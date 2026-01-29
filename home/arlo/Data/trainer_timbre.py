#!/usr/bin/env python3
"""
trainer_timbre.py - Phase 3 Training Script

Replaces discrete instrument IDs (group_id, subgroup_id) and encodec tokens
with continuous 8D TimbreVAE vectors for timbre conditioning.

Key changes from trainer_performerCN2.py:
1. Uses TimbreVAE to encode DCAE latents -> 8D timbre space
2. New ConditioningEncoderTimbre that takes timbre_z instead of discrete IDs
3. Optional density regularization loss
4. Optional envelope conditioning for disentanglement
"""

import sys
sys.path.insert(0, '/home/arlo/Data/dø')
sys.path.append('/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/do-repo/home/arlo/soundspace')

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from datetime import datetime
import argparse, os, json, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
from dataloader import APPROVED_GROUPS, APPROVED_SUBGROUPS
from typing import Optional, Union
import torchaudio
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from contextlib import nullcontext
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F

from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from do.pipeline_do import DoTrainComponents
from dataloader import SimpleLatentDataset, collate_simple

# Import TimbreVAE
from timbre_vae import TimbreVAE

try:
    from loguru import logger
except Exception:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s"
    )
    logger = logging.getLogger("timbre_trainer")


def _get_local_device():
    if torch.cuda.is_available():
        lr = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(lr)
        return f"cuda:{lr}"
    return "cpu"


# Match your grids
DCAE_SR, DCAE_HOP = 44100, 4096
ENC_SR,  ENC_HOP  = 24000,  320
SLOW_HZ = DCAE_SR / DCAE_HOP
FAST_PER_SLOW = (ENC_SR/ENC_HOP) / SLOW_HZ

torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for matmul (faster, less memory)
torch.set_float32_matmul_precision("high")


# ============== New Conditioning Encoder with Timbre Vector ==============

class PerformanceConditionEncoderTimbre(nn.Module):
    """
    Conditioning encoder that uses 8D timbre vector instead of:
    - encodec_tokens
    - group_id
    - subgroup_id

    The timbre_z vector captures the spectral character of the instrument.
    """

    def __init__(
        self,
        d_text: int = 768,
        pr_dim: int = 128,
        timbre_dim: int = 8,
        envelope_dim: int = 0,  # 0 = no envelope conditioning, 4-6 = explicit envelope
        pr_strength: float = 2.0,
        timbre_strength: float = 3.0,
        enable_beat_posenc: bool = True,
    ):
        super().__init__()
        self.d_text = d_text
        self.timbre_dim = timbre_dim
        self.envelope_dim = envelope_dim
        self.pr_strength = float(pr_strength)
        self.timbre_strength = float(timbre_strength)
        self.enable_beat_posenc = bool(enable_beat_posenc)

        # Per-frame projections (piano roll + scalars)
        self.pr_ln = nn.LayerNorm(pr_dim)
        self.pr_proj = nn.Linear(pr_dim, d_text)

        # Scalars: amp, rframe, rbend*mask, voiced, onset = 5 dims
        self.sclr_proj = nn.Sequential(
            nn.Linear(5, d_text), nn.SiLU(), nn.Linear(d_text, d_text)
        )
        self.sclr_ln = nn.LayerNorm(5)

        # NEW: Timbre projection (8D -> d_text)
        # This replaces: group_emb, subgroup_emb, inst_fuse, timbre_pool, timbre_proj
        self.timbre_to_token = nn.Sequential(
            nn.Linear(timbre_dim, d_text),
            nn.LayerNorm(d_text),
            nn.SiLU(),
            nn.Linear(d_text, d_text),
        )

        # Optional: Envelope conditioning for disentanglement
        if envelope_dim > 0:
            self.envelope_proj = nn.Sequential(
                nn.Linear(envelope_dim, d_text // 2),
                nn.SiLU(),
                nn.Linear(d_text // 2, d_text),
            )
        else:
            self.envelope_proj = None

        # Beat positional encoding
        self.beat_pos = nn.Linear(2, d_text) if self.enable_beat_posenc else None

        # Final layer norm and positional embeddings
        self.fuse_ln = nn.LayerNorm(d_text)
        self.cond_pos = nn.Embedding(8192, d_text)
        self.post_token_ln = nn.LayerNorm(d_text)

    def forward(
        self,
        piano_roll: torch.Tensor,     # [B, 128, T_slow]
        amp: torch.Tensor,            # [B, T_slow]
        rframe: torch.Tensor,         # [B, T_slow]
        rbend: torch.Tensor,          # [B, T_slow]
        rbend_mask: torch.Tensor,     # [B, T_slow]
        timbre_z: torch.Tensor,       # [B, 8] - the 8D timbre vector
        envelope_params: Optional[torch.Tensor] = None,  # [B, envelope_dim] optional
    ):
        B, _, T_slow = piano_roll.shape
        D = self.d_text

        # ----- voiced + onset from PR -----
        voiced = (piano_roll > 0).any(dim=1, keepdim=False).float()  # [B, T]
        onset = F.pad((piano_roll[:, :, 1:] > piano_roll[:, :, :-1]).any(dim=1).float(), (1, 0))

        # ----- PR path -----
        pr_in = self.pr_ln(piano_roll.transpose(1, 2))  # [B, T, 128]
        pr_tok = self.pr_proj(pr_in) * self.pr_strength  # [B, T, D]

        # ----- Scalars -----
        sclr_in = torch.stack([amp, rframe, rbend * rbend_mask, voiced, onset], dim=-1)
        sclr_in = self.sclr_ln(sclr_in)
        sclr_tok = self.sclr_proj(sclr_in)  # [B, T, D]

        # ----- Beat positional encoding -----
        if self.enable_beat_posenc and self.beat_pos is not None:
            phi = (rframe - rframe.floor()).unsqueeze(-1)
            beat_feats = torch.cat([torch.sin(2*torch.pi*phi), torch.cos(2*torch.pi*phi)], dim=-1)
            pr_tok = pr_tok + 0.25 * self.beat_pos(beat_feats)

        # Fuse frame tokens
        frame_tok = self.fuse_ln(pr_tok + sclr_tok)  # [B, T, D]

        # ----- Global timbre token from 8D vector -----
        timbre_tok = self.timbre_to_token(timbre_z).unsqueeze(1) * self.timbre_strength  # [B, 1, D]

        # ----- Optional envelope token -----
        if self.envelope_proj is not None and envelope_params is not None:
            env_tok = self.envelope_proj(envelope_params).unsqueeze(1)  # [B, 1, D]
            # Compose: [TIMBRE, ENVELOPE, FRAME_0..T-1]
            tokens = torch.cat([timbre_tok, env_tok, frame_tok], dim=1)
        else:
            # Compose: [TIMBRE, FRAME_0..T-1]
            tokens = torch.cat([timbre_tok, frame_tok], dim=1)

        # Positional embeddings
        L = tokens.shape[1]
        pos = torch.arange(L, device=tokens.device)
        tokens = tokens + self.cond_pos(pos)[None, :, :]

        # Final safety + LN
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=1e3, neginf=-1e3)
        tokens = self.post_token_ln(tokens)

        mask = torch.ones((B, L), device=tokens.device, dtype=torch.bool)
        return tokens, mask


# ============== Helper Modules ==============

class TokenSummarizer(nn.Module):
    def __init__(self, d_text: int, kernel_size: int = 9):
        super().__init__()
        self.ln = nn.LayerNorm(d_text)
        self.dw_conv = nn.Conv1d(d_text, d_text, kernel_size=kernel_size,
                                 padding=kernel_size // 2, groups=d_text, bias=True)
        self.act = nn.SiLU()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.ln(tokens)
        x = x.transpose(1, 2)
        x = self.dw_conv(x)
        x = self.act(x)
        x = x.transpose(1, 2)
        return x.mean(dim=1)


class TemporalCondAdapter(nn.Module):
    def __init__(self, d_text: int, c: int = 8, h: int = 16):
        super().__init__()
        self.pre = nn.Sequential(nn.LayerNorm(d_text), nn.Linear(d_text, d_text), nn.SiLU())
        self.proj = nn.Linear(d_text, c * h)
        self.gain = nn.Parameter(torch.zeros(1))
        self.gain.data.fill_(0.3)
        self.c, self.h = c, h

    def forward(self, tokens: torch.Tensor, T_out: int, scale: float = 1.0) -> torch.Tensor:
        f = self.pre(tokens)
        f = f.transpose(1, 2)
        f = F.interpolate(f, size=T_out, mode="linear", align_corners=False)
        f = f.transpose(1, 2)
        y = self.proj(f)
        y = y.view(y.size(0), T_out, self.c, self.h).permute(0, 2, 3, 1)
        return (self.gain.tanh() * float(scale)) * y


# ============== Main Pipeline ==============

class TimbrePipeline(LightningModule):
    """
    Phase 3 Pipeline with TimbreVAE conditioning.

    Key differences from Pipeline (trainer_performerCN2.py):
    1. Uses frozen TimbreVAE to extract timbre_z from DCAE latents
    2. Uses PerformanceConditionEncoderTimbre instead of PerformanceConditionEncoder
    3. Optional density regularization loss
    4. Optional envelope conditioning
    """

    def __init__(
        self,
        checkpoint_dir: str,
        manifest_json: str,
        timbre_vae_path: str = "/home/arlo/soundspace_checkpoints/timbre_vae_final.pt",
        density_model_path: Optional[str] = None,  # Optional density model for regularization
        learning_rate: float = 1e-4,
        num_workers: int = 8,
        T: int = 1000,
        weight_decay: float = 1e-2,
        every_plot_step: int = 2000,
        shift: float = 3.0,
        cond_cfg_drop_prob: float = 0.15,
        max_steps: int = 200000,
        warmup_steps: int = 10,
        window_slow: int = 256,
        preview_steps: int = 50,
        batch_size: int = 1,
        preview_index: int = 0,
        # Timbre-specific params
        density_loss_weight: float = 0.0,  # 0 = no density loss, 0.01-0.05 = light regularization
        envelope_dim: int = 0,  # 0 = no envelope conditioning
        timbre_strength: float = 3.0,
        partial_mask_prob: float = 0.3,
        control_scale: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=T, shift=shift)

        self._wrote_gt = False
        self._did_preview_once = False
        self.partial_mask_prob = float(partial_mask_prob)

        local_device = _get_local_device()

        # Clear CUDA cache before loading to maximize available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info("Cleared CUDA cache before model loading")

        # Load DCAE and Transformer
        # Use bfloat16 to match inference and save ~7GB VRAM (critical for L4 with 24GB)
        comps = DoTrainComponents(checkpoint_dir=checkpoint_dir, dtype="bfloat16")
        comps.device = torch.device(local_device)

        self.dcae = comps.load_dcae()
        self.dcae.to("cpu")

        logger.info("Loading official ACE-Step weights to FINE-TUNE with timbre conditioning.")
        self.transformers = comps.build_transformer_pretrained()
        self.transformers.to(torch.device(local_device))

        # ============== Load Frozen TimbreVAE ==============
        logger.info(f"Loading TimbreVAE from {timbre_vae_path}")
        vae_ckpt = torch.load(timbre_vae_path, map_location='cpu', weights_only=False)
        vae_config = vae_ckpt['config']

        self.timbre_vae = TimbreVAE(**{k: v for k, v in vae_config.items()
                                        if k in ['input_dim', 'latent_dim', 'hidden_dim',
                                                 'n_residual', 'dropout', 'student_t_df']})
        self.timbre_vae.load_state_dict(vae_ckpt['model'])
        self.timbre_vae.eval()
        for p in self.timbre_vae.parameters():
            p.requires_grad = False

        # Store normalization stats
        self.register_buffer('vae_mean', vae_ckpt['mean'])
        self.register_buffer('vae_std', vae_ckpt['std'])

        self.timbre_dim = vae_config.get('latent_dim', 8)
        logger.info(f"TimbreVAE loaded: {self.timbre_dim}D latent space")

        # ============== Optional Density Model ==============
        self.density_model = None
        self.density_loss_weight = float(density_loss_weight)

        if density_model_path and Path(density_model_path).exists():
            logger.info(f"Loading density model from {density_model_path}")
            from density_estimator import TimbreDensityEstimator
            density_ckpt = torch.load(density_model_path, map_location='cpu', weights_only=False)
            self.density_model = TimbreDensityEstimator(
                latent_channels=8, latent_height=16, hidden_dim=256, n_flows=8
            )
            self.density_model.load_state_dict(density_ckpt['model'])
            self.density_model.eval()
            for p in self.density_model.parameters():
                p.requires_grad = False
            logger.info("Density model loaded for regularization")

        # ============== Freeze/Unfreeze Transformer ==============
        def _set_grad(module, flag):
            if module is None: return
            for p in module.parameters():
                p.requires_grad = flag

        _set_grad(self.transformers, False)

        # Never train unused heads
        for m in [
            getattr(self.transformers, "lyric_embs", None),
            getattr(self.transformers, "lyric_encoder", None),
            getattr(self.transformers, "lyric_proj", None),
            *getattr(self.transformers, "projectors", []),
        ]:
            _set_grad(m, False)

        # Unfreeze some components
        for name in ("genre_embedder", "timestep_embedder", "t_block", "proj_in", "final_layer"):
            m = getattr(self.transformers, name, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = True

        # Unfreeze last 4 blocks (keep x-attn frozen)
        blocks = getattr(self.transformers, "transformer_blocks", [])
        for i in range(max(0, len(blocks)-4), len(blocks)):
            blk = blocks[i]
            for n, mod in blk.named_children():
                if n in ("attn2", "cross_attn"):
                    mod.requires_grad_(False)
                else:
                    for p in mod.parameters():
                        p.requires_grad = True

        # Print trainable params
        trainable = [(n, p.numel()) for n, p in self.transformers.named_parameters() if p.requires_grad]
        total = sum(p.numel() for _, p in self.transformers.named_parameters())
        print(f"[freeze] trainable={sum(n for _,n in trainable)/1e6:.1f}M / total={total/1e6:.1f}M")

        if hasattr(self.transformers, "enable_gradient_checkpointing"):
            self.transformers.enable_gradient_checkpointing()
        self.transformers.train()

        # ============== New Conditioning Encoder ==============
        d_text = getattr(self.transformers.config, "text_embedding_dim", None)
        if d_text is None:
            d_text = self.transformers.config.get("text_embedding_dim", 768)

        self.ctrl_enc = PerformanceConditionEncoderTimbre(
            d_text=d_text,
            timbre_dim=self.timbre_dim,
            envelope_dim=int(envelope_dim),
            timbre_strength=float(timbre_strength),
        )
        # The new ctrl_enc is fully trainable (it's simple enough)
        self.ctrl_enc.requires_grad_(True)

        # Adapter
        self.token_summary = TokenSummarizer(d_text)
        self.cond_adapter = TemporalCondAdapter(d_text=d_text, c=8, h=16)

        # Pitch->height bank
        H_base = int(getattr(self.cond_adapter, "h", 16))
        self.pitch2h_bank = nn.Parameter(0.01 * torch.randn(H_base, 128))

        self.adapter_warmup_steps = 1000

        # Store config
        self.manifest_json = manifest_json
        self.cond_cfg_drop_prob = cond_cfg_drop_prob
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.every_plot_step = every_plot_step

        self._preview_batch = None

    def extract_timbre_z(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Extract 8D timbre vector from DCAE latents.

        Args:
            latents: [B, C, H, T] DCAE latent tensor (C=8, H=16 typically)

        Returns:
            timbre_z: [B, 8] timbre vector (mean over time)
        """
        B, C, H, T = latents.shape

        # Reshape to [B*T, C*H] = [B*T, 128] frames
        frames = latents.permute(0, 3, 1, 2).reshape(B * T, C * H)  # [B*T, 128]

        # Normalize
        frames_norm = (frames - self.vae_mean) / (self.vae_std + 1e-8)

        # Encode with frozen VAE
        with torch.no_grad():
            mu, _ = self.timbre_vae.encode(frames_norm)  # [B*T, 8]

        # Average over time to get per-sample timbre
        mu = mu.view(B, T, -1).mean(dim=1)  # [B, 8]

        return mu

    def extract_envelope_params(self, amp: torch.Tensor) -> torch.Tensor:
        """
        Extract envelope parameters from amplitude curve.

        Args:
            amp: [B, T] amplitude envelope

        Returns:
            envelope_params: [B, 4] (attack_speed, sustain_level, decay_speed, variance)
        """
        B, T = amp.shape

        # Find peak position (attack)
        peak_idx = amp.argmax(dim=1).float() / T  # [B] normalized 0-1

        # Sustain level (mean of middle 50%)
        start, end = int(T * 0.25), int(T * 0.75)
        sustain = amp[:, start:end].mean(dim=1)  # [B]

        # Decay rate (slope after peak to end)
        peak_vals = amp.gather(1, amp.argmax(dim=1, keepdim=True)).squeeze(1)
        end_vals = amp[:, -1]
        decay = (peak_vals - end_vals) / (1 - peak_idx + 1e-6)  # [B]

        # Variance (roughness)
        variance = amp.var(dim=1)  # [B]

        return torch.stack([peak_idx, sustain, decay, variance], dim=1)  # [B, 4]

    def _adapter_gain_scale(self) -> float:
        steps = int(getattr(self, "adapter_warmup_steps", 2000))
        return float(min(1.0, (int(self.global_step) + 1) / max(1, steps)))

    def _match_mod_dtype(self, x, module):
        p = next(module.parameters(), None)
        return x if p is None else x.to(device=p.device, dtype=p.dtype)

    def _bank_softplus_resized(self, H, device, dtype):
        W = self.pitch2h_bank.to(device=device, dtype=dtype)
        if W.shape[0] != H:
            W = F.interpolate(W.T.unsqueeze(0), size=H, mode="linear", align_corners=False).squeeze(0).T
        return F.softplus(W)

    def _amp(self):
        try:
            prec = str(getattr(self.trainer, "precision", "32")).lower()
            if "mixed" in prec:
                return nullcontext()
            else:
                use_bf16 = ("bf16" in prec)
                return torch.autocast("cuda", dtype=torch.bfloat16) if (self.device.type == "cuda" and use_bf16) else nullcontext()
        except Exception:
            return nullcontext()

    def _call_transformer_no_xattn(self, latents, t):
        """Forward pass through transformer using decode() directly with dummy encoder states."""
        B = latents.shape[0]
        device = latents.device
        dtype = latents.dtype

        # Create attention mask for latents (all ones = attend to everything)
        attn_mask = torch.ones(B, latents.shape[-1], device=device, dtype=dtype)

        # Create minimal dummy encoder hidden states for cross-attention
        # Shape: [B, 1, inner_dim] - single token that cross-attention can attend to
        inner_dim = self.transformers.inner_dim
        dummy_encoder_hidden = torch.zeros(B, 1, inner_dim, device=device, dtype=dtype)
        dummy_encoder_mask = torch.ones(B, 1, device=device, dtype=dtype)

        with self._amp():
            # Call decode() directly, bypassing encode() which requires text inputs
            output = self.transformers.decode(
                hidden_states=latents,
                attention_mask=attn_mask,
                encoder_hidden_states=dummy_encoder_hidden,
                encoder_hidden_mask=dummy_encoder_mask,
                timestep=t,
                output_length=latents.shape[-1],
                return_dict=False,
            )
            # decode returns tuple when return_dict=False
            return output[0] if isinstance(output, tuple) else output.sample

    def _partial_mask_control(self, control: torch.Tensor, mask_prob: float = 0.3) -> torch.Tensor:
        """Randomly mask parts of control signal for robustness."""
        if not self.training or torch.rand(()) > mask_prob:
            return control

        B, C, T = control.shape
        mask_len = torch.randint(1, max(2, T // 4), (1,)).item()
        start = torch.randint(0, max(1, T - mask_len), (1,)).item()
        control = control.clone()
        control[:, :, start:start+mask_len] = 0
        return control

    # ============== Data ==============

    def train_dataloader(self):
        ds = SimpleLatentDataset(
            json_path=self.manifest_json,
            window_slow=self.hparams.window_slow,
            filter_groups=["piano", "guitar", "bass", "strings", "brass", "winds"],
        )
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_simple,
            pin_memory=True,
            drop_last=True,
        )

    def configure_optimizers(self):
        # Group params: higher LR for new timbre conditioning, lower for transformer
        timbre_params = list(self.ctrl_enc.parameters()) + \
                        list(self.cond_adapter.parameters()) + \
                        list(self.token_summary.parameters()) + \
                        [self.pitch2h_bank]

        transformer_params = [p for p in self.transformers.parameters() if p.requires_grad]

        param_groups = [
            {"params": timbre_params, "lr": self.learning_rate * 2},  # Higher LR for new modules
            {"params": transformer_params, "lr": self.learning_rate},
        ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        # Linear warmup + cosine decay
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ============== Training Step ==============

    def training_step(self, batch, batch_idx):
        def to_device(x, d):
            if isinstance(x, torch.Tensor): return x.to(d)
            if isinstance(x, dict): return {k: to_device(v, d) for k, v in x.items()}
            if isinstance(x, list): return [to_device(i, d) for i in x]
            return x

        batch = to_device(batch, self.device)
        x0 = batch["latents"]
        B = x0.shape[0]

        # ============== Extract Timbre Z from DCAE latents ==============
        timbre_z = self.extract_timbre_z(x0)  # [B, 8]

        # Optional: add noise to timbre_z during training for robustness
        if self.training and torch.rand(()) < 0.1:
            timbre_z = timbre_z + 0.1 * torch.randn_like(timbre_z)

        # Optional: extract envelope params
        envelope_params = None
        if self.hparams.envelope_dim > 0:
            envelope_params = self.extract_envelope_params(batch["conds"]["amp"])

        # ============== Conditioning dropout ==============
        if self.training:
            p = float(self.cond_cfg_drop_prob)
            if torch.rand(()) < 0.2:
                if torch.rand(()) < 0.5:
                    batch["conds"]["piano_roll"].zero_()
                    batch["conds"]["amp"].zero_()
                    batch["conds"]["rbend"].zero_()
                    batch["conds"]["rbend_mask"].zero_()
            else:
                if torch.rand(()) < p:
                    batch["conds"]["piano_roll"].zero_()
                if torch.rand(()) < p:
                    batch["conds"]["amp"].zero_()

            # Partial masking
            if torch.rand(()) < self.partial_mask_prob:
                batch["conds"]["piano_roll"] = self._partial_mask_control(batch["conds"]["piano_roll"])

            # Timbre dropout (helps with CFG at inference)
            if torch.rand(()) < p:
                timbre_z = torch.zeros_like(timbre_z)

        # ============== Build conditioning tokens ==============
        tokens, mask = self.ctrl_enc(
            piano_roll=batch["conds"]["piano_roll"],
            amp=batch["conds"]["amp"],
            rframe=batch["conds"]["rframe"],
            rbend=batch["conds"]["rbend"],
            rbend_mask=batch["conds"]["rbend_mask"],
            timbre_z=timbre_z,
            envelope_params=envelope_params,
        )

        if torch.isnan(tokens).any():
            tokens = torch.nan_to_num(tokens)
        tokens = tokens.to(dtype=x0.dtype)

        # ============== Flow matching objective ==============
        T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        tau_f32 = torch.rand(B, device=x0.device, dtype=torch.float32).clamp_(1e-4, 1 - 1e-4)
        t_idx = (tau_f32 * (T_train - 1)).to(torch.long)
        sigma = tau_f32.to(x0.dtype).view(B, *([1] * (x0.ndim - 1)))
        z = torch.randn_like(x0)
        x_noisy = (1.0 - sigma) * x0 + sigma * z

        # ============== Adapter ==============
        scale = self._adapter_gain_scale()
        tokens_adapt = self._match_mod_dtype(tokens, self.cond_adapter).clone()
        tokens_adapt[:, 0, :] = tokens_adapt[:, 0, :] * 1.5  # Boost timbre token
        cond_patch = self.cond_adapter(tokens_adapt, T_out=x_noisy.shape[-1], scale=scale)
        cond_patch = cond_patch.to(device=x_noisy.device, dtype=x_noisy.dtype)

        # Pitch-height masking
        B, C, H, T_lat = x0.shape
        pr = batch["conds"]["piano_roll"].to(device=x0.device, dtype=x0.dtype)
        if pr.shape[-1] != T_lat:
            pr = F.interpolate(pr, size=T_lat, mode="nearest")

        W_hp = self._bank_softplus_resized(H, device=x0.device, dtype=x0.dtype)
        Hmap = torch.einsum('bpt,hp->bht', pr, W_hp)
        cond_patch = cond_patch * Hmap.unsqueeze(1)

        x_in = x_noisy + cond_patch

        # ============== Forward pass ==============
        v_pred = self._call_transformer_no_xattn(latents=x_in, t=t_idx)

        x0_hat = x_noisy - sigma * v_pred

        # ============== Loss computation ==============
        # Main reconstruction loss
        pr_tgt = batch["conds"]["piano_roll"].to(x0_hat.device, dtype=x0_hat.dtype)
        if pr_tgt.shape[-1] != T_lat:
            pr_tgt = F.interpolate(pr_tgt, size=T_lat, mode="nearest")

        pr_any = (pr_tgt.amax(dim=1) > 0).to(x0_hat.dtype)
        time_w = 1.0 + 0.5 * pr_any
        w_ex = time_w.mean(dim=1)
        recon_per_ex = (x0_hat - x0).pow(2).flatten(1).mean(dim=1) * w_ex
        recon_loss = recon_per_ex.mean()

        self.log("train/recon_loss", recon_loss.detach(), on_step=True)

        # ============== Optional Density Regularization ==============
        density_loss = torch.zeros((), device=x0.device)

        if self.density_model is not None and self.density_loss_weight > 0:
            # Curriculum: ramp up density loss over training
            density_w = min(self.density_loss_weight,
                           self.global_step / 50000 * self.density_loss_weight)

            if density_w > 0:
                # Extract timbre_z from predicted x0
                with torch.no_grad():
                    timbre_z_pred = self.extract_timbre_z(x0_hat.detach())

                # Density loss: encourage high probability under learned prior
                # This pushes predictions toward the valid timbre manifold
                log_p = self.density_model.flow.log_prob(
                    x0_hat.reshape(B, -1)[:, :128]  # Use first 128 dims
                )
                density_loss = -log_p.mean() * density_w

                self.log("train/density_loss", density_loss.detach(), on_step=True)
                self.log("train/density_weight", density_w, on_step=True)

        # Adapter regularization
        cond_reg = cond_patch.pow(2).mean() * 1e-4

        # Total loss
        loss = recon_loss + density_loss + cond_reg

        self.log("train/loss", loss.detach(), on_step=True)
        self.log("train/timbre_z_norm", timbre_z.norm(dim=1).mean().detach(), on_step=True)

        return loss

    # ============== Preview Generation ==============

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._preview_batch is None:
            self._preview_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v
                                   for k, v in batch.items()}

        if (self.global_step + 1) % self.every_plot_step == 0:
            self._generate_preview()

    @torch.no_grad()
    def _generate_preview(self):
        if self._preview_batch is None:
            return

        batch = self._preview_batch

        def to_device(x, d):
            if isinstance(x, torch.Tensor): return x.to(d)
            if isinstance(x, dict): return {k: to_device(v, d) for k, v in x.items()}
            if isinstance(x, list): return [to_device(i, d) for i in x]
            return x

        batch = to_device(batch, self.device)
        x0 = batch["latents"][:1]  # Take first sample only

        # Extract timbre
        timbre_z = self.extract_timbre_z(x0)

        # Build conditioning
        tokens, mask = self.ctrl_enc(
            piano_roll=batch["conds"]["piano_roll"][:1],
            amp=batch["conds"]["amp"][:1],
            rframe=batch["conds"]["rframe"][:1],
            rbend=batch["conds"]["rbend"][:1],
            rbend_mask=batch["conds"]["rbend_mask"][:1],
            timbre_z=timbre_z,
        )

        # Generate from noise
        x = torch.randn_like(x0)

        steps = self.hparams.preview_steps
        timesteps = torch.linspace(1, 0, steps + 1, device=self.device)

        for i in range(steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            t_idx = (t * 999).long().unsqueeze(0)

            # Adapter
            tokens_adapt = self._match_mod_dtype(tokens, self.cond_adapter)
            cond_patch = self.cond_adapter(tokens_adapt, T_out=x.shape[-1])
            cond_patch = cond_patch.to(device=x.device, dtype=x.dtype)

            x_in = x + cond_patch
            v_pred = self._call_transformer_no_xattn(latents=x_in, t=t_idx)

            x = x + dt * v_pred

        # Decode to audio
        self.dcae.to(self.device)
        audio = self.dcae.decode(x)
        self.dcae.to("cpu")

        # Save
        save_dir = Path(self.hparams.checkpoint_dir) / "previews"
        save_dir.mkdir(exist_ok=True)

        audio_path = save_dir / f"step_{self.global_step:06d}.wav"
        torchaudio.save(str(audio_path), audio[0].cpu(), 44100)

        logger.info(f"Saved preview: {audio_path}")


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="Phase 3: TimbreVAE Training")

    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory for ACE-Step checkpoints")
    parser.add_argument("--manifest_json", type=str, required=True,
                        help="Path to training manifest JSON")
    parser.add_argument("--timbre_vae_path", type=str,
                        default="/home/arlo/soundspace_checkpoints/timbre_vae_final.pt",
                        help="Path to trained TimbreVAE checkpoint")
    parser.add_argument("--density_model_path", type=str, default=None,
                        help="Optional path to density model for regularization")

    # Training params
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--every_plot_step", type=int, default=2000)
    parser.add_argument("--preview_steps", type=int, default=50)

    # Timbre params
    parser.add_argument("--density_loss_weight", type=float, default=0.0,
                        help="Weight for density regularization (0=off, 0.01-0.05=light)")
    parser.add_argument("--envelope_dim", type=int, default=0,
                        help="Envelope conditioning dims (0=off, 4=basic ADSR)")
    parser.add_argument("--timbre_strength", type=float, default=3.0)

    # Hardware
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)

    args = parser.parse_args()

    # Create model
    model = TimbrePipeline(
        checkpoint_dir=args.checkpoint_dir,
        manifest_json=args.manifest_json,
        timbre_vae_path=args.timbre_vae_path,
        density_model_path=args.density_model_path,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        every_plot_step=args.every_plot_step,
        preview_steps=args.preview_steps,
        density_loss_weight=args.density_loss_weight,
        envelope_dim=args.envelope_dim,
        timbre_strength=args.timbre_strength,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.checkpoint_dir) / "timbre_checkpoints",
        filename="timbre-{step:06d}",
        save_top_k=-1,  # Save all checkpoints at every_n_train_steps
        every_n_train_steps=5000,
        save_last=True,
    )

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=Path(args.checkpoint_dir) / "logs",
        name="timbre_training",
    )

    # Trainer
    trainer = Trainer(
        max_steps=args.max_steps,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
