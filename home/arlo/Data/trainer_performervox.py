# ~/Data/trainer_performervox.py
# Apache 2.0 - Vocal-specific trainer with lyric conditioning

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from datetime import datetime
import argparse, os, json, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
from dataloadvervox import APPROVED_GROUPS_VOX as APPROVED_GROUPS, APPROVED_SUBGROUPS_VOX as APPROVED_SUBGROUPS
from typing import Optional, Union, Dict, Any, List
import sys
from pathlib import Path

# Add lyrics processing module
sys.path.append('/home/arlo/Data')
try:
    from lyrics import ACEStepLyricProcessor
    LYRICS_AVAILABLE = True
    print("✅ Lyric conditioning available")
except ImportError as e:
    LYRICS_AVAILABLE = False
    print(f"⚠ Lyric conditioning not available: {e}")
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from contextlib import nullcontext

import torch.nn as nn
import inspect
import torchaudio

from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers import DDPMScheduler

from acestep.pipeline_ace_step import ACEStepTrainComponents
from dataloadvervox import PerformerAIVocalDataset, collate_latent_cond_vocal
from conditioning_encodervox_simple import PerformanceConditionEncoderVocalSimple

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
FAST_PER_SLOW = (ENC_SR/ENC_HOP) / SLOW_HZ  # ~6.96

torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

try:
    from loguru import logger
except Exception:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s"
    )
    logger = logging.getLogger("ace_step")


class TokenSummarizer(nn.Module):
    """
    Preserve temporal cues with a cheap depthwise Conv1d + LN, then mean-pool over time.
    Input:  tokens [B, T, D]
    Output: summary [B, D]
    """
    def __init__(self, d_text: int, kernel_size: int = 9):
        super().__init__()
        self.ln = nn.LayerNorm(d_text)
        # depthwise temporal conv (groups=D) keeps channels independent
        self.dw_conv = nn.Conv1d(d_text, d_text, kernel_size=kernel_size,
                                 padding=kernel_size // 2, groups=d_text, bias=True)
        self.act = nn.SiLU()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.ln(tokens)             # [B, T, D]
        x = x.transpose(1, 2)           # [B, D, T]
        x = self.dw_conv(x)             # [B, D, T]
        x = self.act(x)
        x = x.transpose(1, 2)           # [B, T, D]
        return x.mean(dim=1)            # [B, D]


class TemporalCondAdapter(nn.Module):
    def __init__(self, d_text: int, c: int = 8, h: int = 16):
        super().__init__()
        self.pre  = nn.Sequential(nn.LayerNorm(d_text), nn.Linear(d_text, d_text), nn.SiLU())
        self.proj = nn.Linear(d_text, c * h)  # per-time projection
        self.gain = nn.Parameter(torch.zeros(1))
        # Initialize to non-zero so adapter affects the model from step 0
        self.gain.data.fill_(0.3)  # ~tanh(0.3)=0.29 so patch isn't zero
        self.c, self.h = c, h
        

    def forward(self, tokens: torch.Tensor, T_out: int, scale: float = 1.0) -> torch.Tensor:
        # tokens: [B, T_tok, D]
        f = self.pre(tokens)              # [B, T_tok, D]
        f = f.transpose(1, 2)             # [B, D, T_tok]
        f = F.interpolate(f, size=T_out, mode="linear", align_corners=False)  # [B, D, T_out]
        f = f.transpose(1, 2)             # [B, T_out, D]
        y = self.proj(f)                  # [B, T_out, c*h]
        y = y.view(y.size(0), T_out, self.c, self.h).permute(0, 2, 3, 1)  # [B, c, h, T_out]
        return (self.gain.tanh() * float(scale)) * y


class VocalLyricPRFusion(nn.Module):
    """Fuses lyric and piano roll information for vocal-aware control."""
    def __init__(self, pr_hidden_dim=256, lyric_dim=256):
        super().__init__()
        self.pr_hidden_dim = pr_hidden_dim
        self.lyric_dim = lyric_dim

        # Cross-attention for lyric-PR alignment
        self.lyric_pr_cross_attn = nn.MultiheadAttention(pr_hidden_dim, num_heads=8, batch_first=True)

        # Syllable boundary detection
        self.syllable_detector = nn.Sequential(
            nn.Conv1d(pr_hidden_dim, pr_hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(pr_hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()  # Probability of syllable boundary
        )

        # Multi-note per syllable modeling
        self.note_syllable_fusion = nn.Sequential(
            nn.Linear(pr_hidden_dim + lyric_dim, pr_hidden_dim),
            nn.SiLU(),
            nn.Linear(pr_hidden_dim, pr_hidden_dim)
        )

        # Lyric-conditioned note enhancement
        self.lyric_note_enhancer = nn.Sequential(
            nn.Linear(lyric_dim, pr_hidden_dim),
            nn.SiLU(),
            nn.Linear(pr_hidden_dim, pr_hidden_dim)
        )

    def forward(self, pr_features, lyric_embeddings, lyric_boundaries=None):
        """
        Args:
            pr_features: [B, T, pr_hidden_dim] - Piano roll features
            lyric_embeddings: [B, L, lyric_dim] - Lyric token embeddings
            lyric_boundaries: [B, T] - Optional syllable boundary mask
        Returns:
            fused_features: [B, T, pr_hidden_dim] - Lyric-aware PR features
        """
        B, T, _ = pr_features.shape

        # 1. Cross-attention between lyrics and PR
        lyric_attended_pr, _ = self.lyric_pr_cross_attn(
            query=pr_features,      # [B, T, pr_hidden_dim]
            key=lyric_embeddings,   # [B, L, lyric_dim]
            value=lyric_embeddings  # [B, L, lyric_dim]
        )

        # 2. Detect syllable boundaries from PR patterns
        pr_for_boundary = pr_features.transpose(1, 2)  # [B, pr_hidden_dim, T]
        syllable_probs = self.syllable_detector(pr_for_boundary).squeeze(1)  # [B, T]

        # 3. Enhance PR features with lyric information
        # Pool lyric embeddings for each timestep based on attention weights
        lyric_pooled = lyric_attended_pr  # [B, T, lyric_dim or pr_hidden_dim depending on projection]

        # 4. Fuse PR and lyric information
        if lyric_pooled.shape[-1] != self.lyric_dim:
            # If cross-attention changed dimensionality, project back
            lyric_enhanced = self.lyric_note_enhancer(lyric_embeddings.mean(dim=1, keepdim=True).expand(-1, T, -1))
        else:
            lyric_enhanced = self.lyric_note_enhancer(lyric_pooled)

        # Combine PR features with lyric enhancement
        fused = pr_features + lyric_enhanced

        # 5. Apply syllable boundary weighting
        if lyric_boundaries is not None:
            boundary_weight = lyric_boundaries.unsqueeze(-1)  # [B, T, 1]
        else:
            boundary_weight = syllable_probs.unsqueeze(-1)  # [B, T, 1]

        fused = fused * (1.0 + boundary_weight)  # Emphasize at syllable boundaries

        return fused, syllable_probs


class ControlBranch1D(nn.Module):
    def __init__(self, d_in=128, hidden=256, out_channels_per_block=(512,512,512,512), with_importance=False, instrument_dim=None, vocal_mode=False):
        super().__init__()
        # Per-time mapping M (d_in → hidden)
        self.mapper = nn.Conv1d(d_in, hidden, kernel_size=1)
        self.act    = nn.SiLU()
        self.norm   = nn.GroupNorm(1, hidden)  # LN over channels
        self.with_importance = with_importance
        self.vocal_mode = vocal_mode

        if with_importance:
            self.imp = nn.Conv1d(hidden, 1, kernel_size=1)

        # Instrument-aware FiLM modulation
        self.instrument_dim = instrument_dim
        if instrument_dim is not None:
            # Small MLPs for FiLM conditioning on instrument tokens
            self.film_scale = nn.Linear(instrument_dim, hidden)
            self.film_bias = nn.Linear(instrument_dim, hidden)
            # Initialize to preserve identity when instrument conditioning is zero
            nn.init.zeros_(self.film_scale.weight)
            nn.init.ones_(self.film_scale.bias)  # scale starts at 1
            nn.init.zeros_(self.film_bias.weight)
            nn.init.zeros_(self.film_bias.bias)   # bias starts at 0

        # Vocal-specific lyric-PR fusion
        if vocal_mode:
            self.lyric_pr_fusion = VocalLyricPRFusion(hidden, 256)  # 256 = lyric embedding dim
            self.lyric_projection = nn.Linear(256, hidden)  # Project lyric embeddings to control hidden dim

        self.to_blocks = nn.ModuleList([nn.Conv1d(hidden, oc, 1) for oc in out_channels_per_block])
        for c in self.to_blocks:
            nn.init.zeros_(c.weight);
            if c.bias is not None: nn.init.zeros_(c.bias)

    def forward(self, pr_128t: torch.Tensor, T_out_list: list[int], instrument_token=None, lyric_embeddings=None, lyric_boundaries=None):
        # pr_128t: [B, 129, T]  # 128 PR + 1 AMP
        # instrument_token: [B, D] optional instrument token for FiLM modulation
        # lyric_embeddings: [B, L, 256] optional lyric embeddings for vocal mode
        # lyric_boundaries: [B, T] optional syllable boundary mask
        h = self.act(self.norm(self.mapper(pr_128t)))  # [B, hidden, T]

        # Apply vocal-specific lyric-PR fusion if in vocal mode
        if self.vocal_mode and lyric_embeddings is not None:
            # Convert h to [B, T, hidden] for attention
            h_for_fusion = h.transpose(1, 2)  # [B, T, hidden]

            # Apply lyric-PR fusion
            h_fused, syllable_probs = self.lyric_pr_fusion(
                pr_features=h_for_fusion,
                lyric_embeddings=lyric_embeddings,
                lyric_boundaries=lyric_boundaries
            )

            # Convert back to [B, hidden, T]
            h = h_fused.transpose(1, 2)  # [B, hidden, T]

        # Apply instrument-aware FiLM modulation if available
        if self.instrument_dim is not None and instrument_token is not None:
            # Optional safety: ensure instrument_token matches expected dimension
            if instrument_token.shape[-1] != self.instrument_dim:
                # pad/truncate to match (prevents silent shape bugs if something changes later)
                D = instrument_token.shape[-1]
                if D > self.instrument_dim:
                    instrument_token = instrument_token[:, :self.instrument_dim]
                else:
                    pad = instrument_token.new_zeros(instrument_token.size(0), self.instrument_dim - D)
                    instrument_token = torch.cat([instrument_token, pad], dim=-1)

            # Get scale and bias from instrument token
            scale = self.film_scale(instrument_token)  # [B, hidden]
            bias = self.film_bias(instrument_token)    # [B, hidden]
            # Apply FiLM: h = h * scale + bias
            h = h * scale.unsqueeze(-1) + bias.unsqueeze(-1)  # [B, hidden, T]
        
        outs = []
        for proj, T_i in zip(self.to_blocks, T_out_list):
            h_i = F.interpolate(h, size=T_i, mode="linear", align_corners=False)
            feat = proj(h_i)  # [B, C_i, T_i]
            if self.with_importance:
                m = torch.sigmoid(F.interpolate(self.imp(h), size=T_i, mode="linear", align_corners=False))  # [B,1,T_i]
                feat = feat * m
            outs.append(feat)
        return outs



class Pipeline(LightningModule):
    def __init__(
        self,
        checkpoint_dir: str,
        manifest_json: str,
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
        reinit_heads: bool = False,
        # NEW:
        override_group: Optional[Union[str, int]] = None,
        override_subgroup: Optional[Union[str, int]] = None,
        preview_steps: int = 50,
        batch_size: int = 1,
        preview_index: int = 0,
        train_from_scratch: bool = False,
        encodec_drop_prob: float = 0.10,              # whole-stream
        encodec_channel_drop_prob: float = 0.0,       # per-channel
        encodec_time_mask_prob: float = 0.0,          # per-sample
        encodec_time_mask_max_frac: float = 0.25,
        inst_strength: float = 3.0,
        film_strength: float = 1.0,
        channel_mod_strength: float = 1.0,
        pr_loss_weight: float = 0.6, pr_onset_boost: float = 1.0,
        partial_mask_prob: float = 0.3,
        control_scale: float = 1.0,
        use_ctrl_branch: bool = False, freeze_base_for_ctrl: bool = False,
        # Lyric conditioning parameters
        use_lyric_conditioning: bool = True,
        lyric_conditioning_strength: float = 1.0,
        lyric_dropout_prob: float = 0.1

    ):
        super().__init__()
        self.save_hyperparameters()

        # Memory-safe preview defaults
        self.hparams.enable_preview = getattr(self.hparams, "enable_preview", True)
        # Default to 30 seconds if not specified (was 3 seconds)
        if not hasattr(self.hparams, "preview_seconds") or self.hparams.preview_seconds is None:
            self.hparams.preview_seconds = 30

        self.partial_mask_prob = float(partial_mask_prob)
         # --- 1) Scheduler (ACE FlowMatch Euler)
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=T, shift=shift)

        # self.scheduler = DDPMScheduler(num_train_timesteps=T, prediction_type="epsilon")

        self._wrote_gt = False
        self._did_preview_once = False
        self.encodec_drop_prob = encodec_drop_prob
        self.encodec_channel_drop_prob = encodec_channel_drop_prob
        self.encodec_time_mask_prob = encodec_time_mask_prob
        self.encodec_time_mask_max_frac = encodec_time_mask_max_frac
        self.outlier_disable_steps = 2000   # keep everything for first N steps
        self.outlier_min_batch = 4          # don’t drop if B < 4
        self.outlier_quantile = 0.90 
        self._applied_unfreeze_2k = False
        self._applied_unfreeze_6k = False
        self._resume_unfreeze_done = False 


        self.pr_loss_weight = float(pr_loss_weight)
        self.pr_onset_boost = float(pr_onset_boost)
        self.use_ctrl_branch = bool(use_ctrl_branch)
        self.freeze_base_for_ctrl = bool(freeze_base_for_ctrl)
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="none")
        self.pr_head = None
        self.hparams.control_scale = float(control_scale)

        # Phoneme prediction head for lyric generation
        self.phoneme_head = None
        self.phoneme_loss_weight = 1.0  # From 0.3 (3.3x increase)
        self.num_phonemes = 43  # 41 ARPAbet + PAD + UNK
        print(f"📢 Phoneme prediction enabled (vocab_size={self.num_phonemes}, loss_weight={self.phoneme_loss_weight})")

        # Word prediction head for word-level supervision
        self.word_head = None
        self.word_loss_weight = 0.8  # From 0.5 (1.6x increase)
        # Import MAX_WORD_VOCAB_SIZE from dataloader to ensure consistency
        try:
            from dataloadvervox import MAX_WORD_VOCAB_SIZE
            self.num_words = MAX_WORD_VOCAB_SIZE
        except:
            self.num_words = 8000  # Fallback
        print(f"📢 Word prediction enabled (vocab_size={self.num_words}, loss_weight={self.word_loss_weight})")




#TRAIN FROM SCRATCH
        # --- 2) Build components (NO pretrained weights for transformer)
        # comps = ACEStepTrainComponents(checkpoint_dir=checkpoint_dir, dtype="float32")
        # self.dcae = comps.load_dcae()                            # frozen
        # self.transformers = comps.build_transformer_random()     # RANDOM INIT
        # self.transformers.enable_gradient_checkpointing()
        # self.transformers.train()
        # self._preview_batch = None

        local_device = _get_local_device()

        # one comps; bf16 to cut mem; set device to this rank
        comps = ACEStepTrainComponents(
            checkpoint_dir=self.hparams.checkpoint_dir,
            dtype="float32",
        )
        comps.device = torch.device(local_device)  # <- critical

        # DCAE: put on CPU to save VRAM (we only use it for previews)
        self.dcae = comps.load_dcae()
        self.dcae.to("cpu")

        # Initialize lyric processor for vocal conditioning
        self.lyric_processor = None
        if self.hparams.use_lyric_conditioning and LYRICS_AVAILABLE:
            try:
                self.lyric_processor = ACEStepLyricProcessor()
                print(f"✅ Lyric conditioning initialized (strength: {self.hparams.lyric_conditioning_strength})")
            except Exception as e:
                print(f"⚠ Failed to initialize lyric conditioning: {e}")
                self.hparams.use_lyric_conditioning = False
        else:
            print(f"⚠ Lyric conditioning disabled (use_lyric_conditioning: {self.hparams.use_lyric_conditioning}, available: {LYRICS_AVAILABLE})")

        if self.hparams.train_from_scratch:
            logger.info("Initializing a new Transformer model to train FROM SCRATCH.")
            self.transformers = comps.build_transformer_random()
        else:
            logger.info("Loading official ACE-Step weights to FINE-TUNE.")
            self.transformers = comps.build_transformer_pretrained()

        # ensure on this rank + bf16 (defensive; build_* already used comps.device/dtype)
        self.transformers.to(torch.device(local_device))






        def _set_grad(module, flag):
            if module is None: 
                return
            for p in module.parameters():
                p.requires_grad = flag

        # 0) freeze everything first  ✅ (this was missing)
        _set_grad(self.transformers, False)

        # 1) never train unused heads
        for m in [
            getattr(self.transformers, "lyric_embs", None),
            getattr(self.transformers, "lyric_encoder", None),
            getattr(self.transformers, "lyric_proj", None),
            *getattr(self.transformers, "projectors", []),
        ]:
            _set_grad(m, False)


        for name in ("genre_embedder","timestep_embedder","t_block","proj_in","final_layer"):
            m = getattr(self.transformers, name, None)
            if m is not None:
                for p in m.parameters(): p.requires_grad = True

        # unfreeze last 4 blocks (keep x-attn frozen)
        blocks = getattr(self.transformers, "transformer_blocks", [])
        for i in range(max(0, len(blocks)-4), len(blocks)):
            blk = blocks[i]
            for n, mod in blk.named_children():
                if n in ("attn2", "cross_attn"):      # keep x-attn frozen
                    mod.requires_grad_(False)
                else:
                    for p in mod.parameters(): p.requires_grad = True


            def make_inject(idx):
                def hook(module, inp, out):
                    ctrl_list = getattr(self, "_ctrl_residuals", None)
                    ctrl = None if (ctrl_list is None or idx >= len(ctrl_list)) else ctrl_list[idx]
                    if ctrl is None:
                        return out  # nothing to add this step

                    def add_to(x, add):
                        # x: [B,C,T] or [B,C,H,T]; add: [B,C_res,T] or [B,C_res,H,T]
                        add = add.to(device=x.device, dtype=x.dtype)

                        # if add has 4D, average spatial H to [B,C_res,T]
                        if add.dim() == 4:
                            add = add.mean(dim=2)

                        # time align to x
                        tgt_T = x.shape[-1]
                        if add.shape[-1] != tgt_T:
                            # linear for 3D; if 4D x, we'll lift back after channel proj
                            add = F.interpolate(add, size=tgt_T, mode="linear", align_corners=False)

                        C_res = add.shape[1]
                        C_blk = x.shape[1]

                        # ensure per-block projector exists and is C_res->C_blk
                        while len(self._ctrl_proj) <= idx:
                            self._ctrl_proj.append(nn.Identity())

                        proj = self._ctrl_proj[idx]
                        need_new = not isinstance(proj, nn.Conv1d) or \
                                   getattr(proj, "in_channels", None) != C_res or \
                                   getattr(proj, "out_channels", None) != C_blk
                        if need_new:
                            conv = nn.Conv1d(C_res, C_blk, kernel_size=1)
                            nn.init.zeros_(conv.weight)
                            if conv.bias is not None:
                                nn.init.zeros_(conv.bias)
                            conv = conv.to(device=x.device, dtype=x.dtype)
                            self._ctrl_proj[idx] = conv
                            proj = conv

                        add = proj(add)  # [B,C_blk,T]

                        # gate
                        gate = self._ctrl_gate[idx].tanh() if hasattr(self, "_ctrl_gate") else add.new_ones(())
                        add = gate * add

                        # lift to 4D if x is 4D
                        if x.dim() == 4:
                            add = add.unsqueeze(2).expand(-1, -1, x.shape[2], -1)  # [B,C_blk,H,T]
                        
                        if not hasattr(self, f"_dbg_printed_{idx}"):
                            print(f"[ctrl-hook-{idx}] x", tuple(x.shape), "add", tuple(add.shape))
                            setattr(self, f"_dbg_printed_{idx}", True)
                            
                        return x + add

                    # support various block return types
                    if hasattr(out, "sample") and torch.is_tensor(out.sample):
                        out.sample = add_to(out.sample, ctrl)
                        return out
                    if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                        return (add_to(out[0], ctrl), *out[1:])
                    if torch.is_tensor(out):
                        return add_to(out, ctrl)
                    return out
                return hook

            if self.freeze_base_for_ctrl:
                for p in self.transformers.parameters():
                    p.requires_grad = False



        # sanity print so you SEE it every run
        trainable = [(n,p.numel()) for n,p in self.transformers.named_parameters() if p.requires_grad]
        total = sum(p.numel() for _,p in self.transformers.named_parameters())
        print(f"[freeze] trainable={sum(n for _,n in trainable)/1e6:.1f}M / total={total/1e6:.1f}M")
        # TEMP: force-freeze everything, then selectively unfreeze:
         
        # self.transformers.enable_gradient_checkpointing()
        assert self.transformers is not None
        if hasattr(self.transformers, "enable_gradient_checkpointing"):
            self.transformers.enable_gradient_checkpointing()
        else:
            setattr(self.transformers, "gradient_checkpointing", True)
        self.transformers.train()
        self._preview_batch = None

        # Determine D for the conditioning encoder
        d_text = getattr(self.transformers.config, "text_embedding_dim", None)
        if d_text is None:
            # config may be dict-like
            d_text = self.transformers.config.get("text_embedding_dim", 768)

        # with torch.no_grad():
        #     if hasattr(self.transformers, "cond_gain"):
        #         self.transformers.cond_gain.data.fill_(1.0) 
        
        # x = x + cond_patched
        self.token_summary = TokenSummarizer(d_text)
        self.cond_adapter = TemporalCondAdapter(d_text=d_text, c=8, h=16)
        
        # Initialize pitch->height bank for pitch lock
        H_base = int(getattr(self.cond_adapter, "h", 16))
        self.pitch2h_bank = torch.nn.Parameter(0.01 * torch.randn(H_base, 128))

        self.token_summary.to(self.device)
        self.cond_adapter.to(self.device)
        
        
        

        # Warmup + regularization knobs (kept internal; no CLI needed)
        self.adapter_warmup_steps = 1000   # steps to warm the adapter gain 0→1 (avoid collision with unfreeze)
        self.adapter_l2 = 1e-4  

        self.outlier_thr_lo = 0.30      # relaxed lower bound
        self.outlier_thr_hi = 0.40      # relaxed upper bound during early steps
        self.outlier_relax_steps = 5000 # ~first 3–5k steps
        self.augment_ramp_steps = 5000   




        # Build lookup tables from your approved lists
        
        if isinstance(APPROVED_GROUPS, dict):
            group_names = list(APPROVED_GROUPS.keys())
        else:
            group_names = list(APPROVED_GROUPS)
        sub_names = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})

        self.group2id    = {n: i for i, n in enumerate(group_names)}
        self.subgroup2id = {n: i for i, n in enumerate(sub_names)}

        group_vocab    = len(self.group2id)
        # make sure subgroup vocab covers APPROVED_SUBGROUPS or what's in the manifest
        subgroup_vocab = max(len(self.subgroup2id), self._infer_subgroup_vocab(manifest_json))

        # Initialize class-balanced weights for aux loss
        self.register_buffer("group_ce_w", torch.ones(group_vocab))
        self.register_buffer("sub_ce_w", torch.ones(subgroup_vocab))
        self._setup_class_weights(manifest_json)

        # ---- ctrl_enc now gets the correct sizes ----
        self.ctrl_enc = PerformanceConditionEncoderVocalSimple(
            d_text=d_text,
            enc_channels=8,
            fast_per_slow=FAST_PER_SLOW,
            group_vocab=group_vocab,
            subgroup_vocab=subgroup_vocab,
            inst_emb_dim=384,
            inst_strength=float(getattr(self.hparams, "inst_strength", 3.0)),
            film_strength=float(getattr(self.hparams, "film_strength", 1.0)),
            channel_mod_strength=float(getattr(self.hparams, "channel_mod_strength", 1.0)),
            lyric_strength=float(getattr(self.hparams, "lyric_conditioning_strength", 1.0)),  # Simple additive
        )
        self.ctrl_enc.requires_grad_(False)
        self.ctrl_enc.eval()
        self._set_ctrl_parts_trainable(True)

        # --- ControlBranch (PR+AMP) - initialized after ctrl_enc ---
        if self.use_ctrl_branch:
            blocks = getattr(self.transformers, "transformer_blocks", [])
            tail_blocks = blocks[-4:] if len(blocks) >= 4 else blocks

            # infer latent channel count once, up-front
            C_latent = self._infer_latent_channels()
            n_inject = max(1, len(tail_blocks))
            oc_per = [C_latent] * n_inject  # ensure residual C matches latent C at each injected block

            # d_in = 128 (PR) + 1 (AMP), instrument_dim for FiLM conditioning
            instrument_dim = d_text  # match tokens[:, 0, :].shape[-1]
            # Detect if this model should handle vocals based on manifest content
            is_vocal_model = self._detect_vocal_training()

            self.ctrlnet = ControlBranch1D(
                d_in=129,
                hidden=256,
                out_channels_per_block=oc_per,
                with_importance=True,
                instrument_dim=instrument_dim,
                vocal_mode=is_vocal_model
            ).to(self.device)

            # zero-conv per injection (identity init to unblock gradients) + learnable gate  
            transformer_latent_channels = self._infer_latent_channels()  # dynamically infer actual transformer channel count
            self._ctrl_zero = nn.ModuleList([nn.Conv1d(transformer_latent_channels, transformer_latent_channels, 1) for _ in tail_blocks])
            for z in self._ctrl_zero:
                nn.init.dirac_(z.weight)  # 1x1 identity (out==in) - unblocks gradients
                if z.bias is not None:
                    nn.init.zeros_(z.bias)
            self._ctrl_gate = nn.Parameter(torch.zeros(len(tail_blocks)))  # trainable per-injection
            self._ctrl_gate.data.fill_(0.2)  # small non-zero init to unblock gradients from step 0

            # predeclare per-injection projection; shapes will be fixed lazily later  
            # Project from ControlBranch output channels to transformer latent channels
            # Initialize empty list - will be populated dynamically when we first see the tensor shapes
            self._ctrl_proj = nn.ModuleList()
            # Store the control branch output channels for later initialization
            self._ctrl_branch_channels = C_latent

            # storage for per-block residuals and hooks
            self._ctrl_residuals = [None] * n_inject
            self._ctrl_handles = []

            def make_inject(idx):
                def hook(module, inp, out):
                    ctrl_list = getattr(self, "_ctrl_residuals", None)
                    ctrl = None if (ctrl_list is None or idx >= len(ctrl_list)) else ctrl_list[idx]
                    if ctrl is None:
                        return out  # nothing to add this step

                    def add_to(x, add):
                        # x: [B,C,T] or [B,C,H,T]; add: [B,C_res,T] or [B,C_res,H,T]
                        add = add.to(device=x.device, dtype=x.dtype)

                        # if add has 4D, average spatial H to [B,C_res,T]
                        if add.dim() == 4:
                            add = add.mean(dim=2)

                        # time align to x
                        tgt_T = x.shape[-1]
                        if add.shape[-1] != tgt_T:
                            # linear for 3D; if 4D x, we'll lift back after channel proj
                            add = F.interpolate(add, size=tgt_T, mode="linear", align_corners=False)

                        C_res = add.shape[1]
                        C_blk = x.shape[1]

                        # ensure per-block projector exists and is C_res->C_blk
                        while len(self._ctrl_proj) <= idx:
                            self._ctrl_proj.append(nn.Identity())

                        proj = self._ctrl_proj[idx]
                        need_new = not isinstance(proj, nn.Conv1d) or \
                                   getattr(proj, "in_channels", None) != C_res or \
                                   getattr(proj, "out_channels", None) != C_blk
                        if need_new:
                            conv = nn.Conv1d(C_res, C_blk, kernel_size=1)
                            nn.init.zeros_(conv.weight)
                            if conv.bias is not None:
                                nn.init.zeros_(conv.bias)
                            conv = conv.to(device=x.device, dtype=x.dtype)
                            self._ctrl_proj[idx] = conv
                            proj = conv

                        add = proj(add)  # [B,C_blk,T]

                        # gate
                        gate = self._ctrl_gate[idx].tanh() if hasattr(self, "_ctrl_gate") else add.new_ones(())
                        add = gate * add

                        # lift to 4D if x is 4D
                        if x.dim() == 4:
                            add = add.unsqueeze(2).expand(-1, -1, x.shape[2], -1)  # [B,C_blk,H,T]
                        
                        if not hasattr(self, "_dbg_printed"):
                            print("[ctrl-hook] x", tuple(x.shape), "add", tuple(add.shape))
                            self._dbg_printed = True
                        
                        return x + add

                    # support various block return types
                    if hasattr(out, "sample") and torch.is_tensor(out.sample):
                        out.sample = add_to(out.sample, ctrl)
                        return out
                    if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                        return (add_to(out[0], ctrl), *out[1:])
                    if torch.is_tensor(out):
                        return add_to(out, ctrl)
                    return out
                return hook

            # install
            for i, blk in enumerate(tail_blocks):
                h = blk.register_forward_hook(make_inject(i))
                self._ctrl_handles.append(h)
        self.ctrl_enc.train()




        def _to_id(s, table):
            if s is None: return None
            if isinstance(s, int): return s
            if isinstance(s, str) and s.isdigit(): return int(s)
            return table.get(s, None)

        # USE THE RAW CTOR ARGS (not self.hparams.*) so it also works even if you don’t save hparams
        self.override_gid  = _to_id(override_group, self.group2id)
        self.override_sgid = _to_id(override_subgroup, self.subgroup2id)

        self.manifest_json = manifest_json
        self.cond_cfg_drop_prob = cond_cfg_drop_prob
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.every_plot_step = every_plot_step
        self.group_head = nn.Linear(d_text, len(self.group2id))
        self.sub_head   = nn.Linear(d_text, len(self.subgroup2id))

                

        

        if hasattr(self.scheduler, "config"):
            self.scheduler.config.prediction_type = "v_prediction"

        
        # after loading pretrained and before training, right where you set grads:
        if (not self.hparams.train_from_scratch) and reinit_heads:
            def _reinit_linear(m, gain=0.02):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight, gain=gain)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

            for name in ("genre_embedder","timestep_embedder","t_block","proj_in","final_layer"):
                m = getattr(self.transformers, name, None)
                if m is not None:
                    m.apply(_reinit_linear)


        # self.scheduler = DDPMScheduler(num_train_timesteps=T, prediction_type="epsilon")
      
        # if hasattr(self.scheduler, "config") and hasattr(self.scheduler.config, "prediction_type"):
        #     self.scheduler.config.prediction_type = "epsilon"

    def _set_ctrl_parts_trainable(self, flag=True):
        trainable_parts = [
            "group_emb", "subgroup_emb", "inst_fuse",     # global instrument token
            "timbre_mod_scale", "timbre_mod_bias",        # channel-space FiLM
            "film_scale", "film_bias",                    # D_text FiLM
            "timbre_global"                               # global timbre token proj
        ]
        for name in trainable_parts:
            m = getattr(self.ctrl_enc, name, None)
            if m is not None:
                for p in m.parameters(): p.requires_grad = flag

    # call it:



    # ----- Data -----
    def _infer_subgroup_vocab(self, manifest_json):
        try:
            from pathlib import Path
            data = json.loads(Path(manifest_json).read_text())
            subs = { (it.get("sub_group") or "undefined").lower() for it in data }
            return max(16, len(subs))
        except Exception:
            return 16

    def _setup_class_weights(self, manifest_json):
        """Setup class-balanced weights for aux loss based on manifest hours."""
        try:
            from pathlib import Path
            from collections import Counter
            data = json.loads(Path(manifest_json).read_text())
            
            # Calculate hours per group and subgroup
            group_dur = Counter()
            sub_dur = Counter()
            for it in data:
                dur = float(it.get("duration_sec", it.get("duration", 0)))
                group_name = it.get("group") or it.get("instrument_group") or "unknown"
                sub_name = it.get("sub_group") or it.get("subgroup") or "unknown"
                group_dur[group_name] += dur
                sub_dur[sub_name] += dur
            
            # Convert to hours and compute inverse frequency weights
            alpha = 0.5  # 0=no weight, 1=full inverse frequency
            
            # Group weights
            group_hours = {g: max(1e-6, s/3600.0) for g, s in group_dur.items()}
            for name, hrs in group_hours.items():
                gid = self.group2id.get(name, None)
                if gid is not None:
                    self.group_ce_w[gid] = (1.0 / hrs) ** alpha
            self.group_ce_w /= self.group_ce_w.mean()
            
            # Subgroup weights
            sub_hours = {s: max(1e-6, d/3600.0) for s, d in sub_dur.items()}
            for name, hrs in sub_hours.items():
                sid = self.subgroup2id.get(name, None)
                if sid is not None:
                    self.sub_ce_w[sid] = (1.0 / hrs) ** alpha
            self.sub_ce_w /= self.sub_ce_w.mean()
            
            print(f"[balance] Class weights setup with alpha={alpha}")
            print(f"  Group weights range: [{self.group_ce_w.min():.3f}, {self.group_ce_w.max():.3f}]")
            print(f"  Subgroup weights range: [{self.sub_ce_w.min():.3f}, {self.sub_ce_w.max():.3f}]")
            
        except Exception as e:
            print(f"[balance] Class weights setup failed: {e}. Using uniform weights.")
            # Keep default uniform weights

    def _to_same_device_dtype_as(self, x: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
        p = next(module.parameters(), None)
        if p is None:
            return x
        return x.to(device=p.device, dtype=p.dtype)


    def on_after_backward(self) -> None:
        # Safe to clear per-step state once recomputation+backward are done
        self._ctrl_residuals = None
        
        # Debug: check if control branch gradients are flowing (remove after confirming)
        if hasattr(self, "_ctrl_proj") and self.global_step < 10:  # only log first 10 steps
            for i, conv in enumerate(self._ctrl_proj):
                if conv.weight.grad is not None:
                    self.log(f"dbg/grad_ctrl_proj{i}", conv.weight.grad.norm().detach(), on_step=True)
            
            for i, conv in enumerate(self._ctrl_zero):
                if conv.weight.grad is not None:
                    self.log(f"dbg/grad_ctrl_zero{i}", conv.weight.grad.norm().detach(), on_step=True)
            
            self.log("dbg/ctrl_gate", self._ctrl_gate.data.detach().tanh().mean(), on_step=True)



    def _adapter_gain_scale(self) -> float:
        # linear warmup from 0→1 over adapter_warmup_steps
        steps = int(getattr(self, "adapter_warmup_steps", 2000))
        return float(min(1.0, (int(self.global_step) + 1) / max(1, steps)))


    def on_load_checkpoint(self, checkpoint):
        # Fill in any newly-added ctrl_enc params from the current model init
        sd = checkpoint.get("state_dict", {})
        enc_sd = {f"ctrl_enc.{k}": v.detach().cpu() for k, v in self.ctrl_enc.state_dict().items()}
        added = []
        for k, v in enc_sd.items():
            if k not in sd:
                sd[k] = v
                added.append(k)
        if added:
            print(f"[resume] filled missing ctrl_enc keys: {len(added)}")

        # If pr_head existed in the ckpt, build it now so optimizer groups match.
        if self.pr_head is None:
            w0 = sd.get("pr_head.0.weight")  # Conv1d: [out=256, in_ch, k]
            if w0 is not None:
                in_ch = int(w0.shape[1])
                self.pr_head = nn.Sequential(
                    nn.Conv1d(in_ch, 256, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv1d(256, 128, kernel_size=1)
                ).to(device=self.device, dtype=self.transformers.dtype if hasattr(self.transformers, "dtype") else torch.float32)
                print(f"[resume] reconstructed pr_head with in_ch={in_ch} from checkpoint")

        # If _ctrl_proj existed in the ckpt, reconstruct it with correct shapes
        if hasattr(self, "_ctrl_proj") and self._ctrl_proj is not None:
            ctrl_proj_keys = [k for k in sd.keys() if k.startswith("_ctrl_proj.") and k.endswith(".weight")]
            if ctrl_proj_keys:
                # Clear existing projection layers and rebuild with checkpoint shapes
                self._ctrl_proj = nn.ModuleList([])
                for i, key in enumerate(sorted(ctrl_proj_keys)):
                    w = sd.get(key)
                    if w is not None:
                        out_ch, in_ch, kernel_size = int(w.shape[0]), int(w.shape[1]), int(w.shape[2])
                        conv = nn.Conv1d(in_ch, out_ch, kernel_size)
                        # Initialize to zero like the original lazy creation
                        nn.init.zeros_(conv.weight)
                        if conv.bias is not None:
                            nn.init.zeros_(conv.bias)
                        self._ctrl_proj.append(conv)
                print(f"[resume] reconstructed _ctrl_proj with {len(self._ctrl_proj)} layers from checkpoint shapes")
                
                # Rebuild _ctrl_zero to match _ctrl_proj channel dimensions
                self._ctrl_zero = torch.nn.ModuleList([])
                for proj in self._ctrl_proj:
                    C = int(proj.out_channels)
                    z = torch.nn.Conv1d(C, C, kernel_size=1)
                    torch.nn.init.dirac_(z.weight)
                    if z.bias is not None:
                        torch.nn.init.zeros_(z.bias)
                    self._ctrl_zero.append(z)
                # Make sure gate length matches
                if not isinstance(self._ctrl_gate, torch.nn.Parameter) or self._ctrl_gate.numel() != len(self._ctrl_zero):
                    self._ctrl_gate = torch.nn.Parameter(
                        torch.full((len(self._ctrl_zero),), 0.2, dtype=self._ctrl_proj[0].weight.dtype, device=self._ctrl_proj[0].weight.device)
                    )
                print(f"[resume] rebuilt _ctrl_zero and _ctrl_gate with {len(self._ctrl_zero)} layers matching _ctrl_proj dimensions")

        # Fix optimizer parameter group mismatch
        if "optimizer_states" in checkpoint and len(checkpoint["optimizer_states"]) > 0:
            opt_state = checkpoint["optimizer_states"][0]
            if "param_groups" in opt_state:
                current_groups = 2  # We expect 2 groups: head_params and block_params
                saved_groups = len(opt_state["param_groups"])
                
                if saved_groups > current_groups:
                    print(f"[resume] Merging {saved_groups} saved param groups into {current_groups} current groups")
                    # Merge extra groups (likely pr_head) into the first group (head_params)
                    base_groups = opt_state["param_groups"][:current_groups]
                    extra_groups = opt_state["param_groups"][current_groups:]
                    
                    # The extra groups should be merged into head_params (index 0)
                    # since pr_head and _ctrl_proj are now part of extra_heads in configure_optimizers
                    for extra_group in extra_groups:
                        base_groups[0]["params"].extend(extra_group["params"])
                    
                    opt_state["param_groups"] = base_groups
                    print(f"[resume] Merged optimizer parameter groups successfully")

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        """Override to handle potential optimizer issues during resume"""
        try:
            optimizer.zero_grad(set_to_none=True)
        except Exception as e:
            print(f"[warning] optimizer.zero_grad failed: {e}")
            # Fallback to regular zero_grad
            optimizer.zero_grad()

    def _infer_latent_channels(self) -> int:
        """
        Best-effort inference of the latent channel count C so ControlBranch outputs match.
        Tries common module shapes first, then config hints, then a sane default.
        """
        # 1) Try obvious entry modules
        for name in ("proj_in", "final_layer"):
            m = getattr(self.transformers, name, None)
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                try:
                    return int(m.in_channels)
                except Exception:
                    pass

        # 2) Try config hints
        cfg = getattr(self.transformers, "config", None)
        if cfg is not None:
            for key in ("latent_channels", "in_channels", "channels", "n_channels", "c_in"):
                try:
                    val = getattr(cfg, key, None)
                except Exception:
                    val = None
                if val is None and isinstance(cfg, dict):
                    val = cfg.get(key, None)
                if isinstance(val, int) and val > 0:
                    return int(val)

        # 3) Fallback
        print("[ctrl] Could not infer latent channel count; defaulting to 512")
        return 512



    def _init_pr_head(self, x_latent: torch.Tensor):
        # x_latent: [B, C, H, T]
        _, C, H, _ = x_latent.shape
        in_ch = C * H
        self.pr_head = nn.Sequential(
            nn.Conv1d(in_ch, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(256, 128, kernel_size=1)   # -> [B, 128, T]
        ).to(device=x_latent.device, dtype=x_latent.dtype)

    def _init_phoneme_head(self, x_latent: torch.Tensor):
        """
        Initialize phoneme prediction head.
        Predicts phoneme class at each frame from denoised latent.

        Args:
            x_latent: [B, C, H, T] - denoised latent output
        """
        _, C, H, _ = x_latent.shape
        in_ch = C * H

        # Lightweight phoneme classifier
        self.phoneme_head = nn.Sequential(
            nn.Conv1d(in_ch, 256, kernel_size=5, padding=2),  # Wider kernel for context
            nn.SiLU(),
            nn.GroupNorm(16, 256),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(128, self.num_phonemes, kernel_size=1)  # -> [B, num_phonemes, T]
        ).to(device=x_latent.device, dtype=x_latent.dtype)

        # CRITICAL FIX: Better initialization to prevent early collapse
        # Initialize final layer with small weights and zero bias for better convergence
        with torch.no_grad():
            self.phoneme_head[-1].weight.data *= 0.01  # Small weights
            if self.phoneme_head[-1].bias is not None:
                self.phoneme_head[-1].bias.data.zero_()  # Zero bias

        print(f"[_init_phoneme_head] built phoneme_head with in_ch={in_ch}, num_phonemes={self.num_phonemes}")

    def _phoneme_loss(
        self,
        phoneme_logits: torch.Tensor,
        phoneme_targets: torch.Tensor,
        phoneme_confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute phoneme classification loss with optional confidence weighting.

        Args:
            phoneme_logits: [B, num_phonemes, T] - raw logits from phoneme head
            phoneme_targets: [B, T] - ground truth phoneme indices (long)
            phoneme_confidence: [B, T] - optional confidence weights (0-1)

        Returns:
            Scalar loss
        """
        B, num_classes, T = phoneme_logits.shape

        # Reshape for cross-entropy: [B*T, num_classes], [B*T]
        logits_flat = phoneme_logits.permute(0, 2, 1).reshape(-1, num_classes)  # [B*T, num_classes]
        targets_flat = phoneme_targets.reshape(-1)  # [B*T]

        # CRITICAL FIX: Add label smoothing to prevent early collapse
        # Compute per-frame loss with label smoothing (0.1)
        loss_per_frame = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction='none',
            ignore_index=self.num_phonemes - 2,  # Ignore PAD token (43-2=41)
            label_smoothing=0.1  # Smooth labels to prevent overconfidence
        )  # [B*T]

        # Apply confidence weighting if available
        if phoneme_confidence is not None:
            confidence_flat = phoneme_confidence.reshape(-1)  # [B*T]
            loss_per_frame = loss_per_frame * confidence_flat

        # Return mean loss
        return loss_per_frame.mean()

    def _init_word_head(self, x_latent: torch.Tensor, num_words: int):
        """
        Initialize word prediction head.
        Predicts word class at each frame from denoised latent.

        Args:
            x_latent: [B, C, H, T] - denoised latent output
            num_words: Size of word vocabulary
        """
        _, C, H, _ = x_latent.shape
        in_ch = C * H
        self.num_words = num_words

        # Word classifier (similar to phoneme but for words)
        self.word_head = nn.Sequential(
            nn.Conv1d(in_ch, 512, kernel_size=7, padding=3),  # Wider kernel for word context
            nn.SiLU(),
            nn.GroupNorm(32, 512),
            nn.Dropout(0.1),  # Add dropout for better generalization
            nn.Conv1d(512, 256, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(256, self.num_words, kernel_size=1)  # -> [B, num_words, T]
        ).to(device=x_latent.device, dtype=x_latent.dtype)

        # CRITICAL FIX: Better initialization to prevent early collapse
        # Initialize final layer with small weights and zero bias
        with torch.no_grad():
            self.word_head[-1].weight.data *= 0.01  # Small weights
            if self.word_head[-1].bias is not None:
                self.word_head[-1].bias.data.zero_()  # Zero bias

        print(f"[_init_word_head] built word_head with in_ch={in_ch}, num_words={self.num_words}")

    def _word_loss(
        self,
        word_logits: torch.Tensor,
        word_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute word classification loss with down-weighted <SIL> to prevent collapse.

        Args:
            word_logits: [B, num_words, T] - raw logits from word head
            word_targets: [B, T] - ground truth word indices (long)

        Returns:
            Scalar loss
        """
        B, num_classes, T = word_logits.shape

        # Reshape for cross-entropy: [B*T, num_classes], [B*T]
        logits_flat = word_logits.permute(0, 2, 1).reshape(-1, num_classes)  # [B*T, num_classes]
        targets_flat = word_targets.reshape(-1)  # [B*T]

        # CRITICAL FIX: Add label smoothing to prevent early collapse
        # Compute per-frame loss with label smoothing (0.1)
        loss_per_frame = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction='none',
            ignore_index=0,  # Ignore PAD token (index 0)
            label_smoothing=0.1  # Smooth labels to prevent overconfidence
        )  # [B*T]

        # Down-weight <SIL> tokens (index 2) to prevent collapse
        # Dataset is 62% silence, so down-weight by 0.2x to focus on actual words
        is_sil = (targets_flat == 2)
        weights = torch.where(is_sil, torch.tensor(0.2, device=loss_per_frame.device), torch.tensor(1.0, device=loss_per_frame.device))

        weighted_loss = loss_per_frame * weights
        return weighted_loss.mean()

    def _align_lyrics_with_boundaries(self,
                                      lyric_embeddings: torch.Tensor,  # [B, N_syllables, emb_dim]
                                      boundaries: torch.Tensor,  # [B, T_slow]
                                      T_slow: int) -> torch.Tensor:
        """
        Align lyric embeddings to frame positions using syllable boundaries.
        This preserves timing information instead of destroying it with interpolation.

        Args:
            lyric_embeddings: [B, N_syllables, emb_dim] syllable embeddings
            boundaries: [B, T_slow] with 1.0 at syllable boundaries, 0.0 elsewhere
            T_slow: target sequence length

        Returns:
            aligned: [B, T_slow, emb_dim] frame-aligned lyric features
        """
        B, N_syllables, emb_dim = lyric_embeddings.shape
        device = lyric_embeddings.device

        aligned = torch.zeros(B, T_slow, emb_dim, device=device)

        for b in range(B):
            # Find boundary frame indices (where boundaries > 0.5)
            boundary_frames = torch.where(boundaries[b] > 0.5)[0]

            if len(boundary_frames) == 0:
                # Fallback: distribute syllables evenly across frames
                segment_len = max(1, T_slow // N_syllables)
                for i in range(N_syllables):
                    start = i * segment_len
                    end = min((i + 1) * segment_len, T_slow)
                    if start < T_slow:
                        aligned[b, start:end] = lyric_embeddings[b, i]
            else:
                # Use actual boundary positions to segment the timeline
                # Each syllable occupies frames from its boundary to the next
                for i in range(len(boundary_frames)):
                    start_frame = boundary_frames[i].item()

                    # Determine end frame
                    if i + 1 < len(boundary_frames):
                        end_frame = boundary_frames[i + 1].item()
                    else:
                        end_frame = T_slow

                    # Map to syllable index (handle cases where we have more boundaries than syllables)
                    syllable_idx = min(i, N_syllables - 1)

                    # Fill frames with this syllable's embedding
                    if start_frame < T_slow:
                        aligned[b, start_frame:end_frame] = lyric_embeddings[b, syllable_idx]

        return aligned

    def _align_lyrics_segmented(self,
                                lyric_embeddings: torch.Tensor,  # [B, N_syllables, emb_dim]
                                T_slow: int) -> torch.Tensor:
        """
        Fallback alignment when boundaries are not available.
        Distributes syllables evenly across frames (better than linear interpolation).

        Args:
            lyric_embeddings: [B, N_syllables, emb_dim]
            T_slow: target sequence length

        Returns:
            aligned: [B, T_slow, emb_dim]
        """
        B, N_syllables, emb_dim = lyric_embeddings.shape
        device = lyric_embeddings.device

        aligned = torch.zeros(B, T_slow, emb_dim, device=device)
        segment_len = max(1, T_slow // N_syllables)

        for b in range(B):
            for i in range(N_syllables):
                start = i * segment_len
                end = min((i + 1) * segment_len, T_slow)
                if start < T_slow:
                    aligned[b, start:end] = lyric_embeddings[b, i]

        return aligned

    def _pr_bce_loss(self, pr_logits: torch.Tensor, pr_target: torch.Tensor) -> torch.Tensor:
        """
        pr_logits: [B, 128, T]
        pr_target: [B, 128, T] (0/1)
        Adds optional onset emphasis across time.
        """
        bce = self.bce_logits(pr_logits, pr_target)  # [B,128,T]
        if self.pr_onset_boost > 0.0:
            # time-derivative to get onsets; max over pitch → [B,T]
            d = (pr_target[:, :, 1:] - pr_target[:, :, :-1]).clamp(min=0.0)
            onset = torch.zeros_like(pr_target[:, :, :1]).to(d.dtype)
            onset = torch.cat([onset, d], dim=2).amax(dim=1)  # [B,T]
            w = (1.0 + self.pr_onset_boost * onset).unsqueeze(1)  # [B,1,T]
            bce = bce * w
        return bce.mean()

    def _assert_ctrl_shapes_ok(self):
        """
        Sanity check that control residual channel dimensions are coherent.
        Call this once in on_fit_start() and abort early if mismatched.
        """
        if not hasattr(self, "_ctrl_residuals"): 
            return
        
        n = len(self._ctrl_zero)
        assert len(self._ctrl_proj) in (0, n), f"_ctrl_proj has {len(self._ctrl_proj)} but _ctrl_zero has {n}"
        
        if hasattr(self, "ctrlnet") and hasattr(self.ctrlnet, "to_blocks"):
            assert len(self.ctrlnet.to_blocks) == n, f"ctrlnet.to_blocks has {len(self.ctrlnet.to_blocks)} but _ctrl_zero has {n}"

        for i in range(n):
            if len(self._ctrl_proj) > 0:
                proj_out = self._ctrl_proj[i].out_channels
                zero_in = self._ctrl_zero[i].in_channels
                assert zero_in == proj_out, f"_ctrl_zero[{i}].in_channels={zero_in} != _ctrl_proj[{i}].out_channels={proj_out}"
                # Optional: print shapes once for debugging
                if not hasattr(self, "_shapes_printed"):
                    print(f"[ctrl-shapes] layer {i}: proj {self._ctrl_proj[i].in_channels}→{proj_out}, zero {zero_in}→{self._ctrl_zero[i].out_channels}")
        
        if not hasattr(self, "_shapes_printed"):
            print(f"[ctrl-shapes] All {n} control layers have coherent channel dimensions")
            self._shapes_printed = True

    def _amp(self):  # inside Pipeline
        # When using Lightning's bf16-mixed, don't add manual autocast
        # Lightning already handles precision automatically
        try:
            prec = str(getattr(self.trainer, "precision", "32")).lower()
            if "mixed" in prec:
                # Lightning handles mixed precision automatically
                return nullcontext()
            else:
                # Only use manual autocast for non-mixed precision
                use_bf16 = ("bf16" in prec)
                return torch.autocast("cuda", dtype=torch.bfloat16) if (self.device.type == "cuda" and use_bf16) else nullcontext()
        except Exception:
            return nullcontext()

    def _match_mod_dtype(self, x, module):
        p = next(module.parameters(), None)
        return x if p is None else x.to(device=p.device, dtype=p.dtype)

    def _bank_softplus_resized(self, H, device, dtype):
        """Get pitch->height bank resized to target height H"""
        W = self.pitch2h_bank.to(device=device, dtype=dtype)       # [H_base,128]
        if W.shape[0] != H:
            # resize along height via linear interpolation
            W = F.interpolate(W.T.unsqueeze(0), size=H, mode="linear", align_corners=False).squeeze(0).T
        return F.softplus(W)  # non-negative, smooth

    # REMOVED: _maybe_dropout_encodec - no longer needed with voice reference approach



    # Add this function to your pipeline
    def _partial_mask_control(self, control: torch.Tensor, mask_prob: float = 0.3) -> torch.Tensor:
        B, C, T = control.shape
        out = control.clone()
        for b in range(B):
            if torch.rand(()) < mask_prob:
                t_start = int(torch.randint(0, T, (1,)).item())
                max_len = max(1, int(T * 0.5))
                length  = int(torch.randint(1, max_len + 1, (1,)).item())
                t_end   = min(t_start + length, T)
                out[b, :, t_start:t_end] = 0.0
        return out







    def _has_latents(self, batch):
        return isinstance(batch, dict) and isinstance(batch.get("latents"), torch.Tensor) and batch["latents"].ndim == 4

    def _latents_or_none(self, batch):
        if self._has_latents(batch):
            return batch["latents"]
        return None

    def _move_vocal_cond_to_device(self, vocal_conditioning, device):
        """Helper to move vocal conditioning data to device."""
        if vocal_conditioning is None:
            return None
        vocal_cond_device = []
        for vc in vocal_conditioning:
            if vc is not None:
                vc_device = {
                    "lyrics_data": vc["lyrics_data"],
                    "lyrics_tensors": {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                     for k, v in vc["lyrics_tensors"].items()},
                    "syllable_boundaries": vc["syllable_boundaries"].to(device),
                }
                vocal_cond_device.append(vc_device)
            else:
                vocal_cond_device.append(None)
        return vocal_cond_device

    def on_fit_start(self):
        # one-item, deterministic preview set
        # Use longer window for previews to hear more context
        preview_window = getattr(self.hparams, "preview_window_slow", None)
        if preview_window is None:
            preview_window = getattr(self.hparams, "window_slow", 256)

        ds_prev = PerformerAIVocalDataset(
            json_path=self.manifest_json,
            conditioning_dropout={"piano_roll":0.0, "amp":0.0, "rbend":0.0, "rframe":0.0},
            use_trim=False,                   # <- Don't trim for previews (keep full context)
            pre_roll_seconds=0.0,             # <- No pre-roll needed
            post_roll_seconds=0.0,
            keep_untrimmed_prob=1.0,          # <- Keep full audio
            amp_activity_thr=0.06,
            require_all_core=True,
            collapse_sparse_subgroups_to_any=False,
            static_window=True,               # <- fixed window
            window_slow=preview_window,
            seed=0,
            vocal_conditioning_dropout=0.0,  # NEW: no dropout for preview
        )
        idx = int(self.hparams.preview_index) % len(ds_prev)
        item = ds_prev[idx]
        self._preview_batch = collate_latent_cond_vocal([item])  # stays on CPU; preview fns .to(self.device)
        self._wrote_gt = False  # ensure we write GT for this fixed clip
        self.ctrl_enc.to(self.device)
        self.cond_adapter.to(self.device)
        self.token_summary.to(self.device)
        if hasattr(self, "ctrlnet"):
            self.ctrlnet.to(self.device)


        try:
            print("[sanity] scanning first 100 samples for NaN/Inf/extreme values...")
            ds_check = PerformerAIVocalDataset(
                json_path=self.manifest_json,
                conditioning_dropout={"piano_roll":0.15, "amp":0.10, "rbend":0.10, "rframe":0.05},
                use_trim=True,
                pre_roll_seconds=1.0,
                post_roll_seconds=0.25,
                keep_untrimmed_prob=0.1,
                amp_activity_thr=0.06,
                require_all_core=True,
                collapse_sparse_subgroups_to_any=False,
                static_window=False,
                window_slow=getattr(self.hparams, "window_slow", 256),
                seed=0,
                vocal_conditioning_dropout=0.05,
            )
            bad = 0
            for i in range(min(100, len(ds_check))):
                it = ds_check[i]
                lat = it["latents"]
                if not torch.isfinite(lat).all().item():
                    print(f"[sanity] idx {i}: non-finite in latents")
                    bad += 1
                elif lat.abs().max().item() > 20:
                    print(f"[sanity] idx {i}: unusually large magnitude in latents (max={lat.abs().max().item():.2f})")
                    bad += 1
            if bad == 0:
                print("[sanity] no obvious issues found in first 100 items.")
        except Exception as e:
            print(f"[sanity] dataset scan skipped: {e}")
        
        # Verify control residual channel dimensions are coherent
        self._assert_ctrl_shapes_ok()


    def _save_gt_once(self, batch, sr_out=48000):
        if not getattr(self.trainer, "is_global_zero", True) or self._wrote_gt:
            return
        x0 = self._latents_or_none(batch)
        if x0 is None:
            logger.warning("[preview] no latents in preview batch; skipping GT save this time.")
            return
        B,_,_,T_slow = x0.shape
        audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))

        # cast only for decode
        x_for_dcae = self._match_mod_dtype(x0[:1], self.dcae)
        audio_lengths = torch.tensor([audio_len], device=x_for_dcae.device, dtype=torch.long)
        sr_pred, wav_pred = self.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)

        out_dir = f"{getattr(self.logger, 'log_dir', './logs')}/eval_results/step_{self.global_step}"
        os.makedirs(out_dir, exist_ok=True)
        torchaudio.save(f"{out_dir}/gt.wav", wav_pred[0].float().cpu(), sr_pred)
        self._wrote_gt = True

        

    def _scale_in(self, x, t):
        # Diffusers-style scaling for sigma-param schedulers
        if hasattr(self.scheduler, "scale_model_input"):
            return self.scheduler.scale_model_input(x, t)
        return x
        


    # helper (drop anywhere inside Pipeline, e.g., under other helpers)
    def _preview_sigma0(self):
        if hasattr(self.scheduler, "sigmas"):
            # take the largest sigma in the schedule
            return float(self.scheduler.sigmas.max().item())
        return 1.0


    # def _call_xattn_transformer(self, *, latents, t, tokens, token_mask, attn_mask=None):
    #     B, dtype, dev = latents.shape[0], latents.dtype, latents.device
    #     T_slow = latents.shape[-1]

    #     # latent self-attn mask [B, T_slow]
    #     if attn_mask is None:
    #         attn_mask = torch.ones(B, T_slow, device=dev, dtype=torch.float32)
    #     else:
    #         attn_mask = attn_mask.to(device=dev, dtype=torch.float32)

    #     # cross-attn mask [B, 2+T_slow]
    #     token_mask = token_mask.to(device=dev, dtype=torch.float32)

    #     # dummies ACE expects
    #     spk     = torch.zeros(B, 512, device=dev, dtype=dtype)
    #     lyr_idx = torch.zeros(B, 1, device=dev, dtype=torch.long)
    #     lyr_mask= torch.ones( B, 1, device=dev, dtype=torch.float32)  # float to match speaker/text masks

    #     return self.transformers(
    #         hidden_states=latents,
    #         attention_mask=attn_mask,
    #         encoder_text_hidden_states=tokens,
    #         text_attention_mask=token_mask,
    #         speaker_embeds=spk,
    #         lyric_token_idx=lyr_idx,
    #         lyric_mask=lyr_mask,
    #         timestep=t,
    #     ).sample


    def _make_attn_mask(self, latents):
        # latents: [B, C, H, T_slow] -> mask [B, T_slow]
        B, _, _, T = latents.shape
        return torch.ones(B, T, device=latents.device, dtype=torch.float32)


    def _call_transformer_no_xattn(self, *, latents, t):
        # Minimal, x-attn-free call that still satisfies ACE's forward signature.
        B, dev, dtype = latents.shape[0], latents.device, latents.dtype
        sig = inspect.signature(self.transformers.forward).parameters
        kwargs = {}

        # latents
        if "hidden_states" in sig: kwargs["hidden_states"] = latents
        elif "x" in sig:          kwargs["x"] = latents

        # latent attention mask (time mask)
        if "attention_mask" in sig:
            kwargs["attention_mask"] = self._make_attn_mask(latents)  # [B, T_slow] ones

        # time / sigma arg
        for k in ("timestep", "timesteps", "t", "sigma", "noise_sigma"):
            if k in sig:
                kwargs[k] = t
                break

        # ==== Null out cross-attn inputs (required by ACE) ====
        d_text = getattr(self.transformers.config, "text_embedding_dim", 768)
        # get speaker embed dim if available, fallback 512
        spk_in = 512
        if hasattr(self.transformers, "speaker_embedder"):
            spk_in = getattr(self.transformers.speaker_embedder, "in_features", spk_in)

        # 1 dummy token, fully masked out
        ctx      = torch.zeros(B, 1, d_text, device=dev, dtype=dtype)
        ctx_mask = torch.zeros(B, 1, device=dev, dtype=torch.float32)  # 0 = pad/off

        for k in ("encoder_text_hidden_states", "encoder_hidden_states", "context", "text_embeds"):
            if k in sig:
                kwargs[k] = ctx
        for k in ("text_attention_mask", "encoder_hidden_mask", "encoder_attention_mask", "context_mask"):
            if k in sig:
                kwargs[k] = ctx_mask

        if "speaker_embeds" in sig:
            kwargs["speaker_embeds"] = torch.zeros(B, spk_in, device=dev, dtype=dtype)
        if "lyric_token_idx" in sig:
            kwargs["lyric_token_idx"] = torch.zeros(B, 1, device=dev, dtype=torch.long)
        if "lyric_mask" in sig:
            kwargs["lyric_mask"] = torch.zeros(B, 1, device=dev, dtype=torch.float32)

        out = self.transformers(**kwargs)
        return out.sample if hasattr(out, "sample") else out

    def _sample_ts_and_sigmas(self, bsz, ref_tensor=None):
        # pick device from Lightning
        device = self.device

        # [B] integer timesteps on the training device
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,),
            device=device
        )

        # move the whole sigma schedule to that device *before* indexing
        sigmas_all = self.scheduler.sigmas.to(device=device, dtype=torch.float32)

        # [B] grab the per-sample sigma
        sigmas = sigmas_all.index_select(0, timesteps)  # equivalent to sigmas_all[timesteps]

        # optional: broadcast to match a reference tensor's shape
        if ref_tensor is not None:
            sigmas = sigmas.view((bsz,) + (1,) * (ref_tensor.ndim - 1))

        return timesteps, sigmas


    # In trainer_performer.py

    def train_dataloader(self):
        # For the overfitting test, we create a dummy dataloader that
        # yields our single, pre-loaded preview batch.


#TEST DATALOADER:

        # if self._preview_batch is not None:

        #     dummy_dataset = [self._preview_batch]
            

        #     return DataLoader(
        #         dummy_dataset, 
        #         batch_size=1, 
        #         collate_fn=lambda x: x[0]
        #     )

        # This is the original code for full dataset training.
        # It will only be used when you are NOT overfitting a single file.
        ds = PerformerAIVocalDataset(
            json_path=self.manifest_json,
            conditioning_dropout={"piano_roll":0.15, "amp":0.10, "rbend":0.10, "rframe":0.05},
            use_trim=True,
            pre_roll_seconds=1.0,
            post_roll_seconds=0.25,
            keep_untrimmed_prob=0.1,
            amp_activity_thr=0.06,
            require_all_core=True,
            collapse_sparse_subgroups_to_any=False,
            static_window=False,
            window_slow=getattr(self.hparams, "window_slow", 256),
            seed=0,
            vocal_conditioning_dropout=0.05,
        )

        # --- Weighted sampling to handle strings/brass skew ---
        from collections import Counter
        import numpy as np, json
        from torch.utils.data import WeightedRandomSampler

        try:
            # Check if this is a vocals-only (no-groups) manifest
            with open(self.manifest_json, "r") as f:
                manifest_check = json.load(f)

            # If no entries have 'group' field, skip weighted sampling
            has_groups = any('group' in entry for entry in manifest_check[:100])

            if not has_groups:
                print(f"[balance] No-groups manifest detected - skipping weighted sampling (vocals-only)")
                sampler = None
                use_weighted_sampling = False
            else:
                # Try to get weights from dataset metadata first, fallback to manifest
                if hasattr(ds, 'items_meta') and ds.items_meta:
                    print(f"[balance] Using dataset metadata for weighted sampling ({len(ds.items_meta)} items)")
                    data = ds.items_meta
                else:
                    print(f"[balance] Using manifest for weighted sampling (dataset metadata not available)")
                    data = manifest_check

                # Ensure lengths match
                if len(data) != len(ds):
                    print(f"[balance] Warning: manifest ({len(data)}) and dataset ({len(ds)}) lengths differ. Using dataset length.")
                    data = data[:len(ds)]  # Truncate if manifest is longer

                # Hardcoded group hours based on actual dataset statistics
                # Guitar: 13,455 samples, 1,522,079 seconds, 423 hours
                # Piano: 8,870 samples, 1,224,801 seconds, 340 hours
                # Bass: 6,062 samples, 783,526 seconds, 218 hours
                # Strings: 5,054 samples, 427,570 seconds, 119 hours
                # Winds: 5,126 samples, 348,113 seconds, 97 hours
                # Brass: 4,100 samples, 313,620 seconds, 87 hours
                hours = {
                    'guitar': 423.0,
                    'piano': 340.0,
                    'bass': 218.0,
                    'strings': 119.0,
                    'winds': 97.0,
                    'brass': 87.0
                }
                w = []
                for i, it in enumerate(data):
                    if i >= len(ds):  # Safety check
                        break
                    gid_name = it.get("group") or it.get("instrument_group") or "unknown"
                    w.append(1.0 / hours.get(gid_name, 1.0))
                w = np.asarray(w, dtype=np.float32)
                w = w / w.mean()  # normalize

                sampler = WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True)
                use_weighted_sampling = True
                print(f"[balance] Using weighted sampling. Hours per group: {dict(hours)}")
                for g, h in sorted(hours.items()):
                    weight_boost = (1.0 / h) / (sum(1.0/v for v in hours.values()) / len(hours))
                    print(f"  {g}: {h:.1f}h -> {weight_boost:.2f}x sampling boost")

        except Exception as e:
            print(f"[balance] Weighted sampling failed: {e}. Using shuffle=True instead.")
            sampler = None
            use_weighted_sampling = False

        def _seed_worker(worker_id):
            import numpy as np, random, torch
            seed = torch.initial_seed() % 2**32
            np.random.seed(seed); random.seed(seed)
            info = torch.utils.data.get_worker_info()
            if hasattr(info, "dataset") and hasattr(info.dataset, "rng"):
                info.dataset.rng = np.random.default_rng(seed)

        return DataLoader(
            ds,
            sampler=sampler if use_weighted_sampling else None,
            shuffle=False if use_weighted_sampling else True,  # sampler and shuffle are mutually exclusive
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_latent_cond_vocal,
            batch_size=self.hparams.batch_size,
            worker_init_fn=_seed_worker,   # Fix deterministic randomness across workers
        )

    @torch.no_grad()
    def _preview_x0_direct_rf(self, batch, t_scalar=0.5, sr_out=48000):
        # Early return if preview disabled or memory insufficient
        if not self._can_preview():
            return
        self.transformers.eval()
        with self._amp():
            x0 = self._latents_or_none(batch)
            if x0 is None:
                logger.warning("[preview] no latents; skipping.")
                return

            # Crop to preview_seconds to reduce memory usage
            max_len = int(self.hparams.preview_seconds * SLOW_HZ)
            x0 = x0[..., :max_len]

            x0 = x0.to(self.device)
            B, _, _, T_slow = x0.shape

            # controls on device
            vocal_cond_device = self._move_vocal_cond_to_device(
                batch.get("vocal_conditioning"), self.device
            )

            # NEW: Handle reference_latent
            reference_latent = batch.get("reference_latent")
            if reference_latent is not None:
                reference_latent = reference_latent.to(self.device)

            tokens, mask = self.ctrl_enc(
                piano_roll=batch["conds"]["piano_roll"].to(self.device),
                amp=batch["conds"]["amp"].to(self.device),
                rframe=batch["conds"]["rframe"].to(self.device),
                rbend=batch["conds"]["rbend"].to(self.device),
                rbend_mask=batch["conds"]["rbend_mask"].to(self.device),
                encodec_tokens=batch["encodec_tokens"].to(self.device),
                group_id=batch["group_id"].to(self.device),
                subgroup_id=batch["subgroup_id"].to(self.device),
                vocal_conditioning=vocal_cond_device,
                reference_latent=reference_latent,  # NEW: Voice reference
            )

            # RF pair (x_t, t)
            t = torch.full((B,), float(t_scalar), device=self.device)
            z = torch.randn_like(x0)
            x_t = (1.0 - t.view(B,1,1,1)) * x0 + t.view(B,1,1,1) * z

            T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
            t_idx = (t * (T_train - 1)).long().clamp(0, T_train - 1)

            # adapter patch
            tokens_adapt = self._match_mod_dtype(tokens, self.cond_adapter)
            cond_patch = self.cond_adapter(tokens_adapt, T_out=x_t.shape[-1], scale=self._adapter_gain_scale())
            cond_patch = cond_patch.to(device=x_t.device, dtype=x_t.dtype)

            # Apply pitch-height masking for tighter pitch lock
            B, C, H, T_lat = x0.shape

            # pr: [B, 128, T_pr] -> [B, 128, T_lat]
            pr = batch["conds"]["piano_roll"].to(device=x0.device, dtype=x0.dtype)
            if pr.shape[-1] != T_lat:
                pr = F.interpolate(pr, size=T_lat, mode="nearest")

            # get resized non-negative bank
            W_hp = self._bank_softplus_resized(H, device=x0.device, dtype=x0.dtype)  # [H,128]

            # pitch->height mixing: [B, H, T_lat]
            Hmap = torch.einsum('bpt,hp->bht', pr, W_hp)

            # apply to cond_patch [B, c, h, T]
            cond_patch = cond_patch * Hmap.unsqueeze(1)

            # ---- ControlBranch residuals: PR + AMP + Instrument ----
            if self.use_ctrl_branch:
                pr_128 = batch["conds"]["piano_roll"].to(x_t.device, dtype=x_t.dtype)
                amp_1t = batch["conds"]["amp"].to(x_t.device, dtype=x_t.dtype).unsqueeze(1)
                if amp_1t.shape[-1] != pr_128.shape[-1]:
                    amp_1t = F.interpolate(amp_1t, size=pr_128.shape[-1], mode="nearest")
                ctrl_in = torch.cat([pr_128, amp_1t], dim=1)

                # Get instrument token for FiLM conditioning
                instrument_token = tokens[:, 0, :] if hasattr(self.ctrlnet, 'instrument_dim') and self.ctrlnet.instrument_dim is not None else None

                res_list = self.ctrlnet(ctrl_in, T_out_list=[x_t.shape[-1]] * len(self.ctrlnet.to_blocks), instrument_token=instrument_token)
                scale = float(getattr(self.hparams, "control_scale", 1.0))
                self._ctrl_residuals = [r * scale for r in res_list]


            x_in = x_t + cond_patch
            v_pred = self._call_transformer_no_xattn(latents=x_in, t=t_idx)
            x0_hat = x_t - t.view(B,1,1,1) * v_pred

            # clear residuals so hooks don't leak
            if self.use_ctrl_branch:
                self._ctrl_residuals = None

            # decode short slice
            K = min(T_slow, int(6.0 * SLOW_HZ))
            x_slice = x0_hat[..., :K]
            audio_len = int(round(K * DCAE_HOP * (sr_out / DCAE_SR)))

            x_for_dcae = self._match_mod_dtype(x_slice[:1], self.dcae)
            audio_lengths = torch.tensor([audio_len], device=x_for_dcae.device, dtype=torch.long)
            sr_pred, wav_pred = self.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)

            out_dir = f"{self.logger.log_dir}/eval_results/step_{self.global_step}"
            os.makedirs(out_dir, exist_ok=True)
            if getattr(self.trainer, "is_global_zero", True):
                torchaudio.save(f"{out_dir}/x0_direct_rf_{self.global_step}.wav", wav_pred[0].float().cpu(), sr_pred)
        self.transformers.train()


    @torch.no_grad()
    def _preview_from_noisy_gt(self, batch, t0=0.8, steps=30, sr_out=48000):
        """Start near x0 (RF schedule), take Euler-like steps with no-xattn, then decode."""
        # Early return if preview disabled or memory insufficient
        if not self._can_preview():
            return
        self.transformers.eval()
        with self._amp():
            x0 = self._latents_or_none(batch)
            if x0 is None:
                logger.warning("[preview] no latents; skipping.")
                return

            # Crop to preview_seconds to reduce memory usage
            max_len = int(self.hparams.preview_seconds * SLOW_HZ)
            x0 = x0[..., :max_len]

            x0 = x0.to(self.device)
            B, _, _, T_slow = x0.shape

            vocal_cond_device = self._move_vocal_cond_to_device(
                batch.get("vocal_conditioning"), self.device
            )

            # NEW: Handle reference_latent
            reference_latent = batch.get("reference_latent")
            if reference_latent is not None:
                reference_latent = reference_latent.to(self.device)

            tokens, mask = self.ctrl_enc(
                piano_roll=batch["conds"]["piano_roll"].to(self.device),
                amp=batch["conds"]["amp"].to(self.device),
                rframe=batch["conds"]["rframe"].to(self.device),
                rbend=batch["conds"]["rbend"].to(self.device),
                rbend_mask=batch["conds"]["rbend_mask"].to(self.device),
                encodec_tokens=batch["encodec_tokens"].to(self.device),
                group_id=batch["group_id"].to(self.device),
                subgroup_id=batch["subgroup_id"].to(self.device),
                vocal_conditioning=vocal_cond_device,
                reference_latent=reference_latent,  # NEW: Voice reference
            )

            torch.manual_seed(0)
            z = torch.randn_like(x0)
            x = (1.0 - float(t0)) * x0 + float(t0) * z

            T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
            steps = max(1, int(steps))
            dt = float(t0) / float(steps)

            tokens_adapt = self._match_mod_dtype(tokens, self.cond_adapter)

            # ---- ControlBranch residuals: PR + AMP + Instrument (constant over loop) ----
            if self.use_ctrl_branch:
                pr_128 = batch["conds"]["piano_roll"].to(x.device, dtype=x.dtype)
                amp_1t = batch["conds"]["amp"].to(x.device, dtype=x.dtype).unsqueeze(1)
                if amp_1t.shape[-1] != pr_128.shape[-1]:
                    amp_1t = F.interpolate(amp_1t, size=pr_128.shape[-1], mode="nearest")
                ctrl_in = torch.cat([pr_128, amp_1t], dim=1)

                # Get instrument token for FiLM conditioning
                instrument_token = tokens_adapt[:, 0, :] if hasattr(self.ctrlnet, 'instrument_dim') and self.ctrlnet.instrument_dim is not None else None

                res_list = self.ctrlnet(ctrl_in, T_out_list=[x.shape[-1]] * len(self.ctrlnet.to_blocks), instrument_token=instrument_token)
                scale = float(getattr(self.hparams, "control_scale", 1.0))
                self._ctrl_residuals = [r * scale for r in res_list]


            for i in range(steps, 0, -1):
                t_cont = torch.full((B,), i * dt, device=self.device)
                t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

                cond_patch = self.cond_adapter(tokens_adapt, T_out=x.shape[-1], scale=self._adapter_gain_scale())
                cond_patch = cond_patch.to(device=x.device, dtype=x.dtype)

                # Apply pitch-height masking for tighter pitch lock
                B, C, H, T_lat = x.shape

                # pr: [B, 128, T_pr] -> [B, 128, T_lat]
                pr = batch["conds"]["piano_roll"].to(device=x.device, dtype=x.dtype)
                if pr.shape[-1] != T_lat:
                    pr = F.interpolate(pr, size=T_lat, mode="nearest")

                # get resized non-negative bank
                W_hp = self._bank_softplus_resized(H, device=x.device, dtype=x.dtype)  # [H,128]

                # pitch->height mixing: [B, H, T_lat]
                Hmap = torch.einsum('bpt,hp->bht', pr, W_hp)

                # apply to cond_patch [B, c, h, T]
                cond_patch = cond_patch * Hmap.unsqueeze(1)

                x_in = x + cond_patch
                v_pred = self._call_transformer_no_xattn(latents=x_in, t=t_idx)
                x = x - dt * v_pred

            if self.use_ctrl_branch:
                self._ctrl_residuals = None

            audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
            x_for_dcae = self._match_mod_dtype(x[:1], self.dcae)
            audio_lengths = torch.tensor([audio_len], device=x_for_dcae.device, dtype=torch.long)
            sr_pred, wav_pred = self.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)

            out_dir = f"{getattr(self.logger, 'log_dir', './logs')}/eval_results/step_{self.global_step}"
            os.makedirs(out_dir, exist_ok=True)
            if getattr(self.trainer, "is_global_zero", True):
                torchaudio.save(f"{out_dir}/preview_from_gt_{self.global_step}.wav",
                                wav_pred[0].float().cpu(), sr_pred)
        self.transformers.train()


    @torch.no_grad()
    def _preview(self, batch, steps=30, sr_out=48000, tag="preview"):
        """Full-from-noise preview using RF-style updates and no-xattn path, then decode."""
        # Early return if preview disabled or memory insufficient
        if not self._can_preview():
            return
        self.transformers.eval()
        with self._amp():
            x0 = self._latents_or_none(batch)
            if x0 is None:
                logger.warning("[preview] no latents; skipping.")
                return

            # Crop to preview_seconds to reduce memory usage
            max_len = int(self.hparams.preview_seconds * SLOW_HZ)
            x0 = x0[..., :max_len]

            x0 = x0.to(self.device)
            B, _, _, T_slow = x0.shape

            vocal_cond_device = self._move_vocal_cond_to_device(
                batch.get("vocal_conditioning"), self.device
            )

            # NEW: Handle reference_latent
            reference_latent = batch.get("reference_latent")
            if reference_latent is not None:
                reference_latent = reference_latent.to(self.device)

            tokens, mask = self.ctrl_enc(
                piano_roll=batch["conds"]["piano_roll"].to(self.device),
                amp=batch["conds"]["amp"].to(self.device),
                rframe=batch["conds"]["rframe"].to(self.device),
                rbend=batch["conds"]["rbend"].to(self.device),
                rbend_mask=batch["conds"]["rbend_mask"].to(self.device),
                encodec_tokens=batch["encodec_tokens"].to(self.device),
                group_id=batch["group_id"].to(self.device),
                subgroup_id=batch["subgroup_id"].to(self.device),
                vocal_conditioning=vocal_cond_device,
                reference_latent=reference_latent,  # NEW: Voice reference
            )

            torch.manual_seed(0)
            x = torch.randn_like(x0)

            T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
            steps = max(1, int(steps))
            dt = 1.0 / float(steps)

            tokens_adapt = self._match_mod_dtype(tokens, self.cond_adapter)


            # ---- ControlBranch residuals: PR + AMP + Instrument (constant over loop) ----
            if self.use_ctrl_branch:
                pr_128 = batch["conds"]["piano_roll"].to(x.device, dtype=x.dtype)
                amp_1t = batch["conds"]["amp"].to(x.device, dtype=x.dtype).unsqueeze(1)
                if amp_1t.shape[-1] != pr_128.shape[-1]:
                    amp_1t = F.interpolate(amp_1t, size=pr_128.shape[-1], mode="nearest")
                ctrl_in = torch.cat([pr_128, amp_1t], dim=1)

                # Get instrument token for FiLM conditioning
                instrument_token = tokens_adapt[:, 0, :] if hasattr(self.ctrlnet, 'instrument_dim') and self.ctrlnet.instrument_dim is not None else None

                res_list = self.ctrlnet(ctrl_in, T_out_list=[x.shape[-1]] * len(self.ctrlnet.to_blocks), instrument_token=instrument_token)
                scale = float(getattr(self.hparams, "control_scale", 1.0))
                self._ctrl_residuals = [r * scale for r in res_list]


            for i in range(steps, 0, -1):
                t_cont = torch.full((B,), i * dt, device=self.device)
                t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

                cond_patch = self.cond_adapter(tokens_adapt, T_out=x.shape[-1], scale=self._adapter_gain_scale())
                cond_patch = cond_patch.to(device=x.device, dtype=x.dtype)

                # Apply pitch-height masking for tighter pitch lock
                B, C, H, T_lat = x.shape

                # pr: [B, 128, T_pr] -> [B, 128, T_lat]
                pr = batch["conds"]["piano_roll"].to(device=x.device, dtype=x.dtype)
                if pr.shape[-1] != T_lat:
                    pr = F.interpolate(pr, size=T_lat, mode="nearest")

                # get resized non-negative bank
                W_hp = self._bank_softplus_resized(H, device=x.device, dtype=x.dtype)  # [H,128]

                # pitch->height mixing: [B, H, T_lat]
                Hmap = torch.einsum('bpt,hp->bht', pr, W_hp)

                # apply to cond_patch [B, c, h, T]
                cond_patch = cond_patch * Hmap.unsqueeze(1)

                x_in = x + cond_patch
                v_pred = self._call_transformer_no_xattn(latents=x_in, t=t_idx)
                x = x - dt * v_pred


            audio_len_out = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
            x_for_dcae = self._match_mod_dtype(x[:1], self.dcae)
            audio_lengths = torch.tensor([audio_len_out], device=x_for_dcae.device, dtype=torch.long)
            sr_pred, wav_pred = self.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)

            out_dir = f"{getattr(self.logger, 'log_dir', './logs')}/eval_results/step_{self.global_step}"
            os.makedirs(out_dir, exist_ok=True)
            if getattr(self.trainer, "is_global_zero", True):
                torchaudio.save(f"{out_dir}/{tag}_{self.global_step}.wav",
                                wav_pred[0].float().cpu(), sr_pred)

            if self.use_ctrl_branch:
                self._ctrl_residuals = None


        self.transformers.train()



    @torch.no_grad()
    def _preview_euler_20(self, batch, sr_out=48000):
        """Euler preview with 20 steps."""
        # Early return if preview disabled or memory insufficient
        if not self._can_preview():
            return
        return self._preview(batch, steps=20, sr_out=sr_out, tag="euler_20")

    @torch.no_grad()
    def _preview_euler_40(self, batch, sr_out=48000):
        """Euler preview with 40 steps."""
        # Early return if preview disabled or memory insufficient
        if not self._can_preview():
            return
        return self._preview(batch, steps=40, sr_out=sr_out, tag="euler_40")

    def on_train_batch_start(self, batch, batch_idx):
        # ensure GT save still runs
        if self._preview_batch is not None:
            self._save_gt_once(self._preview_batch)





        # catch-up unfreeze after resume
        if not self._resume_unfreeze_done:
            gs = int(self.global_step)
            blocks = getattr(self.transformers, "transformer_blocks", [])
            if gs >= 6000 and not self._applied_unfreeze_6k:
                self._unfreeze_more(last_k=len(blocks))
                self._applied_unfreeze_6k = True
                self._applied_unfreeze_2k = True
            elif gs >= 2000 and not self._applied_unfreeze_2k:
                self._unfreeze_more(last_k=8)
                self._applied_unfreeze_2k = True
            self._resume_unfreeze_done = True


    def _unfreeze_more(self, last_k):
        if self.freeze_base_for_ctrl:
            return 
        blocks = getattr(self.transformers, "transformer_blocks", [])
        n = len(blocks)
        for i in range(max(0, n-last_k), n):
            blk = blocks[i]
            for name, module in blk.named_children():

                # if name in ("attn2", "cross_attn"):
                #     continue  # keep x-attn frozen
                
                for p in module.parameters():
                    p.requires_grad = True


        if self.global_step > 10000:  # only after model stabilizes
            for i in range(max(0, n-4), n):  # just last 4 blocks
                blk = blocks[i]
                for name, module in blk.named_children():
                    if name in ("attn2", "cross_attn"):
                        for p in module.parameters():
                            p.requires_grad = True




    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        gs = int(self.global_step)
        if gs >= 2000 and not self._applied_unfreeze_2k:
            self._unfreeze_more(last_k=8)
            self._applied_unfreeze_2k = True

        if gs >= 6000 and not self._applied_unfreeze_6k:
            blocks = getattr(self.transformers, "transformer_blocks", [])
            self._unfreeze_more(last_k=len(blocks))
            self._applied_unfreeze_6k = True

        # phase boundaries (tweak if you like)
        if gs == 2000:
            self._unfreeze_more(last_k=8)   # last 8 blocks trainable
        if gs == 6000:
            blocks = getattr(self.transformers, "transformer_blocks", [])
            self._unfreeze_more(last_k=len(blocks))

        # after:
        fixed = self._preview_batch if self._has_latents(self._preview_batch) else batch
        if self._preview_batch is not None and not self._has_latents(self._preview_batch):
            # avoid spamming errors on future steps
            self._preview_batch = None

        preview_now = (self.global_step in (0,1,2,5,10,20)) or (
            ((self.global_step + 1) % max(1, getattr(self.hparams, "every_plot_step", 999999))) == 0
        )

        if preview_now and self._can_preview():
            try:
                self._preview_x0_direct_rf(fixed, t_scalar=0.05, sr_out=32000)
                self._preview_from_noisy_gt(fixed, t0=0.2, steps=min(20, self.hparams.preview_steps), sr_out=32000)
                if self.global_step in (0, 5, 10, 20, 50, 100):
                    # Generate both 20-step and 40-step Euler previews from SAME sample as GT
                    self._preview_euler_20(fixed, sr_out=32000)
                    self._preview_euler_40(fixed, sr_out=32000)
            except Exception as e:
                print(f"[preview] skipped due to error: {e}")


        # optional: full-from-noise preview less frequently
        if (self.global_step + 1) % max(1, getattr(self.hparams, "every_plot_step", 999999)) == 0 and self._can_preview():
            try:
                # Generate both 20-step and 40-step Euler previews for SAME sample as GT
                self._preview_euler_20(fixed, sr_out=48000)
                self._preview_euler_40(fixed, sr_out=48000)
            except Exception as e:
                print(f"[preview] skipped due to error: {e}")

        
    def _resample_curve(self, x_bt, K):
        K = int(K)
        if K <= 0:
            K = x_bt.shape[-1]
        x = x_bt.unsqueeze(1)
        y = F.interpolate(x, size=K, mode="linear", align_corners=False)
        return y.squeeze(1)


    def _ensure_timbre_cond(self, encodec_tokens, Kc, num_embedders):
        """
        encodec_tokens: [B, C_fast, T_fast]
        Returns: [B, num_embedders, Kc] (trim/pad channels, resize time)
        """
        import torch, torch.nn.functional as F
        B, C_fast, T_fast = encodec_tokens.shape
        x = encodec_tokens
        if C_fast > num_embedders:
            x = x[:, :num_embedders, :]
        elif C_fast < num_embedders:
            pad = torch.zeros(B, num_embedders - C_fast, T_fast, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        x = F.interpolate(x, size=Kc, mode="linear", align_corners=False)  # [B,num_embedders,Kc]
        return x

    def _pad_to_len(self, x, L: int):
        # x: [B, N] -> [B, L]
        if x.shape[-1] < L:
            return F.pad(x, (0, L - x.shape[-1]))
        elif x.shape[-1] > L:
            return x[..., :L]
        return x

    def _gather_time_nearest(self, x_bct, K: int):
        """
        x_bct: [B, C, T] of integer codes
        Returns [B, C, K] by nearest-neighbor index selection (no float interpolate).
        """
        B, C, T = x_bct.shape
        # indices 0..T-1 sampled to length K
        idx = torch.linspace(0, T - 1, steps=K, device=x_bct.device)
        idx = idx.round().long().clamp_(0, T - 1)               # [K]
        # gather along time
        return x_bct.index_select(dim=2, index=idx)             # [B, C, K]




    def _resample_channels_time(self, x_ct, K):
        # x_ct: [B,C,T]
        B, C, T = x_ct.shape
        if K % C == 0:
            Tprime = K // C
        else:
            Tprime = max(1, round(K / C))
        y = F.interpolate(x_ct, size=Tprime, mode="linear", align_corners=False)  # [B,C,T']
        vec = y.reshape(B, C*Tprime)
        if vec.shape[-1] < K:
            vec = F.pad(vec, (0, K - vec.shape[-1]))
        elif vec.shape[-1] > K:
            vec = vec[..., :K]
        return vec  # [B,K]

        # ----- Optim -----
  

    def _can_preview(self):
        """Check if we can safely generate previews without OOM."""
        if not getattr(self.hparams, "enable_preview", True):
            return False
        if not getattr(self.trainer, "is_global_zero", True):
            return False
        try:
            free, total = torch.cuda.mem_get_info(self.device)
            return free > 2 * 1024**3  # >2GB free
        except Exception:
            return True

    def configure_optimizers(self):
        head_names = ("final_layer", "proj_in", "genre_embedder", "timestep_embedder", "t_block")

        head_params, block_params = [], []
        for n, p in self.transformers.named_parameters():
            if not p.requires_grad:
                continue
            (head_params if any(h in n for h in head_names) else block_params).append(p)

        # always-optimized small modules
        extra_heads = []
        extra_heads += list(self.cond_adapter.parameters())
        extra_heads += list(self.token_summary.parameters())
        extra_heads += list(self.group_head.parameters()) + list(self.sub_head.parameters())

        # only selected ctrl_enc pieces
        for name in ["group_emb","subgroup_emb","inst_fuse","timbre_mod_scale","timbre_mod_bias",
                    "film_scale","film_bias","timbre_global"]:
            m = getattr(self.ctrl_enc, name, None)
            if m is not None:
                extra_heads += list(m.parameters())

        # control branch
        if hasattr(self, "ctrlnet"):
            extra_heads += list(self.ctrlnet.parameters())




        if hasattr(self, "_ctrl_zero"):
            extra_heads += list(self._ctrl_zero.parameters())
        if hasattr(self, "_ctrl_proj"):
            extra_heads += list(self._ctrl_proj.parameters())
        if hasattr(self, "_ctrl_gate"):
            extra_heads.append(self._ctrl_gate)

        # include the pitch->height bank
        if hasattr(self, "pitch2h_bank") and isinstance(self.pitch2h_bank, torch.nn.Parameter):
            extra_heads.append(self.pitch2h_bank)

        # include pr_head if it exists
        if self.pr_head is not None:
            extra_heads += list(self.pr_head.parameters())

        # Separate phoneme/word heads for higher learning rate
        phoneme_word_params = []
        if self.phoneme_head is not None:
            phoneme_word_params += list(self.phoneme_head.parameters())
        if self.word_head is not None:
            phoneme_word_params += list(self.word_head.parameters())

        # add other heads to head group and dedupe
        head_params.extend(extra_heads)
        head_params = list({id(p): p for p in head_params}.values())
        phoneme_word_params = list({id(p): p for p in phoneme_word_params}.values())
        block_params = list({id(p): p for p in block_params}.values())



        # Even more aggressive for just phoneme/word heads
        groups = [
            {"params": phoneme_word_params, "lr": float(self.hparams.learning_rate) * 8.0},  # 4e-4 for phoneme/word
            {"params": head_params, "lr": float(self.hparams.learning_rate) * 4.0},          # 2e-4 for other heads
            {"params": block_params, "lr": float(self.hparams.learning_rate)},               # 5e-5 for transformer blocks
        ]

        try:
            import bitsandbytes as bnb
            opt = bnb.optim.PagedAdamW8bit(
                groups, betas=(0.9, 0.999), weight_decay=float(self.hparams.weight_decay), eps=1e-8
            )
        except Exception:
            opt = torch.optim.AdamW(
                groups, betas=(0.9, 0.999), weight_decay=float(self.hparams.weight_decay), eps=1e-8, foreach=False
            )

        def lr_lambda(cur):
            if cur < self.hparams.warmup_steps:
                return float(cur) / max(1, self.hparams.warmup_steps)
            progress = float(cur - self.hparams.warmup_steps) / max(1, self.hparams.max_steps - self.hparams.warmup_steps)
            return max(0.0, 1.0 - progress)

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch=-1)
        return [opt], [{"scheduler": sch, "interval": "step"}]



    # ----- Helpers -----
    def get_timestep(self, bsz, device):
        # Logit-normal pick like ACE (defaults)
        u = torch.normal(mean=0.0, std=1.0, size=(bsz,), device="cpu")
        u = torch.sigmoid(u)
        idx = (u * self.scheduler.config.num_train_timesteps).long().clamp(0, self.scheduler.config.num_train_timesteps - 1)
        return self.scheduler.timesteps[idx].to(device)

    def get_sd3_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)              # [T]
        schedule_ts = self.scheduler.timesteps.to(device=device, dtype=timesteps.dtype)  # [T]

        # Find nearest index in schedule for each sampled t
        # result: step_indices shape [B]
        diff = (schedule_ts[None, :] - timesteps[:, None]).abs()
        step_indices = diff.argmin(dim=1)

        sigma = sigmas[step_indices]  # [B]
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)  # -> [B,1,1,1] for 4D latents
        return sigma



    # def _prepare_explicit_conds(self, batch, T_slow):
    #     pr   = batch["conds"]["piano_roll"].to(self.device)              # [B,128,T]
    #     amp  = batch["conds"]["amp"].to(self.device)                     # [B,T]
    #     rb   = (batch["conds"]["rbend"].to(self.device) *
    #             batch["conds"]["rbend_mask"].to(self.device))            # [B,T]
    #     enc  = batch["encodec_tokens"].to(self.device)                   # [B,C_fast,Tfast]

    #     B = pr.shape[0]
    #     Dtext = getattr(self.transformers.config, "text_embedding_dim", 768)
    #     prompt = torch.zeros(B, 1, Dtext, device=self.device, dtype=pr.dtype)

    #     # --- fixed conditioning lengths from the model buffers ---


    #     # Piano roll → [B, 128*PR_T]




    #     # Timbre per-channel → [B, C, K] (indices if Embedding)
    #     num_embedders = len(self.transformers.encodec_embedders)
    #     is_emb = isinstance(self.transformers.encodec_embedders[0], torch.nn.Embedding)

    #     if is_emb:
    #         x = enc.round().long()
    #         C_fast = x.shape[1]
    #         if C_fast > num_embedders: x = x[:, :num_embedders, :]
    #         elif C_fast < num_embedders:
    #             pad = torch.zeros(B, num_embedders - C_fast, x.shape[-1], device=x.device, dtype=x.dtype)
    #             x = torch.cat([x, pad], dim=1)
    #         Kc = x.shape[-1]
    #         timb = self._gather_time_nearest(x, Kc)                                  # [B,C,Kc]
    #         vocab = self.transformers.encodec_embedders[0].num_embeddings
    #         timb = timb.clamp_(0, vocab - 1)
    #     else:
    #         Kc = getattr(self.transformers.encodec_embedders[0], "in_features", T_slow)
    #         x = enc.to(torch.float32)
    #         C_fast = x.shape[1]
    #         if C_fast > num_embedders: x = x[:, :num_embedders, :]
    #         elif C_fast < num_embedders:
    #             pad = torch.zeros(B, num_embedders - C_fast, x.shape[-1], device=x.device, dtype=x.dtype)
    #             x = torch.cat([x, pad], dim=1)
    #         timb = F.interpolate(x, size=Kc, mode="linear", align_corners=False)     # [B,C,Kc]

    #     return {
    #         "prompt_embeds":   prompt,
    #         "piano_roll_cond": pr_vec,
    #         "pitch_bend_cond": pb,
    #         "dynamics_cond":   dyn,
    #         "timbre_cond":     timb,
    #     }



    def _call_transformer(self, latents, tokens, token_mask, timesteps, attn_mask, batch=None):
        """
        Robustly call the transformer no matter which API this build exposes:
        - explicit conditioning API: (prompt_embeds, piano_roll_cond, pitch_bend_cond, dynamics_cond, timbre_cond)
        - cross-attention API: (encoder_text_hidden_states + masks)
        """
        
        if attn_mask is None:
            attn_mask = self._make_attn_mask(latents)

        sig = inspect.signature(self.transformers.forward).parameters
        def has(*names): return all(n in sig for n in names)

        # Branch 1: explicit cond API
        if has("prompt_embeds","piano_roll_cond","pitch_bend_cond","dynamics_cond","timbre_cond"):
            assert batch is not None, "batch is required for explicit cond API"
            B, _, _, T_slow = latents.shape
            conds = self._prepare_explicit_conds(batch, T_slow)

            def try_call(pr, timb):
                return self.transformers(
                    hidden_states=latents,
                    timestep=timesteps,
                    prompt_embeds=conds["prompt_embeds"],
                    piano_roll_cond=pr,
                    pitch_bend_cond=conds["pitch_bend_cond"],
                    dynamics_cond=conds["dynamics_cond"],
                    timbre_cond=timb,
                )

            # try canonical layout first; if shape mismatch, flip time/channel dims
            try:
                out = try_call(conds["piano_roll_cond"], conds["timbre_cond"])
            except RuntimeError:
                pr = conds["piano_roll_cond"].transpose(1,2) if conds["piano_roll_cond"].dim()==3 else conds["piano_roll_cond"]
                timb = conds["timbre_cond"].transpose(1,2)   if conds["timbre_cond"].dim()==3 else conds["timbre_cond"]
                out = try_call(pr, timb)

            return out.sample if hasattr(out, "sample") else out

        # Branch 2: cross-attention API (your original control-token path)
        kwargs = {}
        if "hidden_states" in sig:  kwargs["hidden_states"] = latents
        elif "x" in sig:            kwargs["x"] = latents

        for k in ("timestep","timesteps","t","sigma","noise_sigma"):
            if k in sig:
                kwargs[k] = timesteps
                break

        for k in ("encoder_text_hidden_states","encoder_hidden_states","context","text_embeds"):
            if k in sig:
                kwargs[k] = tokens
                break

        for k in ("text_attention_mask","encoder_hidden_mask","encoder_attention_mask","context_mask"):
            if k in sig:
                kwargs[k] = token_mask
                break

        for k in ("attention_mask","attn_mask","mask"):
            if k in sig:
                kwargs[k] = attn_mask
                break

        if "speaker_embeds" in sig:
            kwargs["speaker_embeds"] = torch.zeros(tokens.shape[0], 512, device=latents.device, dtype=latents.dtype)
        if "lyric_token_idx" in sig:
            kwargs["lyric_token_idx"] = torch.zeros(tokens.shape[0], 1, device=latents.device, dtype=torch.long)
        if "lyric_mask" in sig:
            kwargs["lyric_mask"] = torch.zeros(tokens.shape[0], 1, device=latents.device, dtype=torch.long)

        out = self.transformers(**kwargs)
        return out.sample if hasattr(out, "sample") else out


    # ----- Training core -----

    
    # def training_step(self, batch, batch_idx):
    #     x0 = batch["latents"].to(self.device)
    #     B  = x0.shape[0]

    #     tokens, mask = self.ctrl_enc(
    #         piano_roll=batch["conds"]["piano_roll"].to(self.device),
    #         amp=batch["conds"]["amp"].to(self.device),
    #         rframe=batch["conds"]["rframe"].to(self.device),
    #         rbend=batch["conds"]["rbend"].to(self.device),
    #         rbend_mask=batch["conds"]["rbend_mask"].to(self.device),
    #         encodec_tokens=batch["encodec_tokens"].to(self.device),
    #         group_id=batch["instrument"]["group_id"].to(self.device),
    #         subgroup_id=batch["instrument"]["subgroup_id"].to(self.device),
    #     )

    #     t, sigma = self._sample_ts_and_sigmas(B, ref_tensor=x0)       # [B], [B]
    #     while sigma.ndim < x0.ndim:                       # -> [B,1,1,1]
    #         sigma = sigma.unsqueeze(-1)

    #     eps  = torch.randn_like(x0)
    #     x_t  = x0 + sigma * eps
    #     x_in = self._scale_in(x_t, t)

    #     eps_pred = self.transformers(
    #         hidden_states=x_in,
    #         timestep=t,                      # keep as integer
    #         encoder_hidden_states=tokens,
    #         encoder_attention_mask=mask,
    #     ).sample


    #     loss = F.mse_loss(eps_pred, eps)
    #     self.log("train/loss", loss, on_step=True, prog_bar=True)
    #     self.log("train/lr", self.lr_schedulers().get_last_lr()[0], on_step=True)
    #     return loss



# OG ACE STEP DIFFUSION



    def training_step(self, batch, batch_idx):
        # move to device
        def to_device(x, d):
            if isinstance(x, torch.Tensor): return x.to(d)
            if isinstance(x, dict): return {k: to_device(v,d) for k,v in x.items()}
            if isinstance(x, list): return [to_device(i,d) for i in x]
            return x
        batch = to_device(batch, self.device)

        x0 = batch["latents"]; B = x0.shape[0]

        # Diagnostic: Check phoneme target loading in first 10 steps
        if self.global_step < 10:
            has_phoneme_targets = "phoneme_targets" in batch
            has_phoneme_confidence = "phoneme_confidence" in batch
            has_mhubert = "mhubert_features" in batch
            has_vocal_cond = "vocal_conditioning" in batch

            if self.global_step == 0:
                print(f"\n🔍 [Step {self.global_step}] Phoneme Pipeline Diagnostic:")
                print(f"  ✓ phoneme_targets in batch: {has_phoneme_targets}")
                if has_phoneme_targets:
                    pt = batch["phoneme_targets"]
                    print(f"    - Shape: {pt.shape}")
                    print(f"    - Unique values: {pt.unique().tolist()[:10]}...")
                    print(f"    - Non-PAD ratio: {(pt != 41).float().mean():.2%}")
                print(f"  ✓ phoneme_confidence in batch: {has_phoneme_confidence}")
                if has_phoneme_confidence:
                    pc = batch["phoneme_confidence"]
                    print(f"    - Mean confidence: {pc.mean():.3f}")
                print(f"  ✓ mhubert_features in batch: {has_mhubert}")
                if has_mhubert:
                    mh = batch["mhubert_features"]
                    print(f"    - Shape: {mh.shape}, Norm: {mh.norm():.2f}")
                print(f"  ✓ vocal_conditioning in batch: {has_vocal_cond}")
                if has_vocal_cond:
                    vc_list = batch["vocal_conditioning"]
                    non_none = sum(1 for vc in vc_list if vc is not None)
                    print(f"    - Items with vocal_cond: {non_none}/{len(vc_list)}")

        # controls (swap within-group 10% of the time)
        enc_tok = batch["encodec_tokens"]
        
        if self.training:

            p = float(self.cond_cfg_drop_prob)
            all_mode = (torch.rand(()) < 0.2)
            def maybe_zero(x, prob):
                return x if torch.rand(()) > prob else x.zero_()
            if all_mode:
                if torch.rand(()) < 0.5:
                    batch["conds"]["piano_roll"].zero_(); batch["conds"]["amp"].zero_()
                    batch["conds"]["rbend"].zero_(); batch["conds"]["rbend_mask"].zero_()
            else:
                batch["conds"]["piano_roll"] = maybe_zero(batch["conds"]["piano_roll"], p)
                batch["conds"]["amp"]        = maybe_zero(batch["conds"]["amp"],        p)
                batch["conds"]["rbend"]      = maybe_zero(batch["conds"]["rbend"],      p)

            if torch.rand(1).item() < self.partial_mask_prob:
                batch["conds"]["piano_roll"] = self._partial_mask_control(batch["conds"]["piano_roll"])


            with torch.no_grad():
                swap_p = 0.10
                if torch.rand(()) < swap_p:
                    gid  = batch["group_id"]
                    perm = torch.randperm(gid.shape[0], device=gid.device)
                    same = (gid == gid[perm])
                    if same.any():
                        tmp = enc_tok.clone()
                        tmp[same] = enc_tok[perm][same]
                        enc_tok = tmp

        # NEW: No longer apply EnCodec dropout - voice reference handles conditioning dropout
        # enc_tok = self._maybe_dropout_encodec(enc_tok)  # REMOVED

        # Light data augmentation only for the majority class (strings)
        if self.training:
            strings_gid = self.group2id.get("strings", None)
            if strings_gid is not None:
                mask_major = (batch["group_id"] == strings_gid)
                if mask_major.any():
                    # tiny edge masking for strings only (2% of time dimension)
                    edge_size = max(1, int(enc_tok.shape[-1] * 0.02))
                    enc_tok[mask_major, :, :edge_size] = 0

        ramp = float(min(1.0, (int(self.global_step) + 1) / max(1, int(getattr(self, "augment_ramp_steps", 5000)))))
        self.log("aug/ramp", torch.tensor(ramp, device=x0.device), on_step=True)

        # NEW: Pass reference_latent for voice conditioning
        reference_latent = batch.get("reference_latent")  # [B, 256] speaker embedding or None

        # NEW: Extract mHuBERT features from batch
        mhubert_features = batch.get('mhubert_features')  # [B, T, 768] or None
        if mhubert_features is not None:
            mhubert_features = mhubert_features.to(self.device)
            # Crop temporal dimension if needed (match to reasonable window size)
            if mhubert_features.shape[1] > 512:
                mhubert_features = mhubert_features[:, :512, :]

        tokens, mask = self.ctrl_enc(
            piano_roll=batch["conds"]["piano_roll"],
            amp=batch["conds"]["amp"],
            rframe=batch["conds"]["rframe"],
            rbend=batch["conds"]["rbend"],
            rbend_mask=batch["conds"]["rbend_mask"],
            encodec_tokens=enc_tok,
            group_id=batch["group_id"],
            subgroup_id=batch["subgroup_id"],
            vocal_conditioning=batch.get("vocal_conditioning"),
            reference_latent=reference_latent,  # NEW: Voice reference
            mhubert_features=mhubert_features,  # NEW: mHuBERT features
        )
        if torch.isnan(tokens).any():
            tokens = torch.nan_to_num(tokens)
        tokens = tokens.to(dtype=x0.dtype)  # mixed precision friendly

        # NEW: Log mHuBERT usage statistics
        with torch.no_grad():
            mhubert_available = (mhubert_features is not None)
            self.log("mhubert/available", float(mhubert_available), on_step=True, prog_bar=False)
            if mhubert_available:
                self.log("mhubert/feat_norm", mhubert_features.norm().item(), on_step=True, prog_bar=False)
                self.log("mhubert/seq_len", float(mhubert_features.shape[1]), on_step=True, prog_bar=False)

        # small aux cls heads
        inst_tok     = tokens[:, 0, :]
        group_logits = self.group_head(inst_tok)
        sub_logits   = self.sub_head(inst_tok)
        group_tgt    = batch["group_id"].long().to(group_logits.device)
        sub_tgt      = batch["subgroup_id"].long().to(sub_logits.device)
        # Schedule aux loss weight gradually: start at 0.1, increase to 0.5 over 100k steps
        aux_w        = min(0.5, 0.1 + (self.global_step / 100000) * 0.4)
        aux_loss = F.cross_entropy(group_logits, group_tgt, weight=self.group_ce_w) \
                 + F.cross_entropy(sub_logits, sub_tgt, weight=self.sub_ce_w)

        # ----- RF objective with τ -----
        T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        tau_f32 = torch.rand(B, device=x0.device, dtype=torch.float32).clamp_(1e-4, 1 - 1e-4)
        t_idx   = (tau_f32 * (T_train - 1)).to(torch.long)                  # [B]
        sigma   = tau_f32.to(x0.dtype).view(B, *([1] * (x0.ndim - 1)))      # [B,1,1,1]
        z       = torch.randn_like(x0)
        x_noisy = (1.0 - sigma) * x0 + sigma * z

        # ---- ControlBranch residuals: PR + AMP + Instrument (no dynamic rebuild; sized in __init__) ----
        if self.use_ctrl_branch:
            pr_128 = batch["conds"]["piano_roll"].to(x_noisy.device, dtype=x_noisy.dtype)        # [B,128,Tpr]
            amp_1t = batch["conds"]["amp"].to(x_noisy.device, dtype=x_noisy.dtype).unsqueeze(1)  # [B,1,T?]
            if amp_1t.shape[-1] != pr_128.shape[-1]:
                amp_1t = F.interpolate(amp_1t, size=pr_128.shape[-1], mode="nearest")
            ctrl_in = torch.cat([pr_128, amp_1t], dim=1)                                         # [B,129,Tpr]

            # Get instrument token for FiLM conditioning
            instrument_token = tokens[:, 0, :] if hasattr(self.ctrlnet, 'instrument_dim') and self.ctrlnet.instrument_dim is not None else None

            # Get lyric information for vocal-aware control
            lyric_embeddings = None
            lyric_boundaries = None
            if hasattr(self.ctrlnet, 'vocal_mode') and self.ctrlnet.vocal_mode:
                # Extract real lyric embeddings from vocal_conditioning
                vocal_conditioning_data = batch.get("vocal_conditioning")
                if vocal_conditioning_data is not None and len(vocal_conditioning_data) > 0:
                    # Collect lyric embeddings and boundaries from each batch item
                    lyric_emb_list = []
                    lyric_bound_list = []

                    for b_idx in range(B):
                        vc = vocal_conditioning_data[b_idx] if b_idx < len(vocal_conditioning_data) else None
                        if vc is not None and "lyrics_tensors" in vc:
                            # Extract embeddings from lyrics_tensors
                            lyrics_tensors = vc["lyrics_tensors"]

                            # FIXED: Try multiple keys for embeddings
                            lyric_embs = None
                            if "phoneme_embeddings" in lyrics_tensors:
                                lyric_embs = lyrics_tensors["phoneme_embeddings"]
                            elif "lyrics_embeddings" in lyrics_tensors:
                                lyric_embs = lyrics_tensors["lyrics_embeddings"]
                            elif "lyric_token_id" in lyrics_tensors:
                                # Handle token IDs: create simple embedding
                                token_ids = lyrics_tensors["lyric_token_id"]
                                if isinstance(token_ids, torch.Tensor):
                                    # Create embedding layer if not exists
                                    if not hasattr(self, 'lyric_token_embedder'):
                                        vocab_size = 5000  # Whisper vocab size
                                        self.lyric_token_embedder = nn.Embedding(vocab_size, 256).to(x_noisy.device)
                                    # Clamp token IDs to valid range [0, vocab_size)
                                    token_ids_clamped = torch.clamp(token_ids.long(), 0, 4999)
                                    lyric_embs = self.lyric_token_embedder(token_ids_clamped.to(x_noisy.device))

                            if lyric_embs is not None and isinstance(lyric_embs, torch.Tensor):
                                lyric_emb_list.append(lyric_embs.to(x_noisy.device))
                            else:
                                # Fallback: zeros
                                lyric_emb_list.append(torch.zeros(1, 256, device=x_noisy.device))

                            # Get syllable boundaries
                            if "syllable_boundaries" in vc:
                                syll_bound = vc["syllable_boundaries"].to(x_noisy.device)  # [T_slow]
                                lyric_bound_list.append(syll_bound)
                            else:
                                lyric_bound_list.append(None)
                        else:
                            lyric_emb_list.append(torch.zeros(1, 256, device=x_noisy.device))
                            lyric_bound_list.append(None)

                    # Pad to same length
                    if lyric_emb_list:
                        max_len = max(emb.shape[0] for emb in lyric_emb_list)
                        padded_embs = []
                        for emb in lyric_emb_list:
                            if emb.shape[0] < max_len:
                                pad_len = max_len - emb.shape[0]
                                emb_padded = F.pad(emb, (0, 0, 0, pad_len))
                            else:
                                emb_padded = emb[:max_len]
                            padded_embs.append(emb_padded)
                        lyric_embeddings = torch.stack(padded_embs, dim=0)  # [B, max_len, 256]

                    # FIXED: Stack boundaries with fallback for None values
                    if lyric_bound_list and any(b is not None for b in lyric_bound_list):
                        # Replace None with zeros
                        T_slow = pr_128.shape[-1]  # Get T_slow from piano roll
                        stacked = []
                        for b in lyric_bound_list:
                            if b is not None:
                                # Ensure correct length
                                if b.shape[0] < T_slow:
                                    b = F.pad(b, (0, T_slow - b.shape[0]))
                                elif b.shape[0] > T_slow:
                                    b = b[:T_slow]
                                stacked.append(b)
                            else:
                                stacked.append(torch.zeros(T_slow, device=x_noisy.device))
                        lyric_boundaries = torch.stack(stacked, dim=0)  # [B, T_slow]

            # CRITICAL FIX: Use syllable boundaries for proper alignment (not linear interpolation!)
            if lyric_embeddings is not None:
                T_slow = pr_128.shape[-1]
                L = lyric_embeddings.shape[1]

                # Debug logging (can be removed after validation)
                if self.global_step % 100 == 0:
                    logger.info(f"[Lyric Debug] Lyric seq len: {L}, T_slow: {T_slow}, Boundaries: {lyric_boundaries.shape if lyric_boundaries is not None else 'None'}")

                # Use syllable boundaries for proper alignment (preserves timing!)
                if L != T_slow:
                    if lyric_boundaries is not None:
                        # Use boundary-based alignment
                        lyric_embeddings = self._align_lyrics_with_boundaries(
                            lyric_embeddings,  # [B, L, 256]
                            lyric_boundaries,  # [B, T_slow]
                            T_slow
                        )  # [B, T_slow, 256]
                    else:
                        # Fallback: segment-based alignment (better than linear interpolation)
                        lyric_embeddings = self._align_lyrics_segmented(lyric_embeddings, T_slow)

            res_list = self.ctrlnet(
                ctrl_in,
                T_out_list=[x_noisy.shape[-1]] * len(self.ctrlnet.to_blocks),
                instrument_token=instrument_token,
                lyric_embeddings=lyric_embeddings,
                lyric_boundaries=lyric_boundaries
            )
            scale = float(getattr(self.hparams, "control_scale", 1.0))
            self._ctrl_residuals = [r * scale for r in res_list]


        # adapter (randomized gain)
        import random
        base_gain         = self._adapter_gain_scale()
        random_multiplier = 1
        scale             = base_gain * random_multiplier

        tokens_adapt = self._match_mod_dtype(tokens, self.cond_adapter).clone()
        tokens_adapt[:, 0, :] = tokens_adapt[:, 0, :] * 1.5
        cond_patch = self.cond_adapter(tokens_adapt, T_out=x_noisy.shape[-1], scale=scale)
        cond_patch = cond_patch.to(device=x_noisy.device, dtype=x_noisy.dtype)

        # Apply pitch-height masking for tighter pitch lock
        B, C, H, T_lat = x0.shape

        # pr: [B, 128, T_pr] -> [B, 128, T_lat]
        pr = batch["conds"]["piano_roll"].to(device=x0.device, dtype=x0.dtype)
        if pr.shape[-1] != T_lat:
            pr = F.interpolate(pr, size=T_lat, mode="nearest")

        # get resized non-negative bank
        W_hp = self._bank_softplus_resized(H, device=x0.device, dtype=x0.dtype)  # [H,128]

        # pitch->height mixing: [B, H, T_lat]
        Hmap = torch.einsum('bpt,hp->bht', pr, W_hp)

        # apply to cond_patch [B, c, h, T]
        cond_patch = cond_patch * Hmap.unsqueeze(1)

        x_in = x_noisy + cond_patch

        # Add lyric conditioning for vocal tracks
        lyric_conditioning = None
        if self.hparams.use_lyric_conditioning and self.lyric_processor is not None:
            try:
                # Check if this batch contains vocal data and lyrics
                if "lyrics" in batch and batch["lyrics"] is not None:
                    lyric_conditioning = self._process_lyric_conditioning(batch["lyrics"], x_in.shape[-1])
                elif "audio_path" in batch and any("vox" in str(path).lower() or "vocal" in str(path).lower() for path in batch["audio_path"]):
                    # Extract lyrics on-the-fly for vocal tracks
                    lyric_conditioning = self._extract_and_process_lyrics(batch["audio_path"], x_in.shape[-1])

                if lyric_conditioning is not None:
                    # Apply lyric conditioning to x_in with strength scaling
                    lyric_strength = self.hparams.lyric_conditioning_strength
                    if self.training and torch.rand(1).item() < self.hparams.lyric_dropout_prob:
                        lyric_strength = 0.0  # Dropout for robustness

                    x_in = x_in + lyric_conditioning * lyric_strength

            except Exception as e:
                print(f"⚠ Lyric conditioning failed: {e}")

        # no-x-attn forward
        v_pred = self._call_transformer_no_xattn(latents=x_in, t=t_idx)






        x0_hat = x_noisy - sigma * v_pred

        # PR target (used for both recon weighting and optional PR-BCE)
        B, C, H, T_slow = x0_hat.shape
        pr_tgt = batch["conds"]["piano_roll"].to(x0_hat.device, dtype=x0_hat.dtype)
        if pr_tgt.shape[-1] != T_slow:
            pr_tgt = F.interpolate(pr_tgt, size=T_slow, mode="nearest")

        # recon weighting by activity
        pr_any         = (pr_tgt.amax(dim=1) > 0).to(x0_hat.dtype)  # [B,T]
        time_w         = 1.0 + 0.5 * pr_any
        w_ex           = time_w.mean(dim=1)
        recon_per_ex   = (x0_hat - x0).pow(2).flatten(1).mean(dim=1) * w_ex
        self.log("train/recon_mean", recon_per_ex.mean().detach(), on_step=True)

        # Monitor per-instrument breakdown
        with torch.no_grad():
            # Log fraction of each instrument group in batch
            self.log("breakdown/strings_frac", (group_tgt==self.group2id.get("strings",-1)).float().mean(), on_step=True)
            self.log("breakdown/brass_frac",   (group_tgt==self.group2id.get("brass",-1)).float().mean(),   on_step=True)
            self.log("breakdown/woodwinds_frac", (group_tgt==self.group2id.get("woodwinds",-1)).float().mean(), on_step=True)
            self.log("breakdown/keyboards_frac", (group_tgt==self.group2id.get("keyboards",-1)).float().mean(), on_step=True)
            
            # Per-group reconstruction loss
            for name, gid in self.group2id.items():
                m = (group_tgt == gid)
                if m.any():
                    self.log(f"recon/{name}", recon_per_ex[m].mean(), on_step=True)
                    
            # Per-group aux loss accuracy
            group_pred = group_logits.argmax(dim=1)
            for name, gid in self.group2id.items():
                m = (group_tgt == gid)
                if m.any():
                    acc = (group_pred[m] == group_tgt[m]).float().mean()
                    self.log(f"acc/{name}", acc, on_step=True)

        # optional PR BCE aux with vocal-aware modifications
        pr_loss = x0_hat.new_zeros(())
        if self.pr_loss_weight > 0:
            # ensure pr_head exists by now; if not found in ckpt and you still want it:
            if self.pr_head is None:
                # create a fresh one deterministically using current shapes
                B, C, H, T_slow = x0_hat.shape
                self._init_pr_head(x0_hat)

            x_feat    = x0_hat.reshape(B, C*H, T_slow)
            pr_logits = self.pr_head(x_feat)

            # Vocal-aware PR loss modification
            is_vocal_batch = self._is_vocal_batch(batch)
            if is_vocal_batch and lyric_embeddings is not None:
                # For vocal tracks, apply lyric-aware PR loss weighting
                pr_loss = self._vocal_aware_pr_loss(pr_logits, pr_tgt, lyric_boundaries)
                self.log("aux/vocal_pr_bce", pr_loss.detach(), on_step=True)
            else:
                # Standard PR loss for non-vocal tracks
                pr_loss = self._pr_bce_loss(pr_logits, pr_tgt)
                self.log("aux/pr_bce", pr_loss.detach(), on_step=True)

        # Phoneme prediction loss for vocal intelligibility
        phoneme_loss = x0_hat.new_zeros(())

        # Log phoneme target availability
        phoneme_targets_available = "phoneme_targets" in batch
        self.log("phoneme/targets_available", float(phoneme_targets_available), on_step=True, prog_bar=False)

        if self.phoneme_loss_weight > 0 and "phoneme_targets" in batch:
            # Initialize phoneme head if needed
            if self.phoneme_head is None:
                self._init_phoneme_head(x0_hat)

            # Get phoneme targets and confidence from batch
            phoneme_targets = batch["phoneme_targets"].to(x0_hat.device)  # [B, T_slow]
            phoneme_confidence = batch.get("phoneme_confidence", None)
            if phoneme_confidence is not None:
                phoneme_confidence = phoneme_confidence.to(x0_hat.device)

            # Predict phonemes from denoised latent
            # NOTE: This is currently only used as an auxiliary loss during training.
            # During inference/preview, there's no mechanism to enforce phoneme sequences,
            # so the model might produce mumbled vocals even if phoneme accuracy improves.
            # TODO: Consider adding phoneme-conditioned guidance during sampling for clearer articulation.
            x_feat = x0_hat.reshape(B, C*H, T_slow)
            phoneme_logits = self.phoneme_head(x_feat)  # [B, num_phonemes, T_slow]

            # Align target length if needed
            if phoneme_targets.shape[-1] != T_slow:
                phoneme_targets = F.interpolate(
                    phoneme_targets.unsqueeze(1).float(),
                    size=T_slow,
                    mode='nearest'
                ).squeeze(1).long()
                if phoneme_confidence is not None:
                    phoneme_confidence = F.interpolate(
                        phoneme_confidence.unsqueeze(1),
                        size=T_slow,
                        mode='linear',
                        align_corners=False
                    ).squeeze(1)

            # Compute phoneme loss
            phoneme_loss = self._phoneme_loss(phoneme_logits, phoneme_targets, phoneme_confidence)
            self.log("aux/phoneme_ce", phoneme_loss.detach(), on_step=True)

            # Log phoneme prediction accuracy and diagnostics
            with torch.no_grad():
                phoneme_pred = phoneme_logits.argmax(dim=1)  # [B, T]
                # Ignore PAD tokens in accuracy
                valid_mask = (phoneme_targets != (self.num_phonemes - 2))
                if valid_mask.any():
                    acc = (phoneme_pred[valid_mask] == phoneme_targets[valid_mask]).float().mean()
                    self.log("aux/phoneme_acc", acc, on_step=True, prog_bar=True)
                    # Log non-PAD ratio to track how much real phoneme data we have
                    non_pad_ratio = valid_mask.float().mean()
                    self.log("phoneme/non_pad_ratio", non_pad_ratio, on_step=True, prog_bar=False)
        else:
            # Log when phoneme loss is skipped
            if self.phoneme_loss_weight > 0 and self.global_step % 100 == 0:
                print(f"⚠ [Step {self.global_step}] Phoneme loss enabled (weight={self.phoneme_loss_weight}) but no phoneme_targets in batch!")

        # Word prediction loss for coherent lyrics
        word_loss = x0_hat.new_zeros(())

        # Log word target availability
        word_targets_available = "word_targets" in batch
        self.log("word/targets_available", float(word_targets_available), on_step=True, prog_bar=False)

        if self.word_loss_weight > 0 and "word_targets" in batch:
            # Initialize word head if needed
            if self.word_head is None:
                self._init_word_head(x0_hat, self.num_words)

            # Get word targets from batch
            word_targets = batch["word_targets"].to(x0_hat.device)  # [B, T_slow]

            # Predict words from denoised latent
            x_feat = x0_hat.reshape(B, C*H, T_slow)
            word_logits = self.word_head(x_feat)  # [B, num_words, T_slow]

            # Align target length if needed
            if word_targets.shape[-1] != T_slow:
                word_targets = F.interpolate(
                    word_targets.unsqueeze(1).float(),
                    size=T_slow,
                    mode='nearest'
                ).squeeze(1).long()

            # Compute word loss
            word_loss = self._word_loss(word_logits, word_targets)
            self.log("aux/word_ce", word_loss.detach(), on_step=True)

            # Log word prediction accuracy
            with torch.no_grad():
                word_pred = word_logits.argmax(dim=1)  # [B, T]
                # Ignore PAD (0) and SIL (2) tokens in accuracy - only measure on actual sung words
                valid_mask = (word_targets != 0) & (word_targets != 2)
                if valid_mask.any():
                    acc = (word_pred[valid_mask] == word_targets[valid_mask]).float().mean()
                    self.log("aux/word_acc", acc, on_step=True, prog_bar=True)
                    # Log singing ratio (non-PAD, non-SIL)
                    singing_ratio = valid_mask.float().mean()
                    self.log("word/singing_ratio", singing_ratio, on_step=True, prog_bar=False)

                    # Also log raw accuracy including SIL (for debugging collapse)
                    valid_mask_with_sil = (word_targets != 0)
                    if valid_mask_with_sil.any():
                        acc_with_sil = (word_pred[valid_mask_with_sil] == word_targets[valid_mask_with_sil]).float().mean()
                        self.log("word/acc_with_sil", acc_with_sil, on_step=True, prog_bar=False)
        else:
            # Log when word loss is skipped
            if self.word_loss_weight > 0 and self.global_step % 100 == 0:
                print(f"⚠ [Step {self.global_step}] Word loss enabled (weight={self.word_loss_weight}) but no word_targets in batch!")

        cond_reg = cond_patch.pow(2).mean()

        # --- Outlier filtering ---
        disable = (B < int(getattr(self, "outlier_min_batch", 4))) or \
                (int(self.global_step) < int(getattr(self, "outlier_disable_steps", 2000)))
        if disable:
            keep_mask  = torch.ones_like(recon_per_ex)
            OUTLIER_THR = float("inf")
        else:
            q   = float(getattr(self, "outlier_quantile", 0.90))
            q   = min(max(q, 0.50), 0.99)
            thr = torch.quantile(recon_per_ex.detach().float(), q)
            OUTLIER_THR = float(thr.item())
            keep_mask   = (recon_per_ex <= thr).float()

        kept    = int(keep_mask.sum().item())
        dropped = int(keep_mask.numel() - kept)
        recon_loss = (recon_per_ex * keep_mask).sum() / keep_mask.clamp_min(1).sum()

        # CRITICAL FIX: Add warmup for auxiliary losses to prevent early collapse
        # Set to 3000 steps to match main LR warmup - ensures aux losses ramp up
        # in sync with learning rate, avoiding awkward interaction where aux losses
        # reach full weight while LR is still ramping
        aux_warmup_steps = 3000
        if self.global_step < aux_warmup_steps:
            aux_warmup_factor = float(self.global_step) / aux_warmup_steps
        else:
            aux_warmup_factor = 1.0

        # Apply warmup to phoneme and word losses
        phoneme_weight = self.phoneme_loss_weight * aux_warmup_factor
        word_weight = self.word_loss_weight * aux_warmup_factor

        # Log warmup progress (first 10 steps)
        if self.global_step < 10:
            print(f"[Aux Warmup] Step {self.global_step}: phoneme_weight={phoneme_weight:.4f}, word_weight={word_weight:.4f}")

        loss = recon_loss + self.adapter_l2 * cond_reg + aux_w * aux_loss + self.pr_loss_weight * pr_loss + phoneme_weight * phoneme_loss + word_weight * word_loss

        v_target = z - x0
        v_loss   = F.mse_loss(v_pred, v_target)  # diag only

        self.log("aux/group_ce", aux_loss.detach(), on_step=True)
        self.log("train/outlier_thr", torch.tensor(OUTLIER_THR, device=loss.device), on_step=True)
        self.log("train/outlier_kept", torch.tensor(kept, device=loss.device, dtype=torch.float32), on_step=True)
        self.log("train/outlier_dropped", torch.tensor(dropped, device=loss.device, dtype=torch.float32), on_step=True)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("dbg/cos_vpred_vtgt", F.cosine_similarity(v_pred.flatten(1), v_target.flatten(1)).mean(), on_step=True)
        self.log("dbg/|v_pred|", v_pred.float().pow(2).mean().sqrt(), on_step=True)
        self.log("dbg/|v_tgt|",  v_target.float().pow(2).mean().sqrt(), on_step=True)
        self.log("reg/cond_l2", cond_reg.detach(), on_step=True)
        self.log("reg/scale", torch.tensor(scale, device=loss.device), on_step=True)
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0], on_step=True)
        return loss

    def _process_lyric_conditioning(self, lyrics_data: Dict[str, Any], target_length: int) -> Optional[torch.Tensor]:
        """Process lyric data into conditioning tensor matching latent dimensions."""
        try:
            # Extract lyric tokens and mask
            if isinstance(lyrics_data, dict):
                if "lyric_token_idx" in lyrics_data:
                    token_ids = lyrics_data["lyric_token_idx"]
                    token_mask = lyrics_data.get("lyric_mask", None)
                elif "lyric_token_id" in lyrics_data:
                    token_ids = lyrics_data["lyric_token_id"]
                    token_mask = lyrics_data.get("lyric_mask", None)
                else:
                    return None
            else:
                return None

            # Ensure tensors
            if not isinstance(token_ids, torch.Tensor):
                token_ids = torch.tensor(token_ids, device=self.device)
            if token_mask is not None and not isinstance(token_mask, torch.Tensor):
                token_mask = torch.tensor(token_mask, device=self.device)

            # Create lyric embedding (simplified approach)
            # In a full implementation, you'd use the ACE-Step lyric encoder
            batch_size = 1  # Assuming single item for now
            embed_dim = 256  # Match your latent dimensions

            # Simple lyric conditioning: average token embeddings and project to latent space
            if len(token_ids.shape) == 1:
                token_ids = token_ids.unsqueeze(0)  # Add batch dim

            # Create learnable lyric projection layer if it doesn't exist
            if not hasattr(self, 'lyric_projection'):
                self.lyric_projection = torch.nn.Linear(embed_dim, 8).to(self.device)  # Project to latent channels

            # Simple embedding lookup (would use proper ACE-Step lyric embeddings in full implementation)
            lyric_embed = torch.randn(token_ids.shape[0], token_ids.shape[1], embed_dim, device=self.device)

            # Project to latent dimensions
            lyric_projected = self.lyric_projection(lyric_embed)  # [B, seq_len, 8]

            # Average over sequence dimension and reshape to match latent spatial dims
            lyric_avg = lyric_projected.mean(dim=1)  # [B, 8]

            # Reshape to match x_in: [B, C, H, T]
            lyric_conditioning = lyric_avg.unsqueeze(-1).unsqueeze(-1)  # [B, 8, 1, 1]
            lyric_conditioning = lyric_conditioning.expand(-1, -1, 8, target_length)  # [B, 8, 8, T]

            return lyric_conditioning

        except Exception as e:
            print(f"⚠ Error processing lyric conditioning: {e}")
            return None

    def _detect_vocal_training(self) -> bool:
        """Detect if this training session should handle vocals based on manifest."""
        try:
            # Check manifest path for vocal indicators
            manifest_path = str(self.hparams.manifest_json).lower()
            if 'vocal' in manifest_path or 'voice' in manifest_path:
                return True

            # Check manifest content for vocal entries
            import json
            from pathlib import Path
            if Path(self.hparams.manifest_json).exists():
                with open(self.hparams.manifest_json, 'r') as f:
                    manifest_data = json.load(f)

                # Sample first few entries to check for vocal content
                sample_size = min(10, len(manifest_data))
                vocal_count = 0

                for entry in manifest_data[:sample_size]:
                    if isinstance(entry, dict):
                        # Check group field
                        if entry.get('group') == 'vocal':
                            vocal_count += 1
                        # Check audio path for vocal indicators
                        elif 'audio_path' in entry:
                            audio_path = str(entry['audio_path']).lower()
                            if any(vocal_term in audio_path for vocal_term in ['vox', 'vocal', 'voice', 'sing']):
                                vocal_count += 1

                # If more than 50% of sampled entries are vocal, enable vocal mode
                return vocal_count > (sample_size * 0.5)

            return False

        except Exception as e:
            print(f"⚠ Error detecting vocal training: {e}")
            return False

    def _extract_and_process_lyrics(self, audio_paths: List[str], target_length: int) -> Optional[torch.Tensor]:
        """Extract lyrics from audio and process into conditioning tensor."""
        try:
            if not self.lyric_processor:
                return None

            # Process first audio path (batch processing would handle multiple)
            audio_path = audio_paths[0] if isinstance(audio_paths, list) else audio_paths

            # Extract lyrics using the lyric processor
            lyric_data = self.lyric_processor.process_vocal_file(audio_path)

            if lyric_data and "lyric_token_idx" in lyric_data:
                return self._process_lyric_conditioning(lyric_data, target_length)

            return None

        except Exception as e:
            print(f"⚠ Error extracting lyrics: {e}")
            return None

    def _is_vocal_batch(self, batch) -> bool:
        """Check if current batch contains vocal tracks."""
        try:
            # Check group field in batch
            if "group_id" in batch:
                group_ids = batch["group_id"]
                vocal_id = self.group2id.get("vocal", -1)
                if vocal_id != -1:
                    return (group_ids == vocal_id).any().item()

            # Fallback: check audio paths
            if "audio_path" in batch:
                audio_paths = batch["audio_path"]
                if isinstance(audio_paths, (list, tuple)):
                    return any("vox" in str(path).lower() or "vocal" in str(path).lower() for path in audio_paths)
                else:
                    return "vox" in str(audio_paths).lower() or "vocal" in str(audio_paths).lower()

            return False
        except Exception:
            return False

    def _vocal_aware_pr_loss(self, pr_logits, pr_tgt, lyric_boundaries=None):
        """Compute vocal-aware piano roll loss with syllable boundary weighting and onset alignment."""
        # Standard BCE loss
        base_pr_loss = self._pr_bce_loss(pr_logits, pr_tgt)

        if lyric_boundaries is None:
            return base_pr_loss

        try:
            # Syllable boundary enhancement
            # Weight the loss more heavily at syllable boundaries where note changes are expected
            boundary_weight = 1.0 + lyric_boundaries.unsqueeze(1)  # [B, 1, T]

            # Compute per-timestep loss
            pr_probs = torch.sigmoid(pr_logits)  # [B, 128, T]
            pr_tgt_expanded = pr_tgt.expand_as(pr_probs)  # [B, 128, T]

            # BCE loss per timestep and note
            timestep_loss = F.binary_cross_entropy(pr_probs, pr_tgt_expanded, reduction='none')  # [B, 128, T]

            # Apply boundary weighting
            weighted_loss = timestep_loss * boundary_weight.expand_as(timestep_loss)

            # Average across notes and time, then batch
            vocal_pr_loss = weighted_loss.mean()

            # NEW: Add explicit onset alignment loss
            # Force predicted onsets to align with syllable boundaries
            pr_onset = self._compute_onset_curve(pr_probs)  # [B, T]

            # FIXED: Use correlation-based alignment instead of MSE
            # Normalize both curves to zero mean, unit variance
            onset_norm = (pr_onset - pr_onset.mean(dim=1, keepdim=True)) / (pr_onset.std(dim=1, keepdim=True) + 1e-8)
            bound_norm = (lyric_boundaries - lyric_boundaries.mean(dim=1, keepdim=True)) / (lyric_boundaries.std(dim=1, keepdim=True) + 1e-8)

            # Maximize cosine similarity (correlation) between onset curve and boundaries
            # Loss = 1 - correlation, so minimizing pushes correlation toward 1.0
            onset_alignment_loss = 1.0 - F.cosine_similarity(onset_norm, bound_norm, dim=1).mean()

            # Log onset alignment separately
            self.log("aux/lyric_onset_alignment", onset_alignment_loss.detach(), on_step=True)

            # Combine losses: weighted PR loss + onset alignment
            total_loss = vocal_pr_loss + 0.3 * onset_alignment_loss

            return total_loss

        except Exception as e:
            print(f"⚠ Error in vocal-aware PR loss: {e}")
            return base_pr_loss

    def _compute_onset_curve(self, pr_probs):
        """
        Compute onset curve from piano roll probabilities.

        Args:
            pr_probs: [B, 128, T] piano roll probabilities

        Returns:
            onset_curve: [B, T] onset strength at each timestep
        """
        # Detect note onsets: positive difference between consecutive frames
        # pr_probs[:, :, 1:] - pr_probs[:, :, :-1] gives change
        pr_diff = pr_probs[:, :, 1:] - pr_probs[:, :, :-1]  # [B, 128, T-1]

        # Only keep positive changes (onsets, not offsets)
        pr_onset = torch.clamp(pr_diff, min=0.0)  # [B, 128, T-1]

        # Sum across pitch dimension to get onset strength per timestep
        onset_curve = pr_onset.sum(dim=1)  # [B, T-1]

        # Pad to match original length
        onset_curve = F.pad(onset_curve, (1, 0))  # [B, T]

        # FIXED: Normalize to [0, 1] instead of sigmoid (which saturates)
        onset_curve = onset_curve / (onset_curve.max(dim=1, keepdim=True)[0] + 1e-8)

        return onset_curve




# NEW METHOD

    # def training_step(self, batch, batch_idx):
    #     x0 = batch["latents"].to(self.device)
    #     B = x0.shape[0]

    #     # --- build cond tokens ---
    #     tokens, mask = self.ctrl_enc(
    #         piano_roll=batch["conds"]["piano_roll"].to(self.device),
    #         amp=batch["conds"]["amp"].to(self.device),
    #         rframe=batch["conds"]["rframe"].to(self.device),
    #         rbend=batch["conds"]["rbend"].to(self.device),
    #         rbend_mask=batch["conds"]["rbend_mask"].to(self.device),
    #         encodec_tokens=batch["encodec_tokens"].to(self.device),
    #         group_id=batch["instrument"]["group_id"].to(self.device),
    #         subgroup_id=batch["instrument"]["subgroup_id"].to(self.device),
    #     )

    #     # --- sample t and σ ON THE SAME DEVICE ---
    #     timesteps_all = self.scheduler.timesteps.to(self.device)
    #     sigmas_all    = self.scheduler.sigmas.to(self.device)

    #     idx = torch.randint(0, timesteps_all.numel(), (B,), device=self.device)
    #     t   = timesteps_all[idx]         # [B]
    #     σ   = sigmas_all[idx]            # [B]

    #     # broadcast σ to x0 shape
    #     σ = σ.view(B, *([1] * (x0.ndim - 1)))

    #     # --- epsilon training objective ---
    #     noise   = torch.randn_like(x0)
    #     x_noisy = x0 + σ * noise

    #     eps_hat = self.transformers(
    #         hidden_states=x_noisy,
    #         timestep=t,
    #         encoder_hidden_states=tokens,
    #         encoder_attention_mask=mask,
    #     ).sample

    #     loss = F.mse_loss(eps_hat, noise)
    #     self.log("train/loss", loss, on_step=True, prog_bar=True)
    #     self.log("train/lr", self.lr_schedulers().get_last_lr()[0], on_step=True)
    #     return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Use the same batch as training for overfitting test
        loss = self.training_step(batch, batch_idx)
        self.log("val/loss", loss, on_step=True)
        
        # Calculate instrument classification accuracy
        tokens, mask = self.ctrl_enc(batch["conds"]["piano_roll"], batch["conds"]["rbend"], batch["conds"]["rframe"])
        inst_tok = tokens[:, 0, :]
        
        # Group classification accuracy
        group_logits = self.group_head(inst_tok)
        group_pred = group_logits.argmax(-1)
        group_tgt = batch["group_id"].long().to(group_logits.device)
        group_acc = (group_pred == group_tgt).float().mean()
        self.log("val/group_acc", group_acc, prog_bar=True)

        # Subgroup classification accuracy
        sub_logits = self.sub_head(inst_tok)
        sub_pred = sub_logits.argmax(-1)
        sub_tgt = batch["subgroup_id"].long().to(sub_logits.device)
        sub_acc = (sub_pred == sub_tgt).float().mean()
        self.log("val/subgroup_acc", sub_acc, prog_bar=True)
        
        return loss




 
def plot_loss_from_tfevents(tfevents_path, out_png_path):
    """Reads a TensorBoard tfevents file and plots the train/loss scalar."""
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing import event_accumulator

    print(f"Reading tfevents file: {tfevents_path}")
    if not os.path.exists(tfevents_path):
        print(f"Error: File not found at {tfevents_path}")
        return

    ea = event_accumulator.EventAccumulator(
        tfevents_path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    if 'train/loss' not in ea.Tags()['scalars']:
        print("Error: 'train/loss' tag not found in the tfevents file.")
        print(f"Available tags: {ea.Tags()['scalars']}")
        return

    loss_events = ea.Scalars('train/loss')
    steps = [e.step for e in loss_events]
    values = [e.value for e in loss_events]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, values, label='Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_png_path)
    print(f"Loss curve saved to: {out_png_path}")




def main(args):
    # parse overfit here so it's in-scope
    def _parse_overfit(v):
        if v is None:
            return None
        try:
            f = float(v)   # accepts "1", "1.0", 1, 0.1, etc.
        except ValueError:
            raise ValueError(f"--overfit_batches must be a number (got {v!r})")
        return int(f) if f >= 1 else f

    ov = _parse_overfit(args.overfit_batches)

    model = Pipeline(
        checkpoint_dir=args.checkpoint_dir,
        manifest_json=args.manifest_json,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        shift=args.shift,
        max_steps=args.max_steps,
        every_plot_step=args.every_plot_step,
        cond_cfg_drop_prob=args.cond_cfg_drop_prob,
        warmup_steps=args.warmup_steps,
        override_group=args.override_group,
        override_subgroup=args.override_subgroup,
        preview_steps=args.preview_steps,
        batch_size=args.batch_size,
        preview_index=args.preview_index,
        window_slow=args.window_slow,
        encodec_drop_prob=args.encodec_drop_prob,
        encodec_channel_drop_prob=args.encodec_channel_drop_prob,
        encodec_time_mask_prob=args.encodec_time_mask_prob,
        encodec_time_mask_max_frac=args.encodec_time_mask_max_frac, 
        train_from_scratch=args.train_from_scratch,
        inst_strength=args.inst_strength,
        film_strength=args.film_strength,
        channel_mod_strength=args.channel_mod_strength,
        pr_loss_weight=args.pr_loss_weight,
        pr_onset_boost=args.pr_onset_boost,
        use_ctrl_branch=args.use_ctrl_branch,
        freeze_base_for_ctrl=args.freeze_base_for_ctrl,
        partial_mask_prob=args.partial_mask_prob,
        control_scale=args.control_scale,
        use_lyric_conditioning=getattr(args, 'use_lyric_conditioning', True),
        lyric_conditioning_strength=getattr(args, 'lyric_conditioning_strength', 1.0),
        lyric_dropout_prob=getattr(args, 'lyric_dropout_prob', 0.1),
    )

    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(args.logger_dir, "checkpoints"), monitor=None, every_n_train_steps=args.every_n_train_steps, save_top_k=-1, save_last=False)
    logger_cb = TensorBoardLogger(version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.exp_name,
                                  save_dir=args.logger_dir)

    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy="ddp_find_unused_parameters_true" if int(args.devices) > 1 else "auto",
        max_epochs=-1,
        max_steps=args.max_steps,
        log_every_n_steps=1,
        logger=logger_cb,
        callbacks=[ckpt_cb],
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        val_check_interval=None,     # ← use this instead of overfit mode
        limit_val_batches=0,        # ← no val loop
        num_sanity_val_steps=0
    )

    if args.ckpt_path and args.weights_only:
        import torch
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)

        def _safe_replace(key, target_param):
            """Resize ckpt tensor to target shape by copying overlap and leaving the rest as-initialized."""
            if key not in sd:
                return
            old = sd[key]
            new = target_param.detach().clone()
            if old.shape != new.shape:
                # copy overlap (assume row-major for embeddings/linears)
                if old.ndim >= 2 and new.ndim >= 2:
                    r = min(old.shape[0], new.shape[0])
                    c = min(old.shape[1], new.shape[1])
                    new[:r, :c] = old[:r, :c]
                else:
                    n = min(old.numel(), new.numel())
                    new.view(-1)[:n] = old.view(-1)[:n]
                sd[key] = new
                print(f"[weights-only] resized {key}: ckpt {tuple(old.shape)} -> model {tuple(new.shape)}")

        # Patch known variable-size parts
        _safe_replace("ctrl_enc.subgroup_emb.weight",      model.ctrl_enc.subgroup_emb.weight)
        _safe_replace("ctrl_enc.group_emb.weight",         model.ctrl_enc.group_emb.weight)
        _safe_replace("group_head.weight",                 model.group_head.weight)
        _safe_replace("group_head.bias",                   model.group_head.bias)
        _safe_replace("sub_head.weight",                   model.sub_head.weight)
        _safe_replace("sub_head.bias",                     model.sub_head.bias)

        # Strip incompatible control-related weights from older checkpoints
        bad_prefixes = ("_ctrl_proj.", "ctrlnet.to_blocks.", "_ctrl_zero.", "_ctrl_gate", "ctrl_enc.mhubert_proj.")
        for k in list(sd.keys()):
            if k.startswith(bad_prefixes):
                sd.pop(k)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[weights-only] missing keys: {len(missing)} (ok if new modules). Example: {missing[:8]}...")
        if unexpected:
            print(f"[weights-only] unexpected keys: {len(unexpected)}. Example: {unexpected[:8]}...")

        # ---- Pretend-step + ramp/unfreeze catch-up (for --weights_only) ----
        prev_steps = int(getattr(args, "resume_steps", 0))
        if prev_steps > 0:
            print(f"[weights-only] applying pretend-step state: global_step={prev_steps}")
            # 1) Pretend we're at the old step so step-gated logic doesn't restart
            try:
                model.global_step = prev_steps  # PL will overwrite during fit, but hooks read this early
            except Exception as e:
                print(f"[weights-only] could not set model.global_step: {e}")

            # 2) If you want adapter/augment/outlier ramps effectively 'done'
            model.adapter_warmup_steps   = 1
            model.augment_ramp_steps    = 1
            model.outlier_disable_steps = 1  # enable outlier filter immediately

            # 3) Ensure unfreeze state matches late phase
            model._applied_unfreeze_2k   = True
            model._applied_unfreeze_6k   = True
            model._resume_unfreeze_done  = True

            # Optionally force fully-unfrozen now (safe if you changed nothing else)
            try:
                blocks = getattr(model.transformers, "transformer_blocks", [])
                model._unfreeze_more(last_k=len(blocks))
                print(f"[weights-only] forced unfreeze of last {len(blocks)} blocks")
            except Exception as e:
                print(f"[weights-only] unfreeze skip: {e}")

        # Prevent Lightning from attempting to restore opt/sched
        args.ckpt_path = None


    trainer.fit(model, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_nodes", type=int, default=1)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")


    ap.add_argument("--partial_mask_prob", type=float, default=0.3)
    ap.add_argument("--weights_only", action="store_true",
                help="Load only model weights from the checkpoint; ignore optimizer/scheduler state.")

    ap.add_argument("--inst_strength", type=float, default=3.0,
                    help="Multiplier for the global instrument token.")
    ap.add_argument("--film_strength", type=float, default=1.0,
                    help="Strength of FiLM modulation applied to timbre features.")
    ap.add_argument("--channel_mod_strength", type=float, default=1.0,
                    help="Strength of channel-wise timbre modulation before projection.")



    ap.add_argument("--control_scale", type=float, default=1.0,
                help="Global strength for ControlBranch residuals.")

    # Lyric conditioning arguments
    ap.add_argument("--use_lyric_conditioning", action="store_true", default=True,
                    help="Enable lyric conditioning for vocal tracks")
    ap.add_argument("--lyric_conditioning_strength", type=float, default=1.0,
                    help="Strength of lyric conditioning signal")
    ap.add_argument("--lyric_dropout_prob", type=float, default=0.1,
                    help="Probability of dropping lyric conditioning during training")

    ap.add_argument("--plot_only", type=str, default=None,
                    help="Path to a tfevents file to plot. If provided, skips training.")

    ap.add_argument("--encodec_drop_prob", type=float, default=0.00)
    ap.add_argument("--encodec_channel_drop_prob", type=float, default=0.0)
    ap.add_argument("--encodec_time_mask_prob", type=float, default=0.0)
    ap.add_argument("--encodec_time_mask_max_frac", type=float, default=0.0)


    ap.add_argument("--pr_loss_weight", type=float, default=0.6,
                help="Weight for PR BCE loss (0 disables).")
    ap.add_argument("--pr_onset_boost", type=float, default=1.0,
                    help="Extra weight on onset frames (0 = off).")

    ap.add_argument("--use_ctrl_branch", action="store_true",
                help="Enable ControlBranch1D residual guidance")
    ap.add_argument("--freeze_base_for_ctrl", action="store_true",
                    help="Freeze transformer when control branch is enabled")
    
    ap.add_argument("--resume_steps", type=int, default=0,
                    help="Pretend-training step to restore ramps/unfreeze when using --weights_only.")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--warmup_steps", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=-1)
    ap.add_argument("--max_steps", type=int, default=2000000)
    ap.add_argument("--every_n_train_steps", type=int, default=2000)
    ap.add_argument("--manifest_json", type=str, default="./vocal_training_manifest_yamnet_filtered.json")
    ap.add_argument("--exp_name", type=str, default="scratch_perf_transformer")
    ap.add_argument("--precision", type=str, default="32")
    ap.add_argument("--accumulate_grad_batches", type=int, default=4)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--logger_dir", type=str, default="./exps/logs/")
    ap.add_argument("--ckpt_path", type=str, default=None)
    ap.add_argument("--checkpoint_dir", type=str, required=True, help="Dir that contains ace_step_transformer/config.json + music_dcae_f8c8 + music_vocoder (can be a snapshots/<hash> dir)")
    ap.add_argument("--gradient_clip_val", type=float, default=1.0)
    ap.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    ap.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=1)
    ap.add_argument("--every_plot_step", type=int, default=2000)
    ap.add_argument("--val_check_interval", type=int, default=None)
    ap.add_argument("--cond_cfg_drop_prob", type=float, default=0.15)

    ap.add_argument("--override_group", type=str, default=None,
                help="Instrument group name or int id (e.g., 'strings' or 2)")
    ap.add_argument("--override_subgroup", type=str, default=None,
                    help="Instrument subgroup name or int id (e.g., 'violin' or 7)")

    ap.add_argument("--overfit_batches", type=str, default=None)
    ap.add_argument("--window_slow", type=int, default=2048)

    ap.add_argument("--preview_window_slow", type=int, default=None,
                    help="Window size for previews (in slow frames). If None, uses --window_slow. Use larger values (e.g. 2048) for longer preview clips.")


    ap.add_argument("--preview_steps", type=int, default=50,
                help="Denoising steps for preview sampling")

    ap.add_argument("--preview_seconds", type=int, default=30,
                help="Duration of preview clips in seconds (default: 30). Longer = more memory usage.")

    ap.add_argument("--preview_index", type=int, default=0,
            help="The index of the audio file in the manifest to use for previews.")

    ap.add_argument("--train_from_scratch", action="store_true",
                help="If set, trains a new model from scratch. Otherwise, fine-tunes the pre-trained ACE-Step model.")

 # for this test only



    args = ap.parse_args()

    if args.plot_only:
        output_path = os.path.join(os.path.dirname(args.plot_only), "loss_curve.png")
        plot_loss_from_tfevents(args.plot_only, output_path)
    else:
        main(args)
    
    # main(args)
