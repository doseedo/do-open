# ~/Data/trainer_performer.py
# Apache 2.0

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from datetime import datetime
import argparse, os, json, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from dataloader import APPROVED_GROUPS, APPROVED_SUBGROUPS
from typing import Optional, Union
import os
import torchaudio
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from contextlib import nullcontext

import torch.nn as nn
import inspect
import torch.nn.functional as F
import torch.nn.functional as F

from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers import DDPMScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor


from acestep.pipeline_ace_step import ACEStepTrainComponents
from dataloader import PerformerAIDataset, collate_latent_cond
from conditioning_encoder import PerformanceConditionEncoder
import torchaudio

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


# Adapter classes removed - using cross-attention conditioning only




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
        
    ):
        super().__init__()
        self.save_hyperparameters() 
         # --- 1) Scheduler (ACE FlowMatch Euler)
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=T, shift=shift)

        # Sanitize scheduler sigmas for stability
        with torch.no_grad():
            if hasattr(self.scheduler, "sigmas"):
                if not torch.isfinite(self.scheduler.sigmas).all():
                    print("[sched] non-finite sigmas detected; clamping")
                    self.scheduler.sigmas = torch.nan_to_num(self.scheduler.sigmas, nan=0.0, posinf=1e3, neginf=-1e3)

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
        # Old unfreezing flags removed - using new CA pivot approach 



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
        self.dcae.eval()

        self.dcae.requires_grad_(False)


        self.dcae_device = "cuda" if torch.cuda.is_available() else "cpu"


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

        # Freezing logic moved to freeze_backbone_for_ca_pivot() for consistency
        # Freeze unused heads only
        for m in [
            getattr(self.transformers, "lyric_embs", None),
            getattr(self.transformers, "lyric_encoder", None),
            getattr(self.transformers, "lyric_proj", None),
            *getattr(self.transformers, "projectors", []),
        ]:
            if m is not None:
                for p in m.parameters():
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
        
        # Cross-attention conditioning (no additive adapters needed)
        self.token_summary = TokenSummarizer(d_text)

        # Lightning will handle device placement automatically
        
        
        

        # Cross-attention training settings (no adapter warmup/regularization needed)  

        self.outlier_thr_lo = 0.30      # relaxed lower bound
        self.outlier_thr_hi = 0.40      # relaxed upper bound during early steps
        self.outlier_relax_steps = 5000 # ~first 3–5k steps
        self.augment_ramp_steps = 5000   

        # Freezing will be done after ctrl_enc is built

        self.ctrl_enc = PerformanceConditionEncoder(
            d_text=d_text,
            enc_channels=8,
            fast_per_slow=FAST_PER_SLOW,
            group_vocab=6,
            subgroup_vocab=self._infer_subgroup_vocab(manifest_json),
            inst_emb_dim=384,
            inst_strength=float(getattr(self.hparams, "inst_strength", 3.0)),
            film_strength=float(getattr(self.hparams, "film_strength", 1.0)),
            channel_mod_strength=float(getattr(self.hparams, "channel_mod_strength", 1.0)),
        )

        self.ctrl_enc.requires_grad_(False)
        self.ctrl_enc.eval()

        self._set_ctrl_parts_trainable(True)
        self.ctrl_enc.train()


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

        # Aux classification heads must exist before freezing logic touches them
        self.group_head = nn.Linear(d_text, len(self.group2id))
        self.sub_head   = nn.Linear(d_text, len(self.subgroup2id))

        # ---- ctrl_enc now gets the correct sizes ----
        self.ctrl_enc = PerformanceConditionEncoder(
            d_text=d_text,
            enc_channels=8,
            fast_per_slow=FAST_PER_SLOW,
            group_vocab=group_vocab,
            subgroup_vocab=subgroup_vocab,
            inst_emb_dim=384,
            inst_strength=float(getattr(self.hparams, "inst_strength", 3.0)),
            film_strength=float(getattr(self.hparams, "film_strength", 1.0)),
            channel_mod_strength=float(getattr(self.hparams, "channel_mod_strength", 1.0)),
        )
        self.ctrl_enc.requires_grad_(False)
        self.ctrl_enc.eval()   # keep deterministic while hunting NaNs
        self._set_ctrl_parts_trainable(True)
        # self.ctrl_enc.train()  # keep in eval mode for stability

        # Now freeze backbone for CA pivot (after ctrl_enc and heads exist)
        self.freeze_backbone_for_ca_pivot()

        def _to_id(s, table):
            if s is None: return None
            if isinstance(s, int): return s
            if isinstance(s, str) and s.isdigit(): return int(s)
            return table.get(s, None)

        # USE THE RAW CTOR ARGS (not self.hparams.*) so it also works even if you don't save hparams
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

    def _to_same_device_dtype_as(self, x: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
        p = next(module.parameters(), None)
        if p is None:
            return x
        return x.to(device=p.device, dtype=p.dtype)


    # Adapter gain scaling methods removed (using CA only)


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

    def _maybe_dropout_encodec_masked(self, enc: torch.Tensor, voiced_mask: torch.Tensor) -> torch.Tensor:
        # enc: [B, C_fast, T_fast], int (discrete codes) or float (latents)
        # voiced_mask: [B, T_fast], 1.0 where voiced, 0.0 where unvoiced
        if not self.training:
            return enc

        B, C, T = enc.shape
        out = enc.clone()

        # linear ramp (0->1) over self.augment_ramp_steps
        ramp = float(min(1.0, (int(self.global_step) + 1) / max(1, int(getattr(self, "augment_ramp_steps", 5000)))))

        eff_ch = float(self.encodec_channel_drop_prob) * ramp
        eff_tm = float(self.encodec_time_mask_prob) * ramp
        max_frac = float(self.encodec_time_mask_max_frac)

        # Whole-stream dropout: only apply on **voiced** frames (forces pitch from roll)
        if self.encodec_drop_prob > 0:
            if torch.rand((), device=enc.device) < self.encodec_drop_prob:
                # zero only where voiced
                out = out * (1.0 - voiced_mask.unsqueeze(1))  # zero channels where voiced=1
                # keep unvoiced EnCodec intact

        # Channel dropout (ramped), also only when voiced
        if eff_ch > 0:
            ch_drop = (torch.rand((B, C), device=enc.device) < eff_ch).float().unsqueeze(-1)  # [B,C,1]
            out = out * (1.0 - ch_drop * voiced_mask.unsqueeze(1))

        # Time masking (ramped probability), restrict masked spans to voiced indices
        if eff_tm > 0 and max_frac > 0:
            max_L = max(1, int(T * max_frac))
            for b in range(B):
                if torch.rand((), device=enc.device) < eff_tm:
                    # pick a voiced region to mask; fallback to anywhere if no voiced
                    voiced_idx = (voiced_mask[b] > 0.5).nonzero(as_tuple=False).flatten()
                    if voiced_idx.numel() > 0:
                        s = voiced_idx[torch.randint(0, voiced_idx.numel(), (1,), device=enc.device)].item()
                    else:
                        s = torch.randint(0, T, (1,), device=enc.device).item()
                    L = torch.randint(1, max_L + 1, (1,), device=enc.device).item()
                    e = min(T, s + L)
                    out[b, :, s:e] = 0
        return out




    def _has_latents(self, batch):
        return isinstance(batch, dict) and isinstance(batch.get("latents"), torch.Tensor) and batch["latents"].ndim == 4

    # helpers for CQT->roll BCE
    def _project_cqt_to_roll_bins(self, cqt_mag, fmin=32.7, bins=128):
        # cqt_mag: [B, F, T_audio]
        # Map each MIDI bin to its nearest CQT bin (coarse but cheap)
        B, F, T = cqt_mag.shape
        device = cqt_mag.device
        freqs = torch.logspace(start=torch.log10(torch.tensor(fmin, device=device)),
                               end=torch.log10(torch.tensor(fmin*(2**(bins/12)), device=device)),
                               steps=bins, device=device)
        # assuming CQT bins are log-spaced; build a nearest index LUT once per device shape if you want
        idx = torch.linspace(0, F-1, steps=bins, device=device).long()
        roll_hat = cqt_mag.index_select(1, idx)          # [B, bins, T]
        # normalize per-time
        roll_hat = roll_hat / (roll_hat.amax(dim=1, keepdim=True) + 1e-6)
        return roll_hat


    
# Replace _cqt_roll_bce method:
    def _cqt_roll_bce(self, x_latent_slice, roll_slice, sr=32000):
        """Cheap BCE between CQT(projection(wav(x_latent_slice))) and piano roll."""
        with torch.no_grad():
            device = x_latent_slice.device
            
            frames = int(roll_slice.shape[-1])
            hop_ratio = max(1, int((DCAE_HOP * sr) // DCAE_SR))
            audio_len = torch.tensor([frames * hop_ratio], device=device, dtype=torch.long)
            
            # Use safe decode
            sr_pred, wav = self._dcae_decode_safe(x_latent_slice, audio_len, sr)
            
            # Process wav (already on correct device)
            wav = wav.mean(dim=1, keepdim=True)  # Mono
            
            # Build CQT-like magnitude
            cqt_mag = self._cqt_like(wav, sr=sr, n_bins=84, fmin=32.7)  # [B,F,T]
            
            # Project to roll bins and match time
            bins = roll_slice.shape[1]
            proj = self._project_cqt_to_roll_bins(cqt_mag, fmin=32.7, bins=bins)
            proj = F.interpolate(proj, size=frames, mode="nearest")
            proj = proj / (proj.amax(dim=1, keepdim=True) + 1e-6)
            proj = proj.clamp_(1e-6, 1 - 1e-6)
            
            if int(self.global_step) < 10 and getattr(self.trainer, "is_global_zero", True):
                print(f"[roll_bce] proj mean={proj.mean().item():.4f} max={proj.max().item():.4f} | "
                    f"roll min={roll_slice.min().item():.4f} max={roll_slice.max().item():.4f}")
            
            target = (roll_slice > 0).float()
            voiced = target.any(dim=1, keepdim=True).float()
            weight = 0.5 + 0.5 * voiced
            return F.binary_cross_entropy(proj, target, weight=weight, reduction="mean")



    def _compute_roll_f1(self, x_latent_slice, roll_slice, sr=32000, threshold=0.5):
        """Compute framewise F1 vs roll metric on previews (cheap)"""
        try:
            with torch.no_grad():
                device = x_latent_slice.device
                audio_len = torch.tensor([roll_slice.shape[-1] * (DCAE_HOP * sr // DCAE_SR)], device=device)
                
                # Use safe decode
                sr_pred, wav = self._dcae_decode_safe(x_latent_slice, audio_len, sr)
                wav = wav.mean(dim=1, keepdim=True)  # Mono
                
                # CQT-like magnitude, then project to roll bins
                cqt_mag = self._cqt_like(wav, sr=sr, n_bins=84, fmin=32.7)
                proj = self._project_cqt_to_roll_bins(cqt_mag, fmin=32.7, bins=roll_slice.shape[1])
                proj = F.interpolate(proj, size=roll_slice.shape[-1], mode="nearest")
                
                pred_binary = (proj > threshold).float()
                true_binary = (roll_slice > 0).float()
                pred_any = pred_binary.any(dim=1).float()
                true_any = true_binary.any(dim=1).float()
                
                tp = (pred_any * true_any).sum()
                fp = (pred_any * (1 - true_any)).sum()
                fn = ((1 - pred_any) * true_any).sum()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                return {"f1": f1.item(), "precision": precision.item(), "recall": recall.item()}
        except Exception:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    def _latents_or_none(self, batch):
        if self._has_latents(batch):
            return batch["latents"]
        return None


    def on_fit_start(self):
        # one-item, deterministic preview set
        ds_prev = PerformerAIDataset(
            json_path=self.manifest_json,
            conditioning_dropout={"piano_roll":0.0, "amp":0.0, "rbend":0.0, "rframe":0.0},
            use_trim=True,
            pre_roll_seconds=1.0,
            post_roll_seconds=0.0,
            keep_untrimmed_prob=0.0,         # <- deterministic
            amp_activity_thr=0.06,
            require_all_core=True,
            collapse_sparse_subgroups_to_any=False,
            static_window=True,               # <- fixed window
            window_slow=getattr(self.hparams, "window_slow", 256),
            seed=0
        )
        idx = int(self.hparams.preview_index) % len(ds_prev)
        item = ds_prev[idx]
        self._preview_batch = collate_latent_cond([item])  # stays on CPU; preview fns .to(self.device)
        self._wrote_gt = False  # ensure we write GT for this fixed clip
        self.ctrl_enc.to(self.device)


        try:
            print("[sanity] scanning first 100 samples for NaN/Inf/extreme values...")
            ds_check = PerformerAIDataset(
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
                seed=0
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



    @torch.no_grad()
    def _save_gt_once(self, batch, sr_out=48000):
        if not getattr(self.trainer, "is_global_zero", True) or self._wrote_gt:
            return
        x0 = self._latents_or_none(batch)
        if x0 is None:
            logger.warning("[preview] no latents in preview batch; skipping GT save this time.")
            return
        B, _, _, T_slow = x0.shape
        audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
        
        audio_lengths = torch.tensor([audio_len], device=x0.device, dtype=torch.long)
        sr_pred, wav_pred = self._dcae_decode_safe(x0[:1], audio_lengths, sr_out)
        
        out_dir = f"{getattr(self.logger, 'log_dir', './logs')}/eval_results/step_{self.global_step}"
        os.makedirs(out_dir, exist_ok=True)
        torchaudio.save(f"{out_dir}/gt.wav", wav_pred[0].float().cpu(), sr_pred)
        self._wrote_gt = True


        

    def _scale_in(self, x, t_idx):
        # TEMP: bypass scheduler scaling to prove stability
        return x
        
        # # Guarded scaling option:
        # if hasattr(self.scheduler, "scale_model_input"):
        #     y = self.scheduler.scale_model_input(x, t_idx)
        #     return self._nan_sanitize(y, "latents_scaled", clamp=1e3)
        # return x
        
        # # Safe scaling option (if you want the scaling):
        # try:
        #     sigmas_all = self.scheduler.sigmas.to(device=x.device, dtype=torch.float32)
        #     sig = sigmas_all.index_select(0, t_idx.to(sigmas_all.device))
        #     sig = sig.clamp_min(1e-6)                                 # avoid div-by-zero
        #     # Example SD3-style: scale by (sig^2 + 1)^{-0.5} or whatever your scheduler uses
        #     scale = (sig**2 + 1.0).rsqrt().view(x.shape[0], *([1]*(x.ndim-1)))
        #     x_scaled = x * scale.to(x.dtype)
        #     return x_scaled
        # except Exception:
        #     return x
        


    # helper (drop anywhere inside Pipeline, e.g., under other helpers)
    def _preview_sigma0(self):
        if hasattr(self.scheduler, "sigmas"):
            # take the largest sigma in the schedule
            return float(self.scheduler.sigmas.max().item())
        return 1.0


    def _cqt_like(self, wav: torch.Tensor, sr: int, n_bins: int = 84, fmin: float = 32.7):
        """
        Returns a CQT-like magnitude tensor [B, F, T]. Tries:
        1) torchaudio.transforms.CQT (if available)
        2) librosa.cqt (CPU, numpy -> torch)
        3) torchaudio MelSpectrogram as a fallback (approximate)
        """
        import torch
        B, C, T = wav.shape
        device = wav.device

        # 1) torchaudio CQT if present
        CQT = getattr(__import__("torchaudio").transforms, "CQT", None)
        if CQT is not None:
            cqt = CQT(sr=sr, n_bins=n_bins, fmin=fmin).to(device)(wav)  # complex [B,F,T]
            return (cqt.real.pow(2) + cqt.imag.pow(2)).sqrt()

        # 2) librosa CQT (CPU)
        try:
            import librosa, numpy as np
            mags = []
            wav_cpu = wav.detach().cpu().squeeze(1).numpy()  # [B, T]
            for b in range(B):
                Cx = librosa.cqt(
                    wav_cpu[b], sr=sr, fmin=fmin, n_bins=n_bins, filter_scale=1.0, pad_mode="reflect"
                )  # [F,Tc] complex64
                mags.append(np.abs(Cx))
            mags = torch.from_numpy(np.stack(mags, axis=0))  # [B,F,Tc]
            return mags.to(device=device, dtype=wav.dtype)
        except Exception:
            pass

        # 3) Log-mel fallback (approximate constant-Q)
        import torchaudio
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=2048, hop_length=512, n_mels=n_bins, f_min=fmin, f_max=sr/2
        ).to(device)(wav.squeeze(1))  # [B, F, Tm]
        return mel.clamp_min(1e-8).sqrt().unsqueeze(1).squeeze(1)  # ensure [B,F,T]


    def _make_attn_mask(self, latents):
        # latents: [B, C, H, T_slow] -> mask [B, T_slow]
        B, _, _, T = latents.shape
        return torch.ones(B, T, device=latents.device, dtype=torch.float32)




    def _dcae_decode_safe(self, latents, audio_lengths, sr):
        """
        Safely decode latents with DCAE, handling device placement and dtype.
        Returns (sr, wav) tuple with wav on the same device as latents.
        """
        # Store original device
        orig_device = latents.device
        orig_dtype = latents.dtype
        
        # Move DCAE to the right device if needed (lazy migration)
        if next(self.dcae.parameters()).device != orig_device:
            self.dcae.to(orig_device)
        
        # Cast to DCAE's expected dtype (usually float32)
        with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision for DCAE
            latents_for_dcae = latents.to(dtype=torch.float32)
            sr_out, wav = self.dcae.decode(latents_for_dcae, audio_lengths=audio_lengths, sr=sr)
        
        # Ensure wav is tensor on correct device
        if isinstance(wav, (list, tuple)):
            wav = wav[0]
        if isinstance(wav, dict) and "audio" in wav:
            wav = wav["audio"]
        if not isinstance(wav, torch.Tensor):
            wav = torch.as_tensor(wav, device=orig_device, dtype=torch.float32)
        else:
            wav = wav.to(device=orig_device, dtype=torch.float32)
        
        # Ensure correct shape
        if wav.ndim == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.ndim == 2:
            wav = wav.unsqueeze(0)
        
        return sr_out, wav


    def _normalize_ca_inputs(self, tokens: torch.Tensor, token_mask: Optional[torch.Tensor]):
        """
        Ensures:
        • mask shape matches tokens [B, N]
        • mask is float {0,1}
        • global tokens (idx 0 and 1) are always unmasked
        • at least one token is unmasked per sample
        """
        assert tokens is not None and tokens.ndim == 3, "tokens must be [B,N,D]"
        B, N, _ = tokens.shape
        dev = tokens.device

        if token_mask is None:
            m = torch.ones(B, N, device=dev, dtype=torch.float32)  # all visible
        else:
            # force 2D [B,N]
            m = token_mask
            if m.ndim == 1:
                m = m.unsqueeze(-1)
            if m.ndim == 3:
                m = m.squeeze(-1)
            # clamp to {0,1} and cast
            m = (m > 0.5).to(torch.float32)

            # align length to N (no dropping of global tokens!)
            if m.shape[1] > N:
                m = m[:, :N]
            elif m.shape[1] < N:
                pad = torch.ones(B, N - m.shape[1], device=dev, dtype=torch.float32)  # default visible
                m = torch.cat([m, pad], dim=1)

        # guarantee globals are unmasked
        g = min(2, N)
        m[:, :g] = 1.0

        # guarantee ≥1 unmasked per sample
        bad = (m.sum(dim=1) <= 0.5)
        if bad.any():
            m[bad, 0] = 1.0

        return tokens, m



    
    def _call_transformer(self, latents, tokens, token_mask, timesteps, attn_mask, batch=None):
        """
        Robustly call the transformer no matter which API this build exposes:
        - explicit conditioning API: (prompt_embeds, piano_roll_cond, pitch_bend_cond, dynamics_cond, timbre_cond)
        - cross-attention API: (encoder_text_hidden_states + masks)
        Also fixes the cross-attention mask so global tokens are never dropped.
        """
        if attn_mask is None:
            attn_mask = self._make_attn_mask(latents)

        sig = inspect.signature(self.transformers.forward).parameters
        def has(*names): return all(n in sig for n in names)

        # -------- Branch 1: explicit conditioning API (no CA mask needed) --------
        if has("prompt_embeds","piano_roll_cond","pitch_bend_cond","dynamics_cond","timbre_cond"):
            assert batch is not None, "batch is required for explicit cond API"
            B, _, _, T_slow = latents.shape
            original_dtype = latents.dtype
            conds = self._prepare_explicit_conds(batch, T_slow)

            with torch.autocast("cuda", enabled=False):  # stay in fp32 inside model
                out = self.transformers(
                    hidden_states=latents.float(),
                    timestep=timesteps,
                    prompt_embeds=conds["prompt_embeds"].float(),
                    piano_roll_cond=conds["piano_roll_cond"].float(),
                    pitch_bend_cond=conds["pitch_bend_cond"].float(),
                    dynamics_cond=conds["dynamics_cond"].float(),
                    timbre_cond=conds["timbre_cond"].float(),
                )
            v_pred = out.sample if hasattr(out, "sample") else out
            v_pred = self._nan_sanitize(v_pred, "v_pred_pre_cast", clamp=1e3)
            return v_pred.to(original_dtype)

        # -------- Branch 2: cross-attention API (this is where the bug mattered) --------
        # Normalize tokens & mask so they match and globals are always visible.
        if tokens is not None:
            tokens, token_mask = self._normalize_ca_inputs(tokens, token_mask)

        original_dtype = latents.dtype
        kwargs = {}

        # latents
        if "hidden_states" in sig:  kwargs["hidden_states"] = latents.float()
        elif "x" in sig:            kwargs["x"] = latents.float()

        # time / sigma
        for k in ("timestep","timesteps","t","sigma","noise_sigma"):
            if k in sig:
                kwargs[k] = timesteps
                break

        # CA context
        for k in ("encoder_text_hidden_states","encoder_hidden_states","context","text_embeds"):
            if k in sig:
                kwargs[k] = None if tokens is None else tokens.float()
                break

        # CA mask (kept aligned to tokens length; float {0,1})
        for k in ("text_attention_mask","encoder_hidden_mask","encoder_attention_mask","context_mask"):
            if k in sig:
                if token_mask is None and tokens is not None:
                    # fabricate a full-ones mask if caller didn't provide one
                    token_mask = torch.ones(tokens.shape[0], tokens.shape[1], device=tokens.device, dtype=torch.float32)
                    token_mask[:, :min(2, tokens.shape[1])] = 1.0
                kwargs[k] = token_mask
                break

        # latent-time attention mask
        for k in ("attention_mask","attn_mask","mask"):
            if k in sig:
                kwargs[k] = attn_mask
                break

        # optional extras some builds expect
        if "speaker_embeds" in sig:
            B = tokens.shape[0] if tokens is not None else latents.shape[0]
            kwargs["speaker_embeds"] = torch.zeros(B, 512, device=latents.device, dtype=torch.float32)
        if "lyric_token_idx" in sig:
            B = tokens.shape[0] if tokens is not None else latents.shape[0]
            kwargs["lyric_token_idx"] = torch.zeros(B, 1, device=latents.device, dtype=torch.long)
        if "lyric_mask" in sig:
            B = tokens.shape[0] if tokens is not None else latents.shape[0]
            kwargs["lyric_mask"] = torch.zeros(B, 1, device=latents.device, dtype=torch.float32)

        # NaN guards
        def _check_nan(tag, t):
            if t is None or not torch.is_floating_point(t): return False
            bad = ~torch.isfinite(t)
            if bad.any():
                print(f"[nan-sentry] {tag}: {bad.sum().item()} non-finite at step {int(self.global_step)}")
                return True
            return False
        _check_nan("latents_in", self.first_not_none(kwargs.get("hidden_states"), kwargs.get("x")))
        ctx_in = self.first_not_none(
            kwargs.get("encoder_text_hidden_states"),
            kwargs.get("encoder_hidden_states"),
            kwargs.get("context"),
            kwargs.get("text_embeds"),
        )
        if ctx_in is not None:
            _check_nan("tokens_in", ctx_in)

        with torch.autocast("cuda", enabled=False):  # run model body in fp32
            out = self.transformers(**kwargs)

        v_pred = out.sample if hasattr(out, "sample") else out
        v_pred = self._nan_sanitize(v_pred, "v_pred_pre_cast", clamp=1e3)

        # emergency fallback (very rare)
        if _check_nan("v_pred_pre_cast", v_pred):
            try:
                v_pred = self._call_transformer_no_xattn(
                    latents=self.first_not_none(kwargs.get("hidden_states"), kwargs.get("x")),
                    t=self.first_not_none(kwargs.get("timestep"), kwargs.get("timesteps"), kwargs.get("t"))
                )
            except Exception:
                v_pred = torch.zeros_like(self.first_not_none(kwargs.get("hidden_states"), kwargs.get("x")))

        return v_pred.to(original_dtype)


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
        ds = PerformerAIDataset(
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
            seed=0
        )

        def _seed_worker(worker_id):
            import numpy as np, random, torch
            seed = torch.initial_seed() % 2**32
            np.random.seed(seed); random.seed(seed)
            info = torch.utils.data.get_worker_info()
            if hasattr(info, "dataset") and hasattr(info.dataset, "rng"):
                info.dataset.rng = np.random.default_rng(seed)

        return DataLoader(
            ds,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_latent_cond,
            batch_size=self.hparams.batch_size,
            worker_init_fn=_seed_worker,   # Fix deterministic randomness across workers
        )

    @torch.no_grad()
    def _preview_x0_direct_rf(self, batch, t_scalar=0.5, sr_out=48000):
        self.transformers.eval()
        with self._amp():
            x0 = self._latents_or_none(batch)
            if x0 is None:
                logger.warning("[preview] no latents; skipping.")
                return
            x0 = x0.to(self.device)                               # <<< move to model device
            B, _, _, T_slow = x0.shape

            tokens, mask = self.ctrl_enc(
                piano_roll=batch["conds"]["piano_roll"].to(self.device),
                amp=batch["conds"]["amp"].to(self.device),
                rframe=batch["conds"]["rframe"].to(self.device),
                rbend=batch["conds"]["rbend"].to(self.device),
                rbend_mask=batch["conds"]["rbend_mask"].to(self.device),
                encodec_tokens=batch["encodec_tokens"].to(self.device),
                group_id=batch["instrument"]["group_id"].to(self.device),
                subgroup_id=batch["instrument"]["subgroup_id"].to(self.device),
            )

            # --- ensure valid CA mask: at least one unmasked token (global token at idx 0)
            mask = mask.to(torch.float32)                 # model expects float mask often
            mask[..., 0] = 1.0                            # force global token visible
            # safety: if any sample is fully masked (sum==0), turn on first token
            bad = (mask.sum(dim=-1) <= 0.5)               # [B]
            if bad.any():
                mask[bad, 0] = 1.0

            t = torch.full((B,), float(t_scalar), device=self.device)
            z = torch.randn_like(x0)
            x_t = (1.0 - t.view(B,1,1,1)) * x0 + t.view(B,1,1,1) * z

            T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
            t_idx = (t * (T_train - 1)).long().clamp(0, T_train - 1)

            
            # Use cross-attention conditioning (matching training)
            x_in_scaled = self._scale_in(x_t, t_idx)
            
            # Fix mask-context mismatch for CA
            tokens_ca = tokens
            mask_ca = mask
            
            # Create and validate attention mask
            attn_mask = self._make_attn_mask(x_in_scaled)  # ones
            # if any code path could produce zeros everywhere, enforce at least one:
            if attn_mask.sum(dim=-1).min() <= 0:
                attn_mask[..., 0] = 1.0
            
            # Sanitize inputs before transformer call
            x_in_scaled = self._nan_sanitize(x_in_scaled, "latents_in")
            tokens_ca   = self._nan_sanitize(tokens_ca,   "tokens_in")
            mask_ca     = self._nan_sanitize(mask_ca,     "token_mask")
            t_idx       = self._nan_sanitize(t_idx,       "t_idx", clamp=1e6)

            # Optional stats for first few steps
            if int(self.global_step) <= 5 and self.trainer.is_global_zero:
                self._stats(x_in_scaled, "latents_in")
                self._stats(tokens_ca,   "tokens_in")

            # Use universal caller to handle both CA and explicit APIs
            v_pred = self._call_transformer(
                latents=x_in_scaled,
                tokens=tokens_ca,
                token_mask=mask_ca,
                timesteps=t_idx,
                attn_mask=attn_mask,
                batch=batch,  # needed for explicit-API path (_prepare_explicit_conds)
            )

            # Sanitize v_pred for robustness
            if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
                v_pred = torch.nan_to_num(v_pred, nan=0.0, posinf=1e4, neginf=-1e4)
                v_pred = v_pred.clamp_(-100, 100)

            x0_hat = x_t - t.view(B,1,1,1) * v_pred

            # decode short slice
            K = min(T_slow, int(6.0 * SLOW_HZ))
            x_slice = x0_hat[..., :K]
            audio_len = int(round(K * DCAE_HOP * (sr_out / DCAE_SR)))
            
            audio_lengths = torch.tensor([audio_len], device=x_slice.device, dtype=torch.long)
            sr_pred, wav_pred = self._dcae_decode_safe(x_slice[:1], audio_lengths, sr_out)
            
            out_dir = f"{self.logger.log_dir}/eval_results/step_{self.global_step}"
            os.makedirs(out_dir, exist_ok=True)
            if getattr(self.trainer, "is_global_zero", True):
                torchaudio.save(f"{out_dir}/x0_direct_rf_{self.global_step}.wav", 
                            wav_pred[0].float().cpu(), sr_pred)


    @torch.no_grad()
    def _preview_from_noisy_gt(self, batch, t0=0.8, steps=30, sr_out=48000):
        """Start near x0 (RF schedule), take Euler(-like) steps with the no-xattn path, then decode."""
        self.transformers.eval()
        with self._amp():
            x0 = self._latents_or_none(batch)
            if x0 is None:
                logger.warning("[preview] no latents; skipping.")
                return

            # work on model device
            x0 = x0.to(self.device)
            B, _, _, T_slow = x0.shape

            # build control tokens on model device
            tokens, mask = self.ctrl_enc(
                piano_roll=batch["conds"]["piano_roll"].to(self.device),
                amp=batch["conds"]["amp"].to(self.device),
                rframe=batch["conds"]["rframe"].to(self.device),
                rbend=batch["conds"]["rbend"].to(self.device),
                rbend_mask=batch["conds"]["rbend_mask"].to(self.device),
                encodec_tokens=batch["encodec_tokens"].to(self.device),
                group_id=batch["instrument"]["group_id"].to(self.device),
                subgroup_id=batch["instrument"]["subgroup_id"].to(self.device),
            )

            # --- ensure valid CA mask: at least one unmasked token (global token at idx 0)
            mask = mask.to(torch.float32)                 # model expects float mask often
            mask[..., 0] = 1.0                            # force global token visible
            # safety: if any sample is fully masked (sum==0), turn on first token
            bad = (mask.sum(dim=-1) <= 0.5)               # [B]
            if bad.any():
                mask[bad, 0] = 1.0

            # start near x0 instead of pure noise
            torch.manual_seed(0)
            z = torch.randn_like(x0)
            x = (1.0 - float(t0)) * x0 + float(t0) * z

            T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
            steps = max(1, int(steps))
            dt = float(t0) / float(steps)

            for i in range(steps, 0, -1):
                # continuous t in (0..t0], map to discrete index for model
                t_cont = torch.full((B,), i * dt, device=self.device)
                t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

                # Use cross-attention conditioning (matching training)
                x_in_scaled = self._scale_in(x, t_idx)
                
                # Use universal caller to handle both CA and explicit APIs
                v_pred = self._call_transformer(
                    latents=x_in_scaled,
                    tokens=tokens,
                    token_mask=mask,
                    timesteps=t_idx,
                    attn_mask=None,
                    batch=batch,  # needed for explicit-API path
                )

                
                x      = x - dt * v_pred

            # decode full time span
            audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
            audio_lengths = torch.tensor([audio_len], device=x[:1].device, dtype=torch.long)
            sr_pred, wav_pred = self._dcae_decode_safe(x[:1], audio_lengths, sr_out)
            out_dir = f"{getattr(self.logger, 'log_dir', './logs')}/eval_results/step_{self.global_step}"
            os.makedirs(out_dir, exist_ok=True)
            if getattr(self.trainer, "is_global_zero", True):
                torchaudio.save(f"{out_dir}/preview_from_gt_{self.global_step}.wav",
                                wav_pred[0].float().cpu(), sr_pred)

        self.transformers.train()



    @torch.no_grad()
    def _preview(self, batch, steps=30, sr_out=48000, tag="preview"):
        """Full-from-noise preview using RF-style updates and no-xattn path, then decode."""
        self.transformers.eval()
        with self._amp():
            x0 = self._latents_or_none(batch)
            if x0 is None:
                logger.warning("[preview] no latents; skipping.")
                return

            # work on model device
            x0 = x0.to(self.device)
            B, _, _, T_slow = x0.shape

            # controls on device
            tokens, mask = self.ctrl_enc(
                piano_roll=batch["conds"]["piano_roll"].to(self.device),
                amp=batch["conds"]["amp"].to(self.device),
                rframe=batch["conds"]["rframe"].to(self.device),
                rbend=batch["conds"]["rbend"].to(self.device),
                rbend_mask=batch["conds"]["rbend_mask"].to(self.device),
                encodec_tokens=batch["encodec_tokens"].to(self.device),
                group_id=batch["instrument"]["group_id"].to(self.device),
                subgroup_id=batch["instrument"]["subgroup_id"].to(self.device),
            )

            # --- ensure valid CA mask: at least one unmasked token (global token at idx 0)
            mask = mask.to(torch.float32)                 # model expects float mask often
            mask[..., 0] = 1.0                            # force global token visible
            # safety: if any sample is fully masked (sum==0), turn on first token
            bad = (mask.sum(dim=-1) <= 0.5)               # [B]
            if bad.any():
                mask[bad, 0] = 1.0

            # start from pure noise
            torch.manual_seed(0)
            x = torch.randn_like(x0)

            T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
            steps = max(1, int(steps))
            dt = 1.0 / float(steps)

            for i in range(steps, 0, -1):
                t_cont = torch.full((B,), i * dt, device=self.device)
                t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

                # Use cross-attention conditioning (matching training)
                x_in_scaled = self._scale_in(x, t_idx)
                
                # Use universal caller to handle both CA and explicit APIs
                v_pred = self._call_transformer(
                    latents=x_in_scaled,
                    tokens=tokens,
                    token_mask=mask,
                    timesteps=t_idx,
                    attn_mask=None,
                    batch=batch,  # needed for explicit-API path
                )

               
                x      = x - dt * v_pred

            # decode full time span
            audio_len_out = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
            x_for_dcae = self._match_mod_dtype(x[:1], self.dcae)
            audio_lengths = torch.tensor([audio_len_out], device=x_for_dcae.device, dtype=torch.long)
            sr_pred, wav_pred = self.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)

            out_dir = f"{getattr(self.logger, 'log_dir', './logs')}/eval_results/step_{self.global_step}"
            os.makedirs(out_dir, exist_ok=True)
            if getattr(self.trainer, "is_global_zero", True):
                torchaudio.save(f"{out_dir}/{tag}_{self.global_step}.wav",
                                wav_pred[0].float().cpu(), sr_pred)

        self.transformers.train()



    @torch.no_grad()
    def _preview_euler_20(self, batch, sr_out=48000):
        """Euler preview with 20 steps."""
        return self._preview(batch, steps=20, sr_out=sr_out, tag="euler_20")

    @torch.no_grad()
    def _preview_euler_40(self, batch, sr_out=48000):
        """Euler preview with 40 steps."""
        return self._preview(batch, steps=40, sr_out=sr_out, tag="euler_40") 

    def on_train_batch_start(self, batch, batch_idx):
        # ensure GT save still runs
        if self._preview_batch is not None:
            self._save_gt_once(self._preview_batch)

        # catch-up unfreeze after resume
        # Old resume unfreezing logic removed - using new CA pivot approach

        # Update encoder phase ramps
        gs = int(self.global_step)

        # PR extra boost ramps in fast (first 1k)
        pr_boost = min(1.0, (gs + 1) / 1000.0)
        self.ctrl_enc.phase_pr_boost = torch.tensor(pr_boost, device=self.device)

        # Timbre suppression relaxes after 10k (1.0 -> 0.2 over 5k)
        relax = 0.0 if gs < 10000 else min(1.0, (gs - 10000) / 5000.0)
        self.ctrl_enc.phase_voiced_suppress = torch.tensor(1.0 - 0.8 * relax, device=self.device)


    # Old freezing logic removed - using centralized CA pivot approach

        # after:
        fixed = self._preview_batch if self._has_latents(self._preview_batch) else batch
        if self._preview_batch is not None and not self._has_latents(self._preview_batch):
            # avoid spamming errors on future steps
            self._preview_batch = None

        preview_now = (self.global_step in (0,1,2,5,10,20)) or (
            ((self.global_step + 1) % max(1, getattr(self.hparams, "every_plot_step", 999999))) == 0
        )
        
        if preview_now:
            try:
                self._preview_x0_direct_rf(fixed, t_scalar=0.05, sr_out=32000)
                self._preview_from_noisy_gt(fixed, t0=0.2, steps=min(20, self.hparams.preview_steps), sr_out=32000)
                if self.global_step in (0, 5, 10, 20, 50, 100):
                    # Generate both 20-step and 40-step Euler previews from SAME sample as GT
                    self._preview_euler_20(fixed, sr_out=32000)
                    self._preview_euler_40(fixed, sr_out=32000)
            except Exception as e:
                print(f"[preview] skipped due to error: {e}")

        # Generate full-resolution Euler previews every plot step
        if (self.global_step + 1) % max(1, getattr(self.hparams, "every_plot_step", 999999)) == 0:
            try:
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

    def _prepare_explicit_conds(self, batch, T_slow):
        """Adapter to build explicit conds for pretrained ACE transformer API"""
        B = batch["latents"].shape[0]
        dev = batch["latents"].device
        dtype = batch["latents"].dtype

        pr = batch["conds"]["piano_roll"]                          # [B,128,T_slow]
        amp = batch["conds"]["amp"]                                # [B,T_slow]
        rb  = batch["conds"]["rbend"] * batch["conds"]["rbend_mask"]  # [B,T_slow]

        # Shapes ACE expects
        piano_roll_cond   = pr                                     # [B,128,T_slow]
        dynamics_cond     = amp.unsqueeze(1)                       # [B,1,T_slow]
        pitch_bend_cond   = rb.unsqueeze(1)                        # [B,1,T_slow]

        # timbre on the fast grid; resize to ~fast_per_slow * T_slow and 8 channels
        Kc = int(round(float(self.ctrl_enc.fast_per_slow) * T_slow))
        timbre_cond = self._ensure_timbre_cond(
            batch["encodec_tokens"], Kc=Kc, num_embedders=8
        )                                                          # [B,8,Kc]

        # prompt embeds: 1 token, correct dim
        d_text = getattr(self.transformers.config, "text_embedding_dim", 768)
        prompt_embeds = torch.zeros(B, 1, d_text, device=dev, dtype=dtype)  # [B,1,D]

        return {
            "prompt_embeds":     prompt_embeds.to(dtype=dtype),
            "piano_roll_cond":   piano_roll_cond.to(device=dev, dtype=dtype),
            "pitch_bend_cond":   pitch_bend_cond.to(device=dev, dtype=dtype),
            "dynamics_cond":     dynamics_cond.to(device=dev, dtype=dtype),
            "timbre_cond":       timbre_cond.to(device=dev, dtype=dtype),
        }

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

    def freeze_backbone_for_ca_pivot(self):
        """Freeze most of the backbone for stable cross-attention pivot training"""
        if getattr(self.hparams, "train_from_scratch", False):
            logger.info("Scratch mode — skipping backbone freeze.")
            return

        logger.info("Freezing backbone for CA pivot training")
        
        # Keep these transformer components trainable
        keep = ('attn2', 'norm2', 'timestep', 'final_layer', 'proj_in', 'cond_proj', 'cond_ln', 'cond_gain')
        for name, param in self.transformers.named_parameters():
            param.requires_grad = any(k in name for k in keep)
                
        # DO NOT change ctrl_enc here — it's handled by _set_ctrl_parts_trainable
        # Keep aux heads trainable (guarded)
        if hasattr(self, "group_head"):
            for param in self.group_head.parameters():
                param.requires_grad = True
        if hasattr(self, "sub_head"):
            for param in self.sub_head.parameters():
                param.requires_grad = True
            
        logger.info("Backbone frozen - only CA components and aux heads trainable")

    def unfreeze_progressive(self, step: int):
        """Progressively unfreeze components based on training step"""
        if step == 10000 and not hasattr(self, '_unfroze_10k'):
            logger.info("Step 10k: Unfreezing self-attention layers")
            for name, param in self.transformers.named_parameters():
                if 'attn1' in name or 'norm1' in name:  # self-attention
                    param.requires_grad = True
            self._unfroze_10k = True
            
        elif step == 20000 and not hasattr(self, '_unfroze_20k'):
            logger.info("Step 20k: Unfreezing FFN layers")  
            for name, param in self.transformers.named_parameters():
                if 'ffn' in name or 'norm3' in name:  # FFN layers
                    param.requires_grad = True
            self._unfroze_20k = True

        # ----- Optim -----
  

    def configure_optimizers(self):
        head_names = ("final_layer", "proj_in", "genre_embedder", "timestep_embedder", "t_block")
        head_params, block_params = [], []
        for n, p in self.transformers.named_parameters():
            (head_params if any(h in n for h in head_names) else block_params).append(p)

        # include token_summary (adapters removed for CA pivot)
        extra_heads = list(self.token_summary.parameters())

        # include aux classification heads
        extra_heads += list(self.group_head.parameters()) + list(self.sub_head.parameters())

        # include ONLY the parts of ctrl_enc you set trainable
        ctrl_trainable = []
        for name in ["group_emb","subgroup_emb","inst_fuse","timbre_mod_scale","timbre_mod_bias",
                    "film_scale","film_bias","timbre_global"]:
            m = getattr(self.ctrl_enc, name, None)
            if m is not None:
                ctrl_trainable += list(m.parameters())
        extra_heads += ctrl_trainable

        head_params.extend(extra_heads)

        # Special handling for cond_gain: higher LR, no weight decay
        gain_params = [p for n, p in self.transformers.named_parameters() if n.endswith("cond_gain")]
        
        groups = []
        if gain_params:
            groups.append({
                "params": gain_params,
                "lr": float(self.hparams.learning_rate) * 2.0,
                "weight_decay": 0.0
            })
        
        groups.extend([
            {"params": head_params,  "lr": float(self.hparams.learning_rate) * 2.0},  # e.g. 2e-5 if base is 1e-5
            {"params": block_params, "lr": float(self.hparams.learning_rate)},        # e.g. 1e-5
        ])

        # Force torch.optim.AdamW (avoid bitsandbytes for bf16 stability)
        # try:
        #     import bitsandbytes as bnb
        #     opt = bnb.optim.PagedAdamW8bit(
        #         groups, betas=(0.9, 0.999), weight_decay=float(self.hparams.weight_decay), eps=1e-8
        #     )
        # except Exception:
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

    def first_not_none(self, *xs):
        """Helper to safely get first non-None value, avoiding tensor 'or' chains"""
        for x in xs:
            if x is not None:
                return x
        return None

    def _nan_sanitize(self, x, name, clamp=1e4):
        if x is None:
            return None

        # Integers: cannot have NaN/Inf; just return as-is
        if not torch.is_floating_point(x):
            return x

        bad = ~torch.isfinite(x)
        if bad.any():
            print(f"[nan-sentry] {name}: found {bad.sum().item()} non-finite values at step {int(self.global_step)}")
            x = torch.nan_to_num(x, nan=0.0, posinf=clamp, neginf=-clamp)

        # soft clamp to keep bf16/attn stable
        return x.clamp_(-clamp, clamp)



    def _stats(self, x, name):
        """Quick tensor statistics for debugging"""
        try:
            mx = float(x.abs().max().item())
            mn = float(x.min().item())
            mxv = float(x.max().item())
            print(f"[stats] {name}: min={mn:.3e} max={mxv:.3e} |abs|max={mx:.3e}")
        except Exception:
            pass

    def _all_finite(self, *xs):
        """Check if all tensors have finite values"""
        return all(x is None or torch.isfinite(x).all().item() for x in xs)

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
            original_dtype = latents.dtype
            conds = self._prepare_explicit_conds(batch, T_slow)

            def try_call(pr, timb):
                # Force FP32 inside model to prevent NaNs
                with torch.autocast("cuda", enabled=False):
                    return self.transformers(
                        hidden_states=latents.float(),
                        timestep=timesteps,
                        prompt_embeds=conds["prompt_embeds"].float(),
                        piano_roll_cond=pr.float(),
                        pitch_bend_cond=conds["pitch_bend_cond"].float(),
                        dynamics_cond=conds["dynamics_cond"].float(),
                        timbre_cond=timb.float(),
                    )

            # try canonical layout first; if shape mismatch, flip time/channel dims
            try:
                out = try_call(conds["piano_roll_cond"], conds["timbre_cond"])
            except RuntimeError:
                pr = conds["piano_roll_cond"].transpose(1,2) if conds["piano_roll_cond"].dim()==3 else conds["piano_roll_cond"]
                timb = conds["timbre_cond"].transpose(1,2)   if conds["timbre_cond"].dim()==3 else conds["timbre_cond"]
                out = try_call(pr, timb)

            v_pred = out.sample if hasattr(out, "sample") else out
            v_pred = self._nan_sanitize(v_pred, "v_pred_pre_cast", clamp=1e3)
            return v_pred.to(original_dtype)  # cast back to original dtype

        # Branch 2: cross-attention API (your original control-token path)
        original_dtype = latents.dtype
        kwargs = {}
        if "hidden_states" in sig:  kwargs["hidden_states"] = latents.float()
        elif "x" in sig:            kwargs["x"] = latents.float()

        for k in ("timestep","timesteps","t","sigma","noise_sigma"):
            if k in sig:
                kwargs[k] = timesteps
                break

        for k in ("encoder_text_hidden_states","encoder_hidden_states","context","text_embeds"):
            if k in sig:
                kwargs[k] = tokens.float() if tokens is not None else None
                break

        for k in ("text_attention_mask","encoder_hidden_mask","encoder_attention_mask","context_mask"):
            if k in sig:
                # Safely handle token_mask with proper dtype and shape
                if token_mask is not None:
                    tm = token_mask
                    # clamp to {0,1} and cast to float32 (common ACE convention)
                    tm = (tm > 0.5).to(torch.float32)
                    if tm.ndim == 1:          # [B] -> [B,1]
                        tm = tm.unsqueeze(-1)
                    kwargs[k] = tm
                else:
                    kwargs[k] = token_mask  # keep None if None
                break

        for k in ("attention_mask","attn_mask","mask"):
            if k in sig:
                kwargs[k] = attn_mask
                break

        if "speaker_embeds" in sig:
            kwargs["speaker_embeds"] = torch.zeros(tokens.shape[0], 512, device=latents.device, dtype=torch.float32)
        if "lyric_token_idx" in sig:
            kwargs["lyric_token_idx"] = torch.zeros(tokens.shape[0], 1, device=latents.device, dtype=torch.long)
        if "lyric_mask" in sig:
            kwargs["lyric_mask"] = torch.zeros(tokens.shape[0], 1, device=latents.device, dtype=torch.long)

        # NaN sentry - check inputs before transformer call
        def _check_nan(tag, t):
            if not torch.isfinite(t).all():
                nbad = (~torch.isfinite(t)).sum().item()
                print(f"[nan-sentry] {tag}: found {nbad} non-finite values at step {int(self.global_step)}")
                return True
            return False
        
        # Check key inputs
        _check_nan("latents_in", self.first_not_none(kwargs.get("hidden_states"), kwargs.get("x")))
        ctx_in = self.first_not_none(
            kwargs.get("encoder_text_hidden_states", None),
            kwargs.get("encoder_hidden_states", None),
            kwargs.get("context", None),
            kwargs.get("text_embeds", None),
        )
        if ctx_in is not None: 
            _check_nan("tokens_in", ctx_in)

        # Force FP32 inside model to prevent NaNs
        with torch.autocast("cuda", enabled=False):
            out = self.transformers(**kwargs)
        
        v_pred = out.sample if hasattr(out, "sample") else out
        v_pred = self._nan_sanitize(v_pred, "v_pred_pre_cast", clamp=1e3)
        
        # Check output and provide fallback if needed
        if _check_nan("v_pred_pre_cast", v_pred):
            # last-resort fallback: disable CA for this batch so training can proceed
            try:
                v_pred = self._call_transformer_no_xattn(
                    latents=self.first_not_none(kwargs.get("hidden_states"), kwargs.get("x")),
                    t=self.first_not_none(kwargs.get("timestep"), kwargs.get("timesteps"), kwargs.get("t"))
                )
            except Exception:
                v_pred = torch.zeros_like(self.first_not_none(kwargs.get("hidden_states"), kwargs.get("x")))
        
        return v_pred.to(original_dtype)  # cast back to original dtype



# OG ACE STEP DIFFUSION

    def training_step(self, batch, batch_idx):
        # Progressive unfreezing for CA pivot training
        self.unfreeze_progressive(self.global_step)
        
        # move to device
        def to_device(x, d):
            if isinstance(x, torch.Tensor): return x.to(d)
            if isinstance(x, dict): return {k: to_device(v,d) for k,v in x.items()}
            if isinstance(x, list): return [to_device(i,d) for i in x]
            return x
        batch = to_device(batch, self.device)

        # Early abort of a poisoned batch
        if not self._all_finite(batch["latents"], batch["conds"]["piano_roll"], batch["encodec_tokens"]):
            print(f"[nan-sentry] bad batch at global_step={int(self.global_step)}; skipping")
            return torch.zeros((), device=self.device, requires_grad=True)

        x0 = batch["latents"]; B = x0.shape[0]

        # Sanitize conditioning inputs
        batch["conds"]["amp"] = self._nan_sanitize(batch["conds"]["amp"], "amp_in", clamp=10.0)
        batch["conds"]["rbend"] = self._nan_sanitize(batch["conds"]["rbend"], "rbend_in", clamp=4.0)
        batch["conds"]["rframe"] = self._nan_sanitize(batch["conds"]["rframe"], "rframe_in", clamp=1e3)
        et = batch["encodec_tokens"]
        if torch.is_floating_point(et):
            batch["encodec_tokens"] = self._nan_sanitize(et, "encodec_tokens_in", clamp=8.0)
        else:
            # make sure they are Long (expected by embedding/gather paths)
            batch["encodec_tokens"] = et.long()
        # controls
        enc_tok = batch["encodec_tokens"]
        pr_slow = batch["conds"]["piano_roll"]                        # [B,128,T_slow]
        voiced_slow = (pr_slow > 0).any(dim=1).float()                # [B,T_slow]
        B, C_fast, T_fast = batch["encodec_tokens"].shape
        voiced_fast = F.interpolate(voiced_slow.unsqueeze(1), size=T_fast, mode="nearest").squeeze(1)  # [B,T_fast]
        unvoiced_fast = (voiced_fast < 0.5).bool()  # [B,T_fast]

        # Ramp 'swap_p' from 0 -> 0.10 after 5k steps
        swap_cap = 0.10
        swap_ramp = min(1.0, max(0.0, (int(self.global_step) - 5000) / 3000.0))
        swap_p = swap_cap * swap_ramp
        if self.training and swap_p > 0 and torch.rand(()) < swap_p:
            with torch.no_grad():
                gid = batch["instrument"]["group_id"]
                perm = torch.randperm(gid.shape[0], device=gid.device)
                same = (gid == gid[perm])
                if same.any():
                    tmp = enc_tok.clone()
                    tmp[same][:, :, unvoiced_fast[same]] = enc_tok[perm][same][:, :, unvoiced_fast[same]]
                    enc_tok = tmp

        enc_tok = self._maybe_dropout_encodec_masked(enc_tok, voiced_fast)
        ramp = float(min(1.0, (int(self.global_step) + 1) / max(1, int(getattr(self, "augment_ramp_steps", 5000)))))
        self.log("aug/ramp", torch.tensor(ramp, device=x0.device), on_step=True)


        # tokens for your existing TemporalCondAdapter
        tokens, mask = self.ctrl_enc(
            piano_roll=pr_slow, 
            amp=batch["conds"]["amp"], 
            rframe=batch["conds"]["rframe"],
            rbend=batch["conds"]["rbend"], 
            rbend_mask=batch["conds"]["rbend_mask"],
            encodec_tokens=enc_tok, 
            group_id=batch["instrument"]["group_id"],
            subgroup_id=batch["instrument"]["subgroup_id"],
        )

        # --- ensure valid CA mask: at least one unmasked token (global token at idx 0)
        mask = mask.to(torch.float32)                 # model expects float mask often
        mask[..., 0] = 1.0                            # force global token visible
        # safety: if any sample is fully masked (sum==0), turn on first token
        bad = (mask.sum(dim=-1) <= 0.5)               # [B]
        if bad.any():
            mask[bad, 0] = 1.0

        tokens = self._nan_sanitize(tokens, "tokens_in", clamp=100.0)
        
        # Ensure tokens are in the same dtype as x0 for mixed precision compatibility
        tokens = tokens.to(dtype=x0.dtype)

        inst_tok = tokens[:, 0, :]   # [B, D]
        group_logits = self.group_head(inst_tok)
        sub_logits   = self.sub_head(inst_tok)

        group_tgt = batch["instrument"]["group_id"].long().to(group_logits.device)
        sub_tgt   = batch["instrument"]["subgroup_id"].long().to(sub_logits.device)

        aux_w = 0.1  # small, just a nudge; can tune 0.05–0.2
        aux_loss = F.cross_entropy(group_logits, group_tgt) + F.cross_entropy(sub_logits, sub_tgt)

       # ----- RF objective with τ -----
        T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        
        # Noise curriculum for τ
        gs = int(self.global_step)
        if gs < 5000:
            # favor low noise (Beta(2,5)) in first 5k steps
            u = torch.distributions.Beta(concentration1=2.0, concentration0=5.0).sample((B,)).to(x0.device)
            tau_f32 = u.clamp_(0.02, 0.40)
        else:
            tau_f32 = torch.rand(B, device=x0.device).clamp_(1e-3, 0.95)
        t_idx   = (tau_f32 * (T_train - 1)).to(torch.long)              # [B]
        sigma   = tau_f32.to(x0.dtype).view(B, *([1] * (x0.ndim - 1)))  # [B,1,1,1]
        z       = torch.randn_like(x0)
        x_noisy = (1.0 - sigma) * x0 + sigma * z

        # Use cross-attention conditioning (no scaling needed)
        tokens = tokens.to(dtype=x0.dtype)
        
        x_in_scaled = self._scale_in(x_noisy, t_idx)
        
        # Call transformer with cross-attention conditioning
        # Fix mask-context mismatch: transformer may drop global tokens internally
        tokens_ca = tokens  # [B, 2+T, D] - keep full tokens
        mask_ca = mask      # [B, 2+T] - keep full mask initially
        
        # Use universal caller to handle both CA and explicit APIs
        v_pred = self._call_transformer(
            latents=x_in_scaled,
            tokens=tokens_ca,
            token_mask=mask_ca,
            timesteps=t_idx,
            attn_mask=None,
            batch=batch,  # needed for explicit-API path
        )

        x0_hat = x_noisy - sigma * v_pred
        
        # ---- Outlier diagnostics & safe skip (keep scheduler/step) ----
        # --- Outlier filtering (robust) ---
        B = x0.shape[0]

        # always log the raw per-example recon loss
        recon_per_ex = (x0_hat - x0).pow(2).flatten(1).mean(dim=1)  # [B]
        self.log("train/recon_mean", recon_per_ex.mean().detach(), on_step=True)

        # Conditions to DISABLE dropping:
        disable = (B < int(getattr(self, "outlier_min_batch", 4))) or \
                (int(self.global_step) < int(getattr(self, "outlier_disable_steps", 2000)))

        if disable:
            keep_mask = torch.ones_like(recon_per_ex)
            OUTLIER_THR = float("inf")
        else:
            # quantile threshold on this micro-batch
            q = float(getattr(self, "outlier_quantile", 0.90))
            q = min(max(q, 0.50), 0.99)  # clamp
            thr = torch.quantile(recon_per_ex.detach().float(), q)
            OUTLIER_THR = float(thr.item())
            keep_mask = (recon_per_ex <= thr).float()
            keep_mask = keep_mask.to(dtype=torch.float32)

        kept = int(keep_mask.sum().item()); dropped = int(keep_mask.numel() - kept)

        # weighted recon loss; avoid div-by-zero
        recon_loss = (recon_per_ex * keep_mask).sum() / keep_mask.clamp_min(1).sum()

        # Tiny CQT→roll BCE (throttled, but forced early so we can see it)
        aux_roll = torch.zeros((), device=x0.device)
        force_early = int(self.global_step) < 1000
        do_roll = force_early or (torch.rand((), device=x0.device) < 0.2)



        if do_roll:
            self.ctrl_enc.dbg_roll_calls.add_(1)
            T_slow = x0.shape[-1]
            K = min(T_slow, int(2.0 * SLOW_HZ))
            x_slice = x0[:, :, :, :K]
            roll_slice = batch["conds"]["piano_roll"][..., :K]
            
            if (roll_slice > 0).any():
                try:
                    # Now this stays on GPU - no more device transfers
                    aux_roll = self._cqt_roll_bce(x_slice[:1].detach(), roll_slice[:1].detach(), sr=32000)
                    self.ctrl_enc.dbg_roll_ok.add_(1)
                    
                    # F1 metrics also stay on GPU
                    f1 = self._compute_roll_f1(x_slice[:1].detach(), roll_slice[:1].detach(), sr=32000)
                    self.log("metrics/roll_f1", torch.tensor(f1["f1"], device=x0.device), on_step=True)
                    self.log("metrics/roll_precision", torch.tensor(f1["precision"], device=x0.device), on_step=True)
                    self.log("metrics/roll_recall", torch.tensor(f1["recall"], device=x0.device), on_step=True)
                except Exception as e:
                    self.ctrl_enc.dbg_roll_errors.add_(1)
                    if getattr(self.trainer, "is_global_zero", True):
                        print(f"[aux/roll_bce] exception at step {int(self.global_step)}: {e}")
                    aux_roll = torch.zeros((), device=x0.device)
            else:
                self.ctrl_enc.dbg_roll_empty.add_(1)
                self.log("aux/roll_empty_frame", torch.tensor(1.0, device=x0.device), on_step=True)



        # Always log diagnostics so you can see if the branch fired on rank-0
        self.log("aux/roll_calls",  self.ctrl_enc.dbg_roll_calls.float(),  on_step=True)
        self.log("aux/roll_ok",     self.ctrl_enc.dbg_roll_ok.float(),     on_step=True)
        self.log("aux/roll_empty",  self.ctrl_enc.dbg_roll_empty.float(),  on_step=True)
        self.log("aux/roll_errors", self.ctrl_enc.dbg_roll_errors.float(), on_step=True)


        roll_w = 0.10  # you can temporarily bump to 0.30 to make the effect audible
        loss = recon_loss + (aux_w * aux_loss) + (roll_w * aux_roll)
        self.log("aux/group_ce", aux_loss.detach(), on_step=True)
        self.log("aux/roll_bce", aux_roll.detach(), on_step=True)



        # Adapter regularization logging removed (using CA only)

        self.log("train/outlier_thr", torch.tensor(OUTLIER_THR, device=loss.device), on_step=True)
        self.log("train/outlier_kept", torch.tensor(kept, device=loss.device, dtype=torch.float32), on_step=True)
        self.log("train/outlier_dropped", torch.tensor(dropped, device=loss.device, dtype=torch.float32), on_step=True)

        # Adapter conditioning logging removed (using CA only)

        # optional: spike log only when dropping is enabled
        if (not disable) and dropped > 0:
            try:
                spike_log = os.path.join(getattr(self.logger, "log_dir", "."), "spike_samples.txt")
                with open(spike_log, "a") as f:
                    file_hint = "unknown"
                    meta = batch.get("meta") if isinstance(batch, dict) else None
                    if isinstance(meta, dict):
                        file_hint = meta.get("audio_path") or meta.get("path") or "unknown"
                    elif isinstance(meta, (list, tuple)) and meta and isinstance(meta[0], dict):
                        file_hint = meta[0].get("audio_path") or meta[0].get("path") or "unknown"
                    f.write(f"{int(self.global_step)},{recon_per_ex.max().item():.6f},{file_hint}\n")
            except Exception:
                pass



        v_target = z - x0
        v_loss   = F.mse_loss(v_pred, v_target)  # diag only




        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("dbg/cos_vpred_vtgt", F.cosine_similarity(v_pred.flatten(1), v_target.flatten(1)).mean(), on_step=True)
        self.log("dbg/|v_pred|", v_pred.float().pow(2).mean().sqrt(), on_step=True)
        self.log("dbg/|v_tgt|",  v_target.float().pow(2).mean().sqrt(), on_step=True)
        # Conditioning regularization logging removed (using CA only)
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0], on_step=True)
        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Use the same batch as training for overfitting test
        loss = self.training_step(batch, batch_idx)
        self.log("val/loss", loss, on_step=True)
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
        channel_mod_strength=args.channel_mod_strength
    )

    ckpt_cb = ModelCheckpoint(monitor=None, every_n_train_steps=args.every_n_train_steps, save_top_k=-1, save_last=True)
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



        if args.ckpt_path and args.weights_only:
            import torch
            ckpt = torch.load(args.ckpt_path, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt)

            def _resize_like(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                """Copy the overlapping region from src into a clone of target."""
                new = target.detach().clone()
                if src.ndim == target.ndim == 2:
                    r = min(src.size(0), target.size(0))
                    c = min(src.size(1), target.size(1))
                    new[:r, :c] = src[:r, :c]
                else:
                    n = min(src.numel(), target.numel())
                    new.view(-1)[:n] = src.view(-1)[:n]
                return new

            cur = model.state_dict()
            # list the keys that can change across scripts
            keys_to_resize = [
                "ctrl_enc.sclr_proj.0.weight",
                "ctrl_enc.sclr_proj.0.bias",
                # add any other changed heads here if they pop up:
                # "ctrl_enc.something.weight", "ctrl_enc.something.bias",
                "group_head.weight", "group_head.bias",
                "sub_head.weight",  "sub_head.bias",
            ]

            for k in keys_to_resize:
                if k in sd and k in cur and sd[k].shape != cur[k].shape:
                    print(f"[weights-only] resize {k}: {tuple(sd[k].shape)} -> {tuple(cur[k].shape)}")
                    sd[k] = _resize_like(sd[k], cur[k])

            # Optionally, drop any still-mismatched leftovers
            for k in list(sd.keys()):
                if k in cur and sd[k].shape != cur[k].shape:
                    print(f"[weights-only] drop mismatched {k}: {tuple(sd[k].shape)} vs {tuple(cur[k].shape)}")
                    del sd[k]


        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[weights-only] missing keys: {len(missing)} (ok if new modules). Example: {missing[:8]}...")
        if unexpected:
            print(f"[weights-only] unexpected keys: {len(unexpected)}. Example: {unexpected[:8]}...")

        # Prevent Lightning from attempting to restore opt/sched
        args.ckpt_path = None


    trainer.fit(model, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_nodes", type=int, default=1)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")

    ap.add_argument("--weights_only", action="store_true",
                help="Load only model weights from the checkpoint; ignore optimizer/scheduler state.")

    ap.add_argument("--inst_strength", type=float, default=3.0,
                    help="Multiplier for the global instrument token.")
    ap.add_argument("--film_strength", type=float, default=1.0,
                    help="Strength of FiLM modulation applied to timbre features.")
    ap.add_argument("--channel_mod_strength", type=float, default=1.0,
                    help="Strength of channel-wise timbre modulation before projection.")


    ap.add_argument("--plot_only", type=str, default=None, 
                    help="Path to a tfevents file to plot. If provided, skips training.")

    ap.add_argument("--encodec_drop_prob", type=float, default=0.00)
    ap.add_argument("--encodec_channel_drop_prob", type=float, default=0.0)
    ap.add_argument("--encodec_time_mask_prob", type=float, default=0.0)
    ap.add_argument("--encodec_time_mask_max_frac", type=float, default=0.0)



    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--warmup_steps", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=-1)
    ap.add_argument("--max_steps", type=int, default=2000000)
    ap.add_argument("--every_n_train_steps", type=int, default=2000)
    ap.add_argument("--manifest_json", type=str, default="./final_training_manifest_final.json")
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


    ap.add_argument("--preview_steps", type=int, default=50,
                help="Denoising steps for preview sampling")

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

