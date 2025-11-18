# ~/Data/trainer_performer_backup.py
# Apache 2.0

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from datetime import datetime
import argparse, os, json, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
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


class ControlBranch1D(nn.Module):
    def __init__(self, d_in=128, hidden=256, out_channels_per_block=(512,512,512,512), with_importance=False):
        super().__init__()
        # Per-time mapping M (d_in → hidden)
        self.mapper = nn.Conv1d(d_in, hidden, kernel_size=1)
        self.act    = nn.SiLU()
        self.norm   = nn.GroupNorm(1, hidden)  # LN over channels
        self.with_importance = with_importance
        if with_importance:
            self.imp = nn.Conv1d(hidden, 1, kernel_size=1)

        self.to_blocks = nn.ModuleList([nn.Conv1d(hidden, oc, 1) for oc in out_channels_per_block])
        for c in self.to_blocks:
            nn.init.zeros_(c.weight); 
            if c.bias is not None: nn.init.zeros_(c.bias)

    def forward(self, pr_128t: torch.Tensor, T_out_list: list[int]):
        # pr_128t: [B, 129, T]  # 128 PR + 1 AMP
        h = self.act(self.norm(self.mapper(pr_128t)))  # [B, hidden, T]
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
        use_ctrl_branch: bool = False, freeze_base_for_ctrl: bool = False
        
    ):
        super().__init__()
        self.save_hyperparameters() 
        self.partial_mask_prob = float(partial_mask_prob)
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

        # --- ControlBranch (PR+AMP) ---
        if self.use_ctrl_branch:
            blocks = getattr(self.transformers, "transformer_blocks", [])
            tail_blocks = blocks[-4:] if len(blocks) >= 4 else blocks

            # infer latent channel count once, up-front
            C_latent = self._infer_latent_channels()
            n_inject = max(1, len(tail_blocks))
            oc_per = [C_latent] * n_inject  # ensure residual C matches latent C at each injected block

            # d_in = 128 (PR) + 1 (AMP)
            self.ctrlnet = ControlBranch1D(
                d_in=129,
                hidden=256,
                out_channels_per_block=oc_per,
                with_importance=True
            ).to(self.device)

            # zero-conv per injection (initialized to zero) + learnable gate
            self._ctrl_zero = nn.ModuleList([nn.Conv1d(C_latent, C_latent, 1) for _ in tail_blocks])
            for z in self._ctrl_zero:
                nn.init.zeros_(z.weight)
                if z.bias is not None:
                    nn.init.zeros_(z.bias)
            self._ctrl_gate = nn.Parameter(torch.zeros(len(tail_blocks)))  # trainable per-injection

            # storage for per-block residuals and hooks
            self._ctrl_residuals = [None] * n_inject
            self._ctrl_handles = []
            
            # Initialize _ctrl_proj as empty - will be populated dynamically or from checkpoint
            self._ctrl_proj = nn.ModuleList()

            def make_inject(idx):
                def hook(module, inp, out):
                    ctrl = getattr(self, "_ctrl_residuals", None)
                    ctrl = None if (ctrl is None or idx >= len(ctrl)) else ctrl[idx]
                    if ctrl is None:
                        return out

                    def add_to(x, add, idx):
                        # x: [B,C,H,T] or [B,C,T]; add: [B,C,T]
                        add = add.to(dtype=x.dtype, device=x.device)

                        # 3D time align first (linear is only valid for 3D here)
                        if add.dim() == 3 and add.shape[-1] != x.shape[-1]:
                            add = F.interpolate(add, size=x.shape[-1], mode="linear", align_corners=False)

                        # reduce H if needed; we want [B,C_res,T]
                        add_1d = add.mean(dim=2) if add.dim() == 4 else add  # [B,C_res,T]

                        # --- channel align (e.g., 8 -> 512) with a lazily-created 1x1 conv (zero-init) ---
                        C_in, C_out = add_1d.shape[1], x.shape[1]
                        if not hasattr(self, "_ctrl_proj"):
                            import torch.nn as nn
                            self._ctrl_proj = nn.ModuleList([nn.Identity() for _ in range(len(self._ctrl_gate))])
                        proj = self._ctrl_proj[idx]
                        import torch.nn as nn
                        needs_new = (not isinstance(proj, nn.Conv1d)) or \
                                    (getattr(proj, "in_channels", None) != C_in) or \
                                    (getattr(proj, "out_channels", None) != C_out)
                        if needs_new:
                            conv = nn.Conv1d(C_in, C_out, 1)
                            nn.init.zeros_(conv.weight)
                            if conv.bias is not None:
                                nn.init.zeros_(conv.bias)
                            conv = conv.to(x.device, dtype=x.dtype)
                            self._ctrl_proj[idx] = conv

                        add_1d = self._ctrl_proj[idx](add_1d)                # [B,C_out,T]
                        add_1d = self._ctrl_gate[idx].tanh() * add_1d        # gate

                        # lift back if needed
                        if x.dim() == 4:
                            add = add_1d.unsqueeze(2).expand(-1, -1, x.shape[2], -1)
                        else:
                            add = add_1d
                        return x + add

                    if hasattr(out, "sample"):
                        out.sample = add_to(out.sample, ctrl, idx)
                        return out
                    if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                        return (add_to(out[0], ctrl, idx), *out[1:])
                    if torch.is_tensor(out):
                        return add_to(out, ctrl, idx)
                    return out
                return hook

            for i, blk in enumerate(tail_blocks):
                self._ctrl_handles.append(blk.register_forward_hook(make_inject(i)))

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
        # make sure subgroup vocab covers APPROVED_SUBGROUPS or what’s in the manifest
        subgroup_vocab = max(len(self.subgroup2id), self._infer_subgroup_vocab(manifest_json))

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
        self.ctrl_enc.eval()
        self._set_ctrl_parts_trainable(True)
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
        
        # Initialize pitch->height bank for pitch lock compatibility  
        H_base = 16  # Match checkpoint shape [16, 128]
        self.pitch2h_bank = torch.nn.Parameter(0.01 * torch.randn(H_base, 128))

                

        

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


    def on_after_backward(self) -> None:
        # Safe to clear per-step state once recomputation+backward are done
        self._ctrl_residuals = None



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
        
        # Handle incompatible keys by removing them from state_dict
        # This allows the model to load with current architecture
        incompatible_prefixes = ["_ctrl_proj.", "pr_head."]
        incompatible_keys = ["pitch2h_bank"]
        
        keys_to_remove = []
        for key in sd.keys():
            # Remove keys that start with incompatible prefixes
            if any(key.startswith(prefix) for prefix in incompatible_prefixes):
                keys_to_remove.append(key)
            # Remove exact incompatible keys (but keep if they exist in current model)
            elif key in incompatible_keys and not hasattr(self, key.split('.')[0]):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del sd[key]
            
        if keys_to_remove:
            print(f"[resume] removed incompatible keys: {keys_to_remove}")
            
        # If _ctrl_proj existed in the checkpoint, reconstruct it with correct shapes
        ctrl_proj_keys = [k for k in list(checkpoint.get("state_dict", {}).keys()) if k.startswith("_ctrl_proj.") and k.endswith(".weight")]
        if ctrl_proj_keys and hasattr(self, "use_ctrl_branch") and self.use_ctrl_branch:
            print(f"[resume] found {len(ctrl_proj_keys)} _ctrl_proj layers in checkpoint - will reconstruct after model init")
            # Store for later reconstruction in setup or first forward pass
            self._ctrl_proj_shapes = {}
            for key in ctrl_proj_keys:
                w = checkpoint["state_dict"].get(key)
                if w is not None:
                    layer_idx = key.split('.')[1]  # _ctrl_proj.0.weight -> '0'
                    self._ctrl_proj_shapes[int(layer_idx)] = (w.shape[1], w.shape[0])  # (in_ch, out_ch)
                    
        # Handle pr_head reconstruction if needed
        pr_head_keys = [k for k in list(checkpoint.get("state_dict", {}).keys()) if k.startswith("pr_head.")]
        if pr_head_keys:
            print(f"[resume] found pr_head in checkpoint - will reconstruct when needed")
        
        # Handle pitch2h_bank shape mismatch - resize to match checkpoint
        if "pitch2h_bank" in sd and hasattr(self, "pitch2h_bank"):
            ckpt_shape = sd["pitch2h_bank"].shape
            current_shape = self.pitch2h_bank.shape
            if ckpt_shape != current_shape:
                print(f"[resume] resizing pitch2h_bank from {current_shape} to {ckpt_shape}")
                # Reinitialize with checkpoint shape
                self.pitch2h_bank = torch.nn.Parameter(0.01 * torch.randn(*ckpt_shape))

    def _reconstruct_ctrl_proj_from_checkpoint(self):
        """Reconstruct _ctrl_proj from stored checkpoint shapes if available"""
        if not hasattr(self, "_ctrl_proj_shapes") or not hasattr(self, "_ctrl_proj"):
            return
        
        print(f"[resume] reconstructing _ctrl_proj from checkpoint with {len(self._ctrl_proj_shapes)} layers")
        
        # Ensure _ctrl_proj has enough slots
        max_idx = max(self._ctrl_proj_shapes.keys()) if self._ctrl_proj_shapes else -1
        while len(self._ctrl_proj) <= max_idx:
            self._ctrl_proj.append(None)
        
        # Create conv layers from stored shapes
        for idx, (in_ch, out_ch) in self._ctrl_proj_shapes.items():
            conv = nn.Conv1d(in_ch, out_ch, 1)
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
            conv = conv.to(device=self.device)
            self._ctrl_proj[idx] = conv
            print(f"[resume] reconstructed _ctrl_proj[{idx}]: {in_ch} -> {out_ch}")
        
        # Clean up temporary storage
        delattr(self, "_ctrl_proj_shapes")



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

    def _maybe_dropout_encodec(self, enc: torch.Tensor) -> torch.Tensor:
        # enc: [B, C_fast, T_fast], int (discrete codes) or float (latents)
        if not self.training:
            return enc  # no dropout in eval/preview

        B, C, T = enc.shape
        device = enc.device
        out = enc.clone()

        # ---- NEW: linear ramp for early steps ----
        # goes 0.0 -> 1.0 over self.augment_ramp_steps
        ramp = float(min(1.0, (int(self.global_step) + 1) / max(1, int(getattr(self, "augment_ramp_steps", 5000)))))

        eff_channel_drop_prob = float(self.encodec_channel_drop_prob) * ramp
        eff_time_mask_prob    = float(self.encodec_time_mask_prob) * ramp
        eff_time_mask_max_frac= float(self.encodec_time_mask_max_frac)  # keep as-is (length of mask)

        # Whole-stream dropout (rare; keep constant)
        if self.encodec_drop_prob > 0 and torch.rand((), device=device) < self.encodec_drop_prob:
            out.zero_()
            return out

        # Channel dropout (ramped)
        
        if eff_channel_drop_prob > 0:
            ch_drop = torch.rand((B, C), device=device) < eff_channel_drop_prob  # True = drop
            # broadcast to time and zero in-place without dtype change
            out.masked_fill_(ch_drop.unsqueeze(-1), 0)

        # In the time masking loop, also use masked_fill_:




        # Time masking (ramped probability)
        if eff_time_mask_prob > 0 and eff_time_mask_max_frac > 0:
            max_L = max(1, int(T * eff_time_mask_max_frac))
            for b in range(B):
                if torch.rand((), device=device) < eff_time_mask_prob:
                    L = torch.randint(1, max_L + 1, (1,), device=device).item()
                    s = torch.randint(0, max(1, T - L + 1), (1,), device=device).item()
                    out[b, :, s:s+L] = 0

        return out



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
        self.cond_adapter.to(self.device)
        self.token_summary.to(self.device)
        if hasattr(self, "ctrlnet"):
            self.ctrlnet.to(self.device)


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
        
        # Reconstruct control projection layers from checkpoint if available
        if hasattr(self, "use_ctrl_branch") and self.use_ctrl_branch:
            self._reconstruct_ctrl_proj_from_checkpoint()


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


            # ---- ControlBranch residuals: PR + AMP ----
            if self.use_ctrl_branch:
                pr_128 = batch["conds"]["piano_roll"].to(x_t.device, dtype=x_t.dtype)
                amp_1t = batch["conds"]["amp"].to(x_t.device, dtype=x_t.dtype).unsqueeze(1)
                if amp_1t.shape[-1] != pr_128.shape[-1]:
                    amp_1t = F.interpolate(amp_1t, size=pr_128.shape[-1], mode="nearest")
                ctrl_in = torch.cat([pr_128, amp_1t], dim=1)

                res_list = self.ctrlnet(ctrl_in, T_out_list=[x_t.shape[-1]] * len(self.ctrlnet.to_blocks))
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
        self.transformers.eval()
        with self._amp():
            x0 = self._latents_or_none(batch)
            if x0 is None:
                logger.warning("[preview] no latents; skipping.")
                return

            x0 = x0.to(self.device)
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

            torch.manual_seed(0)
            z = torch.randn_like(x0)
            x = (1.0 - float(t0)) * x0 + float(t0) * z

            T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
            steps = max(1, int(steps))
            dt = float(t0) / float(steps)

            tokens_adapt = self._match_mod_dtype(tokens, self.cond_adapter)

            # ---- ControlBranch residuals: PR + AMP (constant over loop) ----
            if self.use_ctrl_branch:
                pr_128 = batch["conds"]["piano_roll"].to(x.device, dtype=x.dtype)
                amp_1t = batch["conds"]["amp"].to(x.device, dtype=x.dtype).unsqueeze(1)
                if amp_1t.shape[-1] != pr_128.shape[-1]:
                    amp_1t = F.interpolate(amp_1t, size=pr_128.shape[-1], mode="nearest")
                ctrl_in = torch.cat([pr_128, amp_1t], dim=1)

                res_list = self.ctrlnet(ctrl_in, T_out_list=[x.shape[-1]] * len(self.ctrlnet.to_blocks))
                scale = float(getattr(self.hparams, "control_scale", 1.0))
                self._ctrl_residuals = [r * scale for r in res_list]


            for i in range(steps, 0, -1):
                t_cont = torch.full((B,), i * dt, device=self.device)
                t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

                cond_patch = self.cond_adapter(tokens_adapt, T_out=x.shape[-1], scale=self._adapter_gain_scale())
                cond_patch = cond_patch.to(device=x.device, dtype=x.dtype)

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
        self.transformers.eval()
        with self._amp():
            x0 = self._latents_or_none(batch)
            if x0 is None:
                logger.warning("[preview] no latents; skipping.")
                return

            x0 = x0.to(self.device)
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

            torch.manual_seed(0)
            x = torch.randn_like(x0)

            T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
            steps = max(1, int(steps))
            dt = 1.0 / float(steps)

            tokens_adapt = self._match_mod_dtype(tokens, self.cond_adapter)


            # ---- ControlBranch residuals: PR + AMP (constant over loop) ----
            if self.use_ctrl_branch:
                pr_128 = batch["conds"]["piano_roll"].to(x.device, dtype=x.dtype)
                amp_1t = batch["conds"]["amp"].to(x.device, dtype=x.dtype).unsqueeze(1)
                if amp_1t.shape[-1] != pr_128.shape[-1]:
                    amp_1t = F.interpolate(amp_1t, size=pr_128.shape[-1], mode="nearest")
                ctrl_in = torch.cat([pr_128, amp_1t], dim=1)

                res_list = self.ctrlnet(ctrl_in, T_out_list=[x.shape[-1]] * len(self.ctrlnet.to_blocks))
                scale = float(getattr(self.hparams, "control_scale", 1.0))
                self._ctrl_residuals = [r * scale for r in res_list]


            for i in range(steps, 0, -1):
                t_cont = torch.full((B,), i * dt, device=self.device)
                t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

                cond_patch = self.cond_adapter(tokens_adapt, T_out=x.shape[-1], scale=self._adapter_gain_scale())
                cond_patch = cond_patch.to(device=x.device, dtype=x.dtype)

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


        # optional: full-from-noise preview less frequently
        if (self.global_step + 1) % max(1, getattr(self.hparams, "every_plot_step", 999999)) == 0:
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
  

    def configure_optimizers(self):
        head_names = ("final_layer", "proj_in", "genre_embedder", "timestep_embedder", "t_block")

        head_params, block_params = [], []
        for n, p in self.transformers.named_parameters():
            if not p.requires_grad:
                continue
            (head_params if any(h in n for h in head_names) else block_params).append(p)

        # always-optimized small modules (keep these)
        extra_heads = []
        extra_heads += list(self.cond_adapter.parameters())
        extra_heads += list(self.token_summary.parameters())
        extra_heads += list(self.group_head.parameters()) + list(self.sub_head.parameters())

        # only selected ctrl_enc pieces (keep these exactly as before)
        for name in ["group_emb","subgroup_emb","inst_fuse","timbre_mod_scale","timbre_mod_bias",
                    "film_scale","film_bias","timbre_global"]:
            m = getattr(self.ctrl_enc, name, None)
            if m is not None:
                extra_heads += list(m.parameters())
        
        # include the pitch->height bank if it exists
        if hasattr(self, "pitch2h_bank") and isinstance(self.pitch2h_bank, torch.nn.Parameter):
            extra_heads.append(self.pitch2h_bank)

        # --- IMPORTANT ROLLBACK: do NOT optimize any ControlBranch params ---
        # (No ctrlnet, no _ctrl_zero, no _ctrl_gate in the optimizer)
        # if hasattr(self, "ctrlnet"):
        #     extra_heads += list(self.ctrlnet.parameters())
        # if hasattr(self, "_ctrl_zero"):
        #     extra_heads += list(self._ctrl_zero.parameters())
        # if hasattr(self, "_ctrl_gate"):
        #     extra_heads.append(self._ctrl_gate)
        # --------------------------------------------------------------------

        # add to head group and dedupe
        head_params.extend(extra_heads)
        head_params = list({id(p): p for p in head_params}.values())
        block_params = list({id(p): p for p in block_params}.values())

        groups = [
            {"params": head_params,  "lr": float(self.hparams.learning_rate) * 2.0},
            {"params": block_params, "lr": float(self.hparams.learning_rate)},
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
                    gid  = batch["instrument"]["group_id"]
                    perm = torch.randperm(gid.shape[0], device=gid.device)
                    same = (gid == gid[perm])
                    if same.any():
                        tmp = enc_tok.clone()
                        tmp[same] = enc_tok[perm][same]
                        enc_tok = tmp

        enc_tok = self._maybe_dropout_encodec(enc_tok)
        ramp = float(min(1.0, (int(self.global_step) + 1) / max(1, int(getattr(self, "augment_ramp_steps", 5000)))))
        self.log("aug/ramp", torch.tensor(ramp, device=x0.device), on_step=True)

        tokens, mask = self.ctrl_enc(
            piano_roll=batch["conds"]["piano_roll"],
            amp=batch["conds"]["amp"],
            rframe=batch["conds"]["rframe"],
            rbend=batch["conds"]["rbend"],
            rbend_mask=batch["conds"]["rbend_mask"],
            encodec_tokens=enc_tok,
            group_id=batch["instrument"]["group_id"],
            subgroup_id=batch["instrument"]["subgroup_id"],
        )
        if torch.isnan(tokens).any():
            tokens = torch.nan_to_num(tokens)
        tokens = tokens.to(dtype=x0.dtype)  # mixed precision friendly

        # small aux cls heads
        inst_tok     = tokens[:, 0, :]
        group_logits = self.group_head(inst_tok)
        sub_logits   = self.sub_head(inst_tok)
        group_tgt    = batch["instrument"]["group_id"].long().to(group_logits.device)
        sub_tgt      = batch["instrument"]["subgroup_id"].long().to(sub_logits.device)
        aux_w        = 0.1
        aux_loss     = F.cross_entropy(group_logits, group_tgt) + F.cross_entropy(sub_logits, sub_tgt)

        # ----- RF objective with τ -----
        T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        tau_f32 = torch.rand(B, device=x0.device, dtype=torch.float32).clamp_(1e-4, 1 - 1e-4)
        t_idx   = (tau_f32 * (T_train - 1)).to(torch.long)                  # [B]
        sigma   = tau_f32.to(x0.dtype).view(B, *([1] * (x0.ndim - 1)))      # [B,1,1,1]
        z       = torch.randn_like(x0)
        x_noisy = (1.0 - sigma) * x0 + sigma * z

        # ---- ControlBranch residuals: PR + AMP (no dynamic rebuild; sized in __init__) ----
        if self.use_ctrl_branch:
            pr_128 = batch["conds"]["piano_roll"].to(x_noisy.device, dtype=x_noisy.dtype)        # [B,128,Tpr]
            amp_1t = batch["conds"]["amp"].to(x_noisy.device, dtype=x_noisy.dtype).unsqueeze(1)  # [B,1,T?]
            if amp_1t.shape[-1] != pr_128.shape[-1]:
                amp_1t = F.interpolate(amp_1t, size=pr_128.shape[-1], mode="nearest")
            ctrl_in = torch.cat([pr_128, amp_1t], dim=1)                                         # [B,129,Tpr]

            res_list = self.ctrlnet(ctrl_in, T_out_list=[x_noisy.shape[-1]] * len(self.ctrlnet.to_blocks))
            scale = float(getattr(self.hparams, "control_scale", 1.0))
            self._ctrl_residuals = [r * scale for r in res_list]


        # adapter (randomized gain)
        import random
        base_gain         = self._adapter_gain_scale()
        random_multiplier = random.uniform(0.8, 1.8)
        scale             = base_gain * random_multiplier

        tokens_adapt = self._match_mod_dtype(tokens, self.cond_adapter).clone()
        tokens_adapt[:, 0, :] = tokens_adapt[:, 0, :] * 1.5
        cond_patch = self.cond_adapter(tokens_adapt, T_out=x_noisy.shape[-1], scale=scale)
        cond_patch = cond_patch.to(device=x_noisy.device, dtype=x_noisy.dtype)

        x_in = x_noisy + cond_patch

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

        # optional PR BCE aux
        pr_loss = x0_hat.new_zeros(())
        if self.pr_loss_weight > 0:
            if self.pr_head is None:
                self._init_pr_head(x0_hat)

                # robust to 1 or many optimizers, and unwrap PL wrappers
                opt_obj = self.optimizers()
                opt_list = opt_obj if isinstance(opt_obj, (list, tuple)) else [opt_obj]
                for opt in opt_list:
                    base_opt = getattr(opt, "optimizer", opt)  # unwrap Lightning* wrapper if present
                    base_opt.add_param_group({
                        "params": list(self.pr_head.parameters()),
                        "lr": float(self.hparams.learning_rate) * 2.0,
                    })

            x_feat    = x0_hat.reshape(B, C*H, T_slow)
            pr_logits = self.pr_head(x_feat)
            pr_loss   = self._pr_bce_loss(pr_logits, pr_tgt)
            self.log("aux/pr_bce", pr_loss.detach(), on_step=True)

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

        loss = recon_loss + self.adapter_l2 * cond_reg + aux_w * aux_loss + self.pr_loss_weight * pr_loss

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
