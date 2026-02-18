#!/usr/bin/env python3
"""
Gradio Web UI for ACE-Step generation (new Pipeline)

- Loads the new Pipeline and restores hparams from the Lightning ckpt
- Supports ControlBranch residual injection (PR+AMP) when enabled
- Applies pitch→height masking like in training previews
- Instrument-token CFG: guides with instrument ON vs OFF token difference
- Proper timbre muting when encodec_gain=0 (sets film/channel strengths to 0)

Usage (example):
  CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python gen_from_web.py \
    --checkpoint_dir /path/to/ace-step-checkpoints/ \
    --manifest ./final_training_manifest_final.json
"""
import sys
import os
import argparse
import subprocess
from pathlib import Path
import json
import random
import time
import shutil
import tempfile

import numpy as np
import torch
import torchaudio
import gradio as gr

torch.set_float32_matmul_precision("high")

# --- Project imports -----------------------------------------------------------
# Make sure this points to the folder that contains your new trainer file
# e.g. ~/Data/trainer_performer_backup.py
sys.path.append('/home/arlo/Data')

# Import the *new* pipeline (the one you trained with)
try:
    from trainer_performer_backup import Pipeline  # ← new file you shared
except Exception:
    # fallback to original name if you kept it as trainer_performer.py
    from trainer_performer import Pipeline

from dataloader import APPROVED_GROUPS, APPROVED_SUBGROUPS

import torch.nn.functional as F

# --- Constants ----------------------------------------------------------------
DCAE_SR, DCAE_HOP = 44100, 4096
ENC_SR,  ENC_HOP  = 24000,  320
SLOW_HZ = DCAE_SR / DCAE_HOP

# --- Globals ------------------------------------------------------------------
MODEL = None
GROUP_NAMES = []
SUBGROUP_NAMES = []
MANIFEST_PATHS = []
MANIFEST_DATA = []   # filled in main()

# --- Manifest helpers ---------------------------------------------------------
def _find_manifest_record_by_audio(audio_path: str):
    """Find the manifest item for this audio. Match by full path first, then by basename."""
    from pathlib import Path
    if not MANIFEST_DATA:
        return None
    ap = Path(audio_path)
    # 1) full path
    for it in MANIFEST_DATA:
        p = it.get("audio_path")
        if p and os.path.abspath(p) == os.path.abspath(audio_path):
            return it
    # 2) basename match (may be ambiguous, pick the first)
    for it in MANIFEST_DATA:
        p = it.get("audio_path")
        if p and os.path.basename(p) == ap.name:
            return it
    return None

# --- Helpers ------------------------------------------------------------------
def _resize_like(param_from_ckpt: torch.Tensor, target_param: torch.Tensor) -> torch.Tensor:
    """Return a tensor shaped like target_param, filled from param_from_ckpt (copy overlap)."""
    src = param_from_ckpt.detach().cpu()
    tgt = target_param.detach().cpu().clone()
    if tuple(src.shape) == tuple(tgt.shape):
        return src
    common = tuple(min(a, b) for a, b in zip(src.shape, tgt.shape))
    slicers = tuple(slice(0, x) for x in common)
    tgt[slicers] = src[slicers]
    print(f"[compat] resized tensor: ckpt {tuple(src.shape)} -> model {tuple(tgt.shape)}")
    return tgt

def _pipeline_ctor_kwargs_from_ckpt_hparams(hp: dict) -> dict:
    """Filter Lightning hyper_parameters to only those the Pipeline __init__ accepts."""
    import inspect
    sig = inspect.signature(Pipeline.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    out = {}
    for k, v in (hp or {}).items():
        if k in allowed:
            out[k] = v
    return out

def load_model_any_ckpt(checkpoint_path: str, checkpoint_dir: str, manifest_json: str) -> "Pipeline":
    """
    Instantiate the new Pipeline and load a Lightning .ckpt.
    - Restores hparams so ControlBranch etc. are constructed correctly
    - Patches known resizable keys to avoid shape mismatches
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    blob = torch.load(checkpoint_path, map_location="cpu")

    # 1) Pull hparams (Lightning saves as 'hyper_parameters')
    hp = blob.get("hyper_parameters", {})
    ctor_kwargs = _pipeline_ctor_kwargs_from_ckpt_hparams(hp)
    # Always enforce these core args:
    ctor_kwargs["checkpoint_dir"] = checkpoint_dir
    ctor_kwargs["manifest_json"]  = manifest_json

    print("Instantiating Pipeline with restored hyperparameters:")
    for k in sorted(ctor_kwargs.keys()):
        print(f"  - {k} = {ctor_kwargs[k]}")
    model = Pipeline(**ctor_kwargs)
    model.eval()

    # 2) State dict load (with compatibility resizing where needed)
    sd = blob.get("state_dict", blob)

    patch_keys = [
        ("ctrl_enc.subgroup_emb.weight",  model.ctrl_enc.subgroup_emb.weight),
        ("ctrl_enc.group_emb.weight",     model.ctrl_enc.group_emb.weight),
        ("group_head.weight",             model.group_head.weight),
        ("group_head.bias",               model.group_head.bias),
        ("sub_head.weight",               model.sub_head.weight),
        ("sub_head.bias",                 model.sub_head.bias),
        # historical mismatch that sometimes shows up:
        ("ctrl_enc.sclr_proj.0.weight",   getattr(model.ctrl_enc.sclr_proj[0], "weight", None)),
        ("ctrl_enc.sclr_proj.0.bias",     getattr(model.ctrl_enc.sclr_proj[0], "bias",   None)),
    ]
    for k, target in patch_keys:
        if target is None: 
            continue
        if k in sd and tuple(sd[k].shape) != tuple(target.shape):
            sd[k] = _resize_like(sd[k], target)

    print("Loading state dict into model (strict=False)...")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[compat] Missing keys ({len(missing)}). Example: {missing[:8]}...")
    if unexpected:
        print(f"[compat] Unexpected keys ({len(unexpected)}). Example: {unexpected[:8]}...")

    return model

def extract_conditioning_from_audio(audio_path: str, output_dir: str = "./extracted_conditioning") -> dict:
    """
    Prefer manifest-provided conditioning paths. If not found, fall back to local extraction.
    Returns a dict with either:
      - {"paths": {...}} with explicit file paths, or
      - {"dir": "...", "stem": "..."} for locally extracted cache.
    """
    # 1) Try manifest first
    rec = _find_manifest_record_by_audio(audio_path)
    if rec:
        paths = {}
        # piano roll
        prp = rec.get("piano_roll_path") or rec.get("pianoroll_path")
        # conditioning bundle
        c = rec.get("conditioning_paths", {}) or {}
        paths["piano_roll"] = prp or c.get("piano_roll") or c.get("pianoroll")
        paths["amp"]        = c.get("amp")
        paths["rframe"]     = c.get("rframe")
        paths["rbend"]      = c.get("rbend")
        # encodec tokens
        paths["encodec"]    = rec.get("encodec_path") or c.get("encodec")

        # If all required files exist, use them
        required = ["piano_roll", "amp", "rframe", "rbend", "encodec"]
        if all(paths.get(k) and os.path.exists(paths[k]) for k in required):
            print("✅ Using conditioning from manifest paths.")
            return {"paths": paths}

        print("⚠️ Manifest record found but some files missing on disk; falling back to on-the-fly extraction.")

    # 2) Fall back to local extraction
    from pathlib import Path
    p = Path(audio_path)
    stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in p.stem)[:128] or "audio"
    out_dir = Path(output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    req = [out_dir / f"{stem}.pianoroll.npy",
           out_dir / f"{stem}.amp.npy",
           out_dir / f"{stem}.rframe.npy",
           out_dir / f"{stem}.rbend.npy",
           out_dir / f"{stem}.encodec.pt"]

    if all(x.exists() for x in req):
        print(f"✅ Using cached conditioning: {out_dir}")
        return {"dir": str(out_dir), "stem": stem}

    cmd = ["python", "test_extract_local.py", "--input", str(p), "--output", str(out_dir)]
    print(f"Running extraction: {' '.join(cmd)}")
    import subprocess, textwrap
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        print("❌ Extraction failed.")
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError("Extraction failed.")
    print("✅ Conditioning extracted successfully.")
    return {"dir": str(out_dir), "stem": stem}

def _np_load_first(*candidates):
    """Try several filenames; return first that exists."""
    for p in candidates:
        if p is not None and os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError(f"None of the candidates exist: {candidates}")

def _pad_array(x, L, dims):
    if dims == 1:
        return x[:L] if x.shape[0] >= L else np.pad(x, (0, L - x.shape[0]), mode="constant")
    elif dims == 2:
        return x[:, :L] if x.shape[1] >= L else np.pad(x, ((0, 0), (0, L - x.shape[1])), mode="constant")
    return x

def load_conditioning(extraction: dict, window_slow: int):
    """
    Supports two modes:
      - {'paths': {...}}  -> load exactly those files
      - {'dir','stem'}    -> load from extracted cache; accept several filename variants
    Returns: pr [128,T], amp [T], rframe [T], rbend [T], encodec_tokens LongTensor [B=1,C,T]
    """
    if "paths" in extraction:
        paths = extraction["paths"]
        pr  = _np_load_first(paths.get("piano_roll"), paths.get("pianoroll"))
        amp = _np_load_first(paths.get("amp"))
        rfr = _np_load_first(paths.get("rframe"))
        rbd = _np_load_first(paths.get("rbend"))

        enc_path = paths.get("encodec")
        if not enc_path or not os.path.exists(enc_path):
            raise FileNotFoundError(f"encodec tokens not found: {enc_path}")
        enc_data = torch.load(enc_path, map_location="cpu")
    else:
        from pathlib import Path
        out_dir = Path(extraction["dir"])
        stem    = extraction["stem"]

        # Accept both .pianoroll.npy and .piano_roll.npy
        # Also check for nested directory structure (stem/stem/file)
        nested_dir = out_dir / stem
        pr  = _np_load_first(out_dir / f"{stem}.pianoroll.npy",
                             out_dir / f"{stem}.piano_roll.npy",
                             nested_dir / f"{stem}.pianoroll.npy",
                             nested_dir / f"{stem}.piano_roll.npy")
        amp = _np_load_first(out_dir / f"{stem}.amp.npy",
                             nested_dir / f"{stem}.amp.npy")
        rfr = _np_load_first(out_dir / f"{stem}.rframe.npy",
                             nested_dir / f"{stem}.rframe.npy")
        rbd = _np_load_first(out_dir / f"{stem}.rbend.npy",
                             nested_dir / f"{stem}.rbend.npy")

        enc_path = out_dir / f"{stem}.encodec.pt"
        if not enc_path.exists():
            # also try nested directory and .encodec_tokens.pt
            candidates = [
                out_dir / f"{stem}.encodec_tokens.pt",
                nested_dir / f"{stem}.encodec.pt",
                nested_dir / f"{stem}.encodec_tokens.pt"
            ]
            for candidate in candidates:
                if candidate.exists():
                    enc_path = candidate
                    break
        enc_data = torch.load(enc_path, map_location="cpu")

    # Standardize EnCodec tensor
    # common packings: list/tuple nesting, or [C,T] tensor
    if isinstance(enc_data, (list, tuple)):
        # try a few common nestings
        for obj in (enc_data, enc_data[0] if len(enc_data) else None,
                    enc_data[0][0] if len(enc_data) and isinstance(enc_data[0], (list, tuple)) else None):
            if torch.is_tensor(obj):
                enc = obj
                break
        else:
            raise RuntimeError("Unrecognized encodec token structure")
    else:
        enc = enc_data
    if enc.ndim == 2:
        enc = enc.unsqueeze(0)  # [1,C,T]

    # Pad/trim to window_slow
    def _pad_arr(x, L):
        if x.shape[-1] >= L: return x[..., :L]
        pad = [(0,0)]*(x.ndim-1) + [(0, L - x.shape[-1])]
        return np.pad(x, pad, mode="constant")

    pr  = _pad_arr(pr,  window_slow)
    amp = _pad_arr(amp, window_slow)
    rfr = _pad_arr(rfr, window_slow)
    rbd = _pad_arr(rbd, window_slow)

    return pr, amp, rfr, rbd, enc.long()

def _bank_softplus_resized_compat(model, H: int, device, dtype):
    """
    Compatible replacement for model._bank_softplus_resized(H,...).
    If the model exposes pitch2h_bank, we honor it; otherwise fall back to a small positive matrix.
    """
    if hasattr(model, "_bank_softplus_resized"):
        return model._bank_softplus_resized(H, device, dtype)

    if hasattr(model, "pitch2h_bank"):
        W = model.pitch2h_bank.to(device=device, dtype=dtype)  # [H_base, 128]
    else:
        W = torch.ones(H, 128, device=device, dtype=dtype) * 0.01  # tiny positive fallback

    if W.shape[0] != H:
        W = F.interpolate(W.T.unsqueeze(0), size=H, mode="linear", align_corners=False).squeeze(0).T
    return F.softplus(W)

def _adapter_gain_scale_compat(model) -> float:
    """
    Compatible replacement for model._adapter_gain_scale().
    If the model has the method, use it; otherwise return 1.0.
    """
    if hasattr(model, "_adapter_gain_scale"):
        return model._adapter_gain_scale()
    
    # Fallback: try to implement the warmup logic manually
    if hasattr(model, "global_step"):
        steps = int(getattr(model, "adapter_warmup_steps", 2000))
        return float(min(1.0, (int(model.global_step) + 1) / max(1, steps)))
    
    return 1.0  # fallback

# --- Generation core -----------------------------------------------------------
@torch.no_grad()
def _prep_ctrl_residuals_if_enabled(model: Pipeline, pr_128: torch.Tensor, amp_1t: torch.Tensor, T_lat: int):
    """Compute ControlBranch residuals (constant across the loop) if model was built with it."""
    if not getattr(model.hparams, "use_ctrl_branch", False):
        return None
    if not hasattr(model, "ctrlnet"):
        return None

    if amp_1t.shape[-1] != pr_128.shape[-1]:
        amp_1t = torch.nn.functional.interpolate(amp_1t, size=pr_128.shape[-1], mode="nearest")
    ctrl_in = torch.cat([pr_128, amp_1t], dim=1)  # [B,129,Tpr]
    res_list = model.ctrlnet(ctrl_in, T_out_list=[T_lat] * len(model.ctrlnet.to_blocks))
    scale = float(getattr(model.hparams, "control_scale", 1.0))
    res_list = [r * scale for r in res_list]
    return res_list

@torch.no_grad()
def generate(
    model: Pipeline, piano_roll, amp, rframe, rbend, encodec_tokens,
    group, subgroup, steps, seed, adapter_scale, cfg_weight, t0, sr_out,
    instrument_strength=1.0, inst_boost=2.5, piano_roll_gain=1.0, amp_gain=1.0, rframe_gain=1.0, rbend_gain=1.0, encodec_gain=1.0, 
    use_overlap_decoder=True, original_audio_length=None, pitch_fidelity_boost=1.0, onset_guidance_boost=2.0, pitch_snap_strength=0.5
):
    """Full-from-noise sampler with optional CFG and ControlBranch residuals."""
    device = model.device
    model.eval()
    
    # Safety check: remove any override gid/sgid that might force previews to specific instruments
    if hasattr(model, "override_gid"):  model.override_gid  = None
    if hasattr(model, "override_sgid"): model.override_sgid = None

    # Map names → ids with strict validation
    if not hasattr(MODEL, "group2id") or not hasattr(MODEL, "subgroup2id"):
        raise RuntimeError("MODEL missing group2id/subgroup2id mappings.")

    try:
        g_id = MODEL.group2id[group]
    except KeyError:
        raise ValueError(f"Unknown group '{group}'. Valid: {sorted(MODEL.group2id.keys())}")

    try:
        s_id = MODEL.subgroup2id[subgroup]
    except KeyError:
        raise ValueError(f"Unknown subgroup '{subgroup}'. Valid: {sorted(MODEL.subgroup2id.keys())}")

    # ensure subgroup belongs to group (if you have this mapping)
    if subgroup not in APPROVED_SUBGROUPS.get(group, []):
        raise ValueError(f"Subgroup '{subgroup}' not valid for group '{group}'. "
                         f"Valid for {group}: {APPROVED_SUBGROUPS.get(group, [])}")
        
    print(f"[ids] {group}->{g_id}  {subgroup}->{s_id}  (model vocab)")
  
    # Debug: Print the instrument targeting
    print(f"🎯 Target Instrument: {group} (ID: {g_id}) -> {subgroup} (ID: {s_id})")
    print(f"📋 Available groups: {GROUP_NAMES}")
    print(f"📋 Available subgroups: {SUBGROUP_NAMES}")

    # T (slow grid)
    T_slow = int(piano_roll.shape[1])

    # Build conditioning batch on device
    conds = {
        "piano_roll": torch.from_numpy(piano_roll).float().unsqueeze(0).to(device),          # [1,128,T]
        "amp":        torch.from_numpy(amp).float().unsqueeze(0).to(device),                 # [1,T]
        "rframe":     torch.from_numpy(rframe).float().unsqueeze(0).to(device),              # [1,T]
        "rbend":      torch.from_numpy(rbend).float().unsqueeze(0).to(device),               # [1,T]
        "rbend_mask": torch.from_numpy((rframe > 0.5).astype(np.float32)).bool().unsqueeze(0).to(device),
        "encodec_tokens": encodec_tokens.to(device),                                         # [1,C_fast,T_fast]
        "group_id":   torch.tensor([g_id], dtype=torch.long, device=device),
        "subgroup_id":torch.tensor([s_id], dtype=torch.long, device=device),
    }
    
    # Apply individual conditioning gains ONLY to continuous streams (not encodec)
    # When instrument_strength > 1.0, reduce audio conditioning to boost instrument conditioning
    audio_reduction = 2.0 - float(instrument_strength) if instrument_strength > 1.0 else 1.0
    audio_reduction = max(0.1, audio_reduction)  # Don't go below 0.1
    
    conds["piano_roll"] = conds["piano_roll"] * float(piano_roll_gain) * audio_reduction
    conds["amp"] = conds["amp"] * float(amp_gain) * audio_reduction
    conds["rframe"] = conds["rframe"] * float(rframe_gain) * audio_reduction
    conds["rbend"] = conds["rbend"] * float(rbend_gain) * audio_reduction
    
    # EnCodec gating without changing dtype (keep as LongTensor)
    enc = conds["encodec_tokens"].clone()
    if encodec_gain <= 0.0:
        enc.zero_()  # all codes -> 0 (still Long), "force instrument" style
    elif encodec_gain < 1.0:
        # Bernoulli keep/drop per code without changing dtype
        keep = (torch.rand_like(enc.float()) < float(encodec_gain))
        enc = torch.where(keep, enc, enc.new_zeros(()).expand_as(enc))
    conds["encodec_tokens"] = enc

    # Debug: Print the actual conditioning tensor values being used
    print(f"🔍 Conditioning Debug:")
    print(f"  - group_id tensor: {conds['group_id']} (shape: {conds['group_id'].shape})")
    print(f"  - subgroup_id tensor: {conds['subgroup_id']} (shape: {conds['subgroup_id'].shape})")
    print(f"  - piano_roll range: [{conds['piano_roll'].min():.3f}, {conds['piano_roll'].max():.3f}]")
    print(f"  - instrument_strength: {instrument_strength} -> audio_reduction: {audio_reduction}")
    print(f"  - gains applied: PR={piano_roll_gain}, AMP={amp_gain}, RF={rframe_gain}, RB={rbend_gain}, ENC={encodec_gain}")

    # Get tokens (cond_patch will be computed inside the loop like in training)
    # Temporarily modify the conditioning encoder's strengths for timbre control
    orig_inst = getattr(model.ctrl_enc, 'inst_strength', 3.0)
    orig_film = getattr(model.ctrl_enc, 'film_strength', 1.0)
    orig_ch   = getattr(model.ctrl_enc, 'channel_mod_strength', 1.0)

    try:
        model.ctrl_enc.inst_strength = orig_inst * float(instrument_strength)

        # If user wants encodec suppressed, kill the timbre paths explicitly
        if encodec_gain <= 0.0:
            model.ctrl_enc.film_strength = 0.0
            model.ctrl_enc.channel_mod_strength = 0.0
        else:
            # Partially scale timbre by encodec_gain if desired
            model.ctrl_enc.film_strength = orig_film * float(encodec_gain)
            model.ctrl_enc.channel_mod_strength = orig_ch * float(encodec_gain)

        tokens, _ = model.ctrl_enc(**conds)
    finally:
        # restore
        model.ctrl_enc.inst_strength = orig_inst
        model.ctrl_enc.film_strength = orig_film
        model.ctrl_enc.channel_mod_strength = orig_ch

    # Debug: Verify the instrument token actually changes between groups
    with torch.no_grad():
        t = tokens.detach()
        inst_token_norm = t[:,0,:].norm(dim=-1).mean()
        other_tokens_norm = t[:,1:,:].norm(dim=-1).mean()
        print(f"🔍 [Token Debug] inst_token_norm={inst_token_norm:.3f} others_norm={other_tokens_norm:.3f}")

    # Get a sample cond_patch to determine latent shape
    tokens_adapt = tokens.to(dtype=next(model.cond_adapter.parameters()).dtype).clone()
    tokens_adapt[:, 0, :] = tokens_adapt[:, 0, :] * 1.5
    sample_patch = model.cond_adapter(tokens_adapt, T_out=T_slow, scale=_adapter_gain_scale_compat(model))
    
    # Initial latents (same shape as cond_patch)
    torch.manual_seed(int(seed))
    x = torch.randn_like(sample_patch.to(device=conds["piano_roll"].device, dtype=tokens.dtype))

    # Precompute ControlBranch residuals (constant across loop)
    pr_128 = conds["piano_roll"].to(dtype=x.dtype)                       # [B,128,T]
    amp_1t = conds["amp"].unsqueeze(1).to(dtype=x.dtype)                 # [B,1,T]
    ctrl_res_list = _prep_ctrl_residuals_if_enabled(model, pr_128, amp_1t, T_lat=T_slow)

    T_train = int(getattr(model.scheduler.config, "num_train_timesteps", 1000))
    steps   = max(1, int(steps))
    dt      = 1.0 / float(steps)  # Use full noise schedule like training

    for i in range(steps, 0, -1):
        t_cont = torch.full((x.shape[0],), i * dt, device=device, dtype=torch.float32)
        t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

        # ---- Build cond patches (instrument + PR guided CFG) ----
        tokens_adapt = tokens.to(dtype=next(model.cond_adapter.parameters()).dtype)

        # instrument ON + high PR
        tokens_on = tokens_adapt.clone()
        tokens_on[:, 0, :] *= float(inst_boost)  # <- use the UI slider
        cond_on = model.cond_adapter(tokens_on, T_out=x.shape[-1], scale=float(adapter_scale)).to(x)

        # instrument OFF + low PR (for dual guidance)
        tokens_off = tokens_adapt.clone()
        tokens_off[:, 0, :].zero_()
        cond_off = model.cond_adapter(tokens_off, T_out=x.shape[-1], scale=float(adapter_scale)).to(x)

        # Enhanced pitch→height masking with normalization and sharpening
        B, C, H, T_lat = x.shape
        pr = conds["piano_roll"].to(device=x.device, dtype=x.dtype)
        if pr.shape[-1] != T_lat:
            pr = F.interpolate(pr, size=T_lat, mode="nearest")
        
        # PR energy normalization - normalize per-time to fixed max
        pr_norm = pr / (pr.amax(dim=1, keepdim=True) + 1e-6)
        
        # "Pitch snap" - self-consistency correction using PR head
        pr_target = pr_norm.clone()
        if hasattr(model, 'pr_head') and i < steps * 0.8:  # Only in first 80% of steps
            try:
                with torch.no_grad():
                    # Predict current piano roll from latent
                    x_flat = x.reshape(B, C*H, T_lat)
                    pr_logits = model.pr_head(x_flat)
                    pr_prob = pr_logits.sigmoid()
                    # Blend predicted with target (controllable mix for self-consistency)
                    snap_strength = float(pitch_snap_strength)
                    pr_snap = (1.0 - snap_strength) * pr_target + snap_strength * pr_prob
                    pr_target = pr_snap.detach()
            except Exception as e:
                # If PR head fails, just use original target
                if i == steps:  # Log only once
                    print(f"🔍 [Pitch Snap] PR head unavailable or failed: {e}")
        
        # Build height map with sharpening
        W_hp = _bank_softplus_resized_compat(model, H, device=x.device, dtype=x.dtype)
        pr_high = pr_target * (1.0 + float(pitch_fidelity_boost))  # High PR for cond_on
        pr_low = pr_target * (1.0 - float(pitch_fidelity_boost) * 0.5)  # Low PR for cond_off
        Hmap_on = torch.einsum('bpt,hp->bht', pr_high, W_hp)
        Hmap_off = torch.einsum('bpt,hp->bht', pr_low, W_hp)
        
        # Sharpen both maps - suppress off-pitch heights harder (controllable)
        sharpness = 1.0 + float(pitch_fidelity_boost) * 0.5  # 1.0 to 2.0 range
        Hmap_on = (Hmap_on + 1e-6).pow(sharpness)
        Hmap_off = (Hmap_off + 1e-6).pow(sharpness)
        
        # Normalize per-time so active pitches dominate
        Hmap_on = Hmap_on / (Hmap_on.amax(dim=1, keepdim=True) + 1e-6)
        Hmap_off = Hmap_off / (Hmap_off.amax(dim=1, keepdim=True) + 1e-6)
        
        # Hard mask: if frame has zero PR, zero the patch
        active_frames = (pr_norm.amax(dim=1, keepdim=True) > 1e-3).float()
        
        # Dynamic adapter scaling - increase scale when PR is active
        adapt_scale_on = 0.7 + 0.8 * active_frames  # 1.5x on active frames
        adapt_scale_off = 0.5 + 0.3 * active_frames  # 0.8x on active frames
        
        # Apply all enhancements
        cond_on = cond_on * Hmap_on.unsqueeze(1) * adapt_scale_on.unsqueeze(1) * active_frames.unsqueeze(1)
        cond_off = cond_off * Hmap_off.unsqueeze(1) * adapt_scale_off.unsqueeze(1) * active_frames.unsqueeze(1)

        # Onset-weighted guidance schedule - boost guidance at note onsets
        with torch.no_grad():
            # Detect onsets using the corrected piano roll: current frame has notes but previous didn't
            if pr_target.shape[-1] > 1:
                onset = (pr_target[:, :, 1:] > 0.1) & (pr_target[:, :, :-1] <= 0.1)
                onset = F.pad(onset.float().amax(dim=1, keepdim=True), (1,0))  # [B,1,T]
            else:
                onset = torch.zeros_like(pr_target[:, :1, :])
            
            # Interpolate onset to latent resolution if needed
            if onset.shape[-1] != T_lat:
                onset = F.interpolate(onset, size=T_lat, mode="nearest")
            
            # Dynamic guidance: boost at onsets (controllable strength)
            base_guidance = max(1.0, float(cfg_weight))
            onset_boost = float(onset_guidance_boost)
            step_guidance = base_guidance * (1.0 + onset_boost * onset)  # [B,1,T]

        # Debug: Log relative energy in cond_on vs cond_off (first step only)
        if i == steps:
            with torch.no_grad():
                energy_on = cond_on.norm().item()
                energy_off = cond_off.norm().item()
                onset_ratio = onset.mean().item()
                print(f"🔍 [Cond Energy] cond_on={energy_on:.3f} cond_off={energy_off:.3f} ratio={energy_on/(energy_off+1e-8):.3f}")
                print(f"🎵 [Onset Detection] onset_frames={onset_ratio:.3f} guidance_range=[{base_guidance:.1f}, {(base_guidance*3.0):.1f}]")

        # forward both (keep ControlBranch ON for both)
        if ctrl_res_list is not None:
            model._ctrl_residuals = ctrl_res_list
        v_un = model._call_transformer_no_xattn(latents=x + cond_off, t=t_idx)
        v_co = model._call_transformer_no_xattn(latents=x + cond_on,  t=t_idx)

        # Apply onset-weighted guidance per spatial location
        v_diff = v_co - v_un
        v_pred = v_un + step_guidance.unsqueeze(1) * v_diff  # [B,1,1,T] * [B,C,H,T]

        x = x - dt * v_pred
        print(f"  step {steps - i + 1:3d}/{steps}", end="\r")

    # Clear residuals after loop
    model._ctrl_residuals = None
    print("\nDecoding audio...")

    # Decode via DCAE (use proper dtype matching like in training)
    # Use original audio length if provided, otherwise fall back to window-based calculation
    if original_audio_length is not None:
        audio_len = int(round(original_audio_length * sr_out / DCAE_SR))
        print(f"🎵 Using original audio length: {original_audio_length/DCAE_SR:.2f}s -> {audio_len/sr_out:.2f}s at {sr_out}Hz")
    else:
        audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
        print(f"🎵 Using window-based length: {T_slow * DCAE_HOP/DCAE_SR:.2f}s -> {audio_len/sr_out:.2f}s at {sr_out}Hz")
    # Use _match_mod_dtype equivalent
    p = next(model.dcae.parameters(), None)
    if p is not None:
        x_for_dcae = x[:1].to(device=p.device, dtype=p.dtype)
        audio_lengths = torch.tensor([audio_len], dtype=torch.long, device=p.device)
    else:
        x_for_dcae = x[:1].to(device=model.dcae.device, dtype=torch.float32)
        audio_lengths = torch.tensor([audio_len], dtype=torch.long, device=model.dcae.device)
    # Choose decoder based on setting
    if use_overlap_decoder and hasattr(model.dcae, 'decode_overlap'):
        print("🔊 Using overlap decoder for better quality...")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            sr_pred, wav_pred = model.dcae.decode_overlap(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)
    else:
        decoder_type = "standard" if not use_overlap_decoder else "standard (overlap unavailable)"
        print(f"⚡ Using {decoder_type} decoder...")
        sr_pred, wav_pred = model.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)
    wav = wav_pred[0].float().cpu()

    # Save output file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path("./generated_ui"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{timestamp}_seed{seed}_cfg{cfg_weight:.1f}.wav"
    torchaudio.save(str(out_path), wav, sr_pred)
    print(f"✅ Wrote: {out_path}")
    return str(out_path)

# --- Gradio UI ----------------------------------------------------------------
def run_generation(
    audio_file, group, subgroup, seed, steps, adapter_scale, cfg_weight, t0, 
    instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain, use_overlap_decoder, 
    pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength, progress=gr.Progress(track_tqdm=True)
):
    if audio_file is None:
        raise gr.Error("Please upload an audio file or select a random one.")
    progress(0, desc="Extracting conditioning...")
    print(f"\n--- Starting Generation for {Path(audio_file).name} ---")
    print(f"🎯 UI Selection: {group} -> {subgroup}")

    # Get original audio file length
    try:
        original_audio, original_sr = torchaudio.load(audio_file)
        original_length_samples = original_audio.shape[-1]
        print(f"📊 Original audio: {original_length_samples} samples at {original_sr}Hz ({original_length_samples/original_sr:.2f}s)")
    except Exception as e:
        print(f"⚠️ Could not read original audio length: {e}")
        original_length_samples = None

    extraction = extract_conditioning_from_audio(audio_file)
    progress(0.2, desc="Loading conditioning...")
    win_slow = int(MODEL.hparams.window_slow) if hasattr(MODEL, "hparams") else 2048
    pr, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=win_slow)

    progress(0.4, desc="Generating...")
    out = generate(
        MODEL, pr, amp, rfr, rbd, enc,
        group, subgroup, int(steps), int(seed), float(adapter_scale), float(cfg_weight), float(t0),
        sr_out=32000, instrument_strength=float(instrument_strength), inst_boost=float(inst_boost), piano_roll_gain=float(piano_roll_gain), amp_gain=float(amp_gain), 
        rframe_gain=float(rframe_gain), rbend_gain=float(rbend_gain), encodec_gain=float(encodec_gain), use_overlap_decoder=bool(use_overlap_decoder),
        original_audio_length=original_length_samples, pitch_fidelity_boost=float(pitch_fidelity_boost), 
        onset_guidance_boost=float(onset_guidance_boost), pitch_snap_strength=float(pitch_snap_strength)
    )
    progress(1.0, desc="Done!")
    return out

def select_random_file():
    if not MANIFEST_PATHS:
        raise gr.Error("Manifest file could not be read. No random files available.")
    source_path = Path(random.choice(MANIFEST_PATHS))
    print(f"Selected random file: {source_path}")
    temp_dir = tempfile.mkdtemp()
    try:
        dest_path = Path(shutil.copy(source_path, temp_dir))
        print(f"Copied to temporary location: {dest_path}")
        return str(dest_path)
    except Exception as e:
        print(f"Error copying file to temp directory: {e}")
        raise gr.Error(f"Could not copy the random file for processing. Error: {e}")

def select_random_file_by_group(target_group):
    """Select a random file from manifest filtered by instrument group."""
    if not MANIFEST_DATA:
        raise gr.Error("Manifest data not available. No random files available.")
    
    # Filter manifest entries by group
    group_files = []
    for item in MANIFEST_DATA:
        audio_path = item.get('audio_path')
        group = item.get('group')
        if audio_path and group == target_group:
            group_files.append(audio_path)
    
    if not group_files:
        raise gr.Error(f"No files found for instrument group '{target_group}' in manifest.")
    
    source_path = Path(random.choice(group_files))
    print(f"Selected random {target_group} file: {source_path}")
    
    temp_dir = tempfile.mkdtemp()
    try:
        dest_path = Path(shutil.copy(source_path, temp_dir))
        print(f"Copied to temporary location: {dest_path}")
        return str(dest_path)
    except Exception as e:
        print(f"Error copying file to temp directory: {e}")
        raise gr.Error(f"Could not copy the random file for processing. Error: {e}")

def create_ui():
    print("🔧 Creating UI with conditioning gain sliders...")
    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        gr.Markdown("### dø stem v1.1 — new Pipeline")
        gr.Markdown("Upload an audio file to extract conditioning, choose instrument, tweak params, and render.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Input Conditioning")
                audio_input = gr.Audio(type="filepath", label="Upload Audio for Conditioning")
                random_btn = gr.Button("🎤 Use Random File from Manifest", variant="secondary")
                random_group_btn = gr.Button("🎯 Use Random File from Current Group", variant="secondary")

                gr.Markdown("### 2. Instrument Target")
                group_dd = gr.Dropdown(GROUP_NAMES, label="Instrument Group", value=GROUP_NAMES[0] if GROUP_NAMES else None)
                subgroup_dd = gr.Dropdown(SUBGROUP_NAMES, label="Instrument Subgroup", value=SUBGROUP_NAMES[0] if SUBGROUP_NAMES else None)

            with gr.Column(scale=2):
                gr.Markdown("### 3. Generation Parameters")
                with gr.Row():
                    seed_slider  = gr.Slider(0, 10000, value=0, step=1, label="Seed")
                    steps_slider = gr.Slider(10, 100, value=40, step=1, label="Steps")
                with gr.Row():
                    adapter_slider = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Adapter Scale")
                    cfg_slider     = gr.Slider(1.0, 6.0, value=3.0, step=0.1, label="Instrument CFG (Guidance Strength)")
                t0_slider = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="T0 (Noise Horizon) [Fixed at 1.0 for training compatibility]")
                
                gr.Markdown("### 4. Conditioning Stream Gains")
                instrument_strength = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Instrument Conditioning Strength")
                inst_boost = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="Instrument Token Boost (CFG)")
                with gr.Row():
                    piano_roll_gain = gr.Slider(0.0, 4.0, value=1.0, step=0.1, label="Piano Roll Gain")
                    amp_gain = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Amplitude Gain")
                with gr.Row():
                    rframe_gain = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="RFrame Gain")
                    rbend_gain = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="RBend Gain")
                encodec_gain = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="EnCodec Gain")
                
                gr.Markdown("### 5. MIDI Fidelity Enhancements")
                pitch_fidelity_boost = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Pitch Fidelity Boost (Sharper Masking)")
                onset_guidance_boost = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Onset Guidance Boost (3x at Note Onsets)")
                pitch_snap_strength = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Pitch Snap Strength (Self-Consistency)")
                
                gr.Markdown("### 6. Audio Decoder Options")
                use_overlap_decoder = gr.Checkbox(label="Use Overlap Decoder (Better Quality)", value=True)
                
                generate_btn = gr.Button("🎹 Generate Audio", variant="primary")

        with gr.Row():
            audio_output = gr.Audio(label="Generated Output", type="filepath")

        random_btn.click(fn=select_random_file, inputs=[], outputs=[audio_input])
        random_group_btn.click(fn=select_random_file_by_group, inputs=[group_dd], outputs=[audio_input])
        # Debug: Print the number of inputs
        all_inputs = [audio_input, group_dd, subgroup_dd, seed_slider, steps_slider, adapter_slider, cfg_slider, t0_slider, 
                     instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain, use_overlap_decoder,
                     pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength]
        print(f"🔧 Button handler configured with {len(all_inputs)} inputs")
        
        generate_btn.click(
            fn=run_generation,
            inputs=all_inputs,
            outputs=[audio_output]
        )
    return iface

# --- Main ---------------------------------------------------------------------
def main():
    global MODEL, GROUP_NAMES, SUBGROUP_NAMES, MANIFEST_PATHS, MANIFEST_DATA

    DEFAULT_CKPT = "/mnt/msdd/exps/logs_v2/lightning_logs/2025-09-06_16-12-31_all_groups_ft_v3_capivotpitch_ctrl/checkpoints/last.ckpt"

    ap = argparse.ArgumentParser(description="Web UI for ACE-Step generation (new Pipeline).")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Lightning .ckpt path (defaults to your latest).")
    ap.add_argument("--checkpoint_dir", required=True, help="ACEStep snapshots dir (must contain configs and DCAE).")
    ap.add_argument("--manifest", required=True, help="Training manifest json path.")
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    args = ap.parse_args()

    # Load manifest data first
    with open(args.manifest, 'r') as f:
        MANIFEST_DATA = json.load(f)

    print("--- Initializing Model ---")
    MODEL = load_model_any_ckpt(args.checkpoint, args.checkpoint_dir, args.manifest)

    # Move transformer to best device; DCAE stays where Pipeline put it (often CPU)
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL.to(target_device).eval()
    print(f"✅ Model loaded and moved to {target_device}.")

    # Vocab / UI dropdowns
    GROUP_NAMES    = list(APPROVED_GROUPS) if not isinstance(APPROVED_GROUPS, dict) else list(APPROVED_GROUPS.keys())
    SUBGROUP_NAMES = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})
    print(f"Loaded {len(GROUP_NAMES)} groups and {len(SUBGROUP_NAMES)} subgroups.")
    
    # Debug: Show what the default UI selections will be
    default_group = GROUP_NAMES[0] if GROUP_NAMES else "None"
    default_subgroup = SUBGROUP_NAMES[0] if SUBGROUP_NAMES else "None"
    print(f"🎯 UI Defaults will be: Group='{default_group}' (ID: 0), Subgroup='{default_subgroup}' (ID: 0)")
    print(f"📝 First 5 groups: {GROUP_NAMES[:5]}")
    print(f"📝 First 5 subgroups: {SUBGROUP_NAMES[:5]}")
    
    # Check if violin/strings are in the lists and their positions
    if 'strings' in GROUP_NAMES:
        strings_id = GROUP_NAMES.index('strings')
        print(f"🎻 'strings' group is at index {strings_id}")
    if 'violin' in SUBGROUP_NAMES:
        violin_id = SUBGROUP_NAMES.index('violin')
        print(f"🎻 'violin' subgroup is at index {violin_id}")

    # Manifest audio paths (for the random file button)
    try:
        MANIFEST_PATHS = [item['audio_path'] for item in MANIFEST_DATA if 'audio_path' in item]
        print(f"Loaded {len(MANIFEST_PATHS)} audio paths from manifest.")
    except Exception as e:
        print(f"⚠️ Could not load audio paths from manifest: {e}")

    print("\n--- Launching Gradio UI ---")
    ui = create_ui()
    ui.launch(share=args.share, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
