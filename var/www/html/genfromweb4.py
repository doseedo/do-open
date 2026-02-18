#!/usr/bin/env python3
"""
Gradio Web UI for ACE-Step generation (ControlBranch-ready)

- Loads the same Pipeline you trained with and restores hparams from the Lightning .ckpt
- Works with ctrl_enc + ctrlnet residual injection (ControlBranch1D)
- Instrument-token CFG (ON vs OFF) + sharper PR masking like previews
- Proper EnCodec gating (keeps tokens LongTensor)
"""

import sys, os, argparse, subprocess, json, random, time, shutil, tempfile, hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import gradio as gr

torch.set_float32_matmul_precision("high")

# ------------------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------------------
sys.path.append('/home/arlo/Data')  # folder that has trainer_performer.py

try:
    from trainer_performer_backup import Pipeline  # if you kept a backup
except Exception:
    from trainer_performer import Pipeline

from dataloader import APPROVED_GROUPS, APPROVED_SUBGROUPS

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
DCAE_SR, DCAE_HOP = 44100, 4096
ENC_SR,  ENC_HOP  = 24000,  320

# ------------------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------------------
MODEL: Pipeline | None = None
GROUP_NAMES: list[str] = []
SUBGROUP_NAMES: list[str] = []
MANIFEST_PATHS: list[str] = []
MANIFEST_DATA: list[dict] = []

# Cache for conditioning extractions
CONDITIONING_CACHE: dict[str, dict] = {}

# Cache for ground truth latents
LATENT_CACHE: dict[str, torch.Tensor] = {}

# ------------------------------------------------------------------------------
# Caching helpers
# ------------------------------------------------------------------------------
def _get_file_cache_key(audio_path: str) -> str:
    """Generate a cache key based on file path, size, and modification time."""
    try:
        stat = os.stat(audio_path)
        # Use path, size, and mtime for cache key
        key_data = f"{os.path.abspath(audio_path)}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(key_data.encode()).hexdigest()
    except (OSError, IOError):
        # If we can't stat the file, use just the path
        return hashlib.md5(os.path.abspath(audio_path).encode()).hexdigest()

def _is_cache_valid(cache_entry: dict, audio_path: str) -> bool:
    """Check if a cache entry is still valid."""
    try:
        # Check if all cached files still exist
        if "paths" in cache_entry:
            paths = cache_entry["paths"]
            valid = all(
                paths.get(k) and os.path.exists(paths[k])
                for k in ["piano_roll", "amp", "rframe", "rbend", "encodec"]
            )
            print(f"🔍 Cache validation (manifest paths): {valid}")
            if not valid:
                missing = [k for k in ["piano_roll", "amp", "rframe", "rbend", "encodec"]
                          if not (paths.get(k) and os.path.exists(paths[k]))]
                print(f"⚠️ Missing paths: {missing}")
            return valid
        elif "dir" in cache_entry:
            out_dir = Path(cache_entry["dir"])
            stem = cache_entry["stem"]
            req = [
                out_dir / f"{stem}.pianoroll.npy",
                out_dir / f"{stem}.amp.npy",
                out_dir / f"{stem}.rframe.npy",
                out_dir / f"{stem}.rbend.npy",
                out_dir / f"{stem}.encodec.pt"
            ]
            valid = all(x.exists() for x in req)
            print(f"🔍 Cache validation (disk cache): {valid}")
            if not valid:
                missing = [str(x) for x in req if not x.exists()]
                print(f"⚠️ Missing files: {missing[:2]}{'...' if len(missing) > 2 else ''}")
            return valid
    except Exception as e:
        print(f"⚠️ Cache validation error: {e}")
        pass
    return False

def clear_conditioning_cache():
    """Clear the conditioning cache."""
    global CONDITIONING_CACHE
    CONDITIONING_CACHE.clear()
    print("🗑️ Conditioning cache cleared.")

def clear_latent_cache():
    """Clear the latent cache."""
    global LATENT_CACHE
    LATENT_CACHE.clear()
    print("🗑️ Latent cache cleared.")

def clear_all_caches():
    """Clear both conditioning and latent caches."""
    clear_conditioning_cache()
    clear_latent_cache()

def get_cache_stats() -> dict:
    """Get cache statistics."""
    return {
        "conditioning_entries": len(CONDITIONING_CACHE),
        "latent_entries": len(LATENT_CACHE),
        "conditioning_keys": list(CONDITIONING_CACHE.keys())[:3],
        "latent_keys": list(LATENT_CACHE.keys())[:3]
    }

# ------------------------------------------------------------------------------
# Manifest helpers
# ------------------------------------------------------------------------------
def _find_manifest_record_by_audio(audio_path: str):
    if not MANIFEST_DATA:
        return None
    ap = Path(audio_path)
    # full path
    for it in MANIFEST_DATA:
        p = it.get("audio_path")
        if p and os.path.abspath(p) == os.path.abspath(audio_path):
            return it
    # basename
    for it in MANIFEST_DATA:
        p = it.get("audio_path")
        if p and os.path.basename(p) == ap.name:
            return it
    return None

# ------------------------------------------------------------------------------
# Compatibility helpers
# ------------------------------------------------------------------------------
def _resize_like(src_t: torch.Tensor, target_param: torch.Tensor) -> torch.Tensor:
    src = src_t.detach().cpu()
    tgt = target_param.detach().cpu().clone()
    if tuple(src.shape) == tuple(tgt.shape):
        return src
    common = tuple(min(a, b) for a, b in zip(src.shape, tgt.shape))
    slicers = tuple(slice(0, x) for x in common)
    tgt[slicers] = src[slicers]
    print(f"[compat] resized tensor: ckpt {tuple(src.shape)} -> model {tuple(tgt.shape)}")
    return tgt

def _pipeline_ctor_kwargs_from_ckpt_hparams(hp: dict) -> dict:
    import inspect
    sig = inspect.signature(Pipeline.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    out = {}
    for k, v in (hp or {}).items():
        if k in allowed:
            out[k] = v
    return out

def load_model_any_ckpt(ckpt_path: str, checkpoint_dir: str, manifest_json: str) -> Pipeline:
    print(f"Loading checkpoint: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location="cpu")

    hp = blob.get("hyper_parameters", {})
    ctor_kwargs = _pipeline_ctor_kwargs_from_ckpt_hparams(hp)
    ctor_kwargs["checkpoint_dir"] = checkpoint_dir
    ctor_kwargs["manifest_json"]  = manifest_json

    print("Instantiating Pipeline with restored hyperparameters:")
    for k in sorted(ctor_kwargs.keys()):
        print(f"  - {k} = {ctor_kwargs[k]}")
    model = Pipeline(**ctor_kwargs).eval()

    sd = blob.get("state_dict", blob)

    # Patch keys that commonly change size across runs
    def _safe_getattr_weight(obj, attr, sub_attr=None):
        module = getattr(obj, attr, None)
        if module is None:
            return None
        if sub_attr is not None:
            if hasattr(module, '__getitem__'):  # for things like sclr_proj[0]
                try:
                    module = module[0]
                except (IndexError, TypeError):
                    return None
        return getattr(module, "weight", None) if sub_attr == "weight" else getattr(module, "bias", None)
    
    patch_keys = [
        ("ctrl_enc.subgroup_emb.weight",  _safe_getattr_weight(model.ctrl_enc, "subgroup_emb", "weight")),
        ("ctrl_enc.group_emb.weight",     _safe_getattr_weight(model.ctrl_enc, "group_emb", "weight")),
        ("group_head.weight",             _safe_getattr_weight(model, "group_head", "weight")),
        ("group_head.bias",               _safe_getattr_weight(model, "group_head", "bias")),
        ("sub_head.weight",               _safe_getattr_weight(model, "sub_head", "weight")),
        ("sub_head.bias",                 _safe_getattr_weight(model, "sub_head", "bias")),
        ("ctrl_enc.sclr_proj.0.weight",   _safe_getattr_weight(model.ctrl_enc, "sclr_proj", "weight")),
        ("ctrl_enc.sclr_proj.0.bias",     _safe_getattr_weight(model.ctrl_enc, "sclr_proj", "bias")),
    ]
    for k, target in patch_keys:
        if target is None:
            continue
        if k in sd and tuple(sd[k].shape) != tuple(target.shape):
            sd[k] = _resize_like(sd[k], target)

    print("Loading state dict (strict=False)...")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[compat] Missing keys ({len(missing)}). Example: {missing[:8]}...")
    if unexpected:
        print(f"[compat] Unexpected keys ({len(unexpected)}). Example: {unexpected[:8]}...")

    return model

# ------------------------------------------------------------------------------
# Conditioning I/O
# ------------------------------------------------------------------------------
def extract_conditioning_from_audio(audio_path: str, output_dir: str = "./extracted_conditioning") -> dict:
    # Check memory cache first
    cache_key = _get_file_cache_key(audio_path)
    print(f"🔍 Cache key for {Path(audio_path).name}: {cache_key[:8]}...")
    print(f"🔍 Cache contains {len(CONDITIONING_CACHE)} entries: {list(CONDITIONING_CACHE.keys())}")

    if cache_key in CONDITIONING_CACHE:
        cached_result = CONDITIONING_CACHE[cache_key]
        if _is_cache_valid(cached_result, audio_path):
            print(f"✅ Using cached conditioning from memory for: {Path(audio_path).name}")
            return cached_result
        else:
            # Remove invalid cache entry
            print(f"⚠️ Cache entry invalid for {Path(audio_path).name}, removing...")
            del CONDITIONING_CACHE[cache_key]

    # Check manifest paths
    rec = _find_manifest_record_by_audio(audio_path)
    if rec:
        paths = {}
        prp = rec.get("piano_roll_path") or rec.get("pianoroll_path")
        c = rec.get("conditioning_paths", {}) or {}
        paths["piano_roll"] = prp or c.get("piano_roll") or c.get("pianoroll")
        paths["amp"]        = c.get("amp")
        paths["rframe"]     = c.get("rframe")
        paths["rbend"]      = c.get("rbend")
        paths["encodec"]    = rec.get("encodec_path") or c.get("encodec")
        if all(paths.get(k) and os.path.exists(paths[k]) for k in ["piano_roll","amp","rframe","rbend","encodec"]):
            result = {"paths": paths}
            CONDITIONING_CACHE[cache_key] = result
            print("✅ Using conditioning from manifest paths.")
            return result
        print("⚠️ Manifest record found but some files are missing; falling back to local extraction.")

    # Check disk cache
    stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in Path(audio_path).stem)[:128] or "audio"
    out_dir = Path(output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    req = [out_dir/f"{stem}.pianoroll.npy", out_dir/f"{stem}.amp.npy", out_dir/f"{stem}.rframe.npy",
           out_dir/f"{stem}.rbend.npy", out_dir/f"{stem}.encodec.pt"]
    if all(x.exists() for x in req):
        result = {"dir": str(out_dir), "stem": stem}
        CONDITIONING_CACHE[cache_key] = result
        print(f"✅ Using disk-cached conditioning: {out_dir}")
        return result

    # Extract new conditioning
    print(f"🔄 Extracting conditioning for: {Path(audio_path).name}")
    cmd = ["python", "test_extract_local.py", "--input", str(audio_path), "--output", str(out_dir)]
    print(f"Running extraction: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        print(res.stdout); print(res.stderr)
        raise RuntimeError("Extraction failed.")

    result = {"dir": str(out_dir), "stem": stem}
    CONDITIONING_CACHE[cache_key] = result
    print("✅ Conditioning extracted successfully.")
    return result

def _np_load_first(*candidates):
    for p in candidates:
        if p is not None and os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError(f"None of: {candidates}")

def load_conditioning(extraction: dict, window_slow: int):
    if "paths" in extraction:
        paths = extraction["paths"]
        pr  = _np_load_first(paths.get("piano_roll"), paths.get("pianoroll"))
        amp = _np_load_first(paths.get("amp"))
        rfr = _np_load_first(paths.get("rframe"))
        rbd = _np_load_first(paths.get("rbend"))
        enc_data = torch.load(paths["encodec"], map_location="cpu")
    else:
        out_dir = Path(extraction["dir"]); stem = extraction["stem"]
        nested = out_dir / stem
        pr  = _np_load_first(out_dir/f"{stem}.pianoroll.npy", out_dir/f"{stem}.piano_roll.npy",
                             nested/f"{stem}.pianoroll.npy", nested/f"{stem}.piano_roll.npy")
        amp = _np_load_first(out_dir/f"{stem}.amp.npy", nested/f"{stem}.amp.npy")
        rfr = _np_load_first(out_dir/f"{stem}.rframe.npy", nested/f"{stem}.rframe.npy")
        rbd = _np_load_first(out_dir/f"{stem}.rbend.npy", nested/f"{stem}.rbend.npy")
        enc_path = out_dir/f"{stem}.encodec.pt"
        if not enc_path.exists():
            for cand in [out_dir/f"{stem}.encodec_tokens.pt", nested/f"{stem}.encodec.pt", nested/f"{stem}.encodec_tokens.pt"]:
                if cand.exists():
                    enc_path = cand; break
        enc_data = torch.load(enc_path, map_location="cpu")

    # standardize encodec tensor to [1,C,T] long
    if isinstance(enc_data, (list, tuple)):
        enc = None
        for obj in (enc_data, len(enc_data) and enc_data[0], len(enc_data) and isinstance(enc_data[0], (list,tuple)) and enc_data[0][0]):
            if torch.is_tensor(obj):
                enc = obj; break
        if enc is None: raise RuntimeError("Unrecognized encodec token structure")
    else:
        enc = enc_data
    if enc.ndim == 2:
        enc = enc.unsqueeze(0)

    # pad/trim to window_slow
    def _pad_arr(x, L):
        if x.shape[-1] >= L: return x[..., :L]
        pad = [(0,0)]*(x.ndim-1) + [(0, L - x.shape[-1])]
        return np.pad(x, pad, mode="constant")

    pr  = _pad_arr(pr,  window_slow)
    amp = _pad_arr(amp, window_slow)
    rfr = _pad_arr(rfr, window_slow)
    rbd = _pad_arr(rbd, window_slow)

    return pr, amp, rfr, rbd, enc.long()

# ------------------------------------------------------------------------------
# Latent extraction helpers
# ------------------------------------------------------------------------------
def extract_ground_truth_latents(audio_path: str, model: Pipeline) -> torch.Tensor:
    """Extract ground truth latents from audio using the DCAE encoder."""
    # Check latent cache first
    cache_key = _get_file_cache_key(audio_path)
    if cache_key in LATENT_CACHE:
        print(f"✅ Using cached ground truth latents for: {Path(audio_path).name}")
        return LATENT_CACHE[cache_key]

    try:
        print(f"🔄 Extracting ground truth latents for: {Path(audio_path).name}")

        # Load and preprocess audio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to stereo if mono
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # Normalize audio
        waveform = waveform / (waveform.abs().max() + 1e-8)

        # Move to device and add batch dimension
        device = next(model.parameters()).device
        waveform = waveform.to(device)
        audio_batch = waveform.unsqueeze(0).float()
        audio_lengths = torch.tensor([waveform.shape[-1]], device=device)

        # Extract latents using DCAE
        with torch.no_grad():
            latents, latent_lengths = model.dcae.encode(
                audios=audio_batch,
                audio_lengths=audio_lengths,
                sr=sr
            )

        # Remove batch dimension
        latents = latents.squeeze(0)

        # Cache the latents (keep on CPU to save GPU memory)
        LATENT_CACHE[cache_key] = latents.cpu()
        print(f"✅ Extracted and cached ground truth latents: {latents.shape}")
        return latents

    except Exception as e:
        print(f"⚠️ Failed to extract ground truth latents: {e}")
        print("   Falling back to pure noise generation")
        return None

# ------------------------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------------------------
def _bank_softplus_resized_compat(model, H: int, device, dtype):
    if hasattr(model, "_bank_softplus_resized"):
        return model._bank_softplus_resized(H, device, dtype)
    W = getattr(model, "pitch2h_bank", None)
    if W is None:
        W = torch.ones(H, 128, device=device, dtype=dtype) * 0.01
    else:
        W = W.to(device=device, dtype=dtype)
    if W.shape[0] != H:
        W = F.interpolate(W.T.unsqueeze(0), size=H, mode="linear", align_corners=False).squeeze(0).T
    return F.softplus(W)

def _adapter_gain_scale_compat(model) -> float:
    if hasattr(model, "_adapter_gain_scale"):
        return model._adapter_gain_scale()
    steps = int(getattr(model, "adapter_warmup_steps", 2000))
    gstep = int(getattr(model, "global_step", 0))
    return float(min(1.0, (gstep + 1) / max(1, steps)))

@torch.no_grad()
def _prep_ctrl_residuals_if_enabled(model: Pipeline, pr_128: torch.Tensor, amp_1t: torch.Tensor, T_lat: int):
    if not getattr(model.hparams, "use_ctrl_branch", False):
        return None
    if not hasattr(model, "ctrlnet"):
        return None
    if amp_1t.shape[-1] != pr_128.shape[-1]:
        amp_1t = F.interpolate(amp_1t, size=pr_128.shape[-1], mode="nearest")
    ctrl_in = torch.cat([pr_128, amp_1t], dim=1)  # [B,129,T]
    res_list = model.ctrlnet(ctrl_in, T_out_list=[T_lat] * len(model.ctrlnet.to_blocks))
    scale = float(getattr(model.hparams, "control_scale", 1.0))
    return [r * scale for r in res_list]

# ------------------------------------------------------------------------------
# Sampler
# ------------------------------------------------------------------------------
@torch.no_grad()
def generate(
    model: Pipeline, piano_roll, amp, rframe, rbend, encodec_tokens,
    group, subgroup, steps, seed, adapter_scale, cfg_weight, t0, sr_out,
    instrument_strength=1.0, inst_boost=2.5,
    piano_roll_gain=1.0, amp_gain=1.0, rframe_gain=1.0, rbend_gain=1.0, encodec_gain=1.0,
    use_overlap_decoder=True, original_audio_length=None,
    pitch_fidelity_boost=1.0, onset_guidance_boost=2.0, pitch_snap_strength=0.5,
    noise_level=1.0, audio_file=None
):
    device = next(model.parameters()).device
    model.eval()

    # ids
    g2i = getattr(model, "group2id", None)
    s2i = getattr(model, "subgroup2id", None)
    if g2i is None or s2i is None:
        # fallback to approved lists
        g2i = {g:i for i,g in enumerate(list(APPROVED_GROUPS) if not isinstance(APPROVED_GROUPS, dict) else APPROVED_GROUPS.keys())}
        s2i = {}
        i = 0
        for gs in APPROVED_SUBGROUPS.values():
            for sg in gs:
                if sg not in s2i:
                    s2i[sg] = i; i += 1

    if subgroup not in APPROVED_SUBGROUPS.get(group, []):
        raise ValueError(f"Subgroup '{subgroup}' is not valid for group '{group}'. "
                         f"Valid for {group}: {APPROVED_SUBGROUPS.get(group, [])}")

    gid, sgid = int(g2i[group]), int(s2i[subgroup])
    print(f"[ids] {group}->{gid}  {subgroup}->{sgid}")

    # T grid
    T_slow = int(piano_roll.shape[1])

    # build conds
    conds = {
        "piano_roll": torch.from_numpy(piano_roll).float().unsqueeze(0).to(device),  # [1,128,T]
        "amp":        torch.from_numpy(amp).float().unsqueeze(0).to(device),         # [1,T]
        "rframe":     torch.from_numpy(rframe).float().unsqueeze(0).to(device),      # [1,T]
        "rbend":      torch.from_numpy(rbend).float().unsqueeze(0).to(device),       # [1,T]
        "rbend_mask": torch.from_numpy((rframe > 0.5).astype(np.float32)).bool().unsqueeze(0).to(device),
        "encodec_tokens": encodec_tokens.to(device),                                  # [1,C,Tf]
        "group_id":   torch.tensor([gid], dtype=torch.long, device=device),
        "subgroup_id":torch.tensor([sgid], dtype=torch.long, device=device),
    }

    # stream gains (continuous only)
    audio_red = 2.0 - float(instrument_strength) if instrument_strength > 1.0 else 1.0
    audio_red = max(0.1, audio_red)
    conds["piano_roll"] *= float(piano_roll_gain) * audio_red
    conds["amp"]        *= float(amp_gain)        * audio_red
    conds["rframe"]     *= float(rframe_gain)     * audio_red
    conds["rbend"]      *= float(rbend_gain)      * audio_red

    # encodec gating (keep long)
    enc = conds["encodec_tokens"].clone()
    if encodec_gain <= 0.0:
        enc.zero_()
    elif encodec_gain < 1.0:
        keep = (torch.rand_like(enc.float()) < float(encodec_gain))
        enc = torch.where(keep, enc, enc.new_zeros(()).expand_as(enc))
    conds["encodec_tokens"] = enc

    # ctrl_enc strengths (temporarily scaled by instrument_strength / encodec_gain)
    orig_inst = getattr(model.ctrl_enc, 'inst_strength', 3.0)
    orig_film = getattr(model.ctrl_enc, 'film_strength', 1.0)
    orig_ch   = getattr(model.ctrl_enc, 'channel_mod_strength', 1.0)
    try:
        model.ctrl_enc.inst_strength = orig_inst * float(instrument_strength)
        if encodec_gain <= 0.0:
            model.ctrl_enc.film_strength = 0.0
            model.ctrl_enc.channel_mod_strength = 0.0
        else:
            model.ctrl_enc.film_strength        = orig_film * float(encodec_gain)
            model.ctrl_enc.channel_mod_strength = orig_ch   * float(encodec_gain)
        tokens, _ = model.ctrl_enc(**conds)
    finally:
        model.ctrl_enc.inst_strength = orig_inst
        model.ctrl_enc.film_strength = orig_film
        model.ctrl_enc.channel_mod_strength = orig_ch

    # cond adapter sample / latents init
    tokens_adapt = tokens.to(dtype=next(model.cond_adapter.parameters()).dtype)
    sample_patch = model.cond_adapter(tokens_adapt, T_out=T_slow, scale=_adapter_gain_scale_compat(model))
    if int(seed) <= 0:
        seed = torch.seed() % 2**31
    torch.manual_seed(int(seed))

    # Initialize latents based on noise level
    if float(noise_level) >= 1.0:
        # Pure noise (original behavior)
        x = torch.randn_like(sample_patch.to(device=device, dtype=tokens.dtype))
    else:
        # Try to extract ground truth latents for proper noise mixing
        gt_latents = None
        if audio_file is not None:
            gt_latents = extract_ground_truth_latents(audio_file, model)

        if gt_latents is not None:
            # Resize ground truth latents to match expected shape
            gt_latents = gt_latents.to(device=device, dtype=tokens.dtype)
            target_shape = sample_patch.shape  # [1, 8, 16, T]

            # Add batch dimension if missing
            if gt_latents.ndim == 3:  # [8, 16, T] -> [1, 8, 16, T]
                gt_latents = gt_latents.unsqueeze(0)

            # Pad or crop temporal dimension to match target
            if gt_latents.shape[-1] != target_shape[-1]:
                if gt_latents.shape[-1] < target_shape[-1]:
                    # Pad if too short
                    pad_size = target_shape[-1] - gt_latents.shape[-1]
                    gt_latents = F.pad(gt_latents, (0, pad_size), mode='constant', value=0)
                else:
                    # Crop if too long
                    gt_latents = gt_latents[..., :target_shape[-1]]

            # Ensure dimensions match
            if gt_latents.shape != target_shape:
                print(f"⚠️ GT latent shape mismatch: {gt_latents.shape} vs {target_shape}, using noise")
                x = torch.randn_like(sample_patch.to(device=device, dtype=tokens.dtype))
            else:
                if float(noise_level) <= 0.0:
                    # Pure ground truth latents
                    x = gt_latents
                    print(f"✅ Using pure ground truth latents: {x.shape}")
                else:
                    # Mix ground truth latents with noise
                    noise = torch.randn_like(gt_latents)
                    x = (1.0 - float(noise_level)) * gt_latents + float(noise_level) * noise
                    print(f"✅ Mixed GT latents with {float(noise_level):.2f} noise: {x.shape}")
        else:
            # Fallback to pure noise if no ground truth available
            print("⚠️ No ground truth latents available, using pure noise")
            x = torch.randn_like(sample_patch.to(device=device, dtype=tokens.dtype))

    # Control residuals (constant across loop)
    pr_128 = conds["piano_roll"].to(dtype=x.dtype)
    amp_1t = conds["amp"].unsqueeze(1).to(dtype=x.dtype)
    ctrl_res_list = _prep_ctrl_residuals_if_enabled(model, pr_128, amp_1t, T_lat=T_slow)

    # scheduler mapping
    T_train = int(getattr(model.scheduler.config, "num_train_timesteps", 1000))
    steps   = max(1, int(steps))
    dt      = 1.0 / float(steps)

    for i in range(steps, 0, -1):
        t_cont = torch.full((x.shape[0],), i * dt, device=device, dtype=torch.float32)
        t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

        # instrument ON/OFF patches
        tokens_adapt = tokens.to(dtype=next(model.cond_adapter.parameters()).dtype)
        tokens_on  = tokens_adapt.clone(); tokens_on[:, 0, :] *= float(inst_boost)
        tokens_off = tokens_adapt.clone(); tokens_off[:, 0, :].zero_()

        cond_on  = model.cond_adapter(tokens_on,  T_out=x.shape[-1], scale=float(adapter_scale)).to(x)
        cond_off = model.cond_adapter(tokens_off, T_out=x.shape[-1], scale=float(adapter_scale)).to(x)

        # PR-guided masking/sharpening
        B, C, H, T_lat = x.shape
        pr = conds["piano_roll"].to(device=x.device, dtype=x.dtype)
        if pr.shape[-1] != T_lat:
            pr = F.interpolate(pr, size=T_lat, mode="nearest")
        pr_norm = pr / (pr.amax(dim=1, keepdim=True) + 1e-6)

        pr_target = pr_norm.clone()
        if hasattr(model, 'pr_head') and i < steps * 0.8:
            try:
                with torch.no_grad():
                    x_flat = x.reshape(B, C*H, T_lat)
                    pr_logits = model.pr_head(x_flat)
                    pr_prob = pr_logits.sigmoid()
                    snap = float(pitch_snap_strength)
                    pr_target = ((1.0 - snap) * pr_target + snap * pr_prob).detach()
            except Exception as e:
                if i == steps:
                    print(f"[PitchSnap] disabled: {e}")

        W_hp = _bank_softplus_resized_compat(model, H, device=x.device, dtype=x.dtype)
        pr_high = pr_target * (1.0 + float(pitch_fidelity_boost))
        pr_low  = pr_target * (1.0 - float(pitch_fidelity_boost) * 0.5)
        H_on  = torch.einsum('bpt,hp->bht', pr_high, W_hp)
        H_off = torch.einsum('bpt,hp->bht', pr_low,  W_hp)

        sharp = 1.0 + float(pitch_fidelity_boost) * 0.5
        H_on  = (H_on  + 1e-6).pow(sharp)
        H_off = (H_off + 1e-6).pow(sharp)
        H_on  = H_on  / (H_on.amax(dim=1, keepdim=True)  + 1e-6)
        H_off = H_off / (H_off.amax(dim=1, keepdim=True) + 1e-6)

        active = (pr_norm.amax(dim=1, keepdim=True) > 1e-3).float()
        adapt_on  = 0.7 + 0.8 * active
        adapt_off = 0.5 + 0.3 * active

        cond_on  = cond_on  * H_on.unsqueeze(1)  * adapt_on.unsqueeze(1)  * active.unsqueeze(1)
        cond_off = cond_off * H_off.unsqueeze(1) * adapt_off.unsqueeze(1) * active.unsqueeze(1)

        # onset-weighted guidance
        if pr_target.shape[-1] > 1:
            onset = (pr_target[:, :, 1:] > 0.1) & (pr_target[:, :, :-1] <= 0.1)
            onset = F.pad(onset.float().amax(dim=1, keepdim=True), (1,0))
        else:
            onset = torch.zeros_like(pr_target[:, :1, :])
        if onset.shape[-1] != T_lat:
            onset = F.interpolate(onset, size=T_lat, mode="nearest")
        base_guid = max(1.0, float(cfg_weight))
        step_guid = base_guid * (1.0 + float(onset_guidance_boost) * onset)  # [B,1,T]

        # transformer with ControlBranch residuals
        if ctrl_res_list is not None:
            model._ctrl_residuals = ctrl_res_list

        v_un = model._call_transformer_no_xattn(latents=x + cond_off, t=t_idx)
        v_co = model._call_transformer_no_xattn(latents=x + cond_on,  t=t_idx)
        v_pred = v_un + step_guid.unsqueeze(1) * (v_co - v_un)
        x = x - (1.0 / steps) * v_pred
        if i == steps:
            print(f"[CondEnergy] on={cond_on.norm().item():.3f} off={cond_off.norm().item():.3f}")

    model._ctrl_residuals = None
    print("Decoding audio...")

    # decode
    if original_audio_length is not None:
        audio_len = int(round(original_audio_length * sr_out / DCAE_SR))
    else:
        audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))

    p = next(model.dcae.parameters(), None)
    dev = p.device if p is not None else getattr(model.dcae, "device", device)
    dtype = p.dtype if p is not None else torch.float32
    x_for_dcae = x[:1].to(device=dev, dtype=dtype)
    audio_lengths = torch.tensor([audio_len], dtype=torch.long, device=dev)

    if use_overlap_decoder and hasattr(model.dcae, 'decode_overlap'):
        print("🔊 Using overlap decoder")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev.type=="cuda")):
            sr_pred, wav_pred = model.dcae.decode_overlap(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)
    else:
        sr_pred, wav_pred = model.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)

    wav = wav_pred[0].float().cpu()
    out_dir = Path("./generated_ui"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_seed{seed}_cfg{cfg_weight:.1f}.wav"
    torchaudio.save(str(out_path), wav, sr_pred)
    print(f"✅ Wrote: {out_path}")
    return str(out_path)

# ------------------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------------------
def run_generation(
    audio_file, group, subgroup, seed, steps, adapter_scale, cfg_weight, t0,
    instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
    use_overlap_decoder, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength, noise_level,
    progress=gr.Progress(track_tqdm=True)
):
    if audio_file is None:
        raise gr.Error("Please upload an audio file or pick a random one.")
    progress(0, desc="Extracting conditioning…")

    # original len (for exact decode length)
    try:
        wav, sr = torchaudio.load(audio_file)
        orig_len = wav.shape[-1]
    except Exception:
        orig_len = None

    extraction = extract_conditioning_from_audio(audio_file)
    progress(0.25, desc="Loading conditioning…")
    win_slow = int(getattr(MODEL.hparams, "window_slow", 1024))
    pr, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=win_slow)

    progress(0.5, desc="Generating…")
    out = generate(
        MODEL, pr, amp, rfr, rbd, enc,
        group, subgroup, int(steps), int(seed), float(adapter_scale), float(cfg_weight), float(t0),
        sr_out=32000, instrument_strength=float(instrument_strength), inst_boost=float(inst_boost),
        piano_roll_gain=float(piano_roll_gain), amp_gain=float(amp_gain),
        rframe_gain=float(rframe_gain), rbend_gain=float(rbend_gain), encodec_gain=float(encodec_gain),
        use_overlap_decoder=bool(use_overlap_decoder), original_audio_length=orig_len,
        pitch_fidelity_boost=float(pitch_fidelity_boost), onset_guidance_boost=float(onset_guidance_boost),
        pitch_snap_strength=float(pitch_snap_strength), noise_level=float(noise_level),
        audio_file=audio_file
    )
    progress(1.0, desc="Done!")
    return out

def select_random_file():
    if not MANIFEST_PATHS:
        raise gr.Error("Manifest not loaded.")
    src = Path(random.choice(MANIFEST_PATHS))
    tmp = Path(tempfile.mkdtemp())
    return str(Path(shutil.copy(src, tmp)))

def select_random_file_by_group(target_group):
    if not MANIFEST_DATA:
        raise gr.Error("Manifest not loaded.")
    pool = [it["audio_path"] for it in MANIFEST_DATA if it.get("group")==target_group and it.get("audio_path")]
    if not pool:
        raise gr.Error(f"No files for group '{target_group}'")
    src = Path(random.choice(pool))
    tmp = Path(tempfile.mkdtemp())
    return str(Path(shutil.copy(src, tmp)))

def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        gr.Markdown("### dø stem — ControlBranch Pipeline")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(type="filepath", label="Upload Audio for Conditioning")
                random_btn = gr.Button("🎤 Random from Manifest", variant="secondary")
                random_group_btn = gr.Button("🎯 Random from Current Group", variant="secondary")

                group_dd = gr.Dropdown(GROUP_NAMES, label="Instrument Group",
                                       value=GROUP_NAMES[0] if GROUP_NAMES else None)
                subgroup_dd = gr.Dropdown(SUBGROUP_NAMES, label="Instrument Subgroup",
                                          value=SUBGROUP_NAMES[0] if SUBGROUP_NAMES else None)

            with gr.Column(scale=2):
                with gr.Row():
                    seed_slider  = gr.Slider(0, 10000, value=0, step=1, label="Seed (0 = random)")
                    steps_slider = gr.Slider(10, 100, value=40, step=1, label="Steps")
                with gr.Row():
                    adapter_slider = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Adapter Scale")
                    cfg_slider     = gr.Slider(1.0, 6.0, value=1.0, step=0.1, label="Instrument CFG")
                t0_slider = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="T0 (keep 1.0)")

                instrument_strength = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Instrument Conditioning Strength")
                inst_boost = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="Instrument Token Boost")

                with gr.Row():
                    piano_roll_gain = gr.Slider(0.0, 4.0, value=1.0, step=0.1, label="Piano Roll Gain")
                    amp_gain = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Amplitude Gain")
                with gr.Row():
                    rframe_gain = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="RFrame Gain")
                    rbend_gain  = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="RBend Gain")
                encodec_gain = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="EnCodec Gain")

                pitch_fidelity_boost = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Pitch Fidelity Boost")
                onset_guidance_boost = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Onset Guidance Boost")
                pitch_snap_strength  = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Pitch Snap Strength")

                noise_level = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Noise Level (0=pure conditioning, 1=pure noise)")

                use_overlap_decoder = gr.Checkbox(label="Use Overlap Decoder", value=True)
                generate_btn = gr.Button("🎹 Generate", variant="primary")

        audio_output = gr.Audio(label="Generated Output", type="filepath")

        # events
        random_btn.click(fn=select_random_file, inputs=[], outputs=[audio_input])
        random_group_btn.click(fn=select_random_file_by_group, inputs=[group_dd], outputs=[audio_input])

        # dynamic subgroup options by selected group
        def _opts_for_group(g):
            return gr.Dropdown(choices=sorted(APPROVED_SUBGROUPS.get(g, [])),
                               value=(sorted(APPROVED_SUBGROUPS.get(g, [])) or [None])[0])
        group_dd.change(_opts_for_group, inputs=[group_dd], outputs=[subgroup_dd])

        inputs = [audio_input, group_dd, subgroup_dd, seed_slider, steps_slider, adapter_slider, cfg_slider, t0_slider,
                  instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
                  use_overlap_decoder, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength, noise_level]

        generate_btn.click(fn=run_generation, inputs=inputs, outputs=[audio_output])

    return iface

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    global MODEL, GROUP_NAMES, SUBGROUP_NAMES, MANIFEST_PATHS, MANIFEST_DATA

    DEFAULT_CKPT = "/mnt/msdd/exps/logs_v2/lightning_logs/2025-09-06_16-12-31_all_groups_ft_v3_capivotpitch_ctrl/checkpoints/last.ckpt"
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    with open(args.manifest, "r") as f:
        MANIFEST_DATA = json.load(f)

    print("--- Initializing model ---")
    MODEL = load_model_any_ckpt(args.checkpoint, args.checkpoint_dir, args.manifest)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL.to(dev).eval()
    print(f"✅ Model on {dev}")

    GROUP_NAMES = list(APPROVED_GROUPS) if not isinstance(APPROVED_GROUPS, dict) else list(APPROVED_GROUPS.keys())
    SUBGROUP_NAMES = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})
    MANIFEST_PATHS = [it["audio_path"] for it in MANIFEST_DATA if it.get("audio_path")]
    print(f"Groups: {len(GROUP_NAMES)} | Subgroups: {len(SUBGROUP_NAMES)} | Manifest files: {len(MANIFEST_PATHS)}")

    ui = create_ui()
    ui.launch(share=args.share, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
 