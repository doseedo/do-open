#!/usr/bin/env python3
"""
Inference script for the vocal model checkpoint trained with trainer_performervox.py

This script loads the checkpoint at /mnt/msdd/exps/logs_vox_mhubert/checkpoints/epoch=9-step=10000.ckpt
and performs generation with proper vocal conditioning (lyrics, speaker reference, mHuBERT features).

Usage:
    # Basic generation (instrumental mode - no vocals)
    python infer_vox_model.py \\
        --audio /path/to/audio.wav \\
        --group "Drums" \\
        --subgroup "Acoustic Kit" \\
        --output output.wav

    # With vocal conditioning
    python infer_vox_model.py \\
        --audio /path/to/audio.wav \\
        --group "Vocals" \\
        --subgroup "Lead Vocals" \\
        --reference_wav /path/to/speaker_voice.wav \\
        --mhubert_path /path/to/mhubert_features.pt \\
        --output output.wav

    # Custom generation parameters
    python infer_vox_model.py \\
        --audio /path/to/audio.wav \\
        --group "Vocals" \\
        --subgroup "Lead Vocals" \\
        --steps 100 \\
        --seed 42 \\
        --cfg_weight 3.0 \\
        --adapter_scale 0.7 \\
        --output output.wav
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

# Add project path
sys.path.insert(0, '/home/arlo/Data')

# Import the vocal trainer/pipeline
from trainer_performervox import Pipeline
from dataloadvervox import APPROVED_GROUPS_VOX as APPROVED_GROUPS, APPROVED_SUBGROUPS_VOX as APPROVED_SUBGROUPS

# Constants
DCAE_SR = 44100
DCAE_HOP = 4096
ENC_SR = 24000
ENC_HOP = 320
SLOW_HZ = DCAE_SR / DCAE_HOP

# Default checkpoint path
DEFAULT_CHECKPOINT = "/mnt/msdd/exps/logs_vox_mhubert/checkpoints/epoch=9-step=10000.ckpt"
DEFAULT_CHECKPOINT_DIR = "/home/arlo/Data/ACE-Step/checkpoints"
DEFAULT_MANIFEST = "/home/arlo/Data/final_training_manifest_final.json"


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


def load_model(checkpoint_path: str, checkpoint_dir: str, manifest_json: str) -> Pipeline:
    """
    Load the vocal model checkpoint with proper hyperparameter restoration.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    blob = torch.load(checkpoint_path, map_location="cpu")

    # 1) Pull hparams (Lightning saves as 'hyper_parameters')
    hp = blob.get("hyper_parameters", {})
    ctor_kwargs = _pipeline_ctor_kwargs_from_ckpt_hparams(hp)

    # Always enforce these core args:
    ctor_kwargs["checkpoint_dir"] = checkpoint_dir
    ctor_kwargs["manifest_json"] = manifest_json

    print("Instantiating Pipeline with restored hyperparameters:")
    for k in sorted(ctor_kwargs.keys()):
        print(f"  - {k} = {ctor_kwargs[k]}")

    model = Pipeline(**ctor_kwargs)
    model.eval()

    # 2) State dict load (with compatibility resizing where needed)
    sd = blob.get("state_dict", blob)

    patch_keys = [
        ("ctrl_enc.subgroup_emb.weight", model.ctrl_enc.subgroup_emb.weight),
        ("ctrl_enc.group_emb.weight", model.ctrl_enc.group_emb.weight),
        ("group_head.weight", model.group_head.weight),
        ("group_head.bias", model.group_head.bias),
        ("sub_head.weight", model.sub_head.weight),
        ("sub_head.bias", model.sub_head.bias),
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


def extract_conditioning_from_audio(audio_path: str, output_dir: str = "./extracted_conditioning") -> Dict[str, Any]:
    """
    Extract ALL vocal conditioning using extract_vocal_conditioning.py
    Returns dict with paths to all conditioning files.
    """
    from pathlib import Path
    p = Path(audio_path)
    stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in p.stem)[:128] or "audio"
    out_dir = Path(output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted (check summary file)
    summary_path = out_dir / f"{stem}_conditioning_summary.json"
    if summary_path.exists():
        print(f"✅ Using cached conditioning from {out_dir}")
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        return summary.get("conditioning_paths", {})

    print(f"Extracting vocal conditioning from {audio_path}...")
    import subprocess
    cmd = [
        "python", "/home/arlo/Data/extract_vocal_conditioning.py",
        "--audio", str(audio_path),
        "--output", str(out_dir)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print("❌ Extraction failed.")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Vocal conditioning extraction failed: {result.stderr}")

    print(f"✅ Vocal conditioning extracted to {out_dir}")

    # Load summary
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        return summary.get("conditioning_paths", {})
    else:
        raise RuntimeError("Summary file not created by extraction script")


def _np_load_first(*candidates):
    """Try several filenames; return first that exists."""
    for p in candidates:
        if p is not None and os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError(f"None of the candidates exist: {candidates}")


def load_conditioning(extraction: Dict[str, Any], window_slow: int = None):
    """
    Load conditioning from extracted files.
    Supports two modes:
      - {'paths': {...}}  -> load exactly those files
      - {'dir','stem'}    -> load from extracted cache; accept several filename variants
    Returns: pr [128,T], amp [T], rframe [T], rbend [T], encodec_tokens LongTensor [C,T]
    """
    if "paths" in extraction:
        paths = extraction["paths"]
        pr = _np_load_first(paths.get("piano_roll"), paths.get("pianoroll"))
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
        stem = extraction["stem"]

        # Accept both .pianoroll.npy and .piano_roll.npy
        # Also check for nested directory structure (stem/stem/file)
        nested_dir = out_dir / stem
        pr = _np_load_first(
            out_dir / f"{stem}.pianoroll.npy",
            out_dir / f"{stem}.piano_roll.npy",
            nested_dir / f"{stem}.pianoroll.npy",
            nested_dir / f"{stem}.piano_roll.npy"
        )
        amp = _np_load_first(
            out_dir / f"{stem}.amp.npy",
            nested_dir / f"{stem}.amp.npy"
        )
        rfr = _np_load_first(
            out_dir / f"{stem}.rframe.npy",
            nested_dir / f"{stem}.rframe.npy"
        )
        rbd = _np_load_first(
            out_dir / f"{stem}.rbend.npy",
            nested_dir / f"{stem}.rbend.npy"
        )

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
        pass  # [C, T] is correct
    elif enc.ndim == 3 and enc.shape[0] == 1:
        enc = enc.squeeze(0)  # [1, C, T] -> [C, T]
    else:
        raise RuntimeError(f"Unexpected encodec shape: {enc.shape}")

    # Pad/trim to window_slow if specified
    if window_slow is not None:
        def _pad_arr(x, L):
            if x.shape[-1] >= L:
                return x[..., :L]
            pad = [(0, 0)] * (x.ndim - 1) + [(0, L - x.shape[-1])]
            return np.pad(x, pad, mode="constant")

        pr = _pad_arr(pr, window_slow)
        amp = _pad_arr(amp, window_slow)
        rfr = _pad_arr(rfr, window_slow)
        rbd = _pad_arr(rbd, window_slow)

        # Pad encodec tensor
        if enc.shape[-1] < window_slow * 7:  # rough fast/slow ratio
            pad_size = int(window_slow * 7) - enc.shape[-1]
            enc = torch.nn.functional.pad(enc, (0, pad_size), mode='constant', value=0)
        elif enc.shape[-1] > window_slow * 7:
            enc = enc[..., :int(window_slow * 7)]

    return pr, amp, rfr, rbd, enc.long()


def extract_speaker_embedding(audio_path: str) -> torch.Tensor:
    """
    Extract speaker embedding from reference audio using Resemblyzer.
    Returns [256] tensor.
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        print(f"Extracting speaker embedding from {audio_path}...")

        encoder = VoiceEncoder()
        wav = preprocess_wav(audio_path)
        embedding = encoder.embed_utterance(wav)

        return torch.from_numpy(embedding).float()
    except ImportError:
        print("⚠️  Resemblyzer not available. Install with: pip install resemblyzer")
        # Return zero embedding as fallback
        return torch.zeros(256)


def _adapter_gain_scale_compat(model):
    """Get adapter gain/scale from model (compat with different versions)."""
    if hasattr(model, "cond_adapter"):
        return getattr(model.cond_adapter, "gain", torch.tensor(1.0)).item()
    return 1.0


def _bank_softplus_resized_compat(model, H: int, device, dtype) -> torch.Tensor:
    """Get pitch bank weights for height mapping."""
    if hasattr(model.transformers, "bank_softplus"):
        W = model.transformers.bank_softplus.weight.data
        if W.shape[0] != H:
            W = F.interpolate(W.unsqueeze(0).unsqueeze(0), size=(H, W.shape[1]), mode="bilinear", align_corners=False)
            W = W.squeeze(0).squeeze(0)
        return W.to(device=device, dtype=dtype)
    # Fallback: uniform
    return torch.ones(H, 128, device=device, dtype=dtype) / H


def _prep_ctrl_residuals_if_enabled(model, pr_128, amp_1t, T_lat):
    """Prepare ControlBranch residuals if enabled."""
    if not getattr(model, "use_ctrl_branch", False):
        return None
    if not hasattr(model, "ctrlnet"):
        return None

    # Prepare control input [B, 129, T]
    ctrl_in = torch.cat([pr_128, amp_1t], dim=1)

    # Get instrument token for FiLM modulation (from first token)
    instrument_token = None
    if hasattr(model, "_last_tokens"):
        instrument_token = model._last_tokens[:, 0, :]  # [B, D]

    # Get T_out_list for each injection block
    blocks = getattr(model.transformers, "transformer_blocks", [])
    tail_blocks = blocks[-4:] if len(blocks) >= 4 else blocks
    T_out_list = [T_lat] * len(tail_blocks)

    # Call ControlBranch
    ctrl_res_list = model.ctrlnet(
        pr_128t=ctrl_in,
        T_out_list=T_out_list,
        instrument_token=instrument_token
    )

    return ctrl_res_list


def extract_ground_truth_latents(audio_path: str, model: Pipeline) -> Optional[torch.Tensor]:
    """Extract ground truth latents from audio using the DCAE encoder."""
    try:
        print(f"🔄 Extracting ground truth latents from: {Path(audio_path).name}")

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

        print(f"✅ Extracted ground truth latents: {latents.shape}")
        return latents.cpu()

    except Exception as e:
        print(f"⚠️ Failed to extract ground truth latents: {e}")
        print("   Falling back to pure noise generation")
        return None


def generate(
    model: Pipeline,
    piano_roll: np.ndarray,  # [128, T]
    amp: np.ndarray,         # [T]
    rframe: np.ndarray,      # [T]
    rbend: np.ndarray,       # [T]
    encodec_tokens: torch.Tensor,  # [C_fast, T_fast]
    group: str,
    subgroup: str,
    steps: int = 60,
    seed: int = 0,
    adapter_scale: float = 0.7,
    cfg_weight: float = 2.0,
    t0: float = 1.0,  # NEW: from genfromweb5.py
    sr_out: int = 44100,
    instrument_strength: float = 1.0,
    inst_boost: float = 2.5,
    piano_roll_gain: float = 1.0,
    amp_gain: float = 1.0,
    rframe_gain: float = 1.0,
    rbend_gain: float = 1.0,
    encodec_gain: float = 1.0,
    # NEW: Advanced controls from genfromweb5.py
    pitch_fidelity_boost: float = 1.0,
    onset_guidance_boost: float = 2.0,
    pitch_snap_strength: float = 0.5,
    noise_level: float = 1.0,  # 0.0=pure GT latents, 1.0=pure noise
    audio_file: Optional[str] = None,  # For extracting GT latents
    use_overlap_decoder: bool = True,
    # Vocal conditioning (optional)
    reference_latent: Optional[torch.Tensor] = None,  # [256] speaker embedding
    mhubert_features: Optional[torch.Tensor] = None,  # [T_mhubert, 768]
) -> torch.Tensor:
    """
    Generate audio from conditioning with optional vocal-specific inputs.

    Args:
        model: Loaded Pipeline model
        piano_roll: [128, T] piano roll
        amp: [T] amplitude envelope
        rframe: [T] note frame presence
        rbend: [T] pitch bend
        encodec_tokens: [C_fast, T_fast] EnCodec tokens
        group: Instrument group name
        subgroup: Instrument subgroup name
        steps: Number of diffusion steps
        seed: Random seed
        adapter_scale: Conditioning adapter scale
        cfg_weight: Classifier-free guidance weight
        sr_out: Output sample rate
        instrument_strength: Instrument conditioning strength
        inst_boost: Instrument boost multiplier
        piano_roll_gain: Piano roll gain
        amp_gain: Amplitude gain
        rframe_gain: Frame gain
        rbend_gain: Pitch bend gain
        encodec_gain: EnCodec gain
        reference_latent: [256] speaker embedding (for vocals)
        mhubert_features: [T_mhubert, 768] mHuBERT features (for vocals)

    Returns:
        audio: [1, samples] generated audio tensor
    """
    device = model.device
    model.eval()

    # Safety: remove any override gid/sgid
    if hasattr(model, "override_gid"):
        model.override_gid = None
    if hasattr(model, "override_sgid"):
        model.override_sgid = None

    # Map group/subgroup names to IDs
    if not hasattr(model, "group2id") or not hasattr(model, "subgroup2id"):
        raise RuntimeError("MODEL missing group2id/subgroup2id mappings.")

    try:
        g_id = model.group2id[group]
    except KeyError:
        raise ValueError(f"Unknown group '{group}'. Valid: {sorted(model.group2id.keys())}")

    try:
        s_id = model.subgroup2id[subgroup]
    except KeyError:
        raise ValueError(f"Unknown subgroup '{subgroup}'. Valid: {sorted(model.subgroup2id.keys())}")

    # Validate subgroup belongs to group
    if subgroup not in APPROVED_SUBGROUPS.get(group, []):
        raise ValueError(f"Subgroup '{subgroup}' not valid for group '{group}'. "
                         f"Valid for {group}: {APPROVED_SUBGROUPS.get(group, [])}")

    print(f"🎯 Target: {group} (ID: {g_id}) -> {subgroup} (ID: {s_id})")

    # T_slow
    T_slow = int(piano_roll.shape[1])

    # Build conditioning batch
    conds = {
        "piano_roll": torch.from_numpy(piano_roll).float().unsqueeze(0).to(device),  # [1, 128, T]
        "amp": torch.from_numpy(amp).float().unsqueeze(0).to(device),                # [1, T]
        "rframe": torch.from_numpy(rframe).float().unsqueeze(0).to(device),          # [1, T]
        "rbend": torch.from_numpy(rbend).float().unsqueeze(0).to(device),            # [1, T]
        "rbend_mask": torch.from_numpy((rframe > 0.5).astype(np.float32)).bool().unsqueeze(0).to(device),
        "encodec_tokens": encodec_tokens.unsqueeze(0).to(device),                    # [1, C_fast, T_fast]
        "group_id": torch.tensor([g_id], dtype=torch.long, device=device),
        "subgroup_id": torch.tensor([s_id], dtype=torch.long, device=device),
    }

    # Add vocal conditioning if provided
    if reference_latent is not None:
        conds["reference_latent"] = reference_latent.unsqueeze(0).to(device)  # [1, 256]
        print("✅ Using speaker reference conditioning")

    if mhubert_features is not None:
        # Align mHuBERT features to T_slow
        mh = mhubert_features  # [T_mhubert, 768]
        if mh.shape[0] != T_slow:
            mh = F.interpolate(
                mh.T.unsqueeze(0),  # [1, 768, T_mhubert]
                size=T_slow,
                mode="linear",
                align_corners=False
            ).squeeze(0).T  # [T_slow, 768]
        conds["mhubert_features"] = mh.unsqueeze(0).to(device)  # [1, T_slow, 768]
        print("✅ Using mHuBERT phonetic conditioning")

    # Apply conditioning gains
    audio_reduction = 2.0 - float(instrument_strength) if instrument_strength > 1.0 else 1.0
    audio_reduction = max(0.1, audio_reduction)

    conds["piano_roll"] = conds["piano_roll"] * float(piano_roll_gain) * audio_reduction
    conds["amp"] = conds["amp"] * float(amp_gain) * audio_reduction
    conds["rframe"] = conds["rframe"] * float(rframe_gain) * audio_reduction
    conds["rbend"] = conds["rbend"] * float(rbend_gain) * audio_reduction

    # EnCodec gating
    enc = conds["encodec_tokens"].clone()
    if encodec_gain <= 0.0:
        enc.zero_()
    elif encodec_gain < 1.0:
        keep = (torch.rand_like(enc.float()) < float(encodec_gain))
        enc = torch.where(keep, enc, enc.new_zeros(()).expand_as(enc))
    conds["encodec_tokens"] = enc

    # Get conditioning tokens from ctrl_enc
    orig_inst = getattr(model.ctrl_enc, 'inst_strength', 3.0)
    orig_film = getattr(model.ctrl_enc, 'film_strength', 1.0)
    orig_ch = getattr(model.ctrl_enc, 'channel_mod_strength', 1.0)

    try:
        model.ctrl_enc.inst_strength = orig_inst * float(instrument_strength)

        if encodec_gain <= 0.0:
            model.ctrl_enc.film_strength = 0.0
            model.ctrl_enc.channel_mod_strength = 0.0
        else:
            model.ctrl_enc.film_strength = orig_film * float(encodec_gain)
            model.ctrl_enc.channel_mod_strength = orig_ch * float(encodec_gain)

        tokens, _ = model.ctrl_enc(**conds)
        model._last_tokens = tokens  # Store for ControlBranch
    finally:
        model.ctrl_enc.inst_strength = orig_inst
        model.ctrl_enc.film_strength = orig_film
        model.ctrl_enc.channel_mod_strength = orig_ch

    # Get sample cond_patch to determine latent shape
    tokens_adapt = tokens.to(dtype=next(model.cond_adapter.parameters()).dtype).clone()
    tokens_adapt[:, 0, :] = tokens_adapt[:, 0, :] * 1.5
    sample_patch = model.cond_adapter(tokens_adapt, T_out=T_slow, scale=_adapter_gain_scale_compat(model))

    # Initialize latents with noise_level control
    torch.manual_seed(int(seed))

    if float(noise_level) >= 1.0:
        # Pure noise (original behavior)
        x = torch.randn_like(sample_patch.to(device=device, dtype=tokens.dtype))
        print(f"Using pure noise initialization")
    else:
        # Try to extract ground truth latents for proper noise mixing
        gt_latents = None
        if audio_file is not None:
            gt_latents = extract_ground_truth_latents(audio_file, model)

        if gt_latents is not None:
            # Resize ground truth latents to match expected shape
            gt_latents = gt_latents.to(device=device, dtype=tokens.dtype)
            target_shape = sample_patch.shape  # [1, C, H, T]

            # Add batch dimension if missing
            if gt_latents.ndim == 3:  # [C, H, T] -> [1, C, H, T]
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

    # Prepare ControlBranch residuals
    pr_128 = conds["piano_roll"].to(dtype=x.dtype)
    amp_1t = conds["amp"].unsqueeze(1).to(dtype=x.dtype)
    ctrl_res_list = _prep_ctrl_residuals_if_enabled(model, pr_128, amp_1t, T_slow)

    # Sampling loop (Flow Matching Euler)
    T_train = int(getattr(model.scheduler.config, "num_train_timesteps", 1000))
    steps = max(1, int(steps))
    dt = 1.0 / float(steps)

    print(f"Generating with {steps} steps...")
    for i in range(steps, 0, -1):
        t_cont = torch.full((x.shape[0],), i * dt, device=device, dtype=torch.float32)
        t_idx = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

        # Build cond patches
        tokens_adapt = tokens.to(dtype=next(model.cond_adapter.parameters()).dtype)

        # Instrument ON + high PR
        tokens_on = tokens_adapt.clone()
        tokens_on[:, 0, :] *= float(inst_boost)
        cond_on = model.cond_adapter(tokens_on, T_out=x.shape[-1], scale=float(adapter_scale)).to(x)

        # Instrument OFF + low PR
        tokens_off = tokens_adapt.clone()
        tokens_off[:, 0, :].zero_()
        cond_off = model.cond_adapter(tokens_off, T_out=x.shape[-1], scale=float(adapter_scale)).to(x)

        # Enhanced pitch masking with pitch fidelity and snap
        B, C, H, T_lat = x.shape
        pr = conds["piano_roll"].to(device=x.device, dtype=x.dtype)
        if pr.shape[-1] != T_lat:
            pr = F.interpolate(pr, size=T_lat, mode="nearest")

        # PR energy normalization
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
                    # Blend predicted with target
                    snap_strength = float(pitch_snap_strength)
                    pr_snap = (1.0 - snap_strength) * pr_target + snap_strength * pr_prob
                    pr_target = pr_snap.detach()
            except Exception as e:
                if i == steps:  # Log only once
                    print(f"🔍 [Pitch Snap] PR head unavailable: {e}")

        # Build height map with pitch fidelity boost
        W_hp = _bank_softplus_resized_compat(model, H, device=x.device, dtype=x.dtype)
        pr_high = pr_target * (1.0 + float(pitch_fidelity_boost))  # High PR for cond_on
        pr_low = pr_target * (1.0 - float(pitch_fidelity_boost) * 0.5)  # Low PR for cond_off
        Hmap_on = torch.einsum('bpt,hp->bht', pr_high, W_hp)
        Hmap_off = torch.einsum('bpt,hp->bht', pr_low, W_hp)

        # Sharpen both maps
        sharpness = 1.0 + float(pitch_fidelity_boost) * 0.5
        Hmap_on = (Hmap_on + 1e-6).pow(sharpness)
        Hmap_off = (Hmap_off + 1e-6).pow(sharpness)

        # Normalize per-time
        Hmap_on = Hmap_on / (Hmap_on.amax(dim=1, keepdim=True) + 1e-6)
        Hmap_off = Hmap_off / (Hmap_off.amax(dim=1, keepdim=True) + 1e-6)

        # Hard mask: if frame has zero PR, zero the patch
        active_frames = (pr_norm.amax(dim=1, keepdim=True) > 1e-3).float()

        # Dynamic adapter scaling
        adapt_scale_on = 0.7 + 0.8 * active_frames
        adapt_scale_off = 0.5 + 0.3 * active_frames

        # Apply all enhancements
        cond_on = cond_on * Hmap_on.unsqueeze(1) * adapt_scale_on.unsqueeze(1) * active_frames.unsqueeze(1)
        cond_off = cond_off * Hmap_off.unsqueeze(1) * adapt_scale_off.unsqueeze(1) * active_frames.unsqueeze(1)

        # Onset-weighted guidance schedule
        with torch.no_grad():
            # Detect onsets
            if pr_target.shape[-1] > 1:
                onset = (pr_target[:, :, 1:] > 0.1) & (pr_target[:, :, :-1] <= 0.1)
                onset = F.pad(onset.float().amax(dim=1, keepdim=True), (1,0))  # [B,1,T]
            else:
                onset = torch.zeros_like(pr_target[:, :1, :])

            # Interpolate onset to latent resolution if needed
            if onset.shape[-1] != T_lat:
                onset = F.interpolate(onset, size=T_lat, mode="nearest")

            # Dynamic guidance: boost at onsets
            base_guidance = max(1.0, float(cfg_weight))
            onset_boost = float(onset_guidance_boost)
            step_guidance = base_guidance * (1.0 + onset_boost * onset)  # [B,1,T]

        # Forward both (or just conditioned if CFG disabled)
        if ctrl_res_list is not None:
            model._ctrl_residuals = ctrl_res_list

        # Memory optimization: skip CFG if cfg_weight <= 1.0 (like trainer previews)
        if float(cfg_weight) <= 1.0:
            # No CFG mode - single forward pass (saves 50% memory!)
            v_pred = model._call_transformer_no_xattn(latents=x + cond_on, t=t_idx)
        else:
            # Standard CFG mode - double forward pass
            v_un = model._call_transformer_no_xattn(latents=x + cond_off, t=t_idx)
            v_co = model._call_transformer_no_xattn(latents=x + cond_on, t=t_idx)

            # Apply onset-weighted guidance
            v_diff = v_co - v_un
            v_pred = v_un + step_guidance.unsqueeze(1) * v_diff  # [B,1,1,T] * [B,C,H,T]

        # Update
        x = x - dt * v_pred

        # Aggressive memory cleanup for longer runs
        if steps > 12:
            torch.cuda.empty_cache()

        print(f"  step {steps - i + 1:3d}/{steps}", end="\r")

    # Clear residuals
    model._ctrl_residuals = None
    print("\nDecoding audio...")

    # Decode via DCAE (matching trainer format)
    model.dcae.to(device)
    x_decode = x.to(device=device, dtype=next(model.dcae.parameters()).dtype)

    # Calculate output audio length
    T_slow = x_decode.shape[-1]
    DCAE_HOP = 4096  # From DCAE config
    audio_len_out = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
    audio_lengths = torch.tensor([audio_len_out], device=x_decode.device, dtype=torch.long)

    with torch.no_grad():
        sr_pred, wav_pred = model.dcae.decode(x_decode, audio_lengths=audio_lengths, sr=sr_out)
        audio = wav_pred[0]  # Extract first batch element

    # No need to resample - dcae.decode already returns at sr_out
    # if sr_out != DCAE_SR:
    #     audio = torchaudio.functional.resample(audio, DCAE_SR, sr_out)

    return audio.squeeze(0)  # [1, samples]


def main():
    parser = argparse.ArgumentParser(description="Inference for vocal model checkpoint")

    # Required
    parser.add_argument("--audio", type=str, required=True, help="Input audio file for conditioning extraction")
    parser.add_argument("--output", type=str, required=True, help="Output WAV file path")

    # Instrument
    parser.add_argument("--group", type=str, default="vocal", help="Instrument group")
    parser.add_argument("--subgroup", type=str, default="lead_vocal", help="Instrument subgroup")

    # Vocal conditioning (optional)
    parser.add_argument("--reference_wav", type=str, default=None,
                        help="Reference WAV for speaker voice (extracts embedding)")
    parser.add_argument("--mhubert_path", type=str, default=None,
                        help="Path to mHuBERT features (.pt file)")

    # Generation parameters
    parser.add_argument("--steps", type=int, default=60, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cfg_weight", type=float, default=2.0, help="CFG weight")
    parser.add_argument("--adapter_scale", type=float, default=0.7, help="Adapter scale")
    parser.add_argument("--instrument_strength", type=float, default=1.0, help="Instrument strength")
    parser.add_argument("--inst_boost", type=float, default=2.5, help="Instrument boost")

    # NEW: Advanced generation controls from genfromweb5.py
    parser.add_argument("--t0", type=float, default=1.0, help="T0 timestep (keep 1.0 for full generation)")
    parser.add_argument("--noise_level", type=float, default=1.0,
                        help="Noise level: 0.0=pure GT latents, 1.0=pure noise, 0.5=50%% mix")
    parser.add_argument("--pitch_fidelity_boost", type=float, default=1.0,
                        help="Pitch fidelity boost (higher=sharper pitch)")
    parser.add_argument("--onset_guidance_boost", type=float, default=2.0,
                        help="Onset guidance boost (higher=stronger at note starts)")
    parser.add_argument("--pitch_snap_strength", type=float, default=0.5,
                        help="Pitch snap strength (0.0-1.0, uses PR head for self-consistency)")
    parser.add_argument("--use_overlap_decoder", action="store_true", default=True,
                        help="Use overlap decoder for better audio quality")

    # Conditioning gains
    parser.add_argument("--piano_roll_gain", type=float, default=1.0, help="Piano roll gain")
    parser.add_argument("--amp_gain", type=float, default=1.0, help="Amplitude gain")
    parser.add_argument("--rframe_gain", type=float, default=1.0, help="Frame gain")
    parser.add_argument("--rbend_gain", type=float, default=1.0, help="Pitch bend gain")
    parser.add_argument("--encodec_gain", type=float, default=1.0, help="EnCodec gain")

    # Model paths
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Checkpoint path")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint directory")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help="Manifest JSON")

    # Output
    parser.add_argument("--sr_out", type=int, default=44100, help="Output sample rate")

    args = parser.parse_args()

    # Load model
    print("=" * 60)
    print("Loading Vocal Model")
    print("=" * 60)
    model = load_model(args.checkpoint, args.checkpoint_dir, args.manifest)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}")

    # Extract conditioning from input audio
    print("\n" + "=" * 60)
    print("Extracting Conditioning")
    print("=" * 60)
    conditioning_paths = extract_conditioning_from_audio(args.audio)

    # Load standard conditioning
    piano_roll = np.load(conditioning_paths["piano_roll"])
    amp = np.load(conditioning_paths["amp"])
    rframe = np.load(conditioning_paths["rframe"])
    rbend = np.load(conditioning_paths["rbend"])
    encodec_data = torch.load(conditioning_paths["encodec"], map_location="cpu")

    # Standardize encodec_tokens (might be list or tensor)
    if isinstance(encodec_data, (list, tuple)):
        # Extract tensor from nested list/tuple
        for obj in (encodec_data, encodec_data[0] if len(encodec_data) else None,
                    encodec_data[0][0] if len(encodec_data) and isinstance(encodec_data[0], (list, tuple)) else None):
            if torch.is_tensor(obj):
                encodec_tokens = obj
                break
        else:
            raise RuntimeError("Unrecognized encodec token structure")
    else:
        encodec_tokens = encodec_data

    # Ensure correct shape [C, T]
    if encodec_tokens.ndim == 3 and encodec_tokens.shape[0] == 1:
        encodec_tokens = encodec_tokens.squeeze(0)  # [1, C, T] -> [C, T]

    print(f"✅ Standard conditioning loaded:")
    print(f"   Piano roll: {piano_roll.shape}")
    print(f"   Amplitude: {amp.shape}")
    print(f"   Frame: {rframe.shape}")
    print(f"   Pitch bend: {rbend.shape}")
    print(f"   EnCodec: {encodec_tokens.shape}")

    # Load vocal conditioning (automatically extracted!)
    reference_latent = None
    mhubert_features = None

    # Speaker embedding (automatically extracted)
    if "speaker_emb_path" in conditioning_paths:
        print("\n" + "=" * 60)
        print("Loading Speaker Reference (Auto-Extracted)")
        print("=" * 60)
        reference_latent = torch.load(conditioning_paths["speaker_emb_path"], map_location="cpu")
        print(f"✅ Speaker embedding: {reference_latent.shape}")

    # mHuBERT features (automatically extracted if available)
    if "mhubert_features_path" in conditioning_paths:
        print("\n" + "=" * 60)
        print("Loading mHuBERT Features (Auto-Extracted)")
        print("=" * 60)
        mhubert_data = torch.load(conditioning_paths["mhubert_features_path"], map_location="cpu")
        if isinstance(mhubert_data, dict):
            if 'aligned_features' in mhubert_data:
                mhubert_features = mhubert_data['aligned_features']
            elif 'features' in mhubert_data:
                mhubert_features = mhubert_data['features']
        elif isinstance(mhubert_data, torch.Tensor):
            mhubert_features = mhubert_data

        if mhubert_features is not None:
            print(f"✅ mHuBERT features: {mhubert_features.shape}")
        else:
            print("⚠️  Failed to load mHuBERT features")

    # Override with manual references if provided
    if args.reference_wav:
        print("\n" + "=" * 60)
        print("Overriding with Manual Speaker Reference")
        print("=" * 60)
        reference_latent = extract_speaker_embedding(args.reference_wav)
        print(f"✅ Speaker embedding: {reference_latent.shape}")

    if args.mhubert_path:
        print("\n" + "=" * 60)
        print("Overriding with Manual mHuBERT Features")
        print("=" * 60)
        mhubert_data = torch.load(args.mhubert_path, map_location="cpu")
        if isinstance(mhubert_data, dict):
            if 'aligned_features' in mhubert_data:
                mhubert_features = mhubert_data['aligned_features']
            elif 'features' in mhubert_data:
                mhubert_features = mhubert_data['features']
        elif isinstance(mhubert_data, torch.Tensor):
            mhubert_features = mhubert_data

        if mhubert_features is not None:
            print(f"✅ mHuBERT features: {mhubert_features.shape}")
        else:
            print("⚠️  Failed to load mHuBERT features")

    # Generate
    print("\n" + "=" * 60)
    print("Generating Audio")
    print("=" * 60)
    audio = generate(
        model=model,
        piano_roll=piano_roll,
        amp=amp,
        rframe=rframe,
        rbend=rbend,
        encodec_tokens=encodec_tokens,
        group=args.group,
        subgroup=args.subgroup,
        steps=args.steps,
        seed=args.seed,
        adapter_scale=args.adapter_scale,
        cfg_weight=args.cfg_weight,
        t0=args.t0,
        sr_out=args.sr_out,
        instrument_strength=args.instrument_strength,
        inst_boost=args.inst_boost,
        piano_roll_gain=args.piano_roll_gain,
        amp_gain=args.amp_gain,
        rframe_gain=args.rframe_gain,
        rbend_gain=args.rbend_gain,
        encodec_gain=args.encodec_gain,
        # NEW: Advanced controls
        pitch_fidelity_boost=args.pitch_fidelity_boost,
        onset_guidance_boost=args.onset_guidance_boost,
        pitch_snap_strength=args.pitch_snap_strength,
        noise_level=args.noise_level,
        audio_file=args.audio,  # For GT latent extraction
        use_overlap_decoder=args.use_overlap_decoder,
        # Vocal conditioning
        reference_latent=reference_latent,
        mhubert_features=mhubert_features,
    )

    # Save output
    print(f"\nSaving to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torchaudio.save(args.output, audio, args.sr_out)

    print("=" * 60)
    print(f"✅ Generation complete! Saved to {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
