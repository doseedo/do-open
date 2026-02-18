#!/usr/bin/env python3 -u
"""
Inference script for the vocal model with MIDI-only input.

This is a modified version of infer_vox_model.py that accepts just MIDI file
and lyric/syllable map, generating dummy/empty inputs for all other required
conditioning to prevent the model from breaking.

Usage:
    python infer_vox_model_midi_only.py \
        --midi /path/to/file.mid \
        --lyrics "Hello world" \
        --group "Vocals" \
        --subgroup "Lead Vocals" \
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
import pretty_midi

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
SLOW_HZ = DCAE_SR / DCAE_HOP  # ~10.77 Hz

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
    import sys
    from datetime import datetime

    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Loading checkpoint: {checkpoint_path}")
    sys.stdout.flush()
    blob = torch.load(checkpoint_path, map_location="cpu")
    log("Checkpoint blob loaded")

    # 1) Pull hparams (Lightning saves as 'hyper_parameters')
    hp = blob.get("hyper_parameters", {})
    ctor_kwargs = _pipeline_ctor_kwargs_from_ckpt_hparams(hp)

    # Always enforce these core args:
    ctor_kwargs["checkpoint_dir"] = checkpoint_dir
    ctor_kwargs["manifest_json"] = manifest_json

    log("Instantiating Pipeline with restored hyperparameters:")
    for k in sorted(ctor_kwargs.keys()):
        log(f"  - {k} = {ctor_kwargs[k]}")

    log("About to create Pipeline object...")
    sys.stdout.flush()
    model = Pipeline(**ctor_kwargs)
    log("Pipeline object created, setting to eval mode...")
    model.eval()
    log("Model is now in eval mode")

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


def midi_to_piano_roll(midi_path: str, duration_seconds: float) -> tuple:
    """
    Convert MIDI file to piano roll format.

    Returns:
        piano_roll: [128, T] numpy array (pitch x time)
        rframe: [T] numpy array (note presence, 1.0 where notes exist, 0.0 otherwise)
        T: temporal length in "slow" frames (~10.77 Hz)
    """
    # Load MIDI
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Calculate temporal resolution
    T = int(np.ceil(duration_seconds * SLOW_HZ))
    dt = 1.0 / SLOW_HZ  # seconds per frame

    # Initialize piano roll [128, T]
    piano_roll = np.zeros((128, T), dtype=np.float32)

    # Aggregate all instruments
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue  # Skip drum track for melodic piano roll

        for note in instrument.notes:
            # Map note timing to frame indices
            start_frame = int(note.start / dt)
            end_frame = int(note.end / dt)

            # Clip to valid range
            start_frame = max(0, min(start_frame, T - 1))
            end_frame = max(0, min(end_frame, T))

            # Set piano roll (velocity normalized to 0-1)
            pitch = min(127, max(0, note.pitch))
            velocity = note.velocity / 127.0
            piano_roll[pitch, start_frame:end_frame] = max(
                piano_roll[pitch, start_frame:end_frame].max(),
                velocity
            )

    # Generate rframe (note presence indicator)
    # 1.0 where any pitch is active, 0.0 otherwise
    rframe = (piano_roll.max(axis=0) > 0.0).astype(np.float32)

    print(f"✅ Converted MIDI to piano roll: {piano_roll.shape}, duration: {duration_seconds}s, T={T} frames")
    print(f"   Active pitches: {(piano_roll.max(axis=1) > 0).sum()}/128")
    print(f"   Active frames: {rframe.sum():.0f}/{T}")

    return piano_roll, rframe, T


def generate_dummy_conditioning(T: int) -> tuple:
    """
    Generate dummy/empty conditioning inputs to keep model from breaking.

    Args:
        T: Temporal length in "slow" frames (~10.77 Hz)

    Returns:
        amp: [T] amplitude envelope (constant low value)
        rbend: [T] pitch bend (zeros)
        encodec_tokens: [C, T_fast] EnCodec tokens (zeros)
    """
    # Amplitude: constant low value (model expects some amplitude)
    amp = np.ones(T, dtype=np.float32) * 0.1

    # Pitch bend: zeros (no pitch bending)
    rbend = np.zeros(T, dtype=np.float32)

    # EnCodec tokens: zeros
    # EnCodec operates at ~75 Hz (24000 SR / 320 hop)
    # Ratio to slow frames: 75 / 10.77 ≈ 7x
    T_fast = int(T * 7)
    C = 8  # EnCodec uses 8 codebooks
    encodec_tokens = torch.zeros((C, T_fast), dtype=torch.long)

    print(f"✅ Generated dummy conditioning:")
    print(f"   amp: {amp.shape} (constant 0.1)")
    print(f"   rbend: {rbend.shape} (all zeros)")
    print(f"   encodec_tokens: {encodec_tokens.shape} (all zeros)")

    return amp, rbend, encodec_tokens


def generate_lyrics_conditioning(midi_path: str, lyrics_text: str, T_slow: int) -> dict:
    """
    Generate lyrics_tensors and syllable_boundaries from MIDI and lyrics text.

    Args:
        midi_path: Path to MIDI file
        lyrics_text: Space-separated syllables (e.g., "do re mi fa so la ti do")
        T_slow: Number of frames in slow timeline (~10.77 Hz)

    Returns:
        dict with:
            - lyrics_tensors: Dict containing phoneme_embeddings [N_syllables, 256]
            - syllable_boundaries: [T_slow] tensor with 1.0 at syllable starts
    """
    import pretty_midi

    # Load MIDI to get note timings
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Collect all notes (sorted by start time)
    all_notes = []
    for instrument in pm.instruments:
        if not instrument.is_drum:
            all_notes.extend(instrument.notes)
    all_notes.sort(key=lambda n: n.start)

    if not all_notes:
        print("⚠️  No notes found in MIDI file")
        return None

    # Split lyrics into syllables
    syllables = lyrics_text.strip().split()

    # Map syllables to notes (one syllable per note)
    n_syllables = min(len(syllables), len(all_notes))

    print(f"\n📝 Mapping lyrics to MIDI:")
    print(f"   Syllables: {syllables[:n_syllables]}")
    print(f"   Notes: {len(all_notes)}")

    # Create syllable boundaries: [T_slow] with 1.0 at syllable start frames
    syllable_boundaries = torch.zeros(T_slow, dtype=torch.float32)

    dt = 1.0 / SLOW_HZ  # seconds per frame

    syllable_times = []
    for i in range(n_syllables):
        note = all_notes[i]
        syllable = syllables[i]
        start_frame = int(note.start / dt)
        start_frame = min(start_frame, T_slow - 1)  # Clip to valid range

        syllable_boundaries[start_frame] = 1.0
        syllable_times.append((syllable, start_frame, note.start))

        print(f"   '{syllable}' at frame {start_frame} (t={note.start:.2f}s, pitch={note.pitch})")

    # Generate simple phoneme embeddings (one per syllable)
    # Use random embeddings for now - in a real system, you'd use actual phoneme encoder
    # The model will learn to associate these with phonetic content during training
    phoneme_embeddings = torch.randn(n_syllables, 256, dtype=torch.float32)

    # Normalize embeddings
    phoneme_embeddings = F.normalize(phoneme_embeddings, p=2, dim=-1)

    print(f"✅ Generated lyric conditioning:")
    print(f"   phoneme_embeddings: {phoneme_embeddings.shape}")
    print(f"   syllable_boundaries: {syllable_boundaries.shape} ({int(syllable_boundaries.sum())} syllables)")

    return {
        "lyrics_tensors": {
            "phoneme_embeddings": phoneme_embeddings
        },
        "syllable_boundaries": syllable_boundaries
    }


def generate_dummy_vocal_conditioning() -> tuple:
    """
    Generate dummy vocal conditioning (speaker embedding + mHuBERT features).

    Returns:
        reference_latent: [256] zeros (no speaker conditioning)
        mhubert_features: None (no phonetic features)
    """
    reference_latent = torch.zeros(256, dtype=torch.float32)
    mhubert_features = None  # Model should handle None gracefully

    print(f"✅ Generated dummy vocal conditioning:")
    print(f"   reference_latent: {reference_latent.shape} (all zeros)")
    print(f"   mhubert_features: None")

    return reference_latent, mhubert_features


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
    sr_out: int = 44100,
    instrument_strength: float = 1.0,
    inst_boost: float = 2.5,
    reference_latent: Optional[torch.Tensor] = None,
    mhubert_features: Optional[torch.Tensor] = None,
    vocal_conditioning: Optional[dict] = None,  # NEW: Dict with lyrics_tensors and syllable_boundaries
) -> torch.Tensor:
    """
    Generate audio from MIDI-derived conditioning with dummy inputs for missing data.

    This is a simplified version of the full generate() function that doesn't
    include all the advanced controls (noise_level, pitch_fidelity_boost, etc.)
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

    # Add vocal conditioning if provided (even if dummy zeros)
    if reference_latent is not None:
        conds["reference_latent"] = reference_latent.unsqueeze(0).to(device)  # [1, 256]
        print("✅ Using speaker reference conditioning (dummy zeros)")

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

    # NEW: Add lyric/syllable conditioning if provided
    if vocal_conditioning is not None:
        print("✅ Using lyric/syllable conditioning")
        # The vocal_conditioning dict contains lyrics_tensors and syllable_boundaries
        # Move all tensors to device
        vocal_cond_device = {
            "lyrics_tensors": {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in vocal_conditioning["lyrics_tensors"].items()
            },
            "syllable_boundaries": vocal_conditioning["syllable_boundaries"].to(device)
        }
        # We pass it as-is to the model (it will be processed by VocalLyricProcessor)
        conds["vocal_conditioning"] = [vocal_cond_device]  # Wrap in list for batch dimension

    # Get conditioning tokens from ctrl_enc
    tokens, _ = model.ctrl_enc(**conds)
    model._last_tokens = tokens  # Store for ControlBranch

    # Get sample cond_patch to determine latent shape
    tokens_adapt = tokens.to(dtype=next(model.cond_adapter.parameters()).dtype).clone()
    tokens_adapt[:, 0, :] = tokens_adapt[:, 0, :] * 1.5
    sample_patch = model.cond_adapter(tokens_adapt, T_out=T_slow, scale=_adapter_gain_scale_compat(model))

    # Initialize latents (pure noise only - no GT latent mixing for MIDI-only mode)
    torch.manual_seed(int(seed))
    x = torch.randn_like(sample_patch.to(device=device, dtype=tokens.dtype))
    print(f"Using pure noise initialization (MIDI-only mode)")

    # Prepare ControlBranch residuals
    pr_128 = conds["piano_roll"].to(dtype=x.dtype)
    amp_1t = conds["amp"].unsqueeze(1).to(dtype=x.dtype)
    ctrl_res_list = _prep_ctrl_residuals_if_enabled(model, pr_128, amp_1t, T_slow)

    # Sampling loop (Flow Matching Euler) - simplified version
    T_train = int(getattr(model.scheduler.config, "num_train_timesteps", 1000))
    steps = max(1, int(steps))
    dt = 1.0 / float(steps)

    print(f"Generating with {steps} steps...")
    for i in range(steps, 0, -1):
        t_cont = torch.full((x.shape[0],), i * dt, device=device, dtype=torch.float32)
        t_idx = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

        # Build cond patches
        tokens_adapt = tokens.to(dtype=next(model.cond_adapter.parameters()).dtype)

        # Instrument ON
        tokens_on = tokens_adapt.clone()
        tokens_on[:, 0, :] *= float(inst_boost)
        cond_on = model.cond_adapter(tokens_on, T_out=x.shape[-1], scale=float(adapter_scale)).to(x)

        # Instrument OFF
        tokens_off = tokens_adapt.clone()
        tokens_off[:, 0, :].zero_()
        cond_off = model.cond_adapter(tokens_off, T_out=x.shape[-1], scale=float(adapter_scale)).to(x)

        # Simple CFG
        if ctrl_res_list is not None:
            model._ctrl_residuals = ctrl_res_list

        if float(cfg_weight) <= 1.0:
            v_pred = model._call_transformer_no_xattn(latents=x + cond_on, t=t_idx)
        else:
            v_un = model._call_transformer_no_xattn(latents=x + cond_off, t=t_idx)
            v_co = model._call_transformer_no_xattn(latents=x + cond_on, t=t_idx)
            v_pred = v_un + float(cfg_weight) * (v_co - v_un)

        # Update
        x = x - dt * v_pred

        print(f"  step {steps - i + 1:3d}/{steps}", end="\r")

    # Clear residuals
    model._ctrl_residuals = None
    print("\nDecoding audio...")

    # Decode via DCAE
    model.dcae.to(device)
    x_decode = x.to(device=device, dtype=next(model.dcae.parameters()).dtype)

    # Calculate output audio length
    T_slow = x_decode.shape[-1]
    audio_len_out = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
    audio_lengths = torch.tensor([audio_len_out], device=x_decode.device, dtype=torch.long)

    with torch.no_grad():
        sr_pred, wav_pred = model.dcae.decode(x_decode, audio_lengths=audio_lengths, sr=sr_out)
        audio = wav_pred[0]  # Extract first batch element

    return audio.squeeze(0)  # [1, samples]


def main():
    import sys
    from datetime import datetime

    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    log("Script started")
    parser = argparse.ArgumentParser(description="Inference for vocal model with MIDI-only input")

    # Required
    parser.add_argument("--midi", type=str, required=True, help="Input MIDI file")
    parser.add_argument("--output", type=str, required=True, help="Output WAV file path")

    # Optional - lyrics/syllables (for now just stored, not used by model)
    parser.add_argument("--lyrics", type=str, default="", help="Lyrics text (not yet implemented)")
    parser.add_argument("--syllable_map", type=str, default=None, help="Syllable timing map JSON (not yet implemented)")

    # Instrument
    parser.add_argument("--group", type=str, default="vocal", help="Instrument group")
    parser.add_argument("--subgroup", type=str, default="lead_vocal", help="Instrument subgroup")

    # Generation parameters
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (if not specified, uses MIDI duration)")
    parser.add_argument("--steps", type=int, default=60, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cfg_weight", type=float, default=2.0, help="CFG weight")
    parser.add_argument("--adapter_scale", type=float, default=0.7, help="Adapter scale")
    parser.add_argument("--instrument_strength", type=float, default=1.0, help="Instrument strength")
    parser.add_argument("--inst_boost", type=float, default=2.5, help="Instrument boost")

    # Model paths
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Checkpoint path")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint directory")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help="Manifest JSON")

    # Output
    parser.add_argument("--sr_out", type=int, default=44100, help="Output sample rate")

    args = parser.parse_args()

    # Load model
    print("=" * 60)
    print("Loading Vocal Model (MIDI-Only Mode)")
    print("=" * 60)
    model = load_model(args.checkpoint, args.checkpoint_dir, args.manifest)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}")

    # Determine duration
    if args.duration is None:
        # Get MIDI duration
        pm = pretty_midi.PrettyMIDI(args.midi)
        midi_duration = pm.get_end_time()
        duration = max(1.0, midi_duration)  # At least 1 second
        print(f"✅ Using MIDI duration: {duration:.2f}s")
    else:
        duration = args.duration
        print(f"✅ Using specified duration: {duration:.2f}s")

    # Convert MIDI to piano roll
    print("\n" + "=" * 60)
    print("Converting MIDI to Conditioning")
    print("=" * 60)
    piano_roll, rframe, T = midi_to_piano_roll(args.midi, duration)

    # Generate dummy conditioning
    amp, rbend, encodec_tokens = generate_dummy_conditioning(T)

    # Generate dummy vocal conditioning
    reference_latent, mhubert_features = generate_dummy_vocal_conditioning()

    # Generate lyric/syllable conditioning if lyrics provided
    vocal_conditioning = None
    if args.lyrics:
        vocal_conditioning = generate_lyrics_conditioning(args.midi, args.lyrics, T)
        if vocal_conditioning is None:
            print("⚠️  Failed to generate lyric conditioning, proceeding without lyrics")

    # Generate
    print("\n" + "=" * 60)
    print("Generating Audio from MIDI")
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
        sr_out=args.sr_out,
        instrument_strength=args.instrument_strength,
        inst_boost=args.inst_boost,
        reference_latent=reference_latent,
        mhubert_features=mhubert_features,
        vocal_conditioning=vocal_conditioning,
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
