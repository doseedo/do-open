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
from typing import Optional, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
import torchaudio
import gradio as gr
import pretty_midi
import mido
from mido import MidiFile, MidiTrack, MetaMessage

torch.set_float32_matmul_precision("high")

# ------------------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------------------
sys.path.insert(0, '/home/arlo/Data')  # folder that has trainer_performer.py
sys.path.insert(0, '/home/arlo/Data/dø')  # Add dø directory first (has DoTrainComponents for generate-do endpoint)
sys.path.insert(0, '/home/arlo/Data/ACE-Step')  # Add ACE-Step directory for schedulers and generate-ace-step endpoint

try:
    from trainer_performerCN2 import Pipeline  # Use the actual training script
except Exception:
    try:
        from trainer_performer_backup import Pipeline
    except Exception:
        from trainer_performer import Pipeline

from dataloader import APPROVED_GROUPS, APPROVED_SUBGROUPS
from output_paths import get_output_path, ensure_path_exists

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
DCAE_SR, DCAE_HOP = 44100, 4096
ENC_SR,  ENC_HOP  = 24000,  320

# Soundfont mapping for different instrument groups
INSTRUMENT_SOUNDFONTS = {
    # Brass
    "trombone": "/home/arlo/Data/soundfonts/trombone.sf2",
    "trumpet": "/home/arlo/Data/soundfonts/trumpet.sf2",
    "french_horn": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add french_horn.sf2
    "tuba": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add tuba.sf2

    # Winds
    "sax": "/home/arlo/Data/soundfonts/sax.sf2",
    "bassoon": "/home/arlo/Data/soundfonts/bassoon.sf2",
    "clarinet": "/home/arlo/Data/soundfonts/clarinet.sf2",
    "flute": "/home/arlo/Data/soundfonts/flute.sf2",
    "oboe": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add oboe.sf2

    # Strings
    "violin": "/home/arlo/Data/soundfonts/violin.sf2",
    "viola": "/home/arlo/Data/soundfonts/viola.sf2",
    "cello": "/home/arlo/Data/soundfonts/cello.sf2",

    # Piano
    "acoustic_piano": "/home/arlo/Data/soundfonts/Piano.sf2",
    "electric_piano": "/home/arlo/Data/soundfonts/Electric Piano.sf2",
    "keys": "/usr/share/sounds/sf2/FluidR3_GM.sf2",

    # Guitar
    "acoustic_guitar": "/home/arlo/Data/soundfonts/acoustic guitar.sf2",
    "electric_guitar": "/home/arlo/Data/soundfonts/electric guitar.sf2",
    "plucked": "/usr/share/sounds/sf2/FluidR3_GM.sf2",

    # Bass
    "electric_bass": "/home/arlo/Data/soundfonts/electric bass.sf2",
    "bass": "/home/arlo/Data/soundfonts/electric bass.sf2",

    "undefined": "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "default": "/usr/share/sounds/sf2/FluidR3_GM.sf2"  # fallback
}

# Group-level minimum note limits (MIDI note numbers)
# Notes below these values will be transposed up by octaves in monophonic mode
GROUP_NOTE_LIMITS = {
    "strings": 36,   # C2 - no notes below this for strings
    "winds": 34,     # A#1 - bassoon can go this low
    "brass": 28,     # E1 - tuba can go this low
}

# Arrange mode: MIDI note ranges for automatic instrument assignment
# Based on standard orchestral ranges (MIDI note numbers, middle C = 60)
INSTRUMENT_RANGES = {
    "strings": {
        "violin": (55, 103),      # G3-G7 (high register)
        "viola": (48, 84),        # C3-C6 (mid-high register)
        "cello": (36, 72),        # C2-C5 (mid-low register)
    },
    "winds": {
        "flute": (60, 96),        # C4-C7 (very high)
        "clarinet": (50, 91),     # D3-G6 (high to mid-high)
        "sax": (42, 82),          # F#2-A#5 (mid range)
        "bassoon": (34, 70),      # A#1-A#4 (low range)
        "oboe": (58, 91),         # A#3-G6 (high)
    },
    "brass": {
        "trumpet": (55, 82),      # G3-A#5 (high)
        "french_horn": (41, 77),  # F2-F5 (mid-high)
        "trombone": (34, 70),     # A#1-A#4 (mid-low)
        "tuba": (28, 58),         # E1-A#3 (low)
    }
}

# ------------------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------------------
MODEL: Union[Pipeline, None] = None
GROUP_NAMES: list = []
SUBGROUP_NAMES: list = []
MANIFEST_PATHS: list = []
MANIFEST_DATA: list = []

# ACE-Step pipeline for original ACE-Step endpoint
ACE_STEP_PIPELINE = None

# Cache for conditioning extractions
CONDITIONING_CACHE: dict = {}

# Cache for ground truth latents
LATENT_CACHE: dict = {}

# Maximum window_slow to prevent CUDA out-of-bounds errors with long MIDI files
# Model's position embeddings have a maximum sequence length - exceeding this causes:
# "CUDA error: device-side assert triggered"
# Typical model max: 2048-4096 frames. Using 2048 as safe limit (~47.5 seconds at 43.066 fps)
MAX_WINDOW_SLOW = 2048

def clamp_window_slow(window_slow: int, duration_seconds: float = None, fps: float = 43.066) -> int:
    """
    Clamp window_slow to MAX_WINDOW_SLOW to prevent CUDA errors.

    Args:
        window_slow: Calculated window_slow value
        duration_seconds: Optional duration in seconds (for warning message)
        fps: Frames per second (for warning message)

    Returns:
        Clamped window_slow value
    """
    if window_slow > MAX_WINDOW_SLOW:
        if duration_seconds is not None:
            max_duration = MAX_WINDOW_SLOW / fps
            print(f"⚠️  WARNING: MIDI/Audio duration ({duration_seconds:.2f}s) exceeds maximum ({max_duration:.2f}s)")
            print(f"   Calculated window_slow: {window_slow} frames")
            print(f"   Clamping to MAX_WINDOW_SLOW: {MAX_WINDOW_SLOW} frames")
            print(f"   Output will be truncated to {max_duration:.2f}s")
        else:
            print(f"⚠️  WARNING: window_slow ({window_slow}) exceeds MAX_WINDOW_SLOW ({MAX_WINDOW_SLOW})")
            print(f"   Clamping to prevent CUDA errors")
        return MAX_WINDOW_SLOW
    return window_slow

# ------------------------------------------------------------------------------
# Caching helpers
# ------------------------------------------------------------------------------
def _get_file_cache_key(audio_path: str, extra_context: str = None) -> str:
    """Generate a cache key based on file path, size, modification time, and optional context."""
    try:
        stat = os.stat(audio_path)
        # Use path, size, mtime, and optional context for cache key
        key_data = f"{os.path.abspath(audio_path)}_{stat.st_size}_{stat.st_mtime}"
        if extra_context:
            key_data += f"_{extra_context}"
        return hashlib.md5(key_data.encode()).hexdigest()
    except (OSError, IOError):
        # If we can't stat the file, use just the path and context
        key_data = os.path.abspath(audio_path)
        if extra_context:
            key_data += f"_{extra_context}"
        return hashlib.md5(key_data.encode()).hexdigest()

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
# Audio Utility Functions
# ------------------------------------------------------------------------------
def apply_tape_speed_sox(input_path: str, output_path: str, speed: float) -> str:
    """
    Apply tape-style speed change (varispeed) using sox.
    This changes both tempo and pitch like a tape machine.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        speed: Speed factor (e.g., 0.8 = slower, 1.25 = faster)

    Returns:
        Path to the output file
    """
    import subprocess

    if speed == 1.0:
        # No change needed, just copy
        shutil.copy(input_path, output_path)
        return output_path

    print(f"🎞️ Applying tape speed {speed}x: {Path(input_path).name} → {Path(output_path).name}")

    # Use sox speed effect for tape-style slowdown/speedup
    cmd = ["sox", input_path, output_path, "speed", str(speed)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        print(f"✅ Tape speed applied successfully")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Sox command failed: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        raise RuntimeError(f"Failed to apply tape speed: {e}")
    except FileNotFoundError:
        raise RuntimeError("Sox not found. Please install sox: sudo apt-get install sox")

def apply_time_stretch_sox(input_path: str, output_path: str, speed: float) -> str:
    """
    Apply time-stretching (pitch-preserving speed change) using sox.
    This changes tempo but preserves pitch using time-stretching algorithms.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        speed: Speed factor (e.g., 0.8 = slower, 1.25 = faster)

    Returns:
        Path to the output file
    """
    import subprocess

    if speed == 1.0:
        # No change needed, just copy
        shutil.copy(input_path, output_path)
        return output_path

    print(f"🎼 Applying time-stretch {speed}x (pitch preserved): {Path(input_path).name} → {Path(output_path).name}")

    # Use sox tempo effect for pitch-preserving slowdown/speedup
    # tempo changes the speed without changing pitch (time-stretching)
    cmd = ["sox", input_path, output_path, "tempo", str(speed)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        print(f"✅ Time-stretch applied successfully")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Sox command failed: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        raise RuntimeError(f"Failed to apply time-stretch: {e}")
    except FileNotFoundError:
        raise RuntimeError("Sox not found. Please install sox: sudo apt-get install sox")

def apply_pitch_shift_sox(input_path: str, output_path: str, semitones: int) -> str:
    """
    Apply pitch shift using sox.
    This changes pitch without changing duration.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        semitones: Number of semitones to shift (e.g., 12 = up one octave, -12 = down one octave)

    Returns:
        Path to the output file
    """
    import subprocess

    if semitones == 0:
        # No change needed, just copy
        shutil.copy(input_path, output_path)
        return output_path

    print(f"🎹 Pitch shifting {semitones:+d} semitones: {Path(input_path).name} → {Path(output_path).name}")

    # Use sox pitch effect (pitch takes cents, 100 cents = 1 semitone)
    cents = semitones * 100
    cmd = ["sox", input_path, output_path, "pitch", str(cents)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        print(f"✅ Pitch shift applied successfully")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Sox command failed: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        raise RuntimeError(f"Failed to apply pitch shift: {e}")
    except FileNotFoundError:
        raise RuntimeError("Sox not found. Please install sox: sudo apt-get install sox")

def sum_audio_tracks(audio_file_paths: list, output_path: str, normalize: bool = True) -> str:
    """
    Sum/mix multiple audio tracks into a single master track using torchaudio.

    Args:
        audio_file_paths: List of paths to audio files to mix
        output_path: Path where the mixed track should be saved
        normalize: Whether to normalize the output to prevent clipping (default: True)

    Returns:
        Path to the generated master track
    """
    if not audio_file_paths:
        raise ValueError("No audio files provided")

    if len(audio_file_paths) == 1:
        # If only one track, just copy it
        shutil.copy(audio_file_paths[0], output_path)
        return output_path

    print(f"🎚️ Mixing {len(audio_file_paths)} audio tracks...")

    # Load all audio files and find the longest duration
    audio_tensors = []
    sample_rate = None
    max_length = 0

    for i, file_path in enumerate(audio_file_paths):
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File not found: {file_path}")
            continue

        wav, sr = torchaudio.load(file_path)

        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            # Resample if needed
            wav = torchaudio.functional.resample(wav, sr, sample_rate)

        audio_tensors.append(wav)
        max_length = max(max_length, wav.shape[-1])
        print(f"   Track {i+1}: {wav.shape[-1]} samples ({wav.shape[-1]/sr:.2f}s)")

    if not audio_tensors:
        raise ValueError("No valid audio files found")

    # Pad all tracks to the same length
    padded_tensors = []
    for wav in audio_tensors:
        if wav.shape[-1] < max_length:
            pad_length = max_length - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad_length))
        padded_tensors.append(wav)

    # Sum all tracks
    mixed = torch.stack(padded_tensors).sum(dim=0)

    # Normalize if requested
    if normalize:
        max_val = mixed.abs().max()
        if max_val > 0:
            # Leave some headroom (0.95 to avoid clipping)
            mixed = mixed * (0.95 / max_val)
            print(f"   Normalized: peak reduced from {max_val:.3f} to {mixed.abs().max():.3f}")

    # Save the mixed track
    torchaudio.save(output_path, mixed, sample_rate)

    print(f"✅ Mixed track saved: {output_path}")
    print(f"   Duration: {max_length / sample_rate:.2f}s")
    print(f"   Channels: {mixed.shape[0]}")

    return output_path

# ------------------------------------------------------------------------------
# MIDI Processing and FluidSynth Rendering
# ------------------------------------------------------------------------------

def modify_midi_tempo(input_midi_path: str, output_midi_path: str, tempo_scale: float) -> str:
    """
    Modify MIDI file tempo by scaling all tempo change events.

    Args:
        input_midi_path: Path to input MIDI file
        output_midi_path: Path to save modified MIDI file
        tempo_scale: Tempo scaling factor (e.g., 0.75 = 75% speed, slower)

    Returns:
        Path to the modified MIDI file
    """
    import mido

    if tempo_scale == 1.0:
        # No change needed, just copy
        shutil.copy(input_midi_path, output_midi_path)
        return output_midi_path

    print(f"🎼 Modifying MIDI tempo: {tempo_scale}x")
    print(f"   Input:  {Path(input_midi_path).name}")
    print(f"   Output: {Path(output_midi_path).name}")

    midi = mido.MidiFile(input_midi_path)

    # Scale all tempo messages
    # Note: MIDI tempo is in microseconds per quarter note
    # To slow down by factor X, we multiply tempo by 1/X (increase microseconds per beat)
    tempo_multiplier = 1.0 / tempo_scale
    print(f"   🔍 DEBUG: tempo_scale={tempo_scale:.3f}, tempo_multiplier={tempo_multiplier:.3f}")
    print(f"   🔍 DEBUG: If tempo_scale={tempo_scale:.3f}, we want to slow down by {tempo_scale:.3f}x")
    print(f"   🔍 DEBUG: In MIDI, higher µs/qn = slower, so we multiply by {tempo_multiplier:.3f}")

    tempo_found = False
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_found = True
                original_tempo = msg.tempo
                original_bpm = 60_000_000 / original_tempo
                msg.tempo = int(original_tempo * tempo_multiplier)
                new_bpm = 60_000_000 / msg.tempo
                print(f"   Modified tempo: {original_tempo} → {msg.tempo} µs/qn")
                print(f"   🔍 DEBUG: BPM: {original_bpm:.1f} → {new_bpm:.1f} BPM")

    if not tempo_found:
        print(f"   ⚠️ WARNING: No tempo messages found in MIDI file!")

    midi.save(output_midi_path)
    print(f"   ✅ Saved modified MIDI with {tempo_scale}x tempo")

    return output_midi_path

def is_midi_file(file_path: str) -> bool:
    """Check if file is a MIDI file based on extension."""
    if not file_path:
        return False
    return Path(file_path).suffix.lower() in ['.mid', '.midi']

def is_multitrack_midi(midi_path: str) -> tuple:
    """
    Check if MIDI file contains multiple tracks/instruments.
    Returns:
        tuple: (is_multitrack: bool, track_count: int, non_drum_tracks: list)
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        non_drum_instruments = [inst for inst in midi_data.instruments if not inst.is_drum]
        track_count = len(non_drum_instruments)

        # Consider multitrack if more than 1 non-drum instrument
        is_multitrack = track_count > 1

        print(f"🎼 MIDI Analysis: {Path(midi_path).name}")
        print(f"   Total instruments: {len(midi_data.instruments)}")
        print(f"   Non-drum tracks: {track_count}")
        print(f"   Multitrack: {'Yes' if is_multitrack else 'No'}")

        return is_multitrack, track_count, non_drum_instruments
    except Exception as e:
        print(f"⚠️ Error analyzing MIDI file: {e}")
        return False, 0, []

def is_monophonic_track(instrument, fps: float = 43.066, overlap_threshold: float = 0.01) -> bool:
    """
    Check if a MIDI instrument/track is monophonic (no overlapping notes).

    Args:
        instrument: pretty_midi.Instrument object
        fps: Frames per second for time resolution
        overlap_threshold: Minimum overlap time (seconds) to consider polyphonic

    Returns:
        bool: True if track is monophonic, False if polyphonic
    """
    if not instrument.notes or len(instrument.notes) < 2:
        return True  # No notes or single note = monophonic

    # Sort notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda n: n.start)

    # Check for overlapping notes
    for i in range(len(sorted_notes) - 1):
        current_note = sorted_notes[i]
        next_note = sorted_notes[i + 1]

        # Check if current note ends after next note starts (overlap)
        overlap = current_note.end - next_note.start
        if overlap > overlap_threshold:
            return False  # Found overlapping notes = polyphonic

    return True  # No significant overlaps found = monophonic

def extract_midi_tempo(midi_path: str) -> float:
    """
    Extract tempo from MIDI file, handling non-standard tempo placement.

    Args:
        midi_path: Path to MIDI file

    Returns:
        float: Tempo in BPM (defaults to 120.0 if not found)
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        # Method 1: Check for tempo changes (standard location)
        if hasattr(midi_data, 'tempo_changes') and midi_data.tempo_changes:
            initial_tempo = midi_data.tempo_changes[0].tempo
            print(f"🎵 Found tempo from tempo changes: {initial_tempo:.1f} BPM")
            return initial_tempo

        # Method 2: Estimate from note timing patterns
        all_notes = []
        for inst in midi_data.instruments:
            if not inst.is_drum and inst.notes:
                all_notes.extend([note.start for note in inst.notes])

        if len(all_notes) > 10:  # Need enough notes for estimation
            all_notes.sort()
            intervals = []

            # Look for regular time intervals (beats)
            for i in range(1, min(50, len(all_notes))):  # Check first 50 notes
                interval = all_notes[i] - all_notes[i-1]
                if 0.1 < interval < 2.0:  # Reasonable beat intervals
                    intervals.append(interval)

            if intervals:
                import statistics
                median_interval = statistics.median(intervals)
                # Assume the median interval represents quarter notes
                estimated_tempo = 60.0 / median_interval

                # Clamp to reasonable range
                if 40 <= estimated_tempo <= 200:
                    print(f"🎵 Estimated tempo from note timing: {estimated_tempo:.1f} BPM")
                    return estimated_tempo

        print("🎵 No tempo found, using default: 120.0 BPM")
        return 120.0

    except Exception as e:
        print(f"⚠️ Error extracting tempo: {e}, using default: 120.0 BPM")
        return 120.0

def analyze_track_polyphony(midi_path: str) -> dict:
    """
    Analyze polyphony for each track in a MIDI file.

    Returns:
        dict: {
            'total_tracks': int,
            'monophonic_tracks': int,
            'polyphonic_tracks': int,
            'track_analysis': [{'name': str, 'is_monophonic': bool, 'note_count': int}]
        }
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        non_drum_instruments = [inst for inst in midi_data.instruments if not inst.is_drum]

        track_analysis = []
        monophonic_count = 0

        for i, instrument in enumerate(non_drum_instruments):
            is_mono = is_monophonic_track(instrument)
            track_name = instrument.name if instrument.name else f"Track {i+1}"
            note_count = len(instrument.notes)

            track_analysis.append({
                'name': track_name,
                'is_monophonic': is_mono,
                'note_count': note_count,
                'program': instrument.program
            })

            if is_mono:
                monophonic_count += 1

        total_tracks = len(non_drum_instruments)
        polyphonic_count = total_tracks - monophonic_count

        print(f"🎼 Polyphony Analysis: {Path(midi_path).name}")
        print(f"   Total tracks: {total_tracks}")
        print(f"   Monophonic tracks: {monophonic_count}")
        print(f"   Polyphonic tracks: {polyphonic_count}")

        return {
            'total_tracks': total_tracks,
            'monophonic_tracks': monophonic_count,
            'polyphonic_tracks': polyphonic_count,
            'track_analysis': track_analysis
        }

    except Exception as e:
        print(f"⚠️ Error analyzing track polyphony: {e}")
        return {
            'total_tracks': 0,
            'monophonic_tracks': 0,
            'polyphonic_tracks': 0,
            'track_analysis': []
        }

def render_midi_to_audio(midi_path: str, output_dir: str = "./temp_audio", instrument_group: str = None) -> str:
    """
    Render MIDI file to audio using FluidSynth with appropriate soundfont.
    Args:
        midi_path: Path to MIDI file
        output_dir: Directory to save rendered audio
        instrument_group: Instrument group/subgroup to determine soundfont
    Returns:
        Path to rendered audio file
    """
    midi_path = Path(midi_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename with instrument group
    if instrument_group:
        audio_filename = f"{midi_path.stem}_{instrument_group}_rendered.wav"
    else:
        audio_filename = f"{midi_path.stem}_default_rendered.wav"
    audio_path = output_dir / audio_filename

    # Choose appropriate soundfont based on instrument group
    soundfont_path = INSTRUMENT_SOUNDFONTS.get("default")  # default fallback
    matched_instrument = "default"
    if instrument_group:
        # Check if the instrument group matches any of our specific soundfonts
        for instrument, sf_path in INSTRUMENT_SOUNDFONTS.items():
            if instrument != "default" and instrument.lower() in instrument_group.lower():
                soundfont_path = sf_path
                matched_instrument = instrument
                break

    print(f"🎵 Rendering MIDI to audio: {midi_path.name} -> {audio_filename}")
    if instrument_group:
        print(f"   Instrument group: '{instrument_group}' -> Matched: '{matched_instrument}'")
        print(f"   Using soundfont: {soundfont_path}")

    try:
        # Try fluidsynth first (preferred)
        cmd = [
            "fluidsynth",
            "-ni",  # no interactive mode
            "-g", "0.625",  # gain
            "-r", "44100",  # sample rate (CRITICAL: must match expected rate)
            "-F", str(audio_path),  # output file
            soundfont_path,  # soundfont based on instrument group
            str(midi_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0 and audio_path.exists():
            print(f"✅ FluidSynth rendering successful: {audio_path}")
            return str(audio_path)
        else:
            print(f"⚠️ FluidSynth failed with return code {result.returncode}")
            print(f"   STDOUT: {result.stdout[:200]}")
            print(f"   STDERR: {result.stderr[:200]}")
            print(f"   Trying alternative...")
            # Remove failed file
            if audio_path.exists():
                audio_path.unlink()

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"⚠️ FluidSynth error: {e}, trying alternative...")

    # Fallback: Create sine wave audio from MIDI using pretty_midi
    try:
        print("🔄 Fallback: Creating sine wave audio from MIDI...")
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))

        # Synthesize with sine waves (simple but clean)
        audio_data = midi_data.synthesize(fs=44100, wave=np.sin)

        # Normalize audio
        if audio_data.max() > 0:
            audio_data = audio_data / np.abs(audio_data).max() * 0.9

        # Save as WAV using torchaudio
        audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)  # Add channel dimension
        torchaudio.save(str(audio_path), audio_tensor, 44100)

        print(f"✅ Sine wave synthesis successful: {audio_path}")
        return str(audio_path)

    except Exception as e:
        print(f"❌ All audio rendering methods failed: {e}")
        raise RuntimeError(f"Could not render MIDI to audio: {e}")

def split_midi_into_track_files(midi_path: str, output_dir: str = "./temp_tracks") -> list:
    """
    Split a multitrack MIDI file into separate MIDI files, one per track.
    Returns list of individual MIDI file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    midi_data = pretty_midi.PrettyMIDI(midi_path)
    midi_stem = Path(midi_path).stem

    track_files = []

    for i, instrument in enumerate(midi_data.instruments):
        # Process ALL tracks including drums (user may want to process drum patterns)
        # if instrument.is_drum:
        #     continue  # Skip drum tracks

        # Create a new MIDI file with just this track
        single_track_midi = pretty_midi.PrettyMIDI()
        single_track_midi.instruments.append(instrument)

        # Copy basic timing information
        single_track_midi.resolution = midi_data.resolution

        # Save individual track MIDI file
        # Sanitize track name to remove null bytes and invalid filename characters
        raw_track_name = instrument.name if instrument.name else f"Track_{i+1}"
        # Remove null bytes, control characters, and invalid filename characters
        track_name = "".join(c for c in raw_track_name if c.isprintable() and c not in r'\/:*?"<>|').strip()
        # Fallback if name becomes empty after sanitization
        if not track_name:
            track_name = f"Track_{i+1}"

        track_file = output_dir / f"{midi_stem}_{track_name}_track{i+1}.mid"
        single_track_midi.write(str(track_file))

        track_files.append(str(track_file))
        print(f"   Created track MIDI: {track_file.name}")

    return track_files

def process_multitrack_midi_simple(midi_path: str, progress=None, **generation_args) -> tuple:
    """
    Process multitrack MIDI using the simple approach:
    1. Split MIDI into individual track files
    2. Render each track to FluidSynth audio
    3. Process each audio file individually (like audio upload)
    4. Sum the results

    Returns: (mixed_audio_path, individual_audio_paths, info_text)
    """
    print(f"🎼 Processing multitrack MIDI (simple approach): {Path(midi_path).name}")

    if progress:
        progress(0.1, desc="Splitting MIDI into tracks...")

    # Step 1: Split MIDI into individual track files
    track_midi_files = split_midi_into_track_files(midi_path)
    track_count = len(track_midi_files)

    if track_count == 0:
        raise ValueError("No non-drum tracks found in MIDI file")

    print(f"   Split into {track_count} individual track MIDI files")

    if progress:
        progress(0.2, desc="Rendering tracks to audio...")

    # Step 2: Render each track to FluidSynth audio
    individual_audio_files = []
    for i, track_midi in enumerate(track_midi_files):
        if progress:
            progress(0.2 + (i / track_count) * 0.2, desc=f"Rendering track {i+1}/{track_count}...")

        track_name = Path(track_midi).stem
        print(f"🎵 Rendering track {i+1}: {track_name}")

        # Render this track to audio using existing function
        # Pass the subgroup from generation_args for appropriate soundfont selection
        instrument_subgroup = generation_args.get('subgroup', None)
        track_audio = render_midi_to_audio(track_midi, instrument_group=instrument_subgroup)
        individual_audio_files.append(track_audio)
        print(f"   → {Path(track_audio).name}")

    if progress:
        progress(0.4, desc="Processing individual tracks...")

    # Step 3: Process each audio file individually (like regular audio upload)
    individual_outputs = []
    for i, audio_file in enumerate(individual_audio_files):
        if progress:
            progress(0.4 + (i / track_count) * 0.5, desc=f"Generating from track {i+1}/{track_count}...")

        print(f"🎼 Processing track {i+1} audio: {Path(audio_file).name}")

        # Process this audio file using the standard audio processing pipeline
        # Note: We'll call the regular generation function for this audio file
        track_output = process_single_audio_file(audio_file, **generation_args)
        individual_outputs.append(track_output)
        print(f"   → Generated: {Path(track_output).name}")

    if progress:
        progress(0.9, desc="Summing tracks for mixed output...")

    # Step 4: Sum all the individual outputs
    print("🔄 Summing individual tracks to create mixed output...")
    mixed_audio = sum_audio_files(individual_outputs)

    if progress:
        progress(1.0, desc="Done!")

    info_text = f"Generated {track_count} individual tracks from multitrack MIDI (simple approach)"
    return mixed_audio, individual_outputs, info_text

def process_single_audio_file(audio_file: str, **generation_args) -> str:
    """
    Process a single audio file using the standard audio upload pipeline.
    Returns the path to the generated audio file.
    """
    # Extract conditioning from the audio file
    instrument_subgroup = generation_args.get('subgroup', None)
    extract_formats = generation_args.get('extract_formats', None)
    extraction = extract_conditioning_from_audio(audio_file, instrument_group=instrument_subgroup, extract_formats=extract_formats)
    pr, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=generation_args.get('win_slow', 1024))

    # Get the original audio length for correct timing
    try:
        wav, sr = torchaudio.load(audio_file)
        orig_len = wav.shape[-1]
    except Exception:
        orig_len = None

    # Generate using the standard generate function
    output_path = generate(
        generation_args['MODEL'], pr, amp, rfr, rbd, enc,
        generation_args['group'], generation_args['subgroup'],
        generation_args['steps'], generation_args['seed'],
        generation_args['adapter_scale'], generation_args['cfg_weight'],
        generation_args['t0'], sr_out=32000,
        instrument_strength=generation_args.get('instrument_strength', 1.0),
        inst_boost=generation_args.get('inst_boost', 2.5),
        piano_roll_gain=generation_args.get('piano_roll_gain', 1.0),
        amp_gain=generation_args.get('amp_gain', 1.0),
        rframe_gain=generation_args.get('rframe_gain', 1.0),
        rbend_gain=generation_args.get('rbend_gain', 1.0),
        encodec_gain=generation_args.get('encodec_gain', 1.0),
        use_overlap_decoder=generation_args.get('use_overlap_decoder', True),
        original_audio_length=orig_len,
        pitch_fidelity_boost=generation_args.get('pitch_fidelity_boost', 1.0),
        onset_guidance_boost=generation_args.get('onset_guidance_boost', 2.0),
        pitch_snap_strength=generation_args.get('pitch_snap_strength', 0.5),
        noise_level=generation_args.get('noise_level', 1.0),
        audio_file=audio_file
    )

    return output_path

def sum_audio_files(audio_file_paths: list) -> str:
    """
    Load multiple audio files and sum them to create a mixed output.
    Returns the path to the mixed audio file.
    """
    if not audio_file_paths:
        raise ValueError("No audio files to sum")

    print(f"🔄 Summing {len(audio_file_paths)} audio files...")

    mixed_audio = None
    sample_rate = None

    for i, audio_path in enumerate(audio_file_paths):
        print(f"   Loading file {i+1}: {Path(audio_path).name}")

        # Load audio file
        audio, sr = torchaudio.load(audio_path)
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            print(f"⚠️ Sample rate mismatch: {sr} vs {sample_rate}")

        # Convert to numpy and ensure 1D
        audio_numpy = audio.squeeze().numpy()
        if audio_numpy.ndim > 1:
            # If still multi-dimensional, take first channel
            audio_numpy = audio_numpy[0] if audio_numpy.shape[0] < audio_numpy.shape[1] else audio_numpy.flatten()

        if mixed_audio is None:
            mixed_audio = audio_numpy.copy()
        else:
            # Ensure same length (pad shorter one if needed)
            if len(audio_numpy) != len(mixed_audio):
                max_len = max(len(audio_numpy), len(mixed_audio))
                if len(audio_numpy) < max_len:
                    audio_numpy = np.pad(audio_numpy, (0, max_len - len(audio_numpy)))
                if len(mixed_audio) < max_len:
                    mixed_audio = np.pad(mixed_audio, (0, max_len - len(mixed_audio)))

            mixed_audio += audio_numpy

    # Save the mixed output
    mixed_path = Path("./generated_ui") / f"mixed_{int(time.time())}_multitrack_simple.wav"
    mixed_path.parent.mkdir(exist_ok=True)

    # Ensure mixed_audio is 1D and convert to proper 2D tensor for saving
    if mixed_audio.ndim > 1:
        mixed_audio = mixed_audio.flatten()

    # Convert to tensor with proper shape [channels, samples]
    mixed_tensor = torch.from_numpy(mixed_audio).unsqueeze(0)  # Shape: [1, samples]
    torchaudio.save(str(mixed_path), mixed_tensor, sample_rate)

    print(f"✅ Mixed output saved: {mixed_path.name}")
    return str(mixed_path)

def render_multitrack_debug_audio(track_midi_paths: list, debug_dir: str, audio_stem: str, instrument_group: str = None) -> list:
    """
    Render individual MIDI tracks to audio using FluidSynth for debugging multitrack performance.

    Args:
        track_midi_paths: List of paths to individual track MIDI files
        debug_dir: Directory to save debug audio files
        audio_stem: Base name for audio files
        instrument_group: Instrument group/subgroup to determine soundfont

    Returns:
        list: Paths to rendered audio files
    """
    if not track_midi_paths:
        print("🎵 No MIDI tracks to render for debugging")
        return []

    print(f"🎵 Rendering {len(track_midi_paths)} individual tracks with FluidSynth for debugging...")

    try:
        audio_dir = Path(debug_dir) / "debug_audio"
        audio_dir.mkdir(exist_ok=True)
    except Exception as e:
        print(f"❌ Failed to create debug audio directory: {e}")
        return []

    rendered_audio_paths = []

    for i, midi_path in enumerate(track_midi_paths):
        if not midi_path or not Path(midi_path).exists():
            print(f"   ⚠️ Track {i+1}: MIDI file not found: {midi_path}")
            continue

        midi_file = Path(midi_path)
        track_name = midi_file.stem.replace(f"{audio_stem}_track_", "").replace(f"{audio_stem}_", "")

        print(f"   Rendering track {i+1}: {track_name}")

        try:
            # Use the existing render_midi_to_audio function which handles FluidSynth properly
            rendered_audio = render_midi_to_audio(
                midi_path,
                output_dir=str(audio_dir),
                instrument_group=instrument_group
            )

            if rendered_audio and Path(rendered_audio).exists():
                print(f"   ✅ Track {i+1} rendered successfully")
                rendered_audio_paths.append(rendered_audio)
            else:
                print(f"   ❌ Track {i+1} rendering failed")

        except Exception as e:
            print(f"   ❌ Track {i+1} rendering error: {e}")

    print(f"🎵 Successfully rendered {len(rendered_audio_paths)}/{len(track_midi_paths)} tracks")
    return rendered_audio_paths

def midi_to_piano_roll_conditioning(midi_path: str, window_slow: int = 1024, fps: float = 43.066, tempo_override: float = None) -> tuple:
    """
    Convert MIDI file directly to piano roll conditioning (no other conditioning).
    Args:
        midi_path: Path to MIDI file
        window_slow: Target length for conditioning
        fps: Frames per second for piano roll
    Returns:
        tuple: (piano_roll, empty_amp, empty_rframe, empty_rbend, empty_encodec)
    """
    print(f"🎼 Converting MIDI to piano roll conditioning: {Path(midi_path).name}")

    # Load MIDI
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    if not midi_data.instruments:
        raise ValueError("MIDI file contains no instruments")

    # Extract tempo for logging only (not used for timing calculations)
    # CRITICAL: pretty_midi already accounts for tempo in note.start and note.end times
    # DO NOT apply tempo adjustment - it would double-apply the tempo!
    if tempo_override is not None:
        detected_tempo = tempo_override
        print(f"🎵 Tempo override specified: {detected_tempo:.1f} BPM (NOTE: not applied, pretty_midi handles tempo)")
    else:
        detected_tempo = extract_midi_tempo(midi_path)
        print(f"🎵 Detected tempo: {detected_tempo:.1f} BPM (NOTE: already applied by pretty_midi)")

    # Use fps directly - DO NOT adjust for tempo!
    # pretty_midi.get_end_time() and note.start/note.end are already in real-time seconds
    print(f"🎵 Using fps={fps:.3f} (no tempo adjustment needed)")

    # Get duration and create time grid with standard fps
    duration = max(midi_data.get_end_time(), 1.0)  # At least 1 second
    time_steps = int(duration * fps) + 1

    # Create piano roll
    piano_roll = np.zeros((128, time_steps))

    # Convert all instruments to piano roll (merge them)
    total_notes = 0
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue  # Skip drum tracks

        for note in instrument.notes:
            # CRITICAL FIX: note.start and note.end are already in seconds (tempo-adjusted by pretty_midi)
            # Just multiply by fps to get frame numbers
            start_frame = int(note.start * fps)
            end_frame = int(note.end * fps)
            start_frame = max(0, min(start_frame, time_steps - 1))
            end_frame = max(start_frame + 1, min(end_frame, time_steps))

            # Use velocity for intensity (normalized to 0-1)
            intensity = note.velocity / 127.0
            piano_roll[note.pitch, start_frame:end_frame] = intensity
            total_notes += 1

    print(f"✅ Created piano roll: {piano_roll.shape}, {total_notes} notes, {duration:.2f}s")

    # Resize to exact target window
    original_frames = piano_roll.shape[1]
    target_length = window_slow  # Use exact window_slow for consistent sizing

    if piano_roll.shape[1] < target_length:
        # Pad if too short
        pad_width = target_length - piano_roll.shape[1]
        piano_roll = np.pad(piano_roll, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        print(f"🎵 Padded piano roll from {original_frames} to {target_length} frames")
    elif piano_roll.shape[1] > target_length:
        # Crop if too long
        piano_roll = piano_roll[:, :target_length]
        print(f"🎵 Cropped piano roll from {original_frames} to {target_length} frames")

    # Create empty conditioning for other modalities (match piano roll length)
    final_length = piano_roll.shape[1]
    empty_amp = np.zeros(final_length)
    empty_rframe = np.zeros(final_length)
    empty_rbend = np.zeros(final_length)

    # Create minimal encodec tokens (all zeros - will be ignored if encodec_gain=0)
    encodec_length = final_length // 4  # Typical encodec downsampling
    empty_encodec = torch.zeros((1, 8, encodec_length), dtype=torch.long)

    print(f"✅ MIDI conditioning ready: PR={piano_roll.shape}, others empty")

    return piano_roll, empty_amp, empty_rframe, empty_rbend, empty_encodec

def midi_to_multitrack_piano_rolls(midi_path: str, window_slow: int = 1024, fps: float = 43.066, tempo_override: Optional[float] = None) -> dict:
    """
    Convert multitrack MIDI file to separate piano rolls for each track/voice.
    Args:
        midi_path: Path to MIDI file
        window_slow: Target length for conditioning
        fps: Frames per second for piano roll
    Returns:
        dict: {
            'track_piano_rolls': [piano_roll_per_track],
            'track_info': [{'name': str, 'program': int, 'note_count': int}],
            'combined_piano_roll': combined_piano_roll,
            'empty_conditioning': (empty_amp, empty_rframe, empty_rbend, empty_encodec)
        }
    """
    print(f"🎼 Converting multitrack MIDI to separate piano rolls: {Path(midi_path).name}")

    # Load MIDI and check if multitrack
    is_multi, track_count, non_drum_instruments = is_multitrack_midi(midi_path)

    # Extract tempo for logging only (not used for timing calculations)
    # CRITICAL: pretty_midi already accounts for tempo in note.start and note.end times
    if tempo_override is not None:
        original_tempo = tempo_override
        print(f"🎵 Tempo override specified: {original_tempo:.1f} BPM (NOTE: not applied, pretty_midi handles tempo)")
    else:
        original_tempo = extract_midi_tempo(midi_path)
        print(f"🎵 Detected tempo: {original_tempo:.1f} BPM (NOTE: already applied by pretty_midi)")

    # Analyze polyphony for each track
    polyphony_analysis = analyze_track_polyphony(midi_path)

    if not is_multi:
        # Fallback to single track processing
        print("   Single track detected, using standard processing")
        piano_roll, empty_amp, empty_rframe, empty_rbend, empty_encodec = midi_to_piano_roll_conditioning(midi_path, window_slow, fps, tempo_override=tempo_override)
        return {
            'track_piano_rolls': [piano_roll],
            'track_info': [{'name': 'Track 1', 'program': 0, 'note_count': int(np.sum(piano_roll > 0.1))}],
            'combined_piano_roll': piano_roll,
            'empty_conditioning': (empty_amp, empty_rframe, empty_rbend, empty_encodec)
        }

    # Process multitrack MIDI
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    duration = max(midi_data.get_end_time(), 1.0)

    # CRITICAL FIX: DO NOT apply tempo adjustment!
    # pretty_midi already handles tempo in get_end_time() and note.start/note.end
    print(f"🎵 Using fps={fps:.3f} (no tempo adjustment needed)")

    time_steps = int(duration * fps) + 1

    track_piano_rolls = []
    track_info = []
    combined_piano_roll = np.zeros((128, time_steps))

    print(f"   Processing {track_count} tracks, duration: {duration:.2f}s")

    for i, instrument in enumerate(non_drum_instruments):
        # Create piano roll for this track
        track_piano_roll = np.zeros((128, time_steps))
        note_count = 0

        for note in instrument.notes:
            # CRITICAL FIX: note.start and note.end are already in seconds (tempo-adjusted)
            start_frame = int(note.start * fps)
            end_frame = int(note.end * fps)
            start_frame = max(0, min(start_frame, time_steps - 1))
            end_frame = max(start_frame + 1, min(end_frame, time_steps))

            # Use velocity for intensity
            intensity = note.velocity / 127.0
            track_piano_roll[note.pitch, start_frame:end_frame] = intensity
            combined_piano_roll[note.pitch, start_frame:end_frame] = intensity
            note_count += 1

        # Note: Individual track resizing will be done after all tracks are processed
        # to ensure consistent length across all tracks

        track_piano_rolls.append(track_piano_roll)

        # Get track info with polyphony status
        track_name = instrument.name if instrument.name else f"Track {i+1}"

        # Find polyphony info for this track
        is_monophonic = False
        if i < len(polyphony_analysis['track_analysis']):
            is_monophonic = polyphony_analysis['track_analysis'][i]['is_monophonic']

        track_info.append({
            'name': track_name,
            'program': instrument.program,
            'note_count': note_count,
            'is_monophonic': is_monophonic
        })

        mono_status = "monophonic" if is_monophonic else "polyphonic"
        print(f"   Track {i+1}: {track_name} (Program {instrument.program}) - {note_count} notes [{mono_status}]")

    # Resize all piano rolls (individual tracks and combined) consistently
    # Use standard window_slow (1024) for model compatibility
    original_frames = combined_piano_roll.shape[1]
    target_length = window_slow  # Use model training length

    # Resize combined piano roll (truncate or pad to target_length)
    if combined_piano_roll.shape[1] != target_length:
        if combined_piano_roll.shape[1] > target_length:
            # Truncate to target length
            combined_piano_roll = combined_piano_roll[:, :target_length]
        else:
            # Pad to target length
            pad_width = target_length - combined_piano_roll.shape[1]
            combined_piano_roll = np.pad(combined_piano_roll, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    # Resize all individual track piano rolls to match (truncate or pad to target_length)
    for i, track_pr in enumerate(track_piano_rolls):
        if track_pr.shape[1] != target_length:
            if track_pr.shape[1] > target_length:
                # Truncate to target length
                track_piano_rolls[i] = track_pr[:, :target_length]
            else:
                # Pad to target length
                pad_width = target_length - track_pr.shape[1]
                track_piano_rolls[i] = np.pad(track_pr, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    # Create empty conditioning for other modalities (match final length)
    final_length = target_length
    empty_amp = np.zeros(final_length)
    empty_rframe = np.zeros(final_length)
    empty_rbend = np.zeros(final_length)
    encodec_length = final_length // 4
    empty_encodec = torch.zeros((1, 8, encodec_length), dtype=torch.long)

    print(f"✅ Multitrack processing complete: {len(track_piano_rolls)} tracks processed")

    return {
        'track_piano_rolls': track_piano_rolls,
        'track_info': track_info,
        'combined_piano_roll': combined_piano_roll,
        'empty_conditioning': (empty_amp, empty_rframe, empty_rbend, empty_encodec),
        'polyphony_analysis': polyphony_analysis,
        'original_tempo': original_tempo
    }

# ------------------------------------------------------------------------------
# MIDI Generation Functions (from ac.py midigen feature)
# ------------------------------------------------------------------------------
SOUNDFONT_PATH = "/home/arlo/.local/lib/python3.9/site-packages/pretty_midi/TimGM6mb.sf2"
MIDI_CHORD_FOLDER = '/home/arlo/harmonymodule/output/01 - C Major - A minor/4 Progression/Major'

def ensure_midi_above_c2(midi_path: str) -> str:
    """
    Ensure all notes in a MIDI file are at or above C3 (MIDI note 48).
    Transposes any notes below C3 up by octaves until they reach C3 or above.
    Modifies the MIDI file in place.

    Args:
        midi_path: Path to MIDI file to check and fix

    Returns:
        str: Same midi_path (file modified in place)
    """
    C2_MIDI_NOTE = 48  # C3 minimum
    midi = pretty_midi.PrettyMIDI(midi_path)

    notes_transposed = 0
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue  # Skip drum tracks

        for note in instrument.notes:
            original_pitch = note.pitch
            while note.pitch < C2_MIDI_NOTE and note.pitch + 12 <= 127:
                note.pitch += 12
                notes_transposed += 1

            # Critical warning if still below C3
            if note.pitch < C2_MIDI_NOTE:
                print(f"⚠️⚠️⚠️ CRITICAL: Note at {note.pitch} cannot be transposed above C3 (would exceed MIDI 127)")

    if notes_transposed > 0:
        print(f"🎵 ensure_midi_above_c2: Transposed {notes_transposed} notes up to meet C3 minimum")
        # Save the corrected MIDI back
        midi.write(midi_path)

    # Verify and log final pitch range
    all_pitches = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            all_pitches.extend([n.pitch for n in instrument.notes])

    if all_pitches:
        min_pitch = min(all_pitches)
        max_pitch = max(all_pitches)
        if min_pitch < C2_MIDI_NOTE:
            print(f"⚠️ WARNING: MIDI still has notes below C3 (min: {min_pitch})")
        else:
            print(f"✅ MIDI pitch range verified: {min_pitch}-{max_pitch} (C3+ requirement met)")

    return midi_path

def get_random_transposed_midi_wav(tempo: int = 80, skip_wav: bool = False, target_key: str = 'C'):
    """
    Select a random MIDI chord progression, transpose it to target key, and set its tempo.
    Returns paths to both MIDI and WAV files.

    Args:
        tempo: BPM for the MIDI file
        skip_wav: If True, skip FluidSynth WAV rendering (for fast mode)
        target_key: Target key to transpose to (e.g., 'C', 'C#', 'D', etc.)
    """
    output_midi = f'/tmp/transposed_output_{os.getpid()}_{int(time.time())}.mid'
    output_wav = f'/tmp/transposed_output_{os.getpid()}_{int(time.time())}.wav'

    midi_files = [f for f in os.listdir(MIDI_CHORD_FOLDER) if f.endswith('.mid')]
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {MIDI_CHORD_FOLDER}")

    midi_file = random.choice(midi_files)
    midi_path = os.path.join(MIDI_CHORD_FOLDER, midi_file)

    # Calculate semitone transposition from C to target key
    key_to_semitones = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }
    semitone_shift = key_to_semitones.get(target_key, 0)

    print(f"🎼 Selected MIDI: {midi_file} at {tempo} BPM, transposing to {target_key} ({semitone_shift:+d} semitones)")

    # Load and process with pretty_midi
    midi = pretty_midi.PrettyMIDI(midi_path)

    # Log original pitches for debugging
    original_pitches = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            original_pitches.append(note.pitch)
    if original_pitches:
        print(f"   Original pitch range: {min(original_pitches)}-{max(original_pitches)}")

    # Transpose all notes by semitone_shift
    if semitone_shift != 0:
        for instrument in midi.instruments:
            for note in instrument.notes:
                note.pitch = max(0, min(127, note.pitch + semitone_shift))

        # Log transposed pitches for debugging
        transposed_pitches = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                transposed_pitches.append(note.pitch)
        if transposed_pitches:
            print(f"   After transposition ({semitone_shift:+d}): {min(transposed_pitches)}-{max(transposed_pitches)}")

    # Ensure no notes are below C3 (MIDI note 48)
    C2_MIDI_NOTE = 48  # C3 minimum
    notes_transposed_up = 0
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue  # Skip drum tracks
        for note in instrument.notes:
            original_pitch = note.pitch
            # Transpose up octaves until note is at or above C3
            while note.pitch < C2_MIDI_NOTE and note.pitch + 12 <= 127:
                note.pitch += 12
                print(f"      ⬆️  Transposed note from {note.pitch - 12} to {note.pitch}")
            if note.pitch != original_pitch:
                notes_transposed_up += 1
            # Final check: if still below C3, warn
            if note.pitch < C2_MIDI_NOTE:
                print(f"      ⚠️⚠️⚠️ CRITICAL: Note at {note.pitch} is STILL below C3 after transposition!")

    if notes_transposed_up > 0:
        print(f"   🎵 Transposed {notes_transposed_up} notes up to avoid going below C3")

    # Log final pitches after C2 check
    final_pitches = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            final_pitches.append(note.pitch)
    if final_pitches:
        print(f"   Final pitch range before saving: {min(final_pitches)}-{max(final_pitches)}")

    temp_mid_path = f'/tmp/temp_pretty_{os.getpid()}_{int(time.time())}.mid'
    midi.write(temp_mid_path)

    # Reopen with mido to inject fixed tempo
    mid = MidiFile(temp_mid_path)
    new_mid = MidiFile()
    tempo_meta = MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0)

    for i, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        if i == 0:
            new_track.append(tempo_meta)
        for msg in track:
            if not (msg.type == 'set_tempo'):
                new_track.append(msg)
        new_mid.tracks.append(new_track)

    new_mid.save(output_midi)

    # Convert to WAV (skip in fast mode)
    if not skip_wav:
        subprocess.run([
            "fluidsynth", "-T", "wav", "-g", "0.625", "-r", "44100", "-F", output_wav, SOUNDFONT_PATH, output_midi
        ], check=True)
    else:
        print(f"   ⚡ Fast mode: Skipping FluidSynth WAV rendering")
        output_wav = None  # No WAV generated

    return output_midi, output_wav


def compute_best_tempos(scene_changes: list) -> list:
    """
    Compute optimal tempo for each scene to align scene boundaries with musical bars.
    """
    MIN_TEMPO = 70
    MAX_TEMPO = 160
    MAX_TEMPO_JUMP = 20

    tempos = []

    for i in range(len(scene_changes)):
        if i == len(scene_changes) - 1:
            break
        duration = scene_changes[i+1] - scene_changes[i]
        best = None
        best_score = float('inf')

        for bpm in range(MIN_TEMPO, MAX_TEMPO + 1):
            seconds_per_beat = 60 / bpm
            beats = duration / seconds_per_beat

            # Prefer durations that land near exact beats
            residual = abs(round(beats) - beats)
            full_bars = beats / 4
            bar_residual = abs(round(full_bars) - full_bars)

            score = residual + (bar_residual * 0.5)

            # Penalize large tempo jumps
            if tempos:
                jump = abs(bpm - tempos[-1])
                if jump > MAX_TEMPO_JUMP:
                    score += (jump - MAX_TEMPO_JUMP) * 0.3

            if score < best_score:
                best = bpm
                best_score = score

        tempos.append(best)

    return tempos


def apply_automation_to_amp_signal(
    amp_signal: np.ndarray,
    automation_points: list,
    total_duration: float,
    fps: float = 43.066
) -> np.ndarray:
    """
    Apply volume automation envelope to the amp conditioning signal.

    Args:
        amp_signal: Amplitude conditioning array [channels, frames] or [frames]
        automation_points: List of (time_seconds, volume) tuples
        total_duration: Total duration in seconds
        fps: Frames per second of conditioning (43.066 for ACE-Step)

    Returns:
        Modified amp signal with automation applied
    """
    if not automation_points or len(automation_points) == 0:
        print("⚠️  No automation points, returning original amp signal")
        return amp_signal

    # Ensure amp_signal is at least 2D
    original_shape = amp_signal.shape
    is_1d = len(amp_signal.shape) == 1
    if is_1d:
        amp_signal = amp_signal[np.newaxis, :]  # Add channel dimension
        print(f"   Converting 1D amp signal to 2D: {original_shape} -> {amp_signal.shape}")

    # Sort automation points by time
    sorted_automation = sorted(automation_points, key=lambda x: x[0])

    print(f"🎛 Applying automation to amp signal:")
    print(f"   Amp shape: {amp_signal.shape}")
    print(f"   Duration: {total_duration:.2f}s")
    print(f"   FPS: {fps}")
    print(f"   Automation points: {len(sorted_automation)}")

    # Create automation envelope at conditioning frame rate
    num_frames = amp_signal.shape[-1]
    automation_envelope = np.ones(num_frames, dtype=np.float32)

    # Interpolate automation values for each frame
    for frame_idx in range(num_frames):
        time_at_frame = (frame_idx / num_frames) * total_duration

        # Find surrounding automation points
        before_point = None
        after_point = None

        for i, (t, v) in enumerate(sorted_automation):
            if t <= time_at_frame:
                before_point = (t, v)
            if t > time_at_frame and after_point is None:
                after_point = (t, v)
                break

        # Interpolate volume
        if before_point is None and after_point is None:
            # No automation points at all (shouldn't happen)
            volume = 1.0
        elif before_point is None:
            # Before first point
            volume = after_point[1]
        elif after_point is None:
            # After last point
            volume = before_point[1]
        else:
            # Interpolate between points
            t1, v1 = before_point
            t2, v2 = after_point
            if t2 > t1:
                alpha = (time_at_frame - t1) / (t2 - t1)
                volume = v1 + alpha * (v2 - v1)
            else:
                volume = v1

        automation_envelope[frame_idx] = volume

    # Apply envelope to amp signal
    # amp_signal is shape [channels, frames]
    # Expand envelope to match channels
    envelope_expanded = np.tile(automation_envelope, (amp_signal.shape[0], 1))
    amp_modified = amp_signal * envelope_expanded

    # Print statistics
    print(f"   Envelope range: {automation_envelope.min():.3f} - {automation_envelope.max():.3f}")
    print(f"   Envelope mean: {automation_envelope.mean():.3f}")
    print(f"   ✅ Automation applied to amp signal")

    # Return in original shape
    if is_1d:
        amp_modified = amp_modified.squeeze(0)  # Remove channel dimension if input was 1D
        print(f"   Returning 1D amp signal: {amp_modified.shape}")

    return amp_modified


def apply_automation_gain_envelope(
    audio_path: str,
    automation_points: list,
    output_path: str,
    sample_rate: int = 44100
):
    """
    Apply volume automation as a gain envelope to rendered audio.

    Args:
        audio_path: Path to audio file
        automation_points: List of (time_seconds, volume) tuples
        output_path: Where to save the processed audio
        sample_rate: Audio sample rate
    """
    if not automation_points or len(automation_points) == 0:
        print("   ℹ️  No automation points - copying audio unchanged")
        shutil.copy(audio_path, output_path)
        return

    print(f"\n🎛️ APPLYING GAIN ENVELOPE TO AUDIO")
    print(f"   Input: {Path(audio_path).name}")
    print(f"   Automation points: {len(automation_points)}")

    # Load audio
    audio, sr = torchaudio.load(audio_path)
    audio_duration = audio.shape[-1] / sr

    # Create gain envelope
    num_samples = audio.shape[-1]
    gain_envelope = np.ones(num_samples, dtype=np.float32)

    # Sort automation points by time
    sorted_points = sorted(automation_points, key=lambda x: x[0])

    # Log automation ramp levels
    print(f"   Audio duration: {audio_duration:.2f}s")
    print(f"   Gain envelope ramps:")

    for i in range(len(sorted_points) - 1):
        t1, v1 = sorted_points[i]
        t2, v2 = sorted_points[i + 1]

        # Convert times to sample indices
        sample1 = int(t1 * sr)
        sample2 = int(t2 * sr)

        # Clamp to valid range
        sample1 = max(0, min(sample1, num_samples - 1))
        sample2 = max(0, min(sample2, num_samples - 1))

        if sample2 > sample1:
            # Create linear ramp between points
            num_ramp_samples = sample2 - sample1
            ramp = np.linspace(v1, v2, num_ramp_samples, dtype=np.float32)
            gain_envelope[sample1:sample2] = ramp

            # Log this ramp
            print(f"      {t1:.2f}s (gain {v1:.2f}) → {t2:.2f}s (gain {v2:.2f})")

    # Handle before first point and after last point
    if sorted_points[0][0] > 0:
        sample_idx = int(sorted_points[0][0] * sr)
        gain_envelope[:sample_idx] = sorted_points[0][1]
        print(f"      0.00s → {sorted_points[0][0]:.2f}s: constant gain {sorted_points[0][1]:.2f}")

    if sorted_points[-1][0] < audio_duration:
        sample_idx = int(sorted_points[-1][0] * sr)
        gain_envelope[sample_idx:] = sorted_points[-1][1]
        print(f"      {sorted_points[-1][0]:.2f}s → {audio_duration:.2f}s: constant gain {sorted_points[-1][1]:.2f}")

    # Apply gain envelope to audio
    gain_tensor = torch.from_numpy(gain_envelope).float()

    # Handle both mono and stereo
    if audio.shape[0] == 1:
        audio_processed = audio * gain_tensor.unsqueeze(0)
    else:
        # Stereo - apply to both channels
        audio_processed = audio * gain_tensor.unsqueeze(0).expand_as(audio)

    # Save processed audio
    torchaudio.save(output_path, audio_processed, sr)
    print(f"   ✅ Saved gain-automated audio: {Path(output_path).name}\n")

def apply_automation_to_midi(
    midi_path: str,
    scene_start: float,
    scene_duration: float,
    track_automation: list,
    output_path: str,
    total_duration: float,
    scene_tempo: float
):
    """
    Apply volume automation to MIDI notes within a specific scene window.
    Time values in track_automation are normalized 0-1 relative to scene duration.

    This modifies MIDI velocities, which will then be further scaled by the
    gain envelope applied to the FluidSynth render.
    """
    # 1. Convert normalized automation times to absolute seconds
    absolute_automation = [
        (scene_start + (t * scene_duration), v)
        for t, v in track_automation
    ]

    # 2. Ensure coverage of full scene duration
    if not absolute_automation:
        absolute_automation = [
            (scene_start, 0.5),
            (scene_start + scene_duration, 0.5)
        ]
    else:
        # Add start point if missing
        if absolute_automation[0][0] > scene_start:
            absolute_automation.insert(0, (scene_start, absolute_automation[0][1]))
        # Add end point if missing
        if absolute_automation[-1][0] < scene_start + scene_duration:
            absolute_automation.append((scene_start + scene_duration, absolute_automation[-1][1]))

    print(f"🎛 Scene {scene_start:.1f}-{scene_start+scene_duration:.1f}s automation:")
    for t, v in absolute_automation:
        print(f"  {t:.2f}s: {v:.2f}")

    # 3. Process MIDI with tempo and automation
    mid = MidiFile(midi_path)
    new_mid = MidiFile()
    ticks_per_beat = mid.ticks_per_beat

    for track in mid.tracks:
        new_track = MidiTrack()
        abs_time = 0.0  # Tracks absolute time in seconds

        # Inject scene tempo at start
        new_track.append(MetaMessage(
            'set_tempo',
            tempo=mido.bpm2tempo(scene_tempo),
            time=0
        ))
        for msg in track:
            # Skip original tempo messages
            if msg.type == 'set_tempo':
                continue

            # Convert delta time to seconds
            delta_seconds = mido.tick2second(msg.time, ticks_per_beat, mido.bpm2tempo(scene_tempo))
            abs_time += delta_seconds

            # CRITICAL FIX: The incoming MIDI starts at time=0 (it was just generated for this scene)
            # We should keep ALL notes and only check if we're within the scene DURATION
            # Map abs_time (0-based) to scene_time (scene_start-based) for automation lookup
            scene_time = scene_start + abs_time

            if abs_time <= scene_duration:
                # Apply volume to note_on messages using scene_time for automation
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Find surrounding automation points
                    before = [p for p in absolute_automation if p[0] <= scene_time]
                    after = [p for p in absolute_automation if p[0] > scene_time]

                    # Calculate current volume
                    if before and after:
                        prev_time, prev_vol = max(before, key=lambda x: x[0])
                        next_time, next_vol = min(after, key=lambda x: x[0])
                        if next_time > prev_time:
                            ratio = (scene_time - prev_time) / (next_time - prev_time)
                            volume = prev_vol + ratio * (next_vol - prev_vol)
                        else:
                            volume = prev_vol
                    elif before:
                        volume = before[-1][1]
                    else:
                        volume = after[0][1] if after else 0.5

                    # Scale to MIDI velocity (1-127)
                    new_velocity = max(1, min(127, int(volume * 127)))
                    msg = msg.copy(velocity=new_velocity)

                new_track.append(msg)

        new_mid.tracks.append(new_track)

    new_mid.save(output_path)
    print(f"✅ Saved automated MIDI: {output_path}")


def concatenate_midi_scenes(
    scene_midi_paths: dict,
    scene_durations: list,
    output_path: str,
    soundfont_path: str = None
) -> str:
    """
    Concatenate multiple scene MIDIs into one long MIDI file.
    Each scene MIDI is trimmed/padded to match its exact duration.

    Args:
        scene_midi_paths: Dict mapping scene_idx -> midi_file_path
        scene_durations: List of scene durations in seconds
        output_path: Where to save the concatenated MIDI
        soundfont_path: Optional soundfont for debug WAV rendering

    Returns:
        Path to concatenated MIDI file
    """
    from mido import Message

    combined_midi = MidiFile(ticks_per_beat=480)
    combined_track = MidiTrack()
    combined_midi.tracks.append(combined_track)

    print(f"\n🎼 Concatenating {len(scene_midi_paths)} scene MIDIs")
    cumulative_duration = 0.0
    cumulative_ticks = 0

    for scene_idx in sorted(scene_midi_paths.keys()):
        midi_path = scene_midi_paths[scene_idx]
        duration_sec = scene_durations[scene_idx]

        print(f"\n{'─'*60}")
        print(f"🎬 Scene {scene_idx}: Duration = {duration_sec:.2f}s")
        print(f"   MIDI path: {midi_path}")
        print(f"   Concatenation point: {cumulative_duration:.3f}s → {cumulative_duration + duration_sec:.3f}s")
        print(f"   Cumulative ticks before: {cumulative_ticks}")

        # Load scene MIDI
        scene_midi = MidiFile(midi_path)

        # Get tempo from first track
        tempo = 500000  # Default 120 BPM
        for msg in scene_midi.tracks[0]:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

        bpm = mido.tempo2bpm(tempo)
        print(f"   Tempo: {bpm:.1f} BPM")

        # CRITICAL FIX: Add tempo change message at the START of each scene
        # For scene 0: time=0 (first message)
        # For scene 1+: time=0 (happens immediately after last note of previous scene)
        if scene_idx == 0:
            combined_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        else:
            # Insert tempo change with time=0 (relative to end of previous scene)
            combined_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
            print(f"   🎵 Inserted tempo change: {bpm:.1f} BPM at scene boundary")

        # Calculate max ticks for this scene
        ticks_per_second = bpm * combined_midi.ticks_per_beat / 60.0
        max_ticks = int(duration_sec * ticks_per_second)
        print(f"   Max ticks for scene: {max_ticks} ({duration_sec:.3f}s at {ticks_per_second:.1f} ticks/s)")
        current_ticks = 0

        # Track if this is the first message of this scene
        first_message_in_scene = True
        messages_copied = 0

        # Copy messages from scene MIDI - MERGE all tracks into one
        for track in scene_midi.tracks:
            for msg in track:
                if msg.is_meta and msg.type == 'set_tempo':
                    continue  # Skip tempo messages (already added)

                if msg.is_meta:
                    continue  # Skip other meta messages

                # Limit message time to stay within scene duration
                msg_time = msg.time

                # CRITICAL FIX: First message of each scene (except scene 0) should have time=0
                # This ensures the new scene starts immediately after the previous one
                if first_message_in_scene:
                    if scene_idx > 0:
                        print(f"   🎯 First message of scene {scene_idx}: type={msg.type}, setting time=0 (was {msg.time})")
                        msg_time = 0
                    first_message_in_scene = False

                if current_ticks + msg_time > max_ticks:
                    msg_time = max(0, max_ticks - current_ticks)

                msg_copy = msg.copy(time=msg_time)
                if hasattr(msg_copy, 'channel'):
                    msg_copy.channel = 0  # Force channel 0

                # Ensure no notes below C2 (MIDI note 36)
                if hasattr(msg_copy, 'note') and msg_copy.type in ['note_on', 'note_off']:
                    C2_MIDI_NOTE = 36
                    while msg_copy.note < C2_MIDI_NOTE and msg_copy.note < 127 - 12:
                        msg_copy.note += 12

                combined_track.append(msg_copy)
                current_ticks += msg_time
                messages_copied += 1

                if current_ticks >= max_ticks:
                    print(f"   ✅ Reached max_ticks after {messages_copied} messages")
                    break
            # Don't break between tracks - continue copying from all tracks
            # if current_ticks >= max_ticks:
            #     break

        print(f"   📊 Copied {messages_copied} messages from {len(scene_midi.tracks)} tracks")

        # Pad with silence if scene ended early
        if current_ticks < max_ticks:
            padding_ticks = max_ticks - current_ticks
            combined_track.append(Message('note_off', note=0, velocity=0, time=padding_ticks, channel=0))
            print(f"   ⚠️ Padded with {padding_ticks} ticks ({padding_ticks/ticks_per_second:.3f}s)")
            current_ticks = max_ticks

        # Flush all notes at scene boundary
        for note in range(128):
            combined_track.append(Message('note_off', note=note, velocity=0, time=0, channel=0))

        # Update cumulative counters
        cumulative_ticks += current_ticks
        cumulative_duration += duration_sec
        actual_duration = current_ticks / ticks_per_second
        print(f"   ✅ Scene {scene_idx} complete:")
        print(f"      Ticks copied: {current_ticks} (target: {max_ticks})")
        print(f"      Actual duration: {actual_duration:.3f}s (target: {duration_sec:.3f}s)")
        print(f"      Cumulative ticks: {cumulative_ticks}")
        print(f"      Cumulative duration: {cumulative_duration:.3f}s")

        combined_track.append(Message('control_change', control=123, value=0, time=0, channel=0))

    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 CONCATENATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total scenes concatenated: {len(scene_midi_paths)}")
    print(f"Final cumulative duration: {cumulative_duration:.3f}s")
    print(f"Final cumulative ticks: {cumulative_ticks}")
    print(f"{'='*60}\n")

    # Save concatenated MIDI
    combined_midi.save(output_path)
    print(f"✅ Saved concatenated MIDI: {output_path}")

    # Render debug WAV with selected soundfont
    if soundfont_path:
        debug_wav_path = output_path.replace('.mid', '_debug.wav')
        print(f"\n🎼 Rendering debug WAV with soundfont: {Path(soundfont_path).name}")
        try:
            subprocess.run([
                "fluidsynth", "-ni", "-g", "0.625", "-r", "44100", "-F", debug_wav_path,
                soundfont_path, output_path
            ], check=True, capture_output=True)
            print(f"✅ Debug WAV saved: {debug_wav_path}")
        except Exception as e:
            print(f"⚠️ Failed to render debug WAV: {e}")

    # Return MIDI path (no rendering - will be handled by caller with selected soundfont)
    return output_path

# ------------------------------------------------------------------------------
# Conditioning I/O
# ------------------------------------------------------------------------------
def extract_conditioning_from_audio_fast_mode(audio_path: str, output_dir: str = "./extracted_conditioning", instrument_group: str = None, variant: str = "zero") -> dict:
    """
    Fast mode: Extract limited conditioning based on variant.
    - "zero": Only extract piano roll
    - "encodec": Only extract encodec tokens
    This significantly speeds up conditioning extraction at the cost of less detailed control.
    """
    # Convert mp3/m4a/other formats to wav if needed
    audio_ext = Path(audio_path).suffix.lower()
    if audio_ext in ['.mp3', '.m4a', '.aac', '.mp4']:
        print(f"🔄 Converting {audio_ext} to wav using pydub...")
        from pydub import AudioSegment

        # Load audio
        audio = AudioSegment.from_file(audio_path)

        # Create temp wav file
        temp_wav = str(Path(audio_path).with_suffix('.wav'))
        audio.export(temp_wav, format='wav')

        print(f"✅ Converted to: {temp_wav}")
        audio_path = temp_wav

    # Truncate audio to MAX duration to prevent extraction failures
    MAX_AUDIO_DURATION = MAX_WINDOW_SLOW / 43.066  # ~47.5 seconds
    import torchaudio
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_duration = waveform.shape[-1] / sample_rate

        if audio_duration > MAX_AUDIO_DURATION:
            print(f"⚠️  WARNING: Audio duration ({audio_duration:.2f}s) exceeds maximum ({MAX_AUDIO_DURATION:.2f}s)")
            print(f"   Truncating audio to {MAX_AUDIO_DURATION:.2f}s before extraction...")

            # Truncate waveform
            max_samples = int(MAX_AUDIO_DURATION * sample_rate)
            waveform = waveform[:, :max_samples]

            # Save truncated version
            truncated_path = str(Path(audio_path).parent / f"truncated_{Path(audio_path).name}")
            torchaudio.save(truncated_path, waveform, sample_rate)
            audio_path = truncated_path
            print(f"✅ Saved truncated audio to: {truncated_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not check/truncate audio duration: {e}")
        print(f"   Proceeding with original file...")

    # Check memory cache first (include instrument group and variant in cache key)
    cache_context = f"instrument_{instrument_group}_fast_{variant}" if instrument_group else f"no_instrument_fast_{variant}"
    cache_key = _get_file_cache_key(audio_path, cache_context)
    print(f"⚡ Fast mode ({variant.upper()}) extraction - Cache key for {Path(audio_path).name}: {cache_key[:8]}...")

    if cache_key in CONDITIONING_CACHE:
        cached_result = CONDITIONING_CACHE[cache_key]
        if _is_cache_valid(cached_result, audio_path):
            print(f"✅ Using cached fast-mode conditioning from memory for: {Path(audio_path).name}")
            return cached_result
        else:
            del CONDITIONING_CACHE[cache_key]

    # Check manifest paths (only need piano roll for fast mode)
    rec = _find_manifest_record_by_audio(audio_path)
    if rec:
        paths = {}
        prp = rec.get("piano_roll_path") or rec.get("pianoroll_path")
        c = rec.get("conditioning_paths", {}) or {}
        pr_path = prp or c.get("piano_roll") or c.get("pianoroll")

        # Ensure piano_roll is a string path, not a numpy array
        if pr_path and isinstance(pr_path, str) and os.path.exists(pr_path):
            paths["piano_roll"] = pr_path
            result = {"paths": paths}
            CONDITIONING_CACHE[cache_key] = result
            print("✅ Fast mode: Using piano roll from manifest.")
            return result

    # Check disk cache (only need piano roll)
    base_stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in Path(audio_path).stem)[:128] or "audio"
    dir_stem = f"{base_stem}_{instrument_group}" if instrument_group else base_stem
    out_dir = Path(output_dir) / dir_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    pr_file = out_dir / f"{base_stem}.pianoroll.npy"

    if pr_file.exists():
        result = {"dir": str(out_dir), "stem": base_stem}
        CONDITIONING_CACHE[cache_key] = result
        print(f"✅ Fast mode: Using disk-cached piano roll: {pr_file}")
        return result

    # Extract based on variant
    if variant == "encodec":
        # Extract only encodec tokens
        print(f"⚡ Fast mode (ENCODEC): Extracting only encodec tokens for: {Path(audio_path).name}")
        encodec_file = out_dir / f"{base_stem}.encodec.pt"

        if not encodec_file.exists():
            try:
                import torchaudio
                # Extract encodec tokens using the full extraction pipeline
                # We'll call the normal extraction but only save encodec
                full_result = extract_conditioning_from_audio(audio_path, output_dir, instrument_group)
                # The encodec file should now exist from the full extraction
                if encodec_file.exists():
                    print(f"✅ Fast mode (ENCODEC): Encodec tokens extracted")
                else:
                    print(f"⚠️ Fast mode (ENCODEC): Encodec file not found after extraction")

            except Exception as e:
                print(f"⚠️ Fast mode (ENCODEC) extraction failed: {e}")
                print(f"Falling back to full extraction...")
                return extract_conditioning_from_audio(audio_path, output_dir, instrument_group)

        result = {"dir": str(out_dir), "stem": base_stem}
        CONDITIONING_CACHE[cache_key] = result
        print(f"✅ Fast mode (ENCODEC): Using encodec from {encodec_file}")
        return result

    else:  # variant == "zero" or default
        # Extract only piano roll using Basic Pitch
        print(f"⚡ Fast mode (ZERO): Extracting only piano roll for: {Path(audio_path).name}")
        try:
            import basic_pitch
            from basic_pitch.inference import predict
            import librosa

            # Run Basic Pitch to get MIDI (pass file path, not audio array)
            model_output, midi_data, note_events = predict(
                audio_path,
                onset_threshold=0.55,      # Moderately stricter onset detection (reduces false onsets from vibrato)
                frame_threshold=0.35,      # Moderately stronger activation required (smooths out weak pitch variations)
                minimum_note_length=188    # Moderate note duration filtering (filters rapid vibrato fluctuations)
            )

            # Convert to piano roll (128 x T format)
            # model_output contains onset, contour, and note activations
            # We'll use the note activations
            piano_roll_88 = model_output['note'].T  # Transpose to (88, T) - Basic Pitch outputs 88 keys (MIDI 21-108)

            # Pad from 88 keys to 128 MIDI notes (full range 0-127)
            # Basic Pitch covers MIDI notes 21-108 (88 keys)
            # We need to pad: 21 zeros at bottom (MIDI 0-20) and 19 zeros at top (MIDI 109-127)
            T = piano_roll_88.shape[1]
            piano_roll = np.zeros((128, T), dtype=piano_roll_88.dtype)
            piano_roll[21:109, :] = piano_roll_88  # Insert 88 keys at MIDI positions 21-108

            print(f"   Padded piano roll from {piano_roll_88.shape} to {piano_roll.shape}")

            # Save piano roll
            np.save(pr_file, piano_roll)

            result = {"dir": str(out_dir), "stem": base_stem}
            CONDITIONING_CACHE[cache_key] = result
            print(f"✅ Fast mode (ZERO): Piano roll extracted and saved to {pr_file}")
            return result

        except Exception as e:
            print(f"⚠️ Fast mode extraction failed: {e}")
            print(f"Falling back to full extraction...")
            # Fall back to full extraction if fast mode fails
            return extract_conditioning_from_audio(audio_path, output_dir, instrument_group)

def extract_conditioning_from_audio(audio_path: str, output_dir: str = "./extracted_conditioning", instrument_group: str = None, extract_formats: list = None) -> dict:
    # Convert mp3/m4a/other formats to wav if needed (soundfile only supports wav/flac/ogg natively)
    audio_ext = Path(audio_path).suffix.lower()
    if audio_ext in ['.mp3', '.m4a', '.aac', '.mp4']:
        print(f"🔄 Converting {audio_ext} to wav using pydub...")
        from pydub import AudioSegment

        # Load audio
        audio = AudioSegment.from_file(audio_path)

        # Create temp wav file
        temp_wav = str(Path(audio_path).with_suffix('.wav'))
        audio.export(temp_wav, format='wav')

        print(f"✅ Converted to: {temp_wav}")
        audio_path = temp_wav

    # Truncate audio to MAX duration to prevent extraction failures
    MAX_AUDIO_DURATION = MAX_WINDOW_SLOW / 43.066  # ~47.5 seconds
    import torchaudio
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_duration = waveform.shape[-1] / sample_rate

        if audio_duration > MAX_AUDIO_DURATION:
            print(f"⚠️  WARNING: Audio duration ({audio_duration:.2f}s) exceeds maximum ({MAX_AUDIO_DURATION:.2f}s)")
            print(f"   Truncating audio to {MAX_AUDIO_DURATION:.2f}s before extraction...")

            # Truncate waveform
            max_samples = int(MAX_AUDIO_DURATION * sample_rate)
            waveform = waveform[:, :max_samples]

            # Save truncated version
            truncated_path = str(Path(audio_path).parent / f"truncated_{Path(audio_path).name}")
            torchaudio.save(truncated_path, waveform, sample_rate)
            audio_path = truncated_path
            print(f"✅ Saved truncated audio to: {truncated_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not check/truncate audio duration: {e}")
        print(f"   Proceeding with original file...")

    # Check memory cache first (include instrument group in cache key)
    cache_context = f"instrument_{instrument_group}" if instrument_group else "no_instrument"
    cache_key = _get_file_cache_key(audio_path, cache_context)
    print(f"🔍 Cache key for {Path(audio_path).name} ({cache_context}): {cache_key[:8]}...")
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

    # Check disk cache (include instrument group in directory name)
    base_stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in Path(audio_path).stem)[:128] or "audio"
    dir_stem = f"{base_stem}_{instrument_group}" if instrument_group else base_stem
    out_dir = Path(output_dir) / dir_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    req = [out_dir/f"{base_stem}.pianoroll.npy", out_dir/f"{base_stem}.amp.npy", out_dir/f"{base_stem}.rframe.npy",
           out_dir/f"{base_stem}.rbend.npy", out_dir/f"{base_stem}.encodec.pt"]
    if all(x.exists() for x in req):
        result = {"dir": str(out_dir), "stem": base_stem}
        CONDITIONING_CACHE[cache_key] = result
        print(f"✅ Using disk-cached conditioning: {out_dir}")
        return result

    # Extract new conditioning
    print(f"🔄 Extracting conditioning for: {Path(audio_path).name}")
    cmd = ["python", "test_extract_local.py", "--input", str(audio_path), "--output", str(out_dir)]

    # Add formats parameter if specified
    if extract_formats:
        formats_str = ",".join(extract_formats)
        cmd.extend(["--formats", formats_str])
        print(f"📋 Extracting formats: {formats_str}")

    print(f"Running extraction: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        print(res.stdout); print(res.stderr)
        raise RuntimeError("Extraction failed.")

    result = {"dir": str(out_dir), "stem": base_stem}
    CONDITIONING_CACHE[cache_key] = result
    print("✅ Conditioning extracted successfully.")
    return result

def _np_load_first(*candidates):
    for p in candidates:
        if p is not None and os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError(f"None of: {candidates}")

def midi_to_piano_roll_direct(midi_path: str, window_slow: int, fps: float = 43.066):
    """
    Convert MIDI file directly to piano roll without rendering to audio.
    Uses pretty_midi which correctly handles tempo changes to get note times in seconds.
    Used in fast mode to skip expensive FluidSynth rendering and conditioning extraction.
    """
    import pretty_midi

    print(f"⚡ Fast mode: Converting MIDI to piano roll directly (respecting tempo changes)")

    # Load MIDI with pretty_midi - it correctly parses tempo changes
    # and converts all note times to seconds
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Get MIDI duration (accounts for tempo changes)
    midi_duration = pm.get_end_time()

    # Log tempo changes for debugging
    if hasattr(pm, '_tick_scales') and len(pm._tick_scales) > 0:
        print(f"   Found {len(pm._tick_scales)} tempo changes in pretty_midi")
        for i, (tick, scale) in enumerate(pm._tick_scales[:5]):  # Show first 5
            time_sec = pm.tick_to_time(tick)
            # Calculate BPM from tick_scale (scale = seconds per tick)
            # BPM = 60 / (seconds_per_beat) = 60 / (ticks_per_beat * seconds_per_tick)
            if scale > 0:
                bpm = 60.0 / (pm.resolution * scale)
                print(f"     {time_sec:.2f}s: {bpm:.1f} BPM (tick {tick})")
    else:
        print(f"   ⚠️ No tempo changes found in pretty_midi, using default 120 BPM")

    # Also log note count and first few note times for debugging
    total_notes = sum(len(inst.notes) for inst in pm.instruments if not inst.is_drum)
    print(f"   MIDI duration: {midi_duration:.2f}s, {total_notes} notes")
    if total_notes > 0:
        first_note = next((note for inst in pm.instruments if not inst.is_drum for note in inst.notes[:1]), None)
        if first_note:
            print(f"   First note: pitch {first_note.pitch} at {first_note.start:.3f}s - {first_note.end:.3f}s")

    # Create empty piano roll
    piano_roll = np.zeros((128, window_slow), dtype=np.float32)

    # Iterate through all instruments and notes
    # pretty_midi has already converted note.start and note.end to seconds accounting for tempo
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue  # Skip drum tracks

        for note in instrument.notes:
            # Convert seconds to frame numbers
            start_frame = int(note.start * fps)
            end_frame = int(note.end * fps)

            # Clamp to valid range
            start_frame = max(0, min(start_frame, window_slow - 1))
            end_frame = max(0, min(end_frame, window_slow - 1))

            # Fill piano roll for this note's duration
            # Use velocity as intensity (normalized 0-1)
            intensity = note.velocity / 127.0
            for f in range(start_frame, end_frame + 1):
                if f < window_slow:
                    piano_roll[note.pitch, f] = max(piano_roll[note.pitch, f], intensity)

    print(f"   Piano roll shape: {piano_roll.shape}")
    print(f"   Target window_slow: {window_slow} frames")

    # Count active notes for debugging
    active_frames = np.sum(np.any(piano_roll > 0, axis=0))
    print(f"   Active frames: {active_frames}/{window_slow} ({100*active_frames/window_slow:.1f}%)")

    return piano_roll

def automation_points_to_amp_array(automation_points: list, duration: float, fps: float = 43.066):
    """
    Convert automation points [(time, volume), ...] to amp array at specified FPS.
    Args:
        automation_points: List of (time, volume) tuples
        duration: Total duration in seconds
        fps: Frame rate (frames per second)
    Returns:
        amp array with linear interpolation between points
    """
    window_slow = clamp_window_slow(int(duration * fps), duration, fps)
    amp = np.zeros(window_slow, dtype=np.float32)

    if not automation_points or len(automation_points) == 0:
        # No automation - return constant 0.5
        return np.full(window_slow, 0.5, dtype=np.float32)

    # Sort by time
    sorted_points = sorted(automation_points, key=lambda p: p[0])

    # Linear interpolation between points
    for i in range(window_slow):
        time_sec = i / fps

        # Find surrounding points
        before_point = None
        after_point = None

        for point in sorted_points:
            if point[0] <= time_sec:
                before_point = point
            elif after_point is None:
                after_point = point
                break

        # Interpolate
        if before_point is None:
            amp[i] = sorted_points[0][1]
        elif after_point is None:
            amp[i] = before_point[1]
        else:
            # Linear interpolation with exponential curve (25% more exponential)
            t0, v0 = before_point
            t1, v1 = after_point
            alpha = (time_sec - t0) / (t1 - t0) if t1 > t0 else 0
            # Apply exponential curve: alpha^1.25 for 25% more exponential response
            alpha_curved = alpha ** 1.25
            amp[i] = v0 + alpha_curved * (v1 - v0)

    return amp

def generate_conditioning_from_midi_fast(midi_path: str, window_slow: int, fps: float = 43.066, automation_points: list = None, duration: float = None, variant: str = "synthetic"):
    """
    Fast mode: Generate conditioning directly from MIDI without FluidSynth rendering.

    Args:
        midi_path: Path to MIDI file
        window_slow: Number of conditioning frames
        fps: Frames per second (display rate, will be converted to 10.766 Hz in generate())
        automation_points: Optional list of (time, volume) tuples for amp conditioning
        duration: Duration in seconds (required if automation_points provided)
        variant: "zero" (zeros for rframe/rbend/encodec), "encodec" (zeros - can't extract from MIDI), or "synthetic" (synthetic generation)
    Returns: (piano_roll, amp, rframe, rbend, encodec_tokens)
    """
    print(f"⚡ FAST MODE ({variant.upper()}): Converting MIDI to piano roll")
    print(f"   MIDI file: {Path(midi_path).name}")
    print(f"   window_slow: {window_slow}, fps: {fps}, duration: {duration}")

    # Convert MIDI to piano roll (pretty_midi handles tempo changes correctly)
    piano_roll = midi_to_piano_roll_direct(midi_path, window_slow, fps)

    # amp: Use automation envelope if provided
    if automation_points and duration is not None:
        print(f"   Using automation envelope for amp conditioning ({len(automation_points)} points)")
        amp = automation_points_to_amp_array(automation_points, duration, fps)
        # CRITICAL: Ensure amp matches window_slow (may differ due to rounding)
        if amp.shape[0] != window_slow:
            print(f"   ⚠️  WARNING: amp length {amp.shape[0]} != window_slow {window_slow}, resizing...")
            # Resize to match
            if amp.shape[0] < window_slow:
                # Pad with last value
                padding = np.full((window_slow - amp.shape[0],), amp[-1] if len(amp) > 0 else 0.5, dtype=np.float32)
                amp = np.concatenate([amp, padding])
            else:
                # Truncate
                amp = amp[:window_slow]
            print(f"   ✅ Resized amp to {amp.shape[0]}")
    elif variant == "zero" or variant == "encodec":
        # Fast Zero/Encodec: Use zeros for amp (no automation)
        print(f"   Fast {variant.capitalize()}: Using zeros for amp (no automation)")
        amp = np.zeros((window_slow,), dtype=np.float32)
    else:
        # Fast Synthetic: Use preset value
        print(f"   Fast Synthetic: Using preset amp value (0.5)")
        amp = np.full((window_slow,), 0.5, dtype=np.float32)

    # Load MIDI for note information
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Variant-specific conditioning generation
    if variant == "zero":
        # FAST ZERO: Zeros for rframe, rbend, encodec
        print(f"   Fast Zero: Using zeros for rframe, rbend, encodec")
        rfr = np.zeros((window_slow,), dtype=np.float32)
        rbd = np.zeros((window_slow,), dtype=np.float32)

        # encodec: Zero tokens
        encodec_length = window_slow // 2
        enc = torch.zeros((1, 8, encodec_length), dtype=torch.long)
        print(f"   ✅ Fast Zero conditioning: all zeros (relying on piano roll + automation)")

    elif variant == "synthetic":
        # FAST SYNTHETIC: Generate synthetic rframe/rbend/encodec

        # rfr: Generate SYNTHETIC rframe (VOICED MASK) from MIDI notes
        print(f"   🎯 Generating synthetic rframe (voiced mask) from MIDI notes")
        rfr = np.zeros((window_slow,), dtype=np.float32)

        # Create voiced mask: 1.0 during notes, 0.0 during silence
        note_count = 0
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                # Convert note times to frames
                start_frame = int(note.start * fps)
                end_frame = int(note.end * fps)

                # Set voiced mask to 1.0 for entire note duration
                for frame in range(start_frame, min(end_frame, window_slow)):
                    if 0 <= frame < window_slow:
                        rfr[frame] = 1.0
                note_count += 1

        print(f"   ✅ Generated voiced mask from {note_count} notes: {np.sum(rfr > 0.5)}/{window_slow} frames voiced ({100*np.sum(rfr > 0.5)/window_slow:.1f}%)")

        # rbd: Generate SYNTHETIC rbend (pitch bend in semitones relative to A4=440Hz)
        print(f"   🎯 Generating synthetic rbend (pitch bend) from MIDI notes")
        rbd = np.zeros((window_slow,), dtype=np.float32)

        # For each note, compute pitch bend from MIDI pitch
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                start_frame = int(note.start * fps)
                end_frame = int(note.end * fps)

                # MIDI pitch to frequency: f = 440 * 2^((pitch - 69)/12)
                # So: rbend = pitch - 69 (semitones relative to A4)
                rbend_value = float(note.pitch - 69)

                # Set rbend for entire note duration
                for frame in range(start_frame, min(end_frame, window_slow)):
                    if 0 <= frame < window_slow:
                        # If multiple notes overlap, use the first one (or average)
                        if rbd[frame] == 0.0:
                            rbd[frame] = rbend_value

        rbend_active = np.sum(np.abs(rbd) > 0.1)
        print(f"   ✅ Generated rbend: {rbend_active}/{window_slow} frames with pitch info ({100*rbend_active/window_slow:.1f}%)")

        # encodec: Generate SYNTHETIC tokens with temporal patterns
        encodec_length = window_slow // 2
        print(f"   🎯 Generating synthetic encodec tokens with temporal patterns")
        enc = torch.zeros((1, 8, encodec_length), dtype=torch.long)

        # Generate synthetic codes that vary with musical events
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                # Encodec frame at half FPS
                onset_enc_frame = int(note.start * fps / 2)
                offset_enc_frame = int(note.end * fps / 2)

                if 0 <= onset_enc_frame < encodec_length:
                    # Create temporal markers in encodec tokens
                    pitch_code = (note.pitch % 64) * 16  # Map to 0-1008 range

                    # Set tokens for this note's duration
                    for enc_frame in range(onset_enc_frame, min(offset_enc_frame + 1, encodec_length)):
                        for codebook in range(8):
                            # Vary by codebook and frame for temporal structure
                            variation = (codebook * 100 + (enc_frame - onset_enc_frame) * 10) % 200
                            enc[0, codebook, enc_frame] = min(pitch_code + variation, 1023)

        nonzero_tokens = torch.sum(enc > 0).item()
        print(f"   ✅ Generated synthetic encodec: {enc.shape}, non-zero: {nonzero_tokens}/{8*encodec_length} ({100*nonzero_tokens/(8*encodec_length):.1f}%)")

    elif variant == "encodec":
        # FAST ENCODEC: For MIDI-based generation, we can't extract encodec from audio
        # So we'll use zeros for encodec (same as zero mode)
        print(f"   Fast Encodec: Cannot extract encodec from MIDI, using zeros")
        print(f"   Note: Encodec variant is designed for audio input, not MIDI generation")
        rfr = np.zeros((window_slow,), dtype=np.float32)
        rbd = np.zeros((window_slow,), dtype=np.float32)

        # encodec: Zero tokens
        encodec_length = window_slow // 2
        enc = torch.zeros((1, 8, encodec_length), dtype=torch.long)
        print(f"   ✅ Fast Encodec (MIDI mode): all zeros (encodec extraction requires audio input)")

    else:
        raise ValueError(f"Unknown variant: {variant}. Must be 'zero', 'encodec', or 'synthetic'")

    # Summary logging
    encodec_length = window_slow // 2
    print(f"⚡ Fast mode ({variant}) conditioning shapes:")
    print(f"   Piano roll: {piano_roll.shape}")
    if automation_points:
        print(f"   Amp: {amp.shape} (from automation)")
    elif variant == "zero" or variant == "encodec":
        print(f"   Amp: {amp.shape} (zeros)")
    else:
        print(f"   Amp: {amp.shape} (preset 0.5)")
    print(f"   Rframe: {rfr.shape}, active: {np.sum(rfr > 0.5)}/{window_slow} ({100*np.sum(rfr > 0.5)/max(1,window_slow):.1f}%)")
    print(f"   Rbend: {rbd.shape}, active: {np.sum(np.abs(rbd) > 0.1)}/{window_slow} ({100*np.sum(np.abs(rbd) > 0.1)/max(1,window_slow):.1f}%)")
    print(f"   Encodec: {enc.shape}, non-zero: {torch.sum(enc > 0).item()}/{8*encodec_length} ({100*torch.sum(enc > 0).item()/max(1,8*encodec_length):.1f}%)")

    # VERIFY all arrays match expected length
    if piano_roll.shape[1] != window_slow:
        print(f"   ❌ ERROR: piano_roll length {piano_roll.shape[1]} != window_slow {window_slow}")
    if amp.shape[0] != window_slow:
        print(f"   ❌ ERROR: amp length {amp.shape[0]} != window_slow {window_slow}")
    if rfr.shape[0] != window_slow:
        print(f"   ❌ ERROR: rfr length {rfr.shape[0]} != window_slow {window_slow}")

    return piano_roll, amp, rfr, rbd, enc

def load_conditioning_fast_mode(extraction: dict, window_slow: int, variant: str = "zero"):
    """
    Fast mode: Load limited conditioning based on variant.
    - "zero": Only load piano roll, use preset/zeros for everything else
    - "encodec": Only load encodec tokens, use zeros for everything else
    """
    print(f"⚡ Fast mode ({variant.upper()}): Loading conditioning")

    # Pad/trim helper (ORIGINAL CODE)
    def _pad_arr(x, L):
        if x.shape[-1] >= L: return x[..., :L]
        pad = [(0,0)]*(x.ndim-1) + [(0, L - x.shape[-1])]
        return np.pad(x, pad, mode="constant")

    if variant == "encodec":
        # ENCODEC variant: Load only encodec tokens, zero everything else
        print(f"   Loading only encodec tokens, zeroing piano_roll/amp/rframe/rbend")

        # Load encodec
        if "paths" in extraction:
            paths = extraction["paths"]
            enc_data = torch.load(paths.get("encodec"), map_location="cpu")
        else:
            out_dir = Path(extraction["dir"]); stem = extraction["stem"]
            nested = out_dir / stem
            enc_path = None
            for p in [out_dir/f"{stem}.encodec.pt", nested/f"{stem}.encodec.pt"]:
                if p.exists():
                    enc_path = p
                    break
            if enc_path is None:
                raise FileNotFoundError(f"Encodec file not found for {stem}")
            enc_data = torch.load(enc_path, map_location="cpu")

        # Extract encodec tokens (use same logic as load_conditioning for consistency)
        if isinstance(enc_data, (list, tuple)):
            enc = None
            for obj in (enc_data, len(enc_data) and enc_data[0], len(enc_data) and isinstance(enc_data[0], (list,tuple)) and enc_data[0][0]):
                if torch.is_tensor(obj):
                    enc = obj
                    break
            if enc is None:
                raise RuntimeError(f"Unrecognized encodec token structure: {type(enc_data)}")
        else:
            enc = enc_data

        # Ensure correct shape [1, 8, T]
        if enc.ndim == 2:
            enc = enc.unsqueeze(0)

        # Pad/trim encodec to match window_slow//2
        encodec_length = window_slow // 2
        if enc.shape[-1] < encodec_length:
            pad_amt = encodec_length - enc.shape[-1]
            enc = torch.nn.functional.pad(enc, (0, pad_amt), value=0)
        elif enc.shape[-1] > encodec_length:
            enc = enc[:, :, :encodec_length]

        # Create zeros for everything else
        pr = np.zeros((128, window_slow), dtype=np.float32)  # Zero piano roll (128 = full MIDI range)
        amp = np.zeros((window_slow,), dtype=np.float32)     # Zero amp
        rfr = np.zeros((window_slow,), dtype=np.float32)     # Zero rframe
        rbd = np.zeros((window_slow,), dtype=np.float32)     # Zero rbend

        print(f"   PR (zeros): {pr.shape}")
        print(f"   Amp (zeros): {amp.shape}")
        print(f"   Rframe (zeros): {rfr.shape}")
        print(f"   Rbend (zeros): {rbd.shape}")
        print(f"   Encodec (loaded): {enc.shape}")

    else:  # variant == "zero" or default
        # ZERO variant: Load only piano roll from extraction
        if "paths" in extraction:
            paths = extraction["paths"]
            pr = _np_load_first(paths.get("piano_roll"), paths.get("pianoroll"))
        else:
            out_dir = Path(extraction["dir"]); stem = extraction["stem"]
            nested = out_dir / stem
            pr = _np_load_first(out_dir/f"{stem}.pianoroll.npy", out_dir/f"{stem}.piano_roll.npy",
                                 nested/f"{stem}.pianoroll.npy", nested/f"{stem}.piano_roll.npy")

        # If piano roll is 88 keys (MIDI 21-108), pad to 128
        if pr.shape[0] == 88:
            print(f"   ⚠️ Piano roll has 88 keys, padding to 128 MIDI notes")
            T = pr.shape[1]
            pr_padded = np.zeros((128, T), dtype=pr.dtype)
            pr_padded[21:109, :] = pr  # Insert 88 keys at MIDI positions 21-108
            pr = pr_padded

        pr = _pad_arr(pr, window_slow)

        # Create preset values for amp, rfr, rbd (stable/default values)
        # NOTE: amp, rframe, rbend should be 1D arrays of shape (window_slow,), not (128, window_slow)
        # The 128 dimension is only for piano_roll (MIDI note pitches)

        # amp: amplitude envelope - use constant moderate value (0.5)
        amp = np.full((window_slow,), 0.5, dtype=np.float32)

        # rfr: rframe (timbre/roughness) - use zeros (neutral)
        rfr = np.zeros((window_slow,), dtype=np.float32)

        # rbd: rbend (pitch bend) - use zeros (no pitch bend)
        rbd = np.zeros((window_slow,), dtype=np.float32)

        # encodec: use zeros - [1, 8, window_slow//2] (typical encodec shape with 8 codebooks)
        # Encodec downsamples by factor of 2
        encodec_length = window_slow // 2
        enc = torch.zeros((1, 8, encodec_length), dtype=torch.long)

        print(f"   PR (loaded): {pr.shape}")
        print(f"   Amp (preset): {amp.shape}")
        print(f"   Rframe (zeros): {rfr.shape}")
        print(f"   Rbend (zeros): {rbd.shape}")
        print(f"   Encodec (zeros): {enc.shape}")

    return pr, amp, rfr, rbd, enc

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

    # pad/trim to window_slow (ORIGINAL CODE - no interpolation)
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
def extract_ground_truth_latents(audio_path: str, model: Pipeline, target_duration: float = None) -> torch.Tensor:
    """Extract ground truth latents from audio using the DCAE encoder."""
    # Check latent cache first (include target_duration in cache key if provided)
    cache_context = f"duration_{target_duration:.3f}" if target_duration else "no_duration"
    cache_key = _get_file_cache_key(audio_path, cache_context)
    if cache_key in LATENT_CACHE:
        print(f"✅ Using cached ground truth latents for: {Path(audio_path).name}")
        return LATENT_CACHE[cache_key]

    try:
        print(f"🔄 Extracting ground truth latents for: {Path(audio_path).name}")
        if target_duration:
            print(f"   Target duration: {target_duration:.3f}s")

        # Load and preprocess audio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to stereo if mono
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # Pad or crop to target duration if specified
        if target_duration is not None:
            target_samples = int(target_duration * sr)
            current_samples = waveform.shape[-1]

            if current_samples < target_samples:
                # Pad with zeros
                pad_size = target_samples - current_samples
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))
                print(f"   Padded audio: {current_samples} → {target_samples} samples")
            elif current_samples > target_samples:
                # Crop
                waveform = waveform[:, :target_samples]
                print(f"   Cropped audio: {current_samples} → {target_samples} samples")

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
# MIDI conversion and voice separation
# ------------------------------------------------------------------------------
def piano_roll_to_midi(piano_roll, output_path, fps=43.066, program=0, velocity=80, min_note_duration=0.1, tempo=120.0):
    """
    Convert piano roll to MIDI file.
    Args:
        piano_roll: numpy array of shape [128, T] representing MIDI piano roll
        output_path: path to save MIDI file
        fps: frames per second (matches conditioning extraction fps)
        program: MIDI program number (instrument)
        velocity: MIDI velocity
        min_note_duration: minimum note duration in seconds to filter false detections
        tempo: BPM tempo for the MIDI file
    Returns:
        path to saved MIDI file
    """
    # Create a PrettyMIDI object with specified tempo
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # Create an instrument instance
    instrument = pretty_midi.Instrument(program=program)

    # Convert piano roll to notes
    notes_added = 0
    notes_filtered = 0

    for pitch in range(128):
        # Find note onsets and offsets
        note_events = piano_roll[pitch] > 0.1  # threshold for note detection

        if not np.any(note_events):
            continue

        # Find transitions
        diff = np.diff(np.concatenate(([False], note_events, [False])).astype(int))
        onsets = np.where(diff == 1)[0]
        offsets = np.where(diff == -1)[0]

        # Create notes with length filtering
        for onset, offset in zip(onsets, offsets):
            start_time = onset / fps
            end_time = offset / fps
            note_duration = end_time - start_time

            # Filter out very short notes (false detections)
            if note_duration < min_note_duration:
                notes_filtered += 1
                continue

            # Ensure minimum duration for valid notes
            if note_duration < 0.05:
                end_time = start_time + 0.05

            note = pretty_midi.Note(
                velocity=int(velocity * piano_roll[pitch, onset:offset].mean()),
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)
            notes_added += 1

    # Add instrument to MIDI
    midi.instruments.append(instrument)

    # CRITICAL FIX: Ensure MIDI file has correct duration
    # Calculate expected duration from piano roll
    expected_duration = piano_roll.shape[1] / fps

    # If MIDI has notes, check if we need to extend duration
    if len(instrument.notes) > 0:
        last_note_time = max(note.end for note in instrument.notes)
        if last_note_time < expected_duration - 0.1:  # More than 0.1s shorter
            # Add a silent marker note at the end to preserve duration
            marker_note = pretty_midi.Note(
                velocity=1,  # Very quiet
                pitch=0,  # Lowest pitch
                start=expected_duration - 0.01,
                end=expected_duration
            )
            instrument.notes.append(marker_note)
            print(f"   📏 Added end marker: MIDI {last_note_time:.2f}s → {expected_duration:.2f}s")

    # Save MIDI file
    midi.write(str(output_path))
    print(f"✅ Saved MIDI: {output_path} ({notes_added} notes, filtered {notes_filtered} short notes)")
    print(f"   Duration: {expected_duration:.2f}s ({piano_roll.shape[1]} frames @ {fps} fps)")
    return str(output_path)

def save_basic_pitch_midi_with_voices(audio_file, subgroup=None, progress=None, tempo=120.0, separate_voices=True, monophonic=False):
    """
    Save Basic Pitch MIDI from conditioning extraction with optional voice separation.
    Args:
        audio_file: input audio file path
        subgroup: instrument subgroup for soundfont selection
        progress: optional progress callback
        tempo: BPM tempo for output MIDI files
        separate_voices: whether to separate into individual voices (default: True)
        monophonic: if True, use stricter thresholds to extract only one dominant voice (default: False)
    Returns:
        dict with main MIDI path and voice MIDI paths (empty if separate_voices=False)
    """
    if audio_file is None:
        raise gr.Error("Please upload an audio file first.")

    if progress:
        progress(0, desc="Extracting conditioning...")

    # Get actual audio duration
    import torchaudio
    try:
        wav, sr = torchaudio.load(audio_file)
        audio_duration = wav.shape[-1] / sr
        print(f"🎵 Audio duration: {audio_duration:.2f}s ({wav.shape[-1]} samples at {sr}Hz)")
    except Exception as e:
        print(f"⚠️ Could not determine audio duration: {e}, using default")
        audio_duration = 10.0  # fallback

    # Extract conditioning (which includes Basic Pitch piano roll)
    extraction = extract_conditioning_from_audio(audio_file)

    # Calculate window_slow based on actual audio duration (NOT hardcoded!)
    fps = 43.066
    win_slow = clamp_window_slow(int(audio_duration * fps), audio_duration, fps)
    print(f"🎵 Using window_slow = {win_slow} frames for {audio_duration:.2f}s audio")

    pr, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=win_slow)

    # Apply monophonic filtering if requested
    if monophonic:
        print(f"🎵 Applying monophonic filtering (keeping only dominant voice)...")
        # For each time frame, keep only the loudest note
        monophonic_pr = np.zeros_like(pr)
        for t in range(pr.shape[1]):
            frame = pr[:, t]
            if frame.max() > 0:
                # Find the loudest note in this frame
                loudest_pitch = np.argmax(frame)
                monophonic_pr[loudest_pitch, t] = frame[loudest_pitch]
        pr = monophonic_pr
        print(f"   ✅ Filtered to monophonic (single voice)")

    if progress:
        progress(0.3, desc="Creating output directory...")

    # CRITICAL FIX: Always create output in /home/arlo/Data/miditest/ for debugging
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    audio_stem = Path(audio_file).stem
    debug_dir = Path("/home/arlo/Data/miditest")
    debug_dir.mkdir(exist_ok=True)
    output_dir = debug_dir / f"{timestamp}_{audio_stem}"
    voices_dir = output_dir / "voices"
    output_dir.mkdir(parents=True, exist_ok=True)
    voices_dir.mkdir(exist_ok=True)

    # Also create the original location for backward compatibility
    original_output_dir = Path("./midi_exports") / f"{timestamp}_{audio_stem}"
    original_voices_dir = original_output_dir / "voices"
    original_output_dir.mkdir(parents=True, exist_ok=True)
    original_voices_dir.mkdir(exist_ok=True)

    if progress:
        progress(0.5, desc="Saving main MIDI...")

    # Save main MIDI file (both locations)
    # CRITICAL FIX: Basic Pitch piano roll is at 44100/4096 = 10.766 fps, NOT 43.066!
    basic_pitch_fps = 44100 / 4096  # = 10.766 fps
    print(f"🎵 Converting piano roll to MIDI using correct fps: {basic_pitch_fps:.3f}")

    main_midi_path = output_dir / f"{audio_stem}_basicpitch.mid"
    piano_roll_to_midi(pr, main_midi_path, fps=basic_pitch_fps, tempo=tempo)

    # Also save to original location
    original_main_midi = original_output_dir / f"{audio_stem}_basicpitch.mid"
    piano_roll_to_midi(pr, original_main_midi, fps=basic_pitch_fps, tempo=tempo)

    # Voice separation is optional (skip if separate_voices=False)
    voice_midi_paths = []
    debug_audio_paths = []
    voice_count = 0

    if separate_voices:
        if progress:
            progress(0.6, desc="Separating voices...")

        # Separate voices using existing function
        voices = separate_piano_roll_voices(pr)
        voice_count = len(voices)

        if progress:
            progress(0.8, desc="Saving voice MIDI files...")

        # Save individual voice MIDI files with note length filtering (both locations)
        for i, voice_pr in enumerate(voices):
            # Debug location (primary)
            voice_path = voices_dir / f"{audio_stem}_voice_{i+1}.mid"
            piano_roll_to_midi(voice_pr, voice_path, fps=basic_pitch_fps, program=0, velocity=80, min_note_duration=0.1, tempo=tempo)
            voice_midi_paths.append(str(voice_path))

            # Original location (backup)
            original_voice_path = original_voices_dir / f"{audio_stem}_voice_{i+1}.mid"
            piano_roll_to_midi(voice_pr, original_voice_path, fps=basic_pitch_fps, program=0, velocity=80, min_note_duration=0.1, tempo=tempo)

        if progress:
            progress(0.9, desc="Rendering FluidSynth debug audio for voices...")

        # Render FluidSynth debug audio for each voice
        debug_audio_paths = render_multitrack_debug_audio(voice_midi_paths, str(output_dir), audio_stem, instrument_group=subgroup)

        print(f"🎼 Saved {len(voices)} voice MIDI files to: {voices_dir}")
    else:
        print(f"🎼 Voice separation skipped (separate_voices=False)")

    if progress:
        progress(1.0, desc="Done!")

    result = {
        "main_midi": str(main_midi_path),
        "voice_midis": voice_midi_paths,
        "debug_audio_paths": debug_audio_paths,
        "output_dir": str(output_dir),
        "voice_count": voice_count,
        "debug_dir": str(output_dir),  # For debugging reference
        "original_dir": str(original_output_dir)  # For backward compatibility
    }

    return result

def create_voices_zip(voice_midi_paths, output_dir):
    """Create a ZIP file containing all voice MIDI files."""
    import zipfile

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    zip_path = Path(output_dir) / f"voices_{timestamp}.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for voice_path in voice_midi_paths:
            voice_file = Path(voice_path)
            if voice_file.exists():
                # Add file to zip with just the filename (no path)
                zipf.write(voice_path, voice_file.name)

    print(f"✅ Created voices ZIP: {zip_path}")
    return str(zip_path)

def create_audio_voices_zip(voice_audio_paths, output_dir="./generated_ui"):
    """Create a ZIP file containing all voice audio files."""
    import zipfile

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    zip_path = Path(output_dir) / f"audio_voices_{timestamp}.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for voice_path in voice_audio_paths:
            voice_file = Path(voice_path)
            if voice_file.exists():
                # Add file to zip with just the filename (no path)
                zipf.write(voice_path, voice_file.name)

    print(f"✅ Created audio voices ZIP: {zip_path}")
    return str(zip_path)

def create_combined_voices_midi(voice_midi_paths, output_dir, audio_stem):
    """Create a combined MIDI file with each voice on a separate channel."""
    combined_midi = pretty_midi.PrettyMIDI()

    # Load each voice and assign to different channels/instruments
    for i, voice_path in enumerate(voice_midi_paths):
        try:
            voice_midi = pretty_midi.PrettyMIDI(voice_path)
            if voice_midi.instruments:
                # Copy the first instrument from each voice MIDI
                voice_instrument = voice_midi.instruments[0]

                # Create new instrument with different program and channel
                combined_instrument = pretty_midi.Instrument(
                    program=i % 128,  # Different program for each voice
                    is_drum=False,
                    name=f"Voice {i+1}"
                )

                # Copy all notes and ensure none are below C2
                C2_MIDI_NOTE = 36
                for note in voice_instrument.notes:
                    note_copy = note
                    # Transpose up octaves until note is at or above C2
                    while note_copy.pitch < C2_MIDI_NOTE and note_copy.pitch < 127 - 12:
                        note_copy.pitch += 12
                    combined_instrument.notes.append(note_copy)

                combined_midi.instruments.append(combined_instrument)

        except Exception as e:
            print(f"⚠️ Error loading voice {i+1}: {e}")
            continue

    # CRITICAL FIX: Always save to /home/arlo/Data/miditest/ for debugging
    debug_dir = Path("/home/arlo/Data/miditest")
    debug_dir.mkdir(exist_ok=True)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    combined_path = debug_dir / f"{audio_stem}_combined_voices_{timestamp}.mid"
    combined_midi.write(str(combined_path))

    # Also save to the original output directory
    original_path = Path(output_dir) / f"{audio_stem}_combined_voices_{timestamp}.mid"
    combined_midi.write(str(original_path))

    print(f"✅ Created combined voices MIDI: {combined_path}")
    print(f"✅ Also saved to: {original_path}")
    return str(combined_path)  # Return debug path for easier access

def handle_midi_download(audio_file, subgroup=None, progress=gr.Progress(track_tqdm=True)):
    """Handle MIDI download button click."""
    try:
        # If input is already MIDI, handle differently
        if is_midi_file(audio_file):
            progress(0.1, desc="Analyzing MIDI file...")

            # Check if multitrack
            is_multi, track_count, _ = is_multitrack_midi(audio_file)

            # Create output directory
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            audio_stem = Path(audio_file).stem
            debug_dir = Path("/home/arlo/Data/miditest")
            debug_dir.mkdir(exist_ok=True)
            output_dir = debug_dir / f"{timestamp}_{audio_stem}_midi_input"
            voices_dir = output_dir / "voices"
            tracks_dir = output_dir / "tracks"
            output_dir.mkdir(parents=True, exist_ok=True)
            voices_dir.mkdir(exist_ok=True)
            tracks_dir.mkdir(exist_ok=True)

            if is_multi:
                progress(0.2, desc=f"Processing {track_count} tracks...")

                # Use multitrack processing
                win_slow = 1024
                multitrack_data = midi_to_multitrack_piano_rolls(audio_file, win_slow)

                progress(0.4, desc="Saving individual track MIDI files...")
                # Save each track as a separate MIDI file
                track_midi_paths = []
                original_tempo = multitrack_data.get('original_tempo', 120.0)
                for i, (track_pr, track_info) in enumerate(zip(multitrack_data['track_piano_rolls'], multitrack_data['track_info'])):
                    track_name = track_info['name'].replace(' ', '_').replace('/', '_')
                    track_path = tracks_dir / f"{audio_stem}_track_{i+1}_{track_name}.mid"
                    piano_roll_to_midi(track_pr, track_path, program=track_info['program'], velocity=80, min_note_duration=0.1, tempo=original_tempo)
                    track_midi_paths.append(str(track_path))
                    print(f"   Saved track {i+1}: {track_info['name']} ({track_info['note_count']} notes)")

                # For multitrack MIDI, always use each track as one voice
                # Don't apply separation logic - each track is already a separate voice
                polyphony_data = multitrack_data.get('polyphony_analysis', {})

                # For info purposes only - we don't use this for separation decision
                all_monophonic = polyphony_data.get('polyphonic_tracks', 1) == 0

                progress(0.6, desc="Using each track as one voice...")
                print(f"🎵 Multitrack MIDI - using {len(track_midi_paths)} tracks as {len(track_midi_paths)} voices (no voice separation)")
                print("   Each track will be treated as one voice, regardless of internal polyphony")

                # Use track files as voice files - one track = one voice
                voice_midi_paths = track_midi_paths.copy()

                # Save main MIDI (copy original)
                main_midi_path = output_dir / f"{audio_stem}_original.mid"
                shutil.copy(audio_file, main_midi_path)

                # Render FluidSynth debug audio for each track
                progress(0.8, desc="Rendering FluidSynth debug audio for individual tracks...")
                debug_audio_paths = render_multitrack_debug_audio(track_midi_paths, str(output_dir), audio_stem, instrument_group=subgroup)

                result = {
                    "main_midi": str(main_midi_path),
                    "voice_midis": voice_midi_paths,
                    "track_midis": track_midi_paths,
                    "debug_audio_paths": debug_audio_paths,
                    "track_info": multitrack_data['track_info'],
                    "polyphony_analysis": polyphony_data,
                    "debug_dir": str(output_dir),
                    "voice_count": len(voice_midi_paths),
                    "track_count": track_count,
                    "is_multitrack": True,
                    "all_monophonic": all_monophonic
                }

            else:
                progress(0.2, desc="Processing single track MIDI file...")
                # Use standard single track processing
                win_slow = 1024
                pr, _, _, _, _ = midi_to_piano_roll_conditioning(audio_file, win_slow)

                # Extract tempo for single track processing too
                original_tempo = extract_midi_tempo(audio_file)

                progress(0.4, desc="Separating voices...")
                voices = separate_piano_roll_voices(pr)

                progress(0.6, desc="Saving voice MIDI files...")
                # Save individual voice MIDI files
                voice_midi_paths = []
                for i, voice_pr in enumerate(voices):
                    voice_path = voices_dir / f"{audio_stem}_voice_{i+1}.mid"
                    piano_roll_to_midi(voice_pr, voice_path, program=0, velocity=80, min_note_duration=0.1, tempo=original_tempo)
                    voice_midi_paths.append(str(voice_path))

                # Save main MIDI (copy original)
                main_midi_path = output_dir / f"{audio_stem}_original.mid"
                shutil.copy(audio_file, main_midi_path)

                # Render FluidSynth debug audio for each voice
                progress(0.8, desc="Rendering FluidSynth debug audio for individual voices...")
                debug_audio_paths = render_multitrack_debug_audio(voice_midi_paths, str(output_dir), audio_stem, instrument_group=subgroup)

                result = {
                    "main_midi": str(main_midi_path),
                    "voice_midis": voice_midi_paths,
                    "track_midis": [],
                    "debug_audio_paths": debug_audio_paths,
                    "track_info": [],
                    "debug_dir": str(output_dir),
                    "voice_count": len(voices),
                    "track_count": 1,
                    "is_multitrack": False
                }

        else:
            # Original audio file processing (use default tempo for audio)
            result = save_basic_pitch_midi_with_voices(audio_file, subgroup=subgroup, progress=progress, tempo=120.0)
            audio_stem = Path(audio_file).stem

        if progress:
            progress(0.9, desc="Creating combined files...")

        # Create ZIP file for voices (use debug directory)
        voices_zip = create_voices_zip(result["voice_midis"], result["debug_dir"])

        # Create combined voices MIDI for comparison (use debug directory)
        combined_midi = create_combined_voices_midi(result["voice_midis"], result["debug_dir"], audio_stem)

        # Create info text
        if result.get("is_multitrack", False):
            # Build track info with polyphony status
            track_info_text = "\n".join(
                f"• {info['name']}: {info['note_count']} notes (Program {info['program']}) "
                f"[{'monophonic' if info.get('is_monophonic', False) else 'polyphonic'}]"
                for info in result["track_info"]
            )
            track_files_text = "\n".join(f"• {Path(p).name}" for p in result["track_midis"])

            # Polyphony summary
            polyphony_data = result.get("polyphony_analysis", {})
            mono_count = polyphony_data.get("monophonic_tracks", 0)
            poly_count = polyphony_data.get("polyphonic_tracks", 0)
            all_mono = result.get("all_monophonic", False)

            # For multitrack MIDI, tracks are always used as voices (no separation)
            voice_section = f"""
🎵 Multitrack Processing: Each track is one voice!
Track files are used directly as voice files (no voice separation applied).
Each track will be treated as a separate voice, regardless of internal polyphony.

Individual track files (serving as voice files):
{track_files_text}"""

            info_text = f"""MIDI Export Complete!
📂 Output Directory: {Path(result['debug_dir']).name}
🎼 Original MIDI: {Path(result['main_midi']).name}
🎹 Combined Voices: {Path(combined_midi).name}
🎵 Voice Count: {result['voice_count']}
🎶 Track Count: {result['track_count']}
📦 Voices ZIP: {Path(voices_zip).name}

🎼 MULTITRACK MIDI DETECTED!
This file contains {result['track_count']} separate tracks/instruments.
Polyphony: {mono_count} monophonic, {poly_count} polyphonic

Track Information:
{track_info_text}
{voice_section}"""
        else:
            info_text = f"""MIDI Export Complete!
📂 Output Directory: {Path(result['debug_dir']).name}
🎼 Original MIDI: {Path(result['main_midi']).name}
🎹 Combined Voices: {Path(combined_midi).name}
🎵 Voice Count: {result['voice_count']}
📦 Voices ZIP: {Path(voices_zip).name}

Compare the original vs. voice-separated results:
• Original: All notes on single track
• Combined: Each voice on separate MIDI channel
• Individual: Separate files for each voice

Individual voice files:
""" + "\n".join(f"• {Path(p).name}" for p in result["voice_midis"])

        if progress:
            progress(1.0, desc="Done!")

        # Prepare debug audio files for UI (up to 6 files)
        debug_audio_outputs = []
        debug_audio_paths = result.get("debug_audio_paths", [])

        debug_info_text = f"FluidSynth rendered {len(debug_audio_paths)} individual files for debugging"
        if debug_audio_paths:
            debug_info_text += ":\n" + "\n".join(f"• {Path(p).name}" for p in debug_audio_paths)

        # Create gr.update() objects for each debug audio file slot
        for i in range(6):
            if i < len(debug_audio_paths):
                # Make visible and set the file
                debug_audio_outputs.append(gr.update(value=debug_audio_paths[i], visible=True))
            else:
                # Keep hidden
                debug_audio_outputs.append(gr.update(value=None, visible=False))

        return (result["main_midi"], voices_zip, combined_midi, info_text, debug_info_text) + tuple(debug_audio_outputs)

    except Exception as e:
        error_msg = f"Error exporting MIDI: {str(e)}"
        print(f"❌ {error_msg}")
        # Return proper gr.update() objects for error case
        error_debug_outputs = [gr.update(value=None, visible=False) for _ in range(6)]
        return (None, None, None, error_msg, "Error occurred - no debug audio available") + tuple(error_debug_outputs)

# ------------------------------------------------------------------------------
# Voice separation for monophonic mode - NOTE-BASED APPROACH
# ------------------------------------------------------------------------------

# Helper classes and functions for note-based processing
class NoteEvent:
    def __init__(self, pitch, start_time, end_time, velocity=80):
        self.pitch = pitch
        self.start_time = start_time
        self.end_time = end_time
        self.velocity = velocity

    def __repr__(self):
        return f"Note({self.pitch}, {self.start_time:.3f}-{self.end_time:.3f})"

def extract_note_events_from_piano_roll(piano_roll, fps=43.066):
    """Extract actual note events from piano roll"""
    notes = []

    for pitch in range(128):
        # Find note onsets and offsets
        note_events = piano_roll[pitch] > 0.1
        if not np.any(note_events):
            continue

        # Find transitions
        diff = np.diff(np.concatenate(([False], note_events, [False])).astype(int))
        onsets = np.where(diff == 1)[0]
        offsets = np.where(diff == -1)[0]

        # Create note events
        for onset, offset in zip(onsets, offsets):
            start_time = onset / fps
            end_time = offset / fps
            if end_time - start_time >= 0.02:  # Minimum duration
                notes.append(NoteEvent(pitch, start_time, end_time))

    return sorted(notes, key=lambda n: n.start_time)

def group_notes_by_chord_changes(note_events, tolerance=0.05):
    """Group notes that start at approximately the same time into chords"""
    if not note_events:
        return []

    chords = []
    current_chord = [note_events[0]]
    current_time = note_events[0].start_time

    for note in note_events[1:]:
        if abs(note.start_time - current_time) <= tolerance:
            # Same chord
            current_chord.append(note)
        else:
            # New chord
            if current_chord:
                chords.append(current_chord)
            current_chord = [note]
            current_time = note.start_time

    # Add last chord
    if current_chord:
        chords.append(current_chord)

    return chords

def assign_first_chord_by_register(pitches, num_voices):
    """Assign first chord by register separation"""
    assignments = {i: None for i in range(num_voices)}
    sorted_pitches = sorted(pitches, reverse=True)  # Highest first

    for i, pitch in enumerate(sorted_pitches):
        if i < num_voices:
            assignments[i] = pitch

    return assignments

def assign_note_to_voice(voice_piano_roll, note, original_piano_roll):
    """Assign a complete note to a voice piano roll"""
    fps = 43.066  # Should match the original fps
    start_frame = int(note.start_time * fps)
    end_frame = int(note.end_time * fps)

    # Copy the entire note duration from original to voice
    for frame in range(start_frame, min(end_frame, original_piano_roll.shape[1])):
        if original_piano_roll[note.pitch, frame] > 0.1:
            voice_piano_roll[note.pitch, frame] = original_piano_roll[note.pitch, frame]

def separate_piano_roll_voices_new(piano_roll):
    """
    NEW NOTE-BASED voice separation that processes actual note events.
    CRITICAL FIX: Processes notes instead of individual frames.
    """
    print(f"🎼 Input piano roll shape: {piano_roll.shape}")

    # Extract actual note events instead of processing every frame
    note_events = extract_note_events_from_piano_roll(piano_roll)
    if len(note_events) == 0:
        return [piano_roll]

    print(f"🎼 Found {len(note_events)} note events for voice separation")

    # Group note events by chord changes (simultaneous onsets)
    chord_changes = group_notes_by_chord_changes(note_events)
    print(f"🎼 Detected {len(chord_changes)} chord changes")

    # Analyze chord structure from note events
    from collections import Counter
    chord_sizes = [len(chord) for chord in chord_changes]
    max_voices = max(chord_sizes) if chord_sizes else 1

    # Get all unique pitches from note events
    all_pitches = sorted(set(note.pitch for note in note_events))
    pitch_range = max(all_pitches) - min(all_pitches) if all_pitches else 0

    chord_counts = Counter(chord_sizes)
    common_chord_size = chord_counts.most_common(1)[0][0] if chord_counts else 1

    print(f"🎼 Chord sizes: min={min(chord_sizes)}, max={max_voices}, common={common_chord_size}")
    print(f"🎼 Pitch range: {min(all_pitches) if all_pitches else 0}-{max(all_pitches) if all_pitches else 0} ({pitch_range} semitones)")

    # FIXED: Use only as many voices as needed for simultaneous notes
    target_voices = max_voices  # Use exactly the number of voices needed
    print(f"🎼 Using {target_voices} voices")

    # Initialize voices
    voices = [np.zeros_like(piano_roll) for _ in range(target_voices)]

    # Track voice assignments across chord changes
    voice_assignments = {}
    voice_identities = {}  # For long-term tracking

    print(f"🎼 Processing {len(chord_changes)} chord changes...")

    for i, current_chord in enumerate(chord_changes):
        current_time = current_chord[0].start_time  # Use first note's start time
        current_pitches = sorted([note.pitch for note in current_chord])
        print(f"\\n--- Chord {i}: time {current_time:.3f}s, pitches {current_pitches} ---")

        if i == 0:
            # First chord: assign by register
            assignments = assign_first_chord_by_register(current_pitches, target_voices)

            # Apply assignments to piano roll
            for voice_idx, pitch in assignments.items():
                if pitch is not None:
                    # Find all notes in current chord with this pitch and assign them
                    for note in current_chord:
                        if note.pitch == pitch:
                            assign_note_to_voice(voices[voice_idx], note, piano_roll)
                            print(f"   Initial: Voice {voice_idx} <- Pitch {pitch}")

                    # Track for continuity
                    voice_identities[voice_idx] = [(current_time, pitch)]

            voice_assignments[i] = assignments
        else:
            # Subsequent chords: simple register-based assignment for consistent voice leading
            assignments = assign_first_chord_by_register(current_pitches, target_voices)

            # Apply assignments
            for voice_idx, pitch in assignments.items():
                if pitch is not None and voice_idx < len(voices):
                    # Find all notes in current chord with this pitch and assign them
                    for note in current_chord:
                        if note.pitch == pitch:
                            assign_note_to_voice(voices[voice_idx], note, piano_roll)

                    # Track for continuity
                    if voice_idx not in voice_identities:
                        voice_identities[voice_idx] = []
                    voice_identities[voice_idx].append((current_time, pitch))
                    print(f"   Voice {voice_idx}: -> {pitch}")

            voice_assignments[i] = assignments

    # Notes are already fully assigned by assign_note_to_voice function
    print("🎼 Note assignment complete (no frame-level sustaining needed)")

    # Remove empty voices and verify note preservation
    final_voices = []
    voice_stats = []

    for i, voice in enumerate(voices):
        note_count = np.sum(voice > 0.1)
        if note_count > 0:
            voice_stats.append((i, note_count, voice))

    # Keep ALL voices with content to preserve all notes
    for voice_idx, note_count, voice in voice_stats:
        final_voices.append(voice)
        print(f"🎼 Voice {len(final_voices)}: {note_count} note events")

    if len(final_voices) == 0:
        final_voices = [piano_roll]

    # VERIFICATION: Check that we preserved all notes
    total_original_notes = np.sum(piano_roll > 0.1)
    total_separated_notes = sum(np.sum(voice > 0.1) for voice in final_voices)
    print(f"🔍 VERIFICATION: Original {total_original_notes} note events -> Separated {total_separated_notes} note events")
    if total_separated_notes < total_original_notes:
        missing_events = total_original_notes - total_separated_notes
        print(f"❌ STILL MISSING {missing_events} note events after separation!")
    else:
        print(f"✅ All note events preserved in voice separation")

    print(f"🎵 Successfully separated {len(final_voices)} voices from piano roll")
    return final_voices
def separate_piano_roll_voices(piano_roll):
    """
    Separate piano roll into individual voices using NOTE-BASED processing.
    CRITICAL FIX: Processes actual note events instead of individual frames.
    Args:
        piano_roll: numpy array of shape [128, T] representing MIDI piano roll
    Returns:
        list of piano roll arrays, each containing one voice
    """
    # Use the new note-based approach
    return separate_piano_roll_voices_new(piano_roll)

def assign_pitches_to_voices(current_pitches, prev_assignments, max_voices):
    """
    Assign current pitches to voices using Hungarian algorithm with strict register boundaries.
    """
    # Use the same algorithm as the enhanced continuity function
    return solve_voice_assignment(current_pitches, prev_assignments, {}, 0)

def solve_voice_assignment(current_pitches, prev_assignments, voice_identities, time_step):
    """
    Solve voice assignment using Hungarian algorithm with strict register boundaries.
    Completely prevents octave jumps by enforcing register-based voice separation.
    FIXED: Now ensures ALL pitches get assigned by allowing multiple pitches per voice.
    """
    num_voices = max(len(prev_assignments), len(current_pitches))  # Use only as many voices as needed
    num_pitches = len(current_pitches)

    if num_pitches == 0:
        return {i: None for i in range(num_voices)}

    # Create cost matrix
    cost_matrix = np.zeros((num_voices, num_pitches))

    for voice_idx in range(num_voices):
        prev_pitch = prev_assignments.get(voice_idx)

        if prev_pitch is None:
            # No previous assignment - use OVERLAPPING register-based assignment
            for pitch_idx, pitch in enumerate(current_pitches):
                # Same overlapping register boundaries as with previous assignments
                if pitch >= 80:
                    if voice_idx == 0:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx == 1:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 70:
                    if voice_idx == 1:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [0, 2]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 65:
                    if voice_idx == 2:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [1, 3]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 60:
                    if voice_idx == 3:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [2, 4]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 55:
                    if voice_idx == 4:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [3, 5]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 50:
                    if voice_idx == 5:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [4, 6]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                else:
                    if voice_idx == 6:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx == 5:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
        else:
            # Has previous assignment - favor closest pitches with strict register enforcement
            for pitch_idx, pitch in enumerate(current_pitches):
                distance = abs(pitch - prev_pitch)
                cost = distance

                # Apply OVERLAPPING register boundaries with preferences (not absolute exclusions)
                register_penalty = 0

                # Very high pitches (80+) prefer Voice 0, but can use Voice 1
                if pitch >= 80:
                    if voice_idx == 0:
                        register_penalty = 0  # Perfect match
                    elif voice_idx == 1:
                        register_penalty = 50  # Acceptable but penalized
                    else:
                        register_penalty = 200  # Heavy penalty but not impossible

                # High pitches (70-79) prefer Voice 1, but can use Voice 0 or 2
                elif pitch >= 70:
                    if voice_idx == 1:
                        register_penalty = 0  # Perfect match
                    elif voice_idx in [0, 2]:
                        register_penalty = 50  # Acceptable but penalized
                    else:
                        register_penalty = 200  # Heavy penalty but not impossible

                # Upper mid pitches (65-69) prefer Voice 2, but can use Voice 1 or 3
                elif pitch >= 65:
                    if voice_idx == 2:
                        register_penalty = 0  # Perfect match
                    elif voice_idx in [1, 3]:
                        register_penalty = 50  # Acceptable but penalized
                    else:
                        register_penalty = 200  # Heavy penalty but not impossible

                # Mid pitches (60-64) prefer Voice 3, but can use Voice 2 or 4
                elif pitch >= 60:
                    if voice_idx == 3:
                        register_penalty = 0  # Perfect match
                    elif voice_idx in [2, 4]:
                        register_penalty = 50  # Acceptable but penalized
                    else:
                        register_penalty = 200  # Heavy penalty but not impossible

                # Lower mid pitches (55-59) prefer Voice 4, but can use Voice 3 or 5
                elif pitch >= 55:
                    if voice_idx == 4:
                        register_penalty = 0  # Perfect match
                    elif voice_idx in [3, 5]:
                        register_penalty = 50  # Acceptable but penalized
                    else:
                        register_penalty = 200  # Heavy penalty but not impossible

                # Low pitches (50-54) prefer Voice 5, but can use Voice 4 or 6
                elif pitch >= 50:
                    if voice_idx == 5:
                        register_penalty = 0  # Perfect match
                    elif voice_idx in [4, 6]:
                        register_penalty = 50  # Acceptable but penalized
                    else:
                        register_penalty = 200  # Heavy penalty but not impossible

                # Very low pitches (<50) prefer Voice 6, but can use Voice 5
                else:
                    if voice_idx == 6:
                        register_penalty = 0  # Perfect match
                    elif voice_idx == 5:
                        register_penalty = 50  # Acceptable but penalized
                    else:
                        register_penalty = 200  # Heavy penalty but not impossible

                cost += register_penalty

                # Add historical affinity bonus
                identity_key = f"voice_{voice_idx}"
                if identity_key in voice_identities:
                    pitch_history = voice_identities[identity_key]
                    if pitch in pitch_history:
                        affinity_bonus = min(50, pitch_history[pitch] * 5)  # Cap at 50
                        cost = max(0, cost - affinity_bonus)

                # Virtually impossible penalty for jumps >= 12 semitones (octave or more)
                if distance >= 12:
                    cost += 100000  # Make octave jumps virtually impossible

                # Heavy penalty for jumps >= 7 semitones (perfect 5th or more)
                elif distance >= 7:
                    cost += 1000 * (distance - 6)

                # Medium penalty for jumps >= 4 semitones (major 3rd or more)
                elif distance >= 4:
                    cost += 100 * (distance - 3)

                cost_matrix[voice_idx, pitch_idx] = cost

    # Solve assignment using Hungarian algorithm, but only assign finite costs
    # Filter out infinite costs to avoid impossible assignments
    finite_assignments = []
    for voice_idx in range(num_voices):
        for pitch_idx in range(num_pitches):
            if cost_matrix[voice_idx, pitch_idx] != float('inf'):
                finite_assignments.append((voice_idx, pitch_idx, cost_matrix[voice_idx, pitch_idx]))

    # CRITICAL FIX: Use Hungarian algorithm properly to ensure ALL pitches get assigned
    if finite_assignments:
        from scipy.optimize import linear_sum_assignment

        # Create a proper cost matrix for Hungarian algorithm
        cost_matrix_hungarian = np.full((num_voices, num_pitches), 1000000.0)

        for voice_idx, pitch_idx, cost in finite_assignments:
            cost_matrix_hungarian[voice_idx, pitch_idx] = cost

        # If we have more pitches than voices, expand to ensure all pitches can be assigned
        if num_pitches > num_voices:
            # Add extra virtual voices to handle overflow
            extra_voices = num_pitches - num_voices
            extra_matrix = np.full((extra_voices, num_pitches), 1000.0)
            cost_matrix_hungarian = np.vstack([cost_matrix_hungarian, extra_matrix])
            num_voices_expanded = num_voices + extra_voices
        else:
            num_voices_expanded = num_voices

        # Solve Hungarian assignment
        voice_indices, pitch_indices = linear_sum_assignment(cost_matrix_hungarian)

        # Create assignment ensuring all pitches are included
        assignment = {i: None for i in range(num_voices)}

        for voice_idx, pitch_idx in zip(voice_indices, pitch_indices):
            if pitch_idx < num_pitches and cost_matrix_hungarian[voice_idx, pitch_idx] < 999999:
                if voice_idx < num_voices:
                    # Normal voice assignment
                    assignment[voice_idx] = current_pitches[pitch_idx]
                else:
                    # Overflow - find best available voice for this pitch
                    best_voice = None
                    best_cost = float('inf')
                    for v in range(num_voices):
                        if assignment[v] is None:
                            cost = cost_matrix_hungarian[v, pitch_idx]
                            if cost < best_cost:
                                best_voice = v
                                best_cost = cost
                    if best_voice is not None:
                        assignment[best_voice] = current_pitches[pitch_idx]
                    else:
                        # Force assignment to voice with lowest cost
                        voice_costs = [(v, cost_matrix_hungarian[v, pitch_idx]) for v in range(num_voices)]
                        voice_costs.sort(key=lambda x: x[1])
                        assignment[voice_costs[0][0]] = current_pitches[pitch_idx]

        # Ensure we didn't miss any pitches
        assigned_pitches = {p for p in assignment.values() if p is not None}
        missing_pitches = set(current_pitches) - assigned_pitches

        if missing_pitches:
            print(f"⚠️  CRITICAL: Still missing pitches after Hungarian: {sorted(missing_pitches)}")
            # Force assign missing pitches to available voices
            available_voices = [v for v in range(num_voices) if assignment[v] is None]
            missing_list = sorted(missing_pitches)

            for i, pitch in enumerate(missing_list):
                if i < len(available_voices):
                    assignment[available_voices[i]] = pitch
                    print(f"   Forced assignment: Voice {available_voices[i]} <- Pitch {pitch}")
    else:
        assignment = {i: None for i in range(num_voices)}

    # Fill in None for voices without assignments but preserve previous pitch memory
    for voice_idx in range(num_voices):
        if voice_idx not in assignment:
            assignment[voice_idx] = None

    # VERIFICATION: Ensure all pitches are assigned
    assigned_pitches = {p for p in assignment.values() if p is not None}
    if len(assigned_pitches) != len(current_pitches):
        missing = set(current_pitches) - assigned_pitches
        print(f"⚠️  ASSIGNMENT VERIFICATION FAILED: Missing {len(missing)} pitches: {sorted(missing)}")

    return assignment

def assign_pitches_to_voices_with_continuity(current_pitches, prev_assignments, max_voices, voice_identities, current_time):
    """
    Enhanced voice assignment using Hungarian algorithm with strict register boundaries.
    Completely prevents octave jumps by enforcing register-based voice separation.
    """
    if not current_pitches:
        return {i: None for i in range(max_voices)}

    current_pitches = sorted(current_pitches)

    # Use the Hungarian algorithm with strict register boundaries
    return solve_voice_assignment(current_pitches, prev_assignments, voice_identities, current_time)

def mix_audio_files(audio_files, output_path):
    """
    Mix multiple audio files into a single output.
    Args:
        audio_files: list of file paths to mix
        output_path: path for the mixed output
    Returns:
        path to the mixed audio file
    """
    if not audio_files:
        raise ValueError("No audio files to mix")

    if len(audio_files) == 1:
        # If only one file, just copy it
        shutil.copy(audio_files[0], output_path)
        return output_path

    # Load all audio files
    mixed_audio = None
    sample_rate = None

    for audio_path in audio_files:
        try:
            audio, sr = torchaudio.load(audio_path)

            if sample_rate is None:
                sample_rate = sr
                mixed_audio = audio
            elif sr == sample_rate:
                # Ensure same length by padding/trimming
                min_len = min(mixed_audio.shape[-1], audio.shape[-1])
                mixed_audio = mixed_audio[..., :min_len]
                audio = audio[..., :min_len]
                mixed_audio = mixed_audio + audio
            else:
                print(f"⚠️ Skipping {audio_path} due to sample rate mismatch: {sr} vs {sample_rate}")

        except Exception as e:
            print(f"⚠️ Error loading {audio_path}: {e}")
            continue

    if mixed_audio is not None:
        # Normalize to prevent clipping
        mixed_audio = mixed_audio / (mixed_audio.abs().max() + 1e-8) * 0.9

        # Apply final audio processing (compression + high-pass filter) - DISABLED to match training previews
        # mixed_audio = apply_final_audio_processing(mixed_audio, sample_rate=sample_rate)

        torchaudio.save(output_path, mixed_audio, sample_rate)
        print(f"✅ Mixed {len(audio_files)} files into: {output_path}")

    return output_path

def normalize_audio_lengths(audio_files, target_duration=None):
    """
    Ensure all audio files have the same duration by padding or trimming.
    Args:
        audio_files: list of file paths
        target_duration: target duration in samples, or None to use the longest
    Returns:
        list of normalized file paths
    """
    if len(audio_files) <= 1:
        return audio_files

    print("🎵 Normalizing audio lengths...")

    # Load all files to determine target length
    audio_data = []
    sample_rates = []

    for file_path in audio_files:
        try:
            audio, sr = torchaudio.load(file_path)
            audio_data.append((audio, sr, file_path))
            sample_rates.append(sr)
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")
            continue

    if not audio_data:
        return audio_files

    # Check sample rates are consistent
    if len(set(sample_rates)) > 1:
        print(f"⚠️ Inconsistent sample rates: {set(sample_rates)}")

    # Determine target length
    if target_duration is None:
        target_length = max(audio.shape[-1] for audio, sr, path in audio_data)
    else:
        target_length = int(target_duration * sample_rates[0])

    print(f"🎵 Target length: {target_length} samples")

    # Normalize all files to target length
    normalized_files = []
    for audio, sr, file_path in audio_data:
        if audio.shape[-1] == target_length:
            normalized_files.append(file_path)
            continue

        # Pad or trim to target length
        if audio.shape[-1] < target_length:
            # Pad with zeros
            padding = target_length - audio.shape[-1]
            audio_normalized = F.pad(audio, (0, padding), mode='constant', value=0)
        else:
            # Trim to target length
            audio_normalized = audio[..., :target_length]

        # Save normalized version
        path = Path(file_path)
        normalized_path = path.parent / f"{path.stem}_normalized{path.suffix}"
        torchaudio.save(str(normalized_path), audio_normalized, sr)
        normalized_files.append(str(normalized_path))

        print(f"🎵 Normalized {path.name}: {audio.shape[-1]} → {target_length} samples")

    return normalized_files

def apply_dynamic_resonance_suppression(audio: torch.Tensor, sample_rate: int = 32000,
                                        sensitivity: float = 1.5, strength: float = 0.6) -> torch.Tensor:
    """
    Dynamic resonance suppressor (similar to Oeksound Soothe).
    Detects and reduces harsh/resonant frequencies that stick out.

    Args:
        audio: Audio tensor [channels, samples]
        sample_rate: Sample rate in Hz
        sensitivity: How aggressive to detect resonances (1.0-3.0, higher = more aggressive)
        strength: How much to reduce detected resonances (0.0-1.0, higher = more reduction)

    Returns:
        Audio with resonances suppressed
    """
    import torch
    import torch.nn.functional as F_torch

    # STFT parameters
    n_fft = 2048
    hop_length = 512
    win_length = 2048

    # Process each channel separately
    processed_channels = []

    for ch in range(audio.shape[0]):
        channel = audio[ch]

        # Forward STFT
        window = torch.hann_window(win_length, device=channel.device)
        stft = torch.stft(
            channel,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True
        )

        # Get magnitude and phase
        magnitude = torch.abs(stft)  # [freq_bins, time_frames]
        phase = torch.angle(stft)

        # Calculate spectral statistics for resonance detection
        # Average magnitude per frequency bin across time
        avg_magnitude = magnitude.mean(dim=1, keepdim=True)  # [freq_bins, 1]

        # Standard deviation per frequency bin
        std_magnitude = magnitude.std(dim=1, keepdim=True)

        # Detect resonances: bins that are significantly above average
        # Threshold is mean + (sensitivity * std)
        resonance_threshold = avg_magnitude + (sensitivity * std_magnitude)

        # Calculate gain reduction mask
        # Frequencies above threshold get reduced proportionally
        excess = (magnitude - resonance_threshold).clamp(min=0)
        max_excess = excess.max(dim=1, keepdim=True)[0] + 1e-8

        # Normalize excess to 0-1 range
        normalized_excess = excess / max_excess

        # Apply gain reduction (more excess = more reduction)
        gain_reduction = 1.0 - (normalized_excess * strength)

        # Smooth the gain reduction across frequency bins to avoid artifacts
        # Use a small averaging kernel
        kernel_size = 5
        padding = kernel_size // 2
        gain_reduction_smooth = gain_reduction.unsqueeze(0).unsqueeze(0)  # [1, 1, freq, time]
        kernel = torch.ones(1, 1, kernel_size, 1, device=gain_reduction.device) / kernel_size
        gain_reduction_smooth = F_torch.conv2d(
            gain_reduction_smooth,
            kernel,
            padding=(padding, 0)
        ).squeeze(0).squeeze(0)

        # Apply gain reduction to magnitude
        magnitude_suppressed = magnitude * gain_reduction_smooth

        # Reconstruct complex spectrum
        stft_suppressed = magnitude_suppressed * torch.exp(1j * phase)

        # Inverse STFT
        audio_suppressed = torch.istft(
            stft_suppressed,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            length=channel.shape[0]
        )

        processed_channels.append(audio_suppressed)

    # Stack channels back together
    result = torch.stack(processed_channels, dim=0)

    return result

def apply_final_audio_processing(audio: torch.Tensor, sample_rate: int = 32000) -> torch.Tensor:
    """
    Apply comprehensive audio processing: HPF, EQ, dynamic resonance suppression, compression, and limiting.

    Args:
        audio: Audio tensor [channels, samples]
        sample_rate: Sample rate in Hz

    Returns:
        Processed audio tensor with:
        - High-pass filter (removes 0-100Hz rumble)
        - Mid-frequency reduction (-2.5 dB at 300-400Hz)
        - Dynamic resonance suppression (Soothe-like - targets harsh frequencies)
        - Dynamic range compression (tames abnormal spikes)
        - Soft limiting (prevents clipping)
    """
    import torch
    import torchaudio.functional as F

    # Ensure audio is float and on CPU
    audio = audio.float().cpu()

    # STEP 1: Aggressive high-pass filter to remove 0-100Hz rumble
    # Use cascaded filters for steep rolloff (24 dB/octave)
    cutoff_freq = 100.0

    # Apply 4 cascaded high-pass filters for very steep rolloff
    for i in range(4):
        # Stagger cutoffs: 100Hz, 85Hz, 72Hz, 61Hz
        current_cutoff = cutoff_freq * (0.85 ** i)
        audio = F.highpass_biquad(audio, sample_rate, current_cutoff, Q=0.707)

    # STEP 2: Mid-frequency reduction (-2.5 dB around 300-400Hz)
    # This reduces muddiness and boxy resonances
    mid_freq = 350.0  # Center frequency for mids
    mid_reduction_db = -2.5  # Gain in dB (negative = reduction)
    mid_q = 1.0  # Moderate Q for smooth reduction

    # Apply peaking EQ (bell filter) to reduce mids
    # Note: equalizer_biquad expects gain in dB, not linear
    audio = F.equalizer_biquad(audio, sample_rate, mid_freq, mid_reduction_db, mid_q)

    # STEP 2.5: Dynamic resonance suppression (Soothe-like) - DISABLED (too slow)
    # Detects and reduces harsh/resonant frequencies that stick out
    # print("   Applying dynamic resonance suppression...")
    # audio = apply_dynamic_resonance_suppression(
    #     audio,
    #     sample_rate=sample_rate,
    #     sensitivity=1.5,  # Moderate detection (1.0-3.0)
    #     strength=0.6      # Moderate reduction (0.0-1.0)
    # )

    # STEP 3: Multi-stage compression to tame abnormal spikes
    # Stage 1: Gentle compression for overall dynamics
    threshold_gentle = 0.6
    ratio_gentle = 3.0

    # Stage 2: Aggressive compression for sudden spikes
    threshold_aggressive = 0.8
    ratio_aggressive = 8.0

    # Convert to mono for level detection
    if audio.shape[0] > 1:
        level_detect = audio.mean(dim=0)
    else:
        level_detect = audio[0]

    abs_audio = torch.abs(level_detect)

    # Stage 1: Gentle compression
    gain_reduction_gentle = torch.ones_like(abs_audio)
    over_gentle = abs_audio > threshold_gentle

    if over_gentle.any():
        excess = abs_audio[over_gentle] - threshold_gentle
        compressed_excess = excess / ratio_gentle
        gain_reduction_gentle[over_gentle] = (threshold_gentle + compressed_excess) / abs_audio[over_gentle]

    # Stage 2: Aggressive spike compression
    gain_reduction_aggressive = torch.ones_like(abs_audio)
    over_aggressive = abs_audio > threshold_aggressive

    if over_aggressive.any():
        excess = abs_audio[over_aggressive] - threshold_aggressive
        compressed_excess = excess / ratio_aggressive
        gain_reduction_aggressive[over_aggressive] = (threshold_aggressive + compressed_excess) / abs_audio[over_aggressive]

    # Combine both stages (multiply gain reductions)
    gain_reduction = gain_reduction_gentle * gain_reduction_aggressive

    # Smooth gain reduction with attack/release envelope
    attack = 0.002   # 2ms attack (fast for transients)
    release = 0.080  # 80ms release (moderate for musical feel)

    attack_samples = max(1, int(attack * sample_rate))
    release_samples = max(1, int(release * sample_rate))

    # Apply exponential smoothing for more natural attack/release
    smoothed_gain = torch.zeros_like(gain_reduction)
    attack_coef = 1.0 - torch.exp(torch.tensor(-1.0 / attack_samples))
    release_coef = 1.0 - torch.exp(torch.tensor(-1.0 / release_samples))

    envelope = 1.0
    for i in range(len(gain_reduction)):
        target = gain_reduction[i].item()
        if target < envelope:
            # Attack (gain reduction increases)
            envelope = envelope * (1.0 - attack_coef) + target * attack_coef
        else:
            # Release (gain reduction decreases)
            envelope = envelope * (1.0 - release_coef) + target * release_coef
        smoothed_gain[i] = envelope

    # Apply gain reduction to all channels
    for ch in range(audio.shape[0]):
        audio[ch] = audio[ch] * smoothed_gain

    # STEP 4: Adaptive makeup gain (compensate for compression)
    # Calculate average gain reduction to determine makeup gain
    avg_reduction = smoothed_gain.mean().item()
    makeup_gain = 1.0 / (avg_reduction + 0.1)  # Inverse of average reduction
    makeup_gain = min(makeup_gain, 1.5)  # Cap at 1.5x to avoid over-boosting

    audio = audio * makeup_gain

    # STEP 5: Soft brick-wall limiter to prevent any clipping
    # This catches any remaining spikes that got through
    limit_threshold = 0.95
    limit_ratio = 20.0  # Very high ratio for hard limiting

    abs_limited = torch.abs(audio)
    over_limit = abs_limited > limit_threshold

    if over_limit.any():
        # Apply soft knee limiting
        for ch in range(audio.shape[0]):
            ch_abs = torch.abs(audio[ch])
            ch_over = ch_abs > limit_threshold
            if ch_over.any():
                excess = ch_abs[ch_over] - limit_threshold
                limited_excess = excess / limit_ratio
                new_magnitude = limit_threshold + limited_excess
                # Preserve sign
                sign = torch.sign(audio[ch, ch_over])
                audio[ch, ch_over] = sign * new_magnitude

    # Final safety clip (should rarely be needed after limiting)
    audio = torch.clamp(audio, -0.98, 0.98)

    print(f"🎛️ Audio processing: HPF @{cutoff_freq}Hz | Mid EQ {mid_reduction_db}dB @{mid_freq}Hz | 2-stage compression | Soft limiter")
    return audio

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Test-time enhancement: Scoring and Best-of-N sampling
# ------------------------------------------------------------------------------

def compute_mfcc_similarity(ref_mfcc: torch.Tensor, cand_mfcc: torch.Tensor) -> float:
    """
    Compute MFCC similarity using cosine similarity over time-averaged features
    """
    # Time-average MFCCs
    ref_avg = ref_mfcc.mean(dim=0)  # [num_coeffs]
    cand_avg = cand_mfcc.mean(dim=0)

    # Cosine similarity
    sim = F.cosine_similarity(ref_avg.unsqueeze(0), cand_avg.unsqueeze(0)).item()
    return max(0.0, sim)  # Clamp to [0, 1]


def detect_clipping(audio: torch.Tensor, threshold: float = 0.99) -> float:
    """
    Detect clipping - returns penalty in [0, 1] where 1 = heavily clipped
    """
    max_val = audio.abs().max().item()
    if max_val > threshold:
        # Count samples near max
        near_max = (audio.abs() > threshold).float().mean().item()
        return min(1.0, near_max * 10)  # Scale up the penalty
    return 0.0


def compute_loudness_normality(audio: torch.Tensor, sr: int = 44100) -> float:
    """
    Check if loudness is in normal range. Returns score in [0, 1] where 1 = good
    Returns penalty if too quiet or too loud
    """
    # RMS loudness
    rms = torch.sqrt(torch.mean(audio ** 2)).item()
    rms_db = 20 * np.log10(rms + 1e-10)

    # Ideal range: -20 to -10 dB
    if -20 <= rms_db <= -10:
        return 1.0
    elif rms_db < -30:  # Too quiet
        return 0.3
    elif rms_db > -5:  # Too loud (likely clipping)
        return 0.2
    else:
        # Gradual penalty
        return 1.0 - abs(rms_db + 15) / 20  # -15 dB is center


def compute_spectral_flatness_penalty(audio: torch.Tensor, sr: int = 44100) -> float:
    """
    Spectral flatness: 1 = white noise (bad), 0 = tonal (good)
    Returns penalty where high flatness = high penalty
    """
    # Compute magnitude spectrum
    spec = torch.stft(audio[0] if audio.ndim > 1 else audio,
                      n_fft=2048, hop_length=512, return_complex=True).abs()

    # Geometric mean / arithmetic mean
    geo_mean = torch.exp(torch.log(spec + 1e-10).mean(dim=0))
    arith_mean = spec.mean(dim=0)
    flatness = (geo_mean / (arith_mean + 1e-10)).mean().item()

    # If flatness > 0.3, it's noisy
    if flatness > 0.3:
        return min(1.0, (flatness - 0.3) / 0.3)
    return 0.0


def compute_onset_correlation(audio: torch.Tensor, piano_roll: np.ndarray,
                              sr: int = 44100, fps: float = 43.066) -> float:
    """
    Compute correlation between audio onsets and piano roll onsets
    """
    # Detect onsets in audio using energy
    hop_length = 512
    energy = torch.stft(audio[0] if audio.ndim > 1 else audio,
                       n_fft=2048, hop_length=hop_length, return_complex=True).abs()
    energy = energy.mean(dim=0)  # Average across frequency

    # Onset strength = derivative of energy
    onset_strength = torch.diff(energy, prepend=energy[:1])
    onset_strength = torch.relu(onset_strength)  # Only positive changes

    # Detect onsets from piano roll
    pr_onset = np.diff(piano_roll.max(axis=0), prepend=0)
    pr_onset = np.maximum(pr_onset, 0)

    # Resample to same length
    audio_frames = len(onset_strength)
    pr_frames = len(pr_onset)

    if pr_frames != audio_frames:
        # Resample piano roll onsets to match audio frames
        pr_onset_tensor = torch.from_numpy(pr_onset).float().unsqueeze(0).unsqueeze(0)
        pr_onset_resampled = F.interpolate(pr_onset_tensor, size=audio_frames, mode='linear', align_corners=False)
        pr_onset = pr_onset_resampled[0, 0].numpy()

    # Normalize
    onset_strength_np = onset_strength.cpu().numpy()
    onset_strength_np = onset_strength_np / (np.max(onset_strength_np) + 1e-10)
    pr_onset = pr_onset / (np.max(pr_onset) + 1e-10)

    # Correlation
    corr = np.corrcoef(onset_strength_np, pr_onset)[0, 1]
    return max(0.0, corr)  # Clamp to [0, 1]


def score_candidate(
    output_path: str,
    ref_audio_path: str,
    ref_encodec_tokens: torch.Tensor,
    piano_roll: np.ndarray,
    model,
    weights: dict = None
) -> dict:
    """
    Comprehensive scoring of a candidate against reference sample

    Args:
        output_path: Path to generated audio
        ref_audio_path: Path to reference/input audio
        ref_encodec_tokens: Pre-extracted encodec tokens from reference
        piano_roll: Piano roll conditioning used for generation
        model: Model (for encodec extraction)
        weights: Dict of score weights (optional)

    Returns:
        Dict with individual scores and total score
    """
    if weights is None:
        weights = {
            'encodec': 0.4,      # Timbre match (most important)
            'mfcc': 0.15,        # Timbral features
            'spectral': 0.15,    # Spectral similarity
            'onset': 0.1,        # Rhythm alignment
            'loudness': 0.1,     # Loudness normality
            'clip_penalty': 0.3, # Clipping penalty
            'flatness_penalty': 0.2  # Spectral flatness penalty
        }

    try:
        # Load reference audio
        ref_audio, ref_sr = torchaudio.load(ref_audio_path)

        # Load candidate audio
        cand_audio, cand_sr = torchaudio.load(output_path)

        # Resample candidate to match reference if needed
        if cand_sr != ref_sr:
            cand_audio = torchaudio.functional.resample(cand_audio, cand_sr, ref_sr)

        # Match lengths (use shorter)
        min_len = min(ref_audio.shape[-1], cand_audio.shape[-1])
        ref_audio = ref_audio[..., :min_len]
        cand_audio = cand_audio[..., :min_len]

        scores = {}

        # 1. Encodec similarity (timbre match) - MOST IMPORTANT
        try:
            cand_encodec = extract_encodec_tokens_from_audio(cand_audio, ref_sr, model)
            if cand_encodec is not None and ref_encodec_tokens is not None:
                # Flatten and compute cosine similarity
                ref_flat = ref_encodec_tokens.flatten(1).float()
                cand_flat = cand_encodec.flatten(1).float()

                # Match dimensions if needed
                min_dim = min(ref_flat.shape[-1], cand_flat.shape[-1])
                ref_flat = ref_flat[..., :min_dim]
                cand_flat = cand_flat[..., :min_dim]

                encodec_sim = F.cosine_similarity(ref_flat, cand_flat, dim=-1).mean().item()
                scores['encodec'] = max(0.0, min(1.0, encodec_sim))
            else:
                scores['encodec'] = 0.5  # Neutral if extraction fails
        except Exception as e:
            print(f"   ⚠️ Encodec scoring failed: {e}")
            scores['encodec'] = 0.5

        # 2. MFCC similarity
        try:
            ref_mfcc = torchaudio.compliance.kaldi.mfcc(ref_audio, sample_frequency=ref_sr)
            cand_mfcc = torchaudio.compliance.kaldi.mfcc(cand_audio, sample_frequency=ref_sr)
            scores['mfcc'] = compute_mfcc_similarity(ref_mfcc, cand_mfcc)
        except Exception as e:
            print(f"   ⚠️ MFCC scoring failed: {e}")
            scores['mfcc'] = 0.5

        # 3. Spectral similarity
        try:
            ref_spec = torch.stft(ref_audio[0], n_fft=2048, hop_length=512, return_complex=True).abs()
            cand_spec = torch.stft(cand_audio[0], n_fft=2048, hop_length=512, return_complex=True).abs()

            # Flatten and compute cosine similarity
            spec_sim = F.cosine_similarity(
                ref_spec.flatten().unsqueeze(0),
                cand_spec.flatten().unsqueeze(0),
                dim=-1
            ).item()
            scores['spectral'] = max(0.0, min(1.0, spec_sim))
        except Exception as e:
            print(f"   ⚠️ Spectral scoring failed: {e}")
            scores['spectral'] = 0.5

        # 4. Onset correlation (rhythm alignment)
        try:
            scores['onset'] = compute_onset_correlation(cand_audio, piano_roll, sr=ref_sr)
        except Exception as e:
            print(f"   ⚠️ Onset scoring failed: {e}")
            scores['onset'] = 0.5

        # 5. Loudness normality
        try:
            scores['loudness'] = compute_loudness_normality(cand_audio, sr=ref_sr)
        except Exception as e:
            print(f"   ⚠️ Loudness scoring failed: {e}")
            scores['loudness'] = 0.7

        # 6. Clipping penalty
        try:
            scores['clip_penalty'] = detect_clipping(cand_audio)
        except Exception as e:
            print(f"   ⚠️ Clipping detection failed: {e}")
            scores['clip_penalty'] = 0.0

        # 7. Spectral flatness penalty
        try:
            scores['flatness_penalty'] = compute_spectral_flatness_penalty(cand_audio, sr=ref_sr)
        except Exception as e:
            print(f"   ⚠️ Flatness scoring failed: {e}")
            scores['flatness_penalty'] = 0.0

        # Compute weighted total score
        total_score = (
            weights['encodec'] * scores['encodec'] +
            weights['mfcc'] * scores['mfcc'] +
            weights['spectral'] * scores['spectral'] +
            weights['onset'] * scores['onset'] +
            weights['loudness'] * scores['loudness'] -
            weights['clip_penalty'] * scores['clip_penalty'] -
            weights['flatness_penalty'] * scores['flatness_penalty']
        )

        scores['total'] = total_score

        return scores

    except Exception as e:
        print(f"   ❌ Candidate scoring failed: {e}")
        # Return neutral scores
        return {
            'encodec': 0.5, 'mfcc': 0.5, 'spectral': 0.5, 'onset': 0.5,
            'loudness': 0.5, 'clip_penalty': 0.5, 'flatness_penalty': 0.5,
            'total': 0.0
        }


def extract_encodec_tokens_from_audio(audio: torch.Tensor, sr: int, model) -> torch.Tensor:
    """Extract encodec tokens from audio tensor"""
    try:
        # Use model's encodec encoder if available
        if hasattr(model, 'ctrl_enc') and hasattr(model.ctrl_enc, 'encodec'):
            encodec = model.ctrl_enc.encodec

            # Prepare audio
            if audio.ndim == 1:
                audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            elif audio.ndim == 2:
                audio = audio.unsqueeze(0)  # [1, C, T]

            # Resample to 24kHz if needed (encodec native rate)
            if sr != 24000:
                audio = torchaudio.functional.resample(audio, sr, 24000)

            # Extract tokens
            device = next(encodec.parameters()).device
            audio = audio.to(device)

            with torch.no_grad():
                encoded = encodec.encode(audio)
                if isinstance(encoded, tuple):
                    tokens = encoded[0]  # Usually (tokens, scale)
                else:
                    tokens = encoded

            return tokens.cpu()
        else:
            return None
    except Exception as e:
        print(f"   ⚠️ Encodec extraction failed: {e}")
        return None


def generate_best_of_n(
    model,
    audio_file: str,
    piano_roll: np.ndarray,
    amp: np.ndarray,
    rframe: np.ndarray,
    rbend: np.ndarray,
    encodec_tokens: torch.Tensor,
    group: str,
    subgroup: str,
    base_seed: int = 0,
    n_candidates: int = 12,
    **generation_args
) -> tuple:
    """
    Best-of-N sampling with comprehensive reranking

    Generates multiple candidates with parameter variations and returns the one
    that best matches the input audio sample.

    Args:
        model: Pipeline model
        audio_file: Path to reference audio (sample to match)
        piano_roll, amp, rframe, rbend, encodec_tokens: Conditioning inputs
        group, subgroup: Instrument identifiers
        base_seed: Base seed for candidate generation
        n_candidates: Number of candidates to generate (default: 12)
        **generation_args: Additional generation parameters

    Returns:
        (best_output_path, all_candidates_info)
    """
    print(f"\n{'='*80}")
    print(f"🎲 BEST-OF-N SAMPLING (N={n_candidates})")
    print(f"{'='*80}")
    print(f"Reference: {Path(audio_file).name}")
    print(f"Generating {n_candidates} candidates with parameter variations...")

    # Pre-extract reference encodec tokens for scoring
    ref_audio, ref_sr = torchaudio.load(audio_file)
    ref_encodec = extract_encodec_tokens_from_audio(ref_audio, ref_sr, model)

    # Define parameter sweep configurations with SAMPLER DIVERSITY
    # Include: noise levels, instBoost, AND step counts for artifact reduction
    base_steps = generation_args.get('steps', 20)
    base_noise = generation_args.get('noise_level', 0.7)  # Use user's requested noise level
    base_t0 = generation_args.get('t0', 0.7)  # Use user's requested t0
    base_inst_boost = generation_args.get('inst_boost', 2.5)

    # Create variations around the user's requested values (±0.1 for noise, ±0.5 for boost)
    noise_deltas = [-0.1, 0.0, 0.05, 0.1]
    boost_deltas = [-0.5, 0.0, 0.5]

    param_variations = [
        # Core noise/boost variations around user's values
        {"t0": max(0.0, base_t0 + noise_deltas[0]), "noise_level": max(0.0, base_noise + noise_deltas[0]), "inst_boost": base_inst_boost, "steps": base_steps},
        {"t0": base_t0, "noise_level": base_noise, "inst_boost": base_inst_boost, "steps": base_steps},
        {"t0": min(1.0, base_t0 + noise_deltas[2]), "noise_level": min(1.0, base_noise + noise_deltas[2]), "inst_boost": base_inst_boost, "steps": base_steps},
        {"t0": min(1.0, base_t0 + noise_deltas[3]), "noise_level": min(1.0, base_noise + noise_deltas[3]), "inst_boost": base_inst_boost, "steps": base_steps},
        {"t0": base_t0, "noise_level": base_noise, "inst_boost": base_inst_boost + boost_deltas[2], "steps": base_steps},
        {"t0": base_t0, "noise_level": base_noise, "inst_boost": max(1.0, base_inst_boost + boost_deltas[0]), "steps": base_steps},

        # STEP-COUNT SWEEPS (sampler diversity)
        # More steps can reduce artifacts, fewer steps can be faster
        {"t0": base_t0, "noise_level": base_noise, "inst_boost": base_inst_boost, "steps": max(15, base_steps - 10)},
        {"t0": base_t0, "noise_level": base_noise, "inst_boost": base_inst_boost, "steps": base_steps + 10},
        {"t0": base_t0, "noise_level": base_noise, "inst_boost": base_inst_boost, "steps": base_steps + 20},

        # Combined variations
        {"t0": max(0.0, base_t0 - 0.05), "noise_level": max(0.0, base_noise - 0.05), "inst_boost": base_inst_boost, "steps": base_steps + 10},
        {"t0": min(1.0, base_t0 + 0.05), "noise_level": min(1.0, base_noise + 0.05), "inst_boost": base_inst_boost, "steps": base_steps + 15},
        {"t0": base_t0, "noise_level": base_noise, "inst_boost": base_inst_boost + 0.3, "steps": base_steps + 5},
    ]

    # Expand with seed variations
    all_configs = []
    seeds_per_config = max(1, n_candidates // len(param_variations))

    for config in param_variations:
        for i in range(seeds_per_config):
            config_copy = config.copy()
            config_copy['seed'] = base_seed + len(all_configs) * 1000
            all_configs.append(config_copy)
            if len(all_configs) >= n_candidates:
                break
        if len(all_configs) >= n_candidates:
            break

    # Trim to exact count
    all_configs = all_configs[:n_candidates]

    print(f"\n📋 Parameter sweep (with sampler diversity):")
    print(f"   {len(set(c['t0'] for c in all_configs))} noise levels")
    print(f"   {len(set(c['inst_boost'] for c in all_configs))} instBoost values")
    print(f"   {len(set(c['steps'] for c in all_configs))} step counts: {sorted(set(c['steps'] for c in all_configs))}")
    print(f"   {len(all_configs)} total candidates\n")

    # Generate all candidates
    candidates = []

    for idx, config in enumerate(all_configs, 1):
        print(f"   [{idx}/{len(all_configs)}] Generating candidate...")
        print(f"      t0={config['t0']}, noise={config['noise_level']}, instBoost={config['inst_boost']}, steps={config['steps']}, seed={config['seed']}")

        # Merge with base generation args
        gen_args = generation_args.copy()
        gen_args.update(config)

        # Generate
        try:
            output_path = generate(
                model=model,
                piano_roll=piano_roll,
                amp=amp,
                rframe=rframe,
                rbend=rbend,
                encodec_tokens=encodec_tokens,
                group=group,
                subgroup=subgroup,
                audio_file=audio_file,
                **gen_args
            )

            # Score this candidate
            scores = score_candidate(
                output_path=output_path,
                ref_audio_path=audio_file,
                ref_encodec_tokens=ref_encodec,
                piano_roll=piano_roll,
                model=model
            )

            candidates.append({
                'path': output_path,
                'config': config,
                'scores': scores,
                'total_score': scores['total']
            })

            print(f"      ✅ Score: {scores['total']:.3f} (enc={scores['encodec']:.2f}, mfcc={scores['mfcc']:.2f}, spec={scores['spectral']:.2f})")

        except Exception as e:
            print(f"      ❌ Generation failed: {e}")
            continue

    if not candidates:
        raise RuntimeError("All candidates failed to generate")

    # Sort by total score (descending)
    candidates.sort(key=lambda x: x['total_score'], reverse=True)

    # Print ranking
    print(f"\n{'='*80}")
    print(f"🏆 CANDIDATE RANKING")
    print(f"{'='*80}")
    for idx, cand in enumerate(candidates[:5], 1):  # Top 5
        config = cand['config']
        scores = cand['scores']
        print(f"   {idx}. Score: {cand['total_score']:.3f}")
        print(f"      Config: t0={config['t0']}, noise={config['noise_level']}, instBoost={config['inst_boost']}, steps={config['steps']}, seed={config['seed']}")
        print(f"      Scores: enc={scores['encodec']:.2f}, mfcc={scores['mfcc']:.2f}, spec={scores['spectral']:.2f}, onset={scores['onset']:.2f}")
        print(f"      Quality: loud={scores['loudness']:.2f}, clip_pen={scores['clip_penalty']:.2f}, flat_pen={scores['flatness_penalty']:.2f}")
        print(f"      File: {Path(cand['path']).name}")
        print()

    best = candidates[0]
    print(f"✅ Returning best candidate: {Path(best['path']).name}")
    print(f"   Total score: {best['total_score']:.3f}")
    print(f"{'='*80}\n")

    return best['path'], candidates


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
    noise_level=1.0, audio_file=None, fast_mode_variant=None, target_audio_duration=None,
    # Sample recreation enhancement features
    use_time_varying_noise=False, onset_preservation=0.7,
    use_multiresolution_mixing=False,
    use_onset_weighted_encodec=False, encodec_onset_boost=2.0,
    # Test-time adaptation
    use_test_time_adaptation=False, adaptation_steps=10, adaptation_learning_rate=1e-4,
    # Self-consistency ensembling
    use_self_consistency=False, consistency_samples=3, consistency_noise_scale=0.05,
    # Render & extract mode flag
    render_and_extract=False
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
    print(f"🎯 generate() received conditioning with T_slow={T_slow} frames")
    print(f"   piano_roll shape: {piano_roll.shape}")
    print(f"   amp shape: {amp.shape}")
    print(f"   rframe shape: {rframe.shape}")
    print(f"   rbend shape: {rbend.shape}")
    print(f"   encodec shape: {encodec_tokens.shape}")
    if original_audio_length:
        expected_duration = original_audio_length / 44100.0
        expected_frames = int(expected_duration * 43.066)
        print(f"   original_audio_length: {original_audio_length} samples = {expected_duration:.2f}s = {expected_frames} frames @43.066fps")
        if expected_frames != T_slow:
            print(f"   ⚠️  WARNING: Expected {expected_frames} frames but got {T_slow} frames!")

    # build conds
    # Ensure encodec_tokens is a torch tensor (it should be, but convert if needed)
    if isinstance(encodec_tokens, np.ndarray):
        encodec_tokens = torch.from_numpy(encodec_tokens).long()

    # Validate encodec tokens to prevent CUDA indexing errors
    encodec_vocab_size = 1024  # EnCodec uses 1024 tokens per codebook
    if encodec_tokens.numel() > 0:
        min_token = encodec_tokens.min().item()
        max_token = encodec_tokens.max().item()
        if min_token < 0 or max_token >= encodec_vocab_size:
            print(f"❌ ERROR: Invalid encodec tokens detected!")
            print(f"   Min token: {min_token}, Max token: {max_token}")
            print(f"   Valid range: [0, {encodec_vocab_size-1}]")
            print(f"   Clamping tokens to valid range...")
            encodec_tokens = torch.clamp(encodec_tokens, 0, encodec_vocab_size - 1)

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
    # CRITICAL: In render & extract mode, ALWAYS extract GT latents even if noise_level=1.0
    # This allows mixing with FluidSynth-rendered audio GT
    if float(noise_level) >= 1.0 and not render_and_extract:
        # Pure noise (original behavior) - but NOT in render & extract mode
        x = torch.randn_like(sample_patch.to(device=device, dtype=tokens.dtype))
    else:
        # Try to extract ground truth latents for proper noise mixing
        # OR when render_and_extract=True (regardless of noise level)
        gt_latents = None
        if audio_file is not None:
            # Check if audio_file is a MIDI file - if so, render it to audio first
            audio_file_path = audio_file
            if Path(audio_file).suffix.lower() in ['.mid', '.midi']:
                print(f"⚠️  audio_file is MIDI ({Path(audio_file).name}), rendering to audio for latent extraction...")
                try:
                    # Render MIDI to audio using FluidSynth
                    rendered_audio = render_midi_to_audio(audio_file, instrument_group=subgroup)
                    audio_file_path = rendered_audio
                    print(f"✅ Rendered MIDI to audio: {Path(rendered_audio).name}")
                except Exception as e:
                    print(f"❌ Failed to render MIDI to audio: {e}")
                    print(f"   Falling back to pure noise generation")
                    audio_file_path = None

            if audio_file_path is not None:
                gt_latents = extract_ground_truth_latents(audio_file_path, model, target_duration=target_audio_duration)

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
    dt      = float(t0) / float(steps)  # Use t0 to match noise level with timestep schedule

    for i in range(steps, 0, -1):
        t_cont = torch.full((x.shape[0],), i * dt, device=device, dtype=torch.float32)
        t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

        # instrument ON/OFF patches
        tokens_adapt = tokens.to(dtype=next(model.cond_adapter.parameters()).dtype)

        # Simplified: when cfg_weight=1 and no onset boost, use single conditioning like trainer
        use_cfg = float(cfg_weight) > 1.0 or float(onset_guidance_boost) > 0.0

        if use_cfg:
            tokens_on  = tokens_adapt.clone(); tokens_on[:, 0, :] *= float(inst_boost)
            tokens_off = tokens_adapt.clone(); tokens_off[:, 0, :].zero_()
            cond_on  = model.cond_adapter(tokens_on,  T_out=x.shape[-1], scale=float(adapter_scale)).to(x)
            cond_off = model.cond_adapter(tokens_off, T_out=x.shape[-1], scale=float(adapter_scale)).to(x)
        else:
            # Single conditioning path (matches trainer preview behavior)
            cond_patch = model.cond_adapter(tokens_adapt, T_out=x.shape[-1], scale=float(adapter_scale)).to(x)

        # PR-guided masking/sharpening
        B, C, H, T_lat = x.shape
        pr = conds["piano_roll"].to(device=x.device, dtype=x.dtype)
        if pr.shape[-1] != T_lat:
            pr = F.interpolate(pr, size=T_lat, mode="nearest")

        W_hp = _bank_softplus_resized_compat(model, H, device=x.device, dtype=x.dtype)

        if use_cfg:
            # CFG mode: use normalized piano roll with pitch snap
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
            pr_high = pr_target * (1.0 + float(pitch_fidelity_boost))
            pr_low  = pr_target * (1.0 - float(pitch_fidelity_boost) * 0.5)
            H_on  = torch.einsum('bpt,hp->bht', pr_high, W_hp)
            H_off = torch.einsum('bpt,hp->bht', pr_low,  W_hp)

            sharp = 1.0 + float(pitch_fidelity_boost) * 0.5
            H_on  = (H_on  + 1e-6).pow(sharp)
            H_off = (H_off + 1e-6).pow(sharp)
            H_on  = H_on  / (H_on.amax(dim=1, keepdim=True)  + 1e-6)
            H_off = H_off / (H_off.amax(dim=1, keepdim=True) + 1e-6)

            # Simplified conditioning application to match trainer (no active mask or adaptive scaling)
            # This avoids hard discontinuities that can cause clicking artifacts
            cond_on  = cond_on  * H_on.unsqueeze(1)
            cond_off = cond_off * H_off.unsqueeze(1)
        else:
            # Simple pitch-height masking (matches trainer - uses normalized piano roll if trained that way)
            pr_for_mask = pr / (pr.amax(dim=1, keepdim=True) + 1e-6)  # Normalize to match training
            Hmap = torch.einsum('bpt,hp->bht', pr_for_mask, W_hp)
            cond_patch = cond_patch * Hmap.unsqueeze(1)

        # transformer with ControlBranch residuals
        if ctrl_res_list is not None:
            model._ctrl_residuals = ctrl_res_list

        if use_cfg:
            # onset-weighted guidance for CFG
            if pr_target.shape[-1] > 1:
                onset = (pr_target[:, :, 1:] > 0.1) & (pr_target[:, :, :-1] <= 0.1)
                onset = F.pad(onset.float().amax(dim=1, keepdim=True), (1,0))
            else:
                onset = torch.zeros_like(pr_target[:, :1, :])
            if onset.shape[-1] != T_lat:
                onset = F.interpolate(onset, size=T_lat, mode="nearest")
            base_guid = max(1.0, float(cfg_weight))
            step_guid = base_guid * (1.0 + float(onset_guidance_boost) * onset)  # [B,1,T]

            v_un = model._call_transformer_no_xattn(latents=x + cond_off, t=t_idx)
            v_co = model._call_transformer_no_xattn(latents=x + cond_on,  t=t_idx)
            v_pred = v_un + step_guid.unsqueeze(1) * (v_co - v_un)
        else:
            # Simple single-pass like trainer
            x_in = x + cond_patch
            v_pred = model._call_transformer_no_xattn(latents=x_in, t=t_idx)

        # SELF-CONSISTENCY ENSEMBLING: In final 10-20% of steps, run multiple predictions and average
        if use_self_consistency and i <= max(1, int(steps * 0.2)):  # Last 20% of steps
            if i == max(1, int(steps * 0.2)):  # Print once at start of ensembling
                print(f"🔄 Self-consistency ensembling active (last {int(steps * 0.2)} steps, {consistency_samples} samples)")

            # Store original prediction
            predictions = [v_pred]

            # Generate additional predictions with small noise perturbations
            for _ in range(consistency_samples - 1):
                # Add small noise to x for diversity
                x_noisy = x + torch.randn_like(x) * consistency_noise_scale

                if use_cfg:
                    v_un_noisy = model._call_transformer_no_xattn(latents=x_noisy + cond_off, t=t_idx)
                    v_co_noisy = model._call_transformer_no_xattn(latents=x_noisy + cond_on,  t=t_idx)
                    v_pred_noisy = v_un_noisy + step_guid.unsqueeze(1) * (v_co_noisy - v_un_noisy)
                else:
                    x_in_noisy = x_noisy + cond_patch
                    v_pred_noisy = model._call_transformer_no_xattn(latents=x_in_noisy, t=t_idx)

                predictions.append(v_pred_noisy)

            # Average all predictions (not the noisy samples!)
            v_pred = torch.stack(predictions).mean(dim=0)

        x = x - dt * v_pred  # Use dt to match timestep schedule
        if i == steps:
            if use_cfg:
                print(f"[CondEnergy] on={cond_on.norm().item():.3f} off={cond_off.norm().item():.3f}")
            else:
                print(f"[CondEnergy] patch={cond_patch.norm().item():.3f}")

    model._ctrl_residuals = None
    print("Decoding audio...")

    # decode
    if original_audio_length is not None:
        audio_len = int(round(original_audio_length * sr_out / DCAE_SR))
        expected_duration = audio_len / sr_out
        print(f"🎵 DCAE DECODE DIAGNOSTICS:")
        print(f"   Using original_audio_length: {original_audio_length} samples = {original_audio_length/44100:.2f}s")
        print(f"   Calculated audio_len: {audio_len} samples = {expected_duration:.2f}s at {sr_out} Hz")
    else:
        # Calculate correct audio length from latent frames using DCAE hop size
        # Use actual latent temporal dimension from the generated latents
        T_latent_actual = x.shape[-1]
        audio_len = int(round(T_latent_actual * DCAE_HOP * (sr_out / DCAE_SR)))
        expected_duration = audio_len / sr_out
        print(f"🎵 DCAE DECODE DIAGNOSTICS:")
        print(f"   NO original_audio_length - Calculated from latent frames:")
        print(f"   {T_latent_actual} latent frames * {DCAE_HOP} hop * ({sr_out}/{DCAE_SR}) = {audio_len} samples = {expected_duration:.2f}s")

    p = next(model.dcae.parameters(), None)
    dev = p.device if p is not None else getattr(model.dcae, "device", device)
    dtype = p.dtype if p is not None else torch.float32
    x_for_dcae = x[:1].to(device=dev, dtype=dtype)
    audio_lengths = torch.tensor([audio_len], dtype=torch.long, device=dev)

    print(f"   Latent shape to DCAE: {x_for_dcae.shape}")
    print(f"   audio_lengths tensor: {audio_lengths.item()} samples")

    if use_overlap_decoder and hasattr(model.dcae, 'decode_overlap'):
        print("   🔊 Using overlap decoder")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev.type=="cuda")):
            result = model.dcae.decode_overlap(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)

        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 2:
            sr_pred, wav_pred = result
        else:
            sr_pred = sr_out
            wav_pred = result

        # Check if wav_pred is a list or tensor
        if isinstance(wav_pred, list):
            print(f"   ⚠️  decode_overlap returned LIST with {len(wav_pred)} elements")
            if len(wav_pred) > 0:
                if torch.is_tensor(wav_pred[0]):
                    wav_shape = wav_pred[0].shape
                    actual_duration = wav_shape[-1] / sr_pred if len(wav_shape) > 0 else 0
                    print(f"   ✅ DCAE output: sr={sr_pred}, wav[0] shape={wav_shape}, actual duration={actual_duration:.2f}s")
                    if abs(actual_duration - expected_duration) > 1.0:
                        print(f"   ⚠️  WARNING: Output duration ({actual_duration:.2f}s) != expected ({expected_duration:.2f}s)!")
        else:
            wav_shape = wav_pred.shape
            actual_duration = wav_shape[-1] / sr_pred
            print(f"   ✅ DCAE output: sr={sr_pred}, wav shape={wav_shape}, actual duration={actual_duration:.2f}s")
            if abs(actual_duration - expected_duration) > 1.0:
                print(f"   ⚠️  WARNING: Output duration ({actual_duration:.2f}s) != expected ({expected_duration:.2f}s)!")
    else:
        sr_pred, wav_pred = model.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)

    # Handle list or tensor format
    if isinstance(wav_pred, list):
        wav = wav_pred[0].float().cpu() if torch.is_tensor(wav_pred[0]) else wav_pred[0]
    else:
        wav = wav_pred[0].float().cpu()

    # Apply final audio processing (compression + high-pass filter) - DISABLED to match training previews
    # wav = apply_final_audio_processing(wav, sample_rate=sr_pred)

    out_dir = Path("./generated_ui"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_seed{seed}_cfg{cfg_weight:.1f}.wav"
    torchaudio.save(str(out_path), wav, sr_pred)
    print(f"✅ Wrote: {out_path}")
    return str(out_path)

def analyze_voice_pitch_range(voice_piano_roll):
    """
    Analyze the pitch range of a voice from its piano roll.

    Args:
        voice_piano_roll: Piano roll array of shape [88, T] or [1, 88, T]

    Returns:
        tuple: (min_note, max_note, mean_note) as MIDI note numbers
    """
    if voice_piano_roll.ndim == 3:
        voice_piano_roll = voice_piano_roll[0]  # Remove batch dimension

    # Find all active pitches (piano roll is [88, T])
    active_pitches = np.where(voice_piano_roll.sum(axis=1) > 0)[0]

    if len(active_pitches) == 0:
        # No notes, return middle C range as default
        return 60, 60, 60

    # Convert to MIDI note numbers (piano roll starts at A0 = MIDI 21)
    min_note = active_pitches.min() + 21
    max_note = active_pitches.max() + 21

    # Calculate weighted mean based on note activity
    note_activity = voice_piano_roll.sum(axis=1)
    weighted_mean = np.average(active_pitches + 21, weights=note_activity[active_pitches])

    return int(min_note), int(max_note), int(weighted_mean)


def transpose_notes_above_group_minimum(notes, group):
    """
    Transpose notes up by octaves if they fall below the group's minimum note limit.
    Only used in monophonic mode to prevent notes that are too low for the group.

    Args:
        notes: List of pretty_midi.Note objects
        group: Group name (e.g., 'strings', 'winds', 'brass')

    Returns:
        int: Number of notes that were transposed
    """
    group_lower = group.lower()
    if group_lower not in GROUP_NOTE_LIMITS:
        return 0

    min_note = GROUP_NOTE_LIMITS[group_lower]
    transposed_count = 0

    for note in notes:
        original_pitch = note.pitch
        while note.pitch < min_note and note.pitch + 12 <= 127:
            note.pitch += 12  # Transpose up one octave
            transposed_count += 1

        if note.pitch != original_pitch:
            print(f"      ⬆️  Transposed note from {original_pitch} to {note.pitch} (below {group} minimum of {min_note})")

    return transposed_count

def assign_instrument_for_voice(group, voice_range, used_instruments=None):
    """
    Assign the best instrument subgroup for a voice based on its pitch range.

    Args:
        group: Instrument group (strings, winds, brass)
        voice_range: tuple of (min_note, max_note, mean_note)
        used_instruments: list of already assigned instruments to avoid duplicates

    Returns:
        str: Best matching instrument subgroup
    """
    if used_instruments is None:
        used_instruments = []

    group_lower = group.lower()

    # Check if this group has defined ranges
    if group_lower not in INSTRUMENT_RANGES:
        print(f"⚠️ Arrange mode not supported for group '{group}', using default subgroup")
        # Return first available subgroup for this group
        if group_lower in ["strings", "winds", "brass"]:
            return list(INSTRUMENT_RANGES.get(group_lower, {}).keys())[0]
        return group_lower  # Fallback to group name

    min_note, max_note, mean_note = voice_range
    instruments = INSTRUMENT_RANGES[group_lower]

    # Calculate overlap score for each instrument
    scores = {}
    for inst_name, (inst_min, inst_max) in instruments.items():
        # Calculate how well the voice range fits within the instrument range
        overlap_min = max(min_note, inst_min)
        overlap_max = min(max_note, inst_max)

        if overlap_max >= overlap_min:
            # There's overlap
            overlap_range = overlap_max - overlap_min
            voice_range_size = max_note - min_note

            # Score based on percentage of voice range covered
            if voice_range_size > 0:
                coverage = overlap_range / voice_range_size
            else:
                coverage = 1.0

            # Bonus if mean note is within instrument's optimal range
            mean_in_range = inst_min <= mean_note <= inst_max
            mean_bonus = 0.5 if mean_in_range else 0.0

            # Penalty if instrument was already used (encourage variety)
            used_penalty = 0.3 if inst_name in used_instruments else 0.0

            scores[inst_name] = coverage + mean_bonus - used_penalty
        else:
            # No overlap, penalize based on distance
            if mean_note < inst_min:
                distance = inst_min - mean_note
            else:
                distance = mean_note - inst_max
            scores[inst_name] = -distance / 12.0  # Negative score based on octaves away

    # Select instrument with highest score
    if scores:
        best_instrument = max(scores.items(), key=lambda x: x[1])
        # Log all instrument scores for transparency
        print(f"      🎯 Instrument Scores (sorted by best fit):")
        for inst, score in sorted(scores.items(), key=lambda x: -x[1]):
            used_marker = " [ALREADY USED]" if inst in used_instruments else ""
            print(f"         {inst}: {score:.2f}{used_marker}")
        print(f"      ✅ Best match: {best_instrument[0]} (score: {best_instrument[1]:.2f})")
        return best_instrument[0]

    # Fallback: return first instrument in the group
    return list(instruments.keys())[0]


@torch.no_grad()
def generate_monophonic_multiple(
    model, piano_roll, amp, rframe, rbend, encodec_tokens,
    group, subgroup, seed, steps, adapter_scale, cfg_weight, t0, sr_out,
    instrument_strength=1.0, inst_boost=2.5,
    piano_roll_gain=1.0, amp_gain=1.0, rframe_gain=1.0, rbend_gain=1.0, encodec_gain=1.0,
    use_overlap_decoder=True, original_audio_length=None,
    pitch_fidelity_boost=1.0, onset_guidance_boost=2.0, pitch_snap_strength=0.5,
    noise_level=1.0, audio_file=None, progress=None, voice_complete_callback=None,
    enable_voice_separation=True, arrange_mode=False, fast_mode_variant=None,
    fatten_mode=False, fatten_type="fake", inpaint_voice_index=None
):
    """
    Generate multiple monophonic outputs from separated voices and create a mixed sum.

    Args:
        enable_voice_separation: If True, separate voices from piano roll. If False, assume
                                piano roll already contains separated tracks (e.g., multi-track MIDI)
        arrange_mode: If True, automatically assign instrument subgroups based on voice pitch ranges
        fatten_mode: If True, create octave-up versions of each voice to double the track count
        fatten_type: "real" (transpose piano roll) or "fake" (pitch shift audio output)
    """
    print("🎵 Starting monophonic multiple voice generation")
    print(f"   Voice separation enabled: {enable_voice_separation}")
    print(f"   Arrange mode enabled: {arrange_mode}")
    print(f"   🎚️ Fatten mode: {fatten_mode}, Type: {fatten_type}")

    # Separate the piano roll into voices (only if voice separation is enabled)
    if enable_voice_separation:
        print("   Separating piano roll into voices...")
        voices = separate_piano_roll_voices(piano_roll)
    else:
        print("   Skipping voice separation - using tracks as-is (multi-track MIDI)")
        # For multi-track MIDI, piano roll already contains separated tracks
        # We just need to wrap it in a list to process as a single "voice"
        voices = [piano_roll]

    if len(voices) == 1:
        print("⚠️ Only one voice detected, falling back to regular generation")
        return generate(
            model, piano_roll, amp, rframe, rbend, encodec_tokens,
            group, subgroup, steps, seed, adapter_scale, cfg_weight, t0, sr_out,
            instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
            use_overlap_decoder, original_audio_length, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength,
            noise_level, audio_file, fast_mode_variant
        )

    # Generate each voice separately
    voice_outputs = []
    base_seed = int(seed) if seed > 0 else torch.seed() % 2**31

    out_dir = Path("./generated_ui"); out_dir.mkdir(exist_ok=True)
    timestamp = time.strftime('%Y%m%d-%H%M%S')

    # Track used instruments for arrange mode variety
    used_instruments = []

    for i, voice_pr in enumerate(voices):
        # VOICE SELECTION: Skip voices that aren't being regenerated
        if inpaint_voice_index is not None:
            # Voice indices are 1-based (voice 1, voice 2, etc.)
            if (i + 1) != inpaint_voice_index:
                print(f"⏭️  Skipping voice {i + 1} (regenerating voice {inpaint_voice_index} only)")
                continue

        if progress:
            progress_val = 0.5 + (i / len(voices)) * 0.4  # 50-90% range
            progress(progress_val, desc=f"Generating voice {i+1}/{len(voices)}...")

        print(f"🎼 Generating voice {i+1}/{len(voices)}")

        # Arrange mode: auto-assign instrument subgroup based on pitch range
        if arrange_mode:
            voice_range = analyze_voice_pitch_range(voice_pr)
            min_note, max_note, mean_note = voice_range
            voice_subgroup = assign_instrument_for_voice(group, voice_range, used_instruments)
            used_instruments.append(voice_subgroup)
            print(f"   🎼 Arrange mode - Voice {i+1}:")
            print(f"      Range: MIDI {min_note}-{max_note} (mean: {mean_note})")
            print(f"      ✅ Assigned: {voice_subgroup}")
        else:
            voice_subgroup = subgroup

        # Use different seed for each voice for variety
        voice_seed = base_seed + i * 1000

        voice_output = generate(
            model, voice_pr, amp, rframe, rbend, encodec_tokens,
            group, voice_subgroup, steps, voice_seed, adapter_scale, cfg_weight, t0, sr_out,
            instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
            use_overlap_decoder, original_audio_length, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength,
            noise_level, audio_file, fast_mode_variant
        )

        # Rename the output to include voice number
        voice_path = out_dir / f"{timestamp}_voice{i+1}_seed{voice_seed}_cfg{cfg_weight:.1f}.wav"
        shutil.move(voice_output, str(voice_path))
        voice_outputs.append(str(voice_path))
        print(f"✅ Voice {i+1} saved: {voice_path.name}")

        # Notify callback that this voice is complete (for incremental updates)
        if voice_complete_callback:
            voice_complete_callback(i, str(voice_path), len(voices))

    # Fatten mode: create octave-up versions of each voice
    if fatten_mode:
        print(f"\n🎚️ FATTEN MODE ({fatten_type.upper()}): Creating octave-up voices...")
        original_voice_count = len(voice_outputs)

        if fatten_type == "real":
            # Real mode: transpose piano roll up 12 semitones and generate again
            print(f"   Generating {original_voice_count} octave-up voices from transposed piano rolls...")
            for i, voice_pr in enumerate(voices):
                # VOICE SELECTION: Skip voices that aren't being regenerated
                if inpaint_voice_index is not None:
                    if (i + 1) != inpaint_voice_index:
                        print(f"   ⏭️  Skipping octave for voice {i + 1} (regenerating voice {inpaint_voice_index} only)")
                        continue

                if progress:
                    progress_val = 0.5 + ((len(voice_outputs) / (len(voices) * 2))) * 0.4
                    progress(progress_val, desc=f"Generating octave-up voice {i+1}/{len(voices)}...")

                print(f"   🎼 Generating octave-up voice {i+1}/{len(voices)}")

                # Transpose piano roll up one octave (shift all pitch indices up by 12)
                octave_voice_pr = voice_pr.clone()
                # Piano roll shape: (batch, time, pitch)
                # We need to shift the pitch dimension up by 12 semitones
                # Create a new tensor shifted up
                shifted_pr = torch.zeros_like(octave_voice_pr)
                if len(octave_voice_pr.shape) == 3:
                    # Shift pitch up by 12 (move data from index i to i+12)
                    shifted_pr[:, :, 12:] = octave_voice_pr[:, :, :-12]
                else:
                    # Handle 2D case (time, pitch)
                    shifted_pr[:, 12:] = octave_voice_pr[:, :-12]

                octave_voice_pr = shifted_pr

                # Determine subgroup for this octave voice (use same as original if arrange mode)
                if arrange_mode:
                    octave_subgroup = used_instruments[i]
                else:
                    octave_subgroup = subgroup

                # Use different seed for octave voice
                octave_seed = base_seed + (original_voice_count + i) * 1000

                octave_output = generate(
                    model, octave_voice_pr, amp, rframe, rbend, encodec_tokens,
                    group, octave_subgroup, steps, octave_seed, adapter_scale, cfg_weight, t0, sr_out,
                    instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
                    use_overlap_decoder, original_audio_length, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength,
                    noise_level, audio_file, fast_mode_variant
                )

                # Rename the output to include voice number
                octave_path = out_dir / f"{timestamp}_voice{original_voice_count+i+1}_octave_seed{octave_seed}_cfg{cfg_weight:.1f}.wav"
                shutil.move(octave_output, str(octave_path))
                voice_outputs.append(str(octave_path))
                print(f"   ✅ Octave-up voice {i+1} saved: {octave_path.name}")

                # Notify callback for octave voice
                if voice_complete_callback:
                    voice_complete_callback(original_voice_count + i, str(octave_path), len(voices) * 2)

        else:  # fake mode
            # Fake mode: pitch shift existing outputs up an octave
            print(f"   Pitch shifting {original_voice_count} voices up an octave...")
            import librosa
            import soundfile as sf

            for i, voice_path in enumerate(list(voice_outputs)):  # Use list() to avoid modifying during iteration
                print(f"   🎚️ Pitch shifting voice {i+1}/{original_voice_count}...")

                # Load audio
                audio, sr = librosa.load(voice_path, sr=None)

                # Pitch shift up 12 semitones (one octave)
                shifted_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=12)

                # Save shifted version
                octave_path = out_dir / f"{timestamp}_voice{original_voice_count+i+1}_octave_fake.wav"
                sf.write(str(octave_path), shifted_audio, sr)
                voice_outputs.append(str(octave_path))
                print(f"   ✅ Octave-up voice {i+1} saved: {octave_path.name}")

                # Notify callback for octave voice
                if voice_complete_callback:
                    voice_complete_callback(original_voice_count + i, str(octave_path), len(voices) * 2)

        print(f"   ✅ Fatten mode complete: {original_voice_count} original + {original_voice_count} octave = {len(voice_outputs)} total voices")

    # Log arrange mode summary
    if arrange_mode and used_instruments:
        print(f"\n{'='*60}")
        print(f"📋 ARRANGE MODE SUMMARY")
        print(f"{'='*60}")
        print(f"Total voices generated: {len(used_instruments)}")
        print(f"Unique instruments used: {', '.join(set(used_instruments))}")
        print(f"Voice assignments: {', '.join(f'V{i+1}={inst}' for i, inst in enumerate(used_instruments))}")
        print(f"{'='*60}\n")

    # Normalize all voice outputs to same length
    if progress:
        progress(0.90, desc="Normalizing voice lengths...")

    normalized_voice_outputs = normalize_audio_lengths(voice_outputs)

    # Create mixed output
    if progress:
        progress(0.95, desc="Mixing voices...")

    mixed_path = out_dir / f"{timestamp}_mixed_seed{base_seed}_cfg{cfg_weight:.1f}.wav"
    mix_audio_files(normalized_voice_outputs, str(mixed_path))

    if progress:
        progress(1.0, desc="Done!")

    # Return all outputs as a tuple: (mixed_output, individual_voices)
    return {
        "mixed": str(mixed_path),
        "voices": normalized_voice_outputs,
        "voice_count": len(voices)
    }

# ------------------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------------------
def run_generation(
    audio_file, group, subgroup, seed, steps, adapter_scale, cfg_weight, t0,
    instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
    use_overlap_decoder, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength, noise_level,
    monophonic_mode, midi_mode, render_and_extract, render_extract_mono, tempo_override, progress=gr.Progress(track_tqdm=True)
):
    if audio_file is None:
        raise gr.Error("Please upload an audio or MIDI file or pick a random one.")

    # Check if input is MIDI file
    is_midi = is_midi_file(audio_file)

    # CRITICAL FIX: Calculate window_slow based on actual file duration
    # Don't hardcode to 1024 frames (23.78s limit)
    # Use 43.066 fps to match piano roll frame rate
    fps = 43.066
    if is_midi:
        # For MIDI files, get actual duration from MIDI
        try:
            import pretty_midi
            midi_data = pretty_midi.PrettyMIDI(audio_file)
            actual_duration = max(midi_data.get_end_time(), 1.0)
            win_slow = int(actual_duration * fps)
            print(f"🎵 MIDI duration: {actual_duration:.2f}s → window_slow = {win_slow} frames (at {fps} fps)")
        except Exception as e:
            print(f"⚠️ Could not determine MIDI duration: {e}, using default 1024")
            win_slow = int(getattr(MODEL.hparams, "window_slow", 1024))
    else:
        # For audio files, get actual duration from audio
        try:
            import torchaudio
            wav, sr = torchaudio.load(audio_file)
            actual_duration = wav.shape[-1] / sr
            win_slow = int(actual_duration * fps)
            print(f"🎵 Audio duration: {actual_duration:.2f}s → window_slow = {win_slow} frames (at {fps} fps)")
        except Exception as e:
            print(f"⚠️ Could not determine audio duration: {e}, using default 1024")
            win_slow = int(getattr(MODEL.hparams, "window_slow", 1024))

    if is_midi and midi_mode:
        progress(0, desc="Processing MIDI file…")

        if render_and_extract:
            # Mode 1: Render MIDI to audio, then extract full conditioning
            progress(0.1, desc="Rendering MIDI to audio…")

            # CRITICAL WORKAROUND: Stretch MIDI by 4x before FluidSynth rendering
            # FluidSynth renders at the MIDI's internal tempo, which may be 4x too fast
            # if the MIDI was created without proper tempo metadata
            stretched_midi_path = str(Path(audio_file).parent / f"stretched_4x_{Path(audio_file).name}")
            print(f"🐌 Stretching MIDI by 4x (tempo scale 0.25) to fix FluidSynth speed: {stretched_midi_path}")
            modify_midi_tempo(audio_file, stretched_midi_path, tempo_scale=0.25)  # 0.25 = 1/4 speed = 4x slower

            rendered_audio = render_midi_to_audio(stretched_midi_path, instrument_group=subgroup)

            progress(0.3, desc="Extracting conditioning from rendered audio…")
            extraction = extract_conditioning_from_audio(rendered_audio, instrument_group=subgroup,
                                extract_formats=extract_formats)

            # Use stretched MIDI for piano roll (to match FluidSynth render timing)
            progress(0.5, desc="Loading MIDI piano roll…")
            # TIMING FIX: Use stretched MIDI so piano roll matches FluidSynth render
            # to ensure piano roll and audio conditioning have matching timing scales
            pr_midi, _, _, _, _ = midi_to_piano_roll_conditioning(stretched_midi_path, win_slow, fps=43.066, tempo_override=None)
            pr_duration = pr_midi.shape[1] / (43.066 * (tempo_override or 120.0) / 120.0) if tempo_override else pr_midi.shape[1] / 43.066
            print(f"🎵 Piano roll: {pr_midi.shape}, estimated duration: {pr_duration:.2f}s")

            # Load conditioning from rendered audio
            _, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=win_slow)

            # TIMING FIX: Use MIDI piano roll to match individual tracks
            # Resize MIDI piano roll to match conditioning length if needed
            conditioning_length = amp.shape[-1]
            if pr_midi.shape[1] != conditioning_length:
                print(f"🎵 Resizing MIDI piano roll: {pr_midi.shape[1]} → {conditioning_length} frames")
                if pr_midi.shape[1] > conditioning_length:
                    pr_midi = pr_midi[:, :conditioning_length]  # truncate
                else:
                    pad_width = conditioning_length - pr_midi.shape[1]
                    pr_midi = np.pad(pr_midi, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)  # pad

            pr = pr_midi
            print(f"🎵 Using MIDI piano roll: {pr.shape} (to match individual tracks)")

            # TIMING DEBUG: Force use of piano roll calculation instead of FluidSynth audio length
            # Get original length from rendered audio
            try:
                wav, sr = torchaudio.load(rendered_audio)
                fluidsynth_len = wav.shape[-1]
                fluidsynth_duration = fluidsynth_len / sr
                print(f"🎵 FluidSynth audio length: {fluidsynth_len} samples ({fluidsynth_duration:.2f}s at {sr}Hz)")

                # Calculate what the length SHOULD be based on piano roll
                expected_duration = pr.shape[1] / 43.066
                expected_len = int(expected_duration * sr)
                print(f"🎵 Expected audio length: {expected_len} samples ({expected_duration:.2f}s)")

                # FORCE use of piano roll-based calculation
                orig_len = None  # This will force the piano roll calculation path
                print(f"🎵 FORCING piano roll calculation instead of FluidSynth length")

            except Exception:
                orig_len = None
                print("⚠️ Could not determine rendered audio length")

            print("🎼 Mode: MIDI -> Audio -> Full Conditioning (with original MIDI PR)")

        else:
            # Mode 2: Use MIDI directly for piano roll only, empty other conditioning
            progress(0.2, desc="Converting MIDI to piano roll…")
            pr, amp, rfr, rbd, enc = midi_to_piano_roll_conditioning(audio_file, win_slow, tempo_override=tempo_override)
            orig_len = None  # No audio length reference
            print("🎼 Mode: MIDI -> Piano Roll Only")

    else:
        # Original audio file processing
        if render_and_extract:
            # Mode: Extract MIDI from audio with Basic Pitch, render with FluidSynth, then extract conditioning
            progress(0, desc="Extracting MIDI from audio with Basic Pitch…")

            # Extract MIDI using Basic Pitch (no voice separation for render_and_extract mode)
            midi_result = save_basic_pitch_midi_with_voices(
                audio_file,
                subgroup=subgroup,
                progress=None,
                tempo=120.0,
                separate_voices=False,
                monophonic=render_extract_mono
            )
            extracted_midi = midi_result['main_midi']

            print(f"🎵 Basic Pitch extracted MIDI: {extracted_midi}")

            # Render the extracted MIDI to audio using FluidSynth
            progress(0.3, desc="Rendering extracted MIDI to audio with FluidSynth…")
            rendered_audio = render_midi_to_audio(extracted_midi, instrument_group=subgroup)

            print(f"🎵 Rendered audio from extracted MIDI: {rendered_audio}")

            # Extract conditioning from the rendered audio
            progress(0.5, desc="Extracting conditioning from rendered audio…")
            extraction = extract_conditioning_from_audio(rendered_audio, instrument_group=subgroup,
                                extract_formats=extract_formats)

            # Load conditioning (EXACT same as normal MIDI mode)
            progress(0.7, desc="Loading conditioning…")
            _, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=win_slow)

            # Use MIDI for piano roll (EXACT same as normal MIDI mode)
            progress(0.8, desc="Loading MIDI piano roll…")
            pr, _, _, _, _ = midi_to_piano_roll_conditioning(extracted_midi, win_slow, fps=43.066, tempo_override=None)

            # Resize MIDI piano roll to match conditioning length (EXACT same as normal MIDI mode)
            conditioning_length = amp.shape[-1]
            if pr.shape[1] != conditioning_length:
                print(f"🎵 Resizing MIDI piano roll: {pr.shape[1]} → {conditioning_length} frames")
                if pr.shape[1] > conditioning_length:
                    pr = pr[:, :conditioning_length]  # truncate
                else:
                    # pad with zeros
                    pad_amount = conditioning_length - pr.shape[1]
                    pr = np.pad(pr, ((0, 0), (0, pad_amount)), mode='constant', constant_values=0)

            print(f"🎵 Final piano roll: {pr.shape}, amp: {amp.shape}")

            # Get original audio length for exact decode length
            try:
                wav, sr = torchaudio.load(rendered_audio)
                orig_len = wav.shape[-1]
            except Exception:
                orig_len = None

            # IMPORTANT: Replace audio_file with rendered_audio for GT latent extraction (noise < 1.0)
            audio_file = rendered_audio
            print(f"🎵 Render & Extract mode: Audio → Basic Pitch MIDI → FluidSynth → Conditioning + GT Latents")
        else:
            # Normal mode: extract conditioning directly from audio
            progress(0, desc="Extracting conditioning from audio…")

            # original len (for exact decode length)
            try:
                wav, sr = torchaudio.load(audio_file)
                orig_len = wav.shape[-1]
            except Exception:
                orig_len = None

            extraction = extract_conditioning_from_audio(audio_file, instrument_group=subgroup,
                                extract_formats=extract_formats)
            progress(0.25, desc="Loading conditioning…")
            pr, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=win_slow)

    # Check if we should disable voice separation for multitrack MIDI
    should_skip_voice_separation = False
    if midi_mode and is_midi_file(audio_file):
        is_multi, track_count, _ = is_multitrack_midi(audio_file)
        if is_multi:
            should_skip_voice_separation = True
            print(f"🎼 Multitrack MIDI detected ({track_count} tracks) - DISABLING voice separation")
            print("   Individual tracks should be processed separately, not combined and re-separated")

    if monophonic_mode and not should_skip_voice_separation:
        progress(0.5, desc="Generating multiple voices…")
        result = generate_monophonic_multiple(
            MODEL, pr, amp, rfr, rbd, enc,
            group, subgroup, int(seed), int(steps), float(adapter_scale), float(cfg_weight), float(t0),
            sr_out=32000, instrument_strength=float(instrument_strength), inst_boost=float(inst_boost),
            piano_roll_gain=float(piano_roll_gain), amp_gain=float(amp_gain),
            rframe_gain=float(rframe_gain), rbend_gain=float(rbend_gain), encodec_gain=float(encodec_gain),
            use_overlap_decoder=bool(use_overlap_decoder), original_audio_length=orig_len,
            pitch_fidelity_boost=float(pitch_fidelity_boost), onset_guidance_boost=float(onset_guidance_boost),
            pitch_snap_strength=float(pitch_snap_strength), noise_level=float(noise_level),
            audio_file=audio_file, progress=progress
        )
        # Return mixed output, individual voices, and info
        if isinstance(result, dict):
            info_text = f"Generated {result['voice_count']} voices and mixed them."
            # Return mixed, all individual voices, and info
            return result["mixed"], result["voices"], info_text
        else:
            # Fallback case - single output
            return result, [], "Monophonic mode active but only single voice detected."
    else:
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
            audio_file=audio_file,
            render_and_extract=render_and_extract
        )
        progress(1.0, desc="Done!")

        # Check if this was a multitrack MIDI case - use simple approach
        if midi_mode and is_midi_file(audio_file):
            is_multi, track_count, _ = is_multitrack_midi(audio_file)
            if is_multi:
                print(f"🎼 Multitrack MIDI detected - using simple approach")

                # Use the new simple approach
                generation_args = {
                    'MODEL': MODEL,
                    'group': group,
                    'subgroup': subgroup,
                    'steps': int(steps),
                    'seed': int(seed),
                    'adapter_scale': float(adapter_scale),
                    'cfg_weight': float(cfg_weight),
                    't0': float(t0),
                    'instrument_strength': float(instrument_strength),
                    'inst_boost': float(inst_boost),
                    'piano_roll_gain': float(piano_roll_gain),
                    'amp_gain': float(amp_gain),
                    'rframe_gain': float(rframe_gain),
                    'rbend_gain': float(rbend_gain),
                    'encodec_gain': float(encodec_gain),
                    'use_overlap_decoder': bool(use_overlap_decoder),
                    'pitch_fidelity_boost': float(pitch_fidelity_boost),
                    'onset_guidance_boost': float(onset_guidance_boost),
                    'pitch_snap_strength': float(pitch_snap_strength),
                    'noise_level': float(noise_level),
                    'win_slow': win_slow
                }

                return process_multitrack_midi_simple(audio_file, progress=progress, **generation_args)

        return out, [], "Single generation completed."

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
    import gradio as gr  # Lazy import
    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        gr.Markdown("### dø stem — ControlBranch Pipeline")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.File(
                    label="Upload Audio or MIDI File",
                    file_types=[".wav", ".mp3", ".flac", ".m4a", ".mid", ".midi"],
                    type="filepath"
                )
                random_btn = gr.Button("🎤 Random from Manifest", variant="secondary")
                random_group_btn = gr.Button("🎯 Random from Current Group", variant="secondary")

                # MIDI Processing Options (auto-detected)
                gr.Markdown("#### 🎼 Processing Mode")
                file_type_info = gr.Markdown("**Upload a file to see processing mode**")

                midi_mode = gr.Checkbox(
                    label="MIDI Processing Active",
                    value=False,
                    interactive=False,
                    info="Automatically enabled when MIDI file is detected"
                )
                render_and_extract = gr.Checkbox(
                    label="🎵 Render & Extract Full Conditioning",
                    value=True,  # Default to full conditioning
                    info="MIDI → audio → full conditioning + original MIDI piano roll. Uncheck for piano roll only."
                )

                # Tempo Control
                with gr.Row():
                    detected_tempo_display = gr.Textbox(
                        label="Detected Tempo",
                        value="120.0",
                        interactive=False,
                        scale=1,
                        info="Auto-detected from MIDI file"
                    )
                    tempo_override = gr.Number(
                        label="Tempo Override (BPM)",
                        value=120.0,
                        minimum=40.0,
                        maximum=200.0,
                        step=0.1,
                        scale=1,
                        info="Override detected tempo if needed"
                    )

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
                t0_slider = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="T0 (denoising range: 1.0=full, 0.8=partial)")

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
                monophonic_mode = gr.Checkbox(label="🎵 Monophonic Mode (separate voices)", value=False)
                generate_btn = gr.Button("🎹 Generate", variant="primary")
                midi_download_btn = gr.Button("🎼 Download Basic Pitch MIDI + Voices", variant="secondary")

        with gr.Row():
            audio_output = gr.Audio(label="🎵 Mixed/Main Output", type="filepath")

        # Individual voice players (max 6 voices as per our limit)
        gr.Markdown("### Individual Voices")
        voice_players = []
        for i in range(6):
            voice_audio = gr.Audio(label=f"🎼 Voice {i+1}", type="filepath", visible=False)
            voice_players.append(voice_audio)

        # Audio voices ZIP download
        with gr.Row():
            audio_voices_zip_file = gr.File(label="🎵 Download All Generated Voices (ZIP)", type="filepath")

        info_output = gr.Textbox(label="Generation Info", interactive=False, lines=2)

        # MIDI Download section
        gr.Markdown("### MIDI Files")
        with gr.Row():
            midi_info = gr.Textbox(label="MIDI Export Info", interactive=False, lines=3)
        with gr.Row():
            main_midi_file = gr.File(label="🎼 Original MIDI (Basic Pitch)", type="filepath")
            voices_zip_file = gr.File(label="🎵 Voice MIDI Files (ZIP)", type="filepath")
        with gr.Row():
            combined_midi_file = gr.File(label="🎹 Combined Voices MIDI", type="filepath")

        # Debug Audio section
        gr.Markdown("### 🔍 Debug Audio (FluidSynth Renders)")
        debug_audio_info = gr.Textbox(label="FluidSynth Debug Info", interactive=False, lines=2,
                                     value="Individual FluidSynth renders will appear here to help debug multitrack performance")
        debug_audio_files = []
        for i in range(6):  # Support up to 6 debug audio files like voice players
            debug_audio = gr.File(label=f"🎵 Track/Voice {i+1} FluidSynth Render", type="filepath", visible=False)
            debug_audio_files.append(debug_audio)

        # Function to handle generation and update all outputs
        def handle_generation_with_voices(*args):
            # Call run_generation with all arguments
            mixed_audio, voice_files, info_text = run_generation(*args)

            # Prepare outputs for all voice players (up to 6)
            voice_outputs = [None] * 6  # Start with all None
            voice_visibility = [False] * 6  # Start with all hidden
            audio_zip_path = None

            if voice_files and len(voice_files) > 1:
                info_text += f"\n\nGenerated {len(voice_files)} individual voices:"
                for i, voice_file in enumerate(voice_files):
                    if i < 6:  # Don't exceed our maximum
                        voice_outputs[i] = voice_file
                        voice_visibility[i] = True
                        info_text += f"\n• Voice {i+1}: {Path(voice_file).name}"

                # Create ZIP file with all voice audio files
                audio_zip_path = create_audio_voices_zip(voice_files)
                info_text += f"\n\n📦 All voices packaged in ZIP file"

            # Create return tuple: mixed_audio, info_text, then updates for all 6 voice players, then ZIP file
            result = [mixed_audio]
            for i in range(6):
                # Return gr.update() to modify existing components
                result.append(gr.update(value=voice_outputs[i], visible=voice_visibility[i]))
            result.append(audio_zip_path)  # Add ZIP file
            result.append(info_text)  # Add info text last

            return tuple(result)

        # events
        random_btn.click(fn=select_random_file, inputs=[], outputs=[audio_input])
        random_group_btn.click(fn=select_random_file_by_group, inputs=[group_dd], outputs=[audio_input])

        # dynamic subgroup options by selected group
        def _opts_for_group(g):
            return gr.Dropdown(choices=sorted(APPROVED_SUBGROUPS.get(g, [])),
                               value=(sorted(APPROVED_SUBGROUPS.get(g, [])) or [None])[0])
        group_dd.change(_opts_for_group, inputs=[group_dd], outputs=[subgroup_dd])

        # Auto-detect file type and update UI accordingly
        def _detect_and_update_file_type(file_path):
            if not file_path:
                return (
                    gr.update(value="**Upload a file to see processing mode**"),  # file_type_info
                    gr.update(value=False),  # midi_mode
                    gr.update(value=1.0),    # piano_roll_gain
                    gr.update(value=1.0),    # amp_gain
                    gr.update(value=1.0),    # rframe_gain
                    gr.update(value=1.0),    # rbend_gain
                    gr.update(value=1.0),    # encodec_gain
                    gr.update(value="120.0"),  # detected_tempo_display
                    gr.update(value=120.0)     # tempo_override
                )

            is_midi = is_midi_file(file_path)
            file_name = Path(file_path).name

            if is_midi:
                # Extract tempo for MIDI files
                detected_tempo = extract_midi_tempo(file_path)
                info_text = f"**🎼 MIDI File Detected**: `{file_name}`  \n✅ **MIDI processing automatically enabled**  \n🎵 **Detected Tempo**: {detected_tempo:.1f} BPM"
                # Auto-enable MIDI mode with smart gains
                return (
                    gr.update(value=info_text),   # file_type_info
                    gr.update(value=True),        # midi_mode (auto-enable)
                    gr.update(value=1.2),         # piano_roll_gain (emphasize)
                    gr.update(value=0.8),         # amp_gain
                    gr.update(value=0.8),         # rframe_gain
                    gr.update(value=0.8),         # rbend_gain
                    gr.update(value=0.5),         # encodec_gain (lower for MIDI)
                    gr.update(value=f"{detected_tempo:.1f}"),  # detected_tempo_display
                    gr.update(value=float(detected_tempo))     # tempo_override
                )
            else:
                info_text = f"**🎵 Audio File Detected**: `{file_name}`  \n✅ **Standard audio processing mode**"
                # Audio mode - standard gains
                return (
                    gr.update(value=info_text),   # file_type_info
                    gr.update(value=False),       # midi_mode (disable)
                    gr.update(value=1.0),         # piano_roll_gain
                    gr.update(value=1.0),         # amp_gain
                    gr.update(value=1.0),         # rframe_gain
                    gr.update(value=1.0),         # rbend_gain
                    gr.update(value=1.0),         # encodec_gain
                    gr.update(value="N/A"),       # detected_tempo_display
                    gr.update(value=120.0)        # tempo_override (default)
                )

        # Smart gain adjustments when render mode changes (for MIDI only)
        def _update_gains_for_render_mode(midi_enabled, render_enabled):
            if not midi_enabled:
                return tuple(gr.update() for _ in range(5))  # No changes for audio files

            if render_enabled:
                # MIDI render mode - full conditioning available
                return (
                    gr.update(value=1.2),  # piano_roll_gain
                    gr.update(value=0.8),  # amp_gain
                    gr.update(value=0.8),  # rframe_gain
                    gr.update(value=0.8),  # rbend_gain
                    gr.update(value=0.5)   # encodec_gain
                )
            else:
                # MIDI direct mode - only piano roll
                return (
                    gr.update(value=1.0),  # piano_roll_gain
                    gr.update(value=0.0),  # amp_gain = 0
                    gr.update(value=0.0),  # rframe_gain = 0
                    gr.update(value=0.0),  # rbend_gain = 0
                    gr.update(value=0.0)   # encodec_gain = 0
                )

        # Auto-detect file type when file is uploaded
        audio_input.change(
            fn=_detect_and_update_file_type,
            inputs=[audio_input],
            outputs=[file_type_info, midi_mode, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain, detected_tempo_display, tempo_override]
        )

        # Update gains when render mode changes (for MIDI files)
        render_and_extract.change(
            fn=_update_gains_for_render_mode,
            inputs=[midi_mode, render_and_extract],
            outputs=[piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain]
        )

        inputs = [audio_input, group_dd, subgroup_dd, seed_slider, steps_slider, adapter_slider, cfg_slider, t0_slider,
                  instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
                  use_overlap_decoder, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength, noise_level,
                  monophonic_mode, midi_mode, render_and_extract, tempo_override]

        # Create outputs list: mixed output, all 6 voice players, ZIP file, info
        outputs = [audio_output] + voice_players + [audio_voices_zip_file, info_output]

        generate_btn.click(fn=handle_generation_with_voices, inputs=inputs, outputs=outputs)

        # MIDI download button event
        midi_download_btn.click(
            fn=handle_midi_download,
            inputs=[audio_input, subgroup_dd],
            outputs=[main_midi_file, voices_zip_file, combined_midi_file, midi_info, debug_audio_info] + debug_audio_files
        )

    return iface

# ------------------------------------------------------------------------------
# FastAPI Endpoints
# ------------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Depends, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from celery import Celery
from typing import Optional
import uuid
import logging
import openai
import requests as http_requests

# Create FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Celery configuration
celery_app = Celery("ace_step_tasks", broker="pyamqp://guest:guest@localhost//", backend="redis://localhost:6379/0")
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # Only prefetch 1 task at a time for long-running tasks
    broker_heartbeat=0,  # Disable heartbeat to prevent timeouts on long tasks
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    broker_pool_limit=None,  # Unlimited connections
    task_soft_time_limit=1800,  # 30 minutes soft limit
    task_time_limit=2400,  # 40 minutes hard limit
    result_expires=3600,
    task_track_started=True,
    task_send_sent_event=True,
    # Broker transport options for RabbitMQ to prevent connection resets
    broker_transport_options={
        'confirm_publish': True,
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.5,
        'socket_keepalive': True,
        'socket_keepalive_options': {
            1: 1,   # TCP_KEEPIDLE
            2: 10,  # TCP_KEEPINTVL
            3: 6    # TCP_KEEPCNT
        }
    },
    # Result backend options
    result_backend_transport_options={
        'socket_keepalive': True,
        'socket_connect_timeout': 30
    },
    # Task result settings
    task_ignore_result=False,
    result_persistent=True
)

logging.basicConfig(level=logging.INFO)

# Celery worker initialization - load model when worker starts
from celery.signals import worker_process_init

@worker_process_init.connect
def init_worker(**kwargs):
    """Initialize model when Celery worker starts"""
    global MODEL, GROUP_NAMES, SUBGROUP_NAMES, MANIFEST_DATA, ACE_STEP_PIPELINE

    # Get checkpoint paths from environment variables (set in start-logs.sh)
    checkpoint = os.environ.get('ACE_CHECKPOINT', '/mnt/msdd/exps/logs_v2/checkpoints/NEWRUN/epoch=85-step=50000.ckpt')
    checkpoint_dir = os.environ.get('ACE_CHECKPOINT_DIR', '/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c')
    manifest = os.environ.get('ACE_MANIFEST', '/home/arlo/Data/final_training_manifest_final.json')

    print(f"🔧 Initializing ACE-Step model in Celery worker...")
    print(f"   Checkpoint: {checkpoint}")
    print(f"   Checkpoint Dir: {checkpoint_dir}")
    print(f"   Manifest: {manifest}")

    # Manifest not needed - removed random file selection feature
    MANIFEST_DATA = []
    print(f"ℹ️  Manifest loading disabled (random file selection feature removed)")

    # Load model
    MODEL = load_model_any_ckpt(checkpoint, checkpoint_dir, manifest)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL.to(dev).eval()

    GROUP_NAMES = list(APPROVED_GROUPS) if not isinstance(APPROVED_GROUPS, dict) else list(APPROVED_GROUPS.keys())
    SUBGROUP_NAMES = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})

    print(f"✅ Model loaded in Celery worker on {dev}")
    print(f"   Groups: {len(GROUP_NAMES)} | Subgroups: {len(SUBGROUP_NAMES)}")

    # Load ACE-Step pipeline for original ACE-Step endpoint
    print(f"\n🔧 Initializing original ACE-Step pipeline...")
    import time
    pipeline_start = time.time()
    try:
        from acestep.pipeline_ace_step import ACEStepPipeline
        ACE_STEP_PIPELINE = ACEStepPipeline(
            device_id=0,
            dtype="bfloat16"
        )
        # Pre-load the checkpoint to avoid loading it on first generation
        print(f"   Loading checkpoint weights...")
        checkpoint_start = time.time()
        ACE_STEP_PIPELINE.load_checkpoint()
        checkpoint_time = time.time() - checkpoint_start
        pipeline_time = time.time() - pipeline_start
        print(f"   ✓ Checkpoint loaded in {checkpoint_time:.2f}s")
        print(f"✅ Original ACE-Step pipeline fully loaded in {pipeline_time:.2f}s")
    except Exception as e:
        print(f"⚠️  Failed to load ACE-Step pipeline: {e}")
        print(f"   ACE-Step endpoint will use wrapper script instead")
        import traceback
        traceback.print_exc()
        ACE_STEP_PIPELINE = None

# Celery task for audio generation
@celery_app.task(bind=True, name="generate_do_task")
def generate_do_task(
    self,
    audio_file_path: Optional[str],
    description: str,
    duration: float,
    steps: int,
    seed: int,
    adapter_scale: float,
    cfg_weight: float,
    instrument_strength: float,
    noise_level: float,
    piano_roll_gain: float,
    amp_gain: float,
    rframe_gain: float,
    rbend_gain: float,
    encodec_gain: float,
    pitch_fidelity_boost: float,
    onset_guidance_boost: float,
    pitch_snap_strength: float,
    instrument_group: str = None,
    instrument_subgroup: str = None,
    monophonic_mode: bool = False,
    arrange_mode: bool = False,
    fatten_mode: bool = False,
    fatten_type: str = "fake",
    enable_voice_separation: bool = False,
    scene_durations: Optional[list] = None,
    automation_data: Optional[str] = None,
    tape_speed: float = 1.0,
    slowdown_method: str = "stretch",
    upsample_mode: bool = False,
    upsample_noise_level: float = 0.3,
    upsample_steps: int = 20,
    use_overlap_decoder: bool = True,
    inpaint_mode: bool = False,
    inpaint_start_time: Optional[float] = None,
    inpaint_end_time: Optional[float] = None,
    inpaint_voice_index: Optional[int] = None,
    fast_mode_variant: str = None,
    generation_key: str = 'C',
    tempo_override: int = 120,
    # Test-time enhancement parameters
    use_best_of_n: bool = False,
    n_candidates: int = 12,
    use_test_time_adaptation: bool = False,
    adaptation_steps: int = 10,
    adaptation_learning_rate: float = 1e-4,
    use_self_consistency: bool = False,
    consistency_samples: int = 3,
    consistency_noise_scale: float = 0.05,
    use_time_varying_noise: bool = False,
    onset_preservation: float = 0.7,
    use_multiresolution_mixing: bool = False,
    use_onset_weighted_encodec: bool = False,
    encodec_onset_boost: float = 2.0,
    midi_mode: bool = False,
    render_and_extract: bool = False,
    render_extract_mono: bool = False,
    enable_midi_export: bool = False,
    use_chords: bool = False,
    chord_beat_map: Optional[dict] = None,
    extract_formats: Optional[list] = None
):
    """Celery task for ACE-Step audio generation"""
    try:
        import shutil  # Import at top of function
        import subprocess  # Import explicitly to avoid scope issues
        global MODEL
        if MODEL is None:
            raise RuntimeError("Model not loaded. Please start the server first.")

        # Clear conditioning cache to prevent using old cached extractions
        print("🗑️ Clearing conditioning cache for fresh generation...")
        clear_conditioning_cache()

        # Also delete old extracted conditioning directories to prevent reuse
        extracted_cond_dir = Path("extracted_conditioning")
        if extracted_cond_dir.exists():
            try:
                shutil.rmtree(extracted_cond_dir)
                print(f"🗑️ Deleted old extracted conditioning directory")
            except Exception as e:
                print(f"⚠️ Failed to delete extracted conditioning directory: {e}")

        print(f"\n{'='*80}")
        print(f"🎵 Dø GENERATION TASK STARTED")
        print(f"{'='*80}")
        print(f"Description: {description}")
        print(f"Steps: {steps}, Seed: {seed}, CFG: {cfg_weight}")
        print(f"Monophonic mode: {monophonic_mode}, Voice separation: {enable_voice_separation}")
        print(f"🎼 ARRANGE MODE: {arrange_mode}")
        print(f"🎚️ FATTEN MODE: {fatten_mode}, Type: {fatten_type}")
        print(f"Duration parameter: {duration}s")
        print(f"Audio file: {audio_file_path}")
        print(f"🎞️ TAPE SPEED: {tape_speed}x, Method: {slowdown_method}")
        print(f"🔊 USE OVERLAP DECODER: {use_overlap_decoder}")
        print(f"⚡ FAST MODE: {fast_mode_variant} ({fast_mode_variant or 'disabled'})")
        print(f"📋 EXTRACTION FORMATS: {extract_formats or 'all (default)'}")
        if inpaint_mode:
            print(f"\n🎨 INPAINTING MODE:")
            print(f"   Inpaint region: {inpaint_start_time:.2f}s - {inpaint_end_time:.2f}s")
            print(f"   Duration: {inpaint_end_time - inpaint_start_time:.2f}s")
            if inpaint_voice_index is not None:
                print(f"   🎵 Inpaint voice {inpaint_voice_index} only (monophonic mode)")

        # CRITICAL: Log scene_durations received
        print(f"\n📥 SCENE DATA RECEIVED:")
        print(f"   scene_durations type: {type(scene_durations)}")
        print(f"   scene_durations value: {scene_durations}")
        if scene_durations:
            print(f"   scene_durations length: {len(scene_durations)}")
            print(f"   Will use scene-aware generation: {len(scene_durations) > 1}")
        else:
            print(f"   scene_durations is None or empty - using simple generation")
        print(f"{'='*80}\n")

        # Use provided instrument group/subgroup or fall back to defaults
        group = instrument_group if instrument_group else (GROUP_NAMES[0] if GROUP_NAMES else "piano")
        subgroup = instrument_subgroup if instrument_subgroup else (SUBGROUP_NAMES[0] if SUBGROUP_NAMES else "piano")

        print(f"Instrument: {group} / {subgroup}")

        # Parse automation data if provided
        global_automation = []
        if automation_data:
            try:
                automation_json = json.loads(automation_data)
                if isinstance(automation_json, dict) and 'points' in automation_json:
                    global_automation = [
                        (float(point['time']), float(point['volume']))
                        for point in automation_json['points']
                    ]
                    print(f"🎛 Loaded {len(global_automation)} global automation points")
            except Exception as e:
                print(f"⚠️ Automation parse error: {e}")

        # Check if audio file is provided for conditioning
        if audio_file_path is None:
            # INPAINT + MONOPHONIC MODE requires audio file!
            if inpaint_mode and monophonic_mode:
                raise ValueError(
                    "Monophonic inpaint mode requires an audio file (conditioningAudio). "
                    "Please provide the original track you want to inpaint."
                )

            # No audio file provided - generate MIDI conditioning using midigen feature
            print("🎹 No audio file provided - generating MIDI conditioning...")

            # Check if chord progression was provided
            if use_chords and chord_beat_map:
                print(f"\n{'='*80}")
                print(f"🎼 CHORD-BASED MIDI GENERATION")
                print(f"{'='*80}")
                print(f"Using user-specified chord progression:")
                print(f"  Chord beat map: {chord_beat_map}")

                # Import chord progression generator
                from chord_progression_generator import generate_chord_progression_midi

                # Calculate BPM from tempo_override or use default
                bpm = tempo_override if tempo_override else 120
                print(f"  BPM: {bpm}")

                # Generate MIDI from chord progression
                # For now, use default voicing and rhythm (can be extended later)
                # TODO: Add frontend controls for voicing/rhythm/style presets
                chord_midi_path = generate_chord_progression_midi(
                    chord_beat_map=chord_beat_map,
                    bpm=bpm,
                    voicing='drop2',  # Nice jazz voicing
                    rhythm='whole',   # Whole notes for sustained chords
                    style='block',    # Block chords
                    output_path=None  # Auto-generate temp path
                )

                print(f"  ✅ Generated chord MIDI: {chord_midi_path}")
                print(f"{'='*80}\n")

                # Use the generated MIDI as conditioning
                audio_file_path = chord_midi_path

            # Check if we have scene data for multi-scene MIDI generation (ac.py style)
            elif scene_durations and len(scene_durations) > 1:
                print(f"\n{'='*80}")
                print(f"🎬 SCENE-AWARE MIDI GENERATION")
                print(f"{'='*80}")
                print(f"📥 Received {len(scene_durations)} scene durations from frontend:")
                for i, dur in enumerate(scene_durations):
                    print(f"   Scene {i}: {dur:.3f}s")
                total_duration = sum(scene_durations)
                print(f"   TOTAL DURATION: {total_duration:.3f}s")

                # 1. Reconstruct scene_changes from durations
                scene_changes = [0.0]
                for dur in scene_durations:
                    scene_changes.append(scene_changes[-1] + dur)

                print(f"\n📍 Scene change timestamps:")
                for i, timestamp in enumerate(scene_changes):
                    print(f"   Scene {i} starts at: {timestamp:.3f}s")

                # 2. Compute optimal tempo for EACH scene
                tempos = compute_best_tempos(scene_changes)
                print(f"\n🎵 Computed tempos for each scene:")
                for i, tempo in enumerate(tempos):
                    print(f"   Scene {i}: {tempo} BPM")

                # 3. Log tempo analysis
                print("\n🎼 Tempo & Beat Alignment Analysis:")
                for i in range(len(scene_durations)):
                    bpm = tempos[i]
                    seconds_per_beat = 60 / bpm
                    scene_start = scene_changes[i]
                    scene_end = scene_changes[i + 1]
                    scene_duration = scene_end - scene_start

                    beats_in_scene = scene_duration / seconds_per_beat
                    total_beats_before_next_scene = scene_end / seconds_per_beat

                    print(f"▶ Scene {i}")
                    print(f"   Duration: {scene_duration:.3f}s  →  {beats_in_scene:.2f} beats at {bpm} BPM")
                    print(f"   Next scene lands on beat {total_beats_before_next_scene:.2f}")

                # 4. Generate different MIDI for each scene + apply automation
                print(f"\n{'='*80}")
                print(f"🎹 GENERATING MIDI FOR EACH SCENE")
                print(f"{'='*80}")
                scene_midi_paths = {}
                os.makedirs("/tmp/midi_processing", exist_ok=True)

                for scene_idx, scene_dur in enumerate(scene_durations):
                    if scene_dur <= 0:
                        print(f"\n⚠️ Scene {scene_idx}: Skipping (duration {scene_dur:.3f}s <= 0)")
                        continue

                    # Generate unique MIDI at this scene's tempo
                    scene_tempo = tempos[scene_idx]
                    print(f"\n🎼 Scene {scene_idx}:")
                    print(f"   Generating MIDI at {scene_tempo} BPM in key {generation_key}...")
                    # Skip WAV rendering in fast mode (not needed, saves time)
                    original_midi, _ = get_random_transposed_midi_wav(tempo=scene_tempo, skip_wav=bool(fast_mode_variant), target_key=generation_key)

                    # ✅ ENSURE ALL NOTES ARE ABOVE C3 - Single point of enforcement for scene changes
                    print(f"   🔍 Checking and correcting scene MIDI for C3 minimum...")
                    original_midi = ensure_midi_above_c2(original_midi)

                    # Get MIDI duration
                    import pretty_midi
                    midi_data = pretty_midi.PrettyMIDI(original_midi)
                    original_midi_duration = midi_data.get_end_time()
                    print(f"   Generated MIDI duration: {original_midi_duration:.3f}s")
                    print(f"   Target scene duration: {scene_dur:.3f}s")

                    # Calculate scene timing
                    scene_start = sum(scene_durations[:scene_idx])
                    scene_end = scene_start + scene_dur
                    print(f"   Scene window: {scene_start:.3f}s → {scene_end:.3f}s")

                    # Extract automation points for this scene
                    scene_automation = []
                    if global_automation:
                        # Get automation points within scene window
                        for t, v in global_automation:
                            if scene_start <= t <= scene_end:
                                # Normalize time to [0, 1] relative to scene
                                normalized_time = (t - scene_start) / scene_dur
                                scene_automation.append((normalized_time, v))

                        # Ensure we have start/end points
                        if not scene_automation or scene_automation[0][0] > 0:
                            scene_automation.insert(0, (0.0, 0.5))
                        if not scene_automation or scene_automation[-1][0] < 1.0:
                            scene_automation.append((1.0, scene_automation[-1][1] if scene_automation else 0.5))

                    # Apply automation to MIDI
                    processed_midi_path = f"/tmp/midi_processing/scene_{scene_idx}_automated.mid"
                    apply_automation_to_midi(
                        midi_path=original_midi,
                        scene_start=scene_start,
                        scene_duration=scene_dur,
                        track_automation=scene_automation,
                        output_path=processed_midi_path,
                        total_duration=sum(scene_durations),
                        scene_tempo=scene_tempo
                    )

                    # Verify processed MIDI duration
                    processed_midi_data = pretty_midi.PrettyMIDI(processed_midi_path)
                    processed_duration = processed_midi_data.get_end_time()
                    print(f"   ✅ Processed MIDI saved: {Path(processed_midi_path).name}")
                    print(f"   ✅ Processed MIDI duration: {processed_duration:.3f}s")

                    scene_midi_paths[scene_idx] = processed_midi_path

                # 5. Choose soundfont based on selected instrument
                soundfont_path = INSTRUMENT_SOUNDFONTS.get(subgroup, INSTRUMENT_SOUNDFONTS.get("default"))
                print(f"\n🎹 Selected soundfont for '{subgroup}': {soundfont_path}")

                # 6. Concatenate all scene MIDIs into one long file
                print(f"\n{'='*80}")
                print(f"🔗 CONCATENATING SCENE MIDIs")
                print(f"{'='*80}")
                print(f"Concatenating {len(scene_midi_paths)} scene MIDIs...")
                for idx in sorted(scene_midi_paths.keys()):
                    print(f"   Scene {idx}: {scene_durations[idx]:.3f}s → {scene_midi_paths[idx]}")

                concatenated_midi_path = f"/tmp/midi_processing/concatenated_{time.time()}.mid"
                concatenated_midi_path = concatenate_midi_scenes(
                    scene_midi_paths=scene_midi_paths,
                    scene_durations=scene_durations,
                    output_path=concatenated_midi_path,
                    soundfont_path=soundfont_path  # Pass soundfont for debug render
                )

                # Verify final concatenated MIDI
                final_midi_data = pretty_midi.PrettyMIDI(concatenated_midi_path)
                final_midi_duration = final_midi_data.get_end_time()
                print(f"\n✅ Concatenated MIDI saved: {concatenated_midi_path}")
                print(f"✅ Concatenated MIDI duration: {final_midi_duration:.3f}s")
                print(f"{'='*80}\n")

                # Handle monophonic mode: check if MIDI has multiple tracks
                if monophonic_mode:
                    print(f"\n🎵 MONOPHONIC MODE: Analyzing MIDI structure...")
                    print(f"   Group: {group}")
                    print(f"   Subgroup: {subgroup}")
                    print(f"   🎼 Arrange Mode: {arrange_mode}")
                    if arrange_mode:
                        print(f"   ✨ Arrange mode will AUTO-ASSIGN instruments based on pitch ranges!")
                        print(f"   ⚠️  Subgroup '{subgroup}' will be IGNORED for voice assignment")

                    # Create debug folder for voice renders
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    voice_debug_dir = Path("/home/arlo/Data/voice_debug") / timestamp
                    voice_debug_dir.mkdir(parents=True, exist_ok=True)
                    print(f"\n{'='*80}")
                    print(f"📁 DEBUG FILES SAVED TO: {voice_debug_dir}")
                    print(f"{'='*80}")

                    # Copy concatenated MIDI to debug folder
                    debug_concat_midi = voice_debug_dir / "concatenated_master.mid"
                    shutil.copy(concatenated_midi_path, str(debug_concat_midi))
                    print(f"   Copied concatenated MIDI: {debug_concat_midi.name}")

                    # Split MIDI into tracks/voices using pretty_midi
                    pm = pretty_midi.PrettyMIDI(concatenated_midi_path)
                    non_empty_instruments = [inst for inst in pm.instruments if len(inst.notes) > 0]

                    if len(non_empty_instruments) > 1:
                        # Multitrack MIDI: use each track as a voice
                        print(f"   Found {len(non_empty_instruments)} tracks - using each as a voice")
                        voice_midi_paths = []
                        voice_audio_paths = []

                        # Extract tempo track from concatenated MIDI using mido
                        import mido
                        concat_mido = mido.MidiFile(concatenated_midi_path)

                        # Build absolute-time tempo map
                        tempo_events = []  # (absolute_ticks, tempo_value)
                        for track in concat_mido.tracks:
                            current_tick = 0
                            for msg in track:
                                current_tick += msg.time
                                if msg.type == 'set_tempo':
                                    tempo_events.append((current_tick, msg.tempo))

                        # Sort by time and remove duplicates
                        tempo_events = sorted(set(tempo_events), key=lambda x: x[0])
                        print(f"   Extracted {len(tempo_events)} tempo changes from concatenated MIDI")
                        if tempo_events:
                            for i, (tick, tempo) in enumerate(tempo_events[:5]):
                                bpm = mido.tempo2bpm(tempo)
                                print(f"     Tempo {i+1}: {bpm:.1f} BPM at tick {tick}")

                        # Find which concat track corresponds to each pretty_midi instrument
                        # Extract tracks directly from concatenated MIDI to preserve tempo/timing
                        concat_non_tempo_tracks = []
                        for track in concat_mido.tracks:
                            # Check if track has note messages
                            has_notes = any(msg.type in ['note_on', 'note_off'] for msg in track)
                            if has_notes:
                                concat_non_tempo_tracks.append(track)

                        print(f"   Found {len(concat_non_tempo_tracks)} note tracks in concatenated MIDI")

                        for i in range(min(len(non_empty_instruments), len(concat_non_tempo_tracks))):
                            inst = non_empty_instruments[i]
                            source_track = concat_non_tempo_tracks[i]

                            # Apply group-level note limits using pretty_midi (easier for note manipulation)
                            if monophonic_mode and inst.notes:
                                transposed_count = transpose_notes_above_group_minimum(inst.notes, group)
                                if transposed_count > 0:
                                    print(f"   ⬆️  Voice {i+1}: Transposed {transposed_count} note(s) to meet {group} group minimum")

                                    # If we transposed, write with pretty_midi (using correct resolution) then reload with mido
                                    temp_pm = pretty_midi.PrettyMIDI(resolution=concat_mido.ticks_per_beat)
                                    temp_pm.instruments.append(inst)
                                    temp_path = voice_debug_dir / f"voice_{i+1}_transposed.mid"
                                    temp_pm.write(str(temp_path))

                                    # Load transposed MIDI
                                    transposed_mid = mido.MidiFile(str(temp_path))
                                    # Get the note track (skip tempo track if present)
                                    note_track = None
                                    for track in transposed_mid.tracks:
                                        if any(msg.type in ['note_on', 'note_off'] for msg in track):
                                            note_track = track
                                            break

                                    if note_track:
                                        source_track = note_track
                                    temp_path.unlink()

                            # Create voice MIDI with tempo track + this voice's notes
                            voice_mido = mido.MidiFile(ticks_per_beat=concat_mido.ticks_per_beat)
                            voice_track = mido.MidiTrack()
                            voice_mido.tracks.append(voice_track)

                            # Merge tempo events and note events in chronological order
                            all_events = []

                            # Add tempo events
                            for abs_tick, tempo in tempo_events:
                                all_events.append((abs_tick, mido.MetaMessage('set_tempo', tempo=tempo, time=0)))

                            # Add note events from source track
                            current_tick = 0
                            for msg in source_track:
                                current_tick += msg.time
                                if msg.type != 'set_tempo':  # Skip tempo messages
                                    all_events.append((current_tick, msg.copy(time=0)))

                            # Sort by time and convert to relative
                            all_events.sort(key=lambda x: x[0])
                            prev_tick = 0
                            for abs_tick, msg in all_events:
                                delta_tick = abs_tick - prev_tick
                                msg.time = delta_tick
                                voice_track.append(msg)
                                prev_tick = abs_tick

                            # Save voice MIDI
                            voice_midi_path = voice_debug_dir / f"voice_{i+1}_input.mid"
                            voice_mido.save(str(voice_midi_path))
                            voice_midi_paths.append(str(voice_midi_path))
                            print(f"   💾 Voice {i+1}: {len(inst.notes)} notes + {len(tempo_events)} tempo changes → {voice_midi_path}")
                            print(f"      Resolution: {voice_mido.ticks_per_beat} ticks/beat (concat: {concat_mido.ticks_per_beat})")

                        # Track used instruments for arrange mode variety
                        used_instruments = []
                        voice_subgroups = []  # Track actual instrument used for each voice (for logging)
                        voice_soundfonts = []  # Track soundfont used for each voice (for fatten mode)

                        # Render each voice with selected soundfont (skip in fast mode unless noise_level < 1.0 or encodec variant)
                        print(f"\n{'='*80}")
                        if fast_mode_variant and noise_level >= 1.0 and fast_mode_variant != "encodec":
                            print(f"⚡ FAST MODE: SKIPPING FLUIDSYNTH RENDERING - {len(voice_midi_paths)} voices")
                            print(f"{'='*80}")
                            print(f"   Will convert MIDI directly to piano roll")
                            voice_audio_paths = []  # Empty - we won't render audio in fast mode
                            # Populate voice_subgroups for tracking (even though we're not rendering)
                            voice_subgroups = [subgroup] * len(voice_midi_paths)
                            voice_soundfonts = [soundfont_path] * len(voice_midi_paths)
                        elif fast_mode_variant == "encodec":
                            print(f"⚡ FAST MODE (ENCODEC): RENDERING FOR ENCODEC EXTRACTION - {len(voice_midi_paths)} voices")
                            print(f"{'='*80}")
                            print(f"   Will render audio with FluidSynth and extract only encodec tokens")
                        elif fast_mode_variant and noise_level < 1.0:
                            print(f"⚡ FAST MODE + NOISE {noise_level}: RENDERING FOR GT LATENT - {len(voice_midi_paths)} voices")
                            print(f"{'='*80}")
                            print(f"   Will render audio for GT latent extraction, but skip full conditioning extraction")
                        else:
                            print(f"🎼 VOICE RENDERING - {len(voice_midi_paths)} voices")
                            print(f"{'='*80}")
                            if arrange_mode:
                                print(f"✨ ARRANGE MODE ENABLED")
                                print(f"   Instruments will be AUTO-ASSIGNED based on pitch ranges")
                                print(f"   Available: {', '.join(INSTRUMENT_RANGES.get(group.lower(), {}).keys())}")
                            else:
                                print(f"📌 SINGLE INSTRUMENT MODE")
                                print(f"   All voices will use: {subgroup}")
                                soundfont_name = Path(INSTRUMENT_SOUNDFONTS.get(subgroup, soundfont_path)).name
                                print(f"   Soundfont: {soundfont_name}")

                        # Render with FluidSynth if:
                        # 1. Not in fast mode, OR
                        # 2. In fast mode but noise_level < 1.0 (need GT latent), OR
                        # 3. Fast mode encodec variant (needs audio for encodec extraction)
                        should_render = (not fast_mode_variant or
                                       noise_level < 1.0 or
                                       fast_mode_variant == "encodec")

                        if should_render:
                            voice_audio_paths = []  # Initialize here for normal rendering
                            for i, voice_midi in enumerate(voice_midi_paths):
                                # Arrange mode: auto-assign instrument subgroup based on MIDI pitch range
                                if arrange_mode:
                                    # Analyze MIDI pitch range
                                    pm_voice = pretty_midi.PrettyMIDI(voice_midi)
                                    all_pitches = []
                                    for inst in pm_voice.instruments:
                                        all_pitches.extend([note.pitch for note in inst.notes])

                                    if all_pitches:
                                        min_pitch = min(all_pitches)
                                        max_pitch = max(all_pitches)
                                        mean_pitch = sum(all_pitches) / len(all_pitches)
                                        voice_range = (min_pitch, max_pitch, int(mean_pitch))

                                        print(f"\n   🎼 Arrange Mode - Voice {i+1} Analysis:")
                                        print(f"      📊 Note count: {len(all_pitches)}")
                                        print(f"      📏 Range: MIDI {min_pitch}-{max_pitch} (span: {max_pitch - min_pitch} semitones)")
                                        print(f"      📈 Mean pitch: {int(mean_pitch)}")

                                        # Assign instrument for this voice
                                        voice_subgroup = assign_instrument_for_voice(group, voice_range, used_instruments)
                                        used_instruments.append(voice_subgroup)
                                        voice_soundfont = INSTRUMENT_SOUNDFONTS.get(voice_subgroup, INSTRUMENT_SOUNDFONTS.get("default"))
                                        print(f"      🎹 Soundfont: {Path(voice_soundfont).name}")
                                    else:
                                        voice_subgroup = subgroup
                                        voice_soundfont = soundfont_path
                                        print(f"   ⚠️ Voice {i+1}: {subgroup} (no notes - using default)")
                            else:
                                    voice_subgroup = subgroup
                                    voice_soundfont = soundfont_path

                            # Track the subgroup and soundfont used for this voice
                            voice_subgroups.append(voice_subgroup)
                            voice_soundfonts.append(voice_soundfont)

                            voice_audio_raw = voice_debug_dir / f"voice_{i+1}_input_render_raw.wav"
                            if fast_mode_variant == "encodec":
                                print(f"   Rendering voice {i+1}/{len(voice_midi_paths)} with {voice_subgroup} (encodec variant needs audio)...")
                            else:
                                print(f"   Rendering voice {i+1}/{len(voice_midi_paths)} with {voice_subgroup}...")
                            subprocess.run([
                                    "fluidsynth", "-ni", "-g", "0.625", "-r", "44100", "-F", str(voice_audio_raw),
                                    voice_soundfont, voice_midi
                            ], check=True, capture_output=True)
                            print(f"   🎵 Voice {i+1} rendered: {voice_audio_raw}")

                            # Apply automation gain envelope
                            voice_audio_path = voice_debug_dir / f"voice_{i+1}_input_render.wav"
                            if global_automation and len(global_automation) > 0:
                                    apply_automation_gain_envelope(
                                        audio_path=str(voice_audio_raw),
                                        automation_points=global_automation,
                                        output_path=str(voice_audio_path)
                                    )
                            else:
                                    # No automation - just copy
                                    shutil.copy(str(voice_audio_raw), str(voice_audio_path))

                            voice_audio_paths.append(str(voice_audio_path))
                            print(f"   ✅ Added to voice_audio_paths (now has {len(voice_audio_paths)} elements)")

                        print(f"\n📊 Rendering complete:")
                        print(f"   voice_midi_paths: {len(voice_midi_paths)} elements")
                        print(f"   voice_audio_paths: {len(voice_audio_paths)} elements")

                        # Log arrange mode summary
                        if arrange_mode and used_instruments:
                            print(f"\n📋 Arrange Mode Summary:")
                            print(f"   Total voices: {len(used_instruments)}")
                            print(f"   Instruments used: {', '.join(set(used_instruments))}")
                            print(f"   Voice assignments: {', '.join(used_instruments)}")

                        # Fatten mode: create octave-up versions
                        if fatten_mode and fatten_type == "real":
                            print(f"\n🎚️ FATTEN MODE (REAL): Creating octave-up voices...")
                            original_voice_count = len(voice_midi_paths)
                            octave_midi_paths = []
                            octave_audio_paths = []

                            for i, voice_midi in enumerate(voice_midi_paths):
                                # Load MIDI and transpose up 12 semitones
                                pm_voice = pretty_midi.PrettyMIDI(voice_midi)
                                for inst in pm_voice.instruments:
                                    for note in inst.notes:
                                        note.pitch += 12  # Transpose up one octave

                                # Save transposed MIDI
                                octave_midi_path = voice_debug_dir / f"voice_{i+1}_octave.mid"
                                pm_voice.write(str(octave_midi_path))
                                octave_midi_paths.append(str(octave_midi_path))
                                print(f"   💾 Created octave-up voice {i+1}: {octave_midi_path.name}")

                                # Use the same soundfont that was used for this voice (respects arrange mode)
                                octave_soundfont = voice_soundfonts[i] if i < len(voice_soundfonts) else soundfont_path
                                octave_subgroup = voice_subgroups[i] if i < len(voice_subgroups) else subgroup

                                # Render octave-up version
                                octave_audio_raw = voice_debug_dir / f"voice_{i+1}_octave_render_raw.wav"
                                print(f"   Rendering octave-up voice {i+1} with {octave_subgroup}...")
                                subprocess.run([
                                    "fluidsynth", "-ni", "-g", "0.625", "-r", "44100", "-F", str(octave_audio_raw),
                                    octave_soundfont, str(octave_midi_path)
                                ], check=True, capture_output=True)
                                print(f"   🎵 Octave-up voice {i+1} rendered: {octave_audio_raw.name}")

                                # Apply automation gain envelope
                                octave_audio_path = voice_debug_dir / f"voice_{i+1}_octave_render.wav"
                                if global_automation and len(global_automation) > 0:
                                    apply_automation_gain_envelope(
                                        audio_path=str(octave_audio_raw),
                                        automation_points=global_automation,
                                        output_path=str(octave_audio_path)
                                    )
                                else:
                                    # No automation - just copy
                                    shutil.copy(str(octave_audio_raw), str(octave_audio_path))

                                octave_audio_paths.append(str(octave_audio_path))

                            # Add octave voices to the lists
                            voice_midi_paths.extend(octave_midi_paths)
                            voice_audio_paths.extend(octave_audio_paths)
                            # Extend tracking lists with octave voice info (same instruments as originals)
                            voice_subgroups.extend(voice_subgroups[:original_voice_count])
                            voice_soundfonts.extend(voice_soundfonts[:original_voice_count])
                            print(f"\n✅ Doubled to {len(voice_audio_paths)} total voices ({original_voice_count} original + {original_voice_count} octave-up)")

                        # Set audio_file_path to None to trigger voice-by-voice generation
                        audio_file_path = None
                        print(f"\n{'='*80}")
                        print(f"📂 DEBUG FILES SUMMARY:")
                        print(f"   Master MIDI: {debug_concat_midi}")
                        if fast_mode_variant and noise_level >= 1.0 and fast_mode_variant != "encodec":
                            # Fast mode (non-encodec): only MIDI files, no audio
                            for i, midi_path in enumerate(voice_midi_paths, 1):
                                print(f"   Voice {i}: {Path(midi_path).name} (no render - fast mode)")
                        else:
                            # Normal mode, encodec variant, or fast mode with noise < 1.0: show audio files
                            for i, (midi_path, audio_path) in enumerate(zip(voice_midi_paths, voice_audio_paths), 1):
                                print(f"   Voice {i}: {Path(midi_path).name} → {Path(audio_path).name}")
                        print(f"{'='*80}\n")

                    else:
                        # Single track MIDI: split MIDI into voices FIRST, then render each voice
                        print(f"   Single track detected - splitting MIDI into voices...")

                        # Load the concatenated MIDI
                        pm_concat = pretty_midi.PrettyMIDI(concatenated_midi_path)

                        # Split into voices using voice separation algorithm
                        # Get the single instrument
                        if len(pm_concat.instruments) > 0 and len(pm_concat.instruments[0].notes) > 0:
                            instrument = pm_concat.instruments[0]

                            # Sort notes by start time
                            sorted_notes = sorted(instrument.notes, key=lambda n: n.start)

                            # Simple voice separation: group overlapping notes into different voices
                            voices = []
                            for note in sorted_notes:
                                # Find a voice where this note doesn't overlap
                                assigned = False
                                for voice_notes in voices:
                                    # Check if this note overlaps with any note in this voice
                                    if all(note.start >= existing.end for existing in voice_notes):
                                        voice_notes.append(note)
                                        assigned = True
                                        break

                                if not assigned:
                                    # Create new voice
                                    voices.append([note])

                            print(f"   Split into {len(voices)} voices")

                            # Print arrange mode status
                            if arrange_mode:
                                print(f"   ✨ ARRANGE MODE ENABLED - will auto-assign instruments")
                                print(f"   Available: {', '.join(INSTRUMENT_RANGES.get(group.lower(), {}).keys())}")
                            else:
                                print(f"   📌 SINGLE INSTRUMENT MODE - all voices use {subgroup}")

                            # Create voice MIDI files and render each
                            voice_midi_paths = []
                            voice_audio_paths = []

                            # Track used instruments for arrange mode variety
                            used_instruments = []
                            voice_soundfonts = []  # Track soundfont used for each voice
                            voice_subgroups = []  # Track actual instrument used for each voice (for logging)

                            # Get tempo changes from original MIDI with absolute timing
                            import mido
                            concat_mido = mido.MidiFile(concatenated_midi_path)

                            # Build absolute-time tempo map
                            tempo_events = []  # (absolute_ticks, tempo_value)
                            for track in concat_mido.tracks:
                                current_tick = 0
                                for msg in track:
                                    current_tick += msg.time
                                    if msg.type == 'set_tempo':
                                        tempo_events.append((current_tick, msg.tempo))

                            # Sort by time and remove duplicates
                            tempo_events = sorted(set(tempo_events), key=lambda x: x[0])
                            print(f"   Extracted {len(tempo_events)} tempo changes")

                            for voice_idx, voice_notes in enumerate(voices):
                                # Apply group-level note limits: transpose notes up if below minimum
                                if monophonic_mode and voice_notes:
                                    transposed_count = transpose_notes_above_group_minimum(voice_notes, group)
                                    if transposed_count > 0:
                                        print(f"   ⬆️  Voice {voice_idx+1}: Transposed {transposed_count} note(s) to meet {group} group minimum")

                                # Create MIDI for this voice with SAME resolution as concatenated MIDI
                                # AND copy the tempo changes BEFORE adding notes
                                voice_pm = pretty_midi.PrettyMIDI(resolution=concat_mido.ticks_per_beat)

                                # Copy tempo changes from concatenated MIDI so note times are converted correctly
                                if hasattr(pm_concat, '_tick_scales') and len(pm_concat._tick_scales) > 0:
                                    voice_pm._tick_scales = pm_concat._tick_scales.copy()
                                    voice_pm._update_tick_to_time(voice_pm.resolution)
                                    print(f"   Copied {len(pm_concat._tick_scales)} tempo changes to voice {voice_idx + 1}")

                                voice_inst = pretty_midi.Instrument(program=0)
                                voice_inst.notes = voice_notes

                                # Final defensive check: ensure no notes below C2 before saving
                                C2_MIDI_NOTE = 36
                                for note in voice_inst.notes:
                                    while note.pitch < C2_MIDI_NOTE and note.pitch + 12 <= 127:
                                        note.pitch += 12
                                        print(f"      ⚠️  FINAL CHECK: Transposed note up to {note.pitch} (was below C2)")

                                voice_pm.instruments.append(voice_inst)

                                # Save voice MIDI (notes will be converted to ticks using correct tempo map)
                                voice_midi_path = voice_debug_dir / f"voice_{voice_idx + 1}.mid"
                                voice_pm.write(str(voice_midi_path))
                                voice_midi_paths.append(str(voice_midi_path))
                                print(f"   💾 Saved voice {voice_idx + 1} MIDI (resolution={concat_mido.ticks_per_beat}): {voice_midi_path}")

                                # Arrange mode: auto-assign instrument subgroup based on MIDI pitch range
                                if arrange_mode and voice_notes:
                                    # Analyze pitch range from voice notes
                                    all_pitches = [note.pitch for note in voice_notes]
                                    min_pitch = min(all_pitches)
                                    max_pitch = max(all_pitches)
                                    mean_pitch = sum(all_pitches) / len(all_pitches)
                                    voice_range = (min_pitch, max_pitch, int(mean_pitch))

                                    print(f"\n   🎼 Arrange Mode - Voice {voice_idx+1} Analysis:")
                                    print(f"      📊 Note count: {len(all_pitches)}")
                                    print(f"      📏 Range: MIDI {min_pitch}-{max_pitch} (span: {max_pitch - min_pitch} semitones)")
                                    print(f"      📈 Mean pitch: {int(mean_pitch)}")

                                    # Assign instrument for this voice
                                    voice_subgroup = assign_instrument_for_voice(group, voice_range, used_instruments)
                                    used_instruments.append(voice_subgroup)
                                    voice_soundfont = INSTRUMENT_SOUNDFONTS.get(voice_subgroup, INSTRUMENT_SOUNDFONTS.get("default"))
                                    print(f"      🎹 Soundfont: {Path(voice_soundfont).name}")
                                else:
                                    voice_subgroup = subgroup
                                    voice_soundfont = soundfont_path

                                # Track which soundfont is used for this voice (for fatten mode)
                                voice_soundfonts.append(voice_soundfont)
                                voice_subgroups.append(voice_subgroup)

                                # Render voice with FluidSynth (skip in fast mode unless noise < 1.0 or encodec variant)
                                if not fast_mode_variant or noise_level < 1.0 or fast_mode_variant == "encodec":
                                    voice_audio_raw = voice_debug_dir / f"voice_{voice_idx + 1}_rendered_raw.wav"
                                    if fast_mode_variant == "encodec":
                                        print(f"   Rendering voice {voice_idx + 1} with {voice_subgroup} (encodec variant needs audio)...")
                                    else:
                                        print(f"   Rendering voice {voice_idx + 1} with {voice_subgroup}...")
                                    subprocess.run([
                                        "fluidsynth", "-ni", "-g", "0.625", "-r", "44100", "-F", str(voice_audio_raw),
                                        voice_soundfont, str(voice_midi_path)
                                    ], check=True, capture_output=True)
                                    print(f"   🎵 Rendered voice {voice_idx + 1}: {voice_audio_raw}")

                                    # Apply automation gain envelope to the render
                                    voice_audio_path = voice_debug_dir / f"voice_{voice_idx + 1}_rendered.wav"
                                    if global_automation and len(global_automation) > 0:
                                        apply_automation_gain_envelope(
                                            audio_path=str(voice_audio_raw),
                                            automation_points=global_automation,
                                            output_path=str(voice_audio_path)
                                        )
                                    else:
                                        # No automation - just copy
                                        shutil.copy(str(voice_audio_raw), str(voice_audio_path))

                                    voice_audio_paths.append(str(voice_audio_path))
                                    print(f"   ✅ Added to voice_audio_paths (now has {len(voice_audio_paths)} elements)")
                                else:
                                    # Fast mode with noise >= 1.0: Skip rendering
                                    print(f"   ⚡ Fast mode (noise {noise_level}): Skipping render for voice {voice_idx + 1}")

                            print(f"\n📊 Voice preparation complete:")
                            print(f"   voice_midi_paths: {len(voice_midi_paths)} elements")
                            print(f"   voice_audio_paths: {len(voice_audio_paths)} elements")

                            if fast_mode_variant and noise_level >= 1.0 and fast_mode_variant != "encodec":
                                print(f"\n✅ Split {len(voices)} voices into MIDI (rendering skipped - fast mode)")
                            else:
                                print(f"\n✅ Split and rendered {len(voices)} voices")

                            # Log arrange mode summary
                            if arrange_mode and used_instruments:
                                print(f"\n📋 Arrange Mode Summary:")
                                print(f"   Total voices: {len(used_instruments)}")
                                print(f"   Instruments used: {', '.join(set(used_instruments))}")
                                print(f"   Voice assignments: {', '.join(used_instruments)}")

                            # Fatten mode: create octave-up versions
                            if fatten_mode and fatten_type == "real":
                                print(f"\n🎚️ FATTEN MODE (REAL): Creating octave-up voices...")
                                original_voice_count = len(voice_midi_paths)
                                octave_midi_paths = []
                                octave_audio_paths = []

                                for voice_idx, voice_midi in enumerate(voice_midi_paths):
                                    # Load MIDI and transpose up 12 semitones
                                    pm_voice = pretty_midi.PrettyMIDI(voice_midi)
                                    for inst in pm_voice.instruments:
                                        for note in inst.notes:
                                            note.pitch += 12  # Transpose up one octave

                                    # Save transposed MIDI
                                    octave_midi_path = voice_debug_dir / f"voice_{voice_idx+1}_octave.mid"
                                    pm_voice.write(str(octave_midi_path))
                                    octave_midi_paths.append(str(octave_midi_path))
                                    print(f"   💾 Created octave-up voice {voice_idx+1}: {octave_midi_path.name}")

                                    # Use same soundfont as original voice (respects arrange mode)
                                    octave_soundfont = voice_soundfonts[voice_idx] if voice_idx < len(voice_soundfonts) else soundfont_path
                                    octave_subgroup = voice_subgroups[voice_idx] if voice_idx < len(voice_subgroups) else subgroup

                                    # Render octave-up version (skip in fast mode unless noise < 1.0 or encodec variant)
                                    if not fast_mode_variant or noise_level < 1.0 or fast_mode_variant == "encodec":
                                        octave_audio_raw = voice_debug_dir / f"voice_{voice_idx+1}_octave_rendered_raw.wav"
                                        print(f"   Rendering octave-up voice {voice_idx+1} with {octave_subgroup}...")
                                        subprocess.run([
                                            "fluidsynth", "-ni", "-g", "0.625", "-r", "44100", "-F", str(octave_audio_raw),
                                            octave_soundfont, str(octave_midi_path)
                                        ], check=True, capture_output=True)
                                        print(f"   🎵 Octave-up voice {voice_idx+1} rendered: {octave_audio_raw.name}")

                                        # Apply automation gain envelope to the octave render
                                        octave_audio_path = voice_debug_dir / f"voice_{voice_idx+1}_octave_rendered.wav"
                                        if global_automation and len(global_automation) > 0:
                                            apply_automation_gain_envelope(
                                                audio_path=str(octave_audio_raw),
                                                automation_points=global_automation,
                                                output_path=str(octave_audio_path)
                                            )
                                        else:
                                            # No automation - just copy
                                            shutil.copy(str(octave_audio_raw), str(octave_audio_path))

                                        octave_audio_paths.append(str(octave_audio_path))
                                    else:
                                        # Fast mode with noise >= 1.0: Skip rendering
                                        print(f"   ⚡ Fast mode (noise {noise_level}): Skipping octave render for voice {voice_idx+1}")

                                # Add octave voices to the lists
                                voice_midi_paths.extend(octave_midi_paths)
                                voice_audio_paths.extend(octave_audio_paths)
                                # Extend tracking lists with octave voice info (same instruments as originals)
                                voice_subgroups.extend(voice_subgroups[:original_voice_count])
                                voice_soundfonts.extend(voice_soundfonts[:original_voice_count])
                                if fast_mode_variant and noise_level >= 1.0:
                                    print(f"\n✅ Doubled to {len(voice_midi_paths)} total MIDI voices ({original_voice_count} original + {original_voice_count} octave-up)")
                                else:
                                    print(f"\n✅ Doubled to {len(voice_audio_paths)} total voices ({original_voice_count} original + {original_voice_count} octave-up)")

                            print(f"\n{'='*80}")
                            print(f"📂 DEBUG FILES SUMMARY:")
                            print(f"   Master MIDI: {debug_concat_midi}")
                            if fast_mode_variant and noise_level >= 1.0:
                                # Fast mode: only MIDI files, no audio
                                for i, midi_path in enumerate(voice_midi_paths, 1):
                                    print(f"   Voice {i}: {Path(midi_path).name} (no render - fast mode)")
                            else:
                                # Normal mode or fast mode with noise < 1.0: show audio files
                                for i, (midi_path, audio_path) in enumerate(zip(voice_midi_paths, voice_audio_paths), 1):
                                    print(f"   Voice {i}: {Path(midi_path).name} → {Path(audio_path).name}")
                            print(f"{'='*80}\n")
                            # Set audio_file_path to None to trigger voice-by-voice generation
                            audio_file_path = None
                        else:
                            print(f"   ERROR: No notes found in concatenated MIDI")
                            # Fallback to rendering full MIDI
                            audio_file_raw = concatenated_midi_path.replace('.mid', '_raw.wav')
                            subprocess.run([
                                "fluidsynth", "-ni", "-g", "0.625", "-r", "44100", "-F", audio_file_raw,
                                soundfont_path, concatenated_midi_path
                            ], check=True, capture_output=True)

                            # Apply automation gain envelope
                            audio_file_path = concatenated_midi_path.replace('.mid', '.wav')
                            if global_automation and len(global_automation) > 0:
                                apply_automation_gain_envelope(
                                    audio_path=audio_file_raw,
                                    automation_points=global_automation,
                                    output_path=audio_file_path
                                )
                            else:
                                # No automation - just copy
                                shutil.copy(audio_file_raw, audio_file_path)

                else:
                    # Non-monophonic mode: render concatenated MIDI (unless fast mode + noise >= 1.0)
                    if fast_mode_variant and noise_level >= 1.0:
                        print(f"\n⚡ FAST MODE (noise {noise_level}): SKIPPING FLUIDSYNTH RENDERING")
                        print(f"   Will use MIDI directly with automation as amp conditioning")
                        # Don't set audio_file_path - will use MIDI directly
                        audio_file_path = concatenated_midi_path  # Use MIDI path for extraction
                    else:
                        print(f"\n🎼 Rendering concatenated MIDI with {subgroup} soundfont...")
                        audio_file_raw = concatenated_midi_path.replace('.mid', '_raw.wav')
                        subprocess.run([
                            "fluidsynth", "-ni", "-g", "0.625", "-r", "44100", "-F", audio_file_raw,
                            soundfont_path, concatenated_midi_path
                        ], check=True, capture_output=True)
                        print(f"✅ Rendered audio: {audio_file_raw}")

                        # Apply automation gain envelope
                        audio_file_path = concatenated_midi_path.replace('.mid', '.wav')
                        if global_automation and len(global_automation) > 0:
                            apply_automation_gain_envelope(
                                audio_path=audio_file_raw,
                                automation_points=global_automation,
                                output_path=audio_file_path
                            )
                        else:
                            # No automation - just copy
                            shutil.copy(audio_file_raw, audio_file_path)

            else:
                # Simple single-scene MIDI generation (original behavior)
                scene_changes = [0.0, float(duration)]

                # Use tempo override from frontend instead of computing
                selected_tempo = tempo_override

                print(f"🎼 Using tempo override: {selected_tempo} BPM for {duration}s duration")

                # Choose soundfont based on selected instrument
                soundfont_path = INSTRUMENT_SOUNDFONTS.get(subgroup, INSTRUMENT_SOUNDFONTS.get("default"))
                print(f"🎹 Selected soundfont for '{subgroup}': {soundfont_path}")

                # Generate random MIDI file at the computed tempo in target key
                # In fast mode with high noise (>=1.0), skip WAV rendering to use MIDI directly
                # In fast mode with lower noise, render WAV for extraction
                skip_wav = bool(fast_mode_variant) and not monophonic_mode and noise_level >= 1.0
                midi_path, wav_path = get_random_transposed_midi_wav(tempo=selected_tempo, skip_wav=skip_wav, target_key=generation_key)

                # ✅ ENSURE ALL NOTES ARE ABOVE C3 - Single point of enforcement before voice separation
                print(f"🔍 Checking and correcting MIDI for C3 minimum...")
                midi_path = ensure_midi_above_c2(midi_path)

                # Use the generated WAV as conditioning (UNLESS in monophonic mode or fast mode with high noise)
                if not monophonic_mode and not fast_mode_variant:
                    # Normal mode: use rendered WAV
                    audio_file_path = wav_path
                    print(f"✅ Generated MIDI conditioning: {audio_file_path}")
                elif not monophonic_mode and fast_mode_variant and noise_level >= 1.0:
                    # Fast mode with high noise: use MIDI directly (no rendering needed)
                    audio_file_path = midi_path
                    print(f"✅ Generated MIDI for fast mode (high noise): {audio_file_path}")
                elif not monophonic_mode and fast_mode_variant and noise_level < 1.0:
                    # Fast mode with lower noise: use rendered WAV for extraction
                    audio_file_path = wav_path
                    print(f"✅ Generated MIDI conditioning for fast mode (noise {noise_level}): {audio_file_path}")
                else:
                    print(f"✅ Generated MIDI: {midi_path} (monophonic mode - splitting into voices...)")

                    # Create debug folder for voice renders
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    voice_debug_dir = Path("/home/arlo/Data/voice_debug") / timestamp
                    voice_debug_dir.mkdir(parents=True, exist_ok=True)

                    # Split the generated MIDI into voices
                    import pretty_midi
                    pm = pretty_midi.PrettyMIDI(midi_path)

                    if len(pm.instruments) > 0 and len(pm.instruments[0].notes) > 0:
                        instrument = pm.instruments[0]

                        # Sort notes by start time
                        sorted_notes = sorted(instrument.notes, key=lambda n: n.start)

                        # Simple voice separation: group overlapping notes into different voices
                        voices = []
                        for note in sorted_notes:
                            # Find a voice where this note doesn't overlap
                            assigned = False
                            for voice_notes in voices:
                                # Check if this note overlaps with any note in this voice
                                if all(note.start >= existing.end for existing in voice_notes):
                                    voice_notes.append(note)
                                    assigned = True
                                    break

                            if not assigned:
                                # Create new voice
                                voices.append([note])

                        print(f"   Split into {len(voices)} voices")

                        # Create voice MIDI files
                        voice_midi_paths = []
                        voice_audio_paths = []
                        print(f"   🔍 Initialized voice_midi_paths and voice_audio_paths as empty lists")

                        # Get tempo from original MIDI
                        import mido
                        original_mido = mido.MidiFile(midi_path)

                        for voice_idx, voice_notes in enumerate(voices):
                            # Apply group-level note limits
                            if voice_notes:
                                transposed_count = transpose_notes_above_group_minimum(voice_notes, group)
                                if transposed_count > 0:
                                    print(f"   ⬆️  Voice {voice_idx+1}: Transposed {transposed_count} note(s) to meet {group} group minimum")

                            # Create MIDI for this voice
                            voice_pm = pretty_midi.PrettyMIDI(resolution=original_mido.ticks_per_beat)
                            voice_inst = pretty_midi.Instrument(program=0)
                            voice_inst.notes = voice_notes

                            # Final defensive check: ensure no notes below C3 before saving (should be redundant now)
                            C2_MIDI_NOTE = 48  # C3 minimum
                            for note in voice_inst.notes:
                                original_pitch = note.pitch
                                while note.pitch < C2_MIDI_NOTE and note.pitch + 12 <= 127:
                                    note.pitch += 12
                                    print(f"      ⚠️  FINAL CHECK: Transposed note up to {note.pitch} (was {original_pitch})")
                                # Critical warning if still below C3
                                if note.pitch < C2_MIDI_NOTE:
                                    print(f"      ⚠️⚠️⚠️ CRITICAL: Voice {voice_idx+1} note at {note.pitch} is STILL below C3!")

                            voice_pm.instruments.append(voice_inst)

                            # Assign instrument (arrange mode or single instrument)
                            if arrange_mode:
                                # Extract pitch range from notes
                                if voice_notes:
                                    pitches = [note.pitch for note in voice_notes]
                                    min_pitch = min(pitches)
                                    max_pitch = max(pitches)
                                    mean_pitch = sum(pitches) / len(pitches)
                                    voice_range = (min_pitch, max_pitch, mean_pitch)
                                else:
                                    voice_range = (60, 60, 60)  # Default to middle C
                                voice_subgroup = assign_instrument_for_voice(group, voice_range, [])
                                voice_soundfont = INSTRUMENT_SOUNDFONTS.get(voice_subgroup, soundfont_path)
                                print(f"   ✨ Voice {voice_idx+1}: Range MIDI {voice_range[0]}-{voice_range[1]} → {voice_subgroup}")
                            else:
                                voice_subgroup = subgroup
                                voice_soundfont = soundfont_path

                            # Save voice MIDI
                            voice_midi_path = voice_debug_dir / f"voice_{voice_idx + 1}.mid"
                            voice_pm.write(str(voice_midi_path))
                            voice_midi_paths.append(str(voice_midi_path))

                            # Debug: Log min/max pitch after saving
                            if voice_notes:
                                pitches = [n.pitch for n in voice_notes]
                                min_pitch = min(pitches)
                                max_pitch = max(pitches)
                                if min_pitch < 48:
                                    print(f"      ⚠️  WARNING: Voice {voice_idx+1} has notes below C3! Min pitch: {min_pitch}")
                                else:
                                    print(f"      ✅ Voice {voice_idx+1} pitch range: {min_pitch}-{max_pitch}")

                            # Render voice with FluidSynth
                            # Skip rendering only if: fast mode (non-encodec) with noise >= 1.0
                            should_render = (not fast_mode_variant or
                                           noise_level < 1.0 or
                                           fast_mode_variant == "encodec")

                            print(f"   🔍 Voice {voice_idx + 1} render decision:")
                            print(f"      fast_mode_variant={fast_mode_variant}, noise_level={noise_level}")
                            print(f"      should_render={should_render}")

                            if should_render:
                                voice_audio_path = voice_debug_dir / f"voice_{voice_idx + 1}_rendered.wav"
                                if fast_mode_variant == "encodec":
                                    print(f"   Rendering voice {voice_idx + 1} with {voice_subgroup} (encodec variant needs audio)...")
                                else:
                                    print(f"   Rendering voice {voice_idx + 1} with {voice_subgroup}...")
                                subprocess.run([
                                    "fluidsynth", "-ni", "-g", "0.625", "-r", "44100", "-F", str(voice_audio_path),
                                    voice_soundfont, str(voice_midi_path)
                                ], check=True, capture_output=True)
                                voice_audio_paths.append(str(voice_audio_path))
                                print(f"   ✅ Rendered and added to voice_audio_paths (now has {len(voice_audio_paths)} elements)")
                            else:
                                print(f"   ⚡ Fast mode (noise {noise_level}): Skipping render for voice {voice_idx + 1}")

                        print(f"✅ Split and prepared {len(voices)} voices")
                        print(f"   📊 Final counts: voice_midi_paths={len(voice_midi_paths)}, voice_audio_paths={len(voice_audio_paths)}")

                        # FATTEN MODE: Create octave-up versions of each voice
                        if fatten_mode:
                            print(f"\n🎚️ FATTEN MODE ({fatten_type.upper()}): Creating octave-up voices...")
                            original_voice_count = len(voice_midi_paths)

                            if fatten_type == "real":
                                # Real mode: transpose MIDI up 12 semitones and render
                                print(f"   Generating {original_voice_count} octave-up voices from transposed MIDIs...")
                                import pretty_midi

                                for voice_idx, voice_midi_path in enumerate(voice_midi_paths[:original_voice_count]):
                                    print(f"   🎼 Creating octave-up voice {voice_idx+1}/{original_voice_count}...")

                                    # Load MIDI and transpose up 12 semitones
                                    pm_voice = pretty_midi.PrettyMIDI(voice_midi_path)
                                    for inst in pm_voice.instruments:
                                        for note in inst.notes:
                                            # Transpose up one octave, but cap at MIDI 127
                                            new_pitch = min(note.pitch + 12, 127)
                                            note.pitch = new_pitch

                                        # Apply group minimum and C2 check
                                        transposed_count = transpose_notes_above_group_minimum(inst.notes, group)
                                        if transposed_count > 0:
                                            print(f"      ⬆️  Octave voice {voice_idx+1}: Transposed {transposed_count} note(s) to meet {group} minimum")

                                        # Final defensive check: ensure no notes below C3 (should be redundant)
                                        C2_MIDI_NOTE = 48  # C3 minimum
                                        for note in inst.notes:
                                            original_pitch = note.pitch
                                            while note.pitch < C2_MIDI_NOTE and note.pitch + 12 <= 127:
                                                note.pitch += 12
                                                print(f"      ⚠️  OCTAVE VOICE FINAL CHECK: Transposed note up to {note.pitch} (was {original_pitch})")
                                            # Critical warning if still below C3
                                            if note.pitch < C2_MIDI_NOTE:
                                                print(f"      ⚠️⚠️⚠️ CRITICAL: Octave voice {voice_idx+1} note at {note.pitch} is STILL below C3!")

                                    # Save transposed MIDI
                                    octave_midi_path = voice_debug_dir / f"voice_{original_voice_count+voice_idx+1}_octave.mid"
                                    pm_voice.write(str(octave_midi_path))

                                    voice_midi_paths.append(str(octave_midi_path))
                                    print(f"      💾 Saved: {octave_midi_path.name}")

                                    # Debug: Log min/max pitch after all transposition
                                    for inst in pm_voice.instruments:
                                        if inst.notes:
                                            pitches = [n.pitch for n in inst.notes]
                                            min_pitch = min(pitches)
                                            max_pitch = max(pitches)
                                            if min_pitch < 48:
                                                print(f"      ⚠️  WARNING: Octave voice {voice_idx+1} has notes below C3! Min pitch: {min_pitch}")
                                            else:
                                                print(f"      ✅ Octave voice {voice_idx+1} pitch range: {min_pitch}-{max_pitch}")

                                    # Render octave-up MIDI if needed (not in fast mode or noise < 1.0)
                                    should_render_octave = (not fast_mode_variant or noise_level < 1.0 or fast_mode_variant == "encodec")
                                    if should_render_octave:
                                        octave_audio_path = voice_debug_dir / f"voice_{original_voice_count+voice_idx+1}_octave_rendered.wav"
                                        print(f"      🎵 Rendering octave-up voice {voice_idx+1}...")
                                        subprocess.run([
                                            "fluidsynth", "-ni", "-g", "0.5", "-r", "44100", "-F", str(octave_audio_path),
                                            soundfont_path, str(octave_midi_path)
                                        ], check=True, capture_output=True)
                                        voice_audio_paths.append(str(octave_audio_path))
                                        print(f"      ✅ Rendered: {octave_audio_path.name}")
                                    else:
                                        print(f"      ⚡ Fast mode: Skipping render for octave voice {voice_idx+1}")

                                print(f"\n   ✅ Fatten mode complete: {original_voice_count} original + {original_voice_count} octave = {len(voice_midi_paths)} total voices")
                                print(f"      📊 Final counts: voice_midi_paths={len(voice_midi_paths)}, voice_audio_paths={len(voice_audio_paths)}")

                            elif fatten_type == "fake":
                                # Fake mode: we'll pitch shift the generated outputs later (after model generation)
                                print(f"   Fake mode: Will pitch shift outputs after generation")
                                # We'll handle this in the voice processing loop below
                    else:
                        print(f"⚠️ No notes found in generated MIDI, using as single voice")
                        voice_midi_paths = [midi_path]
                        # Include wav_path if: normal mode, or fast mode with noise < 1.0, or encodec variant
                        should_include_wav = (not fast_mode_variant or noise_level < 1.0 or fast_mode_variant == "encodec")
                        voice_audio_paths = [wav_path] if should_include_wav else []

                    # IMPORTANT: Don't discard audio_file_path in inpaint mode!
                    # For inpainting, we need the original voice's audio to extract conditioning
                    if not inpaint_mode:
                        audio_file_path = None  # Keep as None to trigger monophonic voice path
                    else:
                        print(f"🎨 INPAINT MODE: Preserving audio_file_path for voice {inpaint_voice_index}")

        # Setup output directory
        process_id = str(uuid.uuid4())
        output_dir = ensure_path_exists(get_output_path('ace_step_output', process_id=process_id))

        # Handle monophonic mode with scene changes (pre-rendered voices)
        if monophonic_mode and audio_file_path is None and 'voice_midi_paths' in locals():
            # Ensure voice_audio_paths is defined
            if 'voice_audio_paths' not in locals():
                voice_audio_paths = []
                print(f"⚠️ WARNING: voice_audio_paths was not defined, initialized as empty list")

            # Debug: Check voice paths
            print(f"\n🔍 DEBUG: Voice paths check:")
            print(f"   voice_midi_paths: {len(voice_midi_paths)} elements")
            print(f"   voice_audio_paths: {len(voice_audio_paths)} elements")
            if len(voice_audio_paths) > 0:
                print(f"   First audio path: {voice_audio_paths[0]}")
            else:
                print(f"   ⚠️ voice_audio_paths is EMPTY - this will cause issues for encodec variant!")

            if fast_mode_variant:
                print(f"\n⚡ FAST MODE ({fast_mode_variant.upper()}): MONOPHONIC - Processing {len(voice_midi_paths)} MIDI voices...")
                if fast_mode_variant == "encodec":
                    print(f"   ⚠️ ENCODEC mode requires FluidSynth-rendered audio for extraction")
            else:
                print(f"\n🎵 MONOPHONIC + SCENE CHANGES MODE: Generating {len(voice_audio_paths)} voices...")
                if tape_speed < 1.0:
                    print(f"🎞️ TAPE SPEED SLOWDOWN ENABLED: {tape_speed}x using {slowdown_method} method")
                    print(f"   Will slow down FluidSynth-rendered voices, process, then speed back up")

            # Track completed voices for incremental updates
            completed_voices = []
            input_files = {}  # Track input files progressively
            fps = 43.066

            # Determine number of voices to process
            num_voices = len(voice_midi_paths)

            # CRITICAL: Calculate consistent duration for all voices from concatenated MIDI
            # This ensures all voices have the same window_slow value (avoids shape mismatch errors)
            if 'concatenated_midi_path' in locals() and concatenated_midi_path:
                import pretty_midi
                pm_concat = pretty_midi.PrettyMIDI(concatenated_midi_path)
                consistent_duration = pm_concat.get_end_time()
                consistent_window_slow = clamp_window_slow(int(consistent_duration * fps), consistent_duration, fps)
                print(f"\n📏 Using consistent duration for all voices:")
                print(f"   Source: {Path(concatenated_midi_path).name}")
                print(f"   Duration: {consistent_duration:.2f}s")
                print(f"   Window: {consistent_window_slow} frames")
            else:
                consistent_duration = None
                consistent_window_slow = None
                print(f"\n⚠️ No concatenated MIDI found, will use individual voice durations")

            for voice_idx in range(num_voices):
                # VOICE SELECTION: Skip voices that aren't being regenerated (works for both inpaint and regeneration)
                if inpaint_voice_index is not None:
                    # Voice files are 1-indexed (1.wav, 2.wav, etc.)
                    # But voice_idx is 0-indexed, so voice_idx + 1 == file number
                    if (voice_idx + 1) != inpaint_voice_index:
                        mode_str = "inpainting" if inpaint_mode else "regenerating"
                        print(f"\n⏭️  Skipping voice {voice_idx + 1} ({mode_str} voice {inpaint_voice_index} only)")
                        continue

                print(f"\n🎼 Processing voice {voice_idx + 1}/{num_voices}")

                # Fast mode logic
                if fast_mode_variant and noise_level >= 1.0 and fast_mode_variant != "encodec":
                    # FAST MODE (noise >= 1.0, non-encodec): Convert MIDI directly to conditioning, no audio needed
                    voice_midi = voice_midi_paths[voice_idx]
                    print(f"   ⚡ Fast mode (noise {noise_level}): MIDI file: {Path(voice_midi).name}")

                    # Use consistent duration if available, otherwise calculate from MIDI
                    if consistent_duration is not None:
                        midi_duration = consistent_duration
                        window_slow = consistent_window_slow
                        print(f"   Using consistent duration: {midi_duration:.2f}s ({window_slow} frames)")
                    else:
                        import pretty_midi
                        pm = pretty_midi.PrettyMIDI(voice_midi)
                        midi_duration = pm.get_end_time()
                        window_slow = clamp_window_slow(int(midi_duration * fps), midi_duration, fps)
                        print(f"   MIDI duration: {midi_duration:.2f}s ({window_slow} frames)")

                    # Generate conditioning directly from MIDI (with automation if available)
                    # Use fast_mode_variant to determine conditioning strategy
                    piano_roll, amp, rframe, rbend, encodec_tokens = generate_conditioning_from_midi_fast(
                        voice_midi, window_slow, fps,
                        automation_points=global_automation if 'global_automation' in locals() else None,
                        duration=midi_duration,
                        variant=fast_mode_variant  # "zero" or "synthetic"
                    )

                    # Set duration and audio file for generate() call
                    actual_duration = midi_duration
                    voice_audio = None  # No audio file in fast mode with noise >= 1.0

                elif fast_mode_variant and (noise_level < 1.0 or fast_mode_variant == "encodec"):
                    # FAST MODE (noise < 1.0 or encodec variant): Use rendered audio for extraction
                    # Safety check: ensure voice_audio_paths has enough elements
                    if voice_idx >= len(voice_audio_paths):
                        print(f"   ⚠️ ERROR: voice_audio_paths has only {len(voice_audio_paths)} elements, need {voice_idx + 1}")
                        print(f"   ⚠️ This shouldn't happen - voice_midi_paths has {len(voice_midi_paths)} elements")

                        if fast_mode_variant == "encodec":
                            # ENCODEC VARIANT REQUIRES AUDIO - cannot fall back to MIDI
                            raise RuntimeError(
                                f"Encodec variant requires rendered audio but voice_audio_paths is empty! "
                                f"This indicates the FluidSynth rendering was skipped incorrectly. "
                                f"Check the rendering logic around line 6012-6030."
                            )
                        else:
                            # For other fast modes, fall back to MIDI processing
                            print(f"   ⚠️ Falling back to normal mode extraction from MIDI")

                            # Fall back to processing from MIDI
                            voice_midi = voice_midi_paths[voice_idx]
                            if consistent_duration is not None:
                                midi_duration = consistent_duration
                                window_slow = consistent_window_slow
                            else:
                                pm = pretty_midi.PrettyMIDI(voice_midi)
                                midi_duration = pm.get_end_time()
                                window_slow = clamp_window_slow(int(midi_duration * fps), midi_duration, fps)

                            piano_roll, amp, rframe, rbend, encodec_tokens = generate_conditioning_from_midi_fast(
                                voice_midi, window_slow, fps,
                                automation_points=global_automation if 'global_automation' in locals() else None,
                                duration=midi_duration,
                                variant=fast_mode_variant
                            )

                            actual_duration = midi_duration
                            voice_audio = None
                    else:
                        voice_audio = voice_audio_paths[voice_idx]
                        voice_midi = voice_midi_paths[voice_idx]
                        if fast_mode_variant == "encodec":
                            print(f"   ⚡ Fast mode (ENCODEC): Rendered audio: {Path(voice_audio).name}")
                            print(f"   Will extract only encodec tokens, zero piano_roll/amp/rframe/rbend")
                        else:
                            print(f"   ⚡ Fast mode (ZERO, noise {noise_level}): MIDI: {Path(voice_midi).name}, Audio: {Path(voice_audio).name}")
                            print(f"   Will extract piano roll from MIDI (not Basic Pitch!), use preset for amp/rframe/rbend/encodec")

                        # Use consistent duration if available, otherwise calculate from audio
                        if consistent_duration is not None:
                            actual_duration = consistent_duration
                            window_slow = consistent_window_slow
                            print(f"   Using consistent duration: {actual_duration:.2f}s ({window_slow} frames)")
                        else:
                            import torchaudio
                            wav, sr = torchaudio.load(voice_audio)
                            actual_duration = wav.shape[-1] / sr
                            window_slow = clamp_window_slow(int(actual_duration * fps), actual_duration, fps)
                            print(f"   Voice duration from audio: {actual_duration:.2f}s ({window_slow} frames)")

                        # Fast mode with MIDI: Extract piano roll from MIDI, not from audio
                        if fast_mode_variant == "zero" and voice_midi:
                            print(f"   🎼 Extracting piano roll from MIDI: {Path(voice_midi).name}")
                            # Extract piano roll from MIDI directly
                            piano_roll, _, _, _, _ = midi_to_piano_roll_conditioning(voice_midi, window_slow, fps=fps, tempo_override=None)
                            print(f"   ✅ Piano roll from MIDI: {piano_roll.shape}")

                            # Create preset values for other conditioning (same as fast mode)
                            amp = np.full((window_slow,), 0.5, dtype=np.float32)
                            rframe = np.zeros((window_slow,), dtype=np.float32)
                            rbend = np.zeros((window_slow,), dtype=np.float32)
                            encodec_length = window_slow // 2
                            encodec_tokens = torch.zeros((1, 8, encodec_length), dtype=torch.long)
                        else:
                            # Extract conditioning using fast mode variant (for encodec mode)
                            extraction = extract_conditioning_from_audio_fast_mode(
                                voice_audio,
                                instrument_group=subgroup,
                                variant=fast_mode_variant
                            )
                            piano_roll, amp, rframe, rbend, encodec_tokens = load_conditioning_fast_mode(extraction, window_slow, variant=fast_mode_variant)

                else:
                    # NORMAL MODE: Extract from rendered audio
                    voice_audio = voice_audio_paths[voice_idx]
                    print(f"   Original FluidSynth render: {voice_audio}")

                    # Apply speed slowdown if needed
                    if tape_speed < 1.0:
                        print(f"🎞️ Applying {slowdown_method} slowdown ({tape_speed}x) to voice {voice_idx + 1}...")
                        slowed_voice_path = str(Path(voice_audio).parent / f"slowed_{Path(voice_audio).name}")
                        print(f"   Input:  {voice_audio}")
                        print(f"   Output: {slowed_voice_path}")
                        if slowdown_method == "stretch":
                            apply_time_stretch_sox(voice_audio, slowed_voice_path, tape_speed)
                        else:  # tape
                            apply_tape_speed_sox(voice_audio, slowed_voice_path, tape_speed)
                        voice_audio = slowed_voice_path
                        print(f"✅ Voice {voice_idx + 1} slowed file saved: {slowed_voice_path}")
                        print(f"   Now using slowed version for conditioning extraction")

                    # Determine duration from voice audio
                    import torchaudio
                    wav, sr = torchaudio.load(voice_audio)
                    actual_duration = wav.shape[-1] / sr
                    window_slow = clamp_window_slow(int(actual_duration * fps), actual_duration, fps)
                    print(f"   Voice duration: {actual_duration:.2f}s ({window_slow} frames)")

                    # Extract conditioning from rendered audio (with selective format extraction)
                    extraction = extract_conditioning_from_audio(
                        voice_audio,
                        instrument_group=subgroup,
                        extract_formats=extract_formats
                    )
                    piano_roll, amp, rframe, rbend, encodec_tokens = load_conditioning(extraction, window_slow)

                # INPAINT MODE: Slice conditioning to context window
                if inpaint_mode and inpaint_voice_index is not None and (voice_idx + 1) == inpaint_voice_index:
                    print(f"\n{'='*80}")
                    print(f"🎨 INPAINTING VOICE {inpaint_voice_index} - EXTRACTING CONTEXT WINDOW")
                    print(f"{'='*80}")

                    # Calculate frame boundaries
                    context_seconds = 2.0
                    start_frame = int(inpaint_start_time * fps)
                    end_frame = int(inpaint_end_time * fps)
                    context_frames = int(context_seconds * fps)

                    # Get total frames from conditioning
                    total_frames = piano_roll.shape[-1] if len(piano_roll.shape) > 1 else piano_roll.shape[0]
                    context_start = max(0, start_frame - context_frames)
                    context_end = min(total_frames, end_frame + context_frames)

                    print(f"📍 Inpaint region: {inpaint_start_time:.2f}s - {inpaint_end_time:.2f}s")
                    print(f"   Frames: {start_frame} - {end_frame}")
                    print(f"📍 Context window: {context_start} - {context_end} frames")
                    print(f"   Time: {context_start/fps:.2f}s - {context_end/fps:.2f}s")

                    # Helper function to slice conditioning
                    def slice_cond(arr, start, end):
                        if arr is None:
                            return None
                        if len(arr.shape) == 1:
                            return arr[start:end]
                        else:
                            return arr[..., start:end]

                    # Slice all conditioning to context window
                    piano_roll = slice_cond(piano_roll, context_start, context_end)
                    amp = slice_cond(amp, context_start, context_end)
                    rframe = slice_cond(rframe, context_start, context_end)
                    rbend = slice_cond(rbend, context_start, context_end)
                    encodec_tokens = slice_cond(encodec_tokens, context_start, context_end)

                    # Update actual_duration to context window duration
                    context_duration = (context_end - context_start) / fps
                    actual_duration = context_duration
                    window_slow = context_end - context_start

                    print(f"✅ Sliced conditioning to context window:")
                    print(f"   Duration: {context_duration:.2f}s ({window_slow} frames)")
                    print(f"   Piano roll: {piano_roll.shape}")
                    print(f"{'='*80}\n")

                    # Store metadata for post-processing
                    inpaint_metadata = {
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'context_start': context_start,
                        'context_end': context_end,
                        'context_frames': context_frames,
                        'fps': fps,
                        'start_time': inpaint_start_time,
                        'end_time': inpaint_end_time,
                        'context_start_time': context_start / fps,
                        'context_end_time': context_end / fps,
                        'voice_idx': voice_idx
                    }

                # Get MIDI duration for cropping (to avoid white noise from zero conditioning)
                voice_midi = voice_midi_paths[voice_idx]
                import pretty_midi
                pm_voice = pretty_midi.PrettyMIDI(voice_midi)
                midi_end_time = pm_voice.get_end_time()
                print(f"   MIDI end time: {midi_end_time:.2f}s (will crop output to this duration)")

                # Generate audio for this voice
                voice_seed = seed + (voice_idx * 1000)
                print(f"   Generating with seed {voice_seed}...")

                voice_output = generate(
                    model=MODEL,
                    piano_roll=piano_roll,
                    amp=amp,
                    rframe=rframe,
                    rbend=rbend,
                    encodec_tokens=encodec_tokens,
                    group=group,
                    subgroup=subgroup,
                    steps=steps,
                    seed=voice_seed,
                    adapter_scale=adapter_scale,
                    cfg_weight=cfg_weight,
                    t0=1.0,
                    sr_out=44100,
                    instrument_strength=instrument_strength,
                    inst_boost=2.5,
                    piano_roll_gain=piano_roll_gain,
                    amp_gain=amp_gain,
                    rframe_gain=rframe_gain,
                    rbend_gain=rbend_gain,
                    encodec_gain=encodec_gain,
                    use_overlap_decoder=use_overlap_decoder,
                    original_audio_length=int(actual_duration * 44100),
                    pitch_fidelity_boost=pitch_fidelity_boost,
                    onset_guidance_boost=onset_guidance_boost,
                    pitch_snap_strength=pitch_snap_strength,
                    noise_level=noise_level,
                    audio_file=voice_audio,
                    fast_mode_variant=fast_mode_variant,  # Pass fast_mode_variant for T_dcae conversion
                    target_audio_duration=consistent_duration  # Ensure consistent latent extraction
                )

                # Crop audio to MIDI duration (avoid white noise from zero conditioning after MIDI ends)
                import torchaudio
                wav, sr = torchaudio.load(voice_output)
                output_duration = wav.shape[-1] / sr

                if output_duration > midi_end_time + 0.1:  # Add 0.1s tolerance
                    # Crop to MIDI duration
                    crop_samples = int(midi_end_time * sr)
                    wav_cropped = wav[:, :crop_samples]

                    # Save cropped version
                    cropped_path = str(Path(voice_output).parent / f"cropped_{Path(voice_output).name}")
                    torchaudio.save(cropped_path, wav_cropped, sr)
                    voice_output = cropped_path

                    print(f"   ✂️  Cropped output: {output_duration:.2f}s → {midi_end_time:.2f}s (removed {output_duration - midi_end_time:.2f}s of zero-conditioning)")
                else:
                    print(f"   ✅ Output duration ({output_duration:.2f}s) matches MIDI ({midi_end_time:.2f}s), no crop needed")

                # INPAINT MODE: Extract inpainted region from context window
                if inpaint_mode and inpaint_voice_index is not None and (voice_idx + 1) == inpaint_voice_index:
                    print(f"\n{'='*80}")
                    print(f"🎨 INPAINTING POST-PROCESSING - VOICE {inpaint_voice_index}")
                    print(f"{'='*80}")

                    # Reload the audio (it may have been cropped)
                    wav, sr = torchaudio.load(voice_output)
                    print(f"📥 Loaded generated audio: {wav.shape} at {sr}Hz")

                    # Calculate sample positions using time values (no frame rounding!)
                    inpaint_offset_time = inpaint_metadata['start_time'] - inpaint_metadata['context_start_time']
                    inpaint_duration_time = inpaint_metadata['end_time'] - inpaint_metadata['start_time']

                    print(f"\n⏱️  Time-based calculation (PRECISE):")
                    print(f"   Context start: {inpaint_metadata['context_start_time']:.6f}s")
                    print(f"   Inpaint start: {inpaint_metadata['start_time']:.6f}s")
                    print(f"   Inpaint end: {inpaint_metadata['end_time']:.6f}s")
                    print(f"   Offset in generated: {inpaint_offset_time:.6f}s")
                    print(f"   Duration: {inpaint_duration_time:.6f}s")

                    # Convert directly to samples
                    inpaint_start_sample = int(inpaint_offset_time * sr)
                    inpaint_end_sample = int((inpaint_offset_time + inpaint_duration_time) * sr)

                    print(f"   → Start sample: {inpaint_start_sample} ({inpaint_start_sample/sr:.6f}s)")
                    print(f"   → End sample: {inpaint_end_sample} ({inpaint_end_sample/sr:.6f}s)")

                    # Extract inpainted region
                    inpainted_audio = wav[..., inpaint_start_sample:inpaint_end_sample]
                    print(f"\n✂️  Extracted inpainted region: {inpainted_audio.shape}")
                    print(f"   Duration: {inpainted_audio.shape[-1]/sr:.2f}s")
                    print(f"   This will replace {inpaint_metadata['start_time']:.2f}s - {inpaint_metadata['end_time']:.2f}s")

                    # Save only the inpainted segment (frontend will splice)
                    inpaint_output = str(Path(voice_output).parent / f"inpaint_{Path(voice_output).name}")
                    torchaudio.save(inpaint_output, inpainted_audio, sr)
                    voice_output = inpaint_output
                    print(f"💾 Saved inpainted segment: {Path(inpaint_output).name}")
                    print(f"   Frontend will splice this into the original track")
                    print(f"{'='*80}\n")

                # Copy voice to output directory
                voice_output_path = output_dir / f"{voice_idx + 1}.wav"
                shutil.copy(voice_output, str(voice_output_path))

                # Copy to debug folder if it exists
                if 'voice_debug_dir' in locals() and voice_debug_dir.exists():
                    debug_output_path = voice_debug_dir / f"voice_{voice_idx + 1}_output.wav"
                    shutil.copy(voice_output, str(debug_output_path))
                    print(f"   📁 Debug copy: {debug_output_path.name}")

                # Add to completed list
                download_url = f"/download/{process_id}/{voice_idx + 1}.wav"
                completed_voices.append(download_url)

                # Copy input files progressively for this voice
                voice_num = voice_idx + 1
                try:
                    # Copy MIDI input
                    if voice_idx < len(voice_midi_paths) and voice_midi_paths[voice_idx]:
                        midi_src = Path(voice_midi_paths[voice_idx])
                        if midi_src.exists():
                            midi_dest = output_dir / f"{voice_num}_input.mid"
                            shutil.copy(str(midi_src), str(midi_dest))
                            if str(voice_num) not in input_files:
                                input_files[str(voice_num)] = {}
                            input_files[str(voice_num)]["midi_path"] = f"/download/{process_id}/{voice_num}_input.mid"

                    # Copy FluidSynth render if it exists
                    if voice_idx < len(voice_audio_paths) and voice_audio_paths[voice_idx]:
                        audio_src = Path(voice_audio_paths[voice_idx])
                        if audio_src.exists():
                            audio_dest = output_dir / f"{voice_num}_input_render.wav"
                            shutil.copy(str(audio_src), str(audio_dest))
                            if str(voice_num) not in input_files:
                                input_files[str(voice_num)] = {}
                            input_files[str(voice_num)]["render_path"] = f"/download/{process_id}/{voice_num}_input_render.wav"
                            input_files[str(voice_num)]["type"] = "wav"
                        else:
                            if str(voice_num) not in input_files:
                                input_files[str(voice_num)] = {}
                            input_files[str(voice_num)]["type"] = "midi"
                    else:
                        if str(voice_num) not in input_files:
                            input_files[str(voice_num)] = {}
                        input_files[str(voice_num)]["type"] = "midi"
                except Exception as e:
                    print(f"  ⚠️ Error copying input files for voice {voice_num}: {e}")

                # Update Celery task state
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'completed_voices': completed_voices.copy(),
                        'total_voices': num_voices,
                        'progress': len(completed_voices) / num_voices if num_voices > 0 else 0,
                        'input_files': input_files.copy()
                    }
                )
                print(f"   ✅ Voice {voice_idx + 1} complete: {voice_output_path.name}")

            print(f"\n✅ All {num_voices} voices generated successfully")

            # FAKE FATTEN MODE: Pitch shift generated outputs
            if fatten_mode and fatten_type == "fake":
                print(f"\n🎚️ FATTEN MODE (FAKE): Pitch shifting {num_voices} voices up an octave...")
                import librosa
                import soundfile as sf

                original_voice_count = num_voices
                for voice_idx in range(original_voice_count):
                    # VOICE SELECTION: Skip voices that aren't being regenerated (works for both inpaint and regeneration)
                    if inpaint_voice_index is not None:
                        if (voice_idx + 1) != inpaint_voice_index:
                            mode_str = "inpainting" if inpaint_mode else "regenerating"
                            print(f"   ⏭️  Skipping fatten for voice {voice_idx + 1} ({mode_str} voice {inpaint_voice_index} only)")
                            continue

                    print(f"   🎚️ Pitch shifting voice {voice_idx+1}/{original_voice_count}...")

                    # Load the generated voice
                    original_voice_path = output_dir / f"{voice_idx + 1}.wav"
                    audio, sr = librosa.load(str(original_voice_path), sr=None)

                    # Pitch shift up 12 semitones (one octave)
                    shifted_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=12)

                    # Save shifted version
                    octave_voice_path = output_dir / f"{original_voice_count + voice_idx + 1}.wav"
                    sf.write(str(octave_voice_path), shifted_audio, sr)
                    print(f"      ✅ Saved: {octave_voice_path.name}")

                    # Also save to debug folder if it exists
                    if 'voice_debug_dir' in locals() and voice_debug_dir.exists():
                        debug_octave_path = voice_debug_dir / f"voice_{original_voice_count + voice_idx + 1}_octave_fake.wav"
                        sf.write(str(debug_octave_path), shifted_audio, sr)

                    # Add to completed list
                    download_url = f"/download/{process_id}/{original_voice_count + voice_idx + 1}.wav"
                    completed_voices.append(download_url)

                    # Copy input files for octave voice (same as original voice)
                    octave_voice_num = original_voice_count + voice_idx + 1
                    original_voice_num = voice_idx + 1
                    try:
                        if str(original_voice_num) in input_files:
                            # Octave voices share the same input files as their corresponding original voices
                            input_files[str(octave_voice_num)] = input_files[str(original_voice_num)].copy()
                    except Exception as e:
                        print(f"  ⚠️ Error copying input file reference for octave voice {octave_voice_num}: {e}")

                    # Send progress update for this octave voice
                    total_voices_with_octaves = original_voice_count * 2  # Original + octave
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'completed_voices': completed_voices.copy(),
                            'total_voices': total_voices_with_octaves,
                            'progress': len(completed_voices) / total_voices_with_octaves,
                            'input_files': input_files.copy()
                        }
                    )

                print(f"\n   ✅ Fake fatten mode complete: {original_voice_count} original + {original_voice_count} octave = {len(completed_voices)} total voices")

            # Print debug folder summary
            if 'voice_debug_dir' in locals() and voice_debug_dir.exists():
                print(f"\n{'='*60}")
                print(f"📁 VOICE DEBUG FILES SAVED")
                print(f"{'='*60}")
                print(f"Location: {voice_debug_dir}")
                print(f"\nMaster file:")
                print(f"  • concatenated_master.mid  (All voices combined with tempo changes)")
                print(f"\nPer-voice files:")
                for i in range(num_voices):
                    # Use tracked voice subgroup if available, otherwise use global subgroup
                    voice_instrument = voice_subgroups[i] if 'voice_subgroups' in locals() and i < len(voice_subgroups) else subgroup
                    print(f"  Voice {i+1}:")
                    print(f"    • voice_{i+1}_input.mid          (MIDI with tempo changes)")
                    if fast_mode_variant and noise_level >= 1.0:
                        print(f"    • (no render - fast mode)")
                    else:
                        print(f"    • voice_{i+1}_input_render.wav  (Rendered with {voice_instrument} soundfont)")
                    print(f"    • voice_{i+1}_output.wav        (AI-generated output)")
                print(f"{'='*60}\n")

            # Apply speed up if we slowed down the input
            if tape_speed < 1.0:
                print(f"\n{'='*80}")
                print(f"🎞️ SPEED RESTORATION ({slowdown_method.upper()}): {1.0/tape_speed:.2f}x")
                print(f"{'='*80}")
                print(f"Speeding up all {num_voices} voice outputs...")

                speedup_factor = 1.0 / tape_speed
                for voice_idx in range(num_voices):
                    voice_file = output_dir / f"{voice_idx + 1}.wav"
                    if voice_file.exists():
                        print(f"   Processing voice {voice_idx + 1}: {voice_file}")
                        temp_path = str(voice_file.parent / f"temp_{voice_file.name}")
                        print(f"   Input:  {voice_file}")
                        print(f"   Temp:   {temp_path}")
                        if slowdown_method == "stretch":
                            apply_time_stretch_sox(str(voice_file), temp_path, speedup_factor)
                        else:  # tape
                            apply_tape_speed_sox(str(voice_file), temp_path, speedup_factor)
                        shutil.move(temp_path, str(voice_file))
                        print(f"✅ Voice {voice_idx + 1} restored and saved: {voice_file}")

                print(f"✅ All voice outputs restored to original speed")
                print(f"{'='*80}\n")

                # UPSAMPLE MODE: Refine the sped-up audio with additional diffusion steps
                if upsample_mode:
                    print(f"\n{'='*80}")
                    print(f"✨ UPSAMPLE MODE: Refining audio with {upsample_steps} diffusion steps")
                    print(f"{'='*80}")
                    print(f"Noise level: {upsample_noise_level:.2f}")
                    print(f"Processing {num_voices} voices...")

                    for voice_idx in range(num_voices):
                        voice_file = output_dir / f"{voice_idx + 1}.wav"
                        if voice_file.exists():
                            print(f"\n   🎯 Upsampling voice {voice_idx + 1}...")

                            # Extract latent from the sped-up audio
                            print(f"      1. Extracting latent from restored audio...")

                            # Extract conditioning from the sped-up audio (pass file path, not array)
                            extraction = extract_conditioning_from_audio(
                                str(voice_file),
                                instrument_group=subgroup,
                                extract_formats=extract_formats
                            )
                            piano_roll, amp, rframe, rbend, encodec_tokens = load_conditioning(extraction, consistent_window_slow)

                            # Get audio duration
                            voice_audio, sr = torchaudio.load(str(voice_file))
                            audio_duration = voice_audio.shape[-1] / sr

                            print(f"      2. Adding {upsample_noise_level:.2f} noise and running {upsample_steps} diffusion steps...")

                            # Generate with partial noise for upsampling
                            upsampled_output = generate(
                                model=MODEL,
                                piano_roll=piano_roll,
                                amp=amp,
                                rframe=rframe,
                                rbend=rbend,
                                encodec_tokens=encodec_tokens,
                                group=group,
                                subgroup=subgroup,
                                steps=upsample_steps,
                                seed=seed,
                                adapter_scale=adapter_scale,
                                cfg_weight=cfg_weight,
                                t0=1.0,
                                sr_out=44100,
                                instrument_strength=instrument_strength,
                                inst_boost=2.5,
                                piano_roll_gain=piano_roll_gain,
                                amp_gain=amp_gain,
                                rframe_gain=rframe_gain,
                                rbend_gain=rbend_gain,
                                encodec_gain=encodec_gain,
                                use_overlap_decoder=use_overlap_decoder,
                                original_audio_length=int(audio_duration * 44100),
                                pitch_fidelity_boost=pitch_fidelity_boost,
                                onset_guidance_boost=onset_guidance_boost,
                                pitch_snap_strength=pitch_snap_strength,
                                noise_level=upsample_noise_level,
                                audio_file=str(voice_file),
                                target_audio_duration=audio_duration,
                            )

                            # Load and save the upsampled result
                            print(f"      3. Saving upsampled audio...")
                            upsampled_wav, upsample_sr = torchaudio.load(upsampled_output)
                            torchaudio.save(str(voice_file), upsampled_wav, 44100)

                            print(f"   ✅ Voice {voice_idx + 1} upsampled and saved")

                    print(f"\n✅ All voices upsampled successfully")
                    print(f"{'='*80}\n")

            # Input files already copied progressively during voice processing
            # (input_files dict was built incrementally as each voice completed)

                print(f"✅ Input files copied and indexed")

            # Also copy the concatenated MIDI (master) if available
            if 'concatenated_midi_path' in locals() and concatenated_midi_path and os.path.exists(concatenated_midi_path):
                try:
                    concat_midi_src = Path(concatenated_midi_path)
                    concat_midi_dest = output_dir / "0_input_master.mid"
                    shutil.copy(str(concat_midi_src), str(concat_midi_dest))
                    print(f"  ✅ Copied: 0_input_master.mid (concatenated/master MIDI)")
                    # Add to voice 0 or create if doesn't exist
                    if "0" not in input_files:
                        input_files["0"] = {}
                    input_files["0"]["master_midi_path"] = f"/download/{process_id}/0_input_master.mid"
                except Exception as e:
                    print(f"  ⚠️ Error copying concatenated MIDI: {e}")

            return {"file_paths": completed_voices, "input_files": input_files}

        # MULTITRACK MIDI DETECTION: Check if uploaded file is multi-track MIDI with voice separation disabled
        if is_midi_file(audio_file_path) and monophonic_mode and not enable_voice_separation:
            print(f"\n{'='*80}")
            print(f"🎼 MULTI-TRACK MIDI DETECTED (Voice Separation: Disabled)")
            print(f"{'='*80}")
            if tape_speed < 1.0:
                print(f"🎞️ TAPE SPEED ENABLED: {tape_speed}x using {slowdown_method} method")
                print(f"   FluidSynth renders will be slowed, processed, then sped back up")

            is_multi, track_count, _ = is_multitrack_midi(audio_file_path)

            if is_multi:
                print(f"   Detected {track_count} tracks - rendering each with FluidSynth")
                print(f"   Instrument: {subgroup}")

                # Create debug directory in /tmp
                debug_dir = Path(f"/tmp/multitrack_debug_{process_id}")
                debug_dir.mkdir(parents=True, exist_ok=True)
                print(f"   Debug folder: {debug_dir}")

                # Step 1: Split MIDI into separate track files
                print(f"\n📂 Step 1: Splitting MIDI into {track_count} separate files...")
                track_midi_files = split_midi_into_track_files(audio_file_path, output_dir=str(debug_dir))
                print(f"   ✅ Created {len(track_midi_files)} track MIDI files")

                # Step 1.5: Modify MIDI tempo if using tape speed slowdown
                # In stretch mode or fast mode: modify MIDI tempo directly
                # In tape mode with rendering: skip (will slow rendered audio instead)
                should_render_midi = not (fast_mode_variant and noise_level >= 1.0 and fast_mode_variant != "encodec")

                if tape_speed < 1.0:
                    # Modify MIDI tempo if: stretch mode, OR fast mode (any slowdown method)
                    should_modify_midi = slowdown_method == "stretch" or not should_render_midi

                    if should_modify_midi:
                        print(f"\n🎼 Step 1.5: Modifying MIDI tempo ({tape_speed}x)...")
                        if not should_render_midi:
                            print(f"   Fast mode: Slowing MIDI before direct conversion")
                        else:
                            print(f"   Stretch mode: Slowing MIDI tempo")
                        slowed_midi_files = []
                        for i, track_midi in enumerate(track_midi_files):
                            slowed_midi_path = str(Path(track_midi).parent / f"slowed_{Path(track_midi).name}")
                            modify_midi_tempo(track_midi, slowed_midi_path, tape_speed)
                            slowed_midi_files.append(slowed_midi_path)
                        track_midi_files = slowed_midi_files
                        print(f"   ✅ All {len(track_midi_files)} MIDI files slowed to {tape_speed}x tempo")
                    else:
                        print(f"\n🎼 Step 1.5: Skipping MIDI tempo modification (tape mode - will slow renders instead)")

                # Step 2: Render each track with FluidSynth (using correct soundfont)
                # Skip rendering in fast mode with high noise (will convert MIDI directly)
                track_audio_files = []
                should_render_midi = not (fast_mode_variant and noise_level >= 1.0 and fast_mode_variant != "encodec")

                if should_render_midi:
                    print(f"\n🎹 Step 2: Rendering each track with FluidSynth ({subgroup} soundfont)...")
                    for i, track_midi in enumerate(track_midi_files):
                        print(f"   Rendering track {i+1}/{len(track_midi_files)}: {Path(track_midi).name}")
                        track_audio = render_midi_to_audio(
                            track_midi,
                            output_dir=str(debug_dir),
                            instrument_group=subgroup  # Use selected instrument for soundfont
                        )
                        track_audio_files.append(track_audio)
                        print(f"      → {Path(track_audio).name}")
                    print(f"   ✅ All tracks rendered")
                else:
                    print(f"\n⚡ Step 2: FAST MODE ({fast_mode_variant.upper()}): Skipping FluidSynth render")
                    print(f"   Will convert MIDI directly to conditioning for correct timing")
                    # Keep track_audio_files empty - will use track_midi_files directly

                # Step 2.5: Real fatten mode - transpose MIDI and render octave-up versions
                if fatten_mode and fatten_type == "real":
                    print(f"\n🎚️ Step 2.5 REAL FATTEN MODE: Creating octave-up MIDI...")
                    original_track_count = len(track_midi_files)
                    octave_midi_files = []
                    octave_audio_files = []

                    for i, track_midi in enumerate(track_midi_files):
                        # Load MIDI and transpose up 12 semitones (one octave)
                        import pretty_midi
                        pm = pretty_midi.PrettyMIDI(track_midi)
                        for instrument in pm.instruments:
                            for note in instrument.notes:
                                note.pitch += 12  # Transpose up one octave

                        # Save transposed MIDI
                        octave_midi_path = str(Path(track_midi).parent / f"octave_{Path(track_midi).name}")
                        pm.write(octave_midi_path)
                        octave_midi_files.append(octave_midi_path)
                        print(f"   Created octave-up MIDI {i+1}: {Path(octave_midi_path).name}")

                        # Render octave-up version only if we're rendering audio
                        if should_render_midi:
                            octave_audio = render_midi_to_audio(
                                octave_midi_path,
                                output_dir=str(debug_dir),
                                instrument_group=subgroup
                            )
                            octave_audio_files.append(octave_audio)
                            print(f"      → Rendered: {Path(octave_audio).name}")

                    # Add octave files to the processing lists
                    track_midi_files.extend(octave_midi_files)
                    if should_render_midi:
                        track_audio_files.extend(octave_audio_files)
                        print(f"   ✅ Doubled to {len(track_audio_files)} total audio tracks ({original_track_count} original + {original_track_count} octave-up)")
                    else:
                        print(f"   ✅ Doubled to {len(track_midi_files)} total MIDI tracks ({original_track_count} original + {original_track_count} octave-up)")

                # Step 2.6: Apply tape slowdown to renders if using tape mode
                # Only apply if we have rendered audio files (not in fast mode)
                if tape_speed < 1.0 and slowdown_method == "tape" and len(track_audio_files) > 0:
                    print(f"\n🎞️ Step 2.6: Applying tape slowdown ({tape_speed}x) to FluidSynth renders...")
                    slowed_audio_files = []
                    for i, track_audio in enumerate(track_audio_files):
                        slowed_audio_path = str(Path(track_audio).parent / f"slowed_{Path(track_audio).name}")
                        print(f"   Track {i+1}: {Path(track_audio).name} → {Path(slowed_audio_path).name}")
                        apply_tape_speed_sox(track_audio, slowed_audio_path, tape_speed)
                        slowed_audio_files.append(slowed_audio_path)
                    track_audio_files = slowed_audio_files
                    print(f"   ✅ All {len(track_audio_files)} renders slowed with tape method")
                elif tape_speed < 1.0 and slowdown_method == "stretch" and len(track_audio_files) > 0:
                    print(f"   (Renders are at {tape_speed}x tempo from MIDI modification)")

                # Step 3: Extract conditioning and generate for each track
                print(f"\n🎵 Step 3: Generating AI audio for each track...")
                if should_render_midi:
                    print(f"   📊 Total FluidSynth renders to process: {len(track_audio_files)}")
                else:
                    print(f"   📊 Total MIDI tracks to process: {len(track_midi_files)}")
                if fatten_mode and fatten_type == "fake":
                    print(f"   🎚️ FAKE FATTEN MODE: Will pitch shift outputs up an octave (doubling track count)")
                elif fatten_mode and fatten_type == "real":
                    input_count = len(track_audio_files) if track_audio_files else len(track_midi_files)
                    print(f"   🎚️ REAL FATTEN MODE: Octave-up tracks already prepared")
                    print(f"   📊 This includes {input_count//2} original + {input_count//2} octave-up tracks")
                completed_voices = []
                input_files = {}  # Track input files progressively
                voice_outputs = []
                fps = 43.066
                output_track_counter = 1  # Track number for output files (accounts for fatten mode doubling)
                # For real fatten, input is already doubled; for fake fatten, we'll double during generation
                base_track_count = len(track_audio_files) if track_audio_files else len(track_midi_files)
                total_expected_tracks = base_track_count * (2 if (fatten_mode and fatten_type == "fake") else 1)
                print(f"   📊 Expected total output tracks: {total_expected_tracks}")

                # Determine which files to process (MIDI or audio)
                processing_count = len(track_audio_files) if track_audio_files else len(track_midi_files)

                for track_idx in range(processing_count):
                    print(f"\n🎼 Processing track {track_idx + 1}/{processing_count}")

                    # Check if we're using MIDI directly (fast mode) or audio files
                    if not should_render_midi and track_idx < len(track_midi_files):
                        # FAST MODE: Convert MIDI directly to conditioning
                        track_midi = track_midi_files[track_idx]
                        print(f"   Input MIDI: {Path(track_midi).name}")

                        # Get duration from MIDI
                        import pretty_midi
                        pm = pretty_midi.PrettyMIDI(track_midi)
                        actual_duration = pm.get_end_time()
                        window_slow = clamp_window_slow(int(actual_duration * fps), actual_duration, fps)
                        print(f"   Duration: {actual_duration:.2f}s ({window_slow} frames)")

                        # Convert MIDI to conditioning using fast mode function
                        print(f"   ⚡ Converting MIDI to conditioning (fast mode)...")
                        piano_roll, amp, rframe, rbend, encodec_tokens = generate_conditioning_from_midi_fast(
                            track_midi,
                            window_slow=window_slow,
                            fps=fps,
                            automation_points=global_automation if 'global_automation' in locals() and len(global_automation) > 0 else None,
                            duration=actual_duration,
                            variant=fast_mode_variant
                        )
                        print(f"   ✅ MIDI converted: PR={piano_roll.shape}, amp={amp.shape}")

                        # No audio file in fast mode
                        track_audio = None
                        orig_audio_length = None
                    else:
                        # NORMAL MODE: Extract from rendered audio
                        track_audio = track_audio_files[track_idx]
                        print(f"   Input audio: {Path(track_audio).name}")

                        # Determine duration from track audio
                        import torchaudio
                        wav, sr = torchaudio.load(track_audio)
                        actual_duration = wav.shape[-1] / sr
                        window_slow = clamp_window_slow(int(actual_duration * fps), actual_duration, fps)
                        print(f"   Duration: {actual_duration:.2f}s ({window_slow} frames)")

                        # Extract conditioning from rendered track
                        print(f"   Extracting conditioning...")
                        if fast_mode_variant:
                            extraction = extract_conditioning_from_audio_fast_mode(
                                track_audio,
                                instrument_group=subgroup,
                                variant=fast_mode_variant
                            )
                            piano_roll, amp, rframe, rbend, encodec_tokens = load_conditioning_fast_mode(extraction, window_slow, variant=fast_mode_variant)
                        else:
                            extraction = extract_conditioning_from_audio(
                                track_audio,
                                instrument_group=subgroup,
                                extract_formats=extract_formats
                            )
                            piano_roll, amp, rframe, rbend, encodec_tokens = load_conditioning(extraction, window_slow)
                        print(f"   Loaded conditioning: PR={piano_roll.shape}, amp={amp.shape}")

                        orig_audio_length = wav.shape[-1]

                    # Generate audio for this track with FULL conditioning
                    voice_seed = seed + (track_idx * 1000)
                    print(f"   Generating with seed {voice_seed}...")

                    voice_output = generate(
                        model=MODEL,
                        piano_roll=piano_roll,
                        amp=amp,  # REAL conditioning
                        rframe=rframe,  # REAL conditioning
                        rbend=rbend,  # REAL conditioning
                        encodec_tokens=encodec_tokens,  # REAL conditioning
                        group=group,
                        subgroup=subgroup,
                        steps=steps,
                        seed=voice_seed,
                        adapter_scale=adapter_scale,
                        cfg_weight=cfg_weight,
                        t0=1.0,
                        sr_out=44100,
                        instrument_strength=instrument_strength,
                        inst_boost=2.5,
                        piano_roll_gain=piano_roll_gain,
                        amp_gain=amp_gain,
                        rframe_gain=rframe_gain,
                        rbend_gain=rbend_gain,
                        encodec_gain=encodec_gain,
                        use_overlap_decoder=use_overlap_decoder,
                        original_audio_length=orig_audio_length,
                        pitch_fidelity_boost=pitch_fidelity_boost,
                        onset_guidance_boost=onset_guidance_boost,
                        pitch_snap_strength=pitch_snap_strength,
                        noise_level=noise_level,
                        audio_file=track_audio,
                        fast_mode_variant=fast_mode_variant  # Pass fast_mode_variant for correct resolution handling
                    )

                    # Speed up this voice immediately if we slowed it down
                    if tape_speed < 1.0:
                        print(f"   🎞️ Restoring speed ({1.0/tape_speed:.2f}x) for track {track_idx + 1}...")
                        speedup_factor = 1.0 / tape_speed
                        temp_output = str(Path(voice_output).parent / f"temp_sped_{Path(voice_output).name}")
                        if slowdown_method == "stretch":
                            apply_time_stretch_sox(voice_output, temp_output, speedup_factor)
                        else:  # tape
                            apply_tape_speed_sox(voice_output, temp_output, speedup_factor)
                        # Replace voice_output with sped-up version
                        shutil.move(temp_output, voice_output)
                        print(f"   ✅ Track {track_idx + 1} restored to original speed")

                        # UPSAMPLE MODE: Refine the sped-up track
                        if upsample_mode:
                            print(f"   ✨ Upsampling track {track_idx + 1} with {upsample_steps} diffusion steps...")

                            # Extract conditioning from the sped-up audio (pass file path, not array)
                            extraction_u = extract_conditioning_from_audio(
                                voice_output,
                                instrument_group=subgroup,
                                extract_formats=extract_formats
                            )
                            piano_roll_u, amp_u, rframe_u, rbend_u, encodec_tokens_u = load_conditioning(extraction_u, consistent_window_slow)

                            # Get audio duration
                            track_audio_u, sr = torchaudio.load(voice_output)
                            track_duration = track_audio_u.shape[-1] / sr

                            # Generate with partial noise for upsampling
                            upsampled_output = generate(
                                model=MODEL,
                                piano_roll=piano_roll_u,
                                amp=amp_u,
                                rframe=rframe_u,
                                rbend=rbend_u,
                                encodec_tokens=encodec_tokens_u,
                                group=group,
                                subgroup=subgroup,
                                steps=upsample_steps,
                                seed=seed,
                                adapter_scale=adapter_scale,
                                cfg_weight=cfg_weight,
                                t0=1.0,
                                sr_out=44100,
                                instrument_strength=instrument_strength,
                                inst_boost=2.5,
                                piano_roll_gain=piano_roll_gain,
                                amp_gain=amp_gain,
                                rframe_gain=rframe_gain,
                                rbend_gain=rbend_gain,
                                encodec_gain=encodec_gain,
                                use_overlap_decoder=use_overlap_decoder,
                                original_audio_length=int(track_duration * 44100),
                                pitch_fidelity_boost=pitch_fidelity_boost,
                                onset_guidance_boost=onset_guidance_boost,
                                pitch_snap_strength=pitch_snap_strength,
                                noise_level=upsample_noise_level,
                                audio_file=voice_output,
                                target_audio_duration=track_duration,
                            )

                            # Load and save the upsampled result
                            upsampled_wav, _ = torchaudio.load(upsampled_output)
                            torchaudio.save(voice_output, upsampled_wav, 44100)
                            print(f"   ✅ Track {track_idx + 1} upsampled")

                    # Copy original voice to output directory
                    voice_output_path = output_dir / f"{output_track_counter}.wav"
                    shutil.copy(voice_output, str(voice_output_path))
                    voice_outputs.append(str(voice_output_path))

                    # Also copy to debug directory for comparison
                    debug_output = debug_dir / f"track{track_idx + 1}_output.wav"
                    shutil.copy(voice_output, str(debug_output))

                    download_url = f"/download/{process_id}/{output_track_counter}.wav"
                    completed_voices.append(download_url)
                    print(f"   ✅ Track {track_idx + 1} saved as output {output_track_counter}: {voice_output_path.name}")

                    # Copy input files progressively for this track
                    voice_num = output_track_counter
                    try:
                        # Copy MIDI input
                        if track_idx < len(track_midi_files) and track_midi_files[track_idx]:
                            midi_src = Path(track_midi_files[track_idx])
                            if midi_src.exists():
                                midi_dest = output_dir / f"{voice_num}_input.mid"
                                shutil.copy(str(midi_src), str(midi_dest))
                                if str(voice_num) not in input_files:
                                    input_files[str(voice_num)] = {}
                                input_files[str(voice_num)]["midi_path"] = f"/download/{process_id}/{voice_num}_input.mid"

                        # Copy FluidSynth render if it exists
                        if track_audio and Path(track_audio).exists():
                            audio_dest = output_dir / f"{voice_num}_input_render.wav"
                            shutil.copy(str(track_audio), str(audio_dest))
                            if str(voice_num) not in input_files:
                                input_files[str(voice_num)] = {}
                            input_files[str(voice_num)]["render_path"] = f"/download/{process_id}/{voice_num}_input_render.wav"
                            input_files[str(voice_num)]["type"] = "wav"
                        else:
                            if str(voice_num) not in input_files:
                                input_files[str(voice_num)] = {}
                            input_files[str(voice_num)]["type"] = "midi"
                    except Exception as e:
                        print(f"  ⚠️ Error copying input files for track {voice_num}: {e}")

                    output_track_counter += 1

                    # Fake fatten mode: create octave-up version by pitch shifting
                    if fatten_mode and fatten_type == "fake":
                        print(f"   🎚️ FAKE FATTEN MODE: Pitch shifting output up an octave...")
                        octave_output = str(Path(voice_output).parent / f"octave_{Path(voice_output).name}")
                        apply_pitch_shift_sox(voice_output, octave_output, 12)  # +12 semitones = +1 octave

                        # Copy octave version to output directory
                        octave_output_path = output_dir / f"{output_track_counter}.wav"
                        shutil.copy(octave_output, str(octave_output_path))
                        voice_outputs.append(str(octave_output_path))

                        # Also copy to debug directory
                        debug_octave = debug_dir / f"track{track_idx + 1}_octave.wav"
                        shutil.copy(octave_output, str(debug_octave))

                        octave_download_url = f"/download/{process_id}/{output_track_counter}.wav"
                        completed_voices.append(octave_download_url)
                        print(f"   ✅ Octave-up saved as output {output_track_counter}: {octave_output_path.name}")
                        output_track_counter += 1

                    # Update Celery task state with partial results for frontend
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'completed_voices': completed_voices.copy(),
                            'total_voices': total_expected_tracks,
                            'progress': len(completed_voices) / total_expected_tracks,
                            'input_files': input_files.copy()
                        }
                    )

                    print(f"   📊 Progress: {len(completed_voices)}/{total_expected_tracks} tracks complete")

                # Step 4: Create mix of all tracks
                print(f"\n🎚️ Step 4: Creating mix of {len(voice_outputs)} tracks...")
                mix_path = output_dir / "0.wav"
                mix_output = sum_audio_tracks(voice_outputs, str(mix_path), normalize=True)

                # Copy mix to debug directory
                debug_mix = debug_dir / "0_mix.wav"
                shutil.copy(str(mix_path), str(debug_mix))

                file_paths = [f"/download/{process_id}/0.wav"] + completed_voices

                # Print debug summary
                print(f"\n{'='*80}")
                print(f"✅ MULTI-TRACK MIDI GENERATION COMPLETE")
                print(f"{'='*80}")
                print(f"📁 Debug files saved to: {debug_dir}")
                print(f"\n📂 Debug folder contents:")
                print(f"   MIDI tracks:")
                for midi_file in track_midi_files:
                    print(f"      • {Path(midi_file).name}")
                print(f"\n   FluidSynth renders:")
                for audio_file in track_audio_files:
                    print(f"      • {Path(audio_file).name}")
                print(f"\n   AI-generated outputs:")
                for i in range(len(voice_outputs)):
                    print(f"      • track{i+1}_output.wav")
                print(f"\n   Final mix:")
                print(f"      • 0_mix.wav")
                # Copy input files to output directory and build input_files dict
                input_files = {}
                print(f"\n📦 Copying input files to output directory...")

                for i, (midi_file, audio_file) in enumerate(zip(track_midi_files, track_audio_files)):
                    voice_num = i + 1
                    try:
                        # Copy MIDI input file
                        midi_src = Path(midi_file)
                        if midi_src.exists():
                            midi_dest = output_dir / f"{voice_num}_input.mid"
                            shutil.copy(str(midi_src), str(midi_dest))
                            print(f"  ✅ Copied: {voice_num}_input.mid")

                            # Initialize entry for this voice
                            if str(voice_num) not in input_files:
                                input_files[str(voice_num)] = {}
                            input_files[str(voice_num)]["midi_path"] = f"/download/{process_id}/{voice_num}_input.mid"
                            input_files[str(voice_num)]["type"] = "midi"

                        # Copy FluidSynth render
                        audio_src = Path(audio_file)
                        if audio_src.exists():
                            audio_dest = output_dir / f"{voice_num}_input_render.wav"
                            shutil.copy(str(audio_src), str(audio_dest))
                            print(f"  ✅ Copied: {voice_num}_input_render.wav")

                            if str(voice_num) not in input_files:
                                input_files[str(voice_num)] = {}
                            input_files[str(voice_num)]["render_path"] = f"/download/{process_id}/{voice_num}_input_render.wav"

                    except Exception as e:
                        print(f"  ⚠️ Error copying input files for voice {voice_num}: {e}")

                print(f"✅ Input files copied and indexed")

                print(f"\n🎵 Returning to frontend:")
                print(f"   Total tracks: {len(file_paths)}")
                print(f"   Mix (track 0): {file_paths[0]}")
                print(f"   Individual tracks: {completed_voices}")
                print(f"   Input files: {len(input_files)} voices")
                for i, track in enumerate(file_paths):
                    track_type = "MIX (muted)" if i == 0 else f"Voice {i}"
                    print(f"      Track {i}: {track} [{track_type}]")
                print(f"{'='*80}\n")

                if tape_speed < 1.0:
                    print(f"✅ All tracks already restored to original speed (done per-track)")

                return {
                    "file_paths": file_paths,
                    "input_files": input_files,
                    "mainAudio": file_paths[0],
                    "voices": completed_voices  # Individual tracks without mix
                }

        # Apply speed slowdown if needed (before conditioning extraction)
        original_audio_path = audio_file_path
        is_midi_file_check = audio_file_path and audio_file_path.lower().endswith(('.mid', '.midi'))

        if tape_speed < 1.0:
            if is_midi_file_check:
                # For MIDI files: modify tempo instead of using sox
                print(f"\n{'='*80}")
                print(f"🎼 MIDI TEMPO SLOWDOWN: {tape_speed}x")
                print(f"{'='*80}")
                print(f"Modifying MIDI tempo for slower generation...")

                # Create slowed MIDI version
                slowed_path = str(Path(audio_file_path).parent / f"slowed_{Path(audio_file_path).name}")
                modify_midi_tempo(audio_file_path, slowed_path, tape_speed)

                # Use slowed MIDI for processing
                audio_file_path = slowed_path
                print(f"✅ Using slowed MIDI for processing: {Path(audio_file_path).name}")
                print(f"   (Will speed back up to {1.0/tape_speed:.2f}x after generation)")
                print(f"{'='*80}\n")
            else:
                # For audio files: use sox to slow down
                print(f"\n{'='*80}")
                print(f"🎞️ SPEED SLOWDOWN ({slowdown_method.upper()}): {tape_speed}x")
                print(f"{'='*80}")
                print(f"Slowing down input audio for better detail capture...")
                print(f"Method: {slowdown_method} ({'pitch changes' if slowdown_method == 'tape' else 'pitch preserved'})")

                # Create slowed-down version
                slowed_path = str(Path(audio_file_path).parent / f"slowed_{Path(audio_file_path).name}")

                # Apply appropriate slowdown method
                if slowdown_method == "stretch":
                    apply_time_stretch_sox(audio_file_path, slowed_path, tape_speed)
                else:  # tape
                    apply_tape_speed_sox(audio_file_path, slowed_path, tape_speed)

                # Use slowed version for processing
                audio_file_path = slowed_path
                print(f"✅ Using slowed audio for processing: {Path(audio_file_path).name}")
                print(f"   (Will speed back up to {1.0/tape_speed:.2f}x after generation)")
                print(f"{'='*80}\n")

        # Check if audio_file_path is actually a MIDI file (from fast mode no-input generation)
        is_midi_input = audio_file_path.lower().endswith(('.mid', '.midi'))

        print(f"\n🔍 MIDI INPUT CHECK:")
        print(f"   audio_file_path: {audio_file_path}")
        print(f"   is_midi_input: {is_midi_input}")
        print(f"   fast_mode_variant: {fast_mode_variant}")
        print(f"   Will use fast MIDI conversion: {is_midi_input and fast_mode_variant}")

        if is_midi_input and fast_mode_variant:
            # Fast mode with MIDI: convert MIDI directly to conditioning using fast mode function
            print(f"🎼 Fast mode: Converting MIDI directly to conditioning")
            import pretty_midi
            midi_data = pretty_midi.PrettyMIDI(audio_file_path)
            actual_duration = midi_data.get_end_time()
            print(f"🎵 MIDI duration: {actual_duration:.2f}s")

            # Calculate window_slow at T_slow resolution (43.066 fps)
            fps = 43.066
            window_slow = clamp_window_slow(int(actual_duration * fps), actual_duration, fps)
            print(f"🎵 Using window_slow = {window_slow} frames ({actual_duration:.2f}s at {fps} fps)")

            # Convert MIDI using fast mode function (same as monophonic mode uses)
            piano_roll, amp, rframe, rbend, encodec_tokens = generate_conditioning_from_midi_fast(
                audio_file_path,
                window_slow=window_slow,
                fps=fps,
                automation_points=global_automation if global_automation and len(global_automation) > 0 else None,
                duration=actual_duration,
                variant=fast_mode_variant
            )
            print(f"✅ MIDI converted to fast mode conditioning:")
            print(f"   piano_roll: {piano_roll.shape}")
            print(f"   amp: {amp.shape}")
            print(f"   rframe: {rframe.shape}")
            print(f"   rbend: {rbend.shape}")
            print(f"   encodec_tokens: {encodec_tokens.shape if encodec_tokens is not None else 'None'}")

            # Set orig_len to None for MIDI (no original audio length)
            orig_len = None

        elif is_midi_input and monophonic_mode:
            # UPLOADED MIDI FILE WITH MONOPHONIC MODE: Split into voices, render each, then process
            print(f"\n{'='*80}")
            print(f"🎼 UPLOADED MIDI FILE WITH MONOPHONIC MODE - VOICE SEPARATION")
            print(f"{'='*80}")
            print(f"   Input MIDI: {Path(audio_file_path).name}")
            print(f"   Will split into voices, render each with FluidSynth, then generate")
            print(f"{'='*80}\n")

            # Save concatenated_midi_path for later use
            concatenated_midi_path = audio_file_path

            # Create debug folder for voice renders
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            voice_debug_dir = Path("/home/arlo/Data/voice_debug") / timestamp
            voice_debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 DEBUG FILES SAVED TO: {voice_debug_dir}\n")

            # Copy uploaded MIDI to debug folder
            debug_concat_midi = voice_debug_dir / "uploaded_midi.mid"
            shutil.copy(audio_file_path, str(debug_concat_midi))
            print(f"   Copied uploaded MIDI: {debug_concat_midi.name}")

            # Split MIDI into voices using pretty_midi
            import pretty_midi
            pm = pretty_midi.PrettyMIDI(audio_file_path)
            non_empty_instruments = [inst for inst in pm.instruments if len(inst.notes) > 0]

            if len(non_empty_instruments) > 1:
                # Multi-track MIDI: use each track as a voice
                print(f"   Multi-track MIDI: {len(non_empty_instruments)} tracks")
                voice_midi_paths = []
                voice_audio_paths = []

                for i, inst in enumerate(non_empty_instruments):
                    # Create MIDI for this voice
                    voice_pm = pretty_midi.PrettyMIDI(resolution=pm.resolution)
                    voice_pm.instruments.append(inst)

                    # Save voice MIDI
                    voice_midi_path = voice_debug_dir / f"voice_{i+1}.mid"
                    voice_pm.write(str(voice_midi_path))
                    voice_midi_paths.append(str(voice_midi_path))
                    print(f"   💾 Voice {i+1}: {len(inst.notes)} notes → {voice_midi_path.name}")

                    # Render with FluidSynth
                    voice_audio_path = voice_debug_dir / f"voice_{i+1}_render.wav"
                    soundfont_path = INSTRUMENT_SOUNDFONTS.get(subgroup, INSTRUMENT_SOUNDFONTS.get("default"))
                    import subprocess
                    subprocess.run([
                        "fluidsynth", "-ni", "-g", "0.625", "-r", "44100", "-F", str(voice_audio_path),
                        soundfont_path, str(voice_midi_path)
                    ], check=True, capture_output=True)
                    voice_audio_paths.append(str(voice_audio_path))
                    print(f"      🎵 Rendered: {voice_audio_path.name}")

            else:
                # Single-track MIDI: split by note overlap
                print(f"   Single-track MIDI: splitting by note overlap...")
                instrument = non_empty_instruments[0]
                sorted_notes = sorted(instrument.notes, key=lambda n: n.start)

                # Voice separation: group overlapping notes
                voices = []
                for note in sorted_notes:
                    assigned = False
                    for voice_notes in voices:
                        if all(note.start >= existing.end for existing in voice_notes):
                            voice_notes.append(note)
                            assigned = True
                            break
                    if not assigned:
                        voices.append([note])

                print(f"   Split into {len(voices)} voices")

                voice_midi_paths = []
                voice_audio_paths = []

                for voice_idx, voice_notes in enumerate(voices):
                    # Create MIDI for this voice
                    voice_pm = pretty_midi.PrettyMIDI(resolution=pm.resolution)
                    voice_inst = pretty_midi.Instrument(program=0)
                    voice_inst.notes = voice_notes
                    voice_pm.instruments.append(voice_inst)

                    # Save voice MIDI
                    voice_midi_path = voice_debug_dir / f"voice_{voice_idx+1}.mid"
                    voice_pm.write(str(voice_midi_path))
                    voice_midi_paths.append(str(voice_midi_path))
                    print(f"   💾 Voice {voice_idx+1}: {len(voice_notes)} notes → {voice_midi_path.name}")

                    # Render with FluidSynth
                    voice_audio_path = voice_debug_dir / f"voice_{voice_idx+1}_render.wav"
                    soundfont_path = INSTRUMENT_SOUNDFONTS.get(subgroup, INSTRUMENT_SOUNDFONTS.get("default"))
                    import subprocess
                    subprocess.run([
                        "fluidsynth", "-ni", "-g", "0.625", "-r", "44100", "-F", str(voice_audio_path),
                        soundfont_path, str(voice_midi_path)
                    ], check=True, capture_output=True)
                    voice_audio_paths.append(str(voice_audio_path))
                    print(f"      🎵 Rendered: {voice_audio_path.name}")

            print(f"\n✅ Prepared {len(voice_audio_paths)} voices for monophonic processing")
            print(f"   Processing voices now...\n")

            # Setup output directory
            import torchaudio
            import torch
            process_id = str(uuid.uuid4())
            output_dir = ensure_path_exists(get_output_path('ace_step_output', process_id=process_id))

            # Process each voice
            completed_voices = []
            input_files = {}
            fps = 43.066

            # Calculate consistent duration
            pm_concat = pretty_midi.PrettyMIDI(concatenated_midi_path)
            consistent_duration = pm_concat.get_end_time()
            consistent_window_slow = clamp_window_slow(int(consistent_duration * fps), consistent_duration, fps)
            print(f"📏 Consistent duration: {consistent_duration:.2f}s ({consistent_window_slow} frames)\n")

            for voice_idx in range(len(voice_audio_paths)):
                print(f"🎼 Voice {voice_idx + 1}/{len(voice_audio_paths)}")
                voice_audio = voice_audio_paths[voice_idx]
                voice_midi = voice_midi_paths[voice_idx]

                # Extract conditioning
                extraction = extract_conditioning_from_audio(voice_audio, instrument_group=subgroup, extract_formats=extract_formats)
                piano_roll, amp, rframe, rbend, encodec_tokens = load_conditioning(extraction, consistent_window_slow)

                # Generate
                voice_seed = seed + (voice_idx * 1000)
                voice_output = generate(
                    model=MODEL, piano_roll=piano_roll, amp=amp, rframe=rframe, rbend=rbend,
                    encodec_tokens=encodec_tokens, group=group, subgroup=subgroup, steps=steps,
                    seed=voice_seed, adapter_scale=adapter_scale, cfg_weight=cfg_weight, t0=1.0,
                    sr_out=44100, instrument_strength=instrument_strength, noise_level=noise_level,
                    piano_roll_gain=piano_roll_gain, amp_gain=amp_gain, rframe_gain=rframe_gain,
                    rbend_gain=rbend_gain, encodec_gain=encodec_gain,
                    pitch_fidelity_boost=pitch_fidelity_boost, onset_guidance_boost=onset_guidance_boost,
                    pitch_snap_strength=pitch_snap_strength, use_time_varying_noise=use_time_varying_noise,
                    onset_preservation=onset_preservation, use_multiresolution_mixing=use_multiresolution_mixing,
                    use_onset_weighted_encodec=use_onset_weighted_encodec, encodec_onset_boost=encodec_onset_boost,
                    use_overlap_decoder=use_overlap_decoder,
                    audio_file=voice_audio,  # FluidSynth render for GT latent extraction
                    original_audio_length=int(consistent_duration * 44100),  # Audio length for latent extraction
                    target_audio_duration=consistent_duration  # Consistent duration across all voices
                )

                # Save (voice_output is a file path, need to load it first)
                voice_num = voice_idx + 1
                voice_path = output_dir / f"{voice_num}.wav"

                # Load the generated audio from the temp file
                wav, sr = torchaudio.load(voice_output)

                # Save to final location
                torchaudio.save(str(voice_path), wav, 44100)
                voice_download_path = f"/download/{process_id}/{voice_num}.wav"
                completed_voices.append(voice_download_path)
                print(f"   ✅ Saved voice {voice_num}: {voice_path.name}")

                # Copy inputs
                shutil.copy(voice_midi, str(output_dir / f"{voice_num}_input.mid"))
                shutil.copy(voice_audio, str(output_dir / f"{voice_num}_input.wav"))

                input_files[str(voice_num)] = {
                    "midi_path": f"/download/{process_id}/{voice_num}_input.mid",
                    "render_path": f"/download/{process_id}/{voice_num}_input.wav",
                    "type": "midi"
                }

                # Update task state for incremental UI updates
                self.update_state(state='PROGRESS', meta={
                    'status': 'generating',
                    'completed_voices': completed_voices,  # Pass the actual list, not length
                    'total_voices': len(voice_audio_paths),
                    'latest_file': voice_download_path,
                    'input_files': input_files  # Include input files for frontend
                })

            # Create mix
            all_voice_tensors = []
            for voice_path in completed_voices:
                full_path = str(output_dir / Path(voice_path).name)
                wav, sr = torchaudio.load(full_path)
                all_voice_tensors.append(wav)
            mixed = torch.stack(all_voice_tensors).sum(dim=0)
            mixed = mixed / max(mixed.abs().max().item(), 1.0)
            mix_path = output_dir / "0.wav"
            torchaudio.save(str(mix_path), mixed, 44100)
            mix_download_path = f"/download/{process_id}/0.wav"

            print(f"\n🎉 Complete! {len(completed_voices)} voices + mix")
            return {"file_paths": [mix_download_path] + completed_voices, "input_files": input_files}

        # Single-track processing continues below
        skip_to_monophonic = False

        if not skip_to_monophonic:
            # Regular audio file processing
            # RENDER AND EXTRACT MODE: Extract MIDI from audio, render with FluidSynth, then extract
            extracted_midi_path = None  # Track MIDI for later piano roll conversion
            if render_and_extract and not is_midi_input:
                print(f"\n{'='*80}")
                print(f"🎵 RENDER & EXTRACT MODE (Audio → MIDI → FluidSynth → Conditioning)")
                print(f"{'='*80}")
                print(f"   Input audio: {audio_file_path}")
                print(f"   Instrument: {subgroup}")

                # Extract MIDI from audio using Basic Pitch (no voice separation)
                print(f"   [1/3] Extracting MIDI with Basic Pitch...")
                print(f"   Mono mode: {render_extract_mono}")
                midi_result = save_basic_pitch_midi_with_voices(
                    audio_file_path,
                    subgroup=subgroup,
                    progress=None,
                    tempo=120.0,
                    separate_voices=False,
                    monophonic=render_extract_mono
                )
                extracted_midi_path = midi_result['main_midi']  # Save for later
                print(f"   ✅ Extracted MIDI: {extracted_midi_path}")

                # Render MIDI with FluidSynth
                print(f"   [2/3] Rendering MIDI with FluidSynth ({subgroup})...")
                rendered_audio = render_midi_to_audio(extracted_midi_path, instrument_group=subgroup)
                print(f"   ✅ Rendered audio: {rendered_audio}")

                # Replace audio_file_path with rendered audio for conditioning extraction
                audio_file_path = rendered_audio
                print(f"   [3/3] Will extract conditioning from rendered audio")
                print(f"   [NOTE] Piano roll will be derived from MIDI, not re-extracted")
                print(f"{'='*80}\n")

            # Get the original audio length for correct output timing
            # CRITICAL FIX: Determine actual duration from audio file, not parameter
            try:
                import torchaudio
                wav, sr = torchaudio.load(audio_file_path)
                orig_len = wav.shape[-1]
                actual_duration = orig_len / sr
                print(f"🎵 Original audio length: {orig_len} samples ({actual_duration:.2f}s at {sr}Hz)")
            except Exception as e:
                orig_len = None
                actual_duration = duration  # Fallback to parameter
                print(f"⚠️ Could not determine audio length: {e}, using parameter duration {duration}s")

            # Extract conditioning from audio file (either uploaded or generated)
            if fast_mode_variant:
                extraction = extract_conditioning_from_audio_fast_mode(
                    audio_file_path,
                    instrument_group=subgroup,  # Use subgroup as instrument hint
                    variant=fast_mode_variant
                )
            else:
                extraction = extract_conditioning_from_audio(
                    audio_file_path,
                    instrument_group=subgroup,  # Use subgroup as instrument hint
                    extract_formats=extract_formats
                )

            # Load conditioning data - use ACTUAL duration from audio file
            # Use 43.066 fps to match piano roll frame rate
            fps = 43.066
            window_slow = clamp_window_slow(int(actual_duration * fps), actual_duration, fps)
            print(f"🎵 Using window_slow = {window_slow} frames ({actual_duration:.2f}s at {fps} fps)")
            # Load conditioning
            if fast_mode_variant:
                piano_roll, amp, rframe, rbend, encodec_tokens = load_conditioning_fast_mode(extraction, window_slow, variant=fast_mode_variant)
            else:
                piano_roll, amp, rframe, rbend, encodec_tokens = load_conditioning(extraction, window_slow)

            # In render & extract mode, replace piano roll with MIDI-derived one (EXACT same as normal MIDI mode)
            if extracted_midi_path:
                print(f"\n🎵 RENDER & EXTRACT: Using MIDI-derived piano roll (same as normal MIDI mode)")
                print(f"   MIDI path: {extracted_midi_path}")

                # Load MIDI piano roll using midi_to_piano_roll_conditioning (EXACT same as MIDI mode)
                midi_piano_roll, _, _, _, _ = midi_to_piano_roll_conditioning(extracted_midi_path, window_slow, fps=43.066, tempo_override=None)
                print(f"   MIDI piano roll shape: {midi_piano_roll.shape}")

                # Resize MIDI piano roll to match conditioning length (EXACT same as MIDI mode)
                conditioning_length = amp.shape[-1]
                if midi_piano_roll.shape[1] != conditioning_length:
                    print(f"   Resizing MIDI piano roll: {midi_piano_roll.shape[1]} → {conditioning_length} frames")
                    if midi_piano_roll.shape[1] > conditioning_length:
                        midi_piano_roll = midi_piano_roll[:, :conditioning_length]  # truncate
                    else:
                        # pad with zeros
                        pad_amount = conditioning_length - midi_piano_roll.shape[1]
                        midi_piano_roll = np.pad(midi_piano_roll, ((0, 0), (0, pad_amount)), mode='constant', constant_values=0)

                # Use MIDI piano roll
                piano_roll = midi_piano_roll
                print(f"   ✅ Final piano roll: {piano_roll.shape}, amp: {amp.shape}")

            # Debug: Print conditioning shapes
            print(f"📊 Conditioning shapes after loading:")
            print(f"   piano_roll: {piano_roll.shape if piano_roll is not None else 'None'}")
            print(f"   amp: {amp.shape if amp is not None else 'None'}")
            print(f"   rframe: {rframe.shape if rframe is not None else 'None'}")
            print(f"   rbend: {rbend.shape if rbend is not None else 'None'}")
            print(f"   encodec_tokens: {encodec_tokens.shape if encodec_tokens is not None else 'None'}")

        # End of single-track processing - monophonic uploaded MIDI skips above and jumps here

        # ============================================================================
        # APPLY AUTOMATION TO AMP SIGNAL (runs for both single-track and monophonic)
        # ============================================================================
        if global_automation and len(global_automation) > 0:
            print(f"\n{'='*80}")
            print(f"🎛 APPLYING AUTOMATION TO AMP SIGNAL")
            print(f"{'='*80}")
            print(f"📊 Automation points received:")
            for t, v in global_automation[:5]:  # Show first 5 points
                print(f"   {t:.2f}s: {v:.3f}")
            if len(global_automation) > 5:
                print(f"   ... and {len(global_automation) - 5} more points")

            # Apply automation to amp conditioning signal
            amp = apply_automation_to_amp_signal(
                amp_signal=amp,
                automation_points=global_automation,
                total_duration=actual_duration,
                fps=fps
            )

            print(f"{'='*80}\n")

        # ============================================================================
        # INPAINTING MODE - Extract region with context
        # ============================================================================
        if inpaint_mode:
            try:
                print(f"\n{'='*80}")
                print(f"🎨 INPAINTING MODE - EXTRACTING REGION WITH CONTEXT")
                print(f"{'='*80}")

                # Validate inpaint times
                if inpaint_start_time < 0 or inpaint_end_time < 0:
                    raise ValueError(f"Invalid inpaint times: start={inpaint_start_time:.3f}s, end={inpaint_end_time:.3f}s. Both must be >= 0.")
                if inpaint_start_time >= inpaint_end_time:
                    raise ValueError(f"Invalid inpaint times: start={inpaint_start_time:.3f}s >= end={inpaint_end_time:.3f}s")

                # Calculate frame boundaries
                start_frame = int(inpaint_start_time * fps)
                end_frame = int(inpaint_end_time * fps)
                context_seconds = 2.0  # Use 2 seconds of context on each side
                context_frames = int(context_seconds * fps)

                print(f"📍 Inpaint region:")
                print(f"   Time: {inpaint_start_time:.2f}s - {inpaint_end_time:.2f}s")
                print(f"   Frames: {start_frame} - {end_frame}")
                print(f"   Duration: {inpaint_end_time - inpaint_start_time:.2f}s ({end_frame - start_frame} frames)")

                print(f"\n📍 Context windows:")
                print(f"   Context duration: {context_seconds}s ({context_frames} frames)")

                # Get extended region with context
                # For piano_roll, check if 2D to get the right dimension
                total_frames = piano_roll.shape[-1] if len(piano_roll.shape) > 1 else piano_roll.shape[0]
                context_start = max(0, start_frame - context_frames)
                context_end = min(total_frames, end_frame + context_frames)

                print(f"   Total available frames: {total_frames}")
                print(f"   Extended region: {context_start} - {context_end} frames")
                print(f"   Total frames to generate: {context_end - context_start}")

                # Store original full conditioning for reference
                full_piano_roll = piano_roll
                full_amp = amp
                full_rframe = rframe
                full_rbend = rbend
                full_encodec_tokens = encodec_tokens

                # Helper function to slice conditioning arrays handling both 1D and 2D
                def slice_conditioning(arr, start, end):
                    if arr is None:
                        return None
                    if len(arr.shape) == 1:
                        # 1D array - slice directly
                        return arr[start:end]
                    else:
                        # 2D or higher - slice last dimension
                        return arr[..., start:end]

                # Extract region with context for generation
                piano_roll = slice_conditioning(full_piano_roll, context_start, context_end)
                amp = slice_conditioning(full_amp, context_start, context_end)
                rframe = slice_conditioning(full_rframe, context_start, context_end)
                rbend = slice_conditioning(full_rbend, context_start, context_end)
                encodec_tokens = slice_conditioning(full_encodec_tokens, context_start, context_end)

                print(f"\n✅ Extracted conditioning with context:")
                print(f"   piano_roll: {piano_roll.shape if piano_roll is not None else 'None'}")
                print(f"   amp: {amp.shape if amp is not None else 'None'}")
                print(f"   rframe: {rframe.shape if rframe is not None else 'None'}")
                print(f"   rbend: {rbend.shape if rbend is not None else 'None'}")
                print(f"   encodec_tokens: {encodec_tokens.shape if encodec_tokens is not None else 'None'}")

                # Store inpainting metadata for later processing
                # IMPORTANT: Calculate context times directly from original times to avoid rounding errors
                # Don't use context_start/context_end frames - those have rounding errors!
                context_start_time = max(0.0, inpaint_start_time - context_seconds)
                context_end_time = min(actual_duration, inpaint_end_time + context_seconds)

                inpaint_metadata = {
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'context_start': context_start,
                    'context_end': context_end,
                    'context_frames': context_frames,
                    'fps': fps,
                    'original_audio_path': audio_file_path,
                    'sample_rate': sr,
                    # Store original time values for precise sample calculation
                    'start_time': inpaint_start_time,
                    'end_time': inpaint_end_time,
                    'context_start_time': context_start_time,
                    'context_end_time': context_end_time
                }

                print(f"{'='*80}\n")

            except Exception as e:
                print(f"❌ ERROR in inpainting extraction:")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Error message: {str(e)}")
                print(f"   Conditioning shapes:")
                print(f"     piano_roll: {piano_roll.shape if hasattr(piano_roll, 'shape') else type(piano_roll)}")
                print(f"     amp: {amp.shape if hasattr(amp, 'shape') else type(amp)}")
                print(f"     rframe: {rframe.shape if hasattr(rframe, 'shape') else type(rframe)}")
                print(f"     rbend: {rbend.shape if hasattr(rbend, 'shape') else type(rbend)}")
                print(f"     encodec_tokens: {encodec_tokens.shape if hasattr(encodec_tokens, 'shape') else type(encodec_tokens)}")
                raise RuntimeError(f"Inpainting extraction failed: {str(e)}") from e

        # Generate audio - use monophonic mode if enabled
        if monophonic_mode:
            print(f"🎵 Using monophonic mode - separating voices...")

            # Special handling for inpainting a specific voice
            if inpaint_mode and inpaint_voice_index is not None:
                print(f"🎨 INPAINT MODE: Regenerating voice {inpaint_voice_index} only")

                # Separate voices to identify which one to regenerate
                if enable_voice_separation:
                    print(f"   Separating piano roll into voices...")
                    voices = separate_piano_roll_voices(piano_roll)
                else:
                    print(f"   Using tracks as-is (multi-track MIDI)")
                    voices = [piano_roll]

                # Validate voice index
                if inpaint_voice_index < 1 or inpaint_voice_index > len(voices):
                    raise ValueError(f"Invalid voice index {inpaint_voice_index}. Track has {len(voices)} voices.")

                # Convert 1-indexed to 0-indexed
                voice_idx = inpaint_voice_index - 1
                voice_pr = voices[voice_idx]

                print(f"   🎼 Regenerating voice {inpaint_voice_index}/{len(voices)}")

                # Determine subgroup (arrange mode logic if enabled)
                if arrange_mode:
                    voice_range = analyze_voice_pitch_range(voice_pr)
                    min_note, max_note, mean_note = voice_range
                    voice_subgroup = assign_instrument_for_voice(group, voice_range, [])
                    print(f"      🎼 Arrange mode - Voice {inpaint_voice_index}:")
                    print(f"         Range: MIDI {min_note}-{max_note} (mean: {mean_note})")
                    print(f"         ✅ Assigned: {voice_subgroup}")
                else:
                    voice_subgroup = subgroup

                # Generate only this voice
                base_seed = int(seed) if seed > 0 else torch.seed() % 2**31
                voice_seed = base_seed + voice_idx * 1000

                # INPAINT MODE FIX: Calculate context window audio length (not full audio length!)
                context_duration = inpaint_metadata['context_end_time'] - inpaint_metadata['context_start_time']
                context_audio_length = int(context_duration * 44100)
                print(f"🎨 Context window duration: {context_duration:.3f}s = {context_audio_length} samples")
                print(f"   (Not using full audio length: {orig_len} samples)")

                output_audio = generate(
                    model=MODEL,
                    piano_roll=voice_pr,
                    amp=amp,
                    rframe=rframe,
                    rbend=rbend,
                    encodec_tokens=encodec_tokens,
                    group=group,
                    subgroup=voice_subgroup,
                    steps=steps,
                    seed=voice_seed,
                    adapter_scale=adapter_scale,
                    cfg_weight=cfg_weight,
                    t0=1.0,
                    sr_out=44100,
                    instrument_strength=instrument_strength,
                    inst_boost=2.5,
                    piano_roll_gain=piano_roll_gain,
                    amp_gain=amp_gain,
                    rframe_gain=rframe_gain,
                    rbend_gain=rbend_gain,
                    encodec_gain=encodec_gain,
                    use_overlap_decoder=use_overlap_decoder,
                    original_audio_length=context_audio_length,  # Use context window length, not full audio!
                    pitch_fidelity_boost=pitch_fidelity_boost,
                    onset_guidance_boost=onset_guidance_boost,
                    pitch_snap_strength=pitch_snap_strength,
                    noise_level=noise_level,
                    audio_file=audio_file_path,
                    fast_mode_variant=fast_mode_variant  # Pass fast_mode_variant for correct resolution handling
                )

                # Save output as 0.wav (inpaint post-processing expects this)
                output_path = output_dir / "0.wav"
                shutil.copy(output_audio, str(output_path))
                print(f"✅ Voice {inpaint_voice_index} inpainted and saved: {output_path}")

                # Now perform inpaint post-processing for this single voice
                # (same logic as below but needs to happen here for monophonic inpaint)
                print(f"\n{'='*80}")
                print(f"🎨 INPAINTING POST-PROCESSING (Voice {inpaint_voice_index})")
                print(f"{'='*80}")

                # Load the generated audio with context
                generated_wav, gen_sr = torchaudio.load(str(output_path))
                print(f"📥 Loaded generated audio: {generated_wav.shape} at {gen_sr}Hz")

                # Calculate sample positions
                samples_per_second = gen_sr

                # Actual inpaint region within the generated audio
                # Use original time values to avoid rounding errors when converting frames -> samples
                # The generated audio starts at context_start_time, so offset = start_time - context_start_time
                inpaint_offset_time = inpaint_metadata['start_time'] - inpaint_metadata['context_start_time']
                inpaint_duration_time = inpaint_metadata['end_time'] - inpaint_metadata['start_time']

                print(f"\n⏱️  Time-based calculation (PRECISE):")
                print(f"   Context start time: {inpaint_metadata['context_start_time']:.6f}s")
                print(f"   Inpaint start time: {inpaint_metadata['start_time']:.6f}s")
                print(f"   Inpaint end time: {inpaint_metadata['end_time']:.6f}s")
                print(f"   Offset within generated: {inpaint_offset_time:.6f}s")
                print(f"   Duration: {inpaint_duration_time:.6f}s")

                # Convert time directly to samples (no frame conversion = no rounding error)
                inpaint_start_in_gen = int(inpaint_offset_time * samples_per_second)
                inpaint_end_in_gen = int((inpaint_offset_time + inpaint_duration_time) * samples_per_second)

                print(f"   → Start sample: {inpaint_start_in_gen} ({inpaint_start_in_gen / samples_per_second:.6f}s)")
                print(f"   → End sample: {inpaint_end_in_gen} ({inpaint_end_in_gen / samples_per_second:.6f}s)")

                print(f"\n📍 Sample positions in generated audio:")
                print(f"   Total samples: {generated_wav.shape[-1]}")
                print(f"   Inpaint region: {inpaint_start_in_gen} - {inpaint_end_in_gen}")
                print(f"   Inpaint duration: {(inpaint_end_in_gen - inpaint_start_in_gen) / gen_sr:.2f}s")

                # Extract only the inpainted region (discard context-generated parts)
                inpainted_audio = generated_wav[..., inpaint_start_in_gen:inpaint_end_in_gen]
                print(f"✂️  Extracted inpainted region: {inpainted_audio.shape}")

                print(f"\n✅ Inpainted segment ready:")
                print(f"   Shape: {inpainted_audio.shape}")
                print(f"   Duration: {inpainted_audio.shape[-1] / gen_sr:.2f}s")
                print(f"   This segment will replace {inpaint_start_time:.2f}s - {inpaint_end_time:.2f}s in voice {inpaint_voice_index}")
                print(f"   Generated with {context_seconds}s context on each side for smooth blending")

                # Save only the inpainted segment (frontend will splice)
                torchaudio.save(str(output_path), inpainted_audio, gen_sr)
                print(f"💾 Saved inpainted segment: {output_path}")
                print(f"   Frontend will apply crossfades when splicing into original track")
                print(f"{'='*80}\n")

                # Apply speed restoration if needed
                if tape_speed < 1.0:
                    print(f"\n{'='*80}")
                    print(f"🎞️ SPEED RESTORATION ({slowdown_method.upper()}): {1.0/tape_speed:.2f}x")
                    print(f"{'='*80}")
                    print(f"Speeding up generated audio back to original tempo...")

                    speedup_factor = 1.0 / tape_speed
                    temp_path = str(output_path.parent / f"temp_{output_path.name}")
                    if slowdown_method == "stretch":
                        apply_time_stretch_sox(str(output_path), temp_path, speedup_factor)
                    else:  # tape
                        apply_tape_speed_sox(str(output_path), temp_path, speedup_factor)
                    shutil.move(temp_path, str(output_path))
                    print(f"✅ Output restored to original speed: {output_path.name}")
                    print(f"{'='*80}\n")

                    # UPSAMPLE MODE: Refine the sped-up inpainted audio
                    if upsample_mode:
                        print(f"\n{'='*80}")
                        print(f"✨ UPSAMPLE MODE: Refining inpainted audio with {upsample_steps} diffusion steps")
                        print(f"{'='*80}")
                        print(f"Noise level: {upsample_noise_level:.2f}")

                        # Extract latent from the sped-up audio
                        print(f"   🎯 Upsampling {output_path.name}...")
                        print(f"      1. Extracting latent from restored audio...")

                        # Get audio duration and calculate window_slow
                        inpaint_audio, sr = torchaudio.load(str(output_path))
                        inpaint_duration = inpaint_audio.shape[-1] / sr
                        fps = 43.066
                        window_slow_u = clamp_window_slow(int(inpaint_duration * fps), inpaint_duration, fps)

                        # Extract conditioning from the sped-up audio (pass file path, not array)
                        extraction_u = extract_conditioning_from_audio(
                            str(output_path),
                            instrument_group=instrument_subgroup
                        )
                        piano_roll_u, amp_u, rframe_u, rbend_u, encodec_tokens_u = load_conditioning(extraction_u, window_slow_u)

                        print(f"      2. Adding {upsample_noise_level:.2f} noise and running {upsample_steps} diffusion steps...")

                        # Generate with partial noise for upsampling
                        upsampled_output = generate(
                            model=MODEL,
                            piano_roll=piano_roll_u,
                            amp=amp_u,
                            rframe=rframe_u,
                            rbend=rbend_u,
                            encodec_tokens=encodec_tokens_u,
                            group=instrument_group,
                            subgroup=instrument_subgroup,
                            steps=upsample_steps,
                            seed=seed,
                            adapter_scale=adapter_scale,
                            cfg_weight=cfg_weight,
                            t0=1.0,
                            sr_out=44100,
                            instrument_strength=instrument_strength,
                            inst_boost=2.5,
                            piano_roll_gain=piano_roll_gain,
                            amp_gain=amp_gain,
                            rframe_gain=rframe_gain,
                            rbend_gain=rbend_gain,
                            encodec_gain=encodec_gain,
                            use_overlap_decoder=use_overlap_decoder,
                            original_audio_length=int(inpaint_duration * 44100),
                            pitch_fidelity_boost=pitch_fidelity_boost,
                            onset_guidance_boost=onset_guidance_boost,
                            pitch_snap_strength=pitch_snap_strength,
                            noise_level=upsample_noise_level,
                            audio_file=str(output_path),
                            target_audio_duration=inpaint_duration,
                        )

                        # Load and save the upsampled result
                        print(f"      3. Saving upsampled audio...")
                        upsampled_wav, _ = torchaudio.load(upsampled_output)
                        torchaudio.save(str(output_path), upsampled_wav, 44100)

                        print(f"\n✅ Inpainted audio upsampled and saved")
                        print(f"{'='*80}\n")

                logging.info(f"✅ ACE-Step voice {inpaint_voice_index} inpaint complete: {output_path}")

                # Copy input file to output directory and build input_files dict
                input_files = {}
                if audio_file_path and os.path.exists(audio_file_path):
                    print(f"\n📦 Copying input file to output directory...")
                    try:
                        input_src = Path(audio_file_path)
                        # Determine file type
                        if input_src.suffix.lower() in ['.mid', '.midi']:
                            input_dest = output_dir / "0_input.mid"
                            shutil.copy(str(input_src), str(input_dest))
                            print(f"  ✅ Copied: 0_input.mid")
                            input_files["0"] = {
                                "midi_path": f"/download/{process_id}/0_input.mid",
                                "type": "midi"
                            }
                        else:  # WAV file
                            input_dest = output_dir / "0_input.wav"
                            shutil.copy(str(input_src), str(input_dest))
                            print(f"  ✅ Copied: 0_input.wav")
                            input_files["0"] = {
                                "render_path": f"/download/{process_id}/0_input.wav",
                                "type": "wav"
                            }

                            # Also copy the basic pitch MIDI extracted from the audio
                            # Check both extracted_midi_path (render_and_extract mode) and extraction dict
                            midi_copied = False

                            # First try extracted_midi_path (from render_and_extract mode)
                            if 'extracted_midi_path' in locals() and extracted_midi_path and os.path.exists(extracted_midi_path):
                                midi_source = Path(extracted_midi_path)
                                midi_dest = output_dir / "0_input.mid"
                                shutil.copy(str(midi_source), str(midi_dest))
                                print(f"  ✅ Copied: 0_input.mid (from render_and_extract mode)")
                                input_files["0"]["midi_path"] = f"/download/{process_id}/0_input.mid"
                                midi_copied = True

                            # Otherwise try extraction dict (normal mode)
                            if not midi_copied and 'extraction' in locals() and extraction and isinstance(extraction, dict):
                                if 'dir' in extraction and 'stem' in extraction:
                                    basicpitch_midi = Path(extraction['dir']) / f"{extraction['stem']}.mid"
                                    if basicpitch_midi.exists():
                                        # Copy as main MIDI input (for frontend MIDI display)
                                        midi_dest = output_dir / "0_input.mid"
                                        shutil.copy(str(basicpitch_midi), str(midi_dest))
                                        print(f"  ✅ Copied: 0_input.mid (MIDI for display)")
                                        input_files["0"]["midi_path"] = f"/download/{process_id}/0_input.mid"

                                        # Also keep basicpitch copy for reference
                                        basicpitch_dest = output_dir / "0_input_basicpitch.mid"
                                        shutil.copy(str(basicpitch_midi), str(basicpitch_dest))
                                        print(f"  ✅ Copied: 0_input_basicpitch.mid (piano roll source)")
                                        input_files["0"]["basicpitch_midi_path"] = f"/download/{process_id}/0_input_basicpitch.mid"
                    except Exception as e:
                        print(f"  ⚠️ Error copying input file: {e}")

                return {"file_paths": [f"/download/{process_id}/0.wav"], "input_files": input_files}

            else:
                # Normal monophonic mode - generate all voices
                # Track completed voices for incremental updates
                completed_voices = []
                input_files = {}  # Track input files for each voice

                def voice_complete_callback(voice_idx, voice_path, total_voices):
                    """Called when each voice completes generation"""
                    # Copy voice to output directory with sequential numbering
                    voice_output_path = output_dir / f"{voice_idx + 1}.wav"
                    shutil.copy(voice_path, str(voice_output_path))

                    # Add to completed list
                    download_url = f"/download/{process_id}/{voice_idx + 1}.wav"
                    completed_voices.append(download_url)

                    # Copy input files progressively for this voice
                    voice_num = voice_idx + 1
                    try:
                        # Initialize entry for this voice
                        if str(voice_num) not in input_files:
                            input_files[str(voice_num)] = {}

                        # Copy input file based on type
                        if audio_file_path and os.path.exists(audio_file_path):
                            input_src = Path(audio_file_path)

                            if input_src.suffix.lower() in ['.mid', '.midi']:
                                # MIDI input
                                input_dest = output_dir / f"{voice_num}_input.mid"
                                shutil.copy(str(input_src), str(input_dest))
                                input_files[str(voice_num)]["midi_path"] = f"/download/{process_id}/{voice_num}_input.mid"
                                input_files[str(voice_num)]["type"] = "midi"
                            else:
                                # Audio input
                                input_dest = output_dir / f"{voice_num}_input.wav"
                                shutil.copy(str(input_src), str(input_dest))
                                input_files[str(voice_num)]["render_path"] = f"/download/{process_id}/{voice_num}_input.wav"
                                input_files[str(voice_num)]["type"] = "wav"

                                # Also copy BasicPitch MIDI if available
                                if 'extraction' in locals() and extraction and isinstance(extraction, dict):
                                    if 'dir' in extraction and 'stem' in extraction:
                                        basicpitch_midi = Path(extraction['dir']) / f"{extraction['stem']}.mid"
                                        if basicpitch_midi.exists():
                                            # Copy as main MIDI input (for frontend MIDI display)
                                            midi_dest = output_dir / f"{voice_num}_input.mid"
                                            shutil.copy(str(basicpitch_midi), str(midi_dest))
                                            input_files[str(voice_num)]["midi_path"] = f"/download/{process_id}/{voice_num}_input.mid"

                                            # Also keep basicpitch copy for reference
                                            basicpitch_dest = output_dir / f"{voice_num}_input_basicpitch.mid"
                                            shutil.copy(str(basicpitch_midi), str(basicpitch_dest))
                                            input_files[str(voice_num)]["basicpitch_midi_path"] = f"/download/{process_id}/{voice_num}_input_basicpitch.mid"

                    except Exception as e:
                        print(f"  ⚠️ Error copying input files for voice {voice_num}: {e}")

                    # Update Celery task state with partial results
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'completed_voices': completed_voices.copy(),
                            'total_voices': total_voices,
                            'progress': len(completed_voices) / total_voices,
                            'input_files': input_files.copy()
                        }
                    )
                    print(f"📊 Progress update: {len(completed_voices)}/{total_voices} voices completed")

                result = generate_monophonic_multiple(
                model=MODEL,
                piano_roll=piano_roll,
                amp=amp,
                rframe=rframe,
                rbend=rbend,
                encodec_tokens=encodec_tokens,
                group=group,
                subgroup=subgroup,
                steps=steps,
                seed=seed,
                adapter_scale=adapter_scale,
                cfg_weight=cfg_weight,
                t0=1.0,
                sr_out=44100,
                instrument_strength=instrument_strength,
                inst_boost=2.5,
                piano_roll_gain=piano_roll_gain,
                amp_gain=amp_gain,
                rframe_gain=rframe_gain,
                rbend_gain=rbend_gain,
                encodec_gain=encodec_gain,
                use_overlap_decoder=use_overlap_decoder,
                original_audio_length=orig_len,
                pitch_fidelity_boost=pitch_fidelity_boost,
                onset_guidance_boost=onset_guidance_boost,
                pitch_snap_strength=pitch_snap_strength,
                noise_level=noise_level,
                audio_file=audio_file_path,
                progress=None,
                voice_complete_callback=voice_complete_callback,
                enable_voice_separation=enable_voice_separation,
                arrange_mode=arrange_mode,
                fast_mode_variant=fast_mode_variant,
                fatten_mode=fatten_mode,
                fatten_type=fatten_type,
                inpaint_voice_index=inpaint_voice_index
            )

            # Handle monophonic result - individual voices already copied by callback
            # Just need to copy the mixed output as 0.wav
            file_paths = [f"/download/{process_id}/0.wav"] + completed_voices

            if isinstance(result, dict):
                # Copy mixed output as 0.wav
                mixed_path = result.get("mixed")
                if mixed_path and os.path.exists(mixed_path):
                    output_path = output_dir / "0.wav"
                    shutil.copy(mixed_path, str(output_path))
                    print(f"✅ Mixed output saved: {output_path}")
                else:
                    print(f"⚠️ No mixed output found in result")

                logging.info(f"✅ ACE-Step monophonic generation complete: {len(file_paths)} files")
            else:
                # Fallback - single output
                output_path = output_dir / "0.wav"
                shutil.copy(result, str(output_path))
                logging.info(f"✅ ACE-Step generation complete (single voice): {output_path}")

            # Apply speed up if we slowed down the input
            if tape_speed < 1.0:
                print(f"\n{'='*80}")
                print(f"🎞️ SPEED RESTORATION ({slowdown_method.upper()}): {1.0/tape_speed:.2f}x")
                print(f"{'='*80}")
                print(f"Speeding up generated audio back to original tempo...")

                # Speed up all output files
                speedup_factor = 1.0 / tape_speed
                for wav_file in output_dir.glob("*.wav"):
                    temp_path = str(wav_file.parent / f"temp_{wav_file.name}")
                    if slowdown_method == "stretch":
                        apply_time_stretch_sox(str(wav_file), temp_path, speedup_factor)
                    else:  # tape
                        apply_tape_speed_sox(str(wav_file), temp_path, speedup_factor)
                    shutil.move(temp_path, str(wav_file))
                    print(f"✅ Restored speed: {wav_file.name}")

                print(f"✅ All outputs restored to original speed")
                print(f"{'='*80}\n")

                # UPSAMPLE MODE: Refine the sped-up audio with additional diffusion steps
                if upsample_mode:
                    print(f"\n{'='*80}")
                    print(f"✨ UPSAMPLE MODE: Refining audio with {upsample_steps} diffusion steps")
                    print(f"{'='*80}")
                    print(f"Noise level: {upsample_noise_level:.2f}")

                    for wav_file in output_dir.glob("*.wav"):
                        print(f"\n   🎯 Upsampling {wav_file.name}...")

                        # Extract latent from the sped-up audio
                        print(f"      1. Extracting latent from restored audio...")

                        # Get audio duration and calculate window_slow
                        file_audio, sr = torchaudio.load(str(wav_file))
                        audio_duration = file_audio.shape[-1] / sr
                        fps = 43.066
                        window_slow_u = clamp_window_slow(int(audio_duration * fps), audio_duration, fps)

                        # Extract conditioning from the sped-up audio (pass file path, not array)
                        extraction = extract_conditioning_from_audio(
                            str(wav_file),
                            instrument_group=instrument_subgroup
                        )
                        piano_roll_u, amp_u, rframe_u, rbend_u, encodec_tokens_u = load_conditioning(extraction, window_slow_u)

                        print(f"      2. Adding {upsample_noise_level:.2f} noise and running {upsample_steps} diffusion steps...")

                        # Generate with partial noise for upsampling
                        upsampled_output = generate(
                            model=MODEL,
                            piano_roll=piano_roll_u,
                            amp=amp_u,
                            rframe=rframe_u,
                            rbend=rbend_u,
                            encodec_tokens=encodec_tokens_u,
                            group=instrument_group,
                            subgroup=instrument_subgroup,
                            steps=upsample_steps,
                            seed=seed,
                            adapter_scale=adapter_scale,
                            cfg_weight=cfg_weight,
                            t0=1.0,
                            sr_out=44100,
                            instrument_strength=instrument_strength,
                            inst_boost=2.5,
                            piano_roll_gain=piano_roll_gain,
                            amp_gain=amp_gain,
                            rframe_gain=rframe_gain,
                            rbend_gain=rbend_gain,
                            encodec_gain=encodec_gain,
                            use_overlap_decoder=use_overlap_decoder,
                            original_audio_length=int(audio_duration * 44100),
                            pitch_fidelity_boost=pitch_fidelity_boost,
                            onset_guidance_boost=onset_guidance_boost,
                            pitch_snap_strength=pitch_snap_strength,
                            noise_level=upsample_noise_level,
                            audio_file=str(wav_file),
                            target_audio_duration=audio_duration,
                        )

                        # Load and save the upsampled result
                        print(f"      3. Saving upsampled audio...")
                        upsampled_wav, upsample_sr = torchaudio.load(upsampled_output)
                        torchaudio.save(str(wav_file), upsampled_wav, 44100)

                        print(f"   ✅ {wav_file.name} upsampled and saved")

                    print(f"\n✅ All files upsampled successfully")
                    print(f"{'='*80}\n")

            # Input files already copied progressively during voice processing
            # (input_files dict was built incrementally as each voice completed)

                print(f"✅ Input files copied and indexed")

            # Also copy voice 0 (mixed output) input if available
            if 'audio_file_path' in locals() and audio_file_path and os.path.exists(audio_file_path):
                try:
                    input_src = Path(audio_file_path)
                    # Only add voice 0 input if it's different from the voice inputs
                    if input_src.suffix.lower() in ['.mid', '.midi']:
                        input_dest = output_dir / "0_input.mid"
                        shutil.copy(str(input_src), str(input_dest))
                        print(f"  ✅ Copied: 0_input.mid (master)")
                        input_files["0"] = {
                            "midi_path": f"/download/{process_id}/0_input.mid",
                            "type": "midi"
                        }
                    elif input_src.suffix.lower() == '.wav':
                        input_dest = output_dir / "0_input.wav"
                        shutil.copy(str(input_src), str(input_dest))
                        print(f"  ✅ Copied: 0_input.wav (master)")
                        input_files["0"] = {
                            "render_path": f"/download/{process_id}/0_input.wav",
                            "type": "wav"
                        }

                        # Also copy the basic pitch MIDI extracted from the audio
                        if 'extraction' in locals() and extraction and isinstance(extraction, dict):
                            if 'dir' in extraction and 'stem' in extraction:
                                basicpitch_midi = Path(extraction['dir']) / f"{extraction['stem']}.mid"
                                if basicpitch_midi.exists():
                                    # Copy as main MIDI input (for frontend MIDI display)
                                    midi_dest = output_dir / "0_input.mid"
                                    shutil.copy(str(basicpitch_midi), str(midi_dest))
                                    print(f"  ✅ Copied: 0_input.mid (MIDI for display)")
                                    input_files["0"]["midi_path"] = f"/download/{process_id}/0_input.mid"

                                    # Also keep basicpitch copy for reference
                                    basicpitch_dest = output_dir / "0_input_basicpitch.mid"
                                    shutil.copy(str(basicpitch_midi), str(basicpitch_dest))
                                    print(f"  ✅ Copied: 0_input_basicpitch.mid (piano roll source)")
                                    input_files["0"]["basicpitch_midi_path"] = f"/download/{process_id}/0_input_basicpitch.mid"
                except Exception as e:
                    print(f"  ⚠️ Error copying master input file: {e}")

            return {"file_paths": file_paths, "input_files": input_files}

        else:
            # Regular single-voice generation

            # INPAINT MODE FIX: Calculate context window audio length (not full audio length!)
            if inpaint_mode and 'inpaint_metadata' in locals():
                context_duration = inpaint_metadata['context_end_time'] - inpaint_metadata['context_start_time']
                context_audio_length = int(context_duration * 44100)
                print(f"🎨 Context window duration: {context_duration:.3f}s = {context_audio_length} samples")
                print(f"   (Not using full audio length: {orig_len} samples)")
                audio_length_for_generation = context_audio_length
            else:
                audio_length_for_generation = orig_len

            if use_best_of_n and audio_file_path:
                # Best-of-N sampling with reranking
                output_audio, all_candidates = generate_best_of_n(
                    model=MODEL,
                    audio_file=audio_file_path,
                    piano_roll=piano_roll,
                    amp=amp,
                    rframe=rframe,
                    rbend=rbend,
                    encodec_tokens=encodec_tokens,
                    group=group,
                    subgroup=subgroup,
                    base_seed=seed,
                    n_candidates=n_candidates,
                    # Generation parameters
                    steps=steps,
                    adapter_scale=adapter_scale,
                    cfg_weight=cfg_weight,
                    t0=1.0,
                    sr_out=44100,
                    instrument_strength=instrument_strength,
                    inst_boost=2.5,
                    piano_roll_gain=piano_roll_gain,
                    amp_gain=amp_gain,
                    rframe_gain=rframe_gain,
                    rbend_gain=rbend_gain,
                    encodec_gain=encodec_gain,
                    use_overlap_decoder=use_overlap_decoder,
                    original_audio_length=audio_length_for_generation,  # Use context window in inpaint mode!
                    pitch_fidelity_boost=pitch_fidelity_boost,
                    onset_guidance_boost=onset_guidance_boost,
                    pitch_snap_strength=pitch_snap_strength,
                    noise_level=noise_level,
                    fast_mode_variant=fast_mode_variant,
                    # Test-time features disabled for Best-of-N (already doing parameter sweep)
                    use_test_time_adaptation=False,
                    use_self_consistency=use_self_consistency,
                    use_time_varying_noise=use_time_varying_noise,
                    onset_preservation=onset_preservation,
                    use_multiresolution_mixing=use_multiresolution_mixing,
                    use_onset_weighted_encodec=use_onset_weighted_encodec,
                    encodec_onset_boost=encodec_onset_boost
                )
            else:
                # Regular generation
                output_audio = generate(
                    model=MODEL,
                    piano_roll=piano_roll,
                    amp=amp,
                    rframe=rframe,
                    rbend=rbend,
                    encodec_tokens=encodec_tokens,
                    group=group,
                    subgroup=subgroup,
                    steps=steps,
                    seed=seed,
                    adapter_scale=adapter_scale,
                    cfg_weight=cfg_weight,
                    t0=1.0,
                    sr_out=44100,
                    instrument_strength=instrument_strength,
                    inst_boost=2.5,
                    piano_roll_gain=piano_roll_gain,
                    amp_gain=amp_gain,
                    rframe_gain=rframe_gain,
                    rbend_gain=rbend_gain,
                    encodec_gain=encodec_gain,
                    use_overlap_decoder=use_overlap_decoder,
                    original_audio_length=audio_length_for_generation,  # Use context window in inpaint mode!
                    pitch_fidelity_boost=pitch_fidelity_boost,
                    onset_guidance_boost=onset_guidance_boost,
                    pitch_snap_strength=pitch_snap_strength,
                    noise_level=noise_level,
                    audio_file=audio_file_path,
                    fast_mode_variant=fast_mode_variant,
                    # Test-time enhancement parameters
                    use_test_time_adaptation=use_test_time_adaptation,
                    adaptation_steps=adaptation_steps,
                    adaptation_learning_rate=adaptation_learning_rate,
                    use_self_consistency=use_self_consistency,
                    consistency_samples=consistency_samples,
                    consistency_noise_scale=consistency_noise_scale,
                    use_time_varying_noise=use_time_varying_noise,
                    onset_preservation=onset_preservation,
                    use_multiresolution_mixing=use_multiresolution_mixing,
                    use_onset_weighted_encodec=use_onset_weighted_encodec,
                    encodec_onset_boost=encodec_onset_boost,
                    # Render & extract mode flag
                    render_and_extract=render_and_extract
                )

            # Save output
            output_path = output_dir / "0.wav"
            # output_audio is actually a file path returned by generate(), not audio data
            # Copy the generated file to the final output location
            shutil.copy(output_audio, str(output_path))

            # ============================================================================
            # INPAINTING POST-PROCESSING - Extract region and apply crossfading
            # ============================================================================
            if inpaint_mode:
                print(f"\n{'='*80}")
                print(f"🎨 INPAINTING POST-PROCESSING")
                print(f"{'='*80}")

                # Load the generated audio with context
                generated_wav, gen_sr = torchaudio.load(str(output_path))
                print(f"📥 Loaded generated audio: {generated_wav.shape} at {gen_sr}Hz")

                # Calculate sample positions
                samples_per_second = gen_sr

                # Context start/end in samples (for the generated audio)
                context_start_samples = int(max(0, inpaint_metadata['context_start'] - inpaint_metadata['context_start']) * samples_per_second)
                context_end_samples = generated_wav.shape[-1]

                # Actual inpaint region within the generated audio
                # Use original time values to avoid rounding errors when converting frames -> samples
                # The generated audio starts at context_start_time, so offset = start_time - context_start_time
                inpaint_offset_time = inpaint_metadata['start_time'] - inpaint_metadata['context_start_time']
                inpaint_duration_time = inpaint_metadata['end_time'] - inpaint_metadata['start_time']

                print(f"\n⏱️  Time-based calculation (PRECISE):")
                print(f"   Context start time: {inpaint_metadata['context_start_time']:.6f}s")
                print(f"   Inpaint start time: {inpaint_metadata['start_time']:.6f}s")
                print(f"   Inpaint end time: {inpaint_metadata['end_time']:.6f}s")
                print(f"   Offset within generated: {inpaint_offset_time:.6f}s")
                print(f"   Duration: {inpaint_duration_time:.6f}s")

                # Convert time directly to samples (no frame conversion = no rounding error)
                inpaint_start_in_gen = int(inpaint_offset_time * samples_per_second)
                inpaint_end_in_gen = int((inpaint_offset_time + inpaint_duration_time) * samples_per_second)

                print(f"   → Start sample: {inpaint_start_in_gen} ({inpaint_start_in_gen / samples_per_second:.6f}s)")
                print(f"   → End sample: {inpaint_end_in_gen} ({inpaint_end_in_gen / samples_per_second:.6f}s)")

                print(f"\n📍 Sample positions in generated audio:")
                print(f"   Total samples: {generated_wav.shape[-1]}")
                print(f"   Inpaint region: {inpaint_start_in_gen} - {inpaint_end_in_gen}")
                print(f"   Inpaint duration: {(inpaint_end_in_gen - inpaint_start_in_gen) / gen_sr:.2f}s")

                # Extract only the inpainted region (discard context-generated parts)
                inpainted_audio = generated_wav[..., inpaint_start_in_gen:inpaint_end_in_gen]
                print(f"✂️  Extracted inpainted region: {inpainted_audio.shape}")

                # Load original audio for crossfading
                original_wav, orig_sr = torchaudio.load(inpaint_metadata['original_audio_path'])
                print(f"📥 Loaded original audio: {original_wav.shape} at {orig_sr}Hz")

                # Calculate sample positions in original audio
                orig_start_sample = int(inpaint_start_time * orig_sr)
                orig_end_sample = int(inpaint_end_time * orig_sr)

                # Return ONLY the generated inpainted segment
                # The frontend will handle splicing it with the original audio and applying crossfades
                # Since we generated with context, the model already has harmonic/rhythmic awareness

                print(f"\n✅ Inpainted segment ready:")
                print(f"   Shape: {inpainted_audio.shape}")
                print(f"   Duration: {inpainted_audio.shape[-1] / gen_sr:.2f}s")
                print(f"   This segment will replace {inpaint_start_time:.2f}s - {inpaint_end_time:.2f}s in the original")
                print(f"   Generated with {context_seconds}s context on each side for smooth blending")

                # Save only the inpainted segment (frontend will splice)
                torchaudio.save(str(output_path), inpainted_audio, gen_sr)
                print(f"💾 Saved inpainted segment: {output_path}")
                print(f"   Frontend will apply crossfades when splicing into original track")

                print(f"{'='*80}\n")

            # Apply speed up if we slowed down the input
            if tape_speed < 1.0:
                print(f"\n{'='*80}")
                print(f"🎞️ SPEED RESTORATION ({slowdown_method.upper()}): {1.0/tape_speed:.2f}x")
                print(f"{'='*80}")
                print(f"Speeding up generated audio back to original tempo...")

                speedup_factor = 1.0 / tape_speed
                temp_path = str(output_path.parent / f"temp_{output_path.name}")
                if slowdown_method == "stretch":
                    apply_time_stretch_sox(str(output_path), temp_path, speedup_factor)
                else:  # tape
                    apply_tape_speed_sox(str(output_path), temp_path, speedup_factor)
                shutil.move(temp_path, str(output_path))

                print(f"✅ Output restored to original speed: {output_path.name}")
                print(f"{'='*80}\n")

                # UPSAMPLE MODE: Refine the sped-up audio with additional diffusion steps
                if upsample_mode:
                    print(f"\n{'='*80}")
                    print(f"✨ UPSAMPLE MODE: Refining audio with {upsample_steps} diffusion steps")
                    print(f"{'='*80}")
                    print(f"Noise level: {upsample_noise_level:.2f}")
                    print(f"Processing {output_path.name}...")

                    # Extract latent from the sped-up audio
                    print(f"\n   🎯 Upsampling {output_path.name}...")
                    print(f"      1. Extracting latent from restored audio...")

                    # Get audio duration and calculate window_slow
                    file_audio, sr = torchaudio.load(str(output_path))
                    audio_duration = file_audio.shape[-1] / sr
                    fps = 43.066
                    window_slow = clamp_window_slow(int(audio_duration * fps), audio_duration, fps)

                    # Extract conditioning from the sped-up audio (pass file path, not array)
                    extraction = extract_conditioning_from_audio(
                        str(output_path),
                        instrument_group=instrument_subgroup
                    )
                    piano_roll_u, amp_u, rframe_u, rbend_u, encodec_tokens_u = load_conditioning(extraction, window_slow)

                    print(f"      2. Adding {upsample_noise_level:.2f} noise and running {upsample_steps} diffusion steps...")

                    # Generate with partial noise for upsampling
                    upsampled_output = generate(
                        model=MODEL,
                        piano_roll=piano_roll_u,
                        amp=amp_u,
                        rframe=rframe_u,
                        rbend=rbend_u,
                        encodec_tokens=encodec_tokens_u,
                        group=group,
                        subgroup=subgroup,
                        steps=upsample_steps,  # Use upsample steps
                        seed=seed,
                        adapter_scale=adapter_scale,
                        cfg_weight=cfg_weight,
                        t0=1.0,
                        sr_out=44100,
                        instrument_strength=instrument_strength,
                        inst_boost=2.5,
                        piano_roll_gain=piano_roll_gain,
                        amp_gain=amp_gain,
                        rframe_gain=rframe_gain,
                        rbend_gain=rbend_gain,
                        encodec_gain=encodec_gain,
                        use_overlap_decoder=use_overlap_decoder,
                        original_audio_length=int(audio_duration * 44100),
                        pitch_fidelity_boost=pitch_fidelity_boost,
                        onset_guidance_boost=onset_guidance_boost,
                        pitch_snap_strength=pitch_snap_strength,
                        noise_level=upsample_noise_level,  # Use upsample noise level
                        audio_file=str(output_path),  # Use the sped-up audio for GT latents
                        target_audio_duration=audio_duration,
                    )

                    # Load and save the upsampled result
                    print(f"      3. Saving upsampled audio...")
                    upsampled_wav, upsample_sr = torchaudio.load(upsampled_output)
                    torchaudio.save(str(output_path), upsampled_wav, 44100)

                    print(f"\n✅ {output_path.name} upsampled and saved")
                    print(f"{'='*80}\n")

            logging.info(f"✅ ACE-Step generation complete: {output_path}")

            # Copy input file to output directory and build input_files dict
            input_files = {}
            if 'audio_file_path' in locals() and audio_file_path and os.path.exists(audio_file_path):
                print(f"\n📦 Copying input file to output directory...")
                try:
                    input_src = Path(audio_file_path)
                    # Determine file type
                    if input_src.suffix.lower() in ['.mid', '.midi']:
                        input_dest = output_dir / "0_input.mid"
                        shutil.copy(str(input_src), str(input_dest))
                        print(f"  ✅ Copied: 0_input.mid")
                        input_files["0"] = {
                            "midi_path": f"/download/{process_id}/0_input.mid",
                            "type": "midi"
                        }
                    else:  # WAV file
                        input_dest = output_dir / "0_input.wav"
                        shutil.copy(str(input_src), str(input_dest))
                        print(f"  ✅ Copied: 0_input.wav")
                        input_files["0"] = {
                            "render_path": f"/download/{process_id}/0_input.wav",
                            "type": "wav"
                        }

                        # Also copy the basic pitch MIDI extracted from the audio
                        # First check if extraction dict has MIDI (from render_and_extract or other paths)
                        midi_found = False

                        if 'extraction' in locals() and extraction and isinstance(extraction, dict):
                            if 'dir' in extraction and 'stem' in extraction:
                                # Try nested directory first (extraction script creates stem/stem/ structure)
                                basicpitch_midi = Path(extraction['dir']) / extraction['stem'] / f"{extraction['stem']}.mid"
                                if not basicpitch_midi.exists():
                                    # Fallback to flat structure
                                    basicpitch_midi = Path(extraction['dir']) / f"{extraction['stem']}.mid"

                                if basicpitch_midi.exists():
                                    # Copy as main MIDI input (for frontend MIDI display)
                                    midi_dest = output_dir / "0_input.mid"
                                    shutil.copy(str(basicpitch_midi), str(midi_dest))
                                    print(f"  ✅ Copied: 0_input.mid (MIDI for display)")
                                    input_files["0"]["midi_path"] = f"/download/{process_id}/0_input.mid"

                                    # Also keep basicpitch copy for reference
                                    basicpitch_dest = output_dir / "0_input_basicpitch.mid"
                                    shutil.copy(str(basicpitch_midi), str(basicpitch_dest))
                                    print(f"  ✅ Copied: 0_input_basicpitch.mid (piano roll source)")
                                    input_files["0"]["basicpitch_midi_path"] = f"/download/{process_id}/0_input_basicpitch.mid"
                                    midi_found = True
                                    print(f"  ✅ MIDI found in conditioning extraction - skipping re-extraction")

                        # If MIDI not found but enableMidiExport is true, extract it now
                        if not midi_found and enable_midi_export:
                            print(f"  🎵 Extracting MIDI with Basic Pitch (enableMidiExport=True)...")
                            try:
                                midi_result = save_basic_pitch_midi_with_voices(
                                    str(input_src),
                                    subgroup=instrument_subgroup,
                                    progress=None,
                                    tempo=120.0,
                                    separate_voices=False,
                                    monophonic=False
                                )
                                if midi_result and 'main_midi' in midi_result:
                                    extracted_midi = Path(midi_result['main_midi'])
                                    if extracted_midi.exists():
                                        # Copy as main MIDI input
                                        midi_dest = output_dir / "0_input.mid"
                                        shutil.copy(str(extracted_midi), str(midi_dest))
                                        print(f"  ✅ Copied: 0_input.mid (MIDI for display)")
                                        input_files["0"]["midi_path"] = f"/download/{process_id}/0_input.mid"
                            except Exception as e:
                                print(f"  ⚠️ Failed to extract MIDI: {e}")
                except Exception as e:
                    print(f"  ⚠️ Error copying input file: {e}")

            return {"file_paths": [f"/download/{process_id}/0.wav"], "input_files": input_files}

    except Exception as e:
        logging.error(f"❌ Error in ACE-Step generation: {e}")
        raise


# New simple ACE-Step task that uses pre-loaded pipeline
@celery_app.task(bind=True, name="generate_simple_ace_step_task")
def generate_simple_ace_step_task(
    self,
    prompt: str,
    lyrics: str = "",
    steps: int = 100,
    key: str = "C",
    duration: float = 30.0,
    seed: int = 0,
    noise_level: float = 0.8,
    ref_audio_path: str = None,
    detailed_mode: bool = False
):
    """Simple Celery task for ACE-Step generation using pre-loaded pipeline

    If detailed_mode=True, uses phrase-by-phrase generation with noise-to-noise architecture.
    """
    try:
        global ACE_STEP_PIPELINE
        import uuid
        from pathlib import Path
        import time

        print(f"\n{'='*80}")
        mode_label = "DETAILED MODE (Phrase-by-Phrase)" if detailed_mode else "Standard Mode"
        print(f"🎵 ACE-STEP GENERATION TASK STARTED - {mode_label}")
        print(f"{'='*80}")
        print(f"Prompt: {prompt}")
        print(f"Lyrics: {lyrics[:100]}..." if len(lyrics) > 100 else f"Lyrics: {lyrics}")
        print(f"Steps: {steps}, Seed: {seed}, Key: {key}")
        print(f"Duration: {duration}s")
        print(f"Noise Level: {noise_level}")
        print(f"Ref Audio: {ref_audio_path if ref_audio_path else 'None'}")
        print(f"Detailed Mode: {detailed_mode}")
        print(f"{'='*80}\n")

        # Generate unique process ID
        process_id = str(uuid.uuid4())[:8]

        # Create output directory
        output_dir = Path(f"./generated_ui/ace_step_{process_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output path
        output_path = output_dir / f"output_{int(time.time())}.wav"

        # Handle detailed mode (phrase-by-phrase generation)
        if detailed_mode and lyrics and ref_audio_path:
            print(f"🔬 Using DETAILED MODE: phrase-by-phrase generation with noise-to-noise")
            from generate_ace_step_detailed import generate_detailed_mode

            result = generate_detailed_mode(
                pipeline=ACE_STEP_PIPELINE if ACE_STEP_PIPELINE is not None else None,
                lyrics=lyrics,
                ref_audio_path=ref_audio_path,
                output_dir=output_dir,
                prompt=prompt,
                key=key,
                seed=seed,
                noise_level=noise_level,
                steps=steps,
                use_mfa=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            # Copy final output to expected location
            import shutil
            shutil.copy(result['output_path'], str(output_path))

            print(f"✅ Detailed mode generation complete!")
            print(f"   Output: {output_path}")
            print(f"   Generated {len(result['phrase_paths'])} phrases")

        # Use pre-loaded pipeline if available, otherwise fall back to wrapper script
        elif ACE_STEP_PIPELINE is not None:
            print(f"🚀 Using pre-loaded ACE-Step pipeline...")
            generation_start = time.time()

            # Determine task mode and audio2audio parameters
            task = "text2music"
            audio2audio_enable = False
            ref_audio_input = None
            ref_audio_strength = 0.5

            if ref_audio_path:
                task = "audio2audio"
                audio2audio_enable = True
                ref_audio_input = ref_audio_path
                # Map noise_level to ref_audio_strength (invert)
                # noise_level=0.0 means 100% GT, so ref_strength should be high (1.0)
                # noise_level=1.0 means 0% GT, so ref_strength should be low (0.0)
                ref_audio_strength = 1.0 - noise_level
                print(f"   🎵 Audio2Audio mode: ref_audio_strength={ref_audio_strength:.2f} (noise_level={noise_level})")

            ACE_STEP_PIPELINE(
                prompt=prompt,
                lyrics=lyrics,
                audio_duration=duration,
                infer_step=steps,
                manual_seeds=[seed],
                guidance_scale=15.0,
                save_path=str(output_path),
                task=task,
                audio2audio_enable=audio2audio_enable,
                ref_audio_input=ref_audio_input,
                ref_audio_strength=ref_audio_strength
            )

            generation_time = time.time() - generation_start
            print(f"✅ ACE-Step generation complete in {generation_time:.2f}s!")
        else:
            # Fall back to wrapper script if pipeline wasn't loaded
            print(f"⚠️  Pre-loaded pipeline not available, using wrapper script...")
            import subprocess

            # Build ref_audio argument if provided
            ref_audio_arg = f'--ref-audio "{ref_audio_path}"' if ref_audio_path else ''

            cmd = [
                "bash", "-c",
                f"""
                eval "$(conda shell.bash hook)"
                conda activate ace_step
                cd /home/arlo/Data
                python3 ace_step_noise_wrapper.py \
                    --prompt "{prompt}" \
                    --lyrics "{lyrics}" \
                    --steps {steps} \
                    --output "{output_path}" \
                    --duration {duration} \
                    --seed {seed} \
                    --noise-level {noise_level} \
                    {ref_audio_arg}
                """
            ]

            print(f"🚀 Calling ACE-Step wrapper with ace_step environment...")
            if ref_audio_path:
                print(f"   Using reference audio: {ref_audio_path}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                print(f"❌ ACE-Step generation failed!")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise Exception(f"ACE-Step generation failed: {result.stderr}")

            print(f"✅ ACE-Step generation complete!")

        # Check if output file was created
        if not output_path.exists():
            raise Exception("Output file was not created")

        # Return local path immediately (like generate_do does)
        # NOTE: process_id must match directory name ace_step_{process_id}
        local_download_path = f"/download-ace-step/{process_id}/{output_path.name}"
        print(f"📁 Output directory: {output_dir}")
        print(f"📁 Output file: {output_path}")
        print(f"🌐 Download URL: {local_download_path}")

        print(f"✅ ACE-Step generation complete! Returning local path: {local_download_path}")

        # Upload to GCS in background (optional, non-blocking)
        import threading
        def background_upload():
            try:
                from gcs_storage import upload_to_gcs, get_gcs_url
                print(f"📤 Background: Uploading to GCS...")
                gcs_path = upload_to_gcs(
                    str(output_path),
                    prefix="audiofiles/ace_step",
                    user_id="ace_step",
                    make_public=True
                )
                gcs_url = get_gcs_url(gcs_path)
                print(f"✅ Background: Uploaded to GCS: {gcs_url}")
            except Exception as e:
                print(f"⚠️  Background GCS upload failed: {e}")

        # Start background upload thread
        upload_thread = threading.Thread(target=background_upload, daemon=True)
        upload_thread.start()

        # Return local path immediately
        return {
            "file_paths": [local_download_path],
            "input_files": []
        }

    except Exception as e:
        print(f"❌ Error in ACE-Step generation: {e}")
        import traceback
        traceback.print_exc()
        raise


# New simple ACE-Step endpoint
@app.post("/api/generate-ace-step-simple")
async def generate_ace_step_simple(
    steps: int = Form(100),
    prompt: str = Form(""),
    lyrics: str = Form(""),
    key: str = Form("C"),
    duration: float = Form(30.0),
    seed: int = Form(0),
    noise_level: float = Form(0.8),
    ref_audio: UploadFile = File(None),
    midi_lyric_map: str = Form(None)
):
    """
    Simple FastAPI endpoint for ACE-Step text-to-music generation
    Directly calls ace_step_wrapper.py via Celery
    """
    import uuid
    from pathlib import Path

    print(f"\n{'='*80}")
    print(f"📥 RECEIVED /api/generate-ace-step-simple REQUEST")
    print(f"{'='*80}")
    print(f"   steps: {steps}")
    print(f"   prompt: {prompt}")
    print(f"   lyrics: {lyrics[:100]}..." if len(lyrics) > 100 else f"   lyrics: {lyrics}")
    print(f"   key: {key}")
    print(f"   duration: {duration}s")
    print(f"   seed: {seed}")
    print(f"   noise_level: {noise_level}")
    print(f"   ref_audio: {ref_audio.filename if ref_audio else 'None'}")
    print(f"{'='*80}\n")

    # Save reference audio file if provided
    ref_audio_path = None
    if ref_audio:
        # Create temp directory for ref audio
        temp_dir = Path(f"./generated_ui/temp_ref_audio")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save with unique filename
        unique_id = str(uuid.uuid4())[:8]
        ref_audio_path = temp_dir / f"ref_{unique_id}_{ref_audio.filename}"
        with open(ref_audio_path, "wb") as f:
            content = await ref_audio.read()
            f.write(content)
        print(f"💾 Saved reference audio: {ref_audio_path}")

        # Check if uploaded file is a MIDI file - render it through FluidSynth or WORLD vocoder
        if ref_audio_path.suffix.lower() in ['.mid', '.midi']:
            print(f"\n{'='*80}")
            print(f"🎹 MIDI FILE DETECTED")
            print(f"{'='*80}")
            print(f"   Input MIDI: {ref_audio_path.name}")
            print(f"   Lyrics provided: {'Yes' if lyrics else 'No'}")
            print(f"   MIDI lyric map provided: {'Yes' if midi_lyric_map else 'No'}")

            # Create rendered audio path
            rendered_audio_path = temp_dir / f"rendered_{ref_audio_path.stem}.wav"

            # If we have lyrics and a MIDI lyric map, use WORLD vocoder for syllable alignment
            import subprocess
            import json
            import os

            if lyrics and midi_lyric_map:
                print(f"\n   🎤 Using WORLD Vocoder with syllable alignment")
                print(f"{'='*80}")

                try:
                    # Parse MIDI lyric map
                    note_syllable_map = json.loads(midi_lyric_map)
                    print(f"   📝 MIDI-Lyrics mapping ({len(note_syllable_map)} notes):")
                    for note_idx, syllable in sorted(note_syllable_map.items(), key=lambda x: int(x[0])):
                        print(f"      Note {note_idx}: '{syllable}'")

                    # Create temporary JSON file for the mapping
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(note_syllable_map, f)
                        mapping_file = f.name

                    try:
                        # Call WORLD vocoder
                        cmd = [
                            'python3',
                            '/home/arlo/Data/espeak_world_vocoder_aligned.py',
                            '--midi', str(ref_audio_path),
                            '--lyrics-map', mapping_file,
                            '--output', str(rendered_audio_path)
                        ]

                        print(f"\n   🚀 Running WORLD vocoder...")
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                        if result.returncode == 0:
                            print(f"\n   ✅ WORLD vocoder synthesis complete!")
                            print(f"   Output: {rendered_audio_path.name}")
                            print(f"{'='*80}\n")
                            ref_audio_path = rendered_audio_path
                        else:
                            print(f"\n   ❌ WORLD vocoder failed!")
                            print(f"   STDERR: {result.stderr}")
                            print(f"   Falling back to FluidSynth rendering...")
                            raise Exception("WORLD vocoder failed")

                    finally:
                        # Cleanup temp file
                        if os.path.exists(mapping_file):
                            os.remove(mapping_file)

                except Exception as e:
                    print(f"   ⚠️  WORLD vocoder error: {e}")
                    print(f"   Falling back to FluidSynth with vocals soundfont...")

                    # Fallback to FluidSynth
                    soundfont_path = "/home/arlo/Data/soundfonts/vocals1.sf2"
                    try:
                        subprocess.run([
                            "fluidsynth", "-ni", "-g", "0.625", "-r", "44100", "-F", str(rendered_audio_path),
                            soundfont_path, str(ref_audio_path)
                        ], check=True, capture_output=True)

                        print(f"   ✅ Rendered with FluidSynth: {rendered_audio_path.name}")
                        print(f"{'='*80}\n")
                        ref_audio_path = rendered_audio_path

                    except subprocess.CalledProcessError as fs_error:
                        print(f"   ❌ FluidSynth also failed: {fs_error}")
                        raise HTTPException(500, f"Both WORLD vocoder and FluidSynth failed")

            else:
                # No lyrics mapping - use FluidSynth with vocals soundfont
                print(f"\n   🎹 Using FluidSynth with vocals soundfont (no lyrics mapping)")
                print(f"{'='*80}")

                soundfont_path = "/home/arlo/Data/soundfonts/vocals1.sf2"

                try:
                    subprocess.run([
                        "fluidsynth", "-ni", "-g", "0.625", "-r", "44100", "-F", str(rendered_audio_path),
                        soundfont_path, str(ref_audio_path)
                    ], check=True, capture_output=True)

                    print(f"   ✅ Rendered MIDI to audio: {rendered_audio_path.name}")
                    print(f"   Using soundfont: {soundfont_path}")
                    print(f"{'='*80}\n")

                    # Replace ref_audio_path with rendered audio path
                    ref_audio_path = rendered_audio_path

                except subprocess.CalledProcessError as e:
                    print(f"   ❌ FluidSynth rendering failed: {e}")
                    print(f"   STDERR: {e.stderr.decode() if e.stderr else 'N/A'}")
                    print(f"{'='*80}\n")
                    raise HTTPException(500, f"MIDI rendering failed: {e}")

        ref_audio_path = str(ref_audio_path)  # Convert to string for Celery

    # Queue the simple ACE-Step task
    task = generate_simple_ace_step_task.delay(
        prompt=prompt,
        lyrics=lyrics,
        steps=steps,
        key=key,
        duration=duration,
        seed=seed,
        noise_level=noise_level,
        ref_audio_path=ref_audio_path,
        detailed_mode=detailed_mode
    )

    return {
        "task_id": task.id,
        "expected_voices": 1
    }


@app.post("/generate")
async def generate_audio(
    # Support both parameter formats: doseedo2.html sends 'params' JSON, javascript2.js sends individual fields
    params: Optional[str] = Form(None),
    description: str = Form(""),
    duration: Optional[float] = Form(None),  # No default - will be derived from audio or scene data
    steps: int = Form(50),
    seed: int = Form(0),
    adapter_scale: float = Form(1.0),
    cfg_weight: float = Form(3.0),
    instrument_strength: float = Form(1.0),
    noise_level: float = Form(1.0),
    piano_roll_gain: float = Form(1.0),
    amp_gain: float = Form(1.0),
    rframe_gain: float = Form(1.0),
    rbend_gain: float = Form(1.0),
    encodec_gain: float = Form(1.0),
    pitch_fidelity_boost: float = Form(1.0),
    onset_guidance_boost: float = Form(2.0),
    pitch_snap_strength: float = Form(0.5),
    monophonic_mode: bool = Form(False),
    enable_voice_separation: bool = Form(False),
    # Inpainting parameters
    inpaint_mode: bool = Form(False),
    inpaint_start_time: Optional[float] = Form(None),
    inpaint_end_time: Optional[float] = Form(None),
    inpaint_voice_index: Optional[int] = Form(None),
    # Support both file field names
    audio_file: Optional[UploadFile] = File(None),
    conditioningAudio: Optional[UploadFile] = File(None),
    audioFile: Optional[UploadFile] = File(None),
    # Scene-aware MIDI generation parameters
    scene_durations: Optional[str] = Form(None),
    automation_data: Optional[str] = Form(None)
):
    """FastAPI endpoint for ACE-Step generation"""

    print(f"\n{'='*80}")
    print(f"📥 RECEIVED /api/generate-ace-step REQUEST")
    print(f"📍 CODE VERSION: 2025-10-22 18:40 - WITH REGENERATION LOGGING")
    print(f"{'='*80}")
    print(f"   params JSON: {params[:100] if params else 'None'}")
    print(f"   audio_file: {audio_file}")
    print(f"   audio_file.filename: {audio_file.filename if audio_file else 'N/A'}")
    print(f"   conditioningAudio: {conditioningAudio}")
    print(f"   conditioningAudio.filename: {conditioningAudio.filename if conditioningAudio else 'N/A'}")
    print(f"   audioFile: {audioFile}")
    print(f"   audioFile.filename: {audioFile.filename if audioFile else 'N/A'}")
    print(f"{'='*80}\n")

    # Parse params JSON if provided (from doseedo2.html)
    instrument_group = None
    instrument_subgroup = None
    generation_key = 'C'
    tempo_override = 120  # Default tempo
    arrange_mode = False
    fatten_mode = False
    fatten_type = "fake"
    tape_speed = 1.0
    slowdown_method = "tape"
    upsample_mode = False
    upsample_noise_level = 0.3
    upsample_steps = 20
    use_overlap_decoder = True
    fast_mode_variant = None
    midi_mode = False
    render_and_extract = False
    render_extract_mono = False
    # Test-time enhancement parameters
    use_best_of_n = False
    n_candidates = 12
    use_test_time_adaptation = False
    adaptation_steps = 10
    adaptation_learning_rate = 1e-4
    use_self_consistency = False
    consistency_samples = 3
    consistency_noise_scale = 0.05
    use_time_varying_noise = False
    onset_preservation = 0.7
    use_multiresolution_mixing = False
    use_onset_weighted_encodec = False
    encodec_onset_boost = 2.0
    enable_midi_export = False  # Default: don't export MIDI
    extract_formats = None  # Default: extract all formats
    if params:
        params_dict = json.loads(params)
        steps = params_dict.get('steps', steps)
        seed = params_dict.get('seed', seed)
        cfg_weight = params_dict.get('cfgWeight', cfg_weight)
        instrument_strength = params_dict.get('instrumentStrength', instrument_strength)
        noise_level = params_dict.get('noiseLevel', noise_level)
        piano_roll_gain = params_dict.get('pianoRollGain', piano_roll_gain)
        amp_gain = params_dict.get('ampGain', amp_gain)
        rframe_gain = params_dict.get('rframeGain', rframe_gain)
        rbend_gain = params_dict.get('rbendGain', rbend_gain)
        encodec_gain = params_dict.get('encodecGain', encodec_gain)
        pitch_fidelity_boost = params_dict.get('pitchFidelityBoost', pitch_fidelity_boost)
        onset_guidance_boost = params_dict.get('onsetGuidanceBoost', onset_guidance_boost)
        pitch_snap_strength = params_dict.get('pitchSnapStrength', pitch_snap_strength)
        monophonic_mode = params_dict.get('monophonicMode', monophonic_mode)
        arrange_mode = params_dict.get('arrangeMode', False)
        fatten_mode = params_dict.get('fattenMode', False)
        fatten_type = params_dict.get('fattenType', 'fake')
        tape_speed = params_dict.get('tapeSpeed', 1.0)
        slowdown_method = params_dict.get('slowdownMethod', 'tape')
        upsample_mode = params_dict.get('upsampleMode', False)
        upsample_noise_level = params_dict.get('upsampleNoiseLevel', 0.3)
        upsample_steps = params_dict.get('upsampleSteps', 20)
        use_overlap_decoder = params_dict.get('useOverlapDecoder', True)
        midi_mode = params_dict.get('midiMode', midi_mode)
        render_and_extract = params_dict.get('renderAndExtract', render_and_extract)
        render_extract_mono = params_dict.get('renderExtractMono', False)

        # Test-time enhancement parameters
        use_best_of_n = params_dict.get('useBestOfN', use_best_of_n)
        n_candidates = params_dict.get('nCandidates', n_candidates)
        use_test_time_adaptation = params_dict.get('useTestTimeAdaptation', use_test_time_adaptation)
        adaptation_steps = params_dict.get('adaptationSteps', adaptation_steps)
        adaptation_learning_rate = params_dict.get('adaptationLearningRate', adaptation_learning_rate)
        use_self_consistency = params_dict.get('useSelfConsistency', use_self_consistency)
        consistency_samples = params_dict.get('consistencySamples', consistency_samples)
        consistency_noise_scale = params_dict.get('consistencyNoiseScale', consistency_noise_scale)
        use_time_varying_noise = params_dict.get('useTimeVaryingNoise', use_time_varying_noise)
        onset_preservation = params_dict.get('onsetPreservation', onset_preservation)
        use_multiresolution_mixing = params_dict.get('useMultiresolutionMixing', use_multiresolution_mixing)
        use_onset_weighted_encodec = params_dict.get('useOnsetWeightedEncodec', use_onset_weighted_encodec)
        encodec_onset_boost = params_dict.get('encodecOnsetBoost', encodec_onset_boost)

        # DEBUG: Log voice separation parameter parsing
        print(f"\n🔍 DEBUG: Parsing enableVoiceSeparation from params JSON:")
        print(f"   Raw params_dict keys: {list(params_dict.keys())}")
        print(f"   'enableVoiceSeparation' in params_dict: {'enableVoiceSeparation' in params_dict}")
        print(f"   Value in params_dict: {params_dict.get('enableVoiceSeparation', 'KEY_NOT_FOUND')}")
        print(f"   Type: {type(params_dict.get('enableVoiceSeparation'))}")
        print(f"   Default (before parsing): {enable_voice_separation}")

        enable_voice_separation = params_dict.get('enableVoiceSeparation', enable_voice_separation)

        print(f"   Value after parsing: {enable_voice_separation}")
        print(f"   Type after parsing: {type(enable_voice_separation)}\n")

        # Parse MIDI export flag
        enable_midi_export = params_dict.get('enableMidiExport', enable_midi_export)
        print(f"   MIDI Export enabled: {enable_midi_export}")

        # Parse extraction formats (for selective extraction)
        extract_formats = params_dict.get('extractFormats', None)
        if extract_formats:
            print(f"   Extraction formats: {extract_formats}")

        # Parse chord parameters
        use_chords = params_dict.get('useChords', False)
        chord_beat_map = params_dict.get('chordBeatMap', {})
        print(f"\n🎹 CHORD PARAMETERS:")
        print(f"   useChords: {use_chords}")
        print(f"   chordBeatMap type: {type(chord_beat_map)}")
        print(f"   chordBeatMap: {chord_beat_map}")
        if chord_beat_map:
            print(f"   Number of chords: {len(chord_beat_map)}")
            print(f"   Chord progression: {chord_beat_map}")

        # Parse fast mode variant
        fast_mode_variant = params_dict.get('fastModeVariant', None)
        # Convert empty string to None
        if fast_mode_variant == "" or fast_mode_variant == False:
            fast_mode_variant = None

        # Parse inpainting parameters
        inpaint_mode = params_dict.get('inpaintMode', inpaint_mode)
        inpaint_start_time = params_dict.get('inpaintStartTime', inpaint_start_time)
        inpaint_end_time = params_dict.get('inpaintEndTime', inpaint_end_time)
        inpaint_voice_index = params_dict.get('inpaintVoiceIndex', inpaint_voice_index)

        instrument_group = params_dict.get('instrumentGroup')
        instrument_subgroup = params_dict.get('instrumentSubgroup')
        generation_key = params_dict.get('generationKey', 'C')
        tempo_override = params_dict.get('tempoOverride', 120)  # Default 120 BPM
        # Also check for scene data in params JSON
        if 'sceneDurations' in params_dict and not scene_durations:
            scene_durations = json.dumps(params_dict['sceneDurations'])
        if 'automationData' in params_dict and not automation_data:
            automation_data = json.dumps(params_dict['automationData'])

    # Parse scene_durations from JSON string to list
    scene_durations_list = None
    print(f"\n📥 FASTAPI ENDPOINT - SCENE DATA PARSING:")
    print(f"   Raw scene_durations param: {scene_durations}")
    print(f"   Type: {type(scene_durations)}")
    if scene_durations:
        try:
            scene_durations_list = json.loads(scene_durations) if isinstance(scene_durations, str) else scene_durations
            print(f"   ✅ Parsed scene_durations_list: {scene_durations_list}")
            print(f"   ✅ Number of scenes: {len(scene_durations_list)}")
            print(f"   ✅ Total duration: {sum(scene_durations_list):.2f}s")
            for i, dur in enumerate(scene_durations_list):
                print(f"      Scene {i}: {dur:.2f}s")
        except Exception as e:
            print(f"   ❌ Failed to parse scene_durations: {e}")
            scene_durations_list = None
    else:
        print(f"   ℹ️  No scene_durations received - will use simple generation")

    # Keep automation_data as string (will be parsed in task)
    if automation_data:
        print(f"🎚️ Automation data received: {len(automation_data)} chars")

    # Use whichever file was provided
    uploaded_file = audio_file or conditioningAudio or audioFile

    print(f"   Using file: {uploaded_file.filename if uploaded_file else 'None'}")
    print(f"   steps: {steps}, seed: {seed}, cfg: {cfg_weight}")
    print(f"   instrument: {instrument_group} / {instrument_subgroup}")

    # Log inpainting parameters
    if inpaint_mode:
        print(f"\n🎨 INPAINTING MODE ACTIVE:")
        print(f"   Start time: {inpaint_start_time}s")
        print(f"   End time: {inpaint_end_time}s")
        if inpaint_voice_index is not None:
            print(f"   Voice index: {inpaint_voice_index} (will only regenerate this voice)")
        print(f"   Duration to inpaint: {inpaint_end_time - inpaint_start_time:.2f}s")
        print(f"   Full audio file provided: {uploaded_file.filename if uploaded_file else 'None'}\n")
    elif inpaint_voice_index is not None:
        # REGENERATION MODE: Voice selection without time-based inpainting
        print(f"\n🔄 REGENERATION MODE ACTIVE:")
        print(f"   Monophonic mode: {monophonic_mode}")
        print(f"   Voice to regenerate: {inpaint_voice_index}")
        print(f"   Will regenerate FULL TRACK for voice {inpaint_voice_index}")
        print(f"   Input file provided: {uploaded_file.filename if uploaded_file else 'None'}\n")

    # Save uploaded file (audio or video) to shared directory accessible by Celery
    audio_file_path = None

    # DIAGNOSTIC: Log file upload status
    print(f"\n🔍 FILE UPLOAD DIAGNOSTIC:")
    print(f"   uploaded_file: {uploaded_file}")
    print(f"   uploaded_file.filename: {uploaded_file.filename if uploaded_file else 'N/A'}")
    print(f"   uploaded_file.content_type: {uploaded_file.content_type if uploaded_file else 'N/A'}")
    print(f"   inpaint_mode: {inpaint_mode}")
    print(f"   monophonic_mode: {monophonic_mode}")
    print(f"   inpaint_voice_index: {inpaint_voice_index}\n")

    if uploaded_file and uploaded_file.filename:
        file_extension = Path(uploaded_file.filename).suffix.lower()
        # Use shared directory instead of /tmp
        upload_dir = ensure_path_exists(get_output_path('uploads'))

        temp_file_path = str(upload_dir / f"{uuid.uuid4()}{file_extension}")
        with open(temp_file_path, "wb") as f:
            content = await uploaded_file.read()
            f.write(content)

        # Check if this is an edited MIDI JSON file
        if file_extension == '.json':
            print(f"\n{'='*80}")
            print(f"🎹 JSON FILE DETECTED - CHECKING FOR EDITED MIDI")
            print(f"{'='*80}")
            print(f"   File: {uploaded_file.filename}")
            print(f"   Content type: {uploaded_file.content_type if hasattr(uploaded_file, 'content_type') else 'N/A'}")
            print(f"   Size: {len(content)} bytes")

            try:
                with open(temp_file_path, 'r') as f:
                    json_data = json.load(f)

                print(f"   ✅ JSON parsed successfully")
                print(f"   JSON keys: {list(json_data.keys())}")
                print(f"   Type field: {json_data.get('type', 'NOT_FOUND')}")

                if json_data.get('type') == 'edited_midi' and 'notes' in json_data:
                    print(f"\n✅ EDITED MIDI JSON CONFIRMED")
                    print(f"   Notes: {len(json_data['notes'])}")
                    original_tempo = json_data.get('tempo', 120)
                    print(f"   Original Tempo: {original_tempo} BPM")
                    print(f"   Duration: {json_data.get('duration', 'N/A')}s")

                    # Apply tape/stretch slowdown if needed (for regeneration with slowdown)
                    midi_tempo = original_tempo
                    apply_stretch = False

                    if tape_speed != 1.0:
                        print(f"\n🎚️ SLOWDOWN PARAMETERS DETECTED:")
                        print(f"   Tape Speed: {tape_speed}")
                        print(f"   Slowdown Method: {slowdown_method}")

                        if slowdown_method == "tape":
                            # For tape mode: adjust MIDI tempo
                            # If tape_speed = 0.8 (slow down to 80%), we want the MIDI to be 20% slower
                            midi_tempo = original_tempo * tape_speed
                            print(f"   🎵 TAPE MODE: Adjusting MIDI tempo from {original_tempo} to {midi_tempo:.2f} BPM")
                            print(f"      (Audio will be generated at {midi_tempo:.2f} BPM and sped up {1/tape_speed:.2f}x during post-processing)")
                        elif slowdown_method == "stretch":
                            # For stretch mode: keep MIDI tempo same, stretch audio after rendering
                            midi_tempo = original_tempo
                            apply_stretch = True
                            print(f"   🎵 STRETCH MODE: Keeping MIDI tempo at {original_tempo} BPM")
                            print(f"      (Audio will be time-stretched to {tape_speed:.2f}x speed after FluidSynth rendering)")

                    print(f"   Converting to MIDI file...")

                    # Convert JSON to MIDI file
                    import pretty_midi
                    pm = pretty_midi.PrettyMIDI(initial_tempo=midi_tempo)
                    instrument = pretty_midi.Instrument(program=0)  # Piano

                    # If using tape mode, we need to adjust note timings proportionally
                    # If using stretch mode, keep timings as-is (will stretch audio later)
                    time_scale = tape_speed if slowdown_method == "tape" else 1.0

                    for note_data in json_data['notes']:
                        note = pretty_midi.Note(
                            velocity=int(note_data.get('velocity', 100)),
                            pitch=int(note_data['note']),
                            start=float(note_data['time']) * time_scale,
                            end=float(note_data['time'] + note_data['duration']) * time_scale
                        )
                        instrument.notes.append(note)

                    pm.instruments.append(instrument)

                    # Save as MIDI file
                    midi_path = temp_file_path.replace('.json', '_edited.mid')
                    pm.write(midi_path)
                    print(f"   ✅ Converted to MIDI: {midi_path}")
                    if tape_speed != 1.0 and slowdown_method == "tape":
                        print(f"      Applied {tape_speed:.2f}x tape slowdown to MIDI timing and tempo")

                    # Render MIDI with FluidSynth
                    import subprocess
                    soundfont_path = f"/home/arlo/Data/soundfonts/{instrument_subgroup}.sf2"
                    if not Path(soundfont_path).exists():
                        soundfont_path = "/home/arlo/Data/soundfonts/default.sf2"

                    rendered_path = temp_file_path.replace('.json', '_rendered.wav')
                    print(f"   🎼 Rendering with FluidSynth...")
                    print(f"      MIDI: {midi_path}")
                    print(f"      Soundfont: {soundfont_path}")
                    print(f"      Output: {rendered_path}")

                    result = subprocess.run([
                        "fluidsynth", "-ni", "-g", "0.5", "-r", "44100", "-F", rendered_path,
                        soundfont_path, midi_path
                    ], capture_output=True, text=True, timeout=60)

                    if result.returncode == 0 and Path(rendered_path).exists():
                        rendered_size = Path(rendered_path).stat().st_size
                        print(f"\n✅ FLUIDSYNTH RENDER COMPLETE")
                        print(f"   Output: {rendered_path}")
                        print(f"   Size: {rendered_size} bytes")

                        # Apply time-stretching if using stretch mode with slowdown
                        if apply_stretch and tape_speed != 1.0:
                            print(f"\n🎛️ APPLYING TIME-STRETCH (STRETCH MODE)")
                            print(f"   Stretch factor: {tape_speed:.2f}x")

                            try:
                                import librosa
                                import soundfile as sf

                                # Load the rendered audio
                                y, sr = librosa.load(rendered_path, sr=None)
                                print(f"   Loaded audio: {len(y)} samples at {sr} Hz ({len(y)/sr:.2f}s)")

                                # Time-stretch the audio
                                # tape_speed < 1.0 means slow down (e.g., 0.8 = 80% speed = slower)
                                # librosa.effects.time_stretch uses rate parameter where rate > 1.0 means faster
                                # So we need to use 1/tape_speed to invert it
                                stretch_rate = 1.0 / tape_speed
                                print(f"   Applying librosa time_stretch with rate={stretch_rate:.2f}")

                                y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate)
                                print(f"   Stretched audio: {len(y_stretched)} samples ({len(y_stretched)/sr:.2f}s)")

                                # Save stretched audio to a new file
                                stretched_path = rendered_path.replace('_rendered.wav', '_rendered_stretched.wav')
                                sf.write(stretched_path, y_stretched, sr)
                                stretched_size = Path(stretched_path).stat().st_size
                                print(f"   ✅ Saved stretched audio: {stretched_path}")
                                print(f"   Size: {stretched_size} bytes")

                                # Use the stretched audio for conditioning extraction
                                temp_file_path = stretched_path
                                file_extension = '.wav'
                                print(f"   Using stretched audio for conditioning extraction")

                            except Exception as e:
                                print(f"   ❌ Error during time-stretching: {e}")
                                import traceback
                                traceback.print_exc()
                                print(f"   ⚠️ Falling back to non-stretched audio")
                                temp_file_path = rendered_path
                                file_extension = '.wav'
                        else:
                            # No stretch needed, use rendered audio as-is
                            print(f"   Using rendered audio for conditioning extraction")
                            temp_file_path = rendered_path
                            file_extension = '.wav'

                        print(f"{'='*80}\n")
                    else:
                        print(f"   ❌ FluidSynth rendering failed:")
                        print(f"      stdout: {result.stdout}")
                        print(f"      stderr: {result.stderr}")
                        raise HTTPException(500, f"FluidSynth rendering failed: {result.stderr}")
                else:
                    print(f"\n⚠️ JSON file is NOT edited MIDI (type={json_data.get('type', 'NONE')})")
                    print(f"   Skipping MIDI conversion")
                    print(f"{'='*80}\n")

            except Exception as e:
                print(f"❌ Error processing edited MIDI JSON: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(500, f"Error processing edited MIDI: {str(e)}")

        # If it's a MIDI file, check if it's multi-track with voice separation disabled
        midi_extensions = ['.mid', '.midi']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']

        # Set audio_file_path to the processed file
        audio_file_path = temp_file_path

        if file_extension in midi_extensions:
            # Check if it's multi-track MIDI with voice separation disabled
            is_multi, track_count, _ = is_multitrack_midi(temp_file_path)

            # Keep MIDI as MIDI if:
            # 1. Monophonic mode with voice separation disabled (single or multi-track), OR
            # 2. Fast mode is enabled (for proper timing/resolution)
            should_keep_midi = (monophonic_mode and not enable_voice_separation) or fast_mode_variant

            if should_keep_midi:
                if monophonic_mode and not enable_voice_separation:
                    print(f"🎼 MIDI file detected - {track_count} track(s), multitrack={is_multi}")
                    print(f"   Monophonic mode: {monophonic_mode}, Voice separation: {enable_voice_separation}")
                    print(f"   Keeping as MIDI (Celery will split into voices and render)")
                elif fast_mode_variant:
                    print(f"🎼 UPLOAD HANDLER: Fast mode enabled (variant={fast_mode_variant})")
                    print(f"   is_multi={is_multi}, track_count={track_count}")
                    print(f"   Keeping MIDI file for direct conversion (better timing)")
                    print(f"   File will be: {temp_file_path}")
                # Keep the MIDI file, don't render to audio
                audio_file_path = temp_file_path
                print(f"✅ MIDI kept as: {audio_file_path}")
            else:
                print(f"🎹 Rendering MIDI file to audio with {instrument_subgroup} soundfont...")
                # Save original MIDI path before rendering
                original_midi_path = temp_file_path
                # Render MIDI to audio using the existing render_midi_to_audio function
                audio_file_path = render_midi_to_audio(
                    temp_file_path,
                    output_dir=str(upload_dir),
                    instrument_group=instrument_subgroup  # Use instrument_subgroup for correct soundfont
                )
                # DON'T remove temp MIDI file - we need it for the response!
                # Store the MIDI path for later inclusion in input_files
                print(f"✅ MIDI rendered to audio with {instrument_subgroup}: {audio_file_path}")
                print(f"   Original MIDI preserved at: {original_midi_path}")

        elif file_extension in video_extensions:
            print(f"🎬 Extracting audio from video...")
            audio_file_path = str(upload_dir / f"{uuid.uuid4()}.wav")
            # Extract audio using ffmpeg
            result = subprocess.run([
                'ffmpeg', '-y', '-i', temp_file_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                audio_file_path
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ FFmpeg error: {result.stderr}")
                raise RuntimeError(f"Failed to extract audio from video: {result.stderr}")

            # Remove temp video file
            os.remove(temp_file_path)
            print(f"✅ Extracted audio from video to: {audio_file_path}")
        else:
            # It's already an audio file (.wav, .mp3, etc.)
            audio_file_path = temp_file_path
            print(f"✅ Using audio file directly: {audio_file_path}")

    # Calculate duration from audio file or scene data
    calculated_duration = None

    print(f"\n📏 DURATION CALCULATION:")
    print(f"   audio_file_path: {audio_file_path}")
    print(f"   scene_durations_list: {scene_durations_list}")
    print(f"   duration parameter: {duration}")

    if audio_file_path:
        # Priority 1: Get duration from audio file
        try:
            import torchaudio
            wav, sr = torchaudio.load(audio_file_path)
            calculated_duration = wav.shape[-1] / sr
            print(f"   ✅ Using audio file duration: {calculated_duration:.2f}s")
        except Exception as e:
            print(f"   ⚠️ Could not determine audio duration: {e}")

    if calculated_duration is None and scene_durations_list and len(scene_durations_list) > 0:
        # Priority 2: Get duration from scene durations
        calculated_duration = sum(scene_durations_list)
        print(f"   ✅ Using scene durations total: {calculated_duration:.2f}s")

    if calculated_duration is None and duration is not None:
        # Priority 3: Use duration parameter from frontend
        calculated_duration = duration
        print(f"   ✅ Using duration parameter: {calculated_duration:.2f}s")

    if calculated_duration is None:
        # Priority 4: Use default of 30 seconds (for simple MIDI generation without video)
        calculated_duration = 30.0
        print(f"   ℹ️  No duration provided - using default: {calculated_duration:.2f}s")

    print(f"   FINAL DURATION: {calculated_duration:.2f}s\n")

    print(f"🚀 Queueing Celery task with audio_file_path: {audio_file_path}")
    print(f"   Final duration: {calculated_duration:.2f}s")
    print(f"   Monophonic mode: {monophonic_mode}, Voice separation: {enable_voice_separation}")

    # Calculate expected voices for frontend placeholder creation
    expected_voices = 0
    if audio_file_path and monophonic_mode:
        # Check if it's multitrack MIDI
        midi_extensions = ['.mid', '.midi']
        file_ext = Path(audio_file_path).suffix.lower()
        if file_ext in midi_extensions:
            is_multi, track_count, _ = is_multitrack_midi(audio_file_path)
            if is_multi and not enable_voice_separation:
                # Multitrack MIDI - use track count
                expected_voices = track_count
                if fatten_mode:
                    # Double for fatten mode (both real and fake create 2x tracks)
                    expected_voices *= 2
                print(f"   📊 Expected voices: {expected_voices} (multitrack MIDI, fatten={fatten_mode})")
            else:
                # Single track or voice separation mode - default 4 voices
                expected_voices = 4
                if fatten_mode:
                    expected_voices *= 2
                print(f"   📊 Expected voices: {expected_voices} (single-track/separated, fatten={fatten_mode})")
        else:
            # Audio file - default 4 voices
            expected_voices = 4
            if fatten_mode:
                expected_voices *= 2
            print(f"   📊 Expected voices: {expected_voices} (audio file, fatten={fatten_mode})")

    # CRITICAL FIX: In fast mode, use proper gains for conditioning (not all zeros!)
    # Fast mode extracts real encodec tokens, so we need non-zero gains
    if fast_mode_variant and monophonic_mode:
        print(f"\n⚡ FAST MODE GAIN ADJUSTMENT:")
        print(f"   Before: piano_roll={piano_roll_gain}, amp={amp_gain}, rframe={rframe_gain}, rbend={rbend_gain}, encodec={encodec_gain}")

        # Use render mode gains (works with real extracted conditioning)
        if piano_roll_gain == 1.0 and amp_gain == 0.0 and encodec_gain == 0.0:
            piano_roll_gain = 1.2
            amp_gain = 0.8
            rframe_gain = 0.8
            rbend_gain = 0.8
            encodec_gain = 0.5  # CRITICAL: Non-zero for timing!
            print(f"   After:  piano_roll={piano_roll_gain}, amp={amp_gain}, rframe={rframe_gain}, rbend={rbend_gain}, encodec={encodec_gain}")
            print(f"   ✅ Gains adjusted for fast mode with real conditioning")
        else:
            print(f"   ℹ️  Custom gains detected, keeping user settings")

    # Enqueue the task
    task = generate_do_task.delay(
        audio_file_path,
        description,
        calculated_duration,
        steps,
        seed,
        adapter_scale,
        cfg_weight,
        instrument_strength,
        noise_level,
        piano_roll_gain,
        amp_gain,
        rframe_gain,
        rbend_gain,
        encodec_gain,
        pitch_fidelity_boost,
        onset_guidance_boost,
        pitch_snap_strength,
        instrument_group,
        instrument_subgroup,
        monophonic_mode,
        arrange_mode,
        fatten_mode,
        fatten_type,
        enable_voice_separation,
        scene_durations_list,
        automation_data,
        tape_speed,
        slowdown_method,
        upsample_mode,
        upsample_noise_level,
        upsample_steps,
        use_overlap_decoder,
        inpaint_mode,
        inpaint_start_time,
        inpaint_end_time,
        inpaint_voice_index,
        fast_mode_variant,
        generation_key,
        tempo_override,
        # Test-time enhancement parameters
        use_best_of_n,
        n_candidates,
        use_test_time_adaptation,
        adaptation_steps,
        adaptation_learning_rate,
        use_self_consistency,
        consistency_samples,
        consistency_noise_scale,
        use_time_varying_noise,
        onset_preservation,
        use_multiresolution_mixing,
        use_onset_weighted_encodec,
        encodec_onset_boost,
        midi_mode,
        render_and_extract,
        render_extract_mono,
        enable_midi_export,
        use_chords,
        chord_beat_map,
        extract_formats
    )

    return {
        "task_id": task.id,
        "expected_voices": expected_voices
    }

@celery_app.task(bind=True, name="separate_stems_task")
def separate_stems_task(self, temp_input_path: str, process_id: str, mode: str = "6s"):
    """
    Celery task for stem separation using Demucs

    Args:
        temp_input_path: Path to input audio file
        process_id: Unique ID for this separation job
        mode: "6s" for 6-stem (htdemucs_6s) or "2s" for 2-stem (htdemucs vocals/instrumental)
    """
    import subprocess
    import shutil
    from pathlib import Path

    print(f"🎵 Starting stem separation task: {process_id} (mode: {mode})")
    print(f"   Input file: {temp_input_path}")

    # Save to /mnt/models for scalability (same as generate-do endpoint)
    output_base_dir = Path("/mnt/models/stems")
    output_dir = output_base_dir / f"stems_{process_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if mode == "2s":
            # Use htdemucs with --two-stems=vocals for faster 2-stem separation
            model_name = "htdemucs"
            cmd = [
                "demucs",
                "-n", model_name,
                "--two-stems", "vocals",
                "-o", str(output_dir),
                temp_input_path
            ]
            expected_stems = ["vocals", "no_vocals"]
            stems_subdir = model_name
        else:
            # Use htdemucs_6s for full 6-stem separation
            model_name = "htdemucs_6s"
            cmd = [
                "demucs",
                "-n", model_name,
                "-o", str(output_dir),
                temp_input_path
            ]
            expected_stems = ["drums", "bass", "other", "vocals", "guitar", "piano"]
            stems_subdir = model_name

        print(f"   Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"❌ Demucs failed:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            raise Exception(f"Stem separation failed: {result.stderr}")

        print(f"✅ Demucs completed successfully")

        # Find the separated stems
        input_filename = Path(temp_input_path).stem
        stems_dir = output_dir / stems_subdir / input_filename

        if not stems_dir.exists():
            raise Exception(f"Stems directory not found: {stems_dir}")

        # Copy stems to output directory and return local paths immediately
        stem_files = {}
        stem_paths_for_upload = []  # Track paths for background upload

        for stem_name in expected_stems:
            stem_file = stems_dir / f"{stem_name}.wav"
            if stem_file.exists():
                output_stem_path = output_dir / f"{stem_name}.wav"
                shutil.copy(stem_file, output_stem_path)
                print(f"   ✅ {stem_name}: {output_stem_path}")

                # Return local path immediately
                download_url = f"/download-stem/{process_id}/{stem_name}.wav"
                stem_files[stem_name] = download_url

                # Track for background GCS upload
                stem_paths_for_upload.append((stem_name, str(output_stem_path)))

        # For 2-stem mode, also create "instrumental" alias for "no_vocals"
        if mode == "2s" and "no_vocals" in stem_files:
            instrumental_path = output_dir / "instrumental.wav"
            no_vocals_path = output_dir / "no_vocals.wav"
            shutil.copy(no_vocals_path, instrumental_path)
            stem_files["instrumental"] = f"/download-stem/{process_id}/instrumental.wav"
            stem_paths_for_upload.append(("instrumental", str(instrumental_path)))

        # Upload to GCS in background (non-blocking)
        import threading
        def background_upload_stems():
            try:
                from gcs_storage import upload_to_gcs, get_gcs_url
                for stem_name, stem_path in stem_paths_for_upload:
                    try:
                        print(f"   📤 Background: Uploading {stem_name} to GCS...")
                        gcs_path = upload_to_gcs(
                            stem_path,
                            prefix=f"stems/{process_id}",
                            user_id="stems",
                            make_public=True
                        )
                        gcs_url = get_gcs_url(gcs_path)
                        print(f"   ✅ Background: Uploaded {stem_name} to GCS: {gcs_url}")
                    except Exception as e:
                        print(f"   ⚠️  Background: GCS upload failed for {stem_name}: {e}")
            except Exception as e:
                print(f"   ⚠️  Background: GCS upload error: {e}")

        # Start background upload thread
        upload_thread = threading.Thread(target=background_upload_stems, daemon=True)
        upload_thread.start()

        # Clean up temp input file
        import os
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

        print(f"✅ Stem separation complete: {len(stem_files)} stems (mode: {mode})")

        return {
            "process_id": process_id,
            "stems": stem_files,
            "stem_count": len(stem_files),
            "mode": mode
        }

    except Exception as e:
        print(f"❌ Error during stem separation: {e}")
        import traceback
        traceback.print_exc()
        # Clean up temp file on error
        import os
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        raise


# Alias endpoint for frontend compatibility
@app.post("/api/generate-do")
async def generate_do_endpoint(
    params: Optional[str] = Form(None),
    description: str = Form(""),
    duration: Optional[float] = Form(None),
    steps: int = Form(50),
    seed: int = Form(0),
    adapter_scale: float = Form(1.0),
    cfg_weight: float = Form(3.0),
    instrument_strength: float = Form(1.0),
    noise_level: float = Form(1.0),
    piano_roll_gain: float = Form(1.0),
    amp_gain: float = Form(1.0),
    rframe_gain: float = Form(1.0),
    rbend_gain: float = Form(1.0),
    encodec_gain: float = Form(1.0),
    pitch_fidelity_boost: float = Form(1.0),
    onset_guidance_boost: float = Form(2.0),
    pitch_snap_strength: float = Form(0.5),
    monophonic_mode: bool = Form(False),
    enable_voice_separation: bool = Form(False),
    inpaint_mode: bool = Form(False),
    inpaint_start_time: Optional[float] = Form(None),
    inpaint_end_time: Optional[float] = Form(None),
    inpaint_voice_index: Optional[int] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    conditioningAudio: Optional[UploadFile] = File(None),
    audioFile: Optional[UploadFile] = File(None),
    midiFile: Optional[UploadFile] = File(None),
    scene_durations: Optional[str] = Form(None),
    automation_data: Optional[str] = Form(None)
):
    """
    Frontend-compatible endpoint for Dø generation.
    Forwards to generate_audio() with the same parameters.
    """
    return await generate_audio(
        params=params,
        description=description,
        duration=duration,
        steps=steps,
        seed=seed,
        adapter_scale=adapter_scale,
        cfg_weight=cfg_weight,
        instrument_strength=instrument_strength,
        noise_level=noise_level,
        piano_roll_gain=piano_roll_gain,
        amp_gain=amp_gain,
        rframe_gain=rframe_gain,
        rbend_gain=rbend_gain,
        encodec_gain=encodec_gain,
        pitch_fidelity_boost=pitch_fidelity_boost,
        onset_guidance_boost=onset_guidance_boost,
        pitch_snap_strength=pitch_snap_strength,
        monophonic_mode=monophonic_mode,
        enable_voice_separation=enable_voice_separation,
        inpaint_mode=inpaint_mode,
        inpaint_start_time=inpaint_start_time,
        inpaint_end_time=inpaint_end_time,
        inpaint_voice_index=inpaint_voice_index,
        audio_file=audio_file or midiFile,
        conditioningAudio=conditioningAudio,
        audioFile=audioFile,
        scene_durations=scene_durations,
        automation_data=automation_data
    )


# Also add /api/generate-do/task/{task_id} endpoint for status polling
@app.get("/api/generate-do/task/{task_id}")
async def get_do_task_status(task_id: str):
    """
    Frontend-compatible task status endpoint.
    Forwards to the existing task status endpoint.
    """
    return await get_task_status(task_id)


@app.post("/separate-stems")
async def separate_stems(
    audio_file: Optional[UploadFile] = File(None),
    audioFile: Optional[UploadFile] = File(None),
    audioUrl: Optional[str] = Form(None),
    mode: str = Form("6s")
):
    """
    Separate audio into stems using Demucs (async with Celery)
    Accepts either file upload or audio URL

    Args:
        mode: "6s" for 6-stem separation (drums, bass, other, vocals, guitar, piano)
              "2s" for 2-stem separation (vocals, instrumental) - faster

    Returns: Task ID for polling
    """
    import uuid
    import shutil
    import httpx

    print(f"📥 Received /separate-stems request (mode: {mode})")

    # Use whichever file was provided
    uploaded_file = audio_file or audioFile

    upload_dir = ensure_path_exists(get_output_path('uploads'))

    # Handle URL-based separation (for ACE-Step generated files)
    if audioUrl and not uploaded_file:
        print(f"   Processing URL: {audioUrl}")

        # Download the file from the URL
        try:
            # Handle relative URLs by constructing full URL
            if audioUrl.startswith('/'):
                # Use local file path instead of HTTP for local URLs
                if audioUrl.startswith('/download-ace-step/'):
                    # Extract the file path from the URL
                    # /download-ace-step/{filename}
                    filename = audioUrl.split('/')[-1]
                    local_path = Path("/home/arlo/Data/output_audio") / filename

                    if not local_path.exists():
                        raise HTTPException(404, f"Audio file not found: {audioUrl}")

                    print(f"   Using local file: {local_path}")

                    # Copy to temp location
                    file_extension = local_path.suffix.lower()
                    temp_input_path = str(upload_dir / f"{uuid.uuid4()}{file_extension}")
                    shutil.copy(local_path, temp_input_path)

                elif audioUrl.startswith('/download-upload/'):
                    # Handle uploaded files
                    filename = audioUrl.split('/')[-1]
                    local_path = get_output_path('uploads') / filename

                    if not local_path.exists():
                        raise HTTPException(404, f"Audio file not found: {audioUrl}")

                    print(f"   Using local file: {local_path}")

                    # Use directly or copy to temp location
                    file_extension = local_path.suffix.lower()
                    temp_input_path = str(upload_dir / f"{uuid.uuid4()}{file_extension}")
                    shutil.copy(local_path, temp_input_path)
                else:
                    # Try to fetch via HTTP as fallback
                    full_url = f"http://127.0.0.1:8070{audioUrl}"
                    async with httpx.AsyncClient() as client:
                        response = await client.get(full_url)
                        if response.status_code != 200:
                            raise HTTPException(500, f"Failed to download audio from URL: {response.status_code}")

                        # Determine file extension from content-type or URL
                        content_type = response.headers.get('content-type', '')
                        if 'wav' in content_type:
                            file_extension = '.wav'
                        elif 'mp3' in content_type:
                            file_extension = '.mp3'
                        else:
                            file_extension = Path(audioUrl).suffix or '.wav'

                        temp_input_path = str(upload_dir / f"{uuid.uuid4()}{file_extension}")
                        with open(temp_input_path, "wb") as f:
                            f.write(response.content)
            else:
                # External URL - download via HTTP
                async with httpx.AsyncClient() as client:
                    response = await client.get(audioUrl)
                    if response.status_code != 200:
                        raise HTTPException(500, f"Failed to download audio from URL: {response.status_code}")

                    # Determine file extension
                    content_type = response.headers.get('content-type', '')
                    if 'wav' in content_type:
                        file_extension = '.wav'
                    elif 'mp3' in content_type:
                        file_extension = '.mp3'
                    else:
                        file_extension = Path(audioUrl).suffix or '.wav'

                    temp_input_path = str(upload_dir / f"{uuid.uuid4()}{file_extension}")
                    with open(temp_input_path, "wb") as f:
                        f.write(response.content)

        except Exception as e:
            print(f"❌ Error downloading audio from URL: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(500, f"Failed to download audio: {str(e)}")

    # Handle file upload (original behavior)
    elif uploaded_file and uploaded_file.filename:
        print(f"   Processing file: {uploaded_file.filename}")

        # Save uploaded file to temporary location
        file_extension = Path(uploaded_file.filename).suffix.lower()
        temp_input_path = str(upload_dir / f"{uuid.uuid4()}{file_extension}")
        with open(temp_input_path, "wb") as f:
            content = await uploaded_file.read()
            f.write(content)
    else:
        raise HTTPException(400, "No audio file or URL provided")

    print(f"   Saved to: {temp_input_path}")

    # Generate process ID
    process_id = str(uuid.uuid4())

    # Queue Celery task for stem separation
    print(f"🚀 Queuing stem separation task: {process_id} (mode: {mode})")
    task = separate_stems_task.delay(temp_input_path, process_id, mode)

    print(f"✅ Task queued: {task.id}")

    return {
        "task_id": task.id,
        "process_id": process_id,
        "mode": mode,
        "status": "processing"
    }

@app.get("/separate-stems/status/{task_id}")
async def get_stem_separation_status(task_id: str):
    """Get status of stem separation task"""
    task_result = celery_app.AsyncResult(task_id)

    if task_result.state == 'PENDING':
        return {"status": "processing", "task_id": task_id}
    elif task_result.state == 'SUCCESS':
        result = task_result.result
        return {
            "status": "completed",
            "task_id": task_id,
            "process_id": result["process_id"],
            "stems": result["stems"],
            "stem_count": result["stem_count"]
        }
    elif task_result.state == 'FAILURE':
        return {
            "status": "failed",
            "task_id": task_id,
            "error": str(task_result.info)
        }
    else:
        return {"status": task_result.state.lower(), "task_id": task_id}

@app.get("/download-stem/{process_id}/{filename}")
async def download_stem(process_id: str, filename: str):
    """
    Download separated stem file (serves from local or GCS)
    This endpoint is kept for backward compatibility with local storage.
    New stem separations return direct GCS URLs.
    """
    # Try local file first (check both new and old locations for backward compatibility)
    # New location: /mnt/models/stems
    file_path_new = get_output_path('stems', process_id=process_id) / filename
    if file_path_new.exists():
        return FileResponse(file_path_new, media_type="audio/wav", filename=filename)

    # Old location: /home/arlo/ScoreAI/audiofiles (backward compatibility for old files)
    file_path_old = Path("/home/arlo/ScoreAI/audiofiles") / f"stems_{process_id}" / filename
    if file_path_old.exists():
        return FileResponse(file_path_old, media_type="audio/wav", filename=filename)

    # Try GCS if local file doesn't exist
    try:
        from gcs_storage import gcs_file_exists, download_from_gcs, get_gcs_path
        import tempfile

        # Construct GCS path
        gcs_path = f"gs://score-ai-generations/stems/{process_id}/stems/{filename}"

        if gcs_file_exists(gcs_path):
            print(f"📥 Downloading from GCS: {gcs_path}")
            # Download to temp file
            temp_path = download_from_gcs(gcs_path)
            return FileResponse(temp_path, media_type="audio/wav", filename=filename)
    except Exception as e:
        print(f"⚠️  GCS download failed: {e}")

    # Not found
    raise HTTPException(status_code=404, detail=f"Stem file not found: {filename}")

@app.post("/api/upload-audio")
async def upload_audio(
    audioFile: UploadFile = File(...)
):
    """
    Upload an audio file and return a persistent URL
    Used for drag-and-drop uploads to timeline
    """
    import uuid

    print(f"📥 Received /api/upload-audio request")

    if not audioFile or not audioFile.filename:
        raise HTTPException(400, "No audio file provided")

    print(f"   Processing file: {audioFile.filename}")

    # Save uploaded file to uploads directory
    file_extension = Path(audioFile.filename).suffix.lower()
    upload_dir = ensure_path_exists(get_output_path('uploads'))

    # Create unique filename
    unique_id = str(uuid.uuid4())
    safe_filename = f"{unique_id}{file_extension}"
    file_path = upload_dir / safe_filename

    # Save file
    with open(file_path, "wb") as f:
        content = await audioFile.read()
        f.write(content)

    print(f"   Saved to: {file_path}")

    # Return download URL
    download_url = f"/download-upload/{safe_filename}"

    return {
        "url": download_url,
        "filename": audioFile.filename,
        "size": len(content)
    }

@app.get("/download-upload/{filename}")
async def download_upload(filename: str):
    """Download uploaded audio file"""
    file_path = get_output_path('uploads') / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    # Determine media type from extension
    ext = Path(filename).suffix.lower()
    media_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac',
        '.m4a': 'audio/mp4'
    }
    media_type = media_types.get(ext, 'audio/wav')

    return FileResponse(file_path, media_type=media_type)

@app.post("/generate-risers")
async def generate_risers(
    scene_durations: str = Form(...)
):
    """
    Select random riser samples for scene transitions
    Returns: List of riser file URLs
    """
    import glob
    import random
    import subprocess
    import uuid

    # Parse scene durations
    try:
        scene_durations_list = json.loads(scene_durations) if isinstance(scene_durations, str) else scene_durations
    except Exception as e:
        raise HTTPException(400, f"Invalid scene_durations: {e}")

    # Check riser directory
    riser_dir = "/home/arlo/Risers/"
    if not os.path.isdir(riser_dir):
        raise HTTPException(404, f"Riser folder not found: {riser_dir}")

    # Find all .wav riser files
    riser_files = [f for f in os.listdir(riser_dir) if f.endswith(".wav")]
    if not riser_files:
        raise HTTPException(404, "No riser files found.")

    # Calculate number of risers needed (skip first scene)
    scene_count = max(0, len(scene_durations_list) - 1)

    print(f"🎚️ Generating {scene_count} random risers from {len(riser_files)} available files")

    # Create output directory matching the download endpoint path
    process_id = str(uuid.uuid4())
    output_directory = ensure_path_exists(get_output_path('ace_step_output', process_id=process_id))

    # Randomly select and copy risers
    file_paths = []
    for i in range(scene_count):
        chosen_file = random.choice(riser_files)
        src_path = os.path.join(riser_dir, chosen_file)
        dest_path = str(output_directory / f"{i}.wav")
        subprocess.run(["cp", src_path, dest_path], check=True)
        file_paths.append(f"/download/{process_id}/{i}.wav")
        print(f"   Riser {i+1}: {chosen_file} -> {dest_path}")

    print(f"🎈 Returned {scene_count} random risers")

    return {
        "file_paths": file_paths,
        "count": len(file_paths)
    }

@app.post("/generate-drums")
async def generate_drums(
    tempo_override: int = Form(120),
    scene_durations: str = Form(...),
    automation_data: str = Form(None),
    scene_tempos: str = Form(None),  # New parameter: per-scene tempos
    active_samples: str = Form(None)  # New parameter: active sample types
):
    """
    Generate drum hits on downbeats using ORCH samples
    Uses automation envelope to select velocity
    Supports per-scene tempo changes to match music generation
    Supports selecting specific sample types (bass_drum, timpani, cymbals, percussion)
    Returns: Dictionary with sample_tracks containing data for each sample type
    """
    import glob
    import random
    import subprocess
    import uuid

    # Parse scene durations
    try:
        scene_durations_list = json.loads(scene_durations) if isinstance(scene_durations, str) else scene_durations
    except Exception as e:
        raise HTTPException(400, f"Invalid scene_durations: {e}")

    # Parse active samples (default to bass_drum if not provided)
    active_samples_list = ["bass_drum"]  # Default
    if active_samples and active_samples != "null":
        try:
            active_samples_list = json.loads(active_samples) if isinstance(active_samples, str) else active_samples
            print(f"🥁 Active samples: {active_samples_list}")
        except Exception as e:
            print(f"⚠️ Could not parse active_samples, using default [bass_drum]: {e}")

    # Map frontend names to directory names
    sample_dir_map = {
        "bass_drum": "bassdrum",
        "timpani": "timpani",
        "cymbals": "cymbals",
        "percussion": "percussion"
    }

    # Parse scene tempos (if provided, otherwise use tempo_override for all scenes)
    scene_tempos_list = None
    if scene_tempos and scene_tempos != "null":
        try:
            scene_tempos_list = json.loads(scene_tempos) if isinstance(scene_tempos, str) else scene_tempos
            print(f"🎵 Using per-scene tempos: {scene_tempos_list}")
        except Exception as e:
            print(f"⚠️ Could not parse scene_tempos, using tempo_override: {e}")

    # If no scene_tempos provided, compute optimal tempos like the music generation does
    if scene_tempos_list is None:
        # Build scene changes array
        scene_changes = [0.0]
        cumulative = 0.0
        for dur in scene_durations_list:
            cumulative += dur
            scene_changes.append(cumulative)

        # Compute optimal tempos for each scene
        scene_tempos_list = compute_best_tempos(scene_changes)
        print(f"🎵 Computed optimal tempos for {len(scene_tempos_list)} scenes: {scene_tempos_list}")

    # Parse automation data
    automation_points = []
    if automation_data and automation_data != "null":
        try:
            automation_dict = json.loads(automation_data) if isinstance(automation_data, str) else automation_data
            automation_points = automation_dict.get("points", [])
        except Exception as e:
            print(f"⚠️ Could not parse automation data: {e}")

    # Check ORCH directory
    orch_base_dir = "/home/arlo/harmonymodule/ORCH"
    if not os.path.isdir(orch_base_dir):
        raise HTTPException(404, f"ORCH folder not found: {orch_base_dir}")

    # Available velocities
    available_velocities = [80, 90, 100, 110, 120, 127]

    # For each active sample type, find available sample numbers and pick one
    sample_choices = {}  # Maps sample_type -> (dir_path, chosen_sample_number)

    for sample_type in active_samples_list:
        dir_name = sample_dir_map.get(sample_type)
        if not dir_name:
            print(f"⚠️ Unknown sample type: {sample_type}, skipping")
            continue

        sample_dir = os.path.join(orch_base_dir, dir_name)
        if not os.path.isdir(sample_dir):
            print(f"⚠️ Sample directory not found: {sample_dir}, skipping {sample_type}")
            continue

        # Find available sample numbers (check what exists for velocity 100)
        sample_numbers = set()
        for f in os.listdir(sample_dir):
            if f.startswith("100.") and f.endswith(".wav"):
                try:
                    sample_num = int(f.split(".")[1])
                    sample_numbers.add(sample_num)
                except (ValueError, IndexError):
                    pass

        if not sample_numbers:
            print(f"⚠️ No samples found in {sample_dir}, skipping {sample_type}")
            continue

        # Pick a random sample number to use for this sample type
        chosen_sample = random.choice(list(sample_numbers))
        sample_choices[sample_type] = (sample_dir, chosen_sample)
        print(f"🥁 Selected {sample_type} sample #{chosen_sample} from {dir_name}/")

    if not sample_choices:
        raise HTTPException(404, "No valid ORCH samples found for the selected sample types.")

    # Generate downbeat times accounting for tempo changes at scene boundaries
    downbeats = []
    current_time = 0.0

    for scene_idx, scene_duration in enumerate(scene_durations_list):
        scene_tempo = scene_tempos_list[scene_idx] if scene_idx < len(scene_tempos_list) else tempo_override
        seconds_per_beat = 60.0 / scene_tempo
        seconds_per_bar = seconds_per_beat * 4  # 4/4 time

        scene_end_time = current_time + scene_duration

        # Generate downbeats within this scene
        scene_time = current_time
        while scene_time < scene_end_time:
            if scene_time < sum(scene_durations_list):  # Don't exceed total duration
                downbeats.append(scene_time)
            scene_time += seconds_per_bar

        print(f"  Scene {scene_idx}: {scene_tempo} BPM, duration {scene_duration:.2f}s, generated {len([t for t in downbeats if current_time <= t < scene_end_time])} downbeats")
        current_time = scene_end_time

    total_duration = sum(scene_durations_list)
    print(f"🎵 Generated {len(downbeats)} downbeats across {len(scene_durations_list)} scenes over {total_duration:.2f}s")

    # Helper function to interpolate automation value at a given time
    def get_automation_value(time, points):
        if not points or len(points) == 0:
            print(f"   No automation points, using default 0.7")
            return 0.7  # Default medium velocity

        # Sort points by time
        sorted_points = sorted(points, key=lambda p: p.get("time", 0))

        if len(sorted_points) == 0:
            print(f"   Empty sorted points, using default 0.7")
            return 0.7

        # If time is before first point, use first point's value
        if time <= sorted_points[0]["time"]:
            vol = sorted_points[0].get("volume")
            return vol if vol is not None else 0.7

        # If time is after last point, use last point's value
        if time >= sorted_points[-1]["time"]:
            vol = sorted_points[-1].get("volume")
            return vol if vol is not None else 0.7

        # Find surrounding points and interpolate
        for i in range(len(sorted_points) - 1):
            p1 = sorted_points[i]
            p2 = sorted_points[i + 1]

            if p1["time"] <= time <= p2["time"]:
                # Linear interpolation
                t = (time - p1["time"]) / (p2["time"] - p1["time"]) if p2["time"] != p1["time"] else 0
                v1 = p1.get("volume") if p1.get("volume") is not None else 0.7
                v2 = p2.get("volume") if p2.get("volume") is not None else 0.7
                return v1 + (v2 - v1) * t

        return 0.7  # Fallback

    # Map automation value (0-1) to velocity
    def volume_to_velocity(volume):
        # Handle None case (fallback to default)
        if volume is None:
            volume = 0.7

        # Map 0-1 volume to velocities [80, 90, 100, 110, 120, 127]
        if volume < 0.15:
            return 80
        elif volume < 0.35:
            return 90
        elif volume < 0.55:
            return 100
        elif volume < 0.75:
            return 110
        elif volume < 0.90:
            return 120
        else:
            return 127

    # Create output directory
    process_id = str(uuid.uuid4())
    output_directory = ensure_path_exists(get_output_path('ace_step_output', process_id=process_id))

    # Process each sample type separately to create separate tracks
    sample_tracks = {}

    for sample_type, (sample_dir, chosen_sample) in sample_choices.items():
        # For each downbeat, calculate velocity based on automation envelope
        file_paths = []
        velocities = []

        for i, downbeat_time in enumerate(downbeats):
            # Get automation value at this specific downbeat time
            auto_value = get_automation_value(downbeat_time, automation_points)
            if auto_value is None:
                print(f"⚠️  WARNING: get_automation_value returned None for time {downbeat_time:.2f}s")
                print(f"   automation_points type: {type(automation_points)}, length: {len(automation_points) if automation_points else 'N/A'}")
                auto_value = 0.7  # Fallback
            velocity = volume_to_velocity(auto_value)
            velocities.append(velocity)

            # Find the sample file with this velocity and the chosen sample index
            sample_filename = f"{velocity}.{chosen_sample}.wav"
            src_path = os.path.join(sample_dir, sample_filename)

            if not os.path.exists(src_path):
                print(f"⚠️ Sample not found: {sample_filename}, using default velocity 110")
                sample_filename = f"110.{chosen_sample}.wav"
                src_path = os.path.join(sample_dir, sample_filename)
                velocity = 110
                velocities[-1] = 110

            # Copy to output with unique name including sample type
            dest_filename = f"drum_{sample_type}_{i}_{velocity}.wav"
            dest_path = str(output_directory / dest_filename)
            subprocess.run(["cp", src_path, dest_path], check=True)

            file_paths.append(f"/download/{process_id}/{dest_filename}")
            print(f"🥁 {sample_type} hit {i+1} at {downbeat_time:.2f}s: velocity {velocity} (sample #{chosen_sample})")

        sample_tracks[sample_type] = {
            "file_paths": file_paths,
            "start_times": downbeats,
            "velocities": velocities,
            "count": len(downbeats),
            "sample_number": chosen_sample
        }

        print(f"🥁 Created {len(file_paths)} {sample_type} samples with varying velocities")

    return {
        "sample_tracks": sample_tracks,
        "tempos": scene_tempos_list,  # Return list of per-scene tempos
        "total_samples": len(active_samples_list)
    }

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Check task status"""
    task = celery_app.AsyncResult(task_id)
    print(f"📊 Task {task_id} state: {task.state}")

    if task.state == "SUCCESS":
        result = task.result
        print(f"📊 Task result type: {type(result)}, value: {result}")

        # Handle different result formats
        if isinstance(result, dict) and "file_paths" in result:
            file_paths = result["file_paths"]
            input_files = result.get("input_files", {})
            print(f"📊 Returning file_paths: {file_paths}")
            print(f"📦 Returning input_files: {input_files}")
            return {
                "status": "completed",
                "result": file_paths,
                "input_files": input_files
            }
        elif isinstance(result, dict):
            file_paths = result.get("file_paths", [])
            input_files = result.get("input_files", {})
            print(f"📊 Returning file_paths: {file_paths}")
            print(f"📦 Returning input_files: {input_files}")
            return {
                "status": "completed",
                "result": file_paths,
                "input_files": input_files
            }
        else:
            print(f"⚠️ Unexpected result format: {result}")
            return {
                "status": "completed",
                "result": []
            }
    elif task.state == "FAILURE":
        print(f"❌ Task failed: {task.info}")
        return {"status": "failed", "error": str(task.info)}
    elif task.state == "PROGRESS":
        # Return partial results for incremental display
        info = task.info or {}
        completed_voices = info.get("completed_voices", [])
        total_voices = info.get("total_voices", 0)
        input_files = info.get("input_files", {})

        # Calculate progress
        if isinstance(completed_voices, list):
            num_completed = len(completed_voices)
            progress = num_completed / total_voices if total_voices > 0 else 0.0
        else:
            # Fallback if it's an int (backwards compatibility)
            num_completed = completed_voices
            progress = completed_voices / total_voices if total_voices > 0 else 0.0

        print(f"📊 Progress: {num_completed}/{total_voices} voices completed ({progress*100:.1f}%)")
        print(f"📊 Completed voices: {completed_voices}")
        print(f"📦 Input files available: {len(input_files)} voices")

        return {
            "status": "processing",
            "completed_voices": completed_voices,  # Return the list of file paths
            "total_voices": total_voices,
            "progress": progress,
            "input_files": input_files
        }
    else:
        return {"status": task.state}

@app.get("/download/{process_id}/{filename}")
async def download_audio(process_id: str, filename: str):
    """Download generated audio file"""
    file_path = get_output_path('ace_step_output', process_id=process_id) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav")

@app.get("/api/list-midi-files")
async def list_midi_files():
    """List all MIDI files from the harmonymodule directory"""
    midi_dir = Path("/home/arlo/harmonymodule/drummidis")

    if not midi_dir.exists():
        return {"files": [], "error": "MIDI directory not found"}

    try:
        # Get all .mid and .midi files
        midi_files = []
        for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
            midi_files.extend([f.name for f in midi_dir.glob(ext)])

        # Sort alphabetically
        midi_files.sort()

        print(f"📁 Found {len(midi_files)} MIDI files in {midi_dir}")
        return {"files": midi_files}
    except Exception as e:
        print(f"❌ Error listing MIDI files: {e}")
        return {"files": [], "error": str(e)}

@app.get("/api/get-midi-file/{filename}")
async def get_midi_file(filename: str):
    """Serve a specific MIDI file"""
    midi_dir = Path("/home/arlo/harmonymodule/drummidis")
    file_path = midi_dir / filename

    # Security check: ensure the file is within the MIDI directory
    try:
        file_path = file_path.resolve()
        midi_dir = midi_dir.resolve()
        if not str(file_path).startswith(str(midi_dir)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception as e:
        raise HTTPException(status_code=403, detail="Invalid file path")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    print(f"📤 Serving MIDI file: {filename}")
    return FileResponse(file_path, media_type="audio/midi", filename=filename)

@app.post("/api/audio-to-midi")
async def audio_to_midi(
    audioFile: UploadFile = File(...),
    bpm: int = Form(120),
    detailedMode: bool = Form(False)
):
    """
    Convert recorded audio to MIDI using Basic Pitch
    Returns MIDI file data and metadata
    Args:
        audioFile: Audio file to convert
        bpm: Current timeline BPM (default: 120)
        detailedMode: Use detailed processing with slowdown (default: False)
    """
    import uuid
    import base64

    mode_str = "detailed (with slowdown)" if detailedMode else "fast"
    print(f"🎤 Received /api/audio-to-midi request at {bpm} BPM ({mode_str} mode)")

    if not audioFile or not audioFile.filename:
        raise HTTPException(400, "No audio file provided")

    print(f"   Processing file: {audioFile.filename}")

    try:
        # Save uploaded audio temporarily
        upload_id = str(uuid.uuid4())
        temp_dir = ensure_path_exists(get_output_path('temp_recordings'))

        audio_path = temp_dir / f"{upload_id}_{audioFile.filename}"
        with open(audio_path, "wb") as f:
            f.write(await audioFile.read())

        print(f"   Saved to: {audio_path}")
        print(f"   File size: {audio_path.stat().st_size / 1024:.2f} KB")

        # Get actual audio duration first
        import librosa
        import soundfile as sf
        audio_duration = librosa.get_duration(path=str(audio_path))
        print(f"   🎵 Original audio duration: {audio_duration:.2f}s")

        # Use Basic Pitch directly for both modes
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        # Detailed mode: Apply slowdown for better pitch detection
        if detailedMode:
            # SLOWDOWN FEATURE: Slow down audio by 25% for better pitch detection
            slowdown_ratio = 0.25  # 25% slower
            speed_factor = 1.0 - slowdown_ratio  # 0.75x speed

            print(f"   🐌 Detailed mode: Slowing down audio by {slowdown_ratio*100:.0f}% ({speed_factor:.2f}x speed) for processing...")

            # Load audio and apply time stretching
            y, sr = librosa.load(str(audio_path), sr=None)
            y_slowed = librosa.effects.time_stretch(y, rate=speed_factor)

            # Save slowed audio to temporary file
            slowed_audio_path = temp_dir / f"{upload_id}_slowed_{audioFile.filename}"
            sf.write(str(slowed_audio_path), y_slowed, sr)
            print(f"   ✅ Slowed audio saved: {slowed_audio_path}")

            slowed_duration = len(y_slowed) / sr
            print(f"   🎵 Slowed audio duration: {slowed_duration:.2f}s (was {audio_duration:.2f}s)")

            # Extract MIDI using Basic Pitch on SLOWED audio
            print(f"   🎼 Running Basic Pitch inference on slowed audio...")
            model_output, midi_data, note_events = predict(
                str(slowed_audio_path),
                ICASSP_2022_MODEL_PATH,
                onset_threshold=0.55,      # Moderately stricter onset detection (reduces false onsets from vibrato)
                frame_threshold=0.35,      # Moderately stronger activation required (smooths out weak pitch variations)
                minimum_note_length=188    # Moderate note duration filtering (filters rapid vibrato fluctuations)
            )

            # Save MIDI directly to temp file
            import uuid
            midi_id = str(uuid.uuid4())[:8]
            temp_midi_dir = temp_dir / "midi_output"
            temp_midi_dir.mkdir(exist_ok=True)
            main_midi_path = str(temp_midi_dir / f"basic_pitch_detailed_{midi_id}.mid")

            # Write MIDI data
            with open(main_midi_path, 'wb') as f:
                midi_data.write(f)

            print(f"   ✅ Detailed MIDI extracted from slowed audio: {main_midi_path}")

            # Create result dict to match expected format
            result = {
                'main_midi': main_midi_path,
                'voice_midis': []
            }
        else:
            # Fast mode: Use Basic Pitch directly without subprocess
            print(f"   ⚡ Fast mode: Using Basic Pitch library directly (no subprocess)...")
            slowed_audio_path = None
            slowed_duration = audio_duration
            speed_factor = 1.0  # No slowdown

            # Use Basic Pitch directly for fast MIDI extraction
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH

            print(f"   🎼 Running Basic Pitch inference...")
            model_output, midi_data, note_events = predict(
                str(audio_path),
                ICASSP_2022_MODEL_PATH,
                onset_threshold=0.55,      # Moderately stricter onset detection (reduces false onsets from vibrato)
                frame_threshold=0.35,      # Moderately stronger activation required (smooths out weak pitch variations)
                minimum_note_length=188    # Moderate note duration filtering (filters rapid vibrato fluctuations)
            )

            # Save MIDI directly to temp file
            import uuid
            midi_id = str(uuid.uuid4())[:8]
            temp_midi_dir = temp_dir / "midi_output"
            temp_midi_dir.mkdir(exist_ok=True)
            main_midi_path = str(temp_midi_dir / f"basic_pitch_{midi_id}.mid")

            # Write MIDI data
            with open(main_midi_path, 'wb') as f:
                midi_data.write(f)

            print(f"   ✅ Fast MIDI extracted: {main_midi_path}")

            # Create result dict to match expected format
            result = {
                'main_midi': main_midi_path,
                'voice_midis': []
            }

        main_midi_path = result['main_midi']
        print(f"   ✅ MIDI extracted: {main_midi_path}")

        # Parse MIDI to get initial duration
        midi_obj = pretty_midi.PrettyMIDI(main_midi_path)
        midi_duration = midi_obj.get_end_time()

        # First: Match MIDI duration to slowed audio duration
        if midi_duration > 0:
            tempo_scale = slowed_duration / midi_duration
            correct_tempo = 120.0 / tempo_scale
            print(f"   🎵 MIDI duration: {midi_duration:.2f}s, Slowed audio duration: {slowed_duration:.2f}s")
            print(f"   🎵 Tempo scale: {tempo_scale:.3f}x, Correcting tempo to {correct_tempo:.1f} BPM")

            # Modify MIDI tempo to match slowed audio duration
            output_midi_path = str(Path(main_midi_path).parent / f"corrected_{Path(main_midi_path).name}")
            modify_midi_tempo(main_midi_path, output_midi_path, tempo_scale=tempo_scale)
            main_midi_path = output_midi_path

            # Re-parse with corrected tempo
            midi_obj = pretty_midi.PrettyMIDI(main_midi_path)
            duration = midi_obj.get_end_time()
            print(f"   🔍 After tempo correction, MIDI duration: {duration:.2f}s")
            tempo = correct_tempo
        else:
            duration = slowed_duration
            tempo = 120.0

        # SPEEDUP FEATURE: Only apply if detailed mode was used
        if detailedMode and speed_factor != 1.0:
            # We slowed down audio by speed_factor (0.75x), so MIDI is now too long
            # We need to speed up MIDI back to original duration
            # modify_midi_tempo: tempo_scale > 1.0 = speed up, < 1.0 = slow down
            # tempo_multiplier = 1/tempo_scale, so tempo_scale=1.33 → multiplier=0.75 → faster
            actual_speedup = 1.0 / speed_factor  # 1.33x - actual speed increase

            print(f"   🚀 Speeding up MIDI to match original audio duration...")
            print(f"   🔍 Original audio: {audio_duration:.2f}s, Slowed audio: {slowed_duration:.2f}s")
            print(f"   🔍 MIDI duration before speedup: {duration:.2f}s")
            print(f"   🔍 Applying tempo_scale={actual_speedup:.3f} to speed up by {actual_speedup:.2f}x")

            # Speed up MIDI - pass the inverse of slowdown factor
            sped_up_midi_path = str(Path(main_midi_path).parent / f"speedup_{Path(main_midi_path).name}")
            modify_midi_tempo(main_midi_path, sped_up_midi_path, tempo_scale=actual_speedup)
            main_midi_path = sped_up_midi_path

            # Re-parse final MIDI
            midi_obj = pretty_midi.PrettyMIDI(main_midi_path)
            duration = midi_obj.get_end_time()
            tempo = bpm  # Use timeline BPM as final tempo

            print(f"   ✅ Final MIDI duration: {duration:.2f}s (target: {audio_duration:.2f}s)")
            print(f"   ✅ Final tempo: {tempo:.1f} BPM")
        else:
            # Fast mode: No speedup needed
            midi_obj = pretty_midi.PrettyMIDI(main_midi_path)
            duration = midi_obj.get_end_time()
            print(f"   ✅ Fast mode MIDI duration: {duration:.2f}s")
            print(f"   ✅ Tempo: {tempo:.1f} BPM")

        # Read corrected MIDI file as base64 for transmission
        with open(main_midi_path, 'rb') as f:
            midi_data = base64.b64encode(f.read()).decode('utf-8')

        total_notes = sum(len(inst.notes) for inst in midi_obj.instruments if not inst.is_drum)

        # Get note range
        all_notes = [note for inst in midi_obj.instruments if not inst.is_drum for note in inst.notes]
        if all_notes:
            min_pitch = min(note.pitch for note in all_notes)
            max_pitch = max(note.pitch for note in all_notes)
        else:
            min_pitch = max_pitch = 60  # Default to middle C

        print(f"   📊 MIDI info: {total_notes} notes, {duration:.2f}s, tempo {tempo:.1f} BPM")
        print(f"   🎹 Note range: {min_pitch} - {max_pitch}")

        # Clean up temp files
        audio_path.unlink()
        if slowed_audio_path and slowed_audio_path.exists():
            slowed_audio_path.unlink()

        return {
            "success": True,
            "midi_data": midi_data,  # Base64 encoded MIDI file
            "filename": f"recording_{upload_id}.mid",
            "metadata": {
                "duration": duration,
                "total_notes": total_notes,
                "tempo": tempo,
                "min_pitch": min_pitch,
                "max_pitch": max_pitch
            }
        }

    except Exception as e:
        print(f"❌ Audio-to-MIDI conversion error: {e}")
        import traceback
        traceback.print_exc()

        # Clean up temp files if they exist
        if 'audio_path' in locals() and audio_path.exists():
            audio_path.unlink()
        if 'slowed_audio_path' in locals() and slowed_audio_path.exists():
            slowed_audio_path.unlink()

        raise HTTPException(500, f"Audio-to-MIDI conversion error: {str(e)}")

# Cache loaded VST plugins to avoid slow Wine initialization
_VST_PLUGIN_CACHE = {}

def get_cached_plugin(plugin_path: str):
    """Load and cache VST plugins to avoid slow Wine/yabridge initialization"""
    global _VST_PLUGIN_CACHE

    # Normalize path
    plugin_path = str(Path(plugin_path).resolve())

    if plugin_path not in _VST_PLUGIN_CACHE:
        from pedalboard import load_plugin
        print(f"   🔌 Loading plugin (first time): {plugin_path}")
        print(f"   📊 Cache size before: {len(_VST_PLUGIN_CACHE)} plugins")
        _VST_PLUGIN_CACHE[plugin_path] = load_plugin(plugin_path)
        print(f"   📊 Cache size after: {len(_VST_PLUGIN_CACHE)} plugins")
    else:
        print(f"   ⚡ Using cached plugin: {plugin_path}")
        print(f"   📊 Current cache has {len(_VST_PLUGIN_CACHE)} plugins")

    return _VST_PLUGIN_CACHE[plugin_path]

@app.post("/api/download-with-fx")
async def download_with_fx(
    audioFile: UploadFile = File(...),
    trackFX: str = Form("{}"),
    busFX: str = Form("{}"),
    masterFX: str = Form("{}"),
    rc20FX: str = Form("{}"),
    speccraftFX: str = Form("{}")
):
    """
    Process audio through VST plugins and return the processed file.
    Supports RC-20 Retro Color and SpecCraft plugins.
    Uses plugin caching for fast repeated processing.
    """
    import uuid
    from pedalboard import Pedalboard
    from pedalboard.io import AudioFile

    print(f"🎛️ Received /api/download-with-fx request")
    print(f"   File: {audioFile.filename}")

    try:
        # Parse FX parameters
        track_fx = json.loads(trackFX)
        bus_fx = json.loads(busFX)
        master_fx = json.loads(masterFX)
        rc20_fx = json.loads(rc20FX)
        speccraft_fx = json.loads(speccraftFX)

        print(f"   Track FX: {track_fx}")
        print(f"   Bus FX: {bus_fx}")
        print(f"   Master FX: {master_fx}")
        print(f"   RC20 FX: {rc20_fx}")
        print(f"   SpecCraft FX: {speccraft_fx}")

        # Save uploaded audio to temp file
        upload_id = str(uuid.uuid4())
        temp_dir = ensure_path_exists(get_output_path('temp_fx_processing'))

        input_path = temp_dir / f"{upload_id}_input.wav"
        with open(input_path, "wb") as f:
            f.write(await audioFile.read())

        print(f"   💾 Saved input to: {input_path}")

        # Load audio
        with AudioFile(str(input_path)) as f:
            audio = f.read(f.frames)
            sample_rate = f.samplerate
            num_channels = f.num_channels

        print(f"   🎵 Audio: {num_channels} channels, {sample_rate}Hz, {len(audio[0])/sample_rate:.2f}s")

        # Create pedalboard with plugins
        board_plugins = []

        # Add RC-20 Retro Color if enabled
        if rc20_fx.get('enabled', False):
            rc20_path = "/home/arlo/.vst3/yabridge/RC-20 Retro Color.vst3"
            rc20 = get_cached_plugin(rc20_path)

            # Apply RC-20 parameters
            if 'magnitude' in rc20_fx:
                rc20.magnitude = float(rc20_fx['magnitude'])
                print(f"      magnitude = {rc20_fx['magnitude']}")
            if 'noise' in rc20_fx:
                rc20.nois_amount = float(rc20_fx['noise'])
                print(f"      noise = {rc20_fx['noise']}")
            if 'wobble' in rc20_fx:
                rc20.wobb_amount = float(rc20_fx['wobble'])
                print(f"      wobble = {rc20_fx['wobble']}")
            if 'distortion' in rc20_fx:
                rc20.dist_amount = float(rc20_fx['distortion'])
                print(f"      distortion = {rc20_fx['distortion']}")
            if 'magnitudeAmount' in rc20_fx:
                rc20.magn_amount = float(rc20_fx['magnitudeAmount'])
                print(f"      magnitude_amount = {rc20_fx['magnitudeAmount']}")

            board_plugins.append(rc20)
            print("   ✅ RC-20 plugin configured")

        # Add SpecCraft if enabled
        if speccraft_fx.get('enabled', False):
            speccraft_path = "/home/arlo/.vst3/yabridge/SpecCraft(64).vst3"
            speccraft = get_cached_plugin(speccraft_path)

            # Apply SpecCraft compression parameters
            if 'threshold' in speccraft_fx:
                speccraft.threshold_db = float(speccraft_fx['threshold'])
                print(f"      threshold = {speccraft_fx['threshold']} dB")
            if 'ratio' in speccraft_fx:
                speccraft.ratio = float(speccraft_fx['ratio'])
                print(f"      ratio = {speccraft_fx['ratio']}%")
            if 'attack' in speccraft_fx:
                speccraft.attack = float(speccraft_fx['attack'])
                print(f"      attack = {speccraft_fx['attack']} ms")
            if 'knee' in speccraft_fx:
                speccraft.knee = float(speccraft_fx['knee'])
                print(f"      knee = {speccraft_fx['knee']} dB")
            if 'slope' in speccraft_fx:
                speccraft.slope_db = float(speccraft_fx['slope'])
                print(f"      slope = {speccraft_fx['slope']} dB")
            if 'release' in speccraft_fx:
                # SpecCraft uses specific release strings - map to nearest valid value
                release_ms = float(speccraft_fx['release'])
                # Extract valid release values from the actual plugin (these are the EXACT values it accepts)
                # Based on the error message, valid values are very specific decimals
                valid_releases = [
                    10.0, 10.2, 10.3, 10.5, 10.7, 10.8, 11.0, 11.2, 11.3, 11.5, 11.6, 11.8,
                    12.0, 12.1, 12.3, 12.5, 12.6, 12.8, 13.0, 13.1, 13.3, 13.4, 13.6, 13.8, 13.9,
                    14.1, 14.3, 14.4, 14.6, 14.8, 14.9, 15.1, 15.3, 15.4, 15.6, 15.8, 15.9,
                    16.1, 16.2, 16.4, 16.6, 16.7, 16.9, 17.1, 17.2, 17.4, 17.6, 17.7, 17.9,
                    18.0, 18.2, 18.4, 18.5, 18.7, 18.9, 19.0, 19.2, 19.4, 19.5, 19.7, 19.9,
                    20.0, 20.2, 20.4, 20.5, 20.7, 20.8, 21.0, 21.2, 21.3, 21.5, 21.8, 22.1,
                    22.4, 22.8, 23.1, 23.4, 23.7, 24.0, 24.3, 24.6, 25.0, 25.3, 25.6, 25.9,
                    26.2, 26.5, 26.8, 27.1, 27.5, 27.8, 28.1, 28.4, 28.7, 29.0, 29.3, 29.7,
                    30.0, 30.3, 30.6, 30.9, 31.2, 31.5, 31.9, 32.2, 32.5, 32.8, 33.1, 33.4,
                    33.7, 34.0, 34.4, 34.7, 35.0, 35.3, 35.6, 35.9, 36.2, 36.6, 36.9, 37.2,
                    37.5, 37.8, 38.1, 38.4, 38.8, 39.1, 39.4, 39.7, 40.0, 40.3, 40.6, 40.9,
                    41.3, 41.6, 41.9, 42.2, 42.5, 42.8, 43.1, 43.5, 43.8, 44.1, 44.4, 44.7,
                    45.0, 45.3, 45.6, 46.0, 46.3, 46.6, 46.9, 47.2, 47.5, 47.8, 48.2, 48.5,
                    48.8, 49.1, 49.4, 49.7, 50.0, 50.4, 50.7, 51.0, 51.3, 51.6, 51.9, 52.2,
                    52.5, 52.9, 53.2, 53.5, 53.8, 54.1, 54.4, 54.7, 55.1, 55.4, 55.7, 56.0,
                    56.5, 57.0, 57.5, 58.0, 58.5, 59.1, 59.6, 60.1, 60.6, 61.1, 61.6, 62.1,
                    62.6, 63.1, 63.6, 64.1, 64.7, 65.2, 65.7, 66.2, 66.7, 67.2, 67.7, 68.2,
                    68.7, 69.2, 69.7, 70.3, 70.8, 71.3, 71.8, 72.3, 72.8, 73.3, 73.8, 74.3,
                    74.8, 75.3, 75.9, 76.4, 76.9, 77.4, 77.9, 78.4, 78.9, 79.4, 79.9, 80.4,
                    80.9, 81.5, 82.0, 82.5, 83.0, 83.5, 84.0, 84.5, 85.0, 85.5, 86.0, 86.5,
                    87.1, 87.6, 88.1, 88.6, 89.1, 89.6, 90.1, 90.6, 91.1, 91.6, 92.1, 92.7,
                    93.2, 93.7, 94.2, 94.7, 95.2, 95.7, 96.2, 96.7, 97.2, 97.7, 98.3, 98.8,
                    99.3, 99.8, 100.3, 100.8, 101.3, 101.8, 102.3, 102.8, 103.3, 103.9, 104.4,
                    104.9, 105.4, 105.9, 106.4, 106.9, 107.4, 107.9, 108.4, 108.9, 109.5, 110.0,
                    110.5, 111.0, 111.5, 112.0, 112.8, 113.7, 114.5, 115.3, 116.1, 117.0, 117.8,
                    118.6, 119.5, 120.3, 150.2, 180.1, 210.7, 250.4, 300.0, 350.0, 400.0, 450.0,
                    500.0, 501.0, 600.0, 700.0, 800.0, 900.0, 1000.0
                ]
                # Find nearest valid value
                nearest = min(valid_releases, key=lambda x: abs(x - release_ms))
                speccraft.release = f"{nearest}ms"
                print(f"      release = {release_ms} ms (mapped to {nearest}ms)")
            if 'makeup' in speccraft_fx:
                speccraft.makeup_db = float(speccraft_fx['makeup'])
                print(f"      makeup = {speccraft_fx['makeup']} dB")

            # Apply spectral shaping bands
            for i in range(1, 5):
                band_key = f'band{i}'
                if band_key in speccraft_fx and speccraft_fx[band_key].get('enabled', False):
                    band = speccraft_fx[band_key]
                    setattr(speccraft, f'shaper_band{i}_used', True)
                    setattr(speccraft, f'shaper_band{i}_enabled', True)
                    if 'freq' in band:
                        setattr(speccraft, f'shaper_band{i}_freq_hz', float(band['freq']))
                    if 'gain' in band:
                        setattr(speccraft, f'shaper_band{i}_gain_db', float(band['gain']))
                    if 'q' in band:
                        setattr(speccraft, f'shaper_band{i}_q', float(band['q']))
                    print(f"      band{i}: {band.get('freq', 1000)}Hz, {band.get('gain', 0)}dB, Q={band.get('q', 1.0)}")

            board_plugins.append(speccraft)
            print("   ✅ SpecCraft plugin configured")

        # Process audio through plugins
        if board_plugins:
            import time
            print(f"   🔊 Processing audio through {len(board_plugins)} plugin(s)...")
            start_time = time.time()
            board = Pedalboard(board_plugins)
            processed = board(audio, sample_rate)
            elapsed = time.time() - start_time
            print(f"   ⏱️  Processing took {elapsed:.2f} seconds")
        else:
            print("   ℹ️  No plugins enabled, returning original audio")
            processed = audio

        # Save processed audio
        output_path = temp_dir / f"{upload_id}_output.wav"
        with AudioFile(str(output_path), 'w', sample_rate, num_channels) as f:
            f.write(processed)

        print(f"   💾 Saved output to: {output_path}")
        print(f"   ✅ Processing complete!")

        # Return the processed file
        response = FileResponse(
            path=str(output_path),
            media_type="audio/wav",
            filename=f"processed_{audioFile.filename}"
        )

        # Clean up input file (keep output for FileResponse to serve)
        input_path.unlink()

        # Schedule cleanup of output file after response is sent
        # (FileResponse will handle the file, we can delete it after a delay)
        import asyncio
        async def cleanup_later():
            await asyncio.sleep(60)  # Wait 60 seconds
            if output_path.exists():
                output_path.unlink()
                print(f"   🗑️  Cleaned up: {output_path}")

        asyncio.create_task(cleanup_later())

        return response

    except Exception as e:
        print(f"❌ Error processing audio with FX: {e}")
        import traceback
        traceback.print_exc()

        # Clean up temp files
        if 'input_path' in locals() and input_path.exists():
            input_path.unlink()
        if 'output_path' in locals() and output_path.exists():
            output_path.unlink()

        raise HTTPException(500, f"Audio FX processing error: {str(e)}")

@app.post("/api/render-omnisphere")
async def render_omnisphere_midi(
    midiFile: UploadFile = File(...),
    patch: int = Form(0),
    sampleRate: int = Form(44100),
    tailDuration: float = Form(3.0)
):
    """
    Render MIDI through Omnisphere VST3 instrument plugin.

    Args:
        midiFile: MIDI file to render
        patch: Omnisphere patch/program number (0-127)
        sampleRate: Output sample rate in Hz
        tailDuration: Extra seconds after MIDI ends for reverb/delay tails

    Returns:
        FileResponse with rendered audio WAV file
    """
    import uuid
    from render_omnisphere import render_midi_auto

    print(f"🎹 Received /api/render-omnisphere request")
    print(f"   MIDI File: {midiFile.filename}")
    print(f"   Patch: {patch}")
    print(f"   Sample Rate: {sampleRate}Hz")
    print(f"   Tail Duration: {tailDuration}s")

    try:
        # Create temp directory for processing
        upload_id = str(uuid.uuid4())
        temp_dir = ensure_path_exists(get_output_path('temp_omnisphere'))

        # Save uploaded MIDI file
        input_midi_path = temp_dir / f"{upload_id}_input.mid"
        with open(input_midi_path, "wb") as f:
            f.write(await midiFile.read())

        print(f"   💾 Saved MIDI to: {input_midi_path}")

        # Render through Omnisphere (auto-detects best backend)
        output_audio_path = temp_dir / f"{upload_id}_output.wav"

        render_midi_auto(
            midi_path=str(input_midi_path),
            output_path=str(output_audio_path),
            patch_number=patch,
            sample_rate=sampleRate,
            tail_duration=tailDuration,
            verbose=True
        )

        print(f"   ✅ Rendering complete!")

        # Return the rendered audio file
        response = FileResponse(
            path=str(output_audio_path),
            media_type="audio/wav",
            filename=f"omnisphere_{midiFile.filename.replace('.mid', '.wav')}"
        )

        # Clean up input file
        input_midi_path.unlink()

        # Schedule cleanup of output file after response is sent
        import asyncio
        async def cleanup_later():
            await asyncio.sleep(60)  # Wait 60 seconds
            if output_audio_path.exists():
                output_audio_path.unlink()
                print(f"   🗑️  Cleaned up: {output_audio_path}")

        asyncio.create_task(cleanup_later())

        return response

    except Exception as e:
        print(f"❌ Error rendering MIDI with Omnisphere: {e}")
        import traceback
        traceback.print_exc()

        # Clean up temp files
        if 'input_midi_path' in locals() and input_midi_path.exists():
            input_midi_path.unlink()
        if 'output_audio_path' in locals() and output_audio_path.exists():
            output_audio_path.unlink()

        raise HTTPException(500, f"Omnisphere rendering error: {str(e)}")

@app.post("/api/generate-track-image")
async def generate_track_image(
    instrumentGroup: str = Form(...),
    instrumentSubgroup: str = Form(None),
    trackName: str = Form(None)
):
    """
    Generate an AI image for a track using DALL-E based on instrument type.
    Returns the image URL that can be displayed in the frontend.
    """
    try:
        # Set OpenAI API key (read from ac.py or environment)
        openai_key = "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"

        # Create OpenAI client with new API
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)

        print(f"🎨 Generating image for instrument: {instrumentGroup} / {instrumentSubgroup}")

        # Create a detailed prompt based on instrument type
        instrument_prompts = {
            "piano": "A beautiful grand piano in a concert hall with warm lighting, professional music photography, highly detailed",
            "guitar": "An elegant acoustic guitar with beautiful wood grain, studio lighting, professional product photography",
            "bass": "A sleek electric bass guitar with rich wood finish, professional studio photography, dramatic lighting",
            "strings": f"A professional {instrumentSubgroup or 'violin'} in a classical music setting, warm concert hall lighting, highly detailed",
            "brass": f"A gleaming {instrumentSubgroup or 'trumpet'} brass instrument, professional studio photography with dramatic lighting",
            "winds": f"An elegant {instrumentSubgroup or 'saxophone'} woodwind instrument, professional photography with soft lighting",
            "drums": "A professional drum kit in a modern studio, warm lighting, high-end music photography"
        }

        # Get the appropriate prompt
        base_prompt = instrument_prompts.get(
            instrumentGroup.lower(),
            f"A professional {instrumentGroup} musical instrument, studio photography, dramatic lighting"
        )

        # Add subgroup specificity if available
        if instrumentSubgroup and instrumentSubgroup not in ["undefined", "keys"]:
            subgroup_name = instrumentSubgroup.replace("_", " ")
            base_prompt = f"A professional {subgroup_name} musical instrument, concert hall setting, warm dramatic lighting, highly detailed, professional photography"

        # Add style modifiers for better results
        full_prompt = f"{base_prompt}, 4k resolution, cinematic, artistic, elegant composition"

        print(f"   Prompt: {full_prompt}")

        # Call DALL-E API using new OpenAI 2.x client
        response = client.images.generate(
            model="dall-e-3",
            prompt=full_prompt,
            n=1,
            size="1024x1024"
        )

        image_url = response.data[0].url
        print(f"✅ Generated image URL: {image_url}")

        # Download and save the image locally for persistent storage
        process_id = str(uuid.uuid4())
        output_dir = ensure_path_exists(get_output_path('images', process_id=process_id))

        image_filename = f"{instrumentGroup}_{instrumentSubgroup or 'default'}.png"
        image_path = output_dir / image_filename

        # Download the image
        img_response = http_requests.get(image_url)
        with open(image_path, 'wb') as f:
            f.write(img_response.content)

        local_url = f"/download-image/{process_id}/{image_filename}"
        print(f"💾 Saved image locally: {local_url}")

        return {
            "status": "success",
            "imageUrl": local_url,
            "originalUrl": image_url,
            "instrumentGroup": instrumentGroup,
            "instrumentSubgroup": instrumentSubgroup
        }

    except Exception as e:
        print(f"❌ Image generation error: {str(e)}")
        raise HTTPException(500, f"Image generation failed: {str(e)}")

@app.get("/download-image/{process_id}/{filename}")
async def download_image(process_id: str, filename: str):
    """Serve generated track images"""
    file_path = get_output_path('images', process_id=process_id) / filename
    if not file_path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(file_path, media_type="image/png")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    global MODEL, GROUP_NAMES, SUBGROUP_NAMES, MANIFEST_PATHS, MANIFEST_DATA

    DEFAULT_CKPT = "/mnt/msdd/exps/logs_v2/lightning_logs/2025-09-06_16-12-31_all_groups_ft_v3_capivotpitch_ctrl/checkpoints/last.ckpt"
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--manifest", required=False)  # Now optional
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    # Manifest not needed - removed random file selection feature
    MANIFEST_DATA = []
    MANIFEST_PATHS = []
    print(f"ℹ️  Manifest loading disabled (random file selection feature removed)")

    print("--- Initializing model ---")
    MODEL = load_model_any_ckpt(args.checkpoint, args.checkpoint_dir, args.manifest)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL.to(dev).eval()
    print(f"✅ Model on {dev}")

    GROUP_NAMES = list(APPROVED_GROUPS) if not isinstance(APPROVED_GROUPS, dict) else list(APPROVED_GROUPS.keys())
    SUBGROUP_NAMES = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})
    print(f"Groups: {len(GROUP_NAMES)} | Subgroups: {len(SUBGROUP_NAMES)}")

    ui = create_ui()
    ui.launch(share=args.share, server_name="0.0.0.0", server_port=7860)

# FastAPI should be run with: uvicorn genfrominterface:app --host 0.0.0.0 --port 8000
# Celery worker should be run with: celery -A genfrominterface worker --loglevel=info
# Gradio UI (optional) can be run by uncommenting below:

# if __name__ == "__main__":
#     main()
 

# ------------------------------------------------------------------------------
# DrumSampler API Endpoints (replaces HuggingFace Space)
# ------------------------------------------------------------------------------

# Import the drum sampler module (optional)
try:
    import sys
    sys.path.insert(0, "/home/arlo/harmonymodule/harmonymodule")
    from drum_sampler_simple import SimpleDrumSampler
    DRUM_SAMPLER_AVAILABLE = True
except ImportError:
    SimpleDrumSampler = None
    DRUM_SAMPLER_AVAILABLE = False
    print("⚠️  Warning: drum_sampler_simple not available - drum sampling features will be disabled")

# Global drum sampler instance
_drum_sampler = None

def get_drum_sampler():
    """Get or create the global drum sampler instance"""
    global _drum_sampler
    if not DRUM_SAMPLER_AVAILABLE:
        raise ImportError("drum_sampler_simple module is not available")
    if _drum_sampler is None:
        _drum_sampler = SimpleDrumSampler()
    return _drum_sampler

@app.post("/api/drum-sampler/randomize")
async def randomize_drum_pattern():
    """
    Randomize drum pattern - returns a random MIDI file and kit name
    Mimics the /randomize_drums endpoint from HuggingFace Space
    """
    try:
        import random
        import glob
        
        # Get list of available MIDI files
        midi_dir = "/home/arlo/harmonymodule/drummidis"
        midi_files = glob.glob(os.path.join(midi_dir, "*.mid"))
        
        if not midi_files:
            raise HTTPException(404, "No MIDI files found")
        
        # Select random MIDI file
        selected_midi = random.choice(midi_files)
        midi_filename = os.path.basename(selected_midi)
        
        print(f"🎲 Randomized drum pattern: {midi_filename}")
        
        return {
            "success": True,
            "midiFile": midi_filename,
            "drumKit": "default",  # Can be expanded later with actual kit names
            "midiPath": selected_midi
        }
    
    except Exception as e:
        print(f"❌ Error randomizing drums: {str(e)}")
        raise HTTPException(500, f"Randomization failed: {str(e)}")


@app.post("/api/drum-sampler/render")
async def render_drum_midi(
    midiFile: str = Form(...),
    bpm: int = Form(120)
):
    """
    Render MIDI file with drum samples
    Mimics the /render_midi_drums endpoint from HuggingFace Space
    Returns audio file that can be downloaded
    """
    try:
        # Find the MIDI file
        midi_dir = "/home/arlo/harmonymodule/drummidis"
        midi_path = os.path.join(midi_dir, midiFile)
        
        if not os.path.exists(midi_path):
            raise HTTPException(404, f"MIDI file not found: {midiFile}")
        
        print(f"🎵 Rendering drum MIDI: {midiFile}")
        
        # Create output directory
        process_id = str(uuid.uuid4())
        output_dir = ensure_path_exists(get_output_path('drums', process_id=process_id))
        
        # Generate output filename
        output_filename = f"drums_{os.path.splitext(midiFile)[0]}.wav"
        output_path = str(output_dir / output_filename)
        
        # Get drum sampler and render with random kit at specified BPM
        sampler = get_drum_sampler()
        sr, audio, kit_used = sampler.render_midi(midi_path, output_path, bpm=bpm)
        
        # Calculate duration
        duration = len(audio) / sr
        
        print(f"✅ Rendered drum audio: {output_filename} ({duration:.2f}s)")
        
        # Return download URL
        download_url = f"/download-drums/{process_id}/{output_filename}"
        
        return {
            "success": True,
            "audioUrl": download_url,
            "fileName": output_filename,
            "duration": duration,
            "sampleRate": sr,
            "kitUsed": kit_used
        }
    
    except Exception as e:
        print(f"❌ Error rendering drum MIDI: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Render failed: {str(e)}")


@app.get("/download-drums/{process_id}/{filename}")
async def download_drum_audio(process_id: str, filename: str):
    """Serve rendered drum audio files"""
    file_path = get_output_path('drums', process_id=process_id) / filename
    if not file_path.exists():
        raise HTTPException(404, "Drum audio file not found")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

@app.post("/api/generate-ace-step")
async def generate_ace_step_endpoint(
    steps: int = Form(100),
    prompt: str = Form(""),
    lyrics: str = Form(""),
    key: str = Form("C"),
    seed: int = Form(0),
    noise_level: float = Form(0.8),
    ref_audio: UploadFile = File(None),
    midi_lyric_map: str = Form(None),
    detailed_mode: bool = Form(False)
):
    """
    FastAPI endpoint for ACE-Step text-to-music generation

    This endpoint calls the ace_step_noise_wrapper.py script to generate music
    using the ACE-Step model with noise level control.

    Parameters:
        steps: Number of inference steps (default: 100)
        prompt: Text description of the music
        lyrics: Optional lyrics with structure tags
        key: Musical key (C, C#, D, etc.)
        seed: Random seed for reproducibility (default: 0)
        noise_level: Noise level for GT mixing, 0.0=pure GT, 1.0=pure noise (default: 0.8)
        ref_audio: Optional reference audio file for noise mixing
        detailed_mode: If True, use phrase-by-phrase generation with noise-to-noise (default: False)
    """
    import subprocess
    import uuid
    from pathlib import Path
    import time

    print(f"\n{'='*80}")
    mode_str = "DETAILED MODE (Phrase-by-Phrase)" if detailed_mode else "Standard Mode"
    print(f"📥 RECEIVED /api/generate-ace-step REQUEST ({mode_str})")
    print(f"{'='*80}")
    print(f"   steps: {steps}")
    print(f"   prompt: {prompt}")
    print(f"   lyrics: {lyrics[:100]}..." if len(lyrics) > 100 else f"   lyrics: {lyrics}")
    print(f"   key: {key}")
    print(f"   seed: {seed}")
    print(f"   noise_level: {noise_level}")
    print(f"   ref_audio: {ref_audio.filename if ref_audio else 'None'}")
    print(f"   detailed_mode: {detailed_mode}")
    print(f"{'='*80}\n")
    
    try:
        # Generate unique process ID
        process_id = str(uuid.uuid4())[:8]

        # Create output directory
        output_dir = Path(f"./generated_ui/ace_step_{process_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output path
        output_path = output_dir / f"output_{int(time.time())}.wav"

        # Save reference audio file if provided
        ref_audio_path = None
        if ref_audio:
            ref_audio_path = output_dir / f"ref_audio_{ref_audio.filename}"
            with open(ref_audio_path, "wb") as f:
                content = await ref_audio.read()
                f.write(content)
            print(f"💾 Saved reference audio: {ref_audio_path}")

            # Check if uploaded file is a MIDI file - render it through FluidSynth or WORLD vocoder
            if ref_audio_path.suffix.lower() in ['.mid', '.midi']:
                print(f"\n{'='*80}")
                print(f"🎹 MIDI FILE DETECTED")
                print(f"{'='*80}")
                print(f"   Input MIDI: {ref_audio_path.name}")
                print(f"   Lyrics provided: {'Yes' if lyrics else 'No'}")
                print(f"   MIDI lyric map provided: {'Yes' if midi_lyric_map else 'No'}")

                # Create rendered audio path
                rendered_audio_path = output_dir / f"rendered_{ref_audio_path.stem}.wav"

                # If we have lyrics and a MIDI lyric map, use WORLD vocoder for syllable alignment
                import subprocess
                import json
                import os

                if lyrics and midi_lyric_map:
                    print(f"\n   🎤 Using WORLD Vocoder with syllable alignment")
                    print(f"{'='*80}")

                    try:
                        # Parse MIDI lyric map
                        note_syllable_map = json.loads(midi_lyric_map)
                        print(f"   📝 MIDI-Lyrics mapping ({len(note_syllable_map)} notes):")
                        for note_idx, syllable in sorted(note_syllable_map.items(), key=lambda x: int(x[0])):
                            print(f"      Note {note_idx}: '{syllable}'")

                        # Create temporary JSON file for the mapping
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                            json.dump(note_syllable_map, f)
                            mapping_file = f.name

                        try:
                            # Call WORLD vocoder
                            cmd = [
                                'python3',
                                '/home/arlo/Data/espeak_world_vocoder_aligned.py',
                                '--midi', str(ref_audio_path),
                                '--lyrics-map', mapping_file,
                                '--output', str(rendered_audio_path)
                            ]

                            print(f"\n   🚀 Running WORLD vocoder...")
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                            if result.returncode == 0:
                                print(f"\n   ✅ WORLD vocoder synthesis complete!")
                                print(f"   Output: {rendered_audio_path.name}")
                                print(f"{'='*80}\n")
                                ref_audio_path = rendered_audio_path
                            else:
                                print(f"\n   ❌ WORLD vocoder failed!")
                                print(f"   STDERR: {result.stderr}")
                                print(f"   Falling back to FluidSynth rendering...")
                                raise Exception("WORLD vocoder failed")

                        finally:
                            # Cleanup temp file
                            if os.path.exists(mapping_file):
                                os.remove(mapping_file)

                    except Exception as e:
                        print(f"   ⚠️  WORLD vocoder error: {e}")
                        print(f"   Falling back to FluidSynth with vocals soundfont...")

                        # Fallback to FluidSynth
                        soundfont_path = "/home/arlo/Data/soundfonts/vocals1.sf2"
                        try:
                            subprocess.run([
                                "fluidsynth", "-ni", "-g", "0.625", "-r", "44100", "-F", str(rendered_audio_path),
                                soundfont_path, str(ref_audio_path)
                            ], check=True, capture_output=True)

                            print(f"   ✅ Rendered with FluidSynth: {rendered_audio_path.name}")
                            print(f"{'='*80}\n")
                            ref_audio_path = rendered_audio_path

                        except subprocess.CalledProcessError as fs_error:
                            print(f"   ❌ FluidSynth also failed: {fs_error}")
                            raise HTTPException(500, f"Both WORLD vocoder and FluidSynth failed")

                else:
                    # No lyrics mapping - use FluidSynth with vocals soundfont
                    print(f"\n   🎹 Using FluidSynth with vocals soundfont (no lyrics mapping)")
                    print(f"{'='*80}")

                    soundfont_path = "/home/arlo/Data/soundfonts/vocals1.sf2"

                    try:
                        subprocess.run([
                            "fluidsynth", "-ni", "-g", "0.625", "-r", "44100", "-F", str(rendered_audio_path),
                            soundfont_path, str(ref_audio_path)
                        ], check=True, capture_output=True)

                        print(f"   ✅ Rendered MIDI to audio: {rendered_audio_path.name}")
                        print(f"   Using soundfont: {soundfont_path}")
                        print(f"{'='*80}\n")

                        # Replace ref_audio_path with rendered audio path
                        ref_audio_path = rendered_audio_path

                    except subprocess.CalledProcessError as e:
                        print(f"   ❌ FluidSynth rendering failed: {e}")
                        print(f"   STDERR: {e.stderr.decode() if e.stderr else 'N/A'}")
                        print(f"{'='*80}\n")
                        raise HTTPException(500, f"MIDI rendering failed: {e}")

        # Convert ref_audio_path to string for Celery
        ref_audio_path_str = str(ref_audio_path) if ref_audio_path else None

        # Queue the simple ACE-Step task via Celery
        print(f"🚀 Queueing ACE-Step Celery task...")
        task = generate_simple_ace_step_task.delay(
            prompt=prompt,
            lyrics=lyrics,
            steps=steps,
            key=key,
            duration=30.0,
            seed=seed,
            noise_level=noise_level,
            ref_audio_path=ref_audio_path_str
        )

        print(f"✅ Task queued with ID: {task.id}")

        # Return task ID for status polling
        return {
            "task_id": task.id,
            "status": "queued",
            "message": "ACE-Step generation task queued"
        }

    except Exception as e:
        print(f"❌ Error in ACE-Step generation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Generation failed: {str(e)}")


@app.get("/api/generate-ace-step/task/{task_id}")
async def get_ace_step_task_status(task_id: str):
    """Check ACE-Step task status"""
    task = celery_app.AsyncResult(task_id)
    print(f"📊 ACE-Step Task {task_id} state: {task.state}")

    if task.state == "SUCCESS":
        result = task.result
        print(f"📊 ACE-Step Task result: {result}")

        # ACE-Step returns {"file_paths": [...], "input_files": []}
        # Frontend expects {status: 'completed', result: {file_paths: [...], input_files: {}}}
        if isinstance(result, dict):
            file_paths = result.get("file_paths", [])
            input_files = result.get("input_files", [])
            print(f"✅ Returning {len(file_paths)} file(s): {file_paths}")
            return {
                "status": "completed",
                "result": {
                    "file_paths": file_paths,
                    "input_files": input_files
                }
            }
        else:
            print(f"⚠️ Unexpected result format: {result}")
            return {
                "status": "completed",
                "result": {
                    "file_paths": [],
                    "input_files": []
                }
            }
    elif task.state == "FAILURE":
        print(f"❌ ACE-Step Task failed: {task.info}")
        return {"status": "failed", "error": str(task.info)}
    elif task.state == "PENDING":
        return {"status": "queued"}
    else:
        # STARTED, RETRY, etc.
        return {"status": "processing"}


@app.get("/download-ace-step/{process_id}/{filename}")
async def download_ace_step_audio(process_id: str, filename: str):
    """
    Serve ACE-Step generated audio files.
    Converts 32-bit float PCM to 16-bit PCM for browser compatibility.
    """
    import tempfile
    import torchaudio

    # Try local file first
    file_path = Path(f"./generated_ui/ace_step_{process_id}") / filename
    source_path = None

    if file_path.exists():
        source_path = file_path
    else:
        # Try GCS fallback
        try:
            from gcs_storage import gcs_file_exists, download_from_gcs, get_gcs_path

            gcs_path = get_gcs_path(filename, prefix="audiofiles", user_id="ace_step")
            if gcs_file_exists(gcs_path):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    source_path = Path(tmp.name)
                download_from_gcs(gcs_path, str(source_path))
        except Exception as e:
            print(f"⚠️  GCS fallback failed: {e}")

    if not source_path:
        raise HTTPException(404, "ACE-Step audio file not found")

    # Convert to 16-bit PCM for browser compatibility
    try:
        waveform, sample_rate = torchaudio.load(str(source_path))

        # Create temporary file for converted audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            converted_path = tmp.name

        # Save as 16-bit PCM
        torchaudio.save(
            converted_path,
            waveform,
            sample_rate,
            encoding="PCM_S",
            bits_per_sample=16
        )

        return FileResponse(converted_path, media_type="audio/wav", filename=filename)
    except Exception as e:
        print(f"❌ Error converting audio: {e}")
        # Fallback to serving original file
        return FileResponse(str(source_path), media_type="audio/wav", filename=filename)



@app.post("/api/render-chords")
async def render_chords(request: Request):
    """
    Render chord progression as MIDI file
    Input: { chords: {}, bpm: 120, duration: 16 }
    Returns: MIDI file path and metadata
    """
    try:
        data = await request.json()
        chords = data.get("chords", {})  # Dict of beat -> chord name
        bpm = data.get("bpm", 120)
        duration = data.get("duration", 16)  # Total duration in beats
        
        print(f"\n{'='*60}")
        print(f"🎹 CHORD RENDERING REQUEST")
        print(f"{'='*60}")
        print(f"Chords: {chords}")
        print(f"BPM: {bpm}")
        print(f"Duration: {duration} beats")
        
        if not chords:
            raise HTTPException(400, "No chords provided")
        
        # Create unique process ID
        process_id = str(uuid.uuid4())[:8]
        output_dir = Path(f"./generated_ui/chord_render_{process_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert chord dict keys from strings to ints
        chord_map = {int(beat): chord for beat, chord in chords.items()}
        
        # Import chord progression generator
        sys.path.append('/home/arlo/Data')
        from chord_progression_generator import generate_chord_progression_midi

        # Generate MIDI file
        output_file = output_dir / "chord_progression.mid"

        # Get voicing, rhythm, and style from request (default to random)
        import random
        voicing_options = ['close', 'open', 'drop2', 'drop3', 'shell', 'spread']
        rhythm_options = ['whole', 'half', 'quarter', 'eighth', 'syncopated', 'arpeggio', 'dotted']
        style_options = ['block', 'arpeggio']

        requested_voicing = data.get("voicing", "random")
        requested_rhythm = data.get("rhythm", "random")
        requested_style = data.get("style", "random")

        # Select randomly if "random" is specified, otherwise use requested value
        selected_voicing = random.choice(voicing_options) if requested_voicing == "random" else requested_voicing
        selected_rhythm = random.choice(rhythm_options) if requested_rhythm == "random" else requested_rhythm
        selected_style = random.choice(style_options) if requested_style == "random" else requested_style

        print(f"Generating MIDI at: {output_file}")
        print(f"  Voicing: {selected_voicing}, Rhythm: {selected_rhythm}, Style: {selected_style}")

        generate_chord_progression_midi(
            chord_beat_map=chord_map,
            bpm=bpm,
            output_path=str(output_file),
            voicing=selected_voicing,
            rhythm=selected_rhythm,
            style=selected_style,
            auto_detect_scale=True
        )
        
        if not output_file.exists():
            raise HTTPException(500, "MIDI file was not created")
        
        print(f"✅ Chord MIDI generated successfully!")
        print(f"{'='*60}\n")
        
        # Return file info
        return {
            "status": "completed",
            "process_id": process_id,
            "file_path": f"/download-chord-midi/{process_id}/chord_progression.mid",
            "chords": chord_map,
            "bpm": bpm,
            "duration": duration
        }
        
    except Exception as e:
        print(f"❌ Error rendering chords: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Chord rendering failed: {str(e)}")


@app.get("/download-chord-midi/{process_id}/{filename}")
async def download_chord_midi(process_id: str, filename: str):
    """Serve rendered chord MIDI files"""
    file_path = Path(f"./generated_ui/chord_render_{process_id}") / filename
    if not file_path.exists():
        raise HTTPException(404, "Chord MIDI file not found")
    return FileResponse(file_path, media_type="audio/midi", filename=filename)
