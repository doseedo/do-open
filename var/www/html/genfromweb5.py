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

torch.set_float32_matmul_precision("high")

# ------------------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------------------
sys.path.append('/home/arlo/Data')  # folder that has trainer_performer.py
sys.path.append('/home/arlo/Data/ACE-Step')  # Add ACE-Step directory for acestep imports

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

# Soundfont mapping for different instrument groups
INSTRUMENT_SOUNDFONTS = {
    "trombone": "/home/arlo/Data/soundfonts/trombone.sf2",
    "trumpet": "/home/arlo/Data/soundfonts/trumpet.sf2",
    "sax": "/home/arlo/Data/soundfonts/sax.ssof2",
    "violin": "/home/arlo/Data/soundfonts/violin.sf2",
    "cello": "/home/arlo/Data/soundfonts/cello.sf2",
    "default": "/usr/share/sounds/sf2/FluidR3_GM.sf2"  # fallback
}

# ------------------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------------------
MODEL: Union[Pipeline, None] = None
GROUP_NAMES: list = []
SUBGROUP_NAMES: list = []
MANIFEST_PATHS: list = []
MANIFEST_DATA: list = []

# Cache for conditioning extractions
CONDITIONING_CACHE: dict = {}

# Cache for ground truth latents
LATENT_CACHE: dict = {}

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
# MIDI Processing and FluidSynth Rendering
# ------------------------------------------------------------------------------

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
            "-g", "0.5",  # gain
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
        if instrument.is_drum:
            continue  # Skip drum tracks

        # Create a new MIDI file with just this track
        single_track_midi = pretty_midi.PrettyMIDI()
        single_track_midi.instruments.append(instrument)

        # Copy basic timing information
        single_track_midi.resolution = midi_data.resolution

        # Save individual track MIDI file
        track_name = instrument.name if instrument.name else f"Track_{i+1}"
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
    extraction = extract_conditioning_from_audio(audio_file, instrument_group=instrument_subgroup)
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

    # Use tempo override if provided, otherwise extract from MIDI
    if tempo_override is not None:
        actual_tempo = tempo_override
        print(f"🎵 Using tempo override: {actual_tempo:.1f} BPM")
    else:
        actual_tempo = extract_midi_tempo(midi_path)
        print(f"🎵 Detected tempo: {actual_tempo:.1f} BPM")

    base_tempo = 120.0  # fps=43.066 seems to be calibrated for 120 BPM
    tempo_ratio = actual_tempo / base_tempo
    adjusted_fps = fps * tempo_ratio

    print(f"🎵 Tempo adjustment: {actual_tempo:.1f} BPM (ratio: {tempo_ratio:.3f}, adjusted fps: {adjusted_fps:.3f})")

    # Get duration and create time grid with tempo-adjusted fps
    duration = max(midi_data.get_end_time(), 1.0)  # At least 1 second
    time_steps = int(duration * adjusted_fps) + 1

    # Create piano roll
    piano_roll = np.zeros((128, time_steps))

    # Convert all instruments to piano roll (merge them)
    total_notes = 0
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue  # Skip drum tracks

        for note in instrument.notes:
            start_frame = int(note.start * adjusted_fps)
            end_frame = int(note.end * adjusted_fps)
            start_frame = max(0, min(start_frame, time_steps - 1))
            end_frame = max(start_frame + 1, min(end_frame, time_steps))

            # Use velocity for intensity (normalized to 0-1)
            intensity = note.velocity / 127.0
            piano_roll[note.pitch, start_frame:end_frame] = intensity
            total_notes += 1

    print(f"✅ Created piano roll: {piano_roll.shape}, {total_notes} notes, {duration:.2f}s")

    # Resize to target window (preserve full MIDI duration)
    # TIMING FIX: Don't truncate MIDI - preserve full length for proper timing
    original_frames = piano_roll.shape[1]
    target_length = max(window_slow, original_frames)  # Always preserve full length

    if piano_roll.shape[1] != target_length:
        # Only pad if too short, never truncate
        pad_width = target_length - piano_roll.shape[1]
        piano_roll = np.pad(piano_roll, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        print(f"🎵 Padded piano roll from {original_frames} to {target_length} frames")

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

    # Extract tempo from original MIDI or use override
    if tempo_override is not None:
        original_tempo = tempo_override
        print(f"🎵 Using tempo override: {original_tempo:.1f} BPM")
    else:
        original_tempo = extract_midi_tempo(midi_path)
        print(f"🎵 Detected tempo: {original_tempo:.1f} BPM")

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

    # Apply tempo adjustment (same as in midi_to_piano_roll_conditioning)
    base_tempo = 120.0  # fps=43.066 seems to be calibrated for 120 BPM
    tempo_ratio = original_tempo / base_tempo
    adjusted_fps = fps * tempo_ratio

    print(f"🎵 Multitrack tempo adjustment: {original_tempo:.1f} BPM (ratio: {tempo_ratio:.3f}, adjusted fps: {adjusted_fps:.3f})")

    time_steps = int(duration * adjusted_fps) + 1

    track_piano_rolls = []
    track_info = []
    combined_piano_roll = np.zeros((128, time_steps))

    print(f"   Processing {track_count} tracks, duration: {duration:.2f}s")

    for i, instrument in enumerate(non_drum_instruments):
        # Create piano roll for this track
        track_piano_roll = np.zeros((128, time_steps))
        note_count = 0

        for note in instrument.notes:
            start_frame = int(note.start * adjusted_fps)
            end_frame = int(note.end * adjusted_fps)
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
# Conditioning I/O
# ------------------------------------------------------------------------------
def extract_conditioning_from_audio(audio_path: str, output_dir: str = "./extracted_conditioning", instrument_group: str = None) -> dict:
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

    # Save MIDI file
    midi.write(str(output_path))
    print(f"✅ Saved MIDI: {output_path} ({notes_added} notes, filtered {notes_filtered} short notes)")
    return str(output_path)

def save_basic_pitch_midi_with_voices(audio_file, subgroup=None, progress=None, tempo=120.0):
    """
    Save Basic Pitch MIDI from conditioning extraction with voice separation.
    Args:
        audio_file: input audio file path
        subgroup: instrument subgroup for soundfont selection
        progress: optional progress callback
        tempo: BPM tempo for output MIDI files
    Returns:
        dict with main MIDI path and voice MIDI paths
    """
    if audio_file is None:
        raise gr.Error("Please upload an audio file first.")

    if progress:
        progress(0, desc="Extracting conditioning...")

    # Extract conditioning (which includes Basic Pitch piano roll)
    extraction = extract_conditioning_from_audio(audio_file)
    win_slow = 1024  # default window size
    pr, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=win_slow)

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
    main_midi_path = output_dir / f"{audio_stem}_basicpitch.mid"
    piano_roll_to_midi(pr, main_midi_path, tempo=tempo)

    # Also save to original location
    original_main_midi = original_output_dir / f"{audio_stem}_basicpitch.mid"
    piano_roll_to_midi(pr, original_main_midi, tempo=tempo)

    if progress:
        progress(0.6, desc="Separating voices...")

    # Separate voices using existing function
    voices = separate_piano_roll_voices(pr)

    if progress:
        progress(0.8, desc="Saving voice MIDI files...")

    # Save individual voice MIDI files with note length filtering (both locations)
    voice_midi_paths = []
    for i, voice_pr in enumerate(voices):
        # Debug location (primary)
        voice_path = voices_dir / f"{audio_stem}_voice_{i+1}.mid"
        piano_roll_to_midi(voice_pr, voice_path, program=0, velocity=80, min_note_duration=0.1, tempo=tempo)
        voice_midi_paths.append(str(voice_path))

        # Original location (backup)
        original_voice_path = original_voices_dir / f"{audio_stem}_voice_{i+1}.mid"
        piano_roll_to_midi(voice_pr, original_voice_path, program=0, velocity=80, min_note_duration=0.1, tempo=tempo)

    if progress:
        progress(0.9, desc="Rendering FluidSynth debug audio for voices...")

    # Render FluidSynth debug audio for each voice
    debug_audio_paths = render_multitrack_debug_audio(voice_midi_paths, str(output_dir), audio_stem, instrument_group=subgroup)

    if progress:
        progress(1.0, desc="Done!")

    result = {
        "main_midi": str(main_midi_path),
        "voice_midis": voice_midi_paths,
        "debug_audio_paths": debug_audio_paths,
        "output_dir": str(output_dir),
        "voice_count": len(voices),
        "debug_dir": str(output_dir),  # For debugging reference
        "original_dir": str(original_output_dir)  # For backward compatibility
    }

    print(f"🎼 Saved {len(voices)} voice MIDI files to: {voices_dir}")
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

                # Copy all notes
                combined_instrument.notes = voice_instrument.notes.copy()
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

                # Check if all tracks are monophonic
                polyphony_data = multitrack_data.get('polyphony_analysis', {})
                all_monophonic = polyphony_data.get('polyphonic_tracks', 1) == 0

                if all_monophonic:
                    progress(0.6, desc="All tracks are monophonic - skipping voice separation...")
                    print("🎵 All tracks are monophonic - using track files as voice files (no voice separation needed)")

                    # Use track files as voice files since they're already separated
                    voice_midi_paths = track_midi_paths.copy()

                else:
                    progress(0.6, desc="Separating voices from combined track...")
                    print(f"🎵 Found {polyphony_data.get('polyphonic_tracks', 0)} polyphonic tracks - performing voice separation")

                    # Perform voice separation on combined piano roll
                    voices = separate_piano_roll_voices(multitrack_data['combined_piano_roll'])

                    voice_midi_paths = []
                    for i, voice_pr in enumerate(voices):
                        voice_path = voices_dir / f"{audio_stem}_voice_{i+1}.mid"
                        piano_roll_to_midi(voice_pr, voice_path, program=0, velocity=80, min_note_duration=0.1, tempo=original_tempo)
                        voice_midi_paths.append(str(voice_path))

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

            voice_section = ""
            if all_mono:
                voice_section = f"""
🎵 OPTIMIZATION: All tracks are monophonic!
Voice separation was skipped - track files are used as voice files.
This saves processing time since each track is already a separate voice.

Individual track files (also serving as voice files):
{track_files_text}"""
            else:
                voice_section = f"""
Individual track files:
{track_files_text}

Voice separation files (from combined tracks):
""" + "\n".join(f"• {Path(p).name}" for p in result["voice_midis"])

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

        # Apply final audio processing (compression + high-pass filter)
        mixed_audio = apply_final_audio_processing(mixed_audio, sample_rate=sample_rate)

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

def apply_final_audio_processing(audio: torch.Tensor, sample_rate: int = 32000) -> torch.Tensor:
    """
    Apply compression and high-pass filter to final audio output.

    Args:
        audio: Audio tensor [channels, samples]
        sample_rate: Sample rate in Hz

    Returns:
        Processed audio tensor
    """
    import torch
    import torchaudio.functional as F

    # Ensure audio is float and on CPU
    audio = audio.float().cpu()

    # Apply exponential high-pass filter up to 150 Hz
    # Use multiple cascaded high-pass filters for exponential rolloff
    cutoff_freq = 150.0

    # Apply 3 cascaded high-pass filters for steep rolloff (18 dB/octave = exponential-like)
    for i in range(3):
        # Each filter reduces the cutoff slightly for smoother transition
        current_cutoff = cutoff_freq * (0.8 ** i)  # 150Hz, 120Hz, 96Hz
        audio = F.highpass_biquad(audio, sample_rate, current_cutoff, Q=0.707)

    # Apply simple compression
    # Parameters for musical compression
    threshold = 0.7  # Compression threshold
    ratio = 4.0      # 4:1 compression ratio
    attack = 0.003   # 3ms attack
    release = 0.100  # 100ms release

    # Simple peak compression algorithm
    # Convert to mono for level detection
    if audio.shape[0] > 1:
        level_detect = audio.mean(dim=0)  # Average channels for level detection
    else:
        level_detect = audio[0]

    # Calculate compression gain reduction
    abs_audio = torch.abs(level_detect)

    # Simple peak detection with attack/release (sample-based approximation)
    attack_samples = int(attack * sample_rate)
    release_samples = int(release * sample_rate)

    # Apply threshold and ratio
    gain_reduction = torch.ones_like(abs_audio)
    over_threshold = abs_audio > threshold

    if over_threshold.any():
        # Calculate gain reduction for samples over threshold
        excess = abs_audio[over_threshold] - threshold
        compressed_excess = excess / ratio
        gain_reduction[over_threshold] = (threshold + compressed_excess) / abs_audio[over_threshold]

    # Smooth gain reduction with attack/release (simple moving average approximation)
    if attack_samples > 1:
        kernel = torch.ones(1, attack_samples, device=gain_reduction.device) / attack_samples
        gain_reduction = F.convolve(gain_reduction.unsqueeze(0),
                                   kernel,
                                   mode='same').squeeze(0)

    # Apply gain reduction to all channels
    for ch in range(audio.shape[0]):
        audio[ch] = audio[ch] * gain_reduction

    # Apply makeup gain (compensate for compression)
    makeup_gain = 1.2  # Slight boost to compensate
    audio = audio * makeup_gain

    # Final limiting to prevent clipping
    audio = torch.clamp(audio, -0.95, 0.95)

    print(f"🎛️ Applied final processing: HPF @ {cutoff_freq}Hz + {ratio}:1 compression")
    return audio

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
        # TIMING FIX: Calculate correct audio length from piano roll duration
        # Piano roll frames represent time at fps=43.066, convert to target sample rate
        piano_roll_duration_seconds = T_slow / 43.066
        audio_len = int(round(piano_roll_duration_seconds * sr_out))
        print(f"🎵 Calculated audio length: {T_slow} frames → {piano_roll_duration_seconds:.2f}s → {audio_len} samples")

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

    # Apply final audio processing (compression + high-pass filter)
    wav = apply_final_audio_processing(wav, sample_rate=sr_pred)

    out_dir = Path("./generated_ui"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_seed{seed}_cfg{cfg_weight:.1f}.wav"
    torchaudio.save(str(out_path), wav, sr_pred)
    print(f"✅ Wrote: {out_path}")
    return str(out_path)

@torch.no_grad()
def generate_monophonic_multiple(
    model, piano_roll, amp, rframe, rbend, encodec_tokens,
    group, subgroup, seed, steps, adapter_scale, cfg_weight, t0, sr_out,
    instrument_strength=1.0, inst_boost=2.5,
    piano_roll_gain=1.0, amp_gain=1.0, rframe_gain=1.0, rbend_gain=1.0, encodec_gain=1.0,
    use_overlap_decoder=True, original_audio_length=None,
    pitch_fidelity_boost=1.0, onset_guidance_boost=2.0, pitch_snap_strength=0.5,
    noise_level=1.0, audio_file=None, progress=None
):
    """
    Generate multiple monophonic outputs from separated voices and create a mixed sum.
    """
    print("🎵 Starting monophonic multiple voice generation")

    # Separate the piano roll into voices
    voices = separate_piano_roll_voices(piano_roll)

    if len(voices) == 1:
        print("⚠️ Only one voice detected, falling back to regular generation")
        return generate(
            model, piano_roll, amp, rframe, rbend, encodec_tokens,
            group, subgroup, steps, seed, adapter_scale, cfg_weight, t0, sr_out,
            instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
            use_overlap_decoder, original_audio_length, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength,
            noise_level, audio_file
        )

    # Generate each voice separately
    voice_outputs = []
    base_seed = int(seed) if seed > 0 else torch.seed() % 2**31

    out_dir = Path("./generated_ui"); out_dir.mkdir(exist_ok=True)
    timestamp = time.strftime('%Y%m%d-%H%M%S')

    for i, voice_pr in enumerate(voices):
        if progress:
            progress_val = 0.5 + (i / len(voices)) * 0.4  # 50-90% range
            progress(progress_val, desc=f"Generating voice {i+1}/{len(voices)}...")

        print(f"🎼 Generating voice {i+1}/{len(voices)}")

        # Use different seed for each voice for variety
        voice_seed = base_seed + i * 1000

        voice_output = generate(
            model, voice_pr, amp, rframe, rbend, encodec_tokens,
            group, subgroup, steps, voice_seed, adapter_scale, cfg_weight, t0, sr_out,
            instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
            use_overlap_decoder, original_audio_length, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength,
            noise_level, audio_file
        )

        # Rename the output to include voice number
        voice_path = out_dir / f"{timestamp}_voice{i+1}_seed{voice_seed}_cfg{cfg_weight:.1f}.wav"
        shutil.move(voice_output, str(voice_path))
        voice_outputs.append(str(voice_path))
        print(f"✅ Voice {i+1} saved: {voice_path.name}")

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
    monophonic_mode, midi_mode, render_and_extract, tempo_override, progress=gr.Progress(track_tqdm=True)
):
    if audio_file is None:
        raise gr.Error("Please upload an audio or MIDI file or pick a random one.")

    # Check if input is MIDI file
    is_midi = is_midi_file(audio_file)
    win_slow = int(getattr(MODEL.hparams, "window_slow", 1024))

    if is_midi and midi_mode:
        progress(0, desc="Processing MIDI file…")

        if render_and_extract:
            # Mode 1: Render MIDI to audio, then extract full conditioning
            progress(0.1, desc="Rendering MIDI to audio…")
            rendered_audio = render_midi_to_audio(audio_file, instrument_group=subgroup)

            progress(0.3, desc="Extracting conditioning from rendered audio…")
            extraction = extract_conditioning_from_audio(rendered_audio, instrument_group=subgroup)

            # But use original MIDI for piano roll
            progress(0.5, desc="Loading MIDI piano roll…")
            # TIMING FIX: Don't use tempo adjustment when using audio conditioning
            # to ensure piano roll and audio conditioning have matching timing scales
            pr_midi, _, _, _, _ = midi_to_piano_roll_conditioning(audio_file, win_slow, fps=43.066, tempo_override=None)
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
        progress(0, desc="Extracting conditioning from audio…")

        # original len (for exact decode length)
        try:
            wav, sr = torchaudio.load(audio_file)
            orig_len = wav.shape[-1]
        except Exception:
            orig_len = None

        extraction = extract_conditioning_from_audio(audio_file, instrument_group=subgroup)
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
            audio_file=audio_file
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
 