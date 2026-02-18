#!/usr/bin/env python3
"""
Gradio Web UI for ACE-Step generation (ControlBranch-ready)

- Loads the same Pipeline you trained with and restores hparams from the Lightning .ckpt
- Works with ctrl_enc + ctrlnet residual injection (ControlBranch1D)
- Instrument-token CFG (ON vs OFF) + sharper PR masking like previews
- Proper EnCodec gating (keeps tokens LongTensor)
"""

import sys, os, argparse, subprocess, json, random, time, shutil, tempfile
from pathlib import Path
import hashlib
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import gradio as gr
import mido
import fluidsynth
from scipy.io import wavfile
from typing import List, Tuple, Dict, Optional

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
CONDITIONING_CACHE_DIR = "./conditioning_cache"

# ------------------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------------------
MODEL: Pipeline | None = None
GROUP_NAMES: list[str] = []
SUBGROUP_NAMES: list[str] = []
MANIFEST_PATHS: list[str] = []
MANIFEST_DATA: list[dict] = []

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
# MIDI Handling
# ------------------------------------------------------------------------------
def parse_midi_file(midi_path: str) -> Dict:
    """Parse MIDI file and extract note events by track/channel."""
    mid = mido.MidiFile(midi_path)
    tracks_data = []

    for i, track in enumerate(mid.tracks):
        notes = []
        current_time = 0
        active_notes = {}  # note: (start_time, velocity)

        for msg in track:
            current_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[(msg.note, msg.channel)] = (current_time, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                note_key = (msg.note, msg.channel)
                if note_key in active_notes:
                    start_time, velocity = active_notes.pop(note_key)
                    notes.append({
                        'note': msg.note,
                        'channel': msg.channel,
                        'start': start_time,
                        'end': current_time,
                        'velocity': velocity
                    })

        if notes:  # Only include tracks with notes
            tracks_data.append({
                'track_id': i,
                'notes': notes,
                'name': track.name if hasattr(track, 'name') else f'Track {i}'
            })

    return {
        'tracks': tracks_data,
        'ticks_per_beat': mid.ticks_per_beat,
        'length_ticks': current_time if 'current_time' in locals() else max(
            max(note['end'] for note in track['notes']) for track in tracks_data
        ) if tracks_data else 0
    }

def separate_voices(midi_data: Dict, num_voices: int = 4) -> List[Dict]:
    """Separate MIDI into voices with voice leading optimization."""
    all_notes = []
    for track in midi_data['tracks']:
        for note in track['notes']:
            note['track_id'] = track['track_id']
            all_notes.append(note)

    # Sort notes by start time
    all_notes.sort(key=lambda x: x['start'])

    if not all_notes:
        return [{'notes': [], 'range': (60, 72)} for _ in range(num_voices)]

    # Group notes into chords (notes within small time window)
    chord_threshold = midi_data['ticks_per_beat'] // 8  # 32nd note tolerance
    chords = []
    current_chord = []
    current_time = all_notes[0]['start']

    for note in all_notes:
        if note['start'] - current_time <= chord_threshold:
            current_chord.append(note)
        else:
            if current_chord:
                chords.append(current_chord)
            current_chord = [note]
            current_time = note['start']

    if current_chord:
        chords.append(current_chord)

    # Initialize voices
    voices = [{'notes': [], 'last_pitch': None} for _ in range(num_voices)]

    # Assign notes to voices with voice leading
    for chord in chords:
        chord_notes = sorted(chord, key=lambda x: x['note'])

        # Create assignment matrix for voice leading
        assignments = assign_notes_to_voices(chord_notes, voices, num_voices)

        for voice_idx, note in assignments.items():
            if note is not None:
                voices[voice_idx]['notes'].append(note)
                voices[voice_idx]['last_pitch'] = note['note']

    # Calculate ranges for each voice
    for i, voice in enumerate(voices):
        if voice['notes']:
            pitches = [note['note'] for note in voice['notes']]
            voice['range'] = (min(pitches), max(pitches))
        else:
            # Default ranges for SATB
            default_ranges = [(48, 67), (55, 76), (60, 81), (67, 88)]  # Bass, Tenor, Alto, Soprano
            voice['range'] = default_ranges[i % len(default_ranges)]

    return voices

def assign_notes_to_voices(chord_notes: List[Dict], voices: List[Dict], num_voices: int) -> Dict[int, Dict]:
    """Assign chord notes to voices minimizing voice leading distance."""
    assignments = {}

    if not chord_notes:
        return assignments

    # If more notes than voices, take the most important ones (highest and lowest)
    if len(chord_notes) > num_voices:
        # Sort by pitch and take extremes + middle notes
        sorted_notes = sorted(chord_notes, key=lambda x: x['note'])
        selected = []
        indices = np.linspace(0, len(sorted_notes)-1, num_voices, dtype=int)
        chord_notes = [sorted_notes[i] for i in indices]

    # If fewer notes than voices, some voices will be silent
    available_voices = list(range(num_voices))

    for note in chord_notes:
        best_voice = None
        min_distance = float('inf')

        for voice_idx in available_voices:
            distance = 0
            if voices[voice_idx]['last_pitch'] is not None:
                # Voice leading distance (prefer minimal motion)
                distance = abs(note['note'] - voices[voice_idx]['last_pitch'])
            else:
                # No previous note, prefer appropriate range
                default_ranges = [(48, 67), (55, 76), (60, 81), (67, 88)]
                range_center = sum(default_ranges[voice_idx % len(default_ranges)]) / 2
                distance = abs(note['note'] - range_center) / 10  # Weight range preference less

            if distance < min_distance:
                min_distance = distance
                best_voice = voice_idx

        if best_voice is not None:
            assignments[best_voice] = note
            available_voices.remove(best_voice)

    return assignments

def midi_to_piano_roll(voice_data: Dict, ticks_per_beat: int, length_ticks: int,
                      time_resolution: int = 100) -> np.ndarray:
    """Convert MIDI voice to piano roll representation."""
    # Calculate time scale factor
    time_scale = time_resolution / ticks_per_beat
    length_frames = int(length_ticks * time_scale)

    # Create piano roll [128 pitches x time_frames]
    piano_roll = np.zeros((128, length_frames), dtype=np.float32)

    for note in voice_data['notes']:
        pitch = note['note']
        start_frame = int(note['start'] * time_scale)
        end_frame = int(note['end'] * time_scale)
        velocity = note['velocity'] / 127.0  # Normalize velocity

        # Ensure bounds
        start_frame = max(0, min(start_frame, length_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, length_frames))

        piano_roll[pitch, start_frame:end_frame] = velocity

    return piano_roll

def render_midi_with_fluidsynth(voice_data: Dict, ticks_per_beat: int,
                               sample_rate: int = 24000, soundfont_path: str = None) -> np.ndarray:
    """Render MIDI voice to audio using sine wave synthesis (bypassing piano soundfonts)."""
    print("🎵 Using sine wave synthesis instead of piano soundfonts")
    return synthesize_basic_audio(voice_data, ticks_per_beat, sample_rate)

def synthesize_basic_audio(voice_data: Dict, ticks_per_beat: int, sample_rate: int = 24000) -> np.ndarray:
    """Basic sine wave synthesis as fallback."""
    if not voice_data['notes']:
        return np.zeros(sample_rate, dtype=np.float32)  # 1 second of silence

    duration_ticks = max(note['end'] for note in voice_data['notes'])
    duration_seconds = (duration_ticks / ticks_per_beat) * 0.5  # 120 BPM
    total_samples = int(duration_seconds * sample_rate)

    audio = np.zeros(total_samples, dtype=np.float32)

    for note in voice_data['notes']:
        start_time = (note['start'] / ticks_per_beat) * 0.5
        end_time = (note['end'] / ticks_per_beat) * 0.5

        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        if start_sample < total_samples:
            end_sample = min(end_sample, total_samples)
            frequency = 440 * (2 ** ((note['note'] - 69) / 12))
            t = np.linspace(0, (end_sample - start_sample) / sample_rate, end_sample - start_sample)
            amplitude = (note['velocity'] / 127.0) * 0.1  # Quiet to avoid clipping

            # Simple sine wave with envelope
            envelope = np.exp(-t * 2)  # Decay envelope
            wave = amplitude * envelope * np.sin(2 * np.pi * frequency * t)

            audio[start_sample:end_sample] += wave

    return audio

def create_conditioning_from_midi(midi_processed_data: Dict, window_slow: int) -> tuple:
    """Create basic conditioning streams from MIDI data when no audio file is provided."""
    voices = midi_processed_data['voices']

    # Combine all voices for a full piano roll
    combined_pr = np.zeros((128, window_slow), dtype=np.float32)

    for voice in voices:
        if voice['notes_count'] > 0:
            pr = voice['piano_roll']
            if pr.shape[-1] <= window_slow:
                combined_pr[:, :pr.shape[-1]] += pr

    # Create basic amplitude envelope from piano roll activity
    amp = np.sum(combined_pr, axis=0)  # [T]
    amp = amp.reshape(1, -1)  # [1, T]
    amp = np.clip(amp / (amp.max() + 1e-8), 0.0, 1.0)  # Normalize

    # Ensure amp has the right shape
    if amp.shape[-1] != window_slow:
        if amp.shape[-1] < window_slow:
            pad_width = window_slow - amp.shape[-1]
            amp = np.pad(amp, ((0, 0), (0, pad_width)), mode='constant')
        else:
            amp = amp[:, :window_slow]

    # Create basic rhythm frame from note onsets
    rframe = np.zeros((1, window_slow), dtype=np.float32)
    for voice in voices:
        if voice['notes_count'] > 0:
            pr = voice['piano_roll']
            # Detect onsets (where piano roll goes from 0 to >0)
            if pr.shape[-1] > 1:
                onset_mask = (pr[:, 1:] > 0.1) & (pr[:, :-1] <= 0.1)
                onset_strength = np.sum(onset_mask, axis=0)
                frames_to_copy = min(onset_strength.shape[0], window_slow)
                rframe[0, :frames_to_copy] += onset_strength[:frames_to_copy]

    rframe = np.clip(rframe, 0.0, 1.0)

    # Create basic pitch bend (mostly zeros for now)
    rbend = np.zeros((1, window_slow), dtype=np.float32)

    # Create basic EnCodec tokens (dummy tokens, mostly zeros)
    # EnCodec typically has 8 codebooks, we'll create a minimal version
    encodec_frames = max(1, window_slow // 4)  # Rough downsampling factor
    encodec = torch.zeros((1, 8, encodec_frames), dtype=torch.long)

    print(f"✅ Created basic conditioning from MIDI data - shapes: pr={combined_pr.shape}, amp={amp.shape}, rframe={rframe.shape}, rbend={rbend.shape}, encodec={encodec.shape}")
    return combined_pr, amp, rframe, rbend, encodec

# ------------------------------------------------------------------------------
# Conditioning I/O
# ------------------------------------------------------------------------------
def get_audio_file_hash(audio_path: str) -> str:
    """Generate hash for audio file for caching."""
    hasher = hashlib.md5()

    # Include file size and modification time in hash
    stat = os.stat(audio_path)
    hasher.update(f"{stat.st_size}:{stat.st_mtime}".encode())

    # Sample first and last few KB of file for content hash
    with open(audio_path, 'rb') as f:
        f.seek(0)
        hasher.update(f.read(4096))  # First 4KB

        if stat.st_size > 8192:
            f.seek(-4096, 2)  # Last 4KB
            hasher.update(f.read(4096))

    return hasher.hexdigest()

def is_midi_file(file_path: str) -> bool:
    """Check if the uploaded file is a MIDI file."""
    if not file_path or not os.path.exists(file_path):
        return False

    # Check extension
    if file_path.lower().endswith(('.mid', '.midi')):
        return True

    # Check MIDI magic bytes
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'MThd'
    except:
        return False

def process_midi_for_conditioning(midi_path: str, output_dir: str = "./extracted_conditioning", num_voices: int = 4) -> dict:
    """Process MIDI file and return data for separate voice generation."""

    # Parse MIDI
    print(f"📖 Processing MIDI file: {midi_path}")
    midi_data = parse_midi_file(midi_path)

    if not midi_data['tracks']:
        raise ValueError("MIDI file contains no note data")

    # Separate into voices
    print(f"🎵 Separating into {num_voices} voices...")
    voices = separate_voices(midi_data, num_voices)

    # Create output directory
    stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in Path(midi_path).stem)[:128] or "midi"
    out_dir = Path(output_dir) / f"midi_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get model's window size
    win_slow = int(getattr(MODEL.hparams, "window_slow", 1024)) if MODEL else 1024

    # Process each voice and create conditioning
    voice_conditioning_data = []

    for i, voice in enumerate(voices):
        notes_count = len(voice.get('notes', []))
        if notes_count > 0:
            print(f"🎹 Processing voice {i+1} ({notes_count} notes)...")

            # Convert to piano roll
            pr = midi_to_piano_roll(voice, midi_data['ticks_per_beat'], midi_data['length_ticks'])

            # Resize to model's expected window size
            if pr.shape[-1] > win_slow:
                pr = pr[:, :win_slow]
            elif pr.shape[-1] < win_slow:
                pad_width = win_slow - pr.shape[-1]
                pr = np.pad(pr, ((0, 0), (0, pad_width)), mode='constant')

            # Render voice to audio for conditioning extraction
            audio = render_midi_with_fluidsynth(voice, midi_data['ticks_per_beat'], sample_rate=ENC_SR)

            # Save rendered audio
            temp_audio_path = out_dir / f"voice_{i}.wav"
            if len(audio.shape) == 1:
                audio = audio.reshape(1, -1)
            torchaudio.save(str(temp_audio_path), torch.from_numpy(audio), ENC_SR)

            # Create conditioning by running test_extract_local on the rendered audio
            voice_output_dir = out_dir / f"voice_{i}"
            voice_output_dir.mkdir(exist_ok=True)

            cmd = ["python", "test_extract_local.py", "--input", str(temp_audio_path), "--output", str(voice_output_dir)]
            print(f"Extracting conditioning for voice {i}: {' '.join(cmd)}")
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if res.returncode == 0:
                # Successfully extracted conditioning
                voice_conditioning_data.append({
                    'voice_id': i,
                    'piano_roll': pr,
                    'conditioning_dir': str(voice_output_dir),
                    'conditioning_stem': f"voice_{i}",
                    'notes_count': notes_count,
                    'pitch_range': voice.get('range', (60, 72))
                })
            else:
                print(f"Warning: Extraction failed for voice {i}: {res.stderr}")
                # Create basic conditioning for this voice
                voice_conditioning_data.append({
                    'voice_id': i,
                    'piano_roll': pr,
                    'conditioning_dir': None,  # Will use basic conditioning
                    'conditioning_stem': None,
                    'notes_count': notes_count,
                    'pitch_range': voice.get('range', (60, 72))
                })

    return {
        'type': 'midi_voices',
        'voices': voice_conditioning_data,
        'midi_data': midi_data,
        'num_voices': num_voices
    }

def create_basic_midi_conditioning(voices: list, midi_data: dict, out_dir: Path, win_slow: int) -> dict:
    """Create basic conditioning files from MIDI data."""
    stem = "midi_basic"

    # Combine all voices for full piano roll
    combined_pr = np.zeros((128, win_slow), dtype=np.float32)
    for voice in voices:
        notes_count = len(voice.get('notes', []))
        if notes_count > 0:
            pr = midi_to_piano_roll(voice, midi_data['ticks_per_beat'], midi_data['length_ticks'])
            if pr.shape[-1] <= win_slow:
                combined_pr[:, :pr.shape[-1]] += pr

    # Create other conditioning streams
    amp = np.sum(combined_pr, axis=0).reshape(1, -1)  # [1, T]
    amp = np.clip(amp / (amp.max() + 1e-8), 0.0, 1.0)
    if amp.shape[-1] != win_slow:
        if amp.shape[-1] < win_slow:
            amp = np.pad(amp, ((0, 0), (0, win_slow - amp.shape[-1])), mode='constant')
        else:
            amp = amp[:, :win_slow]

    rframe = np.zeros((1, win_slow), dtype=np.float32)
    rbend = np.zeros((1, win_slow), dtype=np.float32)

    # Save conditioning files
    np.save(out_dir / f"{stem}.pianoroll.npy", combined_pr)
    np.save(out_dir / f"{stem}.amp.npy", amp)
    np.save(out_dir / f"{stem}.rframe.npy", rframe)
    np.save(out_dir / f"{stem}.rbend.npy", rbend)

    # Create basic encodec tokens
    encodec_frames = max(1, win_slow // 4)
    encodec = torch.zeros((1, 8, encodec_frames), dtype=torch.long)
    torch.save(encodec, out_dir / f"{stem}.encodec.pt")

    return {"dir": str(out_dir), "stem": stem}

def extract_conditioning_from_audio(audio_path: str, output_dir: str = "./extracted_conditioning", use_cache: bool = True, num_voices: int = 4) -> dict:
    # Check if this is a MIDI file
    if is_midi_file(audio_path):
        print("🎼 MIDI file detected, processing as MIDI...")
        return process_midi_for_conditioning(audio_path, output_dir, num_voices)

    # Check global cache first if enabled
    if use_cache:
        cache_dir = Path(CONDITIONING_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)

        file_hash = get_audio_file_hash(audio_path)
        cache_path = cache_dir / f"{file_hash}.pkl"

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Verify cached files still exist
                    if "paths" in cached_data:
                        if all(cached_data["paths"].get(k) and os.path.exists(cached_data["paths"][k])
                               for k in ["piano_roll","amp","rframe","rbend","encodec"]):
                            print(f"✅ Using cached conditioning (hash: {file_hash[:8]})")
                            return cached_data
                    elif "dir" in cached_data:
                        stem = cached_data["stem"]
                        out_dir = Path(cached_data["dir"])
                        req = [out_dir/f"{stem}.pianoroll.npy", out_dir/f"{stem}.amp.npy", out_dir/f"{stem}.rframe.npy",
                               out_dir/f"{stem}.rbend.npy", out_dir/f"{stem}.encodec.pt"]
                        if all(x.exists() for x in req):
                            print(f"✅ Using cached conditioning (hash: {file_hash[:8]})")
                            return cached_data
                print(f"⚠️ Cache entry exists but files are missing, re-extracting...")
            except Exception as e:
                print(f"⚠️ Cache read error: {e}, re-extracting...")

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
            print("✅ Using conditioning from manifest paths.")
            result = {"paths": paths}
            # Cache the result
            if use_cache:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(result, f)
                except Exception as e:
                    print(f"⚠️ Cache write error: {e}")
            return result
        print("⚠️ Manifest record found but some files are missing; falling back to local extraction.")

    # Extract to local directory
    stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in Path(audio_path).stem)[:128] or "audio"
    out_dir = Path(output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    req = [out_dir/f"{stem}.pianoroll.npy", out_dir/f"{stem}.amp.npy", out_dir/f"{stem}.rframe.npy",
           out_dir/f"{stem}.rbend.npy", out_dir/f"{stem}.encodec.pt"]

    # Check if already extracted locally
    if all(x.exists() for x in req):
        print(f"✅ Using locally cached conditioning: {out_dir}")
        result = {"dir": str(out_dir), "stem": stem}
        # Cache the result
        if use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                print(f"⚠️ Cache write error: {e}")
        return result

    # Run extraction
    cmd = ["python", "test_extract_local.py", "--input", str(audio_path), "--output", str(out_dir)]
    print(f"Running extraction: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        print(res.stdout); print(res.stderr)
        raise RuntimeError("Extraction failed.")
    print("✅ Conditioning extracted successfully.")
    result = {"dir": str(out_dir), "stem": stem}

    # Cache the result
    if use_cache:
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"⚠️ Cache write error: {e}")

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
    pitch_fidelity_boost=1.0, onset_guidance_boost=2.0, pitch_snap_strength=0.5
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

def process_midi_upload(midi_file: str, num_voices: int = 4, render_audio: bool = False) -> Dict:
    """Process uploaded MIDI file and prepare for voice generation."""
    if not midi_file or not os.path.exists(midi_file):
        raise ValueError("No MIDI file provided or file does not exist")

    # Parse MIDI
    print(f"📖 Parsing MIDI file: {midi_file}")
    midi_data = parse_midi_file(midi_file)

    if not midi_data['tracks']:
        raise ValueError("MIDI file contains no note data")

    # Separate into voices
    print(f"🎵 Separating into {num_voices} voices...")
    voices = separate_voices(midi_data, num_voices)

    # Convert each voice to piano roll and optionally audio
    voice_data = []
    win_slow = int(getattr(MODEL.hparams, "window_slow", 1024))

    for i, voice in enumerate(voices):
        print(f"Processing voice {i+1}/{num_voices}...")

        # Convert to piano roll
        pr = midi_to_piano_roll(voice, midi_data['ticks_per_beat'], midi_data['length_ticks'])

        # Resize to model's expected window size
        if pr.shape[-1] > win_slow:
            pr = pr[:, :win_slow]
        elif pr.shape[-1] < win_slow:
            pad_width = win_slow - pr.shape[-1]
            pr = np.pad(pr, ((0, 0), (0, pad_width)), mode='constant')

        voice_entry = {
            'voice_id': i,
            'piano_roll': pr,
            'notes_count': len(voice.get('notes', [])),
            'pitch_range': voice.get('range', (60, 72))
        }

        # Only render audio if requested (when we need audio conditioning)
        if render_audio:
            audio = render_midi_with_fluidsynth(voice, midi_data['ticks_per_beat'], sample_rate=ENC_SR)

            # Save temporary audio file
            temp_audio_dir = Path(tempfile.mkdtemp())
            temp_audio_path = temp_audio_dir / f"voice_{i}.wav"

            # Ensure audio is the right length and shape
            if len(audio.shape) == 1:
                audio = audio.reshape(1, -1)  # Make stereo if needed

            # Save as WAV
            torchaudio.save(str(temp_audio_path), torch.from_numpy(audio), ENC_SR)
            voice_entry['audio_path'] = str(temp_audio_path)
        else:
            voice_entry['audio_path'] = None

        voice_data.append(voice_entry)

    return {
        'voices': voice_data,
        'midi_data': midi_data,
        'num_voices': num_voices
    }

@torch.no_grad()
def generate_voice_separately(
    model: Pipeline, voice_data: Dict, audio_conditioning_path: Optional[str],
    group: str, subgroup: str, steps: int, seed: int, adapter_scale: float,
    cfg_weight: float, t0: float, sr_out: int,
    instrument_strength: float = 1.0, inst_boost: float = 2.5,
    piano_roll_gain: float = 1.0, amp_gain: float = 1.0,
    rframe_gain: float = 1.0, rbend_gain: float = 1.0, encodec_gain: float = 1.0,
    use_overlap_decoder: bool = True, original_audio_length: Optional[int] = None,
    pitch_fidelity_boost: float = 1.0, onset_guidance_boost: float = 2.0,
    pitch_snap_strength: float = 0.5, midi_processed_data: Optional[Dict] = None
) -> str:
    """Generate audio for a single voice using MIDI piano roll + audio conditioning."""

    win_slow = int(getattr(model.hparams, "window_slow", 1024))

    if audio_conditioning_path and os.path.exists(audio_conditioning_path):
        # Extract conditioning from the provided audio
        print(f"🎧 Extracting conditioning from audio for voice {voice_data['voice_id']}...")
        extraction = extract_conditioning_from_audio(audio_conditioning_path, use_cache=True)
        _, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=win_slow)
    else:
        # Create basic conditioning from MIDI data
        print(f"🎼 Creating MIDI-only conditioning for voice {voice_data['voice_id']}...")
        if midi_processed_data is None:
            # Fallback: create minimal conditioning
            amp = np.ones((1, win_slow), dtype=np.float32) * 0.5
            rfr = np.zeros((1, win_slow), dtype=np.float32)
            rbd = np.zeros((1, win_slow), dtype=np.float32)
            encodec_frames = max(1, win_slow // 4)
            enc = torch.zeros((1, 8, encodec_frames), dtype=torch.long)
            print(f"✅ Created fallback conditioning - shapes: amp={amp.shape}, rfr={rfr.shape}, rbd={rbd.shape}, enc={enc.shape}")
        else:
            _, amp, rfr, rbd, enc = create_conditioning_from_midi(midi_processed_data, win_slow)

    # Use the MIDI-derived piano roll
    pr = voice_data['piano_roll']
    print(f"🎹 Using MIDI piano roll for voice {voice_data['voice_id']} "
          f"(notes: {voice_data['notes_count']}, range: {voice_data['pitch_range']}, shape: {pr.shape})")

    # Ensure piano roll has correct shape
    if pr.shape != (128, win_slow):
        print(f"⚠️ Piano roll shape mismatch: expected (128, {win_slow}), got {pr.shape}")
        if pr.shape[0] != 128:
            # Pad or trim pitch dimension
            if pr.shape[0] < 128:
                pr = np.pad(pr, ((0, 128 - pr.shape[0]), (0, 0)), mode='constant')
            else:
                pr = pr[:128, :]
        if pr.shape[1] != win_slow:
            # Pad or trim time dimension
            if pr.shape[1] < win_slow:
                pr = np.pad(pr, ((0, 0), (0, win_slow - pr.shape[1])), mode='constant')
            else:
                pr = pr[:, :win_slow]
        print(f"✅ Piano roll resized to: {pr.shape}")

    print(f"📊 Final conditioning shapes: pr={pr.shape}, amp={amp.shape}, rfr={rfr.shape}, rbd={rbd.shape}, enc={enc.shape}")

    # Generate using the existing generate function with MIDI piano roll
    out_path = generate(
        model, pr, amp, rfr, rbd, enc,
        group, subgroup, steps, seed, adapter_scale, cfg_weight, t0, sr_out,
        instrument_strength=instrument_strength, inst_boost=inst_boost,
        piano_roll_gain=piano_roll_gain, amp_gain=amp_gain,
        rframe_gain=rframe_gain, rbend_gain=rbend_gain, encodec_gain=encodec_gain,
        use_overlap_decoder=use_overlap_decoder, original_audio_length=original_audio_length,
        pitch_fidelity_boost=pitch_fidelity_boost, onset_guidance_boost=onset_guidance_boost,
        pitch_snap_strength=pitch_snap_strength
    )

    # Rename output to include voice ID
    out_path_obj = Path(out_path)
    voice_out_path = out_path_obj.parent / f"voice_{voice_data['voice_id']}_{out_path_obj.name}"
    shutil.move(out_path, voice_out_path)

    return str(voice_out_path)

def generate_all_voices(
    midi_processed_data: Dict, audio_conditioning_path: Optional[str],
    group: str, subgroup: str, steps: int, base_seed: int, adapter_scale: float,
    cfg_weight: float, t0: float, sr_out: int = 32000,
    instrument_strength: float = 1.0, inst_boost: float = 2.5,
    piano_roll_gain: float = 1.0, amp_gain: float = 1.0,
    rframe_gain: float = 1.0, rbend_gain: float = 1.0, encodec_gain: float = 1.0,
    use_overlap_decoder: bool = True, original_audio_length: Optional[int] = None,
    pitch_fidelity_boost: float = 1.0, onset_guidance_boost: float = 2.0,
    pitch_snap_strength: float = 0.5
) -> List[str]:
    """Generate audio for all voices from MIDI."""

    voice_outputs = []
    voices = midi_processed_data['voices']

    for i, voice_data in enumerate(voices):
        if voice_data['notes_count'] == 0:
            print(f"⏭️ Skipping voice {i} (no notes)")
            continue

        print(f"🎵 Generating voice {i+1}/{len(voices)}...")

        # Use different seed for each voice
        voice_seed = base_seed + i * 1000 if base_seed > 0 else 0

        try:
            voice_output = generate_voice_separately(
                MODEL, voice_data, audio_conditioning_path,
                group, subgroup, steps, voice_seed, adapter_scale, cfg_weight, t0, sr_out,
                instrument_strength=instrument_strength, inst_boost=inst_boost,
                piano_roll_gain=piano_roll_gain, amp_gain=amp_gain,
                rframe_gain=rframe_gain, rbend_gain=rbend_gain, encodec_gain=encodec_gain,
                use_overlap_decoder=use_overlap_decoder, original_audio_length=original_audio_length,
                pitch_fidelity_boost=pitch_fidelity_boost, onset_guidance_boost=onset_guidance_boost,
                pitch_snap_strength=pitch_snap_strength, midi_processed_data=midi_processed_data
            )
            voice_outputs.append(voice_output)
            print(f"✅ Voice {i} generated: {voice_output}")

        except Exception as e:
            print(f"❌ Error generating voice {i}: {e}")
            voice_outputs.append(None)

    return voice_outputs

def create_silent_audio_file(voice_id: int, duration: float = 10.0, sample_rate: int = 32000) -> str:
    """Create a silent audio file for empty voices."""
    import time
    out_dir = Path("./generated_ui")
    out_dir.mkdir(exist_ok=True)

    # Generate silent audio
    num_samples = int(duration * sample_rate)
    silent_audio = torch.zeros(1, num_samples, dtype=torch.float32)

    # Create filename with timestamp and voice ID
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    out_path = out_dir / f"voice_{voice_id}_silent_{timestamp}.wav"

    # Save silent audio file
    torchaudio.save(str(out_path), silent_audio, sample_rate)
    print(f"🔇 Created silent audio file for voice {voice_id}: {out_path}")

    return str(out_path)

def combine_voice_outputs(voice_outputs: List[str], output_path: Optional[str] = None) -> str:
    """Combine multiple voice generations into a single audio file."""
    valid_outputs = [path for path in voice_outputs if path and os.path.exists(path)]

    if not valid_outputs:
        raise ValueError("No valid voice outputs to combine")

    if len(valid_outputs) == 1:
        return valid_outputs[0]

    print(f"🎶 Combining {len(valid_outputs)} voice outputs...")

    # Load all audio files
    audio_data = []
    sample_rate = None
    num_channels = None

    for voice_path in valid_outputs:
        wav, sr = torchaudio.load(voice_path)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            # Resample if needed
            wav = torchaudio.functional.resample(wav, sr, sample_rate)

        # Ensure consistent channel format
        if num_channels is None:
            num_channels = wav.shape[0]
        elif wav.shape[0] != num_channels:
            if num_channels == 1 and wav.shape[0] == 2:
                # Convert stereo to mono
                wav = wav.mean(dim=0, keepdim=True)
            elif num_channels == 2 and wav.shape[0] == 1:
                # Convert mono to stereo (duplicate channel)
                wav = wav.repeat(2, 1)

        audio_data.append(wav)

    # Find the maximum length
    max_length = max(audio.shape[-1] for audio in audio_data)

    # Pad all audio to the same length and sum
    combined = torch.zeros(num_channels, max_length)
    for audio in audio_data:
        if audio.shape[-1] < max_length:
            audio = F.pad(audio, (0, max_length - audio.shape[-1]))
        combined += audio

    # Normalize to prevent clipping
    max_val = combined.abs().max()
    if max_val > 0.95:
        combined = combined * (0.95 / max_val)

    # Save combined output
    if output_path is None:
        out_dir = Path("./generated_ui")
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_combined_voices.wav"

    torchaudio.save(str(output_path), combined, sample_rate)
    print(f"✅ Combined audio saved: {output_path}")

    return str(output_path)

# ------------------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------------------
def run_midi_voice_generation(
    midi_file, audio_file, group, subgroup, seed, steps, adapter_scale, cfg_weight, t0,
    instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
    use_overlap_decoder, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength,
    num_voices, separate_voices_enabled, combine_outputs,
    progress=gr.Progress(track_tqdm=True)
):
    if midi_file is None:
        raise gr.Error("Please upload a MIDI file.")

    # Audio file is now optional
    conditioning_mode = "audio" if (audio_file and os.path.exists(audio_file)) else "midi-only"
    print(f"🎛️ Using {conditioning_mode} conditioning mode")

    progress(0, desc="Processing MIDI file...")

    # Process MIDI file
    try:
        # Only render audio if we're not using audio conditioning
        render_audio = (conditioning_mode != "audio")
        midi_processed = process_midi_upload(midi_file, num_voices=int(num_voices), render_audio=render_audio)
    except Exception as e:
        raise gr.Error(f"MIDI processing failed: {str(e)}")

    if separate_voices_enabled:
        progress(0.2, desc="Generating voices separately...")

        # Generate each voice separately
        voice_outputs = generate_all_voices(
            midi_processed, audio_file if conditioning_mode == "audio" else None,
            group, subgroup, int(steps), int(seed), float(adapter_scale), float(cfg_weight), float(t0),
            sr_out=32000, instrument_strength=float(instrument_strength), inst_boost=float(inst_boost),
            piano_roll_gain=float(piano_roll_gain), amp_gain=float(amp_gain),
            rframe_gain=float(rframe_gain), rbend_gain=float(rbend_gain), encodec_gain=float(encodec_gain),
            use_overlap_decoder=bool(use_overlap_decoder),
            pitch_fidelity_boost=float(pitch_fidelity_boost), onset_guidance_boost=float(onset_guidance_boost),
            pitch_snap_strength=float(pitch_snap_strength)
        )

        progress(0.9, desc="Finalizing output...")

        if combine_outputs and len([f for f in voice_outputs if f is not None]) > 1:
            # Combine voices
            combined_output = combine_voice_outputs(voice_outputs)
            progress(1.0, desc="Done!")
            mode_desc = f" ({conditioning_mode} conditioning)"
            return combined_output, f"Generated {len([f for f in voice_outputs if f is not None])} voices and combined them{mode_desc}."
        else:
            # Return first valid voice output
            valid_outputs = [f for f in voice_outputs if f is not None]
            if valid_outputs:
                progress(1.0, desc="Done!")
                mode_desc = f" ({conditioning_mode} conditioning)"
                return valid_outputs[0], f"Generated {len(valid_outputs)} voices separately{mode_desc}. Showing voice 0."
            else:
                raise gr.Error("No voices were successfully generated.")
    else:
        # Use the first voice's piano roll for single generation
        if not midi_processed['voices'] or midi_processed['voices'][0]['notes_count'] == 0:
            raise gr.Error("No notes found in the first voice of the MIDI file.")

        voice_data = midi_processed['voices'][0]
        progress(0.5, desc="Generating with MIDI piano roll...")

        output_path = generate_voice_separately(
            MODEL, voice_data, audio_file if conditioning_mode == "audio" else None,
            group, subgroup, int(steps), int(seed), float(adapter_scale), float(cfg_weight), float(t0),
            sr_out=32000, instrument_strength=float(instrument_strength), inst_boost=float(inst_boost),
            piano_roll_gain=float(piano_roll_gain), amp_gain=float(amp_gain),
            rframe_gain=rframe_gain, rbend_gain=rbend_gain, encodec_gain=encodec_gain,
            use_overlap_decoder=bool(use_overlap_decoder),
            pitch_fidelity_boost=float(pitch_fidelity_boost), onset_guidance_boost=float(onset_guidance_boost),
            pitch_snap_strength=float(pitch_snap_strength), midi_processed_data=midi_processed
        )

        progress(1.0, desc="Done!")
        mode_desc = f" ({conditioning_mode} conditioning)"
        return output_path, f"Generated using MIDI piano roll from voice 0 ({voice_data['notes_count']} notes){mode_desc}."

def generate_single_voice(voice_data: dict, group: str, subgroup: str, seed: int, steps: int,
                         adapter_scale: float, cfg_weight: float, t0: float, sr_out: int,
                         instrument_strength: float, inst_boost: float, piano_roll_gain: float,
                         amp_gain: float, rframe_gain: float, rbend_gain: float, encodec_gain: float,
                         use_overlap_decoder: bool, pitch_fidelity_boost: float,
                         onset_guidance_boost: float, pitch_snap_strength: float, orig_len=None) -> str:
    """Generate audio for a single voice."""

    # Load conditioning for this voice
    win_slow = int(getattr(MODEL.hparams, "window_slow", 1024))

    if voice_data['conditioning_dir']:
        # Use extracted conditioning
        extraction = {"dir": voice_data['conditioning_dir'], "stem": voice_data['conditioning_stem']}
        _, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=win_slow)
    else:
        # Use basic conditioning
        amp = np.ones((1, win_slow), dtype=np.float32) * 0.5
        rfr = np.zeros((1, win_slow), dtype=np.float32)
        rbd = np.zeros((1, win_slow), dtype=np.float32)
        encodec_frames = max(1, win_slow // 4)
        enc = torch.zeros((1, 8, encodec_frames), dtype=torch.long)

    # Use MIDI piano roll
    pr = voice_data['piano_roll']

    print(f"🎵 Generating voice {voice_data['voice_id']} ({voice_data['notes_count']} notes, range: {voice_data['pitch_range']})")

    # Generate audio
    out_path = generate(
        MODEL, pr, amp, rfr, rbd, enc,
        group, subgroup, steps, seed + voice_data['voice_id'] * 1000,  # Different seed per voice
        adapter_scale, cfg_weight, t0, sr_out,
        instrument_strength=instrument_strength, inst_boost=inst_boost,
        piano_roll_gain=piano_roll_gain, amp_gain=amp_gain,
        rframe_gain=rframe_gain, rbend_gain=rbend_gain, encodec_gain=encodec_gain,
        use_overlap_decoder=use_overlap_decoder, original_audio_length=orig_len,
        pitch_fidelity_boost=pitch_fidelity_boost, onset_guidance_boost=onset_guidance_boost,
        pitch_snap_strength=pitch_snap_strength
    )

    # Rename output to include voice ID
    out_path_obj = Path(out_path)
    voice_out_path = out_path_obj.parent / f"voice_{voice_data['voice_id']}_{out_path_obj.name}"
    shutil.move(out_path, voice_out_path)

    return str(voice_out_path)

def run_generation(
    audio_file, group, subgroup, seed, steps, adapter_scale, cfg_weight, t0,
    instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
    use_overlap_decoder, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength,
    progress=gr.Progress(track_tqdm=True)
):
    if audio_file is None:
        raise gr.Error("Please upload an audio file, MIDI file, or pick a random one.")
    progress(0, desc="Extracting conditioning…")

    # Check if this is a MIDI file
    is_midi = is_midi_file(audio_file)
    file_type = "MIDI" if is_midi else "audio"
    print(f"🎵 Processing {file_type} file: {audio_file}")

    extraction = extract_conditioning_from_audio(audio_file, num_voices=4)

    # Check if this is MIDI multi-voice data
    if extraction.get('type') == 'midi_voices':
        progress(0.25, desc="Generating voices separately…")

        voice_outputs = []
        voices = extraction['voices']

        # Calculate actual MIDI duration from the first voice with notes
        orig_len = None
        for voice in voices:
            if voice.get('notes') and len(voice['notes']) > 0:
                # Calculate duration in seconds from MIDI data
                max_end_tick = max(note['end'] for note in voice['notes'])
                tempo_seconds_per_beat = 0.5  # 120 BPM default (60/120 = 0.5)
                midi_duration_seconds = (max_end_tick / extraction['midi_data']['ticks_per_beat']) * tempo_seconds_per_beat
                orig_len = int(midi_duration_seconds * 32000)  # Convert to samples at 32kHz
                print(f"🕐 MIDI duration: {midi_duration_seconds:.2f}s ({orig_len} samples)")
                break

        # Ensure we always process exactly 4 voices
        for i in range(4):
            if i < len(voices) and voices[i]['notes_count'] > 0:
                progress(0.25 + (0.7 * i / 4), desc=f"Generating voice {i+1}…")

                try:
                    voice_output = generate_single_voice(
                        voices[i], group, subgroup, int(seed), int(steps),
                        float(adapter_scale), float(cfg_weight), float(t0), 32000,
                        float(instrument_strength), float(inst_boost), float(piano_roll_gain),
                        float(amp_gain), float(rframe_gain), float(rbend_gain), float(encodec_gain),
                        bool(use_overlap_decoder), float(pitch_fidelity_boost),
                        float(onset_guidance_boost), float(pitch_snap_strength), orig_len
                    )
                    voice_outputs.append(voice_output)
                    print(f"✅ Voice {i+1} generated: {voice_output}")
                except Exception as e:
                    print(f"❌ Error generating voice {i+1}: {e}")
                    # Create silent audio file for failed generation
                    silent_path = create_silent_audio_file(i+1)
                    voice_outputs.append(silent_path)
            else:
                print(f"⏭️ Voice {i+1} has no notes, creating silent audio")
                # Create silent audio file for empty voice
                silent_path = create_silent_audio_file(i+1)
                voice_outputs.append(silent_path)

        # Create combined output
        progress(0.95, desc="Combining voices…")

        # Now all voice_outputs should be valid file paths (including silent files)
        # Filter out any remaining None values just in case
        valid_outputs = [f for f in voice_outputs if f is not None and os.path.exists(f)]
        non_silent_outputs = [f for f in valid_outputs if "silent" not in f]

        if non_silent_outputs:
            # Combine only non-silent voices for the main output
            combined_output = combine_voice_outputs(non_silent_outputs)
            progress(1.0, desc=f"Done! Generated {len(non_silent_outputs)} voices from MIDI (+ {len(valid_outputs) - len(non_silent_outputs)} silent)")
        elif valid_outputs:
            # Only silent outputs available, use the first one as combined
            combined_output = valid_outputs[0]
            progress(1.0, desc="Done! All voices were silent")
        else:
            raise gr.Error("No voice outputs were created.")

        # Ensure we always return exactly 5 outputs: combined + 4 individual voices
        # Fill any missing slots with the last valid output or create a silent one
        while len(voice_outputs) < 4:
            voice_outputs.append(create_silent_audio_file(len(voice_outputs) + 1))

        return [combined_output] + voice_outputs[:4]  # Ensure exactly 4 voice outputs

    else:
        # Regular audio processing
        progress(0.25, desc="Loading conditioning…")
        win_slow = int(getattr(MODEL.hparams, "window_slow", 1024))
        pr, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=win_slow)

        # original len (for exact decode length) - only for audio files
        orig_len = None
        if not is_midi:
            try:
                wav, sr = torchaudio.load(audio_file)
                orig_len = wav.shape[-1]
            except Exception:
                orig_len = None

        progress(0.5, desc="Generating…")
        out = generate(
            MODEL, pr, amp, rfr, rbd, enc,
            group, subgroup, int(steps), int(seed), float(adapter_scale), float(cfg_weight), float(t0),
            sr_out=32000, instrument_strength=float(instrument_strength), inst_boost=float(inst_boost),
            piano_roll_gain=float(piano_roll_gain), amp_gain=float(amp_gain),
            rframe_gain=float(rframe_gain), rbend_gain=float(rbend_gain), encodec_gain=float(encodec_gain),
            use_overlap_decoder=bool(use_overlap_decoder), original_audio_length=orig_len,
            pitch_fidelity_boost=float(pitch_fidelity_boost), onset_guidance_boost=float(onset_guidance_boost),
            pitch_snap_strength=float(pitch_snap_strength)
        )
        progress(1.0, desc=f"Done! Generated from {file_type}")
        return [out]  # Return as list for consistency

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

def play_all_voices(*voice_files):
    """Play all voice files simultaneously by creating a combined audio file."""
    valid_files = [f for f in voice_files if f and os.path.exists(f)]
    if valid_files:
        return combine_voice_outputs(valid_files)
    return None

def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        gr.Markdown("### dø stem — ControlBranch Pipeline")
        gr.Markdown("*Supports both audio and MIDI files. Upload either an audio file (.wav, .mp3, etc.) or MIDI file (.mid, .midi) for generation.*")
        gr.Markdown("*For MIDI files: Generates 4 separate voices plus a combined version. Use 'Play All Voices' to hear them together.*")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(type="filepath", label="Upload Audio or MIDI File")
                with gr.Row():
                    random_btn = gr.Button("🎤 Random from Manifest", variant="secondary")
                    random_group_btn = gr.Button("🎯 Random from Current Group", variant="secondary")

                group_dd = gr.Dropdown(GROUP_NAMES, label="Instrument Group",
                                      value=GROUP_NAMES[0] if GROUP_NAMES else None)
                subgroup_dd = gr.Dropdown(SUBGROUP_NAMES, label="Instrument Subgroup",
                                         value=SUBGROUP_NAMES[0] if SUBGROUP_NAMES else None)

            with gr.Column(scale=2):
                with gr.Row():
                    seed_slider = gr.Slider(0, 10000, value=0, step=1, label="Seed (0 = random)")
                    steps_slider = gr.Slider(10, 100, value=40, step=1, label="Steps")
                with gr.Row():
                    adapter_slider = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Adapter Scale")
                    cfg_slider = gr.Slider(1.0, 6.0, value=3.0, step=0.1, label="Instrument CFG")
                t0_slider = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="T0 (keep 1.0)")

                instrument_strength = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Instrument Conditioning Strength")
                inst_boost = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="Instrument Token Boost")

                with gr.Row():
                    piano_roll_gain = gr.Slider(0.0, 4.0, value=1.0, step=0.1, label="Piano Roll Gain")
                    amp_gain = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Amplitude Gain")
                with gr.Row():
                    rframe_gain = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="RFrame Gain")
                    rbend_gain = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="RBend Gain")
                encodec_gain = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="EnCodec Gain")

                pitch_fidelity_boost = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Pitch Fidelity Boost")
                onset_guidance_boost = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Onset Guidance Boost")
                pitch_snap_strength = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Pitch Snap Strength")

                use_overlap_decoder = gr.Checkbox(label="Use Overlap Decoder", value=True)
                generate_btn = gr.Button("🎹 Generate", variant="primary")

        # Output section with combined and individual voices
        gr.Markdown("### Generated Outputs")

        with gr.Row():
            combined_output = gr.Audio(label="🎼 Combined (All Voices)", type="filepath")

        with gr.Row():
            voice1_output = gr.Audio(label="🎵 Voice 1", type="filepath")
            voice2_output = gr.Audio(label="🎵 Voice 2", type="filepath")

        with gr.Row():
            voice3_output = gr.Audio(label="🎵 Voice 3", type="filepath")
            voice4_output = gr.Audio(label="🎵 Voice 4", type="filepath")

        with gr.Row():
            play_all_btn = gr.Button("🎶 Play All Voices Together", variant="secondary")

        all_voices_output = gr.Audio(label="🎶 All Voices Playing Together", type="filepath", visible=False)

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
                  use_overlap_decoder, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength]

        # Main generation
        def handle_generation(*args):
            results = run_generation(*args)
            if len(results) == 5:  # MIDI with 4 voices + combined
                return results  # combined, voice1, voice2, voice3, voice4
            else:  # Single audio file
                return [results[0], None, None, None, None]

        generate_btn.click(
            fn=handle_generation,
            inputs=inputs,
            outputs=[combined_output, voice1_output, voice2_output, voice3_output, voice4_output]
        )

        # Play all voices together
        def make_all_voices_visible():
            return gr.update(visible=True)

        play_all_btn.click(
            fn=play_all_voices,
            inputs=[voice1_output, voice2_output, voice3_output, voice4_output],
            outputs=[all_voices_output]
        ).then(
            fn=make_all_voices_visible,
            outputs=[all_voices_output]
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
