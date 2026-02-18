#!/usr/bin/env python3
"""
Drum Processor - Session-Aware Drum Transcription

Groups multi-mic drum recordings by session, runs DrumSep stem separation
on each mic, then picks the BEST mic-stem combo per instrument to produce
one merged drum onset grid per session.

Close mic + separation = cleanest onsets (kick mic → kick stem, etc.)
Overheads fill in cymbals/hh. Full-mix bounces use all stems as fallback.

Usage:
  python3 drum_processor.py --limit 50              # process 50 sessions
  python3 drum_processor.py --session /path/to/dir/  # process one session
  python3 drum_processor.py --limit 50 --force       # reprocess done sessions
"""

import sys
import json
import argparse
import re
import time
import shutil
import os
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Add ZFTurbo repo to path
MSST_DIR = Path("/home/arlo/Data/Music-Source-Separation-Training")
sys.path.insert(0, str(MSST_DIR))

# ============================================================
# CONFIG
# ============================================================
DRUMSEP_CONFIG = Path("/home/arlo/gcs-bucket/Models/drumsep/config_mdx23c.yaml")
DRUMSEP_CKPT = Path("/home/arlo/gcs-bucket/Models/drumsep/drumsep_5stems_mdx23c_jarredou.ckpt")
AUDIO_PATH_LIST = Path("/home/arlo/gcs-bucket/Manifests/drums_files.txt")

# Output: session-level merged transcriptions
OUTPUT_DIR = Path("/home/arlo/gcs-bucket/drum_transcriptions")
# Per-file stems still go here (reused if already computed)
STEMS_DIR = Path("/home/arlo/gcs-bucket/drum_stems")

PROGRESS_FILE = Path("/home/arlo/Data/drum_session_progress.json")
GCS_AUDIO_ROOT = Path("/home/arlo/gcs-bucket/protools")

# Local cache to avoid slow GCS FUSE reads during processing
LOCAL_WORK_DIR = Path("/home/arlo/Data/drum_cache")

SAMPLE_RATE = 44100
MIN_DURATION = 5.0
SILENCE_THRESHOLD = 0.001
STEM_NAMES = ["kick", "snare", "toms", "hh", "cymbals"]

# Onset detection params
ONSET_HOP = 512
ONSET_WAIT = 5
ONSET_DELTA = 0.07

# Duration tolerance for grouping same-take files (seconds)
TAKE_DURATION_TOLERANCE = 5.0

# Flush local cache to GCS when it exceeds this size
CACHE_SIZE_LIMIT = 5 * 1024 * 1024 * 1024  # 5 GB


# ============================================================
# MIC CLASSIFICATION
# ============================================================
# Priority-ordered: first match wins. Patterns are checked against
# the filename stem (no extension) in lowercase.
MIC_PATTERNS = [
    # Kick variants
    ("kick", re.compile(r'(?:kick|kik|bass\s*drum|bassdrum|\bbd\b|subkick|sub\s*kick|beater|d112|beta.?52)', re.I)),
    # Snare variants
    ("snare", re.compile(r'(?:snare|sn[_ ]?t(?:op)?|sn[_ ]?b(?:ot|ottom|tm)?|snr|sn[_ ]side)', re.I)),
    # Hi-hat
    ("hihat", re.compile(r'(?:hihat|hi[\.\-_ ]?hat|\bhh\b|\bhat\b|\bhats\b)', re.I)),
    # Floor tom (before generic tom)
    ("tom_floor", re.compile(r'(?:floor[\.\-_ ]?tom|fl[\.\-_ ]?tom|f[\.\-_ ]?tom|\bfloor\b)', re.I)),
    # Rack tom (before generic tom)
    ("tom_rack", re.compile(r'(?:rack[\.\-_ ]?tom|r[\.\-_ ]?tom|\brtom\b|\brack\b)', re.I)),
    # Generic tom
    ("tom", re.compile(r'(?:\btom\b|hi[\.\-_ ]?tom|lo[\.\-_ ]?tom|high[\.\-_ ]?tom|low[\.\-_ ]?tom)', re.I)),
    # Overhead
    ("overhead", re.compile(r'(?:overhead|\boh\b|\bohl\b|\bohr\b|\bohs\b)', re.I)),
    # Room / ambient
    ("room", re.compile(r'(?:\broom\b|\brm\b|front[\.\-_ ]?of[\.\-_ ]?kit|mono[\.\-_ ]?kit|ambient|drum[\.\-_ ]?side)', re.I)),
    # Specific cymbals
    ("cymbal", re.compile(r'(?:\bride\b|\bcrash\b|\bchina\b|\bsplash\b|\bcymbal\b|\bcym\b|\bstack\b)', re.I)),
    # Full kit mix / bounce
    ("mix", re.compile(r'(?:\bdrums\b|\bdrum\b|\bkit\b)', re.I)),
]


def classify_mic(filename: str) -> str:
    """Classify a drum mic filename into a mic type."""
    # Strip Pro Tools suffixes: "Kick In.04_11" → "Kick In"
    stem = Path(filename).stem
    # Remove trailing .NN_NN patterns (PT region numbering like .04_11)
    clean = re.sub(r'\.\d{2}_\d{2,3}$', '', stem)
    # Remove trailing _N, _NN, _N_NN chains (PT take/region: _1, _01, _1_02)
    clean = re.sub(r'(?:_\d+)+$', '', clean)
    # Remove L/R channel suffix
    clean = re.sub(r'[\._][LR]$', '', clean, flags=re.I)
    # Remove _bip suffix
    clean = re.sub(r'_bip$', '', clean, flags=re.I)

    for mic_type, pattern in MIC_PATTERNS:
        if pattern.search(clean):
            return mic_type
    return "unknown"


# Which stem(s) to extract from each mic type
STEM_PRIORITY: Dict[str, List[str]] = {
    "kick":      ["kick"],
    "snare":     ["snare"],
    "hihat":     ["hh"],
    "tom_floor": ["toms"],
    "tom_rack":  ["toms"],
    "tom":       ["toms"],
    "overhead":  ["cymbals", "hh"],
    "cymbal":    ["cymbals"],
    "mix":       ["kick", "snare", "toms", "hh", "cymbals"],
    "room":      [],       # skip — too ambient
    "unknown":   [],       # skip — can't determine
}

# Fallback priority per instrument (which mic type to prefer)
INSTRUMENT_MIC_PRIORITY: Dict[str, List[str]] = {
    "kick":    ["kick", "mix", "overhead"],
    "snare":   ["snare", "mix", "overhead"],
    "toms":    ["tom_floor", "tom_rack", "tom", "mix", "overhead"],
    "hh":      ["hihat", "overhead", "mix"],
    "cymbals": ["cymbal", "overhead", "mix"],
}


# ============================================================
# PROGRESS TRACKER (session-level)
# ============================================================
class ProgressTracker:
    def __init__(self, path: Path):
        self.path = path
        self.done = set()  # session directory paths
        self.stats = {"ok": 0, "skip": 0, "error": 0}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.done = set(data.get("done", []))
                self.stats = data.get("stats", self.stats)
                print(f"Resumed: {len(self.done)} sessions already done")
            except Exception as e:
                print(f"Warning: could not load progress: {e}")

    def save(self):
        data = {
            "done": sorted(self.done),
            "stats": self.stats,
            "last_save": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.path)

    def is_done(self, session_dir: str) -> bool:
        return session_dir in self.done

    def mark_done(self, session_dir: str, status: str):
        self.done.add(session_dir)
        self.stats[status] = self.stats.get(status, 0) + 1


# ============================================================
# CACHE MANAGEMENT (upload stems to GCS when local cache is full)
# ============================================================
_pending_stem_uploads = {}  # local_stem_dir -> gcs_stem_dir


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes (fast os.scandir)."""
    total = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat().st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += get_dir_size(Path(entry.path))
    except (PermissionError, FileNotFoundError):
        pass
    return total


def flush_cache():
    """Upload pending stems to GCS STEMS_DIR and clear local cache."""
    global _pending_stem_uploads

    if _pending_stem_uploads:
        t0 = time.time()
        uploaded = 0
        for local_dir, gcs_dir in _pending_stem_uploads.items():
            local_path = Path(local_dir)
            gcs_path = Path(gcs_dir)
            if not local_path.exists():
                continue
            try:
                gcs_path.mkdir(parents=True, exist_ok=True)
                for f in local_path.iterdir():
                    if f.is_file():
                        shutil.copy2(str(f), str(gcs_path / f.name))
                uploaded += 1
            except Exception as e:
                print(f"    Warning: failed to upload stems to {gcs_path}: {e}")

        dt = time.time() - t0
        if uploaded:
            print(f"  Uploaded {uploaded} stem dirs to GCS in {dt:.1f}s")
        _pending_stem_uploads = {}

    # Clear local cache
    if LOCAL_WORK_DIR.exists():
        shutil.rmtree(LOCAL_WORK_DIR, ignore_errors=True)
    LOCAL_WORK_DIR.mkdir(parents=True, exist_ok=True)


def flush_cache_if_needed():
    """Flush cache if LOCAL_WORK_DIR exceeds CACHE_SIZE_LIMIT."""
    if not LOCAL_WORK_DIR.exists():
        return
    size = get_dir_size(LOCAL_WORK_DIR)
    if size >= CACHE_SIZE_LIMIT:
        size_gb = size / (1024 ** 3)
        print(f"\n  drum_cache at {size_gb:.1f}GB >= 5GB limit, uploading stems to GCS...")
        flush_cache()


# ============================================================
# MODEL LOADING (lazy singleton)
# ============================================================
_model_cache = {}

def get_drumsep_model(device="cuda"):
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["config"]

    from utils.settings import get_model_from_config
    from utils.model_utils import load_start_checkpoint

    print(f"Loading DrumSep model from {DRUMSEP_CKPT}...")
    t0 = time.time()

    model, config = get_model_from_config("mdx23c", str(DRUMSEP_CONFIG))
    checkpoint = torch.load(str(DRUMSEP_CKPT), weights_only=False, map_location="cpu")

    args = argparse.Namespace(
        model_type="mdx23c",
        start_check_point=str(DRUMSEP_CKPT),
        train_lora_peft=False,
        train_lora_loralib=False,
        lora_checkpoint_peft="",
        lora_checkpoint_loralib="",
        load_only_compatible_weights=False,
    )
    load_start_checkpoint(args, model, checkpoint, type_="inference")

    model = model.to(device)
    model.eval()

    _model_cache["model"] = model
    _model_cache["config"] = config

    print(f"DrumSep loaded in {time.time()-t0:.1f}s | Stems: {list(config.training.instruments)}")
    return model, config


# ============================================================
# SEPARATION (per-file, writes to STEMS_DIR)
# ============================================================
def get_stem_dir(audio_path: str) -> Path:
    """Get the stem output directory for an audio file."""
    p = Path(audio_path)
    try:
        rel = p.parent.relative_to(GCS_AUDIO_ROOT.parent)
        return STEMS_DIR / rel / p.stem
    except ValueError:
        return STEMS_DIR / p.parent.name / p.stem


def separate_file(audio_path: str, device="cuda", force=False,
                  read_from: str = None, write_stems_to: Path = None) -> Optional[Dict[str, Path]]:
    """Separate audio into drum stems.

    Args:
        audio_path: Canonical GCS path (used for stem dir mapping + identity).
        read_from: Local path to read audio from (avoids slow GCS FUSE).
        write_stems_to: Local dir to write stems to (if None, writes to GCS STEMS_DIR).
    """
    audio_path = Path(audio_path)
    gcs_stem_dir = get_stem_dir(str(audio_path))
    gcs_done = gcs_stem_dir / ".done"

    # Check GCS cache first (from previous runs)
    if not force and gcs_done.exists():
        stems = {}
        for s in STEM_NAMES:
            p = gcs_stem_dir / f"{s}.wav"
            if p.exists():
                stems[s] = p
        if len(stems) == len(STEM_NAMES):
            return stems

    from utils.model_utils import demix

    model, config = get_drumsep_model(device)
    instruments = list(config.training.instruments)
    sample_rate = config.audio.sample_rate

    actual_read_path = read_from or str(audio_path)
    try:
        mix, sr = librosa.load(actual_read_path, sr=sample_rate, mono=False)
    except Exception as e:
        print(f"    ERROR reading {audio_path.name}: {e}")
        return None

    if mix.ndim == 1:
        mix = np.stack([mix, mix], axis=0)
    elif mix.shape[0] == 1:
        mix = np.concatenate([mix, mix], axis=0)

    duration = mix.shape[1] / sample_rate
    if duration < MIN_DURATION:
        print(f"    SKIP {audio_path.name}: too short ({duration:.1f}s < {MIN_DURATION}s)")
        return None

    rms = np.sqrt(np.mean(mix ** 2))
    if rms < SILENCE_THRESHOLD:
        print(f"    SKIP {audio_path.name}: silence (rms={rms:.6f})")
        return None

    from utils.audio_utils import normalize_audio, denormalize_audio
    do_normalize = getattr(config.inference, "normalize", False)
    norm_params = None
    if do_normalize:
        mix, norm_params = normalize_audio(mix)

    waveforms = demix(config, model, mix, device, model_type="mdx23c", pbar=False)

    stem_dir = write_stems_to or gcs_stem_dir
    stem_dir.mkdir(parents=True, exist_ok=True)
    stems = {}
    for instr_name in instruments:
        estimates = waveforms[instr_name]
        if do_normalize and norm_params is not None:
            estimates = denormalize_audio(estimates, norm_params)
        out_path = stem_dir / f"{instr_name}.wav"
        sf.write(str(out_path), estimates.T, sample_rate)
        stems[instr_name] = out_path

    (stem_dir / ".done").write_text(f"{time.time()}\n")
    return stems


# ============================================================
# ONSET DETECTION (per-stem)
# ============================================================
def detect_onsets_for_stem(stem_path: Path, force=False) -> Optional[np.ndarray]:
    onset_path = stem_path.with_suffix(".onsets.npy")

    if not force and onset_path.exists():
        return np.load(onset_path)

    try:
        y, sr = librosa.load(str(stem_path), sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"    ERROR reading stem {stem_path.name}: {e}")
        return None

    rms = np.sqrt(np.mean(y ** 2))
    if rms < SILENCE_THRESHOLD * 0.5:
        onset_times = np.array([], dtype=np.float32)
        np.save(onset_path, onset_times)
        return onset_times

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=ONSET_HOP)
    onsets = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=ONSET_HOP,
        onset_envelope=onset_env,
        wait=ONSET_WAIT,
        delta=ONSET_DELTA,
        units="time",
    )
    onset_times = np.array(onsets, dtype=np.float32)
    np.save(onset_path, onset_times)
    return onset_times


# ============================================================
# SESSION GROUPING
# ============================================================
def get_file_duration(path: str) -> Optional[float]:
    """Get audio duration without loading the full file."""
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        return None


def fast_silence_check(path: str, threshold: float = SILENCE_THRESHOLD, max_frames: int = 441000) -> bool:
    """Quick silence check by reading up to max_frames (~10s at 44.1kHz).

    Returns True if file is silent (should be skipped).
    """
    try:
        info = sf.info(path)
        frames_to_read = min(info.frames, max_frames)
        data, _ = sf.read(path, frames=frames_to_read, dtype='float32')
        rms = np.sqrt(np.mean(data ** 2))
        if rms < threshold:
            print(f"      SKIP {Path(path).name}: silence (rms={rms:.6f})")
            return True
        return False
    except Exception as e:
        print(f"      SKIP {Path(path).name}: unreadable ({e})")
        return True


def group_sessions(paths: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Group audio paths by parent directory (session).

    Returns: {session_dir: [(path, mic_type), ...]}
    """
    sessions = defaultdict(list)
    for p in paths:
        parent = str(Path(p).parent)
        mic_type = classify_mic(Path(p).name)
        sessions[parent].append((p, mic_type))
    return dict(sessions)


def group_takes(files: List[Tuple[str, str]]) -> List[List[Tuple[str, str, float]]]:
    """Sub-group session files by approximate duration into takes.

    Returns list of take groups, each a list of (path, mic_type, duration).
    """
    # Get durations
    with_dur = []
    for path, mic_type in files:
        dur = get_file_duration(path)
        if dur is not None and dur >= MIN_DURATION:
            with_dur.append((path, mic_type, dur))

    if not with_dur:
        return []

    # Sort by duration
    with_dur.sort(key=lambda x: x[2])

    # Cluster by duration proximity
    takes = []
    current_take = [with_dur[0]]
    for item in with_dur[1:]:
        if abs(item[2] - current_take[0][2]) <= TAKE_DURATION_TOLERANCE:
            current_take.append(item)
        else:
            takes.append(current_take)
            current_take = [item]
    takes.append(current_take)

    return takes


# ============================================================
# SESSION PROCESSING
# ============================================================
def process_session(
    session_dir: str,
    files: List[Tuple[str, str]],
    device: str,
    force: bool,
) -> Optional[dict]:
    """Process all mic files in a session, merge onsets into one transcription."""

    # Group into takes by duration
    takes = group_takes(files)
    if not takes:
        return None

    # Get relative path for output
    try:
        rel = Path(session_dir).relative_to(GCS_AUDIO_ROOT.parent)
        out_base = OUTPUT_DIR / rel
    except ValueError:
        out_base = OUTPUT_DIR / Path(session_dir).name

    session_result = {
        "session_dir": session_dir,
        "session_name": Path(session_dir).parent.name if Path(session_dir).name == "Audio Files" else Path(session_dir).name,
        "takes": [],
    }

    num_takes = len(takes)
    for take_idx, take_files in enumerate(takes):
        take_result = process_take(take_files, take_idx, num_takes, out_base, device, force)
        if take_result:
            session_result["takes"].append(take_result)

    if not session_result["takes"]:
        return None

    return session_result


def process_take(
    take_files: List[Tuple[str, str, float]],
    take_idx: int,
    num_takes: int,
    out_base: Path,
    device: str,
    force: bool,
) -> Optional[dict]:
    """Process one take (group of same-duration mic files) and merge onsets."""

    out_dir = out_base / f"take_{take_idx}" if num_takes > 1 else out_base
    done_marker = out_dir / "session_onsets.json"

    if not force and done_marker.exists():
        try:
            with open(done_marker) as f:
                return json.load(f)
        except Exception:
            pass

    duration = max(f[2] for f in take_files)

    # Log the take
    mic_summary = defaultdict(int)
    for _, mt, _ in take_files:
        mic_summary[mt] += 1
    print(f"  Take {take_idx}: {len(take_files)} mics ({dict(mic_summary)}) ~{duration:.0f}s")

    # ---- Skip takes that can only produce 1 instrument ----
    usable_instruments = set()
    for _, mt, _ in take_files:
        for stem in STEM_PRIORITY.get(mt, []):
            usable_instruments.add(stem)
    if len(usable_instruments) < 2:
        print(f"    SKIP: only {len(usable_instruments)} instrument(s) possible ({usable_instruments or 'none'})")
        return None

    # ---- Pre-filter silence on GCS before copying ----
    non_silent_files = []
    for gcs_path, mic_type, dur in take_files:
        stems_to_use = STEM_PRIORITY.get(mic_type, [])
        if not stems_to_use:
            continue
        # Check if already separated on GCS (cached = not silent)
        gcs_stem_dir = get_stem_dir(gcs_path)
        if not force and (gcs_stem_dir / ".done").exists():
            non_silent_files.append((gcs_path, mic_type, dur, True))  # cached=True
            continue
        # Fast silence check directly on GCS FUSE (reads ~10s of audio)
        if fast_silence_check(gcs_path):
            continue
        non_silent_files.append((gcs_path, mic_type, dur, False))  # cached=False

    if not non_silent_files:
        print(f"    SKIP: all files silent or unreadable")
        return None

    # Re-check instrument coverage after silence filtering
    usable_after = set()
    for _, mt, _, _ in non_silent_files:
        for stem in STEM_PRIORITY.get(mt, []):
            usable_after.add(stem)
    if len(usable_after) < 2:
        print(f"    SKIP: only {len(usable_after)} instrument(s) after silence filter ({usable_after})")
        return None

    # ---- Batch copy non-silent audio to local disk ----
    batch_dir = LOCAL_WORK_DIR / f"batch_{os.getpid()}_{take_idx}"
    audio_dir = batch_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    t_copy = time.time()
    local_map = {}  # gcs_path -> local_path
    cached_count = 0
    for gcs_path, mic_type, dur, cached in non_silent_files:
        if cached:
            cached_count += 1
            continue
        # Copy to local
        local_path = audio_dir / Path(gcs_path).name
        if not local_path.exists():
            try:
                shutil.copy2(gcs_path, str(local_path))
            except Exception as e:
                print(f"      SKIP {Path(gcs_path).name}: copy failed ({e})")
                continue
        local_map[gcs_path] = str(local_path)

    copy_time = time.time() - t_copy
    if local_map:
        print(f"    Copied {len(local_map)} files to local in {copy_time:.1f}s ({cached_count} cached)")
    elif cached_count:
        print(f"    All {cached_count} files cached on GCS")

    # ---- Separate + detect onsets ----
    mic_onsets: Dict[str, Dict[str, Tuple[np.ndarray, str]]] = defaultdict(dict)

    for gcs_path, mic_type, dur, cached in non_silent_files:
        stems_to_use = STEM_PRIORITY.get(mic_type, [])
        if not stems_to_use:
            continue

        # Determine local paths for this file
        local_read = local_map.get(gcs_path)
        local_stem_dir = batch_dir / "stems" / Path(gcs_path).stem if local_read else None

        stems = separate_file(
            gcs_path, device=device, force=force,
            read_from=local_read, write_stems_to=local_stem_dir,
        )
        if stems is None:
            continue

        # Register local stems for deferred GCS upload
        if local_stem_dir and local_stem_dir.exists():
            _pending_stem_uploads[str(local_stem_dir)] = str(get_stem_dir(gcs_path))

        # Detect onsets on the stems we care about (local or GCS, wherever they are)
        for stem_name in stems_to_use:
            stem_path = stems.get(stem_name)
            if stem_path is None or not stem_path.exists():
                continue
            onsets = detect_onsets_for_stem(stem_path, force=force)
            if onsets is not None:
                mic_onsets[stem_name][mic_type] = (onsets, Path(gcs_path).name)

    # ---- Cleanup copied audio (stems kept for deferred GCS upload) ----
    audio_cleanup = batch_dir / "audio"
    if audio_cleanup.exists():
        shutil.rmtree(audio_cleanup, ignore_errors=True)

    # ---- Merge: pick best mic for each instrument ----
    merged = {}
    mics_used = {}
    for instrument, mic_priority in INSTRUMENT_MIC_PRIORITY.items():
        best_onsets = None
        best_source = None

        for preferred_mic in mic_priority:
            if preferred_mic in mic_onsets.get(instrument, {}):
                best_onsets, best_source = mic_onsets[instrument][preferred_mic]
                break

        if best_onsets is not None:
            merged[instrument] = {
                "count": len(best_onsets),
                "times": best_onsets.tolist(),
                "source_mic": best_source,
            }
            mics_used[instrument] = best_source

    if not merged:
        return None

    # Build result
    result = {
        "session_name": out_base.parent.name if out_base.name == "Audio Files" else out_base.name,
        "take": take_idx,
        "duration": round(duration, 2),
        "mic_types": dict(mic_summary),
        "mics_used": mics_used,
        "original_files": {Path(p).name: mt for p, mt, _, _ in non_silent_files},
        "onsets": merged,
        "total_onsets": sum(v["count"] for v in merged.values()),
    }

    # Also store original file paths for audio playback
    result["audio_paths"] = {}
    for p, mt, _, _ in non_silent_files:
        if mt not in result["audio_paths"]:
            result["audio_paths"][mt] = p

    # Write merged output
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "session_onsets.json", "w") as f:
        json.dump(result, f, indent=2)

    total = result["total_onsets"]
    instruments_str = " | ".join(f"{k}:{v['count']}" for k, v in merged.items())
    print(f"    MERGED: {instruments_str} (total: {total})")

    return result


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Session-aware drum transcription: separation + onset detection + merge"
    )
    parser.add_argument("--session", type=str, default=None,
                        help="Process a single session directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max sessions to process")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess already-done sessions")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Inference device (default: cuda)")
    parser.add_argument("--min-mics", type=int, default=2,
                        help="Skip sessions with fewer than N mic files (default: 2)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # --- Single session mode ---
    if args.session:
        session_dir = args.session.rstrip("/")
        if not Path(session_dir).exists():
            print(f"ERROR: Directory not found: {session_dir}")
            sys.exit(1)

        # Find all wav files in the directory
        wav_files = sorted(Path(session_dir).glob("*.wav"))
        if not wav_files:
            print(f"No .wav files in {session_dir}")
            sys.exit(1)

        files = [(str(f), classify_mic(f.name)) for f in wav_files]
        print(f"Session: {session_dir}")
        print(f"Files: {len(files)}")
        for p, mt in files:
            print(f"  {Path(p).name:40s} → {mt}")

        get_drumsep_model(args.device)
        result = process_session(session_dir, files, args.device, args.force)
        if result:
            print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
        else:
            print("\nNo result (all files skipped or failed)")
        return

    # --- Batch mode ---
    print(f"Loading audio paths from {AUDIO_PATH_LIST}...")
    paths = []
    with open(AUDIO_PATH_LIST, "r") as f:
        for line in f:
            p = line.strip()
            if p and p.endswith(".wav"):
                paths.append(p)
    print(f"Total drum files: {len(paths)}")

    # Group by session
    sessions = group_sessions(paths)
    print(f"Total sessions: {len(sessions)}")

    # Filter by min mics
    sessions = {k: v for k, v in sessions.items() if len(v) >= args.min_mics}
    print(f"Sessions with >= {args.min_mics} mics: {len(sessions)}")

    # Sort sessions by path for deterministic order
    session_list = sorted(sessions.items())

    # Load progress tracker
    tracker = ProgressTracker(PROGRESS_FILE)

    # Filter already-done
    if not args.force:
        before = len(session_list)
        session_list = [(d, f) for d, f in session_list if not tracker.is_done(d)]
        skipped = before - len(session_list)
        if skipped > 0:
            print(f"Skipping {skipped} already-completed sessions")

        # Also check for existing output
        still_todo = []
        gcs_checked = 0
        for session_dir, files in session_list:
            try:
                rel = Path(session_dir).relative_to(GCS_AUDIO_ROOT.parent)
                out_path = OUTPUT_DIR / rel / "session_onsets.json"
            except ValueError:
                out_path = OUTPUT_DIR / Path(session_dir).name / "session_onsets.json"

            # Check for take_0 subdirectory too
            out_path_take = out_path.parent / "take_0" / "session_onsets.json"
            if out_path.exists() or out_path_take.exists():
                tracker.mark_done(session_dir, "ok")
                gcs_checked += 1
            else:
                still_todo.append((session_dir, files))

        if gcs_checked > 0:
            print(f"Skipping {gcs_checked} sessions already on GCS")
            tracker.save()
        session_list = still_todo

    if args.limit:
        session_list = session_list[:args.limit]

    if not session_list:
        print("All sessions already processed!")
        return

    # Clear leftover cache from interrupted runs
    if LOCAL_WORK_DIR.exists():
        leftover = get_dir_size(LOCAL_WORK_DIR)
        if leftover > 0:
            print(f"Clearing {leftover / (1024**3):.1f}GB leftover cache from previous run")
            shutil.rmtree(LOCAL_WORK_DIR, ignore_errors=True)
    LOCAL_WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load model
    get_drumsep_model(args.device)

    print(f"\nDevice: {args.device}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Stems: {STEMS_DIR}")
    print(f"Processing {len(session_list)} sessions...")
    print("=" * 60)

    t_start = time.time()
    ok_count = 0
    skip_count = 0
    err_count = 0

    for i, (session_dir, files) in enumerate(session_list):
        session_name = Path(session_dir).parent.name if Path(session_dir).name == "Audio Files" else Path(session_dir).name

        print(f"\n[{i+1}/{len(session_list)}] {session_name} ({len(files)} files)")

        try:
            t0 = time.time()
            result = process_session(session_dir, files, args.device, args.force)
            dt = time.time() - t0

            if result and result["takes"]:
                ok_count += 1
                total_onsets = sum(t.get("total_onsets", 0) for t in result["takes"])
                print(f"  DONE in {dt:.1f}s | {len(result['takes'])} take(s) | {total_onsets} total onsets")
                tracker.mark_done(session_dir, "ok")
            else:
                skip_count += 1
                print(f"  SKIPPED (no usable mics/takes)")
                tracker.mark_done(session_dir, "skip")

        except Exception as e:
            err_count += 1
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            tracker.mark_done(session_dir, "error")

        # Flush cache if over 5GB
        flush_cache_if_needed()

        # Periodic save
        if (i + 1) % 5 == 0:
            tracker.save()

        # Periodic summary
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            eta = (len(session_list) - i - 1) / rate if rate > 0 else 0
            print(f"\n--- {i+1}/{len(session_list)} | {rate:.1f} sessions/min | ETA {eta:.0f}m ---")

    # Upload any remaining cached stems
    flush_cache()

    # Final save
    tracker.save()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  OK: {ok_count} | Skip: {skip_count} | Error: {err_count}")
    print(f"  Progress: {PROGRESS_FILE}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
