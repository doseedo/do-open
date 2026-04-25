"""
Multi-track MIDI conditioning for Stemphonic sub-mix conditioning.

Two data sources:
1. **Pitched instruments**: `.pianoroll.npy` files in Conditioning/ folder
   - Shape [128, T] binary piano roll at some frame rate
   - Derived from BasicPitch transcriptions

2. **Drums**: `session_onsets.json` in drum_transcriptions_v2/
   - Onset times + velocities per drum type (kick, snare, hh, etc.)
   - Converted to a drum onset roll at VAE frame rate

These are kept separate in a multi-channel representation so the model
can distinguish pitched MIDI from drum onset patterns:
  - Channels 0-127: pitched piano roll (from BasicPitch)
  - Channels 128-143: drum onset roll (16 drum classes mapped from onset keys)

Total: [144, T] multitrack MIDI tensor per stem.
Sub-mix = union of all held-out stems' MIDI → perfectly linear, no VAE needed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

PITCHED_BINS = 128
DRUM_BINS = 16
VOCAL_EXTRA_BINS = 2  # word onset + continuous F0
TOTAL_MIDI_BINS = PITCHED_BINS + DRUM_BINS + VOCAL_EXTRA_BINS  # 146
VAE_HZ = 25.0

# Map drum onset keys to drum roll channels
DRUM_KEY_TO_CHANNEL = {
    "kick": 0, "kd": 0, "bd": 0,
    "snare": 1, "sn": 1, "sd": 1,
    "hh": 2, "hihat": 2, "hi-hat": 2, "hi_hat": 2,
    "hh_open": 3, "oh": 3, "open_hihat": 3,
    "tom": 4, "tom1": 4, "high_tom": 4, "rack": 4,
    "tom2": 5, "mid_tom": 5, "floor_tom": 5, "floor": 5,
    "tom3": 6, "low_tom": 6,
    "crash": 7, "cr": 7,
    "ride": 8, "rd": 8,
    "china": 9,
    "splash": 10,
    "bell": 11, "ride_bell": 11,
    "rim": 12, "rimshot": 12, "cross_stick": 12,
    "ghost": 13, "ghost_note": 13,
    "flam": 14,
    "other": 15,
}


MULTIF0_HZ = 86.13  # BasicPitch: sr=22050, hop=256 → 86.13 Hz
MULTIF0_PITCH_OFFSET = 21  # BasicPitch 88-key range starts at MIDI 21 (A0)
MULTIF0_BINS = 88
MULTIF0_CONFIDENCE_FLOOR = 0.15  # Below this = background noise, not a note


def load_multif0_npy(
    multif0_path: str,
    duration_frames: int,
) -> Optional[torch.Tensor]:
    """Load raw BasicPitch MultiF0 posterior as confidence-weighted pianoroll.

    MultiF0 files are [T_86Hz, 88] float32 with per-frame per-pitch confidence
    scores (0-1). This is the raw model output before note extraction — it
    preserves continuous confidence that MIDI conversion discards.

    Returns:
        [128, T] confidence-weighted pianoroll at VAE_HZ, or None if not found
    """
    try:
        raw = np.load(multif0_path)
    except Exception as e:
        logger.debug("Failed to load MultiF0 %s: %s", multif0_path, e)
        return None

    raw = torch.from_numpy(raw).float()
    if raw.dim() != 2 or raw.shape[1] != MULTIF0_BINS:
        return None

    # raw is [T_86Hz, 88]. Transpose to [88, T_86Hz]
    raw = raw.T

    # Map 88 pitch bins (A0-C8) into 128-bin MIDI range
    pr = torch.zeros(PITCHED_BINS, raw.shape[1])
    pr[MULTIF0_PITCH_OFFSET:MULTIF0_PITCH_OFFSET + MULTIF0_BINS] = raw

    # Subtract background noise floor so only real notes have nonzero values
    note_mask = pr > MULTIF0_CONFIDENCE_FLOOR
    pr = (pr - MULTIF0_CONFIDENCE_FLOOR).clamp(min=0)
    # Renormalize to 0-1, then blend: 0.5 base + 0.5 * confidence
    # Detected notes range 0.5-1.0, background stays 0.0
    pr_max = pr.max()
    if pr_max > 0:
        pr = (0.5 + 0.5 * (pr / pr_max)) * note_mask.float()

    # Resample from 86.13Hz to VAE_HZ (25Hz)
    target_len = int(pr.shape[1] * VAE_HZ / MULTIF0_HZ)
    if target_len > 0:
        pr = torch.nn.functional.interpolate(
            pr.unsqueeze(0), size=target_len, mode='linear', align_corners=False
        ).squeeze(0)

    # Crop/pad to duration_frames
    T = pr.shape[1]
    if T >= duration_frames:
        pr = pr[:, :duration_frames]
    else:
        pr = torch.nn.functional.pad(pr, (0, duration_frames - T))

    return pr


_pianoroll_cache: Dict[str, np.ndarray] = {}  # path → raw numpy array (persists per worker)
_drum_onset_cache: Dict[str, dict] = {}  # path → parsed JSON dict (persists per worker)


def load_piano_roll_npy(
    pianoroll_path: str,
    duration_frames: int,
    source_hz: float = 25.0,
) -> torch.Tensor:
    """Load a .pianoroll.npy file and resample to VAE frame rate.

    Args:
        pianoroll_path: path to .pianoroll.npy file
        duration_frames: target number of output frames at VAE_HZ
        source_hz: frame rate of the stored piano roll

    Returns:
        [128, T] piano roll at VAE_HZ (binary — no confidence info)
    """
    if pianoroll_path in _pianoroll_cache:
        pr = torch.from_numpy(_pianoroll_cache[pianoroll_path]).float()
    else:
        try:
            raw = np.load(pianoroll_path)
            _pianoroll_cache[pianoroll_path] = raw
            pr = torch.from_numpy(raw).float()
        except Exception as e:
            logger.debug("Failed to load pianoroll %s: %s", pianoroll_path, e)
            return torch.zeros(PITCHED_BINS, duration_frames)

    # Handle different shapes
    if pr.dim() == 1:
        return torch.zeros(PITCHED_BINS, duration_frames)
    if pr.shape[0] != PITCHED_BINS and pr.shape[1] == PITCHED_BINS:
        pr = pr.T  # Transpose if [T, 128] → [128, T]

    # Resample if different frame rate
    if source_hz != VAE_HZ and pr.shape[-1] > 1:
        target_len = int(pr.shape[-1] * VAE_HZ / source_hz)
        pr = torch.nn.functional.interpolate(
            pr.unsqueeze(0), size=target_len, mode='nearest'
        ).squeeze(0)

    # Crop/pad to duration_frames
    T = pr.shape[-1]
    if T >= duration_frames:
        pr = pr[:, :duration_frames]
    else:
        pr = torch.nn.functional.pad(pr, (0, duration_frames - T))

    # Pass through values: pre-computed .pianoroll.npy are binary (0/1),
    # MIDI-converted .npy have velocity (0.5-1.0). Both work as-is.
    return pr.clamp(0, 1).float()


def drum_onsets_to_roll(
    onsets_path: str,
    duration_frames: int,
    frame_rate: float = VAE_HZ,
    instrument_filter: Optional[set] = None,
) -> torch.Tensor:
    """Convert drum session_onsets.json to a drum onset roll [16, T].

    Args:
        onsets_path: path to session_onsets.json
        duration_frames: number of output frames
        frame_rate: temporal resolution (25Hz)
        instrument_filter: if set, only include these onset keys (e.g. {"kick"}).
            None = include all (for overhead/room mics or full-kit stems).

    Returns:
        [16, T] drum onset roll (1 at onset frames, 0 elsewhere)
    """
    roll = torch.zeros(DRUM_BINS, duration_frames)

    if onsets_path in _drum_onset_cache:
        data = _drum_onset_cache[onsets_path]
    else:
        try:
            with open(onsets_path) as f:
                data = json.load(f)
            _drum_onset_cache[onsets_path] = data
        except Exception as e:
            logger.debug("Failed to load drum onsets %s: %s", onsets_path, e)
            return roll

    onsets = data.get("onsets", {})
    for drum_key, drum_data in onsets.items():
        # Filter: skip instruments not relevant to this mic
        if instrument_filter is not None:
            key_lower = drum_key.lower().strip()
            if not any(filt in key_lower for filt in instrument_filter):
                continue

        # Map drum key to channel
        channel = None
        key_lower = drum_key.lower().strip()
        for pattern, ch in DRUM_KEY_TO_CHANNEL.items():
            if pattern in key_lower:
                channel = ch
                break
        if channel is None:
            channel = DRUM_KEY_TO_CHANNEL["other"]

        # Get onset times
        times = drum_data.get("times", [])
        velocities = drum_data.get("velocities", [1.0] * len(times))

        for t, v in zip(times, velocities):
            frame = int(t * frame_rate)
            if 0 <= frame < duration_frames:
                roll[channel, frame] = max(roll[channel, frame], 0.5 + 0.5 * min(1.0, v))

    return roll


# Map track/filename patterns to which drum onset instruments they capture.
# Close mics hear primarily one instrument; overhead/room mics hear everything.
_DRUM_MIC_TO_INSTRUMENTS = {
    # Close mics → single instrument
    "kick": {"kick"},
    "kd": {"kick"},
    "bd": {"kick"},
    "snare": {"snare"},
    "sn_": {"snare"},
    "snr": {"snare"},
    "hihat": {"hh"},
    "hi_hat": {"hh"},
    "hi-hat": {"hh"},
    "hh": {"hh"},
    "hat": {"hh"},
    "tom": {"toms"},
    "rack": {"toms"},
    "floor": {"toms"},
    # Overhead / room / bus = full kit (None means no filter)
    "oh": None,
    "overhead": None,
    "room": None,
    "amb": None,
    "drum_s": None,
    "drum_bus": None,
    "drum_comp": None,
    "compressed": None,
}


def _get_drum_mic_filter(track_name: str) -> Optional[set]:
    """Determine which drum instruments a mic track captures.

    Returns a set of instrument keys to include, or None for full kit
    (overheads, room mics, drum bus).
    """
    name_lower = track_name.lower().replace(" ", "_")
    for pattern, instruments in _DRUM_MIC_TO_INSTRUMENTS.items():
        if pattern in name_lower:
            return instruments
    # Unknown drum track → full kit (safe default)
    return None


def get_stem_midi_representation(
    stem_info: Dict,
    duration_frames: int,
    conditioning_root: str = "/scratch/Conditioning",
    drum_root: str = "/scratch/drum_transcriptions_v2",
    track_name: str = "",
) -> torch.Tensor:
    """Get the multi-track MIDI representation for a single stem.

    Returns [146, T]:
      channels 0-127:   pitched piano roll (BasicPitch or pianoroll.npy)
      channels 128-143: drum onset roll (16 drum classes)
      channel 144:      word onset markers (vocals only, from lyrics word_timings)
      channel 145:      continuous F0 (vocals only, normalized 0-1)

    For drums: uses track_name to determine which drum instruments this mic
    captures. Close mics (kick, snare) get only their instrument's onsets.
    Overhead/room mics get the full kit.

    For vocals: pitched channels carry the melody from BasicPitch,
    channel 144 carries word timing from lyrics JSON,
    channel 145 carries F0 contour from Conditioning/ if available.
    """
    result = torch.zeros(TOTAL_MIDI_BINS, duration_frames)
    group = stem_info.get("group", "")
    latent_path = stem_info.get("latent_path", "")

    if group in ("drums", "percussion"):
        mic_name = track_name or stem_info.get("filename", "")
        instrument_filter = _get_drum_mic_filter(mic_name)
        is_close = instrument_filter is not None  # Close mic = filtered to specific instruments
        drum_path = _find_drum_onsets(latent_path, drum_root, is_close_mic=is_close)
        if drum_path:
            drum_roll = drum_onsets_to_roll(
                drum_path, duration_frames,
                instrument_filter=instrument_filter,
            )
            result[PITCHED_BINS:PITCHED_BINS + DRUM_BINS] = drum_roll
        # No BasicPitch fallback for drums — drum transcriptions or nothing
    else:
        # Pitched piano roll (binary)
        midi_path = stem_info.get("midi_path", "")
        pr_path = _find_pianoroll(latent_path, conditioning_root, midi_path=midi_path)
        if pr_path:
            pitched = load_piano_roll_npy(pr_path, duration_frames)
            result[:PITCHED_BINS] = pitched

    # Vocal-specific: word onset timing + F0
    if group == "voice":
        vc_path = stem_info.get("vocal_cond_path", "")
        if vc_path and os.path.exists(vc_path):
            result = _add_vocal_timing(result, vc_path, latent_path,
                                        duration_frames, conditioning_root)

    return result


def _add_vocal_timing(
    result: torch.Tensor,
    vc_path: str,
    latent_path: str,
    duration_frames: int,
    conditioning_root: str,
) -> torch.Tensor:
    """Add word onset and F0 channels for vocal stems.

    Channel 144: word onsets (1 at each word start, 0 elsewhere)
    Channel 145: continuous F0 normalized to 0-1 (if available)
    """
    import json as _json

    # Word onsets from lyrics JSON
    try:
        with open(vc_path) as f:
            vc = _json.load(f)
        word_timings = vc.get("word_timings", [])
        for timing in word_timings:
            if len(timing) >= 2:
                onset_time = float(timing[0])
                frame = int(onset_time * VAE_HZ)
                if 0 <= frame < duration_frames:
                    result[PITCHED_BINS + DRUM_BINS, frame] = 1.0
    except Exception as e:
        logger.warning("Failed to load vocal word timings from %s: %s", vc_path, e)

    # Continuous F0 from Conditioning/ (if available)
    try:
        lp = latent_path.replace("/mnt/data/system_home/arlo/", "/home/arlo/")
        if "Latents2/" in lp:
            f0_rel = lp.split("Latents2/")[-1]
            # Try scratch first, then FUSE
            f0_path = None
            for cond_root in ["/scratch/Conditioning"]:
                candidate = os.path.join(cond_root, f0_rel).replace(".vae.pt", ".f0.npy")
                if os.path.exists(candidate):
                    f0_path = candidate
                    break
        else:
            f0_path = lp.replace("Latents2/", "Conditioning/").replace(".vae.pt", ".f0.npy")
        if f0_path and os.path.exists(f0_path):
            import numpy as _np
            f0 = _np.load(f0_path)
            f0_tensor = torch.from_numpy(f0).float()
            # Resample to VAE frame rate if needed
            if len(f0_tensor) != duration_frames:
                f0_tensor = torch.nn.functional.interpolate(
                    f0_tensor.unsqueeze(0).unsqueeze(0),
                    size=duration_frames, mode="linear", align_corners=False,
                ).squeeze()
            # Normalize F0 to 0-1 (log scale, vocal range ~80-1000Hz)
            f0_tensor = f0_tensor.clamp(80, 1000)
            f0_tensor = (torch.log2(f0_tensor) - torch.log2(torch.tensor(80.0))) / \
                        (torch.log2(torch.tensor(1000.0)) - torch.log2(torch.tensor(80.0)))
            result[PITCHED_BINS + DRUM_BINS + 1, :duration_frames] = f0_tensor[:duration_frames]
    except Exception as e:
        logger.warning("Failed to load vocal F0 from %s: %s", latent_path, e)

    return result


_midi_to_pr_cache: Dict[str, Optional[str]] = {}

# Global latent→pianoroll lookup (loaded once from precache index)
_latent_to_pianoroll: Dict[str, str] = {}


def load_pianoroll_cache(cache_path: str = "/scratch/stemphonic_data/pianoroll_cache/latent_to_pianoroll.json"):
    """Load the precached latent→pianoroll mapping. Call once at startup."""
    global _latent_to_pianoroll
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            _latent_to_pianoroll = json.load(f)
        logger.info("Loaded %d precached pianoroll mappings", len(_latent_to_pianoroll))


def _midi_to_pianoroll(mid_path: str, cache_dir: str = "/scratch/midi_pr_cache") -> Optional[str]:
    """Convert a .mid file to [128, T] piano roll .npy, cached on disk."""
    if mid_path in _midi_to_pr_cache:
        cached = _midi_to_pr_cache[mid_path]
        if cached and os.path.exists(cached):
            return cached
        if cached is None:
            return None

    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(mid_path)
    except Exception:
        # Try mido as fallback
        try:
            from mido import MidiFile
            mid = MidiFile(mid_path)
            # Simple conversion: extract note events
            roll = np.zeros((128, 100000), dtype=np.float32)
            t = 0.0
            for msg in mid:
                t += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    frame = int(t * VAE_HZ)
                    if frame < roll.shape[1]:
                        roll[msg.note, frame] = 0.5 + 0.5 * (msg.velocity / 127.0)
                elif msg.type == 'note_on' and msg.velocity == 0 or msg.type == 'note_off':
                    pass  # onset-only representation
            # Trim trailing zeros
            last_frame = max(1, np.max(np.where(roll.sum(axis=0) > 0)[0]) + 1) if roll.sum() > 0 else 1
            roll = roll[:, :last_frame]
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, os.path.basename(mid_path).replace(".mid", ".pianoroll.npy"))
            np.save(cache_path, roll)
            _midi_to_pr_cache[mid_path] = cache_path
            return cache_path
        except Exception:
            _midi_to_pr_cache[mid_path] = None
            return None

    # pretty_midi path
    try:
        end_time = pm.get_end_time()
        n_frames = max(1, int(end_time * VAE_HZ) + 1)
        roll = np.zeros((128, n_frames), dtype=np.float32)
        for inst in pm.instruments:
            for note in inst.notes:
                start_frame = int(note.start * VAE_HZ)
                end_frame = int(note.end * VAE_HZ)
                if start_frame < n_frames:
                    val = 0.5 + 0.5 * (note.velocity / 127.0)
                    roll[note.pitch, start_frame:min(end_frame + 1, n_frames)] = val
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, os.path.basename(mid_path).replace(".mid", ".pianoroll.npy"))
        np.save(cache_path, roll)
        _midi_to_pr_cache[mid_path] = cache_path
        return cache_path
    except Exception:
        _midi_to_pr_cache[mid_path] = None
        return None


def _find_pianoroll(latent_path: str, conditioning_root: str,
                    midi_path: str = "") -> Optional[str]:
    """Find or create a pianoroll for a stem.

    Tries in order:
    0. Precached pianoroll index (fastest — precomputed from BasicPitch .mid)
    1. Pre-computed .pianoroll.npy in Conditioning/ (fast)
    2. Convert .mid file from BasicPitch on the fly (slow, cached)
    """
    if not latent_path:
        return None

    # Try precache index first (populated by precache_pianorolls.py)
    if _latent_to_pianoroll:
        cached = _latent_to_pianoroll.get(latent_path)
        if cached and os.path.exists(cached):
            return cached

    # Try pre-computed pianoroll in Conditioning/
    lp = latent_path.replace("/mnt/data/system_home/arlo/", "/home/arlo/")
    if "Latents2/" in lp:
        rel = lp.split("Latents2/")[-1]
        pr_path = os.path.join(conditioning_root, rel).replace(".vae.pt", ".pianoroll.npy")
        stem_base = Path(pr_path).stem.replace(".pianoroll", "")
        pr_dir = str(Path(pr_path).parent)
        candidates = [
            pr_path,
            os.path.join(pr_dir, f"{stem_base}_bip_1.pianoroll.npy"),
            os.path.join(pr_dir, f"{stem_base}_bip.pianoroll.npy"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

    # Try converting .mid file on the fly from BasicPitch
    # BasicPitch uses DATE/New/SESSION/... (no protools/ prefix)
    BP_ROOTS = ["/scratch/BasicPitch", "/home/arlo/gcs-bucket/BasicPitch"]

    if midi_path:
        if os.path.exists(midi_path):
            return _midi_to_pianoroll(midi_path)
        for bp_root in BP_ROOTS:
            mp = midi_path
            for old in ["/scratch/BasicPitch/", "/home/arlo/gcs-bucket/BasicPitch/"]:
                mp = mp.replace(old, bp_root + "/")
            if os.path.exists(mp):
                return _midi_to_pianoroll(mp)

    # Derive BasicPitch path from latent path
    if "Latents2/" in lp:
        rel = lp.split("Latents2/")[-1]  # protools/DATE/New/SESSION/.../STEM.vae.pt
        stem_mid = rel.replace(".vae.pt", ".mid")
        # Try with protools/ prefix and without (BasicPitch drops the prefix)
        stripped = stem_mid
        for prefix in ["protools/", "protoolsA/"]:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                break
        for bp_root in BP_ROOTS:
            for candidate in [stem_mid, stripped]:
                bp_path = os.path.join(bp_root, candidate)
                if os.path.exists(bp_path):
                    return _midi_to_pianoroll(bp_path)

    return None


def _find_multif0(
    latent_path: str,
    multif0_root: str = "/home/arlo/gcs-bucket/MultiF0",
) -> Optional[str]:
    """Find MultiF0 raw posterior .npy for a stem.

    Path mapping: Latents2/protools/DATE/New/SESSION/Audio Files/STEM.vae.pt
               → MultiF0/protools/DATE/New/SESSION/Audio Files/STEM.npy
    """
    if not latent_path:
        return None
    lp = latent_path.replace("/mnt/data/system_home/arlo/", "/home/arlo/")
    if "Latents2/" not in lp:
        return None
    rel = lp.split("Latents2/")[-1]  # protools/DATE/New/SESSION/.../STEM.vae.pt
    npy_path = os.path.join(multif0_root, rel).replace(".vae.pt", ".npy")
    if os.path.exists(npy_path):
        return npy_path
    return None


_drum_richest_cache: Dict[str, Optional[str]] = {}  # session_dir → richest onset path


def _find_drum_onsets(latent_path: str, drum_root: str = "/scratch/drum_transcriptions_v2",
                      is_close_mic: bool = False) -> Optional[str]:
    """Find drum onset JSON for a stem's session.

    Each take in drum_transcriptions_v2 corresponds to one stem's transcription.
    The take index matches the alphabetical stem order in the session.

    Strategy:
    - Close mics (kick, snare, hh): find THIS stem's take (matched by index)
    - Overheads/room/bus: find the richest take (most instruments = full mix)
    """
    if not latent_path:
        return None
    lp = latent_path.replace("/mnt/data/system_home/arlo/", "/home/arlo/")
    if "Audio Files" in lp:
        session_rel = lp.split("Latents2/")[-1].split("Audio Files")[0]
        stem_name = os.path.basename(lp).replace(".vae.pt", "")
    elif "Bounced Files" in lp:
        session_rel = lp.split("Latents2/")[-1].split("Bounced Files")[0]
        stem_name = os.path.basename(lp).replace(".vae.pt", "")
    else:
        return None

    takes_dir = os.path.join(drum_root, session_rel, "Audio Files")
    if not os.path.exists(takes_dir):
        return None

    try:
        takes = sorted([d for d in os.listdir(takes_dir) if d.startswith("take_")])
    except Exception:
        return None

    if not takes:
        return None

    # Find this stem's take index by matching alphabetical order
    # (stems in Latents2 are indexed alphabetically, same as takes)
    lat_dir = os.path.join("/scratch/Latents2", session_rel, "Audio Files")
    stem_idx = None
    if os.path.exists(lat_dir):
        try:
            all_stems = sorted([f.replace(".vae.pt", "") for f in os.listdir(lat_dir) if f.endswith(".vae.pt")])
            if stem_name in all_stems:
                stem_idx = all_stems.index(stem_name)
        except Exception:
            pass

    if is_close_mic and stem_idx is not None:
        # Close mic: use this stem's own take
        take_name = f"take_{stem_idx}"
        onset_path = os.path.join(takes_dir, take_name, "session_onsets.json")
        if os.path.exists(onset_path):
            return onset_path

    # Overhead/room/bus or fallback: find richest take (most instruments)
    if takes_dir in _drum_richest_cache:
        return _drum_richest_cache[takes_dir]

    best_path, best_count = None, 0
    for take in takes:
        onset_path = os.path.join(takes_dir, take, "session_onsets.json")
        if os.path.exists(onset_path):
            try:
                import json as _json
                with open(onset_path) as f:
                    d = _json.load(f)
                n_inst = len(d.get("onsets", {}))
                n_hits = sum(len(v.get("times", [])) for v in d.get("onsets", {}).values())
                score = n_inst * 1000 + n_hits
                if score > best_count:
                    best_count = score
                    best_path = onset_path
            except Exception:
                if best_path is None:
                    best_path = onset_path
    _drum_richest_cache[takes_dir] = best_path
    return best_path


def sum_midi_representations(rolls: List[torch.Tensor]) -> torch.Tensor:
    """Sum multiple [144, T] MIDI representations — perfectly linear.

    Union of note events: any note active in any stem = active in sum.
    """
    if not rolls:
        return torch.zeros(TOTAL_MIDI_BINS, 1)
    T = min(r.shape[-1] for r in rolls)
    result = torch.zeros(TOTAL_MIDI_BINS, T)
    for r in rolls:
        result += r[:, :T]
    return (result > 0).float()


class MultiTrackMIDIEncoder(nn.Module):
    """Encode [144, T] multitrack MIDI to [embed_dim, T] for conditioning.

    Separate pathways for pitched (128 bins) and drums (16 bins),
    merged into a single embedding.
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.pitched_proj = nn.Sequential(
            nn.Conv1d(PITCHED_BINS, embed_dim, 1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, 1),
        )
        self.drum_proj = nn.Sequential(
            nn.Conv1d(DRUM_BINS, embed_dim, 1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, 1),
        )
        self.merge = nn.Sequential(
            nn.Conv1d(embed_dim * 2, embed_dim, 1),
            nn.GELU(),
        )
        self.embed_dim = embed_dim

    def forward(self, midi_roll: torch.Tensor) -> torch.Tensor:
        """midi_roll: [B, 144, T] → [B, embed_dim, T]"""
        pitched = self.pitched_proj(midi_roll[:, :PITCHED_BINS])
        drums = self.drum_proj(midi_roll[:, PITCHED_BINS:])
        return self.merge(torch.cat([pitched, drums], dim=1))
