#!/usr/bin/env python3
"""
v9 Preprocessing: Generate Paired Latents from Full Audio Files

This processes full audio files to create paired training data:
1. Scans each file for non-silent windows
2. Creates formant-shifted versions using pyworld (proper formant shift)
3. Encodes natural/shifted pairs to latent
4. Saves to manifest for fast training

Uses pyworld for TRUE formant shifting:
- Same pitch, same duration, only formants shifted
"""

import os
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torchaudio
import pyworld as pw
import soundfile as sf
from tqdm import tqdm


# Pitch group ranges (MIDI)
GROUP_RANGES = {
    1: (53, 65),   # F3-F4 (LOW)
    2: (65, 77),   # F4-F5 (MID)
    3: (77, 89),   # F5-F6 (HIGH)
}

MIN_RMS = 0.01
HOP_SIZE = 512
SAMPLE_RATE = 44100


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def get_pitch_group(midi: float) -> Optional[int]:
    """Get group for a MIDI note."""
    for group, (low, high) in GROUP_RANGES.items():
        if low <= midi < high:
            return group
    if midi >= 77 and midi <= 89:
        return 3
    return None


def check_rms(audio: torch.Tensor, min_rms: float = MIN_RMS) -> bool:
    """Check if audio has sufficient RMS."""
    rms = audio.pow(2).mean().sqrt().item()
    return rms >= min_rms


def get_audio_hash(path: str) -> str:
    """Get a short hash for audio path."""
    return hashlib.md5(path.encode()).hexdigest()[:12]


def shift_formants_pyworld(audio: np.ndarray, sr: int, shift_semitones: int) -> np.ndarray:
    """
    Shift formants using pyworld while keeping pitch and duration the same.

    Args:
        audio: mono audio as float64 numpy array
        sr: sample rate
        shift_semitones: positive = formants up, negative = formants down

    Returns:
        Formant-shifted audio (same length, same pitch)
    """
    # Ensure float64 for pyworld
    audio = audio.astype(np.float64)

    # Extract with World vocoder
    f0, t = pw.harvest(audio, sr)  # pitch
    sp = pw.cheaptrick(audio, f0, t, sr)  # spectral envelope (formants)
    ap = pw.d4c(audio, f0, t, sr)  # aperiodicity

    # Shift formants by resampling spectral envelope
    shift_ratio = 2 ** (shift_semitones / 12)
    n_frames, n_bins = sp.shape
    sp_shifted = np.zeros_like(sp)

    for i in range(n_frames):
        old_indices = np.arange(n_bins)
        new_indices = old_indices / shift_ratio

        # Interpolate
        valid = new_indices < n_bins
        sp_shifted[i, valid] = np.interp(new_indices[valid], old_indices, sp[i])

        # Fill remaining frequencies
        if shift_ratio < 1:  # shifting down, need to fill high freqs
            fill_start = int(n_bins * shift_ratio)
            if fill_start < n_bins:
                sp_shifted[i, fill_start:] = sp[i, -1]
        else:  # shifting up, low freqs get compressed
            sp_shifted[i, ~valid] = sp[i, 0]

    # Resynthesize with SAME pitch but shifted formants
    audio_shifted = pw.synthesize(f0, sp_shifted, ap, sr)

    return audio_shifted.astype(np.float32)


def create_shifted_audio(audio_path: str, shift: int, cache_dir: Path) -> Optional[str]:
    """Create formant-shifted version of audio using pyworld."""
    audio_hash = get_audio_hash(audio_path)
    shift_name = f"shift{shift:+d}"
    out_path = cache_dir / f"{audio_hash}_{shift_name}.wav"

    if out_path.exists():
        return str(out_path)

    try:
        # Load audio
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Apply formant shift with pyworld
        audio_shifted = shift_formants_pyworld(audio, sr, shift)

        # Save
        sf.write(str(out_path), audio_shifted, sr)
        return str(out_path) if out_path.exists() else None
    except Exception as e:
        print(f"Formant shift failed for {audio_path}: {e}")
        return None


def load_dcae(device: str = 'cuda'):
    """Load the DCAE encoder."""
    import sys
    sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/dø')
    from do.pipeline_do import DoTrainComponents
    components = DoTrainComponents(
        checkpoint_dir="/home/arlo/Data/ACE-Step/checkpoints",
        device_id=int(device.split(':')[-1]) if ':' in device else 0
    )
    components.load_dcae()
    return components.music_dcae


@torch.no_grad()
def encode_audio(dcae, audio: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
    """Encode audio to latent."""
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)

    # Ensure stereo
    if audio.shape[1] == 1:
        audio = audio.repeat(1, 2, 1)

    audio = audio.to(device)
    result = dcae.encode(audio)
    latent = result[0] if isinstance(result, tuple) else result
    return latent.cpu()


def get_pitch_info_for_window(
    f0_data: np.ndarray,
    start_sample: int,
    end_sample: int,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Get pitch group, direction, and shift for a window."""
    # Convert samples to f0 frames
    start_frame = start_sample // HOP_SIZE
    end_frame = end_sample // HOP_SIZE

    # Get f0 for this window
    if end_frame > len(f0_data):
        end_frame = len(f0_data)
    if start_frame >= end_frame:
        return None, None, None

    f0_window = f0_data[start_frame:end_frame]
    valid = f0_window > 20

    if not valid.any():
        return None, None, None

    # Get median pitch
    midi_values = 69 + 12 * np.log2(f0_window[valid] / 440.0)
    median_midi = np.median(midi_values)

    group = get_pitch_group(median_midi)
    if group is None:
        return None, None, None

    # Direction based on group
    if group == 1:
        direction = 1  # UP only
        shift = 12
    elif group == 3:
        direction = 0  # DOWN only
        shift = -12
    else:  # Group 2
        direction = random.choice([0, 1])
        shift = 12 if direction == 1 else -12

    return group, direction, shift


def process_entry(
    entry: Dict,
    dcae,
    output_dir: Path,
    cache_dir: Path,
    window_seconds: float = 1.0,
    hop_seconds: float = 0.5,
    device: str = 'cuda',
) -> List[Dict]:
    """Process one audio file and extract all valid windows."""
    audio_path = entry['audio_path']
    f0_path = entry['f0_path']

    results = []

    try:
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)

        # Load f0
        f0_data = np.load(f0_path)
        f0_data = np.nan_to_num(f0_data, nan=0.0)

        window_samples = int(window_seconds * SAMPLE_RATE)
        hop_samples = int(hop_seconds * SAMPLE_RATE)
        total_samples = audio.shape[-1]

        if total_samples < window_samples:
            return results

        # Scan all windows
        audio_hash = get_audio_hash(audio_path)

        for start in range(0, total_samples - window_samples + 1, hop_samples):
            end = start + window_samples
            window = audio[:, start:end]

            # RMS check
            if not check_rms(window):
                continue

            # Pitch info
            result = get_pitch_info_for_window(f0_data, start, end)
            if result[0] is None:
                continue

            group, direction, shift = result

            # Get/create shifted audio
            shifted_path = create_shifted_audio(audio_path, shift, cache_dir)
            if shifted_path is None:
                continue

            # Load shifted window
            try:
                shifted_audio, _ = torchaudio.load(shifted_path)
                if shifted_audio.shape[0] > 1:
                    shifted_audio = shifted_audio.mean(0, keepdim=True)

                if shifted_audio.shape[-1] < end:
                    continue

                shifted_window = shifted_audio[:, start:end]
            except Exception:
                continue

            # Make stereo
            natural_stereo = window.repeat(2, 1)
            shifted_stereo = shifted_window.repeat(2, 1)

            # Encode both
            natural_latent = encode_audio(dcae, natural_stereo, device)
            shifted_latent = encode_audio(dcae, shifted_stereo, device)

            # Match lengths (should be same but just in case)
            min_t = min(natural_latent.shape[-1], shifted_latent.shape[-1])
            natural_latent = natural_latent[..., :min_t]
            shifted_latent = shifted_latent[..., :min_t]

            # Save latent pair
            window_id = f"{audio_hash}_{start}_{shift:+d}"
            pair_path = output_dir / f"{window_id}.pt"

            torch.save({
                'natural': natural_latent.squeeze(0),  # [C, H, T]
                'shifted': shifted_latent.squeeze(0),
                'group': group,
                'direction': direction,
                'shift': shift,
                'source_audio': audio_path,
                'start_sample': start,
            }, pair_path)

            results.append({
                'pair_path': str(pair_path),
                'group': group,
                'direction': direction,
                'shift': shift,
                'source_audio': audio_path,
            })

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/msdd2/pitchshift_v9_pyworld_paired')
    parser.add_argument('--cache_dir', type=str,
                        default='/mnt/msdd2/pitchshift_v9_pyworld_cache')
    parser.add_argument('--window_seconds', type=float, default=1.0)
    parser.add_argument('--hop_seconds', type=float, default=0.5)
    parser.add_argument('--instrument', type=str, default='trumpet')
    parser.add_argument('--max_entries', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    with open(args.manifest) as f:
        manifest = json.load(f)

    # Filter entries
    entries = []
    for e in manifest:
        if e.get('sub_group') != args.instrument:
            continue
        if e.get('is_muted', False):
            continue

        audio_path = fix_path(e.get('audio_path', ''))
        f0_path = fix_path(e.get('conditioning_paths', {}).get('f0', ''))

        if not all([audio_path, f0_path]):
            continue
        if not all(os.path.exists(p) for p in [audio_path, f0_path]):
            continue

        entries.append({
            'audio_path': audio_path,
            'f0_path': f0_path,
        })

    if args.max_entries:
        entries = entries[:args.max_entries]

    print(f"Found {len(entries)} valid entries")

    # Load DCAE
    print("Loading DCAE...")
    dcae = load_dcae(args.device)

    # Process all entries
    print(f"\nProcessing {len(entries)} audio files...")
    all_pairs = []
    group_counts = defaultdict(int)
    direction_counts = defaultdict(int)

    for entry in tqdm(entries):
        pairs = process_entry(
            entry, dcae, output_dir, cache_dir,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            device=args.device,
        )
        all_pairs.extend(pairs)

        for p in pairs:
            group_counts[p['group']] += 1
            direction_counts[p['direction']] += 1

    # Save manifest
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(all_pairs, f, indent=2)

    print(f"\n" + "="*60)
    print(f"PREPROCESSING COMPLETE")
    print(f"="*60)
    print(f"Total pairs: {len(all_pairs)}")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"\nGroup distribution:")
    for g in sorted(group_counts.keys()):
        print(f"  Group {g}: {group_counts[g]}")
    print(f"\nDirection distribution:")
    print(f"  Down (0): {direction_counts[0]}")
    print(f"  Up (1): {direction_counts[1]}")


if __name__ == "__main__":
    main()
