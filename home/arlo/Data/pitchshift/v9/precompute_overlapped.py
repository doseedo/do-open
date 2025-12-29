#!/usr/bin/env python3
"""
Precompute sox-shifted latents for v7 formant corrector.

Key: Shift HIGH samples to OVERLAP with LOW range (<55 MIDI).

Logic:
- If original > 79 MIDI: shift -24 (lands in 55-72 range, but we want <55)
- If original > 67 MIDI: shift -24 → lands at 43-55 (good overlap)
- If original > 55 MIDI: shift -12 → lands at 43-55 (good overlap)

Target: shifted samples land in 45-55 MIDI range to match natural LOW (<55).
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/dø')


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def sox_pitch_shift(waveform: torch.Tensor, sr: int, semitones: float) -> torch.Tensor:
    """Apply sox pitch shift."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_in:
        in_path = f_in.name
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_out:
        out_path = f_out.name

    try:
        torchaudio.save(in_path, waveform, sr)
        cents = int(semitones * 100)
        subprocess.run(['sox', in_path, out_path, 'pitch', str(cents)],
                       capture_output=True, check=True)
        shifted, _ = torchaudio.load(out_path)
        return shifted
    finally:
        if os.path.exists(in_path):
            os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def get_median_pitch(f0_path: str) -> float:
    f0 = np.load(f0_path)
    f0 = np.nan_to_num(f0, nan=0.0)
    f0_valid = f0[f0 > 20]
    if len(f0_valid) < 10:
        return None
    return 12 * np.log2(np.median(f0_valid) / 440) + 69


def compute_shift(midi: float, target_center: float = 50.0) -> int:
    """
    Compute shift to land near target_center MIDI.

    Returns shift in semitones (negative = down).
    Rounds to nearest octave (-12 or -24).
    """
    diff = midi - target_center

    # Round to nearest octave
    octaves = round(diff / 12)
    shift = -octaves * 12

    # Clamp to reasonable range
    shift = max(-24, min(-12, shift))

    return shift


def load_ensemble_blacklist(ensemble_results_path: str) -> set:
    """Load set of f0 paths flagged as ensemble/polyphonic."""
    if not os.path.exists(ensemble_results_path):
        print(f"No ensemble detection results at {ensemble_results_path}")
        return set()

    with open(ensemble_results_path) as f:
        data = json.load(f)

    flagged = data.get('flagged_recordings', [])
    # Use f0_path as key since that's what we can match on
    blacklist = set()
    for rec in flagged:
        f0_path = rec.get('f0_path', '')
        if f0_path:
            # Normalize path
            f0_path = fix_path(f0_path)
            blacklist.add(f0_path)

    print(f"Loaded {len(blacklist)} flagged ensemble recordings to filter")
    return blacklist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/msdd2/pitchshift_v7_overlapped')
    parser.add_argument('--low_threshold', type=float, default=55.0,
                        help='LOW register threshold (targets)')
    parser.add_argument('--target_center', type=float, default=50.0,
                        help='Target center MIDI for shifted samples')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--instrument', type=str, default='trumpet')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for GPU decode/encode')
    parser.add_argument('--sox_workers', type=int, default=8,
                        help='Number of parallel sox workers')
    parser.add_argument('--ensemble_results', type=str,
                        default='/home/arlo/Data/pitchshift/v3/ensemble_detection_results.json',
                        help='Path to ensemble detection results for filtering')
    parser.add_argument('--min_high_midi', type=float, default=60.0,
                        help='Minimum MIDI for HIGH register sources (default 60 = C4)')
    args = parser.parse_args()

    # Load ensemble blacklist
    ensemble_blacklist = load_ensemble_blacklist(args.ensemble_results)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load DCAE
    print("Loading DCAE...")
    from do.pipeline_do import DoTrainComponents
    components = DoTrainComponents(
        checkpoint_dir="/home/arlo/Data/ACE-Step/checkpoints",
        device_id=0
    )
    components.load_dcae()
    dcae = components.music_dcae

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    # Categorize entries by pitch
    high_entries = []  # Will be shifted
    low_entries = []   # Natural targets

    print(f"Processing manifest for {args.instrument}...")
    print(f"LOW threshold: < {args.low_threshold} MIDI")
    print(f"Target center after shift: {args.target_center} MIDI")

    skipped_ensemble = 0
    skipped_low_source = 0

    for entry in tqdm(manifest, desc="Scanning manifest"):
        if entry.get('sub_group') != args.instrument:
            continue

        # Skip muted entries - we want dry/open trumpet only
        if entry.get('is_muted', False):
            continue

        latent_path = fix_path(entry.get('latent_path', ''))
        cond = entry.get('conditioning_paths') or {}
        f0_path = fix_path(cond.get('f0', ''))

        if not latent_path or not f0_path:
            continue
        if not os.path.exists(latent_path) or not os.path.exists(f0_path):
            continue

        # Skip ensemble/polyphonic recordings
        if f0_path in ensemble_blacklist:
            skipped_ensemble += 1
            continue

        median_midi = get_median_pitch(f0_path)
        if median_midi is None:
            continue

        entry_data = {
            'latent_path': latent_path,
            'median_midi': median_midi,
        }

        if median_midi < args.low_threshold:
            low_entries.append(entry_data)
        elif median_midi >= args.min_high_midi:  # Use min_high_midi for HIGH sources
            # Anything >= low_threshold could be shifted down
            shift = compute_shift(median_midi, args.target_center)
            target_midi = median_midi + shift

            # Only include if shift lands in LOW range
            if target_midi < args.low_threshold:
                entry_data['shift'] = shift
                entry_data['target_midi'] = target_midi
                high_entries.append(entry_data)

    print(f"\nFound:")
    print(f"  HIGH entries (to be shifted): {len(high_entries)}")
    print(f"  LOW entries (natural targets): {len(low_entries)}")

    if args.max_samples:
        high_entries = high_entries[:args.max_samples]
        print(f"  Limited to {len(high_entries)} samples")

    # Show shift distribution
    shift_counts = {}
    for e in high_entries:
        s = e['shift']
        shift_counts[s] = shift_counts.get(s, 0) + 1
    print(f"\nShift distribution: {shift_counts}")

    # Show target range
    if high_entries:
        target_midis = [e['target_midi'] for e in high_entries]
        print(f"Target MIDI range: {min(target_midis):.1f} - {max(target_midis):.1f}")

    if low_entries:
        low_midis = [e['median_midi'] for e in low_entries]
        print(f"Natural LOW range: {min(low_midis):.1f} - {max(low_midis):.1f}")

    # Process HIGH entries with batching
    print(f"\nProcessing {len(high_entries)} entries...")

    output_manifest = {
        'low_threshold': args.low_threshold,
        'target_center': args.target_center,
        'shifted_entries': [],
        'low_entries': low_entries,
    }

    # Check which are already done
    to_process = []
    for i, entry in enumerate(high_entries):
        output_path = os.path.join(args.output_dir, f"shifted_{i:04d}.pt")
        entry['_output_path'] = output_path
        entry['_idx'] = i

        if os.path.exists(output_path):
            output_manifest['shifted_entries'].append({
                'latent_path': output_path,
                'original_midi': entry['median_midi'],
                'shift': entry['shift'],
                'target_midi': entry['target_midi'],
            })
        else:
            to_process.append(entry)

    print(f"Already done: {len(high_entries) - len(to_process)}, to process: {len(to_process)}")

    # Process in batches for GPU efficiency
    batch_size = args.batch_size

    def process_single_sox(args_tuple):
        """Sox shift in separate thread (CPU bound)."""
        audio, sr, shift, idx = args_tuple
        try:
            shifted = sox_pitch_shift(audio, sr, shift)
            return idx, shifted
        except Exception as e:
            return idx, None

    for batch_start in tqdm(range(0, len(to_process), batch_size), desc="Batches"):
        batch = to_process[batch_start:batch_start + batch_size]

        # Step 1: Load all latents
        latents = []
        valid_entries = []
        for entry in batch:
            try:
                data = torch.load(entry['latent_path'], map_location='cpu', weights_only=False)
                if isinstance(data, dict):
                    latent = data.get('latents', data.get('latent'))
                else:
                    latent = data
                if latent is None:
                    continue
                if latent.dim() == 3:
                    latent = latent.unsqueeze(0)
                latents.append(latent)
                valid_entries.append(entry)
            except Exception as e:
                print(f"Error loading {entry['latent_path']}: {e}")
                continue

        if not latents:
            continue

        # Step 2: Batch decode (GPU)
        audios = []
        with torch.no_grad():
            for latent in latents:
                latent = latent.to(args.device)
                decode_out = dcae.decode(latent)
                # Handle both tuple and direct returns
                if isinstance(decode_out, tuple):
                    sr, wavs = decode_out
                    audio = wavs[0].cpu()
                else:
                    sr = 44100
                    audio = decode_out[0].cpu() if decode_out.dim() == 3 else decode_out.cpu()
                audios.append(audio)

        # Step 3: Parallel sox (CPU, threaded)
        sox_args = [(audios[i], sr, valid_entries[i]['shift'], i) for i in range(len(audios))]
        shifted_audios = [None] * len(audios)

        with ThreadPoolExecutor(max_workers=args.sox_workers) as executor:
            futures = [executor.submit(process_single_sox, arg) for arg in sox_args]
            for future in as_completed(futures):
                idx, shifted = future.result()
                shifted_audios[idx] = shifted

        # Step 4: Batch encode (GPU)
        with torch.no_grad():
            for i, (entry, shifted_audio) in enumerate(zip(valid_entries, shifted_audios)):
                if shifted_audio is None:
                    continue

                try:
                    shifted_audio = shifted_audio.unsqueeze(0).to(args.device)
                    encode_out = dcae.encode(shifted_audio)
                    # Handle tuple return (latents, latent_lengths)
                    if isinstance(encode_out, tuple):
                        shifted_latent = encode_out[0].cpu()
                    else:
                        shifted_latent = encode_out.cpu()

                    # Save
                    output_path = entry['_output_path']
                    torch.save({
                        'latent': shifted_latent,
                        'original_path': entry['latent_path'],
                        'original_midi': entry['median_midi'],
                        'shift': entry['shift'],
                        'target_midi': entry['target_midi'],
                    }, output_path)

                    output_manifest['shifted_entries'].append({
                        'latent_path': output_path,
                        'original_midi': entry['median_midi'],
                        'shift': entry['shift'],
                        'target_midi': entry['target_midi'],
                    })
                except Exception as e:
                    print(f"Error encoding {entry['latent_path']}: {e}")

    # Save manifest
    manifest_path = os.path.join(args.output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(output_manifest, f, indent=2)

    print(f"\nSaved {len(output_manifest['shifted_entries'])} shifted latents")
    print(f"Manifest: {manifest_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Shifted entries: {len(output_manifest['shifted_entries'])}")
    print(f"Natural LOW targets: {len(output_manifest['low_entries'])}")

    if output_manifest['shifted_entries']:
        shifted_targets = [e['target_midi'] for e in output_manifest['shifted_entries']]
        low_midis = [e['median_midi'] for e in output_manifest['low_entries']]

        print(f"\nShifted range: {min(shifted_targets):.1f} - {max(shifted_targets):.1f} MIDI")
        print(f"Natural LOW range: {min(low_midis):.1f} - {max(low_midis):.1f} MIDI")

        # Check overlap
        overlap_min = max(min(shifted_targets), min(low_midis))
        overlap_max = min(max(shifted_targets), max(low_midis))
        if overlap_min < overlap_max:
            print(f"OVERLAP: {overlap_min:.1f} - {overlap_max:.1f} MIDI")
        else:
            print("WARNING: No overlap!")


if __name__ == "__main__":
    main()
