#!/usr/bin/env python3
"""
Precompute sox-shifted latents for v7 formant corrector.

Takes HIGH register samples, sox-shifts them DOWN, re-encodes.
Creates corrupted LOW-pitch latents with HIGH formants.
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path

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
        os.unlink(in_path)
        os.unlink(out_path)


def get_median_pitch(f0_path: str) -> float:
    f0 = np.load(f0_path)
    f0 = np.nan_to_num(f0, nan=0.0)
    f0_valid = f0[f0 > 20]
    if len(f0_valid) < 10:
        return None
    return 12 * np.log2(np.median(f0_valid) / 440) + 69


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/msdd2/pitchshift_shifted_latents_v7')
    parser.add_argument('--high_threshold', type=float, default=70.0)
    parser.add_argument('--shift', type=float, default=-12.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

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

    # Get HIGH register entries
    high_entries = []
    for entry in manifest:
        if entry.get('sub_group') != 'trumpet':
            continue

        latent_path = fix_path(entry.get('latent_path', ''))
        cond = entry.get('conditioning_paths') or {}
        f0_path = fix_path(cond.get('f0', ''))

        if not latent_path or not f0_path:
            continue
        if not os.path.exists(latent_path) or not os.path.exists(f0_path):
            continue

        median_midi = get_median_pitch(f0_path)
        if median_midi is None:
            continue

        if median_midi > args.high_threshold:
            high_entries.append({
                'latent_path': latent_path,
                'median_midi': median_midi,
            })

    print(f"Found {len(high_entries)} HIGH register samples")

    if args.max_samples:
        high_entries = high_entries[:args.max_samples]
        print(f"Limited to {len(high_entries)} samples")

    # Process each
    print(f"Shifting by {args.shift} semitones...")
    for i, entry in enumerate(tqdm(high_entries)):
        output_path = os.path.join(args.output_dir, f"shifted_{i:04d}.pt")

        if os.path.exists(output_path):
            continue

        try:
            # Load latent
            data = torch.load(entry['latent_path'], map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                latent = data.get('latents', data.get('latent'))
            else:
                latent = data
            if latent.dim() == 3:
                latent = latent.unsqueeze(0)

            # Decode to audio
            with torch.no_grad():
                latent = latent.to(args.device)
                sr, wavs = dcae.decode(latent)
                audio = wavs[0].cpu()

            # Sox shift
            shifted_audio = sox_pitch_shift(audio, sr, args.shift)

            # Re-encode
            with torch.no_grad():
                shifted_audio = shifted_audio.unsqueeze(0).to(args.device)
                shifted_latent = dcae.encode(shifted_audio)
                shifted_latent = shifted_latent.cpu()

            # Save
            torch.save({
                'latent': shifted_latent,
                'original_path': entry['latent_path'],
                'original_midi': entry['median_midi'],
                'shift': args.shift,
            }, output_path)

        except Exception as e:
            print(f"Error processing {entry['latent_path']}: {e}")
            continue

    print(f"Saved shifted latents to {args.output_dir}")
    print(f"Total: {len(list(Path(args.output_dir).glob('*.pt')))} files")


if __name__ == "__main__":
    main()
