#!/usr/bin/env python3
"""
Generate Aligned Training Pairs Using Corruption Model

The key insight:
1. Corruption model learned DSP artifact signature
2. Apply corruption model to natural latents
3. Now have (corrupted, natural) pairs from SAME recording
4. Train correction model with frame-level supervision

This is analogous to denoising autoencoders:
- Learn the noise distribution first
- Use it to create paired training data
- Train denoiser on pairs
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm
import numpy as np

from models_corruption import CorruptionModel


def fix_path(path: str) -> str:
    """Fix mount paths."""
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


@torch.no_grad()
def generate_aligned_pairs(
    corruption_checkpoint: str,
    segments_json: str,
    output_dir: str,
    shifts: list = None,
    device: str = 'cuda',
):
    """
    Generate aligned (corrupted, clean) pairs using trained corruption model.

    For each natural segment:
    1. Load latent
    2. Apply corruption model at various shifts
    3. Save corrupted latent
    4. Record pair in manifest

    The result is perfectly aligned pairs for correction training.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    if shifts is None:
        shifts = [-12, -6, -3, 3, 6, 12]

    # Load corruption model
    print(f"Loading corruption model from: {corruption_checkpoint}")
    checkpoint = torch.load(corruption_checkpoint, map_location=device, weights_only=False)
    model = CorruptionModel()
    model.load_state_dict(checkpoint['generator_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded from epoch {checkpoint.get('epoch', '?')}")

    # Load segments
    print(f"Loading segments from: {segments_json}")
    with open(segments_json) as f:
        segments_data = json.load(f)

    segments = []
    for group_id, segs in segments_data.get('segments_by_group', {}).items():
        for seg in segs:
            segments.append({
                'latent_path': fix_path(seg['latent_path']),
                'start_frame': seg['start_frame'],
                'end_frame': seg['end_frame'],
                'pitch': seg['median_midi'],
                'group': int(group_id),
            })

    print(f"Processing {len(segments)} segments with shifts {shifts}")

    # Generate pairs
    pairs_manifest = []
    corrupted_dir = output_dir / 'corrupted_latents'
    corrupted_dir.mkdir(exist_ok=True)

    for seg_idx, seg in enumerate(tqdm(segments, desc="Generating pairs")):
        # Load natural latent
        if not os.path.exists(seg['latent_path']):
            continue

        try:
            data = torch.load(seg['latent_path'], map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                latent = data.get('latents', data.get('latent'))
            else:
                latent = data
            if latent.dim() == 4:
                latent = latent.squeeze(0)
        except:
            continue

        T = latent.shape[-1]
        seg_start = seg['start_frame']
        seg_end = min(seg['end_frame'], T)

        if seg_end - seg_start < 32:  # Min segment length
            continue

        # Extract segment
        segment_latent = latent[:, :, seg_start:seg_end]

        # Apply corruption at each shift
        segment_latent_gpu = segment_latent.unsqueeze(0).to(device)

        for shift in shifts:
            shift_tensor = torch.tensor([float(shift)], device=device)

            # Generate corrupted version
            corrupted_latent = model(segment_latent_gpu, shift_tensor)
            corrupted_latent = corrupted_latent.squeeze(0).cpu()

            # Save corrupted latent
            corrupted_filename = f"seg{seg_idx:05d}_shift{shift:+d}.pt"
            corrupted_path = corrupted_dir / corrupted_filename
            torch.save({
                'latent': corrupted_latent,
                'shift': shift,
            }, corrupted_path)

            # Record pair
            pairs_manifest.append({
                'corrupted_path': str(corrupted_path),
                'clean_path': seg['latent_path'],
                'clean_start': seg_start,
                'clean_end': seg_end,
                'shift': shift,
                'source_pitch': seg['pitch'],
                'target_pitch': seg['pitch'],  # Same! This is the key insight
                'segment_idx': seg_idx,
            })

    # Save manifest
    manifest_path = output_dir / 'aligned_pairs_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({
            'corruption_checkpoint': corruption_checkpoint,
            'shifts': shifts,
            'num_pairs': len(pairs_manifest),
            'pairs': pairs_manifest,
        }, f, indent=2)

    print(f"\nGenerated {len(pairs_manifest)} aligned pairs")
    print(f"Manifest saved to: {manifest_path}")
    print(f"\nNext: Train correction model with frame-level supervision")
    print(f"  python train_correction_aligned.py --pairs_manifest {manifest_path} ...")


def main():
    parser = argparse.ArgumentParser(description="Generate Aligned Pairs")
    parser.add_argument('--corruption_checkpoint', type=str, required=True,
                        help='Path to trained corruption model')
    parser.add_argument('--segments', type=str, required=True,
                        help='Path to segments JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for corrupted latents and manifest')
    parser.add_argument('--shifts', type=int, nargs='+', default=[-12, -6, -3, 3, 6, 12],
                        help='Shift values to generate')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    generate_aligned_pairs(
        corruption_checkpoint=args.corruption_checkpoint,
        segments_json=args.segments,
        output_dir=args.output_dir,
        shifts=args.shifts,
        device=args.device,
    )


if __name__ == "__main__":
    main()
