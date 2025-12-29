#!/usr/bin/env python3
"""
Generate listening test samples from natural pitch-matched pairs.

For each pair:
- Source segment at pitch X
- Target segment at pitch X + shift (naturally occurring)

These are real recordings matched by timbre/dynamics similarity.
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/dø')

import torch
import torchaudio

from do.pipeline_do import DoTrainComponents


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def main():
    parser = argparse.ArgumentParser(description="Listen to natural pairs")
    parser.add_argument('--pairs_json', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_pairs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load pairs
    print(f"Loading pairs from: {args.pairs_json}")
    with open(args.pairs_json) as f:
        data = json.load(f)

    segments = data['segments']
    pairs = data['pairs']

    print(f"Total pairs: {len(pairs)}")
    print(f"Total segments: {len(segments)}")

    # Group by shift
    pairs_by_shift = {}
    for p in pairs:
        shift = p['shift']
        if shift not in pairs_by_shift:
            pairs_by_shift[shift] = []
        pairs_by_shift[shift].append(p)

    print(f"Shifts available: {list(pairs_by_shift.keys())}")

    # Select pairs: prioritize same-recording pairs, then high similarity
    selected_pairs = []

    # Get 5 pairs from +12 and 5 from -12
    for shift in [12, -12]:
        if shift not in pairs_by_shift:
            print(f"No pairs for shift {shift}")
            continue

        shift_pairs = pairs_by_shift[shift]
        # Sort by same_recording first, then by combined_score
        shift_pairs.sort(key=lambda p: (p['same_recording'], p['combined_score']), reverse=True)

        # Take top 5
        for p in shift_pairs[:5]:
            selected_pairs.append(p)

    print(f"\nSelected {len(selected_pairs)} pairs for listening test")

    # Load DCAE
    print("\nLoading DCAE...")
    checkpoint_dir = "/home/arlo/Data/ACE-Step/checkpoints"
    components = DoTrainComponents(checkpoint_dir=checkpoint_dir, device_id=0)
    components.load_dcae()
    dcae = components.music_dcae
    dcae.eval()

    # Cache for latents
    latent_cache = {}

    def load_latent(path):
        path = fix_path(path)
        if path in latent_cache:
            return latent_cache[path]

        data = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            lat = data.get('latents', data.get('latent'))
        else:
            lat = data

        if lat.dim() == 4:
            lat = lat.squeeze(0)

        latent_cache[path] = lat
        return lat

    # Process each pair
    for i, pair in enumerate(selected_pairs):
        shift = pair['shift']
        source_idx = pair['source_idx']
        target_idx = pair['target_idx']

        source_seg = segments[source_idx]
        target_seg = segments[target_idx]

        print(f"\n=== Pair {i+1}: shift {shift:+d} ===")
        print(f"  Source: pitch {pair['source_pitch']:.1f}, frames {source_seg['start_frame']}-{source_seg['end_frame']}")
        print(f"  Target: pitch {pair['target_pitch']:.1f}, frames {target_seg['start_frame']}-{target_seg['end_frame']}")
        print(f"  Same recording: {pair['same_recording']}")
        print(f"  Similarity: {pair['similarity']:.3f}")

        # Load latents
        source_latent = load_latent(source_seg['latent_path'])
        target_latent = load_latent(target_seg['latent_path'])

        # Extract segments
        source_chunk = source_latent[:, :, source_seg['start_frame']:source_seg['end_frame']]
        target_chunk = target_latent[:, :, target_seg['start_frame']:target_seg['end_frame']]

        # Match lengths (use shorter)
        min_len = min(source_chunk.shape[-1], target_chunk.shape[-1])
        source_chunk = source_chunk[:, :, :min_len]
        target_chunk = target_chunk[:, :, :min_len]

        print(f"  Segment length: {min_len} frames")

        # Decode
        with torch.no_grad():
            source_chunk = source_chunk.unsqueeze(0).to(args.device)
            target_chunk = target_chunk.unsqueeze(0).to(args.device)

            # decode returns (sr, [wavs])
            sr, source_wavs = dcae.decode(source_chunk, sr=48000)
            _, target_wavs = dcae.decode(target_chunk, sr=48000)

            source_audio = source_wavs[0]  # [2, samples] stereo
            target_audio = target_wavs[0]

        # Save
        source_path = os.path.join(args.output_dir, f"{i:02d}_source_pitch{pair['source_pitch']:.0f}_shift{shift:+d}.wav")
        target_path = os.path.join(args.output_dir, f"{i:02d}_target_pitch{pair['target_pitch']:.0f}_shift{shift:+d}.wav")

        torchaudio.save(source_path, source_audio.cpu(), sr)
        torchaudio.save(target_path, target_audio.cpu(), sr)

        print(f"  Saved: {Path(source_path).name}")
        print(f"  Saved: {Path(target_path).name}")

    print(f"\n\nListening test saved to: {args.output_dir}")
    print("\nNaming convention:")
    print("  XX_source_pitchNN_shift+/-MM.wav = Original segment at pitch NN")
    print("  XX_target_pitchNN_shift+/-MM.wav = Matched segment at pitch NN (= source + shift)")
    print("\nFor shift +12:")
    print("  source is at lower pitch, target is naturally one octave higher")
    print("  Compare: does target sound like source shifted up an octave? (should, they're matched by timbre)")
    print("\nFor shift -12:")
    print("  source is at higher pitch, target is naturally one octave lower")


if __name__ == "__main__":
    main()
