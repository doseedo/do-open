#!/usr/bin/env python3
"""
Fast Natural Pair Finding using vectorized operations and sampling.

Key optimizations:
1. Precompute feature vectors for all segments
2. Use KD-tree for fast nearest-neighbor search
3. Sample from large pitch bins instead of exhaustive comparison
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def get_recording_id(path: str) -> str:
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


def load_all_segments(segments_json: str):
    """Load all segments with features."""
    print(f"Loading segments from: {segments_json}")
    with open(segments_json) as f:
        data = json.load(f)

    segments = []
    latent_cache = {}

    for group_id, segs in tqdm(data.get('segments_by_group', {}).items(), desc="Groups"):
        for seg in segs:
            latent_path = fix_path(seg['latent_path'])

            # Load latent
            if latent_path not in latent_cache:
                if not os.path.exists(latent_path):
                    continue
                try:
                    ld = torch.load(latent_path, map_location='cpu', weights_only=False)
                    if isinstance(ld, dict):
                        lat = ld.get('latents', ld.get('latent'))
                    else:
                        lat = ld
                    if lat is None:
                        continue
                    if lat.dim() == 4:
                        lat = lat.squeeze(0)
                    latent_cache[latent_path] = lat.numpy()
                except:
                    continue

            latent = latent_cache.get(latent_path)
            if latent is None:
                continue

            start = seg['start_frame']
            end = min(seg['end_frame'], latent.shape[-1])
            if end - start < 32:
                continue

            segment_latent = latent[:, :, start:end]  # [C, H, T]

            # Compute PITCH-INVARIANT feature vector for matching
            # We want: RMS energy, temporal shape (attack/decay), duration
            # NOT: channel means, C3 energy (these differ by register)

            # 1. Total RMS energy (pitch-invariant loudness)
            total_energy = np.sqrt((segment_latent ** 2).mean())

            # 2. Duration (in frames)
            duration = end - start

            # 3. Temporal envelope shape - how energy evolves over time
            # Compute energy per frame, then normalize to get shape
            frame_energy = (segment_latent ** 2).mean(axis=(0, 1))  # [T]
            frame_energy = frame_energy / (frame_energy.max() + 1e-8)  # normalize to [0,1]

            # Summarize temporal shape with a few statistics:
            # - Attack: energy in first 25% vs mean
            # - Sustain: energy variance in middle 50%
            # - Decay: energy in last 25% vs mean
            T = len(frame_energy)
            q1, q2, q3 = T // 4, T // 2, 3 * T // 4
            attack_ratio = frame_energy[:max(1, q1)].mean() / (frame_energy.mean() + 1e-8)
            sustain_var = frame_energy[q1:q3].std() if q3 > q1 else 0.0
            decay_ratio = frame_energy[q3:].mean() / (frame_energy.mean() + 1e-8) if q3 < T else 1.0

            # 4. Spectral centroid proxy - where is energy concentrated in H dim
            # This is somewhat pitch-invariant (formant structure)
            h_energy = (segment_latent ** 2).mean(axis=(0, 2))  # [H=16]
            h_weights = np.arange(len(h_energy))
            spectral_centroid = (h_energy * h_weights).sum() / (h_energy.sum() + 1e-8)
            spectral_spread = np.sqrt(((h_weights - spectral_centroid) ** 2 * h_energy).sum() / (h_energy.sum() + 1e-8))

            feature_vec = np.array([
                np.log1p(total_energy),  # log scale for energy
                np.log1p(duration),      # log scale for duration
                attack_ratio,
                sustain_var,
                decay_ratio,
                spectral_centroid / 16.0,  # normalize to [0,1]
                spectral_spread / 8.0,     # normalize
            ])

            segments.append({
                'idx': len(segments),
                'latent_path': latent_path,
                'start_frame': start,
                'end_frame': end,
                'length': end - start,
                'median_midi': seg['median_midi'],
                'group': int(group_id),
                'recording_id': get_recording_id(latent_path),
                'features': feature_vec,
            })

    print(f"Loaded {len(segments)} valid segments")
    return segments


def find_pairs_fast(segments, target_shifts, pitch_tolerance=1.0, max_pairs_per_shift=1000, k_neighbors=10):
    """Find pairs using nearest neighbor search."""

    # Build feature matrix and index
    features = np.array([s['features'] for s in segments])
    pitches = np.array([s['median_midi'] for s in segments])

    # Normalize features for distance calculation
    features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    # Build NN index
    nn = NearestNeighbors(n_neighbors=min(k_neighbors * 2, len(segments)), algorithm='ball_tree')
    nn.fit(features_norm)

    all_pairs = []

    for shift in target_shifts:
        print(f"\nFinding pairs for shift {shift:+d}...")
        shift_pairs = []

        for i, seg in enumerate(tqdm(segments, desc=f"Shift {shift:+d}")):
            source_pitch = seg['median_midi']
            target_pitch = source_pitch + shift

            # Find segments at target pitch
            pitch_mask = np.abs(pitches - target_pitch) <= pitch_tolerance
            target_indices = np.where(pitch_mask)[0]

            if len(target_indices) == 0:
                continue

            # Get nearest neighbors in feature space from target pitch candidates
            query = features_norm[i:i+1]
            distances, neighbor_indices = nn.kneighbors(query)

            # Filter to only keep neighbors at target pitch
            valid_neighbors = []
            for ni, dist in zip(neighbor_indices[0], distances[0]):
                if ni in target_indices and ni != i:
                    valid_neighbors.append((ni, dist))
                if len(valid_neighbors) >= k_neighbors:
                    break

            for target_idx, dist in valid_neighbors:
                target_seg = segments[target_idx]

                # Compute similarity score (inverse of distance)
                similarity = 1.0 / (1.0 + dist)

                # Compute overlap
                overlap = min(seg['length'], target_seg['length'])

                same_recording = seg['recording_id'] == target_seg['recording_id']

                # Combined score
                score = similarity * (1.2 if same_recording else 1.0)

                shift_pairs.append({
                    'source_idx': seg['idx'],
                    'target_idx': target_seg['idx'],
                    'source_pitch': source_pitch,
                    'target_pitch': target_seg['median_midi'],
                    'shift': shift,
                    'similarity': float(similarity),
                    'overlap_frames': overlap,
                    'same_recording': same_recording,
                    'combined_score': float(score),
                })

        # Sort and take top pairs
        shift_pairs.sort(key=lambda p: p['combined_score'], reverse=True)

        # Deduplicate - keep best match per source
        seen_sources = set()
        unique_pairs = []
        for p in shift_pairs:
            if p['source_idx'] not in seen_sources:
                seen_sources.add(p['source_idx'])
                unique_pairs.append(p)
                if len(unique_pairs) >= max_pairs_per_shift:
                    break

        print(f"  Found {len(unique_pairs)} pairs for shift {shift:+d}")
        all_pairs.extend(unique_pairs)

    return all_pairs


def main():
    parser = argparse.ArgumentParser(description="Fast Natural Pair Finding")
    parser.add_argument('--segments', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--shifts', type=int, nargs='+', default=[-12, -6, 6, 12])
    parser.add_argument('--pitch_tolerance', type=float, default=1.0)
    parser.add_argument('--max_pairs_per_shift', type=int, default=500)
    parser.add_argument('--k_neighbors', type=int, default=5)

    args = parser.parse_args()

    # Make output dir
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load segments
    segments = load_all_segments(args.segments)

    # Find pairs
    pairs = find_pairs_fast(
        segments,
        target_shifts=args.shifts,
        pitch_tolerance=args.pitch_tolerance,
        max_pairs_per_shift=args.max_pairs_per_shift,
        k_neighbors=args.k_neighbors,
    )

    print(f"\nTotal pairs found: {len(pairs)}")

    # Stats
    for shift in args.shifts:
        count = sum(1 for p in pairs if p['shift'] == shift)
        same_rec = sum(1 for p in pairs if p['shift'] == shift and p['same_recording'])
        print(f"  Shift {shift:+d}: {count} pairs ({same_rec} same-recording)")

    # Save
    output_data = {
        'config': vars(args),
        'segments': [
            {k: v for k, v in s.items() if k != 'features'}
            for s in segments
        ],
        'pairs': pairs,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to: {args.output}")

    # Top examples
    print("\n=== Top pairs ===")
    for p in pairs[:10]:
        print(f"  Shift {p['shift']:+d}: pitch {p['source_pitch']:.1f}->{p['target_pitch']:.1f}, "
              f"sim={p['similarity']:.3f}, overlap={p['overlap_frames']}, same_rec={p['same_recording']}")


if __name__ == "__main__":
    main()
