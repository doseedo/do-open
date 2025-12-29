#!/usr/bin/env python3
"""
Find Naturally Occurring Pitch-Matched Pairs

Instead of synthetic corruption, find REAL pairs:
- Segment A at pitch X
- Segment B at pitch X + shift (e.g., +12 semitones)
- Matched in timbre (same/similar instrument, latent statistics)
- Matched in dynamics (similar amplitude profile)

These are natural "before/after" pairs for pitch shift correction training.
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.distance import cosine
from scipy.signal import correlate


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


@dataclass
class Segment:
    """Segment with all metadata for matching."""
    idx: int
    latent_path: str
    start_frame: int
    end_frame: int
    median_midi: float
    group: int
    recording_id: str  # Derived from path

    # Computed features for matching
    latent_mean: Optional[np.ndarray] = None  # [8] per-channel mean
    latent_energy: Optional[np.ndarray] = None  # [8] per-channel energy
    amp_profile: Optional[np.ndarray] = None  # Amplitude envelope
    c3_profile: Optional[np.ndarray] = None  # C3 over time (register signature)


def get_recording_id(path: str) -> str:
    """Extract recording identifier from path."""
    # Use parent folder + filename without extension
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


def load_segment_features(seg: Segment, latent_cache: dict) -> bool:
    """Load and compute features for a segment."""
    path = fix_path(seg.latent_path)

    if path in latent_cache:
        latent = latent_cache[path]
    elif os.path.exists(path):
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                latent = data.get('latents', data.get('latent'))
            else:
                latent = data
            if latent is None:
                return False
            if latent.dim() == 4:
                latent = latent.squeeze(0)
            latent_cache[path] = latent
        except:
            return False
    else:
        return False

    # Extract segment
    start = seg.start_frame
    end = min(seg.end_frame, latent.shape[-1])
    if end - start < 16:
        return False

    segment_latent = latent[:, :, start:end].numpy()  # [C, H, T]

    # Per-channel statistics
    seg.latent_mean = segment_latent.mean(axis=(1, 2))  # [C]
    seg.latent_energy = (segment_latent ** 2).mean(axis=(1, 2))  # [C]

    # C3 profile (register channel over time)
    seg.c3_profile = segment_latent[3].mean(axis=0)  # [T]

    # Amplitude proxy (total energy over time)
    seg.amp_profile = (segment_latent ** 2).mean(axis=(0, 1))  # [T]

    return True


def compute_timbre_similarity(seg_a: Segment, seg_b: Segment) -> float:
    """Compute timbre similarity based on latent statistics."""
    if seg_a.latent_mean is None or seg_b.latent_mean is None:
        return 0.0

    # Compare per-channel energy distribution (normalized)
    energy_a = seg_a.latent_energy / (seg_a.latent_energy.sum() + 1e-8)
    energy_b = seg_b.latent_energy / (seg_b.latent_energy.sum() + 1e-8)

    # 1 - cosine distance = cosine similarity
    try:
        energy_sim = 1 - cosine(energy_a, energy_b)
    except:
        energy_sim = 0.0

    # Compare means (centered around 0)
    mean_diff = np.abs(seg_a.latent_mean - seg_b.latent_mean).mean()
    mean_sim = np.exp(-mean_diff)  # Exponential decay

    return 0.7 * energy_sim + 0.3 * mean_sim


def compute_dynamics_similarity(seg_a: Segment, seg_b: Segment) -> Tuple[float, int]:
    """
    Compute dynamics similarity and find best alignment offset.

    Returns:
        similarity: 0-1 score
        offset: how many frames to shift seg_b to align with seg_a
    """
    if seg_a.amp_profile is None or seg_b.amp_profile is None:
        return 0.0, 0

    amp_a = seg_a.amp_profile
    amp_b = seg_b.amp_profile

    # Normalize
    amp_a = (amp_a - amp_a.mean()) / (amp_a.std() + 1e-8)
    amp_b = (amp_b - amp_b.mean()) / (amp_b.std() + 1e-8)

    # Cross-correlation to find best alignment
    corr = correlate(amp_a, amp_b, mode='full')
    best_offset = corr.argmax() - len(amp_b) + 1

    # Compute similarity at best offset
    min_len = min(len(amp_a), len(amp_b))
    if min_len < 8:
        return 0.0, 0

    # Align and compute correlation
    if best_offset >= 0:
        a_aligned = amp_a[best_offset:best_offset + min_len]
        b_aligned = amp_b[:min_len]
    else:
        a_aligned = amp_a[:min_len]
        b_aligned = amp_b[-best_offset:-best_offset + min_len]

    actual_len = min(len(a_aligned), len(b_aligned))
    if actual_len < 8:
        return 0.0, 0

    a_aligned = a_aligned[:actual_len]
    b_aligned = b_aligned[:actual_len]

    # Pearson correlation
    correlation = np.corrcoef(a_aligned, b_aligned)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0

    return max(0, correlation), best_offset


@dataclass
class MatchedPair:
    """A matched pair of segments."""
    source_idx: int
    target_idx: int
    source_pitch: float
    target_pitch: float
    shift: int  # target_pitch - source_pitch rounded
    timbre_sim: float
    dynamics_sim: float
    alignment_offset: int
    overlap_frames: int
    same_recording: bool
    combined_score: float


def find_pairs(
    segments: List[Segment],
    target_shifts: List[int],
    pitch_tolerance: float = 1.0,
    min_timbre_sim: float = 0.7,
    min_dynamics_sim: float = 0.5,
    min_overlap: int = 32,
) -> List[MatchedPair]:
    """Find all matching pairs."""

    # Group segments by approximate pitch
    pitch_groups = defaultdict(list)
    for seg in segments:
        pitch_bin = round(seg.median_midi)
        pitch_groups[pitch_bin].append(seg)

    print(f"Pitch groups: {len(pitch_groups)}")
    print(f"Pitch range: {min(pitch_groups.keys())} - {max(pitch_groups.keys())}")

    pairs = []

    for shift in target_shifts:
        print(f"\nFinding pairs for shift {shift:+d}...")
        shift_pairs = []

        for source_pitch in tqdm(sorted(pitch_groups.keys())):
            target_pitch = source_pitch + shift

            # Find segments at target pitch (with tolerance)
            target_candidates = []
            for p in range(int(target_pitch - pitch_tolerance), int(target_pitch + pitch_tolerance) + 1):
                target_candidates.extend(pitch_groups.get(p, []))

            if not target_candidates:
                continue

            source_segments = pitch_groups[source_pitch]

            for source_seg in source_segments:
                for target_seg in target_candidates:
                    # Skip same segment
                    if source_seg.idx == target_seg.idx:
                        continue

                    # Check timbre similarity
                    timbre_sim = compute_timbre_similarity(source_seg, target_seg)
                    if timbre_sim < min_timbre_sim:
                        continue

                    # Check dynamics similarity and get alignment
                    dynamics_sim, offset = compute_dynamics_similarity(source_seg, target_seg)
                    if dynamics_sim < min_dynamics_sim:
                        continue

                    # Compute overlap
                    source_len = source_seg.end_frame - source_seg.start_frame
                    target_len = target_seg.end_frame - target_seg.start_frame
                    overlap = min(source_len, target_len) - abs(offset)

                    if overlap < min_overlap:
                        continue

                    same_recording = source_seg.recording_id == target_seg.recording_id

                    # Combined score (weight same-recording pairs higher)
                    combined_score = (
                        0.4 * timbre_sim +
                        0.4 * dynamics_sim +
                        0.1 * min(1.0, overlap / 100) +
                        0.1 * (1.0 if same_recording else 0.0)
                    )

                    pair = MatchedPair(
                        source_idx=source_seg.idx,
                        target_idx=target_seg.idx,
                        source_pitch=source_seg.median_midi,
                        target_pitch=target_seg.median_midi,
                        shift=shift,
                        timbre_sim=timbre_sim,
                        dynamics_sim=dynamics_sim,
                        alignment_offset=offset,
                        overlap_frames=overlap,
                        same_recording=same_recording,
                        combined_score=combined_score,
                    )
                    shift_pairs.append(pair)

        # Sort by score and deduplicate
        shift_pairs.sort(key=lambda p: p.combined_score, reverse=True)

        # Remove duplicate source segments (keep best match)
        seen_sources = set()
        unique_pairs = []
        for pair in shift_pairs:
            if pair.source_idx not in seen_sources:
                seen_sources.add(pair.source_idx)
                unique_pairs.append(pair)

        print(f"  Found {len(unique_pairs)} unique pairs for shift {shift:+d}")
        pairs.extend(unique_pairs)

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Find Natural Pitch-Matched Pairs")
    parser.add_argument('--segments', type=str, required=True,
                        help='Path to segments JSON')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON path')
    parser.add_argument('--shifts', type=int, nargs='+', default=[-12, -6, 6, 12],
                        help='Target shifts to find pairs for')
    parser.add_argument('--pitch_tolerance', type=float, default=1.0,
                        help='Pitch tolerance in semitones')
    parser.add_argument('--min_timbre_sim', type=float, default=0.6,
                        help='Minimum timbre similarity')
    parser.add_argument('--min_dynamics_sim', type=float, default=0.4,
                        help='Minimum dynamics similarity')
    parser.add_argument('--min_overlap', type=int, default=32,
                        help='Minimum overlap in frames')

    args = parser.parse_args()

    # Load segments
    print(f"Loading segments from: {args.segments}")
    with open(args.segments) as f:
        data = json.load(f)

    segments = []
    idx = 0
    for group_id, segs in data.get('segments_by_group', {}).items():
        for seg in segs:
            s = Segment(
                idx=idx,
                latent_path=seg['latent_path'],
                start_frame=seg['start_frame'],
                end_frame=seg['end_frame'],
                median_midi=seg['median_midi'],
                group=int(group_id),
                recording_id=get_recording_id(seg['latent_path']),
            )
            segments.append(s)
            idx += 1

    print(f"Loaded {len(segments)} segments")

    # Load features for all segments
    print("\nLoading segment features...")
    latent_cache = {}
    valid_segments = []
    for seg in tqdm(segments):
        if load_segment_features(seg, latent_cache):
            valid_segments.append(seg)

    print(f"Valid segments with features: {len(valid_segments)}")

    # Find pairs
    pairs = find_pairs(
        valid_segments,
        target_shifts=args.shifts,
        pitch_tolerance=args.pitch_tolerance,
        min_timbre_sim=args.min_timbre_sim,
        min_dynamics_sim=args.min_dynamics_sim,
        min_overlap=args.min_overlap,
    )

    print(f"\nTotal pairs found: {len(pairs)}")

    # Statistics
    same_rec_count = sum(1 for p in pairs if p.same_recording)
    print(f"Same-recording pairs: {same_rec_count}")
    print(f"Cross-recording pairs: {len(pairs) - same_rec_count}")

    for shift in args.shifts:
        shift_count = sum(1 for p in pairs if p.shift == shift)
        print(f"Shift {shift:+d}: {shift_count} pairs")

    # Save results
    output_data = {
        'config': {
            'segments_path': args.segments,
            'shifts': args.shifts,
            'pitch_tolerance': args.pitch_tolerance,
            'min_timbre_sim': args.min_timbre_sim,
            'min_dynamics_sim': args.min_dynamics_sim,
            'min_overlap': args.min_overlap,
        },
        'segments': [
            {
                'idx': seg.idx,
                'latent_path': seg.latent_path,
                'start_frame': seg.start_frame,
                'end_frame': seg.end_frame,
                'median_midi': seg.median_midi,
                'group': seg.group,
                'recording_id': seg.recording_id,
            }
            for seg in valid_segments
        ],
        'pairs': [
            {
                'source_idx': p.source_idx,
                'target_idx': p.target_idx,
                'source_pitch': p.source_pitch,
                'target_pitch': p.target_pitch,
                'shift': p.shift,
                'timbre_sim': p.timbre_sim,
                'dynamics_sim': p.dynamics_sim,
                'alignment_offset': p.alignment_offset,
                'overlap_frames': p.overlap_frames,
                'same_recording': p.same_recording,
                'combined_score': p.combined_score,
            }
            for p in pairs
        ],
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to: {args.output}")

    # Show some examples
    print("\n=== Top pairs ===")
    for p in pairs[:10]:
        src = valid_segments[p.source_idx]
        tgt = valid_segments[p.target_idx]
        print(f"  Shift {p.shift:+d}: pitch {p.source_pitch:.1f}->{p.target_pitch:.1f}, "
              f"timbre={p.timbre_sim:.2f}, dyn={p.dynamics_sim:.2f}, "
              f"overlap={p.overlap_frames}, same_rec={p.same_recording}")


if __name__ == "__main__":
    main()
