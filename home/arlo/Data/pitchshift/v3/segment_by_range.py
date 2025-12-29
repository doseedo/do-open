#!/usr/bin/env python3
"""
Segment recordings into clips by pitch range.

Analyzes f0 data to find sections that stay within ±3 semitones (6 semitone range).
Groups clips by their median pitch into range groups.

This creates the foundation for range-group based training similar to mute_translator.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def fix_mount_path(path: str) -> str:
    """Fix paths that might have wrong mount points."""
    if not path:
        return path
    replacements = [
        ('/mnt/msdd/', '/mnt/msdd2/'),
        ('/home/arlo/gcs-bucket/', '/mnt/gcs-bucket/'),
    ]
    for old, new in replacements:
        if old in path:
            path = path.replace(old, new)
    return path


def hz_to_midi(hz):
    """Convert Hz to MIDI note number. Works with scalars or arrays."""
    hz = np.asarray(hz)
    result = np.zeros_like(hz, dtype=float)
    valid = hz > 0
    result[valid] = 69 + 12 * np.log2(hz[valid] / 440.0)
    return result


def find_stable_segments(
    f0: np.ndarray,
    max_range_semitones: float = 6.0,
    min_segment_frames: int = 16,
    hop_frames: int = 8,
) -> List[Tuple[int, int, float]]:
    """
    Find segments where pitch stays within ±3 semitones of the segment median.

    Args:
        f0: F0 array in Hz
        max_range_semitones: Maximum pitch range within segment (default 6 = ±3)
        min_segment_frames: Minimum segment length
        hop_frames: Step size for segment search

    Returns:
        List of (start_frame, end_frame, median_midi) tuples
    """
    # Convert to MIDI
    midi = np.zeros_like(f0)
    valid_mask = (f0 > 0) & ~np.isnan(f0)
    midi[valid_mask] = hz_to_midi(f0[valid_mask])

    segments = []
    T = len(f0)

    i = 0
    while i < T - min_segment_frames:
        # Skip unvoiced regions
        if not valid_mask[i]:
            i += 1
            continue

        # Try to extend segment as far as possible
        start = i
        end = i + min_segment_frames

        # Get initial segment stats
        seg_midi = midi[start:end]
        seg_valid = valid_mask[start:end]

        if seg_valid.sum() < min_segment_frames // 2:
            i += hop_frames
            continue

        valid_pitches = seg_midi[seg_valid]
        if len(valid_pitches) == 0:
            i += hop_frames
            continue

        median_pitch = np.median(valid_pitches)

        # Extend segment while pitch stays within range
        while end < T:
            if valid_mask[end]:
                pitch_diff = abs(midi[end] - median_pitch)
                if pitch_diff > max_range_semitones / 2:
                    break
            end += 1

            # Recalculate median periodically
            if (end - start) % 16 == 0:
                seg_midi = midi[start:end]
                seg_valid = valid_mask[start:end]
                valid_pitches = seg_midi[seg_valid]
                if len(valid_pitches) > 0:
                    new_median = np.median(valid_pitches)
                    # Check if range is still valid
                    pitch_range = valid_pitches.max() - valid_pitches.min()
                    if pitch_range > max_range_semitones:
                        end -= 16  # Back up
                        break
                    median_pitch = new_median

        # Validate segment
        seg_midi = midi[start:end]
        seg_valid = valid_mask[start:end]
        valid_pitches = seg_midi[seg_valid]

        if len(valid_pitches) >= min_segment_frames // 2:
            pitch_range = valid_pitches.max() - valid_pitches.min()
            if pitch_range <= max_range_semitones:
                final_median = np.median(valid_pitches)
                segments.append((start, end, final_median))
                i = end
                continue

        i += hop_frames

    return segments


def get_range_group(midi_pitch: float, group_size: int = 6, base_pitch: int = 48) -> int:
    """
    Get the range group for a MIDI pitch.

    Args:
        midi_pitch: MIDI note number
        group_size: Size of each group in semitones (default 6)
        base_pitch: Base pitch for group 0 (default 48 = C3)

    Returns:
        Group number (0, 1, 2, ...)
    """
    return int((midi_pitch - base_pitch) // group_size)


def process_manifest(
    manifest_path: str,
    output_path: str,
    instrument: str = 'trumpet',
    group_size: int = 6,
    base_pitch: int = 48,
    max_pitch: int = 96,
    min_segment_frames: int = 32,
    max_range_semitones: float = 6.0,
) -> Dict:
    """
    Process manifest and create range-grouped segments.

    Returns:
        Dictionary with segments grouped by range
    """
    print(f"Loading manifest: {manifest_path}")
    with open(manifest_path, 'r') as f:
        data = json.load(f)

    # Filter for instrument
    entries = []
    for entry in data:
        if entry.get('is_muted', False):
            continue
        sub_group = entry.get('sub_group', '').lower()
        path = entry.get('latent_path', '').lower()
        if sub_group == instrument.lower() or instrument.lower() in path:
            entries.append(entry)

    print(f"Found {len(entries)} {instrument} entries")

    # Calculate number of groups
    num_groups = (max_pitch - base_pitch) // group_size
    print(f"Range groups: {num_groups} (base={base_pitch}, size={group_size})")

    # Process each entry
    segments_by_group = defaultdict(list)
    stats = {
        'total_entries': len(entries),
        'entries_with_f0': 0,
        'total_segments': 0,
        'segments_per_group': defaultdict(int),
    }

    for entry in tqdm(entries, desc="Processing entries"):
        # Get f0 path
        cond = entry.get('conditioning_paths', {})
        f0_path = fix_mount_path(cond.get('f0', ''))
        latent_path = fix_mount_path(entry.get('latent_path', ''))

        if not f0_path or not os.path.exists(f0_path):
            continue
        if not latent_path or not os.path.exists(latent_path):
            continue

        stats['entries_with_f0'] += 1

        try:
            f0 = np.load(f0_path)
        except Exception as e:
            continue

        # Find stable segments
        segments = find_stable_segments(
            f0,
            max_range_semitones=max_range_semitones,
            min_segment_frames=min_segment_frames,
        )

        for start, end, median_midi in segments:
            group = get_range_group(median_midi, group_size, base_pitch)

            # Skip out-of-range groups
            if group < 0 or group >= num_groups:
                continue

            segment_info = {
                'latent_path': latent_path,
                'f0_path': f0_path,
                'audio_path': fix_mount_path(entry.get('audio_path', '')),
                'start_frame': int(start),
                'end_frame': int(end),
                'median_midi': float(median_midi),
                'range_group': int(group),
                'duration_frames': int(end - start),
            }

            segments_by_group[group].append(segment_info)
            stats['total_segments'] += 1
            stats['segments_per_group'][group] += 1

    # Convert to regular dict
    segments_by_group = dict(segments_by_group)

    # Print stats
    print(f"\n{'='*60}")
    print("Segmentation Stats:")
    print(f"  Entries with f0: {stats['entries_with_f0']}")
    print(f"  Total segments: {stats['total_segments']}")
    print(f"\nSegments per group:")
    for group in sorted(stats['segments_per_group'].keys()):
        count = stats['segments_per_group'][group]
        pitch_start = base_pitch + group * group_size
        pitch_end = pitch_start + group_size
        print(f"  Group {group} (MIDI {pitch_start}-{pitch_end}): {count} segments")
    print(f"{'='*60}\n")

    # Build output
    output = {
        'config': {
            'instrument': instrument,
            'group_size': group_size,
            'base_pitch': base_pitch,
            'max_pitch': max_pitch,
            'num_groups': num_groups,
            'min_segment_frames': min_segment_frames,
            'max_range_semitones': max_range_semitones,
        },
        'stats': {
            'total_entries': stats['total_entries'],
            'entries_with_f0': stats['entries_with_f0'],
            'total_segments': stats['total_segments'],
            'segments_per_group': dict(stats['segments_per_group']),
        },
        'segments_by_group': segments_by_group,
    }

    # Save
    print(f"Saving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    return output


def main():
    parser = argparse.ArgumentParser(description="Segment recordings by pitch range")

    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to training manifest JSON')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for segmented data JSON')
    parser.add_argument('--instrument', type=str, default='trumpet',
                        help='Instrument to process')
    parser.add_argument('--group_size', type=int, default=6,
                        help='Size of each range group in semitones (default: 6)')
    parser.add_argument('--base_pitch', type=int, default=48,
                        help='Base MIDI pitch for group 0 (default: 48 = C3)')
    parser.add_argument('--max_pitch', type=int, default=96,
                        help='Maximum MIDI pitch to consider (default: 96 = C7)')
    parser.add_argument('--min_segment_frames', type=int, default=32,
                        help='Minimum segment length in frames (default: 32)')
    parser.add_argument('--max_range', type=float, default=6.0,
                        help='Maximum pitch range within segment in semitones (default: 6)')

    args = parser.parse_args()

    process_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        instrument=args.instrument,
        group_size=args.group_size,
        base_pitch=args.base_pitch,
        max_pitch=args.max_pitch,
        min_segment_frames=args.min_segment_frames,
        max_range_semitones=args.max_range,
    )


if __name__ == "__main__":
    main()
