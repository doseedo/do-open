#!/usr/bin/env python3
"""
Filter trumpet_segments.json to remove flagged ensemble recordings.

Usage:
    python filter_manifest.py --threshold 40  # Remove score >= 40
    python filter_manifest.py --flag high_unvoiced_amplitude  # Remove specific flag
"""

import json
import argparse
from pathlib import Path


def filter_manifest(
    segments_json: str,
    detection_results: str,
    output_path: str,
    score_threshold: int = 40,
    required_flag: str = None,
):
    """Filter segments to remove flagged recordings."""

    # Load detection results
    with open(detection_results) as f:
        detection = json.load(f)

    # Build set of f0_paths to remove
    remove_paths = set()
    for rec in detection['all_recordings']:
        score = rec.get('score', 0)
        flags = rec.get('flags', [])
        f0_path = rec.get('f0_path', '')

        should_remove = False
        if required_flag:
            should_remove = required_flag in flags
        else:
            should_remove = score >= score_threshold

        if should_remove:
            remove_paths.add(f0_path)

    print(f"Recordings to remove: {len(remove_paths)}")

    # Load segments
    with open(segments_json) as f:
        data = json.load(f)

    # Filter segments
    original_total = 0
    filtered_total = 0
    new_segments_by_group = {}

    for group_id, segments in data['segments_by_group'].items():
        original_total += len(segments)
        filtered = [s for s in segments if s['f0_path'] not in remove_paths]
        filtered_total += len(filtered)

        if filtered:
            new_segments_by_group[group_id] = filtered

    print(f"Segments: {original_total} -> {filtered_total} ({original_total - filtered_total} removed)")

    # Update data
    data['segments_by_group'] = new_segments_by_group
    data['stats']['total_segments'] = filtered_total
    data['stats']['filtered'] = {
        'original_segments': original_total,
        'removed_segments': original_total - filtered_total,
        'removed_recordings': len(remove_paths),
        'score_threshold': score_threshold,
        'required_flag': required_flag,
    }

    # Recalculate segments per group
    data['stats']['segments_per_group'] = {
        k: len(v) for k, v in new_segments_by_group.items()
    }

    # Save
    print(f"Saving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Print new distribution
    print("\nSegments per group:")
    for group_id in sorted(new_segments_by_group.keys(), key=int):
        count = len(new_segments_by_group[group_id])
        print(f"  Group {group_id}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Filter trumpet segments manifest")

    parser.add_argument('--segments', type=str,
                        default='/home/arlo/Data/pitchshift/v3/trumpet_segments.json',
                        help='Input segments JSON')
    parser.add_argument('--detection', type=str,
                        default='/home/arlo/Data/pitchshift/v3/ensemble_detection_results.json',
                        help='Detection results JSON')
    parser.add_argument('--output', type=str,
                        default='/home/arlo/Data/pitchshift/v3/trumpet_segments_filtered.json',
                        help='Output filtered JSON')
    parser.add_argument('--threshold', type=int, default=40,
                        help='Score threshold for removal (default: 40)')
    parser.add_argument('--flag', type=str, default=None,
                        help='Remove recordings with this specific flag')

    args = parser.parse_args()

    filter_manifest(
        segments_json=args.segments,
        detection_results=args.detection,
        output_path=args.output,
        score_threshold=args.threshold,
        required_flag=args.flag,
    )


if __name__ == "__main__":
    main()
