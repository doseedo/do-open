#!/usr/bin/env python3
"""
Filter manifest by audio duration - removes entries shorter than minimum duration.
"""

import json
import torchaudio
from pathlib import Path
from tqdm import tqdm


def filter_manifest_by_duration(input_manifest, output_manifest, min_duration=6.0):
    """
    Remove entries from manifest with audio shorter than min_duration seconds.

    Args:
        input_manifest: Input manifest JSON path
        output_manifest: Output manifest JSON path
        min_duration: Minimum duration in seconds (default: 6.0)
    """
    print(f"Loading manifest: {input_manifest}")
    with open(input_manifest, 'r') as f:
        data = json.load(f)

    print(f"Filtering entries shorter than {min_duration} seconds...")
    filtered = []
    removed_count = 0
    removed_paths = []

    for entry in tqdm(data, desc="Checking durations"):
        audio_path = entry.get('audio_path')

        if not audio_path or not Path(audio_path).exists():
            # Keep entries without audio or missing files
            filtered.append(entry)
            continue

        try:
            # Get audio duration
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate

            if duration >= min_duration:
                filtered.append(entry)
            else:
                removed_count += 1
                removed_paths.append((Path(audio_path).name, duration))
        except Exception as e:
            # If we can't read the file, keep it in manifest
            print(f"  Warning: Could not read {audio_path}: {e}")
            filtered.append(entry)

    # Save filtered manifest
    print(f"\nSaving filtered manifest: {output_manifest}")
    with open(output_manifest, 'w') as f:
        json.dump(filtered, f, indent=2)

    # Print summary of removed files
    if removed_paths:
        print(f"\n=== Removed Files ===")
        for name, dur in removed_paths[:20]:  # Show first 20
            print(f"  {name} ({dur:.2f}s)")
        if len(removed_paths) > 20:
            print(f"  ... and {len(removed_paths) - 20} more")

    print(f"\n=== Filter Results ===")
    print(f"Original entries: {len(data)}")
    print(f"Kept:            {len(filtered)}")
    print(f"Removed:         {removed_count}")
    print(f"Saved to: {output_manifest}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter manifest by audio duration")
    parser.add_argument("--input", required=True, help="Input manifest JSON path")
    parser.add_argument("--output", required=True, help="Output manifest JSON path")
    parser.add_argument("--min-duration", type=float, default=6.0,
                       help="Minimum duration in seconds (default: 6.0)")

    args = parser.parse_args()

    filter_manifest_by_duration(args.input, args.output, args.min_duration)
