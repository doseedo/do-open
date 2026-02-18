#!/usr/bin/env python3
"""
Fast conditioning path fixer - uses file existence checks only on broken paths.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# Conditioning folders in priority order
CONDITIONING_FOLDERS = [
    "/mnt/msdd/evenmoreconditioning",
    "/mnt/msdd/moreconditioning",
    "/mnt/msdd/newconditioning"
]

CONDITIONING_TYPES = ["onsets", "rframe", "rbend", "amp", "f0", "f0_masked"]


def extract_relative_path(audio_path: str) -> str:
    """Extract relative path from audio_path."""
    if "protools/" in audio_path:
        parts = audio_path.split("protools/")[1]
        relative_dir = str(Path(parts).parent)
        return relative_dir
    else:
        return str(Path(audio_path).parent)


def get_base_filename(audio_path: str) -> str:
    """Get base filename without extension."""
    return Path(audio_path).stem


def find_conditioning_file(base_filename: str, relative_path: str, cond_type: str) -> Optional[str]:
    """Search for conditioning file across all folders."""
    expected_filename = f"{base_filename}.{cond_type}.npy"

    for cond_folder in CONDITIONING_FOLDERS:
        candidate_path = os.path.join(cond_folder, relative_path, expected_filename)
        if os.path.exists(candidate_path):
            return candidate_path

    return None


def verify_and_fix_entry(entry: Dict) -> Tuple[Dict, int, int]:
    """
    Verify and fix conditioning paths for a single entry.
    Returns: (updated_entry, num_fixed, num_missing)
    """
    audio_path = entry.get("audio_path", "")
    if not audio_path:
        return entry, 0, 0

    cond_paths = entry.get("conditioning_paths", {})
    if not cond_paths:
        return entry, 0, 0

    base_filename = get_base_filename(audio_path)
    relative_path = extract_relative_path(audio_path)

    updated_cond_paths = {}
    num_fixed = 0
    num_missing = 0

    for cond_type in CONDITIONING_TYPES:
        current_path = cond_paths.get(cond_type)

        # Fast path: if current exists, keep it
        if current_path and os.path.exists(current_path):
            updated_cond_paths[cond_type] = current_path
            continue

        # Slow path: search for correct path
        correct_path = find_conditioning_file(base_filename, relative_path, cond_type)

        if correct_path:
            updated_cond_paths[cond_type] = correct_path
            if current_path != correct_path:
                num_fixed += 1
        else:
            num_missing += 1
            # Keep original path for tracking
            if current_path:
                updated_cond_paths[cond_type] = current_path

    entry["conditioning_paths"] = updated_cond_paths
    return entry, num_fixed, num_missing


def main():
    input_manifest = "./vocal_training_manifest_yamnet_filtered.json"
    output_manifest = "./vocal_training_manifest_yamnet_filtered_fixed.json"

    print("=" * 80)
    print("Fast Conditioning Path Fix")
    print("=" * 80)
    print(f"Input:  {input_manifest}")
    print(f"Output: {output_manifest}\n")

    # Load manifest
    print("Loading manifest...")
    with open(input_manifest, 'r') as f:
        manifest = json.load(f)

    print(f"Total entries: {len(manifest)}\n")

    # Process
    updated_manifest = []
    total_fixed = 0
    total_missing = 0
    entries_with_issues = 0

    print("Processing entries...")
    for i, entry in enumerate(manifest):
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(manifest)} ({100*(i+1)/len(manifest):.1f}%)")

        updated_entry, num_fixed, num_missing = verify_and_fix_entry(entry)
        updated_manifest.append(updated_entry)

        total_fixed += num_fixed
        if num_missing > 0:
            total_missing += num_missing
            entries_with_issues += 1

    print(f"  {len(manifest)}/{len(manifest)} (100.0%)\n")

    # Save
    print(f"Saving to {output_manifest}...")
    with open(output_manifest, 'w') as f:
        json.dump(updated_manifest, f, indent=2)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total entries: {len(manifest)}")
    print(f"Paths fixed: {total_fixed}")
    print(f"Entries with missing files: {entries_with_issues}")
    print(f"Total missing files: {total_missing}")
    print(f"\n✅ Updated manifest saved to: {output_manifest}")


if __name__ == "__main__":
    main()
