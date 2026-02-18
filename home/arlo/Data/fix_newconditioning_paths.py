#!/usr/bin/env python3
"""
Fix conditioning paths accounting for newconditioning's different structure.

evenmoreconditioning: /mnt/msdd/evenmoreconditioning/2025-03-31/New/Session/Audio Files/file.amp.npy
newconditioning:      /mnt/msdd/newconditioning/Session/Audio Files/file.amp.npy (NO DATE FOLDER)
"""

import json
from pathlib import Path
import re

COND_TYPES = ["amp", "rbend", "rframe", "onsets", "f0", "f0_masked"]


def extract_session_path(full_path: str) -> str:
    """
    Extract session path without date folders.

    Examples:
        /mnt/msdd/evenmoreconditioning/2025-03-31/New/Session/Audio Files/file.amp.npy
        -> Session/Audio Files/file.amp.npy

        /mnt/msdd/moreconditioning/2025-04-01/New/Another Session/Audio Files/file.amp.npy
        -> Another Session/Audio Files/file.amp.npy
    """
    # Pattern: /2025-XX-XX/New/... -> extract everything after "New/"
    match = re.search(r'/\d{4}-\d{2}-\d{2}/New/(.+)', full_path)
    if match:
        return match.group(1)

    # Fallback: just get everything after the conditioning folder
    for folder in ['/mnt/msdd/evenmoreconditioning/', '/mnt/msdd/moreconditioning/', '/mnt/msdd/newconditioning/']:
        if folder in full_path:
            return full_path.split(folder)[1]

    return None


def find_in_newconditioning(session_relative_path: str, filename: str) -> str:
    """
    Search for file in newconditioning without date structure.

    Args:
        session_relative_path: e.g., "Session/Audio Files"
        filename: e.g., "file.amp.npy"
    """
    # Try direct path
    direct_path = f"/mnt/msdd/newconditioning/{session_relative_path}/{filename}"
    if Path(direct_path).exists():
        return direct_path

    # Try without "Audio Files" subfolder
    session_name = session_relative_path.split('/')[0]
    alt_path = f"/mnt/msdd/newconditioning/{session_name}/{filename}"
    if Path(alt_path).exists():
        return alt_path

    return None


def main():
    input_manifest = "./vocal_training_manifest_yamnet_filtered_FINAL.json"
    output_manifest = "./vocal_training_manifest_yamnet_filtered_FIXED_FINAL.json"

    print("="*80)
    print("Fix newconditioning Paths (No Date Folders)")
    print("="*80)
    print(f"Input:  {input_manifest}")
    print(f"Output: {output_manifest}\n")

    # Load manifest
    with open(input_manifest) as f:
        manifest = json.load(f)

    print(f"Total entries: {len(manifest)}\n")

    stats = {
        'checked': 0,
        'already_exist': 0,
        'found_in_newconditioning': 0,
        'still_missing': 0
    }

    print("Processing...")
    for i, entry in enumerate(manifest):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(manifest)}")

        cond_paths = entry.get("conditioning_paths", {})

        for cond_type in COND_TYPES:
            path = cond_paths.get(cond_type)
            if not path:
                continue

            stats['checked'] += 1

            # Check if already exists
            if Path(path).exists():
                stats['already_exist'] += 1
                continue

            # Extract session path and filename
            session_path = extract_session_path(path)
            filename = Path(path).name

            if not session_path:
                stats['still_missing'] += 1
                continue

            # Try newconditioning
            found_path = find_in_newconditioning(session_path, filename)

            if found_path:
                cond_paths[cond_type] = found_path
                stats['found_in_newconditioning'] += 1
            else:
                stats['still_missing'] += 1

    print(f"  {len(manifest)}/{len(manifest)}\n")

    # Save
    print(f"Saving to {output_manifest}...")
    with open(output_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Report
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Paths checked: {stats['checked']}")
    print(f"Already exist: {stats['already_exist']} ({100*stats['already_exist']/stats['checked']:.1f}%)")
    print(f"Found in newconditioning: {stats['found_in_newconditioning']} ({100*stats['found_in_newconditioning']/stats['checked']:.1f}%)")
    print(f"Still missing: {stats['still_missing']} ({100*stats['still_missing']/stats['checked']:.1f}%)")
    print()

    total_valid = stats['already_exist'] + stats['found_in_newconditioning']
    print(f"Total valid paths: {total_valid} / {stats['checked']} ({100*total_valid/stats['checked']:.1f}%)")
    print(f"\n✅ Saved: {output_manifest}")


if __name__ == "__main__":
    main()
