#!/usr/bin/env python3
"""
Simple conditioning path verifier and fixer.
Only checks files that don't exist, then searches for correct path.
"""

import json
import os
from pathlib import Path

CONDITIONING_FOLDERS = [
    "/mnt/msdd/evenmoreconditioning",
    "/mnt/msdd/moreconditioning",
    "/mnt/msdd/newconditioning"
]

COND_TYPES = ["amp", "rbend", "rframe", "onsets", "f0", "f0_masked"]


def find_correct_path(broken_path: str) -> str:
    """
    Given a broken path, search for the file in other conditioning folders.

    Example:
        /mnt/msdd/evenmoreconditioning/2025-03-30/.../file.amp.npy
        -> /mnt/msdd/moreconditioning/2025-03-30/.../file.amp.npy
    """
    # Extract the relative path after the conditioning folder
    for folder in CONDITIONING_FOLDERS:
        if folder in broken_path:
            relative_path = broken_path.split(folder + "/")[1]

            # Try each conditioning folder
            for try_folder in CONDITIONING_FOLDERS:
                candidate = os.path.join(try_folder, relative_path)
                if os.path.exists(candidate):
                    return candidate
            break

    return broken_path  # Return original if not found


def main():
    input_manifest = "./vocal_training_manifest_yamnet_filtered.json"
    output_manifest = "./vocal_training_manifest_yamnet_filtered_FIXED.json"

    print("Loading manifest...")
    with open(input_manifest) as f:
        manifest = json.load(f)

    print(f"Total entries: {len(manifest)}\n")

    stats = {
        'checked': 0,
        'already_ok': 0,
        'fixed': 0,
        'still_missing': 0,
        'folder_dist': {'evenmoreconditioning': 0, 'moreconditioning': 0, 'newconditioning': 0}
    }

    print("Processing...")
    for i, entry in enumerate(manifest):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(manifest)}")

        cond_paths = entry.get("conditioning_paths", {})
        if not cond_paths:
            continue

        updated = False

        for cond_type in COND_TYPES:
            path = cond_paths.get(cond_type)
            if not path:
                continue

            stats['checked'] += 1

            # Check if current path exists
            if os.path.exists(path):
                stats['already_ok'] += 1

                # Track folder distribution
                for folder in CONDITIONING_FOLDERS:
                    if folder in path:
                        folder_name = folder.split('/')[-1]
                        stats['folder_dist'][folder_name] += 1
                        break
                continue

            # Path broken - search for correct one
            correct_path = find_correct_path(path)

            if correct_path != path and os.path.exists(correct_path):
                cond_paths[cond_type] = correct_path
                stats['fixed'] += 1
                updated = True

                # Track new folder
                for folder in CONDITIONING_FOLDERS:
                    if folder in correct_path:
                        folder_name = folder.split('/')[-1]
                        stats['folder_dist'][folder_name] += 1
                        break
            else:
                stats['still_missing'] += 1

    print(f"  {len(manifest)}/{len(manifest)}\n")

    # Save
    print(f"Saving to {output_manifest}...")
    with open(output_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Report
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Files checked: {stats['checked']}")
    print(f"Already correct: {stats['already_ok']}")
    print(f"Fixed: {stats['fixed']}")
    print(f"Still missing: {stats['still_missing']}")
    print()
    print("Folder distribution:")
    for folder, count in sorted(stats['folder_dist'].items()):
        pct = 100 * count / max(1, stats['checked'])
        print(f"  {folder}: {count} ({pct:.1f}%)")

    print(f"\n✅ Saved: {output_manifest}")


if __name__ == "__main__":
    main()
