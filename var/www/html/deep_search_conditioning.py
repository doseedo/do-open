#!/usr/bin/env python3
"""
Deep search: For missing paths, try swapping the conditioning folder.
"""

import json
from pathlib import Path

COND_FOLDERS = [
    "/mnt/msdd/evenmoreconditioning",
    "/mnt/msdd/moreconditioning",
    "/mnt/msdd/newconditioning"
]


def find_by_folder_swap(path: str) -> str:
    """Try swapping conditioning folder to find the file."""
    if not path:
        return None

    # Identify current folder
    current_folder = None
    for folder in COND_FOLDERS:
        if folder in path:
            current_folder = folder
            break

    if not current_folder:
        return None

    # Try other folders
    for try_folder in COND_FOLDERS:
        if try_folder == current_folder:
            continue

        test_path = path.replace(current_folder, try_folder)
        if Path(test_path).exists():
            return test_path

    return None


def main():
    manifest_path = "./vocal_training_manifest_yamnet_filtered_REBUILT.json"
    output_path = "./vocal_training_manifest_yamnet_filtered_FINAL.json"

    print("="*80)
    print("Deep Search: Swap Conditioning Folders")
    print("="*80)
    print(f"Input:  {manifest_path}")
    print(f"Output: {output_path}\n")

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Total entries: {len(manifest)}\n")

    stats = {
        'checked': 0,
        'already_exist': 0,
        'found_by_swap': 0,
        'still_missing': 0,
        'swaps': {'evenmoreconditioning': 0, 'moreconditioning': 0, 'newconditioning': 0}
    }

    print("Searching...")
    for i, entry in enumerate(manifest):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(manifest)}")

        cond_paths = entry.get("conditioning_paths", {})

        for cond_type, path in list(cond_paths.items()):
            if not path:
                continue

            stats['checked'] += 1

            # Check if exists
            if Path(path).exists():
                stats['already_exist'] += 1
                continue

            # Try folder swap
            found_path = find_by_folder_swap(path)

            if found_path:
                cond_paths[cond_type] = found_path
                stats['found_by_swap'] += 1

                # Track which folder it was found in
                for folder in COND_FOLDERS:
                    if folder in found_path:
                        folder_name = folder.split('/')[-1]
                        stats['swaps'][folder_name] += 1
                        break
            else:
                stats['still_missing'] += 1

    print(f"  {len(manifest)}/{len(manifest)}\n")

    # Save
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Report
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Paths checked: {stats['checked']}")
    print(f"Already exist: {stats['already_exist']} ({100*stats['already_exist']/stats['checked']:.1f}%)")
    print(f"Found by folder swap: {stats['found_by_swap']} ({100*stats['found_by_swap']/stats['checked']:.1f}%)")
    print(f"Still missing: {stats['still_missing']} ({100*stats['still_missing']/stats['checked']:.1f}%)")
    print()

    if stats['found_by_swap'] > 0:
        print("Files found by swapping to:")
        for folder, count in sorted(stats['swaps'].items()):
            if count > 0:
                print(f"  {folder}: {count}")
        print()

    # Final verdict
    total_valid = stats['already_exist'] + stats['found_by_swap']
    print(f"Total valid paths: {total_valid} / {stats['checked']} ({100*total_valid/stats['checked']:.1f}%)")
    print(f"\n✅ Saved: {output_path}")


if __name__ == "__main__":
    main()
