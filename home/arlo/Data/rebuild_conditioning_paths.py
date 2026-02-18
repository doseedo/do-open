#!/usr/bin/env python3
"""
Rebuild conditioning paths by searching for files by name across all folders.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

CONDITIONING_FOLDERS = [
    "/mnt/msdd/evenmoreconditioning",
    "/mnt/msdd/moreconditioning",
    "/mnt/msdd/newconditioning"
]

COND_TYPES = ["amp", "rbend", "rframe", "onsets", "f0", "f0_masked"]


def build_filename_index():
    """Build index: filename -> full_path for all conditioning files."""
    print("Building filename index (this may take a few minutes)...")

    index = defaultdict(list)

    for folder in CONDITIONING_FOLDERS:
        folder_name = folder.split('/')[-1]
        print(f"  Scanning {folder_name}...")

        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.npy'):
                    full_path = os.path.join(root, file)
                    index[file].append(full_path)

    print(f"  Found {len(index)} unique filenames\n")
    return index


def find_conditioning_file(base_filename: str, cond_type: str, index: dict) -> str:
    """Find conditioning file using filename index."""
    expected_filename = f"{base_filename}.{cond_type}.npy"

    if expected_filename in index:
        # Return first match (they should all be the same file)
        return index[expected_filename][0]

    return None


def main():
    input_manifest = "./vocal_training_manifest_yamnet_filtered.json"
    output_manifest = "./vocal_training_manifest_yamnet_filtered_REBUILT.json"

    print("="*80)
    print("Rebuild Conditioning Paths")
    print("="*80)
    print(f"Input:  {input_manifest}")
    print(f"Output: {output_manifest}\n")

    # Build filename index
    index = build_filename_index()

    # Load manifest
    print("Loading manifest...")
    with open(input_manifest) as f:
        manifest = json.load(f)
    print(f"Total entries: {len(manifest)}\n")

    stats = {
        'total': 0,
        'found': 0,
        'missing': 0,
        'by_folder': defaultdict(int),
        'missing_by_type': defaultdict(int)
    }

    print("Processing...")
    for i, entry in enumerate(manifest):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(manifest)}")

        audio_path = entry.get("audio_path", "")
        if not audio_path:
            continue

        base_filename = Path(audio_path).stem
        cond_paths = entry.get("conditioning_paths", {})

        for cond_type in COND_TYPES:
            stats['total'] += 1

            # Search for file
            found_path = find_conditioning_file(base_filename, cond_type, index)

            if found_path:
                cond_paths[cond_type] = found_path
                stats['found'] += 1

                # Track folder
                for folder in CONDITIONING_FOLDERS:
                    if folder in found_path:
                        folder_name = folder.split('/')[-1]
                        stats['by_folder'][folder_name] += 1
                        break
            else:
                stats['missing'] += 1
                stats['missing_by_type'][cond_type] += 1

        entry["conditioning_paths"] = cond_paths

    print(f"  {len(manifest)}/{len(manifest)}\n")

    # Save
    print(f"Saving to {output_manifest}...")
    with open(output_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Report
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files searched: {stats['total']}")
    print(f"Found: {stats['found']} ({100*stats['found']/max(1,stats['total']):.1f}%)")
    print(f"Missing: {stats['missing']} ({100*stats['missing']/max(1,stats['total']):.1f}%)")
    print()

    if stats['by_folder']:
        print("Files by folder:")
        for folder, count in sorted(stats['by_folder'].items()):
            pct = 100 * count / max(1, stats['found'])
            print(f"  {folder}: {count} ({pct:.1f}%)")
        print()

    if stats['missing'] > 0:
        print("Missing by type:")
        for cond_type, count in sorted(stats['missing_by_type'].items()):
            print(f"  {cond_type}: {count}")
        print()

    print(f"✅ Saved: {output_manifest}")


if __name__ == "__main__":
    main()
