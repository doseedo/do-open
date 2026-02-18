#!/usr/bin/env python3
"""
Fix conditioning paths in manifest by searching across all conditioning folders.
Searches: newconditioning, moreconditioning, evenmoreconditioning
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# Conditioning folders in priority order (search newest first)
CONDITIONING_FOLDERS = [
    "/mnt/msdd/evenmoreconditioning",
    "/mnt/msdd/moreconditioning",
    "/mnt/msdd/newconditioning"
]

# Required conditioning files
CONDITIONING_TYPES = ["onsets", "rframe", "rbend", "amp", "f0", "f0_masked"]


def extract_relative_path(audio_path: str) -> str:
    """
    Extract relative path from audio_path.

    Example:
        /home/arlo/gcs-bucket/protools/2025-03-30/New/Session/Audio Files/file.wav
        -> 2025-03-30/New/Session/Audio Files
    """
    # Find "protools/" in path
    if "protools/" in audio_path:
        parts = audio_path.split("protools/")[1]
        # Remove the filename
        relative_dir = str(Path(parts).parent)
        return relative_dir
    else:
        # Fallback: just use parent directory
        return str(Path(audio_path).parent)


def get_base_filename(audio_path: str) -> str:
    """Get base filename without extension."""
    return Path(audio_path).stem


def find_conditioning_file(
    base_filename: str,
    relative_path: str,
    cond_type: str
) -> Optional[str]:
    """
    Search for conditioning file across all conditioning folders.

    Args:
        base_filename: e.g., "LDVOX WET_01"
        relative_path: e.g., "2025-03-30/New/Session/Audio Files"
        cond_type: e.g., "amp", "rframe"

    Returns:
        Full path if found, None otherwise
    """
    expected_filename = f"{base_filename}.{cond_type}.npy"

    # Try each conditioning folder
    for cond_folder in CONDITIONING_FOLDERS:
        candidate_path = os.path.join(cond_folder, relative_path, expected_filename)

        if os.path.exists(candidate_path):
            return candidate_path

    return None


def verify_and_fix_conditioning_paths(entry: Dict) -> Tuple[Dict, Dict]:
    """
    Verify and fix conditioning paths for a single entry.

    Returns:
        (updated_entry, stats_dict)
    """
    stats = {
        "original_folder": None,
        "new_folder": None,
        "fixed": 0,
        "missing": 0,
        "already_correct": 0,
        "missing_files": []
    }

    audio_path = entry.get("audio_path", "")
    if not audio_path:
        return entry, stats

    base_filename = get_base_filename(audio_path)
    relative_path = extract_relative_path(audio_path)

    # Check current conditioning paths
    cond_paths = entry.get("conditioning_paths", {})
    if not cond_paths:
        return entry, stats

    # Detect original folder
    first_path = cond_paths.get("amp", "")
    if "evenmoreconditioning" in first_path:
        stats["original_folder"] = "evenmoreconditioning"
    elif "moreconditioning" in first_path:
        stats["original_folder"] = "moreconditioning"
    elif "newconditioning" in first_path:
        stats["original_folder"] = "newconditioning"

    # Verify and fix each conditioning type
    updated_cond_paths = {}

    for cond_type in CONDITIONING_TYPES:
        current_path = cond_paths.get(cond_type)

        # Check if current path exists
        if current_path and os.path.exists(current_path):
            updated_cond_paths[cond_type] = current_path
            stats["already_correct"] += 1
            continue

        # Search for correct path
        correct_path = find_conditioning_file(base_filename, relative_path, cond_type)

        if correct_path:
            updated_cond_paths[cond_type] = correct_path

            # Detect new folder
            if "evenmoreconditioning" in correct_path:
                stats["new_folder"] = "evenmoreconditioning"
            elif "moreconditioning" in correct_path:
                stats["new_folder"] = "moreconditioning"
            elif "newconditioning" in correct_path:
                stats["new_folder"] = "newconditioning"

            if current_path != correct_path:
                stats["fixed"] += 1
        else:
            # File not found in any folder
            stats["missing"] += 1
            stats["missing_files"].append(cond_type)
            # Keep original path (even if broken) for tracking
            if current_path:
                updated_cond_paths[cond_type] = current_path

    # Update entry
    entry["conditioning_paths"] = updated_cond_paths

    return entry, stats


def main():
    input_manifest = "./vocal_training_manifest_yamnet_filtered.json"
    output_manifest = "./vocal_training_manifest_yamnet_filtered_fixed.json"

    print("=" * 80)
    print("Fixing Conditioning Paths in Manifest")
    print("=" * 80)
    print(f"Input:  {input_manifest}")
    print(f"Output: {output_manifest}")
    print()

    # Load manifest
    print(f"Loading manifest...")
    with open(input_manifest, 'r') as f:
        manifest = json.load(f)

    print(f"Total entries: {len(manifest)}")
    print()

    # Process each entry
    updated_manifest = []
    global_stats = {
        "total": 0,
        "fixed": 0,
        "already_correct": 0,
        "missing": 0,
        "folder_distribution": {
            "newconditioning": 0,
            "moreconditioning": 0,
            "evenmoreconditioning": 0,
            "mixed": 0
        },
        "entries_with_missing": [],
        "missing_by_type": {cond: 0 for cond in CONDITIONING_TYPES}
    }

    print("Processing entries...")
    for i, entry in enumerate(manifest):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(manifest)} entries...")

        updated_entry, stats = verify_and_fix_conditioning_paths(entry)
        updated_manifest.append(updated_entry)

        global_stats["total"] += 1
        global_stats["fixed"] += stats["fixed"]
        global_stats["already_correct"] += stats["already_correct"]

        if stats["missing"] > 0:
            global_stats["missing"] += 1
            global_stats["entries_with_missing"].append({
                "index": i,
                "audio_path": entry.get("audio_path", ""),
                "missing_files": stats["missing_files"]
            })

            for missing_type in stats["missing_files"]:
                global_stats["missing_by_type"][missing_type] += 1

        # Track folder distribution
        if stats["new_folder"]:
            global_stats["folder_distribution"][stats["new_folder"]] += 1

    print(f"  Processed {len(manifest)}/{len(manifest)} entries.")
    print()

    # Save updated manifest
    print(f"Saving updated manifest to {output_manifest}...")
    with open(output_manifest, 'w') as f:
        json.dump(updated_manifest, f, indent=2)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total entries processed: {global_stats['total']}")
    print(f"Files fixed: {global_stats['fixed']}")
    print(f"Already correct: {global_stats['already_correct']}")
    print(f"Entries with missing files: {global_stats['missing']}")
    print()

    print("Conditioning folder distribution:")
    for folder, count in global_stats["folder_distribution"].items():
        if count > 0:
            print(f"  {folder}: {count} entries")
    print()

    if global_stats["missing"] > 0:
        print("Missing files by type:")
        for cond_type, count in global_stats["missing_by_type"].items():
            if count > 0:
                print(f"  {cond_type}: {count} files")
        print()

        print(f"First 10 entries with missing files:")
        for item in global_stats["entries_with_missing"][:10]:
            print(f"  [{item['index']}] {Path(item['audio_path']).name}")
            print(f"      Missing: {', '.join(item['missing_files'])}")

        if len(global_stats["entries_with_missing"]) > 10:
            print(f"  ... and {len(global_stats['entries_with_missing']) - 10} more")
    else:
        print("✅ All conditioning files found!")

    print()
    print(f"✅ Updated manifest saved to: {output_manifest}")
    print()

    # Save detailed report
    report_file = "./conditioning_fix_report.json"
    with open(report_file, 'w') as f:
        json.dump({
            "summary": {
                "total": global_stats["total"],
                "fixed": global_stats["fixed"],
                "already_correct": global_stats["already_correct"],
                "missing": global_stats["missing"]
            },
            "folder_distribution": global_stats["folder_distribution"],
            "missing_by_type": global_stats["missing_by_type"],
            "entries_with_missing": global_stats["entries_with_missing"]
        }, f, indent=2)

    print(f"📋 Detailed report saved to: {report_file}")


if __name__ == "__main__":
    main()
