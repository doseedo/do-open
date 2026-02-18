#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instrument Groups Generator

Creates instrument_groups.json file from categorized instrument paths
using the same groups and subgroups as defined in dataloader.py.

Reads from categorized_instrument_paths_subcats_lists directory structure.
"""

import json
from pathlib import Path
from collections import defaultdict

# Same groups as in dataloader.py
APPROVED_GROUPS = ["piano", "guitar", "bass", "strings", "brass", "winds"]
APPROVED_SUBGROUPS = {
    "piano":   ["acoustic_piano", "keys", "undefined"],
    "guitar":  ["acoustic_guitar", "electric_guitar", "plucked", "undefined"],
    "bass":    ["electric_bass", "upright_bass", "undefined"],
    "strings": ["violin", "viola", "cello", "undefined"],
    "brass":   ["trumpet", "trombone", "french_horn", "tuba", "undefined"],
    "winds":   ["bassoon", "clarinet", "flute", "oboe", "sax"],
}

def load_file_paths_from_txt(txt_path):
    """Load file paths from a text file, one per line"""
    try:
        with open(txt_path, 'r') as f:
            paths = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    paths.append(line)
            return paths
    except Exception as e:
        print(f"Warning: Could not read {txt_path}: {e}")
        return []

def build_instrument_groups(base_dir="/home/arlo/Data/categorized_instrument_paths_subcats_lists"):
    """Build instrument groups from categorized files"""
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return {}

    print(f"Building instrument groups from: {base_dir}")

    instrument_groups = {}

    for group_name in APPROVED_GROUPS:
        print(f"\nProcessing group: {group_name}")

        group_dir = base_path / group_name
        if not group_dir.exists():
            print(f"  Warning: Group directory not found: {group_dir}")
            continue

        group_data = {
            "files": [],
            "subgroups": {},
            "total_files": 0
        }

        # Get approved subgroups for this group
        subgroups = APPROVED_SUBGROUPS.get(group_name, [])

        for subgroup_name in subgroups:
            print(f"  Processing subgroup: {subgroup_name}")

            subgroup_file = group_dir / f"{subgroup_name}.txt"
            if not subgroup_file.exists():
                print(f"    Warning: Subgroup file not found: {subgroup_file}")
                continue

            # Load file paths for this subgroup
            file_paths = load_file_paths_from_txt(subgroup_file)

            if file_paths:
                group_data["subgroups"][subgroup_name] = {
                    "files": file_paths,
                    "count": len(file_paths)
                }

                # Add to group-level files list
                group_data["files"].extend(file_paths)

                print(f"    Loaded {len(file_paths)} files")
            else:
                print(f"    No files found for subgroup: {subgroup_name}")

        # Update total count
        group_data["total_files"] = len(group_data["files"])

        if group_data["total_files"] > 0:
            instrument_groups[group_name] = group_data
            print(f"  Group {group_name}: {group_data['total_files']} total files, {len(group_data['subgroups'])} subgroups")
        else:
            print(f"  Skipping empty group: {group_name}")

    return instrument_groups

def save_instrument_groups(instrument_groups, output_file="instrument_groups.json"):
    """Save instrument groups to JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(instrument_groups, f, indent=2)
        print(f"\nSaved instrument groups to: {output_file}")
        return True
    except Exception as e:
        print(f"Error saving instrument groups: {e}")
        return False

def print_summary(instrument_groups):
    """Print a summary of the instrument groups"""
    print("\n" + "="*60)
    print("INSTRUMENT GROUPS SUMMARY")
    print("="*60)

    total_files = 0
    total_groups = len(instrument_groups)
    total_subgroups = 0

    for group_name, group_data in instrument_groups.items():
        group_files = group_data["total_files"]
        subgroups = len(group_data["subgroups"])
        total_files += group_files
        total_subgroups += subgroups

        print(f"\n{group_name.upper()}:")
        print(f"  Total files: {group_files}")
        print(f"  Subgroups: {subgroups}")

        # Show subgroup breakdown
        for subgroup_name, subgroup_data in group_data["subgroups"].items():
            count = subgroup_data["count"]
            percentage = (count / group_files * 100) if group_files > 0 else 0
            print(f"    {subgroup_name}: {count} files ({percentage:.1f}%)")

    print(f"\n" + "="*60)
    print(f"TOTALS:")
    print(f"  Groups: {total_groups}")
    print(f"  Subgroups: {total_subgroups}")
    print(f"  Files: {total_files}")
    print("="*60)

def main():
    """Main function"""
    print("Instrument Groups Generator")
    print("="*50)

    # Build instrument groups from categorized files
    instrument_groups = build_instrument_groups()

    if not instrument_groups:
        print("Error: No instrument groups created")
        return

    # Save to JSON file
    if not save_instrument_groups(instrument_groups):
        return

    # Print summary
    print_summary(instrument_groups)

    print(f"\nSuccess! Generated instrument_groups.json with {len(instrument_groups)} groups")
    print("You can now run: python instpitch.py")

if __name__ == "__main__":
    main()