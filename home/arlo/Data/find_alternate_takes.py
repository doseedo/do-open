#!/usr/bin/env python3
"""
Find alternate takes in the vocal training manifest.
Alternate takes are files in the same session folder with the same base name
but different numbered extensions (e.g., izaiah.12_17.wav and izaiah.13_18.wav).
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_audio_path(audio_path: str) -> Tuple[str, str, str, str]:
    """
    Parse an audio path into components.
    Returns: (session_folder, base_name, number_extension, full_name)

    Example:
    "/home/arlo/gcs-bucket/protools/2025-06-26/New/Welcome to your life/Audio Files/izaiah.12_17.wav"
    -> ("/.../Welcome to your life/Audio Files", "izaiah", "12_17", "izaiah.12_17.wav")
    """
    path = Path(audio_path)
    full_name = path.name
    session_folder = str(path.parent)

    # Match pattern: basename.number_extension.wav
    # e.g., "izaiah.12_17.wav" -> base="izaiah", ext="12_17"
    # or "BGV H.27_25.wav" -> base="BGV H", ext="27_25"
    match = re.match(r'^(.+?)\.(\d+_\d+)\.wav$', full_name)

    if match:
        base_name = match.group(1)
        number_extension = match.group(2)
        return session_folder, base_name, number_extension, full_name

    # No numbered extension found
    return session_folder, full_name, "", full_name


def find_alternate_takes(manifest_path: str) -> Dict[int, List[Dict]]:
    """
    Find alternate takes for each entry in the manifest.

    Returns a dict mapping manifest_index -> list of alternate take info
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Group files by (session_folder, base_name)
    groups = defaultdict(list)
    seen_paths = {}  # Track unique audio paths

    for idx, entry in enumerate(manifest):
        audio_path = entry.get('audio_path', '')
        if not audio_path:
            continue

        session_folder, base_name, number_ext, full_name = parse_audio_path(audio_path)

        # Only track files with numbered extensions
        if number_ext:
            key = (session_folder, base_name)

            # Only add if we haven't seen this exact path before
            if audio_path not in seen_paths:
                seen_paths[audio_path] = idx
                groups[key].append({
                    'manifest_index': idx,
                    'audio_path': audio_path,
                    'number_extension': number_ext,
                    'entry': entry
                })

    # Find alternates: any group with more than one file
    alternates_map = {}

    for key, files in groups.items():
        if len(files) > 1:
            # Sort by number extension for consistent ordering
            files.sort(key=lambda x: x['number_extension'])

            # For each file, list all OTHER files as alternates
            for file_info in files:
                idx = file_info['manifest_index']
                alternates = [
                    {
                        'manifest_index': alt['manifest_index'],
                        'audio_path': alt['audio_path'],
                        'number_extension': alt['number_extension']
                    }
                    for alt in files if alt['manifest_index'] != idx
                ]
                alternates_map[idx] = alternates

    return alternates_map


def create_alternates_manifest(input_manifest_path: str, output_manifest_path: str):
    """
    Create a new manifest with alternate takes information added.
    """
    with open(input_manifest_path, 'r') as f:
        manifest = json.load(f)

    alternates_map = find_alternate_takes(input_manifest_path)

    # Add alternates to each entry
    new_manifest = []
    for idx, entry in enumerate(manifest):
        new_entry = entry.copy()

        if idx in alternates_map:
            new_entry['alternate_takes'] = alternates_map[idx]
        else:
            new_entry['alternate_takes'] = []

        new_manifest.append(new_entry)

    # Write new manifest
    with open(output_manifest_path, 'w') as f:
        json.dump(new_manifest, f, indent=2)

    # Print statistics
    total_with_alternates = len(alternates_map)
    total_entries = len(manifest)

    print(f"Processed {total_entries} entries")
    print(f"Found {total_with_alternates} entries with alternate takes")

    if total_with_alternates > 0:
        print("\nExample alternate take groups:")
        for idx in list(alternates_map.keys())[:3]:
            entry = manifest[idx]
            alts = alternates_map[idx]
            print(f"\nManifest index {idx}:")
            print(f"  Main: {entry['audio_path']}")
            for alt in alts:
                print(f"  Alt:  {alt['audio_path']}")

    print(f"\nNew manifest written to: {output_manifest_path}")


if __name__ == "__main__":
    input_path = "/home/arlo/Data/vocal_training_manifest_filtered.json"
    output_path = "/home/arlo/Data/vocal_training_manifest_with_alternates.json"

    create_alternates_manifest(input_path, output_path)
