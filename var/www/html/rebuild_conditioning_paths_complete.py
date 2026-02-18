#!/usr/bin/env python3
"""
Rebuild conditioning paths by searching all conditioning folders.
"""

import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

COND_TYPES = ["amp", "rbend", "rframe", "onsets", "f0", "f0_masked"]

COND_FOLDERS = [
    "/mnt/msdd/evenmoreconditioning",
    "/mnt/msdd/moreconditioning",
    "/mnt/msdd/newconditioning",
    "/mnt/msdd/newerconditioning"
]

def build_conditioning_index():
    """Build index of all conditioning files."""
    index = defaultdict(lambda: defaultdict(dict))

    print("Building conditioning index...")

    for cond_folder in COND_FOLDERS:
        cond_root = Path(cond_folder)
        if not cond_root.exists():
            print(f"  Skipping {cond_folder} (doesn't exist)")
            continue

        folder_name = cond_root.name
        print(f"  Indexing {folder_name}...")

        # Find all .npy files
        npy_files = list(cond_root.rglob("*.npy"))

        for npy_file in tqdm(npy_files, desc=f"    {folder_name}", leave=False):
            # Determine conditioning type from filename
            stem = npy_file.stem
            cond_type = None

            for ct in COND_TYPES:
                if stem.endswith(f".{ct}"):
                    cond_type = ct
                    # Get the base filename without the conditioning type suffix
                    base_stem = stem[:-len(f".{ct}")]
                    break

            if not cond_type:
                continue

            # Build a key from the path structure
            # Try to extract session info
            rel_path = npy_file.relative_to(cond_root)
            parts = rel_path.parts

            # Different structures:
            # 1. newconditioning: session/file.cond_type.npy
            # 2. evenmoreconditioning/moreconditioning: date/New/session/Audio Files/file.cond_type.npy

            session = None
            if len(parts) >= 1:
                # Try to find session folder
                if len(parts) >= 4 and parts[1] in ['New', 'Prev']:
                    # Structure: date/New/session/...
                    session = parts[2]
                elif len(parts) >= 1:
                    # Structure: session/...
                    session = parts[0]

            if session:
                # Create key: (session, base_filename)
                key = (session, base_stem)
                index[key][cond_type] = str(npy_file)

    total_files = sum(len(types) for session_files in index.values() for types in session_files.values())
    print(f"\nIndexed {total_files:,} conditioning files for {len(index):,} unique audio files\n")
    return index

def extract_session_info(audio_path):
    """Extract session from audio path."""
    parts = Path(audio_path).parts

    # Find New/Prev and session
    session = None
    if 'New' in parts:
        idx = parts.index('New')
        session = parts[idx + 1]
    elif 'Prev' in parts:
        idx = parts.index('Prev')
        session = parts[idx + 1]

    return session

def find_conditioning_paths(audio_path, index):
    """Find conditioning paths for audio file."""
    audio_file = Path(audio_path)
    base_stem = audio_file.stem

    session = extract_session_info(audio_path)

    if not session:
        return {}

    # Try exact match first
    key = (session, base_stem)
    if key in index:
        return index[key]

    # Try normalized matching (spaces vs underscores)
    base_normalized = base_stem.replace('_', ' ').replace('.', ' ')

    for (idx_session, idx_stem), cond_paths in index.items():
        if idx_session == session:
            idx_normalized = idx_stem.replace('_', ' ').replace('.', ' ')
            if base_normalized == idx_normalized:
                return cond_paths

    return {}

def main():
    input_manifest = "./vocal_training_manifest_yamnet_filtered_PATHS_FIXED.json"
    output_manifest = "./vocal_training_manifest_yamnet_filtered_COMPLETE.json"

    print("="*80)
    print("Rebuild Conditioning Paths")
    print("="*80)
    print(f"Input:  {input_manifest}")
    print(f"Output: {output_manifest}\n")

    # Build conditioning index
    cond_index = build_conditioning_index()

    # Load manifest
    with open(input_manifest) as f:
        manifest = json.load(f)

    print(f"Total manifest entries: {len(manifest)}\n")

    stats = {
        'total': len(manifest),
        'has_all_6': 0,
        'has_some': 0,
        'has_none': 0,
        'per_type': {t: 0 for t in COND_TYPES}
    }

    print("Rebuilding conditioning paths...")
    for entry in tqdm(manifest, desc="Processing"):
        audio_path = entry.get("audio_path", "")

        # Find conditioning paths
        cond_paths = find_conditioning_paths(audio_path, cond_index)

        # Update entry
        entry["conditioning_paths"] = {}
        existing_count = 0

        for cond_type in COND_TYPES:
            if cond_type in cond_paths:
                entry["conditioning_paths"][cond_type] = cond_paths[cond_type]
                stats['per_type'][cond_type] += 1
                existing_count += 1
            else:
                entry["conditioning_paths"][cond_type] = ""

        if existing_count == 6:
            stats['has_all_6'] += 1
        elif existing_count > 0:
            stats['has_some'] += 1
        else:
            stats['has_none'] += 1

    # Save
    print(f"\nSaving to {output_manifest}...")
    with open(output_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Report
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total entries: {stats['total']:,}\n")

    print("Conditioning Status:")
    print(f"  Has all 6 types:  {stats['has_all_6']:,} ({100*stats['has_all_6']/stats['total']:.1f}%)")
    print(f"  Has some types:   {stats['has_some']:,} ({100*stats['has_some']/stats['total']:.1f}%)")
    print(f"  Has none:         {stats['has_none']:,} ({100*stats['has_none']/stats['total']:.1f}%)\n")

    print("By type:")
    for cond_type in COND_TYPES:
        count = stats['per_type'][cond_type]
        print(f"  {cond_type:12s}: {count:,} ({100*count/stats['total']:.1f}%)")

    print(f"\n✅ Saved: {output_manifest}")


if __name__ == "__main__":
    main()
