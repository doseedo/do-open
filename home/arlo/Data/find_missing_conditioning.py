#!/usr/bin/env python3
"""
Smart finder for missing conditioning paths.
Searches across moreconditioning, newconditioning, evenmoreconditioning folders.
"""

import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Paths
MANIFEST_PATH = Path("/home/arlo/Data/vocal_training_manifest_yamnet_filtered_ENCODEC_FIXED.json")
OUTPUT_PATH = Path("/home/arlo/Data/vocal_training_manifest_yamnet_filtered_ALL_FIXED.json")

# Conditioning folders to search
COND_FOLDERS = [
    Path("/mnt/msdd/moreconditioning"),
    Path("/mnt/msdd/newconditioning"),
    Path("/mnt/msdd/evenmoreconditioning"),
    Path("/mnt/msdd/newerconditioning")
]

# Required conditioning types
COND_TYPES = ["amp", "rbend", "rframe", "onsets", "f0", "f0_masked"]

def extract_session_from_path(audio_path):
    """Extract session name from audio path."""
    parts = Path(audio_path).parts

    # Look for session after "New" or "Prev"
    if "New" in parts:
        idx = parts.index("New")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    elif "Prev" in parts:
        idx = parts.index("Prev")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    # Fallback: try to find session by looking for date pattern
    for i, part in enumerate(parts):
        if part.startswith("2025-") and len(part) == 10:
            if i + 1 < len(parts) and parts[i + 1] in ["New", "Prev"]:
                if i + 2 < len(parts):
                    return parts[i + 2]

    return None

def normalize_filename(name):
    """Normalize filename for matching (handle different extensions and suffixes)."""
    # Remove common conditioning suffixes
    for suffix in [".amp", ".rbend", ".rframe", ".onsets", ".f0", ".f0_masked"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    # Remove .npy extension
    if name.endswith(".npy"):
        name = name[:-4]

    return name.lower().strip()

def build_conditioning_index():
    """Build index of all conditioning files by session and filename."""
    print("Building conditioning file index...")

    # Structure: {session: {normalized_filename: {cond_type: path}}}
    index = defaultdict(lambda: defaultdict(dict))

    total_files = 0
    for cond_folder in COND_FOLDERS:
        if not cond_folder.exists():
            print(f"  ⚠️  Folder not found: {cond_folder}")
            continue

        print(f"  Indexing {cond_folder.name}...")

        # Find all .npy files
        npy_files = list(cond_folder.rglob("*.npy"))

        for npy_file in tqdm(npy_files, desc=f"  {cond_folder.name}", leave=False):
            total_files += 1

            # Get session (parent folder name)
            # Could be direct child or in subfolder
            relative = npy_file.relative_to(cond_folder)
            session = relative.parts[0]

            # Determine conditioning type from filename
            stem = npy_file.stem
            cond_type = None

            for ct in COND_TYPES:
                if stem.endswith(f".{ct}"):
                    cond_type = ct
                    break

            if not cond_type:
                continue

            # Get base filename (without conditioning suffix)
            base_stem = stem
            for ct in COND_TYPES:
                if base_stem.endswith(f".{ct}"):
                    base_stem = base_stem[:-len(f".{ct}")]
                    break

            # Normalize for matching
            norm_filename = normalize_filename(base_stem)

            # Index by session and filename
            index[session][norm_filename][cond_type] = str(npy_file)

    print(f"  Total files indexed: {total_files:,}")
    print(f"  Unique sessions: {len(index):,}")

    return index

def find_conditioning_for_entry(audio_path, audio_filename, session, cond_index):
    """Find conditioning files for an audio entry."""
    if not session or session not in cond_index:
        return {}

    # Normalize audio filename
    norm_audio = normalize_filename(audio_filename)

    # Try exact match
    if norm_audio in cond_index[session]:
        return cond_index[session][norm_audio]

    # Try with different normalizations
    # Sometimes files have extra dots or underscores
    alt_names = [
        norm_audio.replace("_", "."),
        norm_audio.replace(".", "_"),
        norm_audio.replace(" ", "_"),
        norm_audio.replace(" ", "."),
    ]

    for alt_name in alt_names:
        if alt_name in cond_index[session]:
            return cond_index[session][alt_name]

    # Try fuzzy match (contains or is contained)
    for indexed_name, cond_files in cond_index[session].items():
        # Check if one is a substring of the other
        if norm_audio in indexed_name or indexed_name in norm_audio:
            # Make sure it's a reasonable match (not too different in length)
            len_diff = abs(len(norm_audio) - len(indexed_name))
            if len_diff <= 5:  # Allow up to 5 character difference
                return cond_files

    return {}

def main():
    print("=" * 80)
    print("Smart Conditioning Path Finder")
    print("=" * 80)
    print(f"Input: {MANIFEST_PATH}")
    print(f"Output: {OUTPUT_PATH}\n")

    # Load manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    print(f"Total entries: {len(manifest)}\n")

    # Build conditioning index
    cond_index = build_conditioning_index()

    # Find missing conditioning
    stats = {
        'had_all_cond': 0,
        'missing_some_cond': 0,
        'found_all_missing': 0,
        'found_partial': 0,
        'found_nothing': 0,
    }

    print("\nSearching for missing conditioning files...")

    for entry in tqdm(manifest, desc="Processing entries"):
        audio_path = entry['audio_path']
        audio_file = Path(audio_path)
        audio_filename = audio_file.stem

        # Extract session
        session = extract_session_from_path(audio_path)

        # Check current conditioning paths
        cond_paths = entry.get('conditioning_paths', {})

        # Check what's missing
        missing_types = []
        for ct in COND_TYPES:
            ct_path = cond_paths.get(ct, '')
            if not ct_path or not Path(ct_path).exists():
                missing_types.append(ct)

        if not missing_types:
            stats['had_all_cond'] += 1
            continue

        stats['missing_some_cond'] += 1

        # Find missing conditioning
        found_cond = find_conditioning_for_entry(audio_path, audio_filename, session, cond_index)

        if not found_cond:
            stats['found_nothing'] += 1
            continue

        # Update entry with found paths
        found_count = 0
        for ct in missing_types:
            if ct in found_cond:
                if 'conditioning_paths' not in entry:
                    entry['conditioning_paths'] = {}
                entry['conditioning_paths'][ct] = found_cond[ct]
                found_count += 1

        if found_count == len(missing_types):
            stats['found_all_missing'] += 1
        elif found_count > 0:
            stats['found_partial'] += 1
        else:
            stats['found_nothing'] += 1

    # Save updated manifest
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Calculate final stats
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Entries that had all conditioning: {stats['had_all_cond']:,}")
    print(f"Entries missing some conditioning: {stats['missing_some_cond']:,}")
    print(f"  → Found all missing: {stats['found_all_missing']:,}")
    print(f"  → Found partial: {stats['found_partial']:,}")
    print(f"  → Found nothing: {stats['found_nothing']:,}")

    # Calculate new totals
    now_complete = stats['had_all_cond'] + stats['found_all_missing']
    pct = (now_complete / len(manifest)) * 100
    print(f"\n✅ Entries with complete conditioning: {now_complete:,} / {len(manifest):,} ({pct:.1f}%)")
    print(f"\n✅ Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
