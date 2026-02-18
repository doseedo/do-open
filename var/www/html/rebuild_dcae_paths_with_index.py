#!/usr/bin/env python3
"""
Rebuild DCAE paths by building an index of all DCAE files and fuzzy matching.
"""

import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def normalize_filename(name):
    """Normalize filename for matching (remove extensions, underscores, dots)."""
    return name.lower().replace('_', ' ').replace('.', ' ').replace('-', ' ')

def build_dcae_index():
    """Build index of all DCAE files."""
    dcae_root = Path("/mnt/msdd/dcae_latentsnew")
    index = defaultdict(list)

    print("Building DCAE index...")

    # Walk through all .pt files
    for pt_file in tqdm(list(dcae_root.rglob("*.pt")), desc="Indexing DCAE files"):
        # Get relative path from dcae_latentsnew
        rel_path = pt_file.relative_to(dcae_root)
        parts = rel_path.parts

        # Handle different structures:
        # 1. date/New|Prev/session/...
        # 2. protools/date/New|Prev/session/...
        # 3. protoolsA/date/New|Prev/session/...

        if len(parts) >= 4:
            # Check if first part is protools/protoolsA or date
            if parts[0] in ['protools', 'protoolsA']:
                # Structure: protools/date/New|Prev/session/...
                if len(parts) >= 5:
                    protools_root = parts[0]  # protools or protoolsA
                    date = parts[1]
                    new_prev = parts[2]
                    session = parts[3]
                    key = (protools_root, date, new_prev, session)
                else:
                    continue
            else:
                # Structure: date/New|Prev/session/...
                date = parts[0]
                new_prev = parts[1]
                session = parts[2]
                key = ('direct', date, new_prev, session)

            # Store filename and full path
            index[key].append({
                'filename': pt_file.name,
                'stem': pt_file.stem,
                'normalized': normalize_filename(pt_file.stem),
                'full_path': str(pt_file)
            })

    print(f"Indexed {sum(len(v) for v in index.values())} DCAE files in {len(index)} sessions\n")
    return index

def extract_session_info(audio_path):
    """Extract date, new/prev, and session from audio path."""
    parts = Path(audio_path).parts

    # Find date
    date = None
    for part in parts:
        if part.startswith('2025-') and len(part) == 10:
            date = part
            break

    # Find New/Prev and session
    new_prev = None
    session = None
    if 'New' in parts:
        idx = parts.index('New')
        new_prev = 'New'
        # Session is the FIRST folder after New (not second)
        session = parts[idx + 1]
    elif 'Prev' in parts:
        idx = parts.index('Prev')
        new_prev = 'Prev'
        session = parts[idx + 1]

    return date, new_prev, session

def find_dcae_match(audio_path, index):
    """Find matching DCAE file using index."""
    audio_file = Path(audio_path)
    audio_stem = audio_file.stem
    audio_normalized = normalize_filename(audio_stem)

    date, new_prev, session = extract_session_info(audio_path)

    if not date or not new_prev or not session:
        return None

    # Determine which protools root (if any) from audio path
    audio_parts = Path(audio_path).parts
    protools_root = None
    if 'gcs-bucket' in audio_parts:
        gcs_idx = audio_parts.index('gcs-bucket')
        if gcs_idx + 1 < len(audio_parts):
            next_part = audio_parts[gcs_idx + 1]
            if next_part in ['protools', 'protoolsA']:
                protools_root = next_part

    # Try different key combinations
    # 1. Try with protools root from audio path
    # 2. Try with opposite protools root
    # 3. Try direct (no protools prefix)
    possible_keys = []

    if protools_root:
        possible_keys.append((protools_root, date, new_prev, session))
        # Also try the other protools root
        other_root = 'protoolsA' if protools_root == 'protools' else 'protools'
        possible_keys.append((other_root, date, new_prev, session))

    # Try direct path
    possible_keys.append(('direct', date, new_prev, session))

    # Try all possible keys
    for key in possible_keys:
        if key not in index:
            continue

        # Try exact stem match first
        for dcae_file in index[key]:
            if dcae_file['stem'] == audio_stem:
                return dcae_file['full_path']

        # Try normalized match
        for dcae_file in index[key]:
            if dcae_file['normalized'] == audio_normalized:
                return dcae_file['full_path']

        # Try prefix match (e.g., "LDVOX WET_01" matches "LDVOX WET.01_02")
        audio_prefix = audio_normalized.split()[0] if audio_normalized else ""
        for dcae_file in index[key]:
            dcae_prefix = dcae_file['normalized'].split()[0] if dcae_file['normalized'] else ""
            if audio_prefix and dcae_prefix and audio_prefix == dcae_prefix:
                # Check if numbers are similar
                audio_nums = ''.join(c for c in audio_stem if c.isdigit())
                dcae_nums = ''.join(c for c in dcae_file['stem'] if c.isdigit())
                if audio_nums and dcae_nums and audio_nums in dcae_nums[:len(audio_nums)+2]:
                    return dcae_file['full_path']

    return None

def find_piano_roll(audio_path):
    """Find piano roll path for audio file."""
    filename = Path(audio_path).stem
    date, new_prev, session = extract_session_info(audio_path)

    if not session:
        return None

    # Try direct session match
    pr_path = Path(f"/mnt/msdd/piano_rolls/{session}/{filename}.pianoroll.npy")
    if pr_path.exists():
        return str(pr_path)

    return None

def main():
    input_manifest = "./vocal_training_manifest_yamnet_filtered.json"
    output_manifest = "./vocal_training_manifest_yamnet_filtered_PATHS_FIXED.json"

    print("="*80)
    print("Rebuild DCAE Paths with Index")
    print("="*80)
    print(f"Input:  {input_manifest}")
    print(f"Output: {output_manifest}\n")

    # Build DCAE index
    dcae_index = build_dcae_index()

    # Load manifest
    with open(input_manifest) as f:
        manifest = json.load(f)

    print(f"Total manifest entries: {len(manifest)}\n")

    stats = {
        'total': len(manifest),
        'pr_found': 0,
        'pr_missing': 0,
        'dcae_found': 0,
        'dcae_missing': 0,
        'both_found': 0
    }

    print("Rebuilding paths...")
    for entry in tqdm(manifest, desc="Processing"):
        audio_path = entry.get("audio_path", "")

        # Find piano roll
        pr_path = find_piano_roll(audio_path)
        if pr_path:
            entry["piano_roll_path"] = pr_path
            stats['pr_found'] += 1
        else:
            entry["piano_roll_path"] = ""
            stats['pr_missing'] += 1

        # Find DCAE latent using index
        dcae_path = find_dcae_match(audio_path, dcae_index)
        if dcae_path:
            entry["dcae_path"] = dcae_path
            stats['dcae_found'] += 1
        else:
            entry["dcae_path"] = ""
            stats['dcae_missing'] += 1

        if pr_path and dcae_path:
            stats['both_found'] += 1

    # Save
    print(f"\nSaving to {output_manifest}...")
    with open(output_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Report
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total entries: {stats['total']:,}\n")

    print("Piano Roll Paths:")
    print(f"  Found:   {stats['pr_found']:,} ({100*stats['pr_found']/stats['total']:.1f}%)")
    print(f"  Missing: {stats['pr_missing']:,} ({100*stats['pr_missing']/stats['total']:.1f}%)\n")

    print("DCAE Paths:")
    print(f"  Found:   {stats['dcae_found']:,} ({100*stats['dcae_found']/stats['total']:.1f}%)")
    print(f"  Missing: {stats['dcae_missing']:,} ({100*stats['dcae_missing']/stats['total']:.1f}%)\n")

    print("Both Found:")
    print(f"  {stats['both_found']:,} ({100*stats['both_found']/stats['total']:.1f}%)")

    if stats['both_found'] == stats['total']:
        print("\n✅ ALL PATHS FOUND - Ready for training!")
    elif stats['both_found'] > 0:
        print(f"\n⚠️  {stats['both_found']:,} entries ready, {stats['total'] - stats['both_found']:,} incomplete")
    else:
        print("\n❌ NO COMPLETE ENTRIES")

    print(f"\n✅ Saved: {output_manifest}")


if __name__ == "__main__":
    main()
