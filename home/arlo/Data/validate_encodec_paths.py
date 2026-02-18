#!/usr/bin/env python3
"""
Validate and rebuild encodec paths in vocal manifest.
Encodec tokens are stored in /mnt/msdd/encodec_tokens/{session_name}/{filename}.pt
"""

import json
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from tqdm import tqdm

# Paths
MANIFEST_PATH = Path("/home/arlo/Data/vocal_training_manifest_yamnet_filtered_COMPLETE.json")
OUTPUT_PATH = Path("/home/arlo/Data/vocal_training_manifest_yamnet_filtered_ENCODEC_FIXED.json")
ENCODEC_ROOT = Path("/mnt/msdd/encodec_tokens")

def normalize_filename(name):
    """Normalize filename for fuzzy matching."""
    return name.lower().replace('_', ' ').replace('.', ' ').strip()

def build_encodec_index():
    """Build index of all encodec files."""
    print("Building encodec index...")
    index = defaultdict(list)

    for pt_file in tqdm(list(ENCODEC_ROOT.rglob("*.pt")), desc="Indexing encodec files"):
        session = pt_file.parent.name
        stem = pt_file.stem

        # Index by (session, filename)
        key = (session, stem)
        index[key].append(str(pt_file))

    print(f"Indexed {len(index)} unique (session, filename) combinations")
    return index

def fuzzy_match_filename(audio_stem, candidates, threshold=0.6):
    """Find best fuzzy match from candidates."""
    if not candidates:
        return None

    norm_audio = normalize_filename(audio_stem)
    best_score = 0
    best_match = None

    for cand_path in candidates:
        cand_stem = Path(cand_path).stem
        norm_cand = normalize_filename(cand_stem)

        score = SequenceMatcher(None, norm_audio, norm_cand).ratio()
        if score > best_score:
            best_score = score
            best_match = cand_path

    if best_score >= threshold:
        return best_match
    return None

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

def find_encodec_path(audio_path, encodec_index):
    """Find encodec path for audio file."""
    audio_file = Path(audio_path)
    audio_stem = audio_file.stem

    # Extract session
    session = extract_session_from_path(audio_path)
    if not session:
        return None

    # Try exact match first
    key = (session, audio_stem)
    if key in encodec_index:
        return encodec_index[key][0]

    # Try fuzzy match within same session
    session_files = [path for (sess, _), paths in encodec_index.items()
                     if sess == session for path in paths]

    return fuzzy_match_filename(audio_stem, session_files)

def main():
    print("=" * 80)
    print("Validate and Rebuild Encodec Paths")
    print("=" * 80)
    print(f"Input: {MANIFEST_PATH}")
    print(f"Output: {OUTPUT_PATH}\n")

    # Load manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    print(f"Total entries: {len(manifest)}\n")

    # Build encodec index
    encodec_index = build_encodec_index()

    # Validate and rebuild
    stats = {
        'had_encodec': 0,
        'encodec_existed': 0,
        'encodec_found': 0,
        'encodec_not_found': 0
    }

    print("\nProcessing entries...")
    for entry in tqdm(manifest, desc="Rebuilding encodec paths"):
        audio_path = entry['audio_path']
        old_encodec = entry.get('encodec_path', '')

        if old_encodec:
            stats['had_encodec'] += 1
            if Path(old_encodec).exists():
                stats['encodec_existed'] += 1
                continue  # Keep existing valid path

        # Find encodec path
        new_encodec = find_encodec_path(audio_path, encodec_index)

        if new_encodec:
            entry['encodec_path'] = new_encodec
            stats['encodec_found'] += 1
        else:
            stats['encodec_not_found'] += 1

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Report
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Entries with encodec_path field: {stats['had_encodec']}")
    print(f"Encodec already existed: {stats['encodec_existed']}")
    print(f"Encodec found and updated: {stats['encodec_found']}")
    print(f"Encodec not found: {stats['encodec_not_found']}")

    total_with_encodec = stats['encodec_existed'] + stats['encodec_found']
    pct = (total_with_encodec / len(manifest)) * 100
    print(f"\nTotal with valid encodec: {total_with_encodec} / {len(manifest)} ({pct:.1f}%)")
    print(f"\n✅ Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
