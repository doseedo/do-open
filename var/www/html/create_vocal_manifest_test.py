#!/usr/bin/env python3
"""
Test vocal manifest generation with first 100 files
"""

import json
from pathlib import Path
import re
import os

def get_vocal_subgroup(audio_path):
    """Determine vocal subgroup based on path."""
    path_upper = audio_path.upper()

    if 'LDVOX' in path_upper or 'LEAD VOX' in path_upper or 'LEAD_VOX' in path_upper:
        return 'lead_vocal'
    elif 'BGV' in path_upper or 'BG VOX' in path_upper or 'BACKING' in path_upper:
        return 'backing_vocal'
    elif 'CHOIR' in path_upper:
        return 'choir'
    elif 'S VOX' in path_upper:
        return 'vocal_stack'
    else:
        return 'vocal_track'

def find_conditioning_base(relative_path, filename_stem):
    """Find which conditioning directory contains the files for this session."""
    conditioning_dirs = [
        "/mnt/msdd/evenmoreconditioning",
        "/mnt/msdd/moreconditioning",
        "/mnt/msdd/newconditioning"
    ]

    # Extract path components from relative_path
    path_parts = Path(relative_path).parts

    if len(path_parts) < 3:
        return None

    # Extract date, "New", and session folder
    date_folder = path_parts[0] if len(path_parts) > 0 else None
    new_folder = path_parts[1] if len(path_parts) > 1 else None
    session_folder = path_parts[2] if len(path_parts) > 2 else None

    # Search for conditioning files in the structured format
    for cond_dir in conditioning_dirs:
        if date_folder and new_folder and session_folder:
            # Construct the expected path: cond_dir/DATE/New/SESSION/Audio Files/
            session_path = Path(cond_dir) / date_folder / new_folder / session_folder / "Audio Files"

            if session_path.exists():
                # Look for the specific conditioning file
                test_file = session_path / f"{filename_stem}.onsets.npy"
                if test_file.exists():
                    return str(session_path / filename_stem)

    return None

def create_manifest_entry(audio_path):
    """Create a manifest entry for a vocal audio path."""
    audio_path = audio_path.strip()

    # Convert audio path to relative parts for generating other paths
    path_obj = Path(audio_path)

    # Extract unique identifier from path for consistent naming
    # Remove /home/arlo/gcs-bucket/protools/ prefix and construct relative path
    relative_path = str(path_obj).replace('/home/arlo/gcs-bucket/protools/', '')

    # Create clean filename without extension
    filename_stem = path_obj.stem

    # Generate a unique hash-like identifier from the full path
    import hashlib
    path_hash = hashlib.md5(relative_path.encode()).hexdigest()[:12]

    # Construct paths for derived files
    piano_roll_path = f"/mnt/msdd/piano_rolls/{path_hash}/{filename_stem}.pianoroll.npy"
    latent_path = f"/mnt/msdd/dcae_latentsnew/{relative_path.replace('.wav', '.pt')}"
    encodec_path = f"/mnt/msdd/encodec_tokens/{path_hash}/{filename_stem}.pt"

    # Find correct conditioning directory
    conditioning_base = find_conditioning_base(relative_path, filename_stem)

    # Vocal conditioning paths (lyrics data)
    vocal_conditioning_base = f"/mnt/msdd/vocal_processing/{filename_stem}"

    entry = {
        "audio_path": audio_path,
        "piano_roll_path": piano_roll_path,
        "latent_path": latent_path,
        "encodec_path": encodec_path,
        "group": "vocal",
        "sub_group": get_vocal_subgroup(audio_path)
    }

    # Only add conditioning paths if they exist
    if conditioning_base:
        entry["conditioning_paths"] = {
            "onsets": f"{conditioning_base}.onsets.npy",
            "rframe": f"{conditioning_base}.rframe.npy",
            "rbend": f"{conditioning_base}.rbend.npy",
            "amp": f"{conditioning_base}.amp.npy",
            "f0": f"{conditioning_base}.f0.npy",
            "f0_masked": f"{conditioning_base}.f0_masked.npy"
        }
    else:
        entry["conditioning_paths"] = {}

    # Add vocal conditioning paths (always include structure even if files don't exist)
    entry["vocal_conditioning_paths"] = {
        "lyrics_data": f"{vocal_conditioning_base}/{filename_stem}_lyrics_ace_step.json",
        "lyrics_tensors": f"{vocal_conditioning_base}/{filename_stem}_tensors.pt",
        "syllable_boundaries": f"{vocal_conditioning_base}/{filename_stem}_syllables.npy"
    }

    return entry

def main():
    """Test vocal manifest generation with first 100 files."""
    vocal_paths_file = "/home/arlo/Data/categorized_instrument_paths_subcats_lists/voice/all.txt"

    if not Path(vocal_paths_file).exists():
        print(f"❌ Vocal paths file not found: {vocal_paths_file}")
        return

    print(f"📁 Loading vocal paths: {vocal_paths_file}")

    with open(vocal_paths_file, 'r') as f:
        vocal_paths = [line.strip() for line in f.readlines() if line.strip()]

    # Take only first 100 files for testing
    vocal_paths = vocal_paths[:100]
    print(f"📊 Testing with first {len(vocal_paths)} vocal paths")

    # Create manifest entries
    print("🎤 Creating vocal manifest entries...")
    vocal_entries = []
    conditioning_stats = {"evenmoreconditioning": 0, "moreconditioning": 0, "newconditioning": 0, "missing": 0}

    for i, audio_path in enumerate(vocal_paths):
        if Path(audio_path).exists():
            entry = create_manifest_entry(audio_path)
            vocal_entries.append(entry)

            # Track conditioning directory usage
            if entry["conditioning_paths"]:
                cond_path = entry["conditioning_paths"].get("onsets", "")
                if "evenmoreconditioning" in cond_path:
                    conditioning_stats["evenmoreconditioning"] += 1
                    print(f"  ✅ Found conditioning: {Path(audio_path).name}")
                elif "moreconditioning" in cond_path:
                    conditioning_stats["moreconditioning"] += 1
                    print(f"  ✅ Found conditioning: {Path(audio_path).name}")
                elif "newconditioning" in cond_path:
                    conditioning_stats["newconditioning"] += 1
                    print(f"  ✅ Found conditioning: {Path(audio_path).name}")
                else:
                    conditioning_stats["missing"] += 1
            else:
                conditioning_stats["missing"] += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(vocal_paths)} files... (Found: {conditioning_stats['evenmoreconditioning'] + conditioning_stats['moreconditioning'] + conditioning_stats['newconditioning']}, Missing: {conditioning_stats['missing']})")

    print(f"✅ Created {len(vocal_entries)} vocal entries")

    # Show statistics
    print(f"\n📁 Conditioning directory usage:")
    for cond_dir, count in conditioning_stats.items():
        print(f"  {cond_dir}: {count} files")

    if vocal_entries:
        print(f"\n📋 Sample entries with conditioning:")
        found_count = 0
        for entry in vocal_entries:
            if entry["conditioning_paths"]:
                filename = Path(entry['audio_path']).name
                onsets_path = entry["conditioning_paths"].get("onsets", "")
                if "evenmoreconditioning" in onsets_path:
                    cond_dir = "evenmoreconditioning"
                elif "moreconditioning" in onsets_path:
                    cond_dir = "moreconditioning"
                elif "newconditioning" in onsets_path:
                    cond_dir = "newconditioning"
                else:
                    cond_dir = "other"
                print(f"  {filename} -> {cond_dir}")
                found_count += 1
                if found_count >= 3:
                    break

if __name__ == "__main__":
    main()