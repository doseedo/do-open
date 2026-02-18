#!/usr/bin/env python3
"""
Create Vocal Training Manifest - Final Version

Creates vocal training manifest entries directly from vocal paths list.
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

    for cond_dir in conditioning_dirs:
        # Strategy 1: Date-based structure (evenmoreconditioning/moreconditioning)
        if date_folder and new_folder and session_folder:
            session_path = Path(cond_dir) / date_folder / new_folder / session_folder / "Audio Files"
            if session_path.exists():
                test_file = session_path / f"{filename_stem}.onsets.npy"
                if test_file.exists():
                    return str(session_path / filename_stem)

        # Strategy 2: Direct session structure (newconditioning)
        # Try the first path part as the session name (for newconditioning structure)
        if len(path_parts) > 0:
            session_name = path_parts[0]  # This could be " 2024_summer in a ghost town_AWelling"

            # Try direct session folder under conditioning dir
            session_path = Path(cond_dir) / session_name / "Audio Files"
            if session_path.exists():
                test_file = session_path / f"{filename_stem}.onsets.npy"
                if test_file.exists():
                    return str(session_path / filename_stem)

            # Try session folder without "Audio Files" subdirectory
            session_path = Path(cond_dir) / session_name
            if session_path.exists():
                test_file = session_path / f"{filename_stem}.onsets.npy"
                if test_file.exists():
                    return str(session_path / filename_stem)

        if session_folder:
            # Try original session folder logic (3rd path part)
            session_path = Path(cond_dir) / session_folder / "Audio Files"
            if session_path.exists():
                test_file = session_path / f"{filename_stem}.onsets.npy"
                if test_file.exists():
                    return str(session_path / filename_stem)

            # Try session folder without "Audio Files" subdirectory
            session_path = Path(cond_dir) / session_folder
            if session_path.exists():
                test_file = session_path / f"{filename_stem}.onsets.npy"
                if test_file.exists():
                    return str(session_path / filename_stem)

        # Strategy 3: newconditioning's flat "Audio Files" structure
        if cond_dir.endswith("newconditioning"):
            session_path = Path(cond_dir) / "Audio Files"
            if session_path.exists():
                test_file = session_path / f"{filename_stem}.onsets.npy"
                if test_file.exists():
                    return str(session_path / filename_stem)

    # If nothing found, return None to indicate missing conditioning
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
    """Create vocal manifest from vocal paths list."""
    vocal_paths_file = "/home/arlo/Data/categorized_instrument_paths_subcats_lists/voice/all.txt"
    output_manifest = "/home/arlo/Data/vocal_training_manifest.json"

    if not Path(vocal_paths_file).exists():
        print(f"❌ Vocal paths file not found: {vocal_paths_file}")
        return

    print(f"📁 Loading vocal paths: {vocal_paths_file}")

    with open(vocal_paths_file, 'r') as f:
        vocal_paths = [line.strip() for line in f.readlines() if line.strip()]

    print(f"📊 Total vocal paths found: {len(vocal_paths)}")

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
                elif "moreconditioning" in cond_path:
                    conditioning_stats["moreconditioning"] += 1
                elif "newconditioning" in cond_path:
                    conditioning_stats["newconditioning"] += 1
                else:
                    conditioning_stats["missing"] += 1
            else:
                conditioning_stats["missing"] += 1

        if (i + 1) % 100 == 0:  # More frequent progress updates
            print(f"  Processed {i + 1}/{len(vocal_paths)} files... (Found: {conditioning_stats['evenmoreconditioning'] + conditioning_stats['moreconditioning'] + conditioning_stats['newconditioning']}, Missing: {conditioning_stats['missing']})")

    print(f"✅ Created {len(vocal_entries)} vocal entries")

    if not vocal_entries:
        print("❌ No valid vocal entries created!")
        return

    # Save vocal manifest
    with open(output_manifest, 'w') as f:
        json.dump(vocal_entries, f, indent=2)

    print(f"🎵 Vocal manifest saved to: {output_manifest}")

    # Show statistics
    sub_group_stats = {}
    for entry in vocal_entries:
        sub_group = entry.get('sub_group', 'unknown')
        sub_group_stats[sub_group] = sub_group_stats.get(sub_group, 0) + 1

    print(f"\n📊 Vocal entry statistics:")
    for sub_group, count in sorted(sub_group_stats.items()):
        print(f"  {sub_group}: {count}")

    print(f"\n📁 Conditioning directory usage:")
    for cond_dir, count in conditioning_stats.items():
        if count > 0:
            print(f"  {cond_dir}: {count} files")

    # Show sample entries
    print(f"\n📋 Sample vocal entries:")
    for i, entry in enumerate(vocal_entries[:3]):
        filename = Path(entry['audio_path']).name
        sub_group = entry.get('sub_group', 'unknown')

        # Determine conditioning directory
        if entry["conditioning_paths"]:
            onsets_path = entry["conditioning_paths"].get("onsets", "")
            if "evenmoreconditioning" in onsets_path:
                cond_dir = "evenmoreconditioning"
            elif "moreconditioning" in onsets_path:
                cond_dir = "moreconditioning"
            elif "newconditioning" in onsets_path:
                cond_dir = "newconditioning"
            else:
                cond_dir = "other"
        else:
            cond_dir = "missing"

        print(f"  {i+1}. {filename} ({sub_group}) -> {cond_dir}")

    print(f"\n✅ Vocal manifest created successfully!")
    print(f"   Total vocal entries: {len(vocal_entries)}")
    print(f"   Output file: {output_manifest}")
    print(f"   Includes vocal conditioning paths for lyrics integration")

if __name__ == "__main__":
    main()