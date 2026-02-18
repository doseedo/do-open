#!/usr/bin/env python3
"""
Create a vocal-only manifest from the full training manifest.
Identifies vocal entries and creates a new manifest in the same format.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import re

def is_vocal_track(audio_path, group=None, sub_group=None):
    """
    Determine if a track is vocal-related based on path patterns and metadata.
    """
    filename = Path(audio_path).name.upper()
    path_parts = Path(audio_path).parts

    # Common vocal patterns in filenames
    vocal_patterns = [
        r'\bVOX\b',           # VOX
        r'\bVOCAL\b',         # VOCAL
        r'\bVOICE\b',         # VOICE
        r'\bLEAD\s*VOX\b',    # LEAD VOX
        r'\bLEAD\s*VOCAL\b',  # LEAD VOCAL
        r'\bBG\s*VOX\b',      # BG VOX
        r'\bBACKING\s*VOX\b', # BACKING VOX
        r'\bCHOIR\b',         # CHOIR
        r'\bCHORUS\b',        # CHORUS (when not referring to song structure)
    ]

    # Check filename patterns
    for pattern in vocal_patterns:
        if re.search(pattern, filename):
            return True

    # Check if path contains vocal session indicators
    path_str = str(audio_path).upper()
    vocal_session_patterns = [
        'VOX',
        'VOCAL',
        'VOICE',
        'LEAD VOX',
        'BACKING',
        'CHOIR'
    ]

    for pattern in vocal_session_patterns:
        if pattern in path_str:
            return True

    # Check group/sub_group metadata if available
    if group and 'vocal' in group.lower():
        return True
    if sub_group and 'vocal' in sub_group.lower():
        return True

    return False

def analyze_manifest_for_vocals(manifest_path):
    """
    Analyze the full manifest to identify vocal patterns and statistics.
    """
    print(f"📊 Analyzing manifest: {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"📁 Total entries in manifest: {len(manifest)}")

    # Analyze patterns
    vocal_entries = []
    group_stats = defaultdict(int)
    sub_group_stats = defaultdict(int)
    filename_patterns = defaultdict(int)

    for entry in manifest:
        audio_path = entry.get('audio_path', '')
        group = entry.get('group', '')
        sub_group = entry.get('sub_group', '')

        group_stats[group] += 1
        sub_group_stats[sub_group] += 1

        # Extract instrument/track name from path
        filename = Path(audio_path).stem
        # Get the part before the first dot or underscore (likely instrument name)
        instrument_name = re.split(r'[._]', filename)[0].upper()
        filename_patterns[instrument_name] += 1

        if is_vocal_track(audio_path, group, sub_group):
            vocal_entries.append(entry)

    print(f"\n🎵 Found {len(vocal_entries)} potential vocal entries")

    print(f"\n📊 Groups in manifest:")
    for group, count in sorted(group_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {group}: {count}")

    print(f"\n📊 Top instrument patterns:")
    top_patterns = sorted(filename_patterns.items(), key=lambda x: x[1], reverse=True)[:20]
    for pattern, count in top_patterns:
        if any(vocal_term in pattern for vocal_term in ['VOX', 'VOCAL', 'VOICE', 'LEAD']):
            print(f"  🎤 {pattern}: {count}")
        else:
            print(f"     {pattern}: {count}")

    return vocal_entries

def validate_conditioning_paths(entry):
    """
    Validate that all required conditioning paths exist for an entry.
    """
    required_conditioning = ['onsets', 'rframe', 'rbend', 'amp', 'f0', 'f0_masked']
    missing_paths = []

    conditioning_paths = entry.get('conditioning_paths', {})

    for cond_type in required_conditioning:
        path = conditioning_paths.get(cond_type)
        if not path or not Path(path).exists():
            missing_paths.append(cond_type)

    # Also check main paths
    main_paths = ['audio_path', 'piano_roll_path', 'latent_path', 'encodec_path']
    for path_type in main_paths:
        path = entry.get(path_type)
        if not path or not Path(path).exists():
            missing_paths.append(path_type)

    return missing_paths

def create_vocal_manifest(input_manifest_path, output_manifest_path):
    """
    Create a vocal-only manifest from the full training manifest.
    """
    print(f"🎤 Creating vocal manifest from {input_manifest_path}")

    # Analyze and find vocal entries
    vocal_entries = analyze_manifest_for_vocals(input_manifest_path)

    if not vocal_entries:
        print("❌ No vocal entries found in the manifest!")
        return

    print(f"\n🔍 Validating paths for {len(vocal_entries)} vocal entries...")

    valid_entries = []
    invalid_entries = []

    for i, entry in enumerate(vocal_entries):
        missing_paths = validate_conditioning_paths(entry)

        if not missing_paths:
            valid_entries.append(entry)
        else:
            invalid_entries.append({
                'entry': entry,
                'missing_paths': missing_paths
            })

        if (i + 1) % 100 == 0:
            print(f"  Validated {i + 1}/{len(vocal_entries)} entries...")

    print(f"\n✅ Valid vocal entries: {len(valid_entries)}")
    print(f"❌ Invalid vocal entries: {len(invalid_entries)}")

    if invalid_entries:
        print(f"\n⚠️  Sample invalid entries:")
        for i, invalid in enumerate(invalid_entries[:5]):
            entry = invalid['entry']
            missing = invalid['missing_paths']
            filename = Path(entry['audio_path']).name
            print(f"  {i+1}. {filename}: missing {', '.join(missing)}")

    if valid_entries:
        # Save vocal manifest
        output_path = Path(output_manifest_path)
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(valid_entries, f, indent=2)

        print(f"\n🎵 Vocal manifest saved to: {output_path}")
        print(f"📊 Contains {len(valid_entries)} vocal entries")

        # Print sample entries
        print(f"\n📋 Sample vocal entries:")
        for i, entry in enumerate(valid_entries[:5]):
            filename = Path(entry['audio_path']).name
            group = entry.get('group', 'unknown')
            sub_group = entry.get('sub_group', 'unknown')
            print(f"  {i+1}. {filename} ({group}/{sub_group})")

    else:
        print("❌ No valid vocal entries found!")

def main():
    """Main function"""
    input_manifest = "/home/arlo/Data/final_training_manifest_final.json"
    output_manifest = "/home/arlo/Data/vocal_training_manifest.json"

    if not Path(input_manifest).exists():
        print(f"❌ Input manifest not found: {input_manifest}")
        return

    create_vocal_manifest(input_manifest, output_manifest)

if __name__ == "__main__":
    main()