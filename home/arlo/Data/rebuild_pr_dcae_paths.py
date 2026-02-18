#!/usr/bin/env python3
"""
Rebuild piano roll and DCAE paths based on actual directory structure.

Piano rolls: /mnt/msdd/piano_rolls/{session}/filename.pianoroll.npy
DCAE latents: /mnt/msdd/dcae_latentsnew/{date}/{New|Prev}/{session}/filename.dcae.npy
"""

import json
from pathlib import Path
from tqdm import tqdm

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
        session = parts[idx + 1]
    elif 'Prev' in parts:
        idx = parts.index('Prev')
        new_prev = 'Prev'
        session = parts[idx + 1]

    return date, new_prev, session


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


def find_dcae_latent(audio_path):
    """Find DCAE latent path for audio file."""
    audio_path_obj = Path(audio_path)
    filename = audio_path_obj.stem
    date, new_prev, session = extract_session_info(audio_path)

    if not date or not new_prev or not session:
        return None

    # DCAE structure: /mnt/msdd/dcae_latentsnew/{date}/{New|Prev}/{session}/Audio Files/filename.pt
    # Need to include "Audio Files" subfolder and use .pt extension
    dcae_path = Path(f"/mnt/msdd/dcae_latentsnew/{date}/{new_prev}/{session}/Audio Files/{filename}.pt")
    if dcae_path.exists():
        return str(dcae_path)

    return None


def main():
    input_manifest = "./vocal_training_manifest_yamnet_filtered.json"
    output_manifest = "./vocal_training_manifest_yamnet_filtered_PATHS_FIXED.json"

    print("="*80)
    print("Rebuild Piano Roll and DCAE Paths")
    print("="*80)
    print(f"Input:  {input_manifest}")
    print(f"Output: {output_manifest}\n")

    # Load manifest
    with open(input_manifest) as f:
        manifest = json.load(f)

    print(f"Total entries: {len(manifest)}\n")

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

        # Find DCAE latent
        dcae_path = find_dcae_latent(audio_path)
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
