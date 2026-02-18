#!/usr/bin/env python3
"""
Verify that piano roll and DCAE paths exist for all manifest entries.
"""

import json
from pathlib import Path
from tqdm import tqdm

def main():
    manifest_path = "./vocal_training_manifest_yamnet_filtered.json"

    print("="*80)
    print("Verify Piano Roll and DCAE Paths")
    print("="*80)
    print(f"Manifest: {manifest_path}\n")

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Total entries: {len(manifest)}\n")

    stats = {
        'total': len(manifest),
        'pr_exist': 0,
        'pr_missing': 0,
        'dcae_exist': 0,
        'dcae_missing': 0,
        'both_exist': 0,
        'both_missing': 0,
        'pr_only': 0,
        'dcae_only': 0
    }

    missing_pr = []
    missing_dcae = []

    print("Checking paths...")
    for entry in tqdm(manifest, desc="Verifying"):
        pr_path = entry.get("piano_roll_path", "")
        dcae_path = entry.get("dcae_path", "")

        pr_exists = bool(pr_path and Path(pr_path).exists())
        dcae_exists = bool(dcae_path and Path(dcae_path).exists())

        if pr_exists:
            stats['pr_exist'] += 1
        else:
            stats['pr_missing'] += 1
            missing_pr.append({
                'audio_path': entry.get('audio_path', ''),
                'pr_path': pr_path
            })

        if dcae_exists:
            stats['dcae_exist'] += 1
        else:
            stats['dcae_missing'] += 1
            missing_dcae.append({
                'audio_path': entry.get('audio_path', ''),
                'dcae_path': dcae_path
            })

        if pr_exists and dcae_exists:
            stats['both_exist'] += 1
        elif not pr_exists and not dcae_exists:
            stats['both_missing'] += 1
        elif pr_exists and not dcae_exists:
            stats['pr_only'] += 1
        elif dcae_exists and not pr_exists:
            stats['dcae_only'] += 1

    # Report
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total entries: {stats['total']}\n")

    print("Piano Roll Paths:")
    print(f"  Exist:   {stats['pr_exist']:,} ({100*stats['pr_exist']/stats['total']:.1f}%)")
    print(f"  Missing: {stats['pr_missing']:,} ({100*stats['pr_missing']/stats['total']:.1f}%)\n")

    print("DCAE Paths:")
    print(f"  Exist:   {stats['dcae_exist']:,} ({100*stats['dcae_exist']/stats['total']:.1f}%)")
    print(f"  Missing: {stats['dcae_missing']:,} ({100*stats['dcae_missing']/stats['total']:.1f}%)\n")

    print("Combined:")
    print(f"  Both exist:    {stats['both_exist']:,} ({100*stats['both_exist']/stats['total']:.1f}%)")
    print(f"  Both missing:  {stats['both_missing']:,} ({100*stats['both_missing']/stats['total']:.1f}%)")
    print(f"  PR only:       {stats['pr_only']:,} ({100*stats['pr_only']/stats['total']:.1f}%)")
    print(f"  DCAE only:     {stats['dcae_only']:,} ({100*stats['dcae_only']/stats['total']:.1f}%)")

    # Save missing lists if any
    if missing_pr:
        with open("./missing_pr_paths.json", 'w') as f:
            json.dump(missing_pr, f, indent=2)
        print(f"\n📝 Saved missing PR paths to: ./missing_pr_paths.json")

    if missing_dcae:
        with open("./missing_dcae_paths.json", 'w') as f:
            json.dump(missing_dcae, f, indent=2)
        print(f"📝 Saved missing DCAE paths to: ./missing_dcae_paths.json")

    # Verdict
    if stats['both_exist'] == stats['total']:
        print("\n✅ ALL PATHS VERIFIED - Ready for training!")
    elif stats['both_exist'] > 0:
        print(f"\n⚠️  {stats['both_exist']:,} entries have both paths, but {stats['total'] - stats['both_exist']:,} need attention")
    else:
        print("\n❌ NO COMPLETE ENTRIES - Cannot train without both PR and DCAE paths")


if __name__ == "__main__":
    main()
