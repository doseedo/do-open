#!/usr/bin/env python3
"""
Create training-ready manifest with only entries that have:
- DCAE path (exists)
- Piano roll path (exists)
- Vocal conditioning paths (exists)

Set encodec/conditioning to null if they don't exist.
"""

import json
from pathlib import Path
from tqdm import tqdm

# Input/output
INPUT_MANIFEST = Path("/home/arlo/Data/vocal_training_manifest_yamnet_filtered_ALL_FIXED.json")
OUTPUT_MANIFEST = Path("/home/arlo/Data/vocal_training_manifest_READY.json")

# Required vocal conditioning paths
VOCAL_COND_REQUIRED = ["lyrics_data", "lyrics_tensors"]  # syllable_boundaries is optional

def main():
    print("=" * 80)
    print("Create Training-Ready Manifest")
    print("=" * 80)
    print(f"Input: {INPUT_MANIFEST}")
    print(f"Output: {OUTPUT_MANIFEST}\n")

    # Load manifest
    with open(INPUT_MANIFEST) as f:
        manifest = json.load(f)

    print(f"Total entries: {len(manifest):,}\n")

    # Filter and clean
    training_ready = []
    
    stats = {
        'total': len(manifest),
        'has_dcae': 0,
        'has_pr': 0,
        'has_vocal_cond': 0,
        'training_ready': 0,
        'encodec_nulled': 0,
        'conditioning_nulled': 0,
    }

    print("Filtering entries...")
    for entry in tqdm(manifest, desc="Processing"):
        # Check required components
        dcae_path = entry.get('dcae_path', '')
        pr_path = entry.get('piano_roll_path', '')
        vocal_cond = entry.get('vocal_conditioning_paths', {})

        has_dcae = dcae_path and Path(dcae_path).exists()
        has_pr = pr_path and Path(pr_path).exists()
        has_vocal_cond = all(
            vocal_cond.get(key) and Path(vocal_cond[key]).exists() 
            for key in VOCAL_COND_REQUIRED
        )

        if has_dcae:
            stats['has_dcae'] += 1
        if has_pr:
            stats['has_pr'] += 1
        if has_vocal_cond:
            stats['has_vocal_cond'] += 1

        # Only include if has all three required
        if has_dcae and has_pr and has_vocal_cond:
            stats['training_ready'] += 1
            
            # Clean entry
            cleaned = entry.copy()
            
            # Check encodec - set to null if doesn't exist
            encodec_path = cleaned.get('encodec_path', '')
            if not encodec_path or not Path(encodec_path).exists():
                cleaned['encodec_path'] = None
                stats['encodec_nulled'] += 1
            
            # Check conditioning - set individual types to null if don't exist
            cond_paths = cleaned.get('conditioning_paths', {})
            if cond_paths:
                has_any_missing = False
                for cond_type in ['amp', 'rbend', 'rframe', 'onsets', 'f0', 'f0_masked']:
                    ct_path = cond_paths.get(cond_type, '')
                    if not ct_path or not Path(ct_path).exists():
                        cond_paths[cond_type] = None
                        has_any_missing = True
                
                if has_any_missing:
                    stats['conditioning_nulled'] += 1
                    
                cleaned['conditioning_paths'] = cond_paths
            else:
                # No conditioning at all - set to empty dict with nulls
                cleaned['conditioning_paths'] = {
                    'amp': None,
                    'rbend': None,
                    'rframe': None,
                    'onsets': None,
                    'f0': None,
                    'f0_masked': None
                }
                stats['conditioning_nulled'] += 1
            
            training_ready.append(cleaned)

    # Save
    print(f"\nSaving to {OUTPUT_MANIFEST}...")
    with open(OUTPUT_MANIFEST, 'w') as f:
        json.dump(training_ready, f, indent=2)

    # Report
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Original entries: {stats['total']:,}")
    print(f"\nHas DCAE: {stats['has_dcae']:,} ({stats['has_dcae']/stats['total']*100:.1f}%)")
    print(f"Has Piano Roll: {stats['has_pr']:,} ({stats['has_pr']/stats['total']*100:.1f}%)")
    print(f"Has Vocal Conditioning: {stats['has_vocal_cond']:,} ({stats['has_vocal_cond']/stats['total']*100:.1f}%)")
    print(f"\n✅ Training Ready (all 3): {stats['training_ready']:,} ({stats['training_ready']/stats['total']*100:.1f}%)")
    print(f"\nEntries with nulled encodec: {stats['encodec_nulled']:,}")
    print(f"Entries with nulled conditioning: {stats['conditioning_nulled']:,}")
    print(f"\n✅ Saved: {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    main()
