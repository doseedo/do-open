#!/usr/bin/env python3
"""
Verify the alternates manifest structure is correct.
"""

import json
from pathlib import Path

def verify_alternates_manifest():
    manifest_path = "/home/arlo/Data/vocal_training_manifest_with_alternates.json"

    print(f"Verifying manifest: {manifest_path}")
    print("=" * 70)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"✅ Total entries: {len(manifest)}")

    # Count entries with alternates
    with_alternates = 0
    total_alternates = 0

    for entry in manifest:
        alts = entry.get('alternate_takes', [])
        if alts:
            with_alternates += 1
            total_alternates += len(alts)

    print(f"✅ Entries with alternate takes: {with_alternates}")
    print(f"✅ Total alternate take references: {total_alternates}")
    print(f"✅ Average alternates per entry with alternates: {total_alternates/with_alternates:.1f}")

    # Show example entries with alternates
    print("\n" + "=" * 70)
    print("Example entries with alternate takes:")
    print("=" * 70)

    count = 0
    for idx, entry in enumerate(manifest):
        alts = entry.get('alternate_takes', [])
        if alts and count < 3:
            print(f"\nEntry {idx}:")
            print(f"  Audio: {entry['audio_path']}")
            print(f"  Alternates ({len(alts)}):")
            for alt in alts[:3]:  # Show first 3 alternates
                print(f"    - Index {alt['manifest_index']}: {alt['audio_path']}")
            if len(alts) > 3:
                print(f"    ... and {len(alts) - 3} more")
            count += 1

    # Verify structure
    print("\n" + "=" * 70)
    print("Verifying data structure:")
    print("=" * 70)

    errors = []

    for idx, entry in enumerate(manifest[:1000]):  # Check first 1000
        # Check all entries have alternate_takes field
        if 'alternate_takes' not in entry:
            errors.append(f"Entry {idx}: Missing 'alternate_takes' field")

        # Check alternate_takes structure
        alts = entry.get('alternate_takes', [])
        if alts:
            for alt in alts:
                if 'manifest_index' not in alt:
                    errors.append(f"Entry {idx}: Alternate missing 'manifest_index'")
                if 'audio_path' not in alt:
                    errors.append(f"Entry {idx}: Alternate missing 'audio_path'")
                if 'number_extension' not in alt:
                    errors.append(f"Entry {idx}: Alternate missing 'number_extension'")

                # Verify manifest_index is valid
                alt_idx = alt.get('manifest_index')
                if alt_idx is not None:
                    if alt_idx < 0 or alt_idx >= len(manifest):
                        errors.append(f"Entry {idx}: Invalid manifest_index {alt_idx}")
                    elif alt_idx == idx:
                        errors.append(f"Entry {idx}: Alternate references itself!")

    if errors:
        print(f"❌ Found {len(errors)} errors:")
        for err in errors[:10]:  # Show first 10
            print(f"  - {err}")
    else:
        print("✅ All structure checks passed!")

    print("\n" + "=" * 70)
    print("✅ Manifest verification complete!")

if __name__ == "__main__":
    verify_alternates_manifest()
