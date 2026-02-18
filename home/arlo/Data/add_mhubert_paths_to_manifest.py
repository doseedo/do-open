#!/usr/bin/env python3
"""
Add mhubert_features paths to manifest.
Sets path if file exists, null otherwise.
"""
import json
from pathlib import Path
from tqdm import tqdm

MANIFEST_PATH = Path('/home/arlo/Data/vocal_training_manifest_READY_NOGROUPS_SPKEMB.json')
OUTPUT_PATH = Path('/home/arlo/Data/vocal_training_manifest_READY_NOGROUPS_SPKEMB_MHUBERT.json')
MHUBERT_BASE = Path('/mnt/msdd/mhubert_features')

print("Loading manifest...")
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

print(f"Loaded {len(manifest)} entries")

print("\nAdding mhubert_features paths...")

found = 0
missing = 0

for entry in tqdm(manifest, desc="Processing entries"):
    audio_path = Path(entry['audio_path'])
    audio_name = audio_path.stem  # e.g., "LDVOX WET_01"

    # Expected mhubert paths
    mhubert_dir = MHUBERT_BASE / audio_name
    mhubert_features_path = mhubert_dir / f"{audio_name}_mhubert_features.pt"
    mhubert_metadata_path = mhubert_dir / f"{audio_name}_mhubert_metadata.json"

    # Check if files exist
    if mhubert_features_path.exists():
        entry['mhubert_features_path'] = str(mhubert_features_path)
        found += 1
    else:
        entry['mhubert_features_path'] = None
        missing += 1

    # Also add metadata path (optional)
    if mhubert_metadata_path.exists():
        entry['mhubert_metadata_path'] = str(mhubert_metadata_path)
    else:
        entry['mhubert_metadata_path'] = None

print(f"\n{'='*70}")
print(f"Results:")
print(f"  ✅ Found mhubert features: {found:,} ({found/len(manifest)*100:.1f}%)")
print(f"  ❌ Missing (set to null): {missing:,} ({missing/len(manifest)*100:.1f}%)")

# Save updated manifest
print(f"\nSaving to: {OUTPUT_PATH}")
with open(OUTPUT_PATH, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Done! Manifest saved with mhubert paths")

# Show sample entry
print(f"\nSample entry with mhubert paths:")
sample = next((e for e in manifest if e.get('mhubert_features_path')), manifest[0])
print(f"  audio_path: {sample['audio_path']}")
print(f"  mhubert_features_path: {sample['mhubert_features_path']}")
print(f"  mhubert_metadata_path: {sample.get('mhubert_metadata_path')}")
