#!/usr/bin/env python3
"""
Remove group/subgroup from manifest - everything is vocals.
"""

import json
from pathlib import Path

INPUT_MANIFEST = Path("/home/arlo/Data/vocal_training_manifest_READY.json")
OUTPUT_MANIFEST = Path("/home/arlo/Data/vocal_training_manifest_READY_NOGROUPS.json")

print("=" * 80)
print("Remove Groups from Vocal Manifest")
print("=" * 80)
print(f"Input: {INPUT_MANIFEST}")
print(f"Output: {OUTPUT_MANIFEST}\n")

with open(INPUT_MANIFEST) as f:
    manifest = json.load(f)

print(f"Total entries: {len(manifest):,}\n")

# Remove group/subgroup from each entry
for entry in manifest:
    entry.pop('group', None)
    entry.pop('sub_group', None)

# Save
with open(OUTPUT_MANIFEST, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Saved: {OUTPUT_MANIFEST}")
print(f"   Removed 'group' and 'sub_group' fields from all entries")
