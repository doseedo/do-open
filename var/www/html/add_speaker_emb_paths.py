#!/usr/bin/env python3
"""
Add speaker embedding paths to the vocal manifest.
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict

INPUT_MANIFEST = Path("/home/arlo/Data/vocal_training_manifest_READY_NOGROUPS.json")
OUTPUT_MANIFEST = Path("/home/arlo/Data/vocal_training_manifest_READY_NOGROUPS_SPKEMB.json")
SPEAKER_EMB_DIR = Path("/mnt/msdd/speaker_embeddings")

print("=" * 80)
print("Add Speaker Embedding Paths to Manifest")
print("=" * 80)
print(f"Input: {INPUT_MANIFEST}")
print(f"Output: {OUTPUT_MANIFEST}")
print(f"Speaker embeddings: {SPEAKER_EMB_DIR}\n")

# Load manifest
with open(INPUT_MANIFEST) as f:
    manifest = json.load(f)

print(f"Total entries: {len(manifest):,}\n")

# Build index of speaker embeddings by hash
print("Building speaker embedding index...")
spk_index = {}
for spk_file in SPEAKER_EMB_DIR.glob("*_spk.pt"):
    # Format: {hash}_{filename}_spk.pt
    parts = spk_file.stem.split("_", 1)
    if len(parts) == 2:
        file_hash = parts[0]
        spk_index[file_hash] = str(spk_file)

print(f"Indexed {len(spk_index):,} speaker embedding files\n")

# Add speaker embedding paths to manifest
stats = {
    'found': 0,
    'not_found': 0
}

for entry in manifest:
    audio_path = entry.get("audio_path", "")

    if not audio_path:
        entry['speaker_emb_path'] = None
        stats['not_found'] += 1
        continue

    # Compute hash of audio path (same as preprocessing script)
    file_hash = hashlib.md5(audio_path.encode()).hexdigest()[:8]

    # Look up speaker embedding
    spk_path = spk_index.get(file_hash)

    if spk_path and Path(spk_path).exists():
        entry['speaker_emb_path'] = spk_path
        stats['found'] += 1
    else:
        entry['speaker_emb_path'] = None
        stats['not_found'] += 1

# Save updated manifest
with open(OUTPUT_MANIFEST, 'w') as f:
    json.dump(manifest, f, indent=2)

print("Results:")
print(f"  Speaker embeddings found: {stats['found']:,} ({stats['found']/len(manifest)*100:.1f}%)")
print(f"  Not found: {stats['not_found']:,} ({stats['not_found']/len(manifest)*100:.1f}%)")
print(f"\n✅ Saved: {OUTPUT_MANIFEST}")
