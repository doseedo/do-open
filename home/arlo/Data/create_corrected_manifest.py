#!/usr/bin/env python3
"""
Create corrected manifest based on merged_manifest_v2:
1. Start with merged_manifest_v2 (has all previous corrections/flags)
2. Fix synth→voice from flagged (484 files)
3. Fix obvious voice filenames in synth group
4. Remove any voice→synth corrections (classifier errors)
"""

import json
import re
from datetime import datetime
from collections import defaultdict

BASE_PATH = "/home/arlo/gcs-bucket/Manifests/merged_manifest_v2.json"
FLAGGED_PATH = "/home/arlo/Data/latent_classifier/flagged_paths.txt"
OUTPUT_PATH = "/home/arlo/gcs-bucket/Manifests/labels_corrected_v1.json"

stats = defaultdict(int)

VOICE_PATTERN = re.compile(
    r'(vox|vocal|voice|bgv|bvox|choir[^a-z]|tenor|soprano|alto|baritone|'
    r'singer|sung|adlib|harmonies|harmony|lead\s*vox|vox\s*lead)', re.I
)

print("Loading merged_manifest_v2...", flush=True)
with open(BASE_PATH) as f:
    base = json.load(f)

entries = base.get('entries', {})
print(f"  Entries: {len(entries)}")
print(f"  Previous stats: {base.get('stats', {})}")

# Load flagged
print("Loading flagged...", flush=True)
flagged = {}
with open(FLAGGED_PATH) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            path, orig, flags, pred_conf = parts[0], parts[1], parts[2], parts[3]
            pred, conf = pred_conf.rsplit(':', 1)
            flagged[path] = {'original': orig, 'flags': flags, 'predicted': pred, 'confidence': float(conf)}
print(f"  Flagged: {len(flagged)}")

print("\nApplying synth→voice corrections...", flush=True)

for path, entry in entries.items():
    orig_group = entry.get('original_group', entry.get('group', ''))
    current_group = entry.get('group', '')
    fname = path.split('/')[-1]

    # 1. Revert any voice→synth changes (these are errors)
    if orig_group == 'voice' and current_group == 'synth':
        entry['group'] = 'voice'
        entry['correction_source'] = 'reverted_voice_to_synth'
        stats['reverted_voice_to_synth'] += 1

    # 2. Fix synth→voice from flagged
    if orig_group == 'synth' and path in flagged:
        f = flagged[path]
        if f['predicted'] == 'voice' and 'disagreement' in f['flags'] and f['confidence'] > 0.7:
            entry['group'] = 'voice'
            entry['correction_source'] = 'classifier_synth_to_voice'
            entry['correction_confidence'] = round(f['confidence'], 3)
            stats['synth_to_voice_flagged'] += 1

    # 3. Fix obvious voice filenames still in synth
    if entry.get('group') == 'synth' and VOICE_PATTERN.search(fname):
        entry['group'] = 'voice'
        entry['correction_source'] = 'filename_voice_in_synth'
        stats['synth_to_voice_filename'] += 1

# Write output
print("\nWriting corrected manifest...", flush=True)
output = {
    'created_at': datetime.now().isoformat(),
    'base_manifest': 'merged_manifest_v2.json',
    'stats': dict(stats),
    'total_entries': len(entries),
    'entries': entries
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(output, f)

print(f"\nDone! Saved to {OUTPUT_PATH}")
print(f"\nCorrection stats:")
for k, v in sorted(stats.items()):
    print(f"  {k}: {v}")
