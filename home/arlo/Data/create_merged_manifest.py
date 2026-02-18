#!/usr/bin/env python3
"""
Create merged manifest - v2
- Keep original labels
- Fill undefined subgroups with classifier predictions
- Apply 972 corrections
- Apply 49,397 flagged disagreements (change group to predicted)
- Add mix=True for 5,329 is_multi=True files
- Add mix=True for files with /mix/ or /room/ in path
"""

import json
from datetime import datetime
from collections import defaultdict
import sys

# Paths
LABELS_PATH = "/home/arlo/gcs-bucket/Manifests/labels.json"
PREDICTIONS_PATH = "/home/arlo/Data/latent_classifier/predictions.json"
CORRECTIONS_PATH = "/home/arlo/gcs-bucket/Manifests/corrections.json"
FLAGGED_PATH = "/home/arlo/Data/latent_classifier/flagged_paths.txt"
OUTPUT_PATH = "/home/arlo/gcs-bucket/Manifests/merged_manifest_v2.json"

stats = defaultdict(int)

print("Loading data sources...", flush=True)

# Load corrections (small)
print("  Loading corrections...", flush=True)
with open(CORRECTIONS_PATH) as f:
    corrections = json.load(f)
print(f"  Corrections: {len(corrections)}", flush=True)

# Load flagged paths with predictions
# Format: path\toriginal_group\tflag_type\tpredicted:confidence
print("  Loading flagged paths...", flush=True)
flagged_data = {}
with open(FLAGGED_PATH) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            path = parts[0]
            original_group = parts[1]
            flag_type = parts[2]
            predicted_conf = parts[3]

            # Parse predicted:confidence
            if ':' in predicted_conf:
                pred_group, conf_str = predicted_conf.rsplit(':', 1)
                try:
                    conf = float(conf_str)
                    flagged_data[path] = {
                        'original': original_group,
                        'flags': flag_type,
                        'predicted': pred_group,
                        'confidence': conf
                    }
                except ValueError:
                    pass
print(f"  Flagged with predictions: {len(flagged_data)}", flush=True)

# Load predictions
print("  Loading predictions...", flush=True)
with open(PREDICTIONS_PATH) as f:
    pred_data = json.load(f)
predictions_list = pred_data.get('predictions', [])
print(f"  Predictions: {len(predictions_list)}", flush=True)

# Index predictions by path
predictions = {}
multi_paths = set()
for p in predictions_list:
    path = p.get('path')
    if path:
        predictions[path] = p
        if p.get('is_multi'):
            multi_paths.add(path)

print(f"  Multi-detected (is_multi=True): {len(multi_paths)}", flush=True)

# Now process labels and write output
print("\nProcessing labels and writing output...", flush=True)

with open(OUTPUT_PATH, 'w') as out:
    # Write header
    out.write('{\n')
    out.write(f'  "created_at": "{datetime.now().isoformat()}",\n')
    out.write('  "entries": {\n')

    first_entry = True
    count = 0

    with open(LABELS_PATH) as f:
        labels = json.load(f)
        total = len(labels)
        print(f"  Total labels: {total}", flush=True)

        for path, entry in labels.items():
            count += 1
            if count % 20000 == 0:
                print(f"  Processed {count}/{total}...", flush=True)

            # Start with original entry
            if isinstance(entry, dict):
                new_entry = dict(entry)
            else:
                new_entry = {'group': entry}

            original_group = new_entry.get('group', '')
            original_subgroup = new_entry.get('subgroup', 'undefined')
            new_entry['original_group'] = original_group
            new_entry['original_subgroup'] = original_subgroup

            # 2. Fill undefined subgroups with predictions
            if original_subgroup in ('undefined', '', None) and path in predictions:
                pred = predictions[path]
                pred_class = pred.get('predicted_group')
                confidence = pred.get('confidence', 0)
                if pred_class and confidence > 0.5:
                    new_entry['subgroup'] = pred_class
                    new_entry['subgroup_source'] = 'classifier'
                    new_entry['subgroup_confidence'] = round(confidence, 3)
                    stats['undefined_filled'] += 1

            # 4. Apply flagged disagreements (only if disagreement flag and high confidence)
            if path in flagged_data:
                flagged = flagged_data[path]
                if 'disagreement' in flagged['flags'] and flagged['confidence'] > 0.7:
                    new_entry['group'] = flagged['predicted']
                    new_entry['group_source'] = 'classifier_correction'
                    new_entry['flagged_original'] = flagged['original']
                    new_entry['flagged_confidence'] = round(flagged['confidence'], 3)
                    stats['flagged_applied'] += 1

            # 5. Mark mix files from is_multi detection
            if path in multi_paths:
                new_entry['mix'] = True
                new_entry['mix_source'] = 'multi_classifier'
                if path in predictions:
                    new_entry['multi_probability'] = round(predictions[path].get('multi_probability', 0), 3)
                stats['mix_multi'] += 1

            # 6. Check filename for mix/room
            path_lower = path.lower()
            if '/mix/' in path_lower or '/room/' in path_lower or 'room mic' in path_lower:
                if not new_entry.get('mix'):
                    new_entry['mix'] = True
                    new_entry['mix_source'] = 'filename'
                    stats['mix_from_filename'] += 1
                else:
                    # Already marked, note it was also in filename
                    new_entry['mix_also_filename'] = True

            # 3. Apply corrections last (override everything)
            if path in corrections:
                correction = corrections[path]
                if isinstance(correction, dict):
                    if correction.get('group'):
                        new_entry['group'] = correction['group']
                        new_entry['group_source'] = 'manual_correction'
                        stats['corrections_group'] += 1
                    if correction.get('subgroup'):
                        new_entry['subgroup'] = correction['subgroup']
                        new_entry['subgroup_source'] = 'manual_correction'
                        stats['corrections_subgroup'] += 1
                stats['corrections_applied'] += 1

            # Write entry
            if not first_entry:
                out.write(',\n')
            first_entry = False

            escaped_path = json.dumps(path)
            out.write(f'    {escaped_path}: {json.dumps(new_entry)}')

    # Write footer with stats
    out.write('\n  },\n')
    out.write(f'  "stats": {json.dumps(dict(stats))},\n')
    out.write(f'  "total_entries": {count}\n')
    out.write('}\n')

print(f"\nDone! Saved to {OUTPUT_PATH}", flush=True)
print(f"\nStats:", flush=True)
for k, v in sorted(stats.items()):
    print(f"  {k}: {v}", flush=True)
print(f"\nTotal entries: {count}", flush=True)

# Summary
print(f"\n=== Summary ===")
print(f"Original labels: {total}")
print(f"Subgroups filled from classifier: {stats['undefined_filled']}")
print(f"Corrections applied: {stats['corrections_applied']}")
print(f"Flagged disagreements applied: {stats['flagged_applied']}")
print(f"Mix from multi-classifier: {stats['mix_multi']}")
print(f"Mix from filename: {stats['mix_from_filename']}")
