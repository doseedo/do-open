#!/usr/bin/env python3
"""
Create Master Manifest - Single Source of Truth

Uses unified_manifest.json (339K filename-derived labels) as base,
then layers on corrections and classifier predictions.

Priority (highest to lowest):
  1. Manual corrections (corrections.json)
  2. Subgroup classifier predictions
  3. Instrument classifier predictions (subgroups only, never overrides group)
  4. Filename re-classification (improved patterns from filename_verify_manifest)
  5. Unified manifest (filename-derived labels — the base)

Schema:
{
  "path": {
    "group": "guitar",
    "subgroup": "acoustic",
    "is_mix": false,
    "roomy": false,
    "bleed_instruments": [],
    "regions": [],
    "source": "manual|classifier|filename",
    "confidence": 1.0,
    "last_modified": "2026-02-16T00:00:00",
    "original_group": "guitar",
    "original_subgroup": "undefined",
    "flags": []
  }
}
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import orjson

# Import canonical filename classifier and group merges
sys.path.insert(0, str(Path(__file__).parent))
from filename_verify_manifest import classify_filename, GROUP_MERGES, disqualify_group

# Paths
MANIFESTS_DIR = Path("/home/arlo/gcs-bucket/Manifests")
DATA_DIR = Path("/home/arlo/Data")

UNIFIED_MANIFEST_PATH = MANIFESTS_DIR / "unified_manifest.json"
CORRECTIONS_PATH = MANIFESTS_DIR / "corrections.json"
INSTRUMENT_PREDICTIONS = DATA_DIR / "latent_classifier/predictions.json"
OUTPUT_PATH = MANIFESTS_DIR / "master_manifest.json"

# Subgroup prediction manifests
SUBGROUP_MANIFESTS = {
    "bass": MANIFESTS_DIR / "bass_subgroup_predictions_manifest.json",
    "guitar": MANIFESTS_DIR / "guitar_subgroup_predictions_manifest.json",
    "strings": MANIFESTS_DIR / "strings_subgroup_predictions_manifest.json",
    "brass": MANIFESTS_DIR / "brass_subgroup_predictions_manifest.json",
    "piano": MANIFESTS_DIR / "piano_subgroup_predictions_manifest.json",
    "winds": MANIFESTS_DIR / "winds_subgroup_predictions_manifest.json",
}


def load_orjson(path):
    if not path.exists():
        return {}
    with open(path, 'rb') as f:
        return orjson.loads(f.read())


def main():
    stats = defaultdict(int)
    now = datetime.now().isoformat()

    print("Creating Master Manifest...\n")

    # 1. Load unified manifest as base (filename-derived, clean labels)
    print("Loading unified manifest (base)...")
    um_data = load_orjson(UNIFIED_MANIFEST_PATH)
    um_entries = um_data.get('entries', um_data) if isinstance(um_data, dict) else um_data
    if isinstance(um_entries, list):
        um_entries = {e.get('audio_path', e.get('path', '')): e
                      for e in um_entries if isinstance(e, dict)}
    print(f"  Loaded {len(um_entries):,} entries")

    # 2. Load corrections
    print("Loading corrections...")
    corrections = load_orjson(CORRECTIONS_PATH)
    print(f"  Loaded {len(corrections):,} corrections")

    # 3. Load instrument predictions (for subgroup enrichment only)
    print("Loading instrument predictions...")
    pred_data = load_orjson(INSTRUMENT_PREDICTIONS)
    predictions = {p['path']: p for p in pred_data.get('predictions', [])
                   } if isinstance(pred_data, dict) else {}
    print(f"  Loaded {len(predictions):,} predictions")

    # 4. Load subgroup predictions
    print("Loading subgroup predictions...")
    subgroup_preds = {}
    for group, manifest_path in SUBGROUP_MANIFESTS.items():
        if manifest_path.exists():
            data = load_orjson(manifest_path)
            items = data if isinstance(data, list) else data.get(
                'predictions', data.get('items', []))
            if isinstance(items, list):
                for item in items:
                    path = item.get('path', item.get('audio_path', ''))
                    if path:
                        subgroup_preds[path] = {
                            'subgroup': item.get('subgroup', item.get(
                                'predicted_subgroup', 'undefined')),
                            'confidence': item.get('confidence', 0),
                            'group': group
                        }
            elif isinstance(items, dict):
                for path, item in items.items():
                    if isinstance(item, dict):
                        subgroup_preds[path] = {
                            'subgroup': item.get('subgroup', 'undefined'),
                            'confidence': item.get('confidence', 0),
                            'group': group
                        }
    print(f"  Loaded {len(subgroup_preds):,} subgroup predictions")

    # Build master manifest from unified base
    print("\nBuilding master manifest...")
    print("  Applying GROUP_MERGES and filename re-classification...")
    master = {}

    for path, entry in um_entries.items():
        group = entry.get('group', 'undefined')
        subgroup = entry.get('subgroup', 'undefined')

        # Apply GROUP_MERGES to normalize non-canonical group names
        original_group = group
        if group in GROUP_MERGES:
            group = GROUP_MERGES[group]
            stats['group_merged'] += 1

        # Re-classify using improved filename patterns
        fn_group, fn_keyword = classify_filename(path)

        # Apply fn_group merge too
        if fn_group in GROUP_MERGES:
            fn_group = GROUP_MERGES[fn_group]

        # If unified manifest group differs from filename classifier,
        # trust the filename classifier (it has better patterns)
        if fn_group != 'undefined' and fn_group != group:
            stats['filename_corrected'] += 1
            group = fn_group
        elif fn_group == 'undefined' and group != 'undefined':
            # Filename classifier can't identify, but check if existing
            # label is contradicted by disqualifier tokens
            if disqualify_group(path, group):
                stats['disqualified'] += 1
                group = 'undefined'

        master_entry = {
            'group': group,
            'subgroup': subgroup,
            'original_group': original_group,
            'original_subgroup': subgroup,
            'is_mix': False,
            'roomy': False,
            'bleed_instruments': [],
            'regions': [],
            'source': 'filename',
            'confidence': 1.0,
            'last_modified': now,
            'flags': [],
            'filename': os.path.basename(path)
        }

        # Track if group was changed
        if group != original_group:
            master_entry['flags'].append(f'was:{original_group}')

        # Check for mix/room in path
        path_lower = path.lower()
        if '/mix/' in path_lower or '/room/' in path_lower or 'room mic' in path_lower:
            master_entry['is_mix'] = True
            master_entry['flags'].append('mix_file')
            stats['mix_from_path'] += 1

        # Enrich subgroup from instrument predictions (never override group)
        if subgroup in ('undefined', '', None) and path in predictions:
            pred = predictions[path]
            if pred.get('confidence', 0) > 0.5:
                master_entry['subgroup'] = pred.get(
                    'predicted_group', 'undefined')
                master_entry['subgroup_confidence'] = pred.get(
                    'confidence', 0)
                master_entry['subgroup_source'] = 'classifier'
                stats['subgroup_from_classifier'] += 1

            # Check multi-instrument detection
            if pred.get('is_multi'):
                master_entry['is_mix'] = True
                master_entry['multi_probability'] = pred.get(
                    'multi_probability', 0)
                master_entry['flags'].append('multi_detected')
                stats['multi_detected'] += 1

        # Apply subgroup predictions
        if path in subgroup_preds:
            sg_pred = subgroup_preds[path]
            if sg_pred['confidence'] > 0.7:
                master_entry['subgroup'] = sg_pred['subgroup']
                master_entry['subgroup_source'] = 'subgroup_classifier'
                master_entry['subgroup_confidence'] = sg_pred['confidence']
                stats['subgroup_from_subgroup_classifier'] += 1

        # Apply manual corrections (highest priority)
        if path in corrections:
            corr = corrections[path]
            if corr.get('group'):
                corr_group = corr['group'].replace('_roomy', '')
                # Apply GROUP_MERGES to manual corrections too
                corr_group = GROUP_MERGES.get(corr_group, corr_group)
                master_entry['group'] = corr_group
                master_entry['source'] = 'manual'
                stats['manual_corrections'] += 1
            if corr.get('subgroup'):
                master_entry['subgroup'] = corr['subgroup']
            if corr.get('roomy') or '_roomy' in corr.get('group', ''):
                master_entry['roomy'] = True
            if corr.get('has_bleed'):
                master_entry['bleed_instruments'] = corr.get(
                    'bleed_instruments', [])
            if corr.get('multi_label') and corr.get('regions'):
                master_entry['regions'] = corr['regions']
                master_entry['flags'].append('has_temporal')
                stats['has_temporal'] += 1
            master_entry['last_modified'] = corr.get('corrected_at', now)

        master[path] = master_entry
        stats['total'] += 1

    # Calculate distributions
    group_counts = defaultdict(int)
    subgroup_counts = defaultdict(int)
    for entry in master.values():
        group_counts[entry['group']] += 1
        if entry['subgroup'] != 'undefined':
            subgroup_counts[entry['subgroup']] += 1

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    output = {
        'created_at': now,
        'version': '2.1',
        'base': 'unified_manifest.json',
        'total_entries': len(master),
        'stats': dict(stats),
        'group_distribution': dict(group_counts),
        'subgroup_distribution': dict(subgroup_counts),
        'entries': master
    }

    with open(OUTPUT_PATH, 'wb') as f:
        f.write(orjson.dumps(output, option=orjson.OPT_INDENT_2))

    # Summary
    print("\n" + "=" * 50)
    print("MASTER MANIFEST CREATED")
    print("=" * 50)
    print(f"\nTotal entries: {stats['total']:,}")
    print(f"\nSources:")
    print(f"  Filename-derived (base): {stats['total'] - stats['manual_corrections']:,}")
    print(f"  Manual corrections: {stats['manual_corrections']:,}")
    print(f"\nFilename fixes:")
    print(f"  Groups merged (GROUP_MERGES): {stats['group_merged']:,}")
    print(f"  Corrected by filename classifier: {stats['filename_corrected']:,}")
    print(f"  Disqualified (bad label → undefined): {stats['disqualified']:,}")
    print(f"\nSubgroups:")
    print(f"  From classifier: {stats['subgroup_from_classifier']:,}")
    print(f"  From subgroup classifier: {stats['subgroup_from_subgroup_classifier']:,}")
    print(f"\nFlags:")
    print(f"  Mix files: {stats['mix_from_path']:,}")
    print(f"  Multi-detected: {stats['multi_detected']:,}")
    print(f"  Has temporal: {stats['has_temporal']:,}")
    print(f"\nTop groups:")
    for group, count in sorted(group_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {group}: {count:,}")
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
