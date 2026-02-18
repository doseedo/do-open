#!/usr/bin/env python3
"""
Prepare training manifest with corrections and validated predictions.

This script:
1. Starts with combined_manifest.json
2. Applies all corrections
3. For undefined files in validated range: uses predicted label as GT
4. Filters out: silent, junk, mix files, etc.
5. Outputs a clean training manifest
"""

import json
from pathlib import Path
from collections import Counter
from datetime import datetime

# Paths
CORRECTIONS_FILE = Path("/home/arlo/gcs-bucket/Manifests/corrections.json")
MANIFEST_FILE = Path("/home/arlo/gcs-bucket/Manifests/combined_manifest.json")
PREDICTIONS_FILE = Path("/home/arlo/Data/latent_classifier/predictions.json")
OUTPUT_FILE = Path("/home/arlo/gcs-bucket/Manifests/training_manifest.json")

# Groups to exclude from training
EXCLUDE_GROUPS = {'silent', 'junk', 'undefined', 'room', 'fx', 'click', 'ensemble', 'full-track'}

# Validated ranges by confidence tier (position in sorted list)
VALIDATED_RANGES = {
    'low': 32010,      # positions 0-32010 in low confidence list
    'medium': 11212,   # positions 0-11212 in medium confidence list
    'high': 42164,     # positions 0-42164 in high confidence list
}

def main():
    print("Loading data...")

    with open(CORRECTIONS_FILE) as f:
        corrections = json.load(f)

    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    with open(PREDICTIONS_FILE) as f:
        predictions = json.load(f)

    pred_list = predictions.get('predictions', [])

    # Split predictions by confidence tier
    low = [(i, p) for i, p in enumerate(pred_list) if p.get('confidence', 0) < 0.65]
    medium = [(i, p) for i, p in enumerate(pred_list) if 0.65 <= p.get('confidence', 0) < 0.85]
    high = [(i, p) for i, p in enumerate(pred_list) if p.get('confidence', 0) >= 0.85]

    # Sort each tier by original position (this is how UI shows them)
    # Actually they're already in order from predictions.json

    # Build set of validated paths with their predicted labels
    validated_predictions = {}

    for tier_name, tier_list, max_pos in [
        ('low', low, VALIDATED_RANGES['low']),
        ('medium', medium, VALIDATED_RANGES['medium']),
        ('high', high, VALIDATED_RANGES['high'])
    ]:
        # Re-index within tier
        for tier_idx, (orig_idx, pred) in enumerate(tier_list):
            if tier_idx <= max_pos:
                path = pred['path']
                # Only add if NOT already corrected (corrections take precedence)
                if path not in corrections:
                    validated_predictions[path] = pred['predicted_group']

    print(f"Validated predictions (skipped = confirmed): {len(validated_predictions)}")

    # Build training manifest
    training_data = {}
    stats = {
        'from_manifest': 0,
        'from_corrections': 0,
        'from_validated_predictions': 0,
        'excluded_mix': 0,
        'excluded_group': 0,
        'excluded_silent': 0,
    }

    # Process all manifest entries
    for path, entry in manifest.items():
        if not isinstance(entry, dict):
            continue

        # Determine the label to use
        if path in corrections:
            # Use correction
            group = corrections[path].get('group', '')
            source = 'correction'
        elif path in validated_predictions:
            # Use validated prediction (user skipped = confirmed)
            group = validated_predictions[path]
            source = 'validated_prediction'
        else:
            # Use original manifest label
            group = entry.get('group', 'undefined')
            source = 'manifest'

        # Skip excluded groups
        if group in EXCLUDE_GROUPS:
            if group == 'silent':
                stats['excluded_silent'] += 1
            else:
                stats['excluded_group'] += 1
            continue

        # Skip mix files (by filename OR by is_multi classification)
        fname = path.lower()
        if 'mix' in fname or 'room' in fname:
            stats['excluded_mix'] += 1
            continue

        # Check if classified as multi-instrument
        pred_entry = next((p for p in pred_list if p['path'] == path), None)
        if pred_entry and pred_entry.get('is_multi'):
            stats['excluded_mix'] += 1
            continue

        # Add to training data
        training_data[path] = {
            'group': group,
            'subgroup': entry.get('subgroup', ''),
            'filename': entry.get('filename', Path(path).name),
            'source': source,
        }

        if source == 'manifest':
            stats['from_manifest'] += 1
        elif source == 'correction':
            stats['from_corrections'] += 1
        else:
            stats['from_validated_predictions'] += 1

    # Print stats
    print(f"\n=== Training Manifest Stats ===")
    print(f"Total entries: {len(training_data)}")
    print(f"  From original manifest: {stats['from_manifest']}")
    print(f"  From corrections: {stats['from_corrections']}")
    print(f"  From validated predictions: {stats['from_validated_predictions']}")
    print(f"\nExcluded:")
    print(f"  Mix files (filename/classified): {stats['excluded_mix']}")
    print(f"  Silent: {stats['excluded_silent']}")
    print(f"  Other excluded groups: {stats['excluded_group']}")

    # Group distribution
    group_counts = Counter(e['group'] for e in training_data.values())
    print(f"\n=== Group Distribution ===")
    for group, count in group_counts.most_common():
        print(f"  {group}: {count}")

    # Save
    output = {
        'entries': training_data,
        'generated_at': datetime.now().isoformat(),
        'stats': stats,
        'total': len(training_data),
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
