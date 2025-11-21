"""
Dataset Verification Script
============================

Verifies the extracted dataset has correct format.

Usage:
    python scripts/verify_dataset.py labeled_dataset_complete.json
"""

import argparse
import json
import sys
from pathlib import Path


def verify_dataset(dataset_file: Path):
    """Verify dataset format"""
    print(f"\n{'='*80}")
    print(f"DATASET VERIFICATION")
    print(f"{'='*80}")
    print(f"File: {dataset_file}")
    print(f"{'='*80}\n")

    # Load dataset
    with open(dataset_file) as f:
        data = json.load(f)

    print(f"✅ Total samples: {len(data)}")

    # Verify first sample
    if not data:
        print("❌ Dataset is empty!")
        return False

    sample = data[0]
    print(f"\n📋 Sample Structure:")
    print(f"   File ID: {sample.get('file_id', 'MISSING')}")
    print(f"   File Path: {sample.get('file_path', 'MISSING')}")

    # Verify features
    if 'features' not in sample:
        print(f"   ❌ Missing 'features' key")
        return False

    features = sample['features']
    if not isinstance(features, list):
        print(f"   ❌ 'features' should be a list, got {type(features)}")
        return False

    if len(features) != 200:
        print(f"   ❌ Expected 200 features, got {len(features)}")
        return False

    print(f"   ✅ Features: {len(features)}D")

    # Verify parameters
    if 'parameters' not in sample:
        print(f"   ❌ Missing 'parameters' key")
        return False

    params = sample['parameters']

    # Check Level 1
    if 'level1_global' not in params:
        print(f"   ❌ Missing 'level1_global'")
        return False

    level1_count = len(params['level1_global'])
    if level1_count != 8:
        print(f"   ❌ Level 1: expected 8 params, got {level1_count}")
        return False

    print(f"   ✅ Level 1: {level1_count} parameters")

    # Check Level 2
    if 'level2_universal' not in params:
        print(f"   ❌ Missing 'level2_universal'")
        return False

    level2_count = sum(len(v) for v in params['level2_universal'].values())
    if level2_count != 20:
        print(f"   ❌ Level 2: expected 20 params, got {level2_count}")
        return False

    print(f"   ✅ Level 2: {level2_count} parameters")

    # Check Level 3
    if 'level3_genre_specific' not in params:
        print(f"   ❌ Missing 'level3_genre_specific'")
        return False

    level3_count = sum(len(v) for v in params['level3_genre_specific'].values())
    if level3_count != 22:
        print(f"   ❌ Level 3: expected 22 params, got {level3_count}")
        print(f"   Level 3 breakdown:")
        for genre, genre_params in params['level3_genre_specific'].items():
            print(f"     - {genre}: {len(genre_params)} params")
        return False

    print(f"   ✅ Level 3: {level3_count} parameters (all genres)")

    # Genre verification
    genre = params['level1_global'].get('genre.primary', 'unknown')
    print(f"   ✅ Detected genre: {genre}")

    # Verify all samples (quick check)
    print(f"\n🔍 Verifying all {len(data)} samples...")
    errors = []

    for i, sample in enumerate(data):
        try:
            # Quick checks
            assert len(sample['features']) == 200
            p = sample['parameters']
            assert len(p['level1_global']) == 8
            assert sum(len(v) for v in p['level2_universal'].values()) == 20
            assert sum(len(v) for v in p['level3_genre_specific'].values()) == 22
        except (AssertionError, KeyError) as e:
            errors.append((i, sample.get('file_id', 'unknown'), str(e)))

    if errors:
        print(f"\n❌ Found {len(errors)} samples with errors:")
        for idx, file_id, error in errors[:10]:
            print(f"   Sample {idx} ({file_id}): {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")
        return False

    print(f"✅ All samples valid!")

    # Summary
    print(f"\n{'='*80}")
    print(f"✅ DATASET VERIFIED")
    print(f"{'='*80}")
    print(f"Total samples: {len(data)}")
    print(f"Features per sample: 200D")
    print(f"Parameters per sample: 50 (8+20+22)")
    print(f"Format: READY FOR TRAINING ✅")
    print(f"{'='*80}\n")

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify extracted dataset format')
    parser.add_argument('dataset', type=str, help='Path to labeled dataset JSON file')

    args = parser.parse_args()

    dataset_file = Path(args.dataset)

    if not dataset_file.exists():
        print(f"❌ Dataset file not found: {dataset_file}")
        sys.exit(1)

    success = verify_dataset(dataset_file)

    sys.exit(0 if success else 1)
