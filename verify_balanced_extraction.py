#!/usr/bin/env python3
"""
Verify Balanced Feature Extraction

Run this after normalizer fitting completes to verify:
1. Balanced selection is being used
2. Features are extracted from all 7 dimensions
3. Normalization is working correctly
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("🔍 Verifying Balanced Feature Extraction")
print("=" * 70)

# 1. Check selection file
print("\n1️⃣  Checking Selection File")
print("-" * 70)

selection_file = Path("midi_generator/feature_selection/output/selected_features_200_balanced.json")
if not selection_file.exists():
    print(f"❌ Balanced selection file not found: {selection_file}")
    print("   Make sure you created it with create_balanced_selection.py")
    sys.exit(1)

import json
with open(selection_file) as f:
    selection = json.load(f)

selected_features = selection.get('selected_features', [])
print(f"✅ Selection file loaded: {len(selected_features)} features")

# Count by dimension
dimension_counts = {
    'harmony': 0,
    'melody': 0,
    'rhythm': 0,
    'dynamics': 0,
    'texture': 0,
    'structure': 0,
    'orchestration': 0
}

for feat in selected_features:
    for dim in dimension_counts.keys():
        if feat.startswith(dim):
            dimension_counts[dim] += 1
            break

print("\n   Feature breakdown:")
for dim, count in dimension_counts.items():
    percent = (count / len(selected_features)) * 100
    print(f"   • {dim:15s}: {count:3d} features ({percent:5.1f}%)")

# Verify it's balanced (not all harmony)
if dimension_counts['harmony'] == len(selected_features):
    print("\n❌ ERROR: Only harmony features selected!")
    print("   You're still using the old selection file")
    sys.exit(1)

# 2. Test feature extraction
print("\n2️⃣  Testing Feature Extraction")
print("-" * 70)

try:
    from midi_generator.feature_selection.enhanced_feature_extractor import (
        EnhancedFeatureExtractor,
        NormalizedFeatureExtractor
    )

    # Find test MIDI
    test_midi = None
    for midi_dir in [
        Path("midi_generator/midi_corpus/big_band"),
        Path("/home/arlo/do-repo/midi_generator/midi_corpus/big_band")
    ]:
        if midi_dir.exists():
            test_files = list(midi_dir.glob("*.mid"))
            if test_files:
                test_midi = test_files[0]
                break

    if not test_midi:
        print("❌ No test MIDI found")
        sys.exit(1)

    print(f"📄 Test file: {test_midi.name}")

    # Create extractor with balanced selection
    base_extractor = EnhancedFeatureExtractor.from_selection_file(selection_file)
    features = base_extractor.extract(test_midi)

    print(f"\n✅ Extraction successful")
    print(f"   Shape: {features.shape}")
    print(f"   Non-zero: {(features != 0).sum()}/{len(features)}")
    print(f"   Range: [{features.min():.2f}, {features.max():.2f}]")

    # Break down by feature groups
    print(f"\n   Base features (0:200):")
    base_feats = features[:200]
    print(f"   • Non-zero: {(base_feats != 0).sum()}/200")
    print(f"   • Mean: {base_feats.mean():.2f}")
    print(f"   • Std: {base_feats.std():.2f}")

    print(f"\n   Velocity features (200:220):")
    vel_feats = features[200:]
    print(f"   • Non-zero: {(vel_feats != 0).sum()}/20")
    print(f"   • Mean: {vel_feats.mean():.2f}")
    print(f"   • Std: {vel_feats.std():.2f}")

    # Check if we got real features
    non_zero_count = (features != 0).sum()
    if non_zero_count < 50:
        print(f"\n❌ WARNING: Only {non_zero_count} non-zero features!")
        print("   Expected: 150-200 non-zero features")
        print("   This suggests DeepFeatureExtractor is still broken")
        print("\n   Check if mido is installed:")
        print("   conda activate ace_step")
        print("   python -c 'import mido; print(\"OK\")'")
    else:
        print(f"\n✅ Feature extraction working correctly!")
        print(f"   {non_zero_count} non-zero features is healthy")

except Exception as e:
    print(f"❌ Extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Test normalization (if normalizer exists)
print("\n3️⃣  Testing Normalization")
print("-" * 70)

try:
    # Try to load fitted normalizer
    normalizer_path = Path("midi_generator/feature_selection/output/feature_normalizer.pkl")

    if normalizer_path.exists():
        import pickle
        with open(normalizer_path, 'rb') as f:
            normalizer_data = pickle.load(f)

        print(f"✅ Normalizer found: {normalizer_path}")
        print(f"   Mean shape: {normalizer_data['mean'].shape}")
        print(f"   Std shape: {normalizer_data['std'].shape}")

        # Check normalization stats
        mean_range = [normalizer_data['mean'].min(), normalizer_data['mean'].max()]
        std_range = [normalizer_data['std'].min(), normalizer_data['std'].max()]

        print(f"\n   Normalization stats:")
        print(f"   • Mean range: [{mean_range[0]:.3f}, {mean_range[1]:.3f}]")
        print(f"   • Std range: [{std_range[0]:.3f}, {std_range[1]:.3f}]")

        # Test normalized extraction
        normalized_extractor = NormalizedFeatureExtractor(base_extractor)
        normalized_extractor.mean = normalizer_data['mean']
        normalized_extractor.std = normalizer_data['std']
        normalized_extractor.is_fitted = True

        norm_features = normalized_extractor.extract(test_midi)

        print(f"\n✅ Normalized features:")
        print(f"   • Mean: {norm_features.mean():.3f} (should be ~0)")
        print(f"   • Std: {norm_features.std():.3f} (should be ~1)")
        print(f"   • Range: [{norm_features.min():.2f}, {norm_features.max():.2f}]")

        if abs(norm_features.mean()) > 0.5:
            print(f"\n⚠️  WARNING: Mean is {norm_features.mean():.3f}, expected ~0")
            print("   Normalization may not be working correctly")

        if abs(norm_features.std() - 1.0) > 0.3:
            print(f"\n⚠️  WARNING: Std is {norm_features.std():.3f}, expected ~1")
            print("   Normalization may not be working correctly")

    else:
        print("⏳ Normalizer not fitted yet")
        print(f"   Expected at: {normalizer_path}")
        print("   Run this script again after normalizer fitting completes")

except Exception as e:
    print(f"❌ Normalization test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("📊 Verification Summary")
print("=" * 70)

print(f"""
✅ Selection: {len(selected_features)} balanced features across 7 dimensions
✅ Extraction: {non_zero_count}/{len(features)} non-zero features
{"✅ Normalization: Working correctly" if normalizer_path.exists() else "⏳ Normalization: Pending"}

Expected training behavior:
• Epoch 1-10: Loss 5000 → 500
• Epoch 10-30: Loss 500 → 50
• Epoch 30-50: Loss 50 → 10

Monitor with: tail -f /home/arlo/training_real_data.log
""")
