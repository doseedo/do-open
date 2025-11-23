#!/usr/bin/env python3
"""
Compare All Feature Extractors

Tests all available extractors to see which one gives real features.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Find a test MIDI file
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
    print("❌ No test MIDI file found")
    sys.exit(1)

print(f"📄 Test MIDI: {test_midi.name}")
print("="*70)

# Test 1: DeepFeatureExtractor (1150D)
print("\n1️⃣  DeepFeatureExtractor (1150D)")
print("-"*70)
try:
    from midi_generator.synthesis.deep_feature_extractor import DeepFeatureExtractor
    extractor = DeepFeatureExtractor()
    features = extractor.extract(test_midi)
    print(f"✅ SUCCESS")
    print(f"   Shape: {features.shape}")
    print(f"   Non-zero: {(features != 0).sum()}/{len(features)}")
    print(f"   Range: [{features.min():.2f}, {features.max():.2f}]")
    print(f"   Mean: {features.mean():.2f}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 2: OptimizedFeatureExtractor (200D)
print("\n2️⃣  OptimizedFeatureExtractor (200D)")
print("-"*70)
try:
    from midi_generator.feature_selection.optimized_feature_extractor import OptimizedFeatureExtractor
    selection_file = Path("midi_generator/feature_selection/output/selected_features_200_actual.json")
    if selection_file.exists():
        extractor = OptimizedFeatureExtractor.from_selection_file(selection_file)
        features = extractor.extract(test_midi)
        print(f"✅ SUCCESS")
        print(f"   Shape: {features.shape}")
        print(f"   Non-zero: {(features != 0).sum()}/{len(features)}")
        print(f"   Range: [{features.min():.2f}, {features.max():.2f}]")
        print(f"   Mean: {features.mean():.2f}")

        # Show which features are non-zero
        nonzero_idx = np.where(features != 0)[0]
        print(f"   Non-zero indices: {nonzero_idx[:20].tolist()}{'...' if len(nonzero_idx) > 20 else ''}")
    else:
        print(f"❌ Selection file not found: {selection_file}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 3: EnhancedFeatureExtractor (220D)
print("\n3️⃣  EnhancedFeatureExtractor (220D)")
print("-"*70)
try:
    from midi_generator.feature_selection.enhanced_feature_extractor import EnhancedFeatureExtractor
    selection_file = Path("midi_generator/feature_selection/output/selected_features_200_actual.json")
    if selection_file.exists():
        extractor = EnhancedFeatureExtractor.from_selection_file(selection_file)
        features = extractor.extract(test_midi)
        print(f"✅ SUCCESS")
        print(f"   Shape: {features.shape}")
        print(f"   Non-zero: {(features != 0).sum()}/{len(features)}")
        print(f"   Range: [{features.min():.2f}, {features.max():.2f}]")
        print(f"   Mean: {features.mean():.2f}")
        print(f"   Base features (0:200): {(features[:200] != 0).sum()}/200 non-zero")
        print(f"   Velocity features (200:220): {(features[200:] != 0).sum()}/20 non-zero")
    else:
        print(f"❌ Selection file not found: {selection_file}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 4: RichMultitrackFeatureExtractor (600D)
print("\n4️⃣  RichMultitrackFeatureExtractor (600D)")
print("-"*70)
try:
    from midi_generator.feature_selection.rich_feature_extractor import RichMultitrackFeatureExtractor
    extractor = RichMultitrackFeatureExtractor()
    features = extractor.extract(str(test_midi))
    print(f"✅ SUCCESS")
    print(f"   Shape: {features.shape}")
    print(f"   Non-zero: {(features != 0).sum()}/{len(features)}")
    print(f"   Range: [{features.min():.2f}, {features.max():.2f}]")
    print(f"   Mean: {features.mean():.2f}")
    print(f"   Global (0:200): {(features[:200] != 0).sum()}/200 non-zero")
    print(f"   Per-track (200:400): {(features[200:400] != 0).sum()}/200 non-zero")
    print(f"   Temporal (400:500): {(features[400:500] != 0).sum()}/100 non-zero")
    print(f"   Orchestration (500:600): {(features[500:] != 0).sum()}/100 non-zero")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)
print("""
For v2.0 training with convergence fixes:

✅ RECOMMENDED: EnhancedFeatureExtractor (220D) + NormalizedFeatureExtractor
   - 200D selected features from DeepFeatureExtractor
   - +20D velocity features
   - Normalized to mean=0, std=1
   - Works with updated encoder configs (input_dim=220)

Alternative (if mido issues): RichMultitrackFeatureExtractor (600D)
   - Uses pretty_midi instead of mido
   - More comprehensive (600D vs 220D)
   - Would require updating encoder config to input_dim=600
""")
