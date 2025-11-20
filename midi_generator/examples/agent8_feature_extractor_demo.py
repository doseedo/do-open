"""
Agent 8: Deep Feature Extractor Demo
=====================================

Demonstrates the 1000+ feature extraction system for Musical Program Synthesis.

This example shows how to:
1. Extract comprehensive features from MIDI files
2. Analyze feature distributions
3. Prepare features for XGBoost parameter prediction

Author: Agent 8 - Deep Feature Extractor Specialist
"""

from pathlib import Path
import numpy as np

# NOTE: Requires 'mido' package: pip install mido
try:
    from midi_generator.synthesis import DeepFeatureExtractor, extract_features
    MIDO_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: 'mido' package not installed")
    print("   Install with: pip install mido")
    MIDO_AVAILABLE = False


def demo_feature_extraction(midi_file: Path):
    """
    Demo: Extract all 1000+ features from a MIDI file.
    """
    print("=" * 80)
    print("AGENT 8: DEEP FEATURE EXTRACTOR DEMO")
    print("=" * 80)

    if not MIDO_AVAILABLE:
        print("\n❌ Cannot run demo: mido package not installed")
        print("   Install with: pip install mido")
        return

    # Initialize extractor
    extractor = DeepFeatureExtractor()

    print(f"\n✅ Feature Extractor Initialized")
    print(f"   Total Features: {extractor.feature_count}")

    print(f"\n📊 Feature Breakdown:")
    print(f"   Harmony:   250 features")
    print(f"   Melody:    200 features")
    print(f"   Rhythm:    250 features")
    print(f"   Dynamics:  150 features")
    print(f"   Texture:   100 features")
    print(f"   Structure:  50 features")
    print(f"   " + "-" * 40)
    print(f"   TOTAL:    1000 features")

    # Extract features
    print(f"\n🔍 Extracting features from: {midi_file.name}")

    try:
        features = extractor.extract(midi_file)

        print(f"\n✅ Feature Extraction Complete!")
        print(f"   Shape: {features.shape}")
        print(f"   Dtype: {features.dtype}")
        print(f"   Memory: {features.nbytes / 1024:.2f} KB")

        # Feature statistics
        print(f"\n📈 Feature Statistics:")
        print(f"   Min value:    {np.min(features):.4f}")
        print(f"   Max value:    {np.max(features):.4f}")
        print(f"   Mean:         {np.mean(features):.4f}")
        print(f"   Std dev:      {np.std(features):.4f}")
        print(f"   Non-zero:     {np.count_nonzero(features)}/{len(features)}")

        # Show sample features
        print(f"\n🎯 Sample Feature Values:")
        print(f"   First 10:  {features[:10]}")
        print(f"   Last 10:   {features[-10:]}")

        return features

    except Exception as e:
        print(f"\n❌ Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_feature_comparison(midi_file1: Path, midi_file2: Path):
    """
    Demo: Compare features from two different MIDI files.
    """
    print("\n" + "=" * 80)
    print("FEATURE COMPARISON DEMO")
    print("=" * 80)

    if not MIDO_AVAILABLE:
        print("\n❌ Cannot run demo: mido package not installed")
        return

    extractor = DeepFeatureExtractor()

    print(f"\nExtracting features from both files...")

    features1 = extractor.extract(midi_file1)
    features2 = extractor.extract(midi_file2)

    print(f"\n✅ Extracted features from both files")

    # Calculate similarity
    # Cosine similarity
    dot_product = np.dot(features1, features2)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    cosine_sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    # Euclidean distance
    euclidean_dist = np.linalg.norm(features1 - features2)

    # Feature-wise differences
    feature_diffs = np.abs(features1 - features2)
    max_diff_idx = np.argmax(feature_diffs)

    print(f"\n📊 Similarity Metrics:")
    print(f"   Cosine Similarity:  {cosine_sim:.4f}")
    print(f"   Euclidean Distance: {euclidean_dist:.4f}")
    print(f"   Max difference:     {feature_diffs[max_diff_idx]:.4f} (feature #{max_diff_idx})")
    print(f"   Mean difference:    {np.mean(feature_diffs):.4f}")


def demo_batch_processing(midi_directory: Path):
    """
    Demo: Process multiple MIDI files and build feature matrix.
    """
    print("\n" + "=" * 80)
    print("BATCH PROCESSING DEMO")
    print("=" * 80)

    if not MIDO_AVAILABLE:
        print("\n❌ Cannot run demo: mido package not installed")
        return

    # Find all MIDI files
    midi_files = list(midi_directory.glob("*.mid")) + list(midi_directory.glob("*.midi"))

    if len(midi_files) == 0:
        print(f"\n❌ No MIDI files found in {midi_directory}")
        return

    print(f"\n📁 Found {len(midi_files)} MIDI files")

    extractor = DeepFeatureExtractor()
    feature_matrix = []
    file_names = []

    print(f"\n🔄 Processing files...")

    for i, midi_file in enumerate(midi_files[:10], 1):  # Limit to 10 for demo
        print(f"   [{i}/{min(10, len(midi_files))}] {midi_file.name}")

        try:
            features = extractor.extract(midi_file)
            feature_matrix.append(features)
            file_names.append(midi_file.name)
        except Exception as e:
            print(f"      ⚠️  Error: {e}")

    if len(feature_matrix) > 0:
        feature_matrix = np.array(feature_matrix)

        print(f"\n✅ Batch Processing Complete!")
        print(f"   Feature Matrix Shape: {feature_matrix.shape}")
        print(f"   (Files × Features): ({len(file_names)} × {extractor.feature_count})")
        print(f"   Memory Size: {feature_matrix.nbytes / (1024*1024):.2f} MB")

        return feature_matrix, file_names
    else:
        print(f"\n❌ No features extracted")
        return None, None


def demo_feature_categories():
    """
    Demo: Show feature organization by category.
    """
    print("\n" + "=" * 80)
    print("FEATURE ORGANIZATION")
    print("=" * 80)

    feature_categories = {
        "Harmony Features": {
            "count": 250,
            "subcategories": [
                "Chord Quality & Extensions (23)",
                "Voicing Characteristics (24)",
                "Harmonic Progression (27)",
                "Voice Leading (25)",
                "Harmonic Rhythm (20)",
                "Tension & Resolution (18)",
                "Extensions & Alterations (25)",
                "Functional Harmony (25)",
                "Modal Harmony (20)",
                "Jazz Harmony (30)",
                "Advanced Harmony (13)"
            ]
        },
        "Melody Features": {
            "count": 200,
            "subcategories": [
                "Contour & Shape (16)",
                "Interval Analysis (24)",
                "Ornamentation (15)",
                "Sequence & Development (10)",
                "Melodic Density (20)",
                "Pitch Statistics (25)",
                "Directional Motion (20)",
                "Chromaticism (20)",
                "Range & Tessitura (15)",
                "Melodic Patterns (35)"
            ]
        },
        "Rhythm Features": {
            "count": 250,
            "subcategories": [
                "Temporal Patterns (13)",
                "Syncopation & Feel (18)",
                "Polyrhythm & Metric (20)",
                "Duration Statistics (30)",
                "Groove Analysis (40)",
                "Rhythmic Patterns (50)",
                "Micro-timing (30)",
                "Metric Structure (49)"
            ]
        },
        "Dynamics Features": {
            "count": 150,
            "subcategories": [
                "Velocity Analysis (17)",
                "Dynamic Shape (17)",
                "Articulation (13)",
                "Dynamic Contrast (20)",
                "Accent Patterns (20)",
                "Envelope Characteristics (20)",
                "Dynamic Transitions (20)",
                "Expression Depth (23)"
            ]
        },
        "Texture Features": {
            "count": 100,
            "subcategories": [
                "Density & Layering (15)",
                "Voice Independence (20)",
                "Vertical Density (20)",
                "Horizontal Density (20)",
                "Texture Type (15)",
                "Layer Interaction (10)"
            ]
        },
        "Structure Features": {
            "count": 50,
            "subcategories": [
                "Form Analysis (16)",
                "Development (10)",
                "Repetition & Variation (12)",
                "Sectional Analysis (12)"
            ]
        }
    }

    total = 0
    for category, info in feature_categories.items():
        print(f"\n{category}: {info['count']} features")
        for subcat in info['subcategories']:
            print(f"  • {subcat}")
        total += info['count']

    print(f"\n{'='*40}")
    print(f"TOTAL: {total} features")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AGENT 8: DEEP FEATURE EXTRACTOR - COMPREHENSIVE DEMO")
    print("=" * 80)

    # Show feature organization
    demo_feature_categories()

    # Check for test MIDI files
    test_files = list(Path("midi_generator").glob("*.mid"))

    if len(test_files) > 0:
        print(f"\n\n📁 Found {len(test_files)} test MIDI files")

        # Run extraction demo on first file
        demo_feature_extraction(test_files[0])

        # Run comparison if we have 2+ files
        if len(test_files) >= 2:
            demo_feature_comparison(test_files[0], test_files[1])

        # Run batch processing demo
        demo_batch_processing(Path("midi_generator"))

    else:
        print("\n\n📝 Usage Examples:")
        print("\nExtract features from a single file:")
        print("```python")
        print("from midi_generator.synthesis import extract_features")
        print("features = extract_features(Path('your_file.mid'))")
        print("print(f'Extracted {len(features)} features')")
        print("```")

        print("\n\nUse the DeepFeatureExtractor class:")
        print("```python")
        print("from midi_generator.synthesis import DeepFeatureExtractor")
        print("extractor = DeepFeatureExtractor()")
        print("features = extractor.extract(Path('your_file.mid'))")
        print("```")

        print("\n\nProcess multiple files:")
        print("```python")
        print("import numpy as np")
        print("from pathlib import Path")
        print("from midi_generator.synthesis import DeepFeatureExtractor")
        print("")
        print("extractor = DeepFeatureExtractor()")
        print("feature_matrix = []")
        print("")
        print("for midi_file in Path('midi_files/').glob('*.mid'):")
        print("    features = extractor.extract(midi_file)")
        print("    feature_matrix.append(features)")
        print("")
        print("feature_matrix = np.array(feature_matrix)")
        print("print(f'Shape: {feature_matrix.shape}')  # (n_files, 1000)")
        print("```")

    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)
