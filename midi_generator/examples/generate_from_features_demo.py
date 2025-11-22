"""
Generate MIDI from Features - Demo Script
==========================================

Demonstrates the complete Features→MIDI conversion pipeline.

This script shows how to:
1. Load or create feature vectors
2. Convert features to MIDI using rule-based approach
3. Save and validate the generated MIDI
4. Measure reconstruction quality

Author: Agent 1 - MIDI Decoder Architecture Lead
Date: November 22, 2025
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from midi_generator.models.rule_based_midi import RuleBasedFeaturesToMIDI

try:
    from midi_generator.synthesis.deep_feature_extractor import DeepFeatureExtractor
    EXTRACTOR_AVAILABLE = True
except ImportError:
    EXTRACTOR_AVAILABLE = False


def demo_basic_generation():
    """
    Demo 1: Basic MIDI generation from random features.
    """
    print("="*70)
    print("Demo 1: Basic MIDI Generation from Random Features")
    print("="*70)

    # Create converter
    converter = RuleBasedFeaturesToMIDI(verbose=True)

    # Create random features (simulating decoded features from DNA)
    print("\n📊 Creating random 1150D feature vector...")
    features = np.random.randn(1150)

    # Convert to MIDI
    print("\n🎵 Converting features to MIDI...")
    midi = converter.features_to_midi(
        features,
        output_path="output/generated_basic.mid"
    )

    print(f"\n✅ Generated MIDI with {len(midi.tracks)} tracks")

    # Validate
    is_valid = converter.validate_output(midi)
    print(f"   Valid: {is_valid}")

    return midi


def demo_parameter_extraction():
    """
    Demo 2: Extract and examine musical parameters from features.
    """
    print("\n" + "="*70)
    print("Demo 2: Musical Parameter Extraction")
    print("="*70)

    converter = RuleBasedFeaturesToMIDI(verbose=False)

    # Create features
    features = np.random.randn(1150)

    # Extract parameters
    print("\n📊 Extracting musical parameters...")
    params = converter.features_to_parameters(features)

    print("\n🎼 Extracted Parameters:")
    print(f"   Key: {params['key']}")
    print(f"   Mode: {params['mode']}")
    print(f"   Tempo: {params['tempo_bpm']:.1f} BPM")
    print(f"   Time Signature: {params['time_signature']}")
    print(f"   Melodic Range: {params['melodic_range']} semitones")
    print(f"   Velocity Mean: {params['velocity_mean']}")
    print(f"   Voice Count: {params['voice_count']}")
    print(f"   Number of Bars: {params['num_bars']}")
    print(f"   Form Type: {params['form_type']}")

    return params


def demo_controlled_generation():
    """
    Demo 3: Generate MIDI with controlled musical parameters.
    """
    print("\n" + "="*70)
    print("Demo 3: Controlled Generation")
    print("="*70)

    converter = RuleBasedFeaturesToMIDI(verbose=True)

    # Create features biased toward specific characteristics
    print("\n📊 Creating features for upbeat major piece...")
    features = np.random.randn(1150)

    # Bias harmony features toward major (positive values)
    features[0:12] = np.abs(np.random.randn(12))  # Positive pitch class
    features[12:20] = np.random.randn(8) + 0.5  # Major mode bias

    # Bias rhythm features toward faster tempo
    features[250:260] = np.random.randn(10) + 0.8  # High tempo

    # Generate MIDI
    print("\n🎵 Generating controlled MIDI...")
    midi = converter.features_to_midi(
        features,
        output_path="output/generated_controlled.mid"
    )

    # Check parameters
    params = converter.features_to_parameters(features)
    print(f"\n🎼 Resulting Parameters:")
    print(f"   Key: {params['key']}")
    print(f"   Mode: {params['mode']} (should tend toward major)")
    print(f"   Tempo: {params['tempo_bpm']:.1f} BPM (should be faster)")


def demo_batch_generation():
    """
    Demo 4: Generate multiple MIDI files from batch of features.
    """
    print("\n" + "="*70)
    print("Demo 4: Batch Generation")
    print("="*70)

    converter = RuleBasedFeaturesToMIDI(verbose=False)

    # Create batch of features
    batch_size = 5
    print(f"\n📊 Creating batch of {batch_size} feature vectors...")
    feature_batch = np.random.randn(batch_size, 1150)

    # Generate multiple MIDI files
    print(f"\n🎵 Generating {batch_size} MIDI files...")
    for i in range(batch_size):
        midi = converter.features_to_midi(
            feature_batch[i],
            output_path=f"output/generated_batch_{i}.mid"
        )
        print(f"   ✅ Generated file {i+1}/{batch_size}")


def demo_roundtrip_reconstruction():
    """
    Demo 5: Full roundtrip - MIDI → Features → MIDI.

    This demonstrates the complete pipeline if DeepFeatureExtractor is available.
    """
    print("\n" + "="*70)
    print("Demo 5: Roundtrip Reconstruction")
    print("="*70)

    if not EXTRACTOR_AVAILABLE:
        print("\n⚠️  DeepFeatureExtractor not available")
        print("   Skipping roundtrip demo")
        return

    converter = RuleBasedFeaturesToMIDI(verbose=True)
    extractor = DeepFeatureExtractor()

    # Generate original MIDI
    print("\n🎵 Step 1: Generate original MIDI...")
    original_features = np.random.randn(1150)
    original_midi = converter.features_to_midi(
        original_features,
        output_path="output/original.mid"
    )

    # Extract features from generated MIDI
    print("\n📊 Step 2: Extract features from generated MIDI...")
    extracted_features = extractor.extract(Path("output/original.mid"))
    print(f"   Extracted {len(extracted_features)} features")

    # Reconstruct MIDI from extracted features
    print("\n🎵 Step 3: Reconstruct MIDI from extracted features...")
    reconstructed_midi = converter.features_to_midi(
        extracted_features,
        output_path="output/reconstructed.mid"
    )

    # Measure reconstruction quality
    print("\n📈 Step 4: Measure reconstruction quality...")
    quality = converter.compute_reconstruction_quality(
        original_midi,
        reconstructed_midi
    )

    print(f"\n🎯 Reconstruction Quality Metrics:")
    print(f"   Note Precision: {quality['note_precision']:.3f}")
    print(f"   Note Recall: {quality['note_recall']:.3f}")
    print(f"   Note F1: {quality['note_f1']:.3f}")
    print(f"   Pitch Accuracy: {quality['pitch_accuracy']:.3f}")
    print(f"   Rhythm Similarity: {quality['rhythm_similarity']:.3f}")
    print(f"   Overall Similarity: {quality['overall_similarity']:.3f}")


def demo_feature_slicing():
    """
    Demo 6: Analyze specific feature dimensions.
    """
    print("\n" + "="*70)
    print("Demo 6: Feature Dimension Analysis")
    print("="*70)

    converter = RuleBasedFeaturesToMIDI(verbose=False)
    features = np.random.randn(1150)

    print("\n📊 Feature Breakdown:")
    breakdown = converter.get_feature_breakdown()

    for category, (start, end) in breakdown.items():
        slice_features = converter.extract_feature_slice(features, category)
        print(f"   {category:15s}: [{start:4d}:{end:4d}] = {len(slice_features):3d}D")
        print(f"      Mean: {np.mean(slice_features):+.3f}, "
              f"Std: {np.std(slice_features):.3f}")


# ============================================================================
# Main Demo Runner
# ============================================================================

def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("Features→MIDI Conversion - Complete Demo Suite")
    print("Agent 1: MIDI Decoder Architecture")
    print("="*70)

    # Create output directory
    Path("output").mkdir(exist_ok=True)

    try:
        # Run demos
        demo_basic_generation()
        demo_parameter_extraction()
        demo_controlled_generation()
        demo_batch_generation()
        demo_feature_slicing()
        demo_roundtrip_reconstruction()

        print("\n" + "="*70)
        print("✅ All demos completed successfully!")
        print("="*70)
        print("\n📂 Generated MIDI files saved to: output/")
        print("\nNext steps:")
        print("1. Listen to the generated MIDI files")
        print("2. Try modifying feature values to control output")
        print("3. Integrate with the full DNA→Features→MIDI pipeline")
        print("4. Train the neural generator for optimal quality")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
