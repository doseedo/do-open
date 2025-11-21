#!/usr/bin/env python3
"""
Example: Semantic Feature Representations - Agent 2
===================================================

Demonstrates usage of semantic features for musical parameter discovery.

This example shows:
1. Creating semantic features
2. Managing feature banks
3. Computing activations
4. Testing locality invariance
5. Finding similar features
6. Detecting redundancy
7. Serialization/deserialization

Author: Agent 2
License: MIT
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from learning.semantic_features import (
        FeatureModality,
        SemanticFeature,
        SemanticFeatureBank,
        cosine_similarity,
        euclidean_distance,
        find_similar_features,
        detect_redundant_features,
        create_semantic_feature
    )
    from learning.musical_locality import (
        LocalityType,
        MusicalLocalityFunctions
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("Make sure you're running from the correct directory")
    IMPORTS_AVAILABLE = False
    sys.exit(1)


def example_1_basic_feature_creation():
    """Example 1: Creating semantic features"""
    print("=" * 70)
    print("Example 1: Creating Semantic Features")
    print("=" * 70)
    print()

    # Create a melodic feature
    melodic_feature = SemanticFeature(
        feature_id='melodic_ascent',
        weight_vector=np.random.randn(200),
        modality=FeatureModality.MELODIC,
        activation_threshold=0.6,
        locality_constraints=[
            LocalityType.TRANSPOSE,
            LocalityType.AUGMENT
        ]
    )

    print(f"✅ Created feature: {melodic_feature.feature_id}")
    print(f"   Modality: {melodic_feature.modality.name}")
    print(f"   Threshold: {melodic_feature.activation_threshold}")
    print(f"   Locality constraints: {[lc.name for lc in melodic_feature.locality_constraints]}")
    print(f"   Weight vector shape: {melodic_feature.weight_vector.shape}")
    print()

    # Compute activation for random input
    input_features = np.random.randn(200)
    activation = melodic_feature.get_activation_strength(input_features)

    print(f"Activation strength: {activation:.4f}")
    print(f"Matches pattern: {melodic_feature.matches_pattern(input_features)}")
    print()


def example_2_feature_bank_operations():
    """Example 2: Managing feature banks"""
    print("=" * 70)
    print("Example 2: Feature Bank Operations")
    print("=" * 70)
    print()

    # Create bank
    bank = SemanticFeatureBank()
    print("✅ Created empty feature bank")
    print()

    # Add features with different modalities
    modalities = [
        FeatureModality.MELODIC,
        FeatureModality.HARMONIC,
        FeatureModality.RHYTHMIC,
        FeatureModality.MELODIC,
        FeatureModality.HARMONIC,
    ]

    for i, modality in enumerate(modalities):
        feature = SemanticFeature(
            feature_id=f'feature_{i}',
            weight_vector=np.random.randn(200),
            modality=modality,
            activation_threshold=0.5
        )
        bank.add_feature(feature)

    print(f"Added {len(bank)} features to bank")
    print(f"Feature dimension: {bank.feature_dimension}D")
    print()

    # Get statistics
    stats = bank.compute_feature_statistics()
    print("Bank statistics:")
    print(f"  Total features: {stats['num_features']}")
    print(f"  Modality distribution: {stats['modality_distribution']}")
    print(f"  Avg weight magnitude: {stats['avg_weight_magnitude']:.4f}")
    print(f"  Avg threshold: {stats['avg_threshold']:.4f}")
    print()

    # Filter by modality
    melodic_features = bank.get_features_by_modality(FeatureModality.MELODIC)
    print(f"Melodic features: {len(melodic_features)}")
    print()

    return bank


def example_3_activation_computation(bank):
    """Example 3: Computing activations"""
    print("=" * 70)
    print("Example 3: Computing Activations")
    print("=" * 70)
    print()

    # Create random input (simulating extracted features)
    input_features = np.random.randn(200)
    print("Computing activations for input features...")
    print()

    # Get all activations
    activations = bank.get_activations(input_features)
    print(f"Total activations: {len(activations)}")
    for feature_id, activation in list(activations.items())[:3]:
        print(f"  {feature_id}: {activation:.4f}")
    print()

    # Get top-k features
    k = 3
    top_features = bank.get_top_k_features(input_features, k=k)
    print(f"Top-{k} activated features:")
    for feature_id, activation, feature in top_features:
        print(f"  {feature_id}: {activation:.4f} (modality: {feature.modality.name})")
    print()

    # Filter by threshold
    high_activations = bank.get_activations(input_features, threshold=0.6)
    print(f"Features with activation > 0.6: {len(high_activations)}")
    print()


def example_4_locality_invariance():
    """Example 4: Testing locality invariance"""
    print("=" * 70)
    print("Example 4: Locality Invariance Testing")
    print("=" * 70)
    print()

    # Create feature with locality constraints
    feature = SemanticFeature(
        feature_id='transpose_invariant',
        weight_vector=np.random.randn(200),
        modality=FeatureModality.HARMONIC,
        locality_constraints=[
            LocalityType.TRANSPOSE,
            LocalityType.REGISTER_SHIFT
        ]
    )

    print(f"Testing feature: {feature.feature_id}")
    print(f"Locality constraints: {[lc.name for lc in feature.locality_constraints]}")
    print()

    # Create input
    input_features = np.random.randn(200)

    # Test invariance
    result = feature.test_locality_invariance(
        input_features,
        num_tests=10,
        tolerance=0.15
    )

    print("Invariance test results:")
    print(f"  Original activation: {result['original_activation']:.4f}")
    print(f"  Mean activation: {result['mean_activation']:.4f}")
    print(f"  Std activation: {result['std_activation']:.4f}")
    print(f"  Max deviation: {result['max_deviation']:.4f}")
    print(f"  Tolerance: {result['tolerance']:.4f}")
    print(f"  Is invariant: {'✅ YES' if result['is_invariant'] else '❌ NO'}")
    print()

    # Generate variants
    print("Generating musical variants...")
    variants = feature.generate_variants(input_features, num_variants=5)
    print(f"Generated {len(variants)} variants:")
    for i, (variant_features, transform, activation) in enumerate(variants):
        print(f"  Variant {i+1}: {transform.transform_type.name} -> activation {activation:.4f}")
    print()


def example_5_similarity_analysis(bank):
    """Example 5: Finding similar features"""
    print("=" * 70)
    print("Example 5: Similarity Analysis")
    print("=" * 70)
    print()

    # Get a target feature
    target_feature = bank.get_feature('feature_0')
    print(f"Target feature: {target_feature.feature_id}")
    print(f"Modality: {target_feature.modality.name}")
    print()

    # Find similar features
    similar = find_similar_features(
        target_feature,
        bank,
        k=3,
        metric='cosine'
    )

    print("Most similar features (cosine similarity):")
    for feature_id, similarity, feature in similar:
        print(f"  {feature_id}: {similarity:.4f} (modality: {feature.modality.name})")
    print()

    # Compute pairwise similarities
    feature_ids = list(bank.features.keys())[:3]
    print("Pairwise similarities:")
    for i, fid1 in enumerate(feature_ids):
        for fid2 in feature_ids[i+1:]:
            f1 = bank.get_feature(fid1)
            f2 = bank.get_feature(fid2)
            sim = cosine_similarity(f1, f2)
            print(f"  {fid1} <-> {fid2}: {sim:.4f}")
    print()


def example_6_redundancy_detection(bank):
    """Example 6: Detecting redundant features"""
    print("=" * 70)
    print("Example 6: Redundancy Detection")
    print("=" * 70)
    print()

    # Add a nearly identical feature to create redundancy
    original_feature = bank.get_feature('feature_0')
    redundant_feature = SemanticFeature(
        feature_id='feature_redundant',
        weight_vector=original_feature.weight_vector + np.random.randn(200) * 0.01,
        modality=original_feature.modality
    )
    bank.add_feature(redundant_feature)

    print(f"Added redundant feature (similar to {original_feature.feature_id})")
    print()

    # Detect redundancy
    redundant_pairs = detect_redundant_features(
        bank,
        similarity_threshold=0.95
    )

    print(f"Redundant pairs found: {len(redundant_pairs)}")
    for fid1, fid2, similarity in redundant_pairs:
        print(f"  {fid1} <-> {fid2}: {similarity:.4f}")
    print()

    if redundant_pairs:
        print("💡 Tip: Consider removing one feature from each redundant pair")
        print()


def example_7_serialization():
    """Example 7: Save and load feature banks"""
    print("=" * 70)
    print("Example 7: Serialization")
    print("=" * 70)
    print()

    # Create a small bank
    bank = SemanticFeatureBank()
    for i in range(5):
        feature = SemanticFeature(
            feature_id=f'feature_{i}',
            weight_vector=np.random.randn(200),
            modality=FeatureModality.MELODIC,
            interpretation=f"Feature {i} interpretation"
        )
        bank.add_feature(feature)

    print(f"Created bank with {len(bank)} features")
    print()

    # Save
    save_path = Path('/tmp/test_semantic_features.pkl')
    bank.save(save_path)
    print(f"✅ Saved to {save_path}")
    print(f"   Also saved JSON to {save_path.with_suffix('.json')}")
    print()

    # Load
    loaded_bank = SemanticFeatureBank.load(save_path)
    print(f"✅ Loaded bank with {len(loaded_bank)} features")
    print()

    # Verify
    print("Verification:")
    print(f"  Original features: {len(bank)}")
    print(f"  Loaded features: {len(loaded_bank)}")
    print(f"  Match: {'✅ YES' if len(bank) == len(loaded_bank) else '❌ NO'}")
    print()

    # Check JSON readability
    json_path = save_path.with_suffix('.json')
    if json_path.exists():
        import json
        with open(json_path) as f:
            data = json.load(f)
        print(f"JSON file size: {json_path.stat().st_size:,} bytes")
        print(f"JSON keys: {list(data.keys())}")
        print()


def example_8_integration_workflow():
    """Example 8: Full integration workflow"""
    print("=" * 70)
    print("Example 8: Full Integration Workflow")
    print("=" * 70)
    print()

    print("This example shows how Agent 2 integrates with other agents:")
    print()

    print("Step 1: Agent 1 provides locality functions")
    print("  → MusicalLocalityFunctions with 12 transformations")
    print()

    print("Step 2: Agent 3 trains neural encoder")
    print("  → Learns 20-30 semantic features from gap dataset")
    print("  → Enforces locality constraints during training")
    print()

    print("Step 3: Agent 3 creates SemanticFeatureBank")
    print("  → Extracts weight vectors from trained encoder")
    print("  → Creates SemanticFeature for each learned feature")
    print("  → Saves bank.save('learned_features.pkl')")
    print()

    print("Step 4: Agent 6 interprets features")
    print("  → Loads bank = SemanticFeatureBank.load('learned_features.pkl')")
    print("  → Tests each feature against musical patterns")
    print("  → Sets feature.interpretation and feature.modality")
    print("  → Sets feature.parameter_mapping for extracted parameters")
    print("  → Saves bank.save('interpreted_features.pkl')")
    print()

    print("Step 5: Use discovered parameters")
    print("  → Load interpreted bank")
    print("  → Extract parameters from new MIDI files")
    print("  → Use for generation or analysis")
    print()

    # Simulate this workflow
    print("Simulating workflow...")
    print()

    # Agent 3: Create learned features
    bank = SemanticFeatureBank()
    print("[Agent 3] Training encoder and extracting features...")
    for i in range(10):
        feature = SemanticFeature(
            feature_id=f'learned_feature_{i}',
            weight_vector=np.random.randn(200),
            modality=FeatureModality.UNKNOWN  # Not yet interpreted
        )
        bank.add_feature(feature)
    print(f"[Agent 3] ✅ Created {len(bank)} learned features")
    print()

    # Agent 6: Interpret features
    print("[Agent 6] Interpreting features...")
    modalities = [
        FeatureModality.MELODIC,
        FeatureModality.HARMONIC,
        FeatureModality.RHYTHMIC,
    ]
    for i, feature in enumerate(bank):
        # Simulate interpretation
        feature.modality = modalities[i % len(modalities)]
        feature.interpretation = f"Simulated interpretation of {feature.feature_id}"
        if i < 3:  # First 3 map to parameters
            feature.parameter_mapping = {
                'parameter_name': f'discovered_param_{i}',
                'extraction_function': f'extract_param_{i}'
            }
    print(f"[Agent 6] ✅ Interpreted {len(bank.get_interpreted_features())} features")
    print(f"[Agent 6] ✅ Mapped {len(bank.get_mapped_parameters())} parameters")
    print()

    # Show results
    stats = bank.compute_feature_statistics()
    print("Final statistics:")
    print(f"  Total features: {stats['num_features']}")
    print(f"  Interpreted: {stats['num_interpreted']} ({stats['interpretation_rate']*100:.0f}%)")
    print(f"  Mapped to parameters: {stats['num_mapped_to_parameters']}")
    print(f"  Modalities: {stats['modality_distribution']}")
    print()


def main():
    """Run all examples"""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "Semantic Features Examples" + " " * 22 + "║")
    print("║" + " " * 28 + "Agent 2" + " " * 31 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    if not IMPORTS_AVAILABLE:
        print("❌ Required imports not available")
        return

    try:
        # Run examples
        example_1_basic_feature_creation()
        input("Press Enter to continue to Example 2...")
        print()

        bank = example_2_feature_bank_operations()
        input("Press Enter to continue to Example 3...")
        print()

        example_3_activation_computation(bank)
        input("Press Enter to continue to Example 4...")
        print()

        example_4_locality_invariance()
        input("Press Enter to continue to Example 5...")
        print()

        example_5_similarity_analysis(bank)
        input("Press Enter to continue to Example 6...")
        print()

        example_6_redundancy_detection(bank)
        input("Press Enter to continue to Example 7...")
        print()

        example_7_serialization()
        input("Press Enter to continue to Example 8...")
        print()

        example_8_integration_workflow()

        print("=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70)
        print()

        print("Next steps:")
        print("  1. Review semantic_features.py for implementation details")
        print("  2. Run test_semantic_features.py for comprehensive tests")
        print("  3. See AGENT_02_SEMANTIC_FEATURES.md for full documentation")
        print()

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
