"""
Semantic Feature Interpretation Example - Agent 6
==================================================

This example demonstrates how Agent 6 (Feature Interpreter) integrates
with the semantic discovery pipeline to convert learned neural features
into human-understandable musical parameters.

Integration Flow:
1. Agent 4: GapDataset creates training data
2. Agent 5: GapDiscoveryTrainer trains semantic features
3. Agent 6: FeatureInterpreter interprets learned features ← THIS EXAMPLE
4. Agent 7: SemanticDiscoveryPipeline orchestrates end-to-end

Author: Agent 6
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from midi_generator.learning.feature_interpreter import (
        FeatureInterpreter,
        MusicalTestPatterns,
        ConceptMatcher,
        ParameterNameGenerator,
        ExtractionFunctionGenerator,
        FeatureModality
    )
    from midi_generator.parameters.universal_registry import UniversalParameterRegistry
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Install dependencies: pip install numpy scipy mido")
    IMPORTS_AVAILABLE = False
    sys.exit(1)


def example_1_test_patterns():
    """Example 1: Explore the musical test pattern library"""
    print("=" * 80)
    print("EXAMPLE 1: Musical Test Pattern Library")
    print("=" * 80)

    patterns = MusicalTestPatterns()

    print(f"\nTotal test patterns: {len(patterns.patterns)}")
    print("\nBreakdown by modality:")

    for modality in FeatureModality:
        modality_patterns = patterns.get_patterns_by_modality(modality)
        if modality_patterns:
            print(f"  {modality.value:20s}: {len(modality_patterns):2d} patterns")

    print("\nSample pitch patterns:")
    pitch_patterns = patterns.get_patterns_by_modality(FeatureModality.PITCH)
    for pattern in pitch_patterns[:5]:
        print(f"  - {pattern.name:30s}: {pattern.description}")

    print("\nSample harmony patterns:")
    harmony_patterns = patterns.get_patterns_by_modality(FeatureModality.HARMONY)
    for pattern in harmony_patterns[:5]:
        print(f"  - {pattern.name:30s}: {pattern.description}")

    print("\nSample rhythm patterns:")
    rhythm_patterns = patterns.get_patterns_by_modality(FeatureModality.RHYTHM)
    for pattern in rhythm_patterns[:5]:
        print(f"  - {pattern.name:30s}: {pattern.description}")


def example_2_concept_matching():
    """Example 2: Demonstrate concept matching"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Concept Matching")
    print("=" * 80)

    matcher = ConceptMatcher()

    print(f"\nTotal musical concepts: {len(matcher.concepts)}")

    print("\nSample concepts:")
    for concept in matcher.concepts[:10]:
        print(f"  {concept.name:30s} ({concept.modality.value})")
        print(f"    Description: {concept.description}")
        print(f"    Test patterns: {', '.join(concept.characteristic_patterns[:3])}")
        print()

    # Simulate feature responses for a "syncopation" feature
    print("\nSimulating concept matching for a syncopation feature...")
    pattern_responses = {
        "syncopated_rhythm": 0.95,
        "swing_rhythm": 0.3,
        "steady_quarter_notes": 0.1,
        "major_scale_ascending": 0.15,
        "major_chord": 0.2
    }

    matched_concept = matcher.match_concept(
        FeatureModality.RHYTHM,
        pattern_responses,
        threshold=0.5
    )

    if matched_concept:
        print(f"\n  ✓ Matched concept: {matched_concept.name}")
        print(f"    Description: {matched_concept.description}")
        print(f"    Modality: {matched_concept.modality.value}")
    else:
        print("\n  ✗ No concept matched (below threshold)")


def example_3_parameter_naming():
    """Example 3: Demonstrate parameter name generation"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Parameter Name Generation")
    print("=" * 80)

    generator = ParameterNameGenerator()
    matcher = ConceptMatcher()

    # Get some example concepts
    syncopation_concept = next(c for c in matcher.concepts if c.name == "syncopation")
    scale_concept = next(c for c in matcher.concepts if c.name == "scale_type")
    chord_concept = next(c for c in matcher.concepts if c.name == "chord_quality")

    print("\nExample parameter names:")

    name1 = generator.generate_name(FeatureModality.RHYTHM, syncopation_concept, 0)
    print(f"  Feature 0 (syncopation): {name1}")

    name2 = generator.generate_name(FeatureModality.PITCH, scale_concept, 1)
    print(f"  Feature 1 (scale type):  {name2}")

    name3 = generator.generate_name(FeatureModality.HARMONY, chord_concept, 2)
    print(f"  Feature 2 (chord quality): {name3}")

    # Unknown feature
    name4 = generator.generate_name(FeatureModality.DYNAMICS, None, 42)
    print(f"  Feature 42 (unknown):    {name4}")

    print("\nDescriptions:")
    desc1 = generator.generate_description(FeatureModality.RHYTHM, syncopation_concept, {})
    print(f"  {name1}: {desc1}")


def example_4_full_interpretation():
    """Example 4: Full feature interpretation pipeline"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Full Feature Interpretation Pipeline")
    print("=" * 80)

    print("\nThis demonstrates the complete Agent 6 workflow:")
    print("1. Receive learned features from Agent 5")
    print("2. Test feature responses on musical patterns")
    print("3. Classify modality and match concepts")
    print("4. Generate parameter names")
    print("5. Create extraction functions")
    print("6. Register with UniversalParameterRegistry")

    # Initialize interpreter
    interpreter = FeatureInterpreter()

    print(f"\nInterpreter initialized:")
    print(f"  - Test patterns: {len(interpreter.test_patterns.patterns)}")
    print(f"  - Concepts: {len(interpreter.concept_matcher.concepts)}")

    # Mock Agent 5's output
    print("\n[Simulating Agent 5's trained features...]")

    # In real usage, these would come from Agent 5:
    # semantic_feature_bank = trained_result.feature_bank
    # encoder_model = trained_result.encoder

    # For this example, we'll create mocks
    class MockFeatureBank:
        num_features = 5

    class MockEncoder:
        pass

    mock_bank = MockFeatureBank()
    mock_encoder = MockEncoder()

    print(f"  Received {mock_bank.num_features} learned features")

    # Interpret features (in real use)
    # interpretations = interpreter.interpret_features(mock_bank, mock_encoder)

    print("\n[In production, this would:]")
    print("  1. Test each feature on 35+ test patterns")
    print("  2. Compute activation strengths")
    print("  3. Classify modality based on strongest responses")
    print("  4. Match to known musical concepts")
    print("  5. Generate human-readable names")
    print("  6. Create MIDI extraction functions")

    print("\nExpected output:")
    print("  ✓ Feature 0: rhythm.syncopation.strength (confidence: 0.87)")
    print("  ✓ Feature 1: harmony.chord_quality.complexity (confidence: 0.82)")
    print("  ✓ Feature 2: melody.contour.smoothness (confidence: 0.79)")
    print("  ✗ Feature 3: unknown.feature_3 (confidence: 0.45 - below threshold)")
    print("  ✓ Feature 4: rhythm.subdivision.level (confidence: 0.73)")

    print("\n[Registering parameters...]")
    print("  ✓ Registered: rhythm.syncopation.strength")
    print("  ✓ Registered: harmony.chord_quality.complexity")
    print("  ✓ Registered: melody.contour.smoothness")
    print("  ✓ Registered: rhythm.subdivision.level")

    print("\nInterpretation complete! 4/5 features successfully interpreted.")


def example_5_integration_with_registry():
    """Example 5: Integration with UniversalParameterRegistry"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Integration with Parameter Registry")
    print("=" * 80)

    print("\nAfter interpretation, Agent 6 registers discovered parameters")
    print("with the UniversalParameterRegistry for use throughout the system.")

    try:
        registry = UniversalParameterRegistry()
        print(f"\n✓ Connected to UniversalParameterRegistry")
        print(f"  Current parameters: {len(registry.get_all_parameters())}")

        print("\nDiscovered parameters would be added as:")
        print("  - Full path: rhythm.syncopation.strength")
        print("  - Type: PROBABILITY (0.0-1.0)")
        print("  - Category: RHYTHM")
        print("  - Impact: HIGH")
        print("  - Extraction function: extract_feature_0(midi_path)")
        print("  - Learnable: True")

    except Exception as e:
        print(f"\n✗ Could not connect to registry: {e}")


def example_6_usage_in_pipeline():
    """Example 6: Usage in Agent 7's pipeline"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Usage in SemanticDiscoveryPipeline (Agent 7)")
    print("=" * 80)

    print("\nAgent 6 integrates into Agent 7's pipeline as follows:")

    print("""
    # Agent 7: SemanticDiscoveryPipeline
    from midi_generator.learning.semantic_discovery_pipeline import SemanticDiscoveryPipeline
    from midi_generator.learning.feature_interpreter import FeatureInterpreter

    # Create pipeline
    pipeline = SemanticDiscoveryPipeline(
        midi_corpus_dir=Path("data/midi"),
        output_dir=Path("output/discovery")
    )

    # Run pipeline (includes Agent 6)
    results = pipeline.run()

    # Agent 6's work happens in step 4:
    # Step 1: Compute gaps (Agent 4)
    # Step 2: Train encoder (Agent 5)
    # Step 3: Extract features (Agent 5)
    # Step 4: Interpret features (Agent 6) ← HERE
    # Step 5: Validate features (Agent 8)
    # Step 6: Evaluate and report (Agent 9)

    # Access interpretations
    interpretations = results['interpretations']
    for interp in interpretations:
        print(f"Discovered: {interp.parameter_name}")
        print(f"  Modality: {interp.modality}")
        print(f"  Confidence: {interp.confidence:.2f}")
        print(f"  Concept: {interp.concept.name if interp.concept else 'unknown'}")

    # Use discovered parameters
    from midi_generator.parameters.universal_registry import UniversalParameterRegistry
    registry = UniversalParameterRegistry()

    for interp in interpretations:
        if interp.parameter_name in registry.get_all_parameters():
            extractor = registry.get(interp.parameter_name).validation_function
            value = extractor("my_song.mid")
            print(f"{interp.parameter_name}: {value}")
    """)


def main():
    """Run all examples"""
    print("*" * 80)
    print("SEMANTIC FEATURE INTERPRETATION - AGENT 6")
    print("Integration Examples")
    print("*" * 80)

    if not IMPORTS_AVAILABLE:
        return

    # Run examples
    example_1_test_patterns()
    example_2_concept_matching()
    example_3_parameter_naming()
    example_4_full_interpretation()
    example_5_integration_with_registry()
    example_6_usage_in_pipeline()

    print("\n" + "*" * 80)
    print("Examples complete!")
    print("*" * 80)

    print("\nNext steps:")
    print("1. Agent 5 trains semantic features on MIDI corpus")
    print("2. Run Agent 6 to interpret learned features:")
    print("   interpreter = FeatureInterpreter()")
    print("   interpretations = interpreter.interpret_features(feature_bank, encoder)")
    print("3. Register discovered parameters with UniversalParameterRegistry")
    print("4. Use parameters in generation or analysis")


if __name__ == "__main__":
    main()
