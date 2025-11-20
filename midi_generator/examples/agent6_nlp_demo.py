#!/usr/bin/env python3
"""
Agent 6 - Natural Language Parameter Predictor - Demonstration
===============================================================

This example demonstrates the Natural Language Parameter Predictor system,
which converts natural language descriptions into precise parameter values
for music generation.

Features Demonstrated:
1. Converting text descriptions to 515+ parameters
2. Style database with 100+ examples
3. Concept extraction from natural language
4. Parameter validation and defaults
5. Similar style matching
6. Integration with music generation

Author: Agent 6 - Natural Language Parameter Predictor
Date: 2025-11-20
"""

import os
import json
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from midi_generator.learning.natural_language_predictor import (
    NaturalLanguageParameterPredictor,
    StyleDatabase,
    ConceptExtractor,
    MusicalConcepts,
    predict_from_text,
    get_style_database
)
from midi_generator.parameters.universal_registry import REGISTRY


def demo_style_database():
    """Demonstrate the style database"""

    print("\n" + "=" * 80)
    print("DEMO 1: Style Database")
    print("=" * 80)

    db = get_style_database()

    print(f"\n📚 Loaded {len(db.examples)} style examples")

    # Show all example names
    print("\n   Available styles:")
    for i, name in enumerate(sorted(db.examples.keys()), 1):
        example = db.examples[name]
        print(f"   {i:2d}. {name:25s} - {example.description[:60]}")

    # Show detailed example
    print("\n📖 Detailed Example: 'sinatra_ballad'")
    example = db.get_example("sinatra_ballad")
    if example:
        print(f"\n   Name: {example.name}")
        print(f"   Description: {example.description}")
        print(f"   Tags: {', '.join(example.tags)}")
        print(f"\n   Concepts:")
        if example.concepts:
            print(f"      Genre: {example.concepts.genre}")
            print(f"      Era: {example.concepts.era}")
            print(f"      Mood: {example.concepts.mood}")
            print(f"      Tempo: {example.concepts.tempo_descriptor}")
            print(f"      Artists: {', '.join(example.concepts.reference_artists)}")

        print(f"\n   Parameters ({len(example.parameters)}):")
        for param_path, value in list(example.parameters.items())[:10]:
            print(f"      {param_path:45s}: {value}")
        if len(example.parameters) > 10:
            print(f"      ...and {len(example.parameters) - 10} more")


def demo_concept_extraction(api_key: str):
    """Demonstrate concept extraction"""

    print("\n" + "=" * 80)
    print("DEMO 2: Concept Extraction")
    print("=" * 80)

    predictor = NaturalLanguageParameterPredictor(api_key=api_key)

    test_prompts = [
        "Generate a Sinatra-style ballad with lush strings and intimate vocals",
        "Fast bebop piano solo with walking bass and brushed drums",
        "Minimalist ambient soundscape with subtle harmonic movement",
        "Upbeat bossa nova with warm guitar and gentle syncopation",
        "Gritty Chicago blues with electric guitar and shuffle feel",
    ]

    print("\n🤖 Extracting concepts from natural language...")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")

        try:
            concepts = predictor.concept_extractor.extract_concepts(prompt)
            print(f"   ✓ Genre: {concepts.genre or 'N/A'}")
            print(f"   ✓ Subgenre: {concepts.subgenre or 'N/A'}")
            print(f"   ✓ Era: {concepts.era or 'N/A'}")
            print(f"   ✓ Mood: {concepts.mood or 'N/A'}")
            print(f"   ✓ Tempo: {concepts.tempo_descriptor or 'N/A'}")
            print(f"   ✓ Rhythmic feel: {concepts.rhythmic_feel or 'N/A'}")
            if concepts.instrumentation:
                print(f"   ✓ Instruments: {', '.join(concepts.instrumentation)}")
            if concepts.technical_terms:
                print(f"   ✓ Technical terms: {', '.join(concepts.technical_terms)}")
            if concepts.reference_artists:
                print(f"   ✓ Artists: {', '.join(concepts.reference_artists)}")

        except Exception as e:
            print(f"   ✗ Error: {e}")


def demo_similar_style_matching():
    """Demonstrate similar style matching"""

    print("\n" + "=" * 80)
    print("DEMO 3: Similar Style Matching")
    print("=" * 80)

    db = get_style_database()

    # Test concepts
    test_cases = [
        MusicalConcepts(
            genre="jazz",
            era="1950s",
            mood="sultry",
            reference_artists=["Frank Sinatra"]
        ),
        MusicalConcepts(
            genre="jazz",
            subgenre="bebop",
            tempo_descriptor="fast",
            technical_terms=["walking bass"]
        ),
        MusicalConcepts(
            genre="contemporary",
            mood="ambient",
            texture="sparse"
        ),
    ]

    descriptions = [
        "Sinatra-style vocal jazz",
        "Fast bebop with walking bass",
        "Sparse ambient soundscape",
    ]

    print("\n🔍 Finding similar styles for test concepts...")

    for desc, concepts in zip(descriptions, test_cases):
        print(f"\n   Query: '{desc}'")
        similar = db.find_similar_examples(concepts, limit=3)

        print(f"   Top {len(similar)} matches:")
        for i, example in enumerate(similar, 1):
            print(f"      {i}. {example.name:20s} - {example.description[:50]}")


def demo_full_prediction(api_key: str):
    """Demonstrate full parameter prediction"""

    print("\n" + "=" * 80)
    print("DEMO 4: Full Parameter Prediction")
    print("=" * 80)

    predictor = NaturalLanguageParameterPredictor(api_key=api_key)

    test_prompts = [
        "Generate a sultry Sinatra-style ballad with lush orchestration",
        "Create a fast bebop piano solo with walking bass",
        "Make a minimalist ambient piece with subtle textures",
    ]

    print("\n🎵 Predicting parameters from natural language...")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")

        try:
            params = predictor.predict_parameters(prompt)

            print(f"   ✓ Predicted {len(params)} parameters")

            # Show sample parameters
            print(f"\n   Sample parameters:")

            # Group by category for display
            categories = {
                "Harmony": [k for k in params.keys() if k.startswith("harmony.")],
                "Rhythm": [k for k in params.keys() if k.startswith("rhythm.")],
                "Melody": [k for k in params.keys() if k.startswith("melody.")],
                "Dynamics": [k for k in params.keys() if k.startswith("dynamics.")],
                "Bass": [k for k in params.keys() if k.startswith("bass.")],
            }

            for cat_name, cat_params in categories.items():
                if cat_params:
                    print(f"\n   {cat_name}:")
                    for param_path in cat_params[:3]:  # Show 3 per category
                        value = params[param_path]
                        print(f"      {param_path:45s}: {value}")
                    if len(cat_params) > 3:
                        print(f"      ...and {len(cat_params) - 3} more {cat_name.lower()} parameters")

            # Save to JSON for inspection
            output_file = f"/tmp/agent6_params_{i}.json"
            with open(output_file, 'w') as f:
                json.dump(params, f, indent=2)
            print(f"\n   💾 Full parameters saved to: {output_file}")

        except Exception as e:
            print(f"   ✗ Error: {e}")


def demo_parameter_validation():
    """Demonstrate parameter validation"""

    print("\n" + "=" * 80)
    print("DEMO 5: Parameter Validation")
    print("=" * 80)

    from midi_generator.learning.natural_language_predictor import ParameterValidator

    validator = ParameterValidator(REGISTRY)

    # Test with valid and invalid parameters
    test_params = {
        "harmony.voicing.spread": 0.6,  # Valid
        "harmony.voicing.type": "close",  # Valid
        "rhythm.swing.amount": 1.5,  # Invalid (out of range)
        "invalid.parameter.path": 0.5,  # Invalid (doesn't exist)
        "harmony.extensions.use_9ths": True,  # Valid
    }

    print("\n✅ Testing parameter validation...")
    print(f"\n   Input parameters ({len(test_params)}):")
    for param_path, value in test_params.items():
        print(f"      {param_path:40s}: {value}")

    validated, warnings = validator.validate_and_fill_defaults(test_params)

    print(f"\n   Validation results:")
    print(f"      ✓ Validated parameters: {len(validated)}")
    print(f"      ⚠ Warnings: {len(warnings)}")

    if warnings:
        print(f"\n   Warnings:")
        for warning in warnings:
            print(f"      - {warning}")

    print(f"\n   Sample validated parameters:")
    for param_path in list(validated.keys())[:10]:
        value = validated[param_path]
        print(f"      {param_path:40s}: {value}")


def demo_integration_with_generation():
    """Demonstrate integration with music generation"""

    print("\n" + "=" * 80)
    print("DEMO 6: Integration with Music Generation")
    print("=" * 80)

    print("\n🎼 Integration workflow:")
    print("\n   1. User provides natural language description")
    print("      → 'Generate a Sinatra-style ballad'")
    print("\n   2. NaturalLanguageParameterPredictor converts to 515 parameters")
    print("      → {harmony.voicing.type: 'close', rhythm.swing.amount: 0.54, ...}")
    print("\n   3. Parameters passed to HarmonyModule generator")
    print("      → HarmonyModule.generate(parameters)")
    print("\n   4. MIDI file generated with predicted parameters")
    print("      → output.mid")
    print("\n   5. If unsatisfactory, analyze MIDI with feature extractor")
    print("      → deep_feature_extractor.py extracts 1000+ features")
    print("\n   6. Gap detection identifies missing/incorrect parameters")
    print("      → LLM proposes new parameters")
    print("\n   7. System expands automatically")
    print("      → New parameters added to registry")

    print("\n   This demonstrates the self-expanding inverse music generation system!")


def main():
    """Run all demonstrations"""

    print("\n" + "=" * 80)
    print("AGENT 6 - NATURAL LANGUAGE PARAMETER PREDICTOR")
    print("Comprehensive Demonstration")
    print("=" * 80)

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Demo 1: Style Database (no API key required)
    demo_style_database()

    # Demo 3: Similar Style Matching (no API key required)
    demo_similar_style_matching()

    # Demo 5: Parameter Validation (no API key required)
    demo_parameter_validation()

    # Demo 6: Integration (no API key required)
    demo_integration_with_generation()

    # Demos requiring API key
    if api_key:
        # Demo 2: Concept Extraction
        demo_concept_extraction(api_key)

        # Demo 4: Full Prediction
        demo_full_prediction(api_key)

        print("\n" + "=" * 80)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 80)

    else:
        print("\n" + "=" * 80)
        print("⚠️  PARTIAL DEMONSTRATIONS COMPLETED")
        print("=" * 80)
        print("\n   Set ANTHROPIC_API_KEY environment variable to enable:")
        print("   - Concept extraction from natural language")
        print("   - Full parameter prediction with LLM")
        print("\n   Example:")
        print("   export ANTHROPIC_API_KEY='your_api_key_here'")

    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)

    print("""
# Example 1: Simple usage
from learning.natural_language_predictor import predict_from_text

params = predict_from_text("Generate a Sinatra-style ballad")
# Returns dictionary with 515+ parameters

# Example 2: Full control
from learning.natural_language_predictor import NaturalLanguageParameterPredictor

predictor = NaturalLanguageParameterPredictor(api_key="your_key")
params = predictor.predict_parameters("Fast bebop piano solo")

# Example 3: Using concepts directly
from learning.natural_language_predictor import MusicalConcepts

concepts = MusicalConcepts(
    genre="jazz",
    era="1950s",
    mood="sultry",
    reference_artists=["Frank Sinatra"]
)
params = predictor.predict_with_concepts(concepts)

# Example 4: Accessing style database
from learning.natural_language_predictor import get_style_database

db = get_style_database()
example = db.get_example("sinatra_ballad")
params = example.parameters  # Pre-defined parameter set
""")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
