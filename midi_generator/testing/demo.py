#!/usr/bin/env python3
"""
Agent 26: Test Case Generator - Demo
=====================================

Demonstration of automated test generation for musical parameters.

This script shows:
1. Creating parameter proposals
2. Generating test suites
3. Validating generated tests
4. Running test suites

Usage:
    python demo.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from midi_generator.testing import (
    TestCaseGenerator,
    ParameterProposal,
    TestConfiguration,
    TestSuiteValidator,
    TestExecutor
)


def demo_continuous_parameter():
    """Demo: Generate tests for a continuous parameter."""
    print("\n" + "="*70)
    print("DEMO 1: Continuous Parameter - Swing Ratio")
    print("="*70)

    # Create parameter proposal
    param = ParameterProposal(
        name="rhythm.swing.enhanced_ratio",
        description="Enhanced swing ratio with extended range (0.5=straight, 0.67=standard, 0.8=extreme)",
        param_type="continuous",
        default=0.67,
        min_value=0.5,
        max_value=0.8,
        category="rhythm",
        impact="critical",
        genres=["jazz", "swing", "blues", "shuffle"],
        affected_features=[
            "rhythmic_feel",
            "groove_quantization",
            "temporal_displacement",
            "swing_percentage"
        ],
        musical_properties=[
            "Defines the swing feel intensity",
            "0.5 = straight eighth notes",
            "0.67 = standard swing (2:1 ratio)",
            "0.75 = hard swing",
            "0.8 = extreme swing"
        ]
    )

    # Generate test suite
    generator = TestCaseGenerator()
    test_path = generator.save_test_suite(param)

    print(f"\n✓ Generated test suite: {test_path}")
    print(f"  Parameter: {param.name}")
    print(f"  Type: {param.param_type}")
    print(f"  Range: [{param.min_value}, {param.max_value}]")

    # Validate
    validator = TestSuiteValidator()
    is_valid = validator.validate_and_report(test_path)

    return test_path, is_valid


def demo_categorical_parameter():
    """Demo: Generate tests for a categorical parameter."""
    print("\n" + "="*70)
    print("DEMO 2: Categorical Parameter - Modal Scale Selection")
    print("="*70)

    param = ParameterProposal(
        name="harmony.modal.scale_type",
        description="Modal scale selection for harmonic generation",
        param_type="categorical",
        default="dorian",
        options=["dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"],
        category="harmony",
        impact="critical",
        genres=["jazz", "modal_jazz", "fusion", "world"],
        affected_features=[
            "scale_type_distribution",
            "characteristic_interval_patterns",
            "modal_cadence_types",
            "harmonic_color"
        ],
        musical_properties=[
            "Defines the modal center and characteristic intervals",
            "Each mode has unique emotional quality",
            "Affects available chord progressions",
            "Common in modal jazz (Miles Davis, John Coltrane)"
        ]
    )

    generator = TestCaseGenerator()
    test_path = generator.save_test_suite(param)

    print(f"\n✓ Generated test suite: {test_path}")
    print(f"  Parameter: {param.name}")
    print(f"  Type: {param.param_type}")
    print(f"  Options: {', '.join(param.options)}")

    validator = TestSuiteValidator()
    is_valid = validator.validate_and_report(test_path)

    return test_path, is_valid


def demo_boolean_parameter():
    """Demo: Generate tests for a boolean parameter."""
    print("\n" + "="*70)
    print("DEMO 3: Boolean Parameter - Voice Leading Optimization")
    print("="*70)

    param = ParameterProposal(
        name="harmony.voice_leading.optimize",
        description="Enable voice leading optimization for smooth chord transitions",
        param_type="boolean",
        default=True,
        category="harmony",
        impact="high",
        genres=["jazz", "classical", "choral"],
        affected_features=[
            "voice_leading_cost",
            "common_tone_retention",
            "stepwise_motion_percentage",
            "parallel_motion_avoidance"
        ],
        musical_properties=[
            "Minimizes voice motion between chords",
            "Retains common tones when possible",
            "Prefers stepwise motion over leaps",
            "Essential for smooth harmonic flow"
        ]
    )

    generator = TestCaseGenerator()
    test_path = generator.save_test_suite(param)

    print(f"\n✓ Generated test suite: {test_path}")
    print(f"  Parameter: {param.name}")
    print(f"  Type: {param.param_type}")
    print(f"  Default: {param.default}")

    validator = TestSuiteValidator()
    is_valid = validator.validate_and_report(test_path)

    return test_path, is_valid


def demo_probability_parameter():
    """Demo: Generate tests for a probability parameter."""
    print("\n" + "="*70)
    print("DEMO 4: Probability Parameter - Chromatic Passing Tone Density")
    print("="*70)

    param = ParameterProposal(
        name="melody.chromatic.passing_tone_density",
        description="Probability of chromatic passing tones in melodic lines",
        param_type="probability",
        default=0.25,
        min_value=0.0,
        max_value=1.0,
        category="melody",
        impact="high",
        genres=["bebop", "jazz", "classical", "romantic"],
        affected_features=[
            "chromatic_note_percentage",
            "melodic_complexity",
            "diatonic_vs_chromatic_ratio",
            "passing_tone_frequency"
        ],
        musical_properties=[
            "Adds chromatic color to melodies",
            "0.0 = fully diatonic",
            "0.25 = moderate chromaticism",
            "1.0 = highly chromatic (bebop style)"
        ]
    )

    generator = TestCaseGenerator()
    test_path = generator.save_test_suite(param)

    print(f"\n✓ Generated test suite: {test_path}")
    print(f"  Parameter: {param.name}")
    print(f"  Type: {param.param_type}")
    print(f"  Range: [0.0, 1.0]")

    validator = TestSuiteValidator()
    is_valid = validator.validate_and_report(test_path)

    return test_path, is_valid


def demo_integer_parameter():
    """Demo: Generate tests for an integer parameter."""
    print("\n" + "="*70)
    print("DEMO 5: Integer Parameter - Harmonic Extension Limit")
    print("="*70)

    param = ParameterProposal(
        name="harmony.extensions.max_extension",
        description="Maximum chord extension (7, 9, 11, 13)",
        param_type="integer",
        default=9,
        min_value=7,
        max_value=13,
        category="harmony",
        impact="high",
        genres=["jazz", "fusion", "contemporary"],
        affected_features=[
            "chord_extension_distribution",
            "harmonic_density",
            "dissonance_level",
            "voicing_complexity"
        ],
        musical_properties=[
            "Controls harmonic sophistication",
            "7 = basic seventh chords",
            "9 = adds ninth extensions",
            "11 = adds eleventh extensions",
            "13 = full extended harmony"
        ]
    )

    generator = TestCaseGenerator()
    test_path = generator.save_test_suite(param)

    print(f"\n✓ Generated test suite: {test_path}")
    print(f"  Parameter: {param.name}")
    print(f"  Type: {param.param_type}")
    print(f"  Range: [{param.min_value}, {param.max_value}]")

    validator = TestSuiteValidator()
    is_valid = validator.validate_and_report(test_path)

    return test_path, is_valid


def demo_from_json():
    """Demo: Load parameter from JSON and generate tests."""
    print("\n" + "="*70)
    print("DEMO 6: Load from JSON - Jazz Tritone Substitution")
    print("="*70)

    import json

    # Load example parameter
    json_path = Path(__file__).parent / "example_parameter.json"

    with open(json_path, 'r') as f:
        param_data = json.load(f)

    param = ParameterProposal.from_dict(param_data)

    generator = TestCaseGenerator()
    test_path = generator.save_test_suite(param)

    print(f"\n✓ Generated test suite: {test_path}")
    print(f"  Loaded from: {json_path}")
    print(f"  Parameter: {param.name}")

    validator = TestSuiteValidator()
    is_valid = validator.validate_and_report(test_path)

    return test_path, is_valid


def demo_custom_configuration():
    """Demo: Generate tests with custom configuration."""
    print("\n" + "="*70)
    print("DEMO 7: Custom Configuration")
    print("="*70)

    # Create custom config
    config = TestConfiguration(
        output_dir=Path("tests/generated/custom"),
        include_fixtures=True,
        include_benchmarks=True,
        generate_integration_tests=True,
        generate_regression_tests=True,
        verbose=True,
        pytest_marks=["slow", "integration"],
        timeout=60
    )

    param = ParameterProposal(
        name="dynamics.crescendo.rate",
        description="Rate of crescendo dynamics (dB per second)",
        param_type="continuous",
        default=2.0,
        min_value=0.5,
        max_value=10.0,
        category="dynamics",
        impact="medium"
    )

    generator = TestCaseGenerator(config)
    test_path = generator.save_test_suite(param)

    print(f"\n✓ Generated test suite: {test_path}")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Timeout: {config.timeout}s")

    validator = TestSuiteValidator()
    is_valid = validator.validate_and_report(test_path)

    return test_path, is_valid


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("AGENT 26: TEST CASE GENERATOR - COMPREHENSIVE DEMO")
    print("="*70)
    print("\nThis demo generates test suites for various parameter types.")
    print("Each test suite includes:")
    print("  - Boundary tests")
    print("  - Musical validity tests")
    print("  - Feature impact tests")
    print("  - Integration tests")
    print("  - Regression tests")

    demos = [
        ("Continuous Parameter", demo_continuous_parameter),
        ("Categorical Parameter", demo_categorical_parameter),
        ("Boolean Parameter", demo_boolean_parameter),
        ("Probability Parameter", demo_probability_parameter),
        ("Integer Parameter", demo_integer_parameter),
        ("Load from JSON", demo_from_json),
        ("Custom Configuration", demo_custom_configuration),
    ]

    results = []

    for name, demo_func in demos:
        try:
            test_path, is_valid = demo_func()
            results.append((name, test_path, is_valid))
        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            results.append((name, None, False))

    # Summary
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)

    for name, test_path, is_valid in results:
        status = "✓" if is_valid else "✗"
        print(f"{status} {name}")
        if test_path:
            print(f"    → {test_path}")

    successful = sum(1 for _, _, valid in results if valid)
    print(f"\nSuccessful: {successful}/{len(results)}")

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Review generated test files in tests/generated/")
    print("2. Run tests with: pytest tests/generated/test_*.py -v")
    print("3. Integrate successful tests into your test suite")
    print("4. Customize test generation for your specific needs")

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
