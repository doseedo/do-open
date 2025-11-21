#!/usr/bin/env python3
"""
Agent 8: Semantic Feature Validation Example
============================================

Demonstrates how to use the SemanticFeatureValidator to validate
discovered features during the semantic discovery pipeline.

This example shows integration with:
- Agent 1: Musical locality functions
- Agent 2: Semantic feature representations
- Agent 6: Feature interpretation

Usage:
    python examples/semantic_discovery/agent8_validation_example.py

Author: Agent 8 - Semantic Feature Discovery
Date: 2025-11-21
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from midi_generator.learning.semantic_constraints import (
    SemanticFeatureValidator,
    MusicalValidityRules,
    ValidationSeverity
)


def example_1_basic_validation():
    """Example 1: Basic feature validation"""
    print("=" * 80)
    print("EXAMPLE 1: BASIC FEATURE VALIDATION")
    print("=" * 80)

    # Create validator
    validator = SemanticFeatureValidator(min_validity_score=0.7)

    # Simulate a discovered feature from Agent 3 (encoder)
    # with interpretation from Agent 6
    feature_interpretation = {
        'name': 'swing_feel',
        'domain': 'rhythm',
        'description': 'Detects swing rhythm patterns with triplet eighth note feel'
    }

    # Simulate feature activations on MIDI corpus
    # In reality, these would come from encoder.get_activations(feature_id)
    feature_activations = np.random.rand(1000)  # 1000 MIDI files

    # Validate the feature
    result = validator.validate_feature(
        feature_id=1,
        activations=feature_activations,
        interpretation=feature_interpretation
    )

    # Display results
    print(f"\nFeature: {feature_interpretation['name']}")
    print(f"Valid: {result.is_valid}")
    print(f"Overall Score: {result.score:.3f}")
    print(f"  Musical Validity: {result.musical_validity_score:.3f}")
    print(f"  Locality Consistency: {result.locality_consistency_score:.3f}")
    print(f"  Redundancy Score: {result.redundancy_score:.3f}")

    if result.issues:
        print(f"\nIssues ({len(result.issues)}):")
        for issue in result.issues:
            print(f"  [{issue.severity.value.upper()}] {issue.message}")

    if result.suggested_fixes:
        print(f"\nSuggested Improvements:")
        for suggestion in result.suggested_fixes:
            print(f"  • {suggestion}")

    return validator


def example_2_invalid_feature(validator):
    """Example 2: Detecting invalid features"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: DETECTING INVALID FEATURES")
    print("=" * 80)

    # Feature with non-musical name
    invalid_interpretation = {
        'name': 'random_activations',
        'domain': 'unknown',
        'description': 'Random pattern'
    }

    result = validator.validate_feature(
        feature_id=2,
        activations=np.random.rand(1000),
        interpretation=invalid_interpretation
    )

    print(f"\nFeature: {invalid_interpretation['name']}")
    print(f"Valid: {result.is_valid}")
    print(f"Score: {result.score:.3f}")

    if result.has_critical_issues:
        print("\n❌ CRITICAL ISSUES DETECTED:")
        for issue in result.issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                print(f"  • {issue.message}")


def example_3_redundancy_detection(validator):
    """Example 3: Redundancy detection"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: REDUNDANCY DETECTION")
    print("=" * 80)

    # First, validate an original feature
    original_interpretation = {
        'name': 'chord_voicing',
        'domain': 'harmony',
        'description': 'Identifies chord voicing patterns'
    }
    original_activations = np.random.rand(1000)

    result1 = validator.validate_feature(
        feature_id=10,
        activations=original_activations,
        interpretation=original_interpretation
    )

    print(f"\n1st Feature: {original_interpretation['name']}")
    print(f"   Valid: {result1.is_valid}")
    print(f"   Redundancy: {result1.redundancy_score:.3f}")

    # Now try a nearly identical feature
    similar_interpretation = {
        'name': 'chord_voicing_variant',
        'domain': 'harmony',
        'description': 'Similar chord voicing detection'
    }
    # Create highly correlated activations
    similar_activations = original_activations + np.random.rand(1000) * 0.05

    result2 = validator.validate_feature(
        feature_id=11,
        activations=similar_activations,
        interpretation=similar_interpretation
    )

    print(f"\n2nd Feature: {similar_interpretation['name']}")
    print(f"   Valid: {result2.is_valid}")
    print(f"   Redundancy: {result2.redundancy_score:.3f}")

    if result2.redundancy_score < 0.7:
        print("\n   ⚠️  REDUNDANCY DETECTED:")
        for issue in result2.issues:
            if 'correlation' in issue.message.lower():
                print(f"      {issue.message}")


def example_4_batch_validation(validator):
    """Example 4: Batch validation of multiple features"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: BATCH VALIDATION")
    print("=" * 80)

    # Simulate multiple discovered features
    discovered_features = [
        {'id': 20, 'name': 'swing_intensity', 'domain': 'rhythm',
         'description': 'Measures strength of swing feel'},
        {'id': 21, 'name': 'chord_complexity', 'domain': 'harmony',
         'description': 'Complexity of chord progressions'},
        {'id': 22, 'name': 'melody_contour', 'domain': 'pitch',
         'description': 'Overall melodic shape and direction'},
        {'id': 23, 'name': 'constant_feature', 'domain': 'unknown',
         'description': 'Always the same'},  # This should be invalid
    ]

    valid_features = []
    invalid_features = []

    for feature in discovered_features:
        interpretation = {
            'name': feature['name'],
            'domain': feature['domain'],
            'description': feature['description']
        }

        # Simulate activations
        activations = np.random.rand(1000)

        result = validator.validate_feature(
            feature_id=feature['id'],
            activations=activations,
            interpretation=interpretation
        )

        if result.is_valid:
            valid_features.append(feature)
        else:
            invalid_features.append((feature, result))

    print(f"\nValidation Results:")
    print(f"  Total features: {len(discovered_features)}")
    print(f"  Valid: {len(valid_features)}")
    print(f"  Invalid: {len(invalid_features)}")

    if invalid_features:
        print(f"\nInvalid Features:")
        for feature, result in invalid_features:
            print(f"  ✗ {feature['name']} (score: {result.score:.2f})")
            if result.has_critical_issues:
                for issue in result.issues:
                    if issue.severity == ValidationSeverity.CRITICAL:
                        print(f"      - {issue.message}")


def example_5_validation_summary(validator):
    """Example 5: Validation summary statistics"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: VALIDATION SUMMARY")
    print("=" * 80)

    # Get overall validation statistics
    summary = validator.get_validation_summary()

    print(f"\nOverall Validation Statistics:")
    print(f"  Total features validated: {summary['total']}")
    print(f"  Valid: {summary['valid']} ({summary['valid_percentage']:.1f}%)")
    print(f"  Invalid: {summary['invalid']}")
    print(f"\nAverage Scores:")
    print(f"  Overall: {summary['average_score']:.3f}")
    print(f"  Musical Validity: {summary['avg_musical_score']:.3f}")
    print(f"  Locality Consistency: {summary['avg_locality_score']:.3f}")
    print(f"  Redundancy: {summary['avg_redundancy_score']:.3f}")


def example_6_musical_domains():
    """Example 6: Understanding musical domains"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: MUSICAL DOMAINS")
    print("=" * 80)

    print("\nValid Musical Domains:")
    for domain, keywords in MusicalValidityRules.VALID_DOMAINS.items():
        print(f"\n  {domain.upper()}:")
        print(f"    Keywords: {', '.join(keywords)}")

    print("\n\nInvalid Pattern Detection:")
    test_patterns = [
        'swing_feel',       # Valid - musical
        'chord_progression', # Valid - musical
        'always_on',        # Invalid - trivial
        'random',           # Invalid - trivial
        'file_size',        # Invalid - non-musical
    ]

    for pattern in test_patterns:
        is_valid = MusicalValidityRules.is_musically_meaningful(pattern)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"  {pattern:20s} → {status}")


def example_7_integration_with_csp():
    """Example 7: Integration with CSP solver"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: CSP INTEGRATION")
    print("=" * 80)

    try:
        from midi_generator.algorithms.constraint_solver import (
            SemanticFeatureConstraint,
            Variable,
            CSPSolver
        )

        print("\n✓ SemanticFeatureConstraint successfully imported")
        print("\nUsage Example:")
        print("""
validator = SemanticFeatureValidator()

# Create constraint for semantic feature validation
constraint = SemanticFeatureConstraint(
    variable='feature_1',
    validator=validator,
    min_validity_score=0.7
)

# Use in CSP
variables = [Variable('feature_1', domain=feature_candidates)]
constraints = [constraint]
solver = CSPSolver(variables, constraints)
solution = solver.solve()  # Only valid features accepted
        """)

    except ImportError as e:
        print(f"\n✗ Could not import CSP integration: {e}")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("AGENT 8: SEMANTIC FEATURE VALIDATION EXAMPLES")
    print("=" * 80)

    # Run examples
    validator = example_1_basic_validation()
    example_2_invalid_feature(validator)
    example_3_redundancy_detection(validator)
    example_4_batch_validation(validator)
    example_5_validation_summary(validator)
    example_6_musical_domains()
    example_7_integration_with_csp()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Validator checks musical validity, locality, and redundancy")
    print("  2. Features need clear musical names and interpretations")
    print("  3. Redundant features are automatically detected")
    print("  4. Validation provides actionable feedback")
    print("  5. Integration with CSP solver available")
    print("\nFor complete documentation, see:")
    print("  docs/AGENT_8_SEMANTIC_CONSTRAINTS.md")
    print("=" * 80)


if __name__ == '__main__':
    main()
