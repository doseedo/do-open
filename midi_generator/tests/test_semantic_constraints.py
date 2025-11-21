#!/usr/bin/env python3
"""
Tests for Semantic Feature Constraint Validation
================================================

Tests for Agent 8 - Semantic Feature Discovery constraint validation system.

Author: Agent 8
Date: 2025-11-21
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from learning.semantic_constraints import (
    SemanticFeatureValidator,
    ValidationResult,
    ValidationSeverity,
    ValidationIssue,
    MusicalValidityRules,
    LocalityConsistencyChecker,
    RedundancyDetector
)


class TestMusicalValidityRules(unittest.TestCase):
    """Test musical validity rules and pattern matching"""

    def test_valid_domains(self):
        """Test that musical domains are properly defined"""
        self.assertIn('pitch', MusicalValidityRules.VALID_DOMAINS)
        self.assertIn('rhythm', MusicalValidityRules.VALID_DOMAINS)
        self.assertIn('harmony', MusicalValidityRules.VALID_DOMAINS)
        self.assertIn('dynamics', MusicalValidityRules.VALID_DOMAINS)

    def test_domain_keywords(self):
        """Test domain keyword retrieval"""
        pitch_keywords = MusicalValidityRules.get_domain_keywords('pitch')
        self.assertIn('melody', pitch_keywords)
        self.assertIn('harmony', pitch_keywords)

        rhythm_keywords = MusicalValidityRules.get_domain_keywords('rhythm')
        self.assertIn('beat', rhythm_keywords)
        self.assertIn('swing', rhythm_keywords)

    def test_musically_meaningful_patterns(self):
        """Test detection of musically meaningful patterns"""
        # Valid musical patterns
        self.assertTrue(MusicalValidityRules.is_musically_meaningful('swing_feel'))
        self.assertTrue(MusicalValidityRules.is_musically_meaningful('chord_progression'))
        self.assertTrue(MusicalValidityRules.is_musically_meaningful('melody_contour'))
        self.assertTrue(MusicalValidityRules.is_musically_meaningful('rhythm_pattern'))

        # Invalid patterns (anti-patterns)
        self.assertFalse(MusicalValidityRules.is_musically_meaningful('always_on'))
        self.assertFalse(MusicalValidityRules.is_musically_meaningful('random'))
        self.assertFalse(MusicalValidityRules.is_musically_meaningful('file_size'))
        self.assertFalse(MusicalValidityRules.is_musically_meaningful('constant'))


class TestRedundancyDetector(unittest.TestCase):
    """Test redundancy detection functionality"""

    def setUp(self):
        """Set up redundancy detector"""
        self.detector = RedundancyDetector(
            correlation_threshold=0.95,
            r2_threshold=0.90
        )

    def test_no_redundancy_first_feature(self):
        """First feature should never be redundant"""
        activations = np.random.rand(100)
        score, issues = self.detector.check_redundancy(1, activations)

        self.assertEqual(score, 1.0)
        self.assertEqual(len(issues), 0)

    def test_redundancy_with_identical_feature(self):
        """Identical features should be detected as redundant"""
        # Add first feature
        activations1 = np.random.rand(100)
        self.detector.add_known_feature(1, activations1)

        # Check identical feature
        score, issues = self.detector.check_redundancy(2, activations1.copy())

        self.assertLess(score, 0.1)  # Should be highly redundant
        self.assertGreater(len(issues), 0)
        self.assertIn('correlation', issues[0].lower())

    def test_no_redundancy_with_different_feature(self):
        """Different features should not be redundant"""
        # Add first feature
        activations1 = np.random.rand(100)
        self.detector.add_known_feature(1, activations1)

        # Create uncorrelated feature
        activations2 = np.random.rand(100)
        score, issues = self.detector.check_redundancy(2, activations2)

        self.assertGreater(score, 0.5)  # Should not be very redundant

    def test_partial_redundancy(self):
        """Partially correlated features should have intermediate score"""
        # Add first feature
        activations1 = np.random.rand(100)
        self.detector.add_known_feature(1, activations1)

        # Create partially correlated feature
        activations2 = 0.7 * activations1 + 0.3 * np.random.rand(100)
        score, issues = self.detector.check_redundancy(2, activations2)

        # Should have some redundancy but not complete
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


class TestLocalityConsistencyChecker(unittest.TestCase):
    """Test locality consistency checking"""

    def setUp(self):
        """Set up locality checker without locality functions"""
        self.checker = LocalityConsistencyChecker(locality_functions=None)

    def test_no_locality_functions_returns_valid(self):
        """Without locality functions, should return valid by default"""
        def dummy_activation(x):
            return 0.5

        score, violations = self.checker.check_locality_consistency(
            feature_id=1,
            activation_function=dummy_activation,
            test_midi_data=None
        )

        self.assertEqual(score, 1.0)
        self.assertEqual(len(violations), 0)


class TestSemanticFeatureValidator(unittest.TestCase):
    """Test complete semantic feature validator"""

    def setUp(self):
        """Set up validator"""
        self.validator = SemanticFeatureValidator(
            locality_functions=None,
            correlation_threshold=0.95,
            min_validity_score=0.7
        )

    def test_valid_feature_with_good_interpretation(self):
        """Feature with good musical interpretation should be valid"""
        interpretation = {
            'name': 'swing_feel',
            'domain': 'rhythm',
            'description': 'Detects swing rhythm patterns with triplet feel'
        }

        activations = np.random.rand(100) * 0.6 + 0.2  # Moderate activations

        result = self.validator.validate_feature(
            feature_id=1,
            activations=activations,
            interpretation=interpretation
        )

        self.assertTrue(result.is_valid)
        self.assertGreater(result.score, 0.7)
        self.assertGreater(result.musical_validity_score, 0.7)

    def test_invalid_feature_with_non_musical_name(self):
        """Feature with non-musical name should be invalid"""
        interpretation = {
            'name': 'random_noise',
            'domain': 'unknown',
            'description': 'Random pattern'
        }

        activations = np.random.rand(100)

        result = self.validator.validate_feature(
            feature_id=1,
            activations=activations,
            interpretation=interpretation
        )

        self.assertFalse(result.is_valid)
        self.assertTrue(result.has_critical_issues)
        self.assertEqual(result.musical_validity_score, 0.0)

    def test_feature_without_interpretation(self):
        """Feature without interpretation should have warning"""
        activations = np.random.rand(100)

        result = self.validator.validate_feature(
            feature_id=1,
            activations=activations,
            interpretation=None
        )

        # Should have low musical validity score
        self.assertEqual(result.musical_validity_score, 0.5)
        self.assertTrue(result.has_warnings)

    def test_redundant_features_detected(self):
        """Redundant features should be flagged"""
        interpretation1 = {
            'name': 'swing_feel',
            'domain': 'rhythm',
            'description': 'Swing rhythm detection'
        }

        activations1 = np.random.rand(100)

        # Add first feature
        result1 = self.validator.validate_feature(
            feature_id=1,
            activations=activations1,
            interpretation=interpretation1
        )
        self.assertTrue(result1.is_valid)

        # Try to add nearly identical feature
        interpretation2 = {
            'name': 'swing_feel_variant',
            'domain': 'rhythm',
            'description': 'Similar swing detection'
        }

        activations2 = activations1 + np.random.rand(100) * 0.01  # Almost identical

        result2 = self.validator.validate_feature(
            feature_id=2,
            activations=activations2,
            interpretation=interpretation2
        )

        # Should detect redundancy
        self.assertLess(result2.redundancy_score, 0.5)

    def test_multiple_features_tracked(self):
        """Validator should track all validated features"""
        for i in range(5):
            interpretation = {
                'name': f'feature_{i}',
                'domain': 'rhythm',
                'description': f'Feature {i} description'
            }
            activations = np.random.rand(100)

            self.validator.validate_feature(
                feature_id=i,
                activations=activations,
                interpretation=interpretation
            )

        # Check validation history
        self.assertEqual(len(self.validator.validation_history), 5)

        # Get summary
        summary = self.validator.get_validation_summary()
        self.assertEqual(summary['total'], 5)
        self.assertIn('valid', summary)
        self.assertIn('invalid', summary)
        self.assertIn('average_score', summary)

    def test_suggested_fixes_generated(self):
        """Validator should generate helpful suggestions"""
        interpretation = {
            'name': 'feature_x',  # Not very musical
            'domain': 'unknown',
            'description': 'Short'  # Too brief
        }

        result = self.validator.validate_feature(
            feature_id=1,
            activations=np.random.rand(100),
            interpretation=interpretation
        )

        # Should have suggestions
        self.assertGreater(len(result.suggested_fixes), 0)

    def test_validation_result_properties(self):
        """Test ValidationResult properties"""
        result = ValidationResult(
            feature_id=1,
            is_valid=False,
            score=0.5,
            issues=[
                ValidationIssue(ValidationSeverity.CRITICAL, 'test', 'Critical issue'),
                ValidationIssue(ValidationSeverity.WARNING, 'test', 'Warning issue')
            ]
        )

        self.assertTrue(result.has_critical_issues)
        self.assertTrue(result.has_warnings)
        self.assertFalse(result.is_valid)

    def test_domain_validation(self):
        """Test that invalid domains trigger warnings"""
        interpretation = {
            'name': 'swing_feel',
            'domain': 'invalid_domain',
            'description': 'Good description'
        }

        result = self.validator.validate_feature(
            feature_id=1,
            activations=np.random.rand(100),
            interpretation=interpretation
        )

        # Should have warning about invalid domain
        self.assertTrue(result.has_warnings)
        domain_warning = any(
            'domain' in issue.message.lower()
            for issue in result.issues
            if issue.severity == ValidationSeverity.WARNING
        )
        self.assertTrue(domain_warning)


class TestValidationSummary(unittest.TestCase):
    """Test validation summary functionality"""

    def test_empty_summary(self):
        """Empty validator should return zero summary"""
        validator = SemanticFeatureValidator()
        summary = validator.get_validation_summary()

        self.assertEqual(summary['total'], 0)
        self.assertEqual(summary['valid'], 0)
        self.assertEqual(summary['invalid'], 0)

    def test_summary_statistics(self):
        """Summary should calculate correct statistics"""
        validator = SemanticFeatureValidator()

        # Add mix of valid and invalid features
        valid_interpretations = [
            {'name': 'swing_feel', 'domain': 'rhythm', 'description': 'Swing detection'},
            {'name': 'chord_voicing', 'domain': 'harmony', 'description': 'Voicing analysis'},
        ]

        invalid_interpretations = [
            {'name': 'random', 'domain': 'unknown', 'description': 'Random'},
        ]

        for i, interp in enumerate(valid_interpretations):
            validator.validate_feature(
                feature_id=i,
                activations=np.random.rand(100),
                interpretation=interp
            )

        for i, interp in enumerate(invalid_interpretations, start=len(valid_interpretations)):
            validator.validate_feature(
                feature_id=i,
                activations=np.random.rand(100),
                interpretation=interp
            )

        summary = validator.get_validation_summary()

        self.assertEqual(summary['total'], 3)
        self.assertEqual(summary['valid'], 2)
        self.assertEqual(summary['invalid'], 1)
        self.assertAlmostEqual(summary['valid_percentage'], 66.67, places=1)
        self.assertGreater(summary['average_score'], 0.0)


class TestIntegrationWithCSP(unittest.TestCase):
    """Test integration with constraint solver"""

    def test_semantic_feature_constraint_import(self):
        """Test that SemanticFeatureConstraint can be imported"""
        try:
            from algorithms.constraint_solver import (
                SemanticFeatureConstraint,
                SEMANTIC_VALIDATION_AVAILABLE
            )
            self.assertTrue(SEMANTIC_VALIDATION_AVAILABLE)
        except ImportError as e:
            self.fail(f"Failed to import SemanticFeatureConstraint: {e}")

    def test_semantic_feature_constraint_creation(self):
        """Test creating SemanticFeatureConstraint"""
        from algorithms.constraint_solver import SemanticFeatureConstraint

        validator = SemanticFeatureValidator()
        constraint = SemanticFeatureConstraint(
            variable='feature_1',
            validator=validator,
            min_validity_score=0.7
        )

        self.assertEqual(constraint.variables, ['feature_1'])
        self.assertEqual(constraint.min_validity_score, 0.7)

    def test_semantic_feature_constraint_validation(self):
        """Test SemanticFeatureConstraint validation logic"""
        from algorithms.constraint_solver import SemanticFeatureConstraint

        validator = SemanticFeatureValidator()
        constraint = SemanticFeatureConstraint(
            variable='feature_1',
            validator=validator,
            min_validity_score=0.7
        )

        # Valid feature
        valid_assignment = {
            'feature_1': {
                'id': 1,
                'interpretation': {
                    'name': 'swing_feel',
                    'domain': 'rhythm',
                    'description': 'Swing rhythm detection'
                },
                'activations': np.random.rand(100)
            }
        }

        self.assertTrue(constraint.is_satisfied(valid_assignment))

        # Invalid feature
        invalid_assignment = {
            'feature_1': {
                'id': 2,
                'interpretation': {
                    'name': 'random_noise',
                    'domain': 'unknown',
                    'description': 'Random'
                },
                'activations': np.random.rand(100)
            }
        }

        self.assertFalse(constraint.is_satisfied(invalid_assignment))


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMusicalValidityRules))
    suite.addTests(loader.loadTestsFromTestCase(TestRedundancyDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestLocalityConsistencyChecker))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticFeatureValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationSummary))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWithCSP))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
