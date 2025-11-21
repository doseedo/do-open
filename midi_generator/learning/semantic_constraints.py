#!/usr/bin/env python3
"""
Semantic Feature Constraint Validation
======================================

Validates discovered semantic features for:
1. Musical validity - represents real musical concepts
2. Locality consistency - respects musical transformations
3. Redundancy detection - not duplicate of existing features

Part of the Semantic Feature Discovery system (Agent 8).

Integration:
- Agent 1: Musical locality transformations
- Agent 2: Semantic feature representations
- Agent 6: Feature interpretation

Author: Agent 8 - Semantic Feature Discovery
Date: 2025-11-21
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple, Callable, Any
from pathlib import Path
import numpy as np
from collections import defaultdict
import warnings


# =============================================================================
# VALIDATION RESULT STRUCTURES
# =============================================================================


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"  # Feature is invalid, must reject
    WARNING = "warning"    # Feature may be problematic
    INFO = "info"          # Informational note


@dataclass
class ValidationIssue:
    """A single validation issue"""
    severity: ValidationSeverity
    check_name: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of semantic feature validation"""
    feature_id: int
    is_valid: bool
    score: float  # 0.0 to 1.0, higher is better
    issues: List[ValidationIssue] = field(default_factory=list)

    # Detailed scores
    musical_validity_score: float = 0.0
    locality_consistency_score: float = 0.0
    redundancy_score: float = 0.0  # 1.0 = not redundant, 0.0 = completely redundant

    # Metadata
    suggested_fixes: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        return f"ValidationResult({status}, score={self.score:.2f}, issues={len(self.issues)})"

    @property
    def has_critical_issues(self) -> bool:
        """Check if any critical issues exist"""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings exist"""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)


# =============================================================================
# MUSICAL VALIDITY RULES
# =============================================================================


class MusicalValidityRules:
    """
    Defines musical validity rules and patterns.

    Features should represent recognizable musical concepts that:
    - Have clear musical meaning
    - Are perceptually significant
    - Follow music theory principles
    - Are compositionally relevant
    """

    # Musical domains that features can represent
    VALID_DOMAINS = {
        'pitch': ['melody', 'harmony', 'chord', 'interval', 'scale', 'mode'],
        'rhythm': ['beat', 'subdivision', 'syncopation', 'swing', 'groove', 'meter'],
        'dynamics': ['volume', 'accent', 'crescendo', 'articulation'],
        'timbre': ['tone', 'texture', 'instrumentation'],
        'structure': ['phrase', 'section', 'form', 'repetition', 'variation'],
        'expression': ['rubato', 'vibrato', 'portamento', 'glissando']
    }

    # Invalid feature patterns (anti-patterns)
    INVALID_PATTERNS = {
        'trivial': [
            'always_on',  # Feature always activated
            'never_on',   # Feature never activated
            'random',     # Random activations
        ],
        'non_musical': [
            'file_size',
            'encoding_artifact',
            'quantization_noise',
        ],
        'degenerate': [
            'constant',
            'linear_time',  # Just tracks time
            'note_count',   # Just counts notes
        ]
    }

    # Musical constraints
    CONSTRAINTS = {
        'pitch': {
            'range': (0, 127),  # MIDI note range
            'interval_max': 48,  # Max 4 octaves
        },
        'rhythm': {
            'min_duration': 0.01,  # 10ms minimum
            'max_duration': 16.0,  # 16 beats maximum
        },
        'dynamics': {
            'velocity_range': (0, 127),
        },
        'tempo': {
            'range': (30, 300),  # BPM
        }
    }

    @staticmethod
    def get_domain_keywords(domain: str) -> Set[str]:
        """Get keywords for a musical domain"""
        keywords = set()
        if domain in MusicalValidityRules.VALID_DOMAINS:
            keywords.update(MusicalValidityRules.VALID_DOMAINS[domain])
        return keywords

    @staticmethod
    def is_musically_meaningful(pattern: str) -> bool:
        """Check if a pattern name suggests musical meaning"""
        pattern_lower = pattern.lower()

        # Check for anti-patterns
        for category, anti_patterns in MusicalValidityRules.INVALID_PATTERNS.items():
            if any(ap in pattern_lower for ap in anti_patterns):
                return False

        # Check for musical domain keywords
        all_keywords = set()
        for domain_keywords in MusicalValidityRules.VALID_DOMAINS.values():
            all_keywords.update(domain_keywords)

        # At least one musical keyword should be present
        return any(keyword in pattern_lower for keyword in all_keywords)


# =============================================================================
# LOCALITY CONSISTENCY CHECKER
# =============================================================================


class LocalityConsistencyChecker:
    """
    Validates that semantic features respect musical locality transformations.

    A musically valid feature should:
    - Be invariant to musically irrelevant transformations
    - Change predictably under musically meaningful transformations
    - Maintain consistency across locality-preserving operations
    """

    def __init__(self, locality_functions: Optional[Any] = None):
        """
        Initialize locality checker.

        Args:
            locality_functions: Agent 1's MusicalLocalityFunctions instance
        """
        self.locality_functions = locality_functions

        # Expected locality behaviors for different feature types
        self.locality_expectations = {
            'rhythm': {
                'transpose': 'invariant',      # Rhythm unchanged by pitch shift
                'time_shift': 'invariant',     # Absolute rhythm preserved
                'retrograde': 'variant',       # Rhythm order matters
            },
            'pitch': {
                'transpose': 'equivariant',    # Shifts with transposition
                'invert_intervals': 'variant', # Changes with inversion
                'time_shift': 'invariant',     # Pitch unchanged by time shift
            },
            'harmony': {
                'transpose': 'equivariant',    # Shifts with transposition
                'voice_permutation': 'invariant',  # Voicing order doesn't matter
            }
        }

    def check_locality_consistency(
        self,
        feature_id: int,
        activation_function: Callable,
        test_midi_data: Any,
        tolerance: float = 0.1
    ) -> Tuple[float, List[str]]:
        """
        Check if feature respects locality constraints.

        Args:
            feature_id: ID of feature to validate
            activation_function: Function that computes feature activation
            test_midi_data: Test MIDI data to transform
            tolerance: Tolerance for activation differences

        Returns:
            (consistency_score, violations)
        """
        violations = []
        consistency_scores = []

        if self.locality_functions is None:
            warnings.warn("No locality functions provided, skipping locality checks")
            return 1.0, []

        # Get baseline activation
        baseline_activation = activation_function(test_midi_data)

        # Test key locality transformations
        transformations = [
            ('transpose', lambda x: self.locality_functions.transpose(x, 2)),
            ('time_shift', lambda x: self.locality_functions.time_shift(x, 0.5)),
            ('retrograde', lambda x: self.locality_functions.retrograde(x)),
        ]

        for transform_name, transform_fn in transformations:
            try:
                # Apply transformation
                transformed_data = transform_fn(test_midi_data)
                transformed_activation = activation_function(transformed_data)

                # Check activation difference
                activation_diff = abs(transformed_activation - baseline_activation)

                # Determine expected behavior (simplified heuristic)
                # In reality, this would be inferred from feature interpretation
                if activation_diff > tolerance:
                    # Feature changes under transformation
                    if transform_name == 'time_shift':
                        # Most musical features should be time-shift invariant
                        violations.append(
                            f"Feature changes under time_shift (diff={activation_diff:.3f}), "
                            "expected invariance"
                        )
                        consistency_scores.append(0.5)
                    else:
                        # Change is acceptable
                        consistency_scores.append(1.0)
                else:
                    # Feature is invariant
                    consistency_scores.append(1.0)

            except Exception as e:
                violations.append(f"Transform '{transform_name}' failed: {e}")
                consistency_scores.append(0.0)

        # Calculate overall consistency score
        if consistency_scores:
            score = np.mean(consistency_scores)
        else:
            score = 0.0

        return score, violations


# =============================================================================
# REDUNDANCY DETECTOR
# =============================================================================


class RedundancyDetector:
    """
    Detects redundant features that duplicate existing or other discovered features.

    Features are considered redundant if:
    - High correlation (>0.95) with existing feature
    - Linear combination of existing features (R² > 0.90)
    - Identical activation patterns
    """

    def __init__(self, correlation_threshold: float = 0.95, r2_threshold: float = 0.90):
        """
        Initialize redundancy detector.

        Args:
            correlation_threshold: Correlation threshold for redundancy
            r2_threshold: R² threshold for linear combination detection
        """
        self.correlation_threshold = correlation_threshold
        self.r2_threshold = r2_threshold
        self.known_features: Dict[int, np.ndarray] = {}

    def add_known_feature(self, feature_id: int, activations: np.ndarray):
        """Register a known feature with its activations"""
        self.known_features[feature_id] = activations

    def check_redundancy(
        self,
        feature_id: int,
        activations: np.ndarray,
        feature_metadata: Optional[Dict] = None
    ) -> Tuple[float, List[str]]:
        """
        Check if feature is redundant with known features.

        Args:
            feature_id: ID of feature to check
            activations: Feature activations on dataset
            feature_metadata: Optional metadata (name, interpretation)

        Returns:
            (redundancy_score, redundancy_issues)
            redundancy_score: 1.0 = not redundant, 0.0 = completely redundant
        """
        redundancy_issues = []
        max_correlation = 0.0

        if len(self.known_features) == 0:
            # No features to compare against
            return 1.0, []

        # Check correlation with each known feature
        for known_id, known_activations in self.known_features.items():
            if len(known_activations) != len(activations):
                continue

            # Pearson correlation
            correlation = np.corrcoef(activations, known_activations)[0, 1]
            max_correlation = max(max_correlation, abs(correlation))

            if abs(correlation) > self.correlation_threshold:
                redundancy_issues.append(
                    f"High correlation ({correlation:.3f}) with feature {known_id}"
                )

        # Check for linear combinations (simplified - would use regression in full version)
        if len(self.known_features) >= 2:
            # Stack known features
            X = np.column_stack(list(self.known_features.values()))
            y = activations

            # Simple linear fit check
            try:
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score

                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)

                if r2 > self.r2_threshold:
                    redundancy_issues.append(
                        f"Feature is linear combination of existing features (R²={r2:.3f})"
                    )
                    max_correlation = max(max_correlation, r2)
            except ImportError:
                # sklearn not available, skip linear combination check
                pass
            except Exception:
                # Other errors, skip
                pass

        # Calculate redundancy score (1.0 = unique, 0.0 = redundant)
        redundancy_score = 1.0 - max_correlation

        return redundancy_score, redundancy_issues


# =============================================================================
# SEMANTIC FEATURE VALIDATOR
# =============================================================================


class SemanticFeatureValidator:
    """
    Main validator for discovered semantic features.

    Performs comprehensive validation:
    1. Musical validity - represents real musical concept
    2. Locality consistency - respects transformations
    3. Redundancy detection - not duplicate

    Usage:
        validator = SemanticFeatureValidator()
        result = validator.validate_feature(
            feature_id=5,
            activation_function=lambda x: encoder.get_activation(x, 5),
            activations=feature_activations,
            interpretation={'name': 'swing_feel', 'domain': 'rhythm'}
        )
    """

    def __init__(
        self,
        locality_functions: Optional[Any] = None,
        correlation_threshold: float = 0.95,
        min_validity_score: float = 0.7
    ):
        """
        Initialize semantic feature validator.

        Args:
            locality_functions: Agent 1's MusicalLocalityFunctions instance
            correlation_threshold: Threshold for redundancy detection
            min_validity_score: Minimum score for feature to be valid
        """
        self.locality_checker = LocalityConsistencyChecker(locality_functions)
        self.redundancy_detector = RedundancyDetector(correlation_threshold)
        self.min_validity_score = min_validity_score
        self.validation_history: List[ValidationResult] = []

    def validate_feature(
        self,
        feature_id: int,
        activation_function: Optional[Callable] = None,
        activations: Optional[np.ndarray] = None,
        interpretation: Optional[Dict] = None,
        test_midi_data: Optional[Any] = None
    ) -> ValidationResult:
        """
        Validate a discovered semantic feature.

        Args:
            feature_id: Unique feature identifier
            activation_function: Function to compute activations (for locality checks)
            activations: Pre-computed activations on dataset (for redundancy)
            interpretation: Feature interpretation from Agent 6
                           {'name': str, 'domain': str, 'description': str}
            test_midi_data: Test MIDI data for locality checks

        Returns:
            ValidationResult with detailed validation information
        """
        issues = []

        # 1. Musical Validity Check
        musical_score, musical_issues = self._check_musical_validity(
            feature_id, interpretation
        )
        issues.extend(musical_issues)

        # 2. Locality Consistency Check
        locality_score = 1.0
        if activation_function is not None and test_midi_data is not None:
            locality_score, locality_violations = self.locality_checker.check_locality_consistency(
                feature_id, activation_function, test_midi_data
            )
            issues.extend([
                ValidationIssue(ValidationSeverity.WARNING, 'locality', violation)
                for violation in locality_violations
            ])

        # 3. Redundancy Check
        redundancy_score = 1.0
        if activations is not None:
            redundancy_score, redundancy_issues = self.redundancy_detector.check_redundancy(
                feature_id, activations, interpretation
            )
            issues.extend([
                ValidationIssue(
                    ValidationSeverity.CRITICAL if redundancy_score < 0.3 else ValidationSeverity.WARNING,
                    'redundancy',
                    issue
                )
                for issue in redundancy_issues
            ])

        # Calculate overall score (weighted average)
        overall_score = (
            0.4 * musical_score +
            0.3 * locality_score +
            0.3 * redundancy_score
        )

        # Determine validity
        is_valid = (
            overall_score >= self.min_validity_score and
            not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        )

        # Generate suggestions
        suggestions = self._generate_suggestions(
            musical_score, locality_score, redundancy_score, interpretation
        )

        # Create result
        result = ValidationResult(
            feature_id=feature_id,
            is_valid=is_valid,
            score=overall_score,
            issues=issues,
            musical_validity_score=musical_score,
            locality_consistency_score=locality_score,
            redundancy_score=redundancy_score,
            suggested_fixes=suggestions
        )

        # Record validation
        self.validation_history.append(result)

        # Register as known feature if valid
        if is_valid and activations is not None:
            self.redundancy_detector.add_known_feature(feature_id, activations)

        return result

    def _check_musical_validity(
        self,
        feature_id: int,
        interpretation: Optional[Dict]
    ) -> Tuple[float, List[ValidationIssue]]:
        """
        Check if feature represents a musically valid concept.

        Returns:
            (validity_score, issues)
        """
        issues = []
        score = 1.0

        if interpretation is None:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                'musical_validity',
                'No interpretation provided, cannot validate musical meaning'
            ))
            return 0.5, issues

        name = interpretation.get('name', '').lower()
        domain = interpretation.get('domain', '').lower()
        description = interpretation.get('description', '').lower()

        # Check if name is musically meaningful
        if not MusicalValidityRules.is_musically_meaningful(name):
            issues.append(ValidationIssue(
                ValidationSeverity.CRITICAL,
                'musical_validity',
                f"Feature name '{name}' does not suggest musical concept"
            ))
            score *= 0.0

        # Check domain validity
        if domain:
            valid_domains = set(MusicalValidityRules.VALID_DOMAINS.keys())
            if domain not in valid_domains:
                issues.append(ValidationIssue(
                    ValidationSeverity.WARNING,
                    'musical_validity',
                    f"Domain '{domain}' not in standard musical domains: {valid_domains}"
                ))
                score *= 0.8

        # Check description quality
        if description and len(description) < 10:
            issues.append(ValidationIssue(
                ValidationSeverity.INFO,
                'musical_validity',
                'Feature description is very brief, may lack clarity'
            ))
            score *= 0.9

        return score, issues

    def _generate_suggestions(
        self,
        musical_score: float,
        locality_score: float,
        redundancy_score: float,
        interpretation: Optional[Dict]
    ) -> List[str]:
        """Generate suggestions for improving feature"""
        suggestions = []

        if musical_score < 0.7:
            suggestions.append(
                "Improve feature interpretation with clearer musical terminology"
            )

        if locality_score < 0.7:
            suggestions.append(
                "Feature may not respect musical transformations - review activation patterns"
            )

        if redundancy_score < 0.7:
            suggestions.append(
                "Feature may be redundant - consider combining with similar features"
            )

        if interpretation and 'description' not in interpretation:
            suggestions.append(
                "Add detailed description of what musical concept this feature captures"
            )

        return suggestions

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed"""
        if not self.validation_history:
            return {'total': 0, 'valid': 0, 'invalid': 0}

        total = len(self.validation_history)
        valid = sum(1 for r in self.validation_history if r.is_valid)
        invalid = total - valid
        avg_score = np.mean([r.score for r in self.validation_history])

        return {
            'total': total,
            'valid': valid,
            'invalid': invalid,
            'valid_percentage': 100.0 * valid / total,
            'average_score': avg_score,
            'avg_musical_score': np.mean([r.musical_validity_score for r in self.validation_history]),
            'avg_locality_score': np.mean([r.locality_consistency_score for r in self.validation_history]),
            'avg_redundancy_score': np.mean([r.redundancy_score for r in self.validation_history]),
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================


if __name__ == '__main__':
    print("=" * 80)
    print("SEMANTIC FEATURE CONSTRAINT VALIDATION")
    print("=" * 80)

    # Create validator
    validator = SemanticFeatureValidator(min_validity_score=0.7)

    # Example 1: Valid feature
    print("\n1. VALIDATING VALID FEATURE (swing_feel)")
    print("-" * 80)

    swing_interpretation = {
        'name': 'swing_feel',
        'domain': 'rhythm',
        'description': 'Detects swing rhythm patterns in eighth notes'
    }

    # Simulate activations (would come from encoder)
    swing_activations = np.random.rand(100) * 0.5 + 0.3  # Moderate activations

    result = validator.validate_feature(
        feature_id=1,
        activations=swing_activations,
        interpretation=swing_interpretation
    )

    print(result)
    print(f"  Musical validity: {result.musical_validity_score:.2f}")
    print(f"  Locality consistency: {result.locality_consistency_score:.2f}")
    print(f"  Redundancy: {result.redundancy_score:.2f}")

    if result.issues:
        print(f"  Issues ({len(result.issues)}):")
        for issue in result.issues:
            print(f"    [{issue.severity.value}] {issue.message}")

    # Example 2: Invalid feature (non-musical)
    print("\n2. VALIDATING INVALID FEATURE (random_noise)")
    print("-" * 80)

    invalid_interpretation = {
        'name': 'random_noise',
        'domain': 'unknown',
        'description': 'Random activations'
    }

    result = validator.validate_feature(
        feature_id=2,
        activations=np.random.rand(100),
        interpretation=invalid_interpretation
    )

    print(result)
    if result.issues:
        print(f"  Issues ({len(result.issues)}):")
        for issue in result.issues:
            print(f"    [{issue.severity.value}] {issue.message}")

    # Example 3: Redundant feature
    print("\n3. VALIDATING REDUNDANT FEATURE")
    print("-" * 80)

    # Create nearly identical activations to feature 1
    redundant_activations = swing_activations + np.random.rand(100) * 0.05

    redundant_interpretation = {
        'name': 'swing_variation',
        'domain': 'rhythm',
        'description': 'Similar to swing_feel but slightly different'
    }

    result = validator.validate_feature(
        feature_id=3,
        activations=redundant_activations,
        interpretation=redundant_interpretation
    )

    print(result)
    if result.issues:
        print(f"  Issues ({len(result.issues)}):")
        for issue in result.issues:
            print(f"    [{issue.severity.value}] {issue.message}")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    summary = validator.get_validation_summary()
    print(f"Total features validated: {summary['total']}")
    print(f"Valid: {summary['valid']} ({summary['valid_percentage']:.1f}%)")
    print(f"Invalid: {summary['invalid']}")
    print(f"Average score: {summary['average_score']:.2f}")
    print(f"  Musical validity: {summary['avg_musical_score']:.2f}")
    print(f"  Locality: {summary['avg_locality_score']:.2f}")
    print(f"  Redundancy: {summary['avg_redundancy_score']:.2f}")

    print("\n" + "=" * 80)
    print("Semantic constraint validation complete!")
    print("=" * 80)
