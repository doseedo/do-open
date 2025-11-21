#!/usr/bin/env python3
"""
Constraint Validator Integration Module - Agent 8
=================================================

Integration utilities for connecting the Musical Constraint Validator
with other components of the Musical Program Synthesis System:

- Phase 1: Universal Parameter Registry (Agents 1-3)
- Phase 2: XGBoost Parameter Synthesizer (Agent 5)
- Phase 2: Program Compiler (Agent 6)
- Phase 2: Real-time Engine (Agent 9)

This module provides:
1. Parameter constraint checking before/after XGBoost prediction
2. Post-processing pipeline for generated parameters
3. Constraint-guided optimization
4. Validation metrics for model training

Author: Agent 8 - Constraint Validator
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from copy import deepcopy
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from constraints.musical_validator import (
    MusicalConstraintValidator,
    ValidationResult,
    ValidationSeverity,
    ViolationType,
    INSTRUMENT_RANGES
)

from constraints.advanced_constraints import (
    JazzVoiceLeadingValidator,
    ExtendedTechniqueValidator,
    PerformancePracticeValidator,
    OrchestrationConstraintValidator,
    create_validator_for_style
)


# =============================================================================
# INTEGRATION WITH PHASE 1: PARAMETER REGISTRY
# =============================================================================

class ParameterRegistryIntegration:
    """
    Integration with Universal Parameter Registry (Agent 3).

    When Phase 1 is complete, this class will:
    - Validate parameters from registry
    - Enforce parameter constraints
    - Provide constraint metadata for parameters
    """

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize registry integration.

        Args:
            registry_path: Path to parameter registry JSON
                          (placeholder until Phase 1 complete)
        """
        self.registry_path = registry_path
        self.registry = self._load_registry()
        self.validator_cache = {}

    def _load_registry(self) -> Dict[str, Any]:
        """Load parameter registry from Phase 1"""
        # Placeholder implementation until Phase 1 is complete
        # Will load from: /home/user/Do/midi_generator/parameters/universal_registry.py

        if self.registry_path and Path(self.registry_path).exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)

        # Return minimal structure for now
        return {
            'parameters': {},
            'constraints': {},
            'dependencies': {}
        }

    def get_constraints_for_parameter(self,
                                     param_name: str) -> Dict[str, Any]:
        """
        Get musical constraints for a specific parameter.

        Args:
            param_name: Parameter name (e.g., 'harmony.jazz.voicing_type')

        Returns:
            Constraint specification dict
        """
        # Extract domain and category from parameter name
        parts = param_name.split('.')

        constraints = {
            'musical_rules': [],
            'valid_ranges': None,
            'dependencies': [],
        }

        # Add domain-specific constraints
        if 'voicing' in param_name:
            constraints['musical_rules'].append('voice_leading')
            constraints['musical_rules'].append('spacing')

        if 'range' in param_name or 'pitch' in param_name:
            constraints['musical_rules'].append('instrument_range')

        if 'harmony' in param_name or 'chord' in param_name:
            constraints['musical_rules'].append('harmonic_rules')

        return constraints

    def validate_parameter_value(self,
                                param_name: str,
                                value: Any,
                                context: Optional[Dict] = None) -> ValidationResult:
        """
        Validate a single parameter value.

        Args:
            param_name: Parameter name
            value: Parameter value
            context: Additional context (key, tempo, etc.)

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        constraints = self.get_constraints_for_parameter(param_name)

        # Apply relevant validators based on constraints
        for rule in constraints['musical_rules']:
            if rule == 'instrument_range' and isinstance(value, (list, tuple)):
                # Value is a list of MIDI notes
                instrument = context.get('instrument', 'default') if context else 'default'
                validator = MusicalConstraintValidator()
                range_result = validator.validate_range(value, instrument)

                for violation in range_result.violations:
                    result.add_violation(violation)

        return result


# =============================================================================
# INTEGRATION WITH PHASE 2: XGBOOST SYNTHESIZER
# =============================================================================

class XGBoostConstraintIntegration:
    """
    Integration with XGBoost Parameter Synthesizer (Agent 5).

    Provides:
    - Post-processing of XGBoost predictions
    - Constraint-guided parameter adjustment
    - Validation metrics for model evaluation
    """

    def __init__(self, style: str = 'common_practice'):
        """
        Initialize XGBoost integration.

        Args:
            style: Musical style for constraint validation
        """
        self.validator = create_validator_for_style(style)
        self.style = style

    def validate_predicted_parameters(self,
                                      predictions: Dict[str, Any],
                                      confidence_scores: Optional[Dict[str, float]] = None
                                      ) -> Tuple[ValidationResult, Dict[str, Any]]:
        """
        Validate parameters predicted by XGBoost.

        Args:
            predictions: Dictionary of predicted parameter values
            confidence_scores: Optional confidence scores for each prediction

        Returns:
            Tuple of (ValidationResult, corrected_parameters)
        """
        # Validate predictions
        result = self.validator.validate_parameters(predictions)

        # If invalid and auto-correction enabled, fix issues
        if not result.is_valid and self.validator.allow_auto_correction:
            corrected, _ = self.validator.validate_and_correct(predictions)
            return result, corrected

        return result, predictions

    def constraint_violation_loss(self,
                                  predictions: Dict[str, Any],
                                  ground_truth: Optional[Dict[str, Any]] = None
                                  ) -> float:
        """
        Calculate loss based on constraint violations.

        This can be used as an additional loss term during XGBoost training
        to penalize predictions that violate music theory rules.

        Args:
            predictions: Predicted parameters
            ground_truth: Optional ground truth parameters

        Returns:
            Constraint violation penalty (0.0 = no violations, 1.0 = severe)
        """
        result = self.validator.validate_parameters(predictions)

        # Calculate penalty based on violation severity
        penalty = 0.0

        for violation in result.violations:
            if violation.severity == ValidationSeverity.INFO:
                penalty += 0.01
            elif violation.severity == ValidationSeverity.WARNING:
                penalty += 0.05
            elif violation.severity == ValidationSeverity.ERROR:
                penalty += 0.15
            elif violation.severity == ValidationSeverity.CRITICAL:
                penalty += 0.30

        return min(1.0, penalty)

    def get_constraint_features(self,
                               parameters: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract constraint-based features for XGBoost.

        These features can help the model learn to avoid violations.

        Args:
            parameters: Parameter dictionary

        Returns:
            Feature dictionary
        """
        features = {}

        # Validate and extract metrics
        result = self.validator.validate_parameters(parameters)

        features['constraint_score'] = result.score
        features['num_violations'] = len(result.violations)
        features['num_warnings'] = result.warnings_count
        features['num_errors'] = result.errors_count
        features['num_critical'] = result.critical_count

        # Count violations by type
        violation_types = {}
        for violation in result.violations:
            vtype = violation.violation_type.value
            violation_types[vtype] = violation_types.get(vtype, 0) + 1

        for vtype, count in violation_types.items():
            features[f'violation_{vtype}'] = count

        return features


# =============================================================================
# POST-PROCESSING PIPELINE
# =============================================================================

class ConstraintPostProcessor:
    """
    Post-processing pipeline for generated parameters.

    Applies constraint validation and correction in a multi-stage pipeline.
    """

    def __init__(self, style: str = 'common_practice', strict: bool = False):
        """
        Initialize post-processor.

        Args:
            style: Musical style
            strict: Use strict validation rules
        """
        self.validator = MusicalConstraintValidator(
            style=style,
            strict_mode=strict,
            allow_auto_correction=True
        )
        self.pipeline_stages = []

    def add_stage(self,
                  stage_name: str,
                  validator_func: Callable,
                  corrector_func: Optional[Callable] = None):
        """
        Add a processing stage to the pipeline.

        Args:
            stage_name: Name of this stage
            validator_func: Function that validates parameters
            corrector_func: Optional function that corrects violations
        """
        self.pipeline_stages.append({
            'name': stage_name,
            'validator': validator_func,
            'corrector': corrector_func
        })

    def process(self,
               parameters: Dict[str, Any],
               max_iterations: int = 3) -> Tuple[Dict[str, Any], List[ValidationResult]]:
        """
        Run full post-processing pipeline.

        Args:
            parameters: Raw parameters to process
            max_iterations: Maximum correction iterations per stage

        Returns:
            Tuple of (processed_parameters, results_per_stage)
        """
        current_params = deepcopy(parameters)
        results_history = []

        for stage in self.pipeline_stages:
            stage_name = stage['name']
            validator = stage['validator']
            corrector = stage['corrector']

            # Validate
            result = validator(current_params)
            results_history.append(result)

            # Correct if needed and possible
            if not result.is_valid and corrector is not None:
                for iteration in range(max_iterations):
                    current_params = corrector(current_params)

                    # Re-validate
                    result = validator(current_params)

                    if result.is_valid:
                        break

        return current_params, results_history

    def create_default_pipeline(self):
        """Create default processing pipeline"""

        # Stage 1: Instrument range validation
        self.add_stage(
            'range_validation',
            lambda p: self.validator.validate_multi_instrument_ranges(
                p.get('instrument_parts', {})
            ) if 'instrument_parts' in p else ValidationResult(is_valid=True),
            lambda p: self._fix_ranges(p)
        )

        # Stage 2: Voice leading validation
        self.add_stage(
            'voice_leading',
            lambda p: self.validator.validate_voice_leading(
                p.get('voices', [])
            ) if 'voices' in p else ValidationResult(is_valid=True),
            lambda p: self._fix_voice_leading(p)
        )

        # Stage 3: Harmonic validation
        self.add_stage(
            'harmony',
            lambda p: self.validator.validate_harmonic_progression(
                p.get('chord_progression', []),
                p.get('key', None)
            ) if 'chord_progression' in p else ValidationResult(is_valid=True),
            None  # No automatic harmonic correction yet
        )

    def _fix_ranges(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fix instrument range issues"""
        fixed = deepcopy(params)

        if 'instrument_parts' in fixed:
            for instrument, notes in fixed['instrument_parts'].items():
                fixed['instrument_parts'][instrument] = \
                    self.validator.fix_out_of_range(notes, instrument)

        return fixed

    def _fix_voice_leading(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fix voice leading issues"""
        fixed = deepcopy(params)

        if 'voices' in fixed:
            fixed['voices'] = self.validator.fix_voice_leading(fixed['voices'])

        return fixed


# =============================================================================
# CONSTRAINT-GUIDED OPTIMIZATION
# =============================================================================

class ConstraintGuidedOptimizer:
    """
    Optimize parameters to satisfy constraints while maintaining similarity
    to XGBoost predictions.

    Uses gradient-free optimization (e.g., simulated annealing, genetic algorithm)
    to find parameters that satisfy constraints.
    """

    def __init__(self, style: str = 'common_practice'):
        """Initialize optimizer"""
        self.validator = create_validator_for_style(style)

    def optimize(self,
                initial_params: Dict[str, Any],
                max_iterations: int = 100) -> Dict[str, Any]:
        """
        Optimize parameters to satisfy constraints.

        Args:
            initial_params: Initial (possibly invalid) parameters
            max_iterations: Maximum optimization iterations

        Returns:
            Optimized parameters
        """
        current = deepcopy(initial_params)
        best = deepcopy(initial_params)
        best_score = self._score_parameters(best)

        for iteration in range(max_iterations):
            # Generate neighbor by small perturbation
            neighbor = self._perturb_parameters(current)

            # Score neighbor
            neighbor_score = self._score_parameters(neighbor)

            # Accept if better
            if neighbor_score > best_score:
                best = deepcopy(neighbor)
                best_score = neighbor_score
                current = neighbor

        return best

    def _score_parameters(self, params: Dict[str, Any]) -> float:
        """
        Score parameters (higher is better).

        Combines constraint satisfaction with other metrics.
        """
        result = self.validator.validate_parameters(params)
        return result.score

    def _perturb_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create small random perturbation of parameters.

        This is a placeholder - real implementation would be parameter-specific.
        """
        perturbed = deepcopy(params)

        # Example: perturb numeric values slightly
        for key, value in perturbed.items():
            if isinstance(value, (int, float)):
                import random
                perturbed[key] = value + random.uniform(-0.1, 0.1)

        return perturbed


# =============================================================================
# VALIDATION METRICS FOR MODEL TRAINING
# =============================================================================

class ConstraintValidationMetrics:
    """
    Metrics for evaluating XGBoost model performance on constraint satisfaction.

    These metrics can be used during model training/evaluation to track
    how well the model learns to generate musically valid parameters.
    """

    @staticmethod
    def constraint_satisfaction_rate(predictions_list: List[Dict[str, Any]],
                                    style: str = 'common_practice') -> float:
        """
        Calculate percentage of predictions that satisfy all constraints.

        Args:
            predictions_list: List of parameter dictionaries
            style: Musical style

        Returns:
            Satisfaction rate (0.0 to 1.0)
        """
        validator = create_validator_for_style(style)

        valid_count = 0
        for predictions in predictions_list:
            result = validator.validate_parameters(predictions)
            if result.is_valid:
                valid_count += 1

        return valid_count / len(predictions_list) if predictions_list else 0.0

    @staticmethod
    def average_constraint_score(predictions_list: List[Dict[str, Any]],
                                 style: str = 'common_practice') -> float:
        """
        Calculate average constraint score across predictions.

        Args:
            predictions_list: List of parameter dictionaries
            style: Musical style

        Returns:
            Average score (0.0 to 1.0)
        """
        validator = create_validator_for_style(style)

        total_score = 0.0
        for predictions in predictions_list:
            result = validator.validate_parameters(predictions)
            total_score += result.score

        return total_score / len(predictions_list) if predictions_list else 0.0

    @staticmethod
    def violation_distribution(predictions_list: List[Dict[str, Any]],
                              style: str = 'common_practice') -> Dict[str, int]:
        """
        Get distribution of violation types across predictions.

        Args:
            predictions_list: List of parameter dictionaries
            style: Musical style

        Returns:
            Dictionary mapping violation types to counts
        """
        validator = create_validator_for_style(style)

        violation_counts = {}

        for predictions in predictions_list:
            result = validator.validate_parameters(predictions)

            for violation in result.violations:
                vtype = violation.violation_type.value
                violation_counts[vtype] = violation_counts.get(vtype, 0) + 1

        return violation_counts


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CONSTRAINT VALIDATOR INTEGRATION - Agent 8")
    print("=" * 80)

    # Example 1: XGBoost Integration
    print("\n1. XGBoost INTEGRATION")
    print("-" * 80)

    xgb_integration = XGBoostConstraintIntegration(style='jazz')

    # Simulated XGBoost predictions
    predictions = {
        'voices': [
            [48, 50, 52],
            [60, 62, 64],
            [64, 66, 68],
            [72, 74, 76],
        ],
        'instrument_parts': {
            'trumpet': [60, 64, 67, 72],
            'trombone': [48, 50, 52, 55],
        }
    }

    result, corrected = xgb_integration.validate_predicted_parameters(predictions)
    print(f"Validation: {result.get_summary()}")

    # Calculate constraint features
    features = xgb_integration.get_constraint_features(predictions)
    print(f"\nConstraint features:")
    for key, value in features.items():
        print(f"  {key}: {value}")

    # Example 2: Post-Processing Pipeline
    print("\n2. POST-PROCESSING PIPELINE")
    print("-" * 80)

    processor = ConstraintPostProcessor(style='common_practice')
    processor.create_default_pipeline()

    processed, history = processor.process(predictions)
    print(f"Processed {len(history)} stages")
    for i, result in enumerate(history):
        print(f"  Stage {i+1}: {result.get_summary()}")

    # Example 3: Validation Metrics
    print("\n3. VALIDATION METRICS")
    print("-" * 80)

    # Simulate multiple predictions
    predictions_batch = [predictions] * 5

    satisfaction_rate = ConstraintValidationMetrics.constraint_satisfaction_rate(
        predictions_batch
    )
    avg_score = ConstraintValidationMetrics.average_constraint_score(
        predictions_batch
    )
    violations = ConstraintValidationMetrics.violation_distribution(
        predictions_batch
    )

    print(f"Constraint satisfaction rate: {satisfaction_rate:.1%}")
    print(f"Average constraint score: {avg_score:.1%}")
    print(f"\nViolation distribution:")
    for vtype, count in violations.items():
        print(f"  {vtype}: {count}")

    print("\n" + "=" * 80)
    print("Integration module ready for Phase 2 components!")
    print("=" * 80)
