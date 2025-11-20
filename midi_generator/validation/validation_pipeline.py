#!/usr/bin/env python3
"""
Agent 08: Validation Pipeline Framework
========================================

Comprehensive validation pipeline for the hierarchical multi-task learning
system for MIDI parameter prediction.

This module provides:
1. Base validation classes and data structures
2. Validation pipeline orchestration
3. Result aggregation and reporting
4. Integration with training pipeline and quality dashboard

Author: Agent 08 - Validation Framework Builder
Date: 2025-11-20
License: MIT
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
from collections import defaultdict


# ==============================================================================
# ENUMS AND CONSTANTS
# ==============================================================================

class ValidationStatus(Enum):
    """Validation status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ParameterLevel(Enum):
    """Hierarchical parameter levels."""
    LEVEL1_GLOBAL = "level1_global"
    LEVEL2_UNIVERSAL = "level2_universal"
    LEVEL3_GENRE_SPECIFIC = "level3_genre_specific"


class ValidationCategory(Enum):
    """Categories of validation."""
    PARAMETER_ACCURACY = "parameter_accuracy"
    MUSICAL_QUALITY = "musical_quality"
    GENRE_SPECIFIC = "genre_specific"
    GENERALIZATION = "generalization"
    REGRESSION = "regression"


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class ParameterValidationResult:
    """Result of validating a single parameter prediction."""
    parameter_name: str
    parameter_path: str  # e.g., "harmony.chord_density"
    parameter_level: ParameterLevel

    # Prediction vs ground truth
    predicted_value: Union[float, int, str, bool]
    ground_truth: Union[float, int, str, bool]

    # Error metrics
    error: Optional[float] = None  # Absolute error (for continuous)
    error_percentage: Optional[float] = None  # Percentage error
    accuracy: Optional[float] = None  # For categorical (0-1)

    # Validation result
    passed: bool = False
    status: ValidationStatus = ValidationStatus.FAILED
    threshold_used: Optional[float] = None

    # Metadata
    validation_time: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result['parameter_level'] = self.parameter_level.value
        result['status'] = self.status.value
        return result


@dataclass
class MusicalValidationResult:
    """Result of musical quality validation."""
    validation_type: str  # 'intervals', 'harmony', 'rhythm', 'voice_leading', etc.
    category: str  # 'intervals', 'harmony', 'rhythm'

    passed: bool
    score: float  # 0-1

    # Detailed metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Thresholds
    threshold: float = 0.85

    # Metadata
    validation_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class GenreValidationResult:
    """Result of genre-specific validation."""
    genre: str  # 'jazz', 'classical', 'rock', etc.

    passed: bool
    authenticity_score: float  # 0-1

    # Genre-specific metrics
    characteristics_validated: Dict[str, bool] = field(default_factory=dict)
    characteristic_scores: Dict[str, float] = field(default_factory=dict)

    # Details
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    validation_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    # Metadata
    report_id: str
    timestamp: str
    model_version: Optional[str] = None
    validation_dataset: Optional[str] = None

    # Overall results
    overall_passed: bool = False
    overall_score: float = 0.0

    # Category results
    parameter_validation_results: List[ParameterValidationResult] = field(default_factory=list)
    musical_validation_results: List[MusicalValidationResult] = field(default_factory=list)
    genre_validation_results: List[GenreValidationResult] = field(default_factory=list)

    # Summary statistics
    total_parameters_validated: int = 0
    parameters_passed: int = 0
    parameters_failed: int = 0

    musical_quality_score: float = 0.0
    genre_authenticity_scores: Dict[str, float] = field(default_factory=dict)

    # Performance
    total_validation_time: float = 0.0

    # Notes and recommendations
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'report_id': self.report_id,
            'timestamp': self.timestamp,
            'model_version': self.model_version,
            'validation_dataset': self.validation_dataset,
            'overall_passed': self.overall_passed,
            'overall_score': self.overall_score,
            'parameter_validation_results': [r.to_dict() for r in self.parameter_validation_results],
            'musical_validation_results': [r.to_dict() for r in self.musical_validation_results],
            'genre_validation_results': [r.to_dict() for r in self.genre_validation_results],
            'total_parameters_validated': self.total_parameters_validated,
            'parameters_passed': self.parameters_passed,
            'parameters_failed': self.parameters_failed,
            'musical_quality_score': self.musical_quality_score,
            'genre_authenticity_scores': self.genre_authenticity_scores,
            'total_validation_time': self.total_validation_time,
            'critical_issues': self.critical_issues,
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }

    def save_to_file(self, filepath: Path):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: Path) -> 'ValidationReport':
        """Load report from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct objects
        report = cls(
            report_id=data['report_id'],
            timestamp=data['timestamp'],
            model_version=data.get('model_version'),
            validation_dataset=data.get('validation_dataset')
        )

        # Add results
        for r in data.get('parameter_validation_results', []):
            report.parameter_validation_results.append(ParameterValidationResult(**r))

        for r in data.get('musical_validation_results', []):
            report.musical_validation_results.append(MusicalValidationResult(**r))

        for r in data.get('genre_validation_results', []):
            report.genre_validation_results.append(GenreValidationResult(**r))

        # Add summary data
        report.overall_passed = data['overall_passed']
        report.overall_score = data['overall_score']
        report.total_parameters_validated = data['total_parameters_validated']
        report.parameters_passed = data['parameters_passed']
        report.parameters_failed = data['parameters_failed']
        report.musical_quality_score = data['musical_quality_score']
        report.genre_authenticity_scores = data['genre_authenticity_scores']
        report.total_validation_time = data['total_validation_time']
        report.critical_issues = data['critical_issues']
        report.warnings = data['warnings']
        report.recommendations = data['recommendations']

        return report


# ==============================================================================
# BASE VALIDATOR CLASSES
# ==============================================================================

class BaseValidator:
    """Base class for all validators."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize validator with optional configuration."""
        self.config = config or {}
        self.validation_count = 0
        self.total_validation_time = 0.0

    def validate(self, *args, **kwargs) -> Union[ParameterValidationResult, MusicalValidationResult]:
        """Perform validation. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement validate()")

    def _start_timer(self) -> float:
        """Start timing a validation."""
        return time.time()

    def _end_timer(self, start_time: float) -> float:
        """End timing and return duration."""
        duration = time.time() - start_time
        self.total_validation_time += duration
        self.validation_count += 1
        return duration


class ParameterValidator(BaseValidator):
    """Base class for parameter validation."""

    def __init__(self, parameter_spec: Dict, config: Optional[Dict] = None):
        """
        Initialize parameter validator.

        Args:
            parameter_spec: Parameter specification including:
                - name: Parameter name
                - path: Parameter path (e.g., 'harmony.chord_density')
                - type: Parameter type ('continuous', 'categorical', 'boolean')
                - range: Valid range or options
                - threshold: Validation threshold
            config: Optional configuration overrides
        """
        super().__init__(config)
        self.parameter_spec = parameter_spec
        self.name = parameter_spec['name']
        self.path = parameter_spec['path']
        self.param_type = parameter_spec.get('type', 'continuous')
        self.valid_range = parameter_spec.get('range', (0.0, 1.0))
        self.threshold = parameter_spec.get('threshold', 0.1)  # Default 10% error

    def validate_prediction(self,
                          predicted: Union[float, int, str, bool],
                          ground_truth: Union[float, int, str, bool]) -> ParameterValidationResult:
        """
        Validate a parameter prediction.

        Args:
            predicted: Predicted value from model
            ground_truth: Ground truth value

        Returns:
            ParameterValidationResult
        """
        start_time = self._start_timer()

        result = ParameterValidationResult(
            parameter_name=self.name,
            parameter_path=self.path,
            parameter_level=self._get_parameter_level(),
            predicted_value=predicted,
            ground_truth=ground_truth,
            threshold_used=self.threshold
        )

        # Validate based on parameter type
        if self.param_type == 'continuous':
            result.error = abs(predicted - ground_truth)
            result.error_percentage = result.error / abs(ground_truth) if ground_truth != 0 else result.error
            result.passed = result.error_percentage <= self.threshold
            result.status = ValidationStatus.PASSED if result.passed else ValidationStatus.FAILED

        elif self.param_type == 'categorical':
            result.accuracy = 1.0 if predicted == ground_truth else 0.0
            result.passed = predicted == ground_truth
            result.status = ValidationStatus.PASSED if result.passed else ValidationStatus.FAILED

        elif self.param_type == 'boolean':
            result.accuracy = 1.0 if predicted == ground_truth else 0.0
            result.passed = predicted == ground_truth
            result.status = ValidationStatus.PASSED if result.passed else ValidationStatus.FAILED

        result.validation_time = self._end_timer(start_time)
        return result

    def validate_distribution(self,
                            predictions: List[Union[float, int, str, bool]],
                            ground_truths: List[Union[float, int, str, bool]]) -> Dict[str, float]:
        """
        Validate distribution of predictions vs ground truths.

        Returns statistical metrics over the dataset.
        """
        if self.param_type == 'continuous':
            errors = [abs(p - gt) for p, gt in zip(predictions, ground_truths)]
            squared_errors = [(p - gt) ** 2 for p, gt in zip(predictions, ground_truths)]

            mae = statistics.mean(errors)
            rmse = statistics.mean(squared_errors) ** 0.5

            # Calculate correlation
            mean_pred = statistics.mean(predictions)
            mean_truth = statistics.mean(ground_truths)
            covariance = sum((p - mean_pred) * (gt - mean_truth)
                           for p, gt in zip(predictions, ground_truths)) / len(predictions)
            std_pred = statistics.stdev(predictions) if len(predictions) > 1 else 0
            std_truth = statistics.stdev(ground_truths) if len(ground_truths) > 1 else 0
            correlation = covariance / (std_pred * std_truth) if (std_pred * std_truth) > 0 else 0

            return {
                'mae': mae,
                'rmse': rmse,
                'correlation': correlation,
                'mean_error': statistics.mean(errors),
                'std_error': statistics.stdev(errors) if len(errors) > 1 else 0
            }

        elif self.param_type in ['categorical', 'boolean']:
            correct = sum(1 for p, gt in zip(predictions, ground_truths) if p == gt)
            accuracy = correct / len(predictions) if predictions else 0

            return {
                'accuracy': accuracy,
                'correct_count': correct,
                'total_count': len(predictions)
            }

        return {}

    def validate_range(self, value: Union[float, int, str, bool]) -> bool:
        """Check if value is within valid range."""
        if self.param_type == 'continuous':
            min_val, max_val = self.valid_range
            return min_val <= value <= max_val
        elif self.param_type == 'categorical':
            return value in self.valid_range
        return True

    def _get_parameter_level(self) -> ParameterLevel:
        """Determine parameter level from path."""
        # Level 1: genre, tempo, time_signature, key, energy, complexity, structure
        level1_prefixes = ['genre', 'tempo', 'time_signature', 'key', 'energy', 'complexity', 'structure']

        # Level 3: genre-specific (jazz., classical., rock., electronic., hiphop., latin.)
        level3_prefixes = ['jazz', 'classical', 'rock', 'electronic', 'hiphop', 'latin']

        path_start = self.path.split('.')[0]

        if path_start in level1_prefixes:
            return ParameterLevel.LEVEL1_GLOBAL
        elif path_start in level3_prefixes:
            return ParameterLevel.LEVEL3_GENRE_SPECIFIC
        else:
            return ParameterLevel.LEVEL2_UNIVERSAL


class MusicalQualityValidator(BaseValidator):
    """Base class for musical quality validation."""

    def __init__(self, validation_type: str, threshold: float = 0.85, config: Optional[Dict] = None):
        """
        Initialize musical quality validator.

        Args:
            validation_type: Type of validation ('intervals', 'harmony', 'rhythm', etc.)
            threshold: Minimum acceptable score (0-1)
            config: Optional configuration
        """
        super().__init__(config)
        self.validation_type = validation_type
        self.threshold = threshold

    def validate_midi(self, midi_data: Any) -> MusicalValidationResult:
        """
        Validate musical quality of MIDI data.

        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement validate_midi()")


class GenreValidator(BaseValidator):
    """Base class for genre-specific validation."""

    def __init__(self, genre: str, characteristics: Dict, config: Optional[Dict] = None):
        """
        Initialize genre validator.

        Args:
            genre: Genre name ('jazz', 'classical', etc.)
            characteristics: Dictionary of genre characteristics to validate
            config: Optional configuration
        """
        super().__init__(config)
        self.genre = genre
        self.characteristics = characteristics

    def validate_style_adherence(self, midi_data: Any, parameters: Dict) -> GenreValidationResult:
        """
        Validate style adherence for the genre.

        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement validate_style_adherence()")


# ==============================================================================
# VALIDATION PIPELINE
# ==============================================================================

class ValidationPipeline:
    """
    Main validation pipeline orchestrator.

    Coordinates all validation stages:
    1. Parameter prediction validation
    2. Musical quality validation
    3. Genre-specific validation
    4. Result aggregation and reporting
    """

    def __init__(self,
                 config_path: Optional[Path] = None,
                 parameter_validators: Optional[List[ParameterValidator]] = None,
                 musical_validators: Optional[List[MusicalQualityValidator]] = None,
                 genre_validators: Optional[List[GenreValidator]] = None):
        """
        Initialize validation pipeline.

        Args:
            config_path: Path to validation configuration file
            parameter_validators: List of parameter validators
            musical_validators: List of musical quality validators
            genre_validators: List of genre validators
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.parameter_validators = parameter_validators or []
        self.musical_validators = musical_validators or []
        self.genre_validators = genre_validators or []

        self.results_storage_dir = Path(self.config.get('results_storage_dir', 'validation_results'))
        self.results_storage_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: Path) -> Dict:
        """Load validation configuration from YAML/JSON file."""
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                return json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML config files")
        return {}

    def validate_predictions(self,
                           predictions: Dict[str, Union[float, int, str, bool]],
                           ground_truths: Dict[str, Union[float, int, str, bool]]) -> List[ParameterValidationResult]:
        """
        Validate parameter predictions against ground truth.

        Args:
            predictions: Dictionary of parameter predictions {path: value}
            ground_truths: Dictionary of ground truth values {path: value}

        Returns:
            List of ParameterValidationResult
        """
        results = []

        for validator in self.parameter_validators:
            path = validator.path

            if path in predictions and path in ground_truths:
                result = validator.validate_prediction(
                    predicted=predictions[path],
                    ground_truth=ground_truths[path]
                )
                results.append(result)

        return results

    def validate_musical_quality(self, midi_data: Any) -> List[MusicalValidationResult]:
        """
        Validate musical quality of generated MIDI.

        Args:
            midi_data: MIDI data to validate

        Returns:
            List of MusicalValidationResult
        """
        results = []

        for validator in self.musical_validators:
            result = validator.validate_midi(midi_data)
            results.append(result)

        return results

    def validate_genre_specific(self,
                               midi_data: Any,
                               parameters: Dict,
                               genre: str) -> Optional[GenreValidationResult]:
        """
        Validate genre-specific characteristics.

        Args:
            midi_data: MIDI data to validate
            parameters: Predicted parameters
            genre: Genre to validate against

        Returns:
            GenreValidationResult or None if genre validator not found
        """
        for validator in self.genre_validators:
            if validator.genre == genre:
                return validator.validate_style_adherence(midi_data, parameters)

        return None

    def validate_complete(self,
                        predictions: Dict[str, Union[float, int, str, bool]],
                        ground_truths: Dict[str, Union[float, int, str, bool]],
                        midi_data: Optional[Any] = None,
                        genre: Optional[str] = None,
                        model_version: Optional[str] = None) -> ValidationReport:
        """
        Run complete validation pipeline.

        Args:
            predictions: Parameter predictions
            ground_truths: Ground truth parameters
            midi_data: Optional MIDI data for musical quality validation
            genre: Optional genre for genre-specific validation
            model_version: Optional model version identifier

        Returns:
            Complete ValidationReport
        """
        start_time = time.time()

        # Create report
        report = ValidationReport(
            report_id=f"validation_{int(time.time())}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            model_version=model_version,
            validation_dataset="validation_set"
        )

        # 1. Validate parameter predictions
        param_results = self.validate_predictions(predictions, ground_truths)
        report.parameter_validation_results = param_results
        report.total_parameters_validated = len(param_results)
        report.parameters_passed = sum(1 for r in param_results if r.passed)
        report.parameters_failed = len(param_results) - report.parameters_passed

        # 2. Validate musical quality (if MIDI data provided)
        if midi_data and self.musical_validators:
            musical_results = self.validate_musical_quality(midi_data)
            report.musical_validation_results = musical_results

            # Calculate overall musical quality score
            scores = [r.score for r in musical_results]
            report.musical_quality_score = statistics.mean(scores) if scores else 0.0

        # 3. Validate genre-specific (if genre specified)
        if genre and midi_data and self.genre_validators:
            genre_result = self.validate_genre_specific(midi_data, predictions, genre)
            if genre_result:
                report.genre_validation_results.append(genre_result)
                report.genre_authenticity_scores[genre] = genre_result.authenticity_score

        # Calculate overall score
        param_accuracy = report.parameters_passed / report.total_parameters_validated if report.total_parameters_validated > 0 else 0
        report.overall_score = (
            param_accuracy * 0.5 +
            report.musical_quality_score * 0.3 +
            (report.genre_authenticity_scores.get(genre, 0.0) if genre else 0) * 0.2
        )

        # Determine overall pass/fail
        report.overall_passed = (
            param_accuracy >= 0.85 and
            report.musical_quality_score >= 0.85
        )

        # Collect issues and recommendations
        report.critical_issues = [
            r.parameter_name for r in param_results
            if not r.passed and r.error_percentage and r.error_percentage > 0.2
        ]

        if param_accuracy < 0.85:
            report.recommendations.append(
                f"Parameter accuracy ({param_accuracy:.2%}) below target 85%. "
                f"Consider retraining or adjusting model architecture."
            )

        if report.musical_quality_score < 0.85:
            report.recommendations.append(
                f"Musical quality score ({report.musical_quality_score:.2f}) below target 0.85. "
                f"Review musical quality validators for specific issues."
            )

        report.total_validation_time = time.time() - start_time

        # Save report
        report_path = self.results_storage_dir / f"{report.report_id}.json"
        report.save_to_file(report_path)

        return report

    def generate_text_report(self, report: ValidationReport) -> str:
        """
        Generate human-readable text report.

        Args:
            report: ValidationReport

        Returns:
            Formatted text report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"\nReport ID: {report.report_id}")
        lines.append(f"Timestamp: {report.timestamp}")
        lines.append(f"Model Version: {report.model_version or 'N/A'}")
        lines.append(f"\nOverall Status: {'✓ PASSED' if report.overall_passed else '✗ FAILED'}")
        lines.append(f"Overall Score: {report.overall_score:.3f}")
        lines.append("")

        # Parameter validation summary
        lines.append("-" * 80)
        lines.append("PARAMETER VALIDATION")
        lines.append("-" * 80)
        lines.append(f"Total Parameters: {report.total_parameters_validated}")
        lines.append(f"Passed: {report.parameters_passed}")
        lines.append(f"Failed: {report.parameters_failed}")
        lines.append(f"Accuracy: {report.parameters_passed / report.total_parameters_validated * 100:.1f}%"
                    if report.total_parameters_validated > 0 else "Accuracy: N/A")
        lines.append("")

        # Show failed parameters
        failed_params = [r for r in report.parameter_validation_results if not r.passed]
        if failed_params:
            lines.append("Failed Parameters:")
            for r in failed_params[:10]:  # Show first 10
                lines.append(f"  - {r.parameter_name}: error={r.error:.3f}" if r.error else f"  - {r.parameter_name}")
        lines.append("")

        # Musical quality
        if report.musical_validation_results:
            lines.append("-" * 80)
            lines.append("MUSICAL QUALITY VALIDATION")
            lines.append("-" * 80)
            lines.append(f"Overall Musical Quality: {report.musical_quality_score:.3f}")
            for r in report.musical_validation_results:
                status = "✓" if r.passed else "✗"
                lines.append(f"  {status} {r.validation_type}: {r.score:.3f}")
            lines.append("")

        # Genre validation
        if report.genre_validation_results:
            lines.append("-" * 80)
            lines.append("GENRE VALIDATION")
            lines.append("-" * 80)
            for r in report.genre_validation_results:
                status = "✓" if r.passed else "✗"
                lines.append(f"  {status} {r.genre}: authenticity={r.authenticity_score:.3f}")
            lines.append("")

        # Critical issues
        if report.critical_issues:
            lines.append("-" * 80)
            lines.append("CRITICAL ISSUES")
            lines.append("-" * 80)
            for issue in report.critical_issues:
                lines.append(f"  ❌ {issue}")
            lines.append("")

        # Recommendations
        if report.recommendations:
            lines.append("-" * 80)
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in report.recommendations:
                lines.append(f"  💡 {rec}")
            lines.append("")

        lines.append("-" * 80)
        lines.append(f"Validation completed in {report.total_validation_time:.2f}s")
        lines.append("=" * 80)

        return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("Agent 08: Validation Pipeline Framework")
    print("=" * 60)
    print("\nThis module provides the core validation infrastructure.")
    print("\nUsage example:")
    print("""
    from validation.validation_pipeline import ValidationPipeline, ParameterValidator

    # Create validators
    param_validators = [
        ParameterValidator({
            'name': 'Tempo',
            'path': 'tempo.bpm',
            'type': 'continuous',
            'threshold': 0.05  # 5% error allowed
        })
    ]

    # Create pipeline
    pipeline = ValidationPipeline(parameter_validators=param_validators)

    # Validate
    report = pipeline.validate_complete(
        predictions={'tempo.bpm': 125},
        ground_truths={'tempo.bpm': 120}
    )

    # Print report
    print(pipeline.generate_text_report(report))
    """)
