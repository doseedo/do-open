#!/usr/bin/env python3
"""
Parameter Coupling Validation - Agent 7
========================================

This module validates musical coherence through parameter coupling constraints.

Musical coupling rules ensure that parameters across dimensions make musical sense together.
For example:
- High harmony complexity should correlate with dense texture
- Form changes should align with orchestration changes
- Rhythmic activity should match harmonic rhythm

Author: Agent 7 - Cross-Dimensional Pattern Discoverer
Date: November 21, 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import json
from pathlib import Path


class CouplingType(Enum):
    """Types of parameter coupling"""
    CORRELATION = "correlation"  # Parameters should correlate
    ANTICORRELATION = "anticorrelation"  # Parameters should anticorrelate
    THRESHOLD = "threshold"  # If one > threshold, other must satisfy condition
    ALIGNMENT = "alignment"  # Parameters should have similar temporal patterns
    HIERARCHY = "hierarchy"  # One parameter should dominate another


@dataclass
class CouplingConstraint:
    """
    Represents a coupling constraint between parameters.

    Example:
        If harmony_complexity > 0.7, then texture_density > 0.5
    """
    name: str
    dimension_a: str
    parameter_a: str
    dimension_b: str
    parameter_b: str
    coupling_type: CouplingType
    strength_threshold: float = 0.3  # Minimum coupling strength to be valid
    condition: Optional[Callable] = None  # Custom validation function
    description: str = ""

    def validate(
        self,
        value_a: float,
        value_b: float,
        tolerance: float = 0.1
    ) -> Tuple[bool, float]:
        """
        Validate if parameter values satisfy coupling constraint.

        Args:
            value_a: Value of parameter A
            value_b: Value of parameter B
            tolerance: Tolerance for constraint satisfaction

        Returns:
            (is_valid, violation_score)
        """
        if self.condition is not None:
            # Custom validation function
            is_valid = self.condition(value_a, value_b)
            violation = 0.0 if is_valid else 1.0
            return is_valid, violation

        # Standard coupling validations
        if self.coupling_type == CouplingType.CORRELATION:
            # Both should be high or both low
            diff = abs(value_a - value_b)
            is_valid = diff < tolerance
            violation = diff

        elif self.coupling_type == CouplingType.ANTICORRELATION:
            # One high → other low
            target_diff = 1.0 - tolerance
            actual_diff = abs(value_a + value_b - 1.0)
            is_valid = actual_diff < tolerance
            violation = actual_diff

        elif self.coupling_type == CouplingType.THRESHOLD:
            # Implemented via custom condition
            is_valid = True
            violation = 0.0

        elif self.coupling_type == CouplingType.ALIGNMENT:
            # Temporal alignment (simplified as correlation here)
            diff = abs(value_a - value_b)
            is_valid = diff < tolerance
            violation = diff

        elif self.coupling_type == CouplingType.HIERARCHY:
            # A should dominate B (A >= B)
            is_valid = value_a >= value_b - tolerance
            violation = max(0.0, value_b - value_a)

        else:
            is_valid = True
            violation = 0.0

        return is_valid, violation

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'dimension_a': self.dimension_a,
            'parameter_a': self.parameter_a,
            'dimension_b': self.dimension_b,
            'parameter_b': self.parameter_b,
            'coupling_type': self.coupling_type.value,
            'strength_threshold': self.strength_threshold,
            'description': self.description
        }


class ParameterCouplingValidator:
    """
    Validates musical coherence through parameter coupling constraints.

    This class maintains a set of coupling constraints derived from music theory
    and validates that cross-dimensional parameters satisfy these constraints.
    """

    def __init__(self):
        """Initialize validator with default musical coupling constraints"""
        self.constraints: List[CouplingConstraint] = []
        self._initialize_default_constraints()

    def _initialize_default_constraints(self):
        """Initialize music theory-based coupling constraints"""

        # 1. Harmony-Texture Coupling
        self.constraints.append(CouplingConstraint(
            name="harmony_texture_density",
            dimension_a="harmony",
            parameter_a="complexity",
            dimension_b="texture",
            parameter_b="density",
            coupling_type=CouplingType.CORRELATION,
            strength_threshold=0.3,
            description="Complex harmony should correlate with dense texture"
        ))

        # 2. Form-Orchestration Coupling
        self.constraints.append(CouplingConstraint(
            name="form_orchestration_change",
            dimension_a="form",
            parameter_a="section_change",
            dimension_b="orchestration",
            parameter_b="instrumentation_change",
            coupling_type=CouplingType.ALIGNMENT,
            strength_threshold=0.4,
            description="Formal section changes should align with orchestration changes"
        ))

        # 3. Rhythm-Harmony Coupling
        self.constraints.append(CouplingConstraint(
            name="rhythm_harmony_activity",
            dimension_a="rhythm",
            parameter_a="syncopation",
            dimension_b="harmony",
            parameter_b="harmonic_rhythm",
            coupling_type=CouplingType.CORRELATION,
            strength_threshold=0.3,
            description="Rhythmic complexity should correlate with harmonic rhythm"
        ))

        # 4. Climax Convergence
        # All dimensions should converge at climax points
        self.constraints.append(CouplingConstraint(
            name="climax_convergence",
            dimension_a="form",
            parameter_a="climax_position",
            dimension_b="orchestration",
            parameter_b="intensity",
            coupling_type=CouplingType.ALIGNMENT,
            strength_threshold=0.5,
            description="Formal climax should align with orchestral intensity peak"
        ))

        # 5. Texture-Orchestration Coupling
        self.constraints.append(CouplingConstraint(
            name="texture_orchestration_balance",
            dimension_a="texture",
            parameter_a="voice_independence",
            dimension_b="orchestration",
            parameter_b="instrument_count",
            coupling_type=CouplingType.CORRELATION,
            strength_threshold=0.3,
            description="Voice independence should correlate with instrument count"
        ))

        # 6. Structural Harmonic Anchoring
        # Harmony should be stable at section boundaries
        def stable_harmony_at_boundaries(form_boundary: float, harmony_stability: float) -> bool:
            """If form boundary (high value), harmony should be stable (high value)"""
            if form_boundary > 0.7:
                return harmony_stability > 0.5
            return True

        self.constraints.append(CouplingConstraint(
            name="structural_harmonic_anchoring",
            dimension_a="form",
            parameter_a="boundary_strength",
            dimension_b="harmony",
            parameter_b="stability",
            coupling_type=CouplingType.THRESHOLD,
            condition=stable_harmony_at_boundaries,
            strength_threshold=0.6,
            description="Harmony should be stable at strong formal boundaries"
        ))

        # 7. Rhythmic-Textural Density
        self.constraints.append(CouplingConstraint(
            name="rhythm_texture_density",
            dimension_a="rhythm",
            parameter_a="density",
            dimension_b="texture",
            parameter_b="density",
            coupling_type=CouplingType.CORRELATION,
            strength_threshold=0.4,
            description="Rhythmic density should correlate with textural density"
        ))

        # 8. Form-Driven Texture Change
        self.constraints.append(CouplingConstraint(
            name="form_texture_variation",
            dimension_a="form",
            parameter_a="contrast_degree",
            dimension_b="texture",
            parameter_b="variation",
            coupling_type=CouplingType.CORRELATION,
            strength_threshold=0.4,
            description="Formal contrast should drive textural variation"
        ))

    def validate_parameters(
        self,
        dimension_parameters: Dict[str, Dict[str, float]],
        tolerance: float = 0.15
    ) -> Dict[str, Any]:
        """
        Validate all coupling constraints.

        Args:
            dimension_parameters: Nested dict like:
                {
                    'harmony': {'complexity': 0.8, 'stability': 0.6, ...},
                    'rhythm': {'syncopation': 0.7, ...},
                    ...
                }
            tolerance: Tolerance for constraint satisfaction

        Returns:
            Validation report with results for each constraint
        """
        results = {
            'overall_valid': True,
            'num_constraints': len(self.constraints),
            'num_satisfied': 0,
            'num_violated': 0,
            'average_violation': 0.0,
            'constraint_results': []
        }

        total_violation = 0.0

        for constraint in self.constraints:
            # Get parameter values
            try:
                # Try to get exact parameter names
                value_a = dimension_parameters.get(constraint.dimension_a, {}).get(
                    constraint.parameter_a, None
                )
                value_b = dimension_parameters.get(constraint.dimension_b, {}).get(
                    constraint.parameter_b, None
                )

                # If exact parameters not found, use aggregates
                if value_a is None:
                    dim_params = dimension_parameters.get(constraint.dimension_a, {})
                    value_a = np.mean(list(dim_params.values())) if dim_params else 0.5

                if value_b is None:
                    dim_params = dimension_parameters.get(constraint.dimension_b, {})
                    value_b = np.mean(list(dim_params.values())) if dim_params else 0.5

                # Validate constraint
                is_valid, violation = constraint.validate(value_a, value_b, tolerance)

                result = {
                    'constraint': constraint.name,
                    'description': constraint.description,
                    'valid': is_valid,
                    'violation': float(violation),
                    'value_a': float(value_a),
                    'value_b': float(value_b)
                }

                results['constraint_results'].append(result)

                if is_valid:
                    results['num_satisfied'] += 1
                else:
                    results['num_violated'] += 1
                    results['overall_valid'] = False

                total_violation += violation

            except Exception as e:
                print(f"Warning: Could not validate constraint {constraint.name}: {e}")

        # Compute average violation
        if len(self.constraints) > 0:
            results['average_violation'] = total_violation / len(self.constraints)

        return results

    def validate_cross_dimensional_parameters(
        self,
        cross_params: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Validate cross-dimensional parameters directly.

        Args:
            cross_params: Array of 10 cross-dimensional parameters
            threshold: Threshold for parameter validity

        Returns:
            Validation results
        """
        # Cross-dimensional parameter names (from CrossDimensionalParameters)
        param_names = [
            'harmonic_rhythmic_coupling',
            'form_driven_texture_change',
            'structural_harmonic_anchoring',
            'orchestral_intensity_gradient',
            'climax_convergence_factor',
            'texture_density_correlation',
            'rhythmic_harmonic_tension',
            'formal_orchestration_coupling',
            'cross_dimensional_coherence',
            'stylistic_consistency_score'
        ]

        # Validation rules for cross-dimensional parameters
        validations = {}

        # 1. Harmonic-rhythmic coupling should be moderate to high
        validations['harmonic_rhythmic_coupling'] = {
            'value': float(cross_params[0]),
            'valid': cross_params[0] > 0.3,
            'rule': 'Should show some coupling (> 0.3)'
        }

        # 2. Form-driven texture change should be significant
        validations['form_driven_texture_change'] = {
            'value': float(cross_params[1]),
            'valid': cross_params[1] > 0.4,
            'rule': 'Texture should change with form (> 0.4)'
        }

        # 3. Structural harmonic anchoring should be strong
        validations['structural_harmonic_anchoring'] = {
            'value': float(cross_params[2]),
            'valid': cross_params[2] > 0.5,
            'rule': 'Harmony should anchor structure (> 0.5)'
        }

        # 4. Orchestral intensity should have clear gradient
        validations['orchestral_intensity_gradient'] = {
            'value': float(cross_params[3]),
            'valid': cross_params[3] > 0.3,
            'rule': 'Should have orchestral arc (> 0.3)'
        }

        # 5. Climax convergence should be strong
        validations['climax_convergence_factor'] = {
            'value': float(cross_params[4]),
            'valid': cross_params[4] > threshold,
            'rule': f'Dimensions should converge at climax (> {threshold})'
        }

        # 6. Texture-density correlation should exist
        validations['texture_density_correlation'] = {
            'value': float(cross_params[5]),
            'valid': cross_params[5] > 0.3,
            'rule': 'Harmony and texture should correlate (> 0.3)'
        }

        # 7. Rhythmic-harmonic tension (can be any value)
        validations['rhythmic_harmonic_tension'] = {
            'value': float(cross_params[6]),
            'valid': True,  # No strict requirement
            'rule': 'No strict requirement'
        }

        # 8. Formal-orchestration coupling
        validations['formal_orchestration_coupling'] = {
            'value': float(cross_params[7]),
            'valid': cross_params[7] > 0.4,
            'rule': 'Form and orchestration should couple (> 0.4)'
        }

        # 9. Overall cross-dimensional coherence (should be high)
        validations['cross_dimensional_coherence'] = {
            'value': float(cross_params[8]),
            'valid': cross_params[8] > 0.6,
            'rule': 'Overall coherence should be strong (> 0.6)'
        }

        # 10. Stylistic consistency (should be high)
        validations['stylistic_consistency_score'] = {
            'value': float(cross_params[9]),
            'valid': cross_params[9] > 0.5,
            'rule': 'Style should be consistent (> 0.5)'
        }

        # Overall validity
        all_valid = all(v['valid'] for v in validations.values())

        return {
            'overall_valid': all_valid,
            'validations': validations,
            'num_valid': sum(1 for v in validations.values() if v['valid']),
            'num_invalid': sum(1 for v in validations.values() if not v['valid'])
        }

    def add_custom_constraint(self, constraint: CouplingConstraint):
        """Add a custom coupling constraint"""
        self.constraints.append(constraint)

    def export_constraints(self, output_path: Path):
        """Export coupling constraints to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        constraints_dict = {
            'num_constraints': len(self.constraints),
            'constraints': [c.to_dict() for c in self.constraints]
        }

        with open(output_path, 'w') as f:
            json.dump(constraints_dict, f, indent=2)

        print(f"✅ Exported {len(self.constraints)} constraints to {output_path}")

    def generate_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable validation report"""
        report = []
        report.append("=" * 80)
        report.append("PARAMETER COUPLING VALIDATION REPORT")
        report.append("=" * 80)

        overall = "✅ VALID" if validation_results['overall_valid'] else "❌ INVALID"
        report.append(f"\nOverall Status: {overall}")
        report.append(f"Satisfied: {validation_results['num_satisfied']}/{validation_results['num_constraints']}")
        report.append(f"Average Violation: {validation_results['average_violation']:.3f}\n")

        # Individual constraints
        report.append("CONSTRAINT DETAILS:")
        report.append("-" * 80)

        for result in validation_results['constraint_results']:
            status = "✓" if result['valid'] else "✗"
            report.append(f"\n{status} {result['constraint']}")
            report.append(f"   {result['description']}")
            report.append(f"   Value A: {result['value_a']:.3f}, Value B: {result['value_b']:.3f}")
            if not result['valid']:
                report.append(f"   Violation: {result['violation']:.3f}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Parameter Coupling Validation - Agent 7")
    print("=" * 80)

    # Create validator
    print("\n1. Creating coupling validator...")
    validator = ParameterCouplingValidator()
    print(f"   ✅ Initialized with {len(validator.constraints)} default constraints")

    # Test dimension parameters
    print("\n2. Testing dimension parameter validation...")
    dimension_parameters = {
        'harmony': {
            'complexity': 0.8,
            'stability': 0.6,
            'harmonic_rhythm': 0.7
        },
        'rhythm': {
            'syncopation': 0.65,
            'density': 0.75
        },
        'form': {
            'section_change': 0.7,
            'boundary_strength': 0.8,
            'contrast_degree': 0.6,
            'climax_position': 0.75
        },
        'orchestration': {
            'instrumentation_change': 0.65,
            'intensity': 0.7,
            'instrument_count': 0.6
        },
        'texture': {
            'density': 0.7,
            'voice_independence': 0.55,
            'variation': 0.65
        }
    }

    validation_results = validator.validate_parameters(dimension_parameters)
    report = validator.generate_report(validation_results)
    print(report)

    # Test cross-dimensional parameters
    print("\n3. Testing cross-dimensional parameter validation...")
    cross_params = np.array([
        0.65,  # harmonic_rhythmic_coupling
        0.55,  # form_driven_texture_change
        0.70,  # structural_harmonic_anchoring
        0.60,  # orchestral_intensity_gradient
        0.75,  # climax_convergence_factor
        0.65,  # texture_density_correlation
        0.50,  # rhythmic_harmonic_tension
        0.60,  # formal_orchestration_coupling
        0.70,  # cross_dimensional_coherence
        0.65   # stylistic_consistency_score
    ])

    cross_validation = validator.validate_cross_dimensional_parameters(cross_params)
    print(f"\n   Overall: {'✅ VALID' if cross_validation['overall_valid'] else '❌ INVALID'}")
    print(f"   Valid parameters: {cross_validation['num_valid']}/10")

    for param_name, result in cross_validation['validations'].items():
        status = "✓" if result['valid'] else "✗"
        print(f"   {status} {param_name}: {result['value']:.3f} ({result['rule']})")

    # Export constraints
    print("\n4. Exporting constraints...")
    output_path = Path("/tmp/coupling_constraints.json")
    validator.export_constraints(output_path)

    print("\n" + "=" * 80)
    print("✅ Coupling validation complete!")
    print("=" * 80)
