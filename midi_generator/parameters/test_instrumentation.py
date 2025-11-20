#!/usr/bin/env python3
"""
Validation Tests for Instrumentation Parameters - Agent 7
=========================================================

Tests that all 50 instrumentation parameters:
1. Are properly defined
2. Have valid default values
3. Work correctly in the parameterized orchestrator
4. Cover all orchestration decisions

Author: Agent 7 - Instrumentation & Orchestration
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from parameters.instrumentation_params import (
    INSTRUMENTATION_PARAMETERS,
    get_instrumentation_parameter,
    get_all_instrumentation_parameters,
    get_parameters_by_category,
    get_default_values,
    validate_parameter_value,
    get_parameter_statistics,
    ParameterType
)

from generators.orchestrator_parameterized import (
    ParameterizedOrchestrator,
    create_orchestrator_from_parameters,
    OrchestrationStyle
)


# =============================================================================
# TEST UTILITIES
# =============================================================================

class TestResult:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def assert_true(self, condition: bool, test_name: str, message: str = ""):
        """Assert condition is true"""
        if condition:
            self.passed += 1
            self.tests.append((test_name, True, message))
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.tests.append((test_name, False, message))
            print(f"  ✗ {test_name}: {message}")

    def assert_equal(self, actual, expected, test_name: str):
        """Assert values are equal"""
        if actual == expected:
            self.passed += 1
            self.tests.append((test_name, True, ""))
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            msg = f"Expected {expected}, got {actual}"
            self.tests.append((test_name, False, msg))
            print(f"  ✗ {test_name}: {msg}")

    def summary(self):
        """Print summary"""
        total = self.passed + self.failed
        print(f"\n{'='*80}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"FAILED: {self.failed}")
            print("\nFailed tests:")
            for name, passed, msg in self.tests:
                if not passed:
                    print(f"  - {name}: {msg}")
        print(f"{'='*80}\n")
        return self.failed == 0


# =============================================================================
# PARAMETER DEFINITION TESTS
# =============================================================================

def test_parameter_count():
    """Test that we have exactly 50 parameters"""
    print("\n" + "="*80)
    print("TEST 1: PARAMETER COUNT")
    print("="*80)

    results = TestResult()

    total = len(INSTRUMENTATION_PARAMETERS)
    results.assert_equal(total, 50, "Total parameter count is 50")

    # Check category counts
    by_category = get_parameter_statistics()['by_category']
    results.assert_equal(by_category['doubling'], 10, "10 doubling parameters")
    results.assert_equal(by_category['voicing'], 12, "12 voicing parameters")
    results.assert_equal(by_category['dynamics'], 8, "8 dynamics parameters")
    results.assert_equal(by_category['selection'], 10, "10 selection parameters")
    results.assert_equal(by_category['register'], 5, "5 register parameters")
    results.assert_equal(by_category['technique'], 5, "5 technique parameters")

    return results


def test_parameter_definitions():
    """Test that all parameters are properly defined"""
    print("\n" + "="*80)
    print("TEST 2: PARAMETER DEFINITIONS")
    print("="*80)

    results = TestResult()

    for name, param in INSTRUMENTATION_PARAMETERS.items():
        # Check name format
        results.assert_true(
            name.startswith('instrumentation.'),
            f"{name}: starts with 'instrumentation.'"
        )

        # Check has description
        results.assert_true(
            len(param.description) > 0,
            f"{name}: has description"
        )

        # Check type-specific properties
        if param.type == ParameterType.CONTINUOUS:
            results.assert_true(
                param.range is not None and len(param.range) == 2,
                f"{name}: continuous has range"
            )
            results.assert_true(
                param.range[0] <= param.default <= param.range[1],
                f"{name}: default in range"
            )

        elif param.type == ParameterType.DISCRETE:
            results.assert_true(
                param.range is not None,
                f"{name}: discrete has range"
            )
            results.assert_true(
                isinstance(param.default, int),
                f"{name}: discrete default is int"
            )

        elif param.type == ParameterType.CATEGORICAL:
            results.assert_true(
                param.options is not None and len(param.options) > 0,
                f"{name}: categorical has options"
            )
            results.assert_true(
                param.default in param.options,
                f"{name}: default in options"
            )

        elif param.type == ParameterType.BOOLEAN:
            results.assert_true(
                isinstance(param.default, bool),
                f"{name}: boolean default is bool"
            )

    return results


def test_parameter_validation():
    """Test parameter value validation"""
    print("\n" + "="*80)
    print("TEST 3: PARAMETER VALIDATION")
    print("="*80)

    results = TestResult()

    # Test continuous parameter
    results.assert_true(
        validate_parameter_value('instrumentation.doubling.octave_probability', 0.5),
        "Valid continuous value accepted"
    )
    results.assert_true(
        not validate_parameter_value('instrumentation.doubling.octave_probability', 1.5),
        "Invalid continuous value rejected"
    )

    # Test discrete parameter
    results.assert_true(
        validate_parameter_value('instrumentation.voicing.density', 4),
        "Valid discrete value accepted"
    )
    results.assert_true(
        not validate_parameter_value('instrumentation.voicing.density', 10),
        "Out-of-range discrete value rejected"
    )

    # Test categorical parameter
    results.assert_true(
        validate_parameter_value('instrumentation.selection.orchestration_size', 'orchestra'),
        "Valid categorical value accepted"
    )
    results.assert_true(
        not validate_parameter_value('instrumentation.selection.orchestration_size', 'invalid'),
        "Invalid categorical value rejected"
    )

    # Test boolean parameter
    results.assert_true(
        validate_parameter_value('instrumentation.doubling.avoid_muddy_bass', True),
        "Valid boolean value accepted"
    )

    return results


# =============================================================================
# ORCHESTRATOR INTEGRATION TESTS
# =============================================================================

def test_orchestrator_creation():
    """Test creating parameterized orchestrator"""
    print("\n" + "="*80)
    print("TEST 4: ORCHESTRATOR CREATION")
    print("="*80)

    results = TestResult()

    # Test with defaults
    orch_default = ParameterizedOrchestrator()
    results.assert_equal(
        len(orch_default.params),
        50,
        "Orchestrator loaded all 50 parameters"
    )

    # Test with custom parameters
    custom_params = {
        'instrumentation.doubling.octave_probability': 0.9,
        'instrumentation.voicing.spread_factor': 0.8,
    }
    orch_custom = create_orchestrator_from_parameters(custom_params)

    results.assert_equal(
        orch_custom.params['instrumentation.doubling.octave_probability'],
        0.9,
        "Custom parameter applied"
    )

    # Test style-based adjustments
    orch_classical = ParameterizedOrchestrator(style=OrchestrationStyle.CLASSICAL)
    orch_romantic = ParameterizedOrchestrator(style=OrchestrationStyle.ROMANTIC)

    results.assert_true(
        orch_classical.params['instrumentation.doubling.octave_probability'] <
        orch_romantic.params['instrumentation.doubling.octave_probability'],
        "Classical has less doubling than Romantic"
    )

    results.assert_true(
        orch_classical.params['instrumentation.dynamics.pp_to_ff_range'] <
        orch_romantic.params['instrumentation.dynamics.pp_to_ff_range'],
        "Classical has narrower dynamic range than Romantic"
    )

    return results


def test_parameter_coverage():
    """Test that parameters cover all orchestration decisions"""
    print("\n" + "="*80)
    print("TEST 5: PARAMETER COVERAGE")
    print("="*80)

    results = TestResult()

    # Check we have parameters for key orchestration decisions
    key_decisions = [
        ('instrumentation.doubling.octave_probability', "Octave doubling"),
        ('instrumentation.voicing.close_position_ratio', "Voicing spacing"),
        ('instrumentation.dynamics.balance_melody_ratio', "Melody balance"),
        ('instrumentation.selection.orchestration_size', "Ensemble size"),
        ('instrumentation.register.prefer_high_melody', "Melody register"),
        ('instrumentation.technique.divisi_probability', "String divisi"),
    ]

    for param_name, description in key_decisions:
        results.assert_true(
            param_name in INSTRUMENTATION_PARAMETERS,
            f"Has parameter for: {description}"
        )

    # Check all categories are represented
    categories = ['doubling', 'voicing', 'dynamics', 'selection', 'register', 'technique']
    for category in categories:
        params_in_category = get_parameters_by_category(category)
        results.assert_true(
            len(params_in_category) > 0,
            f"Category '{category}' has parameters"
        )

    return results


def test_default_values():
    """Test that all default values are reasonable"""
    print("\n" + "="*80)
    print("TEST 6: DEFAULT VALUES")
    print("="*80)

    results = TestResult()

    defaults = get_default_values()

    # Check all defaults are set
    results.assert_equal(
        len(defaults),
        50,
        "All parameters have defaults"
    )

    # Check some specific defaults are reasonable
    results.assert_true(
        0.0 <= defaults['instrumentation.doubling.octave_probability'] <= 1.0,
        "Octave doubling probability in [0,1]"
    )

    results.assert_true(
        defaults['instrumentation.voicing.density'] >= 2,
        "Voicing density at least 2"
    )

    results.assert_true(
        defaults['instrumentation.dynamics.balance_melody_ratio'] >= 1.0,
        "Melody louder than or equal to accompaniment"
    )

    results.assert_true(
        defaults['instrumentation.selection.orchestration_size'] in
        ['solo', 'chamber', 'small_ensemble', 'orchestra', 'large_orchestra'],
        "Orchestration size is valid"
    )

    return results


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_parameter_modification():
    """Test modifying parameters at runtime"""
    print("\n" + "="*80)
    print("TEST 7: RUNTIME PARAMETER MODIFICATION")
    print("="*80)

    results = TestResult()

    orch = ParameterizedOrchestrator()

    original_value = orch.params['instrumentation.doubling.octave_probability']

    # Modify parameter
    orch.set_parameter('instrumentation.doubling.octave_probability', 0.95)

    results.assert_equal(
        orch.params['instrumentation.doubling.octave_probability'],
        0.95,
        "Parameter modified successfully"
    )

    results.assert_true(
        orch.params['instrumentation.doubling.octave_probability'] != original_value,
        "Parameter value changed"
    )

    # Modify multiple parameters
    new_params = {
        'instrumentation.voicing.spread_factor': 0.85,
        'instrumentation.dynamics.balance_melody_ratio': 1.8,
    }
    orch.set_parameters(new_params)

    results.assert_equal(
        orch.params['instrumentation.voicing.spread_factor'],
        0.85,
        "Bulk parameter update works"
    )

    return results


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*80)
    print("INSTRUMENTATION PARAMETER VALIDATION - Agent 7")
    print("="*80)

    all_results = []

    # Run all test suites
    all_results.append(test_parameter_count())
    all_results.append(test_parameter_definitions())
    all_results.append(test_parameter_validation())
    all_results.append(test_orchestrator_creation())
    all_results.append(test_parameter_coverage())
    all_results.append(test_default_values())
    all_results.append(test_parameter_modification())

    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL TEST SUMMARY")
    print("="*80)

    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_tests = total_passed + total_failed

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success Rate: {total_passed/total_tests*100:.1f}%")

    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Agent 7 (Instrumentation & Orchestration) Complete")
        print("   - 50 parameters defined")
        print("   - Orchestrator refactored")
        print("   - All tests passing")
    else:
        print(f"\n⚠️  {total_failed} tests failed")

    print("="*80 + "\n")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
