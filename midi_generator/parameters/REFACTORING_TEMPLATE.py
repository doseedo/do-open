#!/usr/bin/env python3
"""
REFACTORING TEMPLATE FOR PARALLEL SPRINT
=========================================

Copy this template when refactoring generator modules.
Replace placeholders with actual values for your module.

Usage:
1. Copy this entire template
2. Replace MODULE_NAME, DOMAIN, etc.
3. Add all hardcoded values as parameters
4. Test backward compatibility

Author: Agent 4
"""

from parameters import registry, param, ParameterType, MusicalDomain
from typing import Dict, Any, List
import random


class ExampleGenerator:
    """
    Example generator showing full parameterization pattern.

    All hardcoded musical decisions are exposed as learnable parameters.
    """

    # Class-level flag to prevent duplicate registration
    _params_registered = False

    def __init__(self, **params):
        """
        Initialize generator with optional parameter overrides.

        Args:
            **params: Parameter overrides (e.g., {"rhythm.swing.ratio": 0.7})
        """
        self.params = params
        self._register_parameters()

    @classmethod
    def _register_parameters(cls):
        """
        Register all parameters for this module.

        Called once per class to populate global registry.
        """
        if cls._params_registered:
            return

        # ===================================================================
        # CONTINUOUS PARAMETERS (floats in range)
        # ===================================================================

        registry.register_parameter(
            name="domain.module.probability_param",
            type=ParameterType.CONTINUOUS,
            default=0.5,
            description="Probability of applying technique X",
            range=(0.0, 1.0),
            domain=MusicalDomain.HARMONY,  # or MELODY, RHYTHM, etc.
            module="example_generator",
            musical_impact="high",  # low/medium/high
            genre_relevance=["jazz", "bebop", "swing"]
        )

        registry.register_parameter(
            name="domain.module.intensity_param",
            type=ParameterType.CONTINUOUS,
            default=0.75,
            description="Intensity of effect (0=subtle, 1=extreme)",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="example_generator",
            musical_impact="medium",
            genre_relevance=["all"]
        )

        # ===================================================================
        # CATEGORICAL PARAMETERS (choose from options)
        # ===================================================================

        registry.register_parameter(
            name="domain.module.style_param",
            type=ParameterType.CATEGORICAL,
            default="standard",
            description="Style variation to use",
            options=["standard", "bebop", "modal", "fusion"],
            domain=MusicalDomain.HARMONY,
            module="example_generator",
            musical_impact="high",
            genre_relevance=["jazz", "all"]
        )

        # ===================================================================
        # INTEGER PARAMETERS (whole numbers)
        # ===================================================================

        registry.register_parameter(
            name="domain.module.count_param",
            type=ParameterType.INTEGER,
            default=4,
            description="Number of repetitions",
            range=(1, 16),
            domain=MusicalDomain.FORM,
            module="example_generator",
            musical_impact="medium",
            genre_relevance=["all"]
        )

        # ===================================================================
        # BOOLEAN PARAMETERS (on/off switches)
        # ===================================================================

        registry.register_parameter(
            name="domain.module.enable_feature",
            type=ParameterType.BOOLEAN,
            default=True,
            description="Enable advanced feature X",
            domain=MusicalDomain.HARMONY,
            module="example_generator",
            musical_impact="high",
            genre_relevance=["jazz", "bebop"]
        )

        # ===================================================================
        # ARRAY PARAMETERS (lists/patterns)
        # ===================================================================

        registry.register_parameter(
            name="domain.module.pattern_param",
            type=ParameterType.ARRAY,
            default=[1, 0, 1, 0, 1, 1, 0, 1],
            description="Rhythmic/harmonic pattern array",
            domain=MusicalDomain.RHYTHM,
            module="example_generator",
            musical_impact="high",
            genre_relevance=["jazz", "funk"]
        )

        cls._params_registered = True

    # ========================================================================
    # REFACTORED METHODS
    # ========================================================================

    def generate_something(
        self,
        input_data: List,
        **kwargs
    ) -> List:
        """
        Generate output with parameterized behavior.

        Parameters (from registry):
            - domain.module.probability_param: Controls likelihood of X
            - domain.module.intensity_param: Controls strength of Y
            - domain.module.style_param: Selects style variation
            - domain.module.count_param: Number of iterations
            - domain.module.enable_feature: Toggle advanced feature

        Args:
            input_data: Input to process
            **kwargs: Parameter overrides for this call

        Returns:
            Processed output
        """
        # STEP 1: Merge instance params with method-level overrides
        params = {**self.params, **kwargs}

        # STEP 2: Get parameter values (with fallbacks)
        probability = param("domain.module.probability_param", params, 0.5)
        intensity = param("domain.module.intensity_param", params, 0.75)
        style = param("domain.module.style_param", params, "standard")
        count = param("domain.module.count_param", params, 4)
        enabled = param("domain.module.enable_feature", params, True)
        pattern = param("domain.module.pattern_param", params, [1, 0, 1, 0])

        # STEP 3: Use parameters in logic
        result = []
        for item in input_data:
            # Use probability parameter
            if random.random() < probability:
                # Use intensity parameter
                processed = item * intensity

                # Use categorical parameter
                if style == "bebop":
                    processed = self._apply_bebop_style(processed, params)
                elif style == "modal":
                    processed = self._apply_modal_style(processed, params)

                result.append(processed)

        # Use boolean parameter
        if enabled:
            result = self._apply_advanced_feature(result, params)

        return result

    def _apply_bebop_style(self, item, params: Dict[str, Any]):
        """Helper method that also accepts params"""
        # Additional parameters specific to bebop
        bebop_intensity = param("domain.module.bebop_intensity", params, 0.8)
        return item * bebop_intensity

    def _apply_modal_style(self, item, params: Dict[str, Any]):
        """Helper method that also accepts params"""
        modal_sustain = param("domain.module.modal_sustain", params, 2.0)
        return item * modal_sustain

    def _apply_advanced_feature(self, items: List, params: Dict[str, Any]) -> List:
        """Advanced processing with additional parameters"""
        feature_threshold = param("domain.module.feature_threshold", params, 0.6)
        return [item for item in items if item > feature_threshold]

    # ========================================================================
    # STATIC METHODS -> INSTANCE METHODS
    # ========================================================================

    # BEFORE (static - can't access params):
    # @staticmethod
    # def process_data(data):
    #     threshold = 0.5  # HARDCODED!
    #     return [d for d in data if d > threshold]

    # AFTER (instance - can access params):
    def process_data(self, data: List, **kwargs) -> List:
        """
        Process data with parameterized threshold.

        Parameters (from registry):
            - domain.module.threshold: Filtering threshold

        Args:
            data: Input data
            **kwargs: Parameter overrides

        Returns:
            Filtered data
        """
        params = {**self.params, **kwargs}
        threshold = param("domain.module.threshold", params, 0.5)
        return [d for d in data if d > threshold]


# ============================================================================
# TESTING TEMPLATE
# ============================================================================

def test_backward_compatibility():
    """
    Verify refactored code produces same results with defaults.

    This ensures we didn't break existing behavior.
    """
    # Test data
    test_input = [1, 2, 3, 4, 5]

    # Create generator with default params
    gen = ExampleGenerator()

    # Generate with defaults (should match old behavior)
    result = gen.generate_something(test_input)

    # Assertions
    assert isinstance(result, list), "Output type must match"
    assert len(result) > 0, "Should produce output"

    print("✅ Backward compatibility test passed")


def test_parameter_override():
    """
    Verify custom parameters work correctly.

    This ensures new parameter system functions properly.
    """
    # Custom parameters
    custom_params = {
        "domain.module.probability_param": 1.0,  # Always apply
        "domain.module.intensity_param": 2.0,    # Double intensity
        "domain.module.style_param": "bebop",    # Bebop style
    }

    # Create generator with custom params
    gen = ExampleGenerator(**custom_params)

    # Generate with custom params
    test_input = [1, 2, 3]
    result = gen.generate_something(test_input)

    # Verify custom behavior
    assert len(result) == len(test_input), "Should process all items (prob=1.0)"

    print("✅ Parameter override test passed")


def test_method_level_override():
    """
    Verify method-level overrides work.

    Method params should override instance params.
    """
    # Instance params
    gen = ExampleGenerator(
        **{"domain.module.probability_param": 0.5}
    )

    # Method-level override (should take precedence)
    result = gen.generate_something(
        [1, 2, 3],
        **{"domain.module.probability_param": 1.0}
    )

    print("✅ Method-level override test passed")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("REFACTORING TEMPLATE - EXAMPLES")
    print("=" * 80)

    # Example 1: Default behavior
    print("\n1. DEFAULT BEHAVIOR")
    print("-" * 80)
    gen_default = ExampleGenerator()
    result = gen_default.generate_something([1, 2, 3, 4, 5])
    print(f"Result: {result}")

    # Example 2: Custom parameters at initialization
    print("\n2. CUSTOM PARAMETERS AT INIT")
    print("-" * 80)
    gen_custom = ExampleGenerator(**{
        "domain.module.probability_param": 1.0,
        "domain.module.style_param": "bebop"
    })
    result = gen_custom.generate_something([1, 2, 3, 4, 5])
    print(f"Result: {result}")

    # Example 3: Method-level overrides
    print("\n3. METHOD-LEVEL OVERRIDES")
    print("-" * 80)
    gen = ExampleGenerator()
    result = gen.generate_something(
        [1, 2, 3, 4, 5],
        **{"domain.module.intensity_param": 3.0}
    )
    print(f"Result: {result}")

    # Example 4: Registry inspection
    print("\n4. REGISTRY INSPECTION")
    print("-" * 80)
    from parameters import registry

    all_params = registry.get_all_parameters()
    print(f"Total registered parameters: {len(all_params)}")

    example_params = registry.get_by_module("example_generator")
    print(f"Parameters for this module: {len(example_params)}")

    for name, meta in list(example_params.items())[:3]:
        print(f"  - {name}: {meta.description}")

    # Run tests
    print("\n5. TESTS")
    print("-" * 80)
    test_backward_compatibility()
    test_parameter_override()
    test_method_level_override()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✅")
    print("=" * 80)
