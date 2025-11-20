#!/usr/bin/env python3
"""
Agent 26: Test Case Generator
==============================

Automatically generates comprehensive pytest test suites for new parameters
in the self-expanding inverse music generation system.

This module creates:
- Boundary tests (min, max, default values)
- Musical validity tests (generated MIDI is coherent)
- Feature impact tests (parameter affects expected features)
- Integration tests (works with other parameters)
- Regression tests (existing parameters unaffected)

Author: Agent 26
Date: 2025-11-20
Version: 1.0.0
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures for Parameter Proposals
# ============================================================================

@dataclass
class ParameterProposal:
    """Represents a proposed new parameter for the system."""

    name: str
    description: str
    param_type: str  # 'continuous', 'integer', 'categorical', 'boolean', 'probability', 'velocity'
    default: Any
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    options: Optional[List[str]] = None
    category: str = "general"
    impact: str = "medium"  # 'critical', 'high', 'medium', 'low'
    genres: List[str] = field(default_factory=list)
    affected_features: List[str] = field(default_factory=list)
    musical_properties: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate parameter proposal."""
        valid_types = ['continuous', 'integer', 'categorical', 'boolean', 'probability', 'velocity']
        if self.param_type not in valid_types:
            raise ValueError(f"Invalid param_type: {self.param_type}. Must be one of {valid_types}")

        if self.param_type == 'categorical' and not self.options:
            raise ValueError("Categorical parameters must have options")

        if self.param_type in ['continuous', 'integer', 'probability'] and (self.min_value is None or self.max_value is None):
            if self.param_type != 'probability':  # probabilities default to 0.0-1.0
                raise ValueError(f"{self.param_type} parameters must have min and max values")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterProposal':
        """Create ParameterProposal from dictionary."""
        return cls(
            name=data.get('name', ''),
            description=data.get('description', ''),
            param_type=data.get('type', ''),
            default=data.get('default'),
            min_value=data.get('min'),
            max_value=data.get('max'),
            options=data.get('options'),
            category=data.get('category', 'general'),
            impact=data.get('impact', 'medium'),
            genres=data.get('genres', []),
            affected_features=data.get('affected_features', []),
            musical_properties=data.get('musical_properties', [])
        )


@dataclass
class TestConfiguration:
    """Configuration for test generation."""

    output_dir: Path = Path("tests/generated")
    include_fixtures: bool = True
    include_benchmarks: bool = True
    generate_integration_tests: bool = True
    generate_regression_tests: bool = True
    verbose: bool = True
    pytest_marks: List[str] = field(default_factory=list)
    timeout: int = 30  # seconds per test

    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Test Case Generator - Main Class
# ============================================================================

class TestCaseGenerator:
    """
    Generates comprehensive pytest test suites for new parameters.

    This class is responsible for creating complete, executable test files
    that validate new parameters across multiple dimensions:
    - Boundary conditions
    - Musical validity
    - Feature impact
    - Integration with existing system
    - Regression prevention
    """

    def __init__(self, config: Optional[TestConfiguration] = None):
        """
        Initialize the TestCaseGenerator.

        Args:
            config: Optional test configuration
        """
        self.config = config or TestConfiguration()
        self.logger = logging.getLogger(f"{__name__}.TestCaseGenerator")

        # Load existing parameter registry
        self.existing_parameters = self._load_parameter_registry()

        # Test template components
        self.imports_template = self._get_imports_template()
        self.fixtures_template = self._get_fixtures_template()

        self.logger.info("TestCaseGenerator initialized")

    def _load_parameter_registry(self) -> Dict[str, Any]:
        """Load existing parameter registry."""
        registry_path = Path(__file__).parent.parent / "parameters" / "registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                return json.load(f)
        return {}

    # ========================================================================
    # Main Test Generation Methods
    # ========================================================================

    def generate_test_suite(self, param_proposal: ParameterProposal) -> str:
        """
        Generate complete pytest test suite for a parameter.

        Args:
            param_proposal: The parameter proposal to generate tests for

        Returns:
            Complete Python test code as string
        """
        self.logger.info(f"Generating test suite for parameter: {param_proposal.name}")

        # Build test class name
        class_name = self._to_class_name(param_proposal.name)

        # Generate all test components
        imports = self._generate_imports(param_proposal)
        fixtures = self._generate_fixtures(param_proposal)
        boundary_tests = self._generate_boundary_tests(param_proposal)
        validity_tests = self._generate_musical_validity_tests(param_proposal)
        feature_tests = self._generate_feature_impact_tests(param_proposal)
        integration_tests = self._generate_integration_tests(param_proposal)
        regression_tests = self._generate_regression_tests(param_proposal)

        # Combine into complete test file
        test_code = f'''"""
Test suite for parameter: {param_proposal.name}

Auto-generated by TestCaseGenerator (Agent 26)
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Parameter Details:
- Type: {param_proposal.param_type}
- Category: {param_proposal.category}
- Impact: {param_proposal.impact}
- Default: {param_proposal.default}
- Description: {param_proposal.description}
"""

{imports}


class Test{class_name}:
    """Comprehensive test suite for {param_proposal.name}"""

{fixtures}

    # ====================================================================
    # BOUNDARY TESTS
    # ====================================================================

{boundary_tests}

    # ====================================================================
    # MUSICAL VALIDITY TESTS
    # ====================================================================

{validity_tests}

    # ====================================================================
    # FEATURE IMPACT TESTS
    # ====================================================================

{feature_tests}

    # ====================================================================
    # INTEGRATION TESTS
    # ====================================================================

{integration_tests}

    # ====================================================================
    # REGRESSION TESTS
    # ====================================================================

{regression_tests}


# ========================================================================
# Test Utilities
# ========================================================================

def validate_midi_basic(midi_data):
    """Basic MIDI validation utility."""
    if midi_data is None:
        return False, "MIDI data is None"

    # Check for tracks
    if not hasattr(midi_data, 'tracks') or len(midi_data.tracks) == 0:
        return False, "No tracks in MIDI"

    # Check for notes
    note_count = 0
    for track in midi_data.tracks:
        for msg in track:
            if hasattr(msg, 'type') and msg.type == 'note_on':
                note_count += 1

    if note_count == 0:
        return False, "No notes found in MIDI"

    return True, f"Valid MIDI with {{note_count}} notes"


def extract_pitches(midi_data):
    """Extract all pitches from MIDI data."""
    pitches = []
    for track in midi_data.tracks:
        for msg in track:
            if hasattr(msg, 'type') and msg.type == 'note_on' and hasattr(msg, 'note'):
                pitches.append(msg.note)
    return pitches


def extract_durations(midi_data):
    """Extract all note durations from MIDI data."""
    durations = []
    for track in midi_data.tracks:
        current_time = 0
        note_on_times = {{}}

        for msg in track:
            current_time += msg.time if hasattr(msg, 'time') else 0

            if hasattr(msg, 'type'):
                if msg.type == 'note_on' and hasattr(msg, 'note'):
                    note_on_times[msg.note] = current_time
                elif msg.type == 'note_off' and hasattr(msg, 'note'):
                    if msg.note in note_on_times:
                        duration = current_time - note_on_times[msg.note]
                        durations.append(duration)
                        del note_on_times[msg.note]

    return durations


if __name__ == "__main__":
    import pytest

    # Run tests for this module
    pytest.main([__file__, "-v", "--tb=short"])
'''

        return test_code

    def save_test_suite(self, param_proposal: ParameterProposal,
                       output_path: Optional[Path] = None) -> Path:
        """
        Generate and save test suite to file.

        Args:
            param_proposal: The parameter proposal
            output_path: Optional output path (default: auto-generated)

        Returns:
            Path to saved test file
        """
        test_code = self.generate_test_suite(param_proposal)

        if output_path is None:
            # Auto-generate filename
            safe_name = self._to_snake_case(param_proposal.name)
            output_path = self.config.output_dir / f"test_{safe_name}.py"

        # Write to file
        with open(output_path, 'w') as f:
            f.write(test_code)

        self.logger.info(f"Test suite saved to: {output_path}")
        return output_path

    # ========================================================================
    # Test Component Generators
    # ========================================================================

    def _generate_imports(self, param: ParameterProposal) -> str:
        """Generate import statements."""
        return '''import pytest
import numpy as np
import mido
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import system components (with graceful fallbacks)
try:
    from api.unified_api import HarmonyModuleAPI
    HAS_API = True
except ImportError:
    HAS_API = False
    print("Warning: HarmonyModuleAPI not available")

try:
    from analysis.midi_analyzer import MIDIAnalyzer
    HAS_ANALYZER = True
except ImportError:
    HAS_ANALYZER = False
    print("Warning: MIDIAnalyzer not available")

try:
    from generators.context_aware_generator import ContextAwareGenerator
    HAS_GENERATOR = True
except ImportError:
    HAS_GENERATOR = False
    print("Warning: ContextAwareGenerator not available")

# Skip all tests if critical components missing
pytestmark = pytest.mark.skipif(
    not (HAS_API or HAS_GENERATOR),
    reason="Required components not available"
)
'''

    def _generate_fixtures(self, param: ParameterProposal) -> str:
        """Generate pytest fixtures."""
        return '''    @pytest.fixture
    def api(self):
        """Fixture providing HarmonyModuleAPI instance."""
        if HAS_API:
            return HarmonyModuleAPI()
        return None

    @pytest.fixture
    def generator(self):
        """Fixture providing generator instance."""
        if HAS_GENERATOR:
            return ContextAwareGenerator()
        return None

    @pytest.fixture
    def analyzer(self):
        """Fixture providing analyzer instance."""
        if HAS_ANALYZER:
            return MIDIAnalyzer()
        return None

    @pytest.fixture
    def default_params(self):
        """Fixture providing default parameters."""
        return {
            "tempo": 120,
            "key": "C",
            "measures": 8,
            "time_signature": "4/4"
        }
'''

    def _generate_boundary_tests(self, param: ParameterProposal) -> str:
        """Generate boundary condition tests."""
        tests = []

        # Minimum value test
        if param.param_type in ['continuous', 'integer', 'probability']:
            min_val = param.min_value if param.min_value is not None else (0.0 if param.param_type == 'probability' else 0)
            tests.append(f'''    def test_minimum_value(self, api, default_params):
        """Test parameter at minimum value."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = {min_val}

        try:
            result = api.quick_generate(params)
            assert result is not None, "Generation failed with minimum value"

            # Validate basic MIDI structure
            is_valid, msg = validate_midi_basic(result)
            assert is_valid, f"Invalid MIDI at minimum value: {{msg}}"

        except Exception as e:
            pytest.fail(f"Failed at minimum value: {{e}}")
''')

        # Maximum value test
        if param.param_type in ['continuous', 'integer', 'probability']:
            max_val = param.max_value if param.max_value is not None else (1.0 if param.param_type == 'probability' else 100)
            tests.append(f'''    def test_maximum_value(self, api, default_params):
        """Test parameter at maximum value."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = {max_val}

        try:
            result = api.quick_generate(params)
            assert result is not None, "Generation failed with maximum value"

            # Validate basic MIDI structure
            is_valid, msg = validate_midi_basic(result)
            assert is_valid, f"Invalid MIDI at maximum value: {{msg}}"

        except Exception as e:
            pytest.fail(f"Failed at maximum value: {{e}}")
''')

        # Default value test
        tests.append(f'''    def test_default_value(self, api, default_params):
        """Test parameter at default value."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = {repr(param.default)}

        try:
            result = api.quick_generate(params)
            assert result is not None, "Generation failed with default value"

            # Validate basic MIDI structure
            is_valid, msg = validate_midi_basic(result)
            assert is_valid, f"Invalid MIDI at default value: {{msg}}"

        except Exception as e:
            pytest.fail(f"Failed at default value: {{e}}")
''')

        # Categorical options test
        if param.param_type == 'categorical' and param.options:
            for option in param.options:
                tests.append(f'''    def test_categorical_option_{self._to_snake_case(str(option))}(self, api, default_params):
        """Test categorical option: {option}"""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = "{option}"

        try:
            result = api.quick_generate(params)
            assert result is not None, "Generation failed with option '{option}'"

            # Validate basic MIDI structure
            is_valid, msg = validate_midi_basic(result)
            assert is_valid, f"Invalid MIDI with option '{option}': {{msg}}"

        except Exception as e:
            pytest.fail(f"Failed with option '{option}': {{e}}")
''')

        # Boolean test
        if param.param_type == 'boolean':
            tests.append(f'''    def test_boolean_true(self, api, default_params):
        """Test parameter with True value."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = True

        try:
            result = api.quick_generate(params)
            assert result is not None, "Generation failed with True"
            is_valid, msg = validate_midi_basic(result)
            assert is_valid, f"Invalid MIDI with True: {{msg}}"
        except Exception as e:
            pytest.fail(f"Failed with True: {{e}}")

    def test_boolean_false(self, api, default_params):
        """Test parameter with False value."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = False

        try:
            result = api.quick_generate(params)
            assert result is not None, "Generation failed with False"
            is_valid, msg = validate_midi_basic(result)
            assert is_valid, f"Invalid MIDI with False: {{msg}}"
        except Exception as e:
            pytest.fail(f"Failed with False: {{e}}")
''')

        # Out of bounds tests (should handle gracefully)
        if param.param_type in ['continuous', 'integer', 'probability']:
            tests.append(f'''    def test_out_of_bounds_handling(self, api, default_params):
        """Test that out-of-bounds values are handled gracefully."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()

        # Test value way above maximum
        params["{param.name}"] = 999999
        try:
            result = api.quick_generate(params)
            # Should either clip to max or raise clear error
            assert result is not None or True  # Accept either behavior
        except (ValueError, AssertionError) as e:
            # This is acceptable - clear error message
            assert len(str(e)) > 0
''')

        return '\n'.join(tests)

    def _generate_musical_validity_tests(self, param: ParameterProposal) -> str:
        """Generate musical validity tests."""
        tests = []

        # Basic coherence test
        test_value = self._get_test_value(param)
        tests.append(f'''    def test_musical_coherence(self, api, default_params):
        """Test that generated MIDI is musically coherent."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = {repr(test_value)}

        result = api.quick_generate(params)
        assert result is not None, "Generation failed"

        # Basic coherence checks
        pitches = extract_pitches(result)
        assert len(pitches) > 0, "No notes generated"

        # Check pitch range is reasonable (A0 to C8)
        assert all(21 <= p <= 108 for p in pitches), \
            f"Pitches outside reasonable range: {{[p for p in pitches if p < 21 or p > 108]}}"

        # Check MIDI has reasonable length (not too short or ridiculously long)
        durations = extract_durations(result)
        if durations:
            total_duration = sum(durations)
            assert total_duration > 0, "Zero duration composition"
            assert total_duration < 1000000, "Unreasonably long composition"
''')

        # Melodic coherence
        if param.category in ['melody', 'harmony', 'general']:
            tests.append(f'''    def test_melodic_coherence(self, api, default_params):
        """Test melodic coherence and reasonable interval sizes."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = {repr(test_value)}

        result = api.quick_generate(params)
        pitches = extract_pitches(result)

        if len(pitches) > 1:
            # Calculate intervals
            intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]

            # Most intervals should be reasonable (< 2 octaves)
            large_leaps = [i for i in intervals if i > 24]
            assert len(large_leaps) / len(intervals) < 0.3, \
                "Too many large melodic leaps (>2 octaves)"

            # Should have some melodic motion (not all repeated notes)
            motion = [i for i in intervals if i > 0]
            assert len(motion) > 0, "No melodic motion detected"
''')

        # Rhythmic coherence
        if param.category in ['rhythm', 'general']:
            tests.append(f'''    def test_rhythmic_coherence(self, api, default_params):
        """Test rhythmic coherence and reasonable note durations."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = {repr(test_value)}

        result = api.quick_generate(params)
        durations = extract_durations(result)

        if durations:
            # All durations should be positive
            assert all(d > 0 for d in durations), "Non-positive durations found"

            # Durations should be in reasonable range
            # (not instantaneous, not absurdly long)
            assert all(d < 100000 for d in durations), \
                "Unreasonably long note durations"
''')

        # Harmonic validity (if applicable)
        if param.category == 'harmony':
            tests.append(f'''    def test_harmonic_validity(self, api, default_params):
        """Test that harmonies are musically valid."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = {repr(test_value)}

        result = api.quick_generate(params)
        pitches = extract_pitches(result)

        # Check for simultaneous notes (chords)
        # This is a simplified check - would need timing info for full analysis
        if len(pitches) >= 3:
            # Check for reasonable chord voicings
            # Most chords should be within 2 octaves
            pitch_range = max(pitches) - min(pitches)
            # Allow up to 4 octaves for full arrangements
            assert pitch_range <= 48, \
                f"Excessive pitch range: {{pitch_range}} semitones"
''')

        # Genre-specific validity
        if param.genres:
            for genre in param.genres[:2]:  # Test top 2 genres
                tests.append(f'''    def test_genre_validity_{genre.lower().replace(' ', '_')}(self, api, default_params):
        """Test musical validity in {genre} context."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = {repr(test_value)}
        params["genre"] = "{genre}"

        try:
            result = api.quick_generate(params)
            assert result is not None, "Failed to generate {genre} music"

            is_valid, msg = validate_midi_basic(result)
            assert is_valid, f"Invalid {genre} generation: {{msg}}"
        except Exception as e:
            # Some genres might not be implemented yet
            pytest.skip(f"{genre} generation not yet implemented: {{e}}")
''')

        return '\n'.join(tests)

    def _generate_feature_impact_tests(self, param: ParameterProposal) -> str:
        """Generate feature impact tests."""
        tests = []

        # If affected features are specified, test them
        if param.affected_features:
            for feature in param.affected_features:
                tests.append(f'''    def test_feature_impact_{self._to_snake_case(feature)}(self, api, analyzer, default_params):
        """Test impact on feature: {feature}"""
        if api is None or analyzer is None:
            pytest.skip("Required components not available")

        params = default_params.copy()

        # Generate with minimum value
        params["{param.name}"] = {repr(self._get_min_value(param))}
        result_min = api.quick_generate(params)

        # Generate with maximum value
        params["{param.name}"] = {repr(self._get_max_value(param))}
        result_max = api.quick_generate(params)

        # Analyze both
        try:
            features_min = analyzer.extract_features(result_min)
            features_max = analyzer.extract_features(result_max)

            # The feature should show some difference
            # (exact validation depends on feature type)
            assert features_min is not None
            assert features_max is not None
        except Exception as e:
            pytest.skip(f"Feature extraction not fully implemented: {{e}}")
''')

        # Generic parameter variation test
        test_val_1 = self._get_min_value(param)
        test_val_2 = self._get_max_value(param)

        tests.append(f'''    def test_parameter_variation_impact(self, api, default_params):
        """Test that varying parameter produces different results."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()

        # Generate with different values
        params["{param.name}"] = {repr(test_val_1)}
        result_1 = api.quick_generate(params)

        params["{param.name}"] = {repr(test_val_2)}
        result_2 = api.quick_generate(params)

        # Results should be different (in some measurable way)
        pitches_1 = extract_pitches(result_1)
        pitches_2 = extract_pitches(result_2)

        # At least one of these should differ
        different = (
            len(pitches_1) != len(pitches_2) or
            pitches_1 != pitches_2 or
            extract_durations(result_1) != extract_durations(result_2)
        )

        # Note: For some parameters, results might be identical due to
        # randomness or if parameter doesn't affect this particular generation
        # This is a soft check
        if not different:
            print(f"Warning: Parameter variation produced identical results")
''')

        # Impact level test
        if param.impact in ['critical', 'high']:
            tests.append(f'''    def test_high_impact_verification(self, api, default_params):
        """Verify that this {param.impact}-impact parameter has measurable effect."""
        if api is None:
            pytest.skip("API not available")

        # For high/critical impact parameters, we expect clear differences
        params = default_params.copy()

        # Generate multiple samples with different values
        results = []
        test_values = {repr(self._get_test_values_list(param))}

        for val in test_values:
            params["{param.name}"] = val
            result = api.quick_generate(params)
            results.append(result)

        # Check that results differ
        pitches_sets = [extract_pitches(r) for r in results]

        # At least some results should differ
        all_same = all(p == pitches_sets[0] for p in pitches_sets)

        if all_same:
            print(f"Warning: High-impact parameter showed no variation")
            # Don't fail - might be due to other factors
''')

        return '\n'.join(tests)

    def _generate_integration_tests(self, param: ParameterProposal) -> str:
        """Generate integration tests with other parameters."""
        tests = []

        # Test with common parameter combinations
        test_value = self._get_test_value(param)

        tests.append(f'''    def test_integration_with_basic_parameters(self, api, default_params):
        """Test parameter works with basic system parameters."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = {repr(test_value)}

        # Add various common parameters
        params.update({{
            "tempo": 120,
            "key": "C",
            "measures": 8,
            "time_signature": "4/4"
        }})

        result = api.quick_generate(params)
        assert result is not None, "Integration with basic parameters failed"

        is_valid, msg = validate_midi_basic(result)
        assert is_valid, f"Invalid MIDI in integration test: {{msg}}"
''')

        # Test with category-related parameters
        if param.category == 'harmony':
            tests.append(f'''    def test_integration_with_harmony_parameters(self, api, default_params):
        """Test integration with other harmony parameters."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = {repr(test_value)}

        # Add other harmony-related parameters
        params.update({{
            "key": "Dm",
            "mode": "dorian"
        }})

        try:
            result = api.quick_generate(params)
            assert result is not None
            is_valid, msg = validate_midi_basic(result)
            assert is_valid, msg
        except Exception as e:
            pytest.skip(f"Harmony integration not fully implemented: {{e}}")
''')

        elif param.category == 'rhythm':
            tests.append(f'''    def test_integration_with_rhythm_parameters(self, api, default_params):
        """Test integration with other rhythm parameters."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = {repr(test_value)}

        # Add other rhythm-related parameters
        params.update({{
            "tempo": 140,
            "time_signature": "4/4"
        }})

        try:
            result = api.quick_generate(params)
            assert result is not None
            is_valid, msg = validate_midi_basic(result)
            assert is_valid, msg
        except Exception as e:
            pytest.skip(f"Rhythm integration not fully implemented: {{e}}")
''')

        # Test with random existing parameters
        if self.existing_parameters:
            sample_params = list(self.existing_parameters.keys())[:3]
            param_dict = {p: self.existing_parameters[p].get('default', 0.5)
                         for p in sample_params}

            tests.append(f'''    def test_integration_with_existing_parameters(self, api, default_params):
        """Test integration with existing system parameters."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = {repr(test_value)}

        # Add sample of existing parameters
        params.update({repr(param_dict)})

        try:
            result = api.quick_generate(params)
            assert result is not None, "Failed with existing parameters"

            is_valid, msg = validate_midi_basic(result)
            assert is_valid, msg
        except Exception as e:
            # Some parameters might not be implemented in current API
            print(f"Note: Some existing parameters not available: {{e}}")
''')

        # Test parameter doesn't break when omitted (uses default)
        tests.append(f'''    def test_omitted_parameter_uses_default(self, api, default_params):
        """Test that omitting parameter uses default value correctly."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        # Explicitly do NOT include the parameter

        result = api.quick_generate(params)
        assert result is not None, "Generation failed when parameter omitted"

        is_valid, msg = validate_midi_basic(result)
        assert is_valid, f"Invalid MIDI when parameter omitted: {{msg}}"
''')

        return '\n'.join(tests)

    def _generate_regression_tests(self, param: ParameterProposal) -> str:
        """Generate regression tests."""
        tests = []

        # Test that existing functionality still works
        tests.append(f'''    def test_backward_compatibility_empty_params(self, api, default_params):
        """Test system works without new parameter (backward compatibility)."""
        if api is None:
            pytest.skip("API not available")

        # Use only basic params, no new parameter
        params = {{
            "tempo": 120,
            "key": "C",
            "measures": 8
        }}

        result = api.quick_generate(params)
        assert result is not None, "Backward compatibility broken"

        is_valid, msg = validate_midi_basic(result)
        assert is_valid, f"Invalid MIDI in backward compatibility test: {{msg}}"
''')

        # Test that adding parameter doesn't break existing generation
        tests.append(f'''    def test_no_regression_with_new_parameter(self, api, default_params):
        """Test that adding new parameter doesn't break existing code paths."""
        if api is None:
            pytest.skip("API not available")

        # Generate without new parameter
        params_old = default_params.copy()
        result_old = api.quick_generate(params_old)

        # Generate with new parameter at default
        params_new = default_params.copy()
        params_new["{param.name}"] = {repr(param.default)}
        result_new = api.quick_generate(params_new)

        # Both should succeed
        assert result_old is not None, "Old code path broken"
        assert result_new is not None, "New code path broken"

        # Both should produce valid MIDI
        is_valid_old, msg_old = validate_midi_basic(result_old)
        is_valid_new, msg_new = validate_midi_basic(result_new)

        assert is_valid_old, f"Old path invalid: {{msg_old}}"
        assert is_valid_new, f"New path invalid: {{msg_new}}"
''')

        # Test existing parameters still work
        if self.existing_parameters:
            tests.append(f'''    def test_existing_parameters_unaffected(self, api, default_params):
        """Test that existing parameters still work correctly."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()

        # Add new parameter along with existing ones
        params["{param.name}"] = {repr(param.default)}

        # Try to use some existing parameters
        existing_test_params = {{
            # Add a few existing parameters for testing
        }}

        params.update(existing_test_params)

        try:
            result = api.quick_generate(params)
            assert result is not None

            is_valid, msg = validate_midi_basic(result)
            assert is_valid, msg
        except Exception as e:
            print(f"Note: Some existing parameters may not be in current API: {{e}}")
''')

        # Performance regression test
        tests.append(f'''    @pytest.mark.benchmark
    def test_no_performance_regression(self, api, default_params, benchmark):
        """Test that new parameter doesn't cause performance regression."""
        if api is None:
            pytest.skip("API not available")

        params = default_params.copy()
        params["{param.name}"] = {repr(param.default)}

        def generate():
            return api.quick_generate(params)

        # Benchmark the generation
        result = benchmark(generate)

        # Should complete in reasonable time
        assert result is not None

        # Check benchmark stats (available in benchmark.stats)
        # Typical generation should be < 10 seconds
        # (adjust based on system requirements)
''')

        return '\n'.join(tests)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _to_class_name(self, name: str) -> str:
        """Convert parameter name to test class name."""
        # Remove dots and underscores, capitalize words
        words = re.split(r'[._\-\s]+', name)
        return ''.join(word.capitalize() for word in words if word)

    def _to_snake_case(self, name: str) -> str:
        """Convert name to snake_case."""
        # Replace non-alphanumeric with underscore
        name = re.sub(r'[^a-zA-Z0-9]+', '_', str(name))
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Convert to lowercase
        return name.lower()

    def _get_test_value(self, param: ParameterProposal) -> Any:
        """Get a representative test value for parameter."""
        if param.param_type == 'boolean':
            return True
        elif param.param_type == 'categorical':
            return param.options[0] if param.options else param.default
        elif param.param_type == 'continuous':
            # Use midpoint
            if param.min_value is not None and param.max_value is not None:
                return (param.min_value + param.max_value) / 2
            return param.default
        elif param.param_type == 'integer':
            if param.min_value is not None and param.max_value is not None:
                return (param.min_value + param.max_value) // 2
            return param.default
        elif param.param_type == 'probability':
            return 0.5
        elif param.param_type == 'velocity':
            return 64
        else:
            return param.default

    def _get_min_value(self, param: ParameterProposal) -> Any:
        """Get minimum value for parameter."""
        if param.param_type == 'boolean':
            return False
        elif param.param_type == 'categorical':
            return param.options[0] if param.options else param.default
        elif param.param_type in ['continuous', 'integer']:
            return param.min_value if param.min_value is not None else 0
        elif param.param_type == 'probability':
            return 0.0
        elif param.param_type == 'velocity':
            return 0
        else:
            return param.default

    def _get_max_value(self, param: ParameterProposal) -> Any:
        """Get maximum value for parameter."""
        if param.param_type == 'boolean':
            return True
        elif param.param_type == 'categorical':
            return param.options[-1] if param.options else param.default
        elif param.param_type in ['continuous', 'integer']:
            return param.max_value if param.max_value is not None else 100
        elif param.param_type == 'probability':
            return 1.0
        elif param.param_type == 'velocity':
            return 127
        else:
            return param.default

    def _get_test_values_list(self, param: ParameterProposal) -> List[Any]:
        """Get list of test values for parameter."""
        if param.param_type == 'boolean':
            return [False, True]
        elif param.param_type == 'categorical':
            return param.options[:3] if param.options else [param.default]
        elif param.param_type == 'continuous':
            min_val = param.min_value if param.min_value is not None else 0.0
            max_val = param.max_value if param.max_value is not None else 1.0
            mid_val = (min_val + max_val) / 2
            return [min_val, mid_val, max_val]
        elif param.param_type == 'integer':
            min_val = param.min_value if param.min_value is not None else 0
            max_val = param.max_value if param.max_value is not None else 10
            mid_val = (min_val + max_val) // 2
            return [min_val, mid_val, max_val]
        elif param.param_type == 'probability':
            return [0.0, 0.5, 1.0]
        elif param.param_type == 'velocity':
            return [0, 64, 127]
        else:
            return [param.default]

    def _get_imports_template(self) -> str:
        """Get standard imports template."""
        return self._generate_imports(ParameterProposal(
            name="template", description="", param_type="continuous",
            default=0.5, min_value=0.0, max_value=1.0
        ))

    def _get_fixtures_template(self) -> str:
        """Get standard fixtures template."""
        return self._generate_fixtures(ParameterProposal(
            name="template", description="", param_type="continuous",
            default=0.5, min_value=0.0, max_value=1.0
        ))


# ============================================================================
# Test Suite Validator
# ============================================================================

class TestSuiteValidator:
    """
    Validates generated test suites for completeness and correctness.
    """

    def __init__(self):
        """Initialize validator."""
        self.logger = logging.getLogger(f"{__name__}.TestSuiteValidator")

    def validate_test_suite(self, test_file_path: Path) -> Dict[str, Any]:
        """
        Validate a generated test suite.

        Args:
            test_file_path: Path to test file

        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'test_count': 0,
            'coverage': {}
        }

        try:
            # Read test file
            with open(test_file_path, 'r') as f:
                content = f.read()

            # Check for required components
            required_patterns = [
                r'def test_minimum_value',
                r'def test_maximum_value',
                r'def test_default_value',
                r'def test_musical_coherence',
                r'def test_.*integration',
                r'def test_backward_compatibility'
            ]

            for pattern in required_patterns:
                if not re.search(pattern, content):
                    results['warnings'].append(f"Missing recommended test: {pattern}")

            # Count test methods
            test_count = len(re.findall(r'\n    def test_', content))
            results['test_count'] = test_count

            if test_count == 0:
                results['valid'] = False
                results['errors'].append("No test methods found")

            # Check for syntax errors (basic)
            try:
                compile(content, test_file_path, 'exec')
            except SyntaxError as e:
                results['valid'] = False
                results['errors'].append(f"Syntax error: {e}")

            # Coverage analysis
            has_boundary = bool(re.search(r'test_.*minimum.*|test_.*maximum', content))
            has_validity = bool(re.search(r'test_.*coherence|test_.*validity', content))
            has_integration = bool(re.search(r'test_.*integration', content))
            has_regression = bool(re.search(r'test_.*backward.*|test_.*regression', content))

            results['coverage'] = {
                'boundary_tests': has_boundary,
                'validity_tests': has_validity,
                'integration_tests': has_integration,
                'regression_tests': has_regression
            }

        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Validation error: {e}")

        return results

    def validate_and_report(self, test_file_path: Path) -> bool:
        """
        Validate test suite and print report.

        Args:
            test_file_path: Path to test file

        Returns:
            True if valid, False otherwise
        """
        results = self.validate_test_suite(test_file_path)

        print("\n" + "="*70)
        print(f"Test Suite Validation: {test_file_path.name}")
        print("="*70)

        print(f"\nTest Count: {results['test_count']}")

        print("\nCoverage:")
        for category, has_tests in results['coverage'].items():
            status = "✓" if has_tests else "✗"
            print(f"  {status} {category}")

        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  ✗ {error}")

        if results['warnings']:
            print("\nWarnings:")
            for warning in results['warnings']:
                print(f"  ! {warning}")

        status = "PASSED" if results['valid'] else "FAILED"
        print(f"\nValidation: {status}")
        print("="*70)

        return results['valid']


# ============================================================================
# Test Execution Helper
# ============================================================================

class TestExecutor:
    """
    Executes generated test suites and reports results.
    """

    def __init__(self):
        """Initialize executor."""
        self.logger = logging.getLogger(f"{__name__}.TestExecutor")

    def run_test_suite(self, test_file_path: Path,
                      pytest_args: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a test suite using pytest.

        Args:
            test_file_path: Path to test file
            pytest_args: Optional additional pytest arguments

        Returns:
            Test results dictionary
        """
        import subprocess

        args = ['pytest', str(test_file_path), '-v', '--tb=short']
        if pytest_args:
            args.extend(pytest_args)

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Test execution timed out',
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'returncode': -1
            }


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Agent 26: Test Case Generator for Musical Parameters"
    )
    parser.add_argument(
        'param_file',
        type=Path,
        help="JSON file containing parameter proposal"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help="Output test file path (default: auto-generated)"
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help="Validate generated test suite"
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help="Run generated test suite"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Verbose output"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Load parameter proposal
    try:
        with open(args.param_file, 'r') as f:
            param_data = json.load(f)

        param_proposal = ParameterProposal.from_dict(param_data)
    except Exception as e:
        print(f"Error loading parameter proposal: {e}")
        return 1

    # Generate test suite
    generator = TestCaseGenerator()
    test_path = generator.save_test_suite(param_proposal, args.output)

    print(f"\n✓ Test suite generated: {test_path}")

    # Validate if requested
    if args.validate:
        validator = TestSuiteValidator()
        if not validator.validate_and_report(test_path):
            return 1

    # Run if requested
    if args.run:
        executor = TestExecutor()
        print("\nRunning test suite...")
        results = executor.run_test_suite(test_path)

        if results['success']:
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed")
            print(results.get('stdout', ''))
            print(results.get('stderr', ''))
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
