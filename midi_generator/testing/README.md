# Agent 26: Test Case Generator

Comprehensive automated test generation system for musical parameters in the self-expanding inverse music generation framework.

## Overview

The Test Case Generator automatically creates complete pytest test suites for new parameters, ensuring:
- **Boundary Testing**: Min, max, and default values work correctly
- **Musical Validity**: Generated MIDI is musically coherent
- **Feature Impact**: Parameters affect expected musical features
- **Integration**: Works seamlessly with existing parameters
- **Regression Prevention**: New parameters don't break existing functionality

## Architecture

### Core Components

1. **TestCaseGenerator**: Main test generation engine
2. **ParameterProposal**: Data structure for parameter specifications
3. **TestSuiteValidator**: Validates generated test suites
4. **TestExecutor**: Runs and reports test results
5. **TestConfiguration**: Configuration for test generation

### Generated Test Categories

#### 1. Boundary Tests
- Minimum value validation
- Maximum value validation
- Default value validation
- Categorical option testing (for categorical parameters)
- Boolean true/false testing (for boolean parameters)
- Out-of-bounds handling

#### 2. Musical Validity Tests
- Basic musical coherence (pitch ranges, duration validity)
- Melodic coherence (reasonable intervals, melodic motion)
- Rhythmic coherence (valid note durations, timing)
- Harmonic validity (chord voicings, pitch ranges)
- Genre-specific validity (for genre-tagged parameters)

#### 3. Feature Impact Tests
- Parameter variation produces measurable differences
- Specific affected features show expected changes
- High/critical impact parameters have clear effects
- Feature extraction validation

#### 4. Integration Tests
- Works with basic system parameters (tempo, key, measures)
- Category-specific integration (harmony/rhythm/melody)
- Integration with existing parameters
- Omitted parameter uses default correctly

#### 5. Regression Tests
- Backward compatibility (works without new parameter)
- No regression with new parameter added
- Existing parameters still function
- Performance regression detection

## Usage

### Command Line

```bash
# Generate test suite from parameter proposal
python -m midi_generator.testing.test_case_generator example_parameter.json

# Generate and validate
python -m midi_generator.testing.test_case_generator example_parameter.json --validate

# Generate, validate, and run
python -m midi_generator.testing.test_case_generator example_parameter.json --validate --run

# Specify custom output path
python -m midi_generator.testing.test_case_generator example_parameter.json -o tests/test_custom.py
```

### Python API

```python
from midi_generator.testing import TestCaseGenerator, ParameterProposal

# Create parameter proposal
param = ParameterProposal(
    name="harmony.tritone_sub_frequency",
    description="Frequency of tritone substitutions",
    param_type="probability",
    default=0.3,
    min_value=0.0,
    max_value=1.0,
    category="harmony",
    impact="high",
    genres=["jazz", "bebop"],
    affected_features=["harmonic_complexity", "voice_leading_smoothness"]
)

# Generate test suite
generator = TestCaseGenerator()
test_code = generator.generate_test_suite(param)

# Save to file
test_path = generator.save_test_suite(param)
print(f"Tests saved to: {test_path}")

# Validate
from midi_generator.testing import TestSuiteValidator
validator = TestSuiteValidator()
is_valid = validator.validate_and_report(test_path)

# Run tests
from midi_generator.testing import TestExecutor
executor = TestExecutor()
results = executor.run_test_suite(test_path)
```

## Parameter Proposal Format

Parameter proposals are JSON files with the following structure:

```json
{
  "name": "parameter.full.name",
  "description": "Clear description of what parameter does",
  "type": "continuous|integer|categorical|boolean|probability|velocity",
  "default": <default_value>,
  "min": <min_value>,
  "max": <max_value>,
  "options": ["option1", "option2"],  // For categorical only
  "category": "harmony|melody|rhythm|dynamics|etc",
  "impact": "critical|high|medium|low",
  "genres": ["jazz", "classical", "rock"],
  "affected_features": [
    "feature_1",
    "feature_2"
  ],
  "musical_properties": [
    "Description of musical effect 1",
    "Description of musical effect 2"
  ]
}
```

### Parameter Types

- **continuous**: Float values in range [min, max]
- **integer**: Integer values in range [min, max]
- **categorical**: One of specified options
- **boolean**: True or False
- **probability**: Float in range [0.0, 1.0]
- **velocity**: MIDI velocity in range [0, 127]

### Impact Levels

- **critical**: Fundamental parameter affecting core functionality
- **high**: Significant impact on musical output
- **medium**: Moderate impact, refinement parameter
- **low**: Subtle effect, fine-tuning parameter

## Test Suite Structure

Generated test files follow this structure:

```
test_<parameter_name>.py
├── Imports (pytest, numpy, system modules)
├── Fixtures (api, generator, analyzer, default_params)
├── Test Class
│   ├── Boundary Tests (10-15 tests)
│   ├── Musical Validity Tests (5-10 tests)
│   ├── Feature Impact Tests (5-15 tests)
│   ├── Integration Tests (5-8 tests)
│   └── Regression Tests (4-6 tests)
└── Utility Functions (MIDI validation helpers)
```

Each test file is:
- **Self-contained**: Can run independently
- **Well-documented**: Clear docstrings and comments
- **Defensive**: Handles missing components gracefully
- **Informative**: Clear assertion messages

## Integration with System

### Self-Expanding Cycle

The Test Case Generator is part of the self-expanding inverse music generation system:

```
1. Gap Detection → Parameter proposed
2. Agent 26 → Generates test suite
3. Code Generation → Implements parameter
4. Test Execution → Validates implementation
5. System Expansion → Parameter added to registry
```

### Requirements

Test suites require:
- `pytest` for test execution
- `numpy` for numerical operations
- System components:
  - `HarmonyModuleAPI` or `ContextAwareGenerator`
  - `MIDIAnalyzer` (optional, for feature tests)
  - Parameter registry

### Graceful Degradation

Tests are designed to:
- Skip when components unavailable
- Provide informative skip messages
- Allow partial system testing
- Support incremental development

## Examples

### Example 1: Continuous Parameter

```python
# Swing ratio parameter
param = ParameterProposal(
    name="rhythm.swing.ratio",
    description="Swing ratio (0.5=straight, 0.67=standard, 0.75=hard)",
    param_type="continuous",
    default=0.67,
    min_value=0.5,
    max_value=0.8,
    category="rhythm",
    impact="critical",
    genres=["jazz", "swing", "blues"]
)
```

Generates ~35 tests including:
- Boundary tests for 0.5, 0.67, 0.8
- Swing feel validation
- Integration with tempo/time signature
- Jazz genre validation

### Example 2: Categorical Parameter

```python
# Voicing type parameter
param = ParameterProposal(
    name="harmony.voicing.type",
    description="Type of chord voicing",
    param_type="categorical",
    default="close",
    options=["close", "spread", "drop2", "drop3", "rootless"],
    category="harmony",
    impact="high",
    genres=["jazz", "classical"]
)
```

Generates ~40 tests including:
- Test for each categorical option (5 tests)
- Voicing-specific musical validation
- Integration with other harmony parameters
- Genre-specific voicing tests

### Example 3: Boolean Parameter

```python
# Legato parameter
param = ParameterProposal(
    name="articulation.legato",
    description="Use legato articulation",
    param_type="boolean",
    default=False,
    category="articulation",
    impact="medium"
)
```

Generates ~30 tests including:
- True/False boundary tests
- Articulation validation
- Duration ratio checks
- Integration with dynamics

## Best Practices

### Writing Parameter Proposals

1. **Be Specific**: Clear, unambiguous descriptions
2. **Document Impact**: Explain musical effect
3. **List Features**: Specify affected features
4. **Genre Tags**: Add relevant genres
5. **Reasonable Ranges**: Use musically meaningful bounds

### Extending Test Generation

To add new test categories:

```python
class CustomTestGenerator(TestCaseGenerator):
    def _generate_custom_tests(self, param: ParameterProposal) -> str:
        """Generate custom test category."""
        tests = []
        # Add custom test generation logic
        return '\n'.join(tests)

    def generate_test_suite(self, param: ParameterProposal) -> str:
        # Call parent method
        base_suite = super().generate_test_suite(param)
        # Add custom tests
        custom_tests = self._generate_custom_tests(param)
        # Combine and return
        return base_suite + custom_tests
```

### Test Maintenance

- **Review Generated Tests**: Always review before committing
- **Add Manual Tests**: Supplement automated tests with manual ones
- **Update on Failure**: Improve generator when tests fail incorrectly
- **Document Exceptions**: Note parameters with special requirements

## Validation & Quality Assurance

### Automatic Validation

The `TestSuiteValidator` checks:
- ✓ Syntax correctness
- ✓ Test count > 0
- ✓ Required test categories present
- ✓ Code compiles without errors

### Manual Review Checklist

- [ ] Parameter description is clear
- [ ] Test values are musically meaningful
- [ ] Edge cases are covered
- [ ] Integration scenarios make sense
- [ ] Assertions are specific and informative
- [ ] Tests can run independently

## Performance Considerations

### Test Execution Time

- Typical test suite: 30-50 tests
- Execution time: 1-5 minutes (depending on system)
- Use `@pytest.mark.benchmark` for performance tests
- Set appropriate timeouts

### Optimization Tips

1. **Parallel Execution**: Use `pytest -n auto` with pytest-xdist
2. **Selective Running**: Use marks to run subsets
3. **Fixture Caching**: Cache expensive fixtures
4. **Mock Heavy Operations**: Mock slow components when appropriate

## Troubleshooting

### Common Issues

**Issue**: Tests skip with "Required components not available"
- **Solution**: Ensure system components are installed and importable

**Issue**: Musical validity tests fail
- **Solution**: Check parameter ranges are reasonable, review generation logic

**Issue**: Feature impact tests inconclusive
- **Solution**: Verify affected_features list, check feature extraction

**Issue**: Integration tests fail
- **Solution**: Check parameter compatibility, review API usage

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Generate with debug info
generator = TestCaseGenerator()
generator.logger.setLevel(logging.DEBUG)
```

## Statistics

### Code Metrics

- **Total Lines**: 2000+
- **Test Categories**: 5
- **Test Count per Suite**: 30-50 (depending on parameter type)
- **Coverage Types**: Boundary, Validity, Impact, Integration, Regression

### Example Output

For a typical parameter:
- Continuous: ~35 tests
- Categorical (5 options): ~42 tests
- Boolean: ~30 tests
- Probability: ~38 tests

## Contributing

To improve the Test Case Generator:

1. Add new test categories in `_generate_*_tests` methods
2. Enhance musical validation logic
3. Improve feature impact detection
4. Add more integration scenarios
5. Update documentation

## References

- Main System: `/midi_generator/`
- Parameter Registry: `/midi_generator/parameters/registry.json`
- Existing Tests: `/midi_generator/tests/`
- API Documentation: `/midi_generator/api/unified_api.py`

## License

Part of the Do/midi_generator project.

## Contact

For issues or questions about test generation, refer to the main project documentation.
