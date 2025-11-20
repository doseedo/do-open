# Integration Tests

Comprehensive integration testing framework for the Musical Program Synthesis system.

## Overview

This test suite provides comprehensive integration testing across all 35 agents in the system, with special focus on critical pipelines and workflows.

## Quick Start

```bash
# Run all integration tests
python -m midi_generator.testing.integration_test_coordinator

# Run with pytest
pytest tests/integration/

# Run specific test suite
pytest tests/integration/test_feature_extraction.py

# Skip slow tests (for CI)
python -m midi_generator.testing.integration_test_coordinator --skip-slow

# Run only critical tests
pytest tests/integration/ -m critical
```

## Test Organization

### Test Suites

1. **Feature Extraction** (`test_feature_extraction.py`)
   - Agent 8: Deep Feature Extractor
   - 1000+ feature validation
   - Performance benchmarks

2. **Training Pipeline** (`test_training_pipeline.py`)
   - Agent 14: Synthetic Data Generation
   - Agent 15: Model Training
   - End-to-end training workflow

3. **End-to-End** (`test_end_to_end.py`)
   - Complete MIDI → Features → Parameters pipeline
   - Expansion cycle workflows
   - Multi-agent integration

### Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.critical` - Must pass for production
- `@pytest.mark.high` - High priority
- `@pytest.mark.medium` - Medium priority
- `@pytest.mark.low` - Low priority
- `@pytest.mark.integration` - Multi-agent integration
- `@pytest.mark.e2e` - End-to-end workflows
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.regression` - Regression tests

### Running Specific Test Types

```bash
# Critical tests only
pytest tests/integration/ -m critical

# Skip slow tests
pytest tests/integration/ -m "not slow"

# Performance tests
pytest tests/integration/ -m performance

# Regression tests
pytest tests/integration/ -m regression
```

## Test Coverage

### Agents Tested

- ✅ Agent 8: Deep Feature Extractor (1000+ features)
- ✅ Agent 10: Intelligent Gap Detector
- ✅ Agent 11: LLM Parameter Proposer
- ✅ Agent 12: LLM Code Generator
- ✅ Agent 14: Synthetic Training Data Generator
- ✅ Agent 15: Model Training Specialist
- ✅ Agent 16: Expansion Orchestrator
- ✅ Agents 1-7: Core parameter systems (indirect testing)
- ✅ End-to-end workflows

### Coverage Goals

- **Target**: 80%+ code coverage
- **Critical paths**: 100% coverage
- **Integration points**: Full validation
- **Regression**: All known issues tested

## Test Results

Test results are saved to `tests/integration/results/`:

```
tests/integration/results/
├── integration_test_report_20250120_143022.json
├── integration_test_report_20250120_150145.json
└── ...
```

Each report includes:
- Test suite results
- Individual test results
- Performance metrics
- Error details
- System information

## Integration Test Coordinator

The main coordinator (`integration_test_coordinator.py`) provides:

1. **Automated Test Execution**
   - Run all test suites
   - Parallel execution where possible
   - Progress monitoring

2. **Comprehensive Reporting**
   - JSON report generation
   - Console summary
   - Detailed error tracking

3. **Test Management**
   - Priority-based execution
   - Skip conditions for missing agents
   - Regression testing

### Usage

```python
from midi_generator.testing.integration_test_coordinator import IntegrationTestCoordinator

# Initialize coordinator
coordinator = IntegrationTestCoordinator(
    test_data_dir=Path("tests/test_data"),
    output_dir=Path("tests/integration/results"),
    verbose=True
)

# Run all tests
report = coordinator.run_all_tests(skip_slow=False)

# Check results
if report.is_success():
    print("✓ All critical tests passed!")
else:
    print("✗ Some critical tests failed")
```

## Continuous Integration

### CI Configuration

For CI/CD pipelines, use:

```bash
# Fast CI run (skip slow tests)
python -m midi_generator.testing.integration_test_coordinator --skip-slow

# Full CI run
python -m midi_generator.testing.integration_test_coordinator

# With pytest
pytest tests/integration/ -m "not slow" --maxfail=5
```

### Environment Variables

```bash
# Set test data directory
export TEST_DATA_DIR=/path/to/test/data

# Set results directory
export RESULTS_DIR=/path/to/results
```

## Adding New Tests

### 1. Create Test File

```python
# tests/integration/test_my_feature.py

import pytest
from midi_generator.my_module import MyAgent

@pytest.fixture
def my_agent():
    """Create agent instance"""
    return MyAgent()

class TestMyFeature:
    """Test suite for my feature"""

    @pytest.mark.critical
    def test_critical_functionality(self, my_agent):
        """Test critical functionality"""
        result = my_agent.do_something()
        assert result is not None

    @pytest.mark.slow
    def test_expensive_operation(self, my_agent):
        """Test expensive operation"""
        result = my_agent.expensive_operation()
        assert result.success
```

### 2. Add to Coordinator (Optional)

If adding a new test suite to the coordinator:

```python
def test_my_new_feature(self) -> TestSuiteResult:
    """Test my new feature"""
    suite_name = "My New Feature"
    tests: List[TestResult] = []
    start_time = time.time()

    tests.append(self._run_test(
        "My Feature: Basic Test",
        self._test_my_feature_basic,
        priority=TestPriority.HIGH,
        agent_tested="my_agent"
    ))

    total_duration = time.time() - start_time
    return TestSuiteResult(suite_name, tests, total_duration)
```

### 3. Document

Update this README with:
- Test suite description
- Coverage information
- Known issues or limitations

## Troubleshooting

### Common Issues

**Issue**: Tests fail with "Agent X not available"
- **Solution**: Install required dependencies or implement missing agent

**Issue**: MIDI tests fail with "mido not available"
- **Solution**: Install mido: `pip install mido`

**Issue**: Slow test execution
- **Solution**: Use `--skip-slow` flag or run specific test suites

**Issue**: Coverage too low
- **Solution**: Add tests for uncovered code paths

### Debug Mode

Run tests with verbose output:

```bash
# Pytest verbose mode
pytest tests/integration/ -vv -s

# Coordinator verbose mode
python -m midi_generator.testing.integration_test_coordinator --verbose
```

## Performance Benchmarks

Performance targets for integration tests:

- Feature extraction: < 5s per file
- Model training: < 30s for small dataset
- End-to-end pipeline: < 60s
- Full test suite: < 5 minutes (without slow tests)

Monitor performance with:

```bash
pytest tests/integration/ -m performance --durations=10
```

## Contributing

When contributing new tests:

1. Follow existing test patterns
2. Use appropriate markers
3. Include docstrings
4. Test edge cases
5. Document expected behavior
6. Update this README

## References

- [Integration Test Coordinator](../midi_generator/testing/integration_test_coordinator.py)
- [Agent 34 Documentation](../midi_generator/AGENT_34_INTEGRATION_TESTING.md)
- [pytest Documentation](https://docs.pytest.org/)

## License

MIT License - Same as parent project
