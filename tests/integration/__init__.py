"""
Integration Tests for Musical Program Synthesis System

This package contains integration tests that verify the interaction
between different agents and components of the system.

Test suites:
- Feature extraction pipeline (Agent 8)
- Training data generation (Agent 14)
- Model training (Agent 15)
- Expansion orchestration (Agent 16)
- End-to-end workflows

Usage:
    # Run all integration tests
    python -m midi_generator.testing.integration_test_coordinator

    # Run with pytest
    pytest tests/integration/

Author: Agent 34 - Integration Testing Coordinator
"""

from pathlib import Path

# Test configuration
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
RESULTS_DIR = Path(__file__).parent / "results"

# Ensure directories exist
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
