#!/usr/bin/env python3
"""
Testing Module - Agent 26: Test Case Generator
==============================================

Automated test generation for musical parameters in the
self-expanding inverse music generation system.

Modules:
    test_case_generator: Main test generation engine
"""

from .test_case_generator import (
    TestCaseGenerator,
    ParameterProposal,
    TestConfiguration,
    TestSuiteValidator,
    TestExecutor
)

__all__ = [
    'TestCaseGenerator',
    'ParameterProposal',
    'TestConfiguration',
    'TestSuiteValidator',
    'TestExecutor'
]

__version__ = '1.0.0'
__author__ = 'Agent 26'
