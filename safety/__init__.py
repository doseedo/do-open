"""
Safety monitoring and rollback system for the self-expanding music generation framework.

This package provides comprehensive safety monitoring during parameter expansions,
including checkpointing, testing, quality monitoring, and instant rollback capabilities.
"""

from .safety_monitor import (
    SafetyMonitor,
    SafetyConfig,
    SafetyLevel,
    CheckpointStatus,
    TestResult,
    Checkpoint,
    MonitoringResult,
    GitOperations,
    RegistrySnapshot,
    ModelSnapshot,
    QualityMonitor,
    ComprehensiveTestSuite,
    ParameterTester
)

__all__ = [
    'SafetyMonitor',
    'SafetyConfig',
    'SafetyLevel',
    'CheckpointStatus',
    'TestResult',
    'Checkpoint',
    'MonitoringResult',
    'GitOperations',
    'RegistrySnapshot',
    'ModelSnapshot',
    'QualityMonitor',
    'ComprehensiveTestSuite',
    'ParameterTester'
]

__version__ = '1.0.0'
__author__ = 'Agent 17'
