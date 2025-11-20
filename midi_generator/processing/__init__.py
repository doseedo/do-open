"""
Processing Module
=================

Batch processing infrastructure for efficient parallel execution.

This module provides:
- BatchProcessingManager: Main batch processing coordinator
- Parallel feature extraction
- Batch parameter prediction
- Parallel training data generation
- Batch model training
- Progress tracking and monitoring
- Error handling and retry logic

Author: Agent 32 - Batch Processing Manager
License: MIT
"""

from .batch_manager import (
    # Main classes
    BatchProcessingManager,

    # Data structures
    BatchStatus,
    ProcessingMode,
    BatchProgress,
    BatchResult,

    # Utility functions
    create_batch_manager,
)

__all__ = [
    # Main classes
    'BatchProcessingManager',

    # Data structures
    'BatchStatus',
    'ProcessingMode',
    'BatchProgress',
    'BatchResult',

    # Utility functions
    'create_batch_manager',
]
