"""
Training Data Generation Module - Agent 14
===========================================

Synthetic training data generation for the Musical Program Synthesis system.

This module provides comprehensive tools for generating high-quality training
data for new parameters through:

1. **SyntheticTrainingDataGenerator**: Main class for generating training datasets
2. **MusicalCoherenceValidator**: Validates generated MIDI for musical quality
3. **ParameterSpaceSampler**: Intelligent sampling using Latin hypercube
4. **BatchTrainingDataGenerator**: Batch generation for multiple parameters

Key Features:
-------------
- Latin hypercube sampling for even parameter space coverage
- Genre-balanced dataset generation
- Musical coherence validation (pitch, rhythm, harmony, dynamics)
- Diverse parameter variation to prevent overfitting
- Comprehensive metadata and statistics tracking
- Robust error handling and retry logic
- Real-time progress monitoring
- Support for all parameter types

Example Usage:
--------------
```python
from midi_generator.training import SyntheticTrainingDataGenerator
from midi_generator.parameters.universal_registry import UniversalParameterRegistry

# Initialize
registry = UniversalParameterRegistry()
generator = SyntheticTrainingDataGenerator()

# Get parameter definition
param_name = "harmony.jazz.voicing_density"
param_def = registry.parameters[param_name]

# Generate 1000 training examples
training_data = generator.generate_training_data(
    param_name=param_name,
    param_def=param_def,
    n_examples=1000
)

# Or generate genre-balanced dataset
balanced_data = generator.generate_balanced_dataset(
    param_name=param_name,
    param_def=param_def,
    n_per_genre=100
)
```

Author: Agent 14 - Synthetic Training Data Generator
License: MIT
"""

from .synthetic_data_generator import (
    # Main generator class
    SyntheticTrainingDataGenerator,

    # Validator
    MusicalCoherenceValidator,

    # Sampling utilities
    ParameterSpaceSampler,
    DefaultParameterGenerator,

    # Batch generation
    BatchTrainingDataGenerator,

    # Data structures
    TrainingExample,
    DatasetStatistics,

    # Data augmentation
    MIDIDataAugmenter,

    # Cross-validation
    CrossValidationSplitter,

    # Export utilities
    DatasetExporter,

    # Quality analysis
    DatasetQualityAnalyzer,

    # Active learning
    ActiveLearningSelector,
)

__all__ = [
    # Main classes
    'SyntheticTrainingDataGenerator',
    'MusicalCoherenceValidator',
    'ParameterSpaceSampler',
    'DefaultParameterGenerator',
    'BatchTrainingDataGenerator',

    # Data structures
    'TrainingExample',
    'DatasetStatistics',

    # Advanced features
    'MIDIDataAugmenter',
    'CrossValidationSplitter',
    'DatasetExporter',
    'DatasetQualityAnalyzer',
    'ActiveLearningSelector',
]

__version__ = '1.0.0'
__author__ = 'Agent 14'
