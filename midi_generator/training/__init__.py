"""
Training Module - Agents 14 & 15
=================================

This module contains comprehensive training infrastructure for the Musical Program Synthesis system.

Agents:
- Agent 14: Synthetic Training Data Generator
- Agent 15: Model Training Specialist

Features:
---------
1. **Synthetic Data Generation** (Agent 14):
   - Latin hypercube sampling for even parameter space coverage
   - Genre-balanced dataset generation
   - Musical coherence validation
   - Data augmentation and quality analysis

2. **Model Training** (Agent 15):
   - XGBoost model training per parameter
   - Hyperparameter tuning
   - Cross-validation and evaluation
   - Batch training for multiple parameters

Example Usage:
--------------
```python
from midi_generator.training import SyntheticTrainingDataGenerator, ModelTrainingSpecialist
from midi_generator.parameters.universal_registry import UniversalParameterRegistry

# 1. Generate training data
registry = UniversalParameterRegistry()
generator = SyntheticTrainingDataGenerator()
training_data = generator.generate_training_data(
    param_name="harmony.jazz.voicing_density",
    param_def=registry.parameters["harmony.jazz.voicing_density"],
    n_examples=1000
)

# 2. Train model
trainer = ModelTrainingSpecialist()
model, metrics = trainer.train_parameter_model(
    param_name='harmony.jazz.voicing_density',
    param_def=registry.parameters["harmony.jazz.voicing_density"],
    training_data=training_data
)
```

Authors: Agent 14 (Data Generation) & Agent 15 (Model Training)
License: MIT
"""

# Agent 14: Synthetic Data Generation
from .synthetic_data_generator import (
    SyntheticTrainingDataGenerator,
    MusicalCoherenceValidator,
    ParameterSpaceSampler,
    DefaultParameterGenerator,
    BatchTrainingDataGenerator,
    TrainingExample,
    DatasetStatistics,
    MIDIDataAugmenter,
    CrossValidationSplitter,
    DatasetExporter,
    DatasetQualityAnalyzer,
    ActiveLearningSelector,
    TrainingDataset,
)

# Agent 15: Model Training
from .model_trainer import (
    ModelTrainingSpecialist,
    TrainingConfig,
    TrainingMetrics,
    BatchTrainingResults,
    train_single_parameter,
    train_all_parameters,
    ModelTrainingResult
)

__all__ = [
    # Agent 14: Data generation classes
    'SyntheticTrainingDataGenerator',
    'MusicalCoherenceValidator',
    'ParameterSpaceSampler',
    'DefaultParameterGenerator',
    'BatchTrainingDataGenerator',
    'TrainingExample',
    'DatasetStatistics',
    'MIDIDataAugmenter',
    'CrossValidationSplitter',
    'DatasetExporter',
    'DatasetQualityAnalyzer',
    'ActiveLearningSelector',
    'TrainingDataset',

    # Agent 15: Model training classes
    'ModelTrainingSpecialist',
    'TrainingConfig',
    'TrainingMetrics',
    'BatchTrainingResults',
    'train_single_parameter',
    'train_all_parameters',
    'ModelTrainingResult'
]

__version__ = '1.0.0'
__authors__ = 'Agent 14 & Agent 15'
