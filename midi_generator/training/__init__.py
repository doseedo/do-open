"""
Training Module
===============

This module contains agents for training data generation and model training.

Modules:
- synthetic_data_generator: Agent 14 - Synthetic Training Data Generator
- model_trainer: Agent 15 - Model Training Specialist

Author: Musical Program Synthesis Team
License: MIT
"""

from .synthetic_data_generator import (
    SyntheticTrainingDataGenerator,
    TrainingExample,
    TrainingDataset,
    MusicalCoherenceValidator
)

from .model_trainer import (
    ModelTrainingSpecialist,
    TrainingMetrics,
    ModelTrainingResult
)

__all__ = [
    # Data generation
    'SyntheticTrainingDataGenerator',
    'TrainingExample',
    'TrainingDataset',
    'MusicalCoherenceValidator',

    # Model training
    'ModelTrainingSpecialist',
    'TrainingMetrics',
    'ModelTrainingResult'
]
