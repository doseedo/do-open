"""
Training Module - Agent 15
===========================

Model training and evaluation for parameter prediction.

This module provides:
- ModelTrainingSpecialist: Main training class
- TrainingConfig: Configuration management
- TrainingMetrics: Comprehensive metrics tracking
- BatchTrainingResults: Batch training management
- Utility functions for common training tasks

Quick Start:
    from training.model_trainer import ModelTrainingSpecialist, TrainingConfig

    config = TrainingConfig(
        n_estimators=200,
        max_depth=8,
        enable_tuning=True
    )

    trainer = ModelTrainingSpecialist(config)
    model, metrics = trainer.train_parameter_model(
        param_name='harmony.voicing.spread',
        param_def=param_definition,
        training_data=data
    )

Author: Agent 15 - Model Training Specialist
"""

from .model_trainer import (
    ModelTrainingSpecialist,
    TrainingConfig,
    TrainingMetrics,
    BatchTrainingResults,
    train_single_parameter,
    train_all_parameters
)

__all__ = [
    'ModelTrainingSpecialist',
    'TrainingConfig',
    'TrainingMetrics',
    'BatchTrainingResults',
    'train_single_parameter',
    'train_all_parameters'
]

__version__ = '1.0.0'
__author__ = 'Agent 15 - Model Training Specialist'
