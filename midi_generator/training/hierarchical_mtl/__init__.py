"""
Hierarchical Multi-Task Learning Training Infrastructure
==========================================================

Agent 06: Training Pipeline Engineer

This module provides comprehensive training infrastructure for the
Dø MIDI Generator v2.0 hierarchical multi-task learning system.

The hierarchical MTL architecture learns 50 parameters across 3 levels:
- Level 1: Global Context (8 parameters)
- Level 2: Universal Dimensions (20 parameters)
- Level 3: Genre-Specific Details (22 parameters)

Key Features:
-----------
1. **Hierarchical Dataset Handling**
   - Loads 750-file MIDI corpus with hierarchical labels
   - Supports train/val/test splits stratified by genre
   - Handles missing genre-specific parameters gracefully

2. **Advanced Training Loop**
   - Multi-task loss with hierarchical weighting
   - Early stopping with patience
   - Gradient clipping and mixed precision
   - Learning rate scheduling (cosine, step, plateau)

3. **Experiment Tracking**
   - Wandb and MLflow integration
   - Comprehensive metrics logging
   - Model checkpointing

4. **Distributed Training**
   - PyTorch DistributedDataParallel (DDP) support
   - Multi-GPU training

Author: Agent 06 - Training Pipeline Engineer
Date: November 20, 2025
Version: 2.0.0
"""

from midi_generator.training.hierarchical_mtl.config.training_config import (
    HierarchicalMTLConfig,
    OptimizerConfig,
    SchedulerConfig,
    DataConfig
)

from midi_generator.training.hierarchical_mtl.data.dataset import (
    HierarchicalMIDIDataset,
    create_dataloaders
)

from midi_generator.training.hierarchical_mtl.loops.trainer import (
    HierarchicalMTLTrainer
)

from midi_generator.training.hierarchical_mtl.callbacks.early_stopping import (
    EarlyStopping
)

from midi_generator.training.hierarchical_mtl.callbacks.checkpoint import (
    ModelCheckpoint
)

__all__ = [
    # Configuration
    'HierarchicalMTLConfig',
    'OptimizerConfig',
    'SchedulerConfig',
    'DataConfig',

    # Data
    'HierarchicalMIDIDataset',
    'create_dataloaders',

    # Training
    'HierarchicalMTLTrainer',

    # Callbacks
    'EarlyStopping',
    'ModelCheckpoint',
]

__version__ = '2.0.0'
__author__ = 'Agent 06 - Training Pipeline Engineer'
