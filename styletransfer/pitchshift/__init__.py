"""
Register-Aware Pitch Shift Pipeline

A neural network approach to pitch shifting that preserves
register-specific timbre characteristics.

Key components:
- dataset_pitch_shift.py: Dataset with MIDI-based pitch indexing
- models_pitch_shift.py: RegisterAwareTranslator with FiLM conditioning
- train_pitch_shift.py: Training script
- inference_pitch_shift.py: Inference pipeline
- generate_synthetic_pitch_shift.py: Synthetic data generation for student training

The pipeline works in 4 steps:
1. Train pitch-conditioned translator using double-shift strategy
2. Evaluate on various shift amounts
3. Generate synthetic pairs for student training
4. Train lightweight student model for VST export
"""

from .models_pitch_shift import (
    RegisterAwareTranslator,
    RegisterAwareTranslatorDirect,
    RegisterAwareTranslatorLarge,
    CombinedLoss,
    TimbreLoss,
    ContentLoss,
    PitchShiftStudentModel,
)

from .dataset_pitch_shift import (
    PitchShiftDataset,
    PitchShiftDatasetWithRealDegradation,
)

__all__ = [
    'RegisterAwareTranslator',
    'RegisterAwareTranslatorDirect',
    'RegisterAwareTranslatorLarge',
    'CombinedLoss',
    'TimbreLoss',
    'ContentLoss',
    'PitchShiftStudentModel',
    'PitchShiftDataset',
    'PitchShiftDatasetWithRealDegradation',
]
