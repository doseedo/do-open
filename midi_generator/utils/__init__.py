"""
MIDI Generator Utilities - Agent 2
===================================

Differentiable MIDI representations and utilities for neural music generation.

Components:
- differentiable_midi: SoftPianoRoll for differentiable MIDI representation
- soft_sampling: GumbelSoftmaxSampler for differentiable discrete sampling
- midi_assembly: MIDIAssembler for converting decoder outputs to MIDI
- midi_validation: MIDIValidator for validation and quality metrics

Author: Agent 2 - Differentiable MIDI & Utilities Support
Date: November 22, 2025
"""

from .differentiable_midi import SoftPianoRoll, PianoRollConfig
from .soft_sampling import GumbelSoftmaxSampler
from .midi_assembly import MIDIAssembler
from .midi_validation import MIDIValidator

__all__ = [
    'SoftPianoRoll',
    'PianoRollConfig',
    'GumbelSoftmaxSampler',
    'MIDIAssembler',
    'MIDIValidator',
]
