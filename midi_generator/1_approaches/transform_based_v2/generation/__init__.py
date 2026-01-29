"""Generation Pipeline for v53 Pure Contour Grammar.

This module provides tools for generating new MIDI arrangements from
the v53 checkpoint's pattern vocabulary, transforms, and orchestration rules.
"""

from .generator import ArrangementGenerator
from .sampler import PatternSampler, TransformSampler
from .orchestrator import Orchestrator
from .timing import TimingAssigner
from .midi_output import MIDIWriter

__all__ = [
    'ArrangementGenerator',
    'PatternSampler',
    'TransformSampler',
    'Orchestrator',
    'TimingAssigner',
    'MIDIWriter',
]
