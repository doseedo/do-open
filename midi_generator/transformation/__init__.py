"""
Transformation Module - MIDI Transformation Tools

This module provides advanced MIDI transformation capabilities including:
- Style Transfer: Transform music to different styles
- Arrangement: Auto-arrange lead sheets to full arrangements
- Meter Conversion: Convert between time signatures (Agent 7)

Author: Multiple agents - HarmonyModule Enhancement Project
"""

# Import main classes for easy access
from .style_transfer import StyleTransfer, StyleProfile
from .arrangement_engine import ArrangementEngine
from .meter_converter import (
    MeterConverter,
    MetricModulator,
    PhrasePreserver,
    MeterUtilities,
    TimeSignatureInfo,
    ConversionStrategy,
    MeterConversionParams,
    convert_midi_meter
)

__all__ = [
    # Style Transfer
    'StyleTransfer',
    'StyleProfile',

    # Arrangement
    'ArrangementEngine',

    # Meter Conversion (Agent 7)
    'MeterConverter',
    'MetricModulator',
    'PhrasePreserver',
    'MeterUtilities',
    'TimeSignatureInfo',
    'ConversionStrategy',
    'MeterConversionParams',
    'convert_midi_meter',
]
