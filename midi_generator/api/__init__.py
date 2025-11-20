"""
Unified API for HarmonyModule Modular Fusion System

This package provides high-level interfaces for all modular genre fusion capabilities
and big band generation.
"""

from .unified_api import (
    HarmonyModuleAPI,
    QuickFusion,
    GenreBlend,
    ComponentMix,
    ContextGeneration,
    InpaintSection,
    TransformTempo,
    TransformMeter,
    GranularControl,
)

from .big_band_api import (
    BigBandGenerator,
    BigBandConfig,
    generate_big_band,
    list_available_styles,
)

__all__ = [
    # Modular Fusion API
    'HarmonyModuleAPI',
    'QuickFusion',
    'GenreBlend',
    'ComponentMix',
    'ContextGeneration',
    'InpaintSection',
    'TransformTempo',
    'TransformMeter',
    'GranularControl',
    # Big Band API
    'BigBandGenerator',
    'BigBandConfig',
    'generate_big_band',
    'list_available_styles',
]
