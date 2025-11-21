#!/usr/bin/env python3
"""
Parameter Management System for Musical Program Synthesis

Unified parameter registry for the MIDI Generator library.
Part of the Focused Parameter Refactoring project.

This module provides the infrastructure for exposing and managing
all musical parameters across the entire library.

This module will eventually contain ~500-800 parameters across all domains:
- harmony.* (150 params) - Agent 3
- melody.* (100 params) - Agent 4
- rhythm.* (100 params) - Agent 5
- structure.* (50 params) - Agent 6
- instrumentation.* (50 params) - Agent 7 ✅
- dynamics.* (50 params) - Agent 8
- transformation.* (remaining) - Agent 8

Current Status:
- Agent 7 (Instrumentation & Orchestration): ✅ Complete

Author: Focused Parameter Refactoring - Agents 1-10
"""

from .universal_registry import (
    UniversalParameterRegistry,
    ParameterType,
    ParameterDefinition,
    ParameterCategory,
    MusicalImpact,
    REGISTRY,
    get_parameter,
    validate,
    get_default
)

# Import and auto-register expansion modules from multiple agents
from . import registry_expansion  # Agent 1: Core parameters

# Agent 2: Structure expansion
try:
    from .structure_expansion import (
        register_all_structure_parameters,
        get_structure_defaults,
        GENRE_PRESETS
    )
except ImportError:
    pass

# Agent 3: Harmony deep expansion
try:
    from . import harmony_deep_expansion
    harmony_deep_expansion.register_all_harmony_deep_expansion()
except (ImportError, AttributeError):
    pass

# Agent 4: Melody & Rhythm expansion
try:
    from . import melody_rhythm_expansion
except ImportError:
    pass

# Agent 5: Dynamics & Articulation expansion
try:
    from . import dynamics_articulation_expansion
except ImportError:
    pass

# Agent 7: Instrumentation parameters
try:
    from .instrumentation_params import (
        INSTRUMENTATION_PARAMETERS,
        get_instrumentation_parameter,
        get_all_instrumentation_parameters
    )
    __all_instrumentation = [
        'INSTRUMENTATION_PARAMETERS',
        'get_instrumentation_parameter',
        'get_all_instrumentation_parameters'
    ]
except ImportError:
    __all_instrumentation = []

# Register all core expansions
try:
    registry_expansion.register_all_expansions()
except (AttributeError, NameError):
    pass

# Global registry instance (for convenience)
registry = REGISTRY

__all__ = [
    'UniversalParameterRegistry',
    'ParameterType',
    'ParameterDefinition',
    'ParameterCategory',
    'MusicalImpact',
    'REGISTRY',
    'registry',
    'get_parameter',
    'validate',
    'get_default',
]

# Add structure expansion exports if available
try:
    __all__.extend(['register_all_structure_parameters', 'get_structure_defaults', 'GENRE_PRESETS'])
except NameError:
    pass

# Add instrumentation exports if available
__all__.extend(__all_instrumentation)
