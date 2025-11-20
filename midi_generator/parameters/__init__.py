"""
Parameter Management System for Musical Program Synthesis

This module provides the infrastructure for exposing and managing
all musical parameters across the entire library.
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
