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

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Import structure expansion module
from .structure_expansion import (
    register_all_structure_parameters,
    get_structure_defaults,
    GENRE_PRESETS
)
=======
# Import and auto-register expansion modules
from . import registry_expansion
from . import harmony_deep_expansion

# Auto-register all expansion parameters
registry_expansion.register_all_expansions()
harmony_deep_expansion.register_all_harmony_deep_expansion()
>>>>>>> origin/claude/music-generation-agents-01KHsxYd7UXSFQAsHutaMgBi
=======
# Import expansion modules to auto-register parameters
from . import registry_expansion  # Agent 1: Core parameters
from . import melody_rhythm_expansion  # Agent 4: Melody & Rhythm expansion (120 params)
>>>>>>> origin/claude/music-generation-agents-01Hg1HTEAMZ318B1Ad5Zy6mm
=======
# Import expansion modules to register parameters
from . import registry_expansion
from . import dynamics_articulation_expansion
>>>>>>> origin/claude/music-generation-agents-017y2cya6dkgoQQJhEDyjRtP

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
    'register_all_structure_parameters',
    'get_structure_defaults',
    'GENRE_PRESETS'
]
