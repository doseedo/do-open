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

# Import structure expansion module
from .structure_expansion import (
    register_all_structure_parameters,
    get_structure_defaults,
    GENRE_PRESETS
)

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
