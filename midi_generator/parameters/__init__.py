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

# Import expansion modules to auto-register parameters
from . import registry_expansion  # Agent 1: Core parameters
from . import melody_rhythm_expansion  # Agent 4: Melody & Rhythm expansion (120 params)

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
    'get_default'
]
