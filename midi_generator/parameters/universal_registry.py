#!/usr/bin/env python3
"""
Universal Parameter Registry
=============================

Central registry for all learnable parameters across the entire music generation system.
This enables ML-based parameter learning (XGBoost, neural networks) by exposing all
hardcoded decisions as configurable parameters.

Architecture:
- Hierarchical naming: domain.module.parameter (e.g., "jazz.harmony.tritone_sub_probability")
- Type system: categorical, continuous, boolean, integer, array
- Metadata: ranges, defaults, descriptions, musical impact scores
- Validation: constraints, dependencies, music theory rules

Author: Agent 4 - Parameter Refactoring
License: MIT
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class ParameterType(Enum):
    """Parameter type classification"""
    CONTINUOUS = "continuous"  # Float in range [min, max]
    CATEGORICAL = "categorical"  # One of fixed set of options
    BOOLEAN = "boolean"  # True/False
    INTEGER = "integer"  # Integer in range [min, max]
    ARRAY = "array"  # List of values
    PATTERN = "pattern"  # Rhythmic/melodic pattern


class MusicalDomain(Enum):
    """Top-level musical domains"""
    HARMONY = "harmony"
    MELODY = "melody"
    RHYTHM = "rhythm"
    TEXTURE = "texture"
    FORM = "form"
    DYNAMICS = "dynamics"
    ARTICULATION = "articulation"
    TIMBRE = "timbre"


@dataclass
class ParameterMetadata:
    """Complete metadata for a learnable parameter"""

    name: str  # Hierarchical name: domain.module.parameter
    type: ParameterType
    default: Any
    description: str

    # Type-specific constraints
    range: Optional[Tuple[float, float]] = None  # For continuous/integer
    options: Optional[List[Any]] = None  # For categorical

    # Musical metadata
    domain: Optional[MusicalDomain] = None
    module: Optional[str] = None
    musical_impact: str = "medium"  # low/medium/high
    genre_relevance: List[str] = field(default_factory=list)

    # ML metadata
    learnable: bool = True
    feature_importance: float = 0.0  # Updated during training

    # Validation
    constraints: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)

    def validate(self, value: Any) -> bool:
        """Validate a value against parameter constraints"""

        if self.type == ParameterType.CONTINUOUS:
            if self.range:
                return self.range[0] <= value <= self.range[1]

        elif self.type == ParameterType.INTEGER:
            if not isinstance(value, int):
                return False
            if self.range:
                return self.range[0] <= value <= self.range[1]

        elif self.type == ParameterType.CATEGORICAL:
            if self.options:
                return value in self.options

        elif self.type == ParameterType.BOOLEAN:
            return isinstance(value, bool)

        elif self.type == ParameterType.ARRAY:
            return isinstance(value, (list, tuple))

        return True


class UniversalParameterRegistry:
    """
    Global registry of all parameters across the music generation system.

    Singleton pattern - accessed via: registry.get_parameter(name)
    """

    _instance = None
    _parameters: Dict[str, ParameterMetadata] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._parameters = {}
        return cls._instance

    def register_parameter(
        self,
        name: str,
        type: Union[ParameterType, str],
        default: Any,
        description: str = "",
        **kwargs
    ) -> None:
        """
        Register a new parameter in the global registry.

        Args:
            name: Hierarchical name (e.g., "jazz.harmony.tritone_sub_prob")
            type: Parameter type (ParameterType enum or string)
            default: Default value
            description: Human-readable description
            **kwargs: Additional metadata (range, options, domain, etc.)
        """

        # Convert string type to enum
        if isinstance(type, str):
            type = ParameterType(type)

        # Create metadata
        metadata = ParameterMetadata(
            name=name,
            type=type,
            default=default,
            description=description,
            **kwargs
        )

        # Validate default value
        if not metadata.validate(default):
            raise ValueError(f"Default value {default} invalid for parameter {name}")

        # Register
        self._parameters[name] = metadata

    def get_parameter(self, name: str, default: Any = None) -> ParameterMetadata:
        """Get parameter metadata by name"""
        return self._parameters.get(name, default)

    def get_value(self, name: str, params: Dict[str, Any], default: Any = None) -> Any:
        """
        Get parameter value from params dict, falling back to registered default.

        Args:
            name: Parameter name
            params: Dictionary of parameter overrides
            default: Override default if parameter not found

        Returns:
            Parameter value with fallback chain: params[name] -> registry default -> provided default
        """

        # Check if in params dict
        if name in params:
            value = params[name]
            # Validate
            metadata = self.get_parameter(name)
            if metadata and not metadata.validate(value):
                raise ValueError(f"Invalid value {value} for parameter {name}")
            return value

        # Use registry default
        metadata = self.get_parameter(name)
        if metadata:
            return metadata.default

        # Use provided default
        return default

    def get_all_parameters(self) -> Dict[str, ParameterMetadata]:
        """Get all registered parameters"""
        return self._parameters.copy()

    def get_by_domain(self, domain: Union[MusicalDomain, str]) -> Dict[str, ParameterMetadata]:
        """Get all parameters in a musical domain"""
        if isinstance(domain, str):
            domain = MusicalDomain(domain)

        return {
            name: meta
            for name, meta in self._parameters.items()
            if meta.domain == domain
        }

    def get_by_module(self, module: str) -> Dict[str, ParameterMetadata]:
        """Get all parameters in a module"""
        return {
            name: meta
            for name, meta in self._parameters.items()
            if meta.module == module
        }

    def get_by_genre(self, genre: str) -> Dict[str, ParameterMetadata]:
        """Get all parameters relevant to a genre"""
        return {
            name: meta
            for name, meta in self._parameters.items()
            if genre in meta.genre_relevance
        }

    def search(self, pattern: str) -> Dict[str, ParameterMetadata]:
        """Search parameters by name pattern"""
        return {
            name: meta
            for name, meta in self._parameters.items()
            if pattern.lower() in name.lower()
        }

    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save registry to JSON file"""
        path = Path(path)

        data = {
            name: {
                'type': meta.type.value,
                'default': meta.default,
                'description': meta.description,
                'range': meta.range,
                'options': meta.options,
                'domain': meta.domain.value if meta.domain else None,
                'module': meta.module,
                'musical_impact': meta.musical_impact,
                'genre_relevance': meta.genre_relevance,
                'learnable': meta.learnable,
                'feature_importance': meta.feature_importance,
            }
            for name, meta in self._parameters.items()
        }

        path.write_text(json.dumps(data, indent=2))

    def load_from_file(self, path: Union[str, Path]) -> None:
        """Load registry from JSON file"""
        path = Path(path)
        data = json.loads(path.read_text())

        for name, meta_dict in data.items():
            self.register_parameter(
                name=name,
                type=meta_dict['type'],
                default=meta_dict['default'],
                description=meta_dict.get('description', ''),
                range=tuple(meta_dict['range']) if meta_dict.get('range') else None,
                options=meta_dict.get('options'),
                domain=meta_dict.get('domain'),
                module=meta_dict.get('module'),
                musical_impact=meta_dict.get('musical_impact', 'medium'),
                genre_relevance=meta_dict.get('genre_relevance', []),
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        stats = {
            'total_parameters': len(self._parameters),
            'by_type': {},
            'by_domain': {},
            'by_module': {},
            'learnable_count': 0,
        }

        for meta in self._parameters.values():
            # By type
            type_name = meta.type.value
            stats['by_type'][type_name] = stats['by_type'].get(type_name, 0) + 1

            # By domain
            if meta.domain:
                domain_name = meta.domain.value
                stats['by_domain'][domain_name] = stats['by_domain'].get(domain_name, 0) + 1

            # By module
            if meta.module:
                stats['by_module'][meta.module] = stats['by_module'].get(meta.module, 0) + 1

            # Learnable
            if meta.learnable:
                stats['learnable_count'] += 1

        return stats

    def clear(self) -> None:
        """Clear all registered parameters (for testing)"""
        self._parameters.clear()


# Global singleton instance
registry = UniversalParameterRegistry()


# Convenience function for getting parameter values
def param(name: str, params: Dict[str, Any] = None, default: Any = None) -> Any:
    """
    Convenience function to get parameter value.

    Usage:
        swing_ratio = param('jazz.swing.ratio', params, 0.67)

    Args:
        name: Parameter name
        params: Parameter override dictionary
        default: Fallback default value

    Returns:
        Parameter value
    """
    if params is None:
        params = {}
    return registry.get_value(name, params, default)


# Export main classes
__all__ = [
    'ParameterType',
    'MusicalDomain',
    'ParameterMetadata',
    'UniversalParameterRegistry',
    'registry',
    'param',
]
