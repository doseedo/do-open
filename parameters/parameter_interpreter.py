"""
Parameter Interpreter - Runtime Parameter Resolution

Reads .params specifications at runtime and resolves values from multiple sources.
NO CODE EXECUTION - just declarative interpretation.

Author: Musical Program Synthesis Team
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .parameter_registry import get_registry, ParameterRegistry


@dataclass
class ParameterContext:
    """Context for parameter resolution."""
    predicted_params: Optional[Dict[str, Any]] = None
    user_overrides: Optional[Dict[str, Any]] = None
    confidence_scores: Optional[Dict[str, float]] = None
    confidence_threshold: float = 0.7


class ParameterInterpreter:
    """
    Interprets parameter specifications at runtime.

    Resolves parameter values from multiple sources with priority:
    1. User overrides (highest priority)
    2. ML predictions (if confidence > threshold)
    3. Default values (lowest priority)
    """

    def __init__(self, registry: Optional[ParameterRegistry] = None):
        self.registry = registry or get_registry()
        self.confidence_threshold = 0.7

    def get_parameter_value(
        self,
        param_name: str,
        context: Optional[ParameterContext] = None
    ) -> Any:
        """
        Get parameter value with validation and context resolution.

        Args:
            param_name: Fully qualified parameter name
            context: Optional context with predictions/overrides

        Returns:
            Resolved parameter value
        """
        spec = self.registry.get_parameter(param_name)

        if not spec:
            raise ParameterNotFoundError(f"Unknown parameter: {param_name}")

        # Resolve value from context or default
        value = self._resolve_value(spec, context)

        # Validate value
        if not self._is_valid_value(value, spec):
            raise ValueError(f"Invalid value {value} for {param_name}")

        return value

    def get_all_parameters(
        self,
        context: Optional[ParameterContext] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get all parameter values, optionally filtered by domain.

        Args:
            context: Optional context with predictions/overrides
            domain: Optional domain filter (e.g., "harmony", "melody")

        Returns:
            Dictionary of parameter names to values
        """
        param_names = self.registry.list_parameters(domain=domain)

        result = {}
        for param_name in param_names:
            try:
                result[param_name] = self.get_parameter_value(param_name, context)
            except Exception as e:
                print(f"Warning: Failed to resolve {param_name}: {e}")
                # Use default if resolution fails
                result[param_name] = self.registry.get_default(param_name)

        return result

    def _resolve_value(
        self,
        spec: Dict[str, Any],
        context: Optional[ParameterContext]
    ) -> Any:
        """
        Resolve parameter value from multiple sources.

        Priority:
        1. User overrides (explicit)
        2. ML predictions (if high confidence)
        3. Default value
        """
        param_name = spec['name']

        # Priority 1: User overrides
        if context and context.user_overrides and param_name in context.user_overrides:
            return context.user_overrides[param_name]

        # Priority 2: ML predictions (with confidence check)
        if context and context.predicted_params and param_name in context.predicted_params:
            confidence = context.confidence_scores.get(param_name, 0.0) if context.confidence_scores else 0.0
            threshold = context.confidence_threshold

            if confidence >= threshold:
                return context.predicted_params[param_name]

        # Priority 3: Default value
        return spec['default']

    def _is_valid_value(self, value: Any, spec: Dict[str, Any]) -> bool:
        """Validate value against parameter specification."""
        param_type = spec['type']

        if param_type == 'categorical':
            return value in spec.get('values', [])

        elif param_type in ['float', 'probability']:
            if not isinstance(value, (int, float)):
                return False
            if 'range' in spec:
                min_val = spec['range']['min']
                max_val = spec['range']['max']
                return min_val <= value <= max_val
            return True

        elif param_type == 'int':
            if not isinstance(value, int):
                return False
            if 'range' in spec:
                min_val = spec['range']['min']
                max_val = spec['range']['max']
                return min_val <= value <= max_val
            return True

        elif param_type == 'boolean':
            return isinstance(value, bool)

        return False

    def validate_parameter_set(
        self,
        params: Dict[str, Any],
        check_dependencies: bool = True
    ) -> tuple[bool, List[str]]:
        """
        Validate a complete parameter set.

        Args:
            params: Dictionary of parameter names to values
            check_dependencies: Whether to check dependency constraints

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check all parameters individually
        for param_name, value in params.items():
            spec = self.registry.get_parameter(param_name)
            if not spec:
                errors.append(f"Unknown parameter: {param_name}")
                continue

            if not self._is_valid_value(value, spec):
                errors.append(f"Invalid value {value} for {param_name}")

        # Check dependencies if requested
        if check_dependencies:
            dep_errors = self._check_dependency_constraints(params)
            errors.extend(dep_errors)

        return len(errors) == 0, errors

    def _check_dependency_constraints(self, params: Dict[str, Any]) -> List[str]:
        """Check if parameter dependencies are satisfied."""
        errors = []

        for param_name in params.keys():
            spec = self.registry.get_parameter(param_name)
            if not spec:
                continue

            # Check required dependencies
            if 'constraints' in spec and 'requires' in spec['constraints']:
                for required in spec['constraints']['requires']:
                    if required not in params:
                        errors.append(
                            f"{param_name} requires {required} which is not present"
                        )

            # Check conflicts
            if 'constraints' in spec and 'conflicts_with' in spec['constraints']:
                for conflict in spec['constraints']['conflicts_with']:
                    if conflict in params:
                        errors.append(
                            f"{param_name} conflicts with {conflict}"
                        )

        return errors

    def get_parameter_info(self, param_name: str) -> Dict[str, Any]:
        """Get detailed information about a parameter."""
        spec = self.registry.get_parameter(param_name)
        if not spec:
            raise ParameterNotFoundError(f"Unknown parameter: {param_name}")

        info = {
            'name': spec['name'],
            'type': spec['type'],
            'domain': spec['domain'],
            'description': spec['description'],
            'default': spec['default']
        }

        # Add type-specific info
        if spec['type'] == 'categorical':
            info['values'] = spec.get('values', [])
        elif spec['type'] in ['float', 'int', 'probability']:
            info['range'] = spec.get('range')

        # Add optional info
        if 'examples' in spec:
            info['examples'] = spec['examples']
        if 'theory_reference' in spec:
            info['theory_reference'] = spec['theory_reference']
        if 'constraints' in spec:
            info['constraints'] = spec['constraints']

        return info


class ParameterNotFoundError(Exception):
    """Raised when parameter is not found in registry."""
    pass


# Convenience function
def interpret_parameters(
    predicted: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    confidence: Optional[Dict[str, float]] = None,
    confidence_threshold: float = 0.7,
    domain: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to interpret parameters with context.

    Example:
        >>> params = interpret_parameters(
        ...     predicted={'harmony.voicing.type': 'drop_2'},
        ...     confidence={'harmony.voicing.type': 0.85},
        ...     domain='harmony'
        ... )
    """
    context = ParameterContext(
        predicted_params=predicted,
        user_overrides=overrides,
        confidence_scores=confidence,
        confidence_threshold=confidence_threshold
    )

    interpreter = ParameterInterpreter()
    return interpreter.get_all_parameters(context, domain=domain)
