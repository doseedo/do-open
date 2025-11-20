"""
Parameter Registry with Schema Validation

This replaces the old parameter registry that allowed dynamic code generation.
All parameters are now declared in .params files and validated against a schema.

Author: Musical Program Synthesis Team
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import jsonschema
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ParameterRegistry:
    """
    Safe parameter registry using declarative .params files.

    NO CODE GENERATION - only validated YAML specifications.
    """

    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize registry with schema."""
        if schema_path is None:
            schema_path = Path(__file__).parent / "schema.json"

        self.schema = self._load_schema(schema_path)
        self.parameters: Dict[str, Dict[str, Any]] = {}
        self.parameter_files: Dict[str, Path] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}

        # Load all existing parameters
        self._load_all_parameters()

    def _load_schema(self, schema_path: Path) -> Dict:
        """Load JSON Schema for parameter validation."""
        with open(schema_path) as f:
            return json.load(f)

    def _load_all_parameters(self):
        """Load all .params files from parameters directory."""
        params_dir = Path(__file__).parent

        # Load from multiple locations
        for params_file in params_dir.glob("**/*.params"):
            try:
                self.add_parameter(params_file, validate=True)
            except Exception as e:
                print(f"Warning: Failed to load {params_file}: {e}")

    def add_parameter(self, param_spec_path: Path, validate: bool = True) -> bool:
        """
        Add a new parameter from .params file.

        Args:
            param_spec_path: Path to .params YAML file
            validate: Whether to validate against schema

        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Load YAML
            with open(param_spec_path) as f:
                spec = yaml.safe_load(f)

            # Validate against schema
            if validate:
                validation = self.validate_spec(spec)
                if not validation.is_valid:
                    raise ValidationError(f"Invalid spec: {validation.errors}")

            # Check for conflicts with existing params
            conflicts = self._check_conflicts(spec)
            if conflicts:
                raise ConflictError(f"Conflicts with: {conflicts}")

            # Register parameter
            param_name = spec['name']
            self.parameters[param_name] = spec
            self.parameter_files[param_name] = param_spec_path

            # Update dependency graph
            self._update_dependencies(spec)

            return True

        except Exception as e:
            print(f"Error adding parameter from {param_spec_path}: {e}")
            return False

    def validate_spec(self, spec: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameter spec against JSON Schema.

        Returns:
            ValidationResult with detailed errors/warnings
        """
        errors = []
        warnings = []

        try:
            # JSON Schema validation
            jsonschema.validate(instance=spec, schema=self.schema)

        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Additional semantic validation

        # 1. Check default value is valid
        if not self._is_valid_default(spec):
            errors.append(f"Default value {spec['default']} not valid for type {spec['type']}")

        # 2. Check feature mappings exist (if we have feature list)
        if 'feature_mappings' in spec:
            missing_features = self._check_feature_mappings(spec['feature_mappings'])
            if missing_features:
                warnings.append(f"Unknown features: {missing_features}")

        # 3. Check dependencies exist
        if 'constraints' in spec and 'requires' in spec['constraints']:
            missing_deps = self._check_dependencies(spec['constraints']['requires'])
            if missing_deps:
                errors.append(f"Required parameters not found: {missing_deps}")

        # 4. Check for circular dependencies
        if self._has_circular_dependency(spec):
            errors.append(f"Circular dependency detected for {spec['name']}")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def _is_valid_default(self, spec: Dict) -> bool:
        """Check if default value matches type constraints."""
        param_type = spec['type']
        default = spec['default']

        if param_type == 'categorical':
            return default in spec.get('values', [])

        elif param_type in ['float', 'int', 'probability']:
            if 'range' not in spec:
                return False
            min_val, max_val = spec['range']['min'], spec['range']['max']
            return min_val <= default <= max_val

        elif param_type == 'boolean':
            return isinstance(default, bool)

        return True

    def _check_conflicts(self, spec: Dict) -> List[str]:
        """Check for conflicts with existing parameters."""
        conflicts = []
        param_name = spec['name']

        # Check if parameter already exists
        if param_name in self.parameters:
            conflicts.append(f"Parameter {param_name} already exists")

        # Check mutually exclusive parameters
        if 'constraints' in spec and 'mutually_exclusive' in spec['constraints']:
            for exclusive_param in spec['constraints']['mutually_exclusive']:
                if exclusive_param in self.parameters:
                    # Check if both would be active
                    conflicts.append(f"Mutually exclusive with existing {exclusive_param}")

        return conflicts

    def _check_dependencies(self, required_params: List[str]) -> List[str]:
        """Check if required dependencies exist."""
        missing = []
        for req in required_params:
            if req not in self.parameters:
                missing.append(req)
        return missing

    def _check_feature_mappings(self, features: List[str]) -> List[str]:
        """Check if feature names are valid (placeholder - needs feature list)."""
        # TODO: Load feature list from Agent 8
        return []

    def _has_circular_dependency(self, spec: Dict) -> bool:
        """Check for circular dependencies."""
        visited = set()

        def visit(param_name):
            if param_name in visited:
                return True
            visited.add(param_name)

            param_spec = self.parameters.get(param_name)
            if not param_spec:
                return False

            if 'constraints' in param_spec and 'requires' in param_spec['constraints']:
                for dep in param_spec['constraints']['requires']:
                    if visit(dep):
                        return True

            visited.remove(param_name)
            return False

        return visit(spec['name'])

    def _update_dependencies(self, spec: Dict):
        """Update dependency graph."""
        param_name = spec['name']
        self.dependency_graph[param_name] = set()

        if 'constraints' in spec and 'requires' in spec['constraints']:
            self.dependency_graph[param_name].update(spec['constraints']['requires'])

    def get_parameter(self, param_name: str) -> Optional[Dict]:
        """Get parameter specification."""
        return self.parameters.get(param_name)

    def get_default(self, param_name: str) -> Any:
        """Get default value for parameter."""
        spec = self.parameters.get(param_name)
        if spec:
            return spec['default']
        return None

    def get_parameter_type(self, param_name: str) -> Optional[str]:
        """Get parameter type."""
        spec = self.parameters.get(param_name)
        if spec:
            return spec['type']
        return None

    def list_parameters(self, domain: Optional[str] = None) -> List[str]:
        """List all parameter names, optionally filtered by domain."""
        if domain:
            return [
                name for name, spec in self.parameters.items()
                if spec.get('domain') == domain
            ]
        return list(self.parameters.keys())

    def get_dependencies(self, param_name: str) -> Set[str]:
        """Get all dependencies for a parameter (recursive)."""
        if param_name not in self.dependency_graph:
            return set()

        deps = set(self.dependency_graph[param_name])
        for dep in list(deps):
            deps.update(self.get_dependencies(dep))

        return deps

    def export_to_json(self, output_path: Path):
        """Export all parameters to single JSON file."""
        export_data = {
            'parameters': self.parameters,
            'dependency_graph': {
                k: list(v) for k, v in self.dependency_graph.items()
            },
            'exported_at': datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        stats = {
            'total_parameters': len(self.parameters),
            'by_domain': {},
            'by_type': {},
            'with_dependencies': 0,
            'human_created': 0,
            'llm_created': 0
        }

        for spec in self.parameters.values():
            # Count by domain
            domain = spec.get('domain', 'unknown')
            stats['by_domain'][domain] = stats['by_domain'].get(domain, 0) + 1

            # Count by type
            param_type = spec['type']
            stats['by_type'][param_type] = stats['by_type'].get(param_type, 0) + 1

            # Count dependencies
            if 'constraints' in spec and 'requires' in spec['constraints']:
                stats['with_dependencies'] += 1

            # Count creation method
            created_by = spec.get('created_by', 'unknown')
            if created_by == 'human':
                stats['human_created'] += 1
            elif created_by in ['llm_agent', 'self_expansion']:
                stats['llm_created'] += 1

        return stats


class ValidationError(Exception):
    """Raised when parameter validation fails."""
    pass


class ConflictError(Exception):
    """Raised when parameter conflicts with existing parameters."""
    pass


# Singleton instance
_registry = None

def get_registry() -> ParameterRegistry:
    """Get global parameter registry instance."""
    global _registry
    if _registry is None:
        _registry = ParameterRegistry()
    return _registry
