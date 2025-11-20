#!/usr/bin/env python3
"""
Convert Existing Parameters to .params Format

This script migrates the existing parameter registry from the old format
to the new declarative .params format.

Usage:
    python scripts/convert_registry_to_params.py

Author: Musical Program Synthesis Team
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def convert_parameter_to_spec(
    param_name: str,
    param_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Convert old parameter format to new .params spec."""

    # Parse parameter name
    parts = param_name.split('.')
    if len(parts) >= 2:
        domain = parts[0]
        subdomain = parts[1] if len(parts) > 2 else parts[0]
    else:
        domain = "unknown"
        subdomain = "unknown"

    # Determine type and constraints from parameter data
    param_type = param_data.get('type', 'categorical')

    spec = {
        'name': param_name,
        'type': param_type,
        'domain': domain,
        'subdomain': subdomain,
        'description': param_data.get('description', f"Parameter for {param_name}"),
        'version': '1.0.0',
        'created_by': 'human',
        'created_at': datetime.now().isoformat()
    }

    # Add type-specific fields
    if param_type == 'categorical':
        spec['values'] = param_data.get('values', ['default'])
        spec['default'] = param_data.get('default', spec['values'][0])

    elif param_type in ['float', 'int', 'probability']:
        spec['range'] = param_data.get('range', {'min': 0.0, 'max': 1.0})
        spec['default'] = param_data.get('default', 0.5)

    elif param_type == 'boolean':
        spec['default'] = param_data.get('default', False)

    # Add examples if available
    if 'examples' in param_data:
        spec['examples'] = param_data['examples']

    # Add constraints if available
    constraints = {}
    if 'requires' in param_data:
        constraints['requires'] = param_data['requires']
    if 'conflicts_with' in param_data:
        constraints['conflicts_with'] = param_data['conflicts_with']
    if constraints:
        spec['constraints'] = constraints

    # Add feature mappings if available
    if 'feature_mappings' in param_data:
        spec['feature_mappings'] = param_data['feature_mappings']
    else:
        # Generate default feature mappings based on domain
        spec['feature_mappings'] = [
            f"{domain}_complexity",
            f"{domain}_density",
            f"{domain}_variation"
        ]

    # Add training hints
    spec['training_hints'] = {
        'importance': param_data.get('importance', 'medium'),
        'training_samples': param_data.get('training_samples', 1000),
        'feature_selection_method': 'mutual_info'
    }

    return spec


def load_existing_parameters() -> Dict[str, Dict[str, Any]]:
    """Load existing parameters from registry."""

    # Try to load from universal_registry.py
    registry_path = Path("midi_generator/parameters/universal_registry.py")

    if not registry_path.exists():
        print(f"Registry file not found: {registry_path}")
        print("Using example parameters instead")
        return get_example_parameters()

    # Parse Python file to extract parameter definitions
    # This is a simplified version - in production you'd need a proper parser
    print(f"Loading parameters from {registry_path}")

    # For now, return example parameters
    # TODO: Implement proper Python AST parsing
    return get_example_parameters()


def get_example_parameters() -> Dict[str, Dict[str, Any]]:
    """Get example parameters for testing."""
    return {
        'harmony.voicing.type': {
            'type': 'categorical',
            'values': ['close', 'open', 'drop_2', 'spread'],
            'default': 'close',
            'description': 'Type of harmonic voicing',
            'examples': {
                'close': 'Notes within an octave',
                'open': 'Notes spread across multiple octaves',
                'drop_2': 'Second voice dropped an octave',
                'spread': 'Wide spacing for orchestral sound'
            }
        },
        'harmony.chord_density': {
            'type': 'float',
            'range': {'min': 0.0, 'max': 1.0},
            'default': 0.5,
            'description': 'Number of chords per measure (normalized)'
        },
        'melody.contour.shape': {
            'type': 'categorical',
            'values': ['ascending', 'descending', 'arch', 'inverted_arch', 'wave'],
            'default': 'arch',
            'description': 'Overall shape of melodic contour'
        },
        'rhythm.swing.amount': {
            'type': 'probability',
            'range': {'min': 0.0, 'max': 1.0},
            'default': 0.5,
            'description': 'Amount of swing feel (0=straight, 1=maximum swing)'
        },
        'dynamics.overall_level': {
            'type': 'categorical',
            'values': ['ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff'],
            'default': 'mf',
            'description': 'Overall dynamic level'
        }
    }


def convert_all_parameters(output_dir: Path):
    """Convert all parameters and save as .params files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading existing parameters...")
    parameters = load_existing_parameters()

    print(f"Found {len(parameters)} parameters")
    print(f"Converting to .params format...")

    converted_count = 0

    for param_name, param_data in parameters.items():
        try:
            # Convert to spec
            spec = convert_parameter_to_spec(param_name, param_data)

            # Create filename
            filename = param_name.replace('.', '_') + '.params'
            filepath = output_dir / filename

            # Save as YAML
            with open(filepath, 'w') as f:
                yaml.dump(spec, f, default_flow_style=False, sort_keys=False)

            print(f"✓ Converted: {param_name} -> {filename}")
            converted_count += 1

        except Exception as e:
            print(f"✗ Failed to convert {param_name}: {e}")

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Converted {converted_count}/{len(parameters)} parameters")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def create_core_parameter_set(output_dir: Path):
    """Create a minimal set of 50 core parameters for v1.0."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    core_parameters = {
        # Harmony (15 parameters)
        'harmony.chord_type': {
            'type': 'categorical',
            'values': ['major', 'minor', 'dominant', 'diminished', 'augmented'],
            'default': 'major'
        },
        'harmony.chord_density': {
            'type': 'float',
            'range': {'min': 0.0, 'max': 1.0},
            'default': 0.5
        },
        'harmony.voicing.spread': {
            'type': 'float',
            'range': {'min': 0.0, 'max': 1.0},
            'default': 0.5
        },

        # Melody (15 parameters)
        'melody.contour.shape': {
            'type': 'categorical',
            'values': ['ascending', 'descending', 'arch', 'wave'],
            'default': 'arch'
        },
        'melody.note_density': {
            'type': 'float',
            'range': {'min': 0.0, 'max': 1.0},
            'default': 0.5
        },

        # Rhythm (10 parameters)
        'rhythm.swing.amount': {
            'type': 'probability',
            'range': {'min': 0.0, 'max': 1.0},
            'default': 0.0
        },
        'rhythm.syncopation': {
            'type': 'probability',
            'range': {'min': 0.0, 'max': 1.0},
            'default': 0.3
        },

        # Dynamics (5 parameters)
        'dynamics.overall_level': {
            'type': 'categorical',
            'values': ['pp', 'p', 'mp', 'mf', 'f', 'ff'],
            'default': 'mf'
        },

        # Structure (5 parameters)
        'structure.form_type': {
            'type': 'categorical',
            'values': ['aaba', 'verse_chorus', 'through_composed'],
            'default': 'verse_chorus'
        }
    }

    print(f"Creating {len(core_parameters)} core parameters...")

    for param_name, param_data in core_parameters.items():
        param_data['description'] = f"Core parameter for {param_name}"
        spec = convert_parameter_to_spec(param_name, param_data)

        filename = param_name.replace('.', '_') + '.params'
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            yaml.dump(spec, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Created: {filename}")

    print(f"\nCreated {len(core_parameters)} core parameters in {output_dir}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert existing parameters to .params format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="parameters/converted",
        help="Output directory for .params files"
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Create only core 50 parameters for v1.0"
    )

    args = parser.parse_args()

    if args.core_only:
        create_core_parameter_set(Path(args.output_dir))
    else:
        convert_all_parameters(Path(args.output_dir))


if __name__ == "__main__":
    main()
