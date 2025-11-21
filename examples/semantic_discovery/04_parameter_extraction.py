#!/usr/bin/env python3
"""
Example 4: Parameter Extraction

Extract discovered semantic parameters from MIDI files and use them.

Usage:
    python 04_parameter_extraction.py

Author: Agent 10 - Documentation & Examples
Date: November 2025
"""

from pathlib import Path
import json
from midi_generator.parameters.universal_registry import UniversalParameterRegistry
from midi_generator.learning.semantic_features import SemanticFeatureBank


def extract_from_single_file(midi_file: Path, registry: UniversalParameterRegistry):
    """Extract all discovered parameters from a single MIDI file"""
    print(f"\nExtracting parameters from: {midi_file.name}")
    print("-" * 60)

    # Get all discovered parameter names
    discovered_params = [
        name for name in registry.list_parameters()
        if name.startswith("discovered_") or name in [
            "swing_ratio", "chord_density", "rhythmic_complexity",
            # ... add other known discovered parameter names
        ]
    ]

    if not discovered_params:
        print("No discovered parameters found in registry")
        return {}

    # Extract each parameter
    params = {}
    for param_name in discovered_params:
        try:
            value = registry.extract_parameter(param_name, midi_file)
            params[param_name] = float(value)
            print(f"  {param_name:30s}: {value:.3f}")
        except Exception as e:
            print(f"  {param_name:30s}: ERROR - {e}")

    return params


def extract_from_corpus(corpus_dir: Path, registry: UniversalParameterRegistry, limit: int = 10):
    """Extract parameters from multiple files"""
    print(f"\nExtracting from corpus: {corpus_dir}")
    print("=" * 60)

    midi_files = list(corpus_dir.glob("*.mid"))[:limit]

    all_params = []
    for midi_file in midi_files:
        params = extract_from_single_file(midi_file, registry)
        if params:
            params['filename'] = midi_file.name
            all_params.append(params)

    return all_params


def analyze_parameter_statistics(all_params: list):
    """Compute statistics across extracted parameters"""
    import numpy as np

    if not all_params:
        print("No parameters to analyze")
        return

    print("\n" + "=" * 60)
    print("Parameter Statistics")
    print("=" * 60)

    # Get parameter names (excluding filename)
    param_names = [k for k in all_params[0].keys() if k != 'filename']

    print(f"\n{'Parameter':<30s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print("-" * 70)

    for param_name in param_names:
        values = [p[param_name] for p in all_params if param_name in p]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)

            print(f"{param_name:<30s} {mean:>10.3f} {std:>10.3f} {min_val:>10.3f} {max_val:>10.3f}")


def save_parameters(all_params: list, output_file: Path):
    """Save extracted parameters to JSON"""
    print(f"\nSaving parameters to: {output_file}")

    with open(output_file, 'w') as f:
        json.dump(all_params, f, indent=2)

    print(f"Saved {len(all_params)} files' parameters")


def main():
    """Main function"""
    print("="*60)
    print("EXAMPLE 4: Parameter Extraction")
    print("="*60)

    # Load registry (assumes parameters were registered)
    registry = UniversalParameterRegistry()

    # Test on single file
    test_file = Path("data/midi/test/example.mid")
    if test_file.exists():
        params = extract_from_single_file(test_file, registry)
    else:
        print(f"\nTest file not found: {test_file}")
        print("Skipping single file test")

    # Extract from corpus
    corpus_dir = Path("data/midi/test")
    if corpus_dir.exists():
        all_params = extract_from_corpus(corpus_dir, registry, limit=20)

        # Analyze
        analyze_parameter_statistics(all_params)

        # Save
        output_file = Path("output/extracted_parameters.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_parameters(all_params, output_file)
    else:
        print(f"\nCorpus directory not found: {corpus_dir}")

    print("\n" + "="*60)
    print("Extraction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
