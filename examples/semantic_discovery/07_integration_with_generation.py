#!/usr/bin/env python3
"""
Example 7: Integration with MIDI Generation

Use discovered semantic parameters in MIDI generation pipeline.

Usage:
    python 07_integration_with_generation.py

Author: Agent 10 - Documentation & Examples
Date: November 2025
"""

from pathlib import Path
from midi_generator.parameters.universal_registry import UniversalParameterRegistry
from midi_generator.generators import MIDIGenerator


def extract_style_from_midi(midi_file: Path, registry: UniversalParameterRegistry):
    """Extract complete parameter set from a MIDI file"""
    print(f"\nExtracting style from: {midi_file.name}")

    params = {}

    # Extract all parameters (existing + discovered)
    all_param_names = registry.list_parameters()

    for param_name in all_param_names:
        try:
            value = registry.extract_parameter(param_name, midi_file)
            params[param_name] = value
        except Exception as e:
            print(f"  Warning: Could not extract {param_name}: {e}")

    print(f"  Extracted {len(params)} parameters")
    return params


def generate_with_discovered_params(params: dict, output_file: Path):
    """Generate MIDI using discovered parameters"""
    print(f"\nGenerating MIDI with discovered parameters...")

    generator = MIDIGenerator()

    # Generate
    midi = generator.generate(**params)

    # Save
    midi.save(output_file)
    print(f"  Saved: {output_file}")

    return midi


def style_transfer_example():
    """Example: Transfer style from one MIDI to generate new music"""
    print("\n" + "="*60)
    print("STYLE TRANSFER EXAMPLE")
    print("="*60)

    registry = UniversalParameterRegistry()

    # Source: Extract style parameters
    source_file = Path("data/midi/examples/jazz_original.mid")
    if not source_file.exists():
        print(f"\nSource file not found: {source_file}")
        print("Skipping style transfer example")
        return

    params = extract_style_from_midi(source_file, registry)

    # Show key parameters
    print("\nKey discovered parameters:")
    discovered_params = [k for k in params.keys() if k.startswith("discovered_")]
    for key in discovered_params[:10]:
        print(f"  {key:30s}: {params[key]:.3f}")

    # Generate new MIDI with same style
    output_file = Path("output/generated_jazz_style.mid")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    generate_with_discovered_params(params, output_file)

    print(f"\n✓ Style transfer complete!")
    print(f"  Source: {source_file}")
    print(f"  Output: {output_file}")


def parameter_interpolation_example():
    """Example: Interpolate between two musical styles"""
    print("\n" + "="*60)
    print("PARAMETER INTERPOLATION EXAMPLE")
    print("="*60)

    registry = UniversalParameterRegistry()

    # Two source files
    file_a = Path("data/midi/examples/classical.mid")
    file_b = Path("data/midi/examples/jazz.mid")

    if not (file_a.exists() and file_b.exists()):
        print("\nSource files not found:")
        print(f"  {file_a}")
        print(f"  {file_b}")
        print("Skipping interpolation example")
        return

    # Extract parameters from both
    params_a = extract_style_from_midi(file_a, registry)
    params_b = extract_style_from_midi(file_b, registry)

    # Interpolate
    print("\nGenerating interpolated styles...")
    output_dir = Path("output/interpolation")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, alpha in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        # Interpolate parameters
        params_interp = {}
        for key in params_a.keys():
            if key in params_b:
                params_interp[key] = (1 - alpha) * params_a[key] + alpha * params_b[key]
            else:
                params_interp[key] = params_a[key]

        # Generate
        output_file = output_dir / f"interp_{i:02d}_alpha_{alpha:.2f}.mid"
        generate_with_discovered_params(params_interp, output_file)

    print(f"\n✓ Generated 5 interpolated variations")
    print(f"  0% Classical → 100% Jazz")
    print(f"  Output: {output_dir}")


def parameter_variation_example():
    """Example: Generate variations by tweaking discovered parameters"""
    print("\n" + "="*60)
    print("PARAMETER VARIATION EXAMPLE")
    print("="*60)

    registry = UniversalParameterRegistry()

    # Source file
    source_file = Path("data/midi/examples/original.mid")
    if not source_file.exists():
        print(f"\nSource file not found: {source_file}")
        print("Skipping variation example")
        return

    # Extract base parameters
    base_params = extract_style_from_midi(source_file, registry)

    # Generate variations
    print("\nGenerating variations...")
    output_dir = Path("output/variations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Variation 1: Increase swing
    if "swing_ratio" in base_params:
        params_swing = base_params.copy()
        params_swing["swing_ratio"] *= 1.5
        output_file = output_dir / "variation_more_swing.mid"
        generate_with_discovered_params(params_swing, output_file)
        print("  ✓ More swing")

    # Variation 2: Simplify harmony
    if "chord_density" in base_params:
        params_simple = base_params.copy()
        params_simple["chord_density"] *= 0.7
        output_file = output_dir / "variation_simpler_harmony.mid"
        generate_with_discovered_params(params_simple, output_file)
        print("  ✓ Simpler harmony")

    # Variation 3: Increase complexity
    if "rhythmic_complexity" in base_params:
        params_complex = base_params.copy()
        params_complex["rhythmic_complexity"] *= 1.3
        output_file = output_dir / "variation_more_complex.mid"
        generate_with_discovered_params(params_complex, output_file)
        print("  ✓ More rhythmic complexity")

    print(f"\n✓ Generated variations in: {output_dir}")


def main():
    """Main function"""
    print("="*60)
    print("EXAMPLE 7: Integration with MIDI Generation")
    print("="*60)

    print("\nThis example demonstrates how to use discovered semantic")
    print("parameters in MIDI generation:")
    print("  1. Style Transfer: Clone style from existing MIDI")
    print("  2. Interpolation: Blend two musical styles")
    print("  3. Variation: Generate variations by tweaking parameters")
    print()

    # Run examples
    style_transfer_example()
    parameter_interpolation_example()
    parameter_variation_example()

    print("\n" + "="*60)
    print("Integration examples complete!")
    print("="*60)
    print("\nYou can now:")
    print("  - Use discovered parameters in your generation pipeline")
    print("  - Create custom parameter combinations")
    print("  - Build parameter-based music generation UIs")
    print()


if __name__ == "__main__":
    main()
