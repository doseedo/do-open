"""
Modular Semantic Discovery - Comprehensive Usage Examples
=========================================================

Agent 10: Documentation & Deployment Manager

This file demonstrates all major use cases for the Modular Semantic Discovery System.

Examples included:
1. Basic DNA extraction
2. Parameter-based editing
3. Cross-corpus style transfer
4. Batch processing and analysis
5. Real-time DNA editing
6. Training custom modules
7. API usage

Author: Agent 10
Date: November 21, 2025
Version: 1.0.0
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

# Suppress warnings for demo
warnings.filterwarnings('ignore')


# =============================================================================
# EXAMPLE 1: Basic DNA Extraction
# =============================================================================

def example_1_basic_extraction():
    """
    Extract 120-parameter musical DNA from a MIDI file.
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic DNA Extraction")
    print("=" * 80)

    try:
        from midi_generator.learning.modular_discovery_pipeline import ModularSemanticDiscoveryPipeline

        # Load pre-trained pipeline
        print("Loading pre-trained pipeline...")
        pipeline = ModularSemanticDiscoveryPipeline.load_pretrained()

        # Extract DNA from MIDI file
        midi_file = "data/examples/jazz_standard.mid"
        print(f"\nExtracting DNA from: {midi_file}")

        dna = pipeline.extract_dna(midi_file)

        print(f"\n✓ Extracted {len(dna)} parameters")
        print("\nSample parameters:")
        for i, (param, value) in enumerate(list(dna.items())[:10]):
            print(f"  {param:40s} = {value:.3f}")
        print(f"  ... ({len(dna) - 10} more)")

        # Save DNA to file
        output_path = "output/extracted_dna.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dna, f, indent=2)
        print(f"\n✓ DNA saved to: {output_path}")

    except ImportError as e:
        print(f"⚠️  Required modules not available: {e}")
        print("This example requires the trained modular pipeline.")

    print()


# =============================================================================
# EXAMPLE 2: Parameter-Based Editing
# =============================================================================

def example_2_parameter_editing():
    """
    Edit specific musical parameters and regenerate MIDI.
    """
    print("=" * 80)
    print("EXAMPLE 2: Parameter-Based Editing")
    print("=" * 80)

    try:
        from midi_generator.learning.modular_discovery_pipeline import ModularSemanticDiscoveryPipeline

        pipeline = ModularSemanticDiscoveryPipeline.load_pretrained()

        # Load original DNA
        original_file = "data/examples/original.mid"
        print(f"Extracting DNA from original: {original_file}")
        original_dna = pipeline.extract_dna(original_file)

        # Create variations
        variations = []

        # Variation 1: Complex Harmony
        print("\nCreating Variation 1: Complex Harmony")
        complex_harmony = original_dna.copy()
        complex_harmony['harmony.chord_complexity'] = min(1.0, original_dna.get('harmony.chord_complexity', 0.5) + 0.3)
        complex_harmony['harmony.extended_harmonies'] = min(1.0, original_dna.get('harmony.extended_harmonies', 0.5) + 0.4)
        complex_harmony['harmony.voice_leading_smoothness'] = 0.8
        variations.append(('complex_harmony', complex_harmony))
        print("  ✓ Increased chord complexity and extended harmonies")

        # Variation 2: Syncopated Rhythm
        print("\nCreating Variation 2: Syncopated Rhythm")
        syncopated = original_dna.copy()
        syncopated['rhythm.syncopation_intensity'] = 0.9
        syncopated['rhythm.groove_pocket_tightness'] = 0.7
        syncopated['rhythm.swing_ratio'] = 0.6
        variations.append(('syncopated_rhythm', syncopated))
        print("  ✓ Increased syncopation and groove")

        # Variation 3: Sparse Texture
        print("\nCreating Variation 3: Sparse Texture")
        sparse = original_dna.copy()
        sparse['orchestration.instrumentation_density'] = 0.3
        sparse['texture.homophonic_vs_polyphonic'] = 0.2
        sparse['texture.voice_independence_score'] = 0.3
        variations.append(('sparse_texture', sparse))
        print("  ✓ Reduced instrumentation and texture density")

        # Variation 4: Dramatic Arc
        print("\nCreating Variation 4: Dramatic Arc")
        dramatic = original_dna.copy()
        dramatic['form.tension_arc_shape'] = 0.9
        dramatic['form.section_contrast_degree'] = 0.8
        dramatic['form.climax_position_ratio'] = 0.7
        dramatic['cross.climax_convergence_factor'] = 0.9
        variations.append(('dramatic_arc', dramatic))
        print("  ✓ Enhanced dramatic structure and contrast")

        # Generate all variations
        print("\nGenerating variations...")
        for name, dna in variations:
            output_file = f"output/variations/{name}.mid"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            midi = pipeline.generate_from_dna(dna)
            midi.save(output_file)
            print(f"  ✓ Generated: {output_file}")

        print(f"\n✓ Created {len(variations)} variations")

    except ImportError as e:
        print(f"⚠️  Required modules not available: {e}")

    print()


# =============================================================================
# EXAMPLE 3: Cross-Corpus Style Transfer
# =============================================================================

def example_3_style_transfer():
    """
    Combine parameters from different musical styles.
    """
    print("=" * 80)
    print("EXAMPLE 3: Cross-Corpus Style Transfer")
    print("=" * 80)

    try:
        from midi_generator.learning.modular_discovery_pipeline import ModularSemanticDiscoveryPipeline

        pipeline = ModularSemanticDiscoveryPipeline.load_pretrained()

        # Source files
        sources = {
            'jazz': "data/examples/jazz_standard.mid",
            'funk': "data/examples/funk_groove.mid",
            'classical': "data/examples/classical_piece.mid"
        }

        # Extract DNA from all sources
        print("Extracting DNA from source files...")
        source_dnas = {}
        for genre, file in sources.items():
            print(f"  {genre}: {file}")
            source_dnas[genre] = pipeline.extract_dna(file)

        # Create fusion combinations
        fusions = []

        # Fusion 1: Jazz Harmony + Funk Rhythm
        print("\nFusion 1: Jazz Harmony + Funk Rhythm")
        jazz_funk = {}
        for key in source_dnas['jazz']:
            if key.startswith('harmony.') or key.startswith('form.'):
                jazz_funk[key] = source_dnas['jazz'][key]
            elif key.startswith('rhythm.'):
                jazz_funk[key] = source_dnas['funk'][key]
            else:
                # Average for other dimensions
                jazz_funk[key] = (source_dnas['jazz'][key] + source_dnas['funk'][key]) / 2
        fusions.append(('jazz_funk_fusion', jazz_funk))
        print("  ✓ Combined jazz harmony with funk rhythm")

        # Fusion 2: Classical Form + Jazz Harmony
        print("\nFusion 2: Classical Form + Jazz Harmony")
        classical_jazz = {}
        for key in source_dnas['classical']:
            if key.startswith('form.') or key.startswith('orchestration.'):
                classical_jazz[key] = source_dnas['classical'][key]
            elif key.startswith('harmony.'):
                classical_jazz[key] = source_dnas['jazz'][key]
            else:
                classical_jazz[key] = (source_dnas['classical'][key] + source_dnas['jazz'][key]) / 2
        fusions.append(('classical_jazz_fusion', classical_jazz))
        print("  ✓ Combined classical form with jazz harmony")

        # Fusion 3: Weighted blend (60% jazz, 40% funk)
        print("\nFusion 3: Weighted Blend (60% jazz, 40% funk)")
        weighted_blend = {}
        for key in source_dnas['jazz']:
            weighted_blend[key] = (
                0.6 * source_dnas['jazz'][key] +
                0.4 * source_dnas['funk'][key]
            )
        fusions.append(('weighted_jazz_funk', weighted_blend))
        print("  ✓ Created 60/40 weighted blend")

        # Generate fusions
        print("\nGenerating fusion tracks...")
        for name, dna in fusions:
            output_file = f"output/fusions/{name}.mid"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            midi = pipeline.generate_from_dna(dna)
            midi.save(output_file)
            print(f"  ✓ Generated: {output_file}")

        print(f"\n✓ Created {len(fusions)} fusion tracks")

    except ImportError as e:
        print(f"⚠️  Required modules not available: {e}")

    print()


# =============================================================================
# EXAMPLE 4: Batch Processing and Analysis
# =============================================================================

def example_4_batch_analysis():
    """
    Process entire corpus and analyze parameter distributions.
    """
    print("=" * 80)
    print("EXAMPLE 4: Batch Processing and Analysis")
    print("=" * 80)

    try:
        from midi_generator.learning.modular_discovery_pipeline import ModularSemanticDiscoveryPipeline

        pipeline = ModularSemanticDiscoveryPipeline.load_pretrained()

        # Process corpus
        corpus_dir = Path("data/midi_corpus")
        print(f"Processing corpus: {corpus_dir}")

        dna_database = {}
        midi_files = list(corpus_dir.rglob("*.mid"))[:50]  # Limit to 50 for demo

        print(f"Found {len(midi_files)} MIDI files\n")

        for i, midi_file in enumerate(midi_files, 1):
            try:
                dna = pipeline.extract_dna(str(midi_file))
                dna_database[midi_file.name] = dna

                if i % 10 == 0:
                    print(f"  Processed {i}/{len(midi_files)} files...")

            except Exception as e:
                print(f"  ✗ Failed: {midi_file.name} - {e}")
                continue

        print(f"\n✓ Processed {len(dna_database)} files")

        # Save database
        output_path = "output/analysis/dna_database.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dna_database, f, indent=2)
        print(f"✓ Database saved to: {output_path}")

        # Analyze distributions
        print("\n" + "=" * 80)
        print("PARAMETER ANALYSIS")
        print("=" * 80)

        if dna_database:
            param_names = list(next(iter(dna_database.values())).keys())

            # Compute statistics
            param_stats = {}
            for param in param_names:
                values = [dna[param] for dna in dna_database.values()]
                param_stats[param] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

            # Show top varying parameters
            print("\nMost varying parameters:")
            sorted_params = sorted(param_stats.items(), key=lambda x: x[1]['std'], reverse=True)
            for param, stats in sorted_params[:10]:
                print(f"  {param:40s} μ={stats['mean']:.3f} σ={stats['std']:.3f}")

            # Find songs with extreme values
            print("\n" + "-" * 80)
            print("Songs with high syncopation:")
            high_syncopation = [
                (name, dna.get('rhythm.syncopation_intensity', 0))
                for name, dna in dna_database.items()
                if dna.get('rhythm.syncopation_intensity', 0) > 0.8
            ]
            high_syncopation.sort(key=lambda x: x[1], reverse=True)
            for name, value in high_syncopation[:5]:
                print(f"  {name:50s} {value:.3f}")

            print("\nSongs with complex harmony:")
            complex_harmony = [
                (name, dna.get('harmony.chord_complexity', 0))
                for name, dna in dna_database.items()
                if dna.get('harmony.chord_complexity', 0) > 0.7
            ]
            complex_harmony.sort(key=lambda x: x[1], reverse=True)
            for name, value in complex_harmony[:5]:
                print(f"  {name:50s} {value:.3f}")

            # Save analysis report
            analysis_path = "output/analysis/parameter_statistics.json"
            with open(analysis_path, 'w') as f:
                json.dump(param_stats, f, indent=2)
            print(f"\n✓ Analysis saved to: {analysis_path}")

    except ImportError as e:
        print(f"⚠️  Required modules not available: {e}")

    print()


# =============================================================================
# EXAMPLE 5: Training Custom Module
# =============================================================================

def example_5_train_custom_module():
    """
    Train a custom semantic encoder module.
    """
    print("=" * 80)
    print("EXAMPLE 5: Training Custom Module")
    print("=" * 80)

    try:
        from midi_generator.learning import PipelineConfig, ModularSemanticDiscoveryPipeline
        from pathlib import Path

        # Configure training
        print("Configuring training pipeline...")
        config = PipelineConfig(
            midi_corpus_dir=Path("data/midi_corpus"),
            output_dir=Path("output/training"),
            num_semantic_features=120,
            max_epochs=10,  # Reduced for demo
            batch_size=32,
            device='cpu',  # Use 'cuda' for GPU
            verbose=True,
            max_files=100  # Limit corpus size for demo
        )

        print(f"\nConfiguration:")
        print(f"  Corpus: {config.midi_corpus_dir}")
        print(f"  Output: {config.output_dir}")
        print(f"  Features: {config.num_semantic_features}")
        print(f"  Epochs: {config.max_epochs}")
        print(f"  Device: {config.device}")

        # Initialize pipeline
        print("\nInitializing pipeline...")
        pipeline = ModularSemanticDiscoveryPipeline(config)

        # Train (this would take hours in production)
        print("\n" + "=" * 80)
        print("TRAINING (Demo - abbreviated)")
        print("=" * 80)
        print("\n⚠️  Full training takes 18-28 hours with GPU parallelization")
        print("This demo shows the training interface only.\n")

        # Simulate training phases
        phases = [
            ("Phase 1: Architecture Audit", "2-4 hours"),
            ("Phase 2: Module Training (5 parallel)", "8-12 hours"),
            ("Phase 3: Cross-Dimensional Training", "4-6 hours"),
            ("Phase 4: Integration & Validation", "4-6 hours")
        ]

        for phase, duration in phases:
            print(f"{phase:50s} Est: {duration}")

        print("\nTo run full training:")
        print("  python -m midi_generator.learning.train_modular_pipeline \\")
        print("    --corpus data/midi_corpus \\")
        print("    --output output/training \\")
        print("    --features 120 \\")
        print("    --epochs 100 \\")
        print("    --device cuda")

    except ImportError as e:
        print(f"⚠️  Required modules not available: {e}")

    print()


# =============================================================================
# EXAMPLE 6: Interactive DNA Editor
# =============================================================================

def example_6_interactive_editor():
    """
    Demonstrate interactive DNA editing interface.
    """
    print("=" * 80)
    print("EXAMPLE 6: Interactive DNA Editor")
    print("=" * 80)

    print("\nInteractive DNA Editor Interface (Demo)")
    print("-" * 80)

    # Simulate DNA
    sample_dna = {
        'harmony.chord_complexity': 0.73,
        'harmony.tension_resolution_rate': 0.61,
        'harmony.voice_leading_smoothness': 0.89,
        'rhythm.syncopation_intensity': 0.45,
        'rhythm.groove_pocket_tightness': 0.78,
        'rhythm.swing_ratio': 0.52,
        'form.section_contrast_degree': 0.66,
        'orchestration.instrumentation_density': 0.71,
        'texture.homophonic_vs_polyphonic': 0.55,
    }

    print("\nCurrent DNA (showing 9 of 120 parameters):")
    for param, value in sample_dna.items():
        print(f"  {param:40s} [{value:.2f}] {'█' * int(value * 30)}")

    print("\n" + "-" * 80)
    print("Interactive Commands:")
    print("  view [dimension]     - View parameters for dimension")
    print("  edit [param] [value] - Edit parameter value")
    print("  preview             - Generate and play preview")
    print("  save [file]         - Save edited MIDI")
    print("  undo                - Undo last edit")
    print("  diff                - Compare with original")
    print("  quit                - Exit editor")

    print("\nExample session:")
    print("> edit harmony.chord_complexity 0.90")
    print("✓ Updated: harmony.chord_complexity = 0.900")
    print("\n> preview")
    print("▶ Generating preview... Done!")
    print("♪ Playing preview (5 seconds)")
    print("\n> save edited_song.mid")
    print("✓ Saved: edited_song.mid")

    print("\nTo launch interactive editor:")
    print("  python -m midi_generator.tools.dna_editor song.mid")

    print()


# =============================================================================
# EXAMPLE 7: API Usage
# =============================================================================

def example_7_api_usage():
    """
    Demonstrate REST API usage for production deployment.
    """
    print("=" * 80)
    print("EXAMPLE 7: API Usage")
    print("=" * 80)

    print("\nREST API Examples (requires running server)")
    print("-" * 80)

    print("\n1. Start API server:")
    print("   $ uvicorn midi_generator.api.server:app --host 0.0.0.0 --port 8000")

    print("\n2. Extract DNA from MIDI:")
    print("""
   import requests

   with open("song.mid", "rb") as f:
       response = requests.post(
           "http://localhost:8000/extract_dna",
           files={"file": f}
       )
       dna = response.json()["dna"]
       print(f"Extracted {len(dna)} parameters")
    """)

    print("\n3. Generate MIDI from DNA:")
    print("""
   response = requests.post(
       "http://localhost:8000/generate",
       json={"dna": dna}
   )

   with open("generated.mid", "wb") as f:
       f.write(response.content)
    """)

    print("\n4. Edit and regenerate:")
    print("""
   edits = {
       "harmony.chord_complexity": 0.9,
       "rhythm.syncopation_intensity": 0.3
   }

   with open("song.mid", "rb") as f:
       response = requests.post(
           "http://localhost:8000/edit",
           files={"file": f},
           json={"edits": edits}
       )

       with open("edited.mid", "wb") as f_out:
           f_out.write(response.content)
    """)

    print("\n5. API Documentation:")
    print("   Visit: http://localhost:8000/docs")

    print()


# =============================================================================
# Main Runner
# =============================================================================

def main():
    """
    Run all examples.
    """
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  MODULAR SEMANTIC DISCOVERY - COMPREHENSIVE EXAMPLES".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Agent 10: Documentation & Deployment Manager".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    examples = [
        ("Basic DNA Extraction", example_1_basic_extraction),
        ("Parameter-Based Editing", example_2_parameter_editing),
        ("Cross-Corpus Style Transfer", example_3_style_transfer),
        ("Batch Processing and Analysis", example_4_batch_analysis),
        ("Training Custom Module", example_5_train_custom_module),
        ("Interactive DNA Editor", example_6_interactive_editor),
        ("API Usage", example_7_api_usage),
    ]

    print("Available Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nOptions:")
    print("  - Enter number to run specific example")
    print("  - Enter 'all' to run all examples")
    print("  - Enter 'quit' to exit")

    # For automated demo, run all
    print("\n" + "=" * 80)
    print("Running all examples in demo mode...")
    print("=" * 80 + "\n")

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"⚠️  Example failed: {e}\n")

    print("=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nFor more information, see:")
    print("  - Documentation: midi_generator/docs/MODULAR_SEMANTIC_DISCOVERY.md")
    print("  - Training guide: midi_generator/docs/TRAINING_GUIDE.md")
    print("  - API reference: http://localhost:8000/docs (when server running)")
    print()


if __name__ == "__main__":
    main()
