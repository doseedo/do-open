#!/usr/bin/env python3
"""
Example: Run Semantic Discovery Pipeline
=========================================

This script demonstrates how to use the SemanticDiscoveryPipeline to automatically
discover musical parameters from a MIDI corpus.

Usage:
    python run_semantic_discovery.py --corpus data/midi --output output/discovery

Features:
- Automatic parameter discovery from MIDI files
- Configurable number of parameters (default: 25)
- GPU acceleration support
- Resumable from checkpoints

Author: Agent 7 - Integration Pipeline
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from midi_generator.learning.semantic_discovery_pipeline import (
    SemanticDiscoveryPipeline,
    PipelineConfig,
    create_default_config
)


def run_basic_discovery():
    """
    Example 1: Basic discovery with default settings

    Discovers ~25 parameters from MIDI corpus.
    """
    print("=" * 80)
    print("EXAMPLE 1: BASIC SEMANTIC DISCOVERY")
    print("=" * 80)

    # Create configuration
    config = create_default_config(
        midi_corpus_dir="data/midi",
        output_dir="output/basic_discovery",
        num_semantic_features=25
    )

    # Run pipeline
    pipeline = SemanticDiscoveryPipeline(config)
    results = pipeline.run()

    # Display results
    print("\nDiscovered Parameters:")
    for i, param_name in enumerate(results.discovered_parameters, 1):
        print(f"  {i}. {param_name}")

    print(f"\nReconstruction improvement: {results.reconstruction_improvement:.1%}")
    print(f"Interpretability score: {results.interpretability_score:.1%}")

    return results


def run_custom_discovery():
    """
    Example 2: Custom configuration

    Shows how to customize discovery settings.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: CUSTOM CONFIGURATION")
    print("=" * 80)

    # Create custom configuration
    config = PipelineConfig(
        # Paths
        midi_corpus_dir=Path("data/midi/jazz"),
        output_dir=Path("output/jazz_discovery"),
        cache_dir=Path("cache/jazz"),

        # Corpus settings
        max_files=500,  # Limit to 500 files for faster experimentation
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,

        # Discovery settings
        num_semantic_features=30,  # Discover up to 30 parameters
        hidden_dim=512,

        # Training settings
        max_epochs=50,  # Fewer epochs for faster iteration
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=5,

        # Sparsity
        sparsity_weight=0.02,
        target_sparsity=0.1,

        # Locality
        locality_weight=0.1,
        locality_transformations=['transpose', 'invert', 'augment'],

        # Validation
        interpretation_threshold=0.7,  # Higher threshold = more confident interpretations
        redundancy_threshold=0.85,     # Lower = allow more similar parameters

        # Computational
        device="cuda",
        num_workers=4,
        use_mixed_precision=True,

        # Checkpointing
        checkpoint_frequency=5,

        # Logging
        verbose=True,
        log_frequency=10
    )

    # Run pipeline
    pipeline = SemanticDiscoveryPipeline(config)
    results = pipeline.run()

    return results


def run_resume_from_checkpoint():
    """
    Example 3: Resume from checkpoint

    Shows how to resume discovery after interruption.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: RESUME FROM CHECKPOINT")
    print("=" * 80)

    config = create_default_config(
        midi_corpus_dir="data/midi",
        output_dir="output/resumable_discovery",
        resume_from_checkpoint="output/resumable_discovery/checkpoint.json"
    )

    pipeline = SemanticDiscoveryPipeline(config)
    results = pipeline.run()

    return results


def run_genre_specific_discovery():
    """
    Example 4: Genre-specific parameter discovery

    Discover parameters for a specific genre.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: GENRE-SPECIFIC DISCOVERY")
    print("=" * 80)

    genres = ['jazz', 'classical', 'rock', 'electronic']

    all_results = {}

    for genre in genres:
        print(f"\n🎵 Discovering {genre.upper()} parameters...")

        config = create_default_config(
            midi_corpus_dir=f"data/midi/{genre}",
            output_dir=f"output/{genre}_discovery",
            num_semantic_features=20,  # 20 parameters per genre
            max_files=200  # Smaller corpus per genre
        )

        pipeline = SemanticDiscoveryPipeline(config)
        results = pipeline.run()

        all_results[genre] = results

        print(f"   ✅ Discovered {len(results.discovered_parameters)} {genre} parameters")

    # Compare across genres
    print("\n" + "=" * 80)
    print("CROSS-GENRE COMPARISON")
    print("=" * 80)

    for genre, results in all_results.items():
        print(f"\n{genre.upper()}:")
        print(f"  Parameters: {len(results.discovered_parameters)}")
        print(f"  Reconstruction: {results.reconstruction_improvement:.1%}")
        print(f"  Interpretability: {results.interpretability_score:.1%}")

        # Show top 5 parameters
        print(f"  Top parameters:")
        for param in results.discovered_parameters[:5]:
            print(f"    - {param}")

    return all_results


def run_quick_test():
    """
    Example 5: Quick test on small corpus

    For testing the pipeline with minimal computation.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: QUICK TEST")
    print("=" * 80)

    config = PipelineConfig(
        midi_corpus_dir=Path("data/midi"),
        output_dir=Path("output/quick_test"),

        # Minimal settings for fast execution
        max_files=10,           # Only 10 files
        num_semantic_features=5, # Only 5 features
        max_epochs=10,          # Only 10 epochs
        batch_size=8,

        verbose=True
    )

    pipeline = SemanticDiscoveryPipeline(config)
    results = pipeline.run()

    print(f"\n✅ Quick test completed!")
    print(f"   Discovered {len(results.discovered_parameters)} parameters")

    return results


def extract_parameters_from_new_midi(results, midi_path: str):
    """
    Example 6: Use discovered parameters on new MIDI file

    Shows how to apply discovered parameters to extract features from new songs.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: EXTRACT FROM NEW MIDI")
    print("=" * 80)

    print(f"\nExtracting parameters from: {midi_path}")

    # Use extraction functions from discovery results
    extracted_params = {}

    for param_name in results.discovered_parameters:
        extractor = results.extraction_functions.get(param_name)

        if extractor:
            try:
                value = extractor(midi_path)
                extracted_params[param_name] = value
                print(f"  {param_name}: {value:.3f}")
            except Exception as e:
                print(f"  {param_name}: ERROR - {e}")
        else:
            print(f"  {param_name}: No extraction function available")

    return extracted_params


def compare_with_existing_parameters(results):
    """
    Example 7: Compare discovered vs existing parameters

    Shows overlap and novelty of discovered parameters.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: COMPARISON WITH EXISTING PARAMETERS")
    print("=" * 80)

    try:
        from midi_generator.parameters.universal_registry import UniversalParameterRegistry

        registry = UniversalParameterRegistry()
        existing_params = set(registry.get_all_parameters())
        discovered_params = set(results.discovered_parameters)

        overlap = existing_params.intersection(discovered_params)
        novel = discovered_params - existing_params

        print(f"\nExisting parameters in registry: {len(existing_params)}")
        print(f"Discovered parameters: {len(discovered_params)}")
        print(f"Overlap: {len(overlap)}")
        print(f"Novel parameters: {len(novel)}")

        if novel:
            print(f"\n🎉 Novel parameters discovered:")
            for param in list(novel)[:10]:  # Show first 10
                print(f"  - {param}")

        if overlap:
            print(f"\n🔄 Rediscovered existing parameters:")
            for param in list(overlap)[:5]:  # Show first 5
                print(f"  - {param}")

    except ImportError:
        print("⚠️  UniversalParameterRegistry not available")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run semantic discovery pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic discovery
  python run_semantic_discovery.py --corpus data/midi --output output/discovery

  # Custom number of features
  python run_semantic_discovery.py --corpus data/midi --output output/discovery --features 30

  # Quick test
  python run_semantic_discovery.py --corpus data/midi --output output/test --max-files 10 --features 5 --epochs 10

  # Genre-specific
  python run_semantic_discovery.py --corpus data/midi/jazz --output output/jazz --features 20

  # Resume from checkpoint
  python run_semantic_discovery.py --corpus data/midi --output output/discovery --resume
        """
    )

    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to MIDI corpus directory"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results"
    )

    parser.add_argument(
        "--features",
        type=int,
        default=25,
        help="Number of semantic features to discover (default: 25)"
    )

    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of MIDI files to process"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for training (default: cuda)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )

    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific example (1-5)"
    )

    args = parser.parse_args()

    # Run specific example if requested
    if args.example:
        if args.example == 1:
            results = run_basic_discovery()
        elif args.example == 2:
            results = run_custom_discovery()
        elif args.example == 3:
            results = run_resume_from_checkpoint()
        elif args.example == 4:
            results = run_genre_specific_discovery()
        elif args.example == 5:
            results = run_quick_test()
        return

    # Run with command-line arguments
    print("=" * 80)
    print("SEMANTIC PARAMETER DISCOVERY")
    print("=" * 80)
    print(f"\nCorpus: {args.corpus}")
    print(f"Output: {args.output}")
    print(f"Features: {args.features}")
    print(f"Max files: {args.max_files or 'all'}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Create configuration
    config = PipelineConfig(
        midi_corpus_dir=Path(args.corpus),
        output_dir=Path(args.output),
        num_semantic_features=args.features,
        max_files=args.max_files,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        resume_from_checkpoint=f"{args.output}/checkpoint.json" if args.resume else None,
        verbose=True
    )

    # Run pipeline
    pipeline = SemanticDiscoveryPipeline(config)
    results = pipeline.run()

    # Display summary
    print("\n" + "=" * 80)
    print("DISCOVERY SUMMARY")
    print("=" * 80)
    print(f"\n✅ Discovered {len(results.discovered_parameters)} parameters")
    print(f"📊 Reconstruction improvement: {results.reconstruction_improvement:.1%}")
    print(f"🎵 Interpretability score: {results.interpretability_score:.1%}")
    print(f"✓ Musical validity score: {results.musical_validity_score:.1%}")
    print(f"💾 Results saved to: {args.output}")

    print("\nDiscovered parameters:")
    for i, param in enumerate(results.discovered_parameters, 1):
        print(f"  {i:2d}. {param}")

    print("\n" + "=" * 80)
    print("To use these parameters, see the example in the script:")
    print(f"  python -c 'import run_semantic_discovery; "
          f"run_semantic_discovery.extract_parameters_from_new_midi(results, \"song.mid\")'")
    print("=" * 80)


if __name__ == "__main__":
    main()
