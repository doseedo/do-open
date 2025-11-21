#!/usr/bin/env python3
"""
Example 1: Basic Semantic Feature Discovery

This example demonstrates the simplest way to run semantic feature discovery
on a MIDI corpus and examine the results.

Usage:
    python 01_basic_discovery.py

Expected output:
    - Discovered features saved to output/basic_discovery/
    - Console output showing progress and results
    - HTML report with detailed analysis

Author: Agent 10 - Documentation & Examples
Date: November 2025
"""

from pathlib import Path
from midi_generator.learning.semantic_discovery_pipeline import SemanticDiscoveryPipeline


def main():
    """Run basic semantic feature discovery"""

    print("="*60)
    print("EXAMPLE 1: Basic Semantic Feature Discovery")
    print("="*60)
    print()

    # Configuration
    midi_corpus_dir = Path("data/midi/train")
    output_dir = Path("output/basic_discovery")
    num_features = 25

    # Check if corpus exists
    if not midi_corpus_dir.exists():
        print(f"ERROR: Corpus directory not found: {midi_corpus_dir}")
        print()
        print("Please create the directory and add MIDI files:")
        print(f"  mkdir -p {midi_corpus_dir}")
        print(f"  # Copy MIDI files to {midi_corpus_dir}/")
        print()
        print("Recommended: 500-1000 MIDI files for best results")
        return

    # Count files
    midi_files = list(midi_corpus_dir.glob("*.mid"))
    print(f"Found {len(midi_files)} MIDI files in {midi_corpus_dir}")

    if len(midi_files) < 100:
        print()
        print(f"WARNING: Only {len(midi_files)} files found.")
        print("Recommended minimum: 500 files for quality results.")
        print("Continue anyway? (y/n): ", end="")
        response = input().strip().lower()
        if response != 'y':
            print("Aborted.")
            return

    print()
    print(f"Configuration:")
    print(f"  Corpus directory: {midi_corpus_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Number of features to discover: {num_features}")
    print()

    # Create pipeline
    print("Creating discovery pipeline...")
    pipeline = SemanticDiscoveryPipeline(
        midi_corpus_dir=midi_corpus_dir,
        output_dir=output_dir,
        num_features=num_features
    )
    print("Pipeline created successfully!")
    print()

    # Run discovery
    print("Starting discovery process...")
    print("This will take 4-8 hours on GPU, longer on CPU.")
    print()

    results = pipeline.run()

    print()
    print("="*60)
    print("DISCOVERY COMPLETE!")
    print("="*60)
    print()

    # Display results
    print("Results:")
    print(f"  Features discovered: {len(results['features'])}")
    print(f"  Successfully interpreted: {results['auto_interpreted']}/{len(results['features'])}")
    print(f"  Reconstruction quality: {results['reconstruction_score']:.2%}")
    print(f"  Interpretability score: {results['interpretability_score']:.2%}")
    print()

    # List discovered features
    print("Discovered Features:")
    print("-" * 60)
    for i, feature in enumerate(results['features'], 1):
        confidence_str = f"{feature.confidence:.2f}" if feature.confidence else "N/A"
        print(f"{i:2d}. {feature.name:30s} (confidence: {confidence_str})")
        if feature.interpretation:
            print(f"     {feature.interpretation}")
    print()

    # Group by modality
    from collections import defaultdict
    by_modality = defaultdict(list)
    for feature in results['features']:
        by_modality[feature.modality].append(feature)

    print("Features by Modality:")
    print("-" * 60)
    for modality, features in sorted(by_modality.items()):
        print(f"  {modality.value.upper()}: {len(features)} features")
        for feature in features[:3]:  # Show first 3
            print(f"    - {feature.name}")
        if len(features) > 3:
            print(f"    ... and {len(features) - 3} more")
    print()

    # Save results
    results_file = output_dir / "results.pkl"
    pipeline.save_results(results, results_file)
    print(f"Results saved to: {results_file}")
    print()

    # Report
    if 'report_path' in results:
        print(f"Detailed report: {results['report_path']}")
        print(f"Open in browser: file://{results['report_path'].absolute()}")
    print()

    print("="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Review the HTML report for detailed analysis")
    print("2. Run 02_custom_config.py to customize training parameters")
    print("3. Run 04_parameter_extraction.py to use discovered parameters")
    print()


if __name__ == "__main__":
    main()
