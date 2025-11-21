#!/usr/bin/env python3
"""
Example 6: Cross-Corpus Validation

Validate that discovered features generalize across different musical corpora.

Usage:
    python 06_cross_corpus_validation.py

Author: Agent 10 - Documentation & Examples
Date: November 2025
"""

from pathlib import Path
import numpy as np
from midi_generator.learning.semantic_encoder import SemanticFeatureEncoder
from midi_generator.evaluation.semantic_evaluation import SemanticFeatureEvaluator
from midi_generator.learning.gap_dataset import GapDataset


def validate_on_corpus(encoder, corpus_dir: Path, corpus_name: str):
    """Validate encoder on a specific corpus"""
    print(f"\nValidating on {corpus_name}...")
    print("-" * 60)

    # Get MIDI files
    midi_files = list(corpus_dir.glob("*.mid"))
    print(f"  Files: {len(midi_files)}")

    if len(midi_files) == 0:
        print(f"  WARNING: No MIDI files found in {corpus_dir}")
        return None

    # Create dataset
    dataset = GapDataset(
        midi_files,
        cache_dir=Path(f"cache/validation_{corpus_name}"),
        regeneration_method="approximate"
    )

    # Evaluate reconstruction
    evaluator = SemanticFeatureEvaluator(encoder, feature_bank=None)
    metrics = evaluator.evaluate_reconstruction(dataset)

    print(f"  Results:")
    print(f"    R² score:             {metrics['r2_score']:.3f}")
    print(f"    MSE:                  {metrics['mse']:.4f}")
    print(f"    MAE:                  {metrics['mae']:.4f}")
    print(f"    Reconstruction rate:  {metrics['reconstruction_rate']:.1%}")

    return {
        'corpus_name': corpus_name,
        'num_files': len(midi_files),
        **metrics
    }


def compare_corpora(results: list):
    """Compare results across different corpora"""
    print("\n" + "="*60)
    print("Cross-Corpus Comparison")
    print("="*60)

    print(f"\n{'Corpus':<20s} {'Files':>8s} {'R²':>8s} {'MSE':>8s} {'MAE':>8s}")
    print("-" * 60)

    for result in results:
        if result:
            print(f"{result['corpus_name']:<20s} "
                  f"{result['num_files']:>8d} "
                  f"{result['r2_score']:>8.3f} "
                  f"{result['mse']:>8.4f} "
                  f"{result['mae']:>8.4f}")

    # Compute consistency
    r2_scores = [r['r2_score'] for r in results if r]
    if len(r2_scores) > 1:
        print()
        print("Consistency Metrics:")
        print(f"  R² mean:  {np.mean(r2_scores):.3f}")
        print(f"  R² std:   {np.std(r2_scores):.3f}")

        if np.std(r2_scores) < 0.05:
            print("  ✓ Good generalization (low variance)")
        else:
            print("  ⚠ High variance (may be overfitting to training corpus)")


def main():
    """Main function"""
    print("="*60)
    print("EXAMPLE 6: Cross-Corpus Validation")
    print("="*60)

    # Load encoder
    encoder_path = Path("output/basic_discovery/encoder.pt")

    if not encoder_path.exists():
        print(f"\nERROR: Encoder not found at {encoder_path}")
        print("Please run 01_basic_discovery.py first")
        return

    print(f"\nLoading encoder...")
    import torch
    encoder = SemanticFeatureEncoder()
    checkpoint = torch.load(encoder_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    print("Encoder loaded")

    # Define corpora to test
    corpora = [
        (Path("data/midi/train"), "Training Corpus"),
        (Path("data/midi/validation"), "Validation Corpus"),
        (Path("data/midi/test"), "Test Corpus (held-out)"),
        (Path("data/midi/classical"), "Classical Music"),
        (Path("data/midi/jazz"), "Jazz"),
        (Path("data/midi/pop"), "Pop Music"),
    ]

    # Validate on each corpus
    results = []
    for corpus_dir, corpus_name in corpora:
        if corpus_dir.exists():
            result = validate_on_corpus(encoder, corpus_dir, corpus_name)
            if result:
                results.append(result)
        else:
            print(f"\nCorpus not found: {corpus_dir} (skipping)")

    # Compare
    if results:
        compare_corpora(results)
    else:
        print("\nNo corpora found for validation")

    print("\n" + "="*60)
    print("Cross-corpus validation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
