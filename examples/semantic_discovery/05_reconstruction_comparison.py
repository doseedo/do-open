#!/usr/bin/env python3
"""
Example 5: Reconstruction Comparison

Compare reconstruction quality before and after semantic feature discovery.

Usage:
    python 05_reconstruction_comparison.py

Author: Agent 10 - Documentation & Examples
Date: November 2025
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from midi_generator.parameters.optimized_feature_extractor import OptimizedFeatureExtractor
from midi_generator.parameters.hierarchical_parameter_extractor_v2 import HierarchicalParameterExtractorV2
from midi_generator.learning.semantic_encoder import SemanticFeatureEncoder
from midi_generator.learning.gap_dataset import GapAnalyzer


def compare_single_file(midi_file: Path, encoder: SemanticFeatureEncoder):
    """Compare reconstruction for a single MIDI file"""
    print(f"\nAnalyzing: {midi_file.name}")
    print("-" * 60)

    # Extract original features
    feature_extractor = OptimizedFeatureExtractor()
    original_features = feature_extractor.extract(midi_file)

    # Extract with existing parameters only (baseline)
    param_extractor = HierarchicalParameterExtractorV2()
    parameters = param_extractor.extract(midi_file)

    # Regenerate and extract features (baseline reconstruction)
    # This would use existing generation pipeline
    # For now, we'll simulate with approximate method
    from midi_generator.learning.gap_dataset import GapDataset

    # Compute gap
    dataset = GapDataset([midi_file], regeneration_method="approximate")
    gap_data = dataset[0]

    baseline_gap = np.linalg.norm(gap_data['gap'].numpy())
    print(f"  Baseline gap (50 params): {baseline_gap:.4f}")

    # Extract semantic features
    import torch
    x = torch.tensor(original_features, dtype=torch.float32).unsqueeze(0)

    encoder.eval()
    with torch.no_grad():
        outputs = encoder(x)
        semantic_features = outputs['semantic_features']
        reconstructed = outputs['reconstructed']

    # Compute reconstruction error with semantic features
    recon_gap = torch.nn.functional.mse_loss(reconstructed, x).item()
    print(f"  With semantic features:   {recon_gap:.4f}")

    # Improvement
    improvement = (baseline_gap - recon_gap) / baseline_gap * 100
    print(f"  Improvement:              {improvement:.1f}%")

    return {
        'filename': midi_file.name,
        'baseline_gap': baseline_gap,
        'semantic_gap': recon_gap,
        'improvement_pct': improvement
    }


def visualize_comparison(results: list, output_dir: Path):
    """Visualize reconstruction comparison"""
    print("\nCreating visualizations...")

    # Extract data
    filenames = [r['filename'] for r in results]
    baseline_gaps = [r['baseline_gap'] for r in results]
    semantic_gaps = [r['semantic_gap'] for r in results]

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar chart
    x = np.arange(len(filenames))
    width = 0.35

    ax1.bar(x - width/2, baseline_gaps, width, label='Baseline (50 params)', alpha=0.8)
    ax1.bar(x + width/2, semantic_gaps, width, label='With Semantic Features', alpha=0.8)
    ax1.set_xlabel('MIDI Files')
    ax1.set_ylabel('Reconstruction Gap (MSE)')
    ax1.set_title('Reconstruction Quality Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f[:10] for f in filenames], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Improvement histogram
    improvements = [r['improvement_pct'] for r in results]
    ax2.hist(improvements, bins=20, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(improvements), color='red', linestyle='--',
                label=f'Mean: {np.mean(improvements):.1f}%')
    ax2.set_xlabel('Improvement (%)')
    ax2.set_ylabel('Count')
    ax2.set_title('Reconstruction Improvement Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "reconstruction_comparison.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main function"""
    print("="*60)
    print("EXAMPLE 5: Reconstruction Comparison")
    print("="*60)

    # Load trained encoder
    encoder_path = Path("output/basic_discovery/encoder.pt")

    if not encoder_path.exists():
        print(f"\nERROR: Trained encoder not found at {encoder_path}")
        print("Please run 01_basic_discovery.py first")
        return

    print(f"\nLoading encoder from {encoder_path}...")
    import torch
    encoder = SemanticFeatureEncoder()
    checkpoint = torch.load(encoder_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    print("Encoder loaded successfully")

    # Test files
    test_dir = Path("data/midi/test")
    if not test_dir.exists():
        print(f"\nERROR: Test directory not found: {test_dir}")
        return

    test_files = list(test_dir.glob("*.mid"))[:20]  # Test on 20 files
    print(f"\nTesting on {len(test_files)} files...")

    # Compare each file
    results = []
    for midi_file in test_files:
        try:
            result = compare_single_file(midi_file, encoder)
            results.append(result)
        except Exception as e:
            print(f"  ERROR processing {midi_file.name}: {e}")

    # Summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)

    baseline_mean = np.mean([r['baseline_gap'] for r in results])
    semantic_mean = np.mean([r['semantic_gap'] for r in results])
    improvement_mean = np.mean([r['improvement_pct'] for r in results])

    print(f"  Baseline gap (mean):       {baseline_mean:.4f}")
    print(f"  With semantic (mean):      {semantic_mean:.4f}")
    print(f"  Average improvement:       {improvement_mean:.1f}%")
    print()

    files_improved = sum(1 for r in results if r['improvement_pct'] > 0)
    print(f"  Files improved:            {files_improved}/{len(results)}")

    # Visualize
    output_dir = Path("output/reconstruction_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_comparison(results, output_dir)

    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)


if __name__ == "__main__":
    main()
