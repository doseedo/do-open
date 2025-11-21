#!/usr/bin/env python3
"""
Example 3: Feature Visualization

Visualize discovered semantic features to understand what they represent.

Usage:
    python 03_feature_visualization.py

Author: Agent 10 - Documentation & Examples
Date: November 2025
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from midi_generator.learning.semantic_features import SemanticFeatureBank


def visualize_feature_activations(bank: SemanticFeatureBank, output_dir: Path):
    """Plot activation distributions for all features"""
    print("Visualizing feature activations...")

    num_features = len(bank.features)
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    axes = axes.flatten()

    for idx, feature in enumerate(bank.features[:25]):
        ax = axes[idx]
        ax.hist(feature.activation_values, bins=50, alpha=0.7)
        ax.set_title(f"{feature.name}\n({feature.modality.value})", fontsize=10)
        ax.set_xlabel("Activation")
        ax.set_ylabel("Count")

    plt.tight_layout()
    output_file = output_dir / "feature_activations.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def visualize_feature_correlations(bank: SemanticFeatureBank, output_dir: Path):
    """Plot correlation matrix between features"""
    print("Computing feature correlations...")

    # Compute correlation matrix
    activation_matrix = np.array([f.activation_values for f in bank.features])
    corr_matrix = np.corrcoef(activation_matrix)

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        xticklabels=[f.name[:15] for f in bank.features],
        yticklabels=[f.name[:15] for f in bank.features],
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Correlation"}
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()

    output_file = output_dir / "feature_correlations.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def visualize_locality_profiles(bank: SemanticFeatureBank, output_dir: Path):
    """Plot locality profiles for features"""
    print("Visualizing locality profiles...")

    # Select interesting features
    top_features = bank.get_top_k_features(k=6, sort_by="confidence")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    transform_types = [
        "transpose", "rhythm_augment", "velocity_scale",
        "invert", "octave_shift", "time_shift"
    ]

    for idx, feature in enumerate(top_features):
        ax = axes[idx]

        # Get sensitivities
        profile = feature.locality_profile
        sensitivities = [
            getattr(profile, f"{t}_sensitivity", 0.0)
            for t in transform_types
        ]

        # Plot
        ax.bar(range(len(transform_types)), sensitivities, alpha=0.7)
        ax.set_title(f"{feature.name}", fontsize=12)
        ax.set_xticks(range(len(transform_types)))
        ax.set_xticklabels([t.replace("_", "\n") for t in transform_types], fontsize=8)
        ax.set_ylabel("Sensitivity")
        ax.set_ylim([0, 1])

    plt.tight_layout()
    output_file = output_dir / "locality_profiles.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main function"""
    print("="*60)
    print("EXAMPLE 3: Feature Visualization")
    print("="*60)
    print()

    # Load feature bank
    bank_path = Path("output/basic_discovery/feature_bank.pkl")

    if not bank_path.exists():
        print(f"ERROR: Feature bank not found at {bank_path}")
        print("Please run 01_basic_discovery.py first")
        return

    print(f"Loading feature bank from {bank_path}...")
    bank = SemanticFeatureBank()
    bank.load(bank_path)
    print(f"Loaded {len(bank.features)} features")
    print()

    # Create output directory
    output_dir = Path("output/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    visualize_feature_activations(bank, output_dir)
    visualize_feature_correlations(bank, output_dir)
    visualize_locality_profiles(bank, output_dir)

    print()
    print("="*60)
    print("Visualizations complete!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
