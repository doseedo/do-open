#!/usr/bin/env python3
"""
Gap Dataset Example - Agent 4
==============================

Example usage of the Gap Dataset system for semantic feature discovery.

This demonstrates:
1. Creating gap dataset from MIDI corpus
2. Analyzing reconstruction gaps
3. Using dataset for training
4. Cache management

Author: Agent 4 - Gap Dataset Creation
License: MIT
"""

from pathlib import Path
import numpy as np

# Import gap dataset components
from midi_generator.learning.gap_dataset import (
    ParameterMIDIGenerator,
    GapAnalyzer,
    GapCache,
    GapDataset,
    create_gap_dataset_from_directory
)

# Import existing systems
from midi_generator.feature_selection.optimized_feature_extractor import (
    OptimizedFeatureExtractor
)
from midi_generator.parameters.hierarchical_extractor_v2 import (
    HierarchicalParameterExtractorV2
)


# ============================================================================
# Example 1: Quick Start - Create Dataset from Directory
# ============================================================================

def example_1_quick_start():
    """
    Quickest way to create a gap dataset for training.

    This uses the convenience function that sets up everything.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Quick Start - Create Dataset from Directory")
    print("="*70 + "\n")

    # Define paths
    midi_dir = Path('data/midi_corpus')
    cache_dir = Path('data/gap_cache')
    output_dir = Path('output/gap_analysis')

    # Create dataset (this will precompute all gaps and cache them)
    dataset = create_gap_dataset_from_directory(
        midi_dir=midi_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        max_files=100,  # Process first 100 files
        normalize=True,
        verbose=True
    )

    print(f"\n✅ Dataset created with {len(dataset)} files")
    print(f"   Cache location: {cache_dir}")
    print(f"   Statistics saved to: {output_dir}")

    # Get first item
    item = dataset[0]
    print(f"\nDataset item structure:")
    print(f"  features: {item['features'].shape}  # 200D input features")
    print(f"  gaps: {item['gaps'].shape}  # Reconstruction gaps")
    print(f"  parameters_flat: {item['parameters_flat'].shape}  # 50 parameters")
    print(f"  file_id: {item['file_id']}")

    return dataset


# ============================================================================
# Example 2: Step-by-Step Setup
# ============================================================================

def example_2_step_by_step():
    """
    Create dataset with explicit control over each component.

    Use this when you need custom configuration.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Step-by-Step Setup")
    print("="*70 + "\n")

    # Step 1: Initialize feature extractor (200D)
    print("Step 1: Initialize feature extractor...")
    feature_extractor = OptimizedFeatureExtractor.from_selection_file(
        Path('midi_generator/feature_selection/output/selected_features_200_template.json')
    )
    print("  ✅ Feature extractor ready")

    # Step 2: Initialize parameter extractor (50 params)
    print("\nStep 2: Initialize parameter extractor...")
    parameter_extractor = HierarchicalParameterExtractorV2(verbose=False)
    print("  ✅ Parameter extractor ready")

    # Step 3: Initialize MIDI generator
    print("\nStep 3: Initialize MIDI generator...")
    midi_generator = ParameterMIDIGenerator(verbose=True)
    print("  ✅ MIDI generator ready")

    # Step 4: Create gap analyzer
    print("\nStep 4: Create gap analyzer...")
    gap_analyzer = GapAnalyzer(
        feature_extractor=feature_extractor,
        parameter_extractor=parameter_extractor,
        midi_generator=midi_generator,
        verbose=True
    )
    print("  ✅ Gap analyzer ready")

    # Step 5: Create cache
    print("\nStep 5: Create cache...")
    cache = GapCache(
        cache_dir=Path('data/gap_cache'),
        max_size_gb=10.0,  # 10 GB cache
        verbose=True
    )
    print("  ✅ Cache ready")

    # Step 6: Find MIDI files
    print("\nStep 6: Find MIDI files...")
    midi_dir = Path('data/midi_corpus')
    midi_files = list(midi_dir.glob('**/*.mid'))[:50]  # First 50 files
    print(f"  ✅ Found {len(midi_files)} MIDI files")

    # Step 7: Create dataset
    print("\nStep 7: Create dataset...")
    dataset = GapDataset(
        midi_files=midi_files,
        gap_analyzer=gap_analyzer,
        cache=cache,
        precompute=True,  # Precompute all gaps
        normalize_features=True,
        verbose=True
    )
    print(f"  ✅ Dataset ready with {len(dataset)} files")

    return dataset, gap_analyzer, cache


# ============================================================================
# Example 3: Analyze Corpus Gaps
# ============================================================================

def example_3_analyze_corpus():
    """
    Analyze reconstruction gaps across entire corpus.

    This identifies which features are poorly reconstructed.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Analyze Corpus Gaps")
    print("="*70 + "\n")

    # Initialize components
    feature_extractor = OptimizedFeatureExtractor.from_selection_file(
        Path('midi_generator/feature_selection/output/selected_features_200_template.json')
    )
    parameter_extractor = HierarchicalParameterExtractorV2(verbose=False)
    midi_generator = ParameterMIDIGenerator(verbose=False)

    gap_analyzer = GapAnalyzer(
        feature_extractor=feature_extractor,
        parameter_extractor=parameter_extractor,
        midi_generator=midi_generator,
        verbose=True
    )

    # Get MIDI files
    midi_dir = Path('data/midi_corpus')
    midi_files = list(midi_dir.glob('**/*.mid'))[:100]

    # Analyze corpus
    print("\nAnalyzing corpus gaps...")
    statistics = gap_analyzer.analyze_corpus_gaps(
        midi_files,
        output_dir=Path('output/gap_analysis'),
        show_progress=True
    )

    # Print summary
    print(f"\n{'='*70}")
    print("Gap Analysis Results:")
    print(f"{'='*70}")
    print(f"Files processed: {statistics.total_files_processed}/{statistics.n_files}")
    print(f"Failed: {statistics.failed_files}")
    print(f"Mean reconstruction gap: {statistics.mean_total_gap:.4f} ± {statistics.std_total_gap:.4f}")
    print(f"\nPercentiles:")
    print(f"  25th: {statistics.percentiles['p25']:.4f}")
    print(f"  50th: {statistics.percentiles['p50']:.4f}")
    print(f"  75th: {statistics.percentiles['p75']:.4f}")
    print(f"  95th: {statistics.percentiles['p95']:.4f}")

    print(f"\nTop 10 problematic features (largest mean gaps):")
    for idx, gap in statistics.top_gap_features[:10]:
        feature_name = feature_extractor.get_feature_names()[idx]
        print(f"  Feature {idx:3d} ({feature_name[:40]:40s}): {gap:.4f}")

    print(f"\nTop 10 parameter gaps:")
    sorted_param_gaps = sorted(
        statistics.mean_parameter_gaps.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for param_name, gap in sorted_param_gaps:
        print(f"  {param_name:50s}: {gap:.4f}")

    return statistics


# ============================================================================
# Example 4: Use Dataset for Training
# ============================================================================

def example_4_training_usage():
    """
    Use gap dataset for training semantic feature encoder.

    This shows integration with Agent 5's training loop.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Training Usage")
    print("="*70 + "\n")

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("⚠️ PyTorch not available. Skipping training example.")
        return

    # Create dataset
    dataset = create_gap_dataset_from_directory(
        midi_dir=Path('data/midi_corpus'),
        cache_dir=Path('data/gap_cache'),
        max_files=50,
        normalize=True,
        verbose=False
    )

    # Create DataLoader
    dataloader = dataset.get_dataloader(
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: 16")
    print(f"Number of batches: {len(dataloader)}")

    # Example: Simple encoder (this would be Agent 3's SemanticEncoder)
    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(200, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 30)  # 30 semantic features
            )

        def forward(self, x):
            return self.encoder(x)

    model = SimpleEncoder()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop example
    print("\nExample training loop:")
    model.train()

    for epoch in range(2):  # Just 2 epochs for demo
        total_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(dataloader):
            features = batch['features']  # (batch_size, 200)
            gaps = batch['gaps']  # (batch_size, 200)

            # Forward pass
            encoded = model(features)  # (batch_size, 30)

            # Loss: train to predict gaps
            # (This is simplified; Agent 5 will use more sophisticated loss)
            loss = criterion(encoded[:, :20], gaps[:, :20])  # Predict first 20 gaps

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_idx == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}")
                print(f"    Features shape: {features.shape}")
                print(f"    Gaps shape: {gaps.shape}")
                print(f"    Encoded shape: {encoded.shape}")
                print(f"    Loss: {loss.item():.4f}")

        avg_loss = total_loss / batch_count
        print(f"\n  Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")

    print("\n✅ Training loop completed")


# ============================================================================
# Example 5: Cache Management
# ============================================================================

def example_5_cache_management():
    """
    Demonstrate cache management features.

    Shows how to:
    - Check cache statistics
    - Clear cache
    - Manage cache size
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Cache Management")
    print("="*70 + "\n")

    # Create cache
    cache = GapCache(
        cache_dir=Path('data/gap_cache'),
        max_size_gb=5.0,
        verbose=True
    )

    # Get statistics
    stats = cache.get_stats()
    print("Cache Statistics:")
    print(f"  Entries: {stats['entries']}")
    print(f"  Size: {stats['total_size_gb']:.2f} GB / {stats['max_size_gb']:.2f} GB")
    print(f"  Utilization: {stats['utilization']:.1%}")
    print(f"  Hit count: {stats['hit_count']}")
    print(f"  Miss count: {stats['miss_count']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")

    # Clear cache if needed
    if stats['utilization'] > 0.9:
        print("\n⚠️ Cache is > 90% full, clearing...")
        cache.clear()
        print("✅ Cache cleared")

    return cache


# ============================================================================
# Example 6: Single File Gap Analysis
# ============================================================================

def example_6_single_file():
    """
    Analyze gap for a single MIDI file.

    Useful for debugging or analyzing specific files.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Single File Gap Analysis")
    print("="*70 + "\n")

    # Initialize components
    feature_extractor = OptimizedFeatureExtractor.from_selection_file(
        Path('midi_generator/feature_selection/output/selected_features_200_template.json')
    )
    parameter_extractor = HierarchicalParameterExtractorV2(verbose=False)
    midi_generator = ParameterMIDIGenerator(verbose=True)

    gap_analyzer = GapAnalyzer(
        feature_extractor=feature_extractor,
        parameter_extractor=parameter_extractor,
        midi_generator=midi_generator,
        verbose=True
    )

    # Analyze single file
    test_file = Path('data/midi_corpus/song_001.mid')

    if not test_file.exists():
        print(f"⚠️ Test file not found: {test_file}")
        return

    print(f"Analyzing: {test_file.name}")
    gap = gap_analyzer.compute_gap(test_file)

    if gap.success:
        print(f"\n{'='*70}")
        print("Gap Analysis Results:")
        print(f"{'='*70}")
        print(f"File: {gap.file_id}")
        print(f"Total gap: {gap.total_gap:.4f}")
        print(f"Computation time: {gap.computation_time:.2f}s")

        print(f"\nTop 10 feature gaps:")
        for idx, gap_value in gap.get_top_gaps(k=10):
            feature_name = feature_extractor.get_feature_names()[idx]
            print(f"  Feature {idx:3d} ({feature_name[:40]:40s}): {gap_value:.4f}")

        print(f"\nTop 10 parameter gaps:")
        sorted_param_gaps = sorted(
            gap.parameter_gaps.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for param_name, gap_value in sorted_param_gaps:
            print(f"  {param_name:50s}: {gap_value:.4f}")
    else:
        print(f"\n⚠️ Gap computation failed: {gap.error_message}")

    return gap


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples"""
    print("="*70)
    print("GAP DATASET EXAMPLES - AGENT 4")
    print("="*70)

    # Check dependencies
    try:
        import mido
        print("✅ mido available")
    except ImportError:
        print("⚠️ mido not available (required for MIDI generation)")

    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        print("⚠️ PyTorch not available (required for dataset)")

    print("\nAvailable examples:")
    print("  1. Quick Start - Create dataset from directory")
    print("  2. Step-by-Step Setup - Explicit component initialization")
    print("  3. Analyze Corpus - Identify problematic features")
    print("  4. Training Usage - Use dataset for training")
    print("  5. Cache Management - Manage cache")
    print("  6. Single File - Analyze one MIDI file")

    # Note: In actual use, you would run specific examples
    # For demo purposes, we just show the code structure

    print("\n" + "="*70)
    print("To run specific examples, uncomment the function calls below:")
    print("="*70)

    # Uncomment to run specific examples:
    # example_1_quick_start()
    # example_2_step_by_step()
    # example_3_analyze_corpus()
    # example_4_training_usage()
    # example_5_cache_management()
    # example_6_single_file()

    print("\n✅ Example code ready to use!")


if __name__ == '__main__':
    main()
