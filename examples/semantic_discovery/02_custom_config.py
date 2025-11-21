#!/usr/bin/env python3
"""
Example 2: Custom Configuration

This example shows how to customize training parameters for semantic feature
discovery, including loss weights, batch size, learning rate, etc.

Usage:
    python 02_custom_config.py

Author: Agent 10 - Documentation & Examples
Date: November 2025
"""

from pathlib import Path
from midi_generator.learning.semantic_discovery_pipeline import SemanticDiscoveryPipeline
from midi_generator.learning.gap_discovery_trainer import TrainingConfig


def example_quick_development():
    """
    Quick configuration for development/testing.

    Fast training with smaller corpus for rapid iteration.
    """
    print("\n" + "="*60)
    print("CONFIGURATION 1: Quick Development")
    print("="*60)

    # Use smaller corpus for faster iteration
    midi_files = list(Path("data/midi/train").glob("*.mid"))[:200]

    custom_config = {
        # Model
        "num_features": 15,  # Fewer features = faster
        "encoder_hidden_dim": 256,  # Smaller network = faster

        # Training
        "batch_size": 64,  # Larger batch = faster (if GPU allows)
        "num_epochs": 50,  # Fewer epochs = faster
        "early_stopping_patience": 5,

        # Loss weights (balanced)
        "reconstruction_weight": 1.0,
        "locality_weight": 0.3,  # Lower for faster convergence
        "sparsity_weight": 0.005,

        # Hardware
        "num_workers": 8,
    }

    pipeline = SemanticDiscoveryPipeline(
        midi_corpus_dir=Path("data/midi/train"),
        output_dir=Path("output/quick_dev"),
        num_features=custom_config["num_features"],
        config=custom_config
    )

    print("Configuration:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")

    print("\nExpected time: ~1-2 hours on GPU")
    print("Quality: Lower than production (good for testing)")


def example_high_quality():
    """
    Configuration optimized for highest quality features.

    Emphasizes interpretability and musical validity over speed.
    """
    print("\n" + "="*60)
    print("CONFIGURATION 2: High Quality")
    print("="*60)

    custom_config = {
        # Model
        "num_features": 30,  # More features = more coverage
        "encoder_hidden_dim": 512,
        "dropout": 0.1,

        # Training
        "batch_size": 32,
        "learning_rate": 5e-5,  # Lower LR for careful training
        "num_epochs": 150,  # More epochs
        "early_stopping_patience": 20,  # Patient stopping

        # Loss weights (emphasize locality for interpretability)
        "reconstruction_weight": 1.0,
        "locality_weight": 0.8,  # HIGH - very interpretable features
        "sparsity_weight": 0.02,  # Higher sparsity

        # Regularization
        "weight_decay": 1e-4,
        "gradient_clip": 0.5,

        # Hardware
        "num_workers": 4,
    }

    pipeline = SemanticDiscoveryPipeline(
        midi_corpus_dir=Path("data/midi/train"),
        output_dir=Path("output/high_quality"),
        num_features=custom_config["num_features"],
        config=custom_config
    )

    print("Configuration:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")

    print("\nExpected time: ~8-12 hours on GPU")
    print("Quality: Maximum interpretability and musical validity")


def example_fast_reconstruction():
    """
    Configuration optimized for reconstruction quality over interpretability.

    Best for when you need maximum reconstruction accuracy.
    """
    print("\n" + "="*60)
    print("CONFIGURATION 3: Fast Reconstruction")
    print("="*60)

    custom_config = {
        # Model
        "num_features": 40,  # Many features = more expressive
        "encoder_hidden_dim": 512,
        "dropout": 0.05,  # Low dropout = more capacity

        # Training
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 100,

        # Loss weights (emphasize reconstruction)
        "reconstruction_weight": 1.0,
        "locality_weight": 0.2,  # LOW - prioritize reconstruction
        "sparsity_weight": 0.005,  # Low sparsity

        # Hardware
        "num_workers": 4,
    }

    pipeline = SemanticDiscoveryPipeline(
        midi_corpus_dir=Path("data/midi/train"),
        output_dir=Path("output/fast_recon"),
        num_features=custom_config["num_features"],
        config=custom_config
    )

    print("Configuration:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")

    print("\nExpected time: ~5-7 hours on GPU")
    print("Quality: Maximum reconstruction, lower interpretability")


def example_memory_constrained():
    """
    Configuration for systems with limited GPU memory.

    Uses gradient accumulation to simulate larger batches.
    """
    print("\n" + "="*60)
    print("CONFIGURATION 4: Memory Constrained (8GB GPU)")
    print("="*60)

    custom_config = {
        # Model
        "num_features": 25,
        "encoder_hidden_dim": 512,

        # Training (optimized for low memory)
        "batch_size": 8,  # SMALL batch for 8GB GPU
        "gradient_accumulation_steps": 4,  # Effective batch = 8*4 = 32
        "num_epochs": 100,

        # Loss weights
        "reconstruction_weight": 1.0,
        "locality_weight": 0.5,
        "sparsity_weight": 0.01,

        # Hardware
        "num_workers": 2,  # Fewer workers = less memory
        "pin_memory": False,  # Save a bit more memory
    }

    pipeline = SemanticDiscoveryPipeline(
        midi_corpus_dir=Path("data/midi/train"),
        output_dir=Path("output/mem_constrained"),
        num_features=custom_config["num_features"],
        config=custom_config
    )

    print("Configuration:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")

    print("\nExpected time: ~6-9 hours on 8GB GPU")
    print("Quality: Same as standard, just slower")


def example_cpu_only():
    """
    Configuration for CPU-only training.

    Much slower but works without GPU.
    """
    print("\n" + "="*60)
    print("CONFIGURATION 5: CPU Only")
    print("="*60)

    # Use very small corpus for CPU
    midi_files = list(Path("data/midi/train").glob("*.mid"))[:200]

    custom_config = {
        # Model (smaller for CPU)
        "num_features": 15,
        "encoder_hidden_dim": 256,

        # Training
        "batch_size": 16,
        "num_epochs": 50,
        "early_stopping_patience": 5,

        # Loss weights
        "reconstruction_weight": 1.0,
        "locality_weight": 0.4,
        "sparsity_weight": 0.01,

        # Hardware
        "device": "cpu",  # FORCE CPU
        "num_workers": 8,  # More workers help on CPU
    }

    pipeline = SemanticDiscoveryPipeline(
        midi_corpus_dir=Path("data/midi/train"),
        output_dir=Path("output/cpu_only"),
        num_features=custom_config["num_features"],
        config=custom_config
    )

    print("Configuration:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")

    print("\nExpected time: ~24-48 hours on fast CPU")
    print("Quality: Good for testing, slower than GPU")


def run_custom_config():
    """
    Example of running with a custom configuration.
    """
    print("\n" + "="*60)
    print("RUNNING WITH CUSTOM CONFIGURATION")
    print("="*60)

    # Choose configuration based on your needs
    print("\nAvailable configurations:")
    print("1. Quick Development (1-2 hours, lower quality)")
    print("2. High Quality (8-12 hours, maximum interpretability)")
    print("3. Fast Reconstruction (5-7 hours, maximum reconstruction)")
    print("4. Memory Constrained (6-9 hours, works on 8GB GPU)")
    print("5. CPU Only (24-48 hours, no GPU needed)")
    print()
    print("Select configuration (1-5) or 'demo' for demo mode: ", end="")

    choice = input().strip()

    if choice == "demo":
        # Just show configurations without running
        example_quick_development()
        example_high_quality()
        example_fast_reconstruction()
        example_memory_constrained()
        example_cpu_only()
        print("\n" + "="*60)
        print("Demo mode: Configurations shown, not executed")
        print("="*60)
        return

    # Actually run the chosen configuration
    configs = {
        "1": example_quick_development,
        "2": example_high_quality,
        "3": example_fast_reconstruction,
        "4": example_memory_constrained,
        "5": example_cpu_only,
    }

    if choice in configs:
        config_func = configs[choice]
        config_func()

        print("\n" + "="*60)
        print("Ready to run with this configuration")
        print("="*60)
        print("\nUncomment the pipeline.run() line below to execute:")
        print("# results = pipeline.run()")
    else:
        print(f"Invalid choice: {choice}")


def main():
    """Main function"""
    print("="*60)
    print("EXAMPLE 2: Custom Configuration for Semantic Discovery")
    print("="*60)

    run_custom_config()


if __name__ == "__main__":
    main()
