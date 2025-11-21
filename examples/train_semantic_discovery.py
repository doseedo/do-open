"""
Example: Semantic Feature Discovery Training
============================================

Complete example of training the semantic feature discovery system to
automatically discover 20-30 interpretable musical parameters.

This script demonstrates:
    1. Setting up training configuration
    2. Loading/creating dataset (Agent 4 integration)
    3. Training the semantic encoder (Agent 3 integration)
    4. Extracting learned features
    5. Saving results for interpretation (Agent 6)

Usage:
    # Basic training with defaults
    python examples/train_semantic_discovery.py

    # Custom configuration
    python examples/train_semantic_discovery.py \
        --corpus-dir data/my_midi_corpus \
        --output-dir output/my_experiment \
        --num-features 30 \
        --epochs 200 \
        --batch-size 64

    # Resume from checkpoint
    python examples/train_semantic_discovery.py \
        --resume output/my_experiment/checkpoints/checkpoint_epoch_50.pt

Author: Agent 05
Date: November 21, 2025
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

from midi_generator.learning.gap_discovery_trainer import (
    TrainingConfig,
    GapDiscoveryTrainer,
    create_simple_dataset_for_testing,
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train semantic feature discovery system',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths
    parser.add_argument(
        '--corpus-dir',
        type=Path,
        default=Path('data/midi_corpus'),
        help='Directory containing MIDI training corpus'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output/semantic_discovery'),
        help='Output directory for models and logs'
    )

    # Model architecture
    parser.add_argument(
        '--input-dim',
        type=int,
        default=200,
        help='Input feature dimension (from OptimizedFeatureExtractor)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=512,
        help='Hidden layer dimension'
    )
    parser.add_argument(
        '--num-features',
        type=int,
        default=25,
        help='Number of semantic features to discover (target: 20-30)'
    )

    # Training hyperparameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='L2 regularization weight'
    )

    # Loss weights
    parser.add_argument(
        '--reconstruction-weight',
        type=float,
        default=1.0,
        help='Weight for reconstruction loss'
    )
    parser.add_argument(
        '--sparsity-weight',
        type=float,
        default=0.01,
        help='Weight for sparsity constraint'
    )
    parser.add_argument(
        '--locality-weight',
        type=float,
        default=0.5,
        help='Weight for locality constraint'
    )
    parser.add_argument(
        '--orthogonality-weight',
        type=float,
        default=0.1,
        help='Weight for feature orthogonality'
    )

    # Training options
    parser.add_argument(
        '--resume',
        type=Path,
        default=None,
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Training device'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    # Logging
    parser.add_argument(
        '--use-tensorboard',
        action='store_true',
        help='Enable TensorBoard logging'
    )
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='semantic-feature-discovery',
        help='W&B project name'
    )
    parser.add_argument(
        '--wandb-entity',
        type=str,
        default=None,
        help='W&B entity name'
    )

    # Dataset options
    parser.add_argument(
        '--use-synthetic',
        action='store_true',
        help='Use synthetic dataset for testing (no MIDI corpus needed)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5000,
        help='Number of synthetic samples (if --use-synthetic)'
    )

    return parser.parse_args()


def load_dataset(args, config):
    """
    Load dataset for training.

    Integration point for Agent 4's GapDataset.

    Args:
        args: Command line arguments
        config: Training configuration

    Returns:
        train_loader, val_loader, test_loader
    """
    if args.use_synthetic:
        print("\n" + "="*70)
        print("USING SYNTHETIC DATASET FOR TESTING")
        print("="*70)
        print("⚠️  For actual training, use Agent 4's GapDataset with real MIDI corpus")
        print("="*70 + "\n")

        from midi_generator.learning.gap_discovery_trainer import create_simple_dataset_for_testing

        train_loader, val_loader = create_simple_dataset_for_testing(
            n_samples=args.num_samples,
            input_dim=config.input_dim
        )

        # Use val as test for synthetic
        test_loader = val_loader

        return train_loader, val_loader, test_loader

    else:
        # Try to load Agent 4's GapDataset
        try:
            from midi_generator.learning.gap_dataset import GapDataset
            from torch.utils.data import random_split

            print(f"\n📂 Loading MIDI corpus from {args.corpus_dir}")

            # Create full dataset
            full_dataset = GapDataset(
                midi_corpus_dir=args.corpus_dir,
                cache_dir=config.output_dir / 'gap_cache',
                precompute_gaps=True
            )

            # Split into train/val/test
            n_total = len(full_dataset)
            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.15)
            n_test = n_total - n_train - n_val

            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset,
                [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(config.random_seed)
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=torch.cuda.is_available()
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=torch.cuda.is_available()
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=torch.cuda.is_available()
            )

            print(f"✅ Loaded dataset:")
            print(f"   Train: {len(train_dataset)} samples")
            print(f"   Val: {len(val_dataset)} samples")
            print(f"   Test: {len(test_dataset)} samples")

            return train_loader, val_loader, test_loader

        except ImportError:
            print("\n" + "="*70)
            print("ERROR: Agent 4's GapDataset not available")
            print("="*70)
            print("Options:")
            print("  1. Use --use-synthetic flag for testing with synthetic data")
            print("  2. Wait for Agent 4 to complete GapDataset implementation")
            print("  3. Implement your own dataset loader")
            print("="*70 + "\n")
            sys.exit(1)


def main():
    """Main training script"""
    args = parse_args()

    print("\n" + "="*70)
    print("SEMANTIC FEATURE DISCOVERY TRAINING")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    print(f"Target features: {args.num_features}")
    print(f"Max epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("="*70 + "\n")

    # Create configuration
    config = TrainingConfig(
        # Paths
        midi_corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.output_dir / 'checkpoints',
        log_dir=args.output_dir / 'logs',

        # Model architecture
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_semantic_features=args.num_features,

        # Training hyperparameters
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,

        # Loss weights
        reconstruction_weight=args.reconstruction_weight,
        sparsity_weight=args.sparsity_weight,
        locality_weight=args.locality_weight,
        orthogonality_weight=args.orthogonality_weight,

        # Logging
        use_tensorboard=args.use_tensorboard,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,

        # Device
        device=args.device,

        # Reproducibility
        random_seed=args.random_seed,
    )

    # Save configuration
    config.save(config.output_dir / 'training_config.json')
    print(f"📝 Saved config to {config.output_dir / 'training_config.json'}\n")

    # Load dataset
    train_loader, val_loader, test_loader = load_dataset(args, config)

    # Create trainer
    print("🔧 Initializing trainer...")
    trainer = GapDiscoveryTrainer(config)

    # Train model
    print("\n🚀 Starting training...\n")
    summary = trainer.train(
        train_loader,
        val_loader,
        resume_from_checkpoint=args.resume
    )

    # Extract semantic features from validation set
    print("\n📊 Extracting semantic features from validation set...")
    semantic_data = trainer.extract_semantic_features(val_loader)

    # Save semantic feature bank
    bank_path = config.output_dir / 'semantic_feature_bank.npz'
    trainer.save_semantic_feature_bank(semantic_data, bank_path)

    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    test_loss, test_components = trainer.validate(test_loader)

    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Reconstruction Loss: {test_components['reconstruction']:.4f}")
    print(f"Sparsity Loss: {test_components['sparsity']:.4f}")
    print("="*70 + "\n")

    # Save final summary
    import json
    final_summary = {
        'training': summary,
        'test': {
            'test_loss': test_loss,
            'reconstruction_loss': test_components['reconstruction'],
            'sparsity_loss': test_components['sparsity'],
        },
        'config': {
            'num_semantic_features': config.num_semantic_features,
            'input_dim': config.input_dim,
            'hidden_dim': config.hidden_dim,
        }
    }

    summary_path = config.output_dir / 'final_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(final_summary, f, indent=2)

    print(f"✅ Training complete! Results saved to {config.output_dir}\n")

    print("Next steps:")
    print(f"1. Interpret features: Use Agent 6's FeatureInterpreter on {bank_path}")
    print(f"2. Integrate pipeline: Use Agent 7's SemanticDiscoveryPipeline")
    print(f"3. Evaluate quality: Use Agent 9's SemanticFeatureEvaluator")
    print()


if __name__ == "__main__":
    main()
