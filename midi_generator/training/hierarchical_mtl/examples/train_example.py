"""
Example Training Script for Hierarchical MTL.

This script demonstrates how to train the hierarchical multi-task learning
model for MIDI parameter prediction.

Author: Agent 06
Date: November 20, 2025
"""

import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parents[4]))

from midi_generator.training.hierarchical_mtl.config.training_config import (
    HierarchicalMTLConfig,
    get_default_config,
    get_fast_config,
    get_production_config
)
from midi_generator.training.hierarchical_mtl.data.dataset import create_dataloaders
from midi_generator.training.hierarchical_mtl.loops.trainer import HierarchicalMTLTrainer


def create_dummy_model(input_dim: int = 200) -> torch.nn.Module:
    """
    Create a dummy model for testing.

    In production, this would be the actual hierarchical MTL model from Agent 05.
    """
    class DummyHierarchicalModel(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.shared_encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(512, 256)
            )

            # Level 1 heads (dummy - simplified)
            self.level1_heads = torch.nn.ModuleDict({
                'genre.primary': torch.nn.Linear(256, 7),  # 7 genres
                'tempo.bpm': torch.nn.Linear(256, 1),
            })

            # Level 2 heads (dummy - simplified)
            self.level2_heads = torch.nn.ModuleDict({
                'harmony.chord_density': torch.nn.Linear(256, 1),
                'melody.note_density': torch.nn.Linear(256, 1),
            })

            # Level 3 heads (dummy - simplified)
            self.level3_heads = torch.nn.ModuleDict({
                'orchestration.instrument_count': torch.nn.Linear(256, 1),
            })

        def forward(self, x):
            # Encode features
            encoded = self.shared_encoder(x)

            # Level 1 predictions
            level1_preds = {
                name: head(encoded)
                for name, head in self.level1_heads.items()
            }

            # Level 2 predictions
            level2_preds = {
                name: head(encoded)
                for name, head in self.level2_heads.items()
            }

            # Level 3 predictions
            level3_preds = {
                name: head(encoded)
                for name, head in self.level3_heads.items()
            }

            return {
                'level1': level1_preds,
                'level2': level2_preds,
                'level3': level3_preds
            }

    return DummyHierarchicalModel(input_dim)


def main():
    """Main training function."""
    print("\n" + "=" * 80)
    print("HIERARCHICAL MTL TRAINING EXAMPLE")
    print("=" * 80 + "\n")

    # ======================================================================
    # 1. Configuration
    # ======================================================================
    print("Step 1: Setting up configuration...")

    # Choose config preset
    # config = get_default_config()      # Balanced config
    config = get_fast_config()         # Fast experimentation
    # config = get_production_config()   # Production training

    # Customize config
    config.num_epochs = 10
    config.data.batch_size = 32
    config.optimizer.learning_rate = 1e-3
    config.use_wandb = False  # Set to True to enable Wandb tracking
    config.use_mlflow = False  # Set to True to enable MLflow tracking

    # Set paths
    config.data.labeled_dataset_path = Path("path/to/labeled_dataset.json")
    config.data.features_dir = Path("path/to/features")
    config.checkpoint_dir = Path("checkpoints/hierarchical_mtl")
    config.log_dir = Path("logs/hierarchical_mtl")

    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Learning rate: {config.optimizer.learning_rate}")

    # ======================================================================
    # 2. Data Loading
    # ======================================================================
    print("\nStep 2: Creating data loaders...")

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            labeled_dataset_path=config.data.labeled_dataset_path,
            features_dir=config.data.features_dir,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            use_augmentation=config.data.use_augmentation,
            normalize=config.data.normalize_features
        )
        print("  ✓ Data loaders created successfully")
    except FileNotFoundError as e:
        print(f"\n  WARNING: Data files not found: {e}")
        print("  This is expected for the example. In production, ensure:")
        print("    1. Agent 02 has created the MIDI corpus")
        print("    2. Agent 03 has labeled the dataset")
        print("    3. Agent 04 has extracted features")
        print("\n  Creating dummy data loaders for demonstration...")

        # Create dummy loaders for demonstration
        from torch.utils.data import TensorDataset, DataLoader

        # Dummy data: 100 samples, 200 features
        dummy_features = torch.randn(100, 200)
        dummy_labels_l1 = {
            'genre.primary': torch.randint(0, 7, (100,)),
            'tempo.bpm': torch.randn(100, 1)
        }
        dummy_labels_l2 = {
            'harmony.chord_density': torch.randn(100, 1),
            'melody.note_density': torch.randn(100, 1)
        }
        dummy_labels_l3 = {
            'orchestration.instrument_count': torch.randn(100, 1)
        }

        dummy_dataset = TensorDataset(dummy_features)
        train_loader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dummy_dataset, batch_size=32)
        test_loader = DataLoader(dummy_dataset, batch_size=32)

    # ======================================================================
    # 3. Model Creation
    # ======================================================================
    print("\nStep 3: Creating model...")

    # In production, this would be:
    # from midi_generator.training.hierarchical_mtl.models import HierarchicalMTLModel
    # model = HierarchicalMTLModel(
    #     input_dim=200,
    #     shared_encoder_dim=config.shared_encoder_dim,
    #     level1_hidden_dim=config.level1_hidden_dim,
    #     level2_hidden_dim=config.level2_hidden_dim,
    #     level3_hidden_dim=config.level3_hidden_dim,
    #     dropout_rate=config.dropout_rate
    # )

    # For now, use dummy model
    model = create_dummy_model(input_dim=200)
    print("  ✓ Model created")

    # ======================================================================
    # 4. Trainer Setup
    # ======================================================================
    print("\nStep 4: Setting up trainer...")

    trainer = HierarchicalMTLTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    print("  ✓ Trainer initialized")

    # ======================================================================
    # 5. Training
    # ======================================================================
    print("\nStep 5: Starting training...\n")

    try:
        results = trainer.train()

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nBest validation loss: {results['best_val_loss']:.4f}")
        print(f"Final epoch: {results['final_epoch']}")

        if results['test_metrics']:
            print(f"\nTest Results:")
            print(f"  Loss: {results['test_metrics']['loss']:.4f}")
            print(f"  Level 1 Loss: {results['test_metrics']['level1_loss']:.4f}")
            print(f"  Level 2 Loss: {results['test_metrics']['level2_loss']:.4f}")
            print(f"  Level 3 Loss: {results['test_metrics']['level3_loss']:.4f}")

        print(f"\nCheckpoints saved to: {config.checkpoint_dir}")
        print(f"Logs saved to: {config.log_dir}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
