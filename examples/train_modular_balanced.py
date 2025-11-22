#!/usr/bin/env python3
"""
Modular Semantic Discovery Training with Balanced Feature Selection
====================================================================

This script trains 6 separate domain-specific encoders using the balanced
feature selection (50 harmony, 40 melody, 40 rhythm, 30 dynamics, 20 texture,
10 structure, 10 orchestration = 200 base features + 20 velocity = 220D).

Each encoder receives only the features relevant to its domain.

Architecture:
    MIDI Files
        ↓
    EnhancedFeatureExtractor (220D balanced)
        ↓
    Split into domain-specific feature vectors:
      - Harmony encoder: 50D → 30 params
      - Melody encoder: 40D → 20 params
      - Rhythm encoder: 40D → 20 params
      - Dynamics encoder: 30D → 15 params
      - Texture encoder: 20D → 10 params
      - Combined (structure+orchestration): 20D → 15 params
      - Velocity: 20D → 10 params (cross-dimensional)
        ↓
    Total: 120 interpretable parameters
"""

import argparse
import sys
from pathlib import Path
import time
import json
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Feature extraction
from midi_generator.feature_selection.enhanced_feature_extractor import (
    EnhancedFeatureExtractor,
    NormalizedFeatureExtractor
)

# Semantic encoder
from midi_generator.learning.semantic_encoder import (
    SemanticFeatureEncoder,
    EncoderConfig
)


# =============================================================================
# Domain-Specific Feature Splitter
# =============================================================================

class BalancedFeatureSplitter:
    """
    Splits 220D balanced features into domain-specific feature vectors.

    Input: 220D (200 base + 20 velocity)
    Base 200D breakdown:
      - Harmony: 50 features (indices 0-49)
      - Melody: 40 features (indices 50-89)
      - Rhythm: 40 features (indices 90-129)
      - Dynamics: 30 features (indices 130-159)
      - Texture: 20 features (indices 160-179)
      - Structure: 10 features (indices 180-189)
      - Orchestration: 10 features (indices 190-199)
      - Velocity: 20 features (indices 200-219)
    """

    def __init__(self):
        self.domain_ranges = {
            'harmony': (0, 50),
            'melody': (50, 90),
            'rhythm': (90, 130),
            'dynamics': (130, 160),
            'texture': (160, 180),
            'structure': (180, 190),
            'orchestration': (190, 200),
            'velocity': (200, 220)
        }

    def split(self, features_220d: np.ndarray) -> dict:
        """
        Split 220D feature vector into domain-specific vectors.

        Args:
            features_220d: numpy array of shape (220,) or (batch, 220)

        Returns:
            Dictionary mapping domain names to feature vectors
        """
        result = {}
        for domain, (start, end) in self.domain_ranges.items():
            if len(features_220d.shape) == 1:
                result[domain] = features_220d[start:end]
            else:
                result[domain] = features_220d[:, start:end]
        return result

    def get_domain_dim(self, domain: str) -> int:
        """Get feature dimension for a domain"""
        start, end = self.domain_ranges[domain]
        return end - start


# =============================================================================
# Domain-Specific Dataset
# =============================================================================

class DomainSpecificDataset(Dataset):
    """
    Dataset that extracts domain-specific features for training.

    For each MIDI file:
    1. Extract 220D features
    2. Split into domain-specific vectors
    3. Return the requested domain's features
    """

    def __init__(self, midi_files, feature_extractor, domain: str):
        self.midi_files = midi_files
        self.feature_extractor = feature_extractor
        self.domain = domain
        self.splitter = BalancedFeatureSplitter()

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        midi_file = self.midi_files[idx]

        try:
            # Extract 220D features
            features_220d = self.feature_extractor.extract(midi_file)

            # Split into domain-specific features
            domain_features_dict = self.splitter.split(features_220d)
            domain_features = domain_features_dict[self.domain]

            return torch.from_numpy(domain_features).float()

        except Exception as e:
            warnings.warn(f"Failed to extract {self.domain} features from {midi_file}: {e}")
            dim = self.splitter.get_domain_dim(self.domain)
            return torch.zeros(dim, dtype=torch.float32)


# =============================================================================
# Modular Training Configuration
# =============================================================================

class ModularTrainingConfig:
    """Configuration for training 6 separate encoders"""

    def __init__(
        self,
        corpus_dir: Path,
        output_dir: Path,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = 'cuda'
    ):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        # Domain-specific encoder configurations
        # (input_dim, num_semantic_features)
        self.encoder_specs = {
            'harmony': (50, 30),
            'melody': (40, 20),
            'rhythm': (40, 20),
            'dynamics': (30, 15),
            'texture': (20, 10),
            'structure_orchestration': (20, 15),  # Combined 10+10
            'velocity': (20, 10)  # Cross-dimensional
        }


# =============================================================================
# Training Functions
# =============================================================================

def create_encoder(input_dim: int, num_params: int, device: str) -> SemanticFeatureEncoder:
    """Create a semantic encoder for a domain"""
    config = EncoderConfig(
        input_dim=input_dim,
        hidden_dim=512,
        num_semantic_features=num_params,
        num_locality_types=5,
        reconstruction_weight=1.0,
        locality_weight=0.5,
        sparsity_weight=0.01
    )

    encoder = SemanticFeatureEncoder(config)
    encoder.to(device)
    return encoder


def train_encoder(
    encoder: SemanticFeatureEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    domain_name: str,
    output_dir: Path,
    device: str
):
    """Train a single encoder"""
    print(f"\n{'='*70}")
    print(f"Training {domain_name.upper()} Encoder")
    print(f"{'='*70}")

    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 15

    for epoch in range(epochs):
        # Training
        encoder.train()
        train_losses = []

        for batch in train_loader:
            batch = batch.to(device)

            # Forward pass
            loss_dict = encoder.compute_loss(batch)
            loss = loss_dict['total_loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation
        encoder.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss_dict = encoder.compute_loss(batch)
                val_losses.append(loss_dict['total_loss'].item())

        avg_val_loss = np.mean(val_losses)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save best model
            checkpoint_path = output_dir / f"{domain_name}_encoder_best.pt"
            encoder.save(checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    checkpoint_path = output_dir / f"{domain_name}_encoder_best.pt"
    encoder = SemanticFeatureEncoder.load(checkpoint_path, device=device)

    print(f"✅ {domain_name.upper()} encoder training complete")
    print(f"   Best validation loss: {best_val_loss:.4f}")

    return encoder


def main():
    parser = argparse.ArgumentParser(description="Train modular semantic encoders with balanced features")
    parser.add_argument("--corpus-dir", type=Path, required=True, help="MIDI corpus directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    print("="*70)
    print("MODULAR SEMANTIC DISCOVERY - BALANCED FEATURES")
    print("="*70)
    print(f"\nCorpus: {args.corpus_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")

    # Create configuration
    config = ModularTrainingConfig(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )

    # Get MIDI files
    print("\n📂 Loading MIDI corpus...")
    midi_files = list(config.corpus_dir.glob("**/*.mid"))
    print(f"   Found {len(midi_files)} MIDI files")

    # Split into train/val/test
    np.random.seed(42)
    np.random.shuffle(midi_files)

    n_train = int(0.8 * len(midi_files))
    n_val = int(0.1 * len(midi_files))

    train_files = midi_files[:n_train]
    val_files = midi_files[n_train:n_train+n_val]
    test_files = midi_files[n_train+n_val:]

    print(f"   Train: {len(train_files)} files")
    print(f"   Val: {len(val_files)} files")
    print(f"   Test: {len(test_files)} files")

    # Create feature extractor
    print("\n🔧 Initializing feature extractor...")
    selection_file = Path(__file__).parent.parent / "midi_generator/feature_selection/output/selected_features_200_balanced.json"

    base_extractor = EnhancedFeatureExtractor.from_selection_file(str(selection_file))
    normalized_extractor = NormalizedFeatureExtractor(base_extractor)

    # Fit normalizer on sample of training data
    print("   Fitting normalizer...")
    sample_size = min(200, len(train_files))
    normalized_extractor.fit(train_files[:sample_size], sample_size=sample_size)
    print("   ✅ Normalizer fitted")

    # Train each encoder
    trained_encoders = {}

    for domain, (input_dim, num_params) in config.encoder_specs.items():
        print(f"\n{'='*70}")
        print(f"TRAINING {domain.upper()} ENCODER")
        print(f"{'='*70}")
        print(f"  Input dim: {input_dim}D")
        print(f"  Output params: {num_params}")

        # Create datasets
        train_dataset = DomainSpecificDataset(train_files, normalized_extractor, domain.split('_')[0])
        val_dataset = DomainSpecificDataset(val_files, normalized_extractor, domain.split('_')[0])

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)

        # Create encoder
        encoder = create_encoder(input_dim, num_params, config.device)

        # Train
        trained_encoder = train_encoder(
            encoder,
            train_loader,
            val_loader,
            config.epochs,
            config.learning_rate,
            domain,
            config.output_dir,
            config.device
        )

        trained_encoders[domain] = trained_encoder

    # Save configuration
    config_path = config.output_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'corpus_dir': str(config.corpus_dir),
            'output_dir': str(config.output_dir),
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'device': config.device,
            'encoder_specs': {k: list(v) for k, v in config.encoder_specs.items()},
            'num_train_files': len(train_files),
            'num_val_files': len(val_files),
            'num_test_files': len(test_files),
        }, f, indent=2)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"  Trained {len(trained_encoders)} encoders")
    print(f"  Total parameters: {sum(v[1] for v in config.encoder_specs.values())}")
    print(f"  Output directory: {config.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
