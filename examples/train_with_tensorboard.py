#!/usr/bin/env python3
"""
Domain Encoder Training with TensorBoard Monitoring (v2.0)

This script demonstrates how to train domain encoders with:
- Feature normalization (CRITICAL for convergence)
- TensorBoard logging for real-time monitoring
- Updated hyperparameters (1024 hidden_dim, 1e-2 learning_rate)

Usage:
    # Train all domain encoders
    python examples/train_with_tensorboard.py

    # Train specific domain
    python examples/train_with_tensorboard.py --domain harmony

    # Custom configuration
    python examples/train_with_tensorboard.py --epochs 100 --batch-size 64

    # View TensorBoard
    tensorboard --logdir output/semantic_discovery/logs

Requirements:
    - Normalized features (use NormalizedFeatureExtractor)
    - TensorBoard installed (pip install tensorboard)
    - GPU recommended (but not required)

Author: v2.0 Training Pipeline
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Import our modules
from midi_generator.learning.modular_encoder_factory import (
    ModularEncoderFactory,
    MusicalDimension,
    DimensionSpec
)
from midi_generator.learning.semantic_encoder import (
    EncoderConfig,
    SemanticFeatureEncoder
)
from midi_generator.feature_selection.enhanced_feature_extractor import (
    EnhancedFeatureExtractor,
    NormalizedFeatureExtractor
)
from midi_generator.utils.tensorboard_logger import create_logger


def load_normalized_features(
    midi_corpus_dir: Path,
    selection_file: Path,
    normalization_params: Optional[Path] = None,
    sample_size: int = 100
) -> tuple[NormalizedFeatureExtractor, np.ndarray, List[Path]]:
    """
    Load and normalize features from MIDI corpus.

    Args:
        midi_corpus_dir: Directory containing MIDI files
        selection_file: Path to selected_features_200.json
        normalization_params: Path to pre-computed normalization params (optional)
        sample_size: Number of files to use for normalization statistics

    Returns:
        Tuple of (normalized_extractor, features_array, midi_files)
    """
    print("\n" + "="*70)
    print("Loading and Normalizing Features")
    print("="*70)

    # Get MIDI files
    midi_files = list(midi_corpus_dir.glob('**/*.mid')) + list(midi_corpus_dir.glob('**/*.midi'))
    print(f"Found {len(midi_files)} MIDI files")

    if len(midi_files) == 0:
        raise ValueError(f"No MIDI files found in {midi_corpus_dir}")

    # Create base extractor
    base_extractor = EnhancedFeatureExtractor.from_selection_file(selection_file)

    # Create normalized extractor
    normalized_extractor = NormalizedFeatureExtractor(base_extractor)

    # Load or compute normalization parameters
    if normalization_params and normalization_params.exists():
        print(f"\nLoading normalization parameters from {normalization_params}")
        normalized_extractor.load_normalization_params(normalization_params)
    else:
        print(f"\nComputing normalization parameters from {sample_size} samples...")
        normalized_extractor.fit(
            midi_files=midi_files,
            sample_size=sample_size,
            show_progress=True
        )

        # Save for future use
        if normalization_params:
            normalized_extractor.save_normalization_params(normalization_params)

    # Extract normalized features
    print(f"\nExtracting normalized features from all {len(midi_files)} files...")
    features = normalized_extractor.extract_batch(
        midi_files=midi_files,
        show_progress=True
    )

    # Verify normalization
    print(f"\n✅ Feature extraction complete:")
    print(f"   Shape: {features.shape}")
    print(f"   Mean: {features.mean():.4f} (should be ~0.0)")
    print(f"   Std: {features.std():.4f} (should be ~1.0)")
    print(f"   Min: {features.min():.4f}")
    print(f"   Max: {features.max():.4f}")

    return normalized_extractor, features, midi_files


def create_dataloaders(
    features: np.ndarray,
    batch_size: int = 32,
    train_split: float = 0.8,
    random_seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        features: Feature array [n_samples, 220]
        batch_size: Batch size
        train_split: Proportion for training (rest is validation)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Shuffle and split
    n_samples = len(features)
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * train_split)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create tensors
    features_tensor = torch.FloatTensor(features)

    train_features = features_tensor[train_indices]
    val_features = features_tensor[val_indices]

    # Create datasets (autoencoder: input = target)
    train_dataset = TensorDataset(train_features, train_features)
    val_dataset = TensorDataset(val_features, val_features)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"\n📊 Dataset split:")
    print(f"   Training samples: {len(train_features)}")
    print(f"   Validation samples: {len(val_features)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")

    return train_loader, val_loader


def train_encoder(
    encoder: SemanticFeatureEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dimension_name: str,
    config: EncoderConfig,
    num_epochs: int = 50,
    device: str = 'cuda',
    log_dir: Path = Path('output/semantic_discovery/logs')
) -> Dict[str, float]:
    """
    Train a single domain encoder with TensorBoard logging.

    Args:
        encoder: SemanticFeatureEncoder to train
        train_loader: Training data loader
        val_loader: Validation data loader
        dimension_name: Name of musical dimension (for logging)
        config: Encoder configuration
        num_epochs: Number of training epochs
        device: Training device
        log_dir: TensorBoard log directory

    Returns:
        Dictionary with final metrics
    """
    print(f"\n{'='*70}")
    print(f"Training {dimension_name.upper()} Encoder")
    print(f"{'='*70}")

    # Move encoder to device
    encoder = encoder.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Create TensorBoard logger
    logger = create_logger(
        experiment_name=f'{dimension_name}_encoder_v2',
        log_dir=log_dir,
        enabled=True
    )

    # Log hyperparameters
    logger.log_hparams({
        'dimension': dimension_name,
        'input_dim': config.input_dim,
        'hidden_dim': config.hidden_dim,
        'num_semantic_features': config.num_semantic_features,
        'learning_rate': config.learning_rate,
        'dropout': config.dropout,
        'batch_size': config.batch_size,
        'num_epochs': num_epochs
    })

    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        encoder.train()
        train_loss = 0.0
        train_recon_loss = 0.0

        for batch_idx, (batch_features, batch_targets) in enumerate(train_loader):
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            # Forward pass
            encoded = encoder.encoder(batch_features)
            reconstructed = encoder.decoder(encoded)

            # Reconstruction loss
            recon_loss = nn.functional.mse_loss(reconstructed, batch_targets)

            # Sparsity loss (L1 on semantic features)
            sparsity_loss = torch.abs(encoded).mean()

            # Total loss
            loss = (
                config.reconstruction_weight * recon_loss +
                config.sparsity_weight * sparsity_loss
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon_loss += recon_loss.item()

        # Average training losses
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)

        # Validation phase
        encoder.eval()
        val_loss = 0.0
        val_recon_loss = 0.0

        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                # Forward pass
                encoded = encoder.encoder(batch_features)
                reconstructed = encoder.decoder(encoded)

                # Reconstruction loss
                recon_loss = nn.functional.mse_loss(reconstructed, batch_targets)
                val_loss += recon_loss.item()
                val_recon_loss += recon_loss.item()

        # Average validation losses
        val_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)

        # Log to TensorBoard
        logger.log_scalars({
            'Loss/train': train_loss,
            'Loss/val': val_loss,
            'Loss/reconstruction_train': train_recon_loss,
            'Loss/reconstruction_val': val_recon_loss,
            'Learning_Rate': optimizer.param_groups[0]['lr']
        }, epoch)

        # Print progress
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:8.4f} | "
              f"Val Loss: {val_loss:8.4f} | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Early stopping check
        if val_loss < 10.0:
            print(f"\n✅ Converged at epoch {epoch+1}! (loss < 10)")
            break

    # Close logger
    logger.close()

    # Final results
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training Complete: {dimension_name.upper()}")
    print(f"{'='*70}")
    print(f"Final training loss: {train_loss:.4f}")
    print(f"Final validation loss: {val_loss:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_time:.1f}s")
    print(f"{'='*70}\n")

    return {
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'training_time': total_time,
        'converged': val_loss < 10.0
    }


def main():
    parser = argparse.ArgumentParser(description='Train domain encoders with TensorBoard')
    parser.add_argument('--midi-corpus', type=Path,
                        default=Path('data/midi_corpus'),
                        help='Path to MIDI corpus directory')
    parser.add_argument('--selection-file', type=Path,
                        default=Path('output/selected_features_200.json'),
                        help='Path to selected_features_200.json')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('output/semantic_discovery'),
                        help='Output directory for models and logs')
    parser.add_argument('--domain', type=str, choices=['harmony', 'rhythm', 'form', 'orchestration', 'texture', 'cross_dimensional', 'all'],
                        default='all',
                        help='Which domain encoder to train')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')
    parser.add_argument('--normalization-sample-size', type=int, default=100,
                        help='Number of files for normalization statistics')

    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"\n🚀 Domain Encoder Training v2.0")
    print(f"   Device: {device}")
    print(f"   MIDI corpus: {args.midi_corpus}")
    print(f"   Output directory: {args.output_dir}")

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load and normalize features
    normalization_params = args.output_dir / 'normalization_params.json'
    extractor, features, midi_files = load_normalized_features(
        midi_corpus_dir=args.midi_corpus,
        selection_file=args.selection_file,
        normalization_params=normalization_params,
        sample_size=args.normalization_sample_size
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        features=features,
        batch_size=args.batch_size
    )

    # Create encoder factory
    factory = ModularEncoderFactory()

    # Determine which domains to train
    if args.domain == 'all':
        dimensions = [
            MusicalDimension.HARMONY,
            MusicalDimension.RHYTHM,
            MusicalDimension.FORM,
            MusicalDimension.ORCHESTRATION,
            MusicalDimension.TEXTURE,
            MusicalDimension.CROSS_DIMENSIONAL
        ]
    else:
        dimensions = [MusicalDimension(args.domain)]

    # Train each domain encoder
    results = {}
    for dimension in dimensions:
        # Create encoder
        encoder = factory.create_encoder(
            dimension=dimension,
            device=device
        )

        # Train encoder
        result = train_encoder(
            encoder=encoder,
            train_loader=train_loader,
            val_loader=val_loader,
            dimension_name=dimension.value,
            config=encoder.config,
            num_epochs=args.epochs,
            device=device,
            log_dir=log_dir
        )

        results[dimension.value] = result

        # Save model
        model_path = models_dir / f'{dimension.value}_encoder_v2.pt'
        torch.save(encoder.state_dict(), model_path)
        print(f"💾 Model saved to {model_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    for dimension_name, result in results.items():
        status = "✅ CONVERGED" if result['converged'] else "⚠️ NOT CONVERGED"
        print(f"{dimension_name.upper():20s} | Val Loss: {result['final_val_loss']:8.4f} | {status}")
    print(f"{'='*70}")

    print(f"\n🎉 Training complete!")
    print(f"\n📊 View TensorBoard:")
    print(f"   tensorboard --logdir {log_dir}")
    print(f"   Then open: http://localhost:6006")
    print(f"\n💾 Models saved to: {models_dir}")


if __name__ == '__main__':
    main()
