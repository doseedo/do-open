#!/usr/bin/env python3
"""
Training script for Inverse Audio Effects System.

Stage 2+: Full system training with parameter estimation and chain inversion.
Can load pretrained encoder from Stage 1.
"""

import argparse
import os
import yaml
from pathlib import Path

# Parse args FIRST before any imports that trigger CLAP
def parse_args():
    parser = argparse.ArgumentParser(description="Train Inverse AFx System (Stage 2+)")

    # Data arguments
    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default="/mnt/models/inverse_afx_data",
        help="Directory containing generated data",
    )
    parser.add_argument(
        "--manifest", "-m",
        type=str,
        default=None,
        help="Path to manifest.json (default: data_dir/manifest.json)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="precomputed",
        choices=["online", "precomputed"],
        help="Data loading mode",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--max_epochs", "-e",
        type=int,
        default=100,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--learning_rate", "-lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_workers", "-w",
        type=int,
        default=8,
        help="Number of data loader workers",
    )

    # Model arguments
    parser.add_argument(
        "--max_chain_length",
        type=int,
        default=4,
        help="Maximum effect chain length",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=512,
        help="Encoder embedding dimension",
    )
    parser.add_argument(
        "--effect_types",
        type=str,
        nargs="+",
        default=["eq", "compressor", "reverb", "distortion", "chorus", "delay"],
        help="Effect types to train on (use single effect for debugging, e.g., --effect_types reverb)",
    )
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        default=None,
        help="Path to Stage 1 encoder checkpoint to load",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder weights (use with --encoder_ckpt)",
    )

    # Logging arguments
    parser.add_argument(
        "--exp_name", "-n",
        type=str,
        default="inverse_afx_stage2",
        help="Experiment name",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Logging directory",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases logging",
    )

    # Hardware arguments
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices (GPUs)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster training (requires PyTorch 2.0+)",
    )
    parser.add_argument(
        "--accumulate_grad",
        type=int,
        default=1,
        help="Accumulate gradients over N batches (effective batch = batch_size * N)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=48000,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=144000,
        help="Audio segment length in samples",
    )

    return parser.parse_args()

# Parse args before any heavy imports
args = parse_args()

# Now import everything else
import sys
sys.argv = [sys.argv[0]]  # Clear args to prevent CLAP hijacking

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inverse_afx.data.datasets import InverseAFxDataModule
from inverse_afx.training.train_system import InverseAFxSystem, TrainingConfig
from inverse_afx.models.effect_encoder import EffectEncoderConfig
from inverse_afx.models.chain_estimator import ChainEstimatorConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_encoder_weights(model, checkpoint_path):
    """Load Stage 1 encoder weights into the model."""
    print(f"Loading encoder weights from: {checkpoint_path}")

    # Load checkpoint (weights_only=False for Lightning checkpoints)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Extract encoder weights
    encoder_state = {}
    for key, value in state_dict.items():
        if key.startswith('encoder.'):
            new_key = key  # Keep the encoder. prefix
            encoder_state[new_key] = value

    # Load into model's encoder (in chain_estimator)
    missing, unexpected = model.chain_estimator.load_state_dict(encoder_state, strict=False)

    print(f"  Loaded {len(encoder_state)} encoder weights")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    return model


def main():
    global args

    # Setup paths
    data_dir = Path(args.data_dir)
    manifest_path = args.manifest or str(data_dir / "manifest.json")

    if args.mode == 'precomputed' and not Path(manifest_path).exists():
        print(f"Error: Manifest not found at {manifest_path}")
        print("Make sure data generation is complete.")
        return

    # Load configuration
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = {}

    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Create data module
    data_module = InverseAFxDataModule(
        train_dir=data_dir,
        val_dir=data_dir,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_chain_length=args.max_chain_length,
        effect_types=args.effect_types,
        mode=args.mode,
        train_manifest=manifest_path,
        val_manifest=manifest_path,
        load_dry=True,  # Need dry audio for Stage 2
    )

    # Create model configuration
    encoder_config = EffectEncoderConfig(
        embedding_dim=args.embedding_dim,
        num_effect_types=len(args.effect_types),
        sample_rate=args.sample_rate,
    )

    estimator_config = ChainEstimatorConfig(
        encoder_config=encoder_config,
        max_iterations=args.max_chain_length,
        effect_types=args.effect_types,
        sample_rate=args.sample_rate,
    )

    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        use_curriculum=config.get('curriculum', {}).get('stages') is not None,
        curriculum_stages=config.get('curriculum', {}).get('stages'),
    )

    # Create model
    model = InverseAFxSystem(
        encoder_config=encoder_config,
        estimator_config=estimator_config,
        training_config=training_config,
        effect_types=args.effect_types,
        sample_rate=args.sample_rate,
    )

    # Load Stage 1 encoder if provided
    if args.encoder_ckpt:
        model = load_encoder_weights(model, args.encoder_ckpt)

        if args.freeze_encoder:
            print("Freezing encoder weights")
            for param in model.chain_estimator.encoder.parameters():
                param.requires_grad = False

    # Compile model for faster training
    if args.compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"{args.exp_name}-{{epoch:02d}}-{{val/si_sdr:.2f}}",
            monitor="val/si_sdr",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/si_sdr",
            mode="max",
            patience=20,
            check_finite=False,  # Allow -inf early in training
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    if args.wandb:
        logger = WandbLogger(
            project="inverse-afx",
            name=args.exp_name,
            save_dir=args.log_dir,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name=args.exp_name,
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=0.5,  # Lowered from 1.0 to prevent gradient explosion
        log_every_n_steps=50,
        val_check_interval=1.0,  # Validate once per epoch
        accumulate_grad_batches=args.accumulate_grad,
    )

    # Print training info
    print("=" * 60)
    print("Stage 2+: Full Inverse AFx System Training")
    print("=" * 60)
    print(f"Data: {manifest_path}")
    print(f"Mode: {args.mode}")
    print(f"Sample rate: {args.sample_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Effect types: {args.effect_types}")
    if args.encoder_ckpt:
        print(f"Encoder checkpoint: {args.encoder_ckpt}")
        print(f"Freeze encoder: {args.freeze_encoder}")
    print("=" * 60)

    # Train
    trainer.fit(
        model,
        data_module,
        ckpt_path=args.resume,
    )

    print(f"\nTraining complete!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
