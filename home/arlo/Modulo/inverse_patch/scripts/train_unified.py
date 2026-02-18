#!/usr/bin/env python3
"""
Training script for Unified HDemucs Inverter.

Usage:
    python train_unified.py --manifest /path/to/manifest.json --exp_name unified_v1

    # Multi-GPU:
    CUDA_VISIBLE_DEVICES=0,1 python train_unified.py --devices 2 ...
"""

import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inverse_afx.training.train_unified import (
    UnifiedInverterSystem,
    UnifiedTrainingConfig,
    UnifiedDataModule,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Unified Inverter")

    # Data
    parser.add_argument("--manifest", "-m", type=str, required=True,
                        help="Path to manifest.json")
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--segment_length", type=int, default=144000)

    # Training
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--num_workers", "-w", type=int, default=8)
    parser.add_argument("--max_epochs", "-e", type=int, default=50)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--accumulate_grad", type=int, default=4)

    # Model
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["small", "base", "large"])

    # Curriculum learning
    parser.add_argument("--max_chain_length", type=int, default=None,
                        help="Max chain length (1=single effect, None=all)")
    parser.add_argument("--effect_types", type=str, nargs="+", default=None,
                        help="Filter to specific effects (e.g., eq reverb)")

    # Hardware
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"])

    # Logging
    parser.add_argument("--exp_name", "-n", type=str, default="unified_inverter")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Config
    config = UnifiedTrainingConfig(
        learning_rate=args.learning_rate,
        model_size=args.model_size,
        max_steps=args.max_epochs * 5000,  # Approximate
    )

    # Data
    data_module = UnifiedDataModule(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        max_chain_length=args.max_chain_length,
        effect_types=args.effect_types,
    )

    # Model
    model = UnifiedInverterSystem(
        config=config,
        sample_rate=args.sample_rate,
    )

    # Print model info
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Unified HDemucs Inverter Training")
    print(f"{'='*60}")
    print(f"Model size: {args.model_size} ({params:,} params)")
    print(f"Manifest: {args.manifest}")
    print(f"Batch size: {args.batch_size} x {args.accumulate_grad} = {args.batch_size * args.accumulate_grad}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Devices: {args.devices}")
    if args.max_chain_length:
        print(f"Curriculum: max_chain_length={args.max_chain_length}")
    if args.effect_types:
        print(f"Curriculum: effect_types={args.effect_types}")
    print(f"{'='*60}\n")

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
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
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
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad,
        log_every_n_steps=50,
        val_check_interval=1.0,
    )

    # Train
    trainer.fit(model, data_module, ckpt_path=args.resume)

    print(f"\nTraining complete!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
