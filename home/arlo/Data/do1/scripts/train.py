#!/usr/bin/env python3
# Copyright 2024 DO1 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Training script for DO1.

Usage:
    python train.py --config configs/training.yaml --model configs/model_3b.yaml
    python train.py --config configs/training.yaml --model configs/model_small.yaml  # Debug

Multi-GPU:
    torchrun --nproc_per_node=4 train.py --config configs/training.yaml --model configs/model_3b.yaml
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import FSDPStrategy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from do1.training import DO1Pipeline


def load_yaml(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DO1 model")

    # Config files
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="configs/model_3b.yaml",
        help="Model configuration file",
    )
    parser.add_argument(
        "--training", "-t",
        type=str,
        default="configs/training.yaml",
        help="Training configuration file",
    )

    # Data paths
    parser.add_argument(
        "--latents_dir",
        type=str,
        required=True,
        help="Directory containing session latents",
    )
    parser.add_argument(
        "--fx_pairs_dir",
        type=str,
        default=None,
        help="Directory containing FX dry/wet pairs",
    )
    parser.add_argument(
        "--vst_synths_dir",
        type=str,
        default=None,
        help="Directory containing VST synth pairs",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default=None,
        help="Path to instrument labels JSON",
    )
    parser.add_argument(
        "--val_latents_dir",
        type=str,
        default=None,
        help="Directory containing validation latents",
    )

    # Output
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./output",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name (default: timestamp)",
    )

    # Training overrides
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Number of GPUs (default: all available)",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (small model, few steps)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load configurations
    model_config = load_yaml(args.model)
    training_config = load_yaml(args.training)

    # Debug overrides
    if args.debug:
        model_config = load_yaml("configs/model_small.yaml")
        training_config['max_steps'] = 100
        training_config['batch_size'] = 2
        training_config['num_workers'] = 0
        training_config['checkpoint_every'] = 50

    # Command-line overrides
    if args.batch_size is not None:
        training_config['batch_size'] = args.batch_size
    if args.max_steps is not None:
        training_config['max_steps'] = args.max_steps
    if args.gradient_accumulation is not None:
        training_config['gradient_accumulation'] = args.gradient_accumulation

    # Data config
    data_config = {
        'latents_dir': args.latents_dir,
        'fx_pairs_dir': args.fx_pairs_dir,
        'vst_synths_dir': args.vst_synths_dir,
        'labels_path': args.labels_path,
        'val_latents_dir': args.val_latents_dir,
        'max_time_frames': training_config.get('max_time_frames', 4096),
        'samples_per_epoch': training_config.get('samples_per_epoch', 100000),
    }

    # Create experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create output directory
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configs
    with open(output_dir / "model_config.yaml", "w") as f:
        yaml.dump(model_config, f)
    with open(output_dir / "training_config.yaml", "w") as f:
        yaml.dump(training_config, f)

    # Create model
    pipeline = DO1Pipeline(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="do1-{step:07d}",
        every_n_train_steps=training_config.get('checkpoint_every', 5000),
        save_top_k=training_config.get('save_top_k', 3),
        monitor="train/loss",
        mode="min",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
        version="",
    )

    # Determine number of devices
    if args.devices is not None:
        devices = args.devices
    else:
        devices = torch.cuda.device_count() or 1

    # Strategy
    if devices > 1:
        # Use FSDP for multi-GPU training
        from pytorch_lightning.strategies import FSDPStrategy
        from torch.distributed.fsdp import MixedPrecision

        # Import the transformer block for wrapping
        from do1.models.layers import DO1TransformerBlock

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        strategy = FSDPStrategy(
            auto_wrap_policy={DO1TransformerBlock},
            activation_checkpointing_policy={DO1TransformerBlock},
            mixed_precision=mixed_precision,
            sharding_strategy="SHARD_GRAD_OP",
        )
    else:
        strategy = "auto"

    # Trainer
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        strategy=strategy,
        precision=training_config.get('precision', 'bf16-mixed'),
        max_steps=training_config['max_steps'],
        accumulate_grad_batches=training_config.get('gradient_accumulation', 4),
        gradient_clip_val=training_config.get('grad_clip', 1.0),
        log_every_n_steps=training_config.get('log_every', 100),
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    trainer.fit(pipeline, ckpt_path=args.resume)

    print(f"\nTraining complete! Checkpoints saved to: {output_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
