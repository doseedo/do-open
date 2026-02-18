#!/usr/bin/env python3
"""Validate training data and model pipeline."""
import argparse
import json
import sys
import torch

# Add ACE-Step to path
sys.path.insert(0, "/home/arlo/Data/ACE-Step")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='/home/arlo/gcs-bucket/inverse_afx_training_data/manifest.json')
    parser.add_argument('--model-size', default='tiny', choices=['tiny', 'small', 'base'])
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()

    from inverse_afx.training.train_latent_flow import (
        LatentFlowDataModule, LatentFlowTrainingConfig, LatentFlowInverterSystem
    )

    # Data
    print(f"Loading {args.manifest}...")
    dm = LatentFlowDataModule(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        sample_rate=44100,
        segment_length=88200,
        num_workers=0,
    )
    dm.setup()
    print(f"Train: {len(dm.train_dataset)}, Val: {len(dm.val_dataset)}")

    # Batch
    batch = next(iter(dm.train_dataloader()))
    print(f"Batch: wet={batch['wet_audio'].shape}, dry={batch['dry_audio'].shape}")

    # Model
    print(f"\nCreating {args.model_size} model...")
    config = LatentFlowTrainingConfig(model_size=args.model_size, compute_audio_loss=False)
    system = LatentFlowInverterSystem(config, sample_rate=44100)
    print(f"Trainable params: {sum(p.numel() for p in system.parameters() if p.requires_grad):,}")

    # Forward
    print("Testing forward...")
    with torch.no_grad():
        out = system(batch['wet_audio'], batch['effect_types'], batch['effect_params'], n_steps=2)
    print(f"Output: {out.shape}")

    # Training step
    print("Testing training step...")
    loss = system.training_step(batch, 0)
    print(f"Loss: {loss.item():.6f}")

    print("\n✓ Ready to train!")

if __name__ == '__main__':
    main()
