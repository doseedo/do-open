#!/usr/bin/env python3
"""
V5 Pitch Shift Corrector Training

Key changes from V4:
1. Hard loudness lock (mandatory) - kills "boost everything" shortcut
2. Envelope loss - smoothed spectral profile comparison
3. Shift conditioning enabled by default

Losses:
- Distribution loss (normalized shapes)
- Loudness loss (RMS match to SOURCE)
- Envelope loss (smoothed log-spectrum to TARGET)
- Small residual L2 regularization
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset_simple import PitchShiftCorrectionDatasetSimple
from models_simple import (
    PitchShiftCorrectorSimple,
    DistributionLoss,
    LoudnessLoss,
    EnvelopeLoss,
)


class TrainerV5:
    """
    V5 Trainer for pitch-shift register correction.

    Key constraints:
    - Loudness locked to SOURCE (no energy boosting)
    - Envelope matched to TARGET (formant correction)
    - Shift-conditioned by default
    """

    def __init__(
        self,
        shifted_manifest: str,
        segments_json: str,
        output_dir: str,
        batch_size: int = 64,
        num_epochs: int = 100,
        samples_per_epoch: int = 10000,
        learning_rate: float = 1e-4,
        window_frames: int = 64,
        checkpoint_every: int = 10,
        dist_weight: float = 1.0,
        loudness_weight: float = 10.0,  # High weight - hard constraint
        envelope_weight: float = 1.0,
        residual_reg_weight: float = 1e-4,
        device: str = 'cuda',
        num_workers: int = 4,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        flagged_json: str = None,
        pitch_tolerance: float = 1.0,
        conditioning: bool = True,  # Default ON for V5
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.checkpoint_every = checkpoint_every
        self.best_loss = float('inf')
        self.conditioning = conditioning

        # Loss weights
        self.dist_weight = dist_weight
        self.loudness_weight = loudness_weight
        self.envelope_weight = envelope_weight
        self.residual_reg_weight = residual_reg_weight

        # Create dataset
        print("Creating dataset...")
        self.dataset = PitchShiftCorrectionDatasetSimple(
            shifted_manifest=shifted_manifest,
            segments_json=segments_json,
            window_frames=window_frames,
            samples_per_epoch=samples_per_epoch,
            preload_latents=True,
            flagged_json=flagged_json,
            pitch_tolerance=pitch_tolerance,
        )

        # DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Create model
        mode_str = "FiLM-conditioned" if conditioning else "unconditioned"
        print(f"Creating PitchShiftCorrectorSimple model ({mode_str})...")
        self.model = PitchShiftCorrectorSimple(
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            conditioning=conditioning,
        )
        self.model = self.model.to(self.device)
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Loss functions
        self.dist_loss_fn = DistributionLoss()
        self.loudness_loss_fn = LoudnessLoss()
        self.envelope_loss_fn = EnvelopeLoss()

        # Optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=learning_rate / 10)

        # Save config
        config = {
            'shifted_manifest': shifted_manifest,
            'segments_json': segments_json,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'samples_per_epoch': samples_per_epoch,
            'learning_rate': learning_rate,
            'window_frames': window_frames,
            'hidden_dim': hidden_dim,
            'num_blocks': num_blocks,
            'dist_weight': dist_weight,
            'loudness_weight': loudness_weight,
            'envelope_weight': envelope_weight,
            'residual_reg_weight': residual_reg_weight,
            'conditioning': conditioning,
            'version': 'v5-loudness-envelope',
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Logging
        self.log_file = self.output_dir / 'training.log'

    def log(self, message: str):
        """Log to file and stdout."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        valid_mask = batch['valid']
        if not valid_mask.any():
            return {'loss': 0.0}

        # Move to device
        source_latent = batch['source_latent'].to(self.device)
        target_latent = batch['target_latent'].to(self.device)
        shift = batch['shift'].to(self.device)

        # Filter valid samples
        source_latent = source_latent[valid_mask]
        target_latent = target_latent[valid_mask]
        shift = shift[valid_mask]

        # Skip if NaN
        if torch.isnan(source_latent).any() or torch.isnan(target_latent).any():
            return {'loss': 0.0}

        # Forward pass
        self.model.train()
        self.optimizer.zero_grad()

        pred_latent = self.model(source_latent, shift if self.conditioning else None)

        if torch.isnan(pred_latent).any():
            return {'loss': 0.0}

        # Losses
        # 1. Distribution matching (normalized shapes to TARGET)
        dist_losses = self.dist_loss_fn(pred_latent, target_latent)
        dist_loss = dist_losses['total']

        # 2. Loudness lock to SOURCE (hard constraint)
        loudness_loss = self.loudness_loss_fn(pred_latent, source_latent)

        # 3. Envelope matching to TARGET (formant correction)
        envelope_loss = self.envelope_loss_fn(pred_latent, target_latent)

        # 4. Residual regularization
        residual = pred_latent - source_latent
        residual_reg = residual.pow(2).mean()

        # Combined loss
        total_loss = (
            self.dist_weight * dist_loss +
            self.loudness_weight * loudness_loss +
            self.envelope_weight * envelope_loss +
            self.residual_reg_weight * residual_reg
        )

        if torch.isnan(total_loss):
            return {'loss': 0.0}

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Residual magnitude for monitoring
        residual_mag = residual.abs().mean().item()

        return {
            'loss': total_loss.item(),
            'dist': dist_loss.item(),
            'loud': loudness_loss.item(),
            'env': envelope_loss.item(),
            'res_mag': residual_mag,
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        totals = {
            'loss': 0.0, 'dist': 0.0, 'loud': 0.0, 'env': 0.0, 'res_mag': 0.0
        }
        num_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            metrics = self.train_step(batch)

            if metrics['loss'] > 0:
                for k, v in metrics.items():
                    totals[k] += v
                num_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics.get('loss', 0):.4f}",
                'loud': f"{metrics.get('loud', 0):.4f}",
                'env': f"{metrics.get('env', 0):.4f}",
            })

        self.scheduler.step()

        return {k: v / max(num_batches, 1) for k, v in totals.items()}

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'version': 'v5',
        }

        torch.save(checkpoint, self.output_dir / 'latest.pt')

        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pt')

        if epoch % self.checkpoint_every == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch{epoch}.pt')

    def train(self):
        """Full training loop."""
        self.log("=" * 60)
        self.log("PITCH SHIFT CORRECTOR V5")
        self.log("Hard loudness lock + Envelope loss + Conditioning")
        self.log("=" * 60)
        self.log(f"Output: {self.output_dir}")
        self.log(f"Device: {self.device}")
        self.log(f"Conditioning: {self.conditioning}")
        self.log(f"Valid pairs: {len(self.dataset.valid_pairs)}")
        self.log(f"Epochs: {self.num_epochs}")
        self.log(f"Weights: dist={self.dist_weight}, loud={self.loudness_weight}, env={self.envelope_weight}")
        self.log("=" * 60)

        for epoch in range(1, self.num_epochs + 1):
            metrics = self.train_epoch(epoch)

            is_best = metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = metrics['loss']

            self.save_checkpoint(epoch, metrics, is_best)

            self.log(
                f"Epoch {epoch:3d} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Dist: {metrics['dist']:.4f} | "
                f"Loud: {metrics['loud']:.4f} | "
                f"Env: {metrics['env']:.4f}"
                + (" [BEST]" if is_best else "")
            )

        self.log("")
        self.log("Training complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train V5 Pitch Shift Corrector")

    parser.add_argument('--shifted_manifest', type=str, required=True,
                        help='Path to precomputed shifted latents manifest')
    parser.add_argument('--segments', type=str, required=True,
                        help='Path to segments JSON (for target lookup)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--samples_per_epoch', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--window_frames', type=int, default=64)
    parser.add_argument('--checkpoint_every', type=int, default=10)
    parser.add_argument('--dist_weight', type=float, default=1.0)
    parser.add_argument('--loudness_weight', type=float, default=10.0,
                        help='Weight for loudness lock (high = hard constraint)')
    parser.add_argument('--envelope_weight', type=float, default=1.0,
                        help='Weight for envelope/formant matching')
    parser.add_argument('--residual_reg_weight', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_blocks', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--flagged_json', type=str, default=None,
                        help='Path to ensemble_detection_results.json to exclude flagged recordings')
    parser.add_argument('--pitch_tolerance', type=float, default=1.0,
                        help='Pitch tolerance in semitones for target matching')
    parser.add_argument('--no_conditioning', action='store_true',
                        help='Disable FiLM conditioning (not recommended)')

    args = parser.parse_args()

    trainer = TrainerV5(
        shifted_manifest=args.shifted_manifest,
        segments_json=args.segments,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        samples_per_epoch=args.samples_per_epoch,
        learning_rate=args.learning_rate,
        window_frames=args.window_frames,
        checkpoint_every=args.checkpoint_every,
        dist_weight=args.dist_weight,
        loudness_weight=args.loudness_weight,
        envelope_weight=args.envelope_weight,
        residual_reg_weight=args.residual_reg_weight,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        device=args.device,
        num_workers=args.num_workers,
        flagged_json=args.flagged_json,
        pitch_tolerance=args.pitch_tolerance,
        conditioning=not args.no_conditioning,
    )

    trainer.train()


if __name__ == "__main__":
    main()
