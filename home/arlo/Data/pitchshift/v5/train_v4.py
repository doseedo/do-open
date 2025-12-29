#!/usr/bin/env python3
"""
V4 Pitch Shift Corrector Training

Key differences from V3:
1. Input is DSP-pitch-shifted latent (has artifacts)
2. Target is real latent at destination pitch
3. Conditioning on shift amount (continuous) not group (discrete)
4. Identity loss when shift ≈ 0

The model learns: "given audio shifted by s semitones, fix the artifacts"
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

from dataset_v4 import PitchShiftCorrectionDataset
from models_v4 import (
    PitchShiftCorrector,
    DistributionLoss,
    SilenceLoss,
    EnvelopePreservationLoss,
    ContentPreservationLoss,
    AlphaRegularizationLoss,
)


class TrainerV4:
    """
    Trainer for V4 pitch-shift artifact correction.

    Training strategy:
    - Distribution matching to target (real audio at destination pitch)
    - Identity loss when |shift| is small
    - Silence and envelope preservation
    - Residual regularization
    """

    def __init__(
        self,
        segments_json: str,
        output_dir: str,
        batch_size: int = 64,
        num_epochs: int = 100,
        samples_per_epoch: int = 10000,
        learning_rate: float = 1e-4,
        window_frames: int = 64,
        max_shift: int = 12,
        checkpoint_every: int = 10,
        dist_weight: float = 1.0,
        silence_weight: float = 1.0,
        envelope_weight: float = 0.2,
        identity_weight: float = 0.5,
        identity_shift_threshold: int = 2,  # Apply identity loss when |shift| <= this
        residual_reg_weight: float = 1e-4,
        alpha_reg_weight: float = 0.1,  # Regularize alpha to follow expected pattern
        shifted_manifest: str = None,  # Path to precomputed shifted latents
        device: str = 'cuda',
        num_workers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.checkpoint_every = checkpoint_every
        self.best_loss = float('inf')

        # Loss weights
        self.dist_weight = dist_weight
        self.silence_weight = silence_weight
        self.envelope_weight = envelope_weight
        self.identity_weight = identity_weight
        self.identity_shift_threshold = identity_shift_threshold
        self.residual_reg_weight = residual_reg_weight
        self.alpha_reg_weight = alpha_reg_weight
        self.max_shift = max_shift

        # Create dataset
        print("Creating dataset...")
        self.dataset = PitchShiftCorrectionDataset(
            segments_json=segments_json,
            window_frames=window_frames,
            samples_per_epoch=samples_per_epoch,
            max_shift=max_shift,
            min_shift=-max_shift,
            preload_latents=True,
            shifted_manifest=shifted_manifest,
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
        print("Creating PitchShiftCorrector model...")
        self.model = PitchShiftCorrector()
        self.model = self.model.to(self.device)
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Loss functions
        self.dist_loss_fn = DistributionLoss()
        self.silence_loss_fn = SilenceLoss()
        self.envelope_loss_fn = EnvelopePreservationLoss()
        self.alpha_reg_loss_fn = AlphaRegularizationLoss(max_shift=float(max_shift))

        # Optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=learning_rate / 10)

        # Save config
        config = {
            'segments_json': segments_json,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'samples_per_epoch': samples_per_epoch,
            'learning_rate': learning_rate,
            'window_frames': window_frames,
            'max_shift': max_shift,
            'dist_weight': dist_weight,
            'silence_weight': silence_weight,
            'envelope_weight': envelope_weight,
            'identity_weight': identity_weight,
            'identity_shift_threshold': identity_shift_threshold,
            'residual_reg_weight': residual_reg_weight,
            'alpha_reg_weight': alpha_reg_weight,
            'version': 'v4-shift-correction',
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

        pred_latent = self.model(source_latent, shift)

        if torch.isnan(pred_latent).any():
            return {'loss': 0.0}

        # Losses
        # 1. Distribution matching to target
        dist_loss = self.dist_loss_fn(pred_latent, target_latent)

        # 2. Silence preservation
        silence_loss = self.silence_loss_fn(source_latent, pred_latent)

        # 3. Envelope preservation
        envelope_loss = self.envelope_loss_fn(pred_latent, target_latent)

        # 4. Identity loss: when |shift| is small, output should match input
        # This prevents over-correction for small shifts
        small_shift_mask = shift.abs() <= self.identity_shift_threshold
        if small_shift_mask.any():
            identity_loss = torch.nn.functional.l1_loss(
                pred_latent[small_shift_mask],
                source_latent[small_shift_mask]
            )
        else:
            identity_loss = torch.tensor(0.0, device=self.device)

        # 5. Residual regularization
        residual_reg = (pred_latent - source_latent).pow(2).mean()

        # 6. Alpha regularization: encourage alpha to follow expected pattern
        # (high alpha for small shift, low alpha for large shift)
        alphas = self.model.get_alpha(source_latent, shift)
        alpha_reg_loss = self.alpha_reg_loss_fn(alphas, shift)

        # Combined loss
        total_loss = (
            self.dist_weight * dist_loss +
            self.silence_weight * silence_loss +
            self.envelope_weight * envelope_loss +
            self.identity_weight * identity_loss +
            self.residual_reg_weight * residual_reg +
            self.alpha_reg_weight * alpha_reg_loss
        )

        if torch.isnan(total_loss):
            return {'loss': 0.0}

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Get alpha stats for logging
        alpha_mean = alphas.mean().item()

        return {
            'loss': total_loss.item(),
            'dist': dist_loss.item(),
            'silence': silence_loss.item(),
            'envelope': envelope_loss.item(),
            'identity': identity_loss.item(),
            'res_reg': residual_reg.item(),
            'alpha_reg': alpha_reg_loss.item(),
            'alpha': alpha_mean,
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        totals = {
            'loss': 0.0, 'dist': 0.0, 'silence': 0.0,
            'envelope': 0.0, 'identity': 0.0, 'res_reg': 0.0,
            'alpha_reg': 0.0, 'alpha': 0.0
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
                'dist': f"{metrics.get('dist', 0):.4f}",
                'alpha': f"{metrics.get('alpha', 0):.2f}",
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
            'version': 'v4',
        }

        torch.save(checkpoint, self.output_dir / 'latest.pt')

        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pt')

        if epoch % self.checkpoint_every == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch{epoch}.pt')

    def train(self):
        """Full training loop."""
        self.log("=" * 60)
        self.log("PITCH SHIFT CORRECTOR V4")
        self.log("Learns to fix DSP pitch-shift artifacts")
        self.log("=" * 60)
        self.log(f"Output: {self.output_dir}")
        self.log(f"Device: {self.device}")
        self.log(f"Total segments: {len(self.dataset.all_segments)}")
        self.log(f"Valid groups: {self.dataset.valid_groups}")
        self.log(f"Epochs: {self.num_epochs}")
        self.log(f"Weights: dist={self.dist_weight}, silence={self.silence_weight}, "
                 f"envelope={self.envelope_weight}, identity={self.identity_weight}, "
                 f"res_reg={self.residual_reg_weight}")
        self.log(f"Identity threshold: |shift| <= {self.identity_shift_threshold}")
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
                f"Id: {metrics['identity']:.4f} | "
                f"Alpha: {metrics['alpha']:.2f}"
                + (" [BEST]" if is_best else "")
            )

        self.log("")
        self.log("Training complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train V4 Pitch Shift Corrector")

    parser.add_argument('--segments', type=str, required=True,
                        help='Path to segments JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--samples_per_epoch', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--window_frames', type=int, default=64)
    parser.add_argument('--max_shift', type=int, default=12,
                        help='Maximum shift in semitones')
    parser.add_argument('--checkpoint_every', type=int, default=10)
    parser.add_argument('--dist_weight', type=float, default=1.0)
    parser.add_argument('--silence_weight', type=float, default=1.0)
    parser.add_argument('--envelope_weight', type=float, default=0.2)
    parser.add_argument('--identity_weight', type=float, default=0.5)
    parser.add_argument('--identity_shift_threshold', type=int, default=2,
                        help='Apply identity loss when |shift| <= this')
    parser.add_argument('--residual_reg_weight', type=float, default=1e-4)
    parser.add_argument('--alpha_reg_weight', type=float, default=0.1,
                        help='Weight for alpha regularization loss')
    parser.add_argument('--shifted_manifest', type=str, default=None,
                        help='Path to precomputed shifted latents manifest (recommended)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    trainer = TrainerV4(
        segments_json=args.segments,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        samples_per_epoch=args.samples_per_epoch,
        learning_rate=args.learning_rate,
        window_frames=args.window_frames,
        max_shift=args.max_shift,
        checkpoint_every=args.checkpoint_every,
        dist_weight=args.dist_weight,
        silence_weight=args.silence_weight,
        envelope_weight=args.envelope_weight,
        identity_weight=args.identity_weight,
        identity_shift_threshold=args.identity_shift_threshold,
        residual_reg_weight=args.residual_reg_weight,
        alpha_reg_weight=args.alpha_reg_weight,
        shifted_manifest=args.shifted_manifest,
        device=args.device,
        num_workers=args.num_workers,
    )

    trainer.train()


if __name__ == "__main__":
    main()
