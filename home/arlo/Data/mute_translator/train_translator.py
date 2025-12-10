#!/usr/bin/env python3
"""
Train Mute Translator

Step 1 of the mute conversion pipeline.
Trains a latent space translator to map dry trumpet → muted trumpet.

Usage:
    python train_translator.py --manifest /path/to/manifest.json --output_dir ./checkpoints

The translator uses distribution matching:
- Input: dry trumpet latent
- Output: latent that matches muted trumpet distribution
- Loss: MSE to muted distribution + perceptual losses
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/mute_translator')

from models import MuteTranslator, MuteTranslatorLarge, MuteDiscriminator
from dataset import MuteTranslatorDataset, load_manifest


class MuteTranslatorTrainer:
    """
    Trainer for the mute translator.

    Training strategy:
    1. Distribution matching: Make translated latents match muted distribution
    2. Content preservation: Translated latent should preserve structure of input
    3. Optional adversarial: Discriminator to improve realism
    """

    def __init__(
        self,
        manifest_path: str,
        output_dir: str,
        model_type: str = "small",  # "small" or "large"
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        window_frames: int = 128,
        samples_per_epoch: int = 5000,
        use_adversarial: bool = False,
        device: str = "cuda",
        num_workers: int = 4,
    ):
        self.manifest_path = manifest_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch
        self.use_adversarial = use_adversarial
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers

        print(f"Device: {self.device}")

        # Create model
        if model_type == "large":
            self.model = MuteTranslatorLarge().to(self.device)
        else:
            self.model = MuteTranslator().to(self.device)

        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Optional discriminator
        self.discriminator = None
        if use_adversarial:
            self.discriminator = MuteDiscriminator().to(self.device)
            print(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")

        # Create dataset
        self.dataset = MuteTranslatorDataset(
            manifest_path=manifest_path,
            window_frames=window_frames,
            samples_per_epoch=samples_per_epoch,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Compute muted distribution statistics for matching
        self._compute_muted_stats()

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        if self.discriminator:
            self.optimizer_d = AdamW(self.discriminator.parameters(), lr=learning_rate * 0.5)

        # Logging
        self.log_file = self.output_dir / "training.log"
        self.best_loss = float('inf')

    def _compute_muted_stats(self):
        """Compute mean and std of muted latent distribution."""
        print("Computing muted distribution statistics...")

        all_latents = []
        for entry in self.dataset.muted_entries[:100]:  # Sample subset for speed
            latent = self.dataset._load_latent(entry)
            if latent is not None:
                if latent.dim() == 4:
                    latent = latent.squeeze(0)
                all_latents.append(latent)

        if len(all_latents) > 0:
            # Concatenate along time dimension
            concat = torch.cat([l.reshape(-1) for l in all_latents])
            self.muted_mean = concat.mean().item()
            self.muted_std = concat.std().item()

            # Per-channel stats
            stacked = torch.stack([l.mean(dim=(1, 2)) for l in all_latents])  # [N, C]
            self.muted_channel_mean = stacked.mean(dim=0).to(self.device)  # [C]
            self.muted_channel_std = stacked.std(dim=0).to(self.device)  # [C]

            print(f"  Muted mean: {self.muted_mean:.4f}, std: {self.muted_std:.4f}")
        else:
            self.muted_mean = 0.0
            self.muted_std = 1.0
            self.muted_channel_mean = torch.zeros(8, device=self.device)
            self.muted_channel_std = torch.ones(8, device=self.device)

    def distribution_loss(self, pred_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
        """
        Loss to match distribution of predicted latents to muted distribution.

        Uses:
        - MSE between means
        - MSE between stds
        - MMD (Maximum Mean Discrepancy) for distribution matching
        """
        # Per-channel mean/std matching
        pred_mean = pred_latent.mean(dim=(2, 3))  # [B, C]
        pred_std = pred_latent.std(dim=(2, 3))  # [B, C]
        target_mean = target_latent.mean(dim=(2, 3))
        target_std = target_latent.std(dim=(2, 3))

        mean_loss = F.mse_loss(pred_mean, target_mean)
        std_loss = F.mse_loss(pred_std, target_std)

        # Simple MMD approximation
        pred_flat = pred_latent.reshape(pred_latent.size(0), -1)
        target_flat = target_latent.reshape(target_latent.size(0), -1)

        # Gram matrices
        pred_gram = torch.mm(pred_flat, pred_flat.t())
        target_gram = torch.mm(target_flat, target_flat.t())
        cross_gram = torch.mm(pred_flat, target_flat.t())

        mmd = pred_gram.mean() + target_gram.mean() - 2 * cross_gram.mean()

        return mean_loss + std_loss + 0.1 * mmd

    def content_preservation_loss(
        self,
        dry_latent: torch.Tensor,
        pred_latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure translated latent preserves structure of input.

        The translation should modify timbre, not melody/rhythm.
        We compare temporal structure (gradients) between input and output.
        """
        # Temporal gradient (rhythm preservation)
        dry_grad = dry_latent[:, :, :, 1:] - dry_latent[:, :, :, :-1]
        pred_grad = pred_latent[:, :, :, 1:] - pred_latent[:, :, :, :-1]

        # We want similar temporal dynamics
        grad_loss = F.mse_loss(pred_grad.abs(), dry_grad.abs())

        # Spectral gradient (pitch contour preservation)
        dry_spec_grad = dry_latent[:, :, 1:, :] - dry_latent[:, :, :-1, :]
        pred_spec_grad = pred_latent[:, :, 1:, :] - pred_latent[:, :, :-1, :]
        spec_loss = F.mse_loss(pred_spec_grad.abs(), dry_spec_grad.abs())

        return grad_loss + spec_loss

    def adversarial_loss(
        self,
        pred_latent: torch.Tensor,
        real_muted: torch.Tensor,
        train_discriminator: bool = True
    ) -> tuple:
        """Adversarial loss for more realistic translations."""
        if self.discriminator is None:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        if train_discriminator:
            # Train discriminator
            self.discriminator.train()
            real_logits = self.discriminator(real_muted.detach())
            fake_logits = self.discriminator(pred_latent.detach())

            d_loss = F.binary_cross_entropy_with_logits(
                real_logits, torch.ones_like(real_logits)
            ) + F.binary_cross_entropy_with_logits(
                fake_logits, torch.zeros_like(fake_logits)
            )
        else:
            d_loss = torch.tensor(0.0, device=self.device)

        # Generator loss (fool discriminator)
        fake_logits = self.discriminator(pred_latent)
        g_loss = F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)
        )

        return g_loss, d_loss

    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        dry_latent = batch['dry_latent'].to(self.device)
        muted_latent = batch['muted_latent'].to(self.device)
        valid = batch['valid'].to(self.device)

        # Skip invalid samples
        if not valid.any():
            return {'loss': 0.0}

        dry_latent = dry_latent[valid]
        muted_latent = muted_latent[valid]

        # Forward pass
        self.model.train()
        pred_muted = self.model(dry_latent)

        # Losses
        dist_loss = self.distribution_loss(pred_muted, muted_latent)
        content_loss = self.content_preservation_loss(dry_latent, pred_muted)

        total_loss = dist_loss + 0.5 * content_loss

        # Adversarial (if enabled)
        if self.use_adversarial:
            g_loss, d_loss = self.adversarial_loss(pred_muted, muted_latent, train_discriminator=True)
            total_loss = total_loss + 0.1 * g_loss

            # Update discriminator
            if d_loss.item() > 0:
                self.optimizer_d.zero_grad()
                d_loss.backward()
                self.optimizer_d.step()

        # Update generator
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'dist_loss': dist_loss.item(),
            'content_loss': content_loss.item(),
        }

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        total_loss = 0.0
        total_dist = 0.0
        total_content = 0.0
        n_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            total_dist += metrics.get('dist_loss', 0)
            total_content += metrics.get('content_loss', 0)
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'dist': f"{metrics.get('dist_loss', 0):.4f}",
            })

        self.scheduler.step()

        return {
            'loss': total_loss / max(n_batches, 1),
            'dist_loss': total_dist / max(n_batches, 1),
            'content_loss': total_content / max(n_batches, 1),
            'lr': self.scheduler.get_last_lr()[0],
        }

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'muted_stats': {
                'mean': self.muted_mean,
                'std': self.muted_std,
            },
        }

        if self.discriminator:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()

        # Save latest
        torch.save(checkpoint, self.output_dir / "latest.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / "best.pt")

        # Save periodic
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, self.output_dir / f"epoch_{epoch+1}.pt")

    def log(self, message: str):
        """Log message to file and stdout."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def train(self):
        """Full training loop."""
        self.log("=" * 60)
        self.log("MUTE TRANSLATOR TRAINING")
        self.log("=" * 60)
        self.log(f"Output dir: {self.output_dir}")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Learning rate: {self.learning_rate}")
        self.log(f"Epochs: {self.num_epochs}")
        self.log(f"Dry samples: {len(self.dataset.dry_entries)}")
        self.log(f"Muted samples: {len(self.dataset.muted_entries)}")
        self.log("")

        for epoch in range(self.num_epochs):
            metrics = self.train_epoch(epoch)

            is_best = metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = metrics['loss']

            self.save_checkpoint(epoch, metrics, is_best)

            self.log(
                f"Epoch {epoch+1:3d} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Dist: {metrics['dist_loss']:.4f} | "
                f"Content: {metrics['content_loss']:.4f} | "
                f"LR: {metrics['lr']:.2e}"
                + (" [BEST]" if is_best else "")
            )

        self.log("")
        self.log("Training complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")
        self.log(f"Checkpoints saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Mute Translator")
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/Data.backup/final_training_manifest_brass_only.json',
                        help='Path to manifest JSON')
    parser.add_argument('--output_dir', type=str,
                        default='/home/arlo/Data/mute_translator/checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--model_type', type=str, default='small',
                        choices=['small', 'large'],
                        help='Model size')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--window_frames', type=int, default=128)
    parser.add_argument('--samples_per_epoch', type=int, default=5000)
    parser.add_argument('--use_adversarial', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    trainer = MuteTranslatorTrainer(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        window_frames=args.window_frames,
        samples_per_epoch=args.samples_per_epoch,
        use_adversarial=args.use_adversarial,
        device=args.device,
        num_workers=args.num_workers,
    )

    trainer.train()


if __name__ == "__main__":
    main()
