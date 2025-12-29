#!/usr/bin/env python3
"""
Train Register Translator

Trains a latent space translator to map high register → low register trumpet.

Key insight from mute translator: Train on VALID inputs (natural high, natural low),
then apply sox pitch shift AFTER model inference for actual pitch correction.

Usage:
    python train_register.py --manifest /path/to/manifest.json --output_dir ./checkpoints

The translator uses distribution matching:
- Input: high register trumpet latent
- Output: latent that matches low register trumpet distribution
- Loss: MSE to low distribution + content preservation + structure losses
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
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from tqdm import tqdm

# Enable cuDNN autotuning
torch.backends.cudnn.benchmark = True

sys.path.insert(0, '/home/arlo/Data/pitchshift/v6')

from models import (
    RegisterTranslator,
    RegisterTranslatorDirect,
    RegisterTranslatorAdaptive,
    RegisterTranslator2D,
    RegisterDiscriminator,
)
from dataset import RegisterTranslatorDataset, RegisterTranslatorDatasetMultiBin


class RegisterTranslatorTrainer:
    """
    Trainer for the register translator.

    Training strategy (from mute translator):
    1. Distribution matching: Make translated latents match low register distribution
    2. Content preservation: Translated latent should preserve structure of input
    3. Optional adversarial: Discriminator to improve realism
    """

    def __init__(
        self,
        manifest_path: str,
        output_dir: str,
        model_type: str = "residual",
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        window_frames: int = 128,
        samples_per_epoch: int = 5000,
        low_threshold: float = 55.0,
        high_threshold: float = 70.0,
        use_adversarial: bool = False,
        use_amp: bool = True,
        device: str = "cuda",
        num_workers: int = 4,
        dist_weight: float = 0.5,
    ):
        self.manifest_path = manifest_path
        self.use_amp = use_amp
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.use_adversarial = use_adversarial
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        self.model_type = model_type
        self.dist_weight = dist_weight

        print(f"Device: {self.device}")

        # Create model
        if model_type == "2d":
            self.model = RegisterTranslator2D().to(self.device)
        elif model_type == "direct":
            self.model = RegisterTranslatorDirect().to(self.device)
        elif model_type == "adaptive":
            self.model = RegisterTranslatorAdaptive(alpha_init=0.3).to(self.device)
        else:  # residual (default)
            self.model = RegisterTranslator().to(self.device)

        print(f"Model type: {model_type}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Optional discriminator
        self.discriminator = None
        if use_adversarial:
            self.discriminator = RegisterDiscriminator().to(self.device)
            print(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")

        # Create dataset
        self.dataset = RegisterTranslatorDataset(
            manifest_path=manifest_path,
            window_frames=window_frames,
            samples_per_epoch=samples_per_epoch,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Compute target distribution statistics
        self._compute_target_stats()

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        # AMP scaler
        self.scaler = GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print("Using AMP (mixed precision training)")

        if self.discriminator:
            self.optimizer_d = AdamW(self.discriminator.parameters(), lr=learning_rate * 0.5)

        # Logging
        self.log_file = self.output_dir / "training.log"
        self.best_loss = float('inf')

    def _compute_target_stats(self):
        """Compute mean and std of target (low register) distribution."""
        print("Computing target distribution statistics...")

        all_latents = []
        for entry in self.dataset.low_entries[:100]:
            latent = self.dataset._load_latent(entry)
            if latent is not None:
                if latent.dim() == 4:
                    latent = latent.squeeze(0)
                all_latents.append(latent)

        if len(all_latents) > 0:
            concat = torch.cat([l.reshape(-1) for l in all_latents])
            self.target_mean = concat.mean().item()
            self.target_std = concat.std().item()

            stacked = torch.stack([l.mean(dim=(1, 2)) for l in all_latents])
            self.target_channel_mean = stacked.mean(dim=0).to(self.device)
            self.target_channel_std = stacked.std(dim=0).to(self.device)

            print(f"  Target mean: {self.target_mean:.4f}, std: {self.target_std:.4f}")
        else:
            self.target_mean = 0.0
            self.target_std = 1.0
            self.target_channel_mean = torch.zeros(8, device=self.device)
            self.target_channel_std = torch.ones(8, device=self.device)

    def distribution_loss(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Distribution matching loss for register transfer.

        Key insight: formant differences are in spectral envelope, not fine pitch.
        Weight lower frequencies more for formant matching.
        """
        B, C, H, T = pred_latent.shape

        # Frequency band weighting
        # Low-mid frequencies carry more formant information
        freq_weights = torch.ones(H, device=pred_latent.device)
        lf_end = H // 4       # Lower 25%: body resonance
        mf_end = H // 2       # 25-50%: formant region
        freq_weights[:lf_end] = 2.0      # Body: 2x
        freq_weights[lf_end:mf_end] = 1.5  # Mid: 1.5x
        freq_weights = freq_weights.view(1, 1, H, 1)

        # Weighted MSE loss
        weighted_diff = ((pred_latent - target_latent) ** 2) * freq_weights
        mse_loss = weighted_diff.mean()

        # Per-channel mean/std matching
        pred_mean = pred_latent.mean(dim=(2, 3))
        pred_std = pred_latent.std(dim=(2, 3))
        target_mean = target_latent.mean(dim=(2, 3))
        target_std = target_latent.std(dim=(2, 3))

        mean_loss = F.mse_loss(pred_mean, target_mean)
        std_loss = F.mse_loss(pred_std, target_std)

        # Formant region energy (mid-frequencies)
        pred_formant = pred_latent[:, :, lf_end:mf_end, :].abs().mean(dim=(2, 3))
        target_formant = target_latent[:, :, lf_end:mf_end, :].abs().mean(dim=(2, 3))
        formant_loss = F.mse_loss(pred_formant, target_formant)

        # MMD (domain matching)
        pred_flat = pred_latent.reshape(B, -1).float()
        target_flat = target_latent.reshape(B, -1).float()
        scale = pred_flat.shape[1] ** 0.5
        pred_flat = pred_flat / scale
        target_flat = target_flat / scale
        pred_gram = torch.mm(pred_flat, pred_flat.t())
        target_gram = torch.mm(target_flat, target_flat.t())
        cross_gram = torch.mm(pred_flat, target_flat.t())
        mmd = pred_gram.mean() + target_gram.mean() - 2 * cross_gram.mean()

        return (
            mse_loss
            + mean_loss
            + std_loss
            + 0.5 * formant_loss
            + 0.1 * mmd
        )

    def content_preservation_loss(
        self,
        high_latent: torch.Tensor,
        pred_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ensure translated latent preserves melodic structure of input.

        The translation should modify formants, not pitch contour.
        """
        # Temporal gradient (rhythm/note boundaries)
        high_grad = high_latent[:, :, :, 1:] - high_latent[:, :, :, :-1]
        pred_grad = pred_latent[:, :, :, 1:] - pred_latent[:, :, :, :-1]
        grad_loss = F.mse_loss(pred_grad.abs(), high_grad.abs())

        # Spectral gradient (pitch contour)
        high_spec = high_latent[:, :, 1:, :] - high_latent[:, :, :-1, :]
        pred_spec = pred_latent[:, :, 1:, :] - pred_latent[:, :, :-1, :]
        spec_loss = F.mse_loss(pred_spec.abs(), high_spec.abs())

        return grad_loss + spec_loss

    def silence_constraint_loss(
        self,
        high_latent: torch.Tensor,
        pred_latent: torch.Tensor,
        silence_thresh: float = 0.05,
    ) -> torch.Tensor:
        """Penalize output energy when input is silent."""
        high_energy = high_latent.pow(2).mean(dim=(1, 2)).sqrt()
        pred_energy = pred_latent.pow(2).mean(dim=(1, 2)).sqrt()

        silence_mask = (high_energy < silence_thresh).float()
        silence_loss = (silence_mask * pred_energy.pow(2)).sum() / (silence_mask.sum() + 1e-6)

        return silence_loss

    def envelope_preservation_loss(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
        window_size: int = 8,
    ) -> torch.Tensor:
        """Match RMS envelope between predicted and target."""
        B, C, H, T = pred_latent.shape

        pred_energy = pred_latent.pow(2).mean(dim=(1, 2)).unsqueeze(1)
        target_energy = target_latent.pow(2).mean(dim=(1, 2)).unsqueeze(1)

        padding = window_size // 2
        pred_rms = F.avg_pool1d(pred_energy, window_size, stride=1, padding=padding)[:, :, :T].sqrt()
        target_rms = F.avg_pool1d(target_energy, window_size, stride=1, padding=padding)[:, :, :T].sqrt()

        return F.l1_loss(pred_rms, target_rms)

    def adversarial_loss(
        self,
        pred_latent: torch.Tensor,
        real_low: torch.Tensor,
        train_discriminator: bool = True
    ) -> tuple:
        """Adversarial loss for realistic translations."""
        if self.discriminator is None:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        if train_discriminator:
            self.discriminator.train()
            real_logits = self.discriminator(real_low.detach())
            fake_logits = self.discriminator(pred_latent.detach())

            d_loss = F.binary_cross_entropy_with_logits(
                real_logits, torch.ones_like(real_logits)
            ) + F.binary_cross_entropy_with_logits(
                fake_logits, torch.zeros_like(fake_logits)
            )
        else:
            d_loss = torch.tensor(0.0, device=self.device)

        fake_logits = self.discriminator(pred_latent)
        g_loss = F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)
        )

        return g_loss, d_loss

    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        high_latent = batch['high_latent'].to(self.device)
        low_latent = batch['low_latent'].to(self.device)
        valid = batch['valid'].to(self.device)

        if not valid.any():
            return {'loss': 0.0}

        high_latent = high_latent[valid]
        low_latent = low_latent[valid]

        self.model.train()
        self.optimizer.zero_grad()

        amp_context = autocast('cuda') if self.use_amp else nullcontext()

        with amp_context:
            pred_low = self.model(high_latent)

            # Losses
            dist_loss = self.distribution_loss(pred_low, low_latent)
            content_loss = self.content_preservation_loss(high_latent, pred_low)
            silence_loss = self.silence_constraint_loss(high_latent, pred_low)
            envelope_loss = self.envelope_preservation_loss(pred_low, low_latent)

            # Combined loss
            total_loss = (
                self.dist_weight * dist_loss
                + 1.0 * silence_loss
                + 0.2 * envelope_loss
            )

            # Direct model needs content preservation
            if self.model_type == 'direct':
                total_loss = total_loss + 0.5 * content_loss

            if self.use_adversarial:
                g_loss, d_loss = self.adversarial_loss(pred_low, low_latent)
                total_loss = total_loss + 0.1 * g_loss

        # Backward
        if self.use_amp and self.scaler is not None:
            if self.use_adversarial:
                self.optimizer_d.zero_grad()
                self.scaler.scale(d_loss).backward(retain_graph=True)
                self.scaler.step(self.optimizer_d)

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.use_adversarial:
                self.optimizer_d.zero_grad()
                d_loss.backward(retain_graph=True)
                self.optimizer_d.step()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'dist_loss': dist_loss.item(),
            'content_loss': content_loss.item(),
            'silence_loss': silence_loss.item(),
            'envelope_loss': envelope_loss.item(),
        }

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        total_loss = 0.0
        total_dist = 0.0
        total_content = 0.0
        total_silence = 0.0
        total_envelope = 0.0
        n_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            total_dist += metrics.get('dist_loss', 0)
            total_content += metrics.get('content_loss', 0)
            total_silence += metrics.get('silence_loss', 0)
            total_envelope += metrics.get('envelope_loss', 0)
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
            'silence_loss': total_silence / max(n_batches, 1),
            'envelope_loss': total_envelope / max(n_batches, 1),
            'lr': self.scheduler.get_last_lr()[0],
        }

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_type': self.model_type,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'low_threshold': self.low_threshold,
            'high_threshold': self.high_threshold,
            'target_stats': {
                'mean': self.target_mean,
                'std': self.target_std,
            },
        }

        if self.discriminator:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()

        torch.save(checkpoint, self.output_dir / "latest.pt")

        if is_best:
            torch.save(checkpoint, self.output_dir / "best.pt")

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
        self.log("REGISTER TRANSLATOR TRAINING (v6)")
        self.log("=" * 60)
        self.log(f"Output dir: {self.output_dir}")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Learning rate: {self.learning_rate}")
        self.log(f"Epochs: {self.num_epochs}")
        self.log(f"LOW threshold: < {self.low_threshold} MIDI")
        self.log(f"HIGH threshold: > {self.high_threshold} MIDI")
        self.log(f"Gap: {self.high_threshold - self.low_threshold:.0f} semitones")
        self.log(f"HIGH register samples: {len(self.dataset.high_entries)}")
        self.log(f"LOW register samples: {len(self.dataset.low_entries)}")
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
                f"Sil: {metrics['silence_loss']:.4f} | "
                f"Env: {metrics['envelope_loss']:.4f} | "
                f"LR: {metrics['lr']:.2e}"
                + (" [BEST]" if is_best else "")
            )

        self.log("")
        self.log("Training complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")
        self.log(f"Checkpoints saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Register Translator")
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json',
                        help='Path to manifest JSON')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/msdd2/pitchshift_checkpoints/register_v6',
                        help='Output directory for checkpoints')
    parser.add_argument('--model_type', type=str, default='residual',
                        choices=['residual', '2d', 'direct', 'adaptive'],
                        help='Model type')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--window_frames', type=int, default=128)
    parser.add_argument('--samples_per_epoch', type=int, default=10000)
    parser.add_argument('--low_threshold', type=float, default=55.0,
                        help='Below this MIDI = LOW register (default: 55 = G3)')
    parser.add_argument('--high_threshold', type=float, default=70.0,
                        help='Above this MIDI = HIGH register (default: 70 = Bb4)')
    parser.add_argument('--use_adversarial', action='store_true')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dist_weight', type=float, default=0.5)

    args = parser.parse_args()

    trainer = RegisterTranslatorTrainer(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        window_frames=args.window_frames,
        samples_per_epoch=args.samples_per_epoch,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        use_adversarial=args.use_adversarial,
        use_amp=not args.no_amp,
        device=args.device,
        num_workers=args.num_workers,
        dist_weight=args.dist_weight,
    )

    trainer.train()


if __name__ == "__main__":
    main()
