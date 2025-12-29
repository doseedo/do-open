#!/usr/bin/env python3
"""
Adversarial formant corrector training.

Key insight: Use a discriminator to learn what makes natural LOW register sound natural.
The generator learns to fool the discriminator, producing outputs that sound like natural LOW.

This avoids the problem of matching to random mismatched samples.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

sys.path.insert(0, '/home/arlo/Data/pitchshift/v6')
sys.path.insert(0, '/home/arlo/Data/pitchshift/v7')

from models import RegisterTranslator
from dataset_overlapped import OverlappedFormantCorrectorDataset


class FormantDiscriminator(nn.Module):
    """
    Discriminator to distinguish natural LOW from shifted HIGH.

    Learns what makes natural LOW trumpet sound natural:
    - Warmer tone (less HF energy)
    - Correct formant structure for the register
    - Natural harmonic relationships
    """

    def __init__(self, in_channels: int = 8, hidden_dim: int = 64):
        super().__init__()

        # Process [B, C, H, T] latents
        self.net = nn.Sequential(
            # Initial conv
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Downsample
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 4),
            nn.LeakyReLU(0.2),

            # Global pooling and classification
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits: positive = natural LOW, negative = shifted/fake."""
        return self.net(x)


class AdversarialFormantTrainer:
    def __init__(
        self,
        manifest_path: str,
        output_dir: str,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        use_amp: bool = True,
        device: str = "cuda",
        num_workers: int = 4,
        d_steps: int = 1,  # Discriminator steps per generator step
        lambda_adv: float = 0.5,  # Weight for adversarial loss
        lambda_content: float = 1.0,  # Weight for content preservation
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp
        self.device = torch.device(device)
        self.num_epochs = num_epochs
        self.d_steps = d_steps
        self.lambda_adv = lambda_adv
        self.lambda_content = lambda_content

        print(f"Device: {self.device}")

        # Generator (formant corrector)
        self.generator = RegisterTranslator().to(self.device)
        print(f"Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")

        # Discriminator
        self.discriminator = FormantDiscriminator().to(self.device)
        print(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")

        # Dataset
        self.dataset = OverlappedFormantCorrectorDataset(
            manifest_path=manifest_path,
            samples_per_epoch=5000,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Optimizers
        self.opt_g = AdamW(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.opt_d = AdamW(self.discriminator.parameters(), lr=learning_rate * 0.5, betas=(0.5, 0.999))

        self.scheduler_g = CosineAnnealingLR(self.opt_g, T_max=num_epochs, eta_min=1e-6)
        self.scheduler_d = CosineAnnealingLR(self.opt_d, T_max=num_epochs, eta_min=1e-6)

        self.scaler = GradScaler('cuda') if use_amp else None

        self.log_file = self.output_dir / "training.log"
        self.best_loss = float('inf')

        # Pre-compute natural LOW statistics for reference
        self._compute_low_stats()

    def _compute_low_stats(self):
        """Compute statistics of natural LOW latents."""
        print("Computing natural LOW statistics...")

        all_latents = []
        for entry in self.dataset.low_entries[:100]:
            latent = self.dataset._load_latent(entry['latent_path'])
            if latent is not None:
                all_latents.append(latent)

        if all_latents:
            # Per-channel mean/std
            stacked = torch.stack([l.mean(dim=(1, 2)) for l in all_latents])
            self.low_channel_mean = stacked.mean(dim=0).to(self.device)
            self.low_channel_std = stacked.std(dim=0).to(self.device)

            # HF energy (upper half of H dimension)
            hf_energies = []
            for l in all_latents:
                H = l.shape[1]
                hf = l[:, H//2:, :].abs().mean()
                hf_energies.append(hf)
            self.low_hf_mean = torch.tensor(hf_energies).mean().to(self.device)
            self.low_hf_std = torch.tensor(hf_energies).std().to(self.device)

            print(f"  LOW channel mean: {self.low_channel_mean.mean():.4f}")
            print(f"  LOW HF energy: {self.low_hf_mean:.4f} ± {self.low_hf_std:.4f}")

    def content_preservation_loss(self, input_lat: torch.Tensor, output_lat: torch.Tensor) -> torch.Tensor:
        """Preserve temporal and spectral structure."""
        # Temporal gradient
        in_grad = input_lat[:, :, :, 1:] - input_lat[:, :, :, :-1]
        out_grad = output_lat[:, :, :, 1:] - output_lat[:, :, :, :-1]
        temporal = F.mse_loss(out_grad, in_grad)

        # Spectral gradient
        in_spec = input_lat[:, :, 1:, :] - input_lat[:, :, :-1, :]
        out_spec = output_lat[:, :, 1:, :] - output_lat[:, :, :-1, :]
        spectral = F.mse_loss(out_spec, in_spec)

        return temporal + spectral

    def hf_matching_loss(self, output_lat: torch.Tensor, target_lat: torch.Tensor) -> torch.Tensor:
        """Match high-frequency characteristics (formants live here)."""
        H = output_lat.shape[2]

        # Upper half = high frequencies
        out_hf = output_lat[:, :, H//2:, :]
        tgt_hf = target_lat[:, :, H//2:, :]

        # Energy matching
        out_energy = out_hf.abs().mean(dim=(2, 3))
        tgt_energy = tgt_hf.abs().mean(dim=(2, 3))

        return F.mse_loss(out_energy, tgt_energy)

    def train_discriminator_step(self, real_low: torch.Tensor, fake_low: torch.Tensor) -> dict:
        """Train discriminator to distinguish real LOW from generated."""
        self.opt_d.zero_grad()

        amp_ctx = autocast('cuda') if self.use_amp else nullcontext()

        with amp_ctx:
            # Real samples (natural LOW)
            real_pred = self.discriminator(real_low)
            real_loss = F.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            )

            # Fake samples (generated from shifted)
            fake_pred = self.discriminator(fake_low.detach())
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )

            d_loss = (real_loss + fake_loss) / 2

        if self.scaler:
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.opt_d)
            self.scaler.update()
        else:
            d_loss.backward()
            self.opt_d.step()

        return {
            'd_loss': d_loss.item(),
            'd_real': real_pred.mean().item(),
            'd_fake': fake_pred.mean().item(),
        }

    def train_generator_step(
        self,
        shifted: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """Train generator to produce outputs that fool discriminator."""
        self.opt_g.zero_grad()

        amp_ctx = autocast('cuda') if self.use_amp else nullcontext()

        with amp_ctx:
            # Generate corrected output
            output = self.generator(shifted)

            # Adversarial loss: fool discriminator
            fake_pred = self.discriminator(output)
            adv_loss = F.binary_cross_entropy_with_logits(
                fake_pred, torch.ones_like(fake_pred)  # Want discriminator to think it's real
            )

            # Content preservation
            content_loss = self.content_preservation_loss(shifted, output)

            # HF matching to target distribution
            hf_loss = self.hf_matching_loss(output, target)

            # Combined loss
            g_loss = (
                self.lambda_adv * adv_loss +
                self.lambda_content * content_loss +
                0.3 * hf_loss
            )

        if self.scaler:
            self.scaler.scale(g_loss).backward()
            self.scaler.unscale_(self.opt_g)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.scaler.step(self.opt_g)
            self.scaler.update()
        else:
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.opt_g.step()

        return {
            'g_loss': g_loss.item(),
            'adv': adv_loss.item(),
            'content': content_loss.item(),
            'hf': hf_loss.item(),
        }

    def train_epoch(self, epoch: int) -> dict:
        totals = {'d_loss': 0, 'g_loss': 0, 'adv': 0, 'content': 0, 'hf': 0}
        n_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch in pbar:
            shifted = batch['corrupted'].to(self.device)
            target = batch['target'].to(self.device)
            valid = batch['valid'].to(self.device)

            if not valid.any():
                continue

            shifted = shifted[valid]
            target = target[valid]

            # Train discriminator
            with torch.no_grad():
                fake = self.generator(shifted)

            for _ in range(self.d_steps):
                d_metrics = self.train_discriminator_step(target, fake)

            # Train generator
            g_metrics = self.train_generator_step(shifted, target)

            totals['d_loss'] += d_metrics['d_loss']
            totals['g_loss'] += g_metrics['g_loss']
            totals['adv'] += g_metrics['adv']
            totals['content'] += g_metrics['content']
            totals['hf'] += g_metrics['hf']
            n_batches += 1

            pbar.set_postfix({
                'D': f"{d_metrics['d_loss']:.3f}",
                'G': f"{g_metrics['g_loss']:.3f}",
                'adv': f"{g_metrics['adv']:.3f}",
            })

        self.scheduler_g.step()
        self.scheduler_d.step()

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'opt_g_state_dict': self.opt_g.state_dict(),
            'opt_d_state_dict': self.opt_d.state_dict(),
            'metrics': metrics,
        }

        torch.save(checkpoint, self.output_dir / "latest.pt")

        if is_best:
            torch.save(checkpoint, self.output_dir / "best.pt")
            # Also save generator-only for easy loading
            torch.save({
                'model_state_dict': self.generator.state_dict(),
                'epoch': epoch,
                'metrics': metrics,
            }, self.output_dir / "best_generator.pt")

    def log(self, msg: str):
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    def train(self):
        self.log(f"Starting adversarial training")
        self.log(f"  lambda_adv: {self.lambda_adv}")
        self.log(f"  lambda_content: {self.lambda_content}")
        self.log(f"  d_steps: {self.d_steps}")

        for epoch in range(self.num_epochs):
            metrics = self.train_epoch(epoch)

            is_best = metrics['g_loss'] < self.best_loss
            if is_best:
                self.best_loss = metrics['g_loss']

            self.save_checkpoint(epoch, metrics, is_best)

            self.log(
                f"Epoch {epoch+1:3d} | "
                f"D: {metrics['d_loss']:.4f} | "
                f"G: {metrics['g_loss']:.4f} | "
                f"Adv: {metrics['adv']:.4f} | "
                f"Cnt: {metrics['content']:.4f} | "
                f"HF: {metrics['hf']:.4f}"
                + (" [BEST]" if is_best else "")
            )

        self.log("Training complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str,
                        default='/mnt/msdd2/pitchshift_v7_overlapped/manifest.json')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/msdd2/pitchshift_checkpoints/formant_corrector_v7_adversarial')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_adv', type=float, default=0.5)
    parser.add_argument('--lambda_content', type=float, default=1.0)
    parser.add_argument('--d_steps', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    trainer = AdversarialFormantTrainer(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        lambda_adv=args.lambda_adv,
        lambda_content=args.lambda_content,
        d_steps=args.d_steps,
        device=args.device,
        num_workers=args.num_workers,
    )

    trainer.train()


if __name__ == "__main__":
    main()
