#!/usr/bin/env python3
"""
Train Corruption Model: Learn to synthesize DSP pitch shift artifacts

The reverse approach to pitch shift correction:
1. Learn what DSP artifacts look like (distribution matching)
2. Use corruption model to create aligned pairs
3. Train correction model on aligned pairs with frame-level supervision

Training data:
- REAL: Actual DSP-shifted latents (sox/rubberband applied)
- FAKE: Natural latent + corruption model output

The discriminator learns "does this look like real DSP corruption at shift X?"
The generator learns to produce realistic DSP-like artifacts.
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from models_corruption import (
    CorruptionModel,
    CorruptionDiscriminator,
    CorruptionGANLoss,
    ContentPreservationLoss,
    ArtifactMagnitudeLoss,
    C3CorruptionLoss,
)


def fix_path(path: str) -> str:
    """Fix mount paths."""
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


class CorruptionDataset(Dataset):
    """
    Dataset for corruption model training.

    Provides:
    - Natural latents (input to corruption model)
    - Real DSP-shifted latents (target distribution)
    - Shift amounts for conditioning
    """

    def __init__(
        self,
        segments_json: str,
        shifted_manifest: str,
        window_frames: int = 64,
        samples_per_epoch: int = 5000,
    ):
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch

        # Load natural segments
        print("Loading natural segments...")
        with open(segments_json) as f:
            segments_data = json.load(f)

        self.natural_segments = []
        for group_id, segments in segments_data.get('segments_by_group', {}).items():
            for seg in segments:
                self.natural_segments.append({
                    'latent_path': fix_path(seg['latent_path']),
                    'start_frame': seg['start_frame'],
                    'end_frame': seg['end_frame'],
                    'pitch': seg['median_midi'],
                })

        print(f"  {len(self.natural_segments)} natural segments")

        # Load shifted manifest (real DSP-shifted for discriminator)
        print("Loading shifted manifest...")
        with open(shifted_manifest) as f:
            shifted_data = json.load(f)

        self.shifted_entries = []
        for entry in shifted_data.get('entries', []):
            source_midi = entry.get('median_midi', 60.0)
            for shift_str, shifted_path in entry.get('shifted_latents', {}).items():
                shift = int(shift_str)
                if shift == 0:
                    continue  # Skip identity shifts
                self.shifted_entries.append({
                    'shifted_path': fix_path(shifted_path),
                    'shift': shift,
                    'source_pitch': source_midi,
                })

        print(f"  {len(self.shifted_entries)} shifted entries (non-zero shifts)")

        # Get available shift values
        self.available_shifts = list(set(e['shift'] for e in self.shifted_entries))
        print(f"  Available shifts: {sorted(self.available_shifts)}")

        # Preload latents
        print("Preloading latents...")
        self.latent_cache = {}
        self._preload_latents()

    def _preload_latents(self):
        """Preload unique latent files."""
        paths = set()

        for seg in self.natural_segments:
            paths.add(seg['latent_path'])

        for entry in self.shifted_entries:
            paths.add(entry['shifted_path'])

        for path in tqdm(paths, desc="Loading latents"):
            if path and Path(path).exists():
                try:
                    data = torch.load(path, map_location='cpu', weights_only=False)
                    if isinstance(data, dict):
                        latent = data.get('latents', data.get('latent', None))
                    else:
                        latent = data
                    if latent is not None:
                        if latent.dim() == 4:
                            latent = latent.squeeze(0)
                        self.latent_cache[path] = latent
                except:
                    pass

        print(f"  Cached {len(self.latent_cache)} latents")

    def _get_window(self, path: str, start: int = None, end: int = None) -> torch.Tensor:
        """Get a window from a latent."""
        if path not in self.latent_cache:
            return None

        latent = self.latent_cache[path]
        if latent.dim() == 4:
            latent = latent.squeeze(0)

        T = latent.shape[-1]

        if start is not None and end is not None:
            available = end - start
            if available < self.window_frames:
                return None
            ws = np.random.randint(start, end - self.window_frames + 1) if end - start > self.window_frames else start
            return latent[:, :, ws:ws + self.window_frames]
        else:
            if T < self.window_frames:
                return None
            ws = np.random.randint(0, T - self.window_frames + 1)
            return latent[:, :, ws:ws + self.window_frames]

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        """
        Returns:
        - natural_latent: clean trumpet (input to corruption model)
        - dsp_shifted_latent: real DSP-shifted (target distribution)
        - shift: shift amount for conditioning
        - valid: whether data loaded successfully
        """
        # Get natural latent (input to corruption model)
        nat_seg = self.natural_segments[np.random.randint(len(self.natural_segments))]
        natural_latent = self._get_window(
            nat_seg['latent_path'],
            nat_seg['start_frame'],
            nat_seg['end_frame']
        )

        # Get real DSP-shifted latent (target distribution)
        shift_entry = self.shifted_entries[np.random.randint(len(self.shifted_entries))]
        dsp_latent = self._get_window(shift_entry['shifted_path'])

        valid = (natural_latent is not None and dsp_latent is not None)

        if not valid:
            return {
                'natural_latent': torch.zeros(8, 16, self.window_frames),
                'dsp_shifted_latent': torch.zeros(8, 16, self.window_frames),
                'shift': torch.tensor(6.0),
                'valid': False,
            }

        return {
            'natural_latent': natural_latent,
            'dsp_shifted_latent': dsp_latent,
            'shift': torch.tensor(float(shift_entry['shift'])),
            'valid': True,
        }


class CorruptionTrainer:
    """Train corruption model with adversarial loss."""

    def __init__(
        self,
        segments_json: str,
        shifted_manifest: str,
        output_dir: str,
        batch_size: int = 32,
        num_epochs: int = 50,
        samples_per_epoch: int = 5000,
        g_lr: float = 1e-4,
        d_lr: float = 4e-4,
        window_frames: int = 64,
        num_workers: int = 4,
        device: str = 'cuda',
        # Loss weights
        adv_weight: float = 1.0,
        content_weight: float = 1.0,
        magnitude_weight: float = 0.5,
        c3_weight: float = 0.2,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs

        # Loss weights
        self.adv_weight = adv_weight
        self.content_weight = content_weight
        self.magnitude_weight = magnitude_weight
        self.c3_weight = c3_weight

        # Dataset
        print("Creating dataset...")
        self.dataset = CorruptionDataset(
            segments_json=segments_json,
            shifted_manifest=shifted_manifest,
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

        # Models
        print("Creating models...")
        self.generator = CorruptionModel(
            hidden_dim=256,
            num_blocks=6,
        ).to(self.device)

        self.discriminator = CorruptionDiscriminator(
            hidden_dim=128,
        ).to(self.device)

        print(f"Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")

        # Losses
        self.gan_loss = CorruptionGANLoss()
        self.content_loss = ContentPreservationLoss()
        self.magnitude_loss = ArtifactMagnitudeLoss(target_magnitude=0.3)
        self.c3_loss = C3CorruptionLoss()

        # Optimizers
        self.g_optim = AdamW(self.generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
        self.d_optim = AdamW(self.discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))

        self.g_scheduler = CosineAnnealingLR(self.g_optim, T_max=num_epochs, eta_min=g_lr / 10)
        self.d_scheduler = CosineAnnealingLR(self.d_optim, T_max=num_epochs, eta_min=d_lr / 10)

        # Logging
        self.log_file = self.output_dir / 'training.log'
        self.best_g_loss = float('inf')

        # Save config
        config = {
            'model': 'corruption',
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'g_lr': g_lr,
            'd_lr': d_lr,
            'adv_weight': adv_weight,
            'content_weight': content_weight,
            'magnitude_weight': magnitude_weight,
            'c3_weight': c3_weight,
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def log(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def train_discriminator_step(self, batch) -> dict:
        """Train discriminator: real DSP vs synthetic corruption."""
        self.discriminator.train()
        self.d_optim.zero_grad()

        natural_latent = batch['natural_latent'].to(self.device)
        dsp_latent = batch['dsp_shifted_latent'].to(self.device)
        shift = batch['shift'].to(self.device)

        # Generate fake (corrupted natural)
        with torch.no_grad():
            fake_latent = self.generator(natural_latent, shift)

        # Discriminate
        real_logits = self.discriminator(dsp_latent, shift)
        fake_logits = self.discriminator(fake_latent, shift)

        d_loss = self.gan_loss.discriminator_loss(real_logits, fake_logits)

        d_loss.backward()
        self.d_optim.step()

        return {
            'd_loss': d_loss.item(),
            'real_score': real_logits.mean().item(),
            'fake_score': fake_logits.mean().item(),
        }

    def train_generator_step(self, batch) -> dict:
        """Train generator: make corruption look like real DSP."""
        self.generator.train()
        self.g_optim.zero_grad()

        natural_latent = batch['natural_latent'].to(self.device)
        shift = batch['shift'].to(self.device)

        # Generate corrupted
        corrupted_latent = self.generator(natural_latent, shift)

        # Adversarial loss
        fake_logits = self.discriminator(corrupted_latent, shift)
        adv_loss = self.gan_loss.generator_loss(fake_logits)

        # Content preservation
        content_loss = self.content_loss(corrupted_latent, natural_latent)

        # Artifact magnitude
        mag_loss = self.magnitude_loss(corrupted_latent, natural_latent, shift)

        # C3 focus
        c3_loss = self.c3_loss(corrupted_latent, natural_latent)

        # Total
        g_loss = (
            self.adv_weight * adv_loss +
            self.content_weight * content_loss +
            self.magnitude_weight * mag_loss +
            self.c3_weight * c3_loss
        )

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.g_optim.step()

        return {
            'g_loss': g_loss.item(),
            'adv_loss': adv_loss.item(),
            'content_loss': content_loss.item(),
            'mag_loss': mag_loss.item(),
            'c3_loss': c3_loss.item(),
        }

    def train_epoch(self, epoch: int) -> dict:
        """Train one epoch."""
        d_totals = {'d_loss': 0, 'real_score': 0, 'fake_score': 0}
        g_totals = {'g_loss': 0, 'adv_loss': 0, 'content_loss': 0, 'mag_loss': 0, 'c3_loss': 0}
        n_d, n_g = 0, 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")

        for i, batch in enumerate(pbar):
            valid = batch['valid']
            if not valid.any():
                continue

            # Filter valid samples
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v[valid]

            # Train discriminator
            d_metrics = self.train_discriminator_step(batch)
            for k, v in d_metrics.items():
                d_totals[k] += v
            n_d += 1

            # Train generator
            g_metrics = self.train_generator_step(batch)
            for k, v in g_metrics.items():
                g_totals[k] += v
            n_g += 1

            pbar.set_postfix({
                'D': f"{d_metrics['d_loss']:.3f}",
                'G': f"{g_metrics['g_loss']:.3f}",
                'R/F': f"{d_metrics['real_score']:.2f}/{d_metrics['fake_score']:.2f}",
            })

        self.g_scheduler.step()
        self.d_scheduler.step()

        metrics = {}
        for k, v in d_totals.items():
            metrics[k] = v / max(n_d, 1)
        for k, v in g_totals.items():
            metrics[k] = v / max(n_g, 1)

        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save checkpoint."""
        ckpt = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optim_state_dict': self.g_optim.state_dict(),
            'd_optim_state_dict': self.d_optim.state_dict(),
            'metrics': metrics,
        }

        torch.save(ckpt, self.output_dir / 'latest.pt')

        if is_best:
            torch.save(ckpt, self.output_dir / 'best.pt')

        if epoch % 10 == 0:
            torch.save(ckpt, self.output_dir / f'checkpoint_epoch{epoch}.pt')

    def train(self):
        """Full training loop."""
        self.log("=" * 60)
        self.log("CORRUPTION MODEL TRAINING")
        self.log("Learn to synthesize DSP pitch shift artifacts")
        self.log("=" * 60)
        self.log(f"Output: {self.output_dir}")
        self.log(f"Device: {self.device}")
        self.log(f"Weights: adv={self.adv_weight}, content={self.content_weight}, mag={self.magnitude_weight}, c3={self.c3_weight}")
        self.log("=" * 60)

        for epoch in range(1, self.num_epochs + 1):
            metrics = self.train_epoch(epoch)

            is_best = metrics['g_loss'] < self.best_g_loss
            if is_best:
                self.best_g_loss = metrics['g_loss']

            self.save_checkpoint(epoch, metrics, is_best)

            self.log(
                f"Epoch {epoch:3d} | "
                f"D: {metrics['d_loss']:.4f} | "
                f"G: {metrics['g_loss']:.4f} | "
                f"Cont: {metrics['content_loss']:.4f} | "
                f"Mag: {metrics['mag_loss']:.4f} | "
                f"R/F: {metrics['real_score']:.2f}/{metrics['fake_score']:.2f}"
                + (" [BEST]" if is_best else "")
            )

        self.log("\nCorruption model training complete!")
        self.log(f"Best G loss: {self.best_g_loss:.4f}")
        self.log("\nNext steps:")
        self.log("1. Test corruption model quality")
        self.log("2. Generate aligned pairs using corruption model")
        self.log("3. Train correction model on aligned pairs")


def main():
    parser = argparse.ArgumentParser(description="Train Corruption Model")
    parser.add_argument('--segments', type=str, required=True)
    parser.add_argument('--shifted_manifest', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--samples_per_epoch', type=int, default=5000)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=4e-4)
    parser.add_argument('--adv_weight', type=float, default=1.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--magnitude_weight', type=float, default=0.5)
    parser.add_argument('--c3_weight', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    trainer = CorruptionTrainer(
        segments_json=args.segments,
        shifted_manifest=args.shifted_manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        samples_per_epoch=args.samples_per_epoch,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        adv_weight=args.adv_weight,
        content_weight=args.content_weight,
        magnitude_weight=args.magnitude_weight,
        c3_weight=args.c3_weight,
        num_workers=args.num_workers,
        device=args.device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
