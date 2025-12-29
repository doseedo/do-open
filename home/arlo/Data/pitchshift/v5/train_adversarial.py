#!/usr/bin/env python3
"""
Adversarial Training for Pitch Shift Correction

The discriminator learns "what does natural trumpet at pitch X look like?"
The generator learns to fool it while preserving content.

Training data:
- REAL: Natural trumpet latents at various pitches
- FAKE: DSP-shifted latents (before and after correction)

Key insight: discriminator penalizes:
- Noise in silent regions
- Chord-like artifacts on sustained notes
- Trombone-like softness on low trumpet
- Chipmunk artifacts on high trumpet
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

from models_adversarial import (
    RegisterDiscriminator,
    MultiScaleDiscriminator,
    PitchShiftCorrectorGAN,
    GANLoss,
    ContentPreservationLoss,
    FramewiseSilenceLoss,
)


def fix_path(path: str) -> str:
    """Fix mount paths."""
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


class AdversarialDataset(Dataset):
    """
    Dataset for adversarial training.

    Provides:
    - Natural trumpet latents with pitch labels (REAL)
    - DSP-shifted latents with shift labels (FAKE - generator input)
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

        # Load natural segments (for discriminator real samples)
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

        # Load shifted manifest (for generator training)
        print("Loading shifted manifest...")
        with open(shifted_manifest) as f:
            shifted_data = json.load(f)

        self.shifted_entries = []
        # Handle the actual manifest format: entries with shifted_latents dict
        for entry in shifted_data.get('entries', []):
            source_midi = entry.get('median_midi', 60.0)
            source_latent_path = fix_path(entry.get('source_latent_path', ''))

            # Each entry has shifted_latents dict with shift -> path
            for shift_str, shifted_path in entry.get('shifted_latents', {}).items():
                shift = int(shift_str)
                self.shifted_entries.append({
                    'shifted_path': fix_path(shifted_path),
                    'original_path': source_latent_path,
                    'shift': shift,
                    'source_pitch': source_midi,
                    'target_pitch': source_midi + shift,  # CRITICAL: explicit target pitch
                    'start_frame': entry.get('start_frame', 0),
                    'end_frame': entry.get('end_frame', 0),
                })

        print(f"  {len(self.shifted_entries)} shifted entries")

        # Preload latents for speed
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
            paths.add(entry['original_path'])

        for path in tqdm(paths, desc="Loading latents"):
            if path and Path(path).exists():
                try:
                    data = torch.load(path, map_location='cpu', weights_only=False)
                    if isinstance(data, dict):
                        self.latent_cache[path] = data['latents']
                    else:
                        self.latent_cache[path] = data
                except Exception as e:
                    pass

        print(f"  Cached {len(self.latent_cache)} latents")

    def _get_latent_window(self, path: str, start: int = None, end: int = None) -> torch.Tensor:
        """Get a window from a latent."""
        if path not in self.latent_cache:
            return None

        latent = self.latent_cache[path]
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)

        T = latent.shape[-1]

        if start is not None and end is not None:
            # Use specified window
            if end > T:
                return None
            latent = latent[:, :, :, start:end]
        else:
            # Random window
            if T < self.window_frames:
                return None
            start = np.random.randint(0, T - self.window_frames)
            latent = latent[:, :, :, start:start + self.window_frames]

        return latent.squeeze(0)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        """
        Returns dict with:
        - natural_latent: [C, H, T] natural trumpet (REAL for discriminator)
        - natural_pitch: target pitch of natural sample
        - shifted_latent: [C, H, T] DSP-shifted (input to generator)
        - shift: how many semitones shifted
        - target_pitch: what pitch the shifted should sound like
        - valid: whether all data loaded successfully
        """
        # Get natural sample
        nat_seg = self.natural_segments[np.random.randint(len(self.natural_segments))]
        natural_latent = self._get_latent_window(
            nat_seg['latent_path'],
            nat_seg['start_frame'],
            min(nat_seg['end_frame'], nat_seg['start_frame'] + self.window_frames)
        )

        # Get shifted sample
        shift_entry = self.shifted_entries[np.random.randint(len(self.shifted_entries))]
        shifted_latent = self._get_latent_window(shift_entry['shifted_path'])

        valid = (natural_latent is not None and shifted_latent is not None)

        if not valid:
            # Return dummy data
            return {
                'natural_latent': torch.zeros(8, 16, self.window_frames),
                'natural_pitch': torch.tensor(60.0),
                'shifted_latent': torch.zeros(8, 16, self.window_frames),
                'shift': torch.tensor(0.0),
                'source_pitch': torch.tensor(60.0),
                'target_pitch': torch.tensor(60.0),
                'valid': False,
            }

        return {
            'natural_latent': natural_latent,
            'natural_pitch': torch.tensor(nat_seg['pitch']),
            'shifted_latent': shifted_latent,
            'shift': torch.tensor(float(shift_entry['shift'])),
            'source_pitch': torch.tensor(shift_entry.get('source_pitch', 60.0)),
            'target_pitch': torch.tensor(shift_entry['target_pitch']),
            'valid': True,
        }


class AdversarialTrainer:
    """GAN trainer for pitch shift correction."""

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
        content_weight: float = 0.5,
        silence_weight: float = 0.1,
        # Training settings
        n_critic: int = 1,  # D updates per G update
        use_multiscale: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.n_critic = n_critic

        # Loss weights
        self.adv_weight = adv_weight
        self.content_weight = content_weight
        self.silence_weight = silence_weight

        # Dataset
        print("Creating dataset...")
        self.dataset = AdversarialDataset(
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
        self.generator = PitchShiftCorrectorGAN(
            hidden_dim=256,
            num_blocks=6,
        ).to(self.device)

        if use_multiscale:
            self.discriminator = MultiScaleDiscriminator(
                num_scales=3,
                hidden_dim=128,
            ).to(self.device)
        else:
            self.discriminator = RegisterDiscriminator(
                hidden_dim=128,
            ).to(self.device)

        print(f"Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")

        # Losses
        self.gan_loss = GANLoss(loss_type='hinge')
        self.content_loss = ContentPreservationLoss()
        self.silence_loss = FramewiseSilenceLoss()

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
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'g_lr': g_lr,
            'd_lr': d_lr,
            'adv_weight': adv_weight,
            'content_weight': content_weight,
            'silence_weight': silence_weight,
            'n_critic': n_critic,
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
        """Train discriminator: real vs fake."""
        self.discriminator.train()
        self.d_optim.zero_grad()

        # Get data
        natural_latent = batch['natural_latent'].to(self.device)
        natural_pitch = batch['natural_pitch'].to(self.device)
        shifted_latent = batch['shifted_latent'].to(self.device)
        shift = batch['shift'].to(self.device)
        target_pitch = batch['target_pitch'].to(self.device)

        # Generate fake (with target pitch conditioning!)
        with torch.no_grad():
            fake_latent = self.generator(shifted_latent, shift, target_pitch)

        # Discriminate
        real_logits = self.discriminator(natural_latent, natural_pitch)
        fake_logits = self.discriminator(fake_latent, target_pitch)

        # Handle multi-scale discriminator
        if isinstance(real_logits, list):
            d_loss = sum(
                self.gan_loss.discriminator_loss(r, f)
                for r, f in zip(real_logits, fake_logits)
            ) / len(real_logits)
        else:
            d_loss = self.gan_loss.discriminator_loss(real_logits, fake_logits)

        d_loss.backward()
        self.d_optim.step()

        return {
            'd_loss': d_loss.item(),
            'real_score': real_logits[0].mean().item() if isinstance(real_logits, list) else real_logits.mean().item(),
            'fake_score': fake_logits[0].mean().item() if isinstance(fake_logits, list) else fake_logits.mean().item(),
        }

    def train_generator_step(self, batch) -> dict:
        """Train generator: fool discriminator + preserve content."""
        self.generator.train()
        self.g_optim.zero_grad()

        # Get data
        shifted_latent = batch['shifted_latent'].to(self.device)
        shift = batch['shift'].to(self.device)
        target_pitch = batch['target_pitch'].to(self.device)

        # Generate (with target pitch conditioning!)
        fake_latent = self.generator(shifted_latent, shift, target_pitch)

        # Adversarial loss
        fake_logits = self.discriminator(fake_latent, target_pitch)
        if isinstance(fake_logits, list):
            adv_loss = sum(self.gan_loss.generator_loss(f) for f in fake_logits) / len(fake_logits)
        else:
            adv_loss = self.gan_loss.generator_loss(fake_logits)

        # Content preservation
        content_loss = self.content_loss(fake_latent, shifted_latent)

        # Silence preservation
        silence_loss = self.silence_loss(fake_latent, shifted_latent)

        # Total generator loss
        g_loss = (
            self.adv_weight * adv_loss +
            self.content_weight * content_loss +
            self.silence_weight * silence_loss
        )

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.g_optim.step()

        return {
            'g_loss': g_loss.item(),
            'adv_loss': adv_loss.item(),
            'content_loss': content_loss.item(),
            'silence_loss': silence_loss.item(),
        }

    def train_epoch(self, epoch: int) -> dict:
        """Train one epoch."""
        d_totals = {'d_loss': 0, 'real_score': 0, 'fake_score': 0}
        g_totals = {'g_loss': 0, 'adv_loss': 0, 'content_loss': 0, 'silence_loss': 0}
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

            # Train generator every n_critic steps
            if i % self.n_critic == 0:
                g_metrics = self.train_generator_step(batch)
                for k, v in g_metrics.items():
                    g_totals[k] += v
                n_g += 1

            pbar.set_postfix({
                'D': f"{d_metrics['d_loss']:.3f}",
                'G': f"{g_metrics['g_loss']:.3f}" if 'g_loss' in dir() else "...",
                'R': f"{d_metrics['real_score']:.2f}",
                'F': f"{d_metrics['fake_score']:.2f}",
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
        self.log("ADVERSARIAL PITCH SHIFT CORRECTION")
        self.log("=" * 60)
        self.log(f"Output: {self.output_dir}")
        self.log(f"Device: {self.device}")
        self.log(f"Weights: adv={self.adv_weight}, content={self.content_weight}, silence={self.silence_weight}")
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
                f"Adv: {metrics['adv_loss']:.4f} | "
                f"Cont: {metrics['content_loss']:.4f} | "
                f"R/F: {metrics['real_score']:.2f}/{metrics['fake_score']:.2f}"
                + (" [BEST]" if is_best else "")
            )

        self.log("\nTraining complete!")
        self.log(f"Best G loss: {self.best_g_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Adversarial Pitch Shift Correction")
    parser.add_argument('--segments', type=str, required=True)
    parser.add_argument('--shifted_manifest', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--samples_per_epoch', type=int, default=5000)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=4e-4)
    parser.add_argument('--adv_weight', type=float, default=1.0)
    parser.add_argument('--content_weight', type=float, default=0.5)
    parser.add_argument('--silence_weight', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    trainer = AdversarialTrainer(
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
        silence_weight=args.silence_weight,
        num_workers=args.num_workers,
        device=args.device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
