#!/usr/bin/env python3
"""
Distribution Matching Training for Formant Correction

Like mute_translator: learn to match DISTRIBUTION of natural formants,
not exact reconstruction. Random pairing between shifted and natural.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/pitchshift/v9')
from models import FormantCorrectorSimple


class DistributionMatchingDataset(Dataset):
    """
    Dataset that randomly pairs shifted and natural latents.
    Like mute_translator - learns distribution, not exact mapping.
    """

    def __init__(
        self,
        manifest_path: str,
        window_frames: int = 64,
        samples_per_epoch: int = 10000,
    ):
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch

        # Load manifest
        with open(manifest_path) as f:
            data = json.load(f)

        self.pairs = data['pairs']

        # Separate by direction and collect natural/shifted latents
        self.shifted_up = []    # Shifted UP (direction=1)
        self.shifted_down = []  # Shifted DOWN (direction=0)
        self.naturals = []      # All natural latents

        print(f"Loading {len(self.pairs)} pairs...")

        for p in self.pairs:
            pair_path = p['pair_path']
            direction = p['direction']

            # Store reference to pair file
            entry = {'path': pair_path, 'direction': direction}

            if direction == 1:
                self.shifted_up.append(entry)
            else:
                self.shifted_down.append(entry)

        print(f"  Shifted UP: {len(self.shifted_up)}")
        print(f"  Shifted DOWN: {len(self.shifted_down)}")

        # Cache for loaded data
        self._cache = {}

    def _load_pair(self, path):
        if path not in self._cache:
            try:
                data = torch.load(path, map_location='cpu')
                self._cache[path] = data
                # Limit cache size
                if len(self._cache) > 500:
                    # Remove oldest
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
            except:
                return None
        return self._cache.get(path)

    def _crop_window(self, latent):
        """Random crop to window_frames."""
        T = latent.shape[-1]
        if T <= self.window_frames:
            # Pad if needed
            pad = self.window_frames - T
            latent = F.pad(latent, (0, pad))
            return latent

        start = random.randint(0, T - self.window_frames)
        return latent[..., start:start + self.window_frames]

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Randomly pick direction
        direction = random.randint(0, 1)

        # Get a shifted sample
        if direction == 1:
            shifted_entry = random.choice(self.shifted_up)
        else:
            shifted_entry = random.choice(self.shifted_down)

        # Get a RANDOM natural sample (not paired!)
        any_entry = random.choice(self.pairs)

        # Load both
        shifted_data = self._load_pair(shifted_entry['path'])
        natural_data = self._load_pair(any_entry['pair_path'])

        if shifted_data is None or natural_data is None:
            # Fallback
            return {
                'input': torch.zeros(8, 16, self.window_frames),
                'target': torch.zeros(8, 16, self.window_frames),
                'direction': 0,
                'valid': False,
            }

        # Get latents
        shifted = shifted_data['shifted']  # [C, H, T]
        natural = natural_data['natural']  # [C, H, T] - from random sample!

        # Crop windows independently
        shifted_crop = self._crop_window(shifted)
        natural_crop = self._crop_window(natural)

        return {
            'input': shifted_crop,
            'target': natural_crop,
            'direction': direction,
            'valid': True,
        }


class DistributionMatchingTrainer:
    """Train with distribution matching loss only."""

    def __init__(
        self,
        manifest_path: str,
        output_dir: str,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        num_epochs: int = 30,
        window_frames: int = 64,
        samples_per_epoch: int = 10000,
        hidden_channels: int = 128,
        use_amp: bool = True,
        device: str = "cuda",
        num_workers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.num_epochs = num_epochs

        # Model
        self.model = FormantCorrectorSimple(
            hidden_channels=hidden_channels,
            direct_output=False,
            residual_scale_init=0.3,
        ).to(self.device)

        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Dataset
        self.dataset = DistributionMatchingDataset(
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

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        self.scaler = GradScaler('cuda') if use_amp else None

        self.log_file = self.output_dir / "training.log"
        self.best_loss = float('inf')

    def distribution_loss(self, output, target):
        """
        Distribution matching loss.
        Match statistics, not exact values.
        """
        B, C, H, T = output.shape

        # 1. Per-channel mean/std matching (most important)
        out_mean = output.mean(dim=(2, 3))  # [B, C]
        out_std = output.std(dim=(2, 3))
        tgt_mean = target.mean(dim=(2, 3))
        tgt_std = target.std(dim=(2, 3))

        mean_loss = F.mse_loss(out_mean, tgt_mean)
        std_loss = F.mse_loss(out_std, tgt_std)

        # 2. Per-frequency-band energy matching
        # Low (0-25%), Mid (25-50%), High (50-75%), VHF (75-100%)
        band_loss = 0
        for start_frac, end_frac in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
            h_start = int(H * start_frac)
            h_end = int(H * end_frac)
            out_energy = output[:, :, h_start:h_end, :].pow(2).mean(dim=(2, 3))
            tgt_energy = target[:, :, h_start:h_end, :].pow(2).mean(dim=(2, 3))
            band_loss = band_loss + F.mse_loss(out_energy, tgt_energy)

        # 3. Global energy matching
        out_global = output.pow(2).mean()
        tgt_global = target.pow(2).mean()
        energy_loss = (out_global - tgt_global).pow(2)

        # 4. MMD for distribution matching
        out_flat = output.reshape(B, -1)
        tgt_flat = target.reshape(B, -1)

        # Normalize for numerical stability
        scale = out_flat.shape[1] ** 0.5
        out_flat = out_flat / (scale + 1e-8)
        tgt_flat = tgt_flat / (scale + 1e-8)

        # Gram matrices
        out_gram = torch.mm(out_flat, out_flat.t())
        tgt_gram = torch.mm(tgt_flat, tgt_flat.t())
        cross_gram = torch.mm(out_flat, tgt_flat.t())

        mmd = out_gram.mean() + tgt_gram.mean() - 2 * cross_gram.mean()

        # Combine
        total = (
            mean_loss * 1.0 +
            std_loss * 1.0 +
            band_loss * 0.5 +
            energy_loss * 0.3 +
            mmd * 0.2
        )

        return total, {
            'mean': mean_loss.item(),
            'std': std_loss.item(),
            'band': band_loss.item(),
            'energy': energy_loss.item(),
            'mmd': mmd.item(),
        }

    def train_step(self, batch):
        input_latent = batch['input'].to(self.device)
        target = batch['target'].to(self.device)
        direction = batch['direction'].to(self.device)
        valid = batch['valid'].to(self.device)

        if not valid.any():
            return {'loss': 0.0}

        # Filter valid
        input_latent = input_latent[valid]
        target = target[valid]
        direction = direction[valid]

        if input_latent.shape[0] == 0:
            return {'loss': 0.0}

        # Source group (not really used, just pass 1)
        source_group = torch.ones(input_latent.shape[0], dtype=torch.long, device=self.device)

        self.model.train()
        self.optimizer.zero_grad()

        amp_context = autocast('cuda') if self.use_amp else nullcontext()

        with amp_context:
            output = self.model(input_latent, source_group, direction)
            loss, loss_dict = self.distribution_loss(output, target)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return {'loss': loss.item(), **loss_dict}

    def log(self, msg):
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    def save_checkpoint(self, epoch, loss, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }

        torch.save(state, self.output_dir / 'latest.pt')

        if epoch % 10 == 0:
            torch.save(state, self.output_dir / f'epoch_{epoch}.pt')

        if is_best:
            torch.save(state, self.output_dir / 'best.pt')

    def train(self):
        self.log("=" * 60)
        self.log("DISTRIBUTION MATCHING TRAINING")
        self.log("=" * 60)
        self.log(f"Output dir: {self.output_dir}")

        for epoch in range(1, self.num_epochs + 1):
            epoch_losses = []

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
            for batch in pbar:
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])
                pbar.set_postfix(loss=f"{metrics['loss']:.4f}")

            self.scheduler.step()

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            lr = self.scheduler.get_last_lr()[0]

            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss

            self.save_checkpoint(epoch, avg_loss, is_best)

            best_marker = " [BEST]" if is_best else ""
            self.log(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | LR: {lr:.2e}{best_marker}")

        self.log("\nTraining complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str,
                        default='/mnt/msdd2/pitchshift_v9_formant_pairs/manifest.json')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/msdd2/pitchshift_checkpoints/formant_distmatch')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--samples_per_epoch', type=int, default=10000)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    trainer = DistributionMatchingTrainer(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        hidden_channels=args.hidden_channels,
        num_workers=args.num_workers,
    )

    trainer.train()


if __name__ == '__main__':
    main()
