#!/usr/bin/env python3
"""
Train Correction Model on Aligned Pairs

The final step:
1. Corruption model created aligned (corrupted, clean) pairs
2. Now train correction with frame-level supervision
3. This is what mute translator had: same recording, different processing

Key difference from previous approaches:
- NOT distribution matching (sprays energy everywhere)
- NOT adversarial only (mode collapse risk)
- Direct MSE supervision on aligned pairs
- Frame-by-frame "remove THIS artifact at THIS location"
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np


def fix_path(path: str) -> str:
    """Fix mount paths."""
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


# =============================================================================
# Simple Correction Model (pure residual)
# =============================================================================

class ShiftEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, shift: torch.Tensor) -> torch.Tensor:
        if shift.dim() == 1:
            shift = shift.unsqueeze(-1)
        return self.mlp(shift.float() / 12.0)


class FiLMConvBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.film1 = nn.Linear(cond_dim, channels * 2)
        self.film2 = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        gamma1, beta1 = self.film1(cond).chunk(2, dim=-1)
        h = h * (1 + gamma1[:, :, None, None]) + beta1[:, :, None, None]
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        gamma2, beta2 = self.film2(cond).chunk(2, dim=-1)
        h = h * (1 + gamma2[:, :, None, None]) + beta2[:, :, None, None]
        h = F.silu(h)

        return x + h


class CorrectionModel(nn.Module):
    """
    Simple correction model for aligned training.

    Pure residual: output = corrupted + learned_correction
    Learns to remove the specific artifact at each location.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        cond_dim: int = 64,
    ):
        super().__init__()

        self.shift_embed = ShiftEmbedding(cond_dim)

        self.input_proj = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            FiLMConvBlock(hidden_dim, cond_dim)
            for _ in range(num_blocks)
        ])

        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, latent_channels, 1),
        )

        # Initialize small
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, corrupted_latent: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        cond = self.shift_embed(shift)

        h = self.input_proj(corrupted_latent)

        for block in self.blocks:
            h = block(h, cond)

        correction = self.output_proj(h)

        return corrupted_latent + correction


# =============================================================================
# Dataset for Aligned Pairs
# =============================================================================

class AlignedPairsDataset(Dataset):
    """
    Dataset for aligned (corrupted, clean) pairs.

    These are perfectly aligned: same recording, same frames.
    Enables frame-level MSE supervision.
    """

    def __init__(
        self,
        pairs_manifest: str,
        window_frames: int = 64,
        samples_per_epoch: int = 10000,
    ):
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch

        print(f"Loading pairs manifest: {pairs_manifest}")
        with open(pairs_manifest) as f:
            data = json.load(f)

        self.pairs = data['pairs']
        print(f"  {len(self.pairs)} aligned pairs")
        print(f"  Shifts: {data.get('shifts', 'unknown')}")

        # Preload latents
        print("Preloading latents...")
        self.latent_cache = {}
        self._preload_latents()

    def _preload_latents(self):
        paths = set()
        for pair in self.pairs:
            paths.add(pair['corrupted_path'])
            paths.add(fix_path(pair['clean_path']))

        for path in tqdm(paths, desc="Loading"):
            if Path(path).exists():
                try:
                    data = torch.load(path, map_location='cpu', weights_only=False)
                    if isinstance(data, dict):
                        latent = data.get('latent', data.get('latents'))
                    else:
                        latent = data
                    if latent is not None:
                        if latent.dim() == 4:
                            latent = latent.squeeze(0)
                        self.latent_cache[path] = latent
                except:
                    pass

        print(f"  Cached {len(self.latent_cache)} latents")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        pair = self.pairs[np.random.randint(len(self.pairs))]

        corrupted_path = pair['corrupted_path']
        clean_path = fix_path(pair['clean_path'])

        if corrupted_path not in self.latent_cache or clean_path not in self.latent_cache:
            return self._dummy()

        corrupted = self.latent_cache[corrupted_path]
        clean_full = self.latent_cache[clean_path]

        # Extract clean segment
        start = pair['clean_start']
        end = pair['clean_end']
        clean = clean_full[:, :, start:end]

        # Match lengths
        T_corr = corrupted.shape[-1]
        T_clean = clean.shape[-1]
        T = min(T_corr, T_clean)

        if T < self.window_frames:
            return self._dummy()

        # Random window
        ws = np.random.randint(0, T - self.window_frames + 1)

        corrupted_window = corrupted[:, :, ws:ws + self.window_frames]
        clean_window = clean[:, :, ws:ws + self.window_frames]

        return {
            'corrupted': corrupted_window,
            'clean': clean_window,
            'shift': torch.tensor(float(pair['shift'])),
            'valid': True,
        }

    def _dummy(self):
        return {
            'corrupted': torch.zeros(8, 16, self.window_frames),
            'clean': torch.zeros(8, 16, self.window_frames),
            'shift': torch.tensor(0.0),
            'valid': False,
        }


# =============================================================================
# Trainer
# =============================================================================

class AlignedCorrectionTrainer:
    """
    Train correction model on aligned pairs with frame-level supervision.

    This is the key: we have (corrupted, clean) from SAME recording.
    Direct MSE tells the model exactly what to fix at each location.
    """

    def __init__(
        self,
        pairs_manifest: str,
        output_dir: str,
        batch_size: int = 64,
        num_epochs: int = 50,
        samples_per_epoch: int = 10000,
        learning_rate: float = 1e-4,
        window_frames: int = 64,
        num_workers: int = 4,
        device: str = 'cuda',
        # Loss weights
        mse_weight: float = 1.0,
        c3_weight: float = 5.0,  # Extra weight on C3 (register channel)
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.mse_weight = mse_weight
        self.c3_weight = c3_weight

        # Dataset
        print("Creating dataset...")
        self.dataset = AlignedPairsDataset(
            pairs_manifest=pairs_manifest,
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

        # Model
        print("Creating model...")
        self.model = CorrectionModel(
            hidden_dim=256,
            num_blocks=6,
        ).to(self.device)
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=learning_rate / 10)

        # Logging
        self.log_file = self.output_dir / 'training.log'
        self.best_loss = float('inf')

        # Save config
        config = {
            'model': 'correction_aligned',
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'mse_weight': mse_weight,
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

    def train_step(self, batch) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        corrupted = batch['corrupted'].to(self.device)
        clean = batch['clean'].to(self.device)
        shift = batch['shift'].to(self.device)

        # Forward
        corrected = self.model(corrupted, shift)

        # MSE loss (frame-level supervision!)
        mse_loss = F.mse_loss(corrected, clean)

        # Extra weight on C3 (register channel)
        c3_loss = F.mse_loss(corrected[:, 3], clean[:, 3])

        # Total
        total_loss = self.mse_weight * mse_loss + self.c3_weight * c3_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'mse': mse_loss.item(),
            'c3_mse': c3_loss.item(),
        }

    def train_epoch(self, epoch: int) -> dict:
        totals = {'loss': 0, 'mse': 0, 'c3_mse': 0}
        n = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            valid = batch['valid']
            if not valid.any():
                continue

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v[valid]

            metrics = self.train_step(batch)
            for k, v in metrics.items():
                totals[k] += v
            n += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'mse': f"{metrics['mse']:.4f}",
            })

        self.scheduler.step()
        return {k: v / max(n, 1) for k, v in totals.items()}

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }

        torch.save(ckpt, self.output_dir / 'latest.pt')
        if is_best:
            torch.save(ckpt, self.output_dir / 'best.pt')
        if epoch % 10 == 0:
            torch.save(ckpt, self.output_dir / f'checkpoint_epoch{epoch}.pt')

    def train(self):
        self.log("=" * 60)
        self.log("ALIGNED CORRECTION TRAINING")
        self.log("Frame-level supervision on (corrupted, clean) pairs")
        self.log("=" * 60)
        self.log(f"Output: {self.output_dir}")
        self.log(f"Device: {self.device}")
        self.log(f"MSE weight: {self.mse_weight}, C3 weight: {self.c3_weight}")
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
                f"MSE: {metrics['mse']:.4f} | "
                f"C3: {metrics['c3_mse']:.4f}"
                + (" [BEST]" if is_best else "")
            )

        self.log("\nTraining complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")
        self.log(f"\nTest with:")
        self.log(f"  python test_v4.py --checkpoint {self.output_dir}/best.pt ...")


def main():
    parser = argparse.ArgumentParser(description="Train Correction on Aligned Pairs")
    parser.add_argument('--pairs_manifest', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--samples_per_epoch', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--mse_weight', type=float, default=1.0)
    parser.add_argument('--c3_weight', type=float, default=5.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    trainer = AlignedCorrectionTrainer(
        pairs_manifest=args.pairs_manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        samples_per_epoch=args.samples_per_epoch,
        learning_rate=args.learning_rate,
        mse_weight=args.mse_weight,
        c3_weight=args.c3_weight,
        num_workers=args.num_workers,
        device=args.device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
