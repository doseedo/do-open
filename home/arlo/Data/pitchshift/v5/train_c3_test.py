#!/usr/bin/env python3
"""
V5 C3-Focused Training Test

Quick validation: does heavily weighting C3 improve register naturalness?

Based on probe finding:
- C3 encodes most register information
- DSP-shifted C3 is -4.03 from natural target
- This is THE lever for register correction

Train briefly (20 epochs), listen, validate hypothesis.
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset_simple import PitchShiftCorrectionDatasetSimple
from models_simple import (
    PitchShiftCorrectorSimple,
    C3FocusedDistributionLoss,
    LoudnessLoss,
)


class C3TestTrainer:
    """Quick C3-focused training test."""

    def __init__(
        self,
        shifted_manifest: str,
        segments_json: str,
        output_dir: str,
        batch_size: int = 64,
        num_epochs: int = 20,
        samples_per_epoch: int = 5000,
        learning_rate: float = 1e-4,
        window_frames: int = 64,
        c3_weight: float = 10.0,
        loudness_weight: float = 5.0,
        device: str = 'cuda',
        num_workers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.best_loss = float('inf')
        self.c3_weight = c3_weight
        self.loudness_weight = loudness_weight

        # Dataset
        print("Creating dataset...")
        self.dataset = PitchShiftCorrectionDatasetSimple(
            shifted_manifest=shifted_manifest,
            segments_json=segments_json,
            window_frames=window_frames,
            samples_per_epoch=samples_per_epoch,
            preload_latents=True,
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

        # Model (with conditioning)
        print("Creating model...")
        self.model = PitchShiftCorrectorSimple(
            hidden_dim=256,
            num_blocks=6,
            conditioning=True,
        ).to(self.device)
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Losses
        self.c3_loss_fn = C3FocusedDistributionLoss(c3_weight=c3_weight)
        self.loudness_loss_fn = LoudnessLoss()

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=learning_rate / 10)

        # Save config
        config = {
            'experiment': 'c3_focused_test',
            'c3_weight': c3_weight,
            'loudness_weight': loudness_weight,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        self.log_file = self.output_dir / 'training.log'

    def log(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def train_step(self, batch):
        valid_mask = batch['valid']
        if not valid_mask.any():
            return None

        source_latent = batch['source_latent'].to(self.device)[valid_mask]
        target_latent = batch['target_latent'].to(self.device)[valid_mask]
        shift = batch['shift'].to(self.device)[valid_mask]

        if torch.isnan(source_latent).any() or torch.isnan(target_latent).any():
            return None

        self.model.train()
        self.optimizer.zero_grad()

        pred_latent = self.model(source_latent, shift)

        if torch.isnan(pred_latent).any():
            return None

        # C3-focused distribution loss
        c3_losses = self.c3_loss_fn(pred_latent, target_latent)
        c3_loss = c3_losses['total']

        # Loudness lock to source
        loud_loss = self.loudness_loss_fn(pred_latent, source_latent)

        # Total
        total_loss = c3_loss + self.loudness_weight * loud_loss

        if torch.isnan(total_loss):
            return None

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'c3_total': c3_losses['c3_total'].item(),
            'c3_energy': c3_losses['c3_energy'].item(),
            'other': c3_losses['other_total'].item(),
            'loud': loud_loss.item(),
        }

    def train_epoch(self, epoch: int):
        totals = {'loss': 0, 'c3_total': 0, 'c3_energy': 0, 'other': 0, 'loud': 0}
        n = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            metrics = self.train_step(batch)
            if metrics:
                for k, v in metrics.items():
                    totals[k] += v
                n += 1
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'c3': f"{metrics['c3_total']:.4f}",
                })

        self.scheduler.step()
        return {k: v / max(n, 1) for k, v in totals.items()}

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'c3_weight': self.c3_weight,
        }
        torch.save(ckpt, self.output_dir / 'latest.pt')
        if is_best:
            torch.save(ckpt, self.output_dir / 'best.pt')

    def train(self):
        self.log("=" * 60)
        self.log("C3-FOCUSED TRAINING TEST")
        self.log(f"C3 weight: {self.c3_weight}x")
        self.log(f"Loudness weight: {self.loudness_weight}")
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
                f"C3: {metrics['c3_total']:.4f} | "
                f"C3_energy: {metrics['c3_energy']:.4f} | "
                f"Loud: {metrics['loud']:.4f}"
                + (" [BEST]" if is_best else "")
            )

        self.log("\nTraining complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")
        self.log(f"\nRun listening test with:")
        self.log(f"  python test_v4.py --checkpoint {self.output_dir}/best.pt ...")


def main():
    parser = argparse.ArgumentParser(description="C3-Focused Training Test")
    parser.add_argument('--shifted_manifest', type=str, required=True)
    parser.add_argument('--segments', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--samples_per_epoch', type=int, default=5000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--c3_weight', type=float, default=10.0)
    parser.add_argument('--loudness_weight', type=float, default=5.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    trainer = C3TestTrainer(
        shifted_manifest=args.shifted_manifest,
        segments_json=args.segments,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        samples_per_epoch=args.samples_per_epoch,
        learning_rate=args.learning_rate,
        c3_weight=args.c3_weight,
        loudness_weight=args.loudness_weight,
        num_workers=args.num_workers,
        device=args.device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
