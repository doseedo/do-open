#!/usr/bin/env python3
"""
Train Register Translator with Cycle Consistency

Key insight: Pure distribution matching strips pitch because LOW distribution
doesn't have HIGH pitches. Cycle consistency forces pitch preservation:
  HIGH → LOW → HIGH ≈ HIGH (must preserve to reconstruct)

This is the CycleGAN approach adapted for latent register transfer.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

sys.path.insert(0, '/home/arlo/Data/pitchshift/v6')

from models import RegisterTranslatorDirect
from dataset import RegisterTranslatorDataset


class CycleTrainer:
    """
    Cycle-consistent register translator trainer.

    Two networks:
    - G_h2l: HIGH → LOW (apply low formants)
    - G_l2h: LOW → HIGH (apply high formants)

    Losses:
    - Distribution: G_h2l(HIGH) should match LOW distribution
    - Distribution: G_l2h(LOW) should match HIGH distribution
    - Cycle: G_l2h(G_h2l(HIGH)) ≈ HIGH
    - Cycle: G_h2l(G_l2h(LOW)) ≈ LOW
    """

    def __init__(
        self,
        manifest_path: str,
        output_dir: str,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        window_frames: int = 128,
        samples_per_epoch: int = 5000,
        low_threshold: float = 55.0,
        high_threshold: float = 70.0,
        cycle_weight: float = 10.0,  # Weight for cycle consistency
        use_amp: bool = True,
        device: str = "cuda",
        num_workers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.cycle_weight = cycle_weight
        self.use_amp = use_amp
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"Device: {self.device}")
        print(f"Cycle weight: {cycle_weight}")

        # Two translation networks
        self.G_h2l = RegisterTranslatorDirect().to(self.device)  # HIGH → LOW
        self.G_l2h = RegisterTranslatorDirect().to(self.device)  # LOW → HIGH

        total_params = sum(p.numel() for p in self.G_h2l.parameters()) + \
                       sum(p.numel() for p in self.G_l2h.parameters())
        print(f"Total params (2 networks): {total_params:,}")

        # Dataset
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

        # Single optimizer for both networks
        all_params = list(self.G_h2l.parameters()) + list(self.G_l2h.parameters())
        self.optimizer = AdamW(all_params, lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        self.scaler = GradScaler('cuda') if self.use_amp else None

        self.log_file = self.output_dir / "training.log"
        self.best_loss = float('inf')

    def distribution_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Match distribution statistics."""
        # MSE
        mse = F.mse_loss(pred, target)

        # Mean/std matching
        pred_mean = pred.mean(dim=(2, 3))
        pred_std = pred.std(dim=(2, 3))
        target_mean = target.mean(dim=(2, 3))
        target_std = target.std(dim=(2, 3))

        mean_loss = F.mse_loss(pred_mean, target_mean)
        std_loss = F.mse_loss(pred_std, target_std)

        return mse + mean_loss + std_loss

    def cycle_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Cycle consistency: original ≈ reconstructed."""
        return F.l1_loss(reconstructed, original)

    def train_step(self, batch: dict) -> dict:
        """Single training step with cycle consistency."""
        high = batch['high_latent'].to(self.device)
        low = batch['low_latent'].to(self.device)
        valid = batch['valid'].to(self.device)

        if not valid.any():
            return {'loss': 0.0}

        high = high[valid]
        low = low[valid]

        self.G_h2l.train()
        self.G_l2h.train()
        self.optimizer.zero_grad()

        amp_context = autocast('cuda') if self.use_amp else nullcontext()

        with amp_context:
            # Forward translations
            fake_low = self.G_h2l(high)      # HIGH → fake LOW
            fake_high = self.G_l2h(low)      # LOW → fake HIGH

            # Cycle reconstructions
            recon_high = self.G_l2h(fake_low)  # HIGH → LOW → HIGH
            recon_low = self.G_h2l(fake_high)  # LOW → HIGH → LOW

            # Distribution losses (make fakes match target domain)
            dist_h2l = self.distribution_loss(fake_low, low)
            dist_l2h = self.distribution_loss(fake_high, high)

            # Cycle consistency losses (reconstruct originals)
            cycle_h = self.cycle_loss(high, recon_high)
            cycle_l = self.cycle_loss(low, recon_low)

            # Total loss
            dist_loss = dist_h2l + dist_l2h
            cyc_loss = cycle_h + cycle_l
            total_loss = dist_loss + self.cycle_weight * cyc_loss

        # Backward
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.G_h2l.parameters()) + list(self.G_l2h.parameters()), 1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.G_h2l.parameters()) + list(self.G_l2h.parameters()), 1.0
            )
            self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'dist_loss': dist_loss.item(),
            'cycle_loss': cyc_loss.item(),
            'dist_h2l': dist_h2l.item(),
            'dist_l2h': dist_l2h.item(),
        }

    def train_epoch(self, epoch: int) -> dict:
        total_loss = 0.0
        total_dist = 0.0
        total_cycle = 0.0
        n_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            total_dist += metrics.get('dist_loss', 0)
            total_cycle += metrics.get('cycle_loss', 0)
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'dist': f"{metrics.get('dist_loss', 0):.4f}",
                'cyc': f"{metrics.get('cycle_loss', 0):.4f}",
            })

        self.scheduler.step()

        return {
            'loss': total_loss / max(n_batches, 1),
            'dist_loss': total_dist / max(n_batches, 1),
            'cycle_loss': total_cycle / max(n_batches, 1),
            'lr': self.scheduler.get_last_lr()[0],
        }

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'G_h2l_state_dict': self.G_h2l.state_dict(),
            'G_l2h_state_dict': self.G_l2h.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'cycle_weight': self.cycle_weight,
        }

        torch.save(checkpoint, self.output_dir / "latest.pt")

        if is_best:
            torch.save(checkpoint, self.output_dir / "best.pt")

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, self.output_dir / f"epoch_{epoch+1}.pt")

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def train(self):
        self.log("=" * 60)
        self.log("CYCLE-CONSISTENT REGISTER TRANSLATOR (v6)")
        self.log("=" * 60)
        self.log(f"Output dir: {self.output_dir}")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Cycle weight: {self.cycle_weight}")
        self.log(f"HIGH samples: {len(self.dataset.high_entries)}")
        self.log(f"LOW samples: {len(self.dataset.low_entries)}")
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
                f"Cycle: {metrics['cycle_loss']:.4f} | "
                f"LR: {metrics['lr']:.2e}"
                + (" [BEST]" if is_best else "")
            )

        self.log("")
        self.log("Training complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Cycle-Consistent Register Translator")
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/msdd2/pitchshift_checkpoints/register_v6_cycle')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--samples_per_epoch', type=int, default=10000)
    parser.add_argument('--cycle_weight', type=float, default=10.0,
                        help='Weight for cycle consistency loss (default: 10)')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    trainer = CycleTrainer(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        samples_per_epoch=args.samples_per_epoch,
        cycle_weight=args.cycle_weight,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
    )

    trainer.train()


if __name__ == "__main__":
    main()
