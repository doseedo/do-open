#!/usr/bin/env python3
"""
Train Formant Corrector (v7)

Key insight: Both input and target at SAME pitch range.
- Input: sox-shifted HIGH (low pitch, wrong formants)
- Target: natural LOW (low pitch, correct formants)

Model learns to fix formant artifacts, not transfer across pitch ranges.
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

from models import RegisterTranslatorDirect  # Reuse v6 model

# Import v7 dataset directly
import importlib.util
spec = importlib.util.spec_from_file_location("dataset_v7", "/home/arlo/Data/pitchshift/v7/dataset.py")
dataset_v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_v7)
PrecomputedFormantCorrectorDataset = dataset_v7.PrecomputedFormantCorrectorDataset


class FormantCorrectorTrainer:
    """
    Simpler training - just learn corrupted → correct mapping.
    No distribution matching tricks needed since pitches align.
    """

    def __init__(
        self,
        v4_manifest_path: str = '/mnt/msdd2/pitchshift_v4_precomputed/shifted_manifest.json',
        mute_manifest_path: str = '/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json',
        output_dir: str = '/mnt/msdd2/pitchshift_checkpoints/formant_corrector_v7',
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        num_epochs: int = 50,
        window_frames: int = 128,
        samples_per_epoch: int = 5000,
        use_amp: bool = True,
        device: str = "cuda",
        num_workers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.use_amp = use_amp
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"Device: {self.device}")

        # Model (reuse v6 direct architecture)
        self.model = RegisterTranslatorDirect().to(self.device)
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Dataset - uses v4 precomputed shifted latents
        self.dataset = PrecomputedFormantCorrectorDataset(
            v4_manifest_path=v4_manifest_path,
            mute_manifest_path=mute_manifest_path,
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
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        self.scaler = GradScaler('cuda') if self.use_amp else None

        self.log_file = self.output_dir / "training.log"
        self.best_loss = float('inf')

    def train_step(self, batch: dict) -> dict:
        corrupted = batch['corrupted'].to(self.device)
        target = batch['target'].to(self.device)
        valid = batch['valid'].to(self.device)

        if not valid.any():
            return {'loss': 0.0}

        corrupted = corrupted[valid]
        target = target[valid]

        self.model.train()
        self.optimizer.zero_grad()

        amp_context = autocast('cuda') if self.use_amp else nullcontext()

        with amp_context:
            output = self.model(corrupted)

            # Simple L1 + L2 reconstruction loss
            l1_loss = F.l1_loss(output, target)
            l2_loss = F.mse_loss(output, target)

            # Spectral consistency (preserve frequency structure)
            output_spec = torch.fft.rfft(output, dim=-1).abs()
            target_spec = torch.fft.rfft(target, dim=-1).abs()
            spec_loss = F.l1_loss(output_spec, target_spec)

            total_loss = l1_loss + 0.5 * l2_loss + 0.1 * spec_loss

        if self.use_amp and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'l1': l1_loss.item(),
            'l2': l2_loss.item(),
            'spec': spec_loss.item(),
        }

    def train_epoch(self, epoch: int) -> dict:
        total_loss = 0.0
        total_l1 = 0.0
        total_l2 = 0.0
        total_spec = 0.0
        n_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            total_l1 += metrics.get('l1', 0)
            total_l2 += metrics.get('l2', 0)
            total_spec += metrics.get('spec', 0)
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'l1': f"{metrics.get('l1', 0):.4f}",
            })

        self.scheduler.step()

        return {
            'loss': total_loss / max(n_batches, 1),
            'l1': total_l1 / max(n_batches, 1),
            'l2': total_l2 / max(n_batches, 1),
            'spec': total_spec / max(n_batches, 1),
            'lr': self.scheduler.get_last_lr()[0],
        }

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
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
        self.log("FORMANT CORRECTOR TRAINING (v7)")
        self.log("=" * 60)
        self.log(f"Output dir: {self.output_dir}")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Shifted inputs: {len(self.dataset.shifted_entries)}")
        self.log(f"Natural LOW targets: {len(self.dataset.low_entries)}")
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
                f"L1: {metrics['l1']:.4f} | "
                f"L2: {metrics['l2']:.4f} | "
                f"Spec: {metrics['spec']:.4f} | "
                f"LR: {metrics['lr']:.2e}"
                + (" [BEST]" if is_best else "")
            )

        self.log("")
        self.log("Training complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v4_manifest', type=str,
                        default='/mnt/msdd2/pitchshift_v4_precomputed/shifted_manifest.json')
    parser.add_argument('--mute_manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/msdd2/pitchshift_checkpoints/formant_corrector_v7')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--samples_per_epoch', type=int, default=10000)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    trainer = FormantCorrectorTrainer(
        v4_manifest_path=args.v4_manifest,
        mute_manifest_path=args.mute_manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        samples_per_epoch=args.samples_per_epoch,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
    )

    trainer.train()


if __name__ == "__main__":
    main()
