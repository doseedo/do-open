#!/usr/bin/env python3
"""
v9 Training: Conditional Formant Corrector with Full Audio Pairs

Uses pre-computed paired latents from full audio files.
All pairs are formant-shifted (no sox pitch-only data).

Key features:
- Conditional on source group + direction (up/down)
- L1 loss for paired alignment
- Pre-computed latents for fast training
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

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/pitchshift/v9')

from models import ConditionalFormantCorrector, FormantCorrectorSimple
from dataset_full_paired import FullPairedDataset


class FormantCorrectorTrainerV9Full:
    """
    v9 trainer using full audio paired data.
    """

    def __init__(
        self,
        manifest_path: str = '/mnt/msdd2/pitchshift_v9_full_paired/manifest.json',
        output_dir: str = '/mnt/msdd2/pitchshift_checkpoints/formant_corrector_v9_full',
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        num_epochs: int = 50,
        window_frames: int = 6,
        samples_per_epoch: int = 10000,
        use_amp: bool = True,
        device: str = "cuda",
        num_workers: int = 4,
        model_type: str = "simple",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.use_amp = use_amp
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"Device: {self.device}")

        # Model
        if model_type == "film":
            self.model = ConditionalFormantCorrector().to(self.device)
        else:
            self.model = FormantCorrectorSimple().to(self.device)

        print(f"Model type: {model_type}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Dataset
        self.dataset = FullPairedDataset(
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
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        self.scaler = GradScaler('cuda') if self.use_amp else None

        self.log_file = self.output_dir / "training.log"
        self.best_loss = float('inf')

    def paired_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 loss for paired samples."""
        return F.l1_loss(output, target)

    def hf_weighted_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """HF-weighted L1 loss - emphasize formant-relevant frequencies."""
        B, C, H, T = output.shape

        # Higher frequencies (where formants show up) get more weight
        freq_weights = torch.ones(H, device=output.device)
        freq_weights[H//4:H//2] = 1.5   # Mid-high
        freq_weights[H//2:] = 2.0       # High frequency
        freq_weights = freq_weights.view(1, 1, H, 1)

        weighted_diff = (output - target).abs() * freq_weights
        return weighted_diff.mean()

    def train_step(self, batch: dict) -> dict:
        input_latent = batch['input'].to(self.device)
        target = batch['target'].to(self.device)
        source_group = batch['source_group'].to(self.device)
        direction = batch['direction'].to(self.device)
        valid = batch['valid'].to(self.device)

        if not valid.any():
            return {'loss': 0.0}

        # Filter valid samples
        input_latent = input_latent[valid]
        target = target[valid]
        source_group = source_group[valid]
        direction = direction[valid]

        if input_latent.shape[0] == 0:
            return {'loss': 0.0}

        self.model.train()
        self.optimizer.zero_grad()

        amp_context = autocast('cuda') if self.use_amp else nullcontext()

        with amp_context:
            output = self.model(input_latent, source_group, direction)

            # Combined loss: L1 + HF-weighted
            l1_loss = self.paired_loss(output, target)
            hf_loss = self.hf_weighted_loss(output, target)
            total_loss = l1_loss + 0.3 * hf_loss

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
            'hf': hf_loss.item(),
        }

    def train_epoch(self, epoch: int) -> dict:
        total_loss = 0.0
        total_l1 = 0.0
        total_hf = 0.0
        n_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            total_l1 += metrics.get('l1', 0)
            total_hf += metrics.get('hf', 0)
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'l1': f"{metrics.get('l1', 0):.4f}",
                'hf': f"{metrics.get('hf', 0):.4f}",
            })

        self.scheduler.step()

        return {
            'loss': total_loss / max(n_batches, 1),
            'l1': total_l1 / max(n_batches, 1),
            'hf': total_hf / max(n_batches, 1),
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
        self.log("FORMANT CORRECTOR TRAINING (v9 FULL PAIRED)")
        self.log("=" * 60)
        self.log(f"Output dir: {self.output_dir}")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Samples per epoch: {self.dataset.samples_per_epoch}")
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
                f"HF: {metrics['hf']:.4f} | "
                f"LR: {metrics['lr']:.2e}"
                + (" [BEST]" if is_best else "")
            )

        self.log("")
        self.log("Training complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str,
                        default='/mnt/msdd2/pitchshift_v9_full_paired/manifest.json')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/msdd2/pitchshift_checkpoints/formant_corrector_v9_full')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--samples_per_epoch', type=int, default=10000)
    parser.add_argument('--model_type', type=str, default='simple',
                        choices=['simple', 'film'])
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    trainer = FormantCorrectorTrainerV9Full(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        samples_per_epoch=args.samples_per_epoch,
        model_type=args.model_type,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
    )

    trainer.train()


if __name__ == "__main__":
    main()
