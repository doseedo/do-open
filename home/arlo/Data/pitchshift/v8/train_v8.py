"""
Register-Aware Pitch Shift Training V8

DISTRIBUTION MATCHING approach (like mute translator).

Training:
    Input:  source_latent at pitch A
    Target: random real latent at pitch B (DIFFERENT recording!)

The model learns: f(source, src_pitch, tgt_pitch) -> matches distribution of pitch B
while preserving structural content from source.

V8 Changes from V2:
1. Residual model now uses content_loss (was only in direct model before)
2. silence_loss weight increased from 1.0 to 5.0 (reduce noise in silence)
3. models_v8.py has residual_scale=0.3 (was 0.1)

Losses (from mute_translator):
1. Distribution loss: Match statistics of target pitch
2. Content preservation: Preserve temporal/spectral structure from source
3. Silence loss: Penalize output when input is silent
"""

import os
import argparse
from pathlib import Path
from typing import Dict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset_v8 import RegisterTransferDatasetV8
from models_v8 import RegisterTranslator, RegisterTranslatorDirect


class TrainerV8:
    """
    Trainer for V8 register timbre conversion using DISTRIBUTION MATCHING.

    Like mute_translator - learns to match distribution of target pitch
    while preserving content structure from source.

    V8 improvements:
    - Content loss now applied to residual model too (helps preserve pitch)
    - Stronger silence loss (5.0 vs 1.0) to reduce noise in quiet sections
    """

    def __init__(
        self,
        manifest_path: str,
        output_dir: str,
        instrument: str = 'trumpet',
        batch_size: int = 64,
        num_epochs: int = 100,
        samples_per_epoch: int = 10000,
        learning_rate: float = 1e-4,
        model_type: str = 'residual',
        window_frames: int = 64,
        shift_range: tuple = (-12, 12),
        checkpoint_every: int = 10,
        dist_weight: float = 0.5,
        content_weight: float = 0.3,  # V8: content loss for residual model
        silence_weight: float = 5.0,  # V8: increased from 1.0
        device: str = 'cuda',
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.checkpoint_every = checkpoint_every
        self.best_loss = float('inf')
        self.dist_weight = dist_weight
        self.content_weight = content_weight
        self.silence_weight = silence_weight
        self.model_type = model_type

        # Create dataset
        print("Creating dataset...")
        self.dataset = RegisterTransferDatasetV8(
            manifest_path=manifest_path,
            instrument=instrument,
            window_frames=window_frames,
            shift_range=shift_range,
            samples_per_epoch=samples_per_epoch,
            min_samples_per_pitch=5,
            preload_latents=True,
        )

        # DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        # Create model
        print(f"Creating model (type={model_type})...")
        if model_type == 'direct':
            self.model = RegisterTranslatorDirect()
        else:
            self.model = RegisterTranslator()

        self.model = self.model.to(self.device)
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # V8: Print residual scale if applicable
        if hasattr(self.model, 'residual_scale'):
            print(f"Residual scale: {self.model.residual_scale.item()}")

        # Optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=learning_rate / 10)

        # Save config
        config = {
            'manifest_path': manifest_path,
            'instrument': instrument,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'samples_per_epoch': samples_per_epoch,
            'learning_rate': learning_rate,
            'model_type': model_type,
            'window_frames': window_frames,
            'shift_range': list(shift_range),
            'dist_weight': dist_weight,
            'content_weight': content_weight,
            'silence_weight': silence_weight,
            'version': 'v8-distribution-matching-improved',
            'changes_from_v2': [
                'residual_scale increased to 0.3 (was 0.1)',
                'content_loss now used for residual model too',
                'silence_weight increased to 5.0 (was 1.0)',
            ],
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def distribution_loss(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Frequency-weighted distribution matching loss (from mute_translator).

        Emphasizes important frequency bands for timbre conversion.
        """
        B, C, H, T = pred_latent.shape

        # Frequency band weighting (like mute translator)
        # Upper frequencies often carry more timbre information
        freq_weights = torch.ones(H, device=pred_latent.device)
        freq_weights[H // 4:H // 2] = 1.5   # Mid-high
        freq_weights[H // 2:] = 2.0          # High frequencies
        freq_weights = freq_weights.view(1, 1, H, 1)

        # Weighted MSE loss
        weighted_diff = ((pred_latent - target_latent) ** 2) * freq_weights
        mse_loss = weighted_diff.mean()

        # Per-channel statistics matching
        pred_mean = pred_latent.mean(dim=(2, 3))
        pred_std = pred_latent.std(dim=(2, 3)) + 1e-6
        target_mean = target_latent.mean(dim=(2, 3))
        target_std = target_latent.std(dim=(2, 3)) + 1e-6

        mean_loss = F.mse_loss(pred_mean, target_mean)
        std_loss = F.mse_loss(pred_std, target_std)

        # Energy matching per frequency band
        pred_energy = pred_latent.abs().mean(dim=(0, 1, 3))  # [H]
        target_energy = target_latent.abs().mean(dim=(0, 1, 3))  # [H]
        energy_loss = F.l1_loss(pred_energy, target_energy)

        return mse_loss + mean_loss + std_loss + 0.5 * energy_loss

    def content_preservation_loss(
        self,
        source_latent: torch.Tensor,
        pred_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Preserve structural content from source (from mute_translator).

        The transformation should modify timbre, not destroy melody/rhythm.

        V8: Now used for BOTH residual and direct models to help preserve pitch.
        """
        # Temporal gradient (rhythm preservation)
        source_grad = source_latent[:, :, :, 1:] - source_latent[:, :, :, :-1]
        pred_grad = pred_latent[:, :, :, 1:] - pred_latent[:, :, :, :-1]
        grad_loss = F.mse_loss(pred_grad.abs(), source_grad.abs())

        # Spectral gradient (pitch contour preservation)
        source_spec = source_latent[:, :, 1:, :] - source_latent[:, :, :-1, :]
        pred_spec = pred_latent[:, :, 1:, :] - pred_latent[:, :, :-1, :]
        spec_loss = F.mse_loss(pred_spec.abs(), source_spec.abs())

        return grad_loss + spec_loss

    def silence_loss(
        self,
        source_latent: torch.Tensor,
        pred_latent: torch.Tensor,
        threshold: float = 0.05,
    ) -> torch.Tensor:
        """
        Penalize output energy when input is silent (from mute_translator).

        V8: Weight increased to 5.0 (was 1.0) to better suppress noise in silence.
        """
        source_energy = source_latent.pow(2).mean(dim=(1, 2)).sqrt() + 1e-8  # [B, T]
        pred_energy = pred_latent.pow(2).mean(dim=(1, 2)).sqrt() + 1e-8  # [B, T]

        silence_mask = (source_energy < threshold).float()
        mask_sum = silence_mask.sum()

        if mask_sum < 1:
            return torch.tensor(0.0, device=source_latent.device)

        silence_loss = (silence_mask * pred_energy.pow(2)).sum() / (mask_sum + 1e-6)

        return silence_loss

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()

        total_loss = 0.0
        loss_counts = {'dist': 0.0, 'content': 0.0, 'silence': 0.0}
        num_batches = 0

        skip_reasons = {'invalid': 0, 'input_nan': 0, 'output_nan': 0, 'loss_nan': 0}
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            valid_mask = batch['valid']
            if not valid_mask.any():
                skip_reasons['invalid'] += 1
                continue

            # Move to device
            source_latent = batch['source_latent'].to(self.device)
            target_latent = batch['target_latent'].to(self.device)
            source_pitch = batch['source_pitch'].to(self.device)
            target_pitch = batch['target_pitch'].to(self.device)

            # Skip if data has NaN
            if torch.isnan(source_latent).any() or torch.isnan(target_latent).any():
                skip_reasons['input_nan'] += 1
                continue

            # Forward pass
            self.optimizer.zero_grad()
            pred_latent = self.model(source_latent, source_pitch, target_pitch)

            # Skip if output has NaN
            if torch.isnan(pred_latent).any():
                skip_reasons['output_nan'] += 1
                continue

            # Distribution matching loss (match target pitch distribution)
            dist_loss = self.distribution_loss(pred_latent, target_latent)

            # Content preservation (keep structure from source)
            content_loss = self.content_preservation_loss(source_latent, pred_latent)

            # Silence constraint
            silence_loss = self.silence_loss(source_latent, pred_latent)

            # V8: Combined loss - BOTH models now use content_loss
            # Residual model: content helps preserve pitch contour despite residual connection
            # Direct model: content is essential (no passthrough)
            if self.model_type == 'direct':
                # Direct: higher content weight since no residual passthrough
                total = (self.dist_weight * dist_loss +
                        0.5 * content_loss +
                        self.silence_weight * silence_loss)
            else:
                # V8 Residual: Now includes content_loss (was missing in V2!)
                total = (self.dist_weight * dist_loss +
                        self.content_weight * content_loss +
                        self.silence_weight * silence_loss)

            # Skip if loss is NaN
            if torch.isnan(total):
                skip_reasons['loss_nan'] += 1
                continue

            # Backward
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Track
            total_loss += total.item()
            loss_counts['dist'] += dist_loss.item()
            loss_counts['content'] += content_loss.item()
            loss_counts['silence'] += silence_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{total.item():.4f}",
                'dist': f"{dist_loss.item():.4f}",
                'sil': f"{silence_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

        avg_loss = total_loss / max(num_batches, 1)
        avg_counts = {k: v / max(num_batches, 1) for k, v in loss_counts.items()}

        # Print skip stats if many skipped
        total_skipped = sum(skip_reasons.values())
        if total_skipped > 0:
            print(f"  Skipped: {skip_reasons} (trained on {num_batches} batches)")

        return {'loss': avg_loss, **avg_counts}

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }

        if epoch % self.checkpoint_every == 0:
            torch.save(checkpoint, self.output_dir / f"checkpoint_epoch{epoch}.pt")

        if is_best:
            torch.save(checkpoint, self.output_dir / "best.pt")

        torch.save(checkpoint, self.output_dir / "latest.pt")

    def train(self):
        """Run full training loop."""
        print("\n" + "=" * 60)
        print("REGISTER TIMBRE V8 - DISTRIBUTION MATCHING (IMPROVED)")
        print("=" * 60)
        print(f"Output: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Valid pitches: {len(self.dataset.valid_pitches)}")
        print(f"Model type: {self.model_type}")
        print(f"Distribution weight: {self.dist_weight}")
        print(f"Content weight: {self.content_weight}")
        print(f"Silence weight: {self.silence_weight}")
        print("V8 improvements:")
        print("  - Content loss for residual model (pitch preservation)")
        print("  - Stronger silence penalty (noise reduction)")
        print("  - Higher residual_scale=0.3 (stronger transformation)")
        print("=" * 60 + "\n")

        for epoch in range(1, self.num_epochs + 1):
            metrics = self.train_epoch(epoch)
            self.scheduler.step()

            print(f"Epoch {epoch}: loss={metrics['loss']:.4f}, "
                  f"dist={metrics['dist']:.4f}, "
                  f"content={metrics['content']:.4f}, "
                  f"silence={metrics['silence']:.4f}")

            is_best = metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = metrics['loss']
            self.save_checkpoint(epoch, metrics['loss'], is_best)

        print("\nTraining complete!")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Register Timbre V8")

    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--instrument', type=str, default='trumpet')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--samples_per_epoch', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--model_type', type=str, default='residual',
                        choices=['residual', 'direct'])
    parser.add_argument('--window_frames', type=int, default=64)
    parser.add_argument('--shift_range', type=int, nargs=2, default=[-12, 12])
    parser.add_argument('--checkpoint_every', type=int, default=10)
    parser.add_argument('--dist_weight', type=float, default=0.5,
                        help='Weight for distribution loss')
    parser.add_argument('--content_weight', type=float, default=0.3,
                        help='Weight for content preservation loss (V8: now used for residual too)')
    parser.add_argument('--silence_weight', type=float, default=5.0,
                        help='Weight for silence loss (V8: increased from 1.0)')

    args = parser.parse_args()

    trainer = TrainerV8(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        instrument=args.instrument,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        samples_per_epoch=args.samples_per_epoch,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        window_frames=args.window_frames,
        shift_range=tuple(args.shift_range),
        checkpoint_every=args.checkpoint_every,
        dist_weight=args.dist_weight,
        content_weight=args.content_weight,
        silence_weight=args.silence_weight,
    )

    trainer.train()


if __name__ == "__main__":
    main()
