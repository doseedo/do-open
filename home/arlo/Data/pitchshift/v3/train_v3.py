#!/usr/bin/env python3
"""
Range-Group Pitch Shift Training V3

Like mute_translator but with range groups instead of muted/dry.

Training approach:
- Source: random clip from ANY range group
- Target: random clip from TARGET range group (different recording!)
- Conditioning: target group ID

The model learns: f(source, target_group) → characteristics of target_group
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset_v3 import RangeGroupDataset
from models_v3 import (
    RangeGroupTranslator,
    RangeGroupTranslatorDirect,
    RangeGroupTranslatorAdaptive,
    DistributionLoss,
    SilenceLoss,
    ContentPreservationLoss,
    EnvelopePreservationLoss,
)


class TrainerV3:
    """
    Trainer for V3 range-group based pitch shift.

    Uses same training strategy as mute_translator:
    - Distribution matching to target group
    - Silence constraint
    - (Optional) Content preservation for direct models
    """

    def __init__(
        self,
        segments_json: str,
        output_dir: str,
        batch_size: int = 64,
        num_epochs: int = 100,
        samples_per_epoch: int = 10000,
        learning_rate: float = 1e-4,
        model_type: str = 'residual',
        window_frames: int = 64,
        checkpoint_every: int = 10,
        dist_weight: float = 0.5,
        silence_weight: float = 1.0,
        envelope_weight: float = 0.2,
        content_weight: float = 0.5,  # Only for direct models
        identity_weight: float = 0.2,  # When source_group == target_group
        residual_reg_weight: float = 1e-4,  # Regularize residual magnitude
        early_stop_threshold: float = 1.0,
        device: str = 'cuda',
        num_workers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.checkpoint_every = checkpoint_every
        self.model_type = model_type
        self.best_loss = float('inf')
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_saved = False

        # Loss weights
        self.dist_weight = dist_weight
        self.silence_weight = silence_weight
        self.envelope_weight = envelope_weight
        self.content_weight = content_weight
        self.identity_weight = identity_weight
        self.residual_reg_weight = residual_reg_weight

        # Create dataset
        print("Creating dataset...")
        self.dataset = RangeGroupDataset(
            segments_json=segments_json,
            window_frames=window_frames,
            samples_per_epoch=samples_per_epoch,
            preload_latents=True,
        )

        self.num_groups = self.dataset.num_groups

        # DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Create model
        print(f"Creating model (type={model_type}, num_groups={self.num_groups})...")
        if model_type == 'direct':
            self.model = RangeGroupTranslatorDirect(num_groups=self.num_groups)
        elif model_type == 'adaptive':
            self.model = RangeGroupTranslatorAdaptive(num_groups=self.num_groups)
        else:
            self.model = RangeGroupTranslator(num_groups=self.num_groups)

        self.model = self.model.to(self.device)
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Loss functions (from mute_translator)
        self.dist_loss_fn = DistributionLoss()
        self.silence_loss_fn = SilenceLoss()
        self.content_loss_fn = ContentPreservationLoss()
        self.envelope_loss_fn = EnvelopePreservationLoss()

        # Optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=learning_rate / 10)

        # Save config
        config = {
            'segments_json': segments_json,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'samples_per_epoch': samples_per_epoch,
            'learning_rate': learning_rate,
            'model_type': model_type,
            'window_frames': window_frames,
            'num_groups': self.num_groups,
            'dist_weight': dist_weight,
            'silence_weight': silence_weight,
            'envelope_weight': envelope_weight,
            'content_weight': content_weight,
            'identity_weight': identity_weight,
            'residual_reg_weight': residual_reg_weight,
            'version': 'v3-range-groups',
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Logging
        self.log_file = self.output_dir / 'training.log'

    def log(self, message: str):
        """Log to file and stdout."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        valid_mask = batch['valid']
        if not valid_mask.any():
            return {'loss': 0.0}

        # Move to device
        source_latent = batch['source_latent'].to(self.device)
        target_latent = batch['target_latent'].to(self.device)
        source_group = batch['source_group'].to(self.device)
        target_group = batch['target_group'].to(self.device)

        # Filter valid samples
        source_latent = source_latent[valid_mask]
        target_latent = target_latent[valid_mask]
        source_group = source_group[valid_mask]
        target_group = target_group[valid_mask]

        # Skip if NaN
        if torch.isnan(source_latent).any() or torch.isnan(target_latent).any():
            return {'loss': 0.0}

        # Forward pass
        self.model.train()
        self.optimizer.zero_grad()

        pred_latent = self.model(source_latent, target_group)

        if torch.isnan(pred_latent).any():
            return {'loss': 0.0}

        # Losses (same structure as mute_translator)
        dist_loss = self.dist_loss_fn(pred_latent, target_latent)
        silence_loss = self.silence_loss_fn(source_latent, pred_latent)
        envelope_loss = self.envelope_loss_fn(pred_latent, target_latent)

        # Identity loss: when source_group == target_group, output should match input
        # This prevents unnecessary modifications when input already belongs to target group
        same_group_mask = (source_group == target_group)
        if same_group_mask.any():
            identity_loss = torch.nn.functional.l1_loss(
                pred_latent[same_group_mask],
                source_latent[same_group_mask]
            )
        else:
            identity_loss = torch.tensor(0.0, device=self.device)

        # Residual regularization: penalize large residuals to prevent runaway
        # residual = (pred - dry_scale * source) / residual_scale
        # We approximate by just penalizing the deviation from source
        residual_reg = (pred_latent - source_latent).pow(2).mean()

        # Combined loss
        # For RESIDUAL models: no content loss (residual connection preserves it)
        # For DIRECT models: add content loss
        if self.model_type == 'direct':
            content_loss = self.content_loss_fn(source_latent, pred_latent)
            total_loss = (
                self.dist_weight * dist_loss +
                self.silence_weight * silence_loss +
                self.envelope_weight * envelope_loss +
                self.content_weight * content_loss +
                self.identity_weight * identity_loss +
                self.residual_reg_weight * residual_reg
            )
        else:
            content_loss = torch.tensor(0.0, device=self.device)
            total_loss = (
                self.dist_weight * dist_loss +
                self.silence_weight * silence_loss +
                self.envelope_weight * envelope_loss +
                self.identity_weight * identity_loss +
                self.residual_reg_weight * residual_reg
            )

        if torch.isnan(total_loss):
            return {'loss': 0.0}

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'dist': dist_loss.item(),
            'silence': silence_loss.item(),
            'envelope': envelope_loss.item(),
            'identity': identity_loss.item(),
            'res_reg': residual_reg.item(),
            'content': content_loss.item() if self.model_type == 'direct' else 0.0,
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        totals = {'loss': 0.0, 'dist': 0.0, 'silence': 0.0, 'envelope': 0.0, 'identity': 0.0, 'res_reg': 0.0, 'content': 0.0}
        num_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            metrics = self.train_step(batch)

            if metrics['loss'] > 0:
                for k, v in metrics.items():
                    totals[k] += v
                num_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'dist': f"{metrics['dist']:.4f}",
                'sil': f"{metrics['silence']:.4f}",
            })

        self.scheduler.step()

        return {k: v / max(num_batches, 1) for k, v in totals.items()}

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_type': self.model_type,
            'num_groups': self.num_groups,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
        }

        torch.save(checkpoint, self.output_dir / 'latest.pt')

        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pt')

        if epoch % self.checkpoint_every == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch{epoch}.pt')

    def train(self):
        """Full training loop."""
        self.log("=" * 60)
        self.log("RANGE-GROUP PITCH SHIFT V3")
        self.log("Like mute_translator but with range groups!")
        self.log("=" * 60)
        self.log(f"Output: {self.output_dir}")
        self.log(f"Device: {self.device}")
        self.log(f"Model type: {self.model_type}")
        self.log(f"Num groups: {self.num_groups}")
        self.log(f"Valid groups: {self.dataset.valid_groups}")
        self.log(f"Total segments: {len(self.dataset.all_segments)}")
        self.log(f"Epochs: {self.num_epochs}")
        self.log(f"Weights: dist={self.dist_weight}, silence={self.silence_weight}, "
                 f"envelope={self.envelope_weight}, identity={self.identity_weight}, "
                 f"res_reg={self.residual_reg_weight}, content={self.content_weight}")
        self.log("=" * 60)

        for epoch in range(1, self.num_epochs + 1):
            metrics = self.train_epoch(epoch)

            is_best = metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = metrics['loss']

            # Early stop checkpoint
            early_tag = ""
            if not self.early_stop_saved and metrics['loss'] <= self.early_stop_threshold:
                checkpoint = {
                    'epoch': epoch,
                    'model_type': self.model_type,
                    'num_groups': self.num_groups,
                    'model_state_dict': self.model.state_dict(),
                    'metrics': metrics,
                }
                torch.save(checkpoint, self.output_dir / 'early_stop.pt')
                self.early_stop_saved = True
                early_tag = " [EARLY_STOP]"

            self.save_checkpoint(epoch, metrics, is_best)

            self.log(
                f"Epoch {epoch:3d} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Dist: {metrics['dist']:.4f} | "
                f"Sil: {metrics['silence']:.4f} | "
                f"Id: {metrics['identity']:.4f}"
                + (" [BEST]" if is_best else "")
                + early_tag
            )

        self.log("")
        self.log("Training complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Range-Group Pitch Shift V3")

    parser.add_argument('--segments', type=str, required=True,
                        help='Path to segments JSON from segment_by_range.py')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--samples_per_epoch', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--model_type', type=str, default='adaptive',
                        choices=['residual', 'direct', 'adaptive'],
                        help='Model type: residual (bounded), direct (no passthrough), adaptive (gated mixing - recommended)')
    parser.add_argument('--window_frames', type=int, default=64)
    parser.add_argument('--checkpoint_every', type=int, default=10)
    parser.add_argument('--dist_weight', type=float, default=0.5)
    parser.add_argument('--silence_weight', type=float, default=1.0)
    parser.add_argument('--envelope_weight', type=float, default=0.2)
    parser.add_argument('--content_weight', type=float, default=0.5,
                        help='Content preservation weight (only for direct models)')
    parser.add_argument('--identity_weight', type=float, default=0.2,
                        help='Identity loss weight when source_group == target_group')
    parser.add_argument('--residual_reg_weight', type=float, default=1e-4,
                        help='Residual magnitude regularization weight')
    parser.add_argument('--early_stop_threshold', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    trainer = TrainerV3(
        segments_json=args.segments,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        samples_per_epoch=args.samples_per_epoch,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        window_frames=args.window_frames,
        checkpoint_every=args.checkpoint_every,
        dist_weight=args.dist_weight,
        silence_weight=args.silence_weight,
        envelope_weight=args.envelope_weight,
        content_weight=args.content_weight,
        identity_weight=args.identity_weight,
        residual_reg_weight=args.residual_reg_weight,
        early_stop_threshold=args.early_stop_threshold,
        device=args.device,
        num_workers=args.num_workers,
    )

    trainer.train()


if __name__ == "__main__":
    main()
