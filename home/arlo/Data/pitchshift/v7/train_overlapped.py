#!/usr/bin/env python3
"""
Train Formant Corrector (v7) with OVERLAPPED pitch ranges.

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

from models import RegisterTranslator  # Use RESIDUAL model like mute_translator
from dataset_overlapped import OverlappedFormantCorrectorDataset


class FormantCorrectorTrainer:
    """
    Training for formant correction with overlapped pitch ranges.

    Since pitches are matched, this is a true formant-only correction task.
    """

    def __init__(
        self,
        manifest_path: str = '/mnt/msdd2/pitchshift_v7_overlapped/manifest.json',
        output_dir: str = '/mnt/msdd2/pitchshift_checkpoints/formant_corrector_v7_overlapped',
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        num_epochs: int = 50,
        window_frames: int = 128,
        samples_per_epoch: int = 5000,
        pitch_tolerance: float = 3.0,
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

        # Model (RESIDUAL architecture like mute_translator - preserves input structure)
        self.model = RegisterTranslator().to(self.device)
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Dataset
        self.dataset = OverlappedFormantCorrectorDataset(
            manifest_path=manifest_path,
            window_frames=window_frames,
            samples_per_epoch=samples_per_epoch,
            pitch_tolerance=pitch_tolerance,
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

    def content_preservation_loss(self, input_latent: torch.Tensor, output_latent: torch.Tensor) -> torch.Tensor:
        """
        Preserve temporal and spectral structure from input.
        The model should modify formants, NOT melody/rhythm.
        """
        # Temporal gradient (rhythm preservation)
        input_grad = input_latent[:, :, :, 1:] - input_latent[:, :, :, :-1]
        output_grad = output_latent[:, :, :, 1:] - output_latent[:, :, :, :-1]
        temporal_loss = F.mse_loss(output_grad, input_grad)

        # Spectral gradient (pitch contour preservation)
        input_spec_grad = input_latent[:, :, 1:, :] - input_latent[:, :, :-1, :]
        output_spec_grad = output_latent[:, :, 1:, :] - output_latent[:, :, :-1, :]
        spectral_loss = F.mse_loss(output_spec_grad, input_spec_grad)

        return temporal_loss + spectral_loss

    def distribution_matching_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        HF-emphasis distribution loss (same as mute_translator).

        Key: Higher frequencies weighted more heavily - that's where formant
        differences between registers show up.
        """
        B, C, H, T = output.shape

        # Frequency band indices
        mh_start = H // 4       # 25% of H
        mh_end = H // 2         # 50% of H: mid-high
        hf_start = H // 2       # 50-100%: high frequencies (formants!)

        # Create frequency weighting - HF gets more weight
        freq_weights = torch.ones(H, device=output.device)
        freq_weights[mh_start:mh_end] = 2.0  # Mid-high: 2x
        freq_weights[hf_start:] = 3.0        # HF: 3x (formants live here)
        freq_weights = freq_weights.view(1, 1, H, 1)

        # Weighted MSE loss
        weighted_diff = ((output - target) ** 2) * freq_weights
        hf_mse_loss = weighted_diff.mean()

        # Per-channel mean/std matching
        output_mean = output.mean(dim=(2, 3))
        output_std = output.std(dim=(2, 3))
        target_mean = target.mean(dim=(2, 3))
        target_std = target.std(dim=(2, 3))

        mean_loss = F.mse_loss(output_mean, target_mean)
        std_loss = F.mse_loss(output_std, target_std)

        # Mid-high energy (formant character)
        output_mh_energy = output[:, :, mh_start:mh_end, :].abs().mean(dim=(2, 3))
        target_mh_energy = target[:, :, mh_start:mh_end, :].abs().mean(dim=(2, 3))
        mh_energy_loss = F.mse_loss(output_mh_energy, target_mh_energy)

        # High-frequency energy (brightness/formants)
        output_hf_energy = output[:, :, hf_start:, :].abs().mean(dim=(2, 3))
        target_hf_energy = target[:, :, hf_start:, :].abs().mean(dim=(2, 3))
        hf_energy_loss = F.mse_loss(output_hf_energy, target_hf_energy)

        # Simple MMD (fp32 for stability)
        output_flat = output.reshape(B, -1).float()
        target_flat = target.reshape(B, -1).float()
        scale = output_flat.shape[1] ** 0.5
        output_flat = output_flat / scale
        target_flat = target_flat / scale
        output_gram = torch.mm(output_flat, output_flat.t())
        target_gram = torch.mm(target_flat, target_flat.t())
        cross_gram = torch.mm(output_flat, target_flat.t())
        mmd = output_gram.mean() + target_gram.mean() - 2 * cross_gram.mean()

        # Combined loss (same structure as mute_translator)
        return (
            hf_mse_loss
            + mean_loss
            + std_loss
            + 0.3 * mh_energy_loss   # Mid-high formant matching
            + 0.5 * hf_energy_loss   # HF brightness matching
            + 0.1 * mmd
        )

    def silence_constraint_loss(self, input_latent: torch.Tensor, output_latent: torch.Tensor) -> torch.Tensor:
        """
        Penalize output energy when input is silent.
        Prevents artifacts/noise in gaps between notes.
        """
        input_energy = input_latent.pow(2).mean(dim=(1, 2)).sqrt()  # [B, T]
        output_energy = output_latent.pow(2).mean(dim=(1, 2)).sqrt()

        # Silent frames = low input energy
        silence_thresh = input_energy.mean() * 0.1
        silence_mask = (input_energy < silence_thresh).float()

        # Penalize output energy in silent regions
        silence_loss = (silence_mask * output_energy.pow(2)).sum() / (silence_mask.sum() + 1e-6)

        return silence_loss

    def envelope_preservation_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 8,
    ) -> torch.Tensor:
        """
        Match RMS envelope between output and target.
        Prevents compression artifacts (pumping) and preserves dynamics.
        """
        B, C, H, T = output.shape

        # Compute per-frame energy: [B, C, H, T] -> [B, T]
        output_energy = output.pow(2).mean(dim=(1, 2))
        target_energy = target.pow(2).mean(dim=(1, 2))

        # Compute RMS envelope using 1D avg pooling
        output_energy = output_energy.unsqueeze(1)  # [B, 1, T]
        target_energy = target_energy.unsqueeze(1)

        padding = window_size // 2
        output_rms = F.avg_pool1d(output_energy, window_size, stride=1, padding=padding)
        target_rms = F.avg_pool1d(target_energy, window_size, stride=1, padding=padding)

        # Trim to original size
        output_rms = output_rms[:, :, :T].sqrt()
        target_rms = target_rms[:, :, :T].sqrt()

        return F.l1_loss(output_rms, target_rms)

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

            # 1. Distribution matching: match LOW register formant statistics
            # HF-weighted to emphasize formant differences
            dist_loss = self.distribution_matching_loss(output, target)

            # 2. Silence constraint: no artifacts in gaps (strong weight!)
            silence_loss = self.silence_constraint_loss(corrupted, output)

            # 3. Envelope preservation: match dynamics
            envelope_loss = self.envelope_preservation_loss(output, target)

            # Combined loss (same as mute_translator for RESIDUAL model)
            # KEY: NO content_loss - it causes identity collapse for residual models!
            # The residual architecture already preserves structure via the skip connection.
            total_loss = (
                0.5 * dist_loss +         # Distribution matching (formants)
                1.0 * silence_loss +      # Strong silence penalty
                0.2 * envelope_loss       # Dynamics preservation
            )

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
            'dist': dist_loss.item(),
            'silence': silence_loss.item(),
            'envelope': envelope_loss.item(),
        }

    def train_epoch(self, epoch: int) -> dict:
        total_loss = 0.0
        total_dist = 0.0
        total_silence = 0.0
        total_envelope = 0.0
        n_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            total_dist += metrics.get('dist', 0)
            total_silence += metrics.get('silence', 0)
            total_envelope += metrics.get('envelope', 0)
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'dst': f"{metrics.get('dist', 0):.4f}",
                'sil': f"{metrics.get('silence', 0):.4f}",
            })

        self.scheduler.step()

        return {
            'loss': total_loss / max(n_batches, 1),
            'dist': total_dist / max(n_batches, 1),
            'silence': total_silence / max(n_batches, 1),
            'envelope': total_envelope / max(n_batches, 1),
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
        self.log("FORMANT CORRECTOR TRAINING (v7 OVERLAPPED)")
        self.log("=" * 60)
        self.log(f"Output dir: {self.output_dir}")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Shifted inputs: {len(self.dataset.shifted_entries)}")
        self.log(f"Natural LOW targets: {len(self.dataset.low_entries)}")
        self.log(f"Pitch tolerance: {self.dataset.pitch_tolerance}")
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
                f"Dist: {metrics['dist']:.4f} | "
                f"Sil: {metrics['silence']:.4f} | "
                f"Env: {metrics['envelope']:.4f} | "
                f"LR: {metrics['lr']:.2e}"
                + (" [BEST]" if is_best else "")
            )

        self.log("")
        self.log("Training complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str,
                        default='/mnt/msdd2/pitchshift_v7_overlapped/manifest.json')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/msdd2/pitchshift_checkpoints/formant_corrector_v7_overlapped')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--samples_per_epoch', type=int, default=10000)
    parser.add_argument('--pitch_tolerance', type=float, default=3.0)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    trainer = FormantCorrectorTrainer(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        samples_per_epoch=args.samples_per_epoch,
        pitch_tolerance=args.pitch_tolerance,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
    )

    trainer.train()


if __name__ == "__main__":
    main()
