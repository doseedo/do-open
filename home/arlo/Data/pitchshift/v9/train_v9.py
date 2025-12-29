#!/usr/bin/env python3
"""
v9 Training: Conditional Formant Corrector with Mixed Losses

Key features:
- Conditional on source group + direction (up/down)
- Mixed training:
  - Paired samples (formant-shifted): L1 loss for content alignment
  - Unpaired samples (sox-shifted): distribution matching loss
- Architecture from mute_translator (residual, no content loss)

Loss structure:
- Paired (50%): L1 loss (exact content alignment)
- Unpaired (50%): HF-weighted distribution matching + envelope + silence
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
from dataset_paired import FormantCorrectorDatasetPaired


class FormantCorrectorTrainerV9:
    """
    v9 trainer with conditional model and mixed losses.
    """

    def __init__(
        self,
        paired_manifest_path: str = '/mnt/msdd2/pitchshift_v9_paired/manifest.json',
        sox_manifest_path: str = '/mnt/msdd2/pitchshift_v7_overlapped/manifest.json',
        output_dir: str = '/mnt/msdd2/pitchshift_checkpoints/formant_corrector_v9',
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        num_epochs: int = 50,
        window_frames: int = 6,
        samples_per_epoch: int = 10000,
        paired_ratio: float = 0.5,  # 50% paired, 50% sox
        use_amp: bool = True,
        device: str = "cuda",
        num_workers: int = 4,
        model_type: str = "simple",  # "film" or "simple"
        direct_output: bool = False,  # If True, no residual connection
        hidden_channels: int = 64,
        residual_scale: float = 0.5,
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
            self.model = FormantCorrectorSimple(
                hidden_channels=hidden_channels,
                direct_output=direct_output,
                residual_scale_init=residual_scale,
            ).to(self.device)

        print(f"Model type: {model_type}, direct_output: {direct_output}, hidden_channels: {hidden_channels}, residual_scale: {residual_scale}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Dataset - uses ACTUAL paired data + sox unpaired
        self.dataset = FormantCorrectorDatasetPaired(
            paired_manifest_path=paired_manifest_path,
            sox_manifest_path=sox_manifest_path,
            window_frames=window_frames,
            samples_per_epoch=samples_per_epoch,
            paired_ratio=paired_ratio,
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

    def distribution_matching_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        HF-emphasis distribution loss (from mute_translator).

        Higher frequencies weighted more heavily - that's where formant
        differences show up.
        """
        B, C, H, T = output.shape

        # Frequency band indices
        mh_start = H // 4
        mh_end = H // 2
        hf_start = H // 2

        # Frequency weighting
        freq_weights = torch.ones(H, device=output.device)
        freq_weights[mh_start:mh_end] = 2.0  # Mid-high: 2x
        freq_weights[hf_start:] = 3.0        # HF: 3x (formants)
        freq_weights = freq_weights.view(1, 1, H, 1)

        # Weighted MSE
        weighted_diff = ((output - target) ** 2) * freq_weights
        hf_mse_loss = weighted_diff.mean()

        # Per-channel mean/std matching
        output_mean = output.mean(dim=(2, 3))
        output_std = output.std(dim=(2, 3))
        target_mean = target.mean(dim=(2, 3))
        target_std = target.std(dim=(2, 3))

        mean_loss = F.mse_loss(output_mean, target_mean)
        std_loss = F.mse_loss(output_std, target_std)

        # Mid-high energy
        output_mh_energy = output[:, :, mh_start:mh_end, :].abs().mean(dim=(2, 3))
        target_mh_energy = target[:, :, mh_start:mh_end, :].abs().mean(dim=(2, 3))
        mh_energy_loss = F.mse_loss(output_mh_energy, target_mh_energy)

        # HF energy
        output_hf_energy = output[:, :, hf_start:, :].abs().mean(dim=(2, 3))
        target_hf_energy = target[:, :, hf_start:, :].abs().mean(dim=(2, 3))
        hf_energy_loss = F.mse_loss(output_hf_energy, target_hf_energy)

        # MMD (fp32 for stability)
        output_flat = output.reshape(B, -1).float()
        target_flat = target.reshape(B, -1).float()
        scale = output_flat.shape[1] ** 0.5
        output_flat = output_flat / scale
        target_flat = target_flat / scale
        output_gram = torch.mm(output_flat, output_flat.t())
        target_gram = torch.mm(target_flat, target_flat.t())
        cross_gram = torch.mm(output_flat, target_flat.t())
        mmd = output_gram.mean() + target_gram.mean() - 2 * cross_gram.mean()

        return (
            hf_mse_loss
            + mean_loss
            + std_loss
            + 0.3 * mh_energy_loss
            + 0.5 * hf_energy_loss
            + 0.1 * mmd
        )

    def silence_constraint_loss(self, input_latent: torch.Tensor, output_latent: torch.Tensor) -> torch.Tensor:
        """Penalize output energy when input is silent."""
        input_energy = input_latent.pow(2).mean(dim=(1, 2)).sqrt()  # [B, T]
        output_energy = output_latent.pow(2).mean(dim=(1, 2)).sqrt()

        silence_thresh = input_energy.mean() * 0.1
        silence_mask = (input_energy < silence_thresh).float()

        silence_loss = (silence_mask * output_energy.pow(2)).sum() / (silence_mask.sum() + 1e-6)
        return silence_loss

    def envelope_preservation_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 8,
    ) -> torch.Tensor:
        """Match RMS envelope between output and target."""
        B, C, H, T = output.shape

        output_energy = output.pow(2).mean(dim=(1, 2))
        target_energy = target.pow(2).mean(dim=(1, 2))

        output_energy = output_energy.unsqueeze(1)
        target_energy = target_energy.unsqueeze(1)

        padding = window_size // 2
        output_rms = F.avg_pool1d(output_energy, window_size, stride=1, padding=padding)
        target_rms = F.avg_pool1d(target_energy, window_size, stride=1, padding=padding)

        output_rms = output_rms[:, :, :T].sqrt()
        target_rms = target_rms[:, :, :T].sqrt()

        return F.l1_loss(output_rms, target_rms)

    def paired_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 loss for paired samples (exact content alignment)."""
        return F.l1_loss(output, target)

    def train_step(self, batch: dict) -> dict:
        input_latent = batch['input'].to(self.device)
        target = batch['target'].to(self.device)
        source_group = batch['source_group'].to(self.device)
        direction = batch['direction'].to(self.device)
        is_paired = batch['is_paired'].to(self.device)
        valid = batch['valid'].to(self.device)

        if not valid.any():
            return {'loss': 0.0}

        # Filter valid samples
        input_latent = input_latent[valid]
        target = target[valid]
        source_group = source_group[valid]
        direction = direction[valid]
        is_paired = is_paired[valid]

        if input_latent.shape[0] == 0:
            return {'loss': 0.0}

        self.model.train()
        self.optimizer.zero_grad()

        amp_context = autocast('cuda') if self.use_amp else nullcontext()

        with amp_context:
            output = self.model(input_latent, source_group, direction)

            # Split by paired/unpaired
            paired_mask = is_paired.bool()
            unpaired_mask = ~paired_mask

            total_loss = torch.tensor(0.0, device=self.device)
            paired_loss_val = 0.0
            dist_loss_val = 0.0

            # Paired loss: L1 for exact content alignment
            if paired_mask.any():
                paired_output = output[paired_mask]
                paired_target = target[paired_mask]
                p_loss = self.paired_loss(paired_output, paired_target)
                total_loss = total_loss + p_loss
                paired_loss_val = p_loss.item()

            # Unpaired loss: distribution matching
            if unpaired_mask.any():
                unpaired_output = output[unpaired_mask]
                unpaired_target = target[unpaired_mask]
                unpaired_input = input_latent[unpaired_mask]

                dist_loss = self.distribution_matching_loss(unpaired_output, unpaired_target)
                silence_loss = self.silence_constraint_loss(unpaired_input, unpaired_output)
                envelope_loss = self.envelope_preservation_loss(unpaired_output, unpaired_target)

                u_loss = 0.5 * dist_loss + 1.0 * silence_loss + 0.2 * envelope_loss
                total_loss = total_loss + u_loss
                dist_loss_val = dist_loss.item()

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
            'paired': paired_loss_val,
            'dist': dist_loss_val,
        }

    def train_epoch(self, epoch: int) -> dict:
        total_loss = 0.0
        total_paired = 0.0
        total_dist = 0.0
        n_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            total_paired += metrics.get('paired', 0)
            total_dist += metrics.get('dist', 0)
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'pair': f"{metrics.get('paired', 0):.4f}",
                'dist': f"{metrics.get('dist', 0):.4f}",
            })

        self.scheduler.step()

        return {
            'loss': total_loss / max(n_batches, 1),
            'paired': total_paired / max(n_batches, 1),
            'dist': total_dist / max(n_batches, 1),
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
        self.log("FORMANT CORRECTOR TRAINING (v9 CONDITIONAL)")
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
                f"Paired: {metrics['paired']:.4f} | "
                f"Dist: {metrics['dist']:.4f} | "
                f"LR: {metrics['lr']:.2e}"
                + (" [BEST]" if is_best else "")
            )

        self.log("")
        self.log("Training complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired_manifest', type=str,
                        default='/mnt/msdd2/pitchshift_v9_formant_pairs/manifest.json')
    parser.add_argument('--sox_manifest', type=str,
                        default='/mnt/msdd2/pitchshift_v7_overlapped/manifest.json')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/msdd2/pitchshift_checkpoints/formant_corrector_v9')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--samples_per_epoch', type=int, default=10000)
    parser.add_argument('--paired_ratio', type=float, default=0.5,
                        help='Ratio of paired vs sox samples (0.5 = 50/50)')
    parser.add_argument('--model_type', type=str, default='simple',
                        choices=['simple', 'film'])
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--direct_output', action='store_true',
                        help='Use direct output instead of residual connection')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Hidden channels in model (64 or 128)')
    parser.add_argument('--residual_scale', type=float, default=0.5,
                        help='Initial residual scale (0.2-1.0)')

    args = parser.parse_args()

    trainer = FormantCorrectorTrainerV9(
        paired_manifest_path=args.paired_manifest,
        sox_manifest_path=args.sox_manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        samples_per_epoch=args.samples_per_epoch,
        paired_ratio=args.paired_ratio,
        model_type=args.model_type,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        direct_output=args.direct_output,
        hidden_channels=args.hidden_channels,
        residual_scale=args.residual_scale,
    )

    trainer.train()


if __name__ == "__main__":
    main()
