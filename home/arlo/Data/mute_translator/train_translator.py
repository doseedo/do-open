#!/usr/bin/env python3
"""
Train Mute Translator

Step 1 of the mute conversion pipeline.
Trains a latent space translator to map dry trumpet → muted trumpet.

Usage:
    python train_translator.py --manifest /path/to/manifest.json --output_dir ./checkpoints

The translator uses distribution matching:
- Input: dry trumpet latent
- Output: latent that matches muted trumpet distribution
- Loss: MSE to muted distribution + perceptual losses
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from tqdm import tqdm

# Enable cuDNN autotuning for faster convolutions
torch.backends.cudnn.benchmark = True

sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/mute_translator')

from models import MuteTranslator, MuteTranslatorLarge, MuteTranslatorWithEnvelope, MuteTranslatorDirect, MuteTranslatorAdaptive, MuteDiscriminator
from dataset import MuteTranslatorDataset, load_manifest


class MuteTranslatorTrainer:
    """
    Trainer for the mute translator.

    Training strategy:
    1. Distribution matching: Make translated latents match muted distribution
    2. Content preservation: Translated latent should preserve structure of input
    3. Optional adversarial: Discriminator to improve realism
    """

    def __init__(
        self,
        manifest_path: str,
        output_dir: str,
        model_type: str = "envelope",  # "small", "large", "envelope", or "direct"
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        window_frames: int = 128,
        samples_per_epoch: int = 5000,
        use_adversarial: bool = False,
        use_amp: bool = True,
        device: str = "cuda",
        num_workers: int = 4,
        dist_weight: float = 0.5,
        early_stop_threshold: float = 1.0,
        init_checkpoint: str = None,  # Path to pretrained checkpoint for fine-tuning
    ):
        self.manifest_path = manifest_path
        self.use_amp = use_amp
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch
        self.use_adversarial = use_adversarial
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        self.model_type = model_type
        self.dist_weight = dist_weight
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_saved = False

        print(f"Device: {self.device}")

        # Create model
        if model_type == "large":
            self.model = MuteTranslatorLarge().to(self.device)
        elif model_type == "envelope":
            self.model = MuteTranslatorWithEnvelope(use_envelope=True).to(self.device)
        elif model_type == "direct":
            self.model = MuteTranslatorDirect().to(self.device)
        elif model_type == "adaptive":
            self.model = MuteTranslatorAdaptive(alpha_init=0.3).to(self.device)
        else:
            self.model = MuteTranslator().to(self.device)

        print(f"Model type: {model_type}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Load pretrained weights if init_checkpoint provided
        if init_checkpoint:
            print(f"Loading pretrained weights from: {init_checkpoint}")
            ckpt = torch.load(init_checkpoint, map_location=self.device, weights_only=True)
            # Handle missing keys for new parameters (e.g., dry_scale added later)
            state_dict = ckpt['model_state_dict']
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys (using defaults): {missing}")
            epoch = ckpt.get('epoch', '?')
            loss = ckpt.get('loss', None)
            loss_str = f"{loss:.4f}" if loss is not None else "?"
            print(f"  Loaded from epoch {epoch}, loss {loss_str}")

        # Optional discriminator
        self.discriminator = None
        if use_adversarial:
            self.discriminator = MuteDiscriminator().to(self.device)
            print(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")

        # Create dataset with attack-focused sampling
        self.dataset = MuteTranslatorDataset(
            manifest_path=manifest_path,
            window_frames=window_frames,
            samples_per_epoch=samples_per_epoch,
            attack_focus_ratio=0.5,  # 50% of samples focus on attack regions
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Compute muted distribution statistics for matching
        self._compute_muted_stats()

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        # AMP scaler for mixed precision training
        self.scaler = GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print("Using AMP (mixed precision training)")

        if self.discriminator:
            self.optimizer_d = AdamW(self.discriminator.parameters(), lr=learning_rate * 0.5)

        # Logging
        self.log_file = self.output_dir / "training.log"
        self.best_loss = float('inf')

    def _compute_muted_stats(self):
        """Compute mean and std of muted latent distribution."""
        print("Computing muted distribution statistics...")

        all_latents = []
        for entry in self.dataset.muted_entries[:100]:  # Sample subset for speed
            latent = self.dataset._load_latent(entry)
            if latent is not None:
                if latent.dim() == 4:
                    latent = latent.squeeze(0)
                all_latents.append(latent)

        if len(all_latents) > 0:
            # Concatenate along time dimension
            concat = torch.cat([l.reshape(-1) for l in all_latents])
            self.muted_mean = concat.mean().item()
            self.muted_std = concat.std().item()

            # Per-channel stats
            stacked = torch.stack([l.mean(dim=(1, 2)) for l in all_latents])  # [N, C]
            self.muted_channel_mean = stacked.mean(dim=0).to(self.device)  # [C]
            self.muted_channel_std = stacked.std(dim=0).to(self.device)  # [C]

            print(f"  Muted mean: {self.muted_mean:.4f}, std: {self.muted_std:.4f}")
        else:
            self.muted_mean = 0.0
            self.muted_std = 1.0
            self.muted_channel_mean = torch.zeros(8, device=self.device)
            self.muted_channel_std = torch.ones(8, device=self.device)

    def distribution_loss(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
        dry_latent: torch.Tensor = None,
        onsets: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        HF-emphasis distribution loss for Harmon mute translation.

        SIMPLER VERSION that actually trains (no region masking that causes gradient issues).
        Uses global frequency weighting:
        - HF (upper 50%): 3x weight for brightness
        - Mid-high (25-50%): 2x weight for nasal resonance
        - Body (lower 25%): 1x weight
        """
        B, C, H, T = pred_latent.shape

        # Frequency band indices
        mh_start = H // 4       # 25% of H
        mh_end = H // 2         # 50% of H: mid-high (nasal resonance)
        hf_start = H // 2       # 50-100%: high frequencies (brightness)

        # Create frequency weighting - GLOBAL, no masking
        freq_weights = torch.ones(H, device=pred_latent.device)
        freq_weights[mh_start:mh_end] = 2.0  # Mid-high nasal: 2x
        freq_weights[hf_start:] = 3.0        # HF brightness: 3x
        freq_weights = freq_weights.view(1, 1, H, 1)

        # Weighted MSE loss
        weighted_diff = ((pred_latent - target_latent) ** 2) * freq_weights
        hf_mse_loss = weighted_diff.mean()

        # Per-channel mean/std matching
        pred_mean = pred_latent.mean(dim=(2, 3))
        pred_std = pred_latent.std(dim=(2, 3))
        target_mean = target_latent.mean(dim=(2, 3))
        target_std = target_latent.std(dim=(2, 3))

        mean_loss = F.mse_loss(pred_mean, target_mean)
        std_loss = F.mse_loss(pred_std, target_std)

        # Mid-high resonance energy (nasal character)
        pred_mh_energy = pred_latent[:, :, mh_start:mh_end, :].abs().mean(dim=(2, 3))
        target_mh_energy = target_latent[:, :, mh_start:mh_end, :].abs().mean(dim=(2, 3))
        mh_energy_loss = F.mse_loss(pred_mh_energy, target_mh_energy)

        # High-frequency energy (brightness)
        pred_hf_energy = pred_latent[:, :, hf_start:, :].abs().mean(dim=(2, 3))
        target_hf_energy = target_latent[:, :, hf_start:, :].abs().mean(dim=(2, 3))
        hf_energy_loss = F.mse_loss(pred_hf_energy, target_hf_energy)

        # Simple MMD (fp32 for stability)
        pred_flat = pred_latent.reshape(B, -1).float()
        target_flat = target_latent.reshape(B, -1).float()
        scale = pred_flat.shape[1] ** 0.5
        pred_flat = pred_flat / scale
        target_flat = target_flat / scale
        pred_gram = torch.mm(pred_flat, pred_flat.t())
        target_gram = torch.mm(target_flat, target_flat.t())
        cross_gram = torch.mm(pred_flat, target_flat.t())
        mmd = pred_gram.mean() + target_gram.mean() - 2 * cross_gram.mean()

        # Combined loss
        return (
            hf_mse_loss
            + mean_loss
            + std_loss
            + 0.3 * mh_energy_loss   # Mid-high nasal resonance
            + 0.5 * hf_energy_loss   # High frequency brightness
            + 0.1 * mmd
        )

    def content_preservation_loss(
        self,
        dry_latent: torch.Tensor,
        pred_latent: torch.Tensor,
        onsets: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Ensure translated latent preserves structure of input, EXCEPT at attacks.

        The translation should modify timbre, not melody/rhythm.
        We compare temporal structure (gradients) between input and output.

        KEY CHANGE: Reduce weight around onset frames to allow attack reshaping.
        """
        B, C, H, T = dry_latent.shape

        # Temporal gradient (rhythm preservation)
        dry_grad = dry_latent[:, :, :, 1:] - dry_latent[:, :, :, :-1]
        pred_grad = pred_latent[:, :, :, 1:] - pred_latent[:, :, :, :-1]

        # Compute per-frame gradient loss
        grad_diff = (pred_grad.abs() - dry_grad.abs()) ** 2  # [B, C, H, T-1]

        # Create onset-aware weighting mask
        if onsets is not None and onsets.sum() > 0:
            # onsets: [B, T]
            # Create mask that reduces weight around onsets
            # Weight = 0.1 for frames within attack_window of an onset
            attack_window = 20  # frames around onset to allow reshaping

            onset_mask = torch.ones(B, T-1, device=dry_latent.device)

            for b in range(B):
                onset_frames = (onsets[b] > 0.5).nonzero(as_tuple=True)[0]
                for onset_frame in onset_frames:
                    # Reduce weight in window around onset
                    start = max(0, onset_frame.item() - 5)  # Small pre-onset window
                    end = min(T-1, onset_frame.item() + attack_window)
                    onset_mask[b, start:end] = 0.1  # Low weight = allow changes

            # Expand mask for broadcasting: [B, 1, 1, T-1]
            onset_mask = onset_mask.view(B, 1, 1, T-1)
            grad_diff = grad_diff * onset_mask

        grad_loss = grad_diff.mean()

        # Spectral gradient (pitch contour preservation) - always preserve this
        dry_spec_grad = dry_latent[:, :, 1:, :] - dry_latent[:, :, :-1, :]
        pred_spec_grad = pred_latent[:, :, 1:, :] - pred_latent[:, :, :-1, :]
        spec_loss = F.mse_loss(pred_spec_grad.abs(), dry_spec_grad.abs())

        return grad_loss + spec_loss

    def time_weighted_loss(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
        has_attack: torch.Tensor
    ) -> torch.Tensor:
        """
        MSE loss with time weighting to emphasize early frames (attacks).

        The first ~20% of frames get 3x weight, next ~30% get 1.5x weight.
        This forces the model to learn attack characteristics better.
        """
        B, C, H, T = pred_latent.shape

        # Create time weights: higher at the beginning
        # First 20% of frames: weight 3.0
        # Next 30% of frames: weight 1.5
        # Rest: weight 1.0
        weights = torch.ones(T, device=pred_latent.device)

        attack_end = int(T * 0.2)  # First 20% = attack region
        decay_end = int(T * 0.5)   # Next 30% = decay region

        weights[:attack_end] = 3.0
        weights[attack_end:decay_end] = 1.5

        # Normalize so mean weight is ~1
        weights = weights / weights.mean()

        # Expand weights for broadcasting: [1, 1, 1, T]
        weights = weights.view(1, 1, 1, T)

        # Weighted MSE
        diff_sq = (pred_latent - target_latent) ** 2
        weighted_diff = diff_sq * weights

        # For samples with detected attacks, apply even higher weight
        # has_attack: [B] tensor
        if has_attack.any():
            attack_mask = has_attack.float().view(-1, 1, 1, 1)
            # Boost early frames even more for attack-detected samples
            early_boost = torch.ones_like(weights)
            early_boost[:, :, :, :attack_end] = 1.5  # Additional 1.5x on top
            weighted_diff = weighted_diff * (1 + 0.5 * attack_mask * early_boost)

        return weighted_diff.mean()

    def silence_constraint_loss(
        self,
        dry_latent: torch.Tensor,
        pred_latent: torch.Tensor,
        silence_thresh: float = 0.05,
    ) -> torch.Tensor:
        """
        Penalize output energy when input is silent.

        This prevents buzz/noise in gaps between notes.
        When dry input has low energy, output should also have low energy.

        Args:
            dry_latent: Input dry latent [B, C, H, T]
            pred_latent: Predicted muted latent [B, C, H, T]
            silence_thresh: Energy threshold below which input is considered silent

        Returns:
            Scalar loss penalizing output energy in silent regions
        """
        # Compute per-frame energy (RMS-like across channels and height)
        # dry_latent: [B, C, H, T] -> [B, T]
        dry_energy = dry_latent.pow(2).mean(dim=(1, 2)).sqrt()   # [B, T]
        pred_energy = pred_latent.pow(2).mean(dim=(1, 2)).sqrt() # [B, T]

        # Identify silent frames in the input (energy below threshold)
        silence_mask = (dry_energy < silence_thresh).float()  # [B, T]

        # Penalize any output energy in those silent frames
        # pred_energy.pow(2) penalizes harder for louder output in silence
        silence_loss = (silence_mask * pred_energy.pow(2)).sum() / (silence_mask.sum() + 1e-6)

        return silence_loss

    def envelope_preservation_loss(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
        window_size: int = 8,
    ) -> torch.Tensor:
        """
        Match the RMS envelope between predicted and target latents.

        This prevents compression artifacts (pumping) and preserves:
        - Sustain levels
        - Release shape
        - Relative envelope balance

        Args:
            pred_latent: Predicted muted latent [B, C, H, T]
            target_latent: Target muted latent [B, C, H, T]
            window_size: Size of the RMS window (in frames)

        Returns:
            L1 loss between RMS envelopes
        """
        B, C, H, T = pred_latent.shape

        # Compute per-frame energy: [B, C, H, T] -> [B, T]
        pred_energy = pred_latent.pow(2).mean(dim=(1, 2))   # [B, T]
        target_energy = target_latent.pow(2).mean(dim=(1, 2)) # [B, T]

        # Compute RMS envelope using 1D avg pooling over time
        # Reshape for pooling: [B, 1, T]
        pred_energy = pred_energy.unsqueeze(1)
        target_energy = target_energy.unsqueeze(1)

        # Use same padding to keep T dimension
        padding = window_size // 2
        pred_rms = F.avg_pool1d(pred_energy, window_size, stride=1, padding=padding)
        target_rms = F.avg_pool1d(target_energy, window_size, stride=1, padding=padding)

        # Trim to original size if needed
        pred_rms = pred_rms[:, :, :T]
        target_rms = target_rms[:, :, :T]

        # Take sqrt for RMS
        pred_rms = pred_rms.sqrt()
        target_rms = target_rms.sqrt()

        # L1 loss between envelopes
        env_loss = F.l1_loss(pred_rms, target_rms)

        return env_loss

    def adversarial_loss(
        self,
        pred_latent: torch.Tensor,
        real_muted: torch.Tensor,
        train_discriminator: bool = True
    ) -> tuple:
        """Adversarial loss for more realistic translations."""
        if self.discriminator is None:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        if train_discriminator:
            # Train discriminator
            self.discriminator.train()
            real_logits = self.discriminator(real_muted.detach())
            fake_logits = self.discriminator(pred_latent.detach())

            d_loss = F.binary_cross_entropy_with_logits(
                real_logits, torch.ones_like(real_logits)
            ) + F.binary_cross_entropy_with_logits(
                fake_logits, torch.zeros_like(fake_logits)
            )
        else:
            d_loss = torch.tensor(0.0, device=self.device)

        # Generator loss (fool discriminator)
        fake_logits = self.discriminator(pred_latent)
        g_loss = F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)
        )

        return g_loss, d_loss

    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        dry_latent = batch['dry_latent'].to(self.device)
        muted_latent = batch['muted_latent'].to(self.device)
        valid = batch['valid'].to(self.device)
        has_attack = batch.get('has_attack', torch.zeros_like(valid)).to(self.device)

        # Get conditioning data
        dry_onsets = batch.get('dry_onsets')
        dry_amp = batch.get('dry_amp')
        if dry_onsets is not None:
            dry_onsets = dry_onsets.to(self.device)
        if dry_amp is not None:
            dry_amp = dry_amp.to(self.device)

        # Skip invalid samples
        if not valid.any():
            return {'loss': 0.0}

        dry_latent = dry_latent[valid]
        muted_latent = muted_latent[valid]
        has_attack = has_attack[valid]
        if dry_onsets is not None:
            dry_onsets = dry_onsets[valid]
        if dry_amp is not None:
            dry_amp = dry_amp[valid]

        # Forward pass (with optional AMP)
        self.model.train()
        self.optimizer.zero_grad()

        # Context manager for optional AMP
        amp_context = autocast('cuda') if self.use_amp else nullcontext()

        with amp_context:
            # Pass conditioning if model supports it (envelope model)
            if self.model_type == "envelope":
                pred_muted = self.model(dry_latent, dry_onsets, dry_amp)
            else:
                pred_muted = self.model(dry_latent)

            # Losses - attack-aware distribution loss
            dist_loss = self.distribution_loss(pred_muted, muted_latent, dry_latent, dry_onsets)

            # Onset-aware content preservation loss
            content_loss = self.content_preservation_loss(dry_latent, pred_muted, dry_onsets)

            # Time-weighted loss for attack emphasis
            time_loss = self.time_weighted_loss(pred_muted, muted_latent, has_attack)

            # Silence constraint: penalize output energy when input is silent
            # This prevents buzz/noise in gaps between notes
            silence_loss = self.silence_constraint_loss(dry_latent, pred_muted)

            # Envelope preservation: match RMS envelope to prevent compression artifacts
            envelope_loss = self.envelope_preservation_loss(pred_muted, muted_latent)

            # Combined loss: distribution + silence + envelope
            # For RESIDUAL models (small/large/envelope): no content loss (causes identity collapse)
            # For DIRECT models: ADD content loss (no residual shortcut, needs explicit preservation)
            total_loss = (
                self.dist_weight * dist_loss
                + 1.0 * silence_loss      # Strong penalty for buzz in silence
                + 0.2 * envelope_loss     # Match envelope/dynamics
            )

            # Direct model needs content preservation (no dry pass-through)
            if self.model_type == 'direct':
                total_loss = total_loss + 0.5 * content_loss + 0.3 * time_loss

            # Adversarial (if enabled)
            if self.use_adversarial:
                g_loss, d_loss = self.adversarial_loss(pred_muted, muted_latent, train_discriminator=True)
                total_loss = total_loss + 0.1 * g_loss

        # Backward pass
        if self.use_amp and self.scaler is not None:
            # Adversarial discriminator update (outside autocast)
            if self.use_adversarial and d_loss.item() > 0:
                self.optimizer_d.zero_grad()
                self.scaler.scale(d_loss).backward(retain_graph=True)
                self.scaler.step(self.optimizer_d)

            # Update generator with scaled gradients
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard FP32 backward pass
            if self.use_adversarial and d_loss.item() > 0:
                self.optimizer_d.zero_grad()
                d_loss.backward(retain_graph=True)
                self.optimizer_d.step()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'dist_loss': dist_loss.item(),
            'content_loss': content_loss.item(),
            'time_loss': time_loss.item(),
            'silence_loss': silence_loss.item(),
            'envelope_loss': envelope_loss.item(),
        }

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        total_loss = 0.0
        total_dist = 0.0
        total_content = 0.0
        total_time = 0.0
        total_silence = 0.0
        total_envelope = 0.0
        n_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            total_dist += metrics.get('dist_loss', 0)
            total_content += metrics.get('content_loss', 0)
            total_time += metrics.get('time_loss', 0)
            total_silence += metrics.get('silence_loss', 0)
            total_envelope += metrics.get('envelope_loss', 0)
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'dist': f"{metrics.get('dist_loss', 0):.4f}",
                'sil': f"{metrics.get('silence_loss', 0):.4f}",
                'env': f"{metrics.get('envelope_loss', 0):.4f}",
            })

        self.scheduler.step()

        return {
            'loss': total_loss / max(n_batches, 1),
            'dist_loss': total_dist / max(n_batches, 1),
            'content_loss': total_content / max(n_batches, 1),
            'time_loss': total_time / max(n_batches, 1),
            'silence_loss': total_silence / max(n_batches, 1),
            'envelope_loss': total_envelope / max(n_batches, 1),
            'lr': self.scheduler.get_last_lr()[0],
        }

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_type': self.model_type,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'muted_stats': {
                'mean': self.muted_mean,
                'std': self.muted_std,
            },
        }

        if self.discriminator:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()

        # Save latest
        torch.save(checkpoint, self.output_dir / "latest.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / "best.pt")

        # Save periodic
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, self.output_dir / f"epoch_{epoch+1}.pt")

    def log(self, message: str):
        """Log message to file and stdout."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def train(self):
        """Full training loop."""
        self.log("=" * 60)
        self.log("MUTE TRANSLATOR TRAINING")
        self.log("=" * 60)
        self.log(f"Output dir: {self.output_dir}")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Learning rate: {self.learning_rate}")
        self.log(f"Epochs: {self.num_epochs}")
        self.log(f"Dry samples: {len(self.dataset.dry_entries)}")
        self.log(f"Muted samples: {len(self.dataset.muted_entries)}")
        self.log("")

        for epoch in range(self.num_epochs):
            metrics = self.train_epoch(epoch)

            is_best = metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = metrics['loss']

            self.save_checkpoint(epoch, metrics, is_best)

            # Early stop checkpoint: save when loss first drops below threshold
            # This captures the "underfitting sweet spot" before over-regularization
            early_stop_tag = ""
            if not self.early_stop_saved and metrics['loss'] <= self.early_stop_threshold:
                early_stop_ckpt = {
                    'epoch': epoch,
                    'model_type': self.model_type,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'metrics': metrics,
                    'muted_stats': {
                        'mean': self.muted_mean,
                        'std': self.muted_std,
                    },
                }
                torch.save(early_stop_ckpt, self.output_dir / "early_stop.pt")
                self.early_stop_saved = True
                early_stop_tag = " [EARLY_STOP SAVED]"
                self.log(f"  -> Saved early_stop.pt at loss {metrics['loss']:.4f}")

            self.log(
                f"Epoch {epoch+1:3d} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Dist: {metrics['dist_loss']:.4f} | "
                f"Sil: {metrics.get('silence_loss', 0):.4f} | "
                f"Env: {metrics.get('envelope_loss', 0):.4f} | "
                f"LR: {metrics['lr']:.2e}"
                + (" [BEST]" if is_best else "")
                + early_stop_tag
            )

        self.log("")
        self.log("Training complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")
        self.log(f"Checkpoints saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Mute Translator")
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json',
                        help='Path to manifest JSON')
    parser.add_argument('--output_dir', type=str,
                        default='/home/arlo/Data/mute_translator/checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--model_type', type=str, default='envelope',
                        choices=['small', 'large', 'envelope', 'direct', 'adaptive'],
                        help='Model type: small, large, envelope, direct, or adaptive (learnable mixing)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128, can go higher on A100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--window_frames', type=int, default=128)
    parser.add_argument('--samples_per_epoch', type=int, default=10000,
                        help='Samples per epoch (default: 10000)')
    parser.add_argument('--use_adversarial', action='store_true')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision (AMP)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader workers (default: 8)')
    parser.add_argument('--dist_weight', type=float, default=0.5,
                        help='Weight for distribution loss (default: 0.5). Lower = more content preservation = more Harmon-like')
    parser.add_argument('--early_stop_threshold', type=float, default=1.0,
                        help='Save early_stop.pt when loss first drops below this threshold (default: 1.0)')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='Path to pretrained checkpoint to initialize from (for fine-tuning)')

    args = parser.parse_args()

    trainer = MuteTranslatorTrainer(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        window_frames=args.window_frames,
        samples_per_epoch=args.samples_per_epoch,
        use_adversarial=args.use_adversarial,
        use_amp=not args.no_amp,
        device=args.device,
        num_workers=args.num_workers,
        dist_weight=args.dist_weight,
        early_stop_threshold=args.early_stop_threshold,
        init_checkpoint=args.init_checkpoint,
    )

    trainer.train()


if __name__ == "__main__":
    main()
