#!/usr/bin/env python3
"""
Train Audio Enhancer v2 - with proper spectral and adversarial losses.

Key improvements over v1:
- Multi-resolution STFT loss (magnitude, not phase-dependent)
- Mel spectrogram loss (perceptually weighted)
- Optional multi-scale discriminator (adversarial training)
- Feature matching loss (stabilizes GAN training)

Usage:
    # Without discriminator (faster, still much better than L1):
    python train_enhancer_v2.py \
        --pairs_dir /mnt/msdd2/enhancer_pairs_brass \
        --output_dir /mnt/msdd2/audio_enhancer_checkpoints/brass_v2_stft \
        --epochs 50

    # With discriminator (best quality, slower):
    python train_enhancer_v2.py \
        --pairs_dir /mnt/msdd2/enhancer_pairs_brass \
        --output_dir /mnt/msdd2/audio_enhancer_checkpoints/brass_v2_gan \
        --epochs 50 \
        --use_discriminator
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, '/home/arlo/Data/clarifier')

from audio_enhancer import AudioEnhancer, AudioEnhancerLarge


# =============================================================================
# STFT Loss
# =============================================================================

class STFTLoss(nn.Module):
    """Single-resolution STFT loss."""

    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude. x: [B, T] -> [B, F, T]"""
        # Ensure window is on same device
        window = self.window.to(x.device)

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode='reflect',
        )
        return spec.abs()  # [B, F, T]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        Args:
            pred: [B, C, T] predicted audio
            target: [B, C, T] target audio
        Returns:
            sc_loss: spectral convergence loss
            mag_loss: log magnitude loss
        """
        # Process each channel
        sc_loss = 0
        mag_loss = 0

        for c in range(pred.shape[1]):
            pred_mag = self.stft(pred[:, c])
            target_mag = self.stft(target[:, c])

            # Spectral convergence: Frobenius norm of difference / Frobenius norm of target
            sc_loss += torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)

            # Log magnitude loss
            mag_loss += F.l1_loss(torch.log(pred_mag + 1e-8), torch.log(target_mag + 1e-8))

        sc_loss /= pred.shape[1]
        mag_loss /= pred.shape[1]

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss - the key to phase-invariant audio comparison."""

    def __init__(
        self,
        fft_sizes: list = [512, 1024, 2048],
        hop_sizes: list = [128, 256, 512],
        win_sizes: list = [512, 1024, 2048],
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)

        self.stft_losses = nn.ModuleList([
            STFTLoss(n_fft=n, hop_length=h, win_length=w)
            for n, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        Args:
            pred: [B, C, T] predicted audio
            target: [B, C, T] target audio
        Returns:
            sc_loss: total spectral convergence loss
            mag_loss: total log magnitude loss
        """
        sc_loss = 0
        mag_loss = 0

        for stft_loss in self.stft_losses:
            sc, mag = stft_loss(pred, target)
            sc_loss += sc
            mag_loss += mag

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss


# =============================================================================
# Mel Spectrogram Loss
# =============================================================================

class MelSpectrogramLoss(nn.Module):
    """Mel spectrogram loss - perceptually weighted frequency comparison."""

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 20.0,
        f_max: float = 20000.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Create mel filterbank
        self.register_buffer('window', torch.hann_window(n_fft))

        # Mel filterbank matrix
        mel_fb = self._create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer('mel_fb', mel_fb)

    def _create_mel_filterbank(self, sr, n_fft, n_mels, f_min, f_max):
        """Create mel filterbank matrix."""
        # Mel scale conversion
        def hz_to_mel(f):
            return 2595 * torch.log10(1 + f / 700)

        def mel_to_hz(m):
            return 700 * (10 ** (m / 2595) - 1)

        # Mel points
        mel_min = hz_to_mel(torch.tensor(f_min))
        mel_max = hz_to_mel(torch.tensor(f_max))
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # FFT bins
        n_freqs = n_fft // 2 + 1
        fft_freqs = torch.linspace(0, sr / 2, n_freqs)

        # Create filterbank
        fb = torch.zeros(n_mels, n_freqs)
        for i in range(n_mels):
            low = hz_points[i]
            mid = hz_points[i + 1]
            high = hz_points[i + 2]

            # Rising edge
            rising = (fft_freqs - low) / (mid - low + 1e-8)
            # Falling edge
            falling = (high - fft_freqs) / (high - mid + 1e-8)

            fb[i] = torch.clamp(torch.min(rising, falling), min=0)

        return fb

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, T] predicted audio
            target: [B, C, T] target audio
        Returns:
            mel_loss: L1 loss on log mel spectrograms
        """
        loss = 0

        for c in range(pred.shape[1]):
            # STFT
            window = self.window.to(pred.device)

            pred_spec = torch.stft(
                pred[:, c], self.n_fft, self.hop_length,
                window=window, return_complex=True, pad_mode='reflect'
            ).abs()

            target_spec = torch.stft(
                target[:, c], self.n_fft, self.hop_length,
                window=window, return_complex=True, pad_mode='reflect'
            ).abs()

            # Apply mel filterbank: [B, F, T] @ [M, F].T -> [B, M, T]
            mel_fb = self.mel_fb.to(pred.device)
            pred_mel = torch.einsum('bft,mf->bmt', pred_spec, mel_fb)
            target_mel = torch.einsum('bft,mf->bmt', target_spec, mel_fb)

            # Log mel loss
            loss += F.l1_loss(
                torch.log(pred_mel + 1e-8),
                torch.log(target_mel + 1e-8)
            )

        return loss / pred.shape[1]


# =============================================================================
# Discriminator
# =============================================================================

class DiscriminatorBlock(nn.Module):
    """Single discriminator block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(1, out_channels)  # LayerNorm equivalent
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ScaleDiscriminator(nn.Module):
    """Single-scale discriminator."""

    def __init__(self):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(2, 32, 15, 1, padding=7),
            DiscriminatorBlock(32, 64, 41, 4),
            DiscriminatorBlock(64, 128, 41, 4),
            DiscriminatorBlock(128, 256, 41, 4),
            DiscriminatorBlock(256, 512, 41, 4),
            DiscriminatorBlock(512, 512, 5, 1),
        ])
        self.conv_post = nn.Conv1d(512, 1, 3, 1, padding=1)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [B, 2, T] audio
        Returns:
            output: discriminator output
            features: intermediate features for feature matching
        """
        features = []

        for conv in self.convs:
            x = conv(x) if isinstance(conv, nn.Conv1d) else conv(x)
            if not isinstance(conv, nn.Conv1d):  # Skip first conv for features
                features.append(x)

        output = self.conv_post(x)
        features.append(output)

        return output, features


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator - operates at different audio resolutions."""

    def __init__(self, num_scales: int = 3):
        super().__init__()

        self.discriminators = nn.ModuleList([
            ScaleDiscriminator() for _ in range(num_scales)
        ])

        # Downsampling for multi-scale
        self.pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2) for _ in range(num_scales - 1)
        ])

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [B, 2, T] audio
        Returns:
            outputs: list of discriminator outputs
            features: list of feature lists
        """
        outputs = []
        all_features = []

        for i, disc in enumerate(self.discriminators):
            output, features = disc(x)
            outputs.append(output)
            all_features.append(features)

            if i < len(self.pools):
                x = self.pools[i](x)

        return outputs, all_features


# =============================================================================
# GAN Losses
# =============================================================================

def generator_loss(disc_outputs: list) -> torch.Tensor:
    """Generator loss - wants discriminator to output 1 (real)."""
    loss = 0
    for output in disc_outputs:
        loss += torch.mean((output - 1) ** 2)
    return loss / len(disc_outputs)


def discriminator_loss(real_outputs: list, fake_outputs: list) -> torch.Tensor:
    """Discriminator loss - real=1, fake=0."""
    loss = 0
    for real, fake in zip(real_outputs, fake_outputs):
        loss += torch.mean((real - 1) ** 2) + torch.mean(fake ** 2)
    return loss / len(real_outputs)


def feature_matching_loss(real_features: list, fake_features: list) -> torch.Tensor:
    """Feature matching loss - match intermediate discriminator features."""
    loss = 0
    n_features = 0

    for real_feats, fake_feats in zip(real_features, fake_features):
        for real_f, fake_f in zip(real_feats, fake_feats):
            loss += F.l1_loss(fake_f, real_f.detach())
            n_features += 1

    return loss / n_features


# =============================================================================
# Dataset
# =============================================================================

class PreprocessedEnhancerDataset(Dataset):
    """Dataset that loads preprocessed (original, decoded) pairs."""

    def __init__(self, pairs_dir: str):
        self.pairs_dir = Path(pairs_dir)
        self.pair_files = sorted(self.pairs_dir.glob('pair_*.pt'))
        print(f"[Dataset] Found {len(self.pair_files)} pairs in {pairs_dir}")

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, idx: int) -> dict:
        data = torch.load(self.pair_files[idx], map_location='cpu', weights_only=False)
        return {
            'original': data['original'],
            'decoded': data['decoded'],
            'group_id': data['group_id'],
            'subgroup_id': data['subgroup_id'],
        }


def collate_fn(batch):
    return {
        'original': torch.stack([b['original'] for b in batch]),
        'decoded': torch.stack([b['decoded'] for b in batch]),
        'group_id': torch.tensor([b['group_id'] for b in batch]),
        'subgroup_id': torch.tensor([b['subgroup_id'] for b in batch]),
    }


# =============================================================================
# Trainer
# =============================================================================

class EnhancerTrainerV2:
    """Trainer with spectral and optional adversarial losses."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        output_dir: str = './checkpoints',
        device: str = 'cuda',
        use_discriminator: bool = False,
        lambda_stft: float = 1.0,
        lambda_mel: float = 1.0,
        lambda_adv: float = 0.1,
        lambda_fm: float = 2.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_discriminator = use_discriminator

        # Loss weights
        self.lambda_stft = lambda_stft
        self.lambda_mel = lambda_mel
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm

        # Losses
        self.stft_loss = MultiResolutionSTFTLoss().to(device)
        self.mel_loss = MelSpectrogramLoss().to(device)

        # Generator optimizer
        self.optimizer_g = AdamW(model.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=1e-4)

        # Discriminator (optional)
        if use_discriminator:
            self.discriminator = MultiScaleDiscriminator().to(device)
            self.optimizer_d = AdamW(self.discriminator.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=1e-4)

        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        self.model.train()

        decoded = batch['decoded'].to(self.device)
        original = batch['original'].to(self.device)
        group_id = batch['group_id'].to(self.device)
        subgroup_id = batch['subgroup_id'].to(self.device)

        # Forward through generator
        enhanced = self.model(decoded, group_id, subgroup_id)

        # Spectral losses
        sc_loss, mag_loss = self.stft_loss(enhanced, original)
        mel_loss = self.mel_loss(enhanced, original)

        stft_total = sc_loss + mag_loss

        # Generator loss
        g_loss = self.lambda_stft * stft_total + self.lambda_mel * mel_loss

        losses = {
            'sc': sc_loss.item(),
            'mag': mag_loss.item(),
            'mel': mel_loss.item(),
        }

        # Adversarial training
        if self.use_discriminator:
            self.discriminator.train()

            # Discriminator step
            self.optimizer_d.zero_grad()

            real_outputs, real_features = self.discriminator(original)
            fake_outputs, fake_features = self.discriminator(enhanced.detach())

            d_loss = discriminator_loss(real_outputs, fake_outputs)
            d_loss.backward()
            self.optimizer_d.step()

            losses['d_loss'] = d_loss.item()

            # Generator adversarial loss
            fake_outputs, fake_features = self.discriminator(enhanced)
            _, real_features = self.discriminator(original)

            adv_loss = generator_loss(fake_outputs)
            fm_loss = feature_matching_loss(real_features, fake_features)

            g_loss = g_loss + self.lambda_adv * adv_loss + self.lambda_fm * fm_loss

            losses['adv'] = adv_loss.item()
            losses['fm'] = fm_loss.item()

        # Generator backward
        self.optimizer_g.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer_g.step()

        losses['g_loss'] = g_loss.item()
        self.global_step += 1

        return losses

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation."""
        self.model.eval()

        total_sc = 0
        total_mag = 0
        total_mel = 0
        n_batches = 0

        for batch in self.val_loader:
            decoded = batch['decoded'].to(self.device)
            original = batch['original'].to(self.device)
            group_id = batch['group_id'].to(self.device)
            subgroup_id = batch['subgroup_id'].to(self.device)

            enhanced = self.model(decoded, group_id, subgroup_id)

            sc_loss, mag_loss = self.stft_loss(enhanced, original)
            mel_loss = self.mel_loss(enhanced, original)

            total_sc += sc_loss.item()
            total_mag += mag_loss.item()
            total_mel += mel_loss.item()
            n_batches += 1

        n = max(n_batches, 1)
        return {
            'val_sc': total_sc / n,
            'val_mag': total_mag / n,
            'val_mel': total_mel / n,
            'val_loss': (total_sc + total_mag + total_mel) / n,
        }

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }
        if self.use_discriminator:
            ckpt['discriminator_state_dict'] = self.discriminator.state_dict()
            ckpt['optimizer_d_state_dict'] = self.optimizer_d.state_dict()

        torch.save(ckpt, self.output_dir / filename)

    def train(self, epochs: int, save_every: int = 5):
        """Full training loop."""
        print(f"Starting training for {epochs} epochs")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Using discriminator: {self.use_discriminator}")
        print(f"Output: {self.output_dir}")

        scheduler_g = CosineAnnealingLR(self.optimizer_g, T_max=epochs, eta_min=1e-6)
        if self.use_discriminator:
            scheduler_d = CosineAnnealingLR(self.optimizer_d, T_max=epochs, eta_min=1e-6)

        for epoch in range(epochs):
            epoch_losses = {}
            n_batches = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                losses = self.train_step(batch)

                # Accumulate
                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0) + v
                n_batches += 1

                # Display
                display = {k: f"{v:.4f}" for k, v in losses.items() if k in ['g_loss', 'mel', 'd_loss']}
                pbar.set_postfix(display)

            # Average epoch losses
            for k in epoch_losses:
                epoch_losses[k] /= n_batches

            scheduler_g.step()
            if self.use_discriminator:
                scheduler_d.step()

            # Print epoch summary
            print(f"\nEpoch {epoch+1}:")
            print(f"  SC: {epoch_losses.get('sc', 0):.4f}, Mag: {epoch_losses.get('mag', 0):.4f}, Mel: {epoch_losses.get('mel', 0):.4f}")
            if self.use_discriminator:
                print(f"  Adv: {epoch_losses.get('adv', 0):.4f}, FM: {epoch_losses.get('fm', 0):.4f}, D: {epoch_losses.get('d_loss', 0):.4f}")

            # Validation
            val_metrics = self.validate()
            print(f"  Val - SC: {val_metrics['val_sc']:.4f}, Mag: {val_metrics['val_mag']:.4f}, Mel: {val_metrics['val_mel']:.4f}")

            # Save best
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint('best.pt')
                print(f"  New best! Val loss: {self.best_val_loss:.4f}")

            # Periodic save
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch+1:04d}.pt')

        self.save_checkpoint('final.pt')
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Audio Enhancer v2")
    parser.add_argument('--pairs_dir', type=str, required=True,
                        help='Path to preprocessed pairs directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_size', type=str, default='base', choices=['base', 'large'])
    parser.add_argument('--group_vocab', type=int, default=6)
    parser.add_argument('--subgroup_vocab', type=int, default=21)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    # Loss options
    parser.add_argument('--use_discriminator', action='store_true',
                        help='Enable adversarial training')
    parser.add_argument('--lambda_stft', type=float, default=1.0)
    parser.add_argument('--lambda_mel', type=float, default=1.0)
    parser.add_argument('--lambda_adv', type=float, default=0.1)
    parser.add_argument('--lambda_fm', type=float, default=2.0)

    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Dataset
    print(f"Loading pairs from {args.pairs_dir}")
    full_dataset = PreprocessedEnhancerDataset(args.pairs_dir)

    # Split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    if args.model_size == 'large':
        model = AudioEnhancerLarge(
            group_vocab=args.group_vocab,
            subgroup_vocab=args.subgroup_vocab,
        )
    else:
        model = AudioEnhancer(
            group_vocab=args.group_vocab,
            subgroup_vocab=args.subgroup_vocab,
        )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Resume if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    # Trainer
    trainer = EnhancerTrainerV2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        output_dir=args.output_dir,
        device=args.device,
        use_discriminator=args.use_discriminator,
        lambda_stft=args.lambda_stft,
        lambda_mel=args.lambda_mel,
        lambda_adv=args.lambda_adv,
        lambda_fm=args.lambda_fm,
    )

    trainer.train(epochs=args.epochs, save_every=args.save_every)


if __name__ == '__main__':
    main()
