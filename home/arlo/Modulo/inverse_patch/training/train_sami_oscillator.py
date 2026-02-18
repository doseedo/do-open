#!/usr/bin/env python3
"""
Train z_dcae → Sines using Oscillator-based loss (FAST!)

Instead of slow audio synthesis:
  z_dcae → mapper → (freqs, amps) → SineSynth → audio → STFT loss

Use fast latent prediction:
  z_dcae → mapper → (freqs, amps) → SineOscillator → pred_z → MSE(pred_z, z_dcae)

The SineOscillator is pre-trained to predict what z_dcae produces for given sine params.
This is ~10-100x faster than audio synthesis because:
1. No audio synthesis (88200 samples)
2. No STFT computation
3. Just a small neural network forward pass

The oscillator learned how sines combine non-linearly in DCAE's latent space,
which the analytical approach couldn't capture.
"""

import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import torchaudio
import orjson
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True


# ============================================================
# SINE OSCILLATOR (Pre-trained: params → z_dcae)
# ============================================================

class SineOscillator(nn.Module):
    """
    Pre-trained network: (freqs, amps) → z_dcae

    This learns how sines combine in DCAE's latent space.
    Used as a differentiable loss function during SAMI training.
    """

    def __init__(
        self,
        n_sines: int = 4,
        n_channels: int = 8,
        latent_dim: int = 16,
        n_frames: int = 22,
        freq_min: float = 100.0,
        freq_max: float = 2000.0,
    ):
        super().__init__()
        self.n_sines = n_sines
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        self.total_size = n_channels * latent_dim * n_frames

        # Frequency normalization params
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.log_freq_min = np.log10(freq_min)
        self.log_freq_max = np.log10(freq_max)

        # Network: n_sines * 2 (freq + amp per sine) -> full latent
        self.net = nn.Sequential(
            nn.Linear(n_sines * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, self.total_size),
        )

    def normalize_freq(self, freq_hz: torch.Tensor) -> torch.Tensor:
        """Normalize frequency to [0, 1] using log scale."""
        freq_clamped = freq_hz.clamp(min=self.freq_min, max=self.freq_max)
        log_f = torch.log10(freq_clamped)
        return (log_f - self.log_freq_min) / (self.log_freq_max - self.log_freq_min)

    def forward(self, freqs: torch.Tensor, amps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            freqs: [B, n_sines] in Hz
            amps: [B, n_sines] in [0, 1]
        Returns:
            z: [B, n_channels, latent_dim, n_frames]
        """
        # Normalize frequencies
        freqs_norm = self.normalize_freq(freqs)
        # Zero out freqs where amp is 0 (inactive sines)
        freqs_norm = freqs_norm * (amps > 0.01).float()

        # Concatenate [freq1, amp1, freq2, amp2, ...]
        params = torch.stack([freqs_norm, amps], dim=-1).flatten(1)  # [B, n_sines * 2]

        z_flat = self.net(params)
        z = z_flat.view(-1, self.n_channels, self.latent_dim, self.n_frames)
        return z


# ============================================================
# PER-FRAME SINE MAPPER
# ============================================================

class ResBlock(nn.Module):
    """Residual block for deeper network."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class SparseSineMapper(nn.Module):
    """
    Maps z_dcae frames to sine parameters.

    [B, T, 128] → [B, T, n_sines] freqs + [B, T, n_sines] amps

    For oscillator loss, we need FIXED n_sines (matching the oscillator).
    """

    def __init__(
        self,
        frame_dim: int = 128,
        n_sines: int = 4,  # Fixed to match oscillator
        hidden_dim: int = 512,
        n_blocks: int = 4,
        freq_min: float = 100.0,
        freq_max: float = 2000.0,
    ):
        super().__init__()
        self.frame_dim = frame_dim
        self.n_sines = n_sines
        self.freq_min = freq_min
        self.freq_max = freq_max

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.GELU(),
            *[ResBlock(hidden_dim) for _ in range(n_blocks)],
        )

        # Separate heads
        self.freq_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_sines),
        )

        self.amp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_sines),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: [B, T, 128] per-frame latent
        Returns:
            freqs: [B, T, n_sines] in Hz
            amps: [B, T, n_sines] in [0, 1]
        """
        h = self.encoder(z)

        # Frequencies in range [freq_min, freq_max]
        freqs = self.freq_min + torch.sigmoid(self.freq_head(h)) * (self.freq_max - self.freq_min)

        # Amplitudes in [0, 1] - sparsity loss will push unused to 0
        amps = torch.sigmoid(self.amp_head(h))

        return {'freqs': freqs, 'amps': amps}


# ============================================================
# OSCILLATOR-BASED LOSS
# ============================================================

class OscillatorLatentLoss(nn.Module):
    """
    Fast loss using pre-trained oscillator.

    Instead of: synth audio → STFT → spectral loss
    We do: oscillator(freqs, amps) → MSE(pred_z, target_z)

    The oscillator learned how sines combine in DCAE's latent space.
    """

    def __init__(self, oscillator: SineOscillator, n_frames: int = 22):
        super().__init__()
        self.oscillator = oscillator
        self.n_frames = n_frames

        # Freeze oscillator - it's pre-trained
        for param in self.oscillator.parameters():
            param.requires_grad = False

    def forward(
        self,
        freqs: torch.Tensor,
        amps: torch.Tensor,
        z_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            freqs: [B, T, n_sines] predicted frequencies
            amps: [B, T, n_sines] predicted amplitudes
            z_target: [B, 8, 16, T] target DCAE latent

        Returns:
            loss: scalar
        """
        B, T, N = freqs.shape

        # Average over time to get single freq/amp per sine
        # (The oscillator predicts full temporal z from single params)
        freqs_avg = freqs.mean(dim=1)  # [B, n_sines]
        amps_avg = amps.mean(dim=1)    # [B, n_sines]

        # Predict what z should be for these sine params
        z_pred = self.oscillator(freqs_avg, amps_avg)  # [B, 8, 16, 22]

        # Match temporal dimension if needed
        if z_pred.shape[-1] != z_target.shape[-1]:
            z_pred = F.interpolate(
                z_pred.flatten(1, 2),  # [B, 128, T]
                size=z_target.shape[-1],
                mode='linear'
            ).view(B, 8, 16, -1)

        # MSE loss
        mse_loss = F.mse_loss(z_pred, z_target)

        # Cosine similarity loss (important for direction)
        cos_loss = 1 - F.cosine_similarity(
            z_pred.flatten(1), z_target.flatten(1), dim=1
        ).mean()

        return mse_loss + 0.5 * cos_loss


# ============================================================
# DATASET
# ============================================================

class UnifiedLatentDataset(Dataset):
    """Dataset that preloads everything into RAM."""

    def __init__(
        self,
        manifest_path: str,
        max_samples: Optional[int] = None,
        filter_groups: Optional[List[str]] = None,
        sample_rate: int = 44100,
        target_frames: int = 22,
    ):
        self.sample_rate = sample_rate
        self.target_frames = target_frames
        self.target_samples = int(sample_rate * 2.0)  # ~2 seconds

        print(f"Loading manifest from {manifest_path}...")
        sys.stdout.flush()
        with open(manifest_path, 'rb') as f:
            raw = f.read()
        print(f"  Read {len(raw) / 1e6:.1f}MB, parsing...")
        sys.stdout.flush()
        data = orjson.loads(raw)
        del raw

        entries = data.get('entries', data)
        if isinstance(entries, dict):
            entries = list(entries.values())
        print(f"  Found {len(entries)} total entries, filtering...")
        sys.stdout.flush()

        items = []
        for entry in entries:
            if not entry.get('has_latent', False):
                continue
            if entry.get('latent_path') is None:
                continue
            if entry.get('audio_path') is None:
                continue
            if filter_groups and entry.get('group') not in filter_groups:
                continue
            items.append(entry)

        if max_samples:
            items = items[:max_samples]

        # PRELOAD everything
        print(f"  Preloading {len(items)} samples into RAM (parallel)...")
        sys.stdout.flush()

        samples_per_frame = self.target_samples // self.target_frames

        def load_one(item):
            try:
                lat_data = torch.load(item['latent_path'], weights_only=True, map_location='cpu')
                if 'latents' in lat_data:
                    latent = lat_data['latents']
                elif 'latent' in lat_data:
                    latent = lat_data['latent']
                else:
                    return None

                audio, sr = torchaudio.load(item['audio_path'])
                if sr != self.sample_rate:
                    audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0)
                else:
                    audio = audio.squeeze(0)

                C, H, T_lat = latent.shape

                if T_lat < self.target_frames:
                    latent = F.pad(latent, (0, self.target_frames - T_lat))
                    if audio.shape[-1] < self.target_samples:
                        audio = F.pad(audio, (0, self.target_samples - audio.shape[-1]))
                    else:
                        audio = audio[:self.target_samples]
                elif T_lat > self.target_frames:
                    start_frame = (T_lat - self.target_frames) // 2
                    latent = latent[:, :, start_frame:start_frame + self.target_frames]
                    start_sample = start_frame * samples_per_frame
                    end_sample = start_sample + self.target_samples
                    if end_sample <= audio.shape[-1]:
                        audio = audio[start_sample:end_sample]
                    else:
                        audio = audio[-self.target_samples:] if audio.shape[-1] >= self.target_samples else F.pad(audio, (0, self.target_samples - audio.shape[-1]))
                else:
                    if audio.shape[-1] < self.target_samples:
                        audio = F.pad(audio, (0, self.target_samples - audio.shape[-1]))
                    else:
                        audio = audio[:self.target_samples]

                audio = audio / (audio.abs().max() + 1e-8)

                return {
                    'latent': latent,
                    'audio': audio,
                    'group': item.get('group', 'unknown'),
                }
            except:
                return None

        self.data = []
        loaded = 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(load_one, item): i for i, item in enumerate(items)}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    self.data.append(result)
                loaded += 1
                if loaded % 100 == 0:
                    print(f"\r    Loaded {loaded}/{len(items)}...", end="", flush=True)

        print(f"\r    Loaded {len(self.data)} samples into RAM  ")
        sys.stdout.flush()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


def collate_fn(batch):
    return {
        'latent': torch.stack([b['latent'] for b in batch]),
        'audio': torch.stack([b['audio'] for b in batch]),
        'group': [b['group'] for b in batch],
    }


# ============================================================
# TRAINING
# ============================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class OscillatorTrainer:
    """Train SAMI mapper using oscillator-based loss."""

    def __init__(
        self,
        oscillator_path: str,
        n_sines: int = 4,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        sparsity_weight: float = 0.01,
        freq_min: float = 100.0,
        freq_max: float = 2000.0,
        device: str = 'cuda',
    ):
        self.device = device
        self.sparsity_weight = sparsity_weight
        self.n_sines = n_sines

        # Load pre-trained oscillator
        print(f"Loading oscillator from {oscillator_path}...")
        osc_data = torch.load(oscillator_path, map_location=device)

        self.oscillator = SineOscillator(
            n_sines=n_sines,
            n_channels=osc_data.get('n_channels', 8),
            latent_dim=osc_data.get('latent_dim', 16),
            n_frames=osc_data.get('n_frames', 22),
            freq_min=freq_min,
            freq_max=freq_max,
        ).to(device)
        self.oscillator.load_state_dict(osc_data['model'])
        self.oscillator.eval()
        print(f"  Oscillator loaded! (n_sines={n_sines}, frames={osc_data.get('n_frames', 22)})")

        # Create mapper (this is what we're training)
        self.mapper = SparseSineMapper(
            frame_dim=128,
            n_sines=n_sines,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            freq_min=freq_min,
            freq_max=freq_max,
        ).to(device)

        # Loss function using oscillator
        self.loss_fn = OscillatorLatentLoss(self.oscillator, n_frames=22).to(device)

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda')

        params = sum(p.numel() for p in self.mapper.parameters())
        print(f"\nOscillatorTrainer:")
        print(f"  Mapper: z_dcae [B,T,128] → {n_sines} sines (freq, amp)")
        print(f"  Loss: Oscillator-based latent MSE (FAST!)")
        print(f"  Freq range: {freq_min}-{freq_max} Hz")
        print(f"  Sparsity weight: {sparsity_weight}")
        print(f"  Params: {params:,}")

    def train_step(self, batch, optimizer):
        optimizer.zero_grad()

        latent = batch['latent'].to(self.device)  # [B, 8, 16, T]

        with torch.amp.autocast('cuda'):
            B, C, H, T = latent.shape
            z = latent.permute(0, 3, 1, 2).reshape(B, T, C * H)  # [B, T, 128]

            # Map to sine params
            params = self.mapper(z)
            freqs = params['freqs']  # [B, T, n_sines]
            amps = params['amps']

            # Oscillator-based loss (FAST!)
            recon_loss = self.loss_fn(freqs, amps, latent)

            # Sparsity loss on amplitudes
            sparsity_loss = amps.mean()

            loss = recon_loss + self.sparsity_weight * sparsity_loss

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.mapper.parameters(), 1.0)
        self.scaler.step(optimizer)
        self.scaler.update()

        n_active = (amps > 0.1).float().sum(dim=-1).mean().item()

        return {
            'loss': loss.item(),
            'recon': recon_loss.item(),
            'sparsity': sparsity_loss.item(),
            'n_active': n_active,
        }

    def train(self, dataloader, n_epochs: int = 100, lr: float = 1e-3, save_dir: Optional[str] = None):
        optimizer = torch.optim.AdamW(self.mapper.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        best_loss = float('inf')

        print("\n" + "=" * 60)
        print("Training: z_dcae → Sines (Oscillator Loss)")
        print("=" * 60)

        for epoch in range(n_epochs):
            self.mapper.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_active = 0.0
            n_batches = 0

            for batch in dataloader:
                metrics = self.train_step(batch, optimizer)
                epoch_loss += metrics['loss']
                epoch_recon += metrics['recon']
                epoch_active += metrics['n_active']
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_recon = epoch_recon / max(n_batches, 1)
            avg_active = epoch_active / max(n_batches, 1)

            print(f"Epoch {epoch:4d}: loss={avg_loss:.4f} recon={avg_recon:.4f} active={avg_active:.1f}/{self.n_sines} | lr={scheduler.get_last_lr()[0]:.2e}")

            if save_dir and avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'mapper_state_dict': self.mapper.state_dict(),
                    'loss': avg_loss,
                    'n_sines': self.n_sines,
                }, str(save_path / "best_model.pt"))

            if epoch % 20 == 0:
                clear_memory()

        if save_dir:
            torch.save({
                'epoch': n_epochs,
                'mapper_state_dict': self.mapper.state_dict(),
                'loss': avg_loss,
                'n_sines': self.n_sines,
            }, str(save_path / "final_model.pt"))
            print(f"\nSaved to {save_dir}")

        print(f"Training complete! Best loss: {best_loss:.4f}")

        # Analyze predictions
        print("\n" + "=" * 60)
        print("PREDICTION ANALYSIS")
        print("=" * 60)

        self.mapper.eval()
        with torch.no_grad():
            for batch in dataloader:
                latent = batch['latent'].to(self.device)
                B, C, H, T = latent.shape
                z = latent.permute(0, 3, 1, 2).reshape(B, T, C * H)

                params = self.mapper(z)
                freqs = params['freqs']
                amps = params['amps']

                # Show distribution
                print(f"\n  Frequency distribution:")
                for i in range(self.n_sines):
                    f_mean = freqs[:, :, i].mean().item()
                    f_std = freqs[:, :, i].std().item()
                    a_mean = amps[:, :, i].mean().item()
                    print(f"    Sine {i}: freq={f_mean:.0f}±{f_std:.0f} Hz, amp={a_mean:.3f}")

                # Active count at different thresholds
                print(f"\n  Active sines:")
                for thresh in [0.05, 0.1, 0.2, 0.5]:
                    n = (amps > thresh).float().sum(dim=-1).mean().item()
                    print(f"    amp > {thresh}: {n:.1f}/{self.n_sines}")

                break

        return self.mapper


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='/home/arlo/gcs-bucket/Manifests/unified_manifest.json')
    parser.add_argument('--oscillator', type=str, default='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/test_outputs/sine_oscillator_v2/sine_oscillator_v2.pt')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_sines', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--sparsity', type=float, default=0.01)
    parser.add_argument('--freq_min', type=float, default=100.0)
    parser.add_argument('--freq_max', type=float, default=2000.0)
    parser.add_argument('--groups', type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("SAMI Training with Oscillator Loss (FAST!)")
    print("=" * 60)
    print("\nKey advantage: No audio synthesis needed!")
    print("  Old: mapper → synth → STFT → loss (slow)")
    print("  New: mapper → oscillator → latent MSE (fast)")
    sys.stdout.flush()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    sys.stdout.flush()

    filter_groups = args.groups.split(',') if args.groups else None

    print(f"\nLoading dataset...")
    sys.stdout.flush()

    dataset = UnifiedLatentDataset(
        manifest_path=args.manifest,
        max_samples=args.max_samples,
        filter_groups=filter_groups,
    )

    print(f"Creating dataloader with {len(dataset)} samples...")
    sys.stdout.flush()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"Dataloader ready: {len(dataloader)} batches")
    sys.stdout.flush()

    trainer = OscillatorTrainer(
        oscillator_path=args.oscillator,
        n_sines=args.n_sines,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        sparsity_weight=args.sparsity,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        device=device,
    )

    save_dir = "/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/sami_oscillator"

    trainer.train(
        dataloader,
        n_epochs=args.epochs,
        lr=args.lr,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
