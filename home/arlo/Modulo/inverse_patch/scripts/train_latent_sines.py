#!/usr/bin/env python3
"""
Train z_dcae → Sines using LATENT SPACE loss (no audio synthesis!)

Key insight from test_latent_linearity.py:
- DCAE latent space is ~96% linear for sine combinations
- z(sine1 + sine2) ≈ z(sine1) + z(sine2)

Approach:
1. Precompute sine→latent dictionary for grid of (freq, amp)
2. During training: predict sine params → interpolate latents → sum → compare to target
3. ~100x faster than audio synthesis loss!

The "DC offset" (silence latent) needs to be subtracted.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import orjson
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf

sys.path.insert(0, "/home/arlo/Data/ACE-Step")
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

# Paths
DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

SAMPLE_RATE = 44100


# ============================================================
# SINE LATENT DICTIONARY
# ============================================================

class SineLatentDict:
    """
    Precomputed dictionary: (freq, amp) → z_latent

    Allows fast lookup/interpolation during training.
    """

    def __init__(
        self,
        dcae: MusicDCAE,
        freq_bins: int = 256,  # Number of frequency bins
        freq_min: float = 20.0,
        freq_max: float = 16000.0,
        duration: float = 0.5,  # Short duration for dict entries
        device: str = 'cuda',
    ):
        self.device = device
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freq_bins = freq_bins
        self.duration = duration
        self.sample_rate = SAMPLE_RATE

        # Log-spaced frequencies (perceptually uniform)
        self.freqs = torch.logspace(
            np.log10(freq_min), np.log10(freq_max), freq_bins
        ).to(device)

        # Build dictionary
        print(f"Building sine latent dictionary ({freq_bins} frequencies)...")
        self._build_dict(dcae)
        print(f"Dictionary built! Shape: {self.latent_dict.shape}")

    def _generate_sine(self, freq: float) -> np.ndarray:
        """Generate unit amplitude sine."""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        return np.sin(2 * np.pi * freq * t).astype(np.float32)

    def _encode(self, dcae, audio: np.ndarray) -> torch.Tensor:
        """Encode audio to latent."""
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        audio_stereo = audio_tensor.unsqueeze(0).unsqueeze(0).expand(-1, 2, -1)
        audio_lengths = torch.tensor([audio_stereo.shape[-1]], device=self.device)

        with torch.no_grad():
            z, _ = dcae.encode(audio_stereo, audio_lengths=audio_lengths, sr=self.sample_rate)

        return z.squeeze(0)  # [8, 16, T]

    def _build_dict(self, dcae):
        """Build the frequency→latent dictionary."""
        # First encode silence to get DC offset
        silence = np.zeros(int(self.sample_rate * self.duration), dtype=np.float32)
        self.z_silence = self._encode(dcae, silence)

        latents = []
        for i, freq in enumerate(self.freqs.cpu().numpy()):
            audio = self._generate_sine(freq)
            z = self._encode(dcae, audio)
            # Subtract DC offset
            z = z - self.z_silence
            latents.append(z)

            if (i + 1) % 50 == 0:
                print(f"  Encoded {i+1}/{self.freq_bins} frequencies")

        # Stack: [freq_bins, 8, 16, T]
        self.latent_dict = torch.stack(latents, dim=0)
        self.n_frames = self.latent_dict.shape[-1]

    def lookup(self, freqs: torch.Tensor, amps: torch.Tensor) -> torch.Tensor:
        """
        Look up latents for given frequencies and amplitudes.

        Args:
            freqs: [B, T, n_sines] frequencies in Hz
            amps: [B, T, n_sines] amplitudes

        Returns:
            z_pred: [B, 8, 16, T_dict] predicted latent (sum of sine latents)
        """
        B, T, N = freqs.shape

        # Convert freqs to indices (log scale)
        log_freqs = torch.log10(freqs.clamp(min=self.freq_min, max=self.freq_max))
        log_min = np.log10(self.freq_min)
        log_max = np.log10(self.freq_max)

        # Normalize to [0, freq_bins-1]
        indices = (log_freqs - log_min) / (log_max - log_min) * (self.freq_bins - 1)
        indices = indices.clamp(0, self.freq_bins - 1)

        # Bilinear interpolation
        idx_low = indices.floor().long()
        idx_high = (idx_low + 1).clamp(max=self.freq_bins - 1)
        weight_high = indices - idx_low.float()
        weight_low = 1 - weight_high

        # Lookup latents [B, T, N] → [B, T, N, 8, 16, T_dict]
        # This is memory intensive, so we'll process frame by frame

        # For now, average across time frames (simplification)
        # In full version, would interpolate temporally too

        # Take middle frame of each sine's time dimension
        mid_frame = T // 2
        freqs_mid = freqs[:, mid_frame, :]  # [B, N]
        amps_mid = amps[:, mid_frame, :]    # [B, N]

        # Recompute indices for mid frame
        log_freqs_mid = torch.log10(freqs_mid.clamp(min=self.freq_min, max=self.freq_max))
        indices_mid = (log_freqs_mid - log_min) / (log_max - log_min) * (self.freq_bins - 1)
        indices_mid = indices_mid.clamp(0, self.freq_bins - 1)

        idx_low = indices_mid.floor().long()
        idx_high = (idx_low + 1).clamp(max=self.freq_bins - 1)
        weight_high = (indices_mid - idx_low.float()).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weight_low = 1 - weight_high

        # Gather latents
        z_low = self.latent_dict[idx_low]   # [B, N, 8, 16, T_dict]
        z_high = self.latent_dict[idx_high] # [B, N, 8, 16, T_dict]

        # Interpolate
        z_interp = weight_low * z_low + weight_high * z_high  # [B, N, 8, 16, T_dict]

        # Scale by amplitude
        amps_expanded = amps_mid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1, 1]
        z_scaled = z_interp * amps_expanded

        # Sum across sines
        z_sum = z_scaled.sum(dim=1)  # [B, 8, 16, T_dict]

        # Add back DC offset
        z_sum = z_sum + self.z_silence.unsqueeze(0)

        return z_sum


# ============================================================
# MODEL
# ============================================================

class ResBlock(nn.Module):
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


class LatentSineMapper(nn.Module):
    """Maps DCAE latent to sparse sine parameters."""

    def __init__(
        self,
        frame_dim: int = 128,
        max_sines: int = 64,  # Fewer sines since we're more accurate
        hidden_dim: int = 256,
        n_blocks: int = 3,
        freq_min: float = 20.0,
        freq_max: float = 16000.0,
    ):
        super().__init__()
        self.max_sines = max_sines
        self.freq_min = freq_min
        self.freq_max = freq_max

        self.encoder = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.GELU(),
            *[ResBlock(hidden_dim) for _ in range(n_blocks)],
        )

        self.freq_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_sines),
        )

        self.amp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_sines),
        )

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, T, 128] flattened DCAE latent
        Returns:
            freqs: [B, T, max_sines] in Hz
            amps: [B, T, max_sines] in [0, 1]
        """
        h = self.encoder(z)

        # Log-scale frequencies (more resolution at low freqs)
        freq_logits = self.freq_head(h)
        log_freq = torch.sigmoid(freq_logits) * (np.log10(self.freq_max) - np.log10(self.freq_min)) + np.log10(self.freq_min)
        freqs = 10 ** log_freq

        amps = torch.sigmoid(self.amp_head(h))

        return {'freqs': freqs, 'amps': amps}


# ============================================================
# DATASET
# ============================================================

class LatentDataset(Dataset):
    """Load precomputed latents with fixed temporal length."""

    def __init__(self, manifest_path: str, max_samples: int = 1000, target_frames: int = 64):
        self.target_frames = target_frames

        with open(manifest_path, 'rb') as f:
            manifest = orjson.loads(f.read())

        entries = [e for e in manifest.get('entries', []) if e.get('has_latent')][:max_samples * 2]

        self.data = []
        print(f"Loading {min(max_samples, len(entries))} samples (target_frames={target_frames})...")

        for entry in entries[:max_samples * 2]:
            if len(self.data) >= max_samples:
                break
            try:
                latent = torch.load(entry['latent_path'], map_location='cpu', weights_only=False)
                if isinstance(latent, dict):
                    latent = latent.get('latents', latent.get('latent', list(latent.values())[0]))
                if latent.dim() == 4:
                    latent = latent.squeeze(0)
                if latent.shape[0] == 8 and latent.shape[1] == 16:
                    # Resize to fixed temporal length
                    C, H, T = latent.shape
                    if T >= 10:  # Skip very short samples
                        # Reshape for interpolate: [1, C*H, T]
                        latent_flat = latent.reshape(1, C * H, T)
                        latent_resized = F.interpolate(
                            latent_flat,
                            size=target_frames,
                            mode='linear'
                        ).reshape(C, H, target_frames)  # [8, 16, target_frames]
                        self.data.append(latent_resized)
            except:
                continue

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================
# TRAINING
# ============================================================

def train(
    manifest_path: str,
    max_samples: int = 500,
    epochs: int = 50,
    batch_size: int = 4,
    max_sines: int = 64,
    sparsity_weight: float = 0.01,
    device: str = 'cuda',
):
    print("=" * 60)
    print("LATENT SPACE SINE TRAINING")
    print("=" * 60)
    print("\nNo audio synthesis - training in latent space!")
    print(f"Expected speedup: ~100x\n")

    # Load DCAE
    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_PATH,
        vocoder_checkpoint_path=VOCODER_PATH,
    ).to(device)
    dcae.eval()

    # Build sine dictionary
    sine_dict = SineLatentDict(dcae, freq_bins=128, device=device)

    # Load dataset (fixed 64 frames = ~0.75 seconds at 86fps)
    dataset = LatentDataset(manifest_path, max_samples, target_frames=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Model
    mapper = LatentSineMapper(max_sines=max_sines).to(device)
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    print(f"\nMapper params: {sum(p.numel() for p in mapper.parameters()):,}")
    print(f"Max sines: {max_sines}")
    print(f"Sparsity weight: {sparsity_weight}\n")

    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon = 0
        epoch_sparse = 0
        n_batches = 0

        for batch in dataloader:
            batch = batch.to(device)  # [B, 8, 16, T]
            B, C, H, T = batch.shape

            # Flatten for mapper
            z_flat = batch.permute(0, 3, 1, 2).reshape(B, T, C * H)  # [B, T, 128]

            # Predict sine params
            params = mapper(z_flat)
            freqs = params['freqs']  # [B, T, max_sines]
            amps = params['amps']

            # Look up predicted latent from sine dictionary
            z_pred = sine_dict.lookup(freqs, amps)  # [B, 8, 16, T_dict]

            # Interpolate predicted latent to match target resolution
            T_dict = z_pred.shape[-1]
            if T_dict != T:
                z_pred = F.interpolate(
                    z_pred.reshape(B, C * H, T_dict),
                    size=T,
                    mode='linear'
                ).reshape(B, C, H, T)

            z_target = batch

            # Reconstruction loss in latent space
            recon_loss = F.mse_loss(z_pred, z_target)

            # Sparsity loss
            sparsity_loss = amps.mean()

            loss = recon_loss + sparsity_weight * sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_sparse += sparsity_loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_sparse = epoch_sparse / n_batches
        n_active = (amps > 0.1).float().sum(dim=-1).mean().item()

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f} recon={avg_recon:.4f} sparse={avg_sparse:.4f} active={n_active:.0f} lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\nBest loss: {best_loss:.4f}")

    # Save
    save_dir = Path("/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/latent_sines")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': mapper.state_dict(),
        'config': {
            'max_sines': max_sines,
            'freq_min': 20.0,
            'freq_max': 16000.0,
        },
        'best_loss': best_loss,
    }, save_dir / "best_model.pt")
    print(f"Saved to {save_dir}")

    return mapper, sine_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='/home/arlo/gcs-bucket/Manifests/unified_manifest.json')
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_sines', type=int, default=64)
    parser.add_argument('--sparsity', type=float, default=0.01)
    args = parser.parse_args()

    train(
        args.manifest,
        max_samples=args.max_samples,
        epochs=args.epochs,
        max_sines=args.max_sines,
        sparsity_weight=args.sparsity,
    )
