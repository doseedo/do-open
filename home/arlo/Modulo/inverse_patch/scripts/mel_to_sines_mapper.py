#!/usr/bin/env python3
"""
Mel-to-Sines Mapper: Match the decoder's actual structure.

The chain:
  z → decoder → 128 mel bins → vocoder → audio → sines

We do:
  z → mapper → 128 mel bins → deterministic → 64 sines

Key insight from vocoder probe:
  - Mel bin → Hz is deterministic (~10% error)
  - Single bin → single frequency (no harmonics)

So if we predict mel correctly, sines follow deterministically.

Training:
  - Get decoder's mel output as target
  - Train mapper to predict mel from z
  - Convert predicted mel to sines

This uses ALL of z (not just 4 bands) at FULL resolution (128 bins).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import sys
import os

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE


# ============================================================================
# MEL BIN TO HZ CONVERSION (from vocoder probe)
# ============================================================================

def mel_bin_to_hz(bin_idx, n_mels=128, f_min=40, f_max=16000):
    """Convert mel bin index to center frequency in Hz."""
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_val = mel_min + (mel_max - mel_min) * bin_idx / n_mels
    hz = 700 * (10 ** (mel_val / 2595) - 1)
    return hz


# Precompute mel bin to Hz mapping
MEL_BIN_TO_HZ = torch.tensor([mel_bin_to_hz(i) for i in range(128)])


def mel_to_sines(mel_frame, n_sines=64, device='cuda'):
    """
    Convert mel spectrogram frame to sine parameters.

    Args:
        mel_frame: [B, 128] or [B, T, 128] mel energies
        n_sines: number of sines to extract

    Returns:
        freqs: [B, n_sines] or [B, T, n_sines] frequencies in Hz
        amps: [B, n_sines] or [B, T, n_sines] amplitudes
    """
    mel_bin_hz = MEL_BIN_TO_HZ.to(device)

    if mel_frame.dim() == 2:
        # [B, 128]
        B = mel_frame.shape[0]

        # Get top-k bins by energy
        topk_vals, topk_idx = mel_frame.topk(n_sines, dim=-1)  # [B, n_sines]

        # Convert bin indices to Hz
        freqs = mel_bin_hz[topk_idx]  # [B, n_sines]

        # Amplitudes from mel energy (already in reasonable range)
        amps = torch.sigmoid(topk_vals)  # [B, n_sines]

        return freqs, amps

    elif mel_frame.dim() == 3:
        # [B, T, 128]
        B, T, _ = mel_frame.shape

        # Get top-k bins by energy per frame
        topk_vals, topk_idx = mel_frame.topk(n_sines, dim=-1)  # [B, T, n_sines]

        # Convert bin indices to Hz
        freqs = mel_bin_hz[topk_idx.reshape(-1)].reshape(B, T, n_sines)

        # Amplitudes
        amps = torch.sigmoid(topk_vals)

        return freqs, amps


# ============================================================================
# DATASET: Load z and get decoder's mel output as target
# ============================================================================

class ZToMelDataset(Dataset):
    """Dataset that provides z and decoder's mel output."""

    def __init__(self, dcae, sms_manifest_path, max_samples=1000, skip_drums=True, device='cuda'):
        import orjson

        self.dcae = dcae
        self.device = device
        self.cached_data = []  # Pre-compute everything

        with open(sms_manifest_path, 'rb') as f:
            manifest = orjson.loads(f.read())

        count = 0
        print("  Pre-computing mel outputs (one-time)...")
        for entry in manifest['entries']:
            if count >= max_samples:
                break

            path = entry['path']

            if skip_drums:
                if any(kw in path.lower() for kw in ['drum', 'kick', 'snare', 'hat', 'perc']):
                    continue

            if os.path.exists(path):
                try:
                    data = self._load_and_process(path)
                    if data is not None:
                        self.cached_data.append(data)
                        count += 1
                        if count % 50 == 0:
                            print(f"    Processed {count} samples...")
                except Exception as e:
                    continue

        print(f"  Cached {len(self.cached_data)} samples")

    def _load_and_process(self, path):
        """Load and process one sample."""
        data = torch.load(path, weights_only=True, map_location='cpu')

        # Load latent
        latent_path = data.get('latent_path', None)
        if latent_path and os.path.exists(latent_path):
            latent_data = torch.load(latent_path, weights_only=True, map_location='cpu')
            if isinstance(latent_data, dict):
                z = latent_data['latents']
            else:
                z = latent_data
        else:
            return None

        if z.dim() == 4:
            z = z.squeeze(0)

        # Limit sequence length
        max_T = 32
        if z.shape[-1] > max_T:
            z = z[..., :max_T]

        # Get decoder's mel output - use .decoder() not .decode().sample
        with torch.no_grad():
            z_4d = z.unsqueeze(0).to(self.device)
            z_denorm = z_4d / self.dcae.scale_factor + self.dcae.shift_factor
            mel = self.dcae.dcae.decoder(z_denorm)  # [B, C, 128, T]
            mel = mel.mean(dim=1).squeeze(0).cpu()  # [128, T_mel]

        freqs = data.get('freqs', None)
        amps = data.get('amps', None)

        return {
            'z': z.cpu(),
            'mel': mel,
            'freqs': freqs,
            'amps': amps,
        }

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        return self.cached_data[idx]


def collate_fn(batch):
    """Collate with padding."""
    # Find max T
    max_T_z = max(b['z'].shape[-1] for b in batch)
    max_T_mel = max(b['mel'].shape[-1] for b in batch)

    z_batch = []
    mel_batch = []
    freqs_batch = []
    amps_batch = []

    # Also find max T for freqs
    max_T_freqs = 0
    for b in batch:
        if b['freqs'] is not None:
            max_T_freqs = max(max_T_freqs, b['freqs'].shape[0])

    for b in batch:
        z = b['z']
        mel = b['mel']

        # Pad z
        T_z = z.shape[-1]
        if T_z < max_T_z:
            z = F.pad(z, (0, max_T_z - T_z))
        z_batch.append(z)

        # Pad mel
        T_mel = mel.shape[-1]
        if T_mel < max_T_mel:
            mel = F.pad(mel, (0, max_T_mel - T_mel))
        mel_batch.append(mel)

        # Handle sines if present
        if b['freqs'] is not None:
            freqs = b['freqs']
            amps = b['amps']

            T_sines = freqs.shape[0]
            if T_sines < max_T_freqs:
                freqs = F.pad(freqs, (0, 0, 0, max_T_freqs - T_sines))
                amps = F.pad(amps, (0, 0, 0, max_T_freqs - T_sines))

            freqs_batch.append(freqs)
            amps_batch.append(amps)

    result = {
        'z': torch.stack(z_batch),
        'mel': torch.stack(mel_batch),
    }

    if freqs_batch and len(freqs_batch) == len(batch):
        result['freqs'] = torch.stack(freqs_batch)
        result['amps'] = torch.stack(amps_batch)

    return result


# ============================================================================
# MAPPER: z → 128 mel bins
# ============================================================================

class MelMapper(nn.Module):
    """
    Predict 128 mel bins from z.

    Matches decoder structure: z[128] → mel[128]
    Uses all z dims, full mel resolution.
    """

    def __init__(self, hidden_dim=256, n_layers=3):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Temporal attention (decoder uses attention)
        self.temporal_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True),
                'norm1': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
                'norm2': nn.LayerNorm(hidden_dim),
            })
            for _ in range(n_layers)
        ])

        # Output: predict 128 mel bins
        self.mel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
        )

    def forward(self, z):
        """
        z: [B, 8, 16, T]
        Returns: mel [B, T*8, 128] (upsampled to match decoder output)
        """
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T).permute(0, 2, 1)  # [B, T, 128]

        # Encode
        h = self.encoder(z_flat)  # [B, T, hidden]

        # Temporal processing
        for layer in self.temporal_layers:
            h_attn, _ = layer['attn'](h, h, h)
            h = layer['norm1'](h + h_attn)
            h_ff = layer['ff'](h)
            h = layer['norm2'](h + h_ff)

        # Predict mel
        mel = self.mel_head(h)  # [B, T, 128]

        # Upsample 8x to match decoder output resolution
        mel = mel.permute(0, 2, 1)  # [B, 128, T]
        mel = F.interpolate(mel, scale_factor=8, mode='linear', align_corners=False)
        mel = mel.permute(0, 2, 1)  # [B, T*8, 128]

        return mel


class MelMapperV2(nn.Module):
    """
    V2: Predicts mel with discovered structure.

    Uses quadratic on dims 48-63 for band energies,
    but predicts full 128 bins.
    """

    def __init__(self, hidden_dim=256):
        super().__init__()

        # Quadratic encoder for frequency dims
        self.freq_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim),  # dims 48-63
            nn.GELU(),
        )
        self.freq_quad = nn.Linear(16, hidden_dim, bias=False)  # Quadratic term

        # Linear encoder for other dims
        self.other_encoder = nn.Sequential(
            nn.Linear(112, hidden_dim),  # dims 0-47 + 64-127
            nn.GELU(),
        )

        # Combine
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

        # Temporal attention
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Output
        self.mel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
        )

    def forward(self, z):
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T).permute(0, 2, 1)  # [B, T, 128]

        # Split by discovered roles
        z_freq = z_flat[..., 48:64]  # [B, T, 16]
        z_other = torch.cat([z_flat[..., :48], z_flat[..., 64:]], dim=-1)  # [B, T, 112]

        # Encode with quadratic for freq dims
        h_freq = self.freq_encoder(z_freq) + self.freq_quad(z_freq ** 2)
        h_other = self.other_encoder(z_other)

        # Combine
        h = self.combine(torch.cat([h_freq, h_other], dim=-1))

        # Temporal
        h_attn, _ = self.temporal_attn(h, h, h)
        h = self.temporal_norm(h + h_attn)

        # Predict mel
        mel = self.mel_head(h)

        # Upsample
        mel = mel.permute(0, 2, 1)
        mel = F.interpolate(mel, scale_factor=8, mode='linear', align_corners=False)
        mel = mel.permute(0, 2, 1)

        return mel


def train_mel_mapper(model, dataloader, n_epochs=100, lr=1e-3, device='cuda'):
    """Train mapper to predict decoder's mel output."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            z = batch['z'].to(device)
            target_mel = batch['mel'].to(device)  # [B, 128, T_mel]
            target_mel = target_mel.permute(0, 2, 1)  # [B, T_mel, 128]

            pred_mel = model(z)  # [B, T_mel, 128]

            # Match lengths
            min_T = min(pred_mel.shape[1], target_mel.shape[1])
            pred_mel = pred_mel[:, :min_T, :]
            target_mel = target_mel[:, :min_T, :]

            # MSE loss on mel
            loss = F.mse_loss(pred_mel, target_mel)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: mel_loss={total_loss/n_batches:.4f}")

    return model


def evaluate_sine_accuracy(model, dataloader, n_sines=64, device='cuda'):
    """
    Evaluate: predicted mel → sines vs target sines.
    """
    model.eval()

    # Import sinkhorn loss for comparison
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'train_harmonic_ops',
        '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/training/train_harmonic_ops.py'
    )
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    sinkhorn_matching_loss = train_module.sinkhorn_matching_loss

    total_freq_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            z = batch['z'].to(device)

            if 'freqs' not in batch:
                continue

            target_freqs = batch['freqs'].to(device)
            target_amps = batch['amps'].to(device)

            # Predict mel
            pred_mel = model(z)  # [B, T, 128]

            # Convert mel to sines
            pred_freqs, pred_amps = mel_to_sines(pred_mel, n_sines=n_sines, device=device)

            # Match time dimension
            min_T = min(pred_freqs.shape[1], target_freqs.shape[1])
            pred_freqs = pred_freqs[:, :min_T, :]
            pred_amps = pred_amps[:, :min_T, :]
            target_freqs = target_freqs[:, :min_T, :]
            target_amps = target_amps[:, :min_T, :]

            # Compute loss
            _, metrics = sinkhorn_matching_loss(
                pred_freqs, pred_amps,
                target_freqs, target_amps,
            )

            total_freq_loss += metrics['freq_loss']
            n_batches += 1

    return total_freq_loss / n_batches if n_batches > 0 else float('inf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=256)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load DCAE
    print("\nLoading DCAE...")
    DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
    VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    dcae = MusicDCAE(dcae_checkpoint_path=DCAE_PATH, vocoder_checkpoint_path=VOCODER_PATH)
    dcae.dcae.to(device)
    dcae.dcae.eval()

    # Load dataset
    print("\nLoading dataset...")
    dataset = ZToMelDataset(
        dcae=dcae,
        sms_manifest_path='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json',
        max_samples=args.max_samples,
        skip_drums=True,
        device=device,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Can't use multiprocessing with CUDA in dataset
        collate_fn=collate_fn,
    )

    # Test 1: Standard mel mapper (DISABLED)
    # print("\n" + "=" * 70)
    # print("TEST 1: Mel Mapper (z → 128 mel bins → 64 sines)")
    # print("=" * 70)
    # model1 = MelMapper(hidden_dim=args.hidden_dim, n_layers=3)
    # n_params = sum(p.numel() for p in model1.parameters())
    # print(f"Params: {n_params:,}")
    # print("\nTraining to predict decoder's mel output...")
    # model1 = train_mel_mapper(model1, dataloader, n_epochs=args.epochs, lr=args.lr, device=device)
    # print("\nEvaluating sine accuracy...")
    # freq_loss1 = evaluate_sine_accuracy(model1, dataloader, device=device)
    # print(f"  Freq loss: {freq_loss1:.4f} (~{freq_loss1*60:.1f} semitones)")
    freq_loss1 = float('inf')  # Placeholder

    # Test 2: V2 with discovered structure
    print("\n" + "=" * 70)
    print("TEST 2: Mel Mapper V2 (quadratic on dims 48-63)")
    print("=" * 70)

    model2 = MelMapperV2(hidden_dim=args.hidden_dim)
    n_params = sum(p.numel() for p in model2.parameters())
    print(f"Params: {n_params:,}")

    print("\nTraining...")
    model2 = train_mel_mapper(model2, dataloader, n_epochs=args.epochs, lr=args.lr, device=device)

    print("\nEvaluating sine accuracy...")
    freq_loss2 = evaluate_sine_accuracy(model2, dataloader, device=device)
    print(f"  Freq loss: {freq_loss2:.4f} (~{freq_loss2*60:.1f} semitones)")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  Mel Mapper:    {freq_loss1:.4f} (~{freq_loss1*60:.1f} semitones)")
    print(f"  Mel Mapper V2: {freq_loss2:.4f} (~{freq_loss2*60:.1f} semitones)")
    print(f"\n  Previous best (harmonic MLP): 0.046 (~2.8 semitones)")

    best = min(freq_loss1, freq_loss2)
    if best < 0.046:
        print(f"\n✓ BETTER: Mel approach ({best:.4f}) beats direct MLP!")
    elif best < 0.06:
        print(f"\n~ COMPARABLE: Mel approach ({best:.4f})")
    else:
        print(f"\n✗ WORSE: Mel approach ({best:.4f})")


if __name__ == "__main__":
    main()
