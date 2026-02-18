#!/usr/bin/env python3
"""
Train Mel → Full SMS mapper (freqs, amps, phases, noise).

Predicts all 4 components needed for proper SMS synthesis:
- freqs: [T, 128] Hz
- amps: [T, 128]
- phases: [T, 128]
- noise_amps: [T, 8]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import sys
import os
import orjson

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/scripts')
from mel_to_sines_mapper import MelMapperV2


# ============================================================================
# DATASET
# ============================================================================

class MelToFullSMSDataset(Dataset):
    """Dataset that provides mel + full SMS params."""

    def __init__(self, sms_manifest_path, mel_mapper, device='cuda', max_samples=1000):
        self.samples = []
        self.mel_mapper = mel_mapper
        self.device = device

        with open(sms_manifest_path, 'rb') as f:
            manifest = orjson.loads(f.read())

        for entry in manifest['entries'][:max_samples * 2]:
            path = entry['path']
            if any(kw in path.lower() for kw in ['drum', 'kick', 'snare', 'hat', 'perc']):
                continue
            if not os.path.exists(path):
                continue

            try:
                data = torch.load(path, weights_only=True, map_location='cpu')

                # Need latent for mel prediction
                lat_path = data.get('latent_path')
                if not lat_path or not os.path.exists(lat_path):
                    continue

                # Check all required keys
                if not all(k in data for k in ['freqs', 'amps', 'phases', 'noise_amps']):
                    continue

                self.samples.append({
                    'sms_path': path,
                    'latent_path': lat_path
                })

                if len(self.samples) >= max_samples:
                    break
            except:
                continue

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load SMS data
        sms_data = torch.load(sample['sms_path'], weights_only=True, map_location='cpu')
        freqs = sms_data['freqs']  # [T, 128]
        amps = sms_data['amps']
        phases = sms_data['phases']
        noise_amps = sms_data['noise_amps']  # [T, 8]

        # Load latent
        lat_data = torch.load(sample['latent_path'], weights_only=True, map_location='cpu')
        z = lat_data.get('latents', lat_data)
        if z.dim() == 3:
            z = z.unsqueeze(0)

        # Limit length
        T_sms = freqs.shape[0]
        T_z = z.shape[-1]
        T = min(T_sms, T_z * 8, 256)  # mel is 8x upsampled from z

        freqs = freqs[:T]
        amps = amps[:T]
        phases = phases[:T]
        noise_amps = noise_amps[:T]

        # Compute mel from z using mel_mapper
        z_len = (T + 7) // 8
        z = z[:, :, :, :z_len]

        return {
            'z': z.squeeze(0),  # [8, 16, T_z]
            'freqs': freqs,
            'amps': amps,
            'phases': phases,
            'noise_amps': noise_amps,
        }


def collate_fn(batch):
    # Find max lengths
    max_T = max(b['freqs'].shape[0] for b in batch)
    max_T_z = max(b['z'].shape[-1] for b in batch)

    z_batch = []
    freqs_batch = []
    amps_batch = []
    phases_batch = []
    noise_batch = []

    for b in batch:
        # Pad z
        z = b['z']
        if z.shape[-1] < max_T_z:
            z = F.pad(z, (0, max_T_z - z.shape[-1]))
        z_batch.append(z)

        # Pad SMS
        T = b['freqs'].shape[0]
        pad_T = max_T - T

        freqs_batch.append(F.pad(b['freqs'], (0, 0, 0, pad_T)))
        amps_batch.append(F.pad(b['amps'], (0, 0, 0, pad_T)))
        phases_batch.append(F.pad(b['phases'], (0, 0, 0, pad_T)))
        noise_batch.append(F.pad(b['noise_amps'], (0, 0, 0, pad_T)))

    return {
        'z': torch.stack(z_batch),
        'freqs': torch.stack(freqs_batch),
        'amps': torch.stack(amps_batch),
        'phases': torch.stack(phases_batch),
        'noise_amps': torch.stack(noise_batch),
    }


# ============================================================================
# MODEL
# ============================================================================

class FullSMSMapper(nn.Module):
    """
    Predicts full SMS parameters from mel.

    mel [B, T, 128] → freqs [B, T, 128], amps [B, T, 128],
                      phases [B, T, 128], noise [B, T, 8]
    """

    def __init__(self, n_sines=128, n_noise_bands=8, hidden_dim=512):
        super().__init__()
        self.n_sines = n_sines
        self.n_noise_bands = n_noise_bands

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Temporal context (important for phase continuity)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Frequency head - predict which mel bins are active
        self.freq_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_sines),
        )

        # Amplitude head - refine mel energies
        self.amp_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_sines),
        )

        # Phase head - predict phase (important for coherent sound)
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_sines),
        )

        # Noise head - predict noise band amplitudes
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_noise_bands),
        )

        # Frequency scaling
        self.log_freq_min = np.log(20.0)
        self.log_freq_max = np.log(16000.0)

    def forward(self, mel):
        """
        mel: [B, T, 128]
        """
        B, T, _ = mel.shape

        # Encode
        h = self.encoder(mel)  # [B, T, hidden]

        # Temporal context
        h_temporal, _ = self.temporal(h)  # [B, T, hidden*2]

        # Predict all components
        # Frequencies: log scale
        freq_logits = self.freq_head(h_temporal)
        freq_norm = torch.sigmoid(freq_logits)
        log_freqs = self.log_freq_min + freq_norm * (self.log_freq_max - self.log_freq_min)
        freqs = torch.exp(log_freqs)

        # Amplitudes: [0, 0.5] range
        amps = torch.sigmoid(self.amp_head(h_temporal)) * 0.5

        # Phases: [-pi, pi]
        phases = torch.tanh(self.phase_head(h_temporal)) * np.pi

        # Noise: [0, 0.2] range
        noise_amps = torch.sigmoid(self.noise_head(h_temporal)) * 0.2

        return freqs, amps, phases, noise_amps


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def amplitude_weighted_freq_loss(pred_freqs, pred_amps, target_freqs, target_amps):
    """
    Frequency loss weighted by amplitude.
    Errors on loud sines matter more.
    """
    # Log frequency ratio (in semitones / 12)
    freq_ratio = torch.log2(pred_freqs / (target_freqs + 1))
    freq_loss_per_sine = freq_ratio.abs()

    # Weight by target amplitude (loud sines matter more)
    weights = target_amps / (target_amps.sum(dim=-1, keepdim=True) + 1e-8)
    weighted_loss = (freq_loss_per_sine * weights).sum(dim=-1)

    return weighted_loss.mean()


def phase_loss(pred_phases, target_phases, target_amps):
    """
    Phase loss, weighted by amplitude.
    Use circular distance.
    """
    # Circular distance
    phase_diff = pred_phases - target_phases
    phase_dist = torch.abs(torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff)))

    # Weight by amplitude
    weights = target_amps / (target_amps.sum(dim=-1, keepdim=True) + 1e-8)
    weighted_loss = (phase_dist * weights).sum(dim=-1)

    return weighted_loss.mean()


def sms_loss(pred_freqs, pred_amps, pred_phases, pred_noise,
             target_freqs, target_amps, target_phases, target_noise):
    """Combined SMS loss."""

    # Frequency loss (amplitude-weighted)
    freq_loss = amplitude_weighted_freq_loss(pred_freqs, pred_amps, target_freqs, target_amps)

    # Amplitude loss (L1)
    amp_loss = F.l1_loss(pred_amps, target_amps)

    # Phase loss (amplitude-weighted circular)
    ph_loss = phase_loss(pred_phases, target_phases, target_amps)

    # Noise loss (L1)
    noise_loss = F.l1_loss(pred_noise, target_noise)

    # Combine
    total = freq_loss + amp_loss * 10 + ph_loss * 0.1 + noise_loss * 10

    return total, {
        'freq_loss': freq_loss.item(),
        'amp_loss': amp_loss.item(),
        'phase_loss': ph_loss.item(),
        'noise_loss': noise_loss.item(),
    }


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load mel mapper
    print("Loading mel mapper...")
    mel_mapper = MelMapperV2().to(device)
    mel_ckpt = torch.load(
        '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/mel_mapper/best_model.pt',
        weights_only=True
    )
    mel_mapper.load_state_dict(mel_ckpt['model_state_dict'])
    mel_mapper.eval()

    # Dataset
    print("Loading dataset...")
    dataset = MelToFullSMSDataset(
        '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json',
        mel_mapper=mel_mapper,
        device=device,
        max_samples=args.max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Can't use multiprocessing with mel_mapper
        collate_fn=collate_fn,
    )

    # Model
    model = FullSMSMapper(n_sines=128, n_noise_bands=8, hidden_dim=args.hidden_dim).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_freq = 0
        total_amp = 0
        total_phase = 0
        total_noise = 0
        n_batches = 0

        for batch in dataloader:
            z = batch['z'].to(device)
            target_freqs = batch['freqs'].to(device)
            target_amps = batch['amps'].to(device)
            target_phases = batch['phases'].to(device)
            target_noise = batch['noise_amps'].to(device)

            # Get mel from mel_mapper
            with torch.no_grad():
                mel = mel_mapper(z)  # [B, T, 128]

            # Match lengths
            T_mel = mel.shape[1]
            T_sms = target_freqs.shape[1]
            T = min(T_mel, T_sms)
            mel = mel[:, :T, :]
            target_freqs = target_freqs[:, :T, :]
            target_amps = target_amps[:, :T, :]
            target_phases = target_phases[:, :T, :]
            target_noise = target_noise[:, :T, :]

            # Forward
            pred_freqs, pred_amps, pred_phases, pred_noise = model(mel)

            # Loss
            loss, metrics = sms_loss(
                pred_freqs, pred_amps, pred_phases, pred_noise,
                target_freqs, target_amps, target_phases, target_noise
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_freq += metrics['freq_loss']
            total_amp += metrics['amp_loss']
            total_phase += metrics['phase_loss']
            total_noise += metrics['noise_loss']
            n_batches += 1

        scheduler.step()

        avg_loss = total_loss / n_batches
        avg_freq = total_freq / n_batches
        avg_amp = total_amp / n_batches
        avg_phase = total_phase / n_batches
        avg_noise = total_noise / n_batches

        semitones = avg_freq * 12

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, freq={semitones:.1f}st, "
                  f"amp={avg_amp:.4f}, phase={avg_phase:.3f}, noise={avg_noise:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'freq_loss': avg_freq,
            }, '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/mel_to_sms/full_sms_best.pt')

    print(f"\nBest loss: {best_loss:.4f}")
    print("Saved to checkpoints/mel_to_sms/full_sms_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--max_samples', type=int, default=500)
    args = parser.parse_args()

    train(args)
