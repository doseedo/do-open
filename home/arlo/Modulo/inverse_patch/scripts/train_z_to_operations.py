#!/usr/bin/env python3
"""
Train z → operations → sines → audio to match DCAE output.

The entire pipeline is differentiable:
- z → OperationNet → (f0, n_harm, centroid, decay, energy)
- operations → DifferentiableSineSynth → audio
- Loss: mel spectrogram match with DCAE

This learns the causal mapping from z to interpretable operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import sys
import os
from tqdm import tqdm

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
import orjson


# ============================================================================
# DIFFERENTIABLE OPERATION PREDICTION
# ============================================================================

class OperationNet(nn.Module):
    """
    Predicts operation parameters from z latent.

    Input: z [B, 128, T] (flattened latent)
    Output: operation parameters per frame
    """

    def __init__(self, n_sines=32):
        super().__init__()
        self.n_sines = n_sines

        # Temporal encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Operation parameter heads (per-frame predictions)
        # f0: fundamental frequency (Hz)
        self.f0_head = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=1),
        )

        # Partial weights: relative amplitudes of harmonics
        self.partial_head = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, n_sines, kernel_size=1),
        )

        # Decay: per-partial decay rates
        self.decay_head = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, n_sines, kernel_size=1),
        )

        # Inharmonicity: deviation from perfect harmonic ratios
        self.inharm_head = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, n_sines, kernel_size=1),
        )

    def forward(self, z):
        """
        Args:
            z: [B, 128, T]

        Returns:
            dict of parameters, each [B, T, ...]
        """
        h = self.encoder(z)  # [B, 256, T]

        # f0: 50-1000 Hz
        f0_raw = self.f0_head(h).squeeze(1)  # [B, T]
        f0 = 50 + 950 * torch.sigmoid(f0_raw)

        # Partial weights: softmax over partials, scaled by energy
        partial_raw = self.partial_head(h)  # [B, n_sines, T]
        partial_weights = F.softmax(partial_raw, dim=1).permute(0, 2, 1)  # [B, T, n_sines]

        # Decay: 0.01 - 1.0 (slow to fast)
        decay_raw = self.decay_head(h).permute(0, 2, 1)  # [B, T, n_sines]
        decay = 0.01 + 0.99 * torch.sigmoid(decay_raw)

        # Inharmonicity: -0.1 to 0.1 (deviation from integer ratios)
        inharm_raw = self.inharm_head(h).permute(0, 2, 1)  # [B, T, n_sines]
        inharm = 0.2 * torch.tanh(inharm_raw)

        return {
            'f0': f0,  # [B, T]
            'partial_weights': partial_weights,  # [B, T, n_sines]
            'decay': decay,  # [B, T, n_sines]
            'inharmonicity': inharm,  # [B, T, n_sines]
        }


# ============================================================================
# DIFFERENTIABLE ADDITIVE SYNTHESIS
# ============================================================================

class DifferentiableSynth(nn.Module):
    """
    Differentiable additive synthesis.

    Converts operation parameters to audio.
    """

    def __init__(self, n_sines=32, sr=44100, hop_length=512):
        super().__init__()
        self.n_sines = n_sines
        self.sr = sr
        self.hop_length = hop_length

    def forward(self, ops, n_samples):
        """
        Args:
            ops: dict from OperationNet
            n_samples: output audio length

        Returns:
            audio: [B, n_samples]
        """
        f0 = ops['f0']  # [B, T]
        partial_weights = ops['partial_weights']  # [B, T, n_sines]
        decay = ops['decay']  # [B, T, n_sines]
        inharm = ops['inharmonicity']  # [B, T, n_sines]

        B, T, _ = partial_weights.shape
        device = f0.device

        # Generate harmonic ratios with inharmonicity
        ratios = torch.arange(1, self.n_sines + 1, device=device).float()  # [n_sines]
        ratios = ratios.view(1, 1, -1)  # [1, 1, n_sines]
        ratios = ratios + inharm  # Add inharmonicity

        # Frequencies: f0 * ratio
        freqs = f0.unsqueeze(-1) * ratios  # [B, T, n_sines]

        # Upsample to sample rate
        freqs_up = F.interpolate(
            freqs.permute(0, 2, 1),  # [B, n_sines, T]
            size=n_samples,
            mode='linear',
            align_corners=True
        ).permute(0, 2, 1)  # [B, n_samples, n_sines]

        weights_up = F.interpolate(
            partial_weights.permute(0, 2, 1),
            size=n_samples,
            mode='linear',
            align_corners=True
        ).permute(0, 2, 1)

        decay_up = F.interpolate(
            decay.permute(0, 2, 1),
            size=n_samples,
            mode='linear',
            align_corners=True
        ).permute(0, 2, 1)

        # Apply decay envelope
        t = torch.linspace(0, 1, n_samples, device=device).view(1, -1, 1)
        envelope = torch.exp(-decay_up * t * 10)  # Faster decay = lower envelope
        amps = weights_up * envelope

        # Phase accumulation (differentiable)
        dt = 1.0 / self.sr
        phase_inc = 2 * np.pi * freqs_up * dt
        phase = torch.cumsum(phase_inc, dim=1)

        # Synthesize
        sines = amps * torch.sin(phase)
        audio = sines.sum(dim=-1)  # [B, n_samples]

        # Soft clip normalization (differentiable)
        audio = torch.tanh(audio / (audio.abs().max(dim=-1, keepdim=True)[0] + 1e-8))

        return audio


# ============================================================================
# MEL SPECTROGRAM LOSS
# ============================================================================

class MelSpectrogramLoss(nn.Module):
    """Simple spectrogram loss."""

    def __init__(self, sr=44100, n_mels=64):
        super().__init__()
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = 1024
        self.hop = 256

        # Pre-compute mel filterbank
        import librosa
        fb = librosa.filters.mel(sr=sr, n_fft=self.n_fft, n_mels=n_mels, fmax=sr//2)
        self.register_buffer('mel_fb', torch.from_numpy(fb).float())

    def forward(self, pred_audio, target_audio):
        """Compute mel spectrogram loss."""
        pred_mel = self.compute_mel(pred_audio)
        target_mel = self.compute_mel(target_audio)

        # L1 loss on mel
        loss = F.l1_loss(pred_mel, target_mel)

        # Log mel loss
        pred_log = torch.log(pred_mel + 1e-5)
        target_log = torch.log(target_mel + 1e-5)
        loss += F.l1_loss(pred_log, target_log)

        return loss

    def compute_mel(self, audio):
        """Compute mel spectrogram."""
        window = torch.hann_window(self.n_fft, device=audio.device)
        stft = torch.stft(audio, self.n_fft, self.hop, window=window, return_complex=True)
        mag = stft.abs()  # [B, freq, time]

        # Apply mel filterbank
        mel = torch.einsum('mf,bft->bmt', self.mel_fb.to(audio.device), mag)
        return mel


# ============================================================================
# TRAINING
# ============================================================================

def load_dataset(manifest_path, n_samples=500, device='cuda'):
    """Load z latents for training."""
    with open(manifest_path, 'rb') as f:
        manifest = orjson.loads(f.read())

    latents = []
    for entry in manifest['entries'][:n_samples * 2]:
        try:
            data = torch.load(entry['path'], weights_only=True, map_location='cpu')
            lat_path = data.get('latent_path')
            if not lat_path or not os.path.exists(lat_path):
                continue

            lat_data = torch.load(lat_path, weights_only=True, map_location='cpu')
            z = lat_data.get('latents', lat_data)
            if z.dim() == 4:
                z = z.squeeze(0)

            # Take fixed length
            T = min(z.shape[-1], 32)
            z = z[..., :T]

            latents.append(z.to(device))
            if len(latents) >= n_samples:
                break
        except:
            continue

    return latents


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load DCAE (for target generation)
    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8",
        vocoder_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    )
    dcae.dcae.to(device).eval()
    dcae.vocoder.to(device).eval()

    # Load data
    print("Loading latents...")
    latents = load_dataset(
        '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json',
        n_samples=200,
        device=device
    )
    print(f"  Loaded {len(latents)} latents")

    # Models
    op_net = OperationNet(n_sines=32).to(device)
    synth = DifferentiableSynth(n_sines=32, sr=44100).to(device)
    mel_loss = MelSpectrogramLoss(sr=44100).to(device)

    optimizer = torch.optim.AdamW(op_net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    # Training loop
    n_epochs = 50
    batch_size = 4

    print("\nTraining...")
    print(f"  Samples: {len(latents)}, Batch size: {batch_size}")
    sys.stdout.flush()

    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0

        # Shuffle
        indices = torch.randperm(len(latents))

        for i in range(0, len(latents) - batch_size, batch_size):
            batch_z = torch.stack([latents[indices[j]] for j in range(i, i + batch_size)])

            # Flatten z
            B, C, H, T = batch_z.shape
            z_flat = batch_z.reshape(B, 128, T)

            # Generate target audio with DCAE
            with torch.no_grad():
                z_denorm = batch_z / dcae.scale_factor + dcae.shift_factor
                mel = dcae.dcae.decoder(z_denorm).mean(dim=1)
                mel_scaled = mel * 0.5 + 0.5
                mel_scaled = mel_scaled * (dcae.max_mel_value - dcae.min_mel_value) + dcae.min_mel_value
                target_audio = dcae.vocoder.decode(mel_scaled).squeeze(1)

            n_samples = target_audio.shape[-1]

            # Forward: z → ops → audio
            ops = op_net(z_flat)
            pred_audio = synth(ops, n_samples)

            # Match lengths
            min_len = min(pred_audio.shape[-1], target_audio.shape[-1])
            pred_audio = pred_audio[..., :min_len]
            target_audio = target_audio[..., :min_len]

            # Loss
            loss = mel_loss(pred_audio, target_audio)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(op_net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
        sys.stdout.flush()

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'op_net': op_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, f'/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/z_to_ops_{epoch+1}.pt')

            # Generate sample
            with torch.no_grad():
                sample_z = latents[0].unsqueeze(0)
                z_flat = sample_z.reshape(1, 128, -1)
                ops = op_net(z_flat)

                print(f"  Sample ops: f0={ops['f0'][0,0]:.1f}Hz, "
                      f"top_partial={ops['partial_weights'][0,0].argmax()}")

    # Save final
    os.makedirs('/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints', exist_ok=True)
    torch.save({
        'op_net': op_net.state_dict(),
    }, '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/z_to_ops_final.pt')
    print("\nSaved final model!")


if __name__ == "__main__":
    train()
