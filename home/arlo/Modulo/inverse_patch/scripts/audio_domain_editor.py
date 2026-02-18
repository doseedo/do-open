#!/usr/bin/env python3
"""
Audio Domain Editor: learn to apply z-space axis edits directly in STFT domain.

Instead of editing z then decoding through DCAE (which causes frame artifacts),
this model learns the input-dependent spectral transform that each axis produces.

At inference: original audio → STFT → model(stft, axis, delta) → ISTFT → edited audio.
No DCAE decode, no frame boundaries, original fidelity preserved.

Architecture:
  - Input: log-magnitude STFT [B, 1, F, T]
  - Conditioning: axis_idx (embedding) + delta (scalar) → FiLM parameters
  - Network: residual ConvNet predicting spectral delta
  - Output: edited log-magnitude STFT [B, 1, F, T]
  - Phase reused from original (valid for small-to-moderate edits)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "audio_domain_editor"
PAIRS_PATH = OUTPUT_DIR / "training_pairs.pt"
MODEL_PATH = OUTPUT_DIR / "audio_domain_editor.pt"

# STFT parameters
N_FFT = 2048
HOP_LENGTH = 512
SR = 44100
N_FREQ = N_FFT // 2 + 1  # 1025


# ============================================================
# Model
# ============================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: h = gamma * h + beta."""
    def __init__(self, cond_dim, n_channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, n_channels * 2)

    def forward(self, x, cond):
        # x: [B, C, F, T], cond: [B, cond_dim]
        params = self.proj(cond)  # [B, C*2]
        gamma, beta = params.chunk(2, dim=1)
        return gamma[:, :, None, None] * x + beta[:, :, None, None]


class ResBlock(nn.Module):
    """Residual block with FiLM conditioning."""
    def __init__(self, channels, cond_dim, kernel_size=(5, 3)):
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.film = FiLMLayer(cond_dim, channels)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x, cond):
        h = self.norm1(x)
        h = F.gelu(h)
        h = self.conv1(h)
        h = self.film(h, cond)
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.conv2(h)
        return x + h


class AudioDomainEditor(nn.Module):
    """
    Predicts spectral delta for a given axis edit.

    Input: log-magnitude STFT + axis_idx + delta
    Output: edited log-magnitude STFT

    The model predicts a residual: edited = original + predicted_delta.
    Last conv is zero-initialized so the model starts as identity.
    """
    def __init__(self, n_freq=N_FREQ, n_axes=6, hidden=48, cond_dim=64, n_res_blocks=6):
        super().__init__()

        # Conditioning network: axis embedding + delta → conditioning vector
        self.axis_embed = nn.Embedding(n_axes, 32)
        self.cond_net = nn.Sequential(
            nn.Linear(33, cond_dim),  # 32 axis embed + 1 delta
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Input projection
        self.input_conv = nn.Conv2d(1, hidden, (7, 3), padding=(3, 1))

        # Residual blocks with FiLM conditioning
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden, cond_dim) for _ in range(n_res_blocks)
        ])

        # Output projection (zero-initialized for identity start)
        self.output_conv = nn.Conv2d(hidden, 1, (7, 3), padding=(3, 1))
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, log_mag, axis_idx, delta):
        """
        Args:
            log_mag: [B, 1, F, T] log-magnitude STFT
            axis_idx: [B] long tensor — which axis
            delta: [B] float tensor — edit amount (in σ units)
        Returns:
            edited_log_mag: [B, 1, F, T]
        """
        # Conditioning
        ax_emb = self.axis_embed(axis_idx)  # [B, 32]
        cond = self.cond_net(
            torch.cat([ax_emb, delta.unsqueeze(1)], dim=1)
        )  # [B, cond_dim]

        # Process
        h = self.input_conv(log_mag)  # [B, hidden, F, T]

        for block in self.res_blocks:
            h = block(h, cond)

        # Predict spectral delta
        spectral_delta = self.output_conv(h)  # [B, 1, F, T]

        return log_mag + spectral_delta


# ============================================================
# Dataset
# ============================================================

# ============================================================
# Training
# ============================================================

def precompute_stfts(pairs_path, device):
    """Load audio pairs and compute all STFTs on GPU upfront."""
    print(f"Loading training pairs from {pairs_path}...")
    data = torch.load(pairs_path, weights_only=False, map_location='cpu')

    original_audio = data['original_audio']   # [N_samples, L]
    edited_audio = data['edited_audio']        # [N_edits, L]
    sample_idx = data['edit_sample_idx']       # [N_edits]
    axis_idx = data['edit_axis_idx']            # [N_edits]
    delta = data['edit_delta']                  # [N_edits]

    n_samples = len(original_audio)
    n_edits = len(edited_audio)
    print(f"  {n_samples} samples, {n_edits} edits")

    window = torch.hann_window(N_FFT, device=device)

    def batch_stft(audio_batch, batch_size=32):
        """Compute STFTs in batches on GPU."""
        all_stfts = []
        for i in range(0, len(audio_batch), batch_size):
            batch = audio_batch[i:i + batch_size].to(device)
            stft = torch.stft(
                batch, n_fft=N_FFT, hop_length=HOP_LENGTH,
                window=window, return_complex=True,
            )
            log_mag = torch.log1p(stft.abs())  # [B, F, T]
            all_stfts.append(log_mag.cpu())
        return torch.cat(all_stfts, dim=0)

    print("  Computing original STFTs...")
    orig_stfts = batch_stft(original_audio)  # [N_samples, F, T]
    print(f"    Shape: {orig_stfts.shape}")

    print("  Computing edited STFTs...")
    edit_stfts = batch_stft(edited_audio)    # [N_edits, F, T]
    print(f"    Shape: {edit_stfts.shape}")

    # Expand original STFTs to match edits (index by sample_idx)
    orig_stfts_expanded = orig_stfts[sample_idx]  # [N_edits, F, T]

    return {
        'orig_stfts': orig_stfts_expanded,  # [N_edits, F, T]
        'edit_stfts': edit_stfts,            # [N_edits, F, T]
        'axis_idx': axis_idx,                 # [N_edits]
        'delta': delta,                       # [N_edits]
        'n_axes': int(axis_idx.max().item()) + 1,
        'stft_T': orig_stfts.shape[-1],
    }


def train(epochs=20, batch_size=32, lr=3e-4, time_crop=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = precompute_stfts(PAIRS_PATH, device)
    n_edits = len(data['edit_stfts'])
    n_axes = data['n_axes']
    stft_T = data['stft_T']

    # All data as tensors (stay on CPU, move batches to GPU)
    orig_stfts = data['orig_stfts']    # [N, F, T]
    edit_stfts = data['edit_stfts']    # [N, F, T]
    axis_idx = data['axis_idx']         # [N]
    deltas = data['delta']              # [N]

    model = AudioDomainEditor(n_freq=N_FREQ, n_axes=n_axes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(f"Training: {epochs} epochs, batch={batch_size}, lr={lr}, time_crop={time_crop}")
    print(f"Device: {device}, edits: {n_edits}, STFT: [{N_FREQ}, {stft_T}]")
    print()

    best_loss = float('inf')
    indices = torch.arange(n_edits)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        # Shuffle
        perm = torch.randperm(n_edits)

        for batch_start in range(0, n_edits, batch_size):
            batch_idx = perm[batch_start:batch_start + batch_size]

            # Random time crop
            if time_crop and stft_T > time_crop:
                t_start = torch.randint(0, stft_T - time_crop, (1,)).item()
                t_end = t_start + time_crop
            else:
                t_start, t_end = 0, stft_T

            orig = orig_stfts[batch_idx, :, t_start:t_end].unsqueeze(1).to(device)  # [B,1,F,T]
            target = edit_stfts[batch_idx, :, t_start:t_end].unsqueeze(1).to(device)
            ax = axis_idx[batch_idx].to(device)
            d = deltas[batch_idx].to(device)

            pred = model(orig, ax, d)

            # MSE on full STFT + emphasis on the delta
            loss = F.mse_loss(pred, target)
            delta_loss = F.mse_loss(pred - orig, target - orig)
            total_loss_val = loss + 0.5 * delta_loss

            optimizer.zero_grad()
            total_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model': model.state_dict(),
                'n_axes': n_axes,
                'n_freq': N_FREQ,
                'best_loss': best_loss,
                'epoch': epoch + 1,
                'stft_config': {
                    'n_fft': N_FFT,
                    'hop_length': HOP_LENGTH,
                    'sr': SR,
                },
            }, MODEL_PATH)

    print(f"\n  Best loss: {best_loss:.6f}")
    print(f"  Model saved to {MODEL_PATH}")

    # Validate: check identity (delta=0 should produce no change)
    model.eval()
    with torch.no_grad():
        orig = orig_stfts[0:1, :, :time_crop].unsqueeze(1).to(device)
        ax = torch.zeros(1, dtype=torch.long, device=device)
        d = torch.zeros(1, device=device)
        pred = model(orig, ax, d)
        identity_error = F.mse_loss(pred, orig).item()
        print(f"  Identity check (delta=0): MSE={identity_error:.8f}")


# ============================================================
# Inference
# ============================================================

class AudioDomainEditorInference:
    """Inference wrapper: apply axis edits directly to audio."""

    def __init__(self, model_path=MODEL_PATH, device='cuda'):
        ckpt = torch.load(model_path, weights_only=False, map_location='cpu')
        self.model = AudioDomainEditor(
            n_freq=ckpt['n_freq'],
            n_axes=ckpt['n_axes'],
        )
        self.model.load_state_dict(ckpt['model'])
        self.model.to(device).eval()
        self.device = device
        self.window = torch.hann_window(N_FFT).to(device)
        self.stft_config = ckpt['stft_config']

    @torch.no_grad()
    def edit(self, audio, axis_idx, delta):
        """
        Apply axis edit to audio.

        Args:
            audio: [L] float tensor (mono, 44100 Hz)
            axis_idx: int — which axis (0-5)
            delta: float — edit amount in σ units

        Returns:
            edited_audio: [L] float tensor
        """
        audio = audio.to(self.device)

        # STFT
        stft = torch.stft(
            audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
            window=self.window, return_complex=True,
        )  # [F, T]
        mag = stft.abs()
        phase = stft.angle()
        log_mag = torch.log1p(mag)

        # Model prediction
        axis_t = torch.tensor([axis_idx], dtype=torch.long, device=self.device)
        delta_t = torch.tensor([delta], dtype=torch.float32, device=self.device)
        edited_log_mag = self.model(
            log_mag.unsqueeze(0).unsqueeze(0),  # [1, 1, F, T]
            axis_t, delta_t,
        ).squeeze(0).squeeze(0)  # [F, T]

        # Reconstruct: use predicted magnitude + original phase
        edited_mag = torch.expm1(edited_log_mag).clamp(min=0)
        edited_stft = torch.polar(edited_mag, phase)

        # ISTFT
        edited_audio = torch.istft(
            edited_stft, n_fft=N_FFT, hop_length=HOP_LENGTH,
            window=self.window, length=len(audio),
        )

        # Normalize
        peak = edited_audio.abs().max()
        if peak > 0.95:
            edited_audio = edited_audio * (0.95 / peak)

        return edited_audio.cpu()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    if not PAIRS_PATH.exists():
        print(f"Training pairs not found at {PAIRS_PATH}")
        print("Run generate_axis_pairs.py first!")
        sys.exit(1)

    print("=" * 60)
    print("AUDIO DOMAIN EDITOR — TRAINING")
    print("=" * 60)
    train()
