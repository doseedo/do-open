#!/usr/bin/env python3
"""
Sparse Vocoder: Replace neural vocoder with explicit additive synthesis.

The neural vocoder does mel → audio, but essentially:
- Each mel bin maps to a center frequency
- Energy in bin → amplitude of sine at that frequency

We make this explicit:
  mel[128, T] → for each frame, sum sines at mel bin frequencies

This gives us:
1. Full interpretability (we know exactly what frequencies are being synthesized)
2. Differentiable synthesis (can backprop through it)
3. Direct z → sines path when combined with mel mapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
import sys
import os

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE


# ============================================================================
# MEL BIN TO HZ MAPPING
# ============================================================================

def mel_to_hz(mel):
    """Convert mel scale to Hz."""
    return 700 * (10 ** (mel / 2595) - 1)

def hz_to_mel(hz):
    """Convert Hz to mel scale."""
    return 2595 * np.log10(1 + hz / 700)

def get_mel_frequencies(n_mels=128, f_min=40, f_max=16000):
    """Get center frequencies for each mel bin."""
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mels = np.linspace(mel_min, mel_max, n_mels)
    return mel_to_hz(mels)

# Precompute mel bin center frequencies
MEL_FREQS = torch.tensor(get_mel_frequencies(128, 40, 16000), dtype=torch.float32)


# ============================================================================
# SPARSE VOCODER V1: Simple additive synthesis
# ============================================================================

class SparseVocoderV1(nn.Module):
    """
    Simple sparse vocoder: sum of sines at mel bin frequencies.

    mel[B, 128, T] → audio[B, samples]

    For each frame:
      audio += sum over bins: amplitude[bin] * sin(2π * freq[bin] * t + phase[bin])
    """

    def __init__(self, sample_rate=44100, hop_length=512, n_mels=128):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Register mel frequencies as buffer
        self.register_buffer('mel_freqs', MEL_FREQS.clone())

    def forward(self, mel, top_k=64):
        """
        Synthesize audio from mel spectrogram.

        Args:
            mel: [B, 128, T] mel spectrogram (scaled to vocoder range)
            top_k: number of bins to use per frame (sparsity)

        Returns:
            audio: [B, T * hop_length]
        """
        B, n_mels, T = mel.shape
        device = mel.device

        # Output audio length
        n_samples = T * self.hop_length
        audio = torch.zeros(B, n_samples, device=device)

        # Phase accumulator for each mel bin (continuous phase)
        phase = torch.zeros(B, n_mels, device=device)

        # Phase increment per sample for each frequency
        # Δφ = 2π * f / sr
        freq_hz = self.mel_freqs.to(device)
        phase_inc = 2 * np.pi * freq_hz / self.sample_rate  # [128]

        for t in range(T):
            frame_mel = mel[:, :, t]  # [B, 128]

            # Convert mel energy to amplitude
            # Mel is in dB-like scale, convert to linear
            # The neural vocoder mel range is [-11, 3], roughly dB/10
            amp = torch.clamp(frame_mel, min=-11, max=3)
            amp = 10 ** (amp / 10)  # Convert from dB-ish to linear
            amp = amp / amp.max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # Normalize

            # Optional: only keep top-k bins (sparsity)
            if top_k < n_mels:
                topk_vals, topk_idx = amp.topk(top_k, dim=-1)
                amp_sparse = torch.zeros_like(amp)
                amp_sparse.scatter_(-1, topk_idx, topk_vals)
                amp = amp_sparse

            # Generate samples for this frame
            t_samples = torch.arange(self.hop_length, device=device).float()  # [hop]

            # Phase for each sample: phase[bin] + t * phase_inc[bin]
            # Shape: [B, 128, hop]
            sample_phase = phase.unsqueeze(-1) + t_samples.unsqueeze(0).unsqueeze(0) * phase_inc.unsqueeze(0).unsqueeze(-1)

            # Synthesize: sum of sines
            # [B, 128, hop] * [B, 128, 1] → [B, 128, hop] → sum → [B, hop]
            frame_audio = (torch.sin(sample_phase) * amp.unsqueeze(-1)).sum(dim=1)

            # Add to output with overlap
            start = t * self.hop_length
            end = start + self.hop_length
            audio[:, start:end] = frame_audio

            # Update phase (wrap to [0, 2π])
            phase = (phase + self.hop_length * phase_inc.unsqueeze(0)) % (2 * np.pi)

        # Normalize output
        audio = audio / (audio.abs().max(dim=-1, keepdim=True).values + 1e-8) * 0.9

        return audio


# ============================================================================
# SPARSE VOCODER V2: With learned amplitude mapping
# ============================================================================

class SparseVocoderV2(nn.Module):
    """
    Sparse vocoder with learned mel → amplitude mapping.

    The mel values need proper scaling to amplitude.
    Learn this mapping while keeping synthesis explicit.
    """

    def __init__(self, sample_rate=44100, hop_length=512, n_mels=128):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.register_buffer('mel_freqs', MEL_FREQS.clone())

        # Learnable amplitude scaling per bin
        self.amp_scale = nn.Parameter(torch.ones(n_mels))
        self.amp_bias = nn.Parameter(torch.zeros(n_mels))

        # Learnable frequency correction per bin (small adjustments)
        self.freq_correction = nn.Parameter(torch.zeros(n_mels))

    def forward(self, mel, top_k=64):
        B, n_mels, T = mel.shape
        device = mel.device

        n_samples = T * self.hop_length
        audio = torch.zeros(B, n_samples, device=device)

        # Apply learned frequency correction
        freq_hz = self.mel_freqs.to(device) * (1 + 0.1 * torch.tanh(self.freq_correction))
        phase_inc = 2 * np.pi * freq_hz / self.sample_rate

        phase = torch.zeros(B, n_mels, device=device)

        for t in range(T):
            frame_mel = mel[:, :, t]

            # Learned amplitude mapping
            amp = torch.sigmoid(self.amp_scale * frame_mel + self.amp_bias)

            # Sparsity
            if top_k < n_mels:
                topk_vals, topk_idx = amp.topk(top_k, dim=-1)
                amp_sparse = torch.zeros_like(amp)
                amp_sparse.scatter_(-1, topk_idx, topk_vals)
                amp = amp_sparse

            t_samples = torch.arange(self.hop_length, device=device).float()
            sample_phase = phase.unsqueeze(-1) + t_samples.unsqueeze(0).unsqueeze(0) * phase_inc.unsqueeze(0).unsqueeze(-1)
            frame_audio = (torch.sin(sample_phase) * amp.unsqueeze(-1)).sum(dim=1)

            start = t * self.hop_length
            audio[:, start:end] = frame_audio

            phase = (phase + self.hop_length * phase_inc.unsqueeze(0)) % (2 * np.pi)

        audio = audio / (audio.abs().max(dim=-1, keepdim=True).values + 1e-8) * 0.9
        return audio


# ============================================================================
# SPARSE VOCODER V3: Vectorized (faster)
# ============================================================================

class SparseVocoderV3(nn.Module):
    """
    Fully vectorized sparse vocoder for speed.
    """

    def __init__(self, sample_rate=44100, hop_length=512, n_mels=128):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.register_buffer('mel_freqs', MEL_FREQS.clone())

    def forward(self, mel, top_k=None):
        """
        Vectorized synthesis.

        mel: [B, 128, T] in vocoder range [-11, 3]
        """
        B, n_mels, T = mel.shape
        device = mel.device

        n_samples = T * self.hop_length

        # Get frequencies
        freq_hz = self.mel_freqs.to(device)  # [128]

        # Convert mel to amplitude
        # Mel is roughly in dB/10 range [-11, 3] → linear
        amp = 10 ** ((mel - mel.min()) / 20)  # Simple dB to linear
        amp = amp / amp.max()  # Normalize to [0, 1]

        # Interpolate amplitude to sample rate
        # [B, 128, T] → [B, 128, n_samples]
        amp_interp = F.interpolate(amp, size=n_samples, mode='linear', align_corners=False)

        # Generate time axis
        t = torch.arange(n_samples, device=device).float() / self.sample_rate  # [n_samples]

        # Phase: 2π * f * t for each frequency
        # [128] x [n_samples] → [128, n_samples]
        phase = 2 * np.pi * freq_hz.unsqueeze(-1) * t.unsqueeze(0)

        # Synthesize: sum over frequencies
        # [B, 128, n_samples] * sin([128, n_samples]) → [B, n_samples]
        audio = (amp_interp * torch.sin(phase.unsqueeze(0))).sum(dim=1)

        # Normalize
        audio = audio / (audio.abs().max(dim=-1, keepdim=True).values + 1e-8) * 0.9

        return audio


# ============================================================================
# TEST: Compare neural vocoder vs sparse vocoder
# ============================================================================

def test_sparse_vocoder():
    """Compare neural vocoder output with sparse vocoder."""
    device = 'cuda'

    print("Loading DCAE + Neural Vocoder...")
    DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
    VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    dcae = MusicDCAE(dcae_checkpoint_path=DCAE_PATH, vocoder_checkpoint_path=VOCODER_PATH)
    dcae.dcae.to(device).eval()
    dcae.vocoder.to(device).eval()

    # Load a sample latent
    print("\nLoading sample...")
    import orjson
    with open('/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json', 'rb') as f:
        manifest = orjson.loads(f.read())

    # Get first non-drum sample
    sample_path = None
    for entry in manifest['entries']:
        path = entry['path']
        if not any(kw in path.lower() for kw in ['drum', 'kick', 'snare', 'hat', 'perc']):
            sample_path = path
            break

    data = torch.load(sample_path, weights_only=False)
    latent_path = data['latent_path']
    z = torch.load(latent_path, weights_only=True)
    if isinstance(z, dict):
        z = z['latents']
    if z.dim() == 4:
        z = z.squeeze(0)
    z = z[..., :32].unsqueeze(0).to(device)  # Limit length

    print(f"  z shape: {z.shape}")

    # Get mel from decoder
    print("\nDecoding to mel...")
    with torch.no_grad():
        z_denorm = z / dcae.scale_factor + dcae.shift_factor
        mel = dcae.dcae.decoder(z_denorm).mean(dim=1)  # [B, 128, T]

    print(f"  mel shape: {mel.shape}, range: [{mel.min():.2f}, {mel.max():.2f}]")

    # Scale mel for vocoder
    mel_scaled = mel * 0.5 + 0.5
    mel_scaled = mel_scaled * (dcae.max_mel_value - dcae.min_mel_value) + dcae.min_mel_value

    print(f"  mel_scaled range: [{mel_scaled.min():.2f}, {mel_scaled.max():.2f}]")

    # Neural vocoder
    print("\nNeural vocoder...")
    with torch.no_grad():
        audio_neural = dcae.vocoder.decode(mel_scaled).squeeze()
    print(f"  audio shape: {audio_neural.shape}")

    # Sparse vocoder V3 (vectorized)
    print("\nSparse vocoder V3...")
    sparse_voc = SparseVocoderV3(sample_rate=44100, hop_length=512).to(device)
    with torch.no_grad():
        audio_sparse = sparse_voc(mel_scaled).squeeze()
    print(f"  audio shape: {audio_sparse.shape}")

    # Match lengths
    min_len = min(audio_neural.shape[-1], audio_sparse.shape[-1])
    audio_neural = audio_neural[:min_len]
    audio_sparse = audio_sparse[:min_len]

    # Compute metrics
    print("\nComparing outputs...")

    # Spectral convergence
    window = torch.hann_window(2048, device=device)
    spec_neural = torch.stft(audio_neural, 2048, 512, window=window, return_complex=True).abs()
    spec_sparse = torch.stft(audio_sparse, 2048, 512, window=window, return_complex=True).abs()

    sc = torch.norm(spec_neural - spec_sparse, p='fro') / (torch.norm(spec_neural, p='fro') + 1e-8)
    print(f"  Spectral Convergence: {sc.item():.4f}")

    # Correlation
    corr = torch.corrcoef(torch.stack([audio_neural.flatten(), audio_sparse.flatten()]))[0, 1]
    print(f"  Waveform Correlation: {corr.item():.4f}")

    # Save outputs
    output_dir = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/audio_comparison'
    os.makedirs(output_dir, exist_ok=True)

    audio_neural_norm = audio_neural / (audio_neural.abs().max() + 1e-8) * 0.9
    audio_sparse_norm = audio_sparse / (audio_sparse.abs().max() + 1e-8) * 0.9

    torchaudio.save(f'{output_dir}/vocoder_neural.wav', audio_neural_norm.unsqueeze(0).cpu(), 44100)
    torchaudio.save(f'{output_dir}/vocoder_sparse_v3.wav', audio_sparse_norm.unsqueeze(0).cpu(), 44100)

    print(f"\nSaved to {output_dir}/")
    print("  vocoder_neural.wav - Neural vocoder output")
    print("  vocoder_sparse_v3.wav - Sparse (additive) vocoder output")

    # Also test V1 (frame-by-frame)
    print("\nSparse vocoder V1 (frame-by-frame)...")
    sparse_voc1 = SparseVocoderV1(sample_rate=44100, hop_length=512).to(device)
    with torch.no_grad():
        audio_sparse1 = sparse_voc1(mel_scaled, top_k=32).squeeze()

    audio_sparse1_norm = audio_sparse1 / (audio_sparse1.abs().max() + 1e-8) * 0.9
    torchaudio.save(f'{output_dir}/vocoder_sparse_v1_top32.wav', audio_sparse1_norm.unsqueeze(0).cpu(), 44100)
    print("  vocoder_sparse_v1_top32.wav - Sparse with top-32 bins per frame")


if __name__ == "__main__":
    test_sparse_vocoder()
