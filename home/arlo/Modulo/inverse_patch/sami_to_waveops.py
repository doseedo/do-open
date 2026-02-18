#!/usr/bin/env python3
"""
SAMI → WaveOps Bridge

Uses discovered SAMI structure to map z_sami → explicit synthesis parameters.

Discovered dims from SAMI training:
  - Coarse (stable across noise): [189, 83, 168, 116, 85, 233, 213, 209, 161, 250]
  - Fine (noise-sensitive): [251, 248, 81, 174, 91, 30, 104, 221, 119, 198]
  - Spectral control: [16, 17, 24, 31, 30]
  - Energy control: [31, 17, 13, 41, 63]
  - Spread control: [31, 17, 30, 37, 16]
  - Shared timbre core: [16, 17, 31]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


# ============================================================
# DISCOVERED SAMI STRUCTURE
# ============================================================

SAMI_STRUCTURE = {
    # Coarse dims: stable across noise levels → fundamental properties
    'coarse': [189, 83, 168, 116, 85, 233, 213, 209, 161, 250],

    # Fine dims: noise-sensitive → texture, transients
    'fine': [251, 248, 81, 174, 91, 30, 104, 221, 119, 198],

    # Spectral dims: control brightness/pitch center
    'spectral': [16, 17, 24, 31, 30],

    # Energy dims: control amplitude (though mostly 0 effect in VAE)
    'energy': [31, 17, 13, 41, 63],

    # Spread dims: control spectral bandwidth
    'spread': [31, 17, 30, 37, 16],

    # Shared core: dims that affect multiple properties
    'timbre_core': [16, 17, 31],
}


# ============================================================
# WAVEOPS PARAMETER HEADS
# ============================================================

class SAMItoWaveOps(nn.Module):
    """
    Map z_sami → WaveOps synthesis parameters.

    Uses discovered SAMI structure to extract:
      - f0: fundamental frequency (from coarse dims)
      - harmonics: harmonic amplitudes (from spectral dims)
      - noise_envelope: filtered noise (from fine dims)
      - formants: resonance peaks (from spread dims)
    """

    def __init__(
        self,
        sami_dim: int = 256,
        n_harmonics: int = 64,
        n_noise_bands: int = 32,
        n_formants: int = 4,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.sami_dim = sami_dim
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.n_formants = n_formants
        self.sample_rate = sample_rate

        # Store discovered dims as buffers
        self.register_buffer('coarse_idx', torch.tensor(SAMI_STRUCTURE['coarse']))
        self.register_buffer('fine_idx', torch.tensor(SAMI_STRUCTURE['fine']))
        self.register_buffer('spectral_idx', torch.tensor(SAMI_STRUCTURE['spectral']))
        self.register_buffer('spread_idx', torch.tensor(SAMI_STRUCTURE['spread']))
        self.register_buffer('timbre_idx', torch.tensor(SAMI_STRUCTURE['timbre_core']))

        n_coarse = len(SAMI_STRUCTURE['coarse'])
        n_fine = len(SAMI_STRUCTURE['fine'])
        n_spectral = len(SAMI_STRUCTURE['spectral'])
        n_spread = len(SAMI_STRUCTURE['spread'])
        n_timbre = len(SAMI_STRUCTURE['timbre_core'])

        # F0 head: coarse dims → fundamental frequency
        self.f0_head = nn.Sequential(
            nn.Linear(n_coarse, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Harmonics head: spectral + timbre dims → harmonic amplitudes
        self.harmonics_head = nn.Sequential(
            nn.Linear(n_spectral + n_timbre, 128),
            nn.GELU(),
            nn.Linear(128, n_harmonics),
        )

        # Noise head: fine dims → noise envelope per band
        self.noise_head = nn.Sequential(
            nn.Linear(n_fine, 64),
            nn.GELU(),
            nn.Linear(64, n_noise_bands),
        )

        # Formant head: spread dims → formant frequencies and bandwidths
        self.formant_head = nn.Sequential(
            nn.Linear(n_spread, 64),
            nn.GELU(),
            nn.Linear(64, n_formants * 2),  # freq + bandwidth per formant
        )

        # Global amplitude: from full z_sami (since energy dims had weak effect)
        self.amp_head = nn.Sequential(
            nn.Linear(sami_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, z_sami: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Map z_sami to WaveOps parameters.

        Args:
            z_sami: [B, sami_dim] or [B, T, sami_dim]

        Returns:
            dict with synthesis parameters
        """
        # Handle temporal dimension
        has_time = z_sami.dim() == 3
        if has_time:
            B, T, D = z_sami.shape
            z_sami = z_sami.reshape(B * T, D)
        else:
            B = z_sami.shape[0]
            T = 1

        # Extract dim groups
        z_coarse = z_sami[:, self.coarse_idx]
        z_fine = z_sami[:, self.fine_idx]
        z_spectral = z_sami[:, self.spectral_idx]
        z_spread = z_sami[:, self.spread_idx]
        z_timbre = z_sami[:, self.timbre_idx]

        # Compute parameters

        # F0: 20-2000 Hz range
        f0_logits = self.f0_head(z_coarse)
        f0 = 20 + torch.sigmoid(f0_logits) * 1980  # [B*T, 1]

        # Harmonics: softmax for relative amplitudes, then scale
        harmonics_logits = self.harmonics_head(torch.cat([z_spectral, z_timbre], dim=-1))
        harmonics = F.softmax(harmonics_logits, dim=-1)  # [B*T, n_harmonics]

        # Noise envelope per band
        noise_logits = self.noise_head(z_fine)
        noise_env = torch.sigmoid(noise_logits)  # [B*T, n_noise_bands]

        # Formants: frequency (200-5000 Hz) and bandwidth (50-500 Hz)
        formant_raw = self.formant_head(z_spread)
        formant_freqs = 200 + torch.sigmoid(formant_raw[:, :self.n_formants]) * 4800
        formant_bws = 50 + torch.sigmoid(formant_raw[:, self.n_formants:]) * 450

        # Global amplitude
        amp_logits = self.amp_head(z_sami)
        amplitude = torch.sigmoid(amp_logits)  # [B*T, 1]

        # Reshape back if temporal
        if has_time:
            f0 = f0.reshape(B, T, 1)
            harmonics = harmonics.reshape(B, T, self.n_harmonics)
            noise_env = noise_env.reshape(B, T, self.n_noise_bands)
            formant_freqs = formant_freqs.reshape(B, T, self.n_formants)
            formant_bws = formant_bws.reshape(B, T, self.n_formants)
            amplitude = amplitude.reshape(B, T, 1)

        return {
            'f0': f0,
            'harmonics': harmonics,
            'noise_envelope': noise_env,
            'formant_freqs': formant_freqs,
            'formant_bandwidths': formant_bws,
            'amplitude': amplitude,
        }


# ============================================================
# WAVEOPS SYNTHESIZER (Differentiable)
# ============================================================

class WaveOpsSynth(nn.Module):
    """
    Differentiable WaveOps-style synthesizer.

    Generates audio from:
      - f0 + harmonics → additive synthesis
      - noise_envelope → filtered noise
      - formants → resonant filtering
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_size: int = 256,
        n_harmonics: int = 64,
        n_noise_bands: int = 32,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands

        # Harmonic indices
        self.register_buffer('harmonic_idx', torch.arange(1, n_harmonics + 1).float())

        # Noise band frequencies (mel-spaced)
        mel_low = 2595 * np.log10(1 + 20 / 700)
        mel_high = 2595 * np.log10(1 + (sample_rate/2) / 700)
        mel_points = np.linspace(mel_low, mel_high, n_noise_bands + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        self.register_buffer('noise_freqs', torch.tensor(hz_points[1:-1]).float())

    def forward(
        self,
        f0: torch.Tensor,
        harmonics: torch.Tensor,
        noise_envelope: torch.Tensor,
        amplitude: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """
        Synthesize audio from parameters.

        Args:
            f0: [B, T, 1] fundamental frequency in Hz
            harmonics: [B, T, n_harmonics] harmonic amplitudes
            noise_envelope: [B, T, n_noise_bands] noise per band
            amplitude: [B, T, 1] global amplitude
            n_samples: number of output samples

        Returns:
            audio: [B, n_samples]
        """
        B, T, _ = f0.shape
        device = f0.device

        # Upsample parameters to audio rate
        f0_up = F.interpolate(f0.transpose(1, 2), size=n_samples, mode='linear').transpose(1, 2)
        harm_up = F.interpolate(harmonics.transpose(1, 2), size=n_samples, mode='linear').transpose(1, 2)
        noise_up = F.interpolate(noise_envelope.transpose(1, 2), size=n_samples, mode='linear').transpose(1, 2)
        amp_up = F.interpolate(amplitude.transpose(1, 2), size=n_samples, mode='linear').transpose(1, 2)

        # Generate time axis
        t = torch.arange(n_samples, device=device).float() / self.sample_rate
        t = t.unsqueeze(0).unsqueeze(-1)  # [1, n_samples, 1]

        # Additive synthesis: sum of harmonics
        # phase = 2π * f0 * k * t for harmonic k
        freqs = f0_up * self.harmonic_idx  # [B, n_samples, n_harmonics]
        phases = 2 * np.pi * freqs * t  # [B, n_samples, n_harmonics]

        # Anti-aliasing: zero harmonics above Nyquist
        nyquist = self.sample_rate / 2
        harm_up = harm_up * (freqs < nyquist).float()

        # Sum harmonics
        harmonic_audio = (torch.sin(phases) * harm_up).sum(dim=-1)  # [B, n_samples]

        # Filtered noise
        noise = torch.randn(B, n_samples, device=device)
        # Simple bandpass approximation using noise envelope
        # (Full implementation would use proper FIR filters)
        noise_audio = noise * noise_up.mean(dim=-1)  # Simplified

        # Combine and apply amplitude
        audio = (harmonic_audio + 0.1 * noise_audio) * amp_up.squeeze(-1)

        # Normalize
        audio = audio / (audio.abs().max(dim=-1, keepdim=True)[0] + 1e-8)

        return audio


# ============================================================
# FULL SAMI → AUDIO PIPELINE
# ============================================================

class SAMIAudioPipeline(nn.Module):
    """
    Complete pipeline: z_sami → WaveOps params → audio

    Can be trained end-to-end with audio reconstruction loss.
    """

    def __init__(
        self,
        sami_dim: int = 256,
        sample_rate: int = 44100,
        hop_size: int = 256,
        n_harmonics: int = 64,
        n_noise_bands: int = 32,
    ):
        super().__init__()

        self.param_mapper = SAMItoWaveOps(
            sami_dim=sami_dim,
            n_harmonics=n_harmonics,
            n_noise_bands=n_noise_bands,
        )

        self.synth = WaveOpsSynth(
            sample_rate=sample_rate,
            hop_size=hop_size,
            n_harmonics=n_harmonics,
            n_noise_bands=n_noise_bands,
        )

        self.sample_rate = sample_rate

    def forward(
        self,
        z_sami: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        z_sami → params → audio

        Args:
            z_sami: [B, T, sami_dim] SAMI latents
            n_samples: output audio length (default: T * 256)

        Returns:
            dict with 'audio', 'params'
        """
        # Get synthesis parameters
        params = self.param_mapper(z_sami)

        # Default audio length
        if n_samples is None:
            if z_sami.dim() == 3:
                n_samples = z_sami.shape[1] * 256
            else:
                n_samples = 8192

        # Ensure temporal dimension
        if z_sami.dim() == 2:
            for k in params:
                params[k] = params[k].unsqueeze(1)

        # Synthesize
        audio = self.synth(
            f0=params['f0'],
            harmonics=params['harmonics'],
            noise_envelope=params['noise_envelope'],
            amplitude=params['amplitude'],
            n_samples=n_samples,
        )

        return {
            'audio': audio,
            'params': params,
        }


# ============================================================
# TESTING
# ============================================================

def test_pipeline():
    """Test the SAMI → WaveOps pipeline."""

    print("Testing SAMI → WaveOps pipeline...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create pipeline
    pipeline = SAMIAudioPipeline(sami_dim=256).to(device)
    print(f"  Pipeline params: {sum(p.numel() for p in pipeline.parameters()):,}")

    # Test with random z_sami
    B, T = 2, 22  # Match DCAE temporal resolution
    z_sami = torch.randn(B, T, 256, device=device)

    # Forward pass
    out = pipeline(z_sami, n_samples=44100)

    print(f"\n  Input z_sami: {z_sami.shape}")
    print(f"  Output audio: {out['audio'].shape}")
    print(f"  Params:")
    for k, v in out['params'].items():
        print(f"    {k}: {v.shape}")

    # Check parameter ranges
    params = out['params']
    print(f"\n  Parameter ranges:")
    print(f"    f0: {params['f0'].min():.1f} - {params['f0'].max():.1f} Hz")
    print(f"    harmonics sum: {params['harmonics'].sum(dim=-1).mean():.4f}")
    print(f"    noise_envelope: {params['noise_envelope'].min():.4f} - {params['noise_envelope'].max():.4f}")
    print(f"    amplitude: {params['amplitude'].min():.4f} - {params['amplitude'].max():.4f}")

    print("\n  Pipeline test passed!")

    return pipeline


if __name__ == "__main__":
    test_pipeline()
