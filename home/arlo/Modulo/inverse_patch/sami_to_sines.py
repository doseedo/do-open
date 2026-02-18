#!/usr/bin/env python3
"""
SAMI → Sines (True WaveOps)

Atomic unit: sine wave
Discovery: network learns what combinations reconstruct audio
No prescription: no "harmonics", no "formants", no "noise bands"

The network decides what the sines become through reconstruction loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class SAMItoSines(nn.Module):
    """
    z_sami → (frequency, amplitude, phase) for N sine waves.

    No semantic prescription. Just parameters.
    Structure emerges from reconstruction.
    """

    def __init__(self, sami_dim: int = 128, n_sines: int = 128, sample_rate: int = 44100):
        """
        Args:
            sami_dim: Input dimension (128 = 8*16 from DCAE)
            n_sines: Number of sine oscillators
            sample_rate: Audio sample rate
        """
        super().__init__()
        self.sami_dim = sami_dim
        self.n_sines = n_sines
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2

        # Single projection: z_sami → sine params
        # freq, amp, phase per sine = 3 * n_sines
        hidden_dim = max(256, n_sines * 2)
        self.to_params = nn.Sequential(
            nn.Linear(sami_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_sines * 3),
        )

    def forward(self, z_sami: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_sami: [B, sami_dim] or [B, T, sami_dim]

        Returns:
            freqs: [B, n_sines] or [B, T, n_sines] in Hz
            amps: [B, n_sines] or [B, T, n_sines] in [0, 1]
            phases: [B, n_sines] or [B, T, n_sines] in [0, 2π]
        """
        params = self.to_params(z_sami)

        # Split into freq, amp, phase
        freqs_raw = params[..., :self.n_sines]
        amps_raw = params[..., self.n_sines:2*self.n_sines]
        phases_raw = params[..., 2*self.n_sines:]

        # Constrain ranges
        freqs = torch.sigmoid(freqs_raw) * self.nyquist  # 0 to Nyquist
        amps = torch.sigmoid(amps_raw)  # 0 to 1
        phases = torch.sigmoid(phases_raw) * 2 * np.pi  # 0 to 2π

        return {
            'freqs': freqs,
            'amps': amps,
            'phases': phases,
        }


class SineSynth(nn.Module):
    """
    Differentiable additive synthesis.

    freqs, amps, phases → audio
    """

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(
        self,
        freqs: torch.Tensor,
        amps: torch.Tensor,
        phases: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """
        Args:
            freqs: [B, n_sines] or [B, T, n_sines]
            amps: [B, n_sines] or [B, T, n_sines]
            phases: [B, n_sines] or [B, T, n_sines]
            n_samples: output length

        Returns:
            audio: [B, n_samples]
        """
        device = freqs.device

        # Time axis
        t = torch.arange(n_samples, device=device).float() / self.sample_rate

        # Handle time-varying params
        if freqs.dim() == 3:
            B, T, N = freqs.shape

            # Interpolate params to audio rate
            freqs = F.interpolate(freqs.transpose(1, 2), size=n_samples, mode='linear').transpose(1, 2)
            amps = F.interpolate(amps.transpose(1, 2), size=n_samples, mode='linear').transpose(1, 2)
            phases = F.interpolate(phases.transpose(1, 2), size=n_samples, mode='linear').transpose(1, 2)

            # t: [n_samples] → [1, n_samples, 1]
            t = t.unsqueeze(0).unsqueeze(-1)

            # Cumulative phase for time-varying frequency
            # φ(t) = 2π ∫ f(τ) dτ + φ_0
            dt = 1.0 / self.sample_rate
            inst_phase = 2 * np.pi * torch.cumsum(freqs * dt, dim=1) + phases[:, :1, :]

        else:
            B, N = freqs.shape
            # t: [n_samples] → [1, n_samples, 1]
            t = t.unsqueeze(0).unsqueeze(-1)
            # freqs: [B, N] → [B, 1, N]
            freqs = freqs.unsqueeze(1)
            amps = amps.unsqueeze(1)
            phases = phases.unsqueeze(1)

            # Simple phase: φ(t) = 2πft + φ_0
            inst_phase = 2 * np.pi * freqs * t + phases

        # Generate sines and sum
        sines = torch.sin(inst_phase)  # [B, n_samples, n_sines]
        audio = (sines * amps).sum(dim=-1)  # [B, n_samples]

        # Normalize to prevent clipping
        audio = audio / (audio.abs().max(dim=-1, keepdim=True)[0] + 1e-8)

        return audio


class SAMISinePipeline(nn.Module):
    """
    z_sami → sines → audio

    Train with reconstruction loss.
    Structure emerges.
    """

    def __init__(self, sami_dim: int = 128, n_sines: int = 128, sample_rate: int = 44100):
        super().__init__()
        self.mapper = SAMItoSines(sami_dim, n_sines, sample_rate)
        self.synth = SineSynth(sample_rate)
        self.sample_rate = sample_rate
        self.n_sines = n_sines
        self.sami_dim = sami_dim

    def forward(self, z_sami: torch.Tensor, n_samples: int) -> Dict[str, torch.Tensor]:
        params = self.mapper(z_sami)
        audio = self.synth(
            params['freqs'],
            params['amps'],
            params['phases'],
            n_samples,
        )
        return {
            'audio': audio,
            'freqs': params['freqs'],
            'amps': params['amps'],
            'phases': params['phases'],
        }

    def analyze_structure(self, z_sami_samples: torch.Tensor) -> Dict:
        """
        After training, analyze what the network discovered.

        Do sines cluster into harmonics? Noise? Formants?
        We don't prescribe - we observe.
        """
        with torch.no_grad():
            params = self.mapper(z_sami_samples)
            freqs = params['freqs']  # [B, n_sines]
            amps = params['amps']

            # What frequency ranges are active?
            mean_amps = amps.mean(dim=0)  # [n_sines]
            active_mask = mean_amps > 0.1
            active_freqs = freqs[:, active_mask].mean(dim=0)

            # Are there harmonic relationships?
            # (This is observation, not prescription)
            if len(active_freqs) > 1:
                f0_candidates = active_freqs[active_freqs < 500]
                if len(f0_candidates) > 0:
                    f0 = f0_candidates.min()
                    ratios = active_freqs / f0
                    harmonic_ratios = ratios[torch.abs(ratios - ratios.round()) < 0.1]
                else:
                    f0 = None
                    harmonic_ratios = torch.tensor([])
            else:
                f0 = None
                harmonic_ratios = torch.tensor([])

            return {
                'n_active_sines': active_mask.sum().item(),
                'freq_range': (freqs.min().item(), freqs.max().item()),
                'mean_amp': amps.mean().item(),
                'discovered_f0': f0.item() if f0 is not None else None,
                'harmonic_ratios': harmonic_ratios.cpu().numpy() if len(harmonic_ratios) > 0 else [],
            }


def test_pipeline():
    print("Testing SAMI → Sines pipeline...")
    print("  No prescription. Just sines.\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use DCAE dimensions: 8*16 = 128
    sami_dim = 128
    n_sines = 128

    pipeline = SAMISinePipeline(sami_dim=sami_dim, n_sines=n_sines).to(device)
    print(f"  DCAE dim: {sami_dim} (8 channels × 16 height)")
    print(f"  Params: {sum(p.numel() for p in pipeline.parameters()):,}")
    print(f"  Sines: {pipeline.n_sines}")

    # Test forward - match DCAE temporal dimension
    B, T = 2, 22  # 22 frames from DCAE
    z_sami = torch.randn(B, T, sami_dim, device=device)
    out = pipeline(z_sami, n_samples=44100)

    print(f"\n  Input: z_dcae (reshaped) {z_sami.shape}")
    print(f"  Output: audio {out['audio'].shape}")
    print(f"  Freqs: {out['freqs'].shape} range [{out['freqs'].min():.0f}, {out['freqs'].max():.0f}] Hz")
    print(f"  Amps: {out['amps'].shape} range [{out['amps'].min():.3f}, {out['amps'].max():.3f}]")

    # Analyze (before training - should be random)
    analysis = pipeline.analyze_structure(z_sami.reshape(-1, sami_dim)[:10])
    print(f"\n  Structure analysis (untrained):")
    print(f"    Active sines: {analysis['n_active_sines']}/{pipeline.n_sines}")
    print(f"    Freq range: {analysis['freq_range']}")

    print("\n  Next: train with reconstruction loss")
    print("  python training/train_sami_sines.py")
    print("  Structure will emerge from the data.\n")


if __name__ == "__main__":
    test_pipeline()
