"""
Chorus Inverter.

Chorus is mostly invertible as it's a modulated delay effect.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChorusInverter(nn.Module):
    """
    Chorus inverter.

    Chorus adds a modulated delayed copy of the signal.
    Inversion involves estimating and removing this delayed component.
    """

    def __init__(
        self,
        sample_rate: float = 44100,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.sample_rate = sample_rate

        # Network to predict the chorus component to subtract
        self.chorus_predictor = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 1023, padding=511),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 511, padding=255),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 1, 1),
        )

        # Parameter conditioning
        self.param_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, hidden_dim),
            nn.Tanh(),  # Bound to [-1, 1] for stable conditioning
        )

        self.condition_conv = nn.Conv1d(hidden_dim + hidden_dim, hidden_dim, 1)

    def forward(
        self,
        wet_audio: torch.Tensor,
        estimated_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Remove chorus effect from audio.

        Args:
            wet_audio: Audio with chorus [B, 1, T]
            estimated_params: Normalized chorus parameters [B, 4]
                (rate, depth, mix, feedback)

        Returns:
            Dry audio estimate [B, 1, T]
        """
        bs, chs, seq_len = wet_audio.size()

        # Get mix parameter
        mix = estimated_params[:, 2:3].unsqueeze(-1)  # [B, 1, 1]

        # ANALYTICAL BASELINE: Chorus adds a delayed copy
        # wet = (1-mix)*dry + mix*delayed(dry) ≈ dry * (1 + mix*0.5) for small delays
        # So: dry ≈ wet / (1 + mix*0.5)
        # This provides a reasonable starting point before neural refinement
        analytical_dry = wet_audio / (1 + mix * 0.5 + 1e-8)

        # Extract features for neural refinement
        x = wet_audio
        for layer in list(self.chorus_predictor.children())[:-1]:
            x = layer(x)

        # Condition on parameters
        param_features = self.param_encoder(estimated_params)
        param_features = param_features.unsqueeze(-1).expand(-1, -1, x.size(-1))

        conditioned = torch.cat([x, param_features], dim=1)
        conditioned = self.condition_conv(conditioned)
        conditioned = F.silu(conditioned)

        # Predict residual correction (not the full chorus component)
        # This lets the network learn to refine the analytical baseline
        residual = self.chorus_predictor[-1](conditioned)

        # Combine: start with analytical, add learned residual scaled by mix
        # Scale up residual so network can meaningfully contribute
        dry_estimate = analytical_dry + mix * 0.3 * residual

        # Handle NaN and clamp
        dry_estimate = torch.nan_to_num(dry_estimate, nan=0.0, posinf=1.0, neginf=-1.0)
        dry_estimate = torch.clamp(dry_estimate, -1.0, 1.0)

        return dry_estimate


class AnalyticalChorusInverter(nn.Module):
    """
    Analytical chorus inversion using comb filtering.
    """

    def __init__(self, sample_rate: float = 44100):
        super().__init__()
        self.sample_rate = sample_rate
        self.base_delay_ms = 7.0
        self.max_mod_ms = 3.0

    def forward(
        self,
        wet_audio: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Invert chorus using inverse comb filtering.

        Args:
            wet_audio: Chorused audio [B, 1, T]
            params: Chorus parameters [B, 4]

        Returns:
            Dry estimate [B, 1, T]
        """
        bs, chs, seq_len = wet_audio.size()

        # Denormalize parameters
        rate = 0.1 + params[:, 0] * 9.9  # [0.1, 10] Hz
        depth = params[:, 1]  # [0, 1]
        mix = params[:, 2]  # [0, 1]
        feedback = params[:, 3] * 0.9  # [0, 0.9]

        # Generate inverse LFO
        t = torch.arange(seq_len, device=wet_audio.device, dtype=wet_audio.dtype)
        t = t.view(1, 1, -1) / self.sample_rate

        rate_exp = rate.view(bs, 1, 1)
        lfo = torch.sin(2 * math.pi * rate_exp * t)

        # Calculate delay variation
        depth_exp = depth.view(bs, 1, 1)
        base_delay_samples = self.base_delay_ms * self.sample_rate / 1000
        mod_samples = depth_exp * self.max_mod_ms * self.sample_rate / 1000
        delay_samples = base_delay_samples + lfo * mod_samples

        # Create inverse delayed signal
        indices = torch.arange(seq_len, device=wet_audio.device, dtype=wet_audio.dtype)
        indices = indices.view(1, 1, -1).expand(bs, 1, -1)

        # For inverse, we add instead of delay
        write_indices = (indices + delay_samples).long()
        write_indices = torch.clamp(write_indices, 0, seq_len - 1)

        # Subtract estimated chorus component
        mix_exp = mix.view(bs, 1, 1)
        feedback_exp = feedback.view(bs, 1, 1)

        # Simple approximation: subtract delayed version
        delayed = torch.zeros_like(wet_audio)
        for b in range(bs):
            avg_delay = int(base_delay_samples)
            if avg_delay < seq_len:
                delayed[b, :, avg_delay:] = wet_audio[b, :, :seq_len-avg_delay]

        dry_estimate = (wet_audio - mix_exp * delayed) / (1 - mix_exp * feedback_exp + 1e-8)

        dry_estimate = torch.nan_to_num(dry_estimate, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.clamp(dry_estimate, -1.0, 1.0)
