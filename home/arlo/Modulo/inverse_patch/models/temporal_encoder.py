"""
Universal Temporal Pattern Encoder.

General mechanism for handling ANY temporal effect without per-effect engineering.
Works for delay, chorus, flanger, phaser, tremolo, vibrato, multitap, etc.

Key insight: All temporal effects create relationships between samples at different
time positions. This encoder captures those relationships automatically.

Effect          | Temporal Pattern
----------------|--------------------------------------------------
Delay           | Fixed offset: t ↔ t-T
Chorus          | Modulating offset: t ↔ t-T(t) where T oscillates
Flanger         | Short modulating offset with feedback
Phaser          | Phase relationships across frequencies
Tremolo         | Amplitude modulation at rate R
Vibrato         | Pitch modulation at rate R
Multitap delay  | Multiple offsets: t ↔ t-T1, t-T2, t-T3
Slapback        | Short fixed offset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Tuple


class TemporalCorrelationEncoder(nn.Module):
    """
    Computes dense correlation map showing ALL temporal relationships.
    Network learns which relationships matter for each effect.

    This is the most general approach - captures patterns at arbitrary lags
    without hardcoding any effect-specific knowledge.

    How it works:
    - Correlation at lag 22050 = high → delay at 500ms (at 44.1kHz)
    - Correlation at oscillating lags = high → chorus/flanger
    - Amplitude modulation patterns → tremolo

    The network learns to read this map for any effect.
    """

    def __init__(
        self,
        max_lag_ms: float = 2000.0,
        sample_rate: int = 44100,
        n_lags: int = 64,
        output_dim: int = 32,
        learnable_lags: bool = True,
    ):
        """
        Args:
            max_lag_ms: Maximum lag to consider in milliseconds
            sample_rate: Audio sample rate
            n_lags: Number of lag positions to sample
            output_dim: Output feature dimension
            learnable_lags: If True, lag positions are learnable parameters
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.max_lag = int(max_lag_ms * sample_rate / 1000)
        self.n_lags = n_lags
        self.output_dim = output_dim

        # Initialize lags as log-spaced (captures both short and long patterns)
        # Log-spacing is important: flanger needs ~1-10ms, delay needs 100-2000ms
        default_lags = torch.logspace(
            0,
            math.log10(self.max_lag + 1),
            n_lags
        ).long()

        if learnable_lags:
            # Learnable lag positions (continuous, discretized during forward)
            self.lag_offsets = nn.Parameter(default_lags.float())
        else:
            self.register_buffer('lag_offsets', default_lags.float())

        # Correlation processing network
        # Takes n_lags correlation features and produces output_dim features
        self.corr_net = nn.Sequential(
            nn.Conv1d(n_lags, 64, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3, groups=8),  # Grouped for efficiency
            nn.SiLU(),
            nn.Conv1d(64, output_dim, kernel_size=1),
        )

        # Temporal smoothing to reduce noise in correlation estimates
        self.smooth_kernel_size = 127  # ~3ms at 44.1kHz
        self.register_buffer(
            'smooth_kernel',
            torch.hann_window(self.smooth_kernel_size).view(1, 1, -1)
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal correlation features.

        Args:
            audio: [B, 1, T] or [B, T] input audio

        Returns:
            features: [B, output_dim, T] temporal correlation features
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        B, C, T = audio.shape
        device = audio.device

        # Get discrete lag values
        lags = self.lag_offsets.long().clamp(0, self.max_lag)

        correlations = []
        for i, lag in enumerate(lags):
            lag = lag.item()

            if lag == 0:
                # Autocorrelation at lag 0 = energy
                corr = audio * audio
            elif lag < T:
                # Correlation between sample at t and sample at t-lag
                # This reveals periodic patterns at this lag
                shifted = F.pad(audio[:, :, lag:], (0, lag))
                corr = audio * shifted

                # Apply smoothing to get more stable correlation estimates
                if self.smooth_kernel_size < T:
                    corr = F.conv1d(
                        corr,
                        self.smooth_kernel / self.smooth_kernel.sum(),
                        padding=self.smooth_kernel_size // 2,
                    )
            else:
                # Lag exceeds signal length
                corr = torch.zeros_like(audio)

            correlations.append(corr)

        # Stack: [B, n_lags, T]
        corr_map = torch.cat(correlations, dim=1)

        # Normalize correlations (important for stable training)
        # Use instance norm to handle varying audio levels
        # Add small noise to prevent zero variance causing NaN
        corr_map = corr_map + 1e-8 * torch.randn_like(corr_map)
        corr_map = F.instance_norm(corr_map, eps=1e-5)

        # Process to learn which correlations matter for this effect
        features = self.corr_net(corr_map)

        return features

    def get_lag_ms(self) -> torch.Tensor:
        """Return current lag positions in milliseconds."""
        return self.lag_offsets * 1000.0 / self.sample_rate


class MultiScaleTemporalEncoder(nn.Module):
    """
    Analyze audio at multiple time scales automatically.

    This captures patterns across the full temporal hierarchy:
    - Short scale (1-4 samples): catches sample-level distortion artifacts
    - Medium scale (16-256 samples): catches chorus/tremolo modulation
    - Long scale (1024-4096 samples): catches delay echoes

    All scales are analyzed in parallel and combined via cross-scale attention.
    """

    def __init__(
        self,
        scales: List[int] = [1, 4, 16, 64, 256, 1024, 4096],
        hidden_dim: int = 32,
        output_dim: int = 32,
        use_cross_scale_attention: bool = True,
    ):
        """
        Args:
            scales: List of scale factors (in samples) to analyze
            hidden_dim: Hidden dimension per scale
            output_dim: Output feature dimension
            use_cross_scale_attention: Whether to use attention across scales
        """
        super().__init__()

        self.scales = scales
        self.n_scales = len(scales)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_attention = use_cross_scale_attention

        # Per-scale analysis networks
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    1, hidden_dim,
                    kernel_size=min(s * 2 + 1, 1025),  # Cap kernel size
                    stride=max(s, 1),
                    padding=min(s, 512),
                ),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            )
            for s in scales
        ])

        # Cross-scale attention (optional but recommended)
        if use_cross_scale_attention:
            self.cross_scale_attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads=4,
                batch_first=True,
                dropout=0.1,
            )
            self.scale_norm = nn.LayerNorm(hidden_dim)

        # Final projection
        self.output_proj = nn.Conv1d(
            hidden_dim * (2 if use_cross_scale_attention else 1),
            output_dim,
            kernel_size=1
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale temporal features.

        Args:
            audio: [B, 1, T] or [B, T] input audio

        Returns:
            features: [B, output_dim, T] multi-scale features
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        B, C, T = audio.shape

        scale_features = []
        for encoder, scale in zip(self.scale_encoders, self.scales):
            # Analyze at this scale
            feat = encoder(audio)  # [B, hidden_dim, T/scale]

            # Upsample back to original resolution for combination
            feat = F.interpolate(feat, size=T, mode='linear', align_corners=False)
            scale_features.append(feat)

        # Stack scales: [B, n_scales, hidden_dim, T]
        stacked = torch.stack(scale_features, dim=1)

        if self.use_attention:
            # Cross-scale attention at each time position
            # Reshape for attention: [B*T, n_scales, hidden_dim]
            B, S, D, T = stacked.shape
            stacked_flat = stacked.permute(0, 3, 1, 2).reshape(B * T, S, D)

            # Self-attention across scales
            attended, _ = self.cross_scale_attn(
                stacked_flat, stacked_flat, stacked_flat
            )
            attended = self.scale_norm(attended + stacked_flat)

            # Reshape back: [B, T, n_scales, hidden_dim] -> [B, hidden_dim, T]
            attended = attended.reshape(B, T, S, D).permute(0, 3, 1, 2)  # [B, D, S, T]

            # Pool across scales
            scale_pooled = attended.mean(dim=2)  # [B, hidden_dim, T]

            # Also keep the raw mean for residual
            raw_pooled = stacked.mean(dim=1)  # [B, hidden_dim, T]

            # Combine attended and raw
            combined = torch.cat([scale_pooled, raw_pooled], dim=1)  # [B, 2*hidden_dim, T]
        else:
            # Simple mean pooling across scales
            combined = stacked.mean(dim=1)  # [B, hidden_dim, T]

        # Project to output dimension
        features = self.output_proj(combined)

        return features


class TemporalDifferenceEncoder(nn.Module):
    """
    Encode temporal differences at multiple time lags.

    Simpler alternative to correlation - just looks at sample differences.
    Good for detecting sudden changes (transients, echo onsets).
    """

    def __init__(
        self,
        lags: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        output_dim: int = 32,
    ):
        super().__init__()

        self.lags = lags
        self.n_lags = len(lags)

        # Process difference features
        self.diff_net = nn.Sequential(
            nn.Conv1d(len(lags), 64, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(64, output_dim, kernel_size=1),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal difference features.

        Args:
            audio: [B, 1, T] input audio

        Returns:
            features: [B, output_dim, T]
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        B, C, T = audio.shape

        diffs = []
        for lag in self.lags:
            if lag < T:
                # Difference between sample at t and t-lag
                diff = audio[:, :, lag:] - audio[:, :, :-lag]
                diff = F.pad(diff, (lag, 0))  # Pad to maintain length
            else:
                diff = torch.zeros_like(audio)
            diffs.append(diff)

        # Stack: [B, n_lags, T]
        diff_stack = torch.cat(diffs, dim=1)

        # Process
        features = self.diff_net(diff_stack)

        return features


class UnifiedTemporalEncoder(nn.Module):
    """
    Combines multiple temporal analysis strategies for maximum generality.

    Uses:
    1. TemporalCorrelationEncoder - for periodic patterns (delay, chorus)
    2. MultiScaleTemporalEncoder - for multi-resolution analysis
    3. TemporalDifferenceEncoder - for transient detection

    The combination handles ANY temporal effect pattern.
    """

    def __init__(
        self,
        max_lag_ms: float = 2000.0,
        sample_rate: int = 44100,
        output_dim: int = 64,
    ):
        super().__init__()

        self.output_dim = output_dim

        # Three complementary temporal encoders
        self.correlation_enc = TemporalCorrelationEncoder(
            max_lag_ms=max_lag_ms,
            sample_rate=sample_rate,
            n_lags=64,
            output_dim=output_dim // 3,
        )

        self.multiscale_enc = MultiScaleTemporalEncoder(
            scales=[1, 4, 16, 64, 256, 1024, 4096],
            hidden_dim=32,
            output_dim=output_dim // 3,
        )

        self.difference_enc = TemporalDifferenceEncoder(
            lags=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            output_dim=output_dim - 2 * (output_dim // 3),  # Remainder
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=1),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract unified temporal features.

        Args:
            audio: [B, 1, T] input audio

        Returns:
            features: [B, output_dim, T]
        """
        # Get features from all encoders
        corr_feat = self.correlation_enc(audio)
        scale_feat = self.multiscale_enc(audio)
        diff_feat = self.difference_enc(audio)

        # Concatenate and fuse
        combined = torch.cat([corr_feat, scale_feat, diff_feat], dim=1)
        features = self.fusion(combined)

        return features


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional/time embedding."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] or scalar timestep values

        Returns:
            embedding: [B, dim]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(half_dim, device=t.device) / half_dim
        )

        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


if __name__ == '__main__':
    # Test temporal encoders
    print("Testing TemporalCorrelationEncoder...")
    corr_enc = TemporalCorrelationEncoder(max_lag_ms=1000, n_lags=32, output_dim=32)
    audio = torch.randn(2, 1, 44100)  # 1 second
    corr_feat = corr_enc(audio)
    print(f"  Input: {audio.shape} -> Output: {corr_feat.shape}")
    print(f"  Lag range: {corr_enc.get_lag_ms()[0]:.1f}ms to {corr_enc.get_lag_ms()[-1]:.1f}ms")

    print("\nTesting MultiScaleTemporalEncoder...")
    scale_enc = MultiScaleTemporalEncoder(output_dim=32)
    scale_feat = scale_enc(audio)
    print(f"  Input: {audio.shape} -> Output: {scale_feat.shape}")

    print("\nTesting UnifiedTemporalEncoder...")
    unified_enc = UnifiedTemporalEncoder(max_lag_ms=1000, output_dim=64)
    unified_feat = unified_enc(audio)
    print(f"  Input: {audio.shape} -> Output: {unified_feat.shape}")

    print("\nAll tests passed!")
