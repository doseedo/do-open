#!/usr/bin/env python3
"""
Audio Enhancer - Post-DCAE super-resolution model.

Takes decoded DCAE audio (lossy) and outputs high-fidelity audio.
Conditioned on group_id and subgroup_id for instrument-specific restoration.

Input: decoded DCAE audio [B, 2, T] + group_id + subgroup_id
Output: enhanced audio [B, 2, T] matching original pre-encoded quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for conditioning."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.scale = nn.Linear(cond_dim, channels)
        self.shift = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] feature tensor
            cond: [B, cond_dim] conditioning vector
        Returns:
            Modulated features [B, C, T]
        """
        scale = self.scale(cond).unsqueeze(-1)  # [B, C, 1]
        shift = self.shift(cond).unsqueeze(-1)  # [B, C, 1]
        return x * (1 + scale) + shift


class ConditionedResBlock(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.film = FiLM(cond_dim, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.film(x, cond)
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class AudioEnhancer(nn.Module):
    """
    Waveform-domain post-DCAE enhancement model.

    Learns to restore high-frequency detail and fix artifacts
    from DCAE encode/decode cycle, conditioned on instrument type.
    """

    def __init__(
        self,
        in_channels: int = 2,  # stereo
        hidden_channels: int = 64,
        num_blocks: int = 12,
        kernel_size: int = 7,
        group_vocab: int = 6,
        subgroup_vocab: int = 20,
        cond_dim: int = 64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # Conditioning embeddings
        self.group_emb = nn.Embedding(group_vocab, cond_dim)
        self.subgroup_emb = nn.Embedding(subgroup_vocab, cond_dim)
        self.cond_proj = nn.Linear(cond_dim * 2, cond_dim)

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size,
                                     padding=kernel_size // 2)

        # Residual blocks with increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** (i % 5)  # 1, 2, 4, 8, 16, 1, 2, ...
            self.blocks.append(
                ConditionedResBlock(hidden_channels, cond_dim, kernel_size, dilation)
            )

        # Output projection
        self.output_proj = nn.Conv1d(hidden_channels, in_channels, kernel_size,
                                      padding=kernel_size // 2)

        # Initialize output to small values for stable training start
        nn.init.normal_(self.output_proj.weight, std=0.01)
        nn.init.zeros_(self.output_proj.bias)

        # Learnable residual scale (start at 0.5 - spectral losses are stable)
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        audio: torch.Tensor,
        group_id: torch.Tensor,
        subgroup_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio: [B, 2, T] decoded DCAE audio
            group_id: [B] instrument group indices
            subgroup_id: [B] instrument subgroup indices

        Returns:
            enhanced: [B, 2, T] enhanced audio
        """
        # Build conditioning vector
        g_emb = self.group_emb(group_id)  # [B, cond_dim]
        sg_emb = self.subgroup_emb(subgroup_id)  # [B, cond_dim]
        cond = self.cond_proj(torch.cat([g_emb, sg_emb], dim=-1))  # [B, cond_dim]

        # Process audio
        x = self.input_proj(audio)  # [B, hidden, T]

        for block in self.blocks:
            x = block(x, cond)

        # Output residual
        residual = self.output_proj(x)  # [B, 2, T]

        # Residual connection
        enhanced = audio + self.residual_scale * residual

        return enhanced


class AudioEnhancerLarge(AudioEnhancer):
    """Larger version with more capacity."""

    def __init__(
        self,
        group_vocab: int = 6,
        subgroup_vocab: int = 20,
    ):
        super().__init__(
            in_channels=2,
            hidden_channels=128,
            num_blocks=16,
            kernel_size=7,
            group_vocab=group_vocab,
            subgroup_vocab=subgroup_vocab,
            cond_dim=128,
        )


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for adversarial training.
    Operates at multiple audio resolutions.
    """

    def __init__(self, num_scales: int = 3):
        super().__init__()

        self.discriminators = nn.ModuleList([
            self._make_discriminator() for _ in range(num_scales)
        ])
        self.pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2) for _ in range(num_scales - 1)
        ])

    def _make_discriminator(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(2, 32, 15, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 41, stride=4, padding=20, groups=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 1, 3, padding=1),
        )

    def forward(self, audio: torch.Tensor) -> list:
        """Returns list of discriminator outputs at each scale."""
        outputs = []
        x = audio

        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x))
            if i < len(self.pools):
                x = self.pools[i](x)

        return outputs


if __name__ == "__main__":
    # Quick test
    print("Testing AudioEnhancer...")

    model = AudioEnhancer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    B, T = 2, 48000  # 1 second at 48kHz
    audio = torch.randn(B, 2, T)
    group_id = torch.randint(0, 6, (B,))
    subgroup_id = torch.randint(0, 20, (B,))

    out = model(audio, group_id, subgroup_id)
    print(f"Input shape: {audio.shape}")
    print(f"Output shape: {out.shape}")

    # Test large model
    print("\nTesting AudioEnhancerLarge...")
    model_large = AudioEnhancerLarge()
    print(f"Parameters: {sum(p.numel() for p in model_large.parameters()):,}")

    out_large = model_large(audio, group_id, subgroup_id)
    print(f"Output shape: {out_large.shape}")
