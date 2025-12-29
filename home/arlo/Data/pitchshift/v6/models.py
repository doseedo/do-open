#!/usr/bin/env python3
"""
Register Translator Models

Latent space translator for high→low register trumpet conversion.
Works with ACE-Step DCAE latents [B, 8, H, T].

Key insight: Train on VALID inputs (natural high → natural low),
then sox shift AFTER model inference for actual pitch correction.

This matches the mute translator approach: distribution matching,
no corrupted inputs during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResidualBlock1D(nn.Module):
    """1D Residual block for temporal latent processing."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class RegisterTranslator(nn.Module):
    """
    Translates high register latents to low register latents.

    Architecture: Residual network operating on DCAE latent space.
    Input/Output: [B, 8, H, T] where H=height (frequency), T=time

    The model learns a residual mapping: low = high + f(high)
    This preserves the melodic content while modifying formants/timbre.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.latent_channels = latent_channels

        # Project to hidden space (flatten H dimension into channels)
        self.input_proj = nn.Conv1d(latent_channels, hidden_channels, 1)

        # Residual blocks with increasing dilation
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_channels, kernel_size, dilation=2**i)
            for i in range(num_blocks)
        ])

        # Output projection (residual)
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, latent_channels, 1),
        )

        # Small random init (NOT zero - zero causes dead gradients)
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[-1].bias)

        # Learnable residual scale (starts small but not too small)
        self.residual_scale = nn.Parameter(torch.tensor(0.2))

        # Learnable input attenuation
        self.input_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, high_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            high_latent: [B, C, H, T] high register latent

        Returns:
            low_latent: [B, C, H, T] predicted low register latent
        """
        B, C, H, T = high_latent.shape

        # Flatten H into batch for 1D processing: [B*H, C, T]
        x = high_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)

        # Process
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        residual = self.output_proj(x)

        # Reshape back: [B, H, C, T] -> [B, C, H, T]
        residual = residual.reshape(B, H, C, T).permute(0, 2, 1, 3)

        # Apply with learnable scales
        low_latent = self.input_scale * high_latent + self.residual_scale * residual

        return low_latent


class RegisterTranslatorDirect(nn.Module):
    """
    Non-residual translator: output = f(high), not high + f(high).

    Allows the model to REPLACE the formants entirely rather than
    just adding low register character on top.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 7,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, kernel_size, dilation=2**i)
            for i in range(num_blocks)
        ])

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, in_channels, 1),
        )

        # Small init for stability
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, high_latent: torch.Tensor) -> torch.Tensor:
        B, C, H, T = high_latent.shape

        # Flatten H into batch for 1D processing
        x = high_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)

        # Process
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        output = self.output_proj(x)

        # Reshape back
        low_latent = output.reshape(B, H, C, T).permute(0, 2, 1, 3)

        return low_latent


class RegisterTranslatorAdaptive(nn.Module):
    """
    Adaptive mixing: output = alpha(t) * high + (1 - alpha(t)) * transformed

    Learns per-frame mixing between preserving input and applying transformation.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        alpha_init: float = 0.3,
    ):
        super().__init__()

        # Transformation network
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, 1)

        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, kernel_size, dilation=2**i)
            for i in range(num_blocks)
        ])

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, in_channels, 1),
        )

        # Alpha predictor: learns per-frame mixing from input
        self.alpha_net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim // 2, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
        )

        self.alpha_bias = nn.Parameter(torch.tensor(alpha_init))

        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, high_latent: torch.Tensor) -> torch.Tensor:
        B, C, H, T = high_latent.shape

        x = high_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)

        transformed = self.input_proj(x)
        for block in self.blocks:
            transformed = block(transformed)
        low = self.output_proj(transformed)

        low = low.reshape(B, H, C, T).permute(0, 2, 1, 3)

        # Predict alpha from input
        high_for_alpha = high_latent.mean(dim=2)  # [B, C, T]
        alpha_logits = self.alpha_net(high_for_alpha)
        alpha = torch.sigmoid(alpha_logits + self.alpha_bias)
        alpha = alpha.clamp(0.1, 0.9).unsqueeze(2)  # [B, 1, 1, T]

        output = alpha * high_latent + (1 - alpha) * low

        return output


class RegisterTranslator2D(nn.Module):
    """
    2D convolutional translator for better frequency-time interaction.

    Uses full 2D convolutions on [B, C, H, T] tensor.
    Better for capturing cross-frequency register effects.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 32,
        num_blocks: int = 8,
    ):
        super().__init__()

        self.input_proj = nn.Conv2d(latent_channels, hidden_channels, 3, padding=1)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(8, hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(8, hidden_channels),
            ))

        self.output_proj = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_channels, latent_channels, 3, padding=1),
        )

        # Zero init for residual
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, high_latent: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(high_latent)

        for block in self.blocks:
            x = x + block(x)

        residual = self.output_proj(x)
        return high_latent + self.residual_scale * residual


class RegisterDiscriminator(nn.Module):
    """
    Discriminator for adversarial training (optional).
    Classifies whether a latent is real low register or synthetic.
    """

    def __init__(self, latent_channels: int = 8, hidden_channels: int = 32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels * 4, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class CycleConsistentRegisterTranslator(nn.Module):
    """
    Bidirectional translator with cycle consistency.

    Trains two translators:
    - high_to_low: high → low
    - low_to_high: low → high

    With cycle consistency: high → low → high ≈ high
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.high_to_low = RegisterTranslator(**kwargs)
        self.low_to_high = RegisterTranslator(**kwargs)

    def forward(
        self,
        latent: torch.Tensor,
        direction: str = "high_to_low"
    ) -> torch.Tensor:
        if direction == "high_to_low":
            return self.high_to_low(latent)
        else:
            return self.low_to_high(latent)

    def cycle_forward(
        self,
        high_latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (low_pred, high_reconstructed)"""
        low = self.high_to_low(high_latent)
        high_recon = self.low_to_high(low)
        return low, high_recon


if __name__ == "__main__":
    # Test models
    batch = torch.randn(2, 8, 16, 128)  # [B, C, H, T]

    print("Testing RegisterTranslator...")
    model = RegisterTranslator()
    out = model(batch)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting RegisterTranslatorDirect...")
    model_direct = RegisterTranslatorDirect()
    out = model_direct(batch)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model_direct.parameters()):,}")

    print("\nTesting RegisterTranslatorAdaptive...")
    model_adaptive = RegisterTranslatorAdaptive()
    out = model_adaptive(batch)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model_adaptive.parameters()):,}")

    print("\nTesting RegisterTranslator2D...")
    model_2d = RegisterTranslator2D()
    out = model_2d(batch)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model_2d.parameters()):,}")

    print("\nTesting CycleConsistentRegisterTranslator...")
    cycle_model = CycleConsistentRegisterTranslator()
    low, recon = cycle_model.cycle_forward(batch)
    print(f"  Low: {low.shape}, Reconstructed: {recon.shape}")
    print(f"  Params: {sum(p.numel() for p in cycle_model.parameters()):,}")
