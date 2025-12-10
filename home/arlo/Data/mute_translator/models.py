"""
Mute Translator Models

Latent space translator for dry→muted trumpet conversion.
Works with ACE-Step DCAE latents [B, 8, H, T].
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


class MuteTranslator(nn.Module):
    """
    Translates dry trumpet latents to muted trumpet latents.

    Architecture: Residual network operating on DCAE latent space.
    Input/Output: [B, 8, H, T] where H=height (frequency), T=time

    The model learns a residual mapping: muted = dry + f(dry)
    This preserves the melodic content while modifying timbre.
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
        # DCAE latents are [B, 8, H, T] - we process as [B, 8*H, T]
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

        # Initialize output to near-zero for stable training start
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        # Learnable residual scale (starts small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, dry_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dry_latent: [B, C, H, T] dry trumpet latent

        Returns:
            muted_latent: [B, C, H, T] predicted muted trumpet latent
        """
        B, C, H, T = dry_latent.shape

        # Flatten H into batch for 1D processing: [B*H, C, T]
        x = dry_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)

        # Process
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        residual = self.output_proj(x)

        # Reshape back: [B, H, C, T] -> [B, C, H, T]
        residual = residual.reshape(B, H, C, T).permute(0, 2, 1, 3)

        # Apply residual with learnable scale
        muted_latent = dry_latent + self.residual_scale * residual

        return muted_latent


class MuteTranslatorLarge(nn.Module):
    """
    Larger translator with 2D convolutions for better H-T interaction.

    Uses full 2D convolutions on the latent [B, C, H, T] tensor.
    Better for capturing cross-frequency mute effects.
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

        # Zero init for residual learning
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, dry_latent: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(dry_latent)

        for block in self.blocks:
            x = x + block(x)  # Residual connection

        residual = self.output_proj(x)
        return dry_latent + self.residual_scale * residual


class MuteDiscriminator(nn.Module):
    """
    Discriminator for adversarial training (optional).
    Classifies whether a latent is real muted or synthetic muted.
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
        """Returns logit for real/fake classification."""
        return self.net(latent)


class CycleConsistentMuteTranslator(nn.Module):
    """
    Bidirectional translator with cycle consistency.

    Trains two translators:
    - dry_to_muted: dry → muted
    - muted_to_dry: muted → dry

    With cycle consistency: dry → muted → dry ≈ dry
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.dry_to_muted = MuteTranslator(**kwargs)
        self.muted_to_dry = MuteTranslator(**kwargs)

    def forward(
        self,
        latent: torch.Tensor,
        direction: str = "dry_to_muted"
    ) -> torch.Tensor:
        if direction == "dry_to_muted":
            return self.dry_to_muted(latent)
        else:
            return self.muted_to_dry(latent)

    def cycle_forward(
        self,
        dry_latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (muted_pred, dry_reconstructed)"""
        muted = self.dry_to_muted(dry_latent)
        dry_recon = self.muted_to_dry(muted)
        return muted, dry_recon


if __name__ == "__main__":
    # Test models
    batch = torch.randn(2, 8, 16, 128)  # [B, C, H, T]

    print("Testing MuteTranslator...")
    model = MuteTranslator()
    out = model(batch)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting MuteTranslatorLarge...")
    model_large = MuteTranslatorLarge()
    out = model_large(batch)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model_large.parameters()):,}")

    print("\nTesting CycleConsistentMuteTranslator...")
    cycle_model = CycleConsistentMuteTranslator()
    muted, recon = cycle_model.cycle_forward(batch)
    print(f"  Muted: {muted.shape}, Reconstructed: {recon.shape}")
    print(f"  Params: {sum(p.numel() for p in cycle_model.parameters()):,}")
