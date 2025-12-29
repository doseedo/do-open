#!/usr/bin/env python3
"""
Corruption Model: Learn to synthesize DSP pitch shift artifacts

The reverse approach:
1. Train model to corrupt clean audio → DSP-shifted-like artifacts
2. DSP artifacts are algorithmic, should be more pitch-invariant
3. Use corruption model to create aligned pairs for correction training

Training:
- Input: Natural trumpet latent at any pitch
- Target distribution: DSP-shifted latents (existing shifted data)
- Distribution matching learns the artifact signature

Then for correction:
- Take natural recording
- Apply corruption model → synthetic "corrupted" version
- Pair: (corrupted, natural) - perfectly aligned!
- Train correction with frame-level supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ShiftEmbedding(nn.Module):
    """Embed shift amount into conditioning vector."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, shift: torch.Tensor) -> torch.Tensor:
        if shift.dim() == 1:
            shift = shift.unsqueeze(-1)
        shift_norm = shift.float() / 12.0  # Normalize by octave
        return self.mlp(shift_norm)


class FiLMConvBlock(nn.Module):
    """Conv block with FiLM conditioning."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.film1 = nn.Linear(cond_dim, channels * 2)
        self.film2 = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        gamma1, beta1 = self.film1(cond).chunk(2, dim=-1)
        h = h * (1 + gamma1[:, :, None, None]) + beta1[:, :, None, None]
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        gamma2, beta2 = self.film2(cond).chunk(2, dim=-1)
        h = h * (1 + gamma2[:, :, None, None]) + beta2[:, :, None, None]
        h = F.silu(h)

        return x + h


class CorruptionModel(nn.Module):
    """
    Learn to add DSP-like pitch shift artifacts.

    Conditioned on shift amount: learns what +12 corruption vs -12 corruption looks like.

    Key hypothesis: DSP artifacts are algorithmic/pitch-invariant.
    Unlike acoustic muting which changes differently at different pitches,
    phase vocoder artifacts follow consistent mathematical patterns.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        cond_dim: int = 64,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.cond_dim = cond_dim

        # Shift conditioning
        self.shift_embed = ShiftEmbedding(cond_dim)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        # Main blocks with FiLM conditioning
        self.blocks = nn.ModuleList([
            FiLMConvBlock(hidden_dim, cond_dim)
            for _ in range(num_blocks)
        ])

        # Output projection (residual corruption)
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, latent_channels, 1),
        )

        # Initialize small - corruption should be subtle
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        clean_latent: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add DSP-like corruption to clean latent.

        Args:
            clean_latent: [B, 8, 16, T] natural trumpet latent
            shift: [B] shift amount to simulate (e.g., +12 or -12)

        Returns:
            corrupted_latent: [B, 8, 16, T] with DSP-like artifacts
        """
        cond = self.shift_embed(shift)

        h = self.input_proj(clean_latent)

        for block in self.blocks:
            h = block(h, cond)

        corruption = self.output_proj(h)

        # Add corruption as residual
        return clean_latent + corruption


class CorruptionDiscriminator(nn.Module):
    """
    Discriminate real DSP-shifted vs synthetic corruption.

    Conditioned on shift amount.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_dim: int = 128,
        cond_dim: int = 64,
    ):
        super().__init__()

        self.shift_embed = ShiftEmbedding(cond_dim)

        # Conv path
        self.conv = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # FiLM conditioning
        self.film_gamma = nn.Linear(cond_dim, hidden_dim)
        self.film_beta = nn.Linear(cond_dim, hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, 8, 16, T] latent (real DSP-shifted or synthetic)
            shift: [B] shift amount

        Returns:
            [B, 1] real/fake logits
        """
        B = latent.shape[0]

        h = self.conv(latent)

        # FiLM conditioning
        cond = self.shift_embed(shift)
        gamma = self.film_gamma(cond).view(B, -1, 1, 1)
        beta = self.film_beta(cond).view(B, -1, 1, 1)
        h = h * (1 + gamma) + beta

        return self.classifier(h)


# =============================================================================
# Losses
# =============================================================================

class CorruptionGANLoss(nn.Module):
    """Hinge GAN loss for corruption model."""

    def discriminator_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss = F.relu(1.0 + fake_logits).mean()
        return real_loss + fake_loss

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return -fake_logits.mean()


class ContentPreservationLoss(nn.Module):
    """
    Ensure corruption doesn't destroy content.

    Corrupted should still be highly correlated with clean.
    """

    def forward(self, corrupted: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        B = corrupted.shape[0]

        corr_flat = corrupted.reshape(B, -1)
        clean_flat = clean.reshape(B, -1)

        # Center
        corr_centered = corr_flat - corr_flat.mean(dim=1, keepdim=True)
        clean_centered = clean_flat - clean_flat.mean(dim=1, keepdim=True)

        # Correlation
        corr_std = corr_centered.std(dim=1, keepdim=True) + 1e-8
        clean_std = clean_centered.std(dim=1, keepdim=True) + 1e-8

        correlation = (corr_centered * clean_centered).mean(dim=1) / (
            corr_std.squeeze() * clean_std.squeeze()
        )

        # Want high correlation (0.95+), penalize deviation
        target_corr = 0.95
        return F.relu(target_corr - correlation).mean()


class ArtifactMagnitudeLoss(nn.Module):
    """
    Encourage appropriate artifact magnitude.

    Corruption should be noticeable but not overwhelming.
    Calibrate to real DSP shift magnitudes.
    """

    def __init__(self, target_magnitude: float = 0.5):
        super().__init__()
        self.target_magnitude = target_magnitude

    def forward(
        self,
        corrupted: torch.Tensor,
        clean: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        # Corruption magnitude
        diff = (corrupted - clean).abs()
        magnitude = diff.mean(dim=(1, 2, 3))  # Per-sample magnitude

        # Scale target by shift magnitude (larger shifts = more corruption)
        shift_scale = (shift.abs() / 12.0).clamp(0.1, 1.0)
        target = self.target_magnitude * shift_scale

        # Penalize deviation from target
        return F.mse_loss(magnitude, target)


class C3CorruptionLoss(nn.Module):
    """
    DSP artifacts primarily affect C3 (register channel).

    Encourage corruption to focus on C3 like real DSP does.
    """

    def forward(
        self,
        corrupted: torch.Tensor,
        clean: torch.Tensor,
    ) -> torch.Tensor:
        # Compute corruption per channel
        diff = (corrupted - clean).abs()
        channel_corruption = diff.mean(dim=(2, 3))  # [B, C]

        # C3 should have relatively high corruption
        c3_corruption = channel_corruption[:, 3]
        other_corruption = torch.cat([
            channel_corruption[:, :3],
            channel_corruption[:, 4:]
        ], dim=1).mean(dim=1)

        # C3 should be at least as corrupted as average
        # (but don't force it to be ONLY C3)
        return F.relu(other_corruption - c3_corruption).mean()
