#!/usr/bin/env python3
"""
Register-Conditioned Adversarial Models for Pitch Shift Correction

The discriminator learns: "Does this latent look like NATURAL trumpet at pitch X?"
- Penalizes noise/harmonics in wrong places
- Penalizes trombone-like softness on low trumpet
- Learns holistic patterns, not just statistics

Generator (corrector) learns to fool discriminator while preserving content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PitchEmbedding(nn.Module):
    """Embed absolute pitch (MIDI note) into conditioning vector."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        # Pitch range roughly 36-96 (C2 to C7)
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, pitch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pitch: [B] or [B, 1] MIDI pitch (0-127 scale, or normalized)
        Returns:
            [B, embed_dim] conditioning vector
        """
        if pitch.dim() == 1:
            pitch = pitch.unsqueeze(-1)
        # Normalize to roughly [-1, 1] centered around middle C (60)
        pitch_norm = (pitch.float() - 60) / 24.0
        return self.mlp(pitch_norm)


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


class DiscriminatorBlock(nn.Module):
    """Residual conv block for discriminator."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)

        if in_ch != out_ch or downsample:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.leaky_relu(h, 0.2)
        h = self.conv2(h)
        h = self.norm2(h)
        return F.leaky_relu(h + self.shortcut(x), 0.2)


class RegisterDiscriminator(nn.Module):
    """
    Discriminator conditioned on target pitch.

    Learns to distinguish:
    - REAL: Natural trumpet latent at pitch X
    - FAKE: DSP-shifted (or corrected) latent claiming to be at pitch X

    Architecture:
    - C3-focused path (main register signal)
    - Full latent path (context)
    - Pitch conditioning via FiLM
    - Outputs real/fake logits
    """

    def __init__(
        self,
        latent_channels: int = 8,
        latent_height: int = 16,
        hidden_dim: int = 128,
        cond_dim: int = 64,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.cond_dim = cond_dim

        # Pitch embedding
        self.pitch_embed = PitchEmbedding(cond_dim)

        # === C3-FOCUSED PATH ===
        # C3 is channel 3, the main register channel
        self.c3_conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            DiscriminatorBlock(hidden_dim // 2, hidden_dim // 2),
            DiscriminatorBlock(hidden_dim // 2, hidden_dim, downsample=True),
        )

        # === FULL LATENT PATH ===
        self.full_conv = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            DiscriminatorBlock(hidden_dim // 2, hidden_dim // 2),
            DiscriminatorBlock(hidden_dim // 2, hidden_dim, downsample=True),
        )

        # === FUSION + CONDITIONING ===
        # FiLM conditioning on pitch
        self.film_gamma = nn.Linear(cond_dim, hidden_dim * 2)
        self.film_beta = nn.Linear(cond_dim, hidden_dim * 2)

        # Final layers
        self.final_conv = nn.Sequential(
            DiscriminatorBlock(hidden_dim * 2, hidden_dim * 2),
            DiscriminatorBlock(hidden_dim * 2, hidden_dim * 2, downsample=True),
        )

        # Global pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        latent: torch.Tensor,
        target_pitch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latent: [B, 8, 16, T] latent tensor
            target_pitch: [B] target MIDI pitch (what pitch this SHOULD sound like)

        Returns:
            [B, 1] real/fake logits (higher = more real)
        """
        B = latent.shape[0]

        # C3 path
        c3 = latent[:, 3:4, :, :]  # [B, 1, H, T]
        c3_feat = self.c3_conv(c3)  # [B, hidden, H', T']

        # Full latent path
        full_feat = self.full_conv(latent)  # [B, hidden, H', T']

        # Concatenate
        combined = torch.cat([c3_feat, full_feat], dim=1)  # [B, hidden*2, H', T']

        # Pitch conditioning via FiLM
        pitch_emb = self.pitch_embed(target_pitch)  # [B, cond_dim]
        gamma = self.film_gamma(pitch_emb).view(B, -1, 1, 1)
        beta = self.film_beta(pitch_emb).view(B, -1, 1, 1)
        combined = combined * (1 + gamma) + beta

        # Final convolutions
        h = self.final_conv(combined)

        # Classify
        return self.classifier(h)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for better gradient flow.

    Uses multiple discriminators at different temporal scales.
    """

    def __init__(self, num_scales: int = 3, **kwargs):
        super().__init__()

        self.discriminators = nn.ModuleList([
            RegisterDiscriminator(**kwargs)
            for _ in range(num_scales)
        ])

        self.downsamplers = nn.ModuleList([
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
            for _ in range(num_scales - 1)
        ])

    def forward(
        self,
        latent: torch.Tensor,
        target_pitch: torch.Tensor,
    ) -> list:
        """Returns list of logits from each scale."""
        outputs = []
        x = latent

        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x, target_pitch))
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)

        return outputs


# =============================================================================
# Generator (Corrector) - reuse from models_simple but with modifications
# =============================================================================

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


class PitchShiftCorrectorGAN(nn.Module):
    """
    Generator for adversarial pitch shift correction.

    Conditioned on both:
    - shift: how many semitones was the DSP shift
    - target_pitch: what pitch should this sound like (CRITICAL for register correction)

    Key insight: DSP shift corrupts the pitch info in the input latent.
    The model needs EXPLICIT target pitch conditioning to know what register
    the output should sound like. Without this, it can't know where to put energy.

    Pure residual: output = source + learned_correction
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

        # Conditioning embeddings
        self.shift_embed = ShiftEmbedding(cond_dim)
        self.pitch_embed = PitchEmbedding(cond_dim)  # Target pitch embedding

        # Combine shift + pitch conditioning
        self.cond_combine = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
        )

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

        # Output projection (residual)
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, latent_channels, 1),
        )

        # Initialize output to near-zero
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        source_latent: torch.Tensor,
        shift: torch.Tensor,
        target_pitch: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            source_latent: [B, 8, 16, T] DSP-shifted latent
            shift: [B] shift in semitones
            target_pitch: [B] target MIDI pitch (what the output should sound like)
                         If None, uses shift-only conditioning (legacy mode)

        Returns:
            corrected_latent: [B, 8, 16, T]
        """
        # Get conditioning
        shift_emb = self.shift_embed(shift)

        if target_pitch is not None:
            # Full conditioning: shift + target pitch
            pitch_emb = self.pitch_embed(target_pitch)
            cond = self.cond_combine(torch.cat([shift_emb, pitch_emb], dim=-1))
        else:
            # Legacy mode: shift-only
            cond = shift_emb

        # Process
        h = self.input_proj(source_latent)

        for block in self.blocks:
            h = block(h, cond)

        residual = self.output_proj(h)

        # Pure residual connection
        return source_latent + residual


# =============================================================================
# Loss Functions
# =============================================================================

class GANLoss(nn.Module):
    """Standard GAN losses."""

    def __init__(self, loss_type: str = 'hinge'):
        super().__init__()
        self.loss_type = loss_type

    def discriminator_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Discriminator tries to classify real as real, fake as fake."""
        if self.loss_type == 'hinge':
            real_loss = F.relu(1.0 - real_logits).mean()
            fake_loss = F.relu(1.0 + fake_logits).mean()
        elif self.loss_type == 'bce':
            real_loss = F.binary_cross_entropy_with_logits(
                real_logits, torch.ones_like(real_logits))
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_logits, torch.zeros_like(fake_logits))
        else:  # lsgan
            real_loss = F.mse_loss(real_logits, torch.ones_like(real_logits))
            fake_loss = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))

        return real_loss + fake_loss

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Generator tries to make fake look real."""
        if self.loss_type == 'hinge':
            return -fake_logits.mean()
        elif self.loss_type == 'bce':
            return F.binary_cross_entropy_with_logits(
                fake_logits, torch.ones_like(fake_logits))
        else:  # lsgan
            return F.mse_loss(fake_logits, torch.ones_like(fake_logits))


class ContentPreservationLoss(nn.Module):
    """
    Lightweight content preservation.

    Ensures corrected latent doesn't deviate too far from source.
    Uses correlation rather than MSE to allow energy changes.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        B = pred.shape[0]

        # Flatten each sample
        pred_flat = pred.reshape(B, -1)
        source_flat = source.reshape(B, -1)

        # Correlation loss (1 - corr, so higher corr = lower loss)
        pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        source_centered = source_flat - source_flat.mean(dim=1, keepdim=True)

        pred_std = pred_centered.std(dim=1, keepdim=True) + 1e-8
        source_std = source_centered.std(dim=1, keepdim=True) + 1e-8

        correlation = (pred_centered * source_centered).mean(dim=1) / (
            pred_std.squeeze() * source_std.squeeze()
        )

        # Want high correlation, so minimize (1 - corr)
        return (1 - correlation).mean()


class FramewiseSilenceLoss(nn.Module):
    """
    Don't add energy where source is silent.

    Prevents noise in gaps between notes.
    """

    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        # Frame-wise energy
        source_energy = source.pow(2).mean(dim=(1, 2))  # [B, T]
        pred_energy = pred.pow(2).mean(dim=(1, 2))

        # Where source is quiet
        silence_mask = (source_energy < self.threshold).float()

        # Penalize pred energy in silent regions
        silence_violation = (silence_mask * pred_energy).sum() / (silence_mask.sum() + 1e-6)

        return silence_violation
