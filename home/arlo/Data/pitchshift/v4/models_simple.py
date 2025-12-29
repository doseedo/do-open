#!/usr/bin/env python3
"""
V4 Pitch Shift Corrector - With Optional Pitch Conditioning

Two modes:
1. conditioning=False: Mute-style, learns to fix artifacts from appearance
2. conditioning=True: FiLM-conditioned on shift amount for clearer training signal

Architecture: Pure residual, pred = source + f(source, [shift])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv residual block (no conditioning)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return x + h


class FiLMConvBlock(nn.Module):
    """Conv residual block with FiLM conditioning."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

        # FiLM: condition -> scale and shift for each norm
        self.film1 = nn.Linear(cond_dim, channels * 2)
        self.film2 = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # First conv + FiLM
        h = self.conv1(x)
        h = self.norm1(h)
        film_params = self.film1(cond)  # [B, channels*2]
        gamma1, beta1 = film_params.chunk(2, dim=-1)
        h = h * (1 + gamma1[:, :, None, None]) + beta1[:, :, None, None]
        h = F.silu(h)

        # Second conv + FiLM
        h = self.conv2(h)
        h = self.norm2(h)
        film_params = self.film2(cond)
        gamma2, beta2 = film_params.chunk(2, dim=-1)
        h = h * (1 + gamma2[:, :, None, None]) + beta2[:, :, None, None]
        h = F.silu(h)

        return x + h


class ShiftEmbedding(nn.Module):
    """Embed shift value into conditioning vector."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, shift: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shift: [B] or [B, 1] shift in semitones
        Returns:
            [B, embed_dim] conditioning vector
        """
        if shift.dim() == 1:
            shift = shift.unsqueeze(-1)
        # Normalize to [-1, 1] range (assuming max shift is 12)
        shift_norm = shift.float() / 12.0
        return self.mlp(shift_norm)


class PitchShiftCorrectorSimple(nn.Module):
    """
    Artifact corrector with optional pitch conditioning.

    Pure residual: pred = source + residual(source, [shift])

    Args:
        conditioning: If True, use FiLM conditioning on shift amount
                     If False, mute-style (no conditioning)

    Input: latent of DSP-pitch-shifted audio [B, 8, 16, T]
    Output: corrected latent [B, 8, 16, T]
    """

    def __init__(
        self,
        latent_channels: int = 8,
        latent_height: int = 16,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        conditioning: bool = False,
        cond_dim: int = 64,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.conditioning = conditioning
        self.cond_dim = cond_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        # Conditioning embedding (only used if conditioning=True)
        if conditioning:
            self.shift_embed = ShiftEmbedding(cond_dim)
            self.blocks = nn.ModuleList([
                FiLMConvBlock(hidden_dim, cond_dim)
                for _ in range(num_blocks)
            ])
        else:
            self.shift_embed = None
            self.blocks = nn.ModuleList([
                ConvBlock(hidden_dim)
                for _ in range(num_blocks)
            ])

        # Output projection (residual to add to source)
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, latent_channels, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize output to near-zero for identity start."""
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        source_latent: torch.Tensor,
        shift: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            source_latent: [B, C, H, T] pitch-shifted latent with artifacts
            shift: [B] shift in semitones (required if conditioning=True)

        Returns:
            corrected_latent: [B, C, H, T]
        """
        h = self.input_proj(source_latent)

        # Get conditioning if enabled
        cond = None
        if self.conditioning:
            if shift is None:
                raise ValueError("shift required when conditioning=True")
            cond = self.shift_embed(shift)

        for block in self.blocks:
            h = block(h, cond)

        residual = self.output_proj(h)

        # Pure residual connection
        return source_latent + residual


# ============================================================================
# Loss functions (same as V4, but simplified interface)
# ============================================================================

class DistributionLoss(nn.Module):
    """
    Distribution matching loss for non-aligned targets.

    Key insight: Normalize both pred and target to unit variance before comparing.
    This matches SPECTRAL SHAPE only, not absolute energy levels.
    Dynamics are preserved separately via DynamicsLoss.

    Compares:
    - Per-channel mean (after normalization)
    - Relative energy in frequency bands
    - MMD for overall distribution shape
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, T = pred.shape

        # Normalize to unit std (compare shape, not absolute levels)
        pred_std = pred.std() + 1e-8
        target_std = target.std() + 1e-8
        pred_norm = pred / pred_std
        target_norm = target / target_std

        # Frequency band indices
        mh_start = H // 4
        mh_end = H // 2
        hf_start = H // 2

        # Per-channel mean matching (on normalized)
        pred_mean = pred_norm.mean(dim=(2, 3))
        target_mean = target_norm.mean(dim=(2, 3))
        mean_loss = F.mse_loss(pred_mean, target_mean)

        # Per-channel std matching (on normalized - measures relative channel balance)
        pred_ch_std = pred_norm.std(dim=(2, 3))
        target_ch_std = target_norm.std(dim=(2, 3))
        std_loss = F.mse_loss(pred_ch_std, target_ch_std)

        # Mid-high relative energy (character/nasal resonance)
        pred_mh_energy = pred_norm[:, :, mh_start:mh_end, :].abs().mean(dim=(2, 3))
        target_mh_energy = target_norm[:, :, mh_start:mh_end, :].abs().mean(dim=(2, 3))
        mh_energy_loss = F.mse_loss(pred_mh_energy, target_mh_energy)

        # High-frequency relative energy (brightness)
        pred_hf_energy = pred_norm[:, :, hf_start:, :].abs().mean(dim=(2, 3))
        target_hf_energy = target_norm[:, :, hf_start:, :].abs().mean(dim=(2, 3))
        hf_energy_loss = F.mse_loss(pred_hf_energy, target_hf_energy)

        # MMD for distribution matching (on normalized, fp32 for stability)
        pred_flat = pred_norm.reshape(B, -1).float()
        target_flat = target_norm.reshape(B, -1).float()
        scale = pred_flat.shape[1] ** 0.5
        pred_flat = pred_flat / scale
        target_flat = target_flat / scale
        pred_gram = torch.mm(pred_flat, pred_flat.t())
        target_gram = torch.mm(target_flat, target_flat.t())
        cross_gram = torch.mm(pred_flat, target_flat.t())
        mmd = pred_gram.mean() + target_gram.mean() - 2 * cross_gram.mean()

        return {
            'total': mean_loss + std_loss + 0.3 * mh_energy_loss + 0.5 * hf_energy_loss + 0.1 * mmd,
            'mean': mean_loss,
            'std': std_loss,
            'mh_energy': mh_energy_loss,
            'hf_energy': hf_energy_loss,
            'mmd': mmd,
        }


class DynamicsLoss(nn.Module):
    """
    Preserve source dynamics (energy level).

    Pred should have same overall energy as source, even though
    spectral shape matches target.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        pred_std = pred.std()
        source_std = source.std()
        return F.mse_loss(pred_std, source_std)


class DirectionalCentroidLoss(nn.Module):
    """
    Directional loss that pushes H-dimension energy in the correct direction
    based on pitch shift.

    Key insight from DCAE latent analysis:
    - HIGH register (high pitch) → MORE energy in LOW-H, LESS in HIGH-H
    - LOW register (low pitch) → LESS energy in LOW-H, MORE in HIGH-H

    Therefore:
    - +shift (shifting UP) → should DECREASE centroid (move energy to low-H)
    - -shift (shifting DOWN) → should INCREASE centroid (move energy to high-H)

    This is OPPOSITE of acoustic frequency intuition but matches DCAE's encoding.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        source: torch.Tensor,
        shift: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, T] predicted latent
            source: [B, C, H, T] source latent (before correction)
            shift: [B] shift in semitones
        """
        B, C, H, T = pred.shape

        # Compute spectral centroid proxy in latent space
        h_weights = torch.linspace(0, 1, H, device=pred.device).view(1, 1, H, 1)

        # Centroid = weighted mean of energy along H dimension
        pred_energy = pred.abs()
        source_energy = source.abs()

        pred_centroid = (pred_energy * h_weights).sum(dim=2) / (pred_energy.sum(dim=2) + 1e-8)
        source_centroid = (source_energy * h_weights).sum(dim=2) / (source_energy.sum(dim=2) + 1e-8)

        # Average over channels and time
        pred_centroid = pred_centroid.mean(dim=(1, 2))  # [B]
        source_centroid = source_centroid.mean(dim=(1, 2))  # [B]

        # Centroid change
        centroid_diff = pred_centroid - source_centroid  # + means more high-H energy

        # DCAE pattern: +shift → need LOWER centroid (more low-H, less high-H)
        # So +shift and +centroid_diff = WRONG direction
        # Penalize when shift and centroid_diff have SAME sign
        shift_sign = shift.sign()  # +1, 0, or -1
        wrong_direction = F.relu(shift_sign * centroid_diff)

        # Also add low-H energy directional constraint (should increase for +shift)
        lf_end = H // 2
        pred_lf = pred[:, :, :lf_end, :].abs().mean(dim=(1, 2, 3))
        source_lf = source[:, :, :lf_end, :].abs().mean(dim=(1, 2, 3))
        lf_diff = pred_lf - source_lf  # + means more low-H energy

        # +shift should have +lf_diff (more low-H energy)
        # Penalize when signs DON'T match
        wrong_lf_direction = F.relu(-shift_sign * lf_diff)

        return wrong_direction.mean() + 0.5 * wrong_lf_direction.mean()


class SilenceLoss(nn.Module):
    """
    Framewise silence constraint loss.

    Penalizes output energy when input is silent.
    Prevents buzz/noise in gaps between notes.
    """

    def __init__(self, threshold: float = 0.05):
        super().__init__()
        self.threshold = threshold

    def forward(self, source: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        source_energy = source.pow(2).mean(dim=(1, 2)).sqrt()
        pred_energy = pred.pow(2).mean(dim=(1, 2)).sqrt()

        silence_mask = (source_energy < self.threshold).float()
        silence_loss = (silence_mask * pred_energy.pow(2)).sum() / (silence_mask.sum() + 1e-6)

        return silence_loss


class EnvelopePreservationLoss(nn.Module):
    """
    RMS envelope preservation loss.

    Matches RMS envelope between predicted and SOURCE (not target).
    This ensures the model preserves the input's dynamics rather than
    trying to match a non-aligned target's envelope.
    """

    def __init__(self, window_size: int = 8):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, T] predicted/corrected latent
            source: [B, C, H, T] source latent (NOT target!)
        """
        B, C, H, T = pred.shape

        pred_energy = pred.pow(2).mean(dim=(1, 2)).unsqueeze(1)
        source_energy = source.pow(2).mean(dim=(1, 2)).unsqueeze(1)

        padding = self.window_size // 2
        pred_rms = F.avg_pool1d(pred_energy, self.window_size, stride=1, padding=padding)
        source_rms = F.avg_pool1d(source_energy, self.window_size, stride=1, padding=padding)

        pred_rms = pred_rms[:, :, :T].sqrt()
        source_rms = source_rms[:, :, :T].sqrt()

        return F.l1_loss(pred_rms, source_rms)
