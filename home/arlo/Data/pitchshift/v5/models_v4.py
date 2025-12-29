#!/usr/bin/env python3
"""
V4 Pitch Shift Corrector Models

Key difference from V3:
- Conditioning on shift amount (continuous) instead of target group (discrete)
- Input is DSP-pitch-shifted latent (has artifacts)
- Output is corrected latent (should sound like real trumpet at that pitch)

The model learns: "given audio shifted by s semitones, fix the artifacts"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftEmbedding(nn.Module):
    """Embed shift amount (scalar) into a feature vector."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, shift: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shift: [B] or [B, 1] shift amounts normalized to [-1, 1]
        Returns:
            [B, embed_dim] embedding
        """
        if shift.dim() == 1:
            shift = shift.unsqueeze(-1)
        return self.net(shift)


class ConvBlock(nn.Module):
    """Conv block with shift conditioning via FiLM."""

    def __init__(self, channels: int, shift_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

        # FiLM modulation from shift embedding
        self.film = nn.Linear(shift_dim, channels * 2)

    def forward(self, x: torch.Tensor, shift_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, T]
            shift_embed: [B, shift_dim]
        """
        # FiLM parameters
        film_params = self.film(shift_embed)  # [B, C*2]
        gamma, beta = film_params.chunk(2, dim=-1)  # [B, C] each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # First conv with FiLM
        h = self.conv1(x)
        h = self.norm1(h)
        h = h * (1 + gamma) + beta  # FiLM modulation
        h = F.silu(h)

        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return x + h  # Residual


class PitchShiftCorrector(nn.Module):
    """
    Main V4 model: corrects pitch-shift artifacts.

    Architecture: Adaptive gated residual with shift conditioning.
    - Transform branch: learns the correction conditioned on shift amount
    - Alpha gate: decides how much correction to apply (more for larger |s|)

    Input: latent of DSP-pitch-shifted audio [B, 8, 16, T]
    Conditioning: shift amount s (normalized to [-1, 1])
    Output: corrected latent [B, 8, 16, T]
    """

    def __init__(
        self,
        latent_channels: int = 8,
        latent_height: int = 16,
        hidden_dim: int = 256,
        shift_embed_dim: int = 128,
        num_blocks: int = 4,
        alpha_min: float = 0.1,
        alpha_max: float = 0.9,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Summary stats for alpha: per-channel mean + std + global RMS = 2*C + 1 = 17
        alpha_input_dim = shift_embed_dim + 2 * latent_channels + 1

        # Shift embedding
        self.shift_embed = ShiftEmbedding(shift_embed_dim)

        # Input projection (includes shift info)
        self.input_proj = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        # Residual blocks with shift conditioning
        self.blocks = nn.ModuleList([
            ConvBlock(hidden_dim, shift_embed_dim)
            for _ in range(num_blocks)
        ])

        # Transform output
        self.transform_out = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, latent_channels, 1),
        )

        # Alpha gate (how much to correct, conditioned on shift + source stats)
        # Uses meaningful summary statistics instead of arbitrary slice
        self.alpha_net = nn.Sequential(
            nn.Linear(alpha_input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Learnable bias towards identity (start with small corrections)
        self.alpha_bias = nn.Parameter(torch.tensor(1.0))  # sigmoid(1) ≈ 0.73

        self._init_weights()

    def _init_weights(self):
        """Initialize for identity-biased start."""
        # Zero-init transform output for identity start
        nn.init.zeros_(self.transform_out[-1].weight)
        nn.init.zeros_(self.transform_out[-1].bias)

    def forward(
        self,
        source_latent: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            source_latent: [B, C, H, T] pitch-shifted latent with artifacts
            shift: [B] shift amounts in semitones (will be normalized)

        Returns:
            corrected_latent: [B, C, H, T]
        """
        B, C, H, T = source_latent.shape

        # Normalize shift to [-1, 1] (assuming max shift is ±12)
        shift_norm = shift.float() / 12.0
        shift_norm = shift_norm.clamp(-1, 1)

        # Get shift embedding
        shift_emb = self.shift_embed(shift_norm)  # [B, shift_embed_dim]

        # Transform branch
        h = self.input_proj(source_latent)

        for block in self.blocks:
            h = block(h, shift_emb)

        transformed = self.transform_out(h)

        # Alpha gate: how much to correct
        # Use meaningful summary stats instead of arbitrary slice
        # Per-channel mean: [B, C]
        source_mean = source_latent.mean(dim=(2, 3))  # [B, C]
        # Per-channel std: [B, C]
        source_std = source_latent.std(dim=(2, 3))    # [B, C]
        # Global RMS: [B, 1]
        source_rms = source_latent.pow(2).mean().sqrt().view(1, 1).expand(B, 1)

        # Concatenate: [B, shift_embed_dim + 2*C + 1]
        alpha_input = torch.cat([shift_emb, source_mean, source_std, source_rms], dim=-1)
        alpha_logits = self.alpha_net(alpha_input).squeeze(-1)

        # Apply bias and sigmoid, then clamp
        alpha = torch.sigmoid(alpha_logits + self.alpha_bias)
        alpha = alpha.clamp(self.alpha_min, self.alpha_max)
        alpha = alpha.view(B, 1, 1, 1)

        # Gated mixing: alpha=1 means keep source, alpha=0 means use transformed
        # For corrections: we want alpha closer to 0 for large shifts
        # So invert: output = alpha * source + (1 - alpha) * transformed
        # Actually, let's think about this:
        # - shift=0: no correction needed, want output ≈ source, so alpha should be high
        # - shift=±12: big correction needed, want output ≈ transformed, so alpha should be low
        # The alpha_net should learn this from the shift embedding

        output = alpha * source_latent + (1 - alpha) * transformed

        return output

    def get_alpha(
        self,
        source_latent: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        """Get the alpha values for debugging."""
        B = source_latent.shape[0]
        shift_norm = shift.float() / 12.0
        shift_emb = self.shift_embed(shift_norm)

        # Use same summary stats as forward
        source_mean = source_latent.mean(dim=(2, 3))
        source_std = source_latent.std(dim=(2, 3))
        source_rms = source_latent.pow(2).mean().sqrt().view(1, 1).expand(B, 1)

        alpha_input = torch.cat([shift_emb, source_mean, source_std, source_rms], dim=-1)
        alpha_logits = self.alpha_net(alpha_input).squeeze(-1)
        alpha = torch.sigmoid(alpha_logits + self.alpha_bias)
        return alpha.clamp(self.alpha_min, self.alpha_max)


# ============================================================================
# Loss functions (ported from mute_translator's proven approach)
# ============================================================================

class DistributionLoss(nn.Module):
    """
    HF-emphasis distribution loss (ported from mute_translator).

    Uses frequency weighting:
    - HF (upper 50%): 3x weight for brightness
    - Mid-high (25-50%): 2x weight for nasal resonance
    - Body (lower 25%): 1x weight

    Also includes:
    - Per-channel mean/std matching
    - Mid-high energy matching
    - High-frequency energy matching
    - Simple MMD for distribution matching
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, T = pred.shape

        # Frequency band indices
        mh_start = H // 4       # 25% of H
        mh_end = H // 2         # 50% of H: mid-high (nasal resonance)
        hf_start = H // 2       # 50-100%: high frequencies (brightness)

        # Create frequency weighting
        freq_weights = torch.ones(H, device=pred.device)
        freq_weights[mh_start:mh_end] = 2.0  # Mid-high: 2x
        freq_weights[hf_start:] = 3.0        # HF brightness: 3x
        freq_weights = freq_weights.view(1, 1, H, 1)

        # Weighted MSE loss
        weighted_diff = ((pred - target) ** 2) * freq_weights
        hf_mse_loss = weighted_diff.mean()

        # Per-channel mean/std matching
        pred_mean = pred.mean(dim=(2, 3))
        pred_std = pred.std(dim=(2, 3))
        target_mean = target.mean(dim=(2, 3))
        target_std = target.std(dim=(2, 3))

        mean_loss = F.mse_loss(pred_mean, target_mean)
        std_loss = F.mse_loss(pred_std, target_std)

        # Mid-high energy (character)
        pred_mh_energy = pred[:, :, mh_start:mh_end, :].abs().mean(dim=(2, 3))
        target_mh_energy = target[:, :, mh_start:mh_end, :].abs().mean(dim=(2, 3))
        mh_energy_loss = F.mse_loss(pred_mh_energy, target_mh_energy)

        # High-frequency energy (brightness)
        pred_hf_energy = pred[:, :, hf_start:, :].abs().mean(dim=(2, 3))
        target_hf_energy = target[:, :, hf_start:, :].abs().mean(dim=(2, 3))
        hf_energy_loss = F.mse_loss(pred_hf_energy, target_hf_energy)

        # Simple MMD (fp32 for stability)
        pred_flat = pred.reshape(B, -1).float()
        target_flat = target.reshape(B, -1).float()
        scale = pred_flat.shape[1] ** 0.5
        pred_flat = pred_flat / scale
        target_flat = target_flat / scale
        pred_gram = torch.mm(pred_flat, pred_flat.t())
        target_gram = torch.mm(target_flat, target_flat.t())
        cross_gram = torch.mm(pred_flat, target_flat.t())
        mmd = pred_gram.mean() + target_gram.mean() - 2 * cross_gram.mean()

        # Combined loss
        return (
            hf_mse_loss
            + mean_loss
            + std_loss
            + 0.3 * mh_energy_loss
            + 0.5 * hf_energy_loss
            + 0.1 * mmd
        )


class SilenceLoss(nn.Module):
    """
    Framewise silence constraint loss (ported from mute_translator).

    Penalizes output energy when input is silent on a per-frame basis.
    This prevents buzz/noise in gaps between notes.
    """

    def __init__(self, threshold: float = 0.05):
        super().__init__()
        self.threshold = threshold

    def forward(self, source: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        # Compute per-frame energy (RMS-like across channels and height)
        # source: [B, C, H, T] -> [B, T]
        source_energy = source.pow(2).mean(dim=(1, 2)).sqrt()   # [B, T]
        pred_energy = pred.pow(2).mean(dim=(1, 2)).sqrt()       # [B, T]

        # Identify silent frames in the input (energy below threshold)
        silence_mask = (source_energy < self.threshold).float()  # [B, T]

        # Penalize any output energy in those silent frames
        # pred_energy.pow(2) penalizes harder for louder output in silence
        silence_loss = (silence_mask * pred_energy.pow(2)).sum() / (silence_mask.sum() + 1e-6)

        return silence_loss


class EnvelopePreservationLoss(nn.Module):
    """
    RMS envelope preservation loss (ported from mute_translator).

    Matches RMS envelope between predicted and target latents.
    Uses avg_pool1d for smoothing, NOT normalized to max (preserves dynamics).
    """

    def __init__(self, window_size: int = 8):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, T = pred.shape

        # Compute per-frame energy: [B, C, H, T] -> [B, T]
        pred_energy = pred.pow(2).mean(dim=(1, 2))   # [B, T]
        target_energy = target.pow(2).mean(dim=(1, 2)) # [B, T]

        # Compute RMS envelope using 1D avg pooling over time
        # Reshape for pooling: [B, 1, T]
        pred_energy = pred_energy.unsqueeze(1)
        target_energy = target_energy.unsqueeze(1)

        # Use same padding to keep T dimension
        padding = self.window_size // 2
        pred_rms = F.avg_pool1d(pred_energy, self.window_size, stride=1, padding=padding)
        target_rms = F.avg_pool1d(target_energy, self.window_size, stride=1, padding=padding)

        # Trim to original size if needed
        pred_rms = pred_rms[:, :, :T]
        target_rms = target_rms[:, :, :T]

        # Take sqrt for RMS
        pred_rms = pred_rms.sqrt()
        target_rms = target_rms.sqrt()

        # L1 loss between envelopes (NOT normalized - preserves dynamics)
        return F.l1_loss(pred_rms, target_rms)


class ContentPreservationLoss(nn.Module):
    """Preserve content structure via temporal and spectral gradients."""

    def __init__(self):
        super().__init__()

    def forward(self, source: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        # Temporal gradient (rhythm preservation)
        source_grad = source[:, :, :, 1:] - source[:, :, :, :-1]
        pred_grad = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_loss = F.mse_loss(pred_grad.abs(), source_grad.abs())

        # Spectral gradient (pitch contour preservation)
        source_spec_grad = source[:, :, 1:, :] - source[:, :, :-1, :]
        pred_spec_grad = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        spec_loss = F.mse_loss(pred_spec_grad.abs(), source_spec_grad.abs())

        return grad_loss + spec_loss


class AlphaRegularizationLoss(nn.Module):
    """
    Regularize alpha to match expected behavior based on shift magnitude.

    For pitch shift correction:
    - shift=0: alpha should be high (keep source, minimal correction)
    - large |shift|: alpha should be lower (more correction needed)

    Target: alpha_target = alpha_max - (alpha_max - alpha_min) * |shift|/max_shift
    """

    def __init__(self, alpha_max: float = 0.8, alpha_min: float = 0.2, max_shift: float = 12.0):
        super().__init__()
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.max_shift = max_shift

    def forward(self, alpha: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        # Compute target alpha based on shift magnitude
        shift_abs = shift.abs().float()
        shift_ratio = (shift_abs / self.max_shift).clamp(0, 1)

        # Target: high alpha for small shift, low alpha for large shift
        alpha_target = self.alpha_max - (self.alpha_max - self.alpha_min) * shift_ratio

        # L2 loss to encourage alpha to follow expected pattern
        return F.mse_loss(alpha.squeeze(), alpha_target)
