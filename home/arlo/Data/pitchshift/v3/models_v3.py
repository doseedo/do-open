"""
Range-Group Pitch Shift Models V3

Like mute_translator but with group conditioning.

Input: latent from any range group + target group ID
Output: latent with target group's register characteristics

Architecture similar to MuteTranslator but with group embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ResidualBlock1D(nn.Module):
    """1D Residual block for temporal latent processing."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
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


class RangeGroupTranslator(nn.Module):
    """
    Range-group based register translator.

    Like MuteTranslator but with group conditioning:
    - Input: source latent + target group ID
    - Output: latent with target group's characteristics

    Uses residual connection: output = dry_scale * source + residual_scale * residual
    The model learns what each range group "sounds like".
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        num_groups: int = 8,  # Number of range groups
        group_embed_dim: int = 32,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.num_groups = num_groups

        # Group embedding (like pitch embedding but for range groups)
        self.group_embed = nn.Embedding(num_groups, group_embed_dim)

        # Project group embedding to conditioning
        self.group_proj = nn.Sequential(
            nn.Linear(group_embed_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Input projection (latent + group conditioning)
        self.input_proj = nn.Conv1d(latent_channels + hidden_channels, hidden_channels, 1)

        # Residual blocks with increasing dilation
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_channels, kernel_size, dilation=2 ** (i % 4))
            for i in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, latent_channels, 1),
        )

        # Initialize output near-zero for stable residual learning
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        # Learnable scales (like mute_translator)
        self.dry_scale = nn.Parameter(torch.tensor(1.0))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # Hard clamp on residual_scale to prevent runaway (learned from V2 failure)
        self.max_residual_scale = 0.3

    def forward(
        self,
        source_latent: torch.Tensor,
        target_group: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            source_latent: [B, C, H, T] latent from any range group
            target_group: [B] target group ID (0 to num_groups-1)

        Returns:
            output_latent: [B, C, H, T] with target group's characteristics
        """
        B, C, H, T = source_latent.shape

        # Get group conditioning
        group_emb = self.group_embed(target_group)  # [B, embed_dim]
        group_cond = self.group_proj(group_emb)  # [B, hidden]
        group_cond = group_cond.unsqueeze(-1).expand(-1, -1, T)  # [B, hidden, T]

        # Flatten H into batch: [B*H, C, T]
        x = source_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)

        # Expand group conditioning for each H slice
        group_cond_expanded = group_cond.unsqueeze(1).expand(-1, H, -1, -1)
        group_cond_expanded = group_cond_expanded.reshape(B * H, -1, T)

        # Concatenate and process
        x = torch.cat([x, group_cond_expanded], dim=1)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        residual = self.output_proj(x)

        # Reshape back: [B, H, C, T] -> [B, C, H, T]
        residual = residual.reshape(B, H, C, T).permute(0, 2, 1, 3)

        # Apply with learnable scales (like mute_translator)
        # Clamp residual_scale to prevent runaway
        clamped_residual_scale = self.residual_scale.clamp(max=self.max_residual_scale)
        output = self.dry_scale * source_latent + clamped_residual_scale * residual

        return output


class RangeGroupTranslatorDirect(nn.Module):
    """
    Direct (non-residual) range-group translator.

    output = f(input, target_group) - no passthrough of input.
    Better for more aggressive timbre changes.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        num_groups: int = 8,
        group_embed_dim: int = 32,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.num_groups = num_groups

        self.group_embed = nn.Embedding(num_groups, group_embed_dim)

        self.group_proj = nn.Sequential(
            nn.Linear(group_embed_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.input_proj = nn.Sequential(
            nn.Conv1d(latent_channels + hidden_channels, hidden_channels, 1),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_channels, kernel_size, dilation=2 ** (i % 4))
            for i in range(num_blocks)
        ])

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, latent_channels, 1),
        )

        # Small init for stability
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        source_latent: torch.Tensor,
        target_group: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, T = source_latent.shape

        # Group conditioning
        group_emb = self.group_embed(target_group)
        group_cond = self.group_proj(group_emb)
        group_cond = group_cond.unsqueeze(-1).expand(-1, -1, T)

        # Flatten H into batch
        x = source_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)
        group_cond_expanded = group_cond.unsqueeze(1).expand(-1, H, -1, -1).reshape(B * H, -1, T)

        # Process
        x = torch.cat([x, group_cond_expanded], dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        output = self.output_proj(x)

        # Reshape back
        output = output.reshape(B, H, C, T).permute(0, 2, 1, 3)

        return output


class RangeGroupTranslatorAdaptive(nn.Module):
    """
    Adaptive mixing range-group translator (most robust architecture).

    Key insight from mute_translator analysis:
    - Pure residual can still blow up if distribution loss rewards large changes
    - Per-frame adaptive mixing with hard clamps prevents this

    output = alpha(t) * source + (1 - alpha(t)) * transformed

    Where:
    - alpha is learned per-frame, clamped to [0.1, 0.9]
    - alpha_net sees both input content AND target group
    - transform branch sees input AND target group conditioning

    This is the same architecture that made mute_translator robust.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        num_groups: int = 8,
        group_embed_dim: int = 32,
        alpha_init: float = 0.5,  # Start balanced
        alpha_min: float = 0.1,
        alpha_max: float = 0.9,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.num_groups = num_groups
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Group embedding (shared between transform and alpha branches)
        self.group_embed = nn.Embedding(num_groups, group_embed_dim)

        # Project group embedding for transform branch
        self.group_proj = nn.Sequential(
            nn.Linear(group_embed_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Transform branch: learns f(source, target_group)
        self.input_proj = nn.Conv1d(latent_channels + hidden_channels, hidden_channels, 1)

        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_channels, kernel_size, dilation=2 ** (i % 4))
            for i in range(num_blocks)
        ])

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, latent_channels, 1),
        )

        # Alpha gate: learns per-frame mixing based on input + target group
        # Input: (source averaged over H) + group embedding
        self.alpha_net = nn.Sequential(
            nn.Conv1d(latent_channels + group_embed_dim, hidden_channels // 2, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden_channels // 2, hidden_channels // 2, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden_channels // 2, 1, kernel_size=1),
        )

        # Learnable bias toward desired alpha
        self.alpha_bias = nn.Parameter(torch.tensor(alpha_init))

        # Initialize transform output small (not zero - we need to learn full output)
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        source_latent: torch.Tensor,
        target_group: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            source_latent: [B, C, H, T] latent from any range group
            target_group: [B] target group ID

        Returns:
            output_latent: [B, C, H, T] adaptively mixed output
        """
        B, C, H, T = source_latent.shape

        # Get group embedding
        group_emb = self.group_embed(target_group)  # [B, embed_dim]
        group_cond = self.group_proj(group_emb)  # [B, hidden]
        group_cond = group_cond.unsqueeze(-1).expand(-1, -1, T)  # [B, hidden, T]

        # === Transform branch ===
        # Flatten H into batch: [B*H, C, T]
        x = source_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)

        # Expand group conditioning for each H slice
        group_cond_expanded = group_cond.unsqueeze(1).expand(-1, H, -1, -1)
        group_cond_expanded = group_cond_expanded.reshape(B * H, -1, T)

        # Process through transform network
        x = torch.cat([x, group_cond_expanded], dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        transformed = self.output_proj(x)

        # Reshape back: [B, H, C, T] -> [B, C, H, T]
        transformed = transformed.reshape(B, H, C, T).permute(0, 2, 1, 3)

        # === Alpha gate ===
        # Average source over H for alpha prediction: [B, C, T]
        source_for_alpha = source_latent.mean(dim=2)

        # Expand group embedding for concatenation: [B, embed_dim, T]
        group_emb_for_alpha = group_emb.unsqueeze(-1).expand(-1, -1, T)

        # Concatenate and predict alpha
        alpha_input = torch.cat([source_for_alpha, group_emb_for_alpha], dim=1)  # [B, C+embed_dim, T]
        alpha_logits = self.alpha_net(alpha_input)  # [B, 1, T]

        # Apply bias and sigmoid, then hard clamp
        alpha = torch.sigmoid(alpha_logits + self.alpha_bias)
        alpha = alpha.clamp(self.alpha_min, self.alpha_max)

        # Expand alpha for broadcasting: [B, 1, 1, T]
        alpha = alpha.unsqueeze(2)

        # === Adaptive mixing ===
        # alpha=1 → keep source (preserve), alpha=0 → use transformed (modify)
        output = alpha * source_latent + (1 - alpha) * transformed

        return output

    def forward_with_alpha(
        self,
        source_latent: torch.Tensor,
        target_group: torch.Tensor,
    ) -> tuple:
        """Return output and alpha for visualization/debugging."""
        B, C, H, T = source_latent.shape

        group_emb = self.group_embed(target_group)
        group_cond = self.group_proj(group_emb)
        group_cond = group_cond.unsqueeze(-1).expand(-1, -1, T)

        x = source_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)
        group_cond_expanded = group_cond.unsqueeze(1).expand(-1, H, -1, -1).reshape(B * H, -1, T)

        x = torch.cat([x, group_cond_expanded], dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        transformed = self.output_proj(x)
        transformed = transformed.reshape(B, H, C, T).permute(0, 2, 1, 3)

        source_for_alpha = source_latent.mean(dim=2)
        group_emb_for_alpha = group_emb.unsqueeze(-1).expand(-1, -1, T)
        alpha_input = torch.cat([source_for_alpha, group_emb_for_alpha], dim=1)
        alpha_logits = self.alpha_net(alpha_input)
        alpha = torch.sigmoid(alpha_logits + self.alpha_bias)
        alpha = alpha.clamp(self.alpha_min, self.alpha_max)
        alpha_expanded = alpha.unsqueeze(2)

        output = alpha_expanded * source_latent + (1 - alpha_expanded) * transformed

        return output, alpha.squeeze(1)  # [B, T]


class DistributionLoss(nn.Module):
    """
    Distribution matching loss (from mute_translator).

    Matches output distribution to target distribution with frequency weighting.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_latent: [B, C, H, T] model output
            target_latent: [B, C, H, T] target from target group
        """
        B, C, H, T = pred_latent.shape

        # Frequency band weighting (upper frequencies carry more timbre info)
        freq_weights = torch.ones(H, device=pred_latent.device)
        freq_weights[H // 4:H // 2] = 1.5   # Mid-high
        freq_weights[H // 2:] = 2.0          # High frequencies
        freq_weights = freq_weights.view(1, 1, H, 1)

        # Weighted MSE loss
        weighted_diff = ((pred_latent - target_latent) ** 2) * freq_weights
        mse_loss = weighted_diff.mean()

        # Per-channel statistics matching
        pred_mean = pred_latent.mean(dim=(2, 3))
        pred_std = pred_latent.std(dim=(2, 3)) + 1e-6
        target_mean = target_latent.mean(dim=(2, 3))
        target_std = target_latent.std(dim=(2, 3)) + 1e-6

        mean_loss = F.mse_loss(pred_mean, target_mean)
        std_loss = F.mse_loss(pred_std, target_std)

        # Energy matching per frequency band
        pred_energy = pred_latent.abs().mean(dim=(0, 1, 3))  # [H]
        target_energy = target_latent.abs().mean(dim=(0, 1, 3))  # [H]
        energy_loss = F.l1_loss(pred_energy, target_energy)

        return mse_loss + mean_loss + std_loss + 0.5 * energy_loss


class SilenceLoss(nn.Module):
    """
    Silence constraint loss (from mute_translator).

    Penalizes output energy when input is silent.
    """

    def __init__(self, threshold: float = 0.05):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        source_latent: torch.Tensor,
        pred_latent: torch.Tensor,
    ) -> torch.Tensor:
        source_energy = source_latent.pow(2).mean(dim=(1, 2)).sqrt() + 1e-8  # [B, T]
        pred_energy = pred_latent.pow(2).mean(dim=(1, 2)).sqrt() + 1e-8  # [B, T]

        silence_mask = (source_energy < self.threshold).float()
        mask_sum = silence_mask.sum()

        if mask_sum < 1:
            return torch.tensor(0.0, device=source_latent.device)

        silence_loss = (silence_mask * pred_energy.pow(2)).sum() / (mask_sum + 1e-6)
        return silence_loss


class ContentPreservationLoss(nn.Module):
    """
    Content preservation loss (from mute_translator).

    Preserves temporal and spectral structure from source.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        source_latent: torch.Tensor,
        pred_latent: torch.Tensor,
    ) -> torch.Tensor:
        # Temporal gradient (rhythm preservation)
        source_grad = source_latent[:, :, :, 1:] - source_latent[:, :, :, :-1]
        pred_grad = pred_latent[:, :, :, 1:] - pred_latent[:, :, :, :-1]
        grad_loss = F.mse_loss(pred_grad.abs(), source_grad.abs())

        # Spectral gradient (pitch contour preservation)
        source_spec = source_latent[:, :, 1:, :] - source_latent[:, :, :-1, :]
        pred_spec = pred_latent[:, :, 1:, :] - pred_latent[:, :, :-1, :]
        spec_loss = F.mse_loss(pred_spec.abs(), source_spec.abs())

        return grad_loss + spec_loss


class EnvelopePreservationLoss(nn.Module):
    """
    Envelope preservation loss (from mute_translator).

    Matches RMS envelope between predicted and target.
    """

    def __init__(self, window_size: int = 8):
        super().__init__()
        self.window_size = window_size

    def forward(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, T = pred_latent.shape

        # Per-frame energy
        pred_energy = pred_latent.pow(2).mean(dim=(1, 2))  # [B, T]
        target_energy = target_latent.pow(2).mean(dim=(1, 2))  # [B, T]

        # RMS envelope via 1D pooling
        pred_energy = pred_energy.unsqueeze(1)  # [B, 1, T]
        target_energy = target_energy.unsqueeze(1)

        padding = self.window_size // 2
        pred_rms = F.avg_pool1d(pred_energy, self.window_size, stride=1, padding=padding)
        target_rms = F.avg_pool1d(target_energy, self.window_size, stride=1, padding=padding)

        pred_rms = pred_rms[:, :, :T].sqrt()
        target_rms = target_rms[:, :, :T].sqrt()

        return F.l1_loss(pred_rms, target_rms)


if __name__ == "__main__":
    print("Testing RangeGroupTranslator...")
    model = RangeGroupTranslator(num_groups=8)
    x = torch.randn(4, 8, 16, 64)  # [B, C, H, T]
    target_group = torch.randint(0, 8, (4,))
    out = model(x, target_group)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  dry_scale: {model.dry_scale.item():.3f}")
    print(f"  residual_scale: {model.residual_scale.item():.3f}")

    print("\nTesting RangeGroupTranslatorDirect...")
    model_direct = RangeGroupTranslatorDirect(num_groups=8)
    out = model_direct(x, target_group)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model_direct.parameters()):,}")

    print("\nTesting RangeGroupTranslatorAdaptive...")
    model_adaptive = RangeGroupTranslatorAdaptive(num_groups=8)
    out, alpha = model_adaptive.forward_with_alpha(x, target_group)
    print(f"  Input: {x.shape}, Output: {out.shape}, Alpha: {alpha.shape}")
    print(f"  Params: {sum(p.numel() for p in model_adaptive.parameters()):,}")
    print(f"  Alpha mean: {alpha.mean().item():.3f}, min: {alpha.min().item():.3f}, max: {alpha.max().item():.3f}")

    print("\nTesting losses...")
    target = torch.randn_like(x)

    dist_loss = DistributionLoss()
    print(f"  Distribution loss: {dist_loss(out, target).item():.4f}")

    silence_loss = SilenceLoss()
    print(f"  Silence loss: {silence_loss(x, out).item():.4f}")

    content_loss = ContentPreservationLoss()
    print(f"  Content loss: {content_loss(x, out).item():.4f}")

    envelope_loss = EnvelopePreservationLoss()
    print(f"  Envelope loss: {envelope_loss(out, target).item():.4f}")
