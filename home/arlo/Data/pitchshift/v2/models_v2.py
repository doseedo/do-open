"""
Register-Aware Pitch Shift Models V2

Non-linear register transformation, conditioned on pitch.
Same architecture as mute_translator but with pitch conditioning.

Input: latent at source pitch
Output: latent with target pitch's timbre characteristics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


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


class RegisterTranslator(nn.Module):
    """
    Non-linear register transformation conditioned on pitch.

    Like MuteTranslator but with pitch embeddings.
    Learns: output = f(input, source_pitch, target_pitch)

    The model learns the full non-linear transformation between registers,
    not just an additive offset.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        num_pitches: int = 128,
        pitch_embed_dim: int = 32,
    ):
        super().__init__()
        self.latent_channels = latent_channels

        # Pitch embeddings
        self.pitch_embed = nn.Embedding(num_pitches, pitch_embed_dim)

        # Combine source + target pitch into conditioning
        self.pitch_proj = nn.Sequential(
            nn.Linear(pitch_embed_dim * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Input projection (latent + pitch conditioning)
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

        # Initialize output near-zero for stable residual start
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        # Fixed scales (learnable scales cause NaN gradients)
        self.register_buffer('dry_scale', torch.tensor(1.0))
        self.register_buffer('residual_scale', torch.tensor(0.1))

    def forward(
        self,
        source_latent: torch.Tensor,
        source_pitch: torch.Tensor,
        target_pitch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            source_latent: [B, C, H, T] latent (could be from pitch-shifted audio)
            source_pitch: [B] source MIDI pitch
            target_pitch: [B] target MIDI pitch

        Returns:
            output_latent: [B, C, H, T] with target register timbre
        """
        B, C, H, T = source_latent.shape

        # Get pitch conditioning
        src_emb = self.pitch_embed(source_pitch)  # [B, embed_dim]
        tgt_emb = self.pitch_embed(target_pitch)  # [B, embed_dim]
        pitch_cond = self.pitch_proj(torch.cat([src_emb, tgt_emb], dim=-1))  # [B, hidden]
        pitch_cond = pitch_cond.unsqueeze(-1).expand(-1, -1, T)  # [B, hidden, T]

        # Flatten H into batch: [B*H, C, T]
        x = source_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)

        # Expand pitch conditioning for each H slice
        pitch_cond_expanded = pitch_cond.unsqueeze(1).expand(-1, H, -1, -1)
        pitch_cond_expanded = pitch_cond_expanded.reshape(B * H, -1, T)

        # Concatenate and process
        x = torch.cat([x, pitch_cond_expanded], dim=1)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        residual = self.output_proj(x)

        # Reshape back: [B, H, C, T] -> [B, C, H, T]
        residual = residual.reshape(B, H, C, T).permute(0, 2, 1, 3)

        # Apply with learnable scales (like mute_translator)
        output = self.dry_scale * source_latent + self.residual_scale * residual

        return output


class RegisterTranslatorDirect(nn.Module):
    """
    Direct (non-residual) register transformation.

    output = f(input, pitch) - no passthrough of input.
    Better for aggressive timbre changes.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        num_pitches: int = 128,
        pitch_embed_dim: int = 32,
    ):
        super().__init__()
        self.latent_channels = latent_channels

        # Pitch embeddings
        self.pitch_embed = nn.Embedding(num_pitches, pitch_embed_dim)

        self.pitch_proj = nn.Sequential(
            nn.Linear(pitch_embed_dim * 2, hidden_channels),
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

        # Small init for stability but not zero (direct mode needs to learn full output)
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        source_latent: torch.Tensor,
        source_pitch: torch.Tensor,
        target_pitch: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, T = source_latent.shape

        # Pitch conditioning
        src_emb = self.pitch_embed(source_pitch)
        tgt_emb = self.pitch_embed(target_pitch)
        pitch_cond = self.pitch_proj(torch.cat([src_emb, tgt_emb], dim=-1))
        pitch_cond = pitch_cond.unsqueeze(-1).expand(-1, -1, T)

        # Flatten H into batch
        x = source_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)
        pitch_cond_expanded = pitch_cond.unsqueeze(1).expand(-1, H, -1, -1).reshape(B * H, -1, T)

        # Process
        x = torch.cat([x, pitch_cond_expanded], dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        output = self.output_proj(x)

        # Reshape back
        output = output.reshape(B, H, C, T).permute(0, 2, 1, 3)

        return output


class CombinedLossV2(nn.Module):
    """
    Loss function for register transformation.

    Target is computed from codebook during training:
    target = source + (centroid[target_pitch] - centroid[source_pitch])

    But the model learns non-linear transformation to match this target.
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        spectral_weight: float = 0.5,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.spectral_weight = spectral_weight

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            output: [B, C, H, T] model output
            target: [B, C, H, T] target (from codebook)
        """
        # L1 reconstruction
        reconstruction = F.l1_loss(output, target)

        # Spectral envelope (mean over time)
        output_env = output.mean(dim=-1)
        target_env = target.mean(dim=-1)
        spectral = F.l1_loss(output_env, target_env)

        total = (
            self.reconstruction_weight * reconstruction +
            self.spectral_weight * spectral
        )

        return {
            'total_loss': total,
            'reconstruction': reconstruction,
            'spectral': spectral,
        }


if __name__ == "__main__":
    print("Testing RegisterTranslator...")
    model = RegisterTranslator()
    x = torch.randn(4, 8, 16, 64)  # DCAE outputs H=16
    src_pitch = torch.randint(48, 84, (4,))
    tgt_pitch = torch.randint(48, 84, (4,))
    out = model(x, src_pitch, tgt_pitch)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting RegisterTranslatorDirect...")
    model_direct = RegisterTranslatorDirect()
    out = model_direct(x, src_pitch, tgt_pitch)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model_direct.parameters()):,}")

    print("\nTesting CombinedLossV2...")
    loss_fn = CombinedLossV2()
    target = torch.randn_like(x)
    losses = loss_fn(out, target)
    print(f"  Total: {losses['total_loss']:.4f}")
