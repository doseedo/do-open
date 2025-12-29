#!/usr/bin/env python3
"""
v9 Models: Conditional Formant Corrector

Conditions on:
- Starting group (1, 2, or 3)
- Direction (up or down)

This allows the model to learn different transformations for different
register transitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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


class ConditionalFormantCorrector(nn.Module):
    """
    Conditional formant corrector that takes group and direction as input.

    Conditioning:
    - source_group: 1, 2, or 3 (which range the input came from)
    - direction: 0 = down (higher formants → lower), 1 = up (lower → higher)

    Architecture: Residual network with FiLM conditioning.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        num_groups: int = 3,
        num_directions: int = 2,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks

        # Conditioning embeddings
        self.group_embed = nn.Embedding(num_groups, hidden_channels)
        self.direction_embed = nn.Embedding(num_directions, hidden_channels)

        # FiLM conditioning for each block: generates scale and shift
        self.film_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels * 2),  # scale + shift
            )
            for _ in range(num_blocks)
        ])

        # Project to hidden space
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

        # Small init for stability
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[-1].bias)

        # Learnable scales
        self.residual_scale = nn.Parameter(torch.tensor(0.2))
        self.input_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        latent: torch.Tensor,
        source_group: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latent: [B, C, H, T] input latent
            source_group: [B] group indices (0, 1, 2 for groups 1, 2, 3)
            direction: [B] direction (0=down, 1=up)

        Returns:
            corrected: [B, C, H, T] corrected latent
        """
        B, C, H, T = latent.shape

        # Get conditioning
        group_emb = self.group_embed(source_group)  # [B, hidden]
        dir_emb = self.direction_embed(direction)   # [B, hidden]
        cond = torch.cat([group_emb, dir_emb], dim=-1)  # [B, hidden*2]

        # Flatten H into batch for 1D processing: [B*H, C, T]
        x = latent.permute(0, 2, 1, 3).reshape(B * H, C, T)

        # Process
        x = self.input_proj(x)

        for i, block in enumerate(self.blocks):
            x = block(x)

            # Apply FiLM conditioning
            film_params = self.film_layers[i](cond)  # [B, hidden*2]
            scale, shift = film_params.chunk(2, dim=-1)  # [B, hidden] each

            # Expand for broadcasting: [B, hidden] -> [B*H, hidden, 1]
            scale = scale.unsqueeze(1).expand(-1, H, -1).reshape(B * H, -1, 1)
            shift = shift.unsqueeze(1).expand(-1, H, -1).reshape(B * H, -1, 1)

            x = x * (1 + scale) + shift

        residual = self.output_proj(x)

        # Reshape back: [B*H, C, T] -> [B, H, C, T] -> [B, C, H, T]
        residual = residual.reshape(B, H, C, T).permute(0, 2, 1, 3)

        # Apply with learnable scales
        corrected = self.input_scale * latent + self.residual_scale * residual

        return corrected


class FormantCorrectorSimple(nn.Module):
    """
    Simpler version without FiLM - just concatenates conditioning.

    Use this if FiLM causes training instability.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        num_groups: int = 3,
        num_directions: int = 2,
        direct_output: bool = False,  # If True, output directly (no residual)
        residual_scale_init: float = 0.5,  # Initial residual scale (larger = stronger corrections)
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.direct_output = direct_output

        # Conditioning embeddings
        cond_dim = 32
        self.group_embed = nn.Embedding(num_groups, cond_dim)
        self.direction_embed = nn.Embedding(num_directions, cond_dim)

        # Input: latent channels + condition (broadcast over time)
        self.input_proj = nn.Conv1d(latent_channels + cond_dim * 2, hidden_channels, 1)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_channels, kernel_size, dilation=2**i)
            for i in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, latent_channels, 1),
        )

        # For direct output, use normal init; for residual, use small init
        if direct_output:
            nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=1.0)
        else:
            nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[-1].bias)

        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_init))
        self.input_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        latent: torch.Tensor,
        source_group: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, T = latent.shape

        # Get conditioning
        group_emb = self.group_embed(source_group)  # [B, cond_dim]
        dir_emb = self.direction_embed(direction)   # [B, cond_dim]
        cond = torch.cat([group_emb, dir_emb], dim=-1)  # [B, cond_dim*2]

        # Flatten H into batch: [B*H, C, T]
        x = latent.permute(0, 2, 1, 3).reshape(B * H, C, T)

        # Expand condition to match: [B*H, cond_dim*2, T]
        cond_expanded = cond.unsqueeze(1).expand(-1, H, -1).reshape(B * H, -1)
        cond_expanded = cond_expanded.unsqueeze(-1).expand(-1, -1, T)

        # Concatenate input with conditioning
        x = torch.cat([x, cond_expanded], dim=1)

        # Process
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)

        # Reshape back
        output = self.output_proj(x)
        output = output.reshape(B, H, C, T).permute(0, 2, 1, 3)

        if self.direct_output:
            return output  # Direct output, no skip connection
        else:
            return self.input_scale * latent + self.residual_scale * output


# Alias for backward compatibility
RegisterTranslator = FormantCorrectorSimple


if __name__ == "__main__":
    # Test models
    batch = torch.randn(4, 8, 16, 128)  # [B, C, H, T]
    groups = torch.tensor([0, 1, 2, 1])  # Group indices (0-2)
    directions = torch.tensor([0, 1, 0, 1])  # 0=down, 1=up

    print("Testing ConditionalFormantCorrector...")
    model = ConditionalFormantCorrector()
    out = model(batch, groups, directions)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting FormantCorrectorSimple...")
    model_simple = FormantCorrectorSimple()
    out = model_simple(batch, groups, directions)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model_simple.parameters()):,}")
