# Copyright 2024 DO1 Authors. All rights reserved.
# Adapted from ACE-Step's embedding modules.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn


class Qwen2RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.

    Based on Qwen2/LLaMA rotary embeddings for positional encoding
    in attention mechanisms.

    Args:
        dim: Head dimension (must be even)
        max_position_embeddings: Maximum sequence length to cache
        base: Base for computing frequencies (default: 10000)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 1000000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device if device is not None else torch.device("cpu"),
            dtype=torch.float32,
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos/sin embeddings for given sequence length.

        Args:
            x: Input tensor (used for device/dtype)
            seq_len: Sequence length to generate embeddings for

        Returns:
            Tuple of (cos, sin) tensors, each [seq_len, dim]
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class DO1PatchEmbed(nn.Module):
    """
    Patch embedding for DO1's concatenated input.

    Embeds the concatenated [x_noisy, x_cond, mask] tensor into patches.
    Uses a 2-stage convolution with GroupNorm similar to ACE-Step.

    Input shape: [B, 17, 16, T] (8ch noisy + 8ch cond + 1ch mask)
    Output shape: [B, num_patches, embed_dim] where num_patches = (16/patch_h) * (T/patch_w)

    Args:
        in_channels: Number of input channels (default: 17 for noisy+cond+mask)
        embed_dim: Embedding dimension
        patch_size: Tuple of (height, width) patch sizes (default: (4, 4))
        freq_bins: Number of frequency bins (default: 16)
        max_time_frames: Maximum time frames (default: 4096)
        bias: Whether to use bias in conv layers
    """

    def __init__(
        self,
        in_channels: int = 17,
        embed_dim: int = 2560,
        patch_size: Tuple[int, int] = (4, 4),
        freq_bins: int = 16,
        max_time_frames: int = 4096,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.freq_bins = freq_bins

        patch_h, patch_w = patch_size

        # Calculate output spatial dimensions
        self.num_freq_patches = freq_bins // patch_h
        self.num_time_patches = max_time_frames // patch_w

        # Intermediate expansion factor
        # ACE-Step uses in_channels * 256, we scale based on patch size
        intermediate_channels = in_channels * 64

        # Two-stage convolution with GroupNorm
        self.early_conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                intermediate_channels,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                bias=bias,
            ),
            nn.GroupNorm(
                num_groups=min(32, intermediate_channels // 4),
                num_channels=intermediate_channels,
                eps=1e-6,
                affine=True,
            ),
            nn.SiLU(),
            nn.Conv2d(
                intermediate_channels,
                embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed input patches.

        Args:
            x: Input tensor [B, C, H, W] where C=in_channels, H=freq_bins, W=time

        Returns:
            Embedded patches [B, num_patches, embed_dim]
        """
        # x: [B, C, H, W] -> [B, embed_dim, H/patch_h, W/patch_w]
        x = self.early_conv_layers(x)

        # Flatten spatial dimensions: [B, D, H', W'] -> [B, H'*W', D]
        x = x.flatten(2).transpose(1, 2)

        return x


class ReferenceEncoder(nn.Module):
    """
    Encoder for the reference latent (x_ref).

    Embeds x_ref into tokens for cross-attention. Optionally includes
    a learnable CLS token for global context.

    Input shape: [B, 8, 16, T']
    Output shape: [B, num_tokens, embed_dim]

    Args:
        in_channels: Number of input channels (default: 8)
        embed_dim: Embedding dimension
        patch_size: Tuple of (height, width) patch sizes
        include_cls: Whether to include a learnable CLS token
    """

    def __init__(
        self,
        in_channels: int = 8,
        embed_dim: int = 2560,
        patch_size: Tuple[int, int] = (4, 4),
        include_cls: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.include_cls = include_cls

        # Intermediate expansion
        intermediate_channels = in_channels * 64

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                in_channels,
                intermediate_channels,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                bias=True,
            ),
            nn.GroupNorm(
                num_groups=min(32, intermediate_channels // 4),
                num_channels=intermediate_channels,
                eps=1e-6,
                affine=True,
            ),
            nn.SiLU(),
            nn.Conv2d(
                intermediate_channels,
                embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        # Learnable CLS token
        if include_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) / math.sqrt(embed_dim))
        else:
            self.cls_token = None

    def forward(
        self,
        x_ref: torch.Tensor,
        ref_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode reference latent.

        Args:
            x_ref: Reference latent [B, 8, 16, T']
            ref_mask: Optional mask [B, T'] where 1=valid, 0=padding

        Returns:
            Tuple of:
                - Encoded tokens [B, num_tokens, embed_dim]
                - Updated mask [B, num_tokens] if mask provided, else None
        """
        B = x_ref.shape[0]

        # Patch embed: [B, 8, 16, T'] -> [B, D, 4, T'/4]
        x = self.patch_embed(x_ref)

        # Flatten: [B, D, H', W'] -> [B, N', D]
        x = x.flatten(2).transpose(1, 2)

        # Process mask if provided
        if ref_mask is not None:
            # Downsample mask to match patch grid
            # ref_mask: [B, T'] -> [B, T'//patch_w]
            patch_w = self.patch_size[1]
            T_orig = ref_mask.shape[1]
            T_patches = (T_orig + patch_w - 1) // patch_w

            # Use max pooling to get patch-level mask (any valid token in patch = valid)
            ref_mask_4d = ref_mask[:, None, None, :]  # [B, 1, 1, T']
            ref_mask_patches = nn.functional.max_pool2d(
                ref_mask_4d.float(),
                kernel_size=(1, patch_w),
                stride=(1, patch_w),
                padding=0,
            )  # [B, 1, 1, T'//patch_w]

            # Expand for frequency patches: [B, num_freq_patches * num_time_patches]
            num_freq_patches = 16 // self.patch_size[0]
            ref_mask_expanded = ref_mask_patches.squeeze(1).squeeze(1)  # [B, T'//patch_w]
            ref_mask_expanded = ref_mask_expanded.unsqueeze(1).expand(-1, num_freq_patches, -1)
            ref_mask_expanded = ref_mask_expanded.reshape(B, -1)  # [B, N']
        else:
            ref_mask_expanded = None

        # Add CLS token
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

            # Add CLS to mask (always valid)
            if ref_mask_expanded is not None:
                cls_mask = torch.ones(B, 1, device=ref_mask_expanded.device, dtype=ref_mask_expanded.dtype)
                ref_mask_expanded = torch.cat([cls_mask, ref_mask_expanded], dim=1)

        return x, ref_mask_expanded


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding using sinusoidal encoding + MLP.

    Converts scalar timesteps to embedding vectors for AdaLN conditioning.

    Args:
        embedding_dim: Dimension of sinusoidal embedding
        output_dim: Output dimension after MLP
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        output_dim: int = 2560,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # MLP to project to output dimension
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps.

        Args:
            t: Timesteps [B] in range [0, 1]

        Returns:
            Embeddings [B, output_dim]
        """
        # Sinusoidal embedding
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # MLP projection
        emb = self.mlp(emb)

        return emb
