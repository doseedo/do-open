# Copyright 2024 DO1 Authors. All rights reserved.
# Adapted from ACE-Step's transformer layers.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, CustomLiteLAProcessor2_0, CustomerAttnProcessor2_0


def t2i_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply shift and scale modulation (AdaLN)."""
    return x * (1 + scale) + shift


def get_same_padding(kernel_size: int) -> int:
    """Calculate padding for 'same' convolution."""
    assert kernel_size % 2 == 1, f"kernel size {kernel_size} should be odd"
    return kernel_size // 2


class ConvLayer(nn.Module):
    """1D convolution layer with optional normalization and activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding: Optional[int] = None,
        use_bias: bool = False,
        norm: bool = False,
        act: bool = False,
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size) * dilation

        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
        )
        self.norm = nn.RMSNorm(out_dim, elementwise_affine=False) if norm else None
        self.act = nn.SiLU(inplace=True) if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class GLUMBConv(nn.Module):
    """
    Gated Linear Unit with Mobile Bottleneck Convolution.

    A feed-forward network using depthwise separable convolutions
    with GLU gating for efficient computation.

    Architecture:
        1. Inverted bottleneck: expand channels
        2. Depthwise conv: spatial mixing
        3. GLU gate: feature selection
        4. Point conv: project back to original dim
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        kernel_size: int = 3,
        use_bias: Tuple[bool, bool, bool] = (True, True, False),
    ):
        super().__init__()
        out_features = out_features or in_features

        self.glu_act = nn.SiLU(inplace=False)

        # Expand to 2x hidden for GLU
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            kernel_size=1,
            use_bias=use_bias[0],
        )

        # Depthwise conv
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=kernel_size,
            groups=hidden_features * 2,
            use_bias=use_bias[1],
        )

        # Project back
        self.point_conv = ConvLayer(
            hidden_features,
            out_features,
            kernel_size=1,
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D] -> transpose for conv
        x = x.transpose(1, 2)  # [B, D, S]

        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        # GLU gating
        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        x = x.transpose(1, 2)  # [B, S, D]

        return x


class DO1TransformerBlock(nn.Module):
    """
    Transformer block for DO1 with:
    - Self-attention (Linear attention for efficiency)
    - Cross-attention to reference tokens
    - AdaLN-Single conditioning from timestep
    - GLUMBConv feed-forward network

    Args:
        dim: Hidden dimension
        num_attention_heads: Number of attention heads
        attention_head_dim: Dimension per attention head
        mlp_ratio: Expansion ratio for FFN
        cross_attention_dim: Dimension of cross-attention context (x_ref)
        qk_norm: Whether to apply QK normalization
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        cross_attention_dim: Optional[int] = None,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.cross_attention_dim = cross_attention_dim or dim

        # Self-attention with Linear Attention
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm=qk_norm,
            processor=CustomLiteLAProcessor2_0(),
        )

        # Cross-attention to reference
        self.norm_cross = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(
            query_dim=dim,
            cross_attention_dim=self.cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm=qk_norm,
            processor=CustomerAttnProcessor2_0(),
        )

        # FFN
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = GLUMBConv(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            use_bias=(True, True, False),
        )

        # AdaLN-Single: 6 modulation values
        # (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, dim) / dim ** 0.5
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        rotary_freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        temb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            hidden_states: Input tokens [B, S, D]
            encoder_hidden_states: Reference tokens for cross-attention [B, S', D]
            attention_mask: Self-attention mask [B, S]
            encoder_attention_mask: Cross-attention mask [B, S']
            rotary_freqs_cis: RoPE frequencies (cos, sin)
            temb: Timestep embedding [B, 6*D] from t_block

        Returns:
            Output tokens [B, S, D]
        """
        B = hidden_states.shape[0]

        # Parse AdaLN modulation values
        # temb: [B, 6*D] -> [B, 6, D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + temb.reshape(B, 6, -1)
        ).chunk(6, dim=1)

        # squeeze: [B, 1, D] -> [B, D] for broadcasting
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        shift_mlp = shift_mlp.squeeze(1)
        scale_mlp = scale_mlp.squeeze(1)
        gate_mlp = gate_mlp.squeeze(1)

        # Step 1: Self-attention with AdaLN
        norm_hidden = self.norm1(hidden_states)
        norm_hidden = t2i_modulate(norm_hidden, shift_msa[:, None, :], scale_msa[:, None, :])

        attn_output = self.self_attn(
            hidden_states=norm_hidden,
            attention_mask=attention_mask,
            rotary_freqs_cis=rotary_freqs_cis,
        )
        attn_output = gate_msa[:, None, :] * attn_output
        hidden_states = hidden_states + attn_output

        # Step 2: Cross-attention to reference
        norm_hidden = self.norm_cross(hidden_states)
        cross_output = self.cross_attn(
            hidden_states=norm_hidden,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + cross_output

        # Step 3: FFN with AdaLN
        norm_hidden = self.norm2(hidden_states)
        norm_hidden = t2i_modulate(norm_hidden, shift_mlp[:, None, :], scale_mlp[:, None, :])

        ff_output = self.ff(norm_hidden)
        ff_output = gate_mlp[:, None, :] * ff_output
        hidden_states = hidden_states + ff_output

        return hidden_states


class T2IFinalLayer(nn.Module):
    """
    Final layer for DO1 that unpatchifies the output back to latent space.

    Applies final AdaLN modulation and projects patches back to
    the original latent dimensions.

    Args:
        hidden_size: Transformer hidden dimension
        patch_size: Patch size used in PatchEmbed
        out_channels: Output channels (should match input latent channels)
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: Tuple[int, int] = (4, 4),
        out_channels: int = 8,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size

        # Final normalization
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Project to patch dimensions
        patch_dim = patch_size[0] * patch_size[1] * out_channels
        self.linear = nn.Linear(hidden_size, patch_dim, bias=True)

        # AdaLN modulation (2 values: shift, scale)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size ** 0.5
        )

    def unpatchify(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Convert patch tokens back to image format.

        Args:
            hidden_states: [B, num_patches, patch_dim]
            height: Original height (freq bins)
            width: Original width (time frames)

        Returns:
            Output tensor [B, C, H, W]
        """
        B = hidden_states.shape[0]
        patch_h, patch_w = self.patch_size

        # Calculate patch grid dimensions
        num_h_patches = height // patch_h
        num_w_patches = width // patch_w

        # Reshape to patch grid
        # [B, num_patches, patch_dim] -> [B, H', W', Ph, Pw, C]
        hidden_states = hidden_states.reshape(
            B,
            num_h_patches,
            num_w_patches,
            patch_h,
            patch_w,
            self.out_channels,
        )

        # Rearrange to image format: [B, C, H'*Ph, W'*Pw]
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)  # [B, C, H', Ph, W', Pw]
        output = hidden_states.reshape(
            B,
            self.out_channels,
            num_h_patches * patch_h,
            num_w_patches * patch_w,
        )

        return output

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        height: int = 16,
        width: int = None,
    ) -> torch.Tensor:
        """
        Final layer forward pass.

        Args:
            x: Hidden states [B, num_patches, hidden_size]
            temb: Timestep embedding [B, hidden_size] for final AdaLN
            height: Target height (freq bins)
            width: Target width (time frames)

        Returns:
            Output latent [B, out_channels, height, width]
        """
        B, num_patches, _ = x.shape

        # Calculate width from num_patches if not provided
        if width is None:
            num_h_patches = height // self.patch_size[0]
            num_w_patches = num_patches // num_h_patches
            width = num_w_patches * self.patch_size[1]

        # AdaLN modulation
        shift, scale = (self.scale_shift_table[None] + temb[:, None]).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)

        x = self.norm_final(x)
        x = t2i_modulate(x, shift[:, None, :], scale[:, None, :])

        # Project to patch dimensions
        x = self.linear(x)

        # Unpatchify
        output = self.unpatchify(x, height, width)

        return output
