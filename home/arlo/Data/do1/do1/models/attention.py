# Copyright 2024 The HuggingFace Team. All rights reserved.
# Adapted for DO1 from ACE-Step's attention processors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import logging

logger = logging.get_logger(__name__)


class Attention(nn.Module):
    """
    Multi-head attention module supporting both self-attention and cross-attention.

    Adapted from diffusers.models.attention_processor.Attention with modifications
    for DO1's specific needs.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        processor: Optional["AttentionProcessor"] = None,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.is_cross_attention = cross_attention_dim is not None

        # Query, Key, Value projections
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)

        # QK normalization (RMSNorm)
        if qk_norm:
            self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
            self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        else:
            self.norm_q = None
            self.norm_k = None

        # Output projection
        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout),
        ])

        # Set processor
        self.processor = processor if processor is not None else CustomerAttnProcessor2_0()

        # Additional attributes for compatibility
        self.group_norm = None
        self.norm_cross = False
        self.residual_connection = False
        self.rescale_output_factor = 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        rotary_freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            rotary_freqs_cis=rotary_freqs_cis,
            **kwargs,
        )


class CustomLiteLAProcessor2_0:
    """
    Linear Attention processor for efficient O(N) self-attention.

    Uses kernel-based attention: (V @ K^T) @ Q instead of softmax(Q @ K^T) @ V.
    This reduces complexity from O(N^2) to O(N) for long sequences.

    Adapted from ACE-Step's implementation.
    """

    def __init__(self):
        self.kernel_func = nn.ReLU(inplace=False)
        self.eps = 1e-15
        self.pad_val = 1.0

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Apply rotary positional embeddings to input tensor.

        Args:
            x: Input tensor [B, H, S, D]
            freqs_cis: Tuple of (cos, sin) tensors, each [S, D]

        Returns:
            Tensor with rotary embeddings applied [B, H, S, D]
        """
        cos, sin = freqs_cis
        cos = cos[None, None]  # [1, 1, S, D]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        # Split into real and imaginary components
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        rotary_freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Apply linear attention.

        For self-attention: encoder_hidden_states is None
        For cross-attention: encoder_hidden_states contains the context
        """
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # Project query, key, value
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            # Self-attention
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            # Cross-attention
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        # Reshape to multi-head format
        head_dim = attn.dim_head

        # query: [B, S, H*D] -> [B, H, D, S]
        query = query.view(batch_size, -1, attn.heads, head_dim)
        query = query.permute(0, 2, 3, 1)  # [B, H, D, S]

        # key: [B, S, H*D] -> [B, H, S, D]
        key = key.view(batch_size, -1, attn.heads, head_dim)
        key = key.permute(0, 2, 1, 3)  # [B, H, S, D]

        # value: [B, S, H*D] -> [B, H, D, S]
        value = value.view(batch_size, -1, attn.heads, head_dim)
        value = value.permute(0, 2, 3, 1)  # [B, H, D, S]

        # For RoPE: need [B, H, S, D] format
        query_rope = query.permute(0, 1, 3, 2)  # [B, H, S, D]

        # Apply QK normalization
        if attn.norm_q is not None:
            query_rope = attn.norm_q(query_rope)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        if rotary_freqs_cis is not None:
            query_rope = self.apply_rotary_emb(query_rope, rotary_freqs_cis)
            if encoder_hidden_states is None:  # Self-attention
                key = self.apply_rotary_emb(key, rotary_freqs_cis)

        # Convert back to [B, H, D, S]
        query = query_rope.permute(0, 1, 3, 2)

        # Apply attention mask
        if attention_mask is not None:
            # attention_mask: [B, S] -> [B, 1, S, 1]
            mask = attention_mask[:, None, :, None].to(key.dtype)
            query = query * mask.permute(0, 1, 3, 2)
            if encoder_hidden_states is None:
                key = key * mask
                value = value * mask.permute(0, 1, 3, 2)

        if encoder_attention_mask is not None and encoder_hidden_states is not None:
            enc_mask = encoder_attention_mask[:, None, :, None].to(key.dtype)
            key = key * enc_mask
            value = value * enc_mask.permute(0, 1, 3, 2)

        # Apply kernel function (ReLU)
        query = self.kernel_func(query)
        key = self.kernel_func(key)

        # Convert to float for numerical stability
        query, key, value = query.float(), key.float(), value.float()

        # Pad value for normalization
        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=self.pad_val)

        # Linear attention: (V @ K^T) @ Q
        vk = torch.matmul(value, key)  # [B, H, D+1, D]
        hidden_states = torch.matmul(vk, query)  # [B, H, D+1, S]

        # Normalize
        if hidden_states.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.float()
        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)

        # Reshape back: [B, H, D, S] -> [B, S, H*D]
        hidden_states = hidden_states.permute(0, 3, 1, 2)  # [B, S, H, D]
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(dtype)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Clamp for fp16 stability
        if torch.get_autocast_gpu_dtype() == torch.float16:
            hidden_states = hidden_states.clamp(-65504, 65504)

        return hidden_states


class CustomerAttnProcessor2_0:
    """
    Standard scaled dot-product attention processor.

    Uses PyTorch 2.0's F.scaled_dot_product_attention for efficiency.
    Suitable for cross-attention where we want full attention patterns.

    Adapted from ACE-Step's implementation.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "CustomerAttnProcessor2_0 requires PyTorch 2.0+. "
                "Please upgrade PyTorch."
            )

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Apply rotary positional embeddings to input tensor.

        Args:
            x: Input tensor [B, H, S, D]
            freqs_cis: Tuple of (cos, sin) tensors, each [S, D]

        Returns:
            Tensor with rotary embeddings applied [B, H, S, D]
        """
        cos, sin = freqs_cis
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        rotary_freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply scaled dot-product attention.

        For self-attention: encoder_hidden_states is None
        For cross-attention: encoder_hidden_states contains the context
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Get context
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            context_seq_len = seq_len
        else:
            context_seq_len = encoder_hidden_states.shape[1]

        # Project query, key, value
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = attn.dim_head

        # Reshape to [B, H, S, D]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        if rotary_freqs_cis is not None:
            query = self.apply_rotary_emb(query, rotary_freqs_cis)
            # Only apply RoPE to keys for self-attention
            if attn.is_cross_attention is False:
                key = self.apply_rotary_emb(key, rotary_freqs_cis)

        # Prepare attention mask
        attn_mask = None
        if attention_mask is not None and encoder_attention_mask is not None:
            # Cross-attention with both masks
            # attention_mask: [B, S_q], encoder_attention_mask: [B, S_k]
            combined_mask = attention_mask[:, :, None] * encoder_attention_mask[:, None, :]
            attn_mask = torch.where(combined_mask == 1, 0.0, float('-inf'))
            attn_mask = attn_mask[:, None, :, :].expand(-1, attn.heads, -1, -1)
            attn_mask = attn_mask.to(query.dtype)
        elif attention_mask is not None:
            # Self-attention mask
            # Convert [B, S] to [B, 1, 1, S] for broadcasting
            attn_mask = attention_mask[:, None, None, :].to(query.dtype)
            attn_mask = torch.where(attn_mask == 1, 0.0, float('-inf'))

        # Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        # Reshape back: [B, H, S, D] -> [B, S, H*D]
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
