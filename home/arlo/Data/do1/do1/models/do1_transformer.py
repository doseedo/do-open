# Copyright 2024 DO1 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.modeling_utils import ModelMixin

from .embeddings import DO1PatchEmbed, ReferenceEncoder, TimestepEmbedding, Qwen2RotaryEmbedding
from .layers import DO1TransformerBlock, T2IFinalLayer


@dataclass
class DO1TransformerOutput(BaseOutput):
    """
    Output of DO1Transformer2DModel.

    Args:
        sample: Predicted velocity field [B, 8, 16, T]
    """
    sample: torch.Tensor


class DO1Transformer2DModel(ModelMixin, ConfigMixin):
    """
    DO1: Diffusion Transformer for Latent Audio Processing.

    A 3.3B parameter DiT that operates in DCAE latent space via flow matching.
    Takes three latent tensors as input and produces one as output.

    Inputs:
        x_noisy: [B, 8, 16, T] - being denoised via flow matching
        x_cond: [B, 8, 16, T] - primary reference (optionally corrupted)
        x_ref: [B, 8, 16, T'] - secondary reference (style/timbre), or zeros
        mask: [B, 1, 16, T] - 1=preserve x_cond content, 0=generate fresh
        timestep: [B] - flow matching timestep in [0, 1]

    Output:
        v_pred: [B, 8, 16, T] - predicted velocity field

    Architecture:
        - Concatenate [x_noisy, x_cond, mask] -> 17 channels
        - PatchEmbed (4x4 patches) -> sequence of tokens
        - ReferenceEncoder embeds x_ref for cross-attention
        - 32 transformer blocks with self-attention + cross-attention
        - AdaLN conditioning from timestep
        - Unpatchify back to [B, 8, 16, T]
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        # Input/output dimensions
        in_channels_noisy: int = 8,
        in_channels_cond: int = 8,
        in_channels_mask: int = 1,
        in_channels_ref: int = 8,
        out_channels: int = 8,
        # Patch embedding
        patch_size: Tuple[int, int] = (4, 4),
        freq_bins: int = 16,
        max_time_frames: int = 4096,
        # Transformer dimensions
        model_dim: int = 2560,
        num_attention_heads: int = 40,
        attention_head_dim: int = 64,
        num_layers: int = 32,
        mlp_ratio: float = 4.0,
        # Position encoding
        max_position: int = 32768,
        rope_theta: float = 1000000.0,
        # Normalization
        qk_norm: bool = True,
    ):
        super().__init__()

        self.in_channels_noisy = in_channels_noisy
        self.in_channels_cond = in_channels_cond
        self.in_channels_mask = in_channels_mask
        self.in_channels_ref = in_channels_ref
        self.out_channels = out_channels
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.freq_bins = freq_bins

        # Total input channels for main path
        total_in_channels = in_channels_noisy + in_channels_cond + in_channels_mask

        # Main patch embedding (x_noisy + x_cond + mask)
        self.patch_embed = DO1PatchEmbed(
            in_channels=total_in_channels,
            embed_dim=model_dim,
            patch_size=patch_size,
            freq_bins=freq_bins,
            max_time_frames=max_time_frames,
        )

        # Reference encoder (x_ref)
        self.reference_encoder = ReferenceEncoder(
            in_channels=in_channels_ref,
            embed_dim=model_dim,
            patch_size=patch_size,
            include_cls=True,
        )

        # Timestep embedding
        self.timestep_embedder = TimestepEmbedding(
            embedding_dim=256,
            output_dim=model_dim,
        )

        # Project timestep to 6*model_dim for AdaLN
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, 6 * model_dim, bias=True),
        )

        # Rotary position embedding
        self.rotary_emb = Qwen2RotaryEmbedding(
            dim=attention_head_dim,
            max_position_embeddings=max_position,
            base=rope_theta,
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            DO1TransformerBlock(
                dim=model_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                cross_attention_dim=model_dim,
                qk_norm=qk_norm,
            )
            for _ in range(num_layers)
        ])

        # Final layer
        self.final_layer = T2IFinalLayer(
            hidden_size=model_dim,
            patch_size=patch_size,
            out_channels=out_channels,
        )

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        """Enable/disable gradient checkpointing."""
        self.gradient_checkpointing = value

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True

    def forward(
        self,
        x_noisy: torch.Tensor,
        x_cond: torch.Tensor,
        x_ref: torch.Tensor,
        mask: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> DO1TransformerOutput:
        """
        Forward pass.

        Args:
            x_noisy: Noisy latent being denoised [B, 8, 16, T]
            x_cond: Conditioning latent [B, 8, 16, T]
            x_ref: Reference latent for style/timbre [B, 8, 16, T']
            mask: Task mask [B, 1, 16, T]
            timestep: Flow matching timestep [B] in [0, 1]
            attention_mask: Padding mask for main sequence [B, T]
            ref_mask: Padding mask for reference [B, T']
            return_dict: Whether to return DO1TransformerOutput

        Returns:
            Predicted velocity field [B, 8, 16, T]
        """
        B = x_noisy.shape[0]
        T = x_noisy.shape[-1]

        # Step 1: Concatenate inputs along channel dimension
        # [B, 8, 16, T] + [B, 8, 16, T] + [B, 1, 16, T] -> [B, 17, 16, T]
        x_input = torch.cat([x_noisy, x_cond, mask], dim=1)

        # Step 2: Patch embed main input
        # [B, 17, 16, T] -> [B, num_patches, model_dim]
        hidden_states = self.patch_embed(x_input)
        num_patches = hidden_states.shape[1]

        # Step 3: Encode reference
        # [B, 8, 16, T'] -> [B, num_ref_tokens, model_dim]
        ref_tokens, ref_mask_expanded = self.reference_encoder(x_ref, ref_mask)

        # Step 4: Timestep embedding
        # [B] -> [B, model_dim] -> [B, 6*model_dim]
        t_emb = self.timestep_embedder(timestep)
        t_emb = self.t_block(t_emb)

        # Step 5: Compute RoPE frequencies
        seq_len = hidden_states.shape[1]
        rotary_freqs_cis = self.rotary_emb(hidden_states, seq_len)

        # Step 6: Process attention mask
        if attention_mask is not None:
            # Downsample mask to match patch grid
            patch_w = self.patch_size[1]
            # Pool the mask to patch resolution
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
            attention_mask = F.max_pool2d(
                attention_mask.float(),
                kernel_size=(1, patch_w),
                stride=(1, patch_w),
            )
            # Expand for frequency patches
            num_freq_patches = self.freq_bins // self.patch_size[0]
            attention_mask = attention_mask.squeeze(1).squeeze(1)  # [B, T//patch_w]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, num_freq_patches, -1)
            attention_mask = attention_mask.reshape(B, -1)  # [B, num_patches]

        # Step 7: Transformer blocks
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    ref_tokens,
                    attention_mask,
                    ref_mask_expanded,
                    rotary_freqs_cis,
                    t_emb,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=ref_tokens,
                    attention_mask=attention_mask,
                    encoder_attention_mask=ref_mask_expanded,
                    rotary_freqs_cis=rotary_freqs_cis,
                    temb=t_emb,
                )

        # Step 8: Final layer and unpatchify
        # Need final timestep embedding for final AdaLN
        t_emb_final = self.timestep_embedder(timestep)  # [B, model_dim]
        output = self.final_layer(hidden_states, t_emb_final, height=self.freq_bins, width=T)

        if not return_dict:
            return (output,)

        return DO1TransformerOutput(sample=output)


def get_do1_config_3b() -> dict:
    """Get configuration for ~3.3B parameter DO1 model."""
    # Tuned to hit ~3.3B params: model_dim=2304, 36 heads, 32 layers
    return {
        "in_channels_noisy": 8,
        "in_channels_cond": 8,
        "in_channels_mask": 1,
        "in_channels_ref": 8,
        "out_channels": 8,
        "patch_size": (4, 4),
        "freq_bins": 16,
        "max_time_frames": 4096,
        "model_dim": 2304,
        "num_attention_heads": 36,
        "attention_head_dim": 64,
        "num_layers": 32,
        "mlp_ratio": 4.0,
        "max_position": 32768,
        "rope_theta": 1000000.0,
        "qk_norm": True,
    }


def get_do1_config_small() -> dict:
    """Get configuration for small debug/test DO1 model."""
    return {
        "in_channels_noisy": 8,
        "in_channels_cond": 8,
        "in_channels_mask": 1,
        "in_channels_ref": 8,
        "out_channels": 8,
        "patch_size": (4, 4),
        "freq_bins": 16,
        "max_time_frames": 4096,
        "model_dim": 512,
        "num_attention_heads": 8,
        "attention_head_dim": 64,
        "num_layers": 4,
        "mlp_ratio": 4.0,
        "max_position": 8192,
        "rope_theta": 1000000.0,
        "qk_norm": True,
    }


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
