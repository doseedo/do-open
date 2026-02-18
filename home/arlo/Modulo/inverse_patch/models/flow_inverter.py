"""
Unified Flow Matching Inverter for ALL Effects.

Single architecture that handles any effect type including temporal effects.
Uses flow matching for handling multimodality (multiple valid inversions).

Key advantages over per-effect engineering:
- Single architecture for everything
- Learns from data which patterns matter
- Handles unknown effects via generalization
- Variable inference steps (1-4 for easy effects, 8-16 for hard)

Architecture:
- TemporalCorrelationEncoder: Captures ALL temporal patterns automatically
- Mamba/Transformer backbone: Long-range dependencies with efficient O(n) complexity
- Flow matching: Deterministic ODE from wet → dry, handles multimodality

Comparison:
| Approach            | Steps for EQ | Steps for Reverb | Quality |
|---------------------|--------------|------------------|---------|
| Diffusion           | 20+          | 50+              | High    |
| Flow Matching       | 1-4          | 10-20            | High    |
| Consistency Models  | 1-2          | 4-8              | Med-High|
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict, Any

from .temporal_encoder import (
    TemporalCorrelationEncoder,
    UnifiedTemporalEncoder,
    SinusoidalEmbedding,
)


# Try to import Mamba (optional dependency)
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class ChainEncoder(nn.Module):
    """
    Encodes effect chain (types + params) into conditioning vector.
    Reused from unified_inverter but simplified.
    """

    def __init__(
        self,
        n_effects: int = 6,
        max_params: int = 15,
        embed_dim: int = 256,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Effect type embedding
        self.effect_embedding = nn.Embedding(n_effects + 1, embed_dim, padding_idx=n_effects)

        # Parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(max_params, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Combine effect + params
        self.combiner = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Pool across chain
        self.pooler = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            effect_types: [B, max_chain] - indices, -1 for padding
            effect_params: [B, max_chain, max_params]

        Returns:
            condition: [B, embed_dim]
        """
        B, L = effect_types.shape

        # Create mask for valid effects
        mask = effect_types >= 0  # [B, L]

        # Safe indexing
        effect_types_safe = effect_types.clamp(min=0)

        # Get embeddings
        effect_emb = self.effect_embedding(effect_types_safe)  # [B, L, D]
        param_emb = self.param_encoder(effect_params)  # [B, L, D]

        # Combine
        combined = self.combiner(torch.cat([effect_emb, param_emb], dim=-1))  # [B, L, D]

        # Mask padding
        combined = combined * mask.unsqueeze(-1).float()

        # Mean pool (excluding padding)
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = combined.sum(dim=1) / mask_sum  # [B, D]

        return self.pooler(pooled)


class RelativePositionalAttention(nn.Module):
    """
    Self-attention with relative positional encoding.

    The relative position tells the network "how far apart are these samples",
    which is exactly what matters for temporal effects like delay and chorus.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        max_rel_pos: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_rel_pos = max_rel_pos

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Relative position embeddings
        self.rel_pos_emb = nn.Embedding(2 * max_rel_pos + 1, n_heads)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] sequence

        Returns:
            out: [B, N, D]
        """
        B, N, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        positions = torch.arange(N, device=x.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # [N, N]
        rel_pos = rel_pos.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos

        pos_bias = self.rel_pos_emb(rel_pos)  # [N, N, n_heads]
        pos_bias = pos_bias.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, N, N]

        attn = attn + pos_bias

        # Softmax and apply to values
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with relative positional attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        ff_mult: int = 4,
        max_rel_pos: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = RelativePositionalAttention(d_model, n_heads, max_rel_pos, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MambaBlock(nn.Module):
    """Mamba block wrapper for optional Mamba support."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm not installed. Use TransformerBlock instead.")

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class FlowMatchingInverter(nn.Module):
    """
    Flow Matching Inverter for ALL Effects.

    Uses flow matching to learn the velocity field from wet → dry.
    Can use variable steps based on effect difficulty:
    - 1-4 steps for easy effects (EQ, gain)
    - 8-16 steps for hard effects (reverb, distortion)

    Key components:
    1. TemporalCorrelationEncoder: Captures ALL temporal patterns
    2. Backbone (Mamba or Transformer): Long-range dependencies
    3. Flow matching: Handles multimodality naturally
    """

    EFFECT_TYPES = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']

    EFFECT_DIFFICULTY = {
        'eq': 1,        # Very easy, nearly invertible
        'gain': 1,      # Trivial
        'compressor': 4,  # Medium
        'delay': 4,     # Medium (temporal pattern)
        'chorus': 8,    # Harder (modulating temporal)
        'reverb': 12,   # Hard (additive, long)
        'distortion': 16,  # Very hard (info loss)
    }

    def __init__(
        self,
        n_effects: int = 6,
        max_params: int = 15,
        d_model: int = 256,
        n_layers: int = 8,
        sample_rate: int = 44100,
        max_lag_ms: float = 2000.0,
        use_mamba: bool = False,
        patch_size: int = 256,
    ):
        """
        Args:
            n_effects: Number of supported effect types
            max_params: Max parameters per effect
            d_model: Model hidden dimension
            n_layers: Number of backbone layers
            sample_rate: Audio sample rate
            max_lag_ms: Max temporal lag to consider
            use_mamba: Use Mamba backbone (faster) vs Transformer (more stable)
            patch_size: Size of audio patches for tokenization
        """
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.sample_rate = sample_rate
        self.patch_size = patch_size
        self.use_mamba = use_mamba and MAMBA_AVAILABLE

        # Temporal feature extraction - this is key for delay/chorus/etc
        self.temporal_encoder = TemporalCorrelationEncoder(
            max_lag_ms=max_lag_ms,
            sample_rate=sample_rate,
            n_lags=64,
            output_dim=64,
        )

        # Audio + temporal features → model dimension
        # Input: 1 (audio) + 64 (temporal) = 65 channels
        self.input_proj = nn.Conv1d(1 + 64, d_model, kernel_size=1)

        # Patchify audio for efficient processing
        self.patch_embed = nn.Conv1d(
            d_model, d_model,
            kernel_size=patch_size,
            stride=patch_size // 2,
            padding=patch_size // 2,
        )

        # Backbone layers
        if self.use_mamba:
            self.backbone = nn.ModuleList([
                MambaBlock(d_model, d_state=16, d_conv=4, expand=2)
                for _ in range(n_layers)
            ])
        else:
            max_rel_pos = 1024  # ~1 second at 256 patch size
            self.backbone = nn.ModuleList([
                TransformerBlock(d_model, n_heads=8, max_rel_pos=max_rel_pos)
                for _ in range(n_layers)
            ])

        # Conditioning
        self.chain_encoder = ChainEncoder(
            n_effects=n_effects,
            max_params=max_params,
            embed_dim=d_model,
        )

        # Time embedding for flow
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Unpatchify
        self.unpatch = nn.ConvTranspose1d(
            d_model, d_model,
            kernel_size=patch_size,
            stride=patch_size // 2,
            padding=patch_size // 4,
        )

        # Output projection
        self.output_proj = nn.Conv1d(d_model, 1, kernel_size=1)

    def velocity(
        self,
        audio: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity field for flow matching.

        Args:
            audio: [B, 1, T] current audio state
            effect_types: [B, max_chain] effect indices
            effect_params: [B, max_chain, max_params]
            t: [B] timestep (0 = wet, 1 = dry)

        Returns:
            velocity: [B, 1, T] velocity towards dry
        """
        B, _, T = audio.shape

        # Extract temporal features (catches delay, chorus, etc.)
        temporal_feat = self.temporal_encoder(audio)  # [B, 64, T]

        # Combine audio + temporal
        x = torch.cat([audio, temporal_feat], dim=1)  # [B, 65, T]
        x = self.input_proj(x)  # [B, D, T]

        # Add conditioning
        condition = self.chain_encoder(effect_types, effect_params)  # [B, D]
        x = x + condition.unsqueeze(-1)

        # Add time embedding
        t_emb = self.time_embed(t)  # [B, D]
        x = x + t_emb.unsqueeze(-1)

        # Patchify
        x = self.patch_embed(x)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]

        # Backbone
        for layer in self.backbone:
            x = layer(x)

        # Unpatchify
        x = x.transpose(1, 2)  # [B, D, N]
        x = self.unpatch(x)  # [B, D, T']

        # Match length
        if x.shape[-1] != T:
            x = F.interpolate(x, size=T, mode='linear', align_corners=False)

        # Output velocity
        velocity = self.output_proj(x)

        return velocity

    def forward(
        self,
        wet_audio: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Flow from wet → dry.

        Args:
            wet_audio: [B, 1, T] wet audio
            effect_types: [B, max_chain] effect indices
            effect_params: [B, max_chain, max_params]
            n_steps: Override number of steps (auto-selected if None)

        Returns:
            dry_audio: [B, 1, T]
        """
        B = wet_audio.shape[0]
        device = wet_audio.device

        # Auto-select steps based on effect difficulty
        if n_steps is None:
            n_steps = self.estimate_steps(effect_types)

        # Euler integration from wet to dry
        x = wet_audio
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=device)
            v = self.velocity(x, effect_types, effect_params, t)
            x = x + v * dt

        return torch.clamp(x, -1, 1)

    def estimate_steps(self, effect_types: torch.Tensor) -> int:
        """
        Estimate optimal number of steps based on effect difficulty.

        More steps = better quality but slower.
        """
        max_difficulty = 1

        for fx_idx in effect_types[0]:  # Use first batch item
            if fx_idx < 0:
                continue
            if fx_idx < len(self.EFFECT_TYPES):
                fx_name = self.EFFECT_TYPES[fx_idx]
                difficulty = self.EFFECT_DIFFICULTY.get(fx_name, 8)
                max_difficulty = max(max_difficulty, difficulty)

        return max_difficulty

    def training_step(
        self,
        wet_audio: torch.Tensor,
        dry_audio: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flow matching training loss.

        Target: velocity should point from wet → dry.
        """
        B = wet_audio.shape[0]
        device = wet_audio.device

        # Random timestep
        t = torch.rand(B, device=device)

        # Interpolate between wet and dry
        # x_t = t * dry + (1-t) * wet
        x_t = t.view(-1, 1, 1) * dry_audio + (1 - t.view(-1, 1, 1)) * wet_audio

        # Target velocity = dry - wet (constant velocity field)
        target_velocity = dry_audio - wet_audio

        # Predict velocity
        pred_velocity = self.velocity(x_t, effect_types, effect_params, t)

        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)

        return loss

    @classmethod
    def effect_to_idx(cls, effect_name: str) -> int:
        return cls.EFFECT_TYPES.index(effect_name)

    @classmethod
    def idx_to_effect(cls, idx: int) -> str:
        return cls.EFFECT_TYPES[idx]


class LightweightFlowInverter(nn.Module):
    """
    Lightweight version for faster inference.

    Uses simpler architecture:
    - Fewer layers
    - Smaller model dimension
    - No patching (direct convolution)
    """

    EFFECT_TYPES = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']

    def __init__(
        self,
        n_effects: int = 6,
        max_params: int = 15,
        d_model: int = 128,
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.d_model = d_model
        self.sample_rate = sample_rate

        # Temporal encoder
        self.temporal_encoder = TemporalCorrelationEncoder(
            max_lag_ms=1000.0,
            sample_rate=sample_rate,
            n_lags=32,
            output_dim=32,
        )

        # Simple UNet-style architecture
        self.encoder = nn.Sequential(
            nn.Conv1d(1 + 32, 64, 7, padding=3),
            nn.SiLU(),
            nn.Conv1d(64, d_model, 7, stride=2, padding=3),
            nn.SiLU(),
        )

        self.middle = nn.Sequential(
            nn.Conv1d(d_model, d_model, 7, padding=3),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 7, padding=3),
            nn.SiLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(d_model, 64, 7, stride=2, padding=3, output_padding=1),
            nn.SiLU(),
            nn.Conv1d(64, 1, 7, padding=3),
        )

        # Conditioning
        self.chain_encoder = ChainEncoder(n_effects, max_params, d_model)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model),
        )

    def velocity(
        self,
        audio: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        B, _, T = audio.shape

        # Temporal features
        temporal_feat = self.temporal_encoder(audio)
        x = torch.cat([audio, temporal_feat], dim=1)

        # Encode
        x = self.encoder(x)

        # Add conditioning + time
        cond = self.chain_encoder(effect_types, effect_params)
        t_emb = self.time_embed(t)
        x = x + (cond + t_emb).unsqueeze(-1)

        # Middle
        x = self.middle(x)

        # Decode
        x = self.decoder(x)

        # Match length
        if x.shape[-1] != T:
            x = F.interpolate(x, size=T, mode='linear', align_corners=False)

        return x

    def forward(
        self,
        wet_audio: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
        n_steps: int = 4,
    ) -> torch.Tensor:
        B = wet_audio.shape[0]
        device = wet_audio.device

        x = wet_audio
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=device)
            v = self.velocity(x, effect_types, effect_params, t)
            x = x + v * dt

        return torch.clamp(x, -1, 1)

    def training_step(
        self,
        wet_audio: torch.Tensor,
        dry_audio: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> torch.Tensor:
        B = wet_audio.shape[0]
        device = wet_audio.device

        t = torch.rand(B, device=device)
        x_t = t.view(-1, 1, 1) * dry_audio + (1 - t.view(-1, 1, 1)) * wet_audio
        target_velocity = dry_audio - wet_audio
        pred_velocity = self.velocity(x_t, effect_types, effect_params, t)

        return F.mse_loss(pred_velocity, target_velocity)


def create_flow_inverter(
    size: str = 'base',
    sample_rate: int = 44100,
    use_mamba: bool = False,
) -> nn.Module:
    """
    Create flow inverter with preset sizes.

    Args:
        size: 'tiny', 'small', 'base', 'large'
        sample_rate: Audio sample rate
        use_mamba: Use Mamba backbone if available

    Returns:
        FlowMatchingInverter or LightweightFlowInverter
    """
    if size == 'tiny':
        return LightweightFlowInverter(
            d_model=64,
            sample_rate=sample_rate,
        )

    configs = {
        'small': dict(d_model=128, n_layers=4),
        'base': dict(d_model=256, n_layers=8),
        'large': dict(d_model=384, n_layers=12),
    }

    cfg = configs.get(size, configs['base'])

    return FlowMatchingInverter(
        sample_rate=sample_rate,
        use_mamba=use_mamba,
        **cfg,
    )


if __name__ == '__main__':
    print("Testing FlowMatchingInverter...")

    # Create model
    model = create_flow_inverter('base', use_mamba=False)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy input
    B = 2
    T = 44100 * 2  # 2 seconds
    wet = torch.randn(B, 1, T)
    effect_types = torch.tensor([[0, 5, -1, -1, -1, -1], [2, 4, -1, -1, -1, -1]])  # EQ+delay, reverb+chorus
    effect_params = torch.randn(B, 6, 15)

    # Forward pass
    dry = model(wet, effect_types, effect_params, n_steps=4)
    print(f"Input: {wet.shape} -> Output: {dry.shape}")

    # Training step
    dry_gt = torch.randn(B, 1, T)
    loss = model.training_step(wet, dry_gt, effect_types, effect_params)
    print(f"Training loss: {loss.item():.4f}")

    print("\nTesting LightweightFlowInverter...")
    light_model = create_flow_inverter('tiny')
    print(f"Light model params: {sum(p.numel() for p in light_model.parameters()):,}")
    dry_light = light_model(wet, effect_types, effect_params, n_steps=4)
    print(f"Input: {wet.shape} -> Output: {dry_light.shape}")

    print("\nAll tests passed!")
