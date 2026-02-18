"""
Latent Flow Matching Inverter for ALL Effects.

Operates in ACE-Step DCAE latent space instead of raw audio for:
- ~4096x shorter sequences (44100*2 samples → ~21 latent frames for 2s audio)
- Implicit phase reconstruction via HiFi-GAN vocoder
- More efficient training and inference

Architecture:
    wet_audio ──┬──→ [Temporal Encoders (audio domain)] ──→ temporal_cond
                │
                └──→ [Mel Spectrogram] ──→ [DCAE Encoder (frozen)] ──→ wet_latent
                                                                          │
                                          ┌───────────────────────────────┘
                                          ▼
                                 [Mamba/Transformer Flow Net] ←── FiLM(temporal + chain + time)
                                          │
                                          ▼
                                     dry_latent
                                          │
                                          ▼
                                 [DCAE Decoder (frozen)] ──→ mel ──→ [Vocoder] ──→ dry_audio

Key insight: Temporal encoders STAY in audio domain because:
- They detect patterns like "echo at 500ms"
- This info conditions the latent flow
- Moving them to latent would lose timing resolution

DCAE Latent Shape: (batch, channels, height, time/8)
- height relates to mel frequency bands
- time is compressed 8x from mel frames (which are 512 hop from audio)
- Total compression: ~4096x (512 hop * 8 DCAE compression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from typing import List, Tuple, Optional, Dict, Any

from .temporal_encoder import (
    TemporalCorrelationEncoder,
    UnifiedTemporalEncoder,
    SinusoidalEmbedding,
)


# Try to import ACE-Step DCAE
DCAE_PATH = "/home/arlo/Data/ACE-Step"
if DCAE_PATH not in sys.path:
    sys.path.insert(0, DCAE_PATH)

# Default checkpoint paths (HuggingFace cache structure)
DEFAULT_DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
DEFAULT_VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

try:
    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
    DCAE_AVAILABLE = True
except ImportError:
    DCAE_AVAILABLE = False
    print("Warning: ACE-Step DCAE not available. Add /home/arlo/Data/ACE-Step to PYTHONPATH")


# Try to import Mamba (optional dependency)
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.

    Applies conditioning as: output = x * scale + shift
    where scale and shift are learned from the conditioning signal.
    """

    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        # Generate scale and shift from condition
        self.scale_proj = nn.Linear(cond_dim, feature_dim)
        self.shift_proj = nn.Linear(cond_dim, feature_dim)

        # Initialize to identity transform
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.ones_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, T] or [B, T, D] features
            cond: [B, cond_dim] conditioning vector

        Returns:
            modulated: Same shape as x
        """
        scale = self.scale_proj(cond)  # [B, D]
        shift = self.shift_proj(cond)  # [B, D]

        if x.dim() == 3:
            if x.shape[1] == self.feature_dim:
                # [B, D, T] format
                scale = scale.unsqueeze(-1)
                shift = shift.unsqueeze(-1)
            else:
                # [B, T, D] format
                scale = scale.unsqueeze(1)
                shift = shift.unsqueeze(1)

        return x * scale + shift


class ChainEncoder(nn.Module):
    """
    Encodes effect chain (types + params) into conditioning vector.
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

        mask = effect_types >= 0
        effect_types_safe = effect_types.clamp(min=0)

        effect_emb = self.effect_embedding(effect_types_safe)
        param_emb = self.param_encoder(effect_params)

        combined = self.combiner(torch.cat([effect_emb, param_emb], dim=-1))
        combined = combined * mask.unsqueeze(-1).float()

        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = combined.sum(dim=1) / mask_sum

        return self.pooler(pooled)


class TemporalConditionEncoder(nn.Module):
    """
    Encodes temporal features from audio domain into a conditioning vector.
    """

    def __init__(
        self,
        temporal_dim: int = 64,
        output_dim: int = 256,
    ):
        super().__init__()

        self.temporal_pool = nn.Sequential(
            nn.Conv1d(temporal_dim, output_dim, kernel_size=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        pooled = self.temporal_pool(temporal_features).squeeze(-1)
        return self.proj(pooled)


class RelativePositionalAttention(nn.Module):
    """Self-attention with relative positional encoding."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        max_rel_pos: int = 128,
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

        self.rel_pos_emb = nn.Embedding(2 * max_rel_pos + 1, n_heads)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        positions = torch.arange(N, device=x.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        rel_pos = rel_pos.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos

        pos_bias = self.rel_pos_emb(rel_pos)
        pos_bias = pos_bias.permute(2, 0, 1).unsqueeze(0)

        attn = attn + pos_bias
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        return self.out_proj(out)


class TransformerBlockWithFiLM(nn.Module):
    """Transformer block with FiLM conditioning."""

    def __init__(
        self,
        d_model: int,
        cond_dim: int,
        n_heads: int = 8,
        ff_mult: int = 4,
        max_rel_pos: int = 128,
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

        self.film1 = FiLM(d_model, cond_dim)
        self.film2 = FiLM(d_model, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        x_norm = self.film1(x_norm, cond)
        x = x + self.attn(x_norm)

        x_norm = self.norm2(x)
        x_norm = self.film2(x_norm, cond)
        x = x + self.ffn(x_norm)

        return x


class MambaBlockWithFiLM(nn.Module):
    """Mamba block wrapper with FiLM conditioning."""

    def __init__(self, d_model: int, cond_dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm not installed.")

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)
        self.film = FiLM(d_model, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        x_norm = self.film(x_norm, cond)
        return x + self.mamba(x_norm)


class LatentFlowMatchingInverter(nn.Module):
    """
    Latent Flow Matching Inverter using ACE-Step DCAE.

    Operates in DCAE latent space for efficiency:
    - ~4096x compression (512 mel hop * 8 DCAE factor)
    - Implicit phase reconstruction via HiFi-GAN vocoder
    - Handles stereo internally

    Key components:
    1. TemporalCorrelationEncoder: Captures temporal patterns in AUDIO domain
    2. DCAE Encoder/Decoder + Vocoder: Frozen, maps audio ↔ latent
    3. Backbone (Mamba or Transformer): Long-range dependencies in latent
    4. FiLM conditioning: Temporal + chain info modulates each layer
    """

    EFFECT_TYPES = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']

    EFFECT_DIFFICULTY = {
        'eq': 1,
        'gain': 1,
        'compressor': 4,
        'delay': 4,
        'chorus': 8,
        'reverb': 12,
        'distortion': 16,
    }

    # DCAE parameters
    DCAE_SAMPLE_RATE = 44100
    DCAE_HOP_LENGTH = 512
    DCAE_TIME_COMPRESSION = 8  # DCAE compresses mel time by 8x
    DCAE_TOTAL_COMPRESSION = 512 * 8  # = 4096

    def __init__(
        self,
        n_effects: int = 6,
        max_params: int = 15,
        d_model: int = 512,
        n_layers: int = 8,
        sample_rate: int = 44100,
        max_lag_ms: float = 2000.0,
        use_mamba: bool = False,
        dcae_checkpoint_path: Optional[str] = None,
        vocoder_checkpoint_path: Optional[str] = None,
    ):
        """
        Args:
            n_effects: Number of supported effect types
            max_params: Max parameters per effect
            d_model: Model hidden dimension
            n_layers: Number of backbone layers
            sample_rate: Audio sample rate
            max_lag_ms: Max temporal lag to consider
            use_mamba: Use Mamba backbone vs Transformer
            dcae_checkpoint_path: Path to DCAE checkpoint (uses default if None)
            vocoder_checkpoint_path: Path to vocoder checkpoint (uses default if None)
        """
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.sample_rate = sample_rate
        self.use_mamba = use_mamba and MAMBA_AVAILABLE

        # Load frozen DCAE codec
        if DCAE_AVAILABLE:
            dcae_path = dcae_checkpoint_path or DEFAULT_DCAE_PATH
            vocoder_path = vocoder_checkpoint_path or DEFAULT_VOCODER_PATH

            self.codec = MusicDCAE(
                source_sample_rate=sample_rate,
                dcae_checkpoint_path=dcae_path,
                vocoder_checkpoint_path=vocoder_path,
            )
            for param in self.codec.parameters():
                param.requires_grad = False
            self.codec.eval()

            # DCAE latent shape: (batch, channels, height, time)
            # We'll flatten channels*height for the flow network
            # From diffusers AutoencoderDC, typical latent channels is 8-16
            self.latent_channels = 8  # Will verify at runtime
            self.latent_height = 16   # Mel bins / DCAE factor
            self.latent_dim = self.latent_channels * self.latent_height  # Flattened
        else:
            self.codec = None
            self.latent_channels = 8
            self.latent_height = 16
            self.latent_dim = 128

        # Temporal feature extraction in AUDIO domain
        self.temporal_encoder = TemporalCorrelationEncoder(
            max_lag_ms=max_lag_ms,
            sample_rate=sample_rate,
            n_lags=64,
            output_dim=64,
        )

        # Pool temporal features to conditioning vector
        self.temporal_cond_encoder = TemporalConditionEncoder(
            temporal_dim=64,
            output_dim=d_model,
        )

        # Chain encoder
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

        # Combine all conditioning signals
        self.cond_combiner = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Latent input projection: latent_dim → d_model
        self.input_proj = nn.Conv1d(self.latent_dim, d_model, kernel_size=1)

        # Backbone layers with FiLM conditioning
        cond_dim = d_model
        if self.use_mamba:
            self.backbone = nn.ModuleList([
                MambaBlockWithFiLM(d_model, cond_dim, d_state=16, d_conv=4, expand=2)
                for _ in range(n_layers)
            ])
        else:
            max_rel_pos = 128  # Shorter sequences in latent space
            self.backbone = nn.ModuleList([
                TransformerBlockWithFiLM(d_model, cond_dim, n_heads=8, max_rel_pos=max_rel_pos)
                for _ in range(n_layers)
            ])

        # Output projection: d_model → latent_dim
        self.output_proj = nn.Conv1d(d_model, self.latent_dim, kernel_size=1)

    def _mono_to_stereo(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert mono [B, 1, T] to stereo [B, 2, T] for DCAE."""
        if audio.shape[1] == 1:
            return audio.repeat(1, 2, 1)
        return audio

    def _stereo_to_mono(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert stereo [B, 2, T] to mono [B, 1, T]."""
        if audio.shape[1] == 2:
            return audio.mean(dim=1, keepdim=True)
        return audio

    def _flatten_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Flatten DCAE latent from [B, C, H, T] to [B, C*H, T] for flow network.
        """
        B, C, H, T = latent.shape
        return latent.reshape(B, C * H, T)

    def _unflatten_latent(self, latent: torch.Tensor, channels: int, height: int) -> torch.Tensor:
        """
        Unflatten from [B, C*H, T] back to [B, C, H, T] for DCAE decoder.
        """
        B, _, T = latent.shape
        return latent.view(B, channels, height, T)

    @torch.no_grad()
    def encode_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Encode audio to DCAE latent space.

        Args:
            audio: [B, 1, T] mono audio at sample_rate

        Returns:
            latent: [B, latent_dim, T'] flattened latent
            info: dict with original shape info for decoding
        """
        if self.codec is None:
            raise RuntimeError("DCAE codec not loaded.")

        self.codec.eval()
        device = audio.device

        # Convert mono to stereo for DCAE
        audio_stereo = self._mono_to_stereo(audio)  # [B, 2, T]

        # Ensure audio is in valid range
        audio_stereo = torch.clamp(audio_stereo, -1, 1)

        # Get audio lengths
        B, _, T = audio_stereo.shape
        audio_lengths = torch.full((B,), T, device=device)

        # Encode: audio → mel → latent
        # Returns: latent [B, C, H, T'], latent_lengths
        latent, latent_lengths = self.codec.encode(
            audio_stereo,
            audio_lengths=audio_lengths,
            sr=self.sample_rate
        )

        # Store original shape for decoding
        info = {
            'original_shape': latent.shape,
            'channels': latent.shape[1],
            'height': latent.shape[2],
            'audio_length': T,
        }

        # Update instance vars if needed
        if latent.shape[1] != self.latent_channels:
            self.latent_channels = latent.shape[1]
        if latent.shape[2] != self.latent_height:
            self.latent_height = latent.shape[2]

        # Flatten for flow network: [B, C, H, T'] → [B, C*H, T']
        latent_flat = self._flatten_latent(latent)

        return latent_flat, info

    @torch.no_grad()
    def decode_latent(
        self,
        latent: torch.Tensor,
        info: Dict,
        original_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decode DCAE latent to audio.

        Args:
            latent: [B, latent_dim, T'] flattened latent
            info: dict with shape info from encode
            original_length: Original audio length for trimming

        Returns:
            audio: [B, 1, T] mono reconstructed audio
        """
        if self.codec is None:
            raise RuntimeError("DCAE codec not loaded.")

        self.codec.eval()
        device = latent.device

        # Unflatten: [B, C*H, T'] → [B, C, H, T']
        latent_4d = self._unflatten_latent(
            latent,
            channels=info['channels'],
            height=info['height']
        )

        # Get audio lengths for decoding
        B = latent.shape[0]
        audio_length = original_length or info['audio_length']
        audio_lengths = torch.full((B,), audio_length, device=device)

        # Decode: latent → mel → vocoder → audio
        sr, pred_wavs = self.codec.decode(
            latent_4d,
            audio_lengths=audio_lengths,
            sr=self.sample_rate
        )

        # Stack list of waveforms and convert to mono
        # pred_wavs is list of [2, T] tensors
        audio_stereo = torch.stack(pred_wavs, dim=0).to(device)  # [B, 2, T]
        audio_mono = self._stereo_to_mono(audio_stereo)  # [B, 1, T]

        # Trim/pad to original length
        if original_length is not None:
            if audio_mono.shape[-1] > original_length:
                audio_mono = audio_mono[..., :original_length]
            elif audio_mono.shape[-1] < original_length:
                audio_mono = F.pad(audio_mono, (0, original_length - audio_mono.shape[-1]))

        return audio_mono

    def velocity(
        self,
        latent: torch.Tensor,
        temporal_cond: torch.Tensor,
        chain_cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity field for flow matching in latent space.

        Args:
            latent: [B, latent_dim, T'] current latent state (flattened)
            temporal_cond: [B, d_model] temporal conditioning from audio
            chain_cond: [B, d_model] chain conditioning
            t: [B] timestep (0 = wet, 1 = dry)

        Returns:
            velocity: [B, latent_dim, T'] velocity towards dry latent
        """
        B = latent.shape[0]

        # Get time embedding
        time_cond = self.time_embed(t)

        # Combine all conditioning signals
        combined_cond = self.cond_combiner(
            torch.cat([temporal_cond, chain_cond, time_cond], dim=-1)
        )

        # Project latent to model dimension
        x = self.input_proj(latent)  # [B, d_model, T']
        x = x.transpose(1, 2)  # [B, T', d_model]

        # Apply backbone with FiLM conditioning
        for layer in self.backbone:
            x = layer(x, combined_cond)

        # Project back to latent dimension
        x = x.transpose(1, 2)  # [B, d_model, T']
        velocity = self.output_proj(x)  # [B, latent_dim, T']

        return velocity

    def forward(
        self,
        wet_audio: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Flow from wet → dry in latent space.

        Args:
            wet_audio: [B, 1, T] wet audio
            effect_types: [B, max_chain] effect indices
            effect_params: [B, max_chain, max_params]
            n_steps: Override number of steps (auto-selected if None)

        Returns:
            dry_audio: [B, 1, T]
        """
        B = wet_audio.shape[0]
        T_original = wet_audio.shape[-1]
        device = wet_audio.device

        # 1. Extract temporal features from AUDIO (critical for timing info)
        temporal_features = self.temporal_encoder(wet_audio)
        temporal_cond = self.temporal_cond_encoder(temporal_features)

        # 2. Get chain conditioning
        chain_cond = self.chain_encoder(effect_types, effect_params)

        # 3. Encode wet audio to latent
        wet_latent, latent_info = self.encode_audio(wet_audio)

        # 4. Auto-select steps based on effect difficulty
        if n_steps is None:
            n_steps = self.estimate_steps(effect_types)

        # 5. Euler integration from wet to dry IN LATENT SPACE
        z = wet_latent
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=device)
            v = self.velocity(z, temporal_cond, chain_cond, t)
            z = z + v * dt

        # 6. Decode dry latent to audio
        dry_audio = self.decode_latent(z, latent_info, original_length=T_original)

        return torch.clamp(dry_audio, -1, 1)

    def estimate_steps(self, effect_types: torch.Tensor) -> int:
        """Estimate optimal number of steps based on effect difficulty."""
        max_difficulty = 1

        for fx_idx in effect_types[0]:
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
    ) -> Dict[str, torch.Tensor]:
        """
        Flow matching training loss in latent space.
        """
        B = wet_audio.shape[0]
        device = wet_audio.device

        # Convert to mono if stereo
        if wet_audio.shape[1] == 2:
            wet_audio = wet_audio.mean(dim=1, keepdim=True)
        if dry_audio.shape[1] == 2:
            dry_audio = dry_audio.mean(dim=1, keepdim=True)

        # Debug: check input
        if torch.isnan(wet_audio).any():
            print(f"[DEBUG] NaN in wet_audio input")
        if torch.isnan(dry_audio).any():
            print(f"[DEBUG] NaN in dry_audio input")

        # 1. Extract temporal features from WET audio
        temporal_features = self.temporal_encoder(wet_audio)
        if torch.isnan(temporal_features).any():
            print(f"[DEBUG] NaN in temporal_features")
        temporal_cond = self.temporal_cond_encoder(temporal_features)
        if torch.isnan(temporal_cond).any():
            print(f"[DEBUG] NaN in temporal_cond")

        # 2. Get chain conditioning
        chain_cond = self.chain_encoder(effect_types, effect_params)
        if torch.isnan(chain_cond).any():
            print(f"[DEBUG] NaN in chain_cond")

        # 3. Encode both to latent space
        with torch.no_grad():
            wet_latent, _ = self.encode_audio(wet_audio)
            dry_latent, _ = self.encode_audio(dry_audio)
        if torch.isnan(wet_latent).any():
            print(f"[DEBUG] NaN in wet_latent")
        if torch.isnan(dry_latent).any():
            print(f"[DEBUG] NaN in dry_latent")

        # 4. Random timestep
        t = torch.rand(B, device=device)

        # 5. Interpolate between wet and dry latents
        z_t = t.view(-1, 1, 1) * dry_latent + (1 - t.view(-1, 1, 1)) * wet_latent

        # 6. Target velocity = dry - wet (constant velocity field)
        target_velocity = dry_latent - wet_latent

        # 7. Predict velocity
        pred_velocity = self.velocity(z_t, temporal_cond, chain_cond, t)
        if torch.isnan(pred_velocity).any():
            print(f"[DEBUG] NaN in pred_velocity")

        # 8. MSE loss in latent space
        flow_loss = F.mse_loss(pred_velocity, target_velocity)

        return {
            'flow_loss': flow_loss,
            'total': flow_loss,
        }

    @classmethod
    def effect_to_idx(cls, effect_name: str) -> int:
        return cls.EFFECT_TYPES.index(effect_name)

    @classmethod
    def idx_to_effect(cls, idx: int) -> str:
        return cls.EFFECT_TYPES[idx]


class LightweightLatentFlowInverter(nn.Module):
    """
    Lightweight version for faster inference.
    Uses simpler conv backbone instead of transformer.
    """

    EFFECT_TYPES = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']

    def __init__(
        self,
        n_effects: int = 6,
        max_params: int = 15,
        d_model: int = 256,
        sample_rate: int = 44100,
        dcae_checkpoint_path: Optional[str] = None,
        vocoder_checkpoint_path: Optional[str] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.sample_rate = sample_rate

        # Load frozen DCAE codec
        if DCAE_AVAILABLE:
            dcae_path = dcae_checkpoint_path or DEFAULT_DCAE_PATH
            vocoder_path = vocoder_checkpoint_path or DEFAULT_VOCODER_PATH

            self.codec = MusicDCAE(
                source_sample_rate=sample_rate,
                dcae_checkpoint_path=dcae_path,
                vocoder_checkpoint_path=vocoder_path,
            )
            for param in self.codec.parameters():
                param.requires_grad = False
            self.codec.eval()

            self.latent_channels = 8
            self.latent_height = 16
            self.latent_dim = self.latent_channels * self.latent_height
        else:
            self.codec = None
            self.latent_channels = 8
            self.latent_height = 16
            self.latent_dim = 128

        # Temporal encoder (audio domain)
        self.temporal_encoder = TemporalCorrelationEncoder(
            max_lag_ms=1000.0,
            sample_rate=sample_rate,
            n_lags=32,
            output_dim=32,
        )

        self.temporal_cond = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, d_model),
            nn.SiLU(),
        )

        self.chain_encoder = ChainEncoder(n_effects, max_params, d_model)

        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model),
        )

        self.cond_combine = nn.Linear(d_model * 3, d_model)

        # Simple conv backbone
        self.encoder = nn.Sequential(
            nn.Conv1d(self.latent_dim, d_model, 7, padding=3),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 7, stride=2, padding=3),
            nn.SiLU(),
        )

        self.middle = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, 7, padding=3),
                nn.SiLU(),
            )
            for _ in range(4)
        ])

        self.middle_film = nn.ModuleList([
            FiLM(d_model, d_model) for _ in range(4)
        ])

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model, 7, stride=2, padding=3, output_padding=1),
            nn.SiLU(),
            nn.Conv1d(d_model, self.latent_dim, 7, padding=3),
        )

    def _mono_to_stereo(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.shape[1] == 1:
            return audio.repeat(1, 2, 1)
        return audio

    def _stereo_to_mono(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.shape[1] == 2:
            return audio.mean(dim=1, keepdim=True)
        return audio

    def _flatten_latent(self, latent: torch.Tensor) -> torch.Tensor:
        B, C, H, T = latent.shape
        return latent.reshape(B, C * H, T)

    def _unflatten_latent(self, latent: torch.Tensor, channels: int, height: int) -> torch.Tensor:
        B, _, T = latent.shape
        return latent.view(B, channels, height, T)

    @torch.no_grad()
    def encode_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if self.codec is None:
            raise RuntimeError("DCAE codec not loaded.")

        self.codec.eval()
        device = audio.device

        audio_stereo = self._mono_to_stereo(audio)
        audio_stereo = torch.clamp(audio_stereo, -1, 1)

        B, _, T = audio_stereo.shape
        audio_lengths = torch.full((B,), T, device=device)

        latent, _ = self.codec.encode(audio_stereo, audio_lengths=audio_lengths, sr=self.sample_rate)

        info = {
            'original_shape': latent.shape,
            'channels': latent.shape[1],
            'height': latent.shape[2],
            'audio_length': T,
        }

        if latent.shape[1] != self.latent_channels:
            self.latent_channels = latent.shape[1]
        if latent.shape[2] != self.latent_height:
            self.latent_height = latent.shape[2]

        latent_flat = self._flatten_latent(latent)
        return latent_flat, info

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor, info: Dict, original_length: Optional[int] = None) -> torch.Tensor:
        if self.codec is None:
            raise RuntimeError("DCAE codec not loaded.")

        self.codec.eval()
        device = latent.device

        latent_4d = self._unflatten_latent(latent, info['channels'], info['height'])

        B = latent.shape[0]
        audio_length = original_length or info['audio_length']
        audio_lengths = torch.full((B,), audio_length, device=device)

        sr, pred_wavs = self.codec.decode(latent_4d, audio_lengths=audio_lengths, sr=self.sample_rate)

        audio_stereo = torch.stack(pred_wavs, dim=0).to(device)
        audio_mono = self._stereo_to_mono(audio_stereo)

        if original_length is not None:
            if audio_mono.shape[-1] > original_length:
                audio_mono = audio_mono[..., :original_length]
            elif audio_mono.shape[-1] < original_length:
                audio_mono = F.pad(audio_mono, (0, original_length - audio_mono.shape[-1]))

        return audio_mono

    def velocity(
        self,
        latent: torch.Tensor,
        temporal_cond: torch.Tensor,
        chain_cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        B, _, T_lat = latent.shape

        t_emb = self.time_embed(t)
        cond = self.cond_combine(torch.cat([temporal_cond, chain_cond, t_emb], dim=-1))

        x = self.encoder(latent)

        for conv, film in zip(self.middle, self.middle_film):
            x = conv(x)
            x = film(x, cond)

        x = self.decoder(x)

        if x.shape[-1] != T_lat:
            x = F.interpolate(x, size=T_lat, mode='linear', align_corners=False)

        return x

    def forward(
        self,
        wet_audio: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
        n_steps: int = 4,
    ) -> torch.Tensor:
        B = wet_audio.shape[0]
        T_original = wet_audio.shape[-1]
        device = wet_audio.device

        temporal_features = self.temporal_encoder(wet_audio)
        temporal_cond = self.temporal_cond(temporal_features)

        chain_cond = self.chain_encoder(effect_types, effect_params)

        wet_latent, latent_info = self.encode_audio(wet_audio)

        z = wet_latent
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=device)
            v = self.velocity(z, temporal_cond, chain_cond, t)
            z = z + v * dt

        dry_audio = self.decode_latent(z, latent_info, original_length=T_original)
        return torch.clamp(dry_audio, -1, 1)

    def training_step(
        self,
        wet_audio: torch.Tensor,
        dry_audio: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B = wet_audio.shape[0]
        device = wet_audio.device

        # Convert to mono if stereo
        if wet_audio.shape[1] == 2:
            wet_audio = wet_audio.mean(dim=1, keepdim=True)
        if dry_audio.shape[1] == 2:
            dry_audio = dry_audio.mean(dim=1, keepdim=True)

        temporal_features = self.temporal_encoder(wet_audio)
        temporal_cond = self.temporal_cond(temporal_features)

        chain_cond = self.chain_encoder(effect_types, effect_params)

        with torch.no_grad():
            wet_latent, _ = self.encode_audio(wet_audio)
            dry_latent, _ = self.encode_audio(dry_audio)

        t = torch.rand(B, device=device)
        z_t = t.view(-1, 1, 1) * dry_latent + (1 - t.view(-1, 1, 1)) * wet_latent
        target_velocity = dry_latent - wet_latent
        pred_velocity = self.velocity(z_t, temporal_cond, chain_cond, t)

        flow_loss = F.mse_loss(pred_velocity, target_velocity)

        return {
            'flow_loss': flow_loss,
            'total': flow_loss,
        }


def create_latent_flow_inverter(
    size: str = 'base',
    sample_rate: int = 44100,
    use_mamba: bool = False,
    dcae_checkpoint_path: Optional[str] = None,
    vocoder_checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    Create latent flow inverter with preset sizes.

    Args:
        size: 'tiny', 'small', 'base', 'large'
        sample_rate: Audio sample rate
        use_mamba: Use Mamba backbone if available
        dcae_checkpoint_path: Path to DCAE checkpoint
        vocoder_checkpoint_path: Path to vocoder checkpoint

    Returns:
        LatentFlowMatchingInverter or LightweightLatentFlowInverter
    """
    if size == 'tiny':
        return LightweightLatentFlowInverter(
            d_model=128,
            sample_rate=sample_rate,
            dcae_checkpoint_path=dcae_checkpoint_path,
            vocoder_checkpoint_path=vocoder_checkpoint_path,
        )

    configs = {
        'small': dict(d_model=256, n_layers=4),
        'base': dict(d_model=512, n_layers=8),
        'large': dict(d_model=768, n_layers=12),
    }

    cfg = configs.get(size, configs['base'])

    return LatentFlowMatchingInverter(
        sample_rate=sample_rate,
        use_mamba=use_mamba,
        dcae_checkpoint_path=dcae_checkpoint_path,
        vocoder_checkpoint_path=vocoder_checkpoint_path,
        **cfg,
    )


if __name__ == '__main__':
    print("Testing LatentFlowMatchingInverter with ACE-Step DCAE...")
    print(f"DCAE available: {DCAE_AVAILABLE}")
    print(f"Mamba available: {MAMBA_AVAILABLE}")

    if not DCAE_AVAILABLE:
        print("\nDCAE not available. Ensure ACE-Step is in PYTHONPATH:")
        print("  export PYTHONPATH=/home/arlo/Data/ACE-Step:$PYTHONPATH")
        print("Skipping full test")
    else:
        # Create model
        model = create_latent_flow_inverter('small', use_mamba=False)
        print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Dummy input
        B = 2
        T = 44100 * 2  # 2 seconds
        wet = torch.randn(B, 1, T)
        dry_gt = torch.randn(B, 1, T)
        effect_types = torch.tensor([[0, 5, -1, -1, -1, -1], [2, 4, -1, -1, -1, -1]])
        effect_params = torch.randn(B, 6, 15)

        # Test encoding
        print("\nTesting encode...")
        latent, info = model.encode_audio(wet)
        print(f"Latent shape: {latent.shape}")
        print(f"Original shape: {info['original_shape']}")
        print(f"Compression: {T} samples → {latent.shape[-1]} frames ({T / latent.shape[-1]:.1f}x)")

        # Forward pass
        print("\nTesting forward pass...")
        dry = model(wet, effect_types, effect_params, n_steps=2)
        print(f"Input: {wet.shape} → Output: {dry.shape}")

        # Training step
        print("\nTesting training step...")
        losses = model.training_step(wet, dry_gt, effect_types, effect_params)
        print(f"Training loss: {losses['total'].item():.4f}")

        print("\nTesting LightweightLatentFlowInverter...")
        light_model = create_latent_flow_inverter('tiny')
        print(f"Light model params: {sum(p.numel() for p in light_model.parameters()):,}")
        dry_light = light_model(wet, effect_types, effect_params, n_steps=2)
        print(f"Input: {wet.shape} → Output: {dry_light.shape}")

        print("\nAll tests passed!")
