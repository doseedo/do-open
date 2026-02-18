"""
Unified HDemucs-style Inverter.

Single model that inverts ANY effect chain in one forward pass.
Architecture: Hybrid time-frequency processing with chain conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for conditioning."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.scale = nn.Linear(cond_dim, channels)
        self.shift = nn.Linear(cond_dim, channels)

        # Initialize to identity
        nn.init.zeros_(self.scale.weight)
        nn.init.ones_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] or [B, C, H, W]
            condition: [B, cond_dim]
        """
        scale = self.scale(condition)
        shift = self.shift(condition)

        if x.dim() == 3:
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        elif x.dim() == 4:
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)

        return x * scale + shift


class TimeEncoderBlock(nn.Module):
    """Time-domain encoder block with strided convolution."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 4, kernel: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class TimeDecoderBlock(nn.Module):
    """Time-domain decoder block with transposed convolution."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 4, kernel: int = 8):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_ch, out_ch, kernel, stride=stride,
            padding=kernel // 2, output_padding=stride - 1
        )
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class FreqEncoderBlock(nn.Module):
    """Frequency-domain encoder block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=(1, stride), padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class FreqDecoderBlock(nn.Module):
    """Frequency-domain decoder block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_ch, out_ch, 3, stride=(1, stride),
            padding=1, output_padding=(0, stride - 1)
        )
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ChainEncoder(nn.Module):
    """Encodes effect chain (types + params) into conditioning vector."""

    def __init__(
        self,
        n_effects: int = 6,
        max_params: int = 15,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()

        # Effect type embedding (+1 for padding token)
        self.effect_embedding = nn.Embedding(n_effects + 1, embed_dim, padding_idx=n_effects)

        # Parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(max_params, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Positional encoding for chain order
        self.pos_encoding = nn.Parameter(torch.randn(1, 8, embed_dim) * 0.02)

        # Transformer to aggregate chain
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Pool token (learnable)
        self.pool_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(
        self,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            effect_types: [B, max_chain] - indices, -1 for padding
            effect_params: [B, max_chain, max_params] - normalized params

        Returns:
            chain_embedding: [B, embed_dim]
        """
        B, L = effect_types.shape

        # Create padding mask (True = ignore)
        padding_mask = effect_types < 0

        # Clamp effect types for embedding lookup
        effect_types_safe = effect_types.clamp(min=0)

        # Get embeddings
        effect_emb = self.effect_embedding(effect_types_safe)  # [B, L, D]
        param_emb = self.param_encoder(effect_params)          # [B, L, D]

        # Combine
        chain_emb = effect_emb + param_emb + self.pos_encoding[:, :L, :]

        # Zero out padding
        chain_emb = chain_emb.masked_fill(padding_mask.unsqueeze(-1), 0)

        # Prepend pool token
        pool_tokens = self.pool_token.expand(B, -1, -1)
        sequence = torch.cat([pool_tokens, chain_emb], dim=1)  # [B, 1+L, D]

        # Extend padding mask for pool token (never masked)
        extended_mask = torch.cat([
            torch.zeros(B, 1, device=padding_mask.device, dtype=torch.bool),
            padding_mask
        ], dim=1)

        # Transformer
        out = self.transformer(sequence, src_key_padding_mask=extended_mask)

        # Return pool token output
        return out[:, 0, :]  # [B, embed_dim]


class HDemucsInverter(nn.Module):
    """
    Hybrid Demucs-style unified inverter.

    Processes audio through dual time-frequency paths with fusion.
    Conditioned on effect chain via FiLM layers.
    """

    EFFECT_TYPES = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']

    def __init__(
        self,
        n_effects: int = 6,
        max_params: int = 15,
        max_chain_length: int = 6,
        channels: int = 48,
        depth: int = 5,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 48000,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.depth = depth
        self.sample_rate = sample_rate

        # Chain encoder
        self.chain_encoder = ChainEncoder(
            n_effects=n_effects,
            max_params=max_params,
            embed_dim=128,
        )
        cond_dim = 128

        # ===== Time Domain Branch =====
        self.time_encoder = nn.ModuleList()
        self.time_decoder = nn.ModuleList()
        self.time_film = nn.ModuleList()

        ch = 1
        encoder_channels = []
        for i in range(depth):
            out_ch = channels * (2 ** min(i, 3))
            self.time_encoder.append(TimeEncoderBlock(ch, out_ch, stride=4, kernel=8))
            self.time_film.append(FiLM(out_ch, cond_dim))
            encoder_channels.append(out_ch)
            ch = out_ch

        self.time_bottleneck = nn.Sequential(
            nn.Conv1d(ch, ch * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(ch * 2, ch, 3, padding=1),
        )

        for i in range(depth - 1, -1, -1):
            in_ch = encoder_channels[i] * 2  # Skip connection
            out_ch = encoder_channels[i - 1] if i > 0 else 1
            self.time_decoder.append(TimeDecoderBlock(in_ch, out_ch, stride=4, kernel=8))

        # ===== Frequency Domain Branch =====
        self.freq_encoder = nn.ModuleList()
        self.freq_decoder = nn.ModuleList()
        self.freq_film = nn.ModuleList()

        freq_ch = 2  # Real + Imag
        freq_encoder_channels = []
        for i in range(depth):
            out_ch = channels * (2 ** min(i, 3))
            self.freq_encoder.append(FreqEncoderBlock(freq_ch, out_ch, stride=2))
            self.freq_film.append(FiLM(out_ch, cond_dim))
            freq_encoder_channels.append(out_ch)
            freq_ch = out_ch

        self.freq_bottleneck = nn.Sequential(
            nn.Conv2d(freq_ch, freq_ch * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(freq_ch * 2, freq_ch, 3, padding=1),
        )

        for i in range(depth - 1, -1, -1):
            in_ch = freq_encoder_channels[i] * 2
            out_ch = freq_encoder_channels[i - 1] if i > 0 else 2
            self.freq_decoder.append(FreqDecoderBlock(in_ch, out_ch, stride=2))

        # ===== Fusion =====
        self.fusion = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 32, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 1, 1),
        )

        # Residual connection weight - DISABLED (model was learning degenerate solution)
        # self.residual_weight = nn.Parameter(torch.tensor(0.1))
        self.register_buffer('residual_weight', torch.tensor(0.0))

        # Register STFT window
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(
        self,
        wet_audio: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            wet_audio: [B, 1, T] - wet audio
            effect_types: [B, max_chain] - effect indices (-1 for padding)
            effect_params: [B, max_chain, max_params] - normalized params

        Returns:
            dry_audio: [B, 1, T]
        """
        B, _, T = wet_audio.shape

        # Encode chain
        chain_cond = self.chain_encoder(effect_types, effect_params)  # [B, 128]

        # Time domain branch
        time_out = self._process_time(wet_audio, chain_cond)

        # Frequency domain branch
        freq_out = self._process_freq(wet_audio, chain_cond, T)

        # Match lengths before fusion
        min_len = min(time_out.shape[-1], freq_out.shape[-1], T)
        time_out = time_out[..., :min_len]
        freq_out = freq_out[..., :min_len]

        # Fusion
        combined = torch.cat([time_out, freq_out], dim=1)  # [B, 2, T]
        dry_audio = self.fusion(combined)  # [B, 1, min_len]

        # Pad back to original length if needed
        if dry_audio.shape[-1] < T:
            dry_audio = F.pad(dry_audio, (0, T - dry_audio.shape[-1]))

        # Residual connection (helps with identity for small effects)
        dry_audio = dry_audio + self.residual_weight * wet_audio

        return torch.clamp(dry_audio, -1, 1)

    def _process_time(
        self,
        x: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """Process through time-domain UNet."""
        skips = []

        # Encoder
        for enc, film in zip(self.time_encoder, self.time_film):
            x = enc(x)
            x = film(x, cond)
            skips.append(x)

        # Bottleneck
        x = self.time_bottleneck(x)

        # Decoder with skip connections
        for i, dec in enumerate(self.time_decoder):
            skip = skips[-(i + 1)]

            # Handle size mismatch from strided conv
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, (0, diff))

            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return x

    def _process_freq(
        self,
        audio: torch.Tensor,
        cond: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """Process through frequency-domain UNet."""
        # STFT
        spec = torch.stft(
            audio.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )  # [B, F, T]

        x = torch.stack([spec.real, spec.imag], dim=1)  # [B, 2, F, T]

        skips = []

        # Encoder
        for enc, film in zip(self.freq_encoder, self.freq_film):
            x = enc(x)
            x = film(x, cond)
            skips.append(x)

        # Bottleneck
        x = self.freq_bottleneck(x)

        # Decoder
        for i, dec in enumerate(self.freq_decoder):
            skip = skips[-(i + 1)]

            # Handle size mismatch
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        # Match original spec size
        if x.shape[-1] != spec.shape[-1] or x.shape[-2] != spec.shape[-2]:
            x = F.interpolate(x, size=spec.shape[-2:], mode='bilinear', align_corners=False)

        # iSTFT
        spec_out = torch.complex(x[:, 0], x[:, 1])
        audio_out = torch.istft(
            spec_out,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            length=target_length,
        )

        return audio_out.unsqueeze(1)

    @classmethod
    def effect_to_idx(cls, effect_name: str) -> int:
        """Convert effect name to index."""
        return cls.EFFECT_TYPES.index(effect_name)

    @classmethod
    def idx_to_effect(cls, idx: int) -> str:
        """Convert index to effect name."""
        return cls.EFFECT_TYPES[idx]


# ===== Convenience Functions =====

def create_unified_inverter(
    size: str = 'base',
    sample_rate: int = 48000,
) -> HDemucsInverter:
    """Create unified inverter with preset sizes."""

    configs = {
        'small': dict(channels=32, depth=4),
        'base': dict(channels=48, depth=5),
        'large': dict(channels=64, depth=6),
    }

    cfg = configs.get(size, configs['base'])

    return HDemucsInverter(
        sample_rate=sample_rate,
        **cfg,
    )


if __name__ == '__main__':
    # Test
    model = create_unified_inverter('base')
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy input
    B = 2
    T = 48000 * 3  # 3 seconds
    wet = torch.randn(B, 1, T)
    effect_types = torch.tensor([[0, 2, -1, -1, -1, -1], [1, 3, 4, -1, -1, -1]])  # EQ+reverb, comp+dist+chorus
    effect_params = torch.randn(B, 6, 15)

    # Forward
    dry = model(wet, effect_types, effect_params)
    print(f"Input: {wet.shape} -> Output: {dry.shape}")
