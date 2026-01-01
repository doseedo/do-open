"""
Mute Translator Models

Latent space translator for dry→muted trumpet conversion.
Works with ACE-Step DCAE latents [B, 8, H, T].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


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


class EnvelopeModifier(nn.Module):
    """
    Learns to modify attack envelopes in latent space.

    The key insight: muted trumpet attacks are softer/rounder than dry attacks.
    This module learns a time-varying gain that softens transients while
    preserving sustain characteristics.

    Input: latent [B, C, H, T], onsets [B, T], amp [B, T]
    Output: envelope-modified latent [B, C, H, T]
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_dim: int = 32,
        attack_window: int = 16,  # frames to consider around onset
    ):
        super().__init__()
        self.attack_window = attack_window

        # Learn attack envelope transformation from onset/amp conditioning
        # Input: concatenated [onset, amp, energy] features -> envelope modifier
        self.envelope_net = nn.Sequential(
            nn.Conv1d(3, hidden_dim, kernel_size=attack_window*2+1, padding=attack_window),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=attack_window+1, padding=attack_window//2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),  # Output is a gain multiplier 0-1
        )

        # Learnable bias toward 1.0 (no modification by default)
        self.gain_bias = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        latent: torch.Tensor,
        onsets: Optional[torch.Tensor] = None,
        amp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply learned envelope modification.

        Args:
            latent: [B, C, H, T] latent tensor
            onsets: [B, T] onset detection (0-1, higher = onset)
            amp: [B, T] amplitude envelope (0-1)

        Returns:
            Modified latent [B, C, H, T]
        """
        B, C, H, T = latent.shape

        # Compute energy from latent as fallback/additional feature
        energy = (latent ** 2).mean(dim=(1, 2))  # [B, T]
        energy = energy / (energy.max(dim=-1, keepdim=True)[0] + 1e-8)

        # Default onsets/amp if not provided
        if onsets is None:
            onsets = torch.zeros(B, T, device=latent.device)
        if amp is None:
            amp = energy

        # Ensure correct shape
        if onsets.shape[-1] != T:
            onsets = F.interpolate(onsets.unsqueeze(1), size=T, mode='linear', align_corners=False).squeeze(1)
        if amp.shape[-1] != T:
            amp = F.interpolate(amp.unsqueeze(1), size=T, mode='linear', align_corners=False).squeeze(1)

        # Stack features: [B, 3, T]
        features = torch.stack([onsets, amp, energy], dim=1)

        # Compute envelope gain: [B, 1, T]
        gain = self.envelope_net(features)  # [B, 1, T]

        # Bias toward no modification (1.0) with learnable offset
        gain = gain + self.gain_bias
        gain = gain.clamp(0.3, 1.0)  # Don't reduce by more than 70%

        # Apply gain across all channels and heights: [B, 1, 1, T]
        gain = gain.unsqueeze(2)  # [B, 1, 1, T]

        return latent * gain


class MuteTranslator(nn.Module):
    """
    Translates dry trumpet latents to muted trumpet latents.

    Architecture: Residual network operating on DCAE latent space.
    Input/Output: [B, 8, H, T] where H=height (frequency), T=time

    The model learns a residual mapping: muted = dry + f(dry)
    This preserves the melodic content while modifying timbre.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.latent_channels = latent_channels

        # Project to hidden space (flatten H dimension into channels)
        # DCAE latents are [B, 8, H, T] - we process as [B, 8*H, T]
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

        # Initialize output to near-zero for stable training start
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        # Learnable residual scale (starts small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # Learnable dry attenuation (starts at 1.0 = full pass-through for backward compat)
        # Old checkpoints don't have this key, so default must match original behavior
        self.dry_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, dry_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dry_latent: [B, C, H, T] dry trumpet latent

        Returns:
            muted_latent: [B, C, H, T] predicted muted trumpet latent
        """
        B, C, H, T = dry_latent.shape

        # Flatten H into batch for 1D processing: [B*H, C, T]
        x = dry_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)

        # Process
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        residual = self.output_proj(x)

        # Reshape back: [B, H, C, T] -> [B, C, H, T]
        residual = residual.reshape(B, H, C, T).permute(0, 2, 1, 3)

        # Apply with learnable scales: output = dry_scale * dry + residual_scale * residual
        # dry_scale can learn to be < 1 to reduce "both sounds playing" effect
        muted_latent = self.dry_scale * dry_latent + self.residual_scale * residual

        return muted_latent


class MuteTranslatorDirect(nn.Module):
    """
    Non-residual translator: output = f(dry), not dry + f(dry).

    This allows the model to REPLACE the dry timbre entirely rather than
    just adding muted character on top. Better for aggressive transformation.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 7,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, kernel_size, dilation=2**i)
            for i in range(num_blocks)
        ])

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, in_channels, 1),
        )

        # For direct mode, DON'T zero-init output - let it learn full transformation
        # Small init for stability but not zero
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, dry_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dry_latent: [B, C, H, T] dry trumpet latent

        Returns:
            muted_latent: [B, C, H, T] predicted muted trumpet latent (direct, not residual)
        """
        B, C, H, T = dry_latent.shape

        # Flatten H into batch for 1D processing: [B*H, C, T]
        x = dry_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)

        # Process
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        output = self.output_proj(x)

        # Reshape back: [B, H, C, T] -> [B, C, H, T]
        muted_latent = output.reshape(B, H, C, T).permute(0, 2, 1, 3)

        return muted_latent


class MuteTranslatorAdaptive(nn.Module):
    """
    Adaptive mixing translator: output = alpha(t) * dry + (1 - alpha(t)) * muted

    Key insight: The residual architecture forces dry to always pass through.
    The direct architecture destroys pitch entirely.

    This model learns a per-frame mixing coefficient alpha(t) that adaptively
    blends between preserving dry content and applying muted transformation.

    - alpha=1.0: output is pure dry (preserve pitch/content)
    - alpha=0.0: output is pure transformed (full muted character)

    The alpha predictor learns from the input when to preserve vs transform.
    E.g., during attacks it might preserve more (for pitch clarity),
    during sustain it might transform more (for muted timbre).
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        alpha_init: float = 0.3,  # Start with more muted by default
    ):
        super().__init__()

        # Transformation network (learns f(dry))
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, 1)

        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, kernel_size, dilation=2**i)
            for i in range(num_blocks)
        ])

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, in_channels, 1),
        )

        # Alpha predictor: learns per-frame mixing from input content
        # Takes dry latent, outputs alpha per frame
        self.alpha_net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim // 2, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
        )

        # Learnable bias toward desired alpha
        self.alpha_bias = nn.Parameter(torch.tensor(alpha_init))

        # Initialize transformation output small (not zero)
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, dry_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dry_latent: [B, C, H, T] dry trumpet latent

        Returns:
            muted_latent: [B, C, H, T] adaptively mixed output
        """
        B, C, H, T = dry_latent.shape

        # Flatten H into batch for 1D processing: [B*H, C, T]
        x = dry_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)

        # Transform network: learn full muted representation
        transformed = self.input_proj(x)
        for block in self.blocks:
            transformed = block(transformed)
        muted = self.output_proj(transformed)

        # Reshape muted back: [B, H, C, T] -> [B, C, H, T]
        muted = muted.reshape(B, H, C, T).permute(0, 2, 1, 3)

        # Predict alpha from dry input (averaged over H)
        # Input: [B, C, T] (mean over H)
        dry_for_alpha = dry_latent.mean(dim=2)  # [B, C, T]
        alpha_logits = self.alpha_net(dry_for_alpha)  # [B, 1, T]
        alpha = torch.sigmoid(alpha_logits + self.alpha_bias)  # [B, 1, T] in (0, 1)

        # Clamp to reasonable range
        alpha = alpha.clamp(0.1, 0.9)

        # Expand alpha for broadcasting: [B, 1, 1, T]
        alpha = alpha.unsqueeze(2)

        # Adaptive mixing: alpha controls how much dry vs muted
        output = alpha * dry_latent + (1 - alpha) * muted

        return output

    def forward_with_alpha(self, dry_latent: torch.Tensor) -> tuple:
        """Return output and alpha for visualization."""
        B, C, H, T = dry_latent.shape

        x = dry_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)
        transformed = self.input_proj(x)
        for block in self.blocks:
            transformed = block(transformed)
        muted = self.output_proj(transformed)
        muted = muted.reshape(B, H, C, T).permute(0, 2, 1, 3)

        dry_for_alpha = dry_latent.mean(dim=2)
        alpha_logits = self.alpha_net(dry_for_alpha)
        alpha = torch.sigmoid(alpha_logits + self.alpha_bias)
        alpha = alpha.clamp(0.1, 0.9)
        alpha_expanded = alpha.unsqueeze(2)

        output = alpha_expanded * dry_latent + (1 - alpha_expanded) * muted

        return output, alpha.squeeze(1)  # [B, T]


class MuteTranslatorLarge(nn.Module):
    """
    Larger translator with 2D convolutions for better H-T interaction.

    Uses full 2D convolutions on the latent [B, C, H, T] tensor.
    Better for capturing cross-frequency mute effects.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 32,
        num_blocks: int = 8,
    ):
        super().__init__()

        self.input_proj = nn.Conv2d(latent_channels, hidden_channels, 3, padding=1)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(8, hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(8, hidden_channels),
            ))

        self.output_proj = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_channels, latent_channels, 3, padding=1),
        )

        # Zero init for residual learning
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, dry_latent: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(dry_latent)

        for block in self.blocks:
            x = x + block(x)  # Residual connection

        residual = self.output_proj(x)
        return dry_latent + self.residual_scale * residual


class MuteDiscriminator(nn.Module):
    """
    Discriminator for adversarial training (optional).
    Classifies whether a latent is real muted or synthetic muted.
    """

    def __init__(self, latent_channels: int = 8, hidden_channels: int = 32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels * 4, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Returns logit for real/fake classification."""
        return self.net(latent)


class MuteTranslatorWithEnvelope(nn.Module):
    """
    Combined translator with envelope modification for realistic muted attacks.

    Two-stage process:
    1. Envelope modification: Soften attacks using learned envelope transform
    2. Timbral translation: Modify spectral characteristics

    This allows the model to both reshape attacks AND modify timbre.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        use_envelope: bool = True,
    ):
        super().__init__()
        self.use_envelope = use_envelope

        # Envelope modifier (learns attack softening)
        if use_envelope:
            self.envelope_mod = EnvelopeModifier(
                latent_channels=latent_channels,
                hidden_dim=32,
                attack_window=16,
            )

        # Timbral translator (same as before)
        self.timbral = MuteTranslator(
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
        )

    def forward(
        self,
        dry_latent: torch.Tensor,
        onsets: Optional[torch.Tensor] = None,
        amp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            dry_latent: [B, C, H, T] dry trumpet latent
            onsets: [B, T] onset detection
            amp: [B, T] amplitude envelope

        Returns:
            muted_latent: [B, C, H, T] predicted muted trumpet latent
        """
        # Stage 1: Envelope modification (soften attacks)
        if self.use_envelope:
            envelope_modified = self.envelope_mod(dry_latent, onsets, amp)
        else:
            envelope_modified = dry_latent

        # Stage 2: Timbral translation
        muted_latent = self.timbral(envelope_modified)

        return muted_latent

    def forward_with_intermediate(
        self,
        dry_latent: torch.Tensor,
        onsets: Optional[torch.Tensor] = None,
        amp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both intermediate envelope-modified and final output."""
        if self.use_envelope:
            envelope_modified = self.envelope_mod(dry_latent, onsets, amp)
        else:
            envelope_modified = dry_latent

        muted_latent = self.timbral(envelope_modified)
        return envelope_modified, muted_latent


class CycleConsistentMuteTranslator(nn.Module):
    """
    Bidirectional translator with cycle consistency.

    Trains two translators:
    - dry_to_muted: dry → muted
    - muted_to_dry: muted → dry

    With cycle consistency: dry → muted → dry ≈ dry
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.dry_to_muted = MuteTranslator(**kwargs)
        self.muted_to_dry = MuteTranslator(**kwargs)

    def forward(
        self,
        latent: torch.Tensor,
        direction: str = "dry_to_muted"
    ) -> torch.Tensor:
        if direction == "dry_to_muted":
            return self.dry_to_muted(latent)
        else:
            return self.muted_to_dry(latent)

    def cycle_forward(
        self,
        dry_latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (muted_pred, dry_reconstructed)"""
        muted = self.dry_to_muted(dry_latent)
        dry_recon = self.muted_to_dry(muted)
        return muted, dry_recon


if __name__ == "__main__":
    # Test models
    batch = torch.randn(2, 8, 16, 128)  # [B, C, H, T]
    onsets = torch.zeros(2, 128)
    onsets[:, 20] = 1.0  # Onset at frame 20
    onsets[:, 80] = 1.0  # Onset at frame 80
    amp = torch.rand(2, 128)

    print("Testing EnvelopeModifier...")
    env_mod = EnvelopeModifier()
    out = env_mod(batch, onsets, amp)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in env_mod.parameters()):,}")

    print("\nTesting MuteTranslator...")
    model = MuteTranslator()
    out = model(batch)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting MuteTranslatorWithEnvelope...")
    model_env = MuteTranslatorWithEnvelope()
    out = model_env(batch, onsets, amp)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model_env.parameters()):,}")

    print("\nTesting MuteTranslatorLarge...")
    model_large = MuteTranslatorLarge()
    out = model_large(batch)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model_large.parameters()):,}")

    print("\nTesting CycleConsistentMuteTranslator...")
    cycle_model = CycleConsistentMuteTranslator()
    muted, recon = cycle_model.cycle_forward(batch)
    print(f"  Muted: {muted.shape}, Reconstructed: {recon.shape}")
    print(f"  Params: {sum(p.numel() for p in cycle_model.parameters()):,}")
