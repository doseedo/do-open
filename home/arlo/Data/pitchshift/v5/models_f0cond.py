#!/usr/bin/env python3
"""
F0-Conditioned Register Disentanglement for DCAE Latent Space

Key insight: With explicit F0, the content bottleneck only needs to capture
timing/dynamics/articulation - NOT pitch. This allows much tighter bottleneck
without destroying the signal.

Architecture:
- Content Encoder: Very tight bottleneck (timing/dynamics only)
- F0 Embedding: Frame-wise pitch information (externally extracted)
- Register Encoder: Global pool → formant signature
- Decoder: Combines all three to reconstruct
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentEncoder(nn.Module):
    """
    Encodes DCAE latent to tight content bottleneck.

    With F0 provided externally, this only needs to capture:
    - Timing (note onsets/offsets)
    - Dynamics (amplitude envelope)
    - Articulation (attack character)

    Can be VERY tight since pitch is handled by F0.
    """

    def __init__(self, in_channels=8, h_dim=16, hidden_dim=256,
                 bottleneck_dim=32, downsample=8):
        super().__init__()

        self.downsample = downsample
        self.bottleneck_dim = bottleneck_dim
        flat_dim = in_channels * h_dim  # 128

        # Temporal downsampling encoder
        layers = []

        # [B, 128, T] -> [B, hidden, T/2]
        layers.extend([
            nn.Conv1d(flat_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ])

        # [B, hidden, T/2] -> [B, hidden, T/4]
        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ])

        if downsample >= 8:
            # [B, hidden, T/4] -> [B, hidden, T/8]
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])

        if downsample >= 16:
            # [B, hidden, T/8] -> [B, hidden, T/16]
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])

        # Final projection to bottleneck
        layers.extend([
            nn.Conv1d(hidden_dim, bottleneck_dim, kernel_size=3, padding=1),
        ])

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        B, C, H, T = x.shape
        x = x.reshape(B, C * H, T)
        return self.encoder(x)


class F0Embedding(nn.Module):
    """
    Embeds F0 (in Hz) to a learned representation.

    Handles:
    - Log-scale F0 (perceptually uniform)
    - Unvoiced frames (F0=0)
    """

    def __init__(self, f0_dim=64, hidden_dim=128):
        super().__init__()

        # Separate embedding for voiced/unvoiced
        self.voiced_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, f0_dim),
        )

        # Learnable unvoiced token
        self.unvoiced_token = nn.Parameter(torch.randn(f0_dim) * 0.01)

    def forward(self, f0):
        """
        Args:
            f0: [B, T] F0 in Hz, 0 for unvoiced
        Returns:
            f0_emb: [B, f0_dim, T]
        """
        B, T = f0.shape

        # Convert to log scale (add small epsilon to avoid log(0))
        # Use MIDI-like scaling: 12 * log2(f0/440) + 69
        voiced_mask = (f0 > 20).float()  # Below 20Hz is unvoiced

        # Safe log: clamp to avoid log(0)
        f0_safe = torch.clamp(f0, min=20.0)
        f0_log = 12 * torch.log2(f0_safe / 440.0) + 69  # MIDI scale
        f0_log = f0_log / 127.0  # Normalize to ~[0, 1]

        # Embed voiced frames
        f0_input = f0_log.unsqueeze(-1)  # [B, T, 1]
        voiced_emb = self.voiced_embed(f0_input)  # [B, T, f0_dim]

        # Apply unvoiced token where F0 is 0
        unvoiced_emb = self.unvoiced_token.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

        # Blend based on voiced mask
        voiced_mask = voiced_mask.unsqueeze(-1)  # [B, T, 1]
        f0_emb = voiced_mask * voiced_emb + (1 - voiced_mask) * unvoiced_emb

        return f0_emb.transpose(1, 2)  # [B, f0_dim, T]


class RegisterEncoder(nn.Module):
    """
    Extracts global register embedding (formant signature).
    Same as AutoVC version - pools over time.
    """

    def __init__(self, in_channels=8, h_dim=16, hidden_dim=256, register_dim=64):
        super().__init__()

        flat_dim = in_channels * h_dim

        self.conv = nn.Sequential(
            nn.Conv1d(flat_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, register_dim, kernel_size=1),
        )

    def forward(self, x):
        B, C, H, T = x.shape
        x = x.reshape(B, C * H, T)
        x = self.conv(x)
        return x.mean(dim=-1)  # Global average pool


class Decoder(nn.Module):
    """
    Reconstructs from content + F0 + register.

    - Content: upsampled timing/dynamics
    - F0: frame-wise pitch (at full resolution)
    - Register: global formant signature (broadcast)
    """

    def __init__(self, out_channels=8, h_dim=16, hidden_dim=256,
                 bottleneck_dim=32, f0_dim=64, register_dim=64, upsample=8):
        super().__init__()

        self.out_channels = out_channels
        self.h_dim = h_dim
        self.upsample = upsample
        out_dim = out_channels * h_dim

        # Project register to match decoder hidden
        self.register_proj = nn.Linear(register_dim, hidden_dim)

        # Upsample content
        up_layers = []

        # Initial projection
        up_layers.extend([
            nn.Conv1d(bottleneck_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ])

        if upsample >= 16:
            up_layers.extend([
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])

        if upsample >= 8:
            up_layers.extend([
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])

        # x2
        up_layers.extend([
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ])

        # x2
        up_layers.extend([
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ])

        self.content_upsample = nn.Sequential(*up_layers)

        # Final decoder: combines upsampled content + F0 + register
        # Content: hidden_dim, F0: f0_dim, Register: hidden_dim (broadcast)
        combined_dim = hidden_dim + f0_dim + hidden_dim

        self.final_decoder = nn.Sequential(
            nn.Conv1d(combined_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, out_dim, kernel_size=5, padding=2),
        )

    def forward(self, content, f0_emb, register_emb, target_length=None):
        """
        Args:
            content: [B, bottleneck_dim, T_down]
            f0_emb: [B, f0_dim, T]
            register_emb: [B, register_dim]
            target_length: Original T for matching
        Returns:
            reconstructed: [B, C, H, T]
        """
        B = content.shape[0]
        T = f0_emb.shape[-1]

        # Upsample content
        content_up = self.content_upsample(content)  # [B, hidden, T_up]

        # Adjust to target length
        if target_length is not None:
            T_up = content_up.shape[-1]
            if T_up > target_length:
                content_up = content_up[:, :, :target_length]
            elif T_up < target_length:
                content_up = F.pad(content_up, (0, target_length - T_up))

        T_out = content_up.shape[-1]

        # Match F0 length to content
        if f0_emb.shape[-1] != T_out:
            f0_emb = F.interpolate(f0_emb, size=T_out, mode='linear', align_corners=False)

        # Project and broadcast register
        reg_proj = self.register_proj(register_emb)  # [B, hidden_dim]
        reg_broadcast = reg_proj.unsqueeze(-1).expand(-1, -1, T_out)  # [B, hidden_dim, T]

        # Combine all three
        combined = torch.cat([content_up, f0_emb, reg_broadcast], dim=1)

        # Final decode
        out = self.final_decoder(combined)  # [B, out_dim, T]

        return out.view(B, self.out_channels, self.h_dim, T_out)


class F0ConditionedModel(nn.Module):
    """
    Complete F0-conditioned model for register disentanglement.

    Training:
        - Content from segment A (tight bottleneck: timing/dynamics only)
        - F0 from segment A (explicit pitch)
        - Register from segment B (different segment, same pitch bin)
        - Target: reconstruct segment A

    Inference:
        - Extract content from pitch-shifted audio
        - Use F0 of TARGET pitch
        - Use register embedding for TARGET pitch
        - Decode to corrected audio
    """

    def __init__(self,
                 in_channels=8,
                 h_dim=16,
                 hidden_dim=256,
                 bottleneck_dim=16,  # Can be MUCH tighter now
                 f0_dim=64,
                 register_dim=64,
                 downsample=8):
        super().__init__()

        self.content_encoder = ContentEncoder(
            in_channels=in_channels,
            h_dim=h_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            downsample=downsample,
        )

        self.f0_embedding = F0Embedding(
            f0_dim=f0_dim,
            hidden_dim=hidden_dim // 2,
        )

        self.register_encoder = RegisterEncoder(
            in_channels=in_channels,
            h_dim=h_dim,
            hidden_dim=hidden_dim,
            register_dim=register_dim,
        )

        self.decoder = Decoder(
            out_channels=in_channels,
            h_dim=h_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            f0_dim=f0_dim,
            register_dim=register_dim,
            upsample=downsample,
        )

        self.downsample = downsample
        self.bottleneck_dim = bottleneck_dim
        self.f0_dim = f0_dim
        self.register_dim = register_dim

    def forward(self, x_content, f0, x_register):
        """
        Forward pass with separate content, F0, and register sources.

        Args:
            x_content: [B, C, H, T] - Segment for content extraction
            f0: [B, T] - F0 in Hz (from same segment as x_content)
            x_register: [B, C, H, T_r] - Segment for register extraction
        Returns:
            reconstructed: [B, C, H, T]
            content: [B, bottleneck_dim, T//downsample]
            register_emb: [B, register_dim]
        """
        T = x_content.shape[-1]

        # Encode content (tight bottleneck)
        content = self.content_encoder(x_content)

        # Embed F0
        f0_emb = self.f0_embedding(f0)

        # Encode register
        register_emb = self.register_encoder(x_register)

        # Decode
        reconstructed = self.decoder(content, f0_emb, register_emb, target_length=T)

        return reconstructed, content, register_emb

    def encode_content(self, x):
        return self.content_encoder(x)

    def encode_f0(self, f0):
        return self.f0_embedding(f0)

    def encode_register(self, x):
        return self.register_encoder(x)

    def decode(self, content, f0_emb, register_emb, target_length=None):
        return self.decoder(content, f0_emb, register_emb, target_length)


def test_model():
    """Test the F0-conditioned model."""
    print("Testing F0ConditionedModel...")

    model = F0ConditionedModel(
        in_channels=8,
        h_dim=16,
        hidden_dim=256,
        bottleneck_dim=16,  # Very tight!
        f0_dim=64,
        register_dim=64,
        downsample=8,
    )

    # Test inputs
    B, T = 2, 256
    x_content = torch.randn(B, 8, 16, T)
    f0 = torch.rand(B, T) * 500 + 100  # Random F0 in 100-600 Hz
    f0[0, :50] = 0  # Some unvoiced frames
    x_register = torch.randn(B, 8, 16, 128)  # Different length

    # Forward
    reconstructed, content, register_emb = model(x_content, f0, x_register)

    print(f"Content input: {x_content.shape}")
    print(f"F0 input: {f0.shape}")
    print(f"Register input: {x_register.shape}")
    print(f"Content bottleneck: {content.shape}")
    print(f"  -> Temporal compression: {T} → {content.shape[-1]} ({T // content.shape[-1]}x)")
    print(f"  -> Bottleneck capacity: {content.shape[1] * content.shape[-1]} values")
    print(f"Register embedding: {register_emb.shape}")
    print(f"Reconstructed: {reconstructed.shape}")

    # Capacity analysis
    input_capacity = 8 * 16 * T
    bottleneck_capacity = content.shape[1] * content.shape[-1]
    print(f"\nCapacity analysis:")
    print(f"  Input: {input_capacity} values")
    print(f"  Bottleneck: {bottleneck_capacity} values")
    print(f"  Compression ratio: {input_capacity / bottleneck_capacity:.1f}x")
    print(f"  (F0 provides pitch separately: {f0.shape[-1]} frames)")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_model()
