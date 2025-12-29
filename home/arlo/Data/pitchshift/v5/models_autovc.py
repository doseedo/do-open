#!/usr/bin/env python3
"""
AutoVC-style Register Disentanglement for DCAE Latent Space

Key design principles:
1. TEMPORAL DOWNSAMPLING in content encoder - forces information bottleneck
2. CROSS-SEGMENT training - register from different segment at same pitch

Architecture:
- Content Encoder: Temporal downsample → bottleneck (can't encode formants)
- Register Encoder: Global pool → fixed embedding (captures formant signature)
- Decoder: Upsample + register embedding → reconstruct
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentEncoder(nn.Module):
    """
    Encodes DCAE latent to temporally-downsampled content bottleneck.

    Key: Temporal stride reduces capacity so formants CAN'T fit.

    Input: [B, 8, 16, T] - DCAE latent
    Output: [B, bottleneck_dim, T//downsample] - Compressed content
    """

    def __init__(self, in_channels=8, h_dim=16, hidden_dim=256,
                 bottleneck_dim=32, downsample=8):
        super().__init__()

        self.downsample = downsample
        self.bottleneck_dim = bottleneck_dim
        flat_dim = in_channels * h_dim  # 128

        # Temporal downsampling encoder
        # Each strided conv halves temporal resolution
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

        # Final projection to bottleneck dim
        layers.extend([
            nn.Conv1d(hidden_dim, bottleneck_dim, kernel_size=3, padding=1),
        ])

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, T] DCAE latent
        Returns:
            content: [B, bottleneck_dim, T//downsample]
        """
        B, C, H, T = x.shape
        x = x.reshape(B, C * H, T)
        return self.encoder(x)


class RegisterEncoder(nn.Module):
    """
    Extracts global register embedding (formant signature).

    Pools over ALL time - captures average spectral characteristics.

    Input: [B, 8, 16, T] - DCAE latent
    Output: [B, register_dim] - Fixed-size register embedding
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
        """
        Args:
            x: [B, C, H, T]
        Returns:
            register_emb: [B, register_dim]
        """
        B, C, H, T = x.shape
        x = x.reshape(B, C * H, T)
        x = self.conv(x)  # [B, register_dim, T]
        return x.mean(dim=-1)  # Global average pool


class Decoder(nn.Module):
    """
    Reconstructs from content (downsampled) + register embedding.

    Upsamples content back to original resolution.
    """

    def __init__(self, out_channels=8, h_dim=16, hidden_dim=256,
                 bottleneck_dim=32, register_dim=64, upsample=8):
        super().__init__()

        self.out_channels = out_channels
        self.h_dim = h_dim
        self.upsample = upsample
        out_dim = out_channels * h_dim

        # Project register to match bottleneck
        self.register_proj = nn.Linear(register_dim, bottleneck_dim)

        combined_dim = bottleneck_dim * 2

        # Upsampling decoder
        layers = []

        # Initial conv
        layers.extend([
            nn.Conv1d(combined_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ])

        if upsample >= 8:
            # Upsample x2
            layers.extend([
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])

        # Upsample x2
        layers.extend([
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ])

        # Upsample x2
        layers.extend([
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ])

        # Final projection
        layers.extend([
            nn.Conv1d(hidden_dim, out_dim, kernel_size=5, padding=2),
        ])

        self.decoder = nn.Sequential(*layers)

    def forward(self, content, register_emb, target_length=None):
        """
        Args:
            content: [B, bottleneck_dim, T_down]
            register_emb: [B, register_dim]
            target_length: Original temporal length for matching
        Returns:
            reconstructed: [B, C, H, T]
        """
        B, D, T_down = content.shape

        # Project and broadcast register
        reg_proj = self.register_proj(register_emb)  # [B, bottleneck_dim]
        reg_proj = reg_proj.unsqueeze(-1).expand(-1, -1, T_down)

        # Combine
        combined = torch.cat([content, reg_proj], dim=1)

        # Decode with upsampling
        out = self.decoder(combined)  # [B, C*H, T_up]

        # Adjust to exact target length if specified
        if target_length is not None:
            T_out = out.shape[-1]
            if T_out > target_length:
                out = out[:, :, :target_length]
            elif T_out < target_length:
                out = F.pad(out, (0, target_length - T_out))

        return out.view(B, self.out_channels, self.h_dim, -1)


class AutoVCLatent(nn.Module):
    """
    Complete AutoVC-style model with temporal bottleneck.

    Training:
        - Content from segment A
        - Register from segment B (DIFFERENT segment, same pitch)
        - Target: reconstruct segment A

    This forces disentanglement - content can't encode formants.
    """

    def __init__(self,
                 in_channels=8,
                 h_dim=16,
                 hidden_dim=256,
                 bottleneck_dim=32,
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
            register_dim=register_dim,
            upsample=downsample,
        )

        self.downsample = downsample
        self.bottleneck_dim = bottleneck_dim
        self.register_dim = register_dim

    def forward(self, x_content, x_register):
        """
        Forward pass with SEPARATE content and register sources.

        Args:
            x_content: [B, C, H, T] - Segment to extract content from
            x_register: [B, C, H, T_r] - Segment to extract register from (can differ)
        Returns:
            reconstructed: [B, C, H, T]
            content: [B, bottleneck_dim, T//downsample]
            register_emb: [B, register_dim]
        """
        T = x_content.shape[-1]

        # Encode content (with temporal downsampling)
        content = self.content_encoder(x_content)

        # Encode register (global pooling)
        register_emb = self.register_encoder(x_register)

        # Decode
        reconstructed = self.decoder(content, register_emb, target_length=T)

        return reconstructed, content, register_emb

    def encode_content(self, x):
        return self.content_encoder(x)

    def encode_register(self, x):
        return self.register_encoder(x)

    def decode(self, content, register_emb, target_length=None):
        return self.decoder(content, register_emb, target_length)


def test_model():
    """Test the model architecture."""
    print("Testing AutoVCLatent with temporal downsampling...")

    model = AutoVCLatent(
        in_channels=8,
        h_dim=16,
        hidden_dim=256,
        bottleneck_dim=32,
        register_dim=64,
        downsample=8,
    )

    # Test input - content segment
    x_content = torch.randn(2, 8, 16, 128)  # 128 frames

    # Register from different segment (different length)
    x_register = torch.randn(2, 8, 16, 64)  # 64 frames

    # Forward pass
    reconstructed, content, register_emb = model(x_content, x_register)

    print(f"Content input: {x_content.shape}")
    print(f"Register input: {x_register.shape}")
    print(f"Content bottleneck: {content.shape}")
    print(f"  -> Temporal compression: {x_content.shape[-1]} → {content.shape[-1]} ({x_content.shape[-1] // content.shape[-1]}x)")
    print(f"  -> Total bottleneck capacity: {content.shape[1] * content.shape[2]} values")
    print(f"Register embedding: {register_emb.shape}")
    print(f"Reconstructed: {reconstructed.shape}")

    # Verify bottleneck is tight
    input_capacity = 8 * 16 * x_content.shape[-1]  # Original
    bottleneck_capacity = content.shape[1] * content.shape[-1]  # Compressed
    print(f"\nCapacity analysis:")
    print(f"  Input: {input_capacity} values")
    print(f"  Bottleneck: {bottleneck_capacity} values")
    print(f"  Compression ratio: {input_capacity / bottleneck_capacity:.1f}x")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_model()
