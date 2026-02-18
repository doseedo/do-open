"""
Factorized Synth Model Components:
- FactorizedEncoder: mel spectrogram → [z_filter, z_envelope]
- Decoder: [z_filter, z_envelope] → mel spectrogram
- Normalizing Flows: z ↔ interpretable params
- Auxiliary classifiers for disentanglement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from gradient_reversal import GradientReversalLayer


# ============================================================
# Factorized Encoder
# ============================================================

class FactorizedEncoder(nn.Module):
    """
    Encode mel spectrogram into factorized latent codes.

    Output: Two separate latent vectors:
        z_filter: [batch, 16] - filter-related information
        z_envelope: [batch, 16] - envelope-related information
    """

    def __init__(self, n_mels: int = 64, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # Shared backbone (CNN over mel spectrogram)
        self.backbone = nn.Sequential(
            # [B, 1, n_mels, time]
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Separate heads for each factor
        self.head_filter = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )

        self.head_envelope = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel: [B, n_mels, time] mel spectrogram

        Returns:
            z_filter: [B, latent_dim]
            z_envelope: [B, latent_dim]
        """
        # Add channel dim
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)  # [B, 1, n_mels, time]

        features = self.backbone(mel)  # [B, 256]

        z_filter = self.head_filter(features)  # [B, latent_dim]
        z_envelope = self.head_envelope(features)  # [B, latent_dim]

        return z_filter, z_envelope


# ============================================================
# Decoder
# ============================================================

class Decoder(nn.Module):
    """
    Decode factorized latents back to mel spectrogram.
    """

    def __init__(self, n_mels: int = 64, n_frames: int = 345, latent_dim: int = 16):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames

        # Combine latents
        self.combine = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
        )

        # Upsample to mel shape
        self.upsample = nn.Sequential(
            nn.Linear(512, 8 * 4 * 22),  # Start shape
            nn.Unflatten(1, (8, 4, 22)),
            nn.ConvTranspose2d(8, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

        # Final projection to exact shape
        self.final = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, z_filter: torch.Tensor, z_envelope: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_filter: [B, latent_dim]
            z_envelope: [B, latent_dim]

        Returns:
            mel: [B, n_mels, n_frames]
        """
        z = torch.cat([z_filter, z_envelope], dim=-1)  # [B, latent_dim * 2]
        h = self.combine(z)  # [B, 512]
        mel = self.upsample(h)  # [B, 1, H, W]
        mel = self.final(mel)

        # Interpolate to exact shape
        mel = F.interpolate(mel, size=(self.n_mels, self.n_frames), mode='bilinear', align_corners=False)

        return mel.squeeze(1)  # [B, n_mels, n_frames]


# ============================================================
# Normalizing Flows (Simple Affine Coupling)
# ============================================================

class AffineCouplingLayer(nn.Module):
    """Single affine coupling layer."""

    def __init__(self, dim: int, hidden_dim: int = 64, mask_even: bool = True):
        super().__init__()
        self.dim = dim
        self.mask_even = mask_even

        # Create mask
        mask = torch.zeros(dim)
        if mask_even:
            mask[::2] = 1  # Even indices
        else:
            mask[1::2] = 1  # Odd indices
        self.register_buffer('mask', mask)

        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh(),
        )

        self.translate_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: z → z' with log det jacobian."""
        z_masked = z * self.mask

        s = self.scale_net(z_masked) * (1 - self.mask)
        t = self.translate_net(z_masked) * (1 - self.mask)

        z_prime = z_masked + (1 - self.mask) * (z * torch.exp(s) + t)

        log_det = (s * (1 - self.mask)).sum(dim=-1)

        return z_prime, log_det

    def inverse(self, z_prime: torch.Tensor) -> torch.Tensor:
        """Inverse: z' → z."""
        z_masked = z_prime * self.mask

        s = self.scale_net(z_masked) * (1 - self.mask)
        t = self.translate_net(z_masked) * (1 - self.mask)

        z = z_masked + (1 - self.mask) * ((z_prime - t) * torch.exp(-s))

        return z


class NormalizingFlow(nn.Module):
    """
    Simple normalizing flow for mapping latent z ↔ interpretable param.

    Maps: z [B, latent_dim] ↔ param [B, 1] (+ noise dims for invertibility)
    """

    def __init__(self, latent_dim: int = 16, n_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Stack of coupling layers (alternating masks)
        self.layers = nn.ModuleList([
            AffineCouplingLayer(latent_dim, hidden_dim, mask_even=(i % 2 == 0))
            for i in range(n_layers)
        ])

        # Map from single param to latent-dim (for conditioning)
        self.param_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: z → base distribution.
        Returns base samples and total log det jacobian.
        """
        log_det_total = 0

        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det

        return z, log_det_total

    def inverse(self, base: torch.Tensor) -> torch.Tensor:
        """Inverse pass: base distribution → z."""
        z = base
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

    def log_prob(self, z: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of z given the parameter.

        For training, we want to maximize this.
        """
        # Forward through flow
        base, log_det = self.forward(z)

        # The "target" in base space should be param-dependent
        target = self.param_embed(param)  # [B, latent_dim]

        # Gaussian log prob around target
        log_prob_base = -0.5 * ((base - target) ** 2).sum(dim=-1)

        return log_prob_base + log_det

    def sample(self, param: torch.Tensor, noise_scale: float = 0.1) -> torch.Tensor:
        """
        Sample z given parameter value.
        """
        batch_size = param.shape[0]
        device = param.device

        # Get target in base space
        target = self.param_embed(param)  # [B, latent_dim]

        # Sample around target
        base = target + torch.randn_like(target) * noise_scale

        # Inverse flow
        z = self.inverse(base)

        return z


# ============================================================
# Auxiliary Classifiers (for disentanglement)
# ============================================================

class AuxiliaryClassifier(nn.Module):
    """
    Classifier to predict one factor from another factor's latent.

    Used with gradient reversal to encourage disentanglement:
    - z_filter should NOT predict envelope attack
    - z_envelope should NOT predict filter cutoff
    """

    def __init__(self, latent_dim: int = 16, n_classes: int = 4):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=1.0)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, z: torch.Tensor, use_grl: bool = True) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim]
            use_grl: Whether to apply gradient reversal

        Returns:
            logits: [B, n_classes]
        """
        if use_grl:
            z = self.grl(z)
        return self.classifier(z)

    def set_alpha(self, alpha: float):
        """Update GRL strength."""
        self.grl.set_alpha(alpha)


# ============================================================
# Full Model
# ============================================================

class FactorizedSynthModel(nn.Module):
    """
    Complete factorized synth model.

    Components:
    - Encoder: mel → [z_filter, z_envelope]
    - Flows: z_filter ↔ cutoff_param, z_envelope ↔ attack_param
    - Decoder: [z_filter, z_envelope] → mel
    - Aux classifiers: for disentanglement training
    """

    def __init__(self, n_mels: int = 64, n_frames: int = 345, latent_dim: int = 16):
        super().__init__()

        self.encoder = FactorizedEncoder(n_mels, latent_dim)
        self.decoder = Decoder(n_mels, n_frames, latent_dim)

        self.flow_filter = NormalizingFlow(latent_dim, n_layers=4)
        self.flow_envelope = NormalizingFlow(latent_dim, n_layers=4)

        # Auxiliary classifiers for disentanglement
        self.aux_cutoff_from_env = AuxiliaryClassifier(latent_dim, n_classes=4)  # Should FAIL
        self.aux_attack_from_filter = AuxiliaryClassifier(latent_dim, n_classes=4)  # Should FAIL

    def encode(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode mel to factorized latents."""
        return self.encoder(mel)

    def decode(self, z_filter: torch.Tensor, z_envelope: torch.Tensor) -> torch.Tensor:
        """Decode factorized latents to mel."""
        return self.decoder(z_filter, z_envelope)

    def forward(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Returns:
            mel_recon: reconstructed mel
            z_filter: filter latent
            z_envelope: envelope latent
        """
        z_filter, z_envelope = self.encode(mel)
        mel_recon = self.decode(z_filter, z_envelope)
        return mel_recon, z_filter, z_envelope

    def extract_params(self, z_filter: torch.Tensor, z_envelope: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract interpretable parameters from latents using flow inverse.

        This is approximate - we find the param that maximizes flow likelihood.
        """
        # For simplicity, we'll use the first dimension of the base distribution
        base_filter, _ = self.flow_filter(z_filter)
        base_env, _ = self.flow_envelope(z_envelope)

        # The first dim roughly corresponds to the param
        cutoff_est = base_filter[:, 0:1]
        attack_est = base_env[:, 0:1]

        return cutoff_est, attack_est

    def generate_from_params(self, cutoff_norm: torch.Tensor, attack_norm: torch.Tensor,
                              noise_scale: float = 0.1) -> torch.Tensor:
        """
        Generate mel from interpretable parameters.
        """
        z_filter = self.flow_filter.sample(cutoff_norm, noise_scale)
        z_envelope = self.flow_envelope.sample(attack_norm, noise_scale)
        return self.decode(z_filter, z_envelope)

    def edit(self, mel: torch.Tensor, new_cutoff: Optional[torch.Tensor] = None,
             new_attack: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Edit an existing sound by changing specific parameters.

        This is THE KEY TEST: can we change filter without affecting envelope?
        """
        z_filter, z_envelope = self.encode(mel)

        if new_cutoff is not None:
            # Replace z_filter with one generated from new cutoff
            z_filter = self.flow_filter.sample(new_cutoff, noise_scale=0.01)

        if new_attack is not None:
            # Replace z_envelope with one generated from new attack
            z_envelope = self.flow_envelope.sample(new_attack, noise_scale=0.01)

        return self.decode(z_filter, z_envelope)


if __name__ == "__main__":
    # Test model shapes
    model = FactorizedSynthModel(n_mels=64, n_frames=345, latent_dim=16)

    mel = torch.randn(4, 64, 345)
    mel_recon, z_f, z_e = model(mel)

    print(f"Input mel: {mel.shape}")
    print(f"z_filter: {z_f.shape}")
    print(f"z_envelope: {z_e.shape}")
    print(f"Reconstructed mel: {mel_recon.shape}")

    # Test generation
    cutoff = torch.tensor([[0.5], [0.8]])
    attack = torch.tensor([[0.2], [0.9]])
    gen_mel = model.generate_from_params(cutoff, attack)
    print(f"Generated mel: {gen_mel.shape}")
