"""
Distortion Inverter.

Distortion is fundamentally non-invertible as it destroys information.
Uses neural network to estimate a plausible dry signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Optional


class DistortionInverter(nn.Module):
    """
    Neural distortion inverter.

    Distortion/saturation applies nonlinear transformations that destroy
    information. This module estimates a plausible clean signal using
    two-stage processing:
    1. Mel spectrogram restoration
    2. Neural vocoder for waveform reconstruction

    Note: Perfect inversion is impossible for heavy distortion.
    """

    def __init__(
        self,
        sample_rate: float = 44100,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        # Mel restoration network (UNet-style)
        self.mel_restorer = MelRestorer(n_mels, hidden_dim)

        # Simple waveform generator (can be replaced with HiFi-GAN)
        self.waveform_generator = WaveformGenerator(
            n_mels=n_mels,
            hop_length=hop_length,
            hidden_dim=hidden_dim,
        )

        # Parameter conditioning
        self.param_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, hidden_dim),
            nn.Tanh(),  # Bound to [-1, 1] for stable conditioning
        )

    def forward(
        self,
        wet_audio: torch.Tensor,
        estimated_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attempt to remove distortion from audio.

        Args:
            wet_audio: Distorted audio [B, 1, T]
            estimated_params: Normalized distortion parameters [B, 4]
                (drive, tone, mix, output_gain)

        Returns:
            Dry audio estimate [B, 1, T]
        """
        bs, chs, seq_len = wet_audio.size()

        # Compute mel spectrogram
        mel = self.mel_transform(wet_audio.squeeze(1))  # [B, n_mels, T']
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

        # Normalize
        mel_db = (mel_db - mel_db.mean(dim=(1, 2), keepdim=True)) / (
            mel_db.std(dim=(1, 2), keepdim=True) + 1e-8
        )

        # Encode parameters
        param_features = self.param_encoder(estimated_params)

        # Restore mel spectrogram
        clean_mel = self.mel_restorer(mel_db, param_features)

        # Generate waveform
        dry_estimate = self.waveform_generator(clean_mel)

        # Adjust length
        if dry_estimate.size(-1) > seq_len:
            dry_estimate = dry_estimate[..., :seq_len]
        elif dry_estimate.size(-1) < seq_len:
            dry_estimate = F.pad(dry_estimate, (0, seq_len - dry_estimate.size(-1)))

        # Residual blending based on distortion amount
        # Less distortion = more of original signal preserved
        drive = estimated_params[:, 0:1].unsqueeze(-1)  # [B, 1, 1]
        mix = estimated_params[:, 2:3].unsqueeze(-1)

        # Lower drive/mix means less distortion, can keep more original
        blend_factor = drive * mix
        output = (1 - blend_factor * 0.5) * wet_audio + blend_factor * 0.5 * dry_estimate

        # Handle NaN and clamp
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.clamp(output, -1.0, 1.0)


class MelRestorer(nn.Module):
    """
    Restores clean mel spectrogram from distorted one.
    Uses 2D convolutions for time-frequency processing.
    """

    def __init__(self, n_mels: int, hidden_dim: int):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )

        # FiLM conditioning
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.SiLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 4, 1, 1),
        )

    def forward(
        self,
        mel: torch.Tensor,
        param_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Restore mel spectrogram.

        Args:
            mel: Distorted mel [B, n_mels, T]
            param_features: Parameter encoding [B, hidden_dim]

        Returns:
            Clean mel estimate [B, n_mels, T]
        """
        mel = mel.unsqueeze(1)  # [B, 1, n_mels, T]

        # Encoder
        e1 = self.enc1(mel)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck with FiLM conditioning
        x = self.bottleneck(e3)

        gamma = self.film_gamma(param_features).view(-1, x.size(1), 1, 1)
        beta = self.film_beta(param_features).view(-1, x.size(1), 1, 1)
        x = gamma * x + beta

        # Decoder with skip connections
        d3 = self.dec3(x)
        if d3.size() != e2.size():
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)
        if d2.size() != e1.size():
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)

        out = self.dec1(d2)

        return out.squeeze(1)  # [B, n_mels, T]


class WaveformGenerator(nn.Module):
    """
    Simple waveform generator from mel spectrogram.
    Can be replaced with HiFi-GAN or BigVGAN for better quality.
    """

    def __init__(
        self,
        n_mels: int = 80,
        hop_length: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.hop_length = hop_length

        # Upsample network
        self.pre = nn.Conv1d(n_mels, hidden_dim, 7, padding=3)

        # Upsampling layers
        upsample_rates = [8, 8, 4]  # Total: 256x
        self.upsamples = nn.ModuleList()

        in_ch = hidden_dim
        for rate in upsample_rates:
            self.upsamples.append(nn.Sequential(
                nn.ConvTranspose1d(in_ch, in_ch // 2, rate * 2, stride=rate, padding=rate // 2),
                nn.SiLU(),
            ))
            in_ch = in_ch // 2

        self.post = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, 7, padding=3),
            nn.SiLU(),
            nn.Conv1d(in_ch, 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Generate waveform from mel spectrogram.

        Args:
            mel: Mel spectrogram [B, n_mels, T]

        Returns:
            Waveform [B, 1, T * hop_length]
        """
        x = self.pre(mel)

        for upsample in self.upsamples:
            x = upsample(x)

        x = self.post(x)

        return x
