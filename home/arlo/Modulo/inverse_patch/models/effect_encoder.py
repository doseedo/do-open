"""
Effect Encoder Module.

Encodes wet audio into effect-aware embeddings using mel spectrogram features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List


@dataclass
class EffectEncoderConfig:
    """Configuration for EffectEncoder."""
    embedding_dim: int = 512
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    backbone: str = 'efficientnet'  # 'efficientnet', 'resnet', 'convnext'
    pretrained: bool = True
    freeze_backbone: bool = False
    num_effect_types: int = 6


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, c, _, _ = x.size()
        y = self.pool(x).view(bs, c)
        y = self.fc(y).view(bs, c, 1, 1)
        return x * y


class AudioEncoder(nn.Module):
    """
    Custom audio encoder backbone.
    Optimized for mel spectrogram processing.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_layers: int = 4,
        output_dim: int = 1280,
    ):
        super().__init__()

        channels = [base_channels * (2 ** i) for i in range(num_layers)]
        channels = [in_channels] + channels

        layers = []
        for i in range(num_layers):
            layers.append(ConvBlock(
                channels[i], channels[i + 1],
                kernel_size=3, stride=2, padding=1
            ))
            layers.append(SEBlock(channels[i + 1]))
            layers.append(ConvBlock(
                channels[i + 1], channels[i + 1],
                kernel_size=3, stride=1, padding=1
            ))

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


class EffectEncoder(nn.Module):
    """
    Encodes wet audio into effect-aware embedding.

    Architecture: Mel spectrogram -> CNN backbone -> projection -> embedding
    Supports multi-task learning with:
    - Effect type classification
    - Parameter regression
    - Contrastive embedding
    """

    def __init__(self, config: Optional[EffectEncoderConfig] = None):
        super().__init__()
        self.config = config or EffectEncoderConfig()

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Backbone
        self.backbone = self._create_backbone()

        # Projection head
        backbone_dim = self._get_backbone_dim()
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, self.config.embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim),
        )

        # Multi-task heads
        self.effect_classifier = nn.Linear(
            self.config.embedding_dim,
            self.config.num_effect_types
        )

        # Parameter regressors (per effect type)
        self.param_regressors = nn.ModuleDict({
            'eq': nn.Sequential(
                nn.Linear(self.config.embedding_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 15),  # 15 EQ parameters
            ),
            'compressor': nn.Sequential(
                nn.Linear(self.config.embedding_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 6),  # 6 compressor parameters
            ),
            'reverb': nn.Sequential(
                nn.Linear(self.config.embedding_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 4),  # 4 reverb parameters
            ),
            'distortion': nn.Sequential(
                nn.Linear(self.config.embedding_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 4),  # 4 distortion parameters
            ),
            'chorus': nn.Sequential(
                nn.Linear(self.config.embedding_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 4),  # 4 chorus parameters
            ),
            'delay': nn.Sequential(
                nn.Linear(self.config.embedding_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 3),  # 3 delay parameters
            ),
        })

        # Contrastive projection head
        self.contrastive_proj = nn.Sequential(
            nn.Linear(self.config.embedding_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
        )

    def _create_backbone(self) -> nn.Module:
        """Create backbone network."""
        if self.config.backbone == 'efficientnet':
            try:
                import timm
                backbone = timm.create_model(
                    'efficientnet_b0',
                    pretrained=self.config.pretrained,
                    in_chans=1,
                    num_classes=0,
                )
                if self.config.freeze_backbone:
                    for param in backbone.parameters():
                        param.requires_grad = False
                return backbone
            except ImportError:
                print("timm not available, using custom backbone")
                return AudioEncoder(in_channels=1, output_dim=1280)

        elif self.config.backbone == 'resnet':
            try:
                import timm
                backbone = timm.create_model(
                    'resnet34',
                    pretrained=self.config.pretrained,
                    in_chans=1,
                    num_classes=0,
                )
                return backbone
            except ImportError:
                return AudioEncoder(in_channels=1, output_dim=512)

        else:
            return AudioEncoder(in_channels=1, output_dim=1280)

    def _get_backbone_dim(self) -> int:
        """Get output dimension of backbone."""
        if self.config.backbone == 'efficientnet':
            return 1280
        elif self.config.backbone == 'resnet':
            return 512
        else:
            return 1280

    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract mel spectrogram features from audio.

        Args:
            audio: Input audio [B, 1, T] or [B, T]

        Returns:
            Mel spectrogram [B, 1, n_mels, time]
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Compute mel spectrogram
        mel = self.mel_transform(audio.squeeze(1))  # [B, n_mels, time]

        # Clamp to avoid log(0) in amplitude_to_db
        mel = torch.clamp(mel, min=1e-10)
        mel_db = self.amplitude_to_db(mel)

        # Clamp extreme values
        mel_db = torch.clamp(mel_db, min=-100, max=100)

        # Normalize per-sample to handle varying levels
        mel_mean = mel_db.mean(dim=(-2, -1), keepdim=True)
        mel_std = mel_db.std(dim=(-2, -1), keepdim=True)
        mel_db = (mel_db - mel_mean) / (mel_std + 1e-6)

        return mel_db.unsqueeze(1)  # [B, 1, n_mels, time]

    def forward(
        self,
        audio: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio: Input audio [B, 1, T] or [B, T]
            return_all: If True, return all outputs

        Returns:
            Normalized embedding [B, embedding_dim]
        """
        # Extract features
        features = self.extract_features(audio)

        # Check for NaN in features
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)

        # Backbone
        backbone_features = self.backbone(features)

        # Check for NaN in backbone output
        if torch.isnan(backbone_features).any():
            backbone_features = torch.nan_to_num(backbone_features, nan=0.0)

        # Projection
        embedding = self.projection(backbone_features)

        # Check for NaN before normalize
        if torch.isnan(embedding).any():
            embedding = torch.nan_to_num(embedding, nan=0.0)

        # Normalize embedding with epsilon to avoid division by zero
        embedding = F.normalize(embedding, dim=-1, eps=1e-6)

        if return_all:
            return {
                'embedding': embedding,
                'backbone_features': backbone_features,
                'mel_features': features,
            }

        return embedding

    def classify_effects(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Classify which effects are present.

        Args:
            embedding: Effect embedding [B, embedding_dim]

        Returns:
            Effect logits [B, num_effect_types]
        """
        return self.effect_classifier(embedding)

    def estimate_params(
        self,
        embedding: torch.Tensor,
        effect_type: str,
    ) -> torch.Tensor:
        """
        Estimate parameters for a specific effect type.

        Args:
            embedding: Effect embedding [B, embedding_dim]
            effect_type: Type of effect

        Returns:
            Estimated parameters (normalized to [0, 1])
        """
        if effect_type not in self.param_regressors:
            raise ValueError(f"Unknown effect type: {effect_type}")

        params = self.param_regressors[effect_type](embedding)
        return torch.sigmoid(params)  # Normalize to [0, 1]

    def get_contrastive_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Get contrastive embedding for similarity learning.

        Args:
            embedding: Effect embedding [B, embedding_dim]

        Returns:
            Contrastive embedding [B, 128]
        """
        return F.normalize(self.contrastive_proj(embedding), dim=-1, eps=1e-6)


class EffectEncoderWithMultiScale(EffectEncoder):
    """
    Effect encoder with multi-scale feature extraction.
    Uses multiple mel spectrogram resolutions.
    """

    def __init__(self, config: Optional[EffectEncoderConfig] = None):
        super().__init__(config)

        # Additional mel transforms at different resolutions
        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_fft=n_fft,
                hop_length=n_fft // 4,
                n_mels=self.config.n_mels,
            )
            for n_fft in [1024, 2048, 4096]
        ])

        # Multi-scale fusion
        self.scale_fusion = nn.Conv2d(3, 1, kernel_size=1)

    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        audio_1d = audio.squeeze(1)

        # Multi-scale mel spectrograms
        mels = []
        target_time = None

        for mel_transform in self.mel_transforms:
            mel = mel_transform(audio_1d)
            mel_db = self.amplitude_to_db(mel)

            if target_time is None:
                target_time = mel_db.size(-1)

            # Resize to match
            if mel_db.size(-1) != target_time:
                mel_db = F.interpolate(
                    mel_db.unsqueeze(1),
                    size=(self.config.n_mels, target_time),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

            mels.append(mel_db)

        # Stack and fuse
        multi_scale = torch.stack(mels, dim=1)  # [B, 3, n_mels, time]
        fused = self.scale_fusion(multi_scale)  # [B, 1, n_mels, time]

        # Normalize
        fused = (fused - fused.mean()) / (fused.std() + 1e-8)

        return fused
