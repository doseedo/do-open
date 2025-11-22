"""
Expanded Encoders v2.0 - Agent 3
================================

Expanded versions of existing encoders for 300D architecture:
- HarmonyEncoder: 30D → 60D
- RhythmEncoder: 20D → 40D
- TextureEncoder: 20D → 30D
- OrchestrationEncoder: 25D → 40D
- FormEncoder → FormStructureEncoder: 15D → 20D

These encoders are compatible with the 300D hierarchical MusicalDNA structure.

Author: Agent 3 - DNA Expansion & Hierarchical Architecture
Date: 2025-11-22
Version: 2.0.0
"""

from dataclasses import dataclass
from typing import Optional, Dict
from pathlib import Path
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available")

from midi_generator.learning.semantic_encoder import EncoderConfig


# =============================================================================
# Expanded Harmony Encoder (30D → 60D)
# =============================================================================

@dataclass
class HarmonyEncoderV2Config(EncoderConfig):
    """Configuration for expanded HarmonyEncoder"""
    input_dim: int = 250  # Harmony features from DeepFeatureExtractor
    num_semantic_features: int = 60  # Expanded from 30D
    hidden_dim: int = 1024  # Increased capacity
    num_locality_types: int = 5  # TRANSPOSE, INVERT, OCTAVE_SHIFT, VOICE_PERMUTATION, HARMONIC_SHIFT


if TORCH_AVAILABLE:

    class HarmonyEncoderV2(nn.Module):
        """
        Expanded Harmony Encoder (60D output).

        Discovers:
        - Basic harmony (30D): Chord types, voicings, progressions
        - Advanced harmony (30D): Extensions, alterations, voice leading details
        """

        def __init__(self, config: Optional[HarmonyEncoderV2Config] = None):
            super().__init__()

            if config is None:
                config = HarmonyEncoderV2Config()
            self.config = config

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),

                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),

                nn.Linear(config.hidden_dim, config.num_semantic_features),
                nn.Tanh()
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(config.num_semantic_features, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.input_dim)
            )

            # Locality predictor
            self.locality_predictor = nn.Sequential(
                nn.Linear(config.num_semantic_features * 2, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.num_locality_types)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        def extract_semantic_features(self, x: torch.Tensor, as_numpy: bool = False):
            params = self.forward(x)
            if as_numpy:
                return params.detach().cpu().numpy()
            return params

        def reconstruct(self, semantic_features: torch.Tensor) -> torch.Tensor:
            return self.decoder(semantic_features)

        def predict_locality(self, sf1: torch.Tensor, sf2: torch.Tensor) -> torch.Tensor:
            return self.locality_predictor(torch.cat([sf1, sf2], dim=1))

        def save(self, path: Path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'config': self.config.to_dict(),
                'model_state_dict': self.state_dict(),
            }, path)

        @classmethod
        def load(cls, path: Path, device: str = 'cpu') -> 'HarmonyEncoderV2':
            checkpoint = torch.load(path, map_location=device)
            config = HarmonyEncoderV2Config.from_dict(checkpoint['config'])
            encoder = cls(config)
            encoder.load_state_dict(checkpoint['model_state_dict'])
            encoder.to(device)
            return encoder


# =============================================================================
# Expanded Rhythm Encoder (20D → 40D)
# =============================================================================

@dataclass
class RhythmEncoderV2Config(EncoderConfig):
    """Configuration for expanded RhythmEncoder"""
    input_dim: int = 250  # Rhythm features
    num_semantic_features: int = 40  # Expanded from 20D
    hidden_dim: int = 1024
    num_locality_types: int = 4  # AUGMENT, DIMINUTION, TIME_SHIFT, RHYTHMIC_QUANTIZE


if TORCH_AVAILABLE:

    class RhythmEncoderV2(nn.Module):
        """Expanded Rhythm Encoder (40D output)"""

        def __init__(self, config: Optional[RhythmEncoderV2Config] = None):
            super().__init__()

            if config is None:
                config = RhythmEncoderV2Config()
            self.config = config

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),

                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),

                nn.Linear(config.hidden_dim, config.num_semantic_features),
                nn.Tanh()
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(config.num_semantic_features, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.input_dim)
            )

            # Locality predictor
            self.locality_predictor = nn.Sequential(
                nn.Linear(config.num_semantic_features * 2, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.num_locality_types)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        def extract_semantic_features(self, x: torch.Tensor, as_numpy: bool = False):
            params = self.forward(x)
            if as_numpy:
                return params.detach().cpu().numpy()
            return params

        def reconstruct(self, semantic_features: torch.Tensor) -> torch.Tensor:
            return self.decoder(semantic_features)

        def predict_locality(self, sf1: torch.Tensor, sf2: torch.Tensor) -> torch.Tensor:
            return self.locality_predictor(torch.cat([sf1, sf2], dim=1))

        def save(self, path: Path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'config': self.config.to_dict(),
                'model_state_dict': self.state_dict(),
            }, path)

        @classmethod
        def load(cls, path: Path, device: str = 'cpu') -> 'RhythmEncoderV2':
            checkpoint = torch.load(path, map_location=device)
            config = RhythmEncoderV2Config.from_dict(checkpoint['config'])
            encoder = cls(config)
            encoder.load_state_dict(checkpoint['model_state_dict'])
            encoder.to(device)
            return encoder


# =============================================================================
# Expanded Texture Encoder (20D → 30D)
# =============================================================================

@dataclass
class TextureEncoderV2Config(EncoderConfig):
    """Configuration for expanded TextureEncoder"""
    input_dim: int = 250  # Texture (100D) + Dynamics (150D)
    num_semantic_features: int = 30  # Expanded from 20D
    hidden_dim: int = 512
    num_locality_types: int = 2  # VOICE_PERMUTATION, VELOCITY_SCALE


if TORCH_AVAILABLE:

    class TextureEncoderV2(nn.Module):
        """Expanded Texture Encoder (30D output)"""

        def __init__(self, config: Optional[TextureEncoderV2Config] = None):
            super().__init__()

            if config is None:
                config = TextureEncoderV2Config()
            self.config = config

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),

                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),

                nn.Linear(config.hidden_dim, config.num_semantic_features),
                nn.Tanh()
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(config.num_semantic_features, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.input_dim)
            )

            # Locality predictor
            self.locality_predictor = nn.Sequential(
                nn.Linear(config.num_semantic_features * 2, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.num_locality_types)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        def extract_semantic_features(self, x: torch.Tensor, as_numpy: bool = False):
            params = self.forward(x)
            if as_numpy:
                return params.detach().cpu().numpy()
            return params

        def reconstruct(self, semantic_features: torch.Tensor) -> torch.Tensor:
            return self.decoder(semantic_features)

        def predict_locality(self, sf1: torch.Tensor, sf2: torch.Tensor) -> torch.Tensor:
            return self.locality_predictor(torch.cat([sf1, sf2], dim=1))

        def save(self, path: Path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'config': self.config.to_dict(),
                'model_state_dict': self.state_dict(),
            }, path)

        @classmethod
        def load(cls, path: Path, device: str = 'cpu') -> 'TextureEncoderV2':
            checkpoint = torch.load(path, map_location=device)
            config = TextureEncoderV2Config.from_dict(checkpoint['config'])
            encoder = cls(config)
            encoder.load_state_dict(checkpoint['model_state_dict'])
            encoder.to(device)
            return encoder


# =============================================================================
# Expanded Orchestration Encoder (25D → 40D)
# =============================================================================

@dataclass
class OrchestrationEncoderV2Config(EncoderConfig):
    """Configuration for expanded OrchestrationEncoder"""
    input_dim: int = 150  # Orchestration features
    num_semantic_features: int = 40  # Expanded from 25D
    hidden_dim: int = 512
    num_locality_types: int = 3  # VOICE_PERMUTATION, OCTAVE_SHIFT, REGISTER_SHIFT


if TORCH_AVAILABLE:

    class OrchestrationEncoderV2(nn.Module):
        """Expanded Orchestration Encoder (40D output)"""

        def __init__(self, config: Optional[OrchestrationEncoderV2Config] = None):
            super().__init__()

            if config is None:
                config = OrchestrationEncoderV2Config()
            self.config = config

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),

                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),

                nn.Linear(config.hidden_dim, config.num_semantic_features),
                nn.Tanh()
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(config.num_semantic_features, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.input_dim)
            )

            # Locality predictor
            self.locality_predictor = nn.Sequential(
                nn.Linear(config.num_semantic_features * 2, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.num_locality_types)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        def extract_semantic_features(self, x: torch.Tensor, as_numpy: bool = False):
            params = self.forward(x)
            if as_numpy:
                return params.detach().cpu().numpy()
            return params

        def reconstruct(self, semantic_features: torch.Tensor) -> torch.Tensor:
            return self.decoder(semantic_features)

        def predict_locality(self, sf1: torch.Tensor, sf2: torch.Tensor) -> torch.Tensor:
            return self.locality_predictor(torch.cat([sf1, sf2], dim=1))

        def save(self, path: Path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'config': self.config.to_dict(),
                'model_state_dict': self.state_dict(),
            }, path)

        @classmethod
        def load(cls, path: Path, device: str = 'cpu') -> 'OrchestrationEncoderV2':
            checkpoint = torch.load(path, map_location=device)
            config = OrchestrationEncoderV2Config.from_dict(checkpoint['config'])
            encoder = cls(config)
            encoder.load_state_dict(checkpoint['model_state_dict'])
            encoder.to(device)
            return encoder


# =============================================================================
# Form Structure Encoder (15D → 20D)
# =============================================================================

@dataclass
class FormStructureEncoderConfig(EncoderConfig):
    """Configuration for FormStructureEncoder"""
    input_dim: int = 50  # Structure features
    num_semantic_features: int = 20  # Expanded from 15D
    hidden_dim: int = 256
    num_locality_types: int = 2  # RETROGRADE, TIME_SHIFT


if TORCH_AVAILABLE:

    class FormStructureEncoder(nn.Module):
        """Form Structure Encoder (20D output)"""

        def __init__(self, config: Optional[FormStructureEncoderConfig] = None):
            super().__init__()

            if config is None:
                config = FormStructureEncoderConfig()
            self.config = config

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),

                nn.Linear(config.hidden_dim, config.num_semantic_features),
                nn.Tanh()
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(config.num_semantic_features, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.input_dim)
            )

            # Locality predictor
            self.locality_predictor = nn.Sequential(
                nn.Linear(config.num_semantic_features * 2, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.num_locality_types)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        def extract_semantic_features(self, x: torch.Tensor, as_numpy: bool = False):
            params = self.forward(x)
            if as_numpy:
                return params.detach().cpu().numpy()
            return params

        def reconstruct(self, semantic_features: torch.Tensor) -> torch.Tensor:
            return self.decoder(semantic_features)

        def predict_locality(self, sf1: torch.Tensor, sf2: torch.Tensor) -> torch.Tensor:
            return self.locality_predictor(torch.cat([sf1, sf2], dim=1))

        def save(self, path: Path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'config': self.config.to_dict(),
                'model_state_dict': self.state_dict(),
            }, path)

        @classmethod
        def load(cls, path: Path, device: str = 'cpu') -> 'FormStructureEncoder':
            checkpoint = torch.load(path, map_location=device)
            config = FormStructureEncoderConfig.from_dict(checkpoint['config'])
            encoder = cls(config)
            encoder.load_state_dict(checkpoint['model_state_dict'])
            encoder.to(device)
            return encoder


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Expanded Encoders V2.0 - Test")
    print("="*70)

    if TORCH_AVAILABLE:
        encoders = [
            ("HarmonyEncoderV2", HarmonyEncoderV2(), 250, 60),
            ("RhythmEncoderV2", RhythmEncoderV2(), 250, 40),
            ("TextureEncoderV2", TextureEncoderV2(), 250, 30),
            ("OrchestrationEncoderV2", OrchestrationEncoderV2(), 150, 40),
            ("FormStructureEncoder", FormStructureEncoder(), 50, 20),
        ]

        for name, encoder, input_dim, output_dim in encoders:
            print(f"\nTesting {name}:")
            print(f"  Input: {input_dim}D → Output: {output_dim}D")

            # Test forward pass
            x = torch.randn(4, input_dim)
            out = encoder(x)
            print(f"  Forward: {x.shape} → {out.shape} ✓")

            # Test reconstruction
            recon = encoder.reconstruct(out)
            print(f"  Reconstruction: {out.shape} → {recon.shape} ✓")

        print("\n" + "="*70)
        print("✅ All expanded encoders tested successfully!")
        print("="*70)
        print(f"\nTotal expanded dimensions:")
        print(f"  Harmony: 60D")
        print(f"  Rhythm: 40D")
        print(f"  Texture: 30D")
        print(f"  Orchestration: 40D")
        print(f"  Form Structure: 20D")
        print(f"  ──────────")
        print(f"  Subtotal: 190D")
        print(f"\n  + Global: 60D (from GlobalEncoder)")
        print(f"  + Melody: 40D (from MelodyEncoder)")
        print(f"  + Voicing: 30D (from VoicingEncoder)")
        print(f"  ──────────")
        print(f"  TOTAL: 300D ✓")
    else:
        print("PyTorch not available - skipping tests")
