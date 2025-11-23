"""
Voicing Encoder - Agent 3
=========================

Encodes voicing details (30D output) from harmony + dynamics features.

This NEW encoder discovers voicing-specific parameters:
- Spacing (10D): Interval distribution, clusters, spread voicings
- Doubling (10D): Unison, octave doubling, intensity
- Register (10D): Tessitura, range distribution, register shifts

Input: 250D harmony + 150D dynamics = 400D total
Output: 30D voicing parameters

Locality Functions:
- OCTAVE_SHIFT: Octave transposition
- VOICE_PERMUTATION: Voice ordering changes
- REGISTER_SHIFT: Tessitura changes

Author: Agent 3 - DNA Expansion & Hierarchical Architecture
Date: 2025-11-22
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Dict, Optional
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

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


from midi_generator.learning.semantic_encoder import SemanticFeatureEncoder, EncoderConfig


@dataclass
class VoicingEncoderConfig(EncoderConfig):
    """Configuration for VoicingEncoder"""
    input_dim: int = 400  # Harmony (250D) + Dynamics (150D)
    num_semantic_features: int = 30  # Total voicing output
    hidden_dim: int = 512
    num_locality_types: int = 3  # OCTAVE_SHIFT, VOICE_PERMUTATION, REGISTER_SHIFT

    # Component dimensions
    spacing_dim: int = 10
    doubling_dim: int = 10
    register_dim: int = 10


if TORCH_AVAILABLE:

    class VoicingEncoder(nn.Module):
        """
        Voicing encoder for voicing details (30D output).

        Architecture:
            400D (harmony + dynamics) → Shared Encoder → Split into 3 branches:
            - Spacing branch → 10D
            - Doubling branch → 10D
            - Register branch → 10D

        Total output: 30D
        """

        def __init__(self, config: Optional[VoicingEncoderConfig] = None):
            super().__init__()

            if config is None:
                config = VoicingEncoderConfig()
            self.config = config

            # Shared encoder
            self.shared_encoder = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),

                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )

            # Spacing-specific head
            self.spacing_head = nn.Sequential(
                nn.Linear(config.hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(128, config.spacing_dim),
                nn.Tanh()
            )

            # Doubling-specific head
            self.doubling_head = nn.Sequential(
                nn.Linear(config.hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(128, config.doubling_dim),
                nn.Tanh()
            )

            # Register-specific head
            self.register_head = nn.Sequential(
                nn.Linear(config.hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(128, config.register_dim),
                nn.Tanh()
            )

            # Decoder for reconstruction
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
            """
            Args:
                x: Harmony + Dynamics features [batch, 400]

            Returns:
                Voicing parameters [batch, 30]
            """
            # Shared encoding
            h = self.shared_encoder(x)

            # Component heads
            spacing = self.spacing_head(h)      # [batch, 10]
            doubling = self.doubling_head(h)    # [batch, 10]
            register = self.register_head(h)    # [batch, 10]

            # Concatenate all components
            voicing_params = torch.cat([
                spacing,
                doubling,
                register
            ], dim=1)  # [batch, 30]

            return voicing_params

        def extract_semantic_features(self, x: torch.Tensor, as_numpy: bool = False):
            """Extract semantic features (for compatibility with training pipeline)"""
            params = self.forward(x)
            if as_numpy:
                return params.detach().cpu().numpy()
            return params

        def reconstruct(self, semantic_features: torch.Tensor) -> torch.Tensor:
            """Reconstruct input from semantic features"""
            return self.decoder(semantic_features)

        def predict_locality(
            self,
            semantic_features_1: torch.Tensor,
            semantic_features_2: torch.Tensor
        ) -> torch.Tensor:
            """Predict locality transformation type"""
            concatenated = torch.cat([semantic_features_1, semantic_features_2], dim=1)
            return self.locality_predictor(concatenated)

        def extract_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Extract individual components.

            Returns:
                Dictionary with keys: spacing, doubling, register
            """
            h = self.shared_encoder(x)
            return {
                'spacing': self.spacing_head(h),
                'doubling': self.doubling_head(h),
                'register': self.register_head(h),
            }

        def save(self, path: Path):
            """Save encoder"""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'config': self.config.to_dict(),
                'model_state_dict': self.state_dict(),
            }, path)

        @classmethod
        def load(cls, path: Path, device: str = 'cpu') -> 'VoicingEncoder':
            """Load encoder"""
            checkpoint = torch.load(path, map_location=device)
            config = VoicingEncoderConfig.from_dict(checkpoint['config'])
            encoder = cls(config)
            encoder.load_state_dict(checkpoint['model_state_dict'])
            encoder.to(device)
            return encoder


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Voicing Encoder - Test")
    print("="*70)

    if TORCH_AVAILABLE:
        # Create encoder
        config = VoicingEncoderConfig()
        encoder = VoicingEncoder(config)

        print(f"\nConfig:")
        print(f"  Input dim: {config.input_dim} (250D harmony + 150D dynamics)")
        print(f"  Output dim: {config.num_semantic_features}")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Locality types: {config.num_locality_types}")

        print(f"\nComponent dimensions:")
        print(f"  Spacing: {config.spacing_dim}D")
        print(f"  Doubling: {config.doubling_dim}D")
        print(f"  Register: {config.register_dim}D")
        print(f"  Total: {sum([config.spacing_dim, config.doubling_dim, config.register_dim])}D")

        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 400)

        voicing_params = encoder(x)
        print(f"\nForward pass:")
        print(f"  Input: {x.shape}")
        print(f"  Output: {voicing_params.shape}")

        # Test reconstruction
        reconstructed = encoder.reconstruct(voicing_params)
        print(f"\nReconstruction:")
        print(f"  Semantic features: {voicing_params.shape}")
        print(f"  Reconstructed: {reconstructed.shape}")

        # Test locality prediction
        locality_pred = encoder.predict_locality(voicing_params, voicing_params)
        print(f"\nLocality prediction:")
        print(f"  Output: {locality_pred.shape}")
        print(f"  Expected: [{batch_size}, {config.num_locality_types}]")

        # Test component extraction
        components = encoder.extract_components(x)
        print(f"\nComponent extraction:")
        for name, tensor in components.items():
            print(f"  {name}: {tensor.shape}")

        # Test save/load
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "voicing_encoder.pt"
            encoder.save(save_path)
            print(f"\nSaved to: {save_path}")

            loaded = VoicingEncoder.load(save_path)
            print(f"Loaded successfully")

            # Verify output matches
            with torch.no_grad():
                out1 = encoder(x)
                out2 = loaded(x)
                diff = torch.max(torch.abs(out1 - out2))
                print(f"Max diff after load: {diff:.10f}")
                assert diff < 1e-6, "Load failed!"

        print("\n" + "="*70)
        print("✅ All tests passed!")
        print("="*70)
    else:
        print("PyTorch not available - skipping tests")
