"""
Melody Encoder - Agent 3
========================

Encodes melodic parameters (40D output) from 200D melody features.

This NEW encoder discovers melody-specific parameters:
- Contour (15D): Shape, range, intervals, direction
- Motifs (15D): Repetition, development, transformation
- Phrasing (10D): Phrase structure, breath marks, articulation

Input: 200D melody features from DeepFeatureExtractor
Output: 40D melodic parameters

Locality Functions:
- TRANSPOSE: Pitch transposition invariance
- INVERT: Interval inversion
- RETROGRADE: Time reversal
- AUGMENTATION: Rhythmic augmentation

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
class MelodyEncoderConfig(EncoderConfig):
    """Configuration for MelodyEncoder"""
    input_dim: int = 200  # Melody features
    num_semantic_features: int = 40  # Total melody output
    hidden_dim: int = 512
    num_locality_types: int = 4  # TRANSPOSE, INVERT, RETROGRADE, AUGMENT

    # Component dimensions
    contour_dim: int = 15
    motif_dim: int = 15
    phrasing_dim: int = 10


if TORCH_AVAILABLE:

    class MelodyEncoder(nn.Module):
        """
        Melody encoder for melodic parameters (40D output).

        Architecture:
            200D → Shared Encoder → Split into 3 branches:
            - Contour branch → 15D
            - Motif branch → 15D
            - Phrasing branch → 10D

        Total output: 40D
        """

        def __init__(self, config: Optional[MelodyEncoderConfig] = None):
            super().__init__()

            if config is None:
                config = MelodyEncoderConfig()
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

            # Contour-specific head
            self.contour_head = nn.Sequential(
                nn.Linear(config.hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(256, config.contour_dim),
                nn.Tanh()
            )

            # Motif-specific head
            self.motif_head = nn.Sequential(
                nn.Linear(config.hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(256, config.motif_dim),
                nn.Tanh()
            )

            # Phrasing-specific head
            self.phrasing_head = nn.Sequential(
                nn.Linear(config.hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(128, config.phrasing_dim),
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
                x: Melody features [batch, 200]

            Returns:
                Melody parameters [batch, 40]
            """
            # Shared encoding
            h = self.shared_encoder(x)

            # Component heads
            contour = self.contour_head(h)      # [batch, 15]
            motif = self.motif_head(h)          # [batch, 15]
            phrasing = self.phrasing_head(h)    # [batch, 10]

            # Concatenate all components
            melody_params = torch.cat([
                contour,
                motif,
                phrasing
            ], dim=1)  # [batch, 40]

            return melody_params

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
                Dictionary with keys: contour, motif, phrasing
            """
            h = self.shared_encoder(x)
            return {
                'contour': self.contour_head(h),
                'motif': self.motif_head(h),
                'phrasing': self.phrasing_head(h),
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
        def load(cls, path: Path, device: str = 'cpu') -> 'MelodyEncoder':
            """Load encoder"""
            checkpoint = torch.load(path, map_location=device)
            config = MelodyEncoderConfig.from_dict(checkpoint['config'])
            encoder = cls(config)
            encoder.load_state_dict(checkpoint['model_state_dict'])
            encoder.to(device)
            return encoder


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Melody Encoder - Test")
    print("="*70)

    if TORCH_AVAILABLE:
        # Create encoder
        config = MelodyEncoderConfig()
        encoder = MelodyEncoder(config)

        print(f"\nConfig:")
        print(f"  Input dim: {config.input_dim}")
        print(f"  Output dim: {config.num_semantic_features}")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Locality types: {config.num_locality_types}")

        print(f"\nComponent dimensions:")
        print(f"  Contour: {config.contour_dim}D")
        print(f"  Motif: {config.motif_dim}D")
        print(f"  Phrasing: {config.phrasing_dim}D")
        print(f"  Total: {sum([config.contour_dim, config.motif_dim, config.phrasing_dim])}D")

        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 200)

        melody_params = encoder(x)
        print(f"\nForward pass:")
        print(f"  Input: {x.shape}")
        print(f"  Output: {melody_params.shape}")

        # Test reconstruction
        reconstructed = encoder.reconstruct(melody_params)
        print(f"\nReconstruction:")
        print(f"  Semantic features: {melody_params.shape}")
        print(f"  Reconstructed: {reconstructed.shape}")

        # Test locality prediction
        locality_pred = encoder.predict_locality(melody_params, melody_params)
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
            save_path = Path(tmpdir) / "melody_encoder.pt"
            encoder.save(save_path)
            print(f"\nSaved to: {save_path}")

            loaded = MelodyEncoder.load(save_path)
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
