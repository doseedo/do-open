"""
Semantic Decoder - MIDI Reconstruction from Musical DNA
========================================================

Decodes 120D Musical DNA parameters back to MIDI-compatible features.

Architecture:
    Musical DNA (120D) → Hidden Layers → Reconstructed Features → MIDI

This decoder enables the full parameter-guided editing workflow:
1. MIDI → Encoder → DNA (120 params)
2. Edit DNA parameters
3. DNA → Decoder → Reconstructed Features
4. Features → MIDI Synthesis

Author: Architecture Fix - Nov 22, 2025
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available")


@dataclass
class DecoderConfig:
    """Configuration for semantic decoder"""
    # Architecture
    input_dim: int = 120  # Musical DNA parameters
    hidden_dim: int = 1024  # Hidden layer size
    output_dim: int = 1150  # Full feature reconstruction (DeepFeatureExtractor)

    # Training
    dropout_rate: float = 0.1
    use_batch_norm: bool = True

    # Reconstruction target
    reconstruction_type: str = "features"  # 'features' or 'midi'


class SemanticDecoder(nn.Module):
    """
    Decoder that reconstructs musical features from 120D DNA parameters.

    Architecture:
        DNA (120D) → FC(1024) → ReLU → Dropout → FC(1024) → ReLU → FC(1150D)

    This allows:
    - Parameter editing → feature reconstruction → MIDI generation
    - Autoencoder training (encoder + decoder)
    - Interpolation in DNA space
    """

    def __init__(self, config: DecoderConfig):
        """
        Initialize decoder.

        Args:
            config: Decoder configuration
        """
        super().__init__()
        self.config = config

        # Build decoder network
        layers = []

        # Input layer: 120D → 1024D
        layers.append(nn.Linear(config.input_dim, config.hidden_dim))
        if config.use_batch_norm:
            layers.append(nn.BatchNorm1d(config.hidden_dim))
        layers.append(nn.ReLU())
        if config.dropout_rate > 0:
            layers.append(nn.Dropout(config.dropout_rate))

        # Hidden layer: 1024D → 1024D
        layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
        if config.use_batch_norm:
            layers.append(nn.BatchNorm1d(config.hidden_dim))
        layers.append(nn.ReLU())
        if config.dropout_rate > 0:
            layers.append(nn.Dropout(config.dropout_rate))

        # Output layer: 1024D → 1150D
        layers.append(nn.Linear(config.hidden_dim, config.output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, dna_params: torch.Tensor) -> torch.Tensor:
        """
        Decode DNA parameters to reconstructed features.

        Args:
            dna_params: Musical DNA parameters [batch_size, 120]

        Returns:
            Reconstructed features [batch_size, 1150]
        """
        return self.decoder(dna_params)

    def reconstruct(
        self,
        dna_params: torch.Tensor,
        as_numpy: bool = False
    ) -> torch.Tensor:
        """
        Reconstruct features from DNA parameters.

        Args:
            dna_params: DNA parameters [batch_size, 120] or [120]
            as_numpy: Return as numpy array instead of tensor

        Returns:
            Reconstructed features [batch_size, 1150] or [1150]
        """
        # Handle single sample
        if dna_params.dim() == 1:
            dna_params = dna_params.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Reconstruct
        with torch.no_grad():
            self.eval()
            reconstructed = self.forward(dna_params)

        # Squeeze if single sample
        if squeeze_output:
            reconstructed = reconstructed.squeeze(0)

        # Convert to numpy if requested
        if as_numpy:
            return reconstructed.cpu().numpy()

        return reconstructed

    def save(self, path: Path):
        """Save decoder state"""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = 'cpu') -> 'SemanticDecoder':
        """Load decoder from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        decoder = cls(checkpoint['config'])
        decoder.load_state_dict(checkpoint['state_dict'])
        decoder.to(device)
        return decoder


# ============================================================================
# Autoencoder (Encoder + Decoder Together)
# ============================================================================

class SemanticAutoencoder(nn.Module):
    """
    Full autoencoder: MIDI features → DNA params → reconstructed features

    This combines:
    - Modular encoders (6 domain encoders → 120D DNA)
    - Decoder (120D DNA → 1150D features)

    Training objective: Minimize reconstruction error
    """

    def __init__(self, encoders: Dict, decoder: SemanticDecoder):
        """
        Initialize autoencoder.

        Args:
            encoders: Dict of trained modular encoders
            decoder: Semantic decoder
        """
        super().__init__()
        self.encoders = nn.ModuleDict({k.value: v for k, v in encoders.items()})
        self.decoder = decoder

    def forward(self, features: torch.Tensor) -> tuple:
        """
        Full forward pass: features → DNA → reconstructed features

        Args:
            features: Input features [batch_size, 1150]

        Returns:
            (dna_params, reconstructed_features)
        """
        # Extract DNA from features (using encoders)
        # NOTE: This is a simplified version - actual implementation
        # would route dimension-specific features to each encoder
        dna_params_list = []
        for encoder_name, encoder in self.encoders.items():
            if encoder_name != 'cross_dimensional':
                # Extract semantic features from this encoder
                params = encoder.extract_semantic_features(features)
                dna_params_list.append(params)

        # Concatenate all domain parameters
        dna_params = torch.cat(dna_params_list, dim=-1)  # [batch, 110]

        # Add cross-dimensional encoding
        if 'cross_dimensional' in self.encoders:
            cross_encoder = self.encoders['cross_dimensional']
            cross_params = cross_encoder.extract_semantic_features(dna_params)
            dna_params = torch.cat([dna_params, cross_params], dim=-1)  # [batch, 120]

        # Decode DNA to reconstructed features
        reconstructed = self.decoder(dna_params)

        return dna_params, reconstructed

    def reconstruction_loss(
        self,
        original_features: torch.Tensor,
        reconstructed_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss (MSE).

        Args:
            original_features: Original input [batch, 1150]
            reconstructed_features: Reconstructed output [batch, 1150]

        Returns:
            Scalar loss value
        """
        return F.mse_loss(reconstructed_features, original_features)


# ============================================================================
# Utility Functions
# ============================================================================

def create_decoder(
    dna_dim: int = 120,
    feature_dim: int = 1150,
    hidden_dim: int = 1024,
    device: str = 'cpu'
) -> SemanticDecoder:
    """
    Factory function to create decoder.

    Args:
        dna_dim: DNA parameter dimension (default: 120)
        feature_dim: Output feature dimension (default: 1150)
        hidden_dim: Hidden layer size (default: 1024)
        device: Device to create on

    Returns:
        Initialized decoder
    """
    config = DecoderConfig(
        input_dim=dna_dim,
        hidden_dim=hidden_dim,
        output_dim=feature_dim
    )

    decoder = SemanticDecoder(config)
    decoder.to(device)

    return decoder


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Semantic Decoder for MIDI Reconstruction")
    print("="*70)

    if TORCH_AVAILABLE:
        # Create decoder
        print("\n1. Creating decoder...")
        decoder = create_decoder(device='cpu')
        print(f"   ✅ Decoder created: 120D → 1024D → 1150D")

        # Test forward pass
        print("\n2. Testing reconstruction...")
        batch_size = 4
        test_dna = torch.randn(batch_size, 120)
        reconstructed = decoder(test_dna)
        print(f"   Input DNA shape: {test_dna.shape}")
        print(f"   Output features shape: {reconstructed.shape}")
        print(f"   ✅ Reconstruction successful")

        # Test save/load
        print("\n3. Testing save/load...")
        test_path = Path("/tmp/test_decoder.pt")
        decoder.save(test_path)
        loaded_decoder = SemanticDecoder.load(test_path, device='cpu')
        print(f"   ✅ Save/load successful")

        print("\n" + "="*70)
        print("✅ All decoder tests passed!")
        print("="*70)
    else:
        print("\n❌ PyTorch not available")
