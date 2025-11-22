"""
Global Encoder - Agent 3
========================

Encodes global musical context (60D output) from full 1150D features.

This encoder learns high-level musical context that conditions lower-level encoders:
- Key context (12D): Tonal center, modulations, key stability
- Tempo feel (8D): Tempo, variations, metric feel
- Genre style (20D): Jazz/classical/latin style markers
- Form structure (20D): Overall form, sections, proportions

Input: 1150D (all features from DeepFeatureExtractor)
Output: 60D hierarchical global parameters

Architecture uses attention mechanism to focus on relevant features
for each global component.

Author: Agent 3 - DNA Expansion & Hierarchical Architecture
Date: 2025-11-22
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
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


from midi_generator.learning.semantic_encoder import EncoderConfig


@dataclass
class GlobalEncoderConfig(EncoderConfig):
    """Configuration for GlobalEncoder"""
    input_dim: int = 1150  # All features
    num_semantic_features: int = 60  # Total global output
    hidden_dim: int = 2048  # Large hidden for attention
    attention_heads: int = 8  # Multi-head attention

    # Component dimensions
    key_context_dim: int = 12
    tempo_feel_dim: int = 8
    genre_style_dim: int = 20
    form_structure_dim: int = 20


if TORCH_AVAILABLE:

    class MultiHeadAttention(nn.Module):
        """Multi-head attention for feature selection"""

        def __init__(self, embed_dim: int, num_heads: int):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: [batch, features]
            Returns:
                attended: [batch, features]
            """
            # Add sequence dimension
            x = x.unsqueeze(1)  # [batch, 1, features]

            # Self-attention
            attended, _ = self.attention(x, x, x)

            # Remove sequence dimension
            return attended.squeeze(1)


    class GlobalEncoder(nn.Module):
        """
        Global encoder for musical context (60D output).

        Architecture:
            1150D → Attention → Split into 4 branches:
            - Key context branch → 12D
            - Tempo feel branch → 8D
            - Genre style branch → 20D
            - Form structure branch → 20D

        Total output: 60D
        """

        def __init__(self, config: Optional[GlobalEncoderConfig] = None):
            super().__init__()

            if config is None:
                config = GlobalEncoderConfig()
            self.config = config

            # Input projection
            self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
            self.input_bn = nn.BatchNorm1d(config.hidden_dim)

            # Multi-head attention for feature selection
            self.attention = MultiHeadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.attention_heads
            )

            # Shared encoder
            self.shared_encoder = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )

            # Component-specific heads
            self.key_context_head = nn.Sequential(
                nn.Linear(config.hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(256, config.key_context_dim),
                nn.Tanh()
            )

            self.tempo_feel_head = nn.Sequential(
                nn.Linear(config.hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(128, config.tempo_feel_dim),
                nn.Tanh()
            )

            self.genre_style_head = nn.Sequential(
                nn.Linear(config.hidden_dim, 512),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(512, config.genre_style_dim),
                nn.Tanh()
            )

            self.form_structure_head = nn.Sequential(
                nn.Linear(config.hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(256, config.form_structure_dim),
                nn.Tanh()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Input features [batch, 1150]

            Returns:
                Global parameters [batch, 60]
            """
            # Input projection
            h = self.input_proj(x)
            h = self.input_bn(h)
            h = F.relu(h)

            # Attention
            h = self.attention(h)

            # Shared encoding
            h = self.shared_encoder(h)

            # Component heads
            key_context = self.key_context_head(h)      # [batch, 12]
            tempo_feel = self.tempo_feel_head(h)        # [batch, 8]
            genre_style = self.genre_style_head(h)      # [batch, 20]
            form_structure = self.form_structure_head(h)  # [batch, 20]

            # Concatenate all components
            global_params = torch.cat([
                key_context,
                tempo_feel,
                genre_style,
                form_structure
            ], dim=1)  # [batch, 60]

            return global_params

        def extract_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Extract individual components.

            Returns:
                Dictionary with keys: key_context, tempo_feel, genre_style, form_structure
            """
            # Input projection
            h = self.input_proj(x)
            h = self.input_bn(h)
            h = F.relu(h)

            # Attention
            h = self.attention(h)

            # Shared encoding
            h = self.shared_encoder(h)

            # Component heads
            return {
                'key_context': self.key_context_head(h),
                'tempo_feel': self.tempo_feel_head(h),
                'genre_style': self.genre_style_head(h),
                'form_structure': self.form_structure_head(h),
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
        def load(cls, path: Path, device: str = 'cpu') -> 'GlobalEncoder':
            """Load encoder"""
            checkpoint = torch.load(path, map_location=device)
            config = GlobalEncoderConfig.from_dict(checkpoint['config'])
            encoder = cls(config)
            encoder.load_state_dict(checkpoint['model_state_dict'])
            encoder.to(device)
            return encoder


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Global Encoder - Test")
    print("="*70)

    if TORCH_AVAILABLE:
        # Create encoder
        config = GlobalEncoderConfig()
        encoder = GlobalEncoder(config)

        print(f"\nConfig:")
        print(f"  Input dim: {config.input_dim}")
        print(f"  Output dim: {config.num_semantic_features}")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Attention heads: {config.attention_heads}")

        print(f"\nComponent dimensions:")
        print(f"  Key context: {config.key_context_dim}D")
        print(f"  Tempo feel: {config.tempo_feel_dim}D")
        print(f"  Genre style: {config.genre_style_dim}D")
        print(f"  Form structure: {config.form_structure_dim}D")
        print(f"  Total: {sum([config.key_context_dim, config.tempo_feel_dim, config.genre_style_dim, config.form_structure_dim])}D")

        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 1150)

        global_params = encoder(x)
        print(f"\nForward pass:")
        print(f"  Input: {x.shape}")
        print(f"  Output: {global_params.shape}")

        # Test component extraction
        components = encoder.extract_components(x)
        print(f"\nComponent extraction:")
        for name, tensor in components.items():
            print(f"  {name}: {tensor.shape}")

        # Test save/load
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "global_encoder.pt"
            encoder.save(save_path)
            print(f"\nSaved to: {save_path}")

            loaded = GlobalEncoder.load(save_path)
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
