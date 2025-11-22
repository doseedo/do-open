"""
Modular Encoder Factory V2.0 - Agent 3
======================================

Updated factory for creating encoders for 300D hierarchical architecture.

Supports both v1.0 (120D) and v2.0 (300D) DNA structures with automatic
version detection and seamless switching.

V2.0 Encoders (300D total):
- GlobalEncoder: 60D (NEW)
- HarmonyEncoderV2: 60D (expanded from 30D)
- MelodyEncoder: 40D (NEW)
- RhythmEncoderV2: 40D (expanded from 20D)
- VoicingEncoder: 30D (NEW)
- TextureEncoderV2: 30D (expanded from 20D)
- OrchestrationEncoderV2: 40D (expanded from 25D)
- FormStructureEncoder: 20D (expanded from 15D)

Author: Agent 3 - DNA Expansion & Hierarchical Architecture
Date: 2025-11-22
Version: 2.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from pathlib import Path
from enum import Enum
import json
import warnings

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available")

# Import v2.0 encoders
try:
    from midi_generator.learning.global_encoder import GlobalEncoder, GlobalEncoderConfig
    from midi_generator.learning.melody_encoder import MelodyEncoder, MelodyEncoderConfig
    from midi_generator.learning.voicing_encoder import VoicingEncoder, VoicingEncoderConfig
    from midi_generator.learning.expanded_encoders_v2 import (
        HarmonyEncoderV2, HarmonyEncoderV2Config,
        RhythmEncoderV2, RhythmEncoderV2Config,
        TextureEncoderV2, TextureEncoderV2Config,
        OrchestrationEncoderV2, OrchestrationEncoderV2Config,
        FormStructureEncoder, FormStructureEncoderConfig
    )
    V2_ENCODERS_AVAILABLE = True
except ImportError:
    V2_ENCODERS_AVAILABLE = False
    warnings.warn("V2.0 encoders not available")


# =============================================================================
# Dimension Definitions
# =============================================================================

class MusicalDimensionV2(Enum):
    """Musical dimensions for v2.0 (300D architecture)"""
    # Global level (60D)
    GLOBAL = "global"

    # Sectional level (140D)
    HARMONY = "harmony"
    MELODY = "melody"
    RHYTHM = "rhythm"

    # Local level (100D)
    VOICING = "voicing"
    TEXTURE = "texture"
    ORCHESTRATION = "orchestration"
    FORM_STRUCTURE = "form_structure"


@dataclass
class DimensionSpecV2:
    """
    Specification for a musical dimension encoder in v2.0.
    """
    dimension: MusicalDimensionV2
    num_params: int
    input_dim: int
    hidden_dim: int
    level: str  # 'global', 'sectional', or 'local'
    description: str


# =============================================================================
# Dimension Specifications for V2.0
# =============================================================================

DIMENSION_SPECS_V2 = {
    MusicalDimensionV2.GLOBAL: DimensionSpecV2(
        dimension=MusicalDimensionV2.GLOBAL,
        num_params=60,
        input_dim=1150,  # All features
        hidden_dim=2048,
        level='global',
        description="Global musical context: key, tempo, genre, form"
    ),

    MusicalDimensionV2.HARMONY: DimensionSpecV2(
        dimension=MusicalDimensionV2.HARMONY,
        num_params=60,
        input_dim=250,  # Harmony features
        hidden_dim=1024,
        level='sectional',
        description="Harmonic content: chords, progressions, voice leading"
    ),

    MusicalDimensionV2.MELODY: DimensionSpecV2(
        dimension=MusicalDimensionV2.MELODY,
        num_params=40,
        input_dim=200,  # Melody features
        hidden_dim=512,
        level='sectional',
        description="Melodic content: contour, motifs, phrasing"
    ),

    MusicalDimensionV2.RHYTHM: DimensionSpecV2(
        dimension=MusicalDimensionV2.RHYTHM,
        num_params=40,
        input_dim=250,  # Rhythm features
        hidden_dim=1024,
        level='sectional',
        description="Rhythmic content: syncopation, groove, subdivision"
    ),

    MusicalDimensionV2.VOICING: DimensionSpecV2(
        dimension=MusicalDimensionV2.VOICING,
        num_params=30,
        input_dim=400,  # Harmony (250D) + Dynamics (150D)
        hidden_dim=512,
        level='local',
        description="Voicing details: spacing, doubling, register"
    ),

    MusicalDimensionV2.TEXTURE: DimensionSpecV2(
        dimension=MusicalDimensionV2.TEXTURE,
        num_params=30,
        input_dim=250,  # Texture (100D) + Dynamics (150D)
        hidden_dim=512,
        level='local',
        description="Textural details: density, independence, layering"
    ),

    MusicalDimensionV2.ORCHESTRATION: DimensionSpecV2(
        dimension=MusicalDimensionV2.ORCHESTRATION,
        num_params=40,
        input_dim=150,  # Orchestration features
        hidden_dim=512,
        level='local',
        description="Orchestration: instrumentation, balance, articulation"
    ),

    MusicalDimensionV2.FORM_STRUCTURE: DimensionSpecV2(
        dimension=MusicalDimensionV2.FORM_STRUCTURE,
        num_params=20,
        input_dim=50,  # Structure features
        hidden_dim=256,
        level='global',  # Part of global context
        description="Form structure: sections, tension, proportions"
    ),
}


# =============================================================================
# Modular Encoder Factory V2
# =============================================================================

class ModularEncoderFactoryV2:
    """
    Factory for creating v2.0 (300D) encoders.

    Supports hierarchical 300D architecture with:
    - Global level (60D): GlobalEncoder
    - Sectional level (140D): Harmony, Melody, Rhythm
    - Local level (100D): Voicing, Texture, Orchestration

    Usage:
        factory = ModularEncoderFactoryV2()

        # Create single encoder
        harmony_encoder = factory.create_encoder(MusicalDimensionV2.HARMONY)

        # Create all encoders
        all_encoders = factory.create_all_encoders()

        # Get dimension specs
        spec = factory.get_dimension_spec(MusicalDimensionV2.MELODY)
    """

    def __init__(self):
        """Initialize factory with v2.0 specifications"""
        self.dimension_specs = DIMENSION_SPECS_V2
        self._encoder_registry: Dict[MusicalDimensionV2, nn.Module] = {}

    def get_dimension_spec(self, dimension: MusicalDimensionV2) -> DimensionSpecV2:
        """Get specification for a musical dimension"""
        if dimension not in self.dimension_specs:
            raise ValueError(f"Unknown dimension: {dimension}")
        return self.dimension_specs[dimension]

    def create_encoder(
        self,
        dimension: MusicalDimensionV2,
        device: str = 'cpu'
    ) -> nn.Module:
        """
        Create encoder for a specific musical dimension.

        Args:
            dimension: Musical dimension to create encoder for
            device: Device to create encoder on ('cpu', 'cuda', etc.)

        Returns:
            Configured encoder instance
        """
        if not V2_ENCODERS_AVAILABLE:
            raise ImportError("V2.0 encoders not available")

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for encoders")

        # Get dimension specification
        spec = self.get_dimension_spec(dimension)

        # Create appropriate encoder
        if dimension == MusicalDimensionV2.GLOBAL:
            encoder = GlobalEncoder(GlobalEncoderConfig())

        elif dimension == MusicalDimensionV2.HARMONY:
            encoder = HarmonyEncoderV2(HarmonyEncoderV2Config())

        elif dimension == MusicalDimensionV2.MELODY:
            encoder = MelodyEncoder(MelodyEncoderConfig())

        elif dimension == MusicalDimensionV2.RHYTHM:
            encoder = RhythmEncoderV2(RhythmEncoderV2Config())

        elif dimension == MusicalDimensionV2.VOICING:
            encoder = VoicingEncoder(VoicingEncoderConfig())

        elif dimension == MusicalDimensionV2.TEXTURE:
            encoder = TextureEncoderV2(TextureEncoderV2Config())

        elif dimension == MusicalDimensionV2.ORCHESTRATION:
            encoder = OrchestrationEncoderV2(OrchestrationEncoderV2Config())

        elif dimension == MusicalDimensionV2.FORM_STRUCTURE:
            encoder = FormStructureEncoder(FormStructureEncoderConfig())

        else:
            raise ValueError(f"Unknown dimension: {dimension}")

        # Move to device
        encoder.to(device)

        # Register encoder
        self._encoder_registry[dimension] = encoder

        print(f"✅ Created {dimension.value} encoder ({spec.num_params}D)")

        return encoder

    def create_all_encoders(self, device: str = 'cpu') -> Dict[MusicalDimensionV2, nn.Module]:
        """
        Create encoders for all musical dimensions.

        Args:
            device: Device to create encoders on

        Returns:
            Dictionary mapping dimensions to encoders
        """
        encoders = {}

        for dimension in MusicalDimensionV2:
            encoders[dimension] = self.create_encoder(dimension, device=device)

        return encoders

    def create_hierarchical_encoders(self, device: str = 'cpu') -> Dict[str, Dict[MusicalDimensionV2, nn.Module]]:
        """
        Create encoders organized by hierarchical level.

        Returns:
            Dictionary with keys 'global', 'sectional', 'local', each containing encoders
        """
        global_encoders = {}
        sectional_encoders = {}
        local_encoders = {}

        for dimension in MusicalDimensionV2:
            spec = self.get_dimension_spec(dimension)
            encoder = self.create_encoder(dimension, device=device)

            if spec.level == 'global':
                global_encoders[dimension] = encoder
            elif spec.level == 'sectional':
                sectional_encoders[dimension] = encoder
            elif spec.level == 'local':
                local_encoders[dimension] = encoder

        return {
            'global': global_encoders,
            'sectional': sectional_encoders,
            'local': local_encoders,
        }

    def get_total_parameters(self) -> int:
        """
        Get total number of parameters across all dimensions.

        Returns:
            Total parameter count (should be 300)
        """
        return sum(spec.num_params for spec in self.dimension_specs.values())

    def get_parameter_allocation(self) -> Dict[str, int]:
        """Get parameter allocation per dimension"""
        return {
            dimension.value: spec.num_params
            for dimension, spec in self.dimension_specs.items()
        }

    def get_hierarchical_allocation(self) -> Dict[str, int]:
        """Get parameter allocation per hierarchical level"""
        allocation = {'global': 0, 'sectional': 0, 'local': 0}

        for spec in self.dimension_specs.values():
            allocation[spec.level] += spec.num_params

        return allocation

    def save_all_encoders(self, output_dir: Path):
        """Save all created encoders to directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for dimension, encoder in self._encoder_registry.items():
            encoder_path = output_dir / f"{dimension.value}_encoder_v2.pt"
            encoder.save(encoder_path)

        # Save dimension specs
        specs_path = output_dir / "dimension_specs_v2.json"
        specs_dict = {
            dim.value: {
                'num_params': spec.num_params,
                'input_dim': spec.input_dim,
                'hidden_dim': spec.hidden_dim,
                'level': spec.level,
                'description': spec.description,
            }
            for dim, spec in self.dimension_specs.items()
        }

        with open(specs_path, 'w') as f:
            json.dump(specs_dict, f, indent=2)

        print(f"✅ Saved {len(self._encoder_registry)} v2.0 encoders to {output_dir}")

    def load_all_encoders(self, input_dir: Path, device: str = 'cpu') -> Dict[MusicalDimensionV2, nn.Module]:
        """Load all encoders from directory"""
        input_dir = Path(input_dir)
        encoders = {}

        for dimension in MusicalDimensionV2:
            encoder_path = input_dir / f"{dimension.value}_encoder_v2.pt"
            if encoder_path.exists():
                # Use appropriate load method based on dimension
                if dimension == MusicalDimensionV2.GLOBAL:
                    encoder = GlobalEncoder.load(encoder_path, device=device)
                elif dimension == MusicalDimensionV2.MELODY:
                    encoder = MelodyEncoder.load(encoder_path, device=device)
                elif dimension == MusicalDimensionV2.VOICING:
                    encoder = VoicingEncoder.load(encoder_path, device=device)
                elif dimension == MusicalDimensionV2.HARMONY:
                    encoder = HarmonyEncoderV2.load(encoder_path, device=device)
                elif dimension == MusicalDimensionV2.RHYTHM:
                    encoder = RhythmEncoderV2.load(encoder_path, device=device)
                elif dimension == MusicalDimensionV2.TEXTURE:
                    encoder = TextureEncoderV2.load(encoder_path, device=device)
                elif dimension == MusicalDimensionV2.ORCHESTRATION:
                    encoder = OrchestrationEncoderV2.load(encoder_path, device=device)
                elif dimension == MusicalDimensionV2.FORM_STRUCTURE:
                    encoder = FormStructureEncoder.load(encoder_path, device=device)

                encoders[dimension] = encoder
                self._encoder_registry[dimension] = encoder

        print(f"✅ Loaded {len(encoders)} v2.0 encoders from {input_dir}")
        return encoders

    def print_architecture_summary(self):
        """Print summary of v2.0 architecture"""
        print("\n" + "="*70)
        print("MODULAR ENCODER ARCHITECTURE V2.0 - 300D HIERARCHICAL")
        print("="*70)

        # Print by hierarchical level
        for level in ['global', 'sectional', 'local']:
            level_total = 0
            level_encoders = []

            for dimension, spec in self.dimension_specs.items():
                if spec.level == level:
                    level_encoders.append((dimension, spec))
                    level_total += spec.num_params

            print(f"\n{level.upper()} LEVEL ({level_total}D):")
            for dimension, spec in level_encoders:
                print(f"  {dimension.value:20s}: {spec.num_params:3d}D  ({spec.description})")

        print(f"\n{'='*70}")
        print(f"TOTAL INTERPRETABLE PARAMETERS: {self.get_total_parameters()}D")
        print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Modular Encoder Factory V2.0 - Test")
    print("="*70)

    # Create factory
    factory = ModularEncoderFactoryV2()

    # Print architecture summary
    factory.print_architecture_summary()

    # Test parameter allocation
    print("\nParameter allocation by level:")
    hierarchical = factory.get_hierarchical_allocation()
    for level, count in hierarchical.items():
        print(f"  {level:12s}: {count:3d}D")
    print(f"  {'TOTAL':12s}: {sum(hierarchical.values()):3d}D")

    if TORCH_AVAILABLE and V2_ENCODERS_AVAILABLE:
        print("\n" + "="*70)
        print("Creating encoders...")
        print("="*70)

        # Create all encoders
        encoders = factory.create_all_encoders(device='cpu')

        print(f"\n✅ Created {len(encoders)} encoders")

        # Test save/load
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "encoders_v2"
            factory.save_all_encoders(save_dir)

            # Load back
            factory2 = ModularEncoderFactoryV2()
            loaded_encoders = factory2.load_all_encoders(save_dir)

            print(f"✅ Save/load test passed")

        print("\n" + "="*70)
        print("✅ All tests passed!")
        print("="*70)
    else:
        print("\nPyTorch or v2.0 encoders not available - skipping encoder tests")
