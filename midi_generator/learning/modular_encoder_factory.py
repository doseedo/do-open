"""
Modular Encoder Factory - Agent 8: Integration Pipeline Builder
================================================================

Factory for creating domain-specific semantic encoders for modular parameter discovery.

This factory creates specialized encoders for different musical dimensions:
- Harmony Encoder (30 params): Chord progressions, voice leading, harmonic rhythm
- Rhythm Encoder (20 params): Groove, syncopation, swing, polyrhythms
- Form Encoder (15 params): Structure, tension arcs, section relationships
- Orchestration Encoder (25 params): Instrumentation, doubling, voice spacing
- Texture Encoder (20 params): Density, voice independence, layering
- Cross-Dimensional Encoder (10 params): Inter-domain parameter coupling

Total: 120 interpretable musical parameters

Architecture:
    Each encoder inherits from SemanticFeatureEncoder but specializes for its domain.
    The factory configures input dimensions, locality functions, and interpretation
    strategies specific to each musical dimension.

Author: Agent 8 - Integration Pipeline Builder
Date: November 21, 2025
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
import json
import warnings

# Import base encoder
try:
    from midi_generator.learning.semantic_encoder import (
        SemanticFeatureEncoder,
        EncoderConfig
    )
    ENCODER_AVAILABLE = True
except ImportError:
    ENCODER_AVAILABLE = False
    warnings.warn("SemanticFeatureEncoder not available")

# Import PyTorch if available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available")


# =============================================================================
# Musical Dimension Definitions
# =============================================================================

class MusicalDimension(Enum):
    """Musical dimensions for modular discovery"""
    HARMONY = "harmony"
    RHYTHM = "rhythm"
    FORM = "form"
    ORCHESTRATION = "orchestration"
    TEXTURE = "texture"
    CROSS_DIMENSIONAL = "cross_dimensional"


@dataclass
class DimensionSpec:
    """
    Specification for a musical dimension encoder.

    Defines the configuration, locality functions, and interpretation
    strategies specific to each musical dimension.
    """
    dimension: MusicalDimension
    num_params: int  # Number of parameters to discover

    # Feature extraction
    input_dim: int = 200  # Input feature dimension
    hidden_dim: int = 512  # Hidden layer dimension

    # Locality transformations specific to this dimension
    locality_functions: List[str] = field(default_factory=list)

    # Interpretation hints
    expected_concepts: List[str] = field(default_factory=list)

    # Loss weights (dimension-specific tuning)
    reconstruction_weight: float = 1.0
    locality_weight: float = 0.5
    sparsity_weight: float = 0.01

    # Description
    description: str = ""

    def to_encoder_config(self) -> EncoderConfig:
        """Convert to base EncoderConfig"""
        return EncoderConfig(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_semantic_features=self.num_params,
            num_locality_types=len(self.locality_functions),
            reconstruction_weight=self.reconstruction_weight,
            locality_weight=self.locality_weight,
            sparsity_weight=self.sparsity_weight
        )


# =============================================================================
# Dimension Specifications
# =============================================================================

HARMONY_SPEC = DimensionSpec(
    dimension=MusicalDimension.HARMONY,
    num_params=30,
    locality_functions=[
        'transpose',      # Key invariance
        'invert',         # Interval preservation
        'voice_swap',     # Voice leading invariance
        'harmonic_shift', # Cycle of 5ths
        'modal_mixture',  # Mode interchange
    ],
    expected_concepts=[
        'harmonic_complexity',
        'voice_leading_smoothness',
        'chord_density',
        'harmonic_rhythm',
        'tonal_center_stability',
        'dissonance_level',
        'chord_color_richness',
        'bass_motion_pattern',
        'secondary_dominants_frequency',
        'harmonic_tension_resolution',
    ],
    description="Harmony encoder: chord progressions, voice leading, harmonic rhythm"
)

RHYTHM_SPEC = DimensionSpec(
    dimension=MusicalDimension.RHYTHM,
    num_params=20,
    locality_functions=[
        'augment',        # Tempo scaling
        'time_shift',     # Phase invariance
        'retrograde',     # Reverse patterns
        'metric_shift',   # Meter changes
    ],
    expected_concepts=[
        'syncopation_intensity',
        'groove_pocket_tightness',
        'polyrhythmic_complexity',
        'swing_straight_continuum',
        'rhythmic_density',
        'accent_pattern_regularity',
        'subdivision_preference',
        'rhythmic_anticipation',
    ],
    description="Rhythm encoder: groove, syncopation, swing, polyrhythms"
)

FORM_SPEC = DimensionSpec(
    dimension=MusicalDimension.FORM,
    num_params=15,
    locality_functions=[
        'section_permute',     # Rearrange sections
        'section_repeat',      # Repeat sections
        'section_truncate',    # Remove sections
    ],
    expected_concepts=[
        'tension_arc_shape',
        'section_contrast_degree',
        'climax_position_ratio',
        'repetition_variation_balance',
        'golden_ratio_tendency',
        'intro_outro_symmetry',
        'bridge_contrast_level',
    ],
    description="Form encoder: structure, tension arcs, section relationships"
)

ORCHESTRATION_SPEC = DimensionSpec(
    dimension=MusicalDimension.ORCHESTRATION,
    num_params=25,
    locality_functions=[
        'instrument_swap',        # Change instrumentation
        'octave_shift',          # Vertical spacing
        'voice_add',             # Add/remove voices
        'timbre_morph',          # Timbral variation
    ],
    expected_concepts=[
        'instrumentation_density_curve',
        'vertical_spacing_preference',
        'doubling_strategy',
        'timbral_balance_profile',
        'voice_crossing_frequency',
        'register_distribution',
        'instrument_role_clarity',
        'brass_string_balance',
    ],
    description="Orchestration encoder: instrumentation, doubling, voice spacing"
)

TEXTURE_SPEC = DimensionSpec(
    dimension=MusicalDimension.TEXTURE,
    num_params=20,
    locality_functions=[
        'density_morph',          # Change note density
        'layer_add_remove',       # Layering
        'homophonic_polyphonic',  # Texture type
    ],
    expected_concepts=[
        'homophonic_vs_polyphonic',
        'voice_independence_score',
        'textural_density_evolution',
        'call_response_patterns',
        'layer_interaction_complexity',
        'rhythmic_unison_frequency',
        'melodic_divergence',
    ],
    description="Texture encoder: density, voice independence, layering"
)

CROSS_DIMENSIONAL_SPEC = DimensionSpec(
    dimension=MusicalDimension.CROSS_DIMENSIONAL,
    num_params=10,
    input_dim=110,  # Takes concatenated outputs from other 5 encoders (30+20+15+25+20)
    locality_functions=[
        'global_transform',  # Apply transformation across all dimensions
    ],
    expected_concepts=[
        'harmonic_rhythmic_coupling',
        'form_driven_texture_change',
        'structural_harmonic_anchoring',
        'orchestral_intensity_gradient',
        'climax_convergence_factor',
    ],
    description="Cross-dimensional encoder: inter-domain parameter coupling"
)


# =============================================================================
# Modular Encoder Factory
# =============================================================================

class ModularEncoderFactory:
    """
    Factory for creating domain-specific semantic encoders.

    This factory:
    1. Maintains dimension specifications for all musical dimensions
    2. Creates encoders with dimension-specific configurations
    3. Supports custom encoder variants
    4. Provides registry of all created encoders

    Usage:
        factory = ModularEncoderFactory()

        # Create single encoder
        harmony_encoder = factory.create_encoder(MusicalDimension.HARMONY)

        # Create all encoders
        encoders = factory.create_all_encoders()

        # Get dimension spec
        spec = factory.get_dimension_spec(MusicalDimension.RHYTHM)
    """

    def __init__(self):
        """Initialize factory with default dimension specifications"""
        self.dimension_specs: Dict[MusicalDimension, DimensionSpec] = {
            MusicalDimension.HARMONY: HARMONY_SPEC,
            MusicalDimension.RHYTHM: RHYTHM_SPEC,
            MusicalDimension.FORM: FORM_SPEC,
            MusicalDimension.ORCHESTRATION: ORCHESTRATION_SPEC,
            MusicalDimension.TEXTURE: TEXTURE_SPEC,
            MusicalDimension.CROSS_DIMENSIONAL: CROSS_DIMENSIONAL_SPEC,
        }

        # Registry of created encoders
        self._encoder_registry: Dict[MusicalDimension, SemanticFeatureEncoder] = {}

    def get_dimension_spec(self, dimension: MusicalDimension) -> DimensionSpec:
        """Get specification for a musical dimension"""
        if dimension not in self.dimension_specs:
            raise ValueError(f"Unknown dimension: {dimension}")
        return self.dimension_specs[dimension]

    def create_encoder(
        self,
        dimension: MusicalDimension,
        custom_config: Optional[EncoderConfig] = None,
        device: str = 'cpu'
    ) -> SemanticFeatureEncoder:
        """
        Create encoder for a specific musical dimension.

        Args:
            dimension: Musical dimension to create encoder for
            custom_config: Optional custom configuration (overrides dimension spec)
            device: Device to create encoder on ('cpu', 'cuda', etc.)

        Returns:
            Configured SemanticFeatureEncoder instance
        """
        if not ENCODER_AVAILABLE:
            raise ImportError("SemanticFeatureEncoder not available")

        # Get dimension specification
        spec = self.get_dimension_spec(dimension)

        # Create configuration
        if custom_config is None:
            config = spec.to_encoder_config()
        else:
            config = custom_config

        # Create encoder
        encoder = SemanticFeatureEncoder(config)

        if TORCH_AVAILABLE:
            encoder.to(device)

        # Register encoder
        self._encoder_registry[dimension] = encoder

        print(f"✅ Created {dimension.value} encoder ({spec.num_params} params)")

        return encoder

    def create_all_encoders(
        self,
        device: str = 'cpu',
        include_cross_dimensional: bool = True
    ) -> Dict[MusicalDimension, SemanticFeatureEncoder]:
        """
        Create encoders for all musical dimensions.

        Args:
            device: Device to create encoders on
            include_cross_dimensional: Whether to include cross-dimensional encoder

        Returns:
            Dictionary mapping dimensions to encoders
        """
        encoders = {}

        # Create base dimension encoders
        base_dimensions = [
            MusicalDimension.HARMONY,
            MusicalDimension.RHYTHM,
            MusicalDimension.FORM,
            MusicalDimension.ORCHESTRATION,
            MusicalDimension.TEXTURE,
        ]

        for dimension in base_dimensions:
            encoders[dimension] = self.create_encoder(dimension, device=device)

        # Create cross-dimensional encoder (depends on base encoders)
        if include_cross_dimensional:
            encoders[MusicalDimension.CROSS_DIMENSIONAL] = self.create_encoder(
                MusicalDimension.CROSS_DIMENSIONAL,
                device=device
            )

        return encoders

    def get_total_parameters(self, include_cross_dimensional: bool = True) -> int:
        """
        Get total number of parameters across all dimensions.

        Returns:
            Total parameter count (should be 120)
        """
        total = 0
        for dimension, spec in self.dimension_specs.items():
            if dimension == MusicalDimension.CROSS_DIMENSIONAL and not include_cross_dimensional:
                continue
            total += spec.num_params
        return total

    def get_encoder(self, dimension: MusicalDimension) -> Optional[SemanticFeatureEncoder]:
        """Get previously created encoder from registry"""
        return self._encoder_registry.get(dimension)

    def save_all_encoders(self, output_dir: Path):
        """
        Save all created encoders to directory.

        Args:
            output_dir: Directory to save encoders
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for dimension, encoder in self._encoder_registry.items():
            encoder_path = output_dir / f"{dimension.value}_encoder.pt"
            encoder.save(encoder_path)

        # Save dimension specs
        specs_path = output_dir / "dimension_specs.json"
        specs_dict = {
            dim.value: {
                'num_params': spec.num_params,
                'locality_functions': spec.locality_functions,
                'expected_concepts': spec.expected_concepts,
                'description': spec.description,
            }
            for dim, spec in self.dimension_specs.items()
        }

        with open(specs_path, 'w') as f:
            json.dump(specs_dict, f, indent=2)

        print(f"✅ Saved {len(self._encoder_registry)} encoders to {output_dir}")

    def load_all_encoders(self, input_dir: Path, device: str = 'cpu') -> Dict[MusicalDimension, SemanticFeatureEncoder]:
        """
        Load all encoders from directory.

        Args:
            input_dir: Directory containing saved encoders
            device: Device to load encoders on

        Returns:
            Dictionary mapping dimensions to loaded encoders
        """
        input_dir = Path(input_dir)
        encoders = {}

        for dimension in MusicalDimension:
            encoder_path = input_dir / f"{dimension.value}_encoder.pt"
            if encoder_path.exists():
                encoder = SemanticFeatureEncoder.load(encoder_path, device=device)
                encoders[dimension] = encoder
                self._encoder_registry[dimension] = encoder

        print(f"✅ Loaded {len(encoders)} encoders from {input_dir}")
        return encoders

    def print_architecture_summary(self):
        """Print summary of modular architecture"""
        print("\n" + "="*70)
        print("MODULAR SEMANTIC DISCOVERY ARCHITECTURE")
        print("="*70)

        total_params = 0
        for dimension, spec in self.dimension_specs.items():
            print(f"\n{dimension.value.upper()}:")
            print(f"  Parameters: {spec.num_params}")
            print(f"  Locality functions: {', '.join(spec.locality_functions)}")
            print(f"  Description: {spec.description}")
            total_params += spec.num_params

        print(f"\n{'='*70}")
        print(f"TOTAL INTERPRETABLE PARAMETERS: {total_params}")
        print("="*70 + "\n")


# =============================================================================
# Utility Functions
# =============================================================================

def create_default_factory() -> ModularEncoderFactory:
    """Create factory with default configuration"""
    return ModularEncoderFactory()


def get_dimension_from_string(dimension_str: str) -> MusicalDimension:
    """Convert string to MusicalDimension enum"""
    try:
        return MusicalDimension(dimension_str.lower())
    except ValueError:
        raise ValueError(f"Invalid dimension: {dimension_str}")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Modular Encoder Factory - Agent 8")
    print("="*70)

    # Create factory
    print("\n1. Creating modular encoder factory...")
    factory = create_default_factory()
    print("   ✅ Factory created")

    # Print architecture summary
    factory.print_architecture_summary()

    if TORCH_AVAILABLE and ENCODER_AVAILABLE:
        # Create all encoders
        print("\n2. Creating all modular encoders...")
        encoders = factory.create_all_encoders(device='cpu')
        print(f"   ✅ Created {len(encoders)} encoders")

        # Test encoder
        print("\n3. Testing harmony encoder...")
        harmony_encoder = encoders[MusicalDimension.HARMONY]
        test_input = torch.randn(4, 200)  # Batch of 4
        semantic_features = harmony_encoder.extract_semantic_features(test_input)
        print(f"   ✅ Extracted features shape: {semantic_features.shape}")
        print(f"      Expected: [4, 30]")

        # Save encoders
        print("\n4. Testing save/load...")
        save_dir = Path("/tmp/modular_encoders_test")
        factory.save_all_encoders(save_dir)

        # Load encoders
        factory2 = create_default_factory()
        loaded_encoders = factory2.load_all_encoders(save_dir)
        print(f"   ✅ Loaded {len(loaded_encoders)} encoders")
    else:
        print("\n⚠️  PyTorch or SemanticFeatureEncoder not available")
        print("   Skipping encoder creation tests")

    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
