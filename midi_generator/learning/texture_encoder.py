"""
Texture Semantic Encoder - Agent 6
===================================

Neural encoder for discovering semantic texture parameters through reconstruction gaps.

This encoder specializes in texture analysis, discovering 20 interpretable parameters:
- Homophonic vs polyphonic balance
- Voice independence metrics
- Textural density evolution
- Call-response patterns
- Layer interaction complexity

Builds on Agent 3's SemanticFeatureEncoder architecture with texture-specific
locality functions and analysis methods.

Architecture:
    Input: 200D feature vector (from OptimizedFeatureExtractor)
    Encoder: [200] → [512] → [20 texture features]
    Decoder: [20] → [512] → [200]
    Locality Predictor: [20 * 2] → [512] → [6 texture locality types]

Texture-Specific Locality Functions:
    1. DENSITY_SCALE: Scale note density while preserving patterns
    2. VOICE_SWAP: Exchange voices while preserving independence
    3. TEXTURE_INVERT: Convert homophonic ↔ polyphonic
    4. LAYER_SHIFT: Shift layers in time (stagger/align)
    5. REGISTER_SPREAD: Change vertical spacing between voices
    6. ARTICULATION_SYNC: Synchronize/desynchronize articulations

Author: Agent 6 - Texture Encoder Specialist
Date: November 21, 2025
Part of: 10-Agent Modular Semantic Discovery System
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from enum import Enum
import json
import warnings
import copy

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. Neural network functionality will be disabled.")

# NumPy is required
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not installed. Some functionality will be disabled.")

# Import base encoder from Agent 3
from midi_generator.learning.semantic_encoder import (
    SemanticFeatureEncoder,
    EncoderConfig,
    TrainingMetrics
)

# Import musical locality functions from Agent 1
try:
    from midi_generator.learning.musical_locality import (
        MusicalLocalityFunctions,
        MusicalTransform,
        LocalityType
    )
    LOCALITY_AVAILABLE = True
except ImportError:
    LOCALITY_AVAILABLE = False
    warnings.warn("MusicalLocalityFunctions not available")

# Import dynamic shaping from Agent 9
try:
    from midi_generator.transformation.dynamic_shaping import (
        DynamicShaping,
        DynamicLevel,
        PhraseContour,
        AccentPattern
    )
    DYNAMICS_AVAILABLE = True
except ImportError:
    DYNAMICS_AVAILABLE = False
    warnings.warn("DynamicShaping not available")

# Import texture generation infrastructure
try:
    from midi_generator.generators.texture_generator import (
        TextureGenerator,
        TextureType,
        AccompanimentPattern,
        TexturePattern
    )
    TEXTURE_GEN_AVAILABLE = True
except ImportError:
    TEXTURE_GEN_AVAILABLE = False
    warnings.warn("TextureGenerator not available")


# ============================================================================
# Texture-Specific Locality Types
# ============================================================================

class TextureLocalityType(Enum):
    """Texture-specific musical locality transformations"""
    DENSITY_SCALE = "density_scale"           # Scale note density
    VOICE_SWAP = "voice_swap"                 # Exchange voices
    TEXTURE_INVERT = "texture_invert"         # Homophonic ↔ polyphonic
    LAYER_SHIFT = "layer_shift"               # Temporal layer staggering
    REGISTER_SPREAD = "register_spread"       # Vertical spacing change
    ARTICULATION_SYNC = "articulation_sync"   # Sync/desync articulations


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TextureEncoderConfig(EncoderConfig):
    """Configuration for TextureSemanticEncoder"""

    # Override defaults for texture-specific settings
    num_semantic_features: int = 20  # 20 texture parameters
    num_locality_types: int = 6      # 6 texture locality types

    # Texture-specific hyperparameters
    texture_focus_weight: float = 1.5  # Extra weight for texture features
    voice_independence_weight: float = 1.0  # Weight for voice independence loss
    density_consistency_weight: float = 0.5  # Weight for density consistency

    # Feature interpretation
    texture_parameter_names: List[str] = None

    def __post_init__(self):
        """Initialize texture parameter names"""
        if self.texture_parameter_names is None:
            self.texture_parameter_names = [
                # Texture Type (4 params)
                "homophonic_polyphonic_balance",
                "monophonic_tendency",
                "heterophonic_variation",
                "texture_consistency",

                # Voice Independence (4 params)
                "voice_independence_score",
                "rhythmic_independence",
                "melodic_independence",
                "harmonic_independence",

                # Density & Complexity (4 params)
                "textural_density_mean",
                "textural_density_variance",
                "vertical_density",
                "horizontal_density",

                # Layering & Interaction (4 params)
                "layer_count",
                "layer_interaction_complexity",
                "foreground_background_separation",
                "voice_crossing_frequency",

                # Temporal Patterns (4 params)
                "call_response_strength",
                "imitation_frequency",
                "texture_evolution_rate",
                "stagger_synchronization_balance"
            ]


# ============================================================================
# Texture Analysis Methods
# ============================================================================

class TextureAnalyzer:
    """
    Texture analysis methods for extracting texture features from MIDI.

    These methods compute the 20 texture parameters that the encoder
    will learn to extract.
    """

    @staticmethod
    def analyze_texture(midi_data: Any) -> Dict[str, float]:
        """
        Analyze texture properties of MIDI data.

        Args:
            midi_data: MIDI file or note list

        Returns:
            Dictionary of 20 texture parameters (0.0 to 1.0)
        """
        # Initialize with default values
        texture_params = {
            "homophonic_polyphonic_balance": 0.5,
            "monophonic_tendency": 0.0,
            "heterophonic_variation": 0.0,
            "texture_consistency": 0.5,
            "voice_independence_score": 0.5,
            "rhythmic_independence": 0.5,
            "melodic_independence": 0.5,
            "harmonic_independence": 0.5,
            "textural_density_mean": 0.5,
            "textural_density_variance": 0.3,
            "vertical_density": 0.5,
            "horizontal_density": 0.5,
            "layer_count": 0.5,
            "layer_interaction_complexity": 0.5,
            "foreground_background_separation": 0.5,
            "voice_crossing_frequency": 0.2,
            "call_response_strength": 0.3,
            "imitation_frequency": 0.2,
            "texture_evolution_rate": 0.4,
            "stagger_synchronization_balance": 0.5
        }

        # TODO: Implement actual analysis
        # For now, return placeholder values

        return texture_params

    @staticmethod
    def compute_voice_independence(notes_by_voice: Dict[int, List]) -> float:
        """
        Compute voice independence score.

        Measures how independently voices move rhythmically and melodically.

        Args:
            notes_by_voice: Dictionary mapping voice ID to list of notes

        Returns:
            Voice independence score (0.0 = homophonic, 1.0 = polyphonic)
        """
        if len(notes_by_voice) <= 1:
            return 0.0  # Single voice = no independence

        # Analyze rhythmic independence
        rhythmic_independence = 0.0
        voice_pairs = []

        for v1, notes1 in notes_by_voice.items():
            for v2, notes2 in notes_by_voice.items():
                if v1 < v2:
                    # Compute rhythmic correlation
                    # Higher correlation = lower independence
                    correlation = TextureAnalyzer._compute_rhythmic_correlation(notes1, notes2)
                    rhythmic_independence += (1.0 - correlation)
                    voice_pairs.append((v1, v2))

        if voice_pairs:
            rhythmic_independence /= len(voice_pairs)

        return rhythmic_independence

    @staticmethod
    def _compute_rhythmic_correlation(notes1: List, notes2: List) -> float:
        """
        Compute rhythmic correlation between two voices.

        Returns:
            Correlation (0.0 = independent, 1.0 = synchronized)
        """
        # Simplified: check onset alignment
        # TODO: Implement proper correlation analysis
        return 0.5

    @staticmethod
    def compute_textural_density(notes: List, time_window: float = 1.0) -> List[float]:
        """
        Compute textural density over time.

        Measures how many notes are sounding per unit time.

        Args:
            notes: List of notes with start_time and duration
            time_window: Time window for density calculation (seconds)

        Returns:
            List of density values over time
        """
        if not notes:
            return [0.0]

        # Determine time range
        start_time = min(n.get('start_time', 0) for n in notes)
        end_time = max(n.get('start_time', 0) + n.get('duration', 0) for n in notes)

        # Compute density in time windows
        densities = []
        current_time = start_time

        while current_time < end_time:
            # Count notes active in this window
            active_notes = 0
            for note in notes:
                note_start = note.get('start_time', 0)
                note_end = note_start + note.get('duration', 0)

                # Check if note overlaps with window
                if note_start < current_time + time_window and note_end > current_time:
                    active_notes += 1

            densities.append(active_notes)
            current_time += time_window

        return densities

    @staticmethod
    def detect_call_response(notes: List) -> float:
        """
        Detect call-and-response patterns.

        Args:
            notes: List of notes

        Returns:
            Call-response strength (0.0 to 1.0)
        """
        # TODO: Implement call-response detection
        # Look for alternating phrase patterns
        return 0.3

    @staticmethod
    def compute_layer_interaction(layers: Dict[str, List]) -> float:
        """
        Compute complexity of layer interactions.

        Args:
            layers: Dictionary of layer name to note list

        Returns:
            Interaction complexity (0.0 = independent, 1.0 = highly interactive)
        """
        if len(layers) <= 1:
            return 0.0

        # Analyze timing relationships between layers
        # TODO: Implement proper layer interaction analysis

        return 0.5


# ============================================================================
# Main Texture Encoder
# ============================================================================

class TextureSemanticEncoder(SemanticFeatureEncoder):
    """
    Texture-specialized semantic feature encoder.

    Discovers 20 interpretable texture parameters using:
    - Texture-specific locality functions
    - Voice independence analysis
    - Density calculation
    - Layer interaction metrics

    Usage:
        # Initialize
        config = TextureEncoderConfig()
        encoder = TextureSemanticEncoder(config)

        # Extract texture features
        features = torch.randn(32, 200)  # Batch of 32 feature vectors
        output = encoder(features)

        # Get texture parameters
        texture_params = encoder.extract_texture_parameters(features)
    """

    def __init__(self, config: Optional[TextureEncoderConfig] = None):
        """
        Initialize texture encoder.

        Args:
            config: TextureEncoderConfig or None (uses defaults)
        """
        if config is None:
            config = TextureEncoderConfig()

        # Initialize base encoder
        super().__init__(config)

        self.texture_config = config
        self.texture_analyzer = TextureAnalyzer()

        # Initialize dynamic shaping agent connection
        if DYNAMICS_AVAILABLE:
            self.dynamics_agent = DynamicShaping()
        else:
            self.dynamics_agent = None

    def extract_texture_parameters(
        self,
        x: torch.Tensor,
        as_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, float]]:
        """
        Extract texture parameters from input features.

        Args:
            x: Input features [batch_size, 200] or [200]
            as_dict: Return as dictionary with parameter names

        Returns:
            Texture parameters as tensor or dictionary
        """
        # Extract semantic features
        texture_features = self.extract_semantic_features(x, as_numpy=False)

        # Convert to dictionary if requested
        if as_dict:
            # Handle single sample
            if texture_features.dim() == 1:
                values = texture_features.detach().cpu().numpy()
            else:
                values = texture_features[0].detach().cpu().numpy()

            return {
                name: float(values[i])
                for i, name in enumerate(self.texture_config.texture_parameter_names)
            }

        return texture_features

    def apply_texture_locality(
        self,
        features: torch.Tensor,
        locality_type: TextureLocalityType,
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        Apply texture-specific locality transformation.

        Args:
            features: Input features [batch_size, 200]
            locality_type: Type of texture transformation
            strength: Transformation strength (0.0 to 1.0)

        Returns:
            Transformed features
        """
        # TODO: Implement texture-specific transformations
        # For now, return slightly perturbed features
        noise = torch.randn_like(features) * 0.1 * strength
        return features + noise

    def connect_dynamic_shaping(
        self,
        notes: List,
        texture_params: Dict[str, float]
    ) -> List:
        """
        Apply dynamic shaping based on texture parameters.

        Connects to Agent 9 (Dynamic Shaping) to apply appropriate
        dynamics based on texture characteristics.

        Args:
            notes: List of notes (NoteEvent or JazzNote)
            texture_params: Texture parameters from encoder

        Returns:
            Notes with dynamics applied
        """
        if not DYNAMICS_AVAILABLE or self.dynamics_agent is None:
            warnings.warn("DynamicShaping not available")
            return notes

        if not notes:
            return notes

        # Determine appropriate dynamic contour based on texture
        homophonic_balance = texture_params.get("homophonic_polyphonic_balance", 0.5)
        density = texture_params.get("textural_density_mean", 0.5)
        call_response = texture_params.get("call_response_strength", 0.0)

        # Choose contour based on texture type
        if homophonic_balance > 0.7:
            # Homophonic texture: use arch contour
            contour = PhraseContour.ARCH
        elif call_response > 0.5:
            # Call-response: use alternating dynamics
            shaped = self.dynamics_agent.apply_accent_pattern(
                notes,
                pattern=AccentPattern.ALTERNATING,
                accent_amount=15
            )
            return shaped
        else:
            # Polyphonic texture: flatter dynamics
            contour = PhraseContour.FLAT

        # Apply phrase contour
        base_velocity = int(50 + (density * 40))  # 50-90 based on density
        shaped = self.dynamics_agent.apply_phrase_contour(
            notes,
            phrase_length_bars=4,
            contour=contour,
            base_velocity=base_velocity,
            variation_range=20
        )

        return shaped

    def analyze_midi_texture(self, midi_path: str) -> Dict[str, float]:
        """
        Analyze texture of a MIDI file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Dictionary of 20 texture parameters
        """
        # Use texture analyzer to extract ground truth texture features
        return self.texture_analyzer.analyze_texture(midi_path)

    def save(self, path: Path, include_config: bool = True):
        """
        Save texture encoder checkpoint.

        Args:
            path: Path to save model
            include_config: Whether to save config alongside model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.texture_config.to_dict(),
            'training_step': self.training_step,
            'encoder_type': 'texture'
        }

        torch.save(checkpoint, path)

        # Save config separately
        if include_config:
            config_path = path.with_suffix('.json')
            with open(config_path, 'w') as f:
                json.dump(self.texture_config.to_dict(), f, indent=2)

        print(f"✅ Texture encoder saved to {path}")

    @classmethod
    def load(cls, path: Path, device: str = 'cpu') -> 'TextureSemanticEncoder':
        """
        Load texture encoder checkpoint.

        Args:
            path: Path to model checkpoint
            device: Device to load model on

        Returns:
            Loaded texture encoder
        """
        checkpoint = torch.load(path, map_location=device)

        # Create config
        config = TextureEncoderConfig.from_dict(checkpoint['config'])

        # Create model
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.training_step = checkpoint.get('training_step', 0)

        model.to(device)
        model.eval()

        print(f"✅ Texture encoder loaded from {path}")
        print(f"   Training step: {model.training_step}")
        print(f"   Texture parameters: {config.num_semantic_features}")

        return model


# ============================================================================
# Utility Functions
# ============================================================================

def create_default_texture_encoder(device: str = 'cpu') -> TextureSemanticEncoder:
    """
    Create texture encoder with default configuration.

    Args:
        device: Device to create model on

    Returns:
        Initialized texture encoder
    """
    config = TextureEncoderConfig()
    encoder = TextureSemanticEncoder(config)
    encoder.to(device)
    return encoder


def extract_texture_from_midi(
    encoder: TextureSemanticEncoder,
    midi_path: str,
    feature_extractor: Optional[Any] = None
) -> Dict[str, float]:
    """
    Extract texture parameters from MIDI file.

    Args:
        encoder: Trained texture encoder
        midi_path: Path to MIDI file
        feature_extractor: Optional feature extractor (200D features)

    Returns:
        Dictionary of 20 texture parameters
    """
    # TODO: Implement MIDI → features → texture params pipeline
    # For now, use analyzer directly
    return encoder.analyze_midi_texture(midi_path)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Texture Semantic Encoder - Agent 6")
    print("="*70)

    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch is not installed.")
        print("   Install PyTorch to use this module:")
        print("   pip install torch")
        exit(1)

    # Create texture encoder
    print("\n1. Creating texture encoder...")
    config = TextureEncoderConfig()
    encoder = create_default_texture_encoder()
    print(f"   ✅ Created texture encoder with {config.num_semantic_features} texture parameters")
    print(f"   ✅ Texture-specific locality types: {config.num_locality_types}")

    # List texture parameters
    print("\n2. Texture parameters to discover:")
    for i, param_name in enumerate(config.texture_parameter_names, 1):
        print(f"   {i:2d}. {param_name}")

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 16
    x = torch.randn(batch_size, 200)
    output = encoder(x)
    print(f"   ✅ Forward pass successful")
    print(f"      Semantic features shape: {output['semantic_features'].shape}")
    print(f"      Reconstructed shape: {output['reconstructed'].shape}")

    # Extract texture parameters
    print("\n4. Extracting texture parameters...")
    texture_params = encoder.extract_texture_parameters(x[0], as_dict=True)
    print(f"   ✅ Extracted {len(texture_params)} texture parameters")
    print(f"   Sample parameters:")
    for i, (name, value) in enumerate(list(texture_params.items())[:5]):
        print(f"      {name}: {value:.3f}")
    print(f"      ...")

    # Test save/load
    print("\n5. Testing save/load...")
    save_path = Path("/tmp/texture_encoder.pt")
    encoder.save(save_path)
    encoder_loaded = TextureSemanticEncoder.load(save_path)
    print(f"   ✅ Save/load successful")

    # Test dynamic shaping connection
    print("\n6. Testing dynamic shaping connection...")
    if DYNAMICS_AVAILABLE:
        # Create dummy notes
        from midi_generator.analysis.midi_analyzer import NoteEvent
        test_notes = []
        for i in range(16):
            note = NoteEvent(
                start_time=float(i),
                duration=0.9,
                start_tick=i * 480,
                duration_ticks=int(0.9 * 480),
                pitch=60 + (i % 8),
                velocity=75,
                channel=0,
                track_idx=0
            )
            test_notes.append(note)

        shaped_notes = encoder.connect_dynamic_shaping(test_notes, texture_params)
        print(f"   ✅ Dynamic shaping applied to {len(shaped_notes)} notes")
        velocities = [n.velocity for n in shaped_notes[:8]]
        print(f"      Sample velocities: {velocities}")
    else:
        print(f"   ⚠️  DynamicShaping not available (skipped)")

    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Train encoder on MIDI corpus (see gap_discovery_trainer.py)")
    print("  2. Analyze learned texture features (see feature_interpreter.py)")
    print("  3. Integrate with other modular encoders (Agent 8)")
    print("="*70)
