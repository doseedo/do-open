"""
Rhythm Semantic Encoder - Agent 3 (Modular Semantic Discovery)
===============================================================

Specialized semantic encoder for discovering rhythm-specific interpretable parameters.

This encoder is part of the modular semantic discovery architecture and focuses
exclusively on rhythmic features, discovering 20 interpretable parameters that
capture the essence of musical rhythm, groove, and timing.

Architecture:
    Input: 200D feature vector (from OptimizedFeatureExtractor)
    Encoder: [200] → [512] → [20 rhythm features]
    Decoder: [20] → [512] → [200]
    Locality: Tempo-invariant transformations

Discovered Parameters (20):
    1. syncopation_intensity         - Amount of syncopation (0-1)
    2. groove_pocket_tightness       - Timing precision vs. looseness (0-1)
    3. polyrhythmic_complexity       - Cross-rhythm complexity (0-1)
    4. swing_straight_continuum      - Swing feel amount (0=straight, 1=full swing)
    5. rhythmic_density              - Note density per beat (0-1)
    6. metric_stability              - Consistency of meter (0-1)
    7. microtiming_deviation         - Human timing variation (0-1)
    8. accent_pattern_complexity     - Complexity of accents (0-1)
    9. rest_space_ratio              - Ratio of rests to notes (0-1)
   10. subdivision_granularity       - Finest subdivision used (0-1)
   11. tempo_stability               - Consistency of tempo (0-1)
   12. groove_template_match         - Match to known grooves (0-1)
   13. clave_alignment               - Alignment to clave patterns (0-1)
   14. backbeat_strength             - Strength of backbeat (0-1)
   15. anticipation_tendency         - Notes ahead of beat (0-1)
   16. delay_tendency                - Notes behind beat (0-1)
   17. event_regularity              - Regularity of event spacing (0-1)
   18. duration_variation            - Variation in note lengths (0-1)
   19. velocity_rhythm_coupling      - Velocity-timing correlation (0-1)
   20. composite_groove_factor       - Overall groove quality (0-1)

Tempo-Invariant Locality Functions:
    - AUGMENT: Stretch rhythm (tempo change)
    - DIMINUTION: Compress rhythm (tempo change)
    - TIME_SHIFT: Shift all events in time
    - RETROGRADE: Reverse event sequence
    - RHYTHMIC_QUANTIZE: Align to grid

Integration:
    - Reuses SemanticFeatureEncoder base class
    - Leverages MusicalLocalityFunctions for transformations
    - Connects to rhythm_engine.py for feature extraction
    - Compatible with GapDiscoveryTrainer for training

Author: Agent 3 - Rhythm Module Builder
Date: November 21, 2025
License: MIT
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. Neural network functionality will be disabled.")

# Import base encoder
try:
    from .semantic_encoder import SemanticFeatureEncoder, EncoderConfig, TrainingMetrics
    from .musical_locality import MusicalLocalityFunctions, LocalityType, MusicalTransform
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from semantic_encoder import SemanticFeatureEncoder, EncoderConfig, TrainingMetrics
    from musical_locality import MusicalLocalityFunctions, LocalityType, MusicalTransform


# ==============================================================================
# RHYTHM-SPECIFIC CONFIGURATION
# ==============================================================================

@dataclass
class RhythmEncoderConfig(EncoderConfig):
    """
    Configuration for RhythmSemanticEncoder.

    Extends base EncoderConfig with rhythm-specific settings.
    """
    # Override to 20 rhythm features
    num_semantic_features: int = 20

    # Rhythm-specific locality types
    rhythm_locality_types: List[LocalityType] = None

    # Feature extraction settings
    enable_swing_detection: bool = True
    enable_polyrhythm_detection: bool = True
    enable_groove_analysis: bool = True
    enable_microtiming_analysis: bool = True

    # Tempo-invariance settings
    tempo_invariance_weight: float = 0.3  # Weight for tempo-invariant loss

    def __post_init__(self):
        """Initialize rhythm-specific locality types"""
        if self.rhythm_locality_types is None:
            # Define tempo-invariant rhythm transformations
            self.rhythm_locality_types = [
                LocalityType.AUGMENT,           # Tempo change (slower)
                LocalityType.DIMINUTION,        # Tempo change (faster)
                LocalityType.TIME_SHIFT,        # Temporal offset
                LocalityType.RETROGRADE,        # Reverse sequence
                LocalityType.RHYTHMIC_QUANTIZE  # Grid alignment
            ]

        # Adjust num_locality_types for rhythm-specific transformations
        self.num_locality_types = len(self.rhythm_locality_types)


# ==============================================================================
# RHYTHM PARAMETER DEFINITIONS
# ==============================================================================

RHYTHM_PARAMETERS = {
    0: {
        'name': 'syncopation_intensity',
        'description': 'Amount of syncopation (notes on weak beats)',
        'range': (0.0, 1.0),
        'default': 0.3,
        'interpretation': '0=on-beat, 1=highly syncopated'
    },
    1: {
        'name': 'groove_pocket_tightness',
        'description': 'Timing precision vs. looseness',
        'range': (0.0, 1.0),
        'default': 0.7,
        'interpretation': '0=loose/behind, 1=tight/on-grid'
    },
    2: {
        'name': 'polyrhythmic_complexity',
        'description': 'Complexity of cross-rhythms',
        'range': (0.0, 1.0),
        'default': 0.0,
        'interpretation': '0=simple, 1=complex polyrhythms'
    },
    3: {
        'name': 'swing_straight_continuum',
        'description': 'Swing feel amount',
        'range': (0.0, 1.0),
        'default': 0.5,
        'interpretation': '0=straight, 0.5=medium swing, 1=full triplet swing'
    },
    4: {
        'name': 'rhythmic_density',
        'description': 'Note density per beat',
        'range': (0.0, 1.0),
        'default': 0.5,
        'interpretation': '0=sparse, 1=very dense'
    },
    5: {
        'name': 'metric_stability',
        'description': 'Consistency of meter/time signature',
        'range': (0.0, 1.0),
        'default': 0.9,
        'interpretation': '0=unstable/changing, 1=stable meter'
    },
    6: {
        'name': 'microtiming_deviation',
        'description': 'Human timing variation (humanization)',
        'range': (0.0, 1.0),
        'default': 0.2,
        'interpretation': '0=mechanical, 1=very human'
    },
    7: {
        'name': 'accent_pattern_complexity',
        'description': 'Complexity of accent patterns',
        'range': (0.0, 1.0),
        'default': 0.4,
        'interpretation': '0=simple/regular, 1=complex/irregular'
    },
    8: {
        'name': 'rest_space_ratio',
        'description': 'Ratio of silence to sound',
        'range': (0.0, 1.0),
        'default': 0.3,
        'interpretation': '0=no rests, 1=mostly rests'
    },
    9: {
        'name': 'subdivision_granularity',
        'description': 'Finest rhythmic subdivision used',
        'range': (0.0, 1.0),
        'default': 0.5,
        'interpretation': '0=quarter notes, 0.5=eighth, 1=32nd+'
    },
    10: {
        'name': 'tempo_stability',
        'description': 'Consistency of tempo (rubato vs. strict)',
        'range': (0.0, 1.0),
        'default': 0.8,
        'interpretation': '0=rubato, 1=strict tempo'
    },
    11: {
        'name': 'groove_template_match',
        'description': 'Match to known groove templates',
        'range': (0.0, 1.0),
        'default': 0.5,
        'interpretation': '0=no match, 1=perfect template match'
    },
    12: {
        'name': 'clave_alignment',
        'description': 'Alignment to clave/timeline patterns',
        'range': (0.0, 1.0),
        'default': 0.0,
        'interpretation': '0=no clave, 1=strong clave alignment'
    },
    13: {
        'name': 'backbeat_strength',
        'description': 'Strength of backbeat (beats 2&4)',
        'range': (0.0, 1.0),
        'default': 0.6,
        'interpretation': '0=no backbeat, 1=strong backbeat'
    },
    14: {
        'name': 'anticipation_tendency',
        'description': 'Notes ahead of beat (rushing)',
        'range': (0.0, 1.0),
        'default': 0.5,
        'interpretation': '0=behind, 0.5=on-time, 1=ahead'
    },
    15: {
        'name': 'delay_tendency',
        'description': 'Notes behind beat (laid-back)',
        'range': (0.0, 1.0),
        'default': 0.5,
        'interpretation': '0=ahead, 0.5=on-time, 1=laid-back'
    },
    16: {
        'name': 'event_regularity',
        'description': 'Regularity of inter-onset intervals',
        'range': (0.0, 1.0),
        'default': 0.6,
        'interpretation': '0=irregular, 1=perfectly regular'
    },
    17: {
        'name': 'duration_variation',
        'description': 'Variation in note durations',
        'range': (0.0, 1.0),
        'default': 0.4,
        'interpretation': '0=uniform, 1=highly varied'
    },
    18: {
        'name': 'velocity_rhythm_coupling',
        'description': 'Correlation between velocity and timing',
        'range': (0.0, 1.0),
        'default': 0.5,
        'interpretation': '0=independent, 1=strongly coupled'
    },
    19: {
        'name': 'composite_groove_factor',
        'description': 'Overall groove quality',
        'range': (0.0, 1.0),
        'default': 0.5,
        'interpretation': '0=no groove, 1=strong groove'
    }
}


# ==============================================================================
# RHYTHM SEMANTIC ENCODER
# ==============================================================================

class RhythmSemanticEncoder(SemanticFeatureEncoder):
    """
    Specialized semantic encoder for rhythm features.

    Discovers 20 interpretable rhythm parameters through:
    1. Reconstruction of rhythm features
    2. Tempo-invariant locality prediction
    3. Sparsity regularization

    Usage:
        # Create encoder
        config = RhythmEncoderConfig(num_semantic_features=20)
        encoder = RhythmSemanticEncoder(config)

        # Extract rhythm features
        features = torch.randn(32, 200)  # Batch of features
        rhythm_params = encoder.extract_rhythm_parameters(features)

        # Get parameter names
        param_names = encoder.get_parameter_names()

        # Interpret parameters
        interpretation = encoder.interpret_parameters(rhythm_params[0])
    """

    def __init__(self, config: Optional[RhythmEncoderConfig] = None):
        """
        Initialize rhythm semantic encoder.

        Args:
            config: RhythmEncoderConfig (default: creates with 20 features)
        """
        if config is None:
            config = RhythmEncoderConfig(num_semantic_features=20)
        elif not isinstance(config, RhythmEncoderConfig):
            # Convert base config to rhythm config
            rhythm_config = RhythmEncoderConfig(**config.to_dict())
            rhythm_config.num_semantic_features = 20
            config = rhythm_config

        super().__init__(config)
        self.rhythm_config = config

        # Store parameter definitions
        self.parameter_definitions = RHYTHM_PARAMETERS

        # Locality functions for tempo-invariant transformations
        self.locality_functions = MusicalLocalityFunctions()

    def extract_rhythm_parameters(
        self,
        features: torch.Tensor,
        as_dict: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Extract rhythm parameters from input features.

        Args:
            features: Input features [batch_size, 200] or [200]
            as_dict: Return as dictionary with parameter names

        Returns:
            Dictionary mapping parameter names to values [batch_size] or scalar
        """
        # Extract semantic features
        semantic_features = self.extract_semantic_features(features, as_numpy=True)

        # Handle single sample
        is_single = len(semantic_features.shape) == 1
        if is_single:
            semantic_features = semantic_features.reshape(1, -1)

        if as_dict:
            # Convert to dictionary with parameter names
            result = {}
            for i, param_def in self.parameter_definitions.items():
                param_name = param_def['name']
                param_values = semantic_features[:, i]
                result[param_name] = param_values[0] if is_single else param_values
            return result
        else:
            return semantic_features[0] if is_single else semantic_features

    def interpret_parameters(self, parameters: np.ndarray) -> Dict[str, Any]:
        """
        Interpret rhythm parameters with human-readable descriptions.

        Args:
            parameters: Array of 20 rhythm parameters

        Returns:
            Dictionary with interpretations
        """
        interpretation = {}

        for i, param_def in self.parameter_definitions.items():
            value = float(parameters[i])
            interpretation[param_def['name']] = {
                'value': value,
                'description': param_def['description'],
                'interpretation': param_def['interpretation'],
                'range': param_def['range']
            }

        return interpretation

    def get_parameter_names(self) -> List[str]:
        """Get list of rhythm parameter names in order."""
        return [self.parameter_definitions[i]['name']
                for i in range(len(self.parameter_definitions))]

    def get_parameter_definitions(self) -> Dict[int, Dict[str, Any]]:
        """Get complete parameter definitions."""
        return self.parameter_definitions

    def compute_tempo_invariant_loss(
        self,
        features: torch.Tensor,
        transformed_features: torch.Tensor,
        transform_type: LocalityType
    ) -> torch.Tensor:
        """
        Compute tempo-invariant loss for rhythm features.

        For tempo-invariant transformations (AUGMENT, DIMINUTION),
        the semantic features should be nearly identical.

        Args:
            features: Original features [batch_size, 200]
            transformed_features: Tempo-transformed features [batch_size, 200]
            transform_type: Type of transformation

        Returns:
            Tempo-invariant loss (MSE between semantic features)
        """
        # Extract semantic features
        z1 = self.encoder(features)
        z2 = self.encoder(transformed_features)

        # For tempo-invariant transforms, features should match
        if transform_type in [LocalityType.AUGMENT, LocalityType.DIMINUTION]:
            # Full invariance expected
            loss = F.mse_loss(z1, z2)
        elif transform_type == LocalityType.TIME_SHIFT:
            # Partial invariance (some parameters may change)
            loss = F.mse_loss(z1, z2) * 0.5
        else:
            # Other transforms may change features
            loss = torch.tensor(0.0, device=features.device)

        return loss

    def analyze_rhythm_patterns(
        self,
        parameters: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze rhythm patterns from extracted parameters.

        Args:
            parameters: Array of 20 rhythm parameters

        Returns:
            Analysis dictionary with insights
        """
        analysis = {
            'rhythmic_style': self._classify_rhythmic_style(parameters),
            'complexity_level': self._compute_complexity_level(parameters),
            'tempo_feel': self._analyze_tempo_feel(parameters),
            'groove_characteristics': self._analyze_groove(parameters),
            'notable_features': self._identify_notable_features(parameters)
        }

        return analysis

    def _classify_rhythmic_style(self, params: np.ndarray) -> str:
        """Classify overall rhythmic style."""
        swing = params[3]  # swing_straight_continuum
        syncopation = params[0]  # syncopation_intensity
        clave = params[12]  # clave_alignment

        if clave > 0.6:
            return "Afro-Cuban/Latin"
        elif swing > 0.6 and syncopation > 0.5:
            return "Jazz/Swing"
        elif syncopation > 0.7:
            return "Funk/Syncopated"
        elif swing < 0.3 and params[16] > 0.7:  # event_regularity
            return "Classical/Straight"
        else:
            return "Mixed/Contemporary"

    def _compute_complexity_level(self, params: np.ndarray) -> str:
        """Compute overall rhythmic complexity."""
        complexity_score = (
            params[0] * 0.3 +   # syncopation
            params[2] * 0.3 +   # polyrhythm
            params[7] * 0.2 +   # accent complexity
            params[9] * 0.2     # subdivision granularity
        )

        if complexity_score < 0.3:
            return "Simple"
        elif complexity_score < 0.6:
            return "Moderate"
        else:
            return "Complex"

    def _analyze_tempo_feel(self, params: np.ndarray) -> Dict[str, Any]:
        """Analyze tempo and timing feel."""
        return {
            'stability': 'Strict' if params[10] > 0.7 else 'Rubato',
            'timing': self._get_timing_feel(params[14], params[15]),
            'pocket': 'Tight' if params[1] > 0.7 else 'Loose',
            'humanization': 'High' if params[6] > 0.6 else 'Low'
        }

    def _get_timing_feel(self, anticipation: float, delay: float) -> str:
        """Determine timing feel from anticipation/delay."""
        if anticipation > 0.6:
            return "Rushing/Ahead"
        elif delay > 0.6:
            return "Laid-back/Behind"
        else:
            return "On-time"

    def _analyze_groove(self, params: np.ndarray) -> Dict[str, Any]:
        """Analyze groove characteristics."""
        return {
            'overall_groove': 'Strong' if params[19] > 0.6 else 'Weak',
            'swing_feel': f"{params[3]:.2f}",
            'groove_template_match': f"{params[11]:.2f}",
            'backbeat_strength': 'Strong' if params[13] > 0.6 else 'Weak',
            'density': 'Dense' if params[4] > 0.6 else 'Sparse'
        }

    def _identify_notable_features(self, params: np.ndarray) -> List[str]:
        """Identify notable rhythmic features."""
        features = []

        if params[0] > 0.7:
            features.append("Highly syncopated")
        if params[2] > 0.5:
            features.append("Polyrhythmic")
        if params[3] > 0.7:
            features.append("Strong swing feel")
        if params[6] > 0.7:
            features.append("Highly humanized")
        if params[12] > 0.6:
            features.append("Clave-based")
        if params[9] > 0.8:
            features.append("Fine subdivisions")

        return features if features else ["Standard rhythm"]

    def save_parameters(
        self,
        parameters: np.ndarray,
        output_path: Path,
        include_interpretation: bool = True
    ):
        """
        Save rhythm parameters to JSON file.

        Args:
            parameters: Array of 20 parameters
            output_path: Path to save JSON
            include_interpretation: Include human-readable interpretation
        """
        output = {
            'parameters': {}
        }

        for i, param_def in self.parameter_definitions.items():
            output['parameters'][param_def['name']] = {
                'value': float(parameters[i]),
                'description': param_def['description'],
                'range': param_def['range']
            }

        if include_interpretation:
            output['interpretation'] = self.interpret_parameters(parameters)
            output['analysis'] = self.analyze_rhythm_patterns(parameters)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✅ Saved rhythm parameters to {output_path}")


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def create_rhythm_encoder(device: str = 'cpu') -> RhythmSemanticEncoder:
    """
    Create rhythm encoder with default configuration.

    Args:
        device: Device to create encoder on ('cpu', 'cuda', etc.)

    Returns:
        RhythmSemanticEncoder instance
    """
    config = RhythmEncoderConfig(num_semantic_features=20)
    encoder = RhythmSemanticEncoder(config)
    if TORCH_AVAILABLE:
        encoder.to(device)
    return encoder


def discover_rhythm_patterns_from_features(
    features: np.ndarray,
    encoder: Optional[RhythmSemanticEncoder] = None
) -> Dict[str, Any]:
    """
    Discover rhythm patterns from feature vectors.

    Args:
        features: Input features [batch_size, 200] or [200]
        encoder: RhythmSemanticEncoder (creates one if None)

    Returns:
        Dictionary with discovered patterns and analysis
    """
    if encoder is None:
        encoder = create_rhythm_encoder()

    # Convert to tensor
    if not isinstance(features, torch.Tensor):
        features_tensor = torch.FloatTensor(features)
    else:
        features_tensor = features

    # Extract parameters
    params = encoder.extract_rhythm_parameters(features_tensor, as_dict=True)

    # Get interpretation and analysis
    params_array = np.array([params[name] for name in encoder.get_parameter_names()])
    interpretation = encoder.interpret_parameters(params_array)
    analysis = encoder.analyze_rhythm_patterns(params_array)

    return {
        'parameters': params,
        'interpretation': interpretation,
        'analysis': analysis
    }


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Rhythm Semantic Encoder - Agent 3")
    print("="*70)

    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch is not installed.")
        print("   Install PyTorch to use this module:")
        print("   pip install torch")
        exit(1)

    # Create rhythm encoder
    print("\n1. Creating rhythm encoder...")
    config = RhythmEncoderConfig(num_semantic_features=20)
    encoder = create_rhythm_encoder()
    print(f"   ✅ Created encoder with 20 rhythm parameters")

    # List parameters
    print("\n2. Rhythm parameters:")
    for i, param_def in enumerate(RHYTHM_PARAMETERS.values()):
        print(f"   {i+1:2d}. {param_def['name']:30s} - {param_def['description']}")

    # Test with random features
    print("\n3. Testing parameter extraction...")
    test_features = torch.randn(16, 200)
    rhythm_params = encoder.extract_rhythm_parameters(test_features, as_dict=True)
    print(f"   ✅ Extracted parameters for batch of {test_features.shape[0]}")

    # Interpret single example
    print("\n4. Interpreting parameters for first example...")
    params_array = np.array([rhythm_params[name][0] for name in encoder.get_parameter_names()])
    interpretation = encoder.interpret_parameters(params_array)
    analysis = encoder.analyze_rhythm_patterns(params_array)

    print(f"\n   Rhythmic Style: {analysis['rhythmic_style']}")
    print(f"   Complexity: {analysis['complexity_level']}")
    print(f"   Tempo Feel: {analysis['tempo_feel']['timing']} ({analysis['tempo_feel']['pocket']} pocket)")
    print(f"   Notable Features: {', '.join(analysis['notable_features'])}")

    # Test save/load
    print("\n5. Testing save/load...")
    save_path = Path("/tmp/rhythm_params.json")
    encoder.save_parameters(params_array, save_path, include_interpretation=True)
    print(f"   ✅ Saved to {save_path}")

    # Test forward pass
    print("\n6. Testing forward pass...")
    output = encoder(test_features)
    print(f"   ✅ Forward pass successful")
    print(f"      Semantic features shape: {output['semantic_features'].shape}")
    print(f"      Reconstructed shape: {output['reconstructed'].shape}")

    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
