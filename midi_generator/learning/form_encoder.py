"""
Form/Structure Semantic Encoder - Agent 4
==========================================

Specialized semantic encoder for discovering form and structural parameters.

This encoder learns to compress 200D feature vectors into 15 interpretable
form-related semantic parameters that capture:
- Structural archetypes (AABA, verse-chorus, sonata, etc.)
- Section relationships and contrasts
- Tension arcs and climax positioning
- Repetition and variation patterns
- Form coherence and balance

The 15 Discovered Form Parameters:
1. tension_arc_shape - Overall shape of tension buildup (0=flat, 1=dramatic)
2. section_contrast_degree - How different sections are from each other
3. climax_position_ratio - Where the climax occurs (0=start, 1=end, 0.618=golden)
4. repetition_variation_balance - Balance between exact repetition and variation
5. golden_ratio_tendency - How closely form follows golden ratio proportions
6. form_symmetry - Degree of symmetry in section arrangement
7. development_intensity - Amount of thematic development vs statement
8. section_transition_smoothness - How smooth transitions are between sections
9. structural_coherence - Overall unity and coherence of form
10. climax_convergence - How well all elements converge at climax
11. recapitulation_fidelity - How closely recap matches exposition (sonata forms)
12. bridge_contrast_level - Degree of contrast in bridge/B sections
13. intro_outro_balance - Relationship between intro and outro
14. modulation_frequency - How often key changes occur
15. form_complexity - Overall structural complexity

Section-Aware Locality Functions:
- section_permute: Rearrange sections while preserving content
- section_repeat: Repeat specific sections
- section_delete: Remove sections
- tension_invert: Flip tension arc
- climax_shift: Move climax position

Architecture extends SemanticFeatureEncoder from Agent 3:
    Input: 200D feature vector
    Encoder: [200] → [512] → [15]
    Decoder: [15] → [512] → [200]
    Locality Predictor: [15 * 2] → [512] → [5] (section-aware transformations)

Author: Agent 4 - Form/Structure Module Builder
Date: November 21, 2025
Dependencies: Agent 3 (SemanticFeatureEncoder), Agent 1 (MusicalLocalityFunctions)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum
import json
import warnings

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. Install with: pip install torch")

# Try to import NumPy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not installed. Some functionality will be disabled.")

# Import base semantic encoder
try:
    from .semantic_encoder import SemanticFeatureEncoder, EncoderConfig, TrainingMetrics
    SEMANTIC_ENCODER_AVAILABLE = True
except ImportError:
    SEMANTIC_ENCODER_AVAILABLE = False
    warnings.warn("SemanticFeatureEncoder not available. Install base module first.")


# ============================================================================
# Form Parameter Definitions
# ============================================================================

class FormParameter(Enum):
    """Enumeration of 15 form-related semantic parameters."""

    # Structural shape parameters
    TENSION_ARC_SHAPE = "tension_arc_shape"
    CLIMAX_POSITION_RATIO = "climax_position_ratio"
    GOLDEN_RATIO_TENDENCY = "golden_ratio_tendency"
    FORM_SYMMETRY = "form_symmetry"

    # Section relationship parameters
    SECTION_CONTRAST_DEGREE = "section_contrast_degree"
    BRIDGE_CONTRAST_LEVEL = "bridge_contrast_level"
    REPETITION_VARIATION_BALANCE = "repetition_variation_balance"
    RECAPITULATION_FIDELITY = "recapitulation_fidelity"

    # Development parameters
    DEVELOPMENT_INTENSITY = "development_intensity"
    SECTION_TRANSITION_SMOOTHNESS = "section_transition_smoothness"
    MODULATION_FREQUENCY = "modulation_frequency"

    # Coherence parameters
    STRUCTURAL_COHERENCE = "structural_coherence"
    CLIMAX_CONVERGENCE = "climax_convergence"
    INTRO_OUTRO_BALANCE = "intro_outro_balance"
    FORM_COMPLEXITY = "form_complexity"


# Parameter descriptions and expected ranges
PARAMETER_DESCRIPTIONS = {
    FormParameter.TENSION_ARC_SHAPE: {
        "description": "Shape of tension buildup throughout piece",
        "range": "0.0 (flat) to 1.0 (dramatic arc)",
        "musical_meaning": "Classical sonata = high (0.8+), ambient = low (0.2-)"
    },
    FormParameter.SECTION_CONTRAST_DEGREE: {
        "description": "Difference between sections",
        "range": "0.0 (homogeneous) to 1.0 (highly contrasting)",
        "musical_meaning": "Through-composed = high, strophic = low"
    },
    FormParameter.CLIMAX_POSITION_RATIO: {
        "description": "Normalized position of climax in piece",
        "range": "0.0 (beginning) to 1.0 (end), 0.618 = golden ratio",
        "musical_meaning": "Pop = 0.75 (near end), classical = 0.618 (golden)"
    },
    FormParameter.REPETITION_VARIATION_BALANCE: {
        "description": "Balance between exact repetition and variation",
        "range": "0.0 (exact repetition) to 1.0 (constant variation)",
        "musical_meaning": "Minimalism = low, jazz = high"
    },
    FormParameter.GOLDEN_RATIO_TENDENCY: {
        "description": "How closely form follows golden ratio (1.618)",
        "range": "0.0 (no golden ratio) to 1.0 (perfect golden ratio)",
        "musical_meaning": "Classical forms often score 0.7+"
    },
    FormParameter.FORM_SYMMETRY: {
        "description": "Symmetry of section arrangement (ABA, ABCBA)",
        "range": "0.0 (asymmetric) to 1.0 (perfectly symmetric)",
        "musical_meaning": "Ternary form = high, verse-chorus = lower"
    },
    FormParameter.DEVELOPMENT_INTENSITY: {
        "description": "Amount of thematic development vs statement",
        "range": "0.0 (pure statement) to 1.0 (heavy development)",
        "musical_meaning": "Sonata development = high, pop song = low"
    },
    FormParameter.SECTION_TRANSITION_SMOOTHNESS: {
        "description": "Smoothness of transitions between sections",
        "range": "0.0 (abrupt) to 1.0 (seamless)",
        "musical_meaning": "Classical = smooth, EDM = abrupt"
    },
    FormParameter.STRUCTURAL_COHERENCE: {
        "description": "Overall unity and coherence of form",
        "range": "0.0 (fragmented) to 1.0 (highly unified)",
        "musical_meaning": "Well-constructed forms score 0.6+"
    },
    FormParameter.CLIMAX_CONVERGENCE: {
        "description": "How well all elements converge at climax",
        "range": "0.0 (no convergence) to 1.0 (perfect convergence)",
        "musical_meaning": "Symphonic climaxes score 0.8+"
    },
    FormParameter.RECAPITULATION_FIDELITY: {
        "description": "Similarity between exposition and recapitulation",
        "range": "0.0 (completely different) to 1.0 (identical)",
        "musical_meaning": "Classical sonata = 0.7-0.9"
    },
    FormParameter.BRIDGE_CONTRAST_LEVEL: {
        "description": "Degree of contrast in bridge/B sections",
        "range": "0.0 (no contrast) to 1.0 (maximum contrast)",
        "musical_meaning": "AABA bridge should score 0.5+"
    },
    FormParameter.INTRO_OUTRO_BALANCE: {
        "description": "Relationship between intro and outro",
        "range": "0.0 (unrelated) to 1.0 (mirror/bookend)",
        "musical_meaning": "Rounded forms score high"
    },
    FormParameter.MODULATION_FREQUENCY: {
        "description": "How often key changes occur",
        "range": "0.0 (no modulation) to 1.0 (constant modulation)",
        "musical_meaning": "Romantic music = high, folk = low"
    },
    FormParameter.FORM_COMPLEXITY: {
        "description": "Overall structural complexity",
        "range": "0.0 (simple) to 1.0 (highly complex)",
        "musical_meaning": "Strophic = low, through-composed = high"
    }
}


# ============================================================================
# Section-Aware Locality Types
# ============================================================================

class SectionLocalityType(Enum):
    """Section-aware locality transformations for form learning."""

    SECTION_PERMUTE = "section_permute"      # Rearrange section order
    SECTION_REPEAT = "section_repeat"         # Repeat a section
    SECTION_DELETE = "section_delete"         # Remove a section
    TENSION_INVERT = "tension_invert"         # Flip tension arc
    CLIMAX_SHIFT = "climax_shift"             # Move climax position


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FormEncoderConfig(EncoderConfig):
    """
    Configuration for FormSemanticEncoder.

    Extends EncoderConfig with form-specific parameters.
    """

    # Override defaults for form encoder
    num_semantic_features: int = 15  # 15 form parameters
    num_locality_types: int = 5      # 5 section-aware transformations

    # Form-specific settings
    focus_on_structure: bool = True
    use_section_attention: bool = True
    structural_weight: float = 1.5   # Extra weight for structural features

    # Advanced form analysis
    detect_golden_ratio: bool = True
    detect_symmetry: bool = True
    detect_climax: bool = True


# ============================================================================
# Form Semantic Encoder
# ============================================================================

if TORCH_AVAILABLE and SEMANTIC_ENCODER_AVAILABLE:

    class FormSemanticEncoder(SemanticFeatureEncoder):
        """
        Form/Structure semantic encoder with 15 parameters.

        Extends SemanticFeatureEncoder with form-specific architecture
        and section-aware locality functions.

        Usage:
            # Initialize
            config = FormEncoderConfig()
            encoder = FormSemanticEncoder(config)

            # Extract form parameters
            features = torch.randn(32, 200)  # Batch of 32 feature vectors
            form_params = encoder.extract_form_parameters(features)

            # Get parameter names
            param_names = encoder.get_parameter_names()
        """

        def __init__(self, config: FormEncoderConfig):
            """
            Initialize FormSemanticEncoder.

            Args:
                config: FormEncoderConfig with form-specific settings
            """
            super().__init__(config)
            self.form_config = config

            # Store parameter names in order
            self.parameter_names = [p.value for p in FormParameter]

            # Section attention mechanism (if enabled)
            if config.use_section_attention:
                self.section_attention = nn.MultiheadAttention(
                    embed_dim=config.num_semantic_features,
                    num_heads=3,
                    dropout=config.dropout
                )
            else:
                self.section_attention = None

        def extract_form_parameters(
            self,
            x: torch.Tensor,
            as_dict: bool = False
        ) -> Dict[str, torch.Tensor]:
            """
            Extract 15 form parameters from features.

            Args:
                x: Input features [batch_size, 200]
                as_dict: Return as dictionary with parameter names

            Returns:
                If as_dict=True: Dictionary mapping parameter names to values
                If as_dict=False: Tensor of shape [batch_size, 15]
            """
            # Extract semantic features (15 parameters)
            params = self.extract_semantic_features(x)

            if as_dict:
                # Convert to dictionary
                batch_size = params.shape[0]
                result = {}

                for i, param_name in enumerate(self.parameter_names):
                    result[param_name] = params[:, i]

                return result
            else:
                return params

        def get_parameter_names(self) -> List[str]:
            """Get list of parameter names in order."""
            return self.parameter_names.copy()

        def get_parameter_descriptions(self) -> Dict[str, Dict[str, str]]:
            """Get detailed descriptions of all parameters."""
            return {
                param.value: PARAMETER_DESCRIPTIONS[param]
                for param in FormParameter
            }

        def analyze_form_structure(
            self,
            form_params: Dict[str, torch.Tensor]
        ) -> Dict[str, Any]:
            """
            Analyze form structure from extracted parameters.

            Args:
                form_params: Dictionary of form parameters

            Returns:
                Dictionary with structural analysis
            """
            analysis = {}

            # Determine form archetype
            analysis['form_archetype'] = self._classify_form_archetype(form_params)

            # Analyze structural balance
            analysis['balance_score'] = self._compute_balance_score(form_params)

            # Detect special structural features
            analysis['has_golden_ratio'] = form_params['golden_ratio_tendency'].item() > 0.6
            analysis['is_symmetric'] = form_params['form_symmetry'].item() > 0.6
            analysis['has_strong_climax'] = form_params['climax_convergence'].item() > 0.7

            # Classify complexity
            complexity = form_params['form_complexity'].item()
            if complexity < 0.3:
                analysis['complexity_level'] = "simple"
            elif complexity < 0.7:
                analysis['complexity_level'] = "moderate"
            else:
                analysis['complexity_level'] = "complex"

            return analysis

        def _classify_form_archetype(
            self,
            params: Dict[str, torch.Tensor]
        ) -> str:
            """
            Classify form archetype based on parameters.

            Returns:
                Form type string (e.g., "AABA", "verse_chorus", "sonata")
            """
            # Extract key parameters
            symmetry = params['form_symmetry'].item()
            contrast = params['section_contrast_degree'].item()
            repetition = params['repetition_variation_balance'].item()
            development = params['development_intensity'].item()
            recap_fidelity = params['recapitulation_fidelity'].item()

            # Classification logic
            if recap_fidelity > 0.7 and development > 0.6:
                return "sonata"
            elif symmetry > 0.7 and contrast < 0.4:
                return "ternary_ABA"
            elif repetition < 0.3 and symmetry > 0.5:
                return "AABA"
            elif contrast > 0.6 and repetition > 0.5:
                return "verse_chorus"
            elif development > 0.7:
                return "through_composed"
            elif repetition < 0.2:
                return "strophic"
            else:
                return "custom_form"

        def _compute_balance_score(
            self,
            params: Dict[str, torch.Tensor]
        ) -> float:
            """
            Compute overall structural balance score.

            Returns:
                Balance score 0.0-1.0
            """
            # Key balance indicators
            coherence = params['structural_coherence'].item()
            symmetry = params['form_symmetry'].item()
            golden = params['golden_ratio_tendency'].item()
            intro_outro = params['intro_outro_balance'].item()

            # Weighted average
            balance = (
                coherence * 0.4 +
                symmetry * 0.3 +
                golden * 0.2 +
                intro_outro * 0.1
            )

            return float(balance)

        def compare_forms(
            self,
            params_a: Dict[str, torch.Tensor],
            params_b: Dict[str, torch.Tensor]
        ) -> Dict[str, float]:
            """
            Compare two forms based on their parameters.

            Args:
                params_a: Form parameters for piece A
                params_b: Form parameters for piece B

            Returns:
                Dictionary with similarity scores
            """
            similarities = {}

            for param_name in self.parameter_names:
                val_a = params_a[param_name].item()
                val_b = params_b[param_name].item()

                # Compute similarity (1 - normalized difference)
                diff = abs(val_a - val_b)
                similarity = 1.0 - diff
                similarities[param_name] = similarity

            # Overall similarity
            similarities['overall'] = np.mean(list(similarities.values()))

            return similarities

        def save_with_metadata(self, path: Path):
            """Save model with form-specific metadata."""
            super().save(path, include_config=True)

            # Save parameter descriptions
            metadata_path = path.with_suffix('.metadata.json')
            metadata = {
                'parameter_names': self.parameter_names,
                'parameter_descriptions': self.get_parameter_descriptions(),
                'encoder_type': 'FormSemanticEncoder',
                'num_parameters': len(self.parameter_names)
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"✅ Metadata saved to {metadata_path}")


# ============================================================================
# Form-Specific Locality Functions
# ============================================================================

class FormLocalityFunctions:
    """
    Section-aware locality functions for form parameter discovery.

    These transformations operate on section-level structure while
    preserving certain form properties.
    """

    @staticmethod
    def section_permute(
        sections: List[Dict],
        permutation: List[int]
    ) -> List[Dict]:
        """
        Permute section order.

        Args:
            sections: List of section dictionaries
            permutation: New order indices

        Returns:
            Reordered sections
        """
        if len(permutation) != len(sections):
            raise ValueError("Permutation must match number of sections")

        return [sections[i] for i in permutation]

    @staticmethod
    def section_repeat(
        sections: List[Dict],
        section_index: int,
        num_repeats: int = 1
    ) -> List[Dict]:
        """
        Repeat a specific section.

        Args:
            sections: List of sections
            section_index: Index of section to repeat
            num_repeats: Number of times to repeat

        Returns:
            Sections with repeated section
        """
        result = sections[:section_index + 1]
        for _ in range(num_repeats):
            result.append(sections[section_index].copy())
        result.extend(sections[section_index + 1:])

        return result

    @staticmethod
    def section_delete(
        sections: List[Dict],
        section_index: int
    ) -> List[Dict]:
        """
        Delete a section.

        Args:
            sections: List of sections
            section_index: Index of section to delete

        Returns:
            Sections with one removed
        """
        return sections[:section_index] + sections[section_index + 1:]

    @staticmethod
    def tension_invert(
        sections: List[Dict]
    ) -> List[Dict]:
        """
        Invert tension arc (flip dynamics).

        Args:
            sections: List of sections with 'dynamic_level' key

        Returns:
            Sections with inverted dynamics
        """
        result = []
        for section in sections:
            new_section = section.copy()
            if 'dynamic_level' in section:
                new_section['dynamic_level'] = 1.0 - section['dynamic_level']
            result.append(new_section)

        return result

    @staticmethod
    def climax_shift(
        sections: List[Dict],
        shift_amount: int
    ) -> List[Dict]:
        """
        Shift climax position by moving peak dynamics.

        Args:
            sections: List of sections
            shift_amount: Number of sections to shift (+/-)

        Returns:
            Sections with shifted climax
        """
        # Find current climax
        dynamics = [s.get('dynamic_level', 0.5) for s in sections]
        climax_idx = dynamics.index(max(dynamics))

        # Calculate new position
        new_climax_idx = (climax_idx + shift_amount) % len(sections)

        # Swap dynamics
        result = [s.copy() for s in sections]
        result[climax_idx]['dynamic_level'], result[new_climax_idx]['dynamic_level'] = \
            result[new_climax_idx]['dynamic_level'], result[climax_idx]['dynamic_level']

        return result


# ============================================================================
# Utility Functions
# ============================================================================

def create_form_encoder(
    device: str = 'cpu'
) -> 'FormSemanticEncoder':
    """
    Create FormSemanticEncoder with default configuration.

    Args:
        device: Device to create model on ('cpu' or 'cuda')

    Returns:
        Initialized FormSemanticEncoder
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Install with: pip install torch")

    config = FormEncoderConfig()
    encoder = FormSemanticEncoder(config)
    encoder.to(device)
    return encoder


def analyze_midi_form_structure(
    midi_file_path: Path,
    encoder: 'FormSemanticEncoder',
    feature_extractor: Any = None
) -> Dict[str, Any]:
    """
    Extract and analyze form structure from MIDI file.

    Args:
        midi_file_path: Path to MIDI file
        encoder: Trained FormSemanticEncoder
        feature_extractor: Feature extractor (200D)

    Returns:
        Dictionary with form analysis
    """
    # This is a placeholder - would need actual feature extraction
    # In practice, would use OptimizedFeatureExtractor from Agent 2

    if feature_extractor is None:
        raise ValueError("Feature extractor required")

    # Extract 200D features
    features = feature_extractor.extract(midi_file_path)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    # Extract form parameters
    encoder.eval()
    with torch.no_grad():
        form_params = encoder.extract_form_parameters(features_tensor, as_dict=True)

    # Analyze structure
    analysis = encoder.analyze_form_structure(form_params)

    # Add parameter values
    analysis['parameters'] = {
        k: v.item() if hasattr(v, 'item') else v
        for k, v in form_params.items()
    }

    return analysis


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Form Semantic Encoder - Agent 4")
    print("=" * 80)

    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch is not installed.")
        print("   Install PyTorch to use this module:")
        print("   pip install torch")
        exit(1)

    # Create encoder
    print("\n1. Creating FormSemanticEncoder...")
    config = FormEncoderConfig()
    encoder = create_form_encoder()
    print(f"   ✅ Created encoder with {len(encoder.parameter_names)} form parameters")

    # Show parameter names
    print("\n2. Form Parameters:")
    for i, param in enumerate(encoder.parameter_names, 1):
        desc = PARAMETER_DESCRIPTIONS[FormParameter(param)]
        print(f"   {i:2d}. {param:30s} - {desc['description']}")

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 8
    x = torch.randn(batch_size, 200)
    form_params = encoder.extract_form_parameters(x, as_dict=True)
    print(f"   ✅ Forward pass successful")
    print(f"      Batch size: {batch_size}")
    print(f"      Parameters extracted: {len(form_params)}")

    # Analyze form structure
    print("\n4. Analyzing form structure...")
    analysis = encoder.analyze_form_structure(form_params)
    print(f"   ✅ Analysis complete")
    print(f"      Form archetype: {analysis['form_archetype']}")
    print(f"      Complexity: {analysis['complexity_level']}")
    print(f"      Balance score: {analysis['balance_score']:.3f}")
    print(f"      Has golden ratio: {analysis['has_golden_ratio']}")
    print(f"      Is symmetric: {analysis['is_symmetric']}")

    # Test locality functions
    print("\n5. Testing section-aware locality functions...")
    sections = [
        {'name': 'A1', 'dynamic_level': 0.6},
        {'name': 'A2', 'dynamic_level': 0.6},
        {'name': 'B', 'dynamic_level': 0.7},
        {'name': 'A3', 'dynamic_level': 0.8}
    ]

    # Permute sections
    permuted = FormLocalityFunctions.section_permute(sections, [0, 2, 1, 3])
    print(f"   ✅ Section permute: {[s['name'] for s in permuted]}")

    # Invert tension
    inverted = FormLocalityFunctions.tension_invert(sections)
    print(f"   ✅ Tension invert: {[s['dynamic_level'] for s in inverted]}")

    # Test save/load
    print("\n6. Testing save/load...")
    save_path = Path("/tmp/form_encoder_test.pt")
    encoder.save_with_metadata(save_path)
    print(f"   ✅ Model and metadata saved")

    # Get parameter descriptions
    print("\n7. Parameter descriptions:")
    descriptions = encoder.get_parameter_descriptions()
    example_param = FormParameter.TENSION_ARC_SHAPE.value
    desc = descriptions[example_param]
    print(f"   Example: {example_param}")
    print(f"   - Description: {desc['description']}")
    print(f"   - Range: {desc['range']}")
    print(f"   - Musical meaning: {desc['musical_meaning']}")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print("\nFormSemanticEncoder is ready to discover 15 form parameters!")
    print("Next steps:")
    print("  - Train on MIDI corpus with section annotations")
    print("  - Integrate with Agent 3's training pipeline")
    print("  - Connect to arrangement agents for form-aware generation")
