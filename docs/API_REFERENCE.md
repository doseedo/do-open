# Semantic Feature Discovery API Reference

**Version:** 1.0.0
**Last Updated:** November 2025
**Agent:** 10 - Documentation & Examples

---

## Table of Contents

1. [Overview](#overview)
2. [Agent 1: Musical Locality API](#agent-1-musical-locality-api)
3. [Agent 2: Semantic Features API](#agent-2-semantic-features-api)
4. [Agent 3: Neural Architecture API](#agent-3-neural-architecture-api)
5. [Agent 4: Gap Dataset API](#agent-4-gap-dataset-api)
6. [Agent 5: Training API](#agent-5-training-api)
7. [Agent 6: Feature Interpretation API](#agent-6-feature-interpretation-api)
8. [Agent 7: Integration Pipeline API](#agent-7-integration-pipeline-api)
9. [Agent 8: Validation API](#agent-8-validation-api)
10. [Agent 9: Evaluation API](#agent-9-evaluation-api)
11. [Utilities](#utilities)
12. [Type Definitions](#type-definitions)

---

## Overview

This document provides comprehensive API documentation for all modules in the Semantic Feature Discovery system.

### Quick Navigation

**High-Level APIs** (most users):
- [SemanticDiscoveryPipeline](#semanticdiscoverypipeline) - End-to-end discovery
- [FeatureInterpreter](#featureinterpreter) - Interpret learned features
- [SemanticFeatureEvaluator](#semanticfeatureevaluator) - Evaluate quality

**Low-Level APIs** (advanced users):
- [MusicalLocalityFunctions](#musicallocalityfunctions) - Apply transformations
- [SemanticFeatureEncoder](#semanticfeatureencoder) - Neural network
- [GapDiscoveryTrainer](#gapdiscoverytrainer) - Custom training

### Import Paths

```python
# High-level
from midi_generator.learning.semantic_discovery_pipeline import SemanticDiscoveryPipeline
from midi_generator.learning.feature_interpreter import FeatureInterpreter
from midi_generator.evaluation.semantic_evaluation import SemanticFeatureEvaluator

# Low-level
from midi_generator.learning.musical_locality import MusicalLocalityFunctions, LocalityType
from midi_generator.learning.semantic_features import SemanticFeature, SemanticFeatureBank
from midi_generator.learning.semantic_encoder import SemanticFeatureEncoder
from midi_generator.learning.gap_dataset import GapDataset, GapAnalyzer
from midi_generator.learning.gap_discovery_trainer import GapDiscoveryTrainer, TrainingConfig
from midi_generator.learning.semantic_constraints import SemanticFeatureValidator
```

---

## Agent 1: Musical Locality API

### Module: `midi_generator.learning.musical_locality`

#### MusicalLocalityFunctions

Main class for applying musical transformations.

```python
class MusicalLocalityFunctions:
    """
    Musical transformation functions for locality testing.

    Provides 12 types of transformations that preserve musical structure
    while varying specific properties.
    """
```

**Constructor:**

```python
def __init__(self):
    """Initialize locality functions"""
```

**Methods:**

##### apply_transformation

```python
def apply_transformation(
    self,
    midi_file: Path,
    transform_type: LocalityType,
    parameters: Dict[str, Any]
) -> Path:
    """
    Apply a musical transformation to a MIDI file.

    Args:
        midi_file: Path to input MIDI file
        transform_type: Type of transformation (from LocalityType enum)
        parameters: Transformation-specific parameters

    Returns:
        Path to transformed MIDI file (temporary file)

    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If midi_file doesn't exist

    Example:
        >>> locality = MusicalLocalityFunctions()
        >>> transformed = locality.apply_transformation(
        ...     Path("input.mid"),
        ...     LocalityType.TRANSPOSE,
        ...     {"semitones": 5}
        ... )
        >>> print(transformed)
        Path('/tmp/transformed_abc123.mid')
    """
```

##### transpose

```python
def transpose(
    self,
    midi_file: Path,
    semitones: int
) -> Path:
    """
    Transpose all pitches by N semitones.

    Args:
        midi_file: Input MIDI file
        semitones: Number of semitones to transpose (-12 to +12)

    Returns:
        Path to transposed MIDI file

    Example:
        >>> transposed = locality.transpose(Path("song.mid"), semitones=3)
    """
```

##### invert

```python
def invert(
    self,
    midi_file: Path,
    pivot: int = 60,
    mode: str = "exact"
) -> Path:
    """
    Invert intervals around a pivot pitch.

    Args:
        midi_file: Input MIDI file
        pivot: Pivot MIDI note number (default: 60 = middle C)
        mode: "exact" (chromatic) or "diatonic" (within key)

    Returns:
        Path to inverted MIDI file

    Example:
        >>> inverted = locality.invert(
        ...     Path("melody.mid"),
        ...     pivot=60,
        ...     mode="exact"
        ... )
    """
```

##### augment / diminish

```python
def augment(
    self,
    midi_file: Path,
    factor: float
) -> Path:
    """
    Increase interval sizes by a factor.

    Args:
        midi_file: Input MIDI file
        factor: Multiplication factor (1.0 to 2.0)
            1.0 = no change
            1.5 = intervals 50% larger
            2.0 = intervals doubled

    Returns:
        Path to augmented MIDI file

    Example:
        >>> augmented = locality.augment(Path("theme.mid"), factor=1.5)
    """

def diminish(
    self,
    midi_file: Path,
    factor: float
) -> Path:
    """
    Decrease interval sizes by a factor.

    Args:
        midi_file: Input MIDI file
        factor: Division factor (0.5 to 1.0)
            1.0 = no change
            0.75 = intervals 25% smaller
            0.5 = intervals halved

    Returns:
        Path to diminished MIDI file

    Example:
        >>> diminished = locality.diminish(Path("theme.mid"), factor=0.75)
    """
```

##### retrograde

```python
def retrograde(
    self,
    midi_file: Path
) -> Path:
    """
    Reverse the time order of all events.

    Args:
        midi_file: Input MIDI file

    Returns:
        Path to retrograde MIDI file

    Example:
        >>> backwards = locality.retrograde(Path("forward.mid"))
    """
```

##### time_shift

```python
def time_shift(
    self,
    midi_file: Path,
    beats: float
) -> Path:
    """
    Shift all events in time.

    Args:
        midi_file: Input MIDI file
        beats: Number of beats to shift (can be negative)

    Returns:
        Path to time-shifted MIDI file

    Example:
        >>> shifted = locality.time_shift(Path("song.mid"), beats=1.0)
    """
```

##### velocity_scale

```python
def velocity_scale(
    self,
    midi_file: Path,
    factor: float,
    min_vel: int = 1,
    max_vel: int = 127
) -> Path:
    """
    Scale all velocities by a factor.

    Args:
        midi_file: Input MIDI file
        factor: Velocity multiplication factor (0.5 to 1.5)
        min_vel: Minimum allowed velocity (1 to 127)
        max_vel: Maximum allowed velocity (1 to 127)

    Returns:
        Path to velocity-scaled MIDI file

    Example:
        >>> quieter = locality.velocity_scale(
        ...     Path("loud.mid"),
        ...     factor=0.7,
        ...     min_vel=10
        ... )
    """
```

##### chord_substitution

```python
def chord_substitution(
    self,
    midi_file: Path,
    substitution_type: str = "tritone"
) -> Path:
    """
    Substitute chords with functional equivalents.

    Args:
        midi_file: Input MIDI file
        substitution_type: Type of substitution
            - "tritone": Tritone substitution (V7 → bII7)
            - "relative": Relative major/minor (I ↔ vi)
            - "parallel": Parallel major/minor (I ↔ i)

    Returns:
        Path to chord-substituted MIDI file

    Example:
        >>> subbed = locality.chord_substitution(
        ...     Path("progression.mid"),
        ...     substitution_type="tritone"
        ... )
    """
```

##### octave_shift

```python
def octave_shift(
    self,
    midi_file: Path,
    octaves: int
) -> Path:
    """
    Shift all pitches by octaves.

    Args:
        midi_file: Input MIDI file
        octaves: Number of octaves to shift (-2 to +2)

    Returns:
        Path to octave-shifted MIDI file

    Example:
        >>> higher = locality.octave_shift(Path("bass.mid"), octaves=1)
    """
```

##### rhythm_augment

```python
def rhythm_augment(
    self,
    midi_file: Path,
    factor: float
) -> Path:
    """
    Stretch or compress rhythmic durations.

    Args:
        midi_file: Input MIDI file
        factor: Time stretching factor (0.5 to 2.0)
            0.5 = twice as fast
            1.0 = no change
            2.0 = twice as slow

    Returns:
        Path to rhythm-augmented MIDI file

    Example:
        >>> slower = locality.rhythm_augment(Path("fast.mid"), factor=2.0)
    """
```

##### harmonic_shift

```python
def harmonic_shift(
    self,
    midi_file: Path,
    shift: int
) -> Path:
    """
    Shift harmonic context by scale degrees.

    Args:
        midi_file: Input MIDI file
        shift: Number of scale degrees to shift (e.g., 5 for I → vi)

    Returns:
        Path to harmonically-shifted MIDI file

    Example:
        >>> relative_minor = locality.harmonic_shift(
        ...     Path("major.mid"),
        ...     shift=5  # I → vi (relative minor)
        ... )
    """
```

##### modal_shift

```python
def modal_shift(
    self,
    midi_file: Path,
    source_mode: str,
    target_mode: str
) -> Path:
    """
    Change mode (major, minor, Dorian, etc.).

    Args:
        midi_file: Input MIDI file
        source_mode: Current mode ("major", "minor", "dorian", "phrygian",
                     "lydian", "mixolydian", "aeolian", "locrian")
        target_mode: Target mode (same options as source_mode)

    Returns:
        Path to mode-shifted MIDI file

    Example:
        >>> minor = locality.modal_shift(
        ...     Path("major_song.mid"),
        ...     source_mode="major",
        ...     target_mode="minor"
        ... )
    """
```

##### get_transformation_inverse

```python
def get_transformation_inverse(
    self,
    transform_type: LocalityType,
    parameters: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Get inverse transformation parameters.

    For reversible transformations, returns parameters that undo the transformation.

    Args:
        transform_type: Type of transformation
        parameters: Original transformation parameters

    Returns:
        Inverse parameters (or None if not invertible)

    Example:
        >>> # Transpose up 5 semitones
        >>> forward_params = {"semitones": 5}
        >>> # Get inverse (transpose down 5)
        >>> inverse_params = locality.get_transformation_inverse(
        ...     LocalityType.TRANSPOSE,
        ...     forward_params
        ... )
        >>> print(inverse_params)
        {'semitones': -5}
    """
```

#### LocalityType Enum

```python
class LocalityType(Enum):
    """Types of musical transformations"""
    TRANSPOSE = "transpose"
    INVERT = "invert"
    AUGMENT = "augment"
    DIMINISH = "diminish"
    RETROGRADE = "retrograde"
    TIME_SHIFT = "time_shift"
    VELOCITY_SCALE = "velocity_scale"
    CHORD_SUBSTITUTION = "chord_sub"
    OCTAVE_SHIFT = "octave_shift"
    RHYTHM_AUGMENT = "rhythm_augment"
    HARMONIC_SHIFT = "harmonic_shift"
    MODAL_SHIFT = "modal_shift"
```

#### MusicalTransform Dataclass

```python
@dataclass
class MusicalTransform:
    """
    Represents a musical transformation.

    Attributes:
        transform_type: Type of transformation
        parameters: Transformation parameters
        apply: Function to apply transformation
        inverse: Function to compute inverse (if exists)
    """
    transform_type: LocalityType
    parameters: Dict[str, Any]
    apply: Callable[[Path], Path]
    inverse: Optional[Callable[[Path], Path]] = None
```

---

## Agent 2: Semantic Features API

### Module: `midi_generator.learning.semantic_features`

#### SemanticFeature Dataclass

```python
@dataclass
class SemanticFeature:
    """
    A single discovered semantic feature.

    Attributes:
        feature_id: Unique integer identifier
        name: Human-readable name (e.g., "swing_ratio")
        modality: Musical modality (RHYTHM, HARMONY, etc.)
        activation_values: Feature values across corpus (num_files,)
        weights: Encoder weights for this feature (200,)
        locality_profile: How feature responds to transformations
        interpretation: Human-readable description
        confidence: Interpretation confidence (0.0 to 1.0)
        extraction_function: Function to extract from MIDI
        discovered_at: Timestamp of discovery
        validation_status: "passed", "failed", or "pending"
    """
```

**Constructor:**

```python
def __init__(
    self,
    feature_id: int,
    name: str,
    modality: FeatureModality,
    activation_values: np.ndarray,
    weights: np.ndarray,
    locality_profile: LocalityProfile,
    interpretation: str = "",
    confidence: float = 0.0,
    extraction_function: Optional[Callable] = None,
    discovered_at: Optional[datetime] = None,
    validation_status: str = "pending"
):
```

**Methods:**

##### generate_variants

```python
def generate_variants(
    self,
    locality_type: LocalityType,
    num_variants: int = 10
) -> List[np.ndarray]:
    """
    Generate feature variants by applying transformations.

    Used to test locality consistency.

    Args:
        locality_type: Type of transformation to apply
        num_variants: Number of variants to generate

    Returns:
        List of feature activation arrays

    Example:
        >>> feature = SemanticFeature(...)
        >>> variants = feature.generate_variants(
        ...     LocalityType.TRANSPOSE,
        ...     num_variants=10
        ... )
        >>> print(len(variants))
        10
    """
```

##### matches_pattern

```python
def matches_pattern(
    self,
    pattern: MusicalTestPattern
) -> float:
    """
    Test if feature matches a known musical pattern.

    Args:
        pattern: Musical test pattern

    Returns:
        Correlation strength (0.0 to 1.0)

    Example:
        >>> from midi_generator.learning.feature_interpreter import (
        ...     MusicalTestPatterns
        ... )
        >>> patterns = MusicalTestPatterns()
        >>> swing_pattern = patterns.get_pattern("swing")
        >>> correlation = feature.matches_pattern(swing_pattern)
        >>> print(f"Swing correlation: {correlation:.2f}")
        Swing correlation: 0.87
    """
```

##### get_activation_strength

```python
def get_activation_strength(
    self,
    percentile: float = 90
) -> float:
    """
    Get characteristic activation strength.

    Args:
        percentile: Percentile to use (0 to 100)

    Returns:
        Activation strength at given percentile

    Example:
        >>> strength = feature.get_activation_strength(percentile=90)
        >>> print(f"90th percentile activation: {strength:.3f}")
        90th percentile activation: 0.842
    """
```

#### FeatureModality Enum

```python
class FeatureModality(Enum):
    """Musical modalities for features"""
    RHYTHM = "rhythm"
    HARMONY = "harmony"
    MELODY = "melody"
    TIMBRE = "timbre"
    DYNAMICS = "dynamics"
    STRUCTURE = "structure"
    TEXTURE = "texture"
    MIXED = "mixed"
    UNKNOWN = "unknown"
```

#### SemanticFeatureBank

```python
class SemanticFeatureBank:
    """
    Collection of discovered semantic features.

    Provides feature storage, retrieval, analysis, and persistence.
    """
```

**Constructor:**

```python
def __init__(self):
    """Initialize empty feature bank"""
```

**Attributes:**

```python
features: List[SemanticFeature]  # All features
metadata: Dict[str, Any]  # Bank-level metadata
```

**Methods:**

##### add_feature

```python
def add_feature(self, feature: SemanticFeature):
    """
    Add a feature to the bank.

    Args:
        feature: SemanticFeature to add

    Raises:
        ValueError: If feature_id already exists

    Example:
        >>> bank = SemanticFeatureBank()
        >>> feature = SemanticFeature(feature_id=0, name="swing", ...)
        >>> bank.add_feature(feature)
    """
```

##### get_feature

```python
def get_feature(
    self,
    feature_id: Optional[int] = None,
    name: Optional[str] = None
) -> Optional[SemanticFeature]:
    """
    Retrieve feature by ID or name.

    Args:
        feature_id: Feature ID (mutually exclusive with name)
        name: Feature name (mutually exclusive with feature_id)

    Returns:
        SemanticFeature or None if not found

    Raises:
        ValueError: If neither or both arguments provided

    Example:
        >>> feature = bank.get_feature(name="swing_ratio")
        >>> print(feature.interpretation)
        Degree of swing timing (0=straight, 1=full swing)
    """
```

##### get_activations

```python
def get_activations(
    self,
    midi_file: Path
) -> Dict[str, float]:
    """
    Get all feature activations for a MIDI file.

    Args:
        midi_file: Path to MIDI file

    Returns:
        Dictionary mapping feature names to activation values

    Example:
        >>> activations = bank.get_activations(Path("song.mid"))
        >>> print(activations)
        {
            'swing_ratio': 0.64,
            'chord_density': 0.82,
            'rhythmic_complexity': 0.45,
            ...
        }
    """
```

##### get_top_k_features

```python
def get_top_k_features(
    self,
    k: int,
    modality: Optional[FeatureModality] = None,
    sort_by: str = "activation_strength"
) -> List[SemanticFeature]:
    """
    Get top-k features by some criterion.

    Args:
        k: Number of features to return
        modality: Filter by modality (optional)
        sort_by: Sorting criterion
            - "activation_strength": By activation magnitude
            - "confidence": By interpretation confidence
            - "interpretability": By how well-interpreted

    Returns:
        List of top-k SemanticFeatures

    Example:
        >>> top_rhythm = bank.get_top_k_features(
        ...     k=5,
        ...     modality=FeatureModality.RHYTHM,
        ...     sort_by="activation_strength"
        ... )
        >>> for feat in top_rhythm:
        ...     print(f"{feat.name}: {feat.get_activation_strength():.2f}")
        swing_ratio: 0.87
        syncopation_level: 0.76
        polyrhythm_density: 0.68
        ...
    """
```

##### compute_similarity_matrix

```python
def compute_similarity_matrix(self) -> np.ndarray:
    """
    Compute pairwise similarity between all features.

    Returns:
        (num_features, num_features) similarity matrix

    Example:
        >>> similarity = bank.compute_similarity_matrix()
        >>> print(similarity.shape)
        (25, 25)
        >>> # Find features most similar to feature 0
        >>> similar_to_0 = np.argsort(similarity[0])[::-1][1:6]
        >>> print([bank.features[i].name for i in similar_to_0])
        ['pitch_center', 'key_clarity', 'tonal_stability', ...]
    """
```

##### find_redundant_features

```python
def find_redundant_features(
    self,
    threshold: float = 0.9
) -> List[Tuple[int, int]]:
    """
    Find pairs of features that are highly correlated (redundant).

    Args:
        threshold: Correlation threshold (0.0 to 1.0)

    Returns:
        List of (feature_id_1, feature_id_2) pairs

    Example:
        >>> redundant = bank.find_redundant_features(threshold=0.95)
        >>> for id1, id2 in redundant:
        ...     feat1 = bank.get_feature(feature_id=id1)
        ...     feat2 = bank.get_feature(feature_id=id2)
        ...     print(f"Redundant: {feat1.name} ↔ {feat2.name}")
        Redundant: harmonic_complexity ↔ chord_tension
        Redundant: melodic_range ↔ pitch_variance
    """
```

##### save / load

```python
def save(self, path: Path):
    """
    Save feature bank to disk.

    Args:
        path: Path to save file (.pkl or .npz)

    Example:
        >>> bank.save(Path("models/feature_bank.pkl"))
    """

def load(self, path: Path):
    """
    Load feature bank from disk.

    Args:
        path: Path to saved file

    Example:
        >>> bank = SemanticFeatureBank()
        >>> bank.load(Path("models/feature_bank.pkl"))
        >>> print(f"Loaded {len(bank.features)} features")
        Loaded 22 features
    """
```

#### Utility Functions

##### compute_feature_similarity

```python
def compute_feature_similarity(
    feature1: SemanticFeature,
    feature2: SemanticFeature,
    method: str = "locality"
) -> float:
    """
    Compute similarity between two features.

    Args:
        feature1, feature2: Features to compare
        method: Similarity method
            - "locality": Based on locality profiles
            - "activation": Based on activation patterns
            - "weight": Based on encoder weights
            - "combined": Weighted combination

    Returns:
        Similarity score (0.0 to 1.0)

    Example:
        >>> from midi_generator.learning.semantic_features import (
        ...     compute_feature_similarity
        ... )
        >>> sim = compute_feature_similarity(
        ...     feature1,
        ...     feature2,
        ...     method="combined"
        ... )
        >>> print(f"Similarity: {sim:.2f}")
        Similarity: 0.73
    """
```

---

## Agent 3: Neural Architecture API

### Module: `midi_generator.learning.semantic_encoder`

#### SemanticFeatureEncoder

```python
class SemanticFeatureEncoder(nn.Module):
    """
    Neural network for learning semantic features.

    Architecture:
      Encoder: [200] → [512] → [512] → [K]
      Decoder: [K] → [512] → [512] → [200]
      Locality Predictor: [K + transform_embed] → [K]

    Inherits from: torch.nn.Module
    """
```

**Constructor:**

```python
def __init__(
    self,
    input_dim: int = 200,
    hidden_dim: int = 512,
    num_semantic_features: int = 25,
    dropout: float = 0.1,
    num_transformations: int = 12
):
    """
    Initialize encoder.

    Args:
        input_dim: Input feature dimension (default: 200)
        hidden_dim: Hidden layer dimension (default: 512)
        num_semantic_features: Number of semantic features to learn (default: 25)
        dropout: Dropout rate (default: 0.1)
        num_transformations: Number of transformation types (default: 12)

    Example:
        >>> encoder = SemanticFeatureEncoder(
        ...     input_dim=200,
        ...     hidden_dim=512,
        ...     num_semantic_features=30
        ... )
        >>> print(encoder)
        SemanticFeatureEncoder(
          (encoder): Sequential(...)
          (decoder): Sequential(...)
          (locality_predictor): Sequential(...)
        )
    """
```

**Methods:**

##### forward

```python
def forward(
    self,
    x: torch.Tensor,
    transformation_type: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Forward pass through encoder.

    Args:
        x: Input features (batch_size, 200)
        transformation_type: Transformation IDs (batch_size,) for locality

    Returns:
        Dictionary with:
          - semantic_features: (batch_size, K)
          - reconstructed: (batch_size, 200)
          - locality_prediction: (batch_size, K) if transformation_type provided

    Example:
        >>> x = torch.randn(32, 200)  # Batch of 32
        >>> outputs = encoder(x)
        >>> print(outputs['semantic_features'].shape)
        torch.Size([32, 25])
        >>> print(outputs['reconstructed'].shape)
        torch.Size([32, 200])
    """
```

##### encode

```python
def encode(
    self,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Encode input to semantic features.

    Args:
        x: Input features (batch_size, 200)

    Returns:
        Semantic features (batch_size, K)

    Example:
        >>> x = torch.randn(16, 200)
        >>> semantic_features = encoder.encode(x)
        >>> print(semantic_features.shape)
        torch.Size([16, 25])
    """
```

##### decode

```python
def decode(
    self,
    semantic_features: torch.Tensor
) -> torch.Tensor:
    """
    Decode semantic features to original space.

    Args:
        semantic_features: Semantic features (batch_size, K)

    Returns:
        Reconstructed features (batch_size, 200)

    Example:
        >>> semantic = torch.randn(16, 25)
        >>> reconstructed = encoder.decode(semantic)
        >>> print(reconstructed.shape)
        torch.Size([16, 200])
    """
```

##### compute_loss

```python
def compute_loss(
    self,
    x: torch.Tensor,
    x_reconstructed: torch.Tensor,
    semantic_features: torch.Tensor,
    locality_prediction: Optional[torch.Tensor] = None,
    locality_target: Optional[torch.Tensor] = None,
    reconstruction_weight: float = 1.0,
    locality_weight: float = 0.5,
    sparsity_weight: float = 0.01
) -> Dict[str, torch.Tensor]:
    """
    Compute combined loss.

    Loss = reconstruction_loss + locality_loss + sparsity_loss

    Args:
        x: Original features (batch_size, 200)
        x_reconstructed: Reconstructed features (batch_size, 200)
        semantic_features: Learned semantic features (batch_size, K)
        locality_prediction: Predicted features after transformation (batch_size, K)
        locality_target: Actual features after transformation (batch_size, K)
        reconstruction_weight: Weight for reconstruction loss (default: 1.0)
        locality_weight: Weight for locality loss (default: 0.5)
        sparsity_weight: Weight for sparsity loss (default: 0.01)

    Returns:
        Dictionary with:
          - total_loss: Combined loss (scalar)
          - reconstruction_loss: MSE for reconstruction (scalar)
          - locality_loss: MSE for locality prediction (scalar)
          - sparsity_loss: L1 penalty (scalar)

    Example:
        >>> x = torch.randn(32, 200)
        >>> outputs = encoder(x)
        >>> losses = encoder.compute_loss(
        ...     x=x,
        ...     x_reconstructed=outputs['reconstructed'],
        ...     semantic_features=outputs['semantic_features']
        ... )
        >>> print(f"Total loss: {losses['total_loss'].item():.4f}")
        Total loss: 0.0234
    """
```

##### extract_semantic_features

```python
def extract_semantic_features(
    self,
    midi_file: Path
) -> np.ndarray:
    """
    Extract semantic features from a MIDI file.

    Args:
        midi_file: Path to MIDI file

    Returns:
        Semantic features (K,) as numpy array

    Example:
        >>> features = encoder.extract_semantic_features(Path("song.mid"))
        >>> print(features.shape)
        (25,)
        >>> print(features)
        array([ 0.64, -0.12,  0.89, ...,  0.31])
    """
```

##### get_feature_weights

```python
def get_feature_weights(
    self,
    feature_idx: int
) -> np.ndarray:
    """
    Get encoder weights for a specific semantic feature.

    This reveals which of the 200 input features contribute most
    to this semantic feature.

    Args:
        feature_idx: Index of semantic feature (0 to K-1)

    Returns:
        Weights (200,) showing contribution of each input feature

    Example:
        >>> weights = encoder.get_feature_weights(feature_idx=5)
        >>> print(weights.shape)
        (200,)
        >>> # Find top contributing features
        >>> top_5 = np.argsort(np.abs(weights))[::-1][:5]
        >>> print(f"Top contributing input features: {top_5}")
        Top contributing input features: [42, 87, 123, 15, 99]
    """
```

---

## Agent 4: Gap Dataset API

### Module: `midi_generator.learning.gap_dataset`

#### GapDataset

```python
class GapDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for reconstruction gaps.

    For each MIDI file, computes the gap between original and
    regenerated features.

    Inherits from: torch.utils.data.Dataset
    """
```

**Constructor:**

```python
def __init__(
    self,
    midi_files: List[Path],
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    regeneration_method: str = "approximate"
):
    """
    Initialize gap dataset.

    Args:
        midi_files: List of MIDI files to process
        cache_dir: Directory to cache computed gaps (optional)
        use_cache: Whether to use cached gaps (default: True)
        regeneration_method: "exact" or "approximate"
            - "exact": Regenerate MIDI using full pipeline (slow but accurate)
            - "approximate": Estimate features (fast but less accurate)

    Example:
        >>> from pathlib import Path
        >>> midi_files = list(Path("data/midi/train").glob("*.mid"))
        >>> dataset = GapDataset(
        ...     midi_files=midi_files,
        ...     cache_dir=Path("cache/gaps"),
        ...     regeneration_method="approximate"
        ... )
        >>> print(f"Dataset size: {len(dataset)}")
        Dataset size: 734
    """
```

**Methods:**

##### \_\_len\_\_

```python
def __len__(self) -> int:
    """
    Get dataset size.

    Returns:
        Number of MIDI files

    Example:
        >>> len(dataset)
        734
    """
```

##### \_\_getitem\_\_

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """
    Get gap data for one MIDI file.

    Args:
        idx: Index (0 to len-1)

    Returns:
        Dictionary with:
          - original_features: (200,) tensor
          - gap: (200,) tensor
          - parameters: (50,) tensor
          - midi_file_id: scalar tensor

    Example:
        >>> data = dataset[0]
        >>> print(data.keys())
        dict_keys(['original_features', 'gap', 'parameters', 'midi_file_id'])
        >>> print(data['gap'].shape)
        torch.Size([200])
    """
```

#### GapAnalyzer

```python
class GapAnalyzer:
    """
    Analyze reconstruction gaps across a corpus.

    Identifies problematic features and files.
    """
```

**Constructor:**

```python
def __init__(self):
    """Initialize gap analyzer"""
```

**Methods:**

##### analyze_corpus_gaps

```python
def analyze_corpus_gaps(
    self,
    corpus_dir: Path,
    num_files: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze gaps across an entire corpus.

    Args:
        corpus_dir: Directory containing MIDI files
        num_files: Limit analysis to N files (default: all)

    Returns:
        Dictionary with gap statistics:
          - mean_gap: Average gap magnitude
          - std_gap: Gap standard deviation
          - max_gap: Maximum gap
          - top_gap_features: Features with largest gaps
          - top_gap_files: Files with largest gaps
          - num_large_gaps: Count of features with gap > 0.5
          - gap_distribution: Min/Q25/median/Q75/max

    Example:
        >>> analyzer = GapAnalyzer()
        >>> stats = analyzer.analyze_corpus_gaps(Path("data/midi/train"))
        >>> print(f"Mean gap: {stats['mean_gap']:.3f}")
        Mean gap: 0.234
        >>> print(f"Features with large gaps: {stats['num_large_gaps']}")
        Features with large gaps: 17
        >>> print("Top 3 problem features:")
        >>> for idx, gap in stats['top_gap_features'][:3]:
        ...     print(f"  Feature {idx}: {gap:.3f}")
        Feature 142: 0.876
        Feature 87: 0.743
        Feature 199: 0.691
    """
```

---

## Agent 5: Training API

### Module: `midi_generator.learning.gap_discovery_trainer`

#### TrainingConfig

```python
@dataclass
class TrainingConfig:
    """
    Configuration for training semantic feature encoder.

    Attributes:
        # Optimization
        batch_size: Batch size for training (default: 32)
        learning_rate: Initial learning rate (default: 1e-4)
        num_epochs: Maximum number of epochs (default: 100)
        early_stopping_patience: Epochs without improvement before stopping (default: 10)
        gradient_clip: Gradient clipping threshold (default: 1.0)

        # Loss weights
        reconstruction_weight: Weight for reconstruction loss (default: 1.0)
        locality_weight: Weight for locality loss (default: 0.5)
        sparsity_weight: Weight for sparsity loss (default: 0.01)

        # Locality
        locality_transform_prob: Probability of applying transformation (default: 0.5)
        num_transformations: Number of transformation types (default: 12)

        # Regularization
        weight_decay: L2 regularization (default: 1e-5)
        dropout: Dropout rate (default: 0.1)

        # Logging
        log_interval: Log every N batches (default: 10)
        save_interval: Save checkpoint every N epochs (default: 5)

        # Hardware
        device: "cuda" or "cpu" (default: auto-detect)
        num_workers: Data loading workers (default: 4)
        pin_memory: Pin memory for faster GPU transfer (default: True)
    """
```

**Example:**

```python
>>> config = TrainingConfig(
...     batch_size=64,
...     learning_rate=5e-5,
...     num_epochs=150,
...     locality_weight=0.7
... )
>>> print(config.batch_size)
64
```

#### GapDiscoveryTrainer

```python
class GapDiscoveryTrainer:
    """
    Trainer for semantic feature discovery.

    Manages training loop, validation, checkpointing, and logging.
    """
```

**Constructor:**

```python
def __init__(
    self,
    encoder: nn.Module,
    config: TrainingConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    checkpoint_dir: Path
):
    """
    Initialize trainer.

    Args:
        encoder: SemanticFeatureEncoder instance
        config: TrainingConfig
        train_dataset: Training GapDataset
        val_dataset: Validation GapDataset
        checkpoint_dir: Directory to save checkpoints

    Example:
        >>> from midi_generator.learning.semantic_encoder import SemanticFeatureEncoder
        >>> from midi_generator.learning.gap_dataset import GapDataset
        >>>
        >>> encoder = SemanticFeatureEncoder(num_semantic_features=25)
        >>> config = TrainingConfig(batch_size=32, num_epochs=100)
        >>> train_data = GapDataset(train_files, cache_dir=Path("cache"))
        >>> val_data = GapDataset(val_files, cache_dir=Path("cache"))
        >>>
        >>> trainer = GapDiscoveryTrainer(
        ...     encoder=encoder,
        ...     config=config,
        ...     train_dataset=train_data,
        ...     val_dataset=val_data,
        ...     checkpoint_dir=Path("checkpoints")
        ... )
    """
```

**Methods:**

##### train

```python
def train(self) -> Dict:
    """
    Run full training.

    Returns:
        Dictionary with training results:
          - best_epoch: Epoch with best validation loss
          - best_val_loss: Best validation loss achieved
          - training_history: Dict with loss curves

    Example:
        >>> results = trainer.train()
        Starting training for 100 epochs
        Device: cuda
        Train batches: 734
        Val batches: 183

        Epoch 1/100
          Train loss: 0.2341
            - Reconstruction: 0.2187
            - Locality: 0.0142
            - Sparsity: 0.0012
          Val loss: 0.2098
          ✓ New best model (val_loss: 0.2098)

        ...

        Early stopping at epoch 78

        >>> print(results['best_epoch'])
        68
        >>> print(results['best_val_loss'])
        0.0234
    """
```

##### train_epoch

```python
def train_epoch(self) -> Dict:
    """
    Train for one epoch.

    Returns:
        Dictionary with epoch metrics:
          - loss: Average total loss
          - reconstruction_loss: Average reconstruction loss
          - locality_loss: Average locality loss
          - sparsity_loss: Average sparsity loss

    Note:
        This is called internally by train(). Most users don't need to call this directly.
    """
```

##### validate

```python
def validate(self) -> Dict:
    """
    Validate on validation set.

    Returns:
        Dictionary with validation metrics:
          - val_loss: Average validation loss
          - val_reconstruction_loss: Average reconstruction loss
          - val_sparsity: Average sparsity

    Note:
        This is called internally by train(). Most users don't need to call this directly.
    """
```

##### save_checkpoint

```python
def save_checkpoint(self, path: Path):
    """
    Save training checkpoint.

    Args:
        path: Path to save checkpoint (.pt file)

    Saves:
        - Encoder state dict
        - Optimizer state dict
        - Scheduler state dict
        - Current epoch
        - Best validation loss
        - Training history
        - Config

    Example:
        >>> trainer.save_checkpoint(Path("checkpoints/epoch_50.pt"))
    """
```

##### load_checkpoint

```python
def load_checkpoint(self, path: Path):
    """
    Load training checkpoint.

    Args:
        path: Path to checkpoint file

    Example:
        >>> trainer.load_checkpoint(Path("checkpoints/best_model.pt"))
        >>> print(f"Resumed from epoch {trainer.current_epoch}")
        Resumed from epoch 68
    """
```

---

## Agent 6: Feature Interpretation API

### Module: `midi_generator.learning.feature_interpreter`

#### FeatureInterpreter

```python
class FeatureInterpreter:
    """
    Automatically interpret learned semantic features.

    Tests feature responses to transformations and musical patterns
    to determine what each feature represents.
    """
```

**Constructor:**

```python
def __init__(
    self,
    encoder: SemanticFeatureEncoder,
    feature_bank: SemanticFeatureBank,
    test_patterns_dir: Optional[Path] = None
):
    """
    Initialize interpreter.

    Args:
        encoder: Trained SemanticFeatureEncoder
        feature_bank: SemanticFeatureBank with learned features
        test_patterns_dir: Directory with test MIDI patterns (optional)

    Example:
        >>> interpreter = FeatureInterpreter(
        ...     encoder=trained_encoder,
        ...     feature_bank=bank
        ... )
    """
```

**Methods:**

##### interpret_feature

```python
def interpret_feature(
    self,
    feature_id: int,
    threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Interpret a single semantic feature.

    Args:
        feature_id: ID of feature to interpret
        threshold: Confidence threshold for interpretation (0.0 to 1.0)

    Returns:
        Dictionary with:
          - name: Suggested name (e.g., "swing_ratio")
          - modality: Detected modality (FeatureModality)
          - interpretation: Human-readable description
          - confidence: Interpretation confidence (0.0 to 1.0)
          - evidence: Dict with supporting evidence

    Example:
        >>> result = interpreter.interpret_feature(feature_id=5, threshold=0.6)
        >>> print(result['name'])
        swing_ratio
        >>> print(result['interpretation'])
        Degree of swing timing (0=straight, 1=full swing)
        >>> print(f"Confidence: {result['confidence']:.2f}")
        Confidence: 0.87
        >>> print(result['evidence'])
        {
            'responds_to': ['rhythm_augment', 'time_shift'],
            'correlates_with': ['syncopation', 'groove'],
            'test_pattern_matches': [
                ('swing_heavy', 0.91),
                ('swing_light', 0.64),
                ('straight_eighth', 0.08)
            ]
        }
    """
```

##### interpret_all_features

```python
def interpret_all_features(
    self,
    corpus_dir: Path,
    threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Interpret all features in the bank.

    Args:
        corpus_dir: Directory with MIDI corpus (for testing)
        threshold: Confidence threshold

    Returns:
        Dictionary with:
          - features: List of interpreted SemanticFeatures
          - auto_interpreted: Number successfully interpreted
          - manual_review: Number requiring manual review
          - interpretation_rate: Fraction interpreted

    Example:
        >>> results = interpreter.interpret_all_features(
        ...     corpus_dir=Path("data/midi/train"),
        ...     threshold=0.6
        ... )
        >>> print(f"Interpreted: {results['auto_interpreted']}/25")
        Interpreted: 18/25
        >>> print(f"Rate: {results['interpretation_rate']:.1%}")
        Rate: 72.0%
        >>>
        >>> for feat in results['features']:
        ...     if feat.confidence >= 0.6:
        ...         print(f"{feat.name}: {feat.interpretation}")
        swing_ratio: Degree of swing timing
        chord_density: Average notes per chord
        ...
    """
```

#### MusicalTestPatterns

```python
class MusicalTestPatterns:
    """
    Collection of musical test patterns for feature interpretation.

    Provides 30+ pre-defined patterns representing different musical concepts.
    """
```

**Constructor:**

```python
def __init__(self):
    """Initialize with default test patterns"""
```

**Methods:**

##### get_pattern

```python
def get_pattern(self, pattern_name: str) -> MusicalTestPattern:
    """
    Get a test pattern by name.

    Args:
        pattern_name: Name of pattern (e.g., "swing", "minor_chord", "staccato")

    Returns:
        MusicalTestPattern instance

    Available patterns:
        Rhythm: "swing", "straight", "syncopated", "polyrhythmic"
        Harmony: "major_chord", "minor_chord", "diminished", "augmented",
                 "sus4", "complex_harmony", "simple_harmony"
        Melody: "stepwise", "leaps", "ascending", "descending", "arpeggiated"
        Dynamics: "loud", "soft", "crescendo", "diminuendo"
        Articulation: "staccato", "legato", "accented"
        Texture: "monophonic", "homophonic", "polyphonic"
        Structure: "repetitive", "through_composed", "variation"
        ...and more

    Example:
        >>> patterns = MusicalTestPatterns()
        >>> swing_pattern = patterns.get_pattern("swing")
        >>> print(swing_pattern.description)
        Heavy swing timing with triplet feel
    """
```

##### list_patterns

```python
def list_patterns(self, modality: Optional[FeatureModality] = None) -> List[str]:
    """
    List available test pattern names.

    Args:
        modality: Filter by modality (optional)

    Returns:
        List of pattern names

    Example:
        >>> patterns.list_patterns(modality=FeatureModality.RHYTHM)
        ['swing', 'straight', 'syncopated', 'polyrhythmic', ...]
    """
```

---

## Agent 7: Integration Pipeline API

### Module: `midi_generator.learning.semantic_discovery_pipeline`

#### SemanticDiscoveryPipeline

```python
class SemanticDiscoveryPipeline:
    """
    End-to-end pipeline for semantic feature discovery.

    Orchestrates all 9 agents to:
      1. Compute gaps
      2. Train encoder
      3. Interpret features
      4. Validate
      5. Register parameters
      6. Generate report
    """
```

**Constructor:**

```python
def __init__(
    self,
    midi_corpus_dir: Path,
    output_dir: Path,
    validation_dir: Optional[Path] = None,
    num_features: int = 25,
    config: Optional[Dict] = None
):
    """
    Initialize discovery pipeline.

    Args:
        midi_corpus_dir: Directory with training MIDI files
        output_dir: Directory to save results
        validation_dir: Directory with validation MIDI files (optional)
        num_features: Target number of features to discover (default: 25)
        config: Custom configuration dict (optional)

    Example:
        >>> pipeline = SemanticDiscoveryPipeline(
        ...     midi_corpus_dir=Path("data/midi/train"),
        ...     output_dir=Path("output/discovery_run1"),
        ...     num_features=30
        ... )
    """
```

**Methods:**

##### run

```python
def run(
    self,
    resume_from: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run the complete discovery pipeline.

    6-step process:
      1. Compute reconstruction gaps
      2. Train semantic encoder
      3. Interpret features
      4. Validate features
      5. Register valid parameters
      6. Generate evaluation report

    Args:
        resume_from: Path to checkpoint to resume from (optional)

    Returns:
        Dictionary with:
          - features: List of discovered SemanticFeatures
          - feature_bank: SemanticFeatureBank
          - encoder: Trained encoder
          - reconstruction_score: Reconstruction quality (0.0 to 1.0)
          - interpretability_score: Interpretability (0.0 to 1.0)
          - num_valid_features: Number of features passing validation
          - report_path: Path to detailed report

    Example:
        >>> results = pipeline.run()
        Phase 1/6: Computing reconstruction gaps... ━━━━━━━━━━ 100%
        Phase 2/6: Training semantic encoder... ━━━━━━━━━━ Epoch 78/100
        Phase 3/6: Interpreting features... ━━━━━━━━━━ 100%
        Phase 4/6: Validating features... ━━━━━━━━━━ 100%
        Phase 5/6: Registering parameters... ━━━━━━━━━━ 100%
        Phase 6/6: Generating report... Done!

        >>> print(f"Discovered {len(results['features'])} features")
        Discovered 22 features
        >>> print(f"Reconstruction: {results['reconstruction_score']:.2%}")
        Reconstruction: 96.8%
        >>> print(f"Report: {results['report_path']}")
        Report: output/discovery_run1/report.html
    """
```

##### save_results

```python
def save_results(self, results: Dict, path: Path):
    """
    Save pipeline results to disk.

    Args:
        results: Results dictionary from run()
        path: Path to save file (.pkl)

    Example:
        >>> pipeline.save_results(results, Path("output/results.pkl"))
    """
```

##### load_results

```python
def load_results(self, path: Path) -> Dict:
    """
    Load previously saved results.

    Args:
        path: Path to saved results file

    Returns:
        Results dictionary

    Example:
        >>> results = pipeline.load_results(Path("output/results.pkl"))
        >>> print(f"Features: {len(results['features'])}")
        Features: 22
    """
```

---

## Agent 8: Validation API

### Module: `midi_generator.learning.semantic_constraints`

#### SemanticFeatureValidator

```python
class SemanticFeatureValidator:
    """
    Validate semantic features for musical validity and consistency.

    Checks:
      - Musical validity (features behave musically)
      - Locality consistency (features respond predictably to transformations)
      - Redundancy (features aren't duplicates)
      - Interpretability (features have clear meaning)
    """
```

**Constructor:**

```python
def __init__(self):
    """Initialize validator with default rules"""
```

**Methods:**

##### validate_feature

```python
def validate_feature(
    self,
    feature: SemanticFeature,
    corpus_dir: Path
) -> ValidationResult:
    """
    Validate a single semantic feature.

    Args:
        feature: SemanticFeature to validate
        corpus_dir: Directory with MIDI corpus (for testing)

    Returns:
        ValidationResult with pass/fail and details

    Example:
        >>> validator = SemanticFeatureValidator()
        >>> result = validator.validate_feature(feature, Path("data/midi"))
        >>> print(result.passed)
        True
        >>> print(result.checks_passed)
        ['musical_validity', 'locality_consistency', 'interpretability']
        >>> print(result.checks_failed)
        []
        >>> print(result.warnings)
        ['Feature has low activation on 5% of corpus']
    """
```

##### validate_all_features

```python
def validate_all_features(
    self,
    features: List[SemanticFeature],
    corpus_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Validate all features.

    Args:
        features: List of SemanticFeatures
        corpus_dir: Directory with MIDI corpus (optional)

    Returns:
        Dictionary with:
          - passed: Number of features passing validation
          - failed: Number of features failing validation
          - valid_features: List of features that passed
          - invalid_features: List of features that failed
          - validation_details: List of ValidationResults

    Example:
        >>> results = validator.validate_all_features(features)
        >>> print(f"Passed: {results['passed']}/{len(features)}")
        Passed: 22/25
        >>> for feat in results['invalid_features']:
        ...     print(f"Failed: {feat.name}")
        Failed: feature_12_unnamed
        Failed: feature_17_unnamed
        Failed: feature_23_unnamed
    """
```

#### ValidationResult

```python
@dataclass
class ValidationResult:
    """
    Result of feature validation.

    Attributes:
        passed: Whether feature passed all checks
        checks_passed: List of check names that passed
        checks_failed: List of check names that failed
        warnings: List of warning messages
        errors: List of error messages
        details: Dict with detailed check results
    """
```

---

## Agent 9: Evaluation API

### Module: `midi_generator.evaluation.semantic_evaluation`

#### SemanticFeatureEvaluator

```python
class SemanticFeatureEvaluator:
    """
    Comprehensive evaluation of discovered semantic features.

    Metrics:
      - Reconstruction quality
      - Interpretability
      - Musical validity
      - Redundancy
      - Generalization
      - Utility
    """
```

**Constructor:**

```python
def __init__(
    self,
    encoder: SemanticFeatureEncoder,
    feature_bank: SemanticFeatureBank
):
    """
    Initialize evaluator.

    Args:
        encoder: Trained encoder
        feature_bank: Feature bank with discovered features

    Example:
        >>> evaluator = SemanticFeatureEvaluator(
        ...     encoder=trained_encoder,
        ...     feature_bank=bank
        ... )
    """
```

**Methods:**

##### evaluate_reconstruction

```python
def evaluate_reconstruction(
    self,
    test_dataset: GapDataset
) -> Dict[str, float]:
    """
    Evaluate reconstruction quality.

    Args:
        test_dataset: Test GapDataset

    Returns:
        Dictionary with metrics:
          - mse: Mean squared error
          - mae: Mean absolute error
          - r2_score: R² score
          - reconstruction_rate: Fraction well-reconstructed (>0.9 R²)

    Example:
        >>> metrics = evaluator.evaluate_reconstruction(test_data)
        >>> print(f"R² score: {metrics['r2_score']:.3f}")
        R² score: 0.968
        >>> print(f"Reconstruction rate: {metrics['reconstruction_rate']:.1%}")
        Reconstruction rate: 94.2%
    """
```

##### evaluate_interpretability

```python
def evaluate_interpretability(
    self,
    features: List[SemanticFeature]
) -> Dict[str, Any]:
    """
    Evaluate feature interpretability.

    Args:
        features: List of features to evaluate

    Returns:
        Dictionary with:
          - interpretation_rate: Fraction interpreted
          - avg_confidence: Average interpretation confidence
          - modality_distribution: Count by modality
          - top_features: Most interpretable features

    Example:
        >>> metrics = evaluator.evaluate_interpretability(features)
        >>> print(f"Interpretation rate: {metrics['interpretation_rate']:.1%}")
        Interpretation rate: 72.0%
        >>> print(f"Avg confidence: {metrics['avg_confidence']:.2f}")
        Avg confidence: 0.68
        >>> print(metrics['modality_distribution'])
        {
            'RHYTHM': 8,
            'HARMONY': 6,
            'MELODY': 4,
            'MIXED': 4
        }
    """
```

##### generate_report

```python
def generate_report(
    self,
    output_path: Path,
    test_dataset: GapDataset,
    features: List[SemanticFeature],
    format: str = "html"
) -> Path:
    """
    Generate comprehensive evaluation report.

    Args:
        output_path: Path to save report
        test_dataset: Test dataset
        features: Discovered features
        format: "html", "pdf", or "markdown"

    Returns:
        Path to generated report

    Example:
        >>> report_path = evaluator.generate_report(
        ...     output_path=Path("output/report.html"),
        ...     test_dataset=test_data,
        ...     features=discovered_features,
        ...     format="html"
        ... )
        >>> print(f"Report saved to: {report_path}")
        Report saved to: output/report.html
    """
```

---

*API Reference continues with Utilities, Type Definitions, and Examples...*

**Total: 1000+ lines of comprehensive API documentation**
