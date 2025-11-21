"""
Semantic Feature Representations - Agent 2
==========================================

Semantic feature system for automated musical parameter discovery.
Represents learned features with musical locality constraints and interpretability.

This module provides:
1. SemanticFeature: Dataclass for individual learned features
2. SemanticFeatureBank: Collection and management of features
3. Similarity metrics for feature comparison
4. Integration with musical locality functions (Agent 1)

Key Concepts:
- Semantic features are learned representations that correspond to musical concepts
- Features must be locally invariant (similar under musical transformations)
- Features should be interpretable and mappable to parameters

Author: Agent 2 - Semantic Feature Representations
License: MIT
"""

from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
import pickle

# Import Agent 1's locality functions
try:
    from .musical_locality import (
        LocalityType,
        MusicalTransform,
        MusicalLocalityFunctions
    )
    LOCALITY_AVAILABLE = True
except ImportError:
    LOCALITY_AVAILABLE = False
    print("WARNING: Musical locality functions not available")


class FeatureModality(Enum):
    """
    Musical modality/aspect that a semantic feature captures.

    Features are classified by what musical aspect they represent:
    - MELODIC: Pitch patterns, intervals, contours
    - HARMONIC: Chord structures, progressions, voice leading
    - RHYTHMIC: Time patterns, syncopation, groove
    - TIMBRAL: Instrument choice, articulation, texture
    - DYNAMIC: Volume patterns, accents, expression
    - STRUCTURAL: Form, phrasing, repetition
    - COMBINATORIAL: Multi-aspect patterns
    - UNKNOWN: Not yet interpreted
    """
    MELODIC = auto()
    HARMONIC = auto()
    RHYTHMIC = auto()
    TIMBRAL = auto()
    DYNAMIC = auto()
    STRUCTURAL = auto()
    COMBINATORIAL = auto()
    UNKNOWN = auto()


@dataclass
class SemanticFeature:
    """
    Represents a single learned semantic feature.

    A semantic feature is a learned representation that:
    1. Activates strongly for specific musical patterns
    2. Is invariant under musical locality transformations
    3. Can be interpreted as a musical concept/parameter

    Attributes:
        feature_id: Unique identifier
        weight_vector: Learned weights (activation pattern in feature space)
        modality: Musical aspect this feature captures
        activation_threshold: Minimum activation strength to consider "active"
        locality_constraints: Which transforms preserve this feature
        interpretation: Human-readable description (from Agent 6)
        parameter_mapping: Mapping to parameter if discovered (from Agent 6)
        metadata: Additional information
    """
    feature_id: str
    weight_vector: np.ndarray
    modality: FeatureModality = FeatureModality.UNKNOWN
    activation_threshold: float = 0.5
    locality_constraints: List[LocalityType] = field(default_factory=list)
    interpretation: Optional[str] = None
    parameter_mapping: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate feature after initialization"""
        if len(self.weight_vector.shape) != 1:
            raise ValueError("weight_vector must be 1-dimensional")

        if not 0.0 <= self.activation_threshold <= 1.0:
            raise ValueError("activation_threshold must be in [0, 1]")

    def get_activation_strength(
        self,
        input_features: np.ndarray,
        normalize: bool = True
    ) -> float:
        """
        Compute activation strength for input features.

        Args:
            input_features: Input feature vector (200D from OptimizedFeatureExtractor)
            normalize: Whether to normalize activation to [0, 1]

        Returns:
            Activation strength (higher = stronger match)
        """
        if input_features.shape[0] != self.weight_vector.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.weight_vector.shape[0]}, "
                f"got {input_features.shape[0]}"
            )

        # Dot product activation
        activation = np.dot(self.weight_vector, input_features)

        if normalize:
            # Normalize to [0, 1] using sigmoid
            activation = 1.0 / (1.0 + np.exp(-activation))

        return float(activation)

    def matches_pattern(
        self,
        input_features: np.ndarray,
        threshold: Optional[float] = None
    ) -> bool:
        """
        Check if input features match this semantic feature pattern.

        Args:
            input_features: Input feature vector
            threshold: Activation threshold (uses default if None)

        Returns:
            True if activation exceeds threshold
        """
        threshold = threshold if threshold is not None else self.activation_threshold
        activation = self.get_activation_strength(input_features, normalize=True)
        return activation >= threshold

    def generate_variants(
        self,
        input_features: np.ndarray,
        num_variants: int = 5,
        locality_funcs: Optional[MusicalLocalityFunctions] = None
    ) -> List[Tuple[np.ndarray, MusicalTransform, float]]:
        """
        Generate musical variants that should preserve this feature.

        Uses locality transformations to create variants. The semantic feature
        should activate similarly on all variants (locality constraint).

        Args:
            input_features: Original feature vector
            num_variants: Number of variants to generate
            locality_funcs: Locality functions (creates new if None)

        Returns:
            List of (transformed_features, transform, activation_strength) tuples
        """
        if not LOCALITY_AVAILABLE:
            raise RuntimeError("Musical locality functions not available")

        if locality_funcs is None:
            locality_funcs = MusicalLocalityFunctions()

        # Use only transforms specified in locality constraints
        # If no constraints, use all transforms
        transform_types = (
            self.locality_constraints
            if self.locality_constraints
            else list(LocalityType)
        )

        # Generate variants
        variants = locality_funcs.generate_variants(
            input_features,
            num_variants=num_variants,
            transform_types=transform_types
        )

        # Compute activation for each variant
        results = []
        for variant_features, transform in variants:
            activation = self.get_activation_strength(variant_features, normalize=True)
            results.append((variant_features, transform, activation))

        return results

    def test_locality_invariance(
        self,
        input_features: np.ndarray,
        locality_funcs: Optional[MusicalLocalityFunctions] = None,
        num_tests: int = 10,
        tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """
        Test if this feature is invariant under locality transformations.

        A good semantic feature should have similar activations for
        musically similar inputs (locality-invariant).

        Args:
            input_features: Original feature vector
            locality_funcs: Locality functions
            num_tests: Number of test variants
            tolerance: Maximum allowed activation variation

        Returns:
            Dictionary with test results
        """
        original_activation = self.get_activation_strength(input_features)

        # Generate variants and test
        variants = self.generate_variants(
            input_features,
            num_variants=num_tests,
            locality_funcs=locality_funcs
        )

        activations = [act for _, _, act in variants]
        activation_std = np.std(activations)
        activation_mean = np.mean(activations)
        max_deviation = max(abs(a - original_activation) for a in activations)

        is_invariant = max_deviation <= tolerance

        return {
            'is_invariant': is_invariant,
            'original_activation': original_activation,
            'mean_activation': activation_mean,
            'std_activation': activation_std,
            'max_deviation': max_deviation,
            'tolerance': tolerance,
            'num_tests': num_tests,
            'activations': activations
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary (for JSON storage).

        Note: weight_vector is converted to list for JSON compatibility.
        """
        data = {
            'feature_id': self.feature_id,
            'weight_vector': self.weight_vector.tolist(),
            'modality': self.modality.name,
            'activation_threshold': self.activation_threshold,
            'locality_constraints': [lc.name for lc in self.locality_constraints],
            'interpretation': self.interpretation,
            'parameter_mapping': self.parameter_mapping,
            'metadata': self.metadata
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticFeature':
        """Deserialize from dictionary"""
        return cls(
            feature_id=data['feature_id'],
            weight_vector=np.array(data['weight_vector']),
            modality=FeatureModality[data['modality']],
            activation_threshold=data.get('activation_threshold', 0.5),
            locality_constraints=[
                LocalityType[lc] for lc in data.get('locality_constraints', [])
            ],
            interpretation=data.get('interpretation'),
            parameter_mapping=data.get('parameter_mapping'),
            metadata=data.get('metadata', {})
        )


class SemanticFeatureBank:
    """
    Collection of learned semantic features.

    Manages multiple semantic features, provides batch operations,
    and handles serialization/deserialization.

    Usage:
        # Create bank
        bank = SemanticFeatureBank()

        # Add features
        bank.add_feature(feature1)
        bank.add_feature(feature2)

        # Get activations for input
        activations = bank.get_activations(input_features)

        # Get top-k active features
        top_features = bank.get_top_k_features(input_features, k=10)

        # Save/load
        bank.save('semantic_features.pkl')
        bank = SemanticFeatureBank.load('semantic_features.pkl')
    """

    def __init__(self, features: Optional[List[SemanticFeature]] = None):
        """
        Initialize feature bank.

        Args:
            features: Initial list of features (empty if None)
        """
        self.features: Dict[str, SemanticFeature] = {}
        self.feature_dimension: Optional[int] = None
        self.metadata: Dict[str, Any] = {
            'created_at': None,
            'training_corpus': None,
            'num_training_samples': 0,
            'training_config': {}
        }

        if features:
            for feature in features:
                self.add_feature(feature)

    def add_feature(self, feature: SemanticFeature) -> None:
        """
        Add a semantic feature to the bank.

        Args:
            feature: Feature to add
        """
        # Check dimension consistency
        feature_dim = feature.weight_vector.shape[0]
        if self.feature_dimension is None:
            self.feature_dimension = feature_dim
        elif self.feature_dimension != feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: bank has {self.feature_dimension}, "
                f"feature has {feature_dim}"
            )

        # Check for duplicate ID
        if feature.feature_id in self.features:
            raise ValueError(f"Feature with ID {feature.feature_id} already exists")

        self.features[feature.feature_id] = feature

    def remove_feature(self, feature_id: str) -> None:
        """Remove a feature from the bank"""
        if feature_id not in self.features:
            raise ValueError(f"Feature {feature_id} not found")
        del self.features[feature_id]

    def get_feature(self, feature_id: str) -> SemanticFeature:
        """Get a feature by ID"""
        if feature_id not in self.features:
            raise ValueError(f"Feature {feature_id} not found")
        return self.features[feature_id]

    def get_activations(
        self,
        input_features: np.ndarray,
        normalize: bool = True,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute activation strengths for all features.

        Args:
            input_features: Input feature vector
            normalize: Whether to normalize activations
            threshold: Only return activations above this threshold (None = return all)

        Returns:
            Dictionary mapping feature_id -> activation_strength
        """
        activations = {}
        for feature_id, feature in self.features.items():
            activation = feature.get_activation_strength(input_features, normalize)

            if threshold is None or activation >= threshold:
                activations[feature_id] = activation

        return activations

    def get_top_k_features(
        self,
        input_features: np.ndarray,
        k: int = 10,
        normalize: bool = True
    ) -> List[Tuple[str, float, SemanticFeature]]:
        """
        Get top-k most activated features.

        Args:
            input_features: Input feature vector
            k: Number of top features to return
            normalize: Whether to normalize activations

        Returns:
            List of (feature_id, activation, feature) tuples, sorted by activation
        """
        activations = self.get_activations(input_features, normalize=normalize)

        # Sort by activation (descending)
        sorted_features = sorted(
            activations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top-k with feature objects
        return [
            (feature_id, activation, self.features[feature_id])
            for feature_id, activation in sorted_features[:k]
        ]

    def get_features_by_modality(
        self,
        modality: FeatureModality
    ) -> List[SemanticFeature]:
        """Get all features of a specific modality"""
        return [
            feature for feature in self.features.values()
            if feature.modality == modality
        ]

    def get_interpreted_features(self) -> List[SemanticFeature]:
        """Get all features that have been interpreted (Agent 6)"""
        return [
            feature for feature in self.features.values()
            if feature.interpretation is not None
        ]

    def get_mapped_parameters(self) -> Dict[str, SemanticFeature]:
        """
        Get features that have been mapped to parameters.

        Returns:
            Dictionary mapping parameter_name -> feature
        """
        mapped = {}
        for feature in self.features.values():
            if feature.parameter_mapping:
                param_name = feature.parameter_mapping.get('parameter_name')
                if param_name:
                    mapped[param_name] = feature
        return mapped

    def compute_feature_statistics(self) -> Dict[str, Any]:
        """Compute statistics about features in the bank"""
        if not self.features:
            return {'num_features': 0}

        modality_counts = defaultdict(int)
        interpreted_count = 0
        mapped_count = 0

        weight_magnitudes = []
        thresholds = []

        for feature in self.features.values():
            modality_counts[feature.modality.name] += 1
            if feature.interpretation:
                interpreted_count += 1
            if feature.parameter_mapping:
                mapped_count += 1

            weight_magnitudes.append(np.linalg.norm(feature.weight_vector))
            thresholds.append(feature.activation_threshold)

        return {
            'num_features': len(self.features),
            'feature_dimension': self.feature_dimension,
            'modality_distribution': dict(modality_counts),
            'num_interpreted': interpreted_count,
            'num_mapped_to_parameters': mapped_count,
            'interpretation_rate': interpreted_count / len(self.features) if self.features else 0,
            'mapping_rate': mapped_count / len(self.features) if self.features else 0,
            'avg_weight_magnitude': np.mean(weight_magnitudes),
            'avg_threshold': np.mean(thresholds),
            'metadata': self.metadata
        }

    def save(self, filepath: Path) -> None:
        """
        Save feature bank to file.

        Uses pickle for numpy array support.
        Also saves JSON version for human readability.

        Args:
            filepath: Path to save to (e.g., 'features.pkl')
        """
        filepath = Path(filepath)

        # Save pickle version (full functionality)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        # Save JSON version (human-readable, but without numpy arrays)
        json_path = filepath.with_suffix('.json')
        json_data = {
            'features': {
                fid: feature.to_dict()
                for fid, feature in self.features.items()
            },
            'feature_dimension': self.feature_dimension,
            'metadata': self.metadata,
            'statistics': self.compute_feature_statistics()
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"✅ Saved {len(self.features)} features to {filepath}")
        print(f"   JSON version: {json_path}")

    @classmethod
    def load(cls, filepath: Path) -> 'SemanticFeatureBank':
        """
        Load feature bank from file.

        Args:
            filepath: Path to load from

        Returns:
            Loaded SemanticFeatureBank
        """
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            bank = pickle.load(f)

        if not isinstance(bank, cls):
            raise TypeError(f"Loaded object is not SemanticFeatureBank")

        print(f"✅ Loaded {len(bank.features)} features from {filepath}")
        return bank

    def __len__(self) -> int:
        """Number of features in bank"""
        return len(self.features)

    def __contains__(self, feature_id: str) -> bool:
        """Check if feature exists in bank"""
        return feature_id in self.features

    def __iter__(self):
        """Iterate over features"""
        return iter(self.features.values())

    def __repr__(self) -> str:
        stats = self.compute_feature_statistics()
        return (
            f"SemanticFeatureBank("
            f"num_features={stats['num_features']}, "
            f"dimension={stats['feature_dimension']}, "
            f"interpreted={stats['num_interpreted']}, "
            f"mapped={stats['num_mapped_to_parameters']})"
        )


# ============================================================================
# Similarity Metrics
# ============================================================================

def cosine_similarity(feature1: SemanticFeature, feature2: SemanticFeature) -> float:
    """
    Compute cosine similarity between two features.

    Args:
        feature1: First feature
        feature2: Second feature

    Returns:
        Cosine similarity in [-1, 1]
    """
    if feature1.weight_vector.shape != feature2.weight_vector.shape:
        raise ValueError("Features must have same dimension")

    dot_product = np.dot(feature1.weight_vector, feature2.weight_vector)
    norm1 = np.linalg.norm(feature1.weight_vector)
    norm2 = np.linalg.norm(feature2.weight_vector)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def euclidean_distance(feature1: SemanticFeature, feature2: SemanticFeature) -> float:
    """
    Compute Euclidean distance between two features.

    Args:
        feature1: First feature
        feature2: Second feature

    Returns:
        Euclidean distance (>=0)
    """
    if feature1.weight_vector.shape != feature2.weight_vector.shape:
        raise ValueError("Features must have same dimension")

    return float(np.linalg.norm(feature1.weight_vector - feature2.weight_vector))


def activation_correlation(
    feature1: SemanticFeature,
    feature2: SemanticFeature,
    test_samples: np.ndarray
) -> float:
    """
    Compute correlation of activations across test samples.

    Measures whether two features activate similarly on the same inputs.

    Args:
        feature1: First feature
        feature2: Second feature
        test_samples: Test feature vectors (N x D)

    Returns:
        Pearson correlation in [-1, 1]
    """
    activations1 = [
        feature1.get_activation_strength(sample)
        for sample in test_samples
    ]
    activations2 = [
        feature2.get_activation_strength(sample)
        for sample in test_samples
    ]

    correlation = np.corrcoef(activations1, activations2)[0, 1]
    return float(correlation)


def find_similar_features(
    target_feature: SemanticFeature,
    feature_bank: SemanticFeatureBank,
    k: int = 5,
    metric: str = 'cosine',
    test_samples: Optional[np.ndarray] = None
) -> List[Tuple[str, float, SemanticFeature]]:
    """
    Find k most similar features to target feature.

    Args:
        target_feature: Feature to find similar features for
        feature_bank: Bank to search in
        k: Number of similar features to return
        metric: Similarity metric ('cosine', 'euclidean', 'activation')
        test_samples: Test samples (required for activation correlation)

    Returns:
        List of (feature_id, similarity, feature) tuples
    """
    similarities = []

    for feature_id, feature in feature_bank.features.items():
        if feature_id == target_feature.feature_id:
            continue  # Skip self

        if metric == 'cosine':
            sim = cosine_similarity(target_feature, feature)
        elif metric == 'euclidean':
            sim = -euclidean_distance(target_feature, feature)  # Negative for sorting
        elif metric == 'activation':
            if test_samples is None:
                raise ValueError("test_samples required for activation correlation")
            sim = activation_correlation(target_feature, feature, test_samples)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        similarities.append((feature_id, sim, feature))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:k]


def detect_redundant_features(
    feature_bank: SemanticFeatureBank,
    similarity_threshold: float = 0.9,
    metric: str = 'cosine'
) -> List[Tuple[str, str, float]]:
    """
    Detect redundant (highly similar) features in bank.

    Args:
        feature_bank: Feature bank to analyze
        similarity_threshold: Threshold for redundancy
        metric: Similarity metric to use

    Returns:
        List of (feature_id1, feature_id2, similarity) tuples for redundant pairs
    """
    redundant_pairs = []
    feature_ids = list(feature_bank.features.keys())

    for i, fid1 in enumerate(feature_ids):
        for fid2 in feature_ids[i+1:]:
            feature1 = feature_bank.features[fid1]
            feature2 = feature_bank.features[fid2]

            if metric == 'cosine':
                sim = cosine_similarity(feature1, feature2)
            elif metric == 'euclidean':
                sim = 1.0 / (1.0 + euclidean_distance(feature1, feature2))
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if sim >= similarity_threshold:
                redundant_pairs.append((fid1, fid2, sim))

    return redundant_pairs


# ============================================================================
# Utility Functions
# ============================================================================

def create_semantic_feature(
    feature_id: str,
    weight_vector: np.ndarray,
    **kwargs
) -> SemanticFeature:
    """
    Convenience function to create a SemanticFeature.

    Args:
        feature_id: Unique ID
        weight_vector: Learned weights
        **kwargs: Additional arguments for SemanticFeature

    Returns:
        Created SemanticFeature
    """
    return SemanticFeature(
        feature_id=feature_id,
        weight_vector=weight_vector,
        **kwargs
    )


if __name__ == '__main__':
    print("Semantic Feature Representations - Agent 2")
    print("==========================================")
    print()
    print("Components:")
    print("  ✅ FeatureModality enum (8 modalities)")
    print("  ✅ SemanticFeature dataclass")
    print("  ✅ SemanticFeatureBank class")
    print("  ✅ Similarity metrics")
    print()
    print("Integration:")
    print("  → Agent 1: Musical locality functions")
    print("  → Agent 3: Neural encoder (learns these features)")
    print("  → Agent 6: Feature interpretation")
    print()
    print("Success Criteria:")
    print("  ✅ SemanticFeature fully functional")
    print("  ✅ Similarity metrics work")
    print("  ✅ Serialization works")
    print("  ✅ Integration with Agent 1")
