"""
Dimension-Specific Feature Router
==================================

Routes features from DeepFeatureExtractor to dimension-specific encoders.

This ensures each encoder receives ONLY features relevant to its musical dimension:
- Harmony encoder ← harmony features (250D)
- Rhythm encoder ← rhythm features (250D)
- Form encoder ← structure features (50D)
- Orchestration encoder ← orchestration features (150D)
- Texture encoder ← texture features (100D)
- Dynamics encoder ← dynamics features (150D)
- Melody encoder ← melody features (200D)

Author: Architecture Fix - Nov 22, 2025
"""

from pathlib import Path
from typing import Dict, Optional
from enum import Enum
import numpy as np
import warnings

try:
    from midi_generator.synthesis.deep_feature_extractor import DeepFeatureExtractor
    DEEP_EXTRACTOR_AVAILABLE = True
except ImportError:
    DEEP_EXTRACTOR_AVAILABLE = False
    warnings.warn("DeepFeatureExtractor not available")


class MusicalDimension(Enum):
    """Musical dimensions for feature routing"""
    HARMONY = "harmony"
    MELODY = "melody"
    RHYTHM = "rhythm"
    DYNAMICS = "dynamics"
    TEXTURE = "texture"
    FORM = "form"  # Maps to 'structure' features
    ORCHESTRATION = "orchestration"
    CROSS_DIMENSIONAL = "cross_dimensional"


class DimensionFeatureRouter:
    """
    Routes dimension-specific features to encoders.

    Feature Mapping (from DeepFeatureExtractor 1150D):
    - harmony_0 to harmony_249     → HARMONY encoder (250D)
    - melody_0 to melody_199       → MELODY encoder (200D)
    - rhythm_0 to rhythm_249       → RHYTHM encoder (250D)
    - dynamics_0 to dynamics_149   → DYNAMICS encoder (150D)
    - texture_0 to texture_99      → TEXTURE encoder (100D)
    - structure_0 to structure_49  → FORM encoder (50D)
    - orchestration_0 to orchestration_149 → ORCHESTRATION encoder (150D)

    Total: 1150 features across 7 dimensions
    """

    # Feature ranges for each dimension in DeepFeatureExtractor
    DIMENSION_RANGES = {
        MusicalDimension.HARMONY: (0, 250),           # 250 features
        MusicalDimension.MELODY: (250, 450),          # 200 features
        MusicalDimension.RHYTHM: (450, 700),          # 250 features
        MusicalDimension.DYNAMICS: (700, 850),        # 150 features
        MusicalDimension.TEXTURE: (850, 950),         # 100 features
        MusicalDimension.FORM: (950, 1000),           # 50 features (structure)
        MusicalDimension.ORCHESTRATION: (1000, 1150), # 150 features
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize feature router.

        Args:
            cache_dir: Optional directory for caching extracted features
        """
        if not DEEP_EXTRACTOR_AVAILABLE:
            raise ImportError("DeepFeatureExtractor required for dimension routing")

        self.extractor = DeepFeatureExtractor()
        self.cache_dir = cache_dir
        self._cache = {}  # In-memory cache: {midi_path: features_1150d}

    def extract_for_dimension(
        self,
        midi_file: Path,
        dimension: MusicalDimension,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Extract features for a specific musical dimension.

        Args:
            midi_file: Path to MIDI file
            dimension: Which dimension to extract features for
            use_cache: Whether to use cached features

        Returns:
            numpy array of dimension-specific features

        Example:
            router = DimensionFeatureRouter()
            harmony_features = router.extract_for_dimension(
                Path("song.mid"),
                MusicalDimension.HARMONY
            )
            # Returns 250D harmony features
        """
        # Get full 1150D feature vector
        full_features = self._get_full_features(midi_file, use_cache)

        # Handle cross-dimensional specially (no feature routing needed)
        if dimension == MusicalDimension.CROSS_DIMENSIONAL:
            raise ValueError("Cross-dimensional encoder uses encoder outputs, not raw features")

        # Get dimension-specific slice
        start_idx, end_idx = self.DIMENSION_RANGES[dimension]
        dimension_features = full_features[start_idx:end_idx]

        return dimension_features

    def get_dimension_size(self, dimension: MusicalDimension) -> int:
        """Get feature count for a dimension"""
        if dimension == MusicalDimension.CROSS_DIMENSIONAL:
            # Cross-dimensional takes concatenated encoder outputs
            # Assuming: harmony=30, rhythm=20, form=15, orch=25, texture=20 = 110D
            return 110

        start_idx, end_idx = self.DIMENSION_RANGES[dimension]
        return end_idx - start_idx

    def _get_full_features(self, midi_file: Path, use_cache: bool) -> np.ndarray:
        """Extract full 1150D feature vector"""
        cache_key = str(midi_file)

        # Check cache
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Extract features
        try:
            features = self.extractor.extract(midi_file)
        except Exception as e:
            warnings.warn(f"Feature extraction failed for {midi_file}: {e}")
            features = np.zeros(1150, dtype=np.float32)

        # Cache
        if use_cache:
            self._cache[cache_key] = features

        return features

    def clear_cache(self):
        """Clear in-memory feature cache"""
        self._cache.clear()

    def get_feature_names_for_dimension(self, dimension: MusicalDimension) -> list:
        """Get feature names for a specific dimension"""
        if dimension == MusicalDimension.CROSS_DIMENSIONAL:
            return []  # No raw features

        start_idx, end_idx = self.DIMENSION_RANGES[dimension]
        count = end_idx - start_idx

        # Map back to original feature names
        dim_name = "structure" if dimension == MusicalDimension.FORM else dimension.value
        return [f"{dim_name}_{i}" for i in range(count)]


# ============================================================================
# Integration with Existing Pipeline
# ============================================================================

def create_dimension_router(cache_dir: Optional[Path] = None) -> DimensionFeatureRouter:
    """Factory function to create feature router"""
    return DimensionFeatureRouter(cache_dir=cache_dir)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Dimension-Specific Feature Router")
    print("="*70)

    if DEEP_EXTRACTOR_AVAILABLE:
        router = create_dimension_router()

        print("\nFeature dimensions for each encoder:")
        for dimension in MusicalDimension:
            if dimension == MusicalDimension.CROSS_DIMENSIONAL:
                print(f"  {dimension.value}: 110D (from encoder outputs)")
            else:
                size = router.get_dimension_size(dimension)
                print(f"  {dimension.value}: {size}D")

        print("\n✅ Feature router initialized successfully")
    else:
        print("\n❌ DeepFeatureExtractor not available")
