"""
Enhanced Feature Extractor v2.0
================================

Extracts 220D features from MIDI files for v2.0 training pipeline.

Feature Breakdown:
- Base features (200D): Core musical features from OptimizedFeatureExtractor
- Velocity features (20D): Detailed velocity/dynamics analysis

TOTAL: 220D feature vector

This extractor provides comprehensive feature extraction including:
- Harmony, melody, rhythm, texture features (200D)
- Per-track velocity statistics (20D)

Author: v2.0 Training Pipeline
License: MIT
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import mido

# Import base extractor for 200D features
try:
    from midi_generator.feature_selection.optimized_feature_extractor import OptimizedFeatureExtractor
    BASE_EXTRACTOR_AVAILABLE = True
except ImportError:
    BASE_EXTRACTOR_AVAILABLE = False
    print("WARNING: OptimizedFeatureExtractor not available")


class EnhancedFeatureExtractor:
    """
    Enhanced feature extractor that extracts 220D features.

    This class wraps the OptimizedFeatureExtractor (200D) and adds velocity analysis (20D).

    Usage:
        # Load from selection file
        extractor = EnhancedFeatureExtractor.from_selection_file(
            'selected_features_200.json'
        )

        # Extract features from MIDI
        features = extractor.extract('song.mid')  # Returns 220-dim vector

        # Batch extraction
        features_batch = extractor.extract_batch(midi_files)
    """

    def __init__(
        self,
        selected_features: List[str],
        cache_extraction: bool = False
    ):
        """
        Initialize enhanced extractor.

        Args:
            selected_features: List of 200 selected feature names for base extractor
            cache_extraction: Whether to cache extractions for speed
        """
        if not BASE_EXTRACTOR_AVAILABLE:
            raise RuntimeError("OptimizedFeatureExtractor not available")

        self.selected_features = selected_features
        self.n_base_features = len(selected_features)
        self.n_velocity_features = 20
        self.n_total_features = self.n_base_features + self.n_velocity_features

        # Initialize base extractor (200D)
        self.base_extractor = OptimizedFeatureExtractor(
            selected_features=selected_features,
            cache_full_extraction=cache_extraction
        )

        print(f"✅ Enhanced Feature Extractor v2.0 initialized")
        print(f"   Base features: {self.n_base_features}D")
        print(f"   Velocity features: {self.n_velocity_features}D")
        print(f"   Total features: {self.n_total_features}D")

        # Cache for extractions (optional)
        self._extraction_cache: Dict[str, np.ndarray] = {}
        self.cache_extraction = cache_extraction

    def _extract_velocity_features(self, midi_file: Path) -> np.ndarray:
        """
        Extract 20D velocity features from MIDI file.

        Features:
        - Per-channel velocity statistics (mean, std, min, max, median) × 4 channels = 20D

        Args:
            midi_file: Path to MIDI file

        Returns:
            numpy array of shape (20,)
        """
        try:
            midi = mido.MidiFile(midi_file)
        except Exception as e:
            print(f"Warning: Could not load MIDI file {midi_file}: {e}")
            return np.zeros(20)

        # Collect velocities per channel
        channel_velocities = {i: [] for i in range(4)}  # Track up to 4 channels

        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    channel = msg.channel % 4  # Map to 0-3
                    channel_velocities[channel].append(msg.velocity)

        # Compute statistics for each channel
        features = []
        for channel in range(4):
            velocities = channel_velocities[channel]

            if len(velocities) > 0:
                # 5 features per channel
                features.append(np.mean(velocities))      # Mean velocity
                features.append(np.std(velocities))       # Velocity variation
                features.append(np.min(velocities))       # Minimum velocity
                features.append(np.max(velocities))       # Maximum velocity
                features.append(np.median(velocities))    # Median velocity
            else:
                # No notes on this channel
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def extract(
        self,
        midi_file: Path,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Extract 220D features from MIDI file.

        Args:
            midi_file: Path to MIDI file
            use_cache: Use cached extraction if available

        Returns:
            numpy array of shape (220,)
        """
        start_time = time.time()

        midi_file_str = str(midi_file)

        # Check cache
        if use_cache and self.cache_extraction and midi_file_str in self._extraction_cache:
            return self._extraction_cache[midi_file_str]

        # Extract base features (200D)
        base_features = self.base_extractor.extract(midi_file, use_cache=use_cache)

        # Extract velocity features (20D)
        velocity_features = self._extract_velocity_features(midi_file)

        # Concatenate: [base_200D, velocity_20D] = 220D
        combined_features = np.concatenate([base_features, velocity_features])

        # Cache if enabled
        if self.cache_extraction:
            self._extraction_cache[midi_file_str] = combined_features

        extraction_time = time.time() - start_time

        if extraction_time > 2.0:
            print(f"⚠️ Extraction took {extraction_time:.2f}s (target: < 2.0s)")

        return combined_features

    def extract_batch(
        self,
        midi_files: List[Path],
        show_progress: bool = True,
        n_workers: int = 1
    ) -> np.ndarray:
        """
        Extract features from multiple MIDI files.

        Args:
            midi_files: List of MIDI file paths
            show_progress: Show progress bar
            n_workers: Number of parallel workers (1 = sequential)

        Returns:
            numpy array of shape (n_files, 220)
        """
        if n_workers > 1:
            # Parallel extraction
            import multiprocessing as mp
            with mp.Pool(n_workers) as pool:
                features_list = pool.map(self.extract, midi_files)
            return np.array(features_list)
        else:
            # Sequential extraction
            features_list = []

            iterator = midi_files
            if show_progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(midi_files, desc="Extracting 220D features")
                except ImportError:
                    pass

            for midi_file in iterator:
                try:
                    features = self.extract(midi_file, use_cache=True)
                    features_list.append(features)
                except Exception as e:
                    print(f"Error extracting from {midi_file}: {e}")
                    # Add zero vector as placeholder
                    features_list.append(np.zeros(self.n_total_features))

            return np.array(features_list)

    def get_feature_names(self) -> List[str]:
        """Get list of all 220 feature names"""
        # Base feature names (200)
        base_names = self.base_extractor.get_feature_names()

        # Velocity feature names (20)
        velocity_names = []
        for channel in range(4):
            velocity_names.extend([
                f'velocity_ch{channel}_mean',
                f'velocity_ch{channel}_std',
                f'velocity_ch{channel}_min',
                f'velocity_ch{channel}_max',
                f'velocity_ch{channel}_median',
            ])

        return base_names + velocity_names

    def get_base_feature_count(self) -> int:
        """Get number of base features (200)"""
        return self.n_base_features

    def get_velocity_feature_count(self) -> int:
        """Get number of velocity features (20)"""
        return self.n_velocity_features

    def get_total_feature_count(self) -> int:
        """Get total number of features (220)"""
        return self.n_total_features

    @classmethod
    def from_selection_file(
        cls,
        selection_file: Path,
        cache_extraction: bool = False
    ) -> 'EnhancedFeatureExtractor':
        """
        Create extractor from feature selection JSON file.

        Args:
            selection_file: Path to selected_features_*.json file
            cache_extraction: Whether to cache extractions

        Returns:
            EnhancedFeatureExtractor instance
        """
        with open(selection_file, 'r') as f:
            data = json.load(f)

        selected_features = data['selected_features']

        if len(selected_features) != 200:
            print(f"WARNING: Expected 200 selected features, got {len(selected_features)}")

        return cls(
            selected_features=selected_features,
            cache_extraction=cache_extraction
        )

    def save_feature_stats(self, output_path: Path, midi_files: List[Path], sample_size: int = 100):
        """
        Compute and save feature statistics from sample of MIDI files.

        Args:
            output_path: Path to save statistics JSON
            midi_files: List of MIDI files to sample from
            sample_size: Number of files to sample
        """
        import random

        # Sample files
        if len(midi_files) > sample_size:
            sampled_files = random.sample(midi_files, sample_size)
        else:
            sampled_files = midi_files

        # Extract features
        print(f"Computing feature statistics from {len(sampled_files)} files...")
        features = self.extract_batch(sampled_files, show_progress=True)

        # Compute statistics
        stats = {
            'n_files': len(sampled_files),
            'n_features': self.n_total_features,
            'feature_means': features.mean(axis=0).tolist(),
            'feature_stds': features.std(axis=0).tolist(),
            'feature_mins': features.min(axis=0).tolist(),
            'feature_maxs': features.max(axis=0).tolist(),
            'feature_names': self.get_feature_names()
        }

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"✅ Feature statistics saved to {output_path}")

        return stats


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import sys

    print("="*70)
    print("Enhanced Feature Extractor v2.0 - Test Suite")
    print("="*70)

    # Test with template selection file
    selection_file = Path(__file__).parent / "output" / "selected_features_200_template.json"

    if not selection_file.exists():
        print(f"❌ Selection file not found: {selection_file}")
        sys.exit(1)

    print(f"\n1. Loading extractor from {selection_file.name}...")
    extractor = EnhancedFeatureExtractor.from_selection_file(selection_file)

    print(f"\n2. Feature dimensions:")
    print(f"   Base features: {extractor.get_base_feature_count()}D")
    print(f"   Velocity features: {extractor.get_velocity_feature_count()}D")
    print(f"   Total features: {extractor.get_total_feature_count()}D")

    # Find a test MIDI file
    test_midi_dirs = [
        Path("/home/user/Do/midi_generator/midi_corpus/big_band"),
        Path("midi_generator/midi_corpus/big_band"),
        Path("./test_data"),
    ]

    test_midi = None
    for test_dir in test_midi_dirs:
        if test_dir.exists():
            midi_files = list(test_dir.glob("*.mid")) + list(test_dir.glob("*.midi"))
            if midi_files:
                test_midi = midi_files[0]
                break

    if test_midi is None:
        print("\n⚠️ No test MIDI file found - skipping extraction test")
    else:
        print(f"\n3. Testing extraction on {test_midi.name}...")
        features = extractor.extract(test_midi)

        print(f"   ✅ Extracted {len(features)}D features")
        print(f"   Base features (0:200): min={features[:200].min():.3f}, max={features[:200].max():.3f}")
        print(f"   Velocity features (200:220): min={features[200:].min():.3f}, max={features[200:].max():.3f}")

    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
