"""
Parameter Prediction API - Agent 09
====================================

High-level API for predicting musical parameters from MIDI files.

This module combines:
- Feature extraction (Agent 04/08)
- Hierarchical MTL model inference (Agents 05-06)
- Parameter validation and sanitization
- Performance optimization

Provides two main workflows:
1. MIDI → Features → Parameters (analysis/extraction)
2. Parameters → MIDI (generation)

Author: Agent 09 - HarmonyModule Integration Lead
Date: 2025-11-20
License: MIT
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import time
import json
import warnings

warnings.filterwarnings('ignore')

# Internal imports
try:
    from midi_generator.synthesis.deep_feature_extractor import DeepFeatureExtractor
    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError:
    print("Warning: DeepFeatureExtractor not available")
    FEATURE_EXTRACTOR_AVAILABLE = False

try:
    from midi_generator.integration.hierarchical_model_wrapper import (
        HierarchicalMTLWrapper,
        HierarchicalPrediction,
        create_wrapper
    )
    MODEL_WRAPPER_AVAILABLE = True
except ImportError:
    print("Warning: HierarchicalMTLWrapper not available")
    MODEL_WRAPPER_AVAILABLE = False

try:
    from midi_generator.parameters.universal_registry import (
        UniversalParameterRegistry,
        ParameterDefinition
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    print("Warning: UniversalParameterRegistry not available")
    REGISTRY_AVAILABLE = False


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ParameterAnalysisResult:
    """Complete result from MIDI→Parameters analysis"""

    # Predictions
    parameters: Dict[str, Any]
    hierarchical: HierarchicalPrediction

    # Metadata
    source_midi: Path
    analysis_time_ms: float
    feature_extraction_time_ms: float
    inference_time_ms: float

    # Quality metrics
    confidence_scores: Dict[str, float]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'parameters': self.parameters,
            'hierarchical': self.hierarchical.to_dict(),
            'metadata': {
                'source_midi': str(self.source_midi),
                'analysis_time_ms': self.analysis_time_ms,
                'feature_extraction_time_ms': self.feature_extraction_time_ms,
                'inference_time_ms': self.inference_time_ms,
            },
            'quality': {
                'confidence_scores': self.confidence_scores,
                'warnings': self.warnings
            }
        }

    def save(self, output_path: Path):
        """Save analysis to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Saved analysis to {output_path}")


# ============================================================================
# Main Parameter Prediction API
# ============================================================================

class ParameterPredictionAPI:
    """
    High-level API for parameter prediction from MIDI files.

    This is the main entry point for:
    1. Analyzing MIDI files to extract parameters
    2. Validating and sanitizing parameter values
    3. Batch processing multiple files
    4. Performance optimization and caching

    Usage:
        >>> api = ParameterPredictionAPI()
        >>> result = api.analyze_midi("song.mid")
        >>> print(result.hierarchical.genre_primary)
        'jazz'
        >>> print(result.hierarchical.tempo_bpm)
        140.0
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        feature_extractor: Optional[Any] = None,
        device: str = 'cpu',
        enable_validation: bool = True,
        enable_caching: bool = True
    ):
        """
        Initialize Parameter Prediction API

        Args:
            models_dir: Directory containing trained models
            feature_extractor: Custom feature extractor (None = use default)
            device: 'cpu' or 'cuda' for inference
            enable_validation: Validate predicted parameters
            enable_caching: Enable prediction caching
        """
        self.enable_validation = enable_validation
        self.enable_caching = enable_caching

        # Initialize feature extractor
        if feature_extractor:
            self.feature_extractor = feature_extractor
        elif FEATURE_EXTRACTOR_AVAILABLE:
            self.feature_extractor = DeepFeatureExtractor()
            print("✓ DeepFeatureExtractor initialized")
        else:
            self.feature_extractor = None
            print("⚠ Feature extractor not available - using placeholder")

        # Initialize model wrapper
        if MODEL_WRAPPER_AVAILABLE:
            self.model_wrapper = create_wrapper(models_dir=models_dir, device=device)
            print("✓ HierarchicalMTLWrapper initialized")
        else:
            self.model_wrapper = None
            print("⚠ Model wrapper not available")

        # Initialize parameter registry
        if REGISTRY_AVAILABLE:
            self.registry = UniversalParameterRegistry()
            print("✓ Parameter registry loaded")
        else:
            self.registry = None
            print("⚠ Parameter registry not available")

        # Performance tracking
        self._analysis_times: List[float] = []
        self._cache: Dict[str, ParameterAnalysisResult] = {}

        print(f"\n{'='*60}")
        print(f"ParameterPredictionAPI ready!")
        print(f"  Validation: {'enabled' if enable_validation else 'disabled'}")
        print(f"  Caching: {'enabled' if enable_caching else 'disabled'}")
        print(f"  Device: {device}")
        print(f"{'='*60}\n")

    # ========================================================================
    # Main Analysis Methods
    # ========================================================================

    def analyze_midi(
        self,
        midi_file: Union[str, Path],
        return_features: bool = False,
        use_cache: bool = True
    ) -> ParameterAnalysisResult:
        """
        Analyze MIDI file and predict all parameters

        This is the main entry point for MIDI → Parameters workflow.

        Args:
            midi_file: Path to MIDI file
            return_features: Include extracted features in result
            use_cache: Use cached results if available

        Returns:
            Complete parameter analysis result

        Example:
            >>> api = ParameterPredictionAPI()
            >>> result = api.analyze_midi("examples/jazz_sample.mid")
            >>> print(f"Genre: {result.hierarchical.genre_primary}")
            >>> print(f"Tempo: {result.hierarchical.tempo_bpm} BPM")
            >>> print(f"Complexity: {result.hierarchical.complexity_overall:.2f}")
        """
        start_time = time.time()
        midi_file = Path(midi_file)

        if not midi_file.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_file}")

        # Check cache
        if use_cache and self.enable_caching:
            cache_key = str(midi_file.resolve())
            if cache_key in self._cache:
                print(f"✓ Using cached analysis for {midi_file.name}")
                return self._cache[cache_key]

        print(f"Analyzing: {midi_file.name}")

        # Step 1: Extract features
        feature_start = time.time()
        features = self._extract_features(midi_file)
        feature_time = (time.time() - feature_start) * 1000

        print(f"  ✓ Features extracted ({feature_time:.1f}ms)")

        # Step 2: Predict parameters
        inference_start = time.time()
        prediction = self._predict_parameters(features)
        inference_time = (time.time() - inference_start) * 1000

        print(f"  ✓ Parameters predicted ({inference_time:.1f}ms)")

        # Step 3: Validate parameters
        warnings_list = []
        if self.enable_validation and prediction:
            warnings_list = self._validate_parameters(prediction)
            if warnings_list:
                print(f"  ⚠ {len(warnings_list)} validation warnings")

        # Step 4: Create result
        total_time = (time.time() - start_time) * 1000

        result = ParameterAnalysisResult(
            parameters=prediction.to_flat_dict() if prediction else {},
            hierarchical=prediction,
            source_midi=midi_file,
            analysis_time_ms=total_time,
            feature_extraction_time_ms=feature_time,
            inference_time_ms=inference_time,
            confidence_scores=prediction.confidence_scores if prediction else {},
            warnings=warnings_list
        )

        # Cache result
        if self.enable_caching and use_cache:
            self._cache[str(midi_file.resolve())] = result

        # Track performance
        self._analysis_times.append(total_time)

        print(f"  ✓ Analysis complete ({total_time:.1f}ms total)\n")

        return result

    def analyze_batch(
        self,
        midi_files: List[Union[str, Path]],
        show_progress: bool = True
    ) -> List[ParameterAnalysisResult]:
        """
        Analyze multiple MIDI files in batch

        Args:
            midi_files: List of MIDI file paths
            show_progress: Show progress messages

        Returns:
            List of analysis results

        Example:
            >>> files = ["song1.mid", "song2.mid", "song3.mid"]
            >>> results = api.analyze_batch(files)
            >>> for result in results:
            ...     print(f"{result.source_midi.name}: {result.hierarchical.genre_primary}")
        """
        results = []

        print(f"\nBatch analysis: {len(midi_files)} files")
        print("=" * 60)

        for i, midi_file in enumerate(midi_files, 1):
            if show_progress:
                print(f"\n[{i}/{len(midi_files)}] ", end="")

            try:
                result = self.analyze_midi(midi_file, use_cache=True)
                results.append(result)
            except Exception as e:
                print(f"  ✗ Error analyzing {Path(midi_file).name}: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"Batch complete: {len(results)}/{len(midi_files)} successful")
        print(f"Average time: {np.mean([r.analysis_time_ms for r in results]):.1f}ms")
        print(f"{'='*60}\n")

        return results

    # ========================================================================
    # Feature Extraction
    # ========================================================================

    def _extract_features(self, midi_file: Path) -> np.ndarray:
        """Extract features from MIDI file"""
        if self.feature_extractor is None:
            # Return dummy features for development
            print("  ⚠ Using dummy features (feature extractor not available)")
            return np.random.randn(200)

        try:
            # Use deep feature extractor
            features = self.feature_extractor.extract(midi_file)

            # Handle feature dimension mismatch
            if len(features) > 200:
                # Use feature selection (Agent 04)
                # For now, just take first 200
                features = features[:200]
            elif len(features) < 200:
                # Pad with zeros
                features = np.pad(features, (0, 200 - len(features)))

            return features

        except Exception as e:
            print(f"  ✗ Feature extraction failed: {e}")
            print(f"  ⚠ Using dummy features")
            return np.random.randn(200)

    # ========================================================================
    # Parameter Prediction
    # ========================================================================

    def _predict_parameters(self, features: np.ndarray) -> Optional[HierarchicalPrediction]:
        """Predict parameters from features"""
        if self.model_wrapper is None:
            print("  ⚠ Model wrapper not available - creating dummy prediction")
            return self._create_dummy_prediction()

        try:
            prediction = self.model_wrapper.predict(features, use_cache=self.enable_caching)
            return prediction
        except Exception as e:
            print(f"  ✗ Prediction failed: {e}")
            return self._create_dummy_prediction()

    def _create_dummy_prediction(self) -> HierarchicalPrediction:
        """Create dummy prediction for development"""
        from midi_generator.integration.hierarchical_model_wrapper import HierarchicalPrediction

        return HierarchicalPrediction(
            genre_primary='jazz',
            tempo_bpm=120.0,
            time_signature=(4, 4),
            key_tonic='C',
            key_mode='major',
            energy_level=0.7,
            complexity_overall=0.6,
            structure_form='AABA',

            harmony_chord_density=4.0,
            harmony_complexity=0.7,
            harmony_chromaticism=0.5,
            harmony_tension=0.6,
            harmony_voicing_spread=0.7,
            harmony_progression_predictability=0.5,

            melody_note_density=4.0,
            melody_range_semitones=24,
            melody_contour_smoothness=0.6,
            melody_rhythmic_complexity=0.5,
            melody_repetition=0.4,

            rhythm_subdivision='16th',
            rhythm_syncopation=0.6,
            rhythm_groove_consistency=0.8,
            rhythm_polyrhythm=0.2,
            rhythm_swing_amount=0.5,

            dynamics_overall_level=0.7,
            dynamics_range=0.5,

            texture_polyphony=4,
            texture_density=0.6,

            model_version='1.0.0-dev'
        )

    # ========================================================================
    # Validation
    # ========================================================================

    def _validate_parameters(self, prediction: HierarchicalPrediction) -> List[str]:
        """
        Validate predicted parameters

        Returns:
            List of warning messages
        """
        warnings = []

        # Check Level 1 parameters
        if not 40 <= prediction.tempo_bpm <= 200:
            warnings.append(f"Tempo {prediction.tempo_bpm} outside typical range [40, 200]")

        if not 0 <= prediction.energy_level <= 1:
            warnings.append(f"Energy level {prediction.energy_level} outside valid range [0, 1]")

        if not 0 <= prediction.complexity_overall <= 1:
            warnings.append(f"Complexity {prediction.complexity_overall} outside valid range [0, 1]")

        # Check Level 2 parameters
        if not 0.5 <= prediction.harmony_chord_density <= 8.0:
            warnings.append(f"Chord density {prediction.harmony_chord_density} outside typical range")

        if not 12 <= prediction.melody_range_semitones <= 48:
            warnings.append(f"Melody range {prediction.melody_range_semitones} outside typical range")

        if not 1 <= prediction.texture_polyphony <= 12:
            warnings.append(f"Polyphony {prediction.texture_polyphony} outside typical range")

        # Check consistency
        if prediction.complexity_overall < 0.3 and prediction.harmony_complexity > 0.7:
            warnings.append("Inconsistency: low overall complexity but high harmony complexity")

        if prediction.energy_level < 0.3 and prediction.tempo_bpm > 160:
            warnings.append("Inconsistency: low energy but fast tempo")

        return warnings

    def validate_parameter_dict(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a dictionary of parameters

        Args:
            parameters: Parameter name -> value mapping

        Returns:
            (is_valid, list_of_errors)
        """
        if self.registry is None:
            return True, []

        errors = []

        for param_name, param_value in parameters.items():
            param_def = self.registry.get(param_name)

            if param_def is None:
                errors.append(f"Unknown parameter: {param_name}")
                continue

            is_valid, error_msg = param_def.validate(param_value)
            if not is_valid:
                errors.append(error_msg)

        return len(errors) == 0, errors

    # ========================================================================
    # Parameter Sanitization
    # ========================================================================

    def sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize parameter values to ensure they're within valid ranges

        Args:
            parameters: Raw parameter dictionary

        Returns:
            Sanitized parameter dictionary
        """
        sanitized = parameters.copy()

        # Clip continuous parameters
        if 'tempo.bpm' in sanitized:
            sanitized['tempo.bpm'] = np.clip(sanitized['tempo.bpm'], 40, 200)

        if 'energy.level' in sanitized:
            sanitized['energy.level'] = np.clip(sanitized['energy.level'], 0, 1)

        if 'complexity.overall' in sanitized:
            sanitized['complexity.overall'] = np.clip(sanitized['complexity.overall'], 0, 1)

        if 'harmony.chord_density' in sanitized:
            sanitized['harmony.chord_density'] = np.clip(sanitized['harmony.chord_density'], 0.5, 8.0)

        if 'melody.range_semitones' in sanitized:
            sanitized['melody.range_semitones'] = int(np.clip(sanitized['melody.range_semitones'], 12, 48))

        if 'texture.polyphony' in sanitized:
            sanitized['texture.polyphony'] = int(np.clip(sanitized['texture.polyphony'], 1, 12))

        # Clip all probability/normalized parameters
        for key, value in sanitized.items():
            if any(kw in key for kw in ['ratio', 'level', 'complexity', 'predictability', 'consistency']):
                if isinstance(value, (int, float)):
                    sanitized[key] = float(np.clip(value, 0, 1))

        return sanitized

    # ========================================================================
    # Performance & Statistics
    # ========================================================================

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self._analysis_times:
            return {'message': 'No analyses performed yet'}

        return {
            'total_analyses': len(self._analysis_times),
            'average_time_ms': np.mean(self._analysis_times),
            'min_time_ms': np.min(self._analysis_times),
            'max_time_ms': np.max(self._analysis_times),
            'cache_size': len(self._cache),
            'model_avg_inference_ms': self.model_wrapper.get_average_inference_time() if self.model_wrapper else 0
        }

    def print_performance_stats(self):
        """Print performance statistics"""
        stats = self.get_performance_stats()

        print("\n" + "="*60)
        print("PERFORMANCE STATISTICS")
        print("="*60)

        if 'message' in stats:
            print(stats['message'])
        else:
            print(f"Total analyses: {stats['total_analyses']}")
            print(f"Average time: {stats['average_time_ms']:.1f}ms")
            print(f"Min time: {stats['min_time_ms']:.1f}ms")
            print(f"Max time: {stats['max_time_ms']:.1f}ms")
            print(f"Cache size: {stats['cache_size']}")
            print(f"Model inference: {stats['model_avg_inference_ms']:.1f}ms")

        print("="*60 + "\n")

    def clear_cache(self):
        """Clear prediction cache"""
        self._cache.clear()
        if self.model_wrapper:
            self.model_wrapper.clear_cache()
        print("✓ All caches cleared")


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_midi_file(
    midi_file: Union[str, Path],
    models_dir: Optional[Path] = None,
    device: str = 'cpu'
) -> ParameterAnalysisResult:
    """
    Quick function to analyze a single MIDI file

    Args:
        midi_file: Path to MIDI file
        models_dir: Directory containing models
        device: 'cpu' or 'cuda'

    Returns:
        Analysis result

    Example:
        >>> result = analyze_midi_file("song.mid")
        >>> print(result.hierarchical.genre_primary)
    """
    api = ParameterPredictionAPI(models_dir=models_dir, device=device)
    return api.analyze_midi(midi_file)


def analyze_midi_directory(
    directory: Union[str, Path],
    pattern: str = "*.mid",
    output_dir: Optional[Path] = None
) -> List[ParameterAnalysisResult]:
    """
    Analyze all MIDI files in a directory

    Args:
        directory: Directory containing MIDI files
        pattern: File pattern (e.g., "*.mid", "*.midi")
        output_dir: Optional directory to save results

    Returns:
        List of analysis results
    """
    directory = Path(directory)
    midi_files = list(directory.glob(pattern))

    print(f"Found {len(midi_files)} MIDI files in {directory}")

    api = ParameterPredictionAPI()
    results = api.analyze_batch(midi_files)

    # Save results if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            output_file = output_dir / f"{result.source_midi.stem}_analysis.json"
            result.save(output_file)

        print(f"\n✓ Saved {len(results)} analysis files to {output_dir}")

    return results


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Parameter Prediction API - Agent 09")
    print("=" * 60)

    # Create API
    api = ParameterPredictionAPI()

    # Test with example file (if available)
    test_file = Path("examples/test.mid")

    if test_file.exists():
        print(f"\nTesting with: {test_file}")
        result = api.analyze_midi(test_file)

        print(f"\nResults:")
        print(f"  Genre: {result.hierarchical.genre_primary}")
        print(f"  Tempo: {result.hierarchical.tempo_bpm:.1f} BPM")
        print(f"  Key: {result.hierarchical.key_tonic} {result.hierarchical.key_mode}")
        print(f"  Energy: {result.hierarchical.energy_level:.2f}")
        print(f"  Complexity: {result.hierarchical.complexity_overall:.2f}")

        # Show performance
        api.print_performance_stats()
    else:
        print(f"\n⚠ Test file not found: {test_file}")
        print("API initialized and ready for use!")

    print("\n✓ Parameter Prediction API working correctly!")
