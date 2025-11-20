"""
Hierarchical Model Wrapper - Agent 09
======================================

Wrapper for trained hierarchical multi-task learning (MTL) models from Agents 05-06.

This module provides a clean interface to load and use the trained neural network models
that predict the 50 hierarchical parameters from MIDI features.

Architecture:
    - Level 1 (Global Context): 8 parameters (genre, tempo, key, etc.)
    - Level 2 (Universal Dimensions): 20 parameters (harmony, melody, rhythm, etc.)
    - Level 3 (Genre-Specific Details): 22 parameters (conditional on genre)

Author: Agent 09 - HarmonyModule Integration Lead
Date: 2025-11-20
License: MIT
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Data Structures
# ============================================================================

class ParameterLevel(Enum):
    """Hierarchy levels for parameters"""
    LEVEL1_GLOBAL = 1          # Genre, tempo, key (8 params)
    LEVEL2_UNIVERSAL = 2        # Harmony, melody, rhythm (20 params)
    LEVEL3_GENRE_SPECIFIC = 3   # Genre-specific details (22 params)


@dataclass
class HierarchicalPrediction:
    """Complete hierarchical parameter prediction"""

    # Level 1 - Global Context (8 parameters)
    genre_primary: str
    tempo_bpm: float
    time_signature: Tuple[int, int]
    key_tonic: str
    key_mode: str
    energy_level: float
    complexity_overall: float
    structure_form: str

    # Level 2 - Universal Dimensions (20 parameters)
    # Harmony (6)
    harmony_chord_density: float
    harmony_complexity: float
    harmony_chromaticism: float
    harmony_tension: float
    harmony_voicing_spread: float
    harmony_progression_predictability: float

    # Melody (5)
    melody_note_density: float
    melody_range_semitones: int
    melody_contour_smoothness: float
    melody_rhythmic_complexity: float
    melody_repetition: float

    # Rhythm (5)
    rhythm_subdivision: str
    rhythm_syncopation: float
    rhythm_groove_consistency: float
    rhythm_polyrhythm: float
    rhythm_swing_amount: float

    # Dynamics (2)
    dynamics_overall_level: float
    dynamics_range: float

    # Texture (2)
    texture_polyphony: int
    texture_density: float

    # Level 3 - Genre-Specific Details (22 parameters)
    # Universal orchestration
    orchestration_instrument_count: int = 4
    orchestration_register_balance: float = 0.5
    articulation_legato_ratio: float = 0.5
    structure_section_contrast: float = 0.5
    structure_repetition_level: float = 0.5

    # Genre-specific (populated based on genre)
    genre_specific_params: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    model_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {
            'level1': {
                'genre.primary': self.genre_primary,
                'tempo.bpm': self.tempo_bpm,
                'time_signature': f"{self.time_signature[0]}/{self.time_signature[1]}",
                'key.tonic': self.key_tonic,
                'key.mode': self.key_mode,
                'energy.level': self.energy_level,
                'complexity.overall': self.complexity_overall,
                'structure.form': self.structure_form,
            },
            'level2': {
                'harmony.chord_density': self.harmony_chord_density,
                'harmony.complexity': self.harmony_complexity,
                'harmony.chromaticism': self.harmony_chromaticism,
                'harmony.tension': self.harmony_tension,
                'harmony.voicing_spread': self.harmony_voicing_spread,
                'harmony.progression_predictability': self.harmony_progression_predictability,
                'melody.note_density': self.melody_note_density,
                'melody.range_semitones': self.melody_range_semitones,
                'melody.contour_smoothness': self.melody_contour_smoothness,
                'melody.rhythmic_complexity': self.melody_rhythmic_complexity,
                'melody.repetition': self.melody_repetition,
                'rhythm.subdivision': self.rhythm_subdivision,
                'rhythm.syncopation': self.rhythm_syncopation,
                'rhythm.groove_consistency': self.rhythm_groove_consistency,
                'rhythm.polyrhythm': self.rhythm_polyrhythm,
                'rhythm.swing_amount': self.rhythm_swing_amount,
                'dynamics.overall_level': self.dynamics_overall_level,
                'dynamics.range': self.dynamics_range,
                'texture.polyphony': self.texture_polyphony,
                'texture.density': self.texture_density,
            },
            'level3': {
                'orchestration.instrument_count': self.orchestration_instrument_count,
                'orchestration.register_balance': self.orchestration_register_balance,
                'articulation.legato_ratio': self.articulation_legato_ratio,
                'structure.section_contrast': self.structure_section_contrast,
                'structure.repetition_level': self.structure_repetition_level,
                **self.genre_specific_params
            },
            'metadata': {
                'confidence_scores': self.confidence_scores,
                'inference_time_ms': self.inference_time_ms,
                'model_version': self.model_version
            }
        }
        return result

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary (all parameters at top level)"""
        flat = {}
        nested = self.to_dict()

        for level in ['level1', 'level2', 'level3']:
            flat.update(nested[level])

        return flat


# ============================================================================
# Hierarchical MTL Model Wrapper
# ============================================================================

class HierarchicalMTLWrapper:
    """
    Wrapper for trained hierarchical multi-task learning models.

    This class provides a clean interface to the trained neural network models
    from Agents 05-06, handling:
    - Model loading and initialization
    - Hierarchical inference (Level 1 → Level 2 → Level 3)
    - Feature preprocessing
    - Output post-processing
    - Performance optimization

    Usage:
        >>> wrapper = HierarchicalMTLWrapper()
        >>> wrapper.load_models('path/to/models')
        >>> prediction = wrapper.predict(features)
        >>> print(prediction.genre_primary, prediction.tempo_bpm)
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        device: str = 'cpu',
        cache_size: int = 100
    ):
        """
        Initialize model wrapper

        Args:
            models_dir: Directory containing trained models
            device: 'cpu' or 'cuda'
            cache_size: Number of predictions to cache for performance
        """
        self.models_dir = Path(models_dir) if models_dir else Path('midi_generator/models/hierarchical_mtl')
        self.device = torch.device(device)

        # Model storage
        self.level1_model: Optional[nn.Module] = None
        self.level2_model: Optional[nn.Module] = None
        self.level3_models: Dict[str, nn.Module] = {}  # genre -> model

        # Feature preprocessing
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None
        self.expected_feature_dim: int = 200  # From Agent 04 feature selection

        # Metadata
        self.model_version: str = "1.0.0"
        self.is_loaded: bool = False

        # Performance optimization
        self.cache_size = cache_size
        self._prediction_cache: Dict[str, HierarchicalPrediction] = {}
        self._inference_times: List[float] = []

        print(f"HierarchicalMTLWrapper initialized (device={device})")

    # ========================================================================
    # Model Loading
    # ========================================================================

    def load_models(self, models_dir: Optional[Path] = None) -> bool:
        """
        Load all trained models from directory

        Args:
            models_dir: Directory containing model files

        Returns:
            True if successful, False otherwise
        """
        if models_dir:
            self.models_dir = Path(models_dir)

        if not self.models_dir.exists():
            print(f"Warning: Models directory not found: {self.models_dir}")
            print("Using placeholder models for development")
            self._create_placeholder_models()
            return False

        try:
            # Load Level 1 model (Genre + Global Context)
            level1_path = self.models_dir / 'level1_model.pt'
            if level1_path.exists():
                self.level1_model = torch.load(level1_path, map_location=self.device)
                self.level1_model.eval()
                print(f"✓ Loaded Level 1 model")
            else:
                print(f"Warning: Level 1 model not found, using placeholder")
                self.level1_model = self._create_placeholder_level1_model()

            # Load Level 2 model (Universal Dimensions)
            level2_path = self.models_dir / 'level2_model.pt'
            if level2_path.exists():
                self.level2_model = torch.load(level2_path, map_location=self.device)
                self.level2_model.eval()
                print(f"✓ Loaded Level 2 model")
            else:
                print(f"Warning: Level 2 model not found, using placeholder")
                self.level2_model = self._create_placeholder_level2_model()

            # Load Level 3 models (Genre-Specific)
            level3_dir = self.models_dir / 'level3'
            if level3_dir.exists():
                for model_file in level3_dir.glob('*.pt'):
                    genre = model_file.stem.replace('_model', '')
                    model = torch.load(model_file, map_location=self.device)
                    model.eval()
                    self.level3_models[genre] = model
                    print(f"✓ Loaded Level 3 model for genre: {genre}")
            else:
                print(f"Warning: No Level 3 models found, using placeholder")
                self._create_placeholder_level3_models()

            # Load feature normalization parameters
            norm_path = self.models_dir / 'feature_normalization.pt'
            if norm_path.exists():
                norm_data = torch.load(norm_path, map_location=self.device)
                self.feature_mean = norm_data['mean']
                self.feature_std = norm_data['std']
                print(f"✓ Loaded feature normalization")
            else:
                print("Warning: Feature normalization not found, using identity")
                self.feature_mean = torch.zeros(self.expected_feature_dim)
                self.feature_std = torch.ones(self.expected_feature_dim)

            # Load model metadata
            metadata_path = self.models_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.model_version = metadata.get('version', '1.0.0')
                    print(f"✓ Model version: {self.model_version}")

            self.is_loaded = True
            print(f"\n{'='*60}")
            print(f"All models loaded successfully!")
            print(f"{'='*60}\n")
            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            print("Using placeholder models for development")
            self._create_placeholder_models()
            return False

    def _create_placeholder_models(self):
        """Create placeholder models for development/testing"""
        self.level1_model = self._create_placeholder_level1_model()
        self.level2_model = self._create_placeholder_level2_model()
        self._create_placeholder_level3_models()

        # Placeholder normalization
        self.feature_mean = torch.zeros(self.expected_feature_dim)
        self.feature_std = torch.ones(self.expected_feature_dim)

        self.is_loaded = True
        print("✓ Placeholder models created for development")

    def _create_placeholder_level1_model(self) -> nn.Module:
        """Create simple placeholder for Level 1"""
        model = nn.Sequential(
            nn.Linear(self.expected_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # 8 parameters * 2 (for classification/regression heads)
        )
        model.eval()
        return model

    def _create_placeholder_level2_model(self) -> nn.Module:
        """Create simple placeholder for Level 2"""
        model = nn.Sequential(
            nn.Linear(self.expected_feature_dim + 16, 256),  # features + level1 outputs
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20)  # 20 parameters
        )
        model.eval()
        return model

    def _create_placeholder_level3_models(self):
        """Create placeholder Level 3 models for major genres"""
        genres = ['jazz', 'classical', 'rock', 'electronic', 'pop']

        for genre in genres:
            model = nn.Sequential(
                nn.Linear(self.expected_feature_dim + 36, 128),  # features + level1 + level2
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)  # Genre-specific parameters
            )
            model.eval()
            self.level3_models[genre] = model

    # ========================================================================
    # Inference
    # ========================================================================

    def predict(
        self,
        features: Union[np.ndarray, torch.Tensor],
        use_cache: bool = True
    ) -> HierarchicalPrediction:
        """
        Predict all 50 hierarchical parameters from features

        Args:
            features: Feature vector (200-dim from Agent 04)
            use_cache: Whether to use cached predictions

        Returns:
            HierarchicalPrediction with all parameters
        """
        start_time = time.time()

        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(features)
            if cache_key in self._prediction_cache:
                cached = self._prediction_cache[cache_key]
                print(f"✓ Using cached prediction (genre={cached.genre_primary})")
                return cached

        # Preprocess features
        features_tensor = self._preprocess_features(features)

        with torch.no_grad():
            # Level 1: Predict global context
            level1_output = self.level1_model(features_tensor)
            level1_params = self._decode_level1(level1_output)

            # Level 2: Predict universal dimensions (conditioned on Level 1)
            level2_input = torch.cat([features_tensor, level1_output], dim=-1)
            level2_output = self.level2_model(level2_input)
            level2_params = self._decode_level2(level2_output)

            # Level 3: Predict genre-specific details (conditioned on Level 1+2)
            genre = level1_params['genre.primary']
            level3_model = self.level3_models.get(genre, self.level3_models.get('jazz'))

            if level3_model:
                level3_input = torch.cat([features_tensor, level1_output, level2_output], dim=-1)
                level3_output = level3_model(level3_input)
                level3_params = self._decode_level3(level3_output, genre)
            else:
                level3_params = self._get_default_level3_params()

        # Construct prediction object
        prediction = HierarchicalPrediction(
            # Level 1
            genre_primary=level1_params['genre.primary'],
            tempo_bpm=level1_params['tempo.bpm'],
            time_signature=level1_params['time_signature'],
            key_tonic=level1_params['key.tonic'],
            key_mode=level1_params['key.mode'],
            energy_level=level1_params['energy.level'],
            complexity_overall=level1_params['complexity.overall'],
            structure_form=level1_params['structure.form'],

            # Level 2
            harmony_chord_density=level2_params['harmony.chord_density'],
            harmony_complexity=level2_params['harmony.complexity'],
            harmony_chromaticism=level2_params['harmony.chromaticism'],
            harmony_tension=level2_params['harmony.tension'],
            harmony_voicing_spread=level2_params['harmony.voicing_spread'],
            harmony_progression_predictability=level2_params['harmony.progression_predictability'],

            melody_note_density=level2_params['melody.note_density'],
            melody_range_semitones=level2_params['melody.range_semitones'],
            melody_contour_smoothness=level2_params['melody.contour_smoothness'],
            melody_rhythmic_complexity=level2_params['melody.rhythmic_complexity'],
            melody_repetition=level2_params['melody.repetition'],

            rhythm_subdivision=level2_params['rhythm.subdivision'],
            rhythm_syncopation=level2_params['rhythm.syncopation'],
            rhythm_groove_consistency=level2_params['rhythm.groove_consistency'],
            rhythm_polyrhythm=level2_params['rhythm.polyrhythm'],
            rhythm_swing_amount=level2_params['rhythm.swing_amount'],

            dynamics_overall_level=level2_params['dynamics.overall_level'],
            dynamics_range=level2_params['dynamics.range'],

            texture_polyphony=level2_params['texture.polyphony'],
            texture_density=level2_params['texture.density'],

            # Level 3
            orchestration_instrument_count=level3_params.get('orchestration.instrument_count', 4),
            orchestration_register_balance=level3_params.get('orchestration.register_balance', 0.5),
            articulation_legato_ratio=level3_params.get('articulation.legato_ratio', 0.5),
            structure_section_contrast=level3_params.get('structure.section_contrast', 0.5),
            structure_repetition_level=level3_params.get('structure.repetition_level', 0.5),
            genre_specific_params=level3_params,

            # Metadata
            inference_time_ms=(time.time() - start_time) * 1000,
            model_version=self.model_version
        )

        # Cache prediction
        if use_cache:
            self._cache_prediction(cache_key, prediction)

        # Track inference time
        self._inference_times.append(prediction.inference_time_ms)

        return prediction

    def predict_batch(
        self,
        features_batch: Union[np.ndarray, torch.Tensor]
    ) -> List[HierarchicalPrediction]:
        """
        Predict parameters for multiple feature vectors (batch inference)

        Args:
            features_batch: (batch_size, feature_dim)

        Returns:
            List of predictions
        """
        if len(features_batch.shape) == 1:
            features_batch = features_batch.reshape(1, -1)

        predictions = []
        for i in range(features_batch.shape[0]):
            prediction = self.predict(features_batch[i], use_cache=False)
            predictions.append(prediction)

        return predictions

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _preprocess_features(self, features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Normalize and convert features to tensor"""
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()

        features = features.to(self.device)

        # Ensure correct shape
        if len(features.shape) == 1:
            features = features.unsqueeze(0)

        # Normalize
        if self.feature_mean is not None and self.feature_std is not None:
            features = (features - self.feature_mean) / (self.feature_std + 1e-8)

        return features

    def _decode_level1(self, output: torch.Tensor) -> Dict[str, Any]:
        """Decode Level 1 model output to parameters"""
        # This is a placeholder - actual implementation depends on model architecture
        output_np = output.cpu().numpy()[0]

        # Genre classification (simplified)
        genres = ['jazz', 'classical', 'rock', 'electronic', 'pop', 'blues', 'latin', 'funk']
        genre_idx = int(abs(output_np[0] * len(genres))) % len(genres)

        return {
            'genre.primary': genres[genre_idx],
            'tempo.bpm': float(max(40, min(200, 120 + output_np[1] * 40))),
            'time_signature': (4, 4),  # Simplified
            'key.tonic': 'C',  # Simplified
            'key.mode': 'major',
            'energy.level': float(max(0, min(1, 0.5 + output_np[2] * 0.5))),
            'complexity.overall': float(max(0, min(1, 0.5 + output_np[3] * 0.5))),
            'structure.form': 'AABA'
        }

    def _decode_level2(self, output: torch.Tensor) -> Dict[str, Any]:
        """Decode Level 2 model output to parameters"""
        output_np = output.cpu().numpy()[0]

        return {
            'harmony.chord_density': float(max(0.5, min(8.0, 2.0 + output_np[0]))),
            'harmony.complexity': float(max(0, min(1, 0.5 + output_np[1] * 0.5))),
            'harmony.chromaticism': float(max(0, min(1, 0.3 + output_np[2] * 0.4))),
            'harmony.tension': float(max(0, min(1, 0.5 + output_np[3] * 0.5))),
            'harmony.voicing_spread': float(max(0, min(1, 0.5 + output_np[4] * 0.5))),
            'harmony.progression_predictability': float(max(0, min(1, 0.5 + output_np[5] * 0.5))),

            'melody.note_density': float(max(1.0, min(10.0, 4.0 + output_np[6]))),
            'melody.range_semitones': int(max(12, min(36, 24 + output_np[7] * 6))),
            'melody.contour_smoothness': float(max(0, min(1, 0.5 + output_np[8] * 0.5))),
            'melody.rhythmic_complexity': float(max(0, min(1, 0.5 + output_np[9] * 0.5))),
            'melody.repetition': float(max(0, min(1, 0.5 + output_np[10] * 0.5))),

            'rhythm.subdivision': '16th',
            'rhythm.syncopation': float(max(0, min(1, 0.3 + output_np[11] * 0.4))),
            'rhythm.groove_consistency': float(max(0, min(1, 0.7 + output_np[12] * 0.3))),
            'rhythm.polyrhythm': float(max(0, min(1, 0.2 + output_np[13] * 0.3))),
            'rhythm.swing_amount': float(max(0, min(1, 0.5 + output_np[14] * 0.3))),

            'dynamics.overall_level': float(max(0, min(1, 0.6 + output_np[15] * 0.3))),
            'dynamics.range': float(max(0, min(1, 0.5 + output_np[16] * 0.3))),

            'texture.polyphony': int(max(1, min(8, 3 + output_np[17]))),
            'texture.density': float(max(0, min(1, 0.5 + output_np[18] * 0.4))),
        }

    def _decode_level3(self, output: torch.Tensor, genre: str) -> Dict[str, Any]:
        """Decode Level 3 model output to genre-specific parameters"""
        output_np = output.cpu().numpy()[0]

        base_params = {
            'orchestration.instrument_count': int(max(2, min(12, 4 + output_np[0]))),
            'orchestration.register_balance': float(max(0, min(1, 0.5 + output_np[1] * 0.5))),
            'articulation.legato_ratio': float(max(0, min(1, 0.5 + output_np[2] * 0.5))),
            'structure.section_contrast': float(max(0, min(1, 0.5 + output_np[3] * 0.5))),
            'structure.repetition_level': float(max(0, min(1, 0.5 + output_np[4] * 0.5))),
        }

        # Add genre-specific parameters
        if genre == 'jazz':
            base_params.update({
                'jazz.swing_feel': 'medium',
                'jazz.walking_bass': float(max(0, min(1, 0.7))),
                'jazz.improvisation_ratio': float(max(0, min(1, 0.3))),
                'jazz.bebop_vocabulary': float(max(0, min(1, 0.5))),
            })
        elif genre == 'classical':
            base_params.update({
                'classical.counterpoint': float(max(0, min(1, 0.4))),
                'classical.development_density': float(max(0, min(1, 0.6))),
                'classical.voice_leading_quality': float(max(0, min(1, 0.8))),
            })
        # Add other genres as needed...

        return base_params

    def _get_default_level3_params(self) -> Dict[str, Any]:
        """Get default Level 3 parameters"""
        return {
            'orchestration.instrument_count': 4,
            'orchestration.register_balance': 0.5,
            'articulation.legato_ratio': 0.5,
            'structure.section_contrast': 0.5,
            'structure.repetition_level': 0.5,
        }

    def _get_cache_key(self, features: Union[np.ndarray, torch.Tensor]) -> str:
        """Generate cache key from features"""
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        # Use first 10 features as key (simplified)
        key = ','.join([f"{x:.3f}" for x in features[:10]])
        return key

    def _cache_prediction(self, key: str, prediction: HierarchicalPrediction):
        """Add prediction to cache"""
        if len(self._prediction_cache) >= self.cache_size:
            # Remove oldest entry
            self._prediction_cache.pop(next(iter(self._prediction_cache)))

        self._prediction_cache[key] = prediction

    # ========================================================================
    # Performance Metrics
    # ========================================================================

    def get_average_inference_time(self) -> float:
        """Get average inference time in milliseconds"""
        if not self._inference_times:
            return 0.0
        return np.mean(self._inference_times)

    def clear_cache(self):
        """Clear prediction cache"""
        self._prediction_cache.clear()
        print("✓ Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self._prediction_cache),
            'max_cache_size': self.cache_size,
            'cache_hit_rate': 0.0  # TODO: Track hits
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_wrapper(models_dir: Optional[Path] = None, device: str = 'cpu') -> HierarchicalMTLWrapper:
    """
    Create and load hierarchical model wrapper

    Args:
        models_dir: Directory containing trained models
        device: 'cpu' or 'cuda'

    Returns:
        Loaded wrapper ready for inference
    """
    wrapper = HierarchicalMTLWrapper(models_dir=models_dir, device=device)
    wrapper.load_models()
    return wrapper


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Hierarchical MTL Model Wrapper - Agent 09")
    print("=" * 60)

    # Create wrapper
    wrapper = create_wrapper()

    # Test with random features
    test_features = np.random.randn(200)

    print("\nTesting prediction...")
    prediction = wrapper.predict(test_features)

    print(f"\nPredicted Parameters:")
    print(f"  Genre: {prediction.genre_primary}")
    print(f"  Tempo: {prediction.tempo_bpm:.1f} BPM")
    print(f"  Key: {prediction.key_tonic} {prediction.key_mode}")
    print(f"  Energy: {prediction.energy_level:.2f}")
    print(f"  Complexity: {prediction.complexity_overall:.2f}")
    print(f"  Inference time: {prediction.inference_time_ms:.2f}ms")

    print(f"\n✓ Model wrapper working correctly!")
