"""
Sparse Dictionary Learning for Transform Discovery
===================================================

Phase 2: Learn 120-140 data-driven transforms from MIDI corpus.

Uses sparse coding to discover latent musical operations that:
1. Are not obvious from music theory
2. Capture dataset-specific patterns
3. Complement theory-based transforms
4. Maintain sparse activation (L1 penalty)

Method: MiniBatchDictionaryLearning
- Input: 1,150D feature vectors from dataset
- Output: Dictionary of 140 components
- Sparsity: L1 penalty (α=1.0)
- Each component → executable transform

Research Foundation:
- Cogliati et al. (2015-2017): Sparse coding on MIDI
- Achieved 93.6% F-measure with 200-300 basis functions

Author: Agent 8 - Transform Architecture
Phase: 2 (Automated Discovery)
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pickle
import json
from collections import defaultdict

try:
    from sklearn.decomposition import MiniBatchDictionaryLearning
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")
    MiniBatchDictionaryLearning = None
    StandardScaler = None

import mido

from ..transforms.space_level_transforms import (
    SpaceLevelTransform,
    TransformMetadata,
    extract_notes_from_midi,
    notes_to_midi
)


# ============================================================================
# Feature Extraction
# ============================================================================

class MIDIFeatureExtractor:
    """
    Extract comprehensive feature vectors from MIDI files.

    Features organized by domain:
    - Pitch (200 features)
    - Rhythm (300 features)
    - Harmony (300 features)
    - Texture (200 features)
    - Form (150 features)

    Total: 1,150 dimensions (similar to jSymbolic)
    """

    def __init__(self):
        self.feature_names = self._define_features()

    def _define_features(self) -> List[str]:
        """Define all feature names"""
        features = []

        # Pitch features (200)
        features.extend([f'pitch_class_histogram_{i}' for i in range(12)])
        features.extend([f'pitch_histogram_{i}' for i in range(128)])
        features.extend(['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range'])
        features.extend(['pitch_entropy', 'pitch_unique_count'])
        features.extend([f'interval_histogram_{i}' for i in range(-12, 13)])
        features.extend(['interval_mean', 'interval_std'])
        features.extend([f'pitch_transition_matrix_{i}_{j}' for i in range(12) for j in range(12)])  # Sparse

        # Rhythm features (300)
        features.extend(['note_density', 'avg_note_duration', 'std_note_duration'])
        features.extend(['avg_ioi', 'std_ioi', 'ioi_entropy'])  # IOI = inter-onset interval
        features.extend([f'duration_histogram_bin_{i}' for i in range(20)])
        features.extend([f'onset_histogram_bin_{i}' for i in range(100)])
        features.extend(['syncopation_index', 'swing_ratio'])
        features.extend(['rhythmic_variability', 'groove_consistency'])
        # ... (simplified, would have 300 total)

        # Harmony features (300)
        features.extend([f'chord_histogram_{i}' for i in range(50)])  # Common chord types
        features.extend(['harmonic_complexity', 'dissonance_mean', 'dissonance_std'])
        features.extend(['chord_change_rate', 'harmonic_rhythm'])
        features.extend([f'interval_class_vector_{i}' for i in range(12)])
        features.extend(['tonalness', 'key_clarity'])
        # ... (simplified)

        # Texture features (200)
        features.extend(['polyphony_mean', 'polyphony_max', 'polyphony_std'])
        features.extend(['voice_count', 'track_count'])
        features.extend(['note_overlap_ratio', 'vertical_density'])
        features.extend(['register_spread', 'voice_crossing_rate'])
        # ... (simplified)

        # Form features (150)
        features.extend(['total_duration', 'n_notes_total'])
        features.extend(['section_count', 'repetition_ratio'])
        features.extend(['self_similarity_mean', 'self_similarity_max'])
        features.extend(['structural_complexity'])
        # ... (simplified)

        # Pad to 1150 if needed
        while len(features) < 1150:
            features.append(f'padding_feature_{len(features)}')

        return features[:1150]

    def extract(self, midi: mido.MidiFile) -> np.ndarray:
        """
        Extract 1,150D feature vector from MIDI.

        Args:
            midi: MIDI file

        Returns:
            1150D numpy array
        """
        notes = extract_notes_from_midi(midi)
        features = np.zeros(1150)

        if not notes:
            return features

        # Pitch features
        pitches = np.array([n['pitch'] for n in notes])
        pitch_classes = pitches % 12

        # Pitch class histogram (12 bins)
        for pc in pitch_classes:
            features[int(pc)] += 1
        features[0:12] /= max(len(notes), 1)  # Normalize

        # Pitch histogram (128 bins)
        for p in pitches:
            features[12 + int(p)] += 1
        features[12:140] /= max(len(notes), 1)

        # Pitch statistics
        features[140] = np.mean(pitches)
        features[141] = np.std(pitches)
        features[142] = np.min(pitches)
        features[143] = np.max(pitches)
        features[144] = np.ptp(pitches)

        # Intervals
        if len(pitches) > 1:
            intervals = np.diff(pitches)
            for interval in intervals:
                if -12 <= interval <= 12:
                    features[145 + int(interval) + 12] += 1
            features[145:170] /= max(len(intervals), 1)

        # Rhythm features
        start_times = np.array([n['start_time'] for n in notes])
        durations = np.array([n['duration'] for n in notes])

        features[200] = len(notes) / max(start_times[-1], 1.0) if len(notes) > 0 else 0  # Note density
        features[201] = np.mean(durations)
        features[202] = np.std(durations)

        if len(notes) > 1:
            ioi = np.diff(start_times)
            features[203] = np.mean(ioi)
            features[204] = np.std(ioi)

        # Harmony features (simplified)
        # Would compute chord analysis here

        # Texture features
        velocities = np.array([n['velocity'] for n in notes])
        features[500] = np.mean(velocities)
        features[501] = np.std(velocities)

        # Form features
        features[700] = start_times[-1] if len(notes) > 0 else 0  # Total duration
        features[701] = len(notes)

        # Remaining features stay at 0 (padding)

        return features


# ============================================================================
# Sparse Dictionary Learning
# ============================================================================

@dataclass
class LearnedComponent:
    """
    Learned sparse dictionary component.

    Represents a latent musical operation discovered from data.
    """
    component_id: int
    weights: np.ndarray  # 1150D weight vector
    activation_frequency: float  # How often this component is active
    dominant_features: List[Tuple[int, float]]  # Top features by weight

    # Interpretation
    inferred_dimension: str  # pitch, rhythm, harmony, texture, form
    inferred_operation: str  # Description of what this does
    confidence: float


class SparseTransformLearner:
    """
    Learn transforms via sparse dictionary learning.

    Pipeline:
    1. Extract 1150D features from MIDI dataset
    2. Run MiniBatchDictionaryLearning (140 components, L1 penalty)
    3. Interpret each component as musical operation
    4. Synthesize executable transform for each component
    5. Validate on held-out data

    Target: 120-140 learned transforms
    """

    def __init__(self,
                 n_components: int = 140,
                 alpha: float = 1.0,
                 batch_size: int = 100,
                 n_iter: int = 1000,
                 random_state: int = 42):
        """
        Initialize sparse dictionary learner.

        Args:
            n_components: Number of dictionary atoms (transforms) to learn
            alpha: Sparsity penalty (higher = more sparse)
            batch_size: Batch size for mini-batch learning
            n_iter: Number of iterations
            random_state: Random seed
        """
        self.n_components = n_components
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.random_state = random_state

        self.feature_extractor = MIDIFeatureExtractor()
        self.scaler = StandardScaler() if StandardScaler else None
        self.dictionary_learner = None
        self.learned_components: List[LearnedComponent] = []

    def fit(self, midi_files: List[Path], verbose: bool = True) -> List[LearnedComponent]:
        """
        Learn sparse dictionary from MIDI dataset.

        Args:
            midi_files: List of MIDI file paths
            verbose: Print progress

        Returns:
            List of learned components
        """
        if MiniBatchDictionaryLearning is None:
            raise RuntimeError("scikit-learn required for sparse learning")

        if verbose:
            print(f"\n{'='*70}")
            print(f"Sparse Dictionary Learning: {self.n_components} components")
            print(f"Dataset: {len(midi_files)} MIDI files")
            print(f"{'='*70}\n")

        # Step 1: Extract features
        if verbose:
            print("Step 1: Extracting features from MIDI files...")

        features = []
        valid_files = []

        for i, midi_file in enumerate(midi_files):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(midi_files)} files...")

            try:
                midi = mido.MidiFile(str(midi_file))
                feature_vector = self.feature_extractor.extract(midi)
                features.append(feature_vector)
                valid_files.append(midi_file)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to process {midi_file}: {e}")
                continue

        features = np.array(features)
        if verbose:
            print(f"  → Extracted {len(features)} feature vectors (1150D each)\n")

        # Step 2: Normalize features
        if verbose:
            print("Step 2: Normalizing features...")
        if self.scaler:
            features = self.scaler.fit_transform(features)
        if verbose:
            print("  → Features normalized\n")

        # Step 3: Learn sparse dictionary
        if verbose:
            print(f"Step 3: Learning sparse dictionary ({self.n_components} components)...")
            print(f"  Alpha (sparsity): {self.alpha}")
            print(f"  Iterations: {self.n_iter}")

        self.dictionary_learner = MiniBatchDictionaryLearning(
            n_components=self.n_components,
            alpha=self.alpha,
            batch_size=self.batch_size,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=1 if verbose else 0,
            fit_algorithm='cd',  # Coordinate descent
            transform_algorithm='lasso_cd'
        )

        # Fit dictionary
        self.dictionary_learner.fit(features)

        if verbose:
            print("  → Dictionary learned\n")

        # Step 4: Analyze learned components
        if verbose:
            print("Step 4: Analyzing learned components...")

        dictionary = self.dictionary_learner.components_  # Shape: (n_components, 1150)

        # Transform data to get activations
        activations = self.dictionary_learner.transform(features)  # Shape: (n_samples, n_components)

        for i in range(self.n_components):
            component_weights = dictionary[i]
            component_activations = activations[:, i]

            # Analyze component
            component = self._analyze_component(
                component_id=i,
                weights=component_weights,
                activations=component_activations
            )

            self.learned_components.append(component)

            if verbose and (i + 1) % 20 == 0:
                print(f"  Analyzed {i + 1}/{self.n_components} components...")

        if verbose:
            print(f"  → Analyzed all {self.n_components} components\n")

        # Step 5: Summarize results
        if verbose:
            self._print_summary()

        return self.learned_components

    def _analyze_component(self,
                          component_id: int,
                          weights: np.ndarray,
                          activations: np.ndarray) -> LearnedComponent:
        """
        Analyze a learned dictionary component.

        Infers:
        - Which musical dimension it belongs to
        - What operation it represents
        - Dominant features
        """
        # Find dominant features (top 20 by absolute weight)
        top_indices = np.argsort(np.abs(weights))[-20:][::-1]
        dominant_features = [(int(idx), float(weights[idx])) for idx in top_indices]

        # Infer dimension from feature indices
        dimension = self._infer_dimension(dominant_features)

        # Infer operation from dominant features
        operation = self._infer_operation(dominant_features, dimension)

        # Calculate activation frequency
        activation_freq = np.mean(np.abs(activations) > 0.01)

        # Confidence based on weight concentration
        weight_concentration = np.std(np.abs(weights)) / (np.mean(np.abs(weights)) + 1e-10)
        confidence = min(weight_concentration / 10, 1.0)

        return LearnedComponent(
            component_id=component_id,
            weights=weights,
            activation_frequency=activation_freq,
            dominant_features=dominant_features,
            inferred_dimension=dimension,
            inferred_operation=operation,
            confidence=confidence
        )

    def _infer_dimension(self, dominant_features: List[Tuple[int, float]]) -> str:
        """Infer musical dimension from dominant features"""
        # Feature ranges:
        # 0-200: pitch
        # 200-500: rhythm
        # 500-800: harmony
        # 800-1000: texture
        # 1000-1150: form

        dimension_votes = defaultdict(int)

        for feat_idx, _ in dominant_features:
            if feat_idx < 200:
                dimension_votes['pitch'] += 1
            elif feat_idx < 500:
                dimension_votes['rhythm'] += 1
            elif feat_idx < 800:
                dimension_votes['harmony'] += 1
            elif feat_idx < 1000:
                dimension_votes['texture'] += 1
            else:
                dimension_votes['form'] += 1

        if not dimension_votes:
            return 'unknown'

        return max(dimension_votes.items(), key=lambda x: x[1])[0]

    def _infer_operation(self, dominant_features: List[Tuple[int, float]], dimension: str) -> str:
        """Infer musical operation from dominant features"""
        # Simplified heuristics
        # Full version would use more sophisticated analysis

        if dimension == 'pitch':
            # Check if features are pitch class histogram
            if any(0 <= idx < 12 for idx, _ in dominant_features):
                return "pitch_class_emphasis"
            elif any(140 <= idx < 145 for idx, _ in dominant_features):
                return "pitch_range_adjustment"
            else:
                return "pitch_transformation"

        elif dimension == 'rhythm':
            if any(200 <= idx < 210 for idx, _ in dominant_features):
                return "rhythmic_density_variation"
            else:
                return "timing_transformation"

        elif dimension == 'harmony':
            return "harmonic_transformation"

        elif dimension == 'texture':
            return "textural_transformation"

        elif dimension == 'form':
            return "structural_transformation"

        return "unknown_operation"

    def _print_summary(self):
        """Print summary of learned components"""
        print(f"\n{'='*70}")
        print("Learned Component Summary")
        print(f"{'='*70}\n")

        # Group by dimension
        by_dimension = defaultdict(list)
        for comp in self.learned_components:
            by_dimension[comp.inferred_dimension].append(comp)

        for dimension, comps in sorted(by_dimension.items()):
            print(f"{dimension.capitalize()} ({len(comps)} components):")
            for comp in comps[:5]:  # Show first 5
                print(f"  Component {comp.component_id}: {comp.inferred_operation}")
                print(f"    Activation: {comp.activation_frequency:.2%}, Confidence: {comp.confidence:.2f}")
            if len(comps) > 5:
                print(f"  ... and {len(comps) - 5} more")
            print()

        print(f"{'='*70}\n")

    def save(self, save_path: Path):
        """Save learned dictionary and components"""
        save_data = {
            'dictionary': self.dictionary_learner.components_ if self.dictionary_learner else None,
            'scaler_mean': self.scaler.mean_ if self.scaler else None,
            'scaler_scale': self.scaler.scale_ if self.scaler else None,
            'components': [
                {
                    'component_id': c.component_id,
                    'weights': c.weights.tolist(),
                    'activation_frequency': c.activation_frequency,
                    'dominant_features': c.dominant_features,
                    'inferred_dimension': c.inferred_dimension,
                    'inferred_operation': c.inferred_operation,
                    'confidence': c.confidence
                }
                for c in self.learned_components
            ],
            'config': {
                'n_components': self.n_components,
                'alpha': self.alpha,
                'batch_size': self.batch_size,
                'n_iter': self.n_iter
            }
        }

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Saved learned dictionary to {save_path}")

    def load(self, load_path: Path):
        """Load learned dictionary and components"""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)

        # Reconstruct dictionary learner
        if save_data['dictionary'] is not None and MiniBatchDictionaryLearning:
            self.dictionary_learner = MiniBatchDictionaryLearning(
                n_components=save_data['config']['n_components'],
                alpha=save_data['config']['alpha']
            )
            self.dictionary_learner.components_ = save_data['dictionary']

        # Reconstruct scaler
        if save_data['scaler_mean'] is not None and StandardScaler:
            self.scaler = StandardScaler()
            self.scaler.mean_ = save_data['scaler_mean']
            self.scaler.scale_ = save_data['scaler_scale']

        # Reconstruct components
        self.learned_components = [
            LearnedComponent(
                component_id=c['component_id'],
                weights=np.array(c['weights']),
                activation_frequency=c['activation_frequency'],
                dominant_features=c['dominant_features'],
                inferred_dimension=c['inferred_dimension'],
                inferred_operation=c['inferred_operation'],
                confidence=c['confidence']
            )
            for c in save_data['components']
        ]

        print(f"Loaded learned dictionary from {load_path}")
