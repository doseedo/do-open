#!/usr/bin/env python3
"""
Corpus-Based Learning for Musical Style Modeling
=================================================

This module implements statistical learning from large MIDI corpora to model
musical styles, composers, and genres. It builds probabilistic models that
capture stylistic characteristics and can generate music in learned styles.

Features:
- Corpus ingestion and preprocessing
- Statistical style modeling (interval, rhythm, harmonic distributions)
- Composer/genre classification
- Style interpolation and hybridization
- Markov chain and n-gram models
- Feature extraction and analysis

Research foundations:
- Cope's Experiments in Musical Intelligence (EMI)
- Music21 corpus analysis methods
- Statistical modeling of musical style (Conklin, 2003)
- Style recognition and classification (Saunders et al., 2004)

Author: Agent 9 - ML Integration & Pattern Discovery
License: MIT
"""

from typing import List, Dict, Tuple, Optional, Set, Any, Callable
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import math
from enum import Enum

try:
    import numpy as np
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn scipy numpy")


class StyleFeature(Enum):
    """Enumeration of musical style features."""
    PITCH_DISTRIBUTION = "pitch_distribution"
    INTERVAL_DISTRIBUTION = "interval_distribution"
    RHYTHM_DISTRIBUTION = "rhythm_distribution"
    CHORD_DISTRIBUTION = "chord_distribution"
    MELODIC_RANGE = "melodic_range"
    HARMONIC_COMPLEXITY = "harmonic_complexity"
    RHYTHMIC_COMPLEXITY = "rhythmic_complexity"
    TEMPO = "tempo"
    KEY_PREFERENCE = "key_preference"
    CADENCE_PATTERNS = "cadence_patterns"


@dataclass
class MIDIFile:
    """Represents a MIDI file with metadata."""
    filepath: str
    composer: Optional[str] = None
    genre: Optional[str] = None
    era: Optional[str] = None
    tempo: Optional[float] = None
    key: Optional[str] = None
    time_signature: Optional[Tuple[int, int]] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StyleModel:
    """Statistical model of a musical style."""
    name: str
    pitch_dist: Counter = field(default_factory=Counter)
    interval_dist: Counter = field(default_factory=Counter)
    rhythm_dist: Counter = field(default_factory=Counter)
    chord_dist: Counter = field(default_factory=Counter)
    n_gram_models: Dict[int, Dict] = field(default_factory=dict)
    statistics: Dict[str, float] = field(default_factory=dict)
    num_pieces: int = 0


class CorpusAnalyzer:
    """
    Analyze musical corpus to extract statistical features.

    Processes MIDI files to extract pitch, rhythm, and harmonic features
    for style characterization.
    """

    def __init__(self):
        """Initialize corpus analyzer."""
        self.files: List[MIDIFile] = []
        self.pitch_sequences: List[List[int]] = []
        self.interval_sequences: List[List[int]] = []
        self.duration_sequences: List[List[float]] = []
        self.chord_sequences: List[List[str]] = []

    def add_file(self, filepath: str, **metadata):
        """
        Add a MIDI file to the corpus.

        Args:
            filepath: Path to MIDI file
            **metadata: Optional metadata (composer, genre, etc.)
        """
        midi_file = MIDIFile(filepath=filepath, **metadata)
        self.files.append(midi_file)

    def analyze_pitch_distribution(self, pitches: List[int]) -> Counter:
        """
        Analyze pitch class distribution.

        Returns:
            Counter of pitch classes (0-11)
        """
        pitch_classes = [p % 12 for p in pitches]
        return Counter(pitch_classes)

    def analyze_interval_distribution(self, pitches: List[int]) -> Counter:
        """
        Analyze melodic interval distribution.

        Returns:
            Counter of intervals (in semitones)
        """
        intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]
        return Counter(intervals)

    def analyze_rhythm_distribution(self, durations: List[float]) -> Counter:
        """
        Analyze rhythmic value distribution.

        Returns:
            Counter of quantized durations
        """
        quantized = [self._quantize_duration(d) for d in durations]
        return Counter(quantized)

    def compute_melodic_range(self, pitches: List[int]) -> int:
        """Compute melodic range in semitones."""
        if not pitches:
            return 0
        return max(pitches) - min(pitches)

    def compute_interval_entropy(self, intervals: List[int]) -> float:
        """
        Compute Shannon entropy of interval distribution.

        Higher entropy indicates more diverse/unpredictable melodic motion.
        """
        if not intervals:
            return 0.0

        counter = Counter(intervals)
        total = len(intervals)

        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def compute_rhythmic_complexity(self, durations: List[float]) -> float:
        """
        Compute rhythmic complexity metric.

        Based on duration variety and syncopation.
        """
        if not durations:
            return 0.0

        # Number of unique duration values
        unique_durations = len(set(durations))

        # Entropy of duration distribution
        counter = Counter(durations)
        total = len(durations)
        entropy = sum(-count/total * math.log2(count/total)
                     for count in counter.values() if count > 0)

        # Combine metrics
        complexity = (unique_durations / 10.0) * entropy
        return complexity

    def _quantize_duration(self, duration: float) -> float:
        """Quantize duration to nearest common rhythmic value."""
        common_durations = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        return min(common_durations, key=lambda x: abs(x - duration))

    def extract_features(self, pitches: List[int],
                        durations: List[float] = None) -> Dict[str, float]:
        """
        Extract comprehensive feature vector from sequence.

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        # Pitch features
        if pitches:
            pitch_dist = self.analyze_pitch_distribution(pitches)
            intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]

            features['melodic_range'] = self.compute_melodic_range(pitches)
            features['avg_pitch'] = np.mean(pitches)
            features['pitch_std'] = np.std(pitches)

            # Interval features
            if intervals:
                features['avg_interval'] = np.mean([abs(i) for i in intervals])
                features['interval_entropy'] = self.compute_interval_entropy(intervals)
                features['stepwise_ratio'] = sum(1 for i in intervals if abs(i) <= 2) / len(intervals)
                features['leap_ratio'] = sum(1 for i in intervals if abs(i) > 4) / len(intervals)

            # Pitch class distribution (normalized)
            total_pitches = len(pitches)
            for pc in range(12):
                features[f'pc_{pc}'] = pitch_dist.get(pc, 0) / total_pitches

        # Rhythm features
        if durations:
            features['rhythmic_complexity'] = self.compute_rhythmic_complexity(durations)
            features['avg_duration'] = np.mean(durations)
            features['duration_std'] = np.std(durations)

        return features

    def compare_distributions(self, dist1: Counter, dist2: Counter) -> float:
        """
        Compare two distributions using Kullback-Leibler divergence.

        Returns:
            KL divergence (lower = more similar)
        """
        # Get all keys
        all_keys = set(dist1.keys()) | set(dist2.keys())

        # Normalize distributions
        total1 = sum(dist1.values())
        total2 = sum(dist2.values())

        if total1 == 0 or total2 == 0:
            return float('inf')

        kl_div = 0.0
        epsilon = 1e-10  # Smoothing

        for key in all_keys:
            p = (dist1.get(key, 0) + epsilon) / (total1 + epsilon * len(all_keys))
            q = (dist2.get(key, 0) + epsilon) / (total2 + epsilon * len(all_keys))
            kl_div += p * math.log2(p / q)

        return kl_div


class StyleLearner:
    """
    Learn musical styles from corpus using statistical methods.

    Builds probabilistic models of different styles (composers, genres, eras)
    and can generate music or classify pieces by style.
    """

    def __init__(self):
        """Initialize style learner."""
        self.styles: Dict[str, StyleModel] = {}
        self.classifier = None
        self.feature_scaler = None

    def learn_style(self, name: str,
                   pitch_sequences: List[List[int]],
                   duration_sequences: List[List[float]] = None,
                   chord_sequences: List[List[str]] = None) -> StyleModel:
        """
        Learn a style model from training sequences.

        Args:
            name: Style name (composer, genre, etc.)
            pitch_sequences: List of melody sequences
            duration_sequences: Optional rhythm sequences
            chord_sequences: Optional chord progressions

        Returns:
            Learned StyleModel
        """
        model = StyleModel(name=name, num_pieces=len(pitch_sequences))

        # Aggregate pitch distribution
        all_pitches = []
        all_intervals = []
        all_durations = []

        for pitches in pitch_sequences:
            all_pitches.extend(pitches)
            intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]
            all_intervals.extend(intervals)

        if duration_sequences:
            for durations in duration_sequences:
                all_durations.extend(durations)

        # Build distributions
        model.pitch_dist = Counter([p % 12 for p in all_pitches])
        model.interval_dist = Counter(all_intervals)

        if all_durations:
            quantized = [self._quantize_duration(d) for d in all_durations]
            model.rhythm_dist = Counter(quantized)

        if chord_sequences:
            all_chords = [chord for seq in chord_sequences for chord in seq]
            model.chord_dist = Counter(all_chords)

        # Build n-gram models
        model.n_gram_models = self._build_ngram_models(pitch_sequences)

        # Compute statistics
        model.statistics = {
            'avg_pitch': np.mean(all_pitches) if all_pitches else 0,
            'pitch_std': np.std(all_pitches) if all_pitches else 0,
            'avg_interval': np.mean([abs(i) for i in all_intervals]) if all_intervals else 0,
            'interval_entropy': self._compute_entropy(model.interval_dist),
            'melodic_range_avg': np.mean([max(seq) - min(seq) for seq in pitch_sequences if seq]),
            'stepwise_ratio': sum(1 for i in all_intervals if abs(i) <= 2) / len(all_intervals) if all_intervals else 0,
        }

        if all_durations:
            model.statistics['avg_duration'] = np.mean(all_durations)
            model.statistics['rhythmic_complexity'] = self._compute_rhythmic_complexity(all_durations)

        self.styles[name] = model
        return model

    def _build_ngram_models(self, sequences: List[List[int]],
                           max_n: int = 4) -> Dict[int, Dict]:
        """
        Build n-gram transition probabilities.

        Returns:
            Dictionary mapping n -> {context: {next_note: probability}}
        """
        ngram_models = {}

        for n in range(1, max_n + 1):
            transitions = defaultdict(Counter)

            for seq in sequences:
                # Extract intervals
                intervals = [seq[i+1] - seq[i] for i in range(len(seq) - 1)]

                # Build n-grams
                for i in range(len(intervals) - n):
                    context = tuple(intervals[i:i+n])
                    next_interval = intervals[i+n]
                    transitions[context][next_interval] += 1

            # Convert to probabilities
            prob_model = {}
            for context, counter in transitions.items():
                total = sum(counter.values())
                prob_model[context] = {interval: count / total
                                      for interval, count in counter.items()}

            ngram_models[n] = prob_model

        return ngram_models

    def _compute_entropy(self, distribution: Counter) -> float:
        """Compute Shannon entropy of distribution."""
        total = sum(distribution.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in distribution.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _compute_rhythmic_complexity(self, durations: List[float]) -> float:
        """Compute rhythmic complexity metric."""
        if not durations:
            return 0.0

        unique_durations = len(set(durations))
        counter = Counter(durations)
        total = len(durations)
        entropy = sum(-count/total * math.log2(count/total)
                     for count in counter.values() if count > 0)

        return (unique_durations / 10.0) * entropy

    def _quantize_duration(self, duration: float) -> float:
        """Quantize duration to nearest common value."""
        common_durations = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        return min(common_durations, key=lambda x: abs(x - duration))

    def generate_in_style(self, style_name: str,
                         length: int = 16,
                         start_pitch: int = 60,
                         temperature: float = 1.0) -> List[int]:
        """
        Generate a melodic sequence in the learned style.

        Args:
            style_name: Name of style to generate in
            length: Number of notes to generate
            start_pitch: Starting pitch (MIDI note number)
            temperature: Randomness (higher = more random)

        Returns:
            List of MIDI pitch values
        """
        if style_name not in self.styles:
            raise ValueError(f"Style '{style_name}' not learned")

        model = self.styles[style_name]
        pitches = [start_pitch]

        # Use highest order n-gram model available
        max_n = max(model.n_gram_models.keys()) if model.n_gram_models else 1
        ngram_model = model.n_gram_models.get(max_n, {})

        for _ in range(length - 1):
            # Get context from recent intervals
            intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]
            context_length = min(max_n, len(intervals))

            if context_length > 0:
                context = tuple(intervals[-context_length:])

                # Try to find matching context
                if context in ngram_model:
                    next_interval = self._sample_from_distribution(
                        ngram_model[context], temperature
                    )
                else:
                    # Fall back to interval distribution
                    next_interval = self._sample_from_counter(
                        model.interval_dist, temperature
                    )
            else:
                # First interval - sample from distribution
                next_interval = self._sample_from_counter(
                    model.interval_dist, temperature
                )

            next_pitch = pitches[-1] + next_interval

            # Keep in reasonable range
            next_pitch = max(36, min(84, next_pitch))

            pitches.append(next_pitch)

        return pitches

    def _sample_from_distribution(self, prob_dist: Dict[int, float],
                                  temperature: float = 1.0) -> int:
        """Sample from probability distribution with temperature."""
        if not prob_dist:
            return 0

        # Apply temperature
        items = list(prob_dist.items())
        values = [item[0] for item in items]
        probs = [item[1] for item in items]

        if temperature != 1.0:
            # Adjust probabilities by temperature
            probs = [p ** (1.0 / temperature) for p in probs]
            total = sum(probs)
            probs = [p / total for p in probs]

        # Sample
        return np.random.choice(values, p=probs)

    def _sample_from_counter(self, counter: Counter,
                            temperature: float = 1.0) -> int:
        """Sample from Counter distribution."""
        if not counter:
            return 0

        total = sum(counter.values())
        prob_dist = {k: v / total for k, v in counter.items()}
        return self._sample_from_distribution(prob_dist, temperature)

    def interpolate_styles(self, style1: str, style2: str,
                          alpha: float = 0.5) -> StyleModel:
        """
        Create hybrid style by interpolating between two styles.

        Args:
            style1, style2: Names of styles to interpolate
            alpha: Interpolation factor (0 = pure style1, 1 = pure style2)

        Returns:
            New hybrid StyleModel
        """
        if style1 not in self.styles or style2 not in self.styles:
            raise ValueError("Both styles must be learned first")

        m1 = self.styles[style1]
        m2 = self.styles[style2]

        hybrid_name = f"{style1}_{alpha:.2f}_{style2}"
        hybrid = StyleModel(name=hybrid_name)

        # Interpolate distributions
        hybrid.pitch_dist = self._interpolate_counters(
            m1.pitch_dist, m2.pitch_dist, alpha
        )
        hybrid.interval_dist = self._interpolate_counters(
            m1.interval_dist, m2.interval_dist, alpha
        )
        hybrid.rhythm_dist = self._interpolate_counters(
            m1.rhythm_dist, m2.rhythm_dist, alpha
        )

        # Interpolate statistics
        for key in m1.statistics.keys() & m2.statistics.keys():
            hybrid.statistics[key] = (
                (1 - alpha) * m1.statistics[key] +
                alpha * m2.statistics[key]
            )

        return hybrid

    def _interpolate_counters(self, c1: Counter, c2: Counter,
                             alpha: float) -> Counter:
        """Interpolate between two Counters."""
        result = Counter()
        all_keys = set(c1.keys()) | set(c2.keys())

        for key in all_keys:
            val1 = c1.get(key, 0)
            val2 = c2.get(key, 0)
            result[key] = int((1 - alpha) * val1 + alpha * val2)

        return result

    def compare_styles(self, style1: str, style2: str) -> Dict[str, float]:
        """
        Compare two learned styles using multiple metrics.

        Returns:
            Dictionary of similarity metrics
        """
        if style1 not in self.styles or style2 not in self.styles:
            raise ValueError("Both styles must be learned first")

        m1 = self.styles[style1]
        m2 = self.styles[style2]

        metrics = {}

        # KL divergence of distributions
        metrics['pitch_kl_div'] = self._kl_divergence(m1.pitch_dist, m2.pitch_dist)
        metrics['interval_kl_div'] = self._kl_divergence(m1.interval_dist, m2.interval_dist)
        metrics['rhythm_kl_div'] = self._kl_divergence(m1.rhythm_dist, m2.rhythm_dist)

        # Statistical differences
        for stat_name in ['avg_pitch', 'pitch_std', 'avg_interval', 'stepwise_ratio']:
            if stat_name in m1.statistics and stat_name in m2.statistics:
                diff = abs(m1.statistics[stat_name] - m2.statistics[stat_name])
                metrics[f'{stat_name}_diff'] = diff

        return metrics

    def _kl_divergence(self, dist1: Counter, dist2: Counter) -> float:
        """Compute KL divergence between two distributions."""
        all_keys = set(dist1.keys()) | set(dist2.keys())

        total1 = sum(dist1.values())
        total2 = sum(dist2.values())

        if total1 == 0 or total2 == 0:
            return float('inf')

        kl_div = 0.0
        epsilon = 1e-10

        for key in all_keys:
            p = (dist1.get(key, 0) + epsilon) / (total1 + epsilon * len(all_keys))
            q = (dist2.get(key, 0) + epsilon) / (total2 + epsilon * len(all_keys))
            kl_div += p * math.log2(p / q)

        return kl_div

    def save_model(self, filepath: str):
        """Save learned styles to file."""
        data = {
            name: {
                'pitch_dist': dict(model.pitch_dist),
                'interval_dist': dict(model.interval_dist),
                'rhythm_dist': dict(model.rhythm_dist),
                'chord_dist': dict(model.chord_dist),
                'statistics': model.statistics,
                'num_pieces': model.num_pieces,
            }
            for name, model in self.styles.items()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_model(self, filepath: str):
        """Load learned styles from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        for name, style_data in data.items():
            model = StyleModel(
                name=name,
                pitch_dist=Counter(style_data['pitch_dist']),
                interval_dist=Counter(style_data['interval_dist']),
                rhythm_dist=Counter(style_data['rhythm_dist']),
                chord_dist=Counter(style_data['chord_dist']),
                statistics=style_data['statistics'],
                num_pieces=style_data['num_pieces'],
            )
            self.styles[name] = model


class StyleClassifier:
    """
    Classify musical pieces by style using machine learning.

    Uses feature extraction and supervised learning to identify composers,
    genres, or eras from musical content.
    """

    def __init__(self):
        """Initialize style classifier."""
        self.classifier = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.label_encoder: Dict[str, int] = {}
        self.label_decoder: Dict[int, str] = {}

    def train(self, training_data: List[Tuple[List[int], str]],
             classifier_type: str = 'random_forest'):
        """
        Train classifier on labeled musical sequences.

        Args:
            training_data: List of (pitch_sequence, label) tuples
            classifier_type: 'naive_bayes' or 'random_forest'
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for classification")

        # Extract features
        X = []
        y = []
        analyzer = CorpusAnalyzer()

        for pitches, label in training_data:
            features = analyzer.extract_features(pitches)
            feature_vector = list(features.values())
            X.append(feature_vector)
            y.append(label)

            if not self.feature_names:
                self.feature_names = list(features.keys())

        X = np.array(X)

        # Encode labels
        unique_labels = sorted(set(y))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        y_encoded = np.array([self.label_encoder[label] for label in y])

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train classifier
        if classifier_type == 'naive_bayes':
            self.classifier = GaussianNB()
        elif classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.classifier.fit(X_scaled, y_encoded)

        # Cross-validation score
        scores = cross_val_score(self.classifier, X_scaled, y_encoded, cv=5)
        print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    def predict(self, pitches: List[int]) -> Tuple[str, float]:
        """
        Predict style of a musical sequence.

        Returns:
            Tuple of (predicted_label, confidence)
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained")

        analyzer = CorpusAnalyzer()
        features = analyzer.extract_features(pitches)
        feature_vector = [features.get(name, 0) for name in self.feature_names]
        feature_vector = np.array([feature_vector])

        # Predict
        feature_vector_scaled = self.scaler.transform(feature_vector)
        prediction = self.classifier.predict(feature_vector_scaled)[0]
        probabilities = self.classifier.predict_proba(feature_vector_scaled)[0]

        label = self.label_decoder[prediction]
        confidence = probabilities[prediction]

        return label, confidence

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (for RandomForest only)."""
        if not isinstance(self.classifier, RandomForestClassifier):
            return {}

        importances = self.classifier.feature_importances_
        return {name: imp for name, imp in zip(self.feature_names, importances)}


# Example usage
if __name__ == "__main__":
    print("Corpus-Based Style Learner")
    print("=" * 60)

    # Example: Learn styles from different composers
    bach_sequences = [
        [60, 62, 64, 65, 67, 65, 64, 62, 60],  # Bach-like stepwise motion
        [64, 62, 60, 62, 64, 65, 67, 69, 67],
        [67, 65, 64, 62, 64, 65, 67, 65, 64],
    ]

    mozart_sequences = [
        [60, 64, 67, 72, 67, 64, 60, 64, 67],  # Mozart-like arpeggios
        [65, 69, 72, 69, 65, 69, 72, 76, 72],
        [62, 65, 69, 74, 69, 65, 62, 65, 69],
    ]

    # Initialize learner
    learner = StyleLearner()

    # Learn styles
    print("\nLearning Bach style...")
    bach_model = learner.learn_style("Bach", bach_sequences)
    print(f"  Learned from {bach_model.num_pieces} pieces")
    print(f"  Average interval: {bach_model.statistics['avg_interval']:.2f}")
    print(f"  Stepwise ratio: {bach_model.statistics['stepwise_ratio']:.2f}")

    print("\nLearning Mozart style...")
    mozart_model = learner.learn_style("Mozart", mozart_sequences)
    print(f"  Learned from {mozart_model.num_pieces} pieces")
    print(f"  Average interval: {mozart_model.statistics['avg_interval']:.2f}")
    print(f"  Stepwise ratio: {mozart_model.statistics['stepwise_ratio']:.2f}")

    # Compare styles
    print("\n\nComparing styles:")
    print("-" * 60)
    comparison = learner.compare_styles("Bach", "Mozart")
    for metric, value in comparison.items():
        print(f"  {metric}: {value:.4f}")

    # Generate in styles
    print("\n\nGenerating melodies:")
    print("-" * 60)

    print("\nBach-style melody:")
    bach_gen = learner.generate_in_style("Bach", length=16, start_pitch=60)
    print(f"  {bach_gen}")

    print("\nMozart-style melody:")
    mozart_gen = learner.generate_in_style("Mozart", length=16, start_pitch=60)
    print(f"  {mozart_gen}")

    # Style interpolation
    print("\n\nStyle interpolation:")
    print("-" * 60)
    hybrid = learner.interpolate_styles("Bach", "Mozart", alpha=0.5)
    print(f"Created hybrid style: {hybrid.name}")
    print(f"  Average interval: {hybrid.statistics.get('avg_interval', 0):.2f}")

    # Classification
    if SKLEARN_AVAILABLE:
        print("\n\nStyle Classification:")
        print("-" * 60)

        # Prepare training data
        training_data = []
        for seq in bach_sequences:
            training_data.append((seq, "Bach"))
        for seq in mozart_sequences:
            training_data.append((seq, "Mozart"))

        # Train classifier
        classifier = StyleClassifier()
        print("Training classifier...")
        classifier.train(training_data, classifier_type='random_forest')

        # Test prediction
        test_sequence = [60, 62, 64, 65, 64, 62, 60, 59]  # Bach-like
        label, confidence = classifier.predict(test_sequence)
        print(f"\nTest sequence classified as: {label}")
        print(f"Confidence: {confidence:.2%}")

        # Feature importance
        print("\nFeature importance:")
        importance = classifier.get_feature_importance()
        for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feature}: {imp:.4f}")

    print("\n" + "=" * 60)
    print("Style learning complete!")
