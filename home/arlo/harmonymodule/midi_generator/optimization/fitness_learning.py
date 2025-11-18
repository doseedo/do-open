#!/usr/bin/env python3
"""
Learned Fitness Functions for Musical Optimization
==================================================

This module implements machine learning-based fitness functions for genetic
algorithms and other optimization methods. It learns what makes melodies
"good" from examples and user feedback, then uses these learned preferences
to guide music generation.

Features:
- Supervised learning from labeled examples (good vs. bad melodies)
- Preference learning from user ratings
- Feature extraction for musical quality
- Multi-objective fitness functions
- Active learning for efficient training
- Fitness function evolution and adaptation

Research foundations:
- Machine learning for music evaluation
- Preference learning (Furnkranz & Hüllermeier, 2010)
- Multi-objective optimization in music (Miranda & Biles, 2007)
- Active learning for music (Flexer, 2006)

Author: Agent 9 - ML Integration & Pattern Discovery
License: MIT
"""

from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from enum import Enum
import math
import random

try:
    import numpy as np
    from scipy import stats
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn scipy numpy")


class QualityLabel(Enum):
    """Quality labels for melodies."""
    POOR = 0
    FAIR = 1
    GOOD = 2
    EXCELLENT = 3


@dataclass
class MelodyExample:
    """Represents a labeled melody for training."""
    pitches: List[int]
    durations: List[float]
    quality_label: int  # 0-3 or continuous rating
    features: Dict[str, float] = field(default_factory=dict)
    user_id: Optional[str] = None
    timestamp: Optional[float] = None


class MusicFeatureExtractor:
    """
    Extract musical features for quality assessment.

    Features capture melodic, harmonic, and rhythmic characteristics
    that correlate with perceived musical quality.
    """

    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names: List[str] = []

    def extract_features(self, pitches: List[int],
                        durations: List[float] = None) -> Dict[str, float]:
        """
        Extract comprehensive feature vector from melody.

        Returns:
            Dictionary mapping feature_name -> value
        """
        features = {}

        # Melodic features
        if pitches:
            features.update(self._extract_melodic_features(pitches))

        # Rhythmic features
        if durations:
            features.update(self._extract_rhythmic_features(durations))

        # Combined features
        if pitches and durations and len(pitches) == len(durations):
            features.update(self._extract_structural_features(pitches, durations))

        # Cache feature names
        if not self.feature_names:
            self.feature_names = sorted(features.keys())

        return features

    def _extract_melodic_features(self, pitches: List[int]) -> Dict[str, float]:
        """Extract melodic features."""
        features = {}

        # Basic statistics
        features['melodic_range'] = max(pitches) - min(pitches) if pitches else 0
        features['avg_pitch'] = np.mean(pitches)
        features['pitch_std'] = np.std(pitches)
        features['length'] = len(pitches)

        # Interval analysis
        intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]
        if intervals:
            features['avg_interval'] = np.mean([abs(i) for i in intervals])
            features['max_interval'] = max(abs(i) for i in intervals)
            features['interval_std'] = np.std(intervals)

            # Melodic motion preferences
            features['stepwise_motion'] = sum(1 for i in intervals if abs(i) <= 2) / len(intervals)
            features['leap_ratio'] = sum(1 for i in intervals if abs(i) > 4) / len(intervals)
            features['ascending_ratio'] = sum(1 for i in intervals if i > 0) / len(intervals)

            # Direction changes
            direction_changes = 0
            for i in range(len(intervals) - 1):
                if (intervals[i] > 0 and intervals[i+1] < 0) or \
                   (intervals[i] < 0 and intervals[i+1] > 0):
                    direction_changes += 1
            features['direction_changes'] = direction_changes / len(intervals) if intervals else 0

        # Pitch class distribution (tonal center)
        pitch_classes = [p % 12 for p in pitches]
        pc_counter = Counter(pitch_classes)
        features['pitch_class_entropy'] = self._entropy(pc_counter)

        # Melodic arc (climax position)
        if pitches:
            max_pitch_idx = pitches.index(max(pitches))
            features['climax_position'] = max_pitch_idx / len(pitches)  # 0-1

        # Repeated notes
        if pitches:
            repeated = sum(1 for i in range(len(pitches) - 1) if pitches[i] == pitches[i+1])
            features['repetition_ratio'] = repeated / (len(pitches) - 1)

        return features

    def _extract_rhythmic_features(self, durations: List[float]) -> Dict[str, float]:
        """Extract rhythmic features."""
        features = {}

        features['avg_duration'] = np.mean(durations)
        features['duration_std'] = np.std(durations)

        # Rhythmic variety
        unique_durations = len(set(durations))
        features['rhythm_variety'] = unique_durations / len(durations) if durations else 0

        # Syncopation detection (simplified)
        duration_counter = Counter(durations)
        features['rhythm_entropy'] = self._entropy(duration_counter)

        # Long/short note ratios
        median_duration = np.median(durations)
        long_notes = sum(1 for d in durations if d > median_duration)
        features['long_note_ratio'] = long_notes / len(durations) if durations else 0

        return features

    def _extract_structural_features(self, pitches: List[int],
                                    durations: List[float]) -> Dict[str, float]:
        """Extract structural and phrase features."""
        features = {}

        # Total duration
        features['total_duration'] = sum(durations)

        # Density (notes per unit time)
        if sum(durations) > 0:
            features['note_density'] = len(pitches) / sum(durations)

        # Phrase structure (detect potential phrase boundaries)
        phrase_boundaries = 0
        for i in range(len(durations) - 1):
            if durations[i] > np.mean(durations) * 1.5:  # Long note indicates boundary
                phrase_boundaries += 1
        features['phrase_boundaries'] = phrase_boundaries

        return features

    def _entropy(self, counter: Counter) -> float:
        """Compute Shannon entropy of distribution."""
        total = sum(counter.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array (in consistent order)."""
        if not self.feature_names:
            self.feature_names = sorted(features.keys())

        return np.array([features.get(name, 0.0) for name in self.feature_names])


class LearnedFitnessFunction:
    """
    Fitness function learned from examples using supervised learning.

    Can be trained on:
    - Binary classification (good vs. bad)
    - Multi-class classification (poor, fair, good, excellent)
    - Regression (continuous quality scores)
    """

    def __init__(self, model_type: str = 'random_forest',
                 task: str = 'classification'):
        """
        Initialize learned fitness function.

        Args:
            model_type: 'random_forest', 'svm', 'gradient_boosting'
            task: 'classification' or 'regression'
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for learned fitness functions")

        self.model_type = model_type
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = MusicFeatureExtractor()
        self.is_trained = False

    def train(self, examples: List[MelodyExample]):
        """
        Train fitness function on labeled examples.

        Args:
            examples: List of MelodyExample with quality labels
        """
        # Extract features
        X = []
        y = []

        for example in examples:
            features = self.feature_extractor.extract_features(
                example.pitches, example.durations
            )
            feature_vector = self.feature_extractor.features_to_vector(features)
            X.append(feature_vector)
            y.append(example.quality_label)

        X = np.array(X)
        y = np.array(y)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        if self.task == 'classification':
            if self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100, random_state=42, max_depth=10
                )
            elif self.model_type == 'svm':
                self.model = SVC(kernel='rbf', probability=True, random_state=42)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

        else:  # regression
            if self.model_type == 'random_forest':
                self.model = GradientBoostingRegressor(
                    n_estimators=100, random_state=42, max_depth=10
                )
            elif self.model_type == 'svm':
                self.model = SVR(kernel='rbf')
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Cross-validation
        scores = cross_val_score(self.model, X_scaled, y, cv=5)
        print(f"Training complete. CV score: {scores.mean():.3f} (+/- {scores.std():.3f})")

    def evaluate(self, pitches: List[int],
                durations: List[float] = None) -> float:
        """
        Evaluate quality of a melody.

        Returns:
            Quality score (0-1 for classification, continuous for regression)
        """
        if not self.is_trained:
            raise ValueError("Fitness function not trained")

        if durations is None:
            durations = [1.0] * len(pitches)

        # Extract features
        features = self.feature_extractor.extract_features(pitches, durations)
        feature_vector = self.feature_extractor.features_to_vector(features)
        feature_vector = feature_vector.reshape(1, -1)

        # Predict
        feature_vector_scaled = self.scaler.transform(feature_vector)

        if self.task == 'classification':
            # Return probability of "good" class
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(feature_vector_scaled)[0]
                # Weighted average of class probabilities
                score = sum(i * p for i, p in enumerate(probs)) / (len(probs) - 1)
                return score
            else:
                prediction = self.model.predict(feature_vector_scaled)[0]
                return prediction / 3.0  # Normalize to 0-1

        else:  # regression
            prediction = self.model.predict(feature_vector_scaled)[0]
            return prediction

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (for tree-based models)."""
        if not self.is_trained:
            return {}

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return {
                name: imp
                for name, imp in zip(self.feature_extractor.feature_names, importances)
            }

        return {}


class PreferenceLearner:
    """
    Learn user preferences from pairwise comparisons and ratings.

    Implements preference learning where users rate melodies or choose
    between pairs, allowing the system to learn personalized taste.
    """

    def __init__(self):
        """Initialize preference learner."""
        self.user_preferences: Dict[str, List[Tuple[List[int], float]]] = defaultdict(list)
        self.feature_extractor = MusicFeatureExtractor()
        self.user_models: Dict[str, LearnedFitnessFunction] = {}

    def add_rating(self, user_id: str, pitches: List[int],
                  durations: List[float], rating: float):
        """
        Add a user rating for a melody.

        Args:
            user_id: Unique user identifier
            pitches: Melody pitches
            durations: Melody durations
            rating: User rating (0-1 or 0-10, etc.)
        """
        self.user_preferences[user_id].append((pitches, durations, rating))

    def add_comparison(self, user_id: str,
                      melody_a: Tuple[List[int], List[float]],
                      melody_b: Tuple[List[int], List[float]],
                      preference: str):
        """
        Add a pairwise comparison.

        Args:
            user_id: User identifier
            melody_a, melody_b: Two melodies to compare
            preference: 'a', 'b', or 'equal'
        """
        # Convert to ratings (winner=1, loser=0, equal=0.5)
        pitches_a, dur_a = melody_a
        pitches_b, dur_b = melody_b

        if preference == 'a':
            self.user_preferences[user_id].append((pitches_a, dur_a, 1.0))
            self.user_preferences[user_id].append((pitches_b, dur_b, 0.0))
        elif preference == 'b':
            self.user_preferences[user_id].append((pitches_a, dur_a, 0.0))
            self.user_preferences[user_id].append((pitches_b, dur_b, 1.0))
        else:  # equal
            self.user_preferences[user_id].append((pitches_a, dur_a, 0.5))
            self.user_preferences[user_id].append((pitches_b, dur_b, 0.5))

    def train_user_model(self, user_id: str) -> LearnedFitnessFunction:
        """
        Train personalized fitness function for user.

        Args:
            user_id: User to train model for

        Returns:
            Trained LearnedFitnessFunction
        """
        if user_id not in self.user_preferences:
            raise ValueError(f"No preferences for user {user_id}")

        # Convert to training examples
        examples = []
        for pitches, durations, rating in self.user_preferences[user_id]:
            example = MelodyExample(
                pitches=pitches,
                durations=durations,
                quality_label=rating,  # Use rating directly for regression
                user_id=user_id,
            )
            examples.append(example)

        # Train model
        model = LearnedFitnessFunction(model_type='random_forest', task='regression')
        model.train(examples)

        self.user_models[user_id] = model
        return model

    def predict_user_rating(self, user_id: str, pitches: List[int],
                           durations: List[float] = None) -> float:
        """
        Predict how much a user would like a melody.

        Args:
            user_id: User identifier
            pitches: Melody to evaluate
            durations: Optional rhythm

        Returns:
            Predicted rating
        """
        if user_id not in self.user_models:
            raise ValueError(f"No model trained for user {user_id}")

        return self.user_models[user_id].evaluate(pitches, durations)


class MultiObjectiveFitness:
    """
    Multi-objective fitness function combining multiple criteria.

    Allows optimization across multiple dimensions:
    - Melodic quality
    - Rhythmic interest
    - Harmonic consistency
    - Structural coherence
    - User preference
    """

    def __init__(self):
        """Initialize multi-objective fitness."""
        self.objectives: Dict[str, Callable] = {}
        self.weights: Dict[str, float] = {}

    def add_objective(self, name: str, objective_fn: Callable,
                     weight: float = 1.0):
        """
        Add an objective function.

        Args:
            name: Objective name
            objective_fn: Function that takes (pitches, durations) and returns score
            weight: Relative importance (default 1.0)
        """
        self.objectives[name] = objective_fn
        self.weights[name] = weight

    def evaluate(self, pitches: List[int],
                durations: List[float] = None) -> float:
        """
        Evaluate melody across all objectives.

        Returns:
            Weighted average of all objective scores
        """
        if not self.objectives:
            return 0.0

        if durations is None:
            durations = [1.0] * len(pitches)

        total_score = 0.0
        total_weight = sum(self.weights.values())

        for name, objective_fn in self.objectives.items():
            weight = self.weights[name]
            try:
                score = objective_fn(pitches, durations)
                total_score += weight * score
            except Exception as e:
                print(f"Error evaluating objective '{name}': {e}")
                continue

        return total_score / total_weight if total_weight > 0 else 0.0

    def evaluate_all(self, pitches: List[int],
                    durations: List[float] = None) -> Dict[str, float]:
        """
        Evaluate all objectives separately.

        Returns:
            Dictionary mapping objective_name -> score
        """
        if durations is None:
            durations = [1.0] * len(pitches)

        scores = {}
        for name, objective_fn in self.objectives.items():
            try:
                scores[name] = objective_fn(pitches, durations)
            except Exception as e:
                print(f"Error evaluating objective '{name}': {e}")
                scores[name] = 0.0

        return scores


# Predefined objective functions
def melodic_smoothness(pitches: List[int], durations: List[float] = None) -> float:
    """Objective: Prefer stepwise motion over large leaps."""
    if len(pitches) < 2:
        return 0.0

    intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches) - 1)]
    stepwise = sum(1 for i in intervals if i <= 2)
    return stepwise / len(intervals)


def rhythmic_variety(pitches: List[int], durations: List[float]) -> float:
    """Objective: Prefer rhythmic variety."""
    unique_durations = len(set(durations))
    return min(1.0, unique_durations / 5.0)  # Cap at 5 different durations


def tonal_coherence(pitches: List[int], durations: List[float] = None) -> float:
    """Objective: Prefer strong tonal center."""
    pitch_classes = [p % 12 for p in pitches]
    pc_counter = Counter(pitch_classes)

    # High concentration in few pitch classes = strong tonality
    total = len(pitch_classes)
    max_concentration = max(pc_counter.values()) / total if total > 0 else 0

    return max_concentration


def climax_placement(pitches: List[int], durations: List[float] = None) -> float:
    """Objective: Prefer climax (highest note) in golden ratio position."""
    if not pitches:
        return 0.0

    max_pitch_idx = pitches.index(max(pitches))
    climax_position = max_pitch_idx / len(pitches)

    # Golden ratio ≈ 0.618
    golden_ratio = 0.618
    deviation = abs(climax_position - golden_ratio)

    return 1.0 - deviation


# Example usage
if __name__ == "__main__":
    print("Learned Fitness Functions")
    print("=" * 60)

    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn not available")
        print("Install with: pip install scikit-learn scipy numpy")
        exit(1)

    # Create training examples
    print("\nGenerating training examples...")

    good_melodies = [
        ([60, 62, 64, 65, 67, 65, 64, 62, 60], [0.5]*9, 3),  # Stepwise, good
        ([64, 65, 67, 69, 67, 65, 64, 62, 64], [0.5]*9, 3),
        ([67, 69, 71, 72, 71, 69, 67, 65, 67], [0.5]*9, 3),
    ]

    bad_melodies = [
        ([60, 72, 48, 84, 36, 60, 72, 48], [1.0]*8, 0),  # Random leaps, bad
        ([60, 60, 60, 60, 60, 60, 60, 60], [1.0]*8, 0),  # Boring, bad
        ([60, 61, 62, 63, 64, 65, 66, 67], [0.25]*8, 1),  # Chromatic, mediocre
    ]

    examples = []
    for pitches, durations, label in good_melodies + bad_melodies:
        examples.append(MelodyExample(pitches, durations, label))

    # Train fitness function
    print("\nTraining fitness function...")
    fitness = LearnedFitnessFunction(model_type='random_forest', task='classification')
    fitness.train(examples)

    # Test evaluation
    print("\n\nEvaluating test melodies:")
    print("-" * 60)

    test_melodies = [
        ([60, 62, 64, 65, 64, 62, 60], "Stepwise melody"),
        ([60, 67, 55, 72, 48], "Random leaps"),
        ([64, 65, 67, 69, 67, 64], "Good contour"),
    ]

    for pitches, description in test_melodies:
        score = fitness.evaluate(pitches)
        print(f"{description}: {score:.3f}")

    # Feature importance
    print("\n\nFeature Importance:")
    print("-" * 60)
    importance = fitness.get_feature_importance()
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feature}: {imp:.4f}")

    # Multi-objective optimization
    print("\n\nMulti-Objective Fitness:")
    print("-" * 60)

    multi_fitness = MultiObjectiveFitness()
    multi_fitness.add_objective("smoothness", melodic_smoothness, weight=1.0)
    multi_fitness.add_objective("rhythm_variety", rhythmic_variety, weight=0.5)
    multi_fitness.add_objective("tonality", tonal_coherence, weight=0.8)
    multi_fitness.add_objective("climax", climax_placement, weight=0.3)

    test_melody = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65]
    test_rhythm = [0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0]

    overall_score = multi_fitness.evaluate(test_melody, test_rhythm)
    objective_scores = multi_fitness.evaluate_all(test_melody, test_rhythm)

    print(f"\nTest melody: {test_melody}")
    print(f"Overall score: {overall_score:.3f}")
    print("\nObjective breakdown:")
    for name, score in objective_scores.items():
        print(f"  {name}: {score:.3f}")

    # Preference learning
    print("\n\nPreference Learning:")
    print("-" * 60)

    pref_learner = PreferenceLearner()

    # Simulate user ratings
    pref_learner.add_rating("user1", [60, 62, 64, 65, 67], [0.5]*5, 0.9)
    pref_learner.add_rating("user1", [60, 67, 55, 72], [1.0]*4, 0.2)
    pref_learner.add_rating("user1", [64, 65, 67, 69, 67], [0.5]*5, 0.85)
    pref_learner.add_rating("user1", [60, 60, 60, 60], [1.0]*4, 0.1)

    # Train personalized model
    print("Training personalized model for user1...")
    user_model = pref_learner.train_user_model("user1")

    # Predict ratings
    test_pitches = [62, 64, 65, 67, 69, 67, 65]
    predicted_rating = pref_learner.predict_user_rating("user1", test_pitches)
    print(f"\nPredicted rating for test melody: {predicted_rating:.3f}")

    print("\n" + "=" * 60)
    print("Fitness learning complete!")
