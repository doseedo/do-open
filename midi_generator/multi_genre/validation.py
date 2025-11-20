"""
Validation and Quality Assurance Module
Validates augmented data and provides genre-specific validation

Author: Agent 07
Date: November 20, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AugmentationValidator:
    """
    Validate quality of augmented MIDI data.

    Checks:
    - File integrity (valid MIDI structure)
    - Musical validity (valid pitches, velocities, timing)
    - Harmonic validity (coherent harmony after transposition)
    - Rhythmic validity (coherent rhythms after tempo scaling)
    - Parameter drift (parameters haven't changed too much)

    Example:
        >>> validator = AugmentationValidator()
        >>> is_valid = validator.validate_augmentation(
        ...     original_midi,
        ...     augmented_midi
        ... )
    """

    def __init__(
        self,
        parameter_drift_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize validator.

        Args:
            parameter_drift_thresholds: Max allowed drift per parameter
        """
        # Default drift thresholds
        self.drift_thresholds = parameter_drift_thresholds or {
            'tempo_bpm': 0.15,  # Within 15% change
            'complexity': 0.10,  # Within 10%
            'harmony.chord_density': 0.10,
            'melody.contour_smoothness': 0.05,
            'rhythm.syncopation': 0.05,
        }

        logger.info("AugmentationValidator initialized")

    def validate_augmentation(
        self,
        original: Dict[str, Any],
        augmented: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of augmented data.

        Args:
            original: Original MIDI data
            augmented: Augmented MIDI data

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # 1. File integrity
        if not self.validate_file_integrity(augmented):
            issues.append("File integrity check failed")

        # 2. Musical validity
        musical_issues = self.validate_musical_validity(augmented)
        issues.extend(musical_issues)

        # 3. Harmonic validity (if applicable)
        if 'key' in augmented:
            harmonic_issues = self.validate_harmonic_validity(augmented)
            issues.extend(harmonic_issues)

        # 4. Rhythmic validity
        rhythmic_issues = self.validate_rhythmic_validity(augmented)
        issues.extend(rhythmic_issues)

        # 5. Parameter drift
        if 'parameters' in original and 'parameters' in augmented:
            drift_issues = self.check_parameter_drift(
                original['parameters'],
                augmented['parameters']
            )
            issues.extend(drift_issues)

        is_valid = len(issues) == 0

        if not is_valid:
            logger.warning(f"Validation failed with {len(issues)} issues")

        return is_valid, issues

    def validate_file_integrity(self, midi_data: Dict[str, Any]) -> bool:
        """Check basic file structure."""
        try:
            # Check required fields
            if 'notes' not in midi_data:
                logger.error("Missing 'notes' field")
                return False

            if not isinstance(midi_data['notes'], list):
                logger.error("'notes' is not a list")
                return False

            if len(midi_data['notes']) == 0:
                logger.error("No notes in file")
                return False

            return True

        except Exception as e:
            logger.error(f"File integrity check failed: {e}")
            return False

    def validate_musical_validity(self, midi_data: Dict[str, Any]) -> List[str]:
        """Check musical validity."""
        issues = []

        for i, note in enumerate(midi_data['notes']):
            # Check pitch range
            if not (0 <= note['pitch'] <= 127):
                issues.append(f"Note {i}: Invalid pitch {note['pitch']}")

            # Check velocity range
            if not (1 <= note['velocity'] <= 127):
                issues.append(f"Note {i}: Invalid velocity {note['velocity']}")

            # Check timing
            if note['start'] < 0:
                issues.append(f"Note {i}: Negative start time {note['start']}")

            if note['end'] <= note['start']:
                issues.append(f"Note {i}: Invalid duration (end <= start)")

        # Check tempo if present
        if 'tempo_bpm' in midi_data:
            tempo = midi_data['tempo_bpm']
            if not (20 <= tempo <= 300):
                issues.append(f"Invalid tempo: {tempo} BPM")

        return issues

    def validate_harmonic_validity(self, midi_data: Dict[str, Any]) -> List[str]:
        """Check harmonic validity."""
        issues = []

        # Check for extremely large intervals (> 2 octaves) in melody
        # which might indicate transposition errors
        notes = sorted(midi_data['notes'], key=lambda n: n['start'])

        for i in range(len(notes) - 1):
            curr_pitch = notes[i]['pitch']
            next_pitch = notes[i + 1]['pitch']
            interval = abs(next_pitch - curr_pitch)

            if interval > 24:  # More than 2 octaves
                issues.append(
                    f"Large melodic interval: {interval} semitones "
                    f"(pitches {curr_pitch} -> {next_pitch})"
                )

        # TODO: More sophisticated harmonic analysis
        # - Detect chords and check validity
        # - Check for coherent key/mode
        # - Verify voice leading if polyphonic

        return issues

    def validate_rhythmic_validity(self, midi_data: Dict[str, Any]) -> List[str]:
        """Check rhythmic validity."""
        issues = []

        # Check for reasonable note durations
        for i, note in enumerate(midi_data['notes']):
            duration = note['end'] - note['start']

            if duration < 0.001:  # Less than 1ms
                issues.append(f"Note {i}: Duration too short ({duration:.4f}s)")

            if duration > 60:  # More than 1 minute
                issues.append(f"Note {i}: Duration too long ({duration:.1f}s)")

        # Check for reasonable time signature if present
        if 'time_signature' in midi_data:
            num, denom = midi_data['time_signature']
            if not (1 <= num <= 32 and denom in [2, 4, 8, 16, 32]):
                issues.append(f"Invalid time signature: {num}/{denom}")

        return issues

    def check_parameter_drift(
        self,
        original_params: Dict[str, Any],
        augmented_params: Dict[str, Any]
    ) -> List[str]:
        """
        Check if parameters have drifted too much.

        Args:
            original_params: Parameters of original MIDI
            augmented_params: Parameters of augmented MIDI

        Returns:
            List of drift issues
        """
        issues = []

        for param, threshold in self.drift_thresholds.items():
            if param not in original_params or param not in augmented_params:
                continue

            original_val = original_params[param]
            augmented_val = augmented_params[param]

            # Skip if values are equal or zero
            if original_val == augmented_val:
                continue

            if original_val == 0:
                continue

            # Compute relative drift
            drift = abs(augmented_val - original_val) / abs(original_val)

            if drift > threshold:
                issues.append(
                    f"Parameter {param} drifted by {drift:.1%} "
                    f"(threshold: {threshold:.1%})"
                )

        return issues


class GenreValidationSplitter:
    """
    Create genre-stratified validation sets.

    Provides methods for:
    - K-fold cross-validation with genre stratification
    - Genre-specific test sets for detailed analysis
    - Hold-out validation sets

    Example:
        >>> splitter = GenreValidationSplitter()
        >>> folds = splitter.k_fold_split(dataset, k=5)
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize splitter.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

        logger.info("GenreValidationSplitter initialized")

    def k_fold_split(
        self,
        dataset: List[Dict[str, Any]],
        k: int = 5
    ) -> List[Tuple[List[Dict], List[Dict]]]:
        """
        Create k-fold splits with genre stratification.

        Args:
            dataset: Full dataset
            k: Number of folds

        Returns:
            List of (train, val) tuples for each fold
        """
        # Group by genre
        genre_samples = defaultdict(list)
        for sample in dataset:
            genre = sample.get('genre', 'unknown')
            genre_samples[genre].append(sample)

        # Create folds for each genre
        genre_folds = {}
        for genre, samples in genre_samples.items():
            # Shuffle
            indices = np.random.permutation(len(samples))
            shuffled = [samples[i] for i in indices]

            # Split into k folds
            fold_size = len(samples) // k
            folds = []
            for i in range(k):
                start = i * fold_size
                end = start + fold_size if i < k - 1 else len(samples)
                folds.append(shuffled[start:end])

            genre_folds[genre] = folds

        # Combine folds across genres
        splits = []
        for fold_idx in range(k):
            val_data = []
            train_data = []

            for genre, folds in genre_folds.items():
                # This fold is validation
                val_data.extend(folds[fold_idx])

                # Other folds are training
                for i in range(k):
                    if i != fold_idx:
                        train_data.extend(folds[i])

            splits.append((train_data, val_data))

        logger.info(f"Created {k}-fold splits")

        return splits

    def create_genre_specific_test_sets(
        self,
        dataset: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict]]:
        """
        Create separate test sets for each genre.

        Useful for detailed per-genre analysis.

        Args:
            dataset: Test dataset

        Returns:
            Dictionary mapping genre -> test samples
        """
        genre_test_sets = defaultdict(list)

        for sample in dataset:
            genre = sample.get('genre', 'unknown')
            genre_test_sets[genre].append(sample)

        logger.info(f"Created genre-specific test sets:")
        for genre, samples in genre_test_sets.items():
            logger.info(f"  {genre}: {len(samples)} samples")

        return dict(genre_test_sets)


class GenreDataStatistics:
    """
    Track and visualize dataset statistics across genres.

    Example:
        >>> stats = GenreDataStatistics()
        >>> summary = stats.compute_genre_distribution(dataset)
        >>> stats.compute_parameter_distributions(dataset, by_genre=True)
    """

    def __init__(self):
        """Initialize statistics tracker."""
        logger.info("GenreDataStatistics initialized")

    def compute_genre_distribution(
        self,
        dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute genre distribution statistics.

        Args:
            dataset: Dataset to analyze

        Returns:
            Dictionary with distribution statistics
        """
        genre_counts = defaultdict(int)
        total = len(dataset)

        for sample in dataset:
            genre = sample.get('genre', 'unknown')
            genre_counts[genre] += 1

        # Compute proportions
        genre_proportions = {
            genre: count / total
            for genre, count in genre_counts.items()
        }

        # Compute imbalance ratio
        max_count = max(genre_counts.values())
        min_count = min(genre_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else 0

        stats = {
            'total_samples': total,
            'genre_counts': dict(genre_counts),
            'genre_proportions': genre_proportions,
            'num_genres': len(genre_counts),
            'imbalance_ratio': imbalance_ratio
        }

        logger.info(f"Genre distribution: {genre_counts}")
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")

        return stats

    def compute_parameter_distributions(
        self,
        dataset: List[Dict[str, Any]],
        by_genre: bool = True
    ) -> Dict[str, Any]:
        """
        Compute parameter distribution statistics.

        Args:
            dataset: Dataset to analyze
            by_genre: Whether to compute per-genre distributions

        Returns:
            Dictionary with parameter statistics
        """
        if not by_genre:
            # Overall statistics
            return self._compute_overall_stats(dataset)
        else:
            # Per-genre statistics
            genre_samples = defaultdict(list)
            for sample in dataset:
                genre = sample.get('genre', 'unknown')
                genre_samples[genre].append(sample)

            genre_stats = {}
            for genre, samples in genre_samples.items():
                genre_stats[genre] = self._compute_overall_stats(samples)

            return genre_stats

    def _compute_overall_stats(
        self,
        dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute overall statistics."""
        stats = {}

        # Collect tempo values
        tempos = [s.get('tempo_bpm', 120) for s in dataset]
        stats['tempo'] = {
            'mean': np.mean(tempos),
            'std': np.std(tempos),
            'min': np.min(tempos),
            'max': np.max(tempos),
            'median': np.median(tempos)
        }

        # Collect complexity values if available
        if any('complexity' in s for s in dataset):
            complexities = [s.get('complexity', 0.5) for s in dataset]
            stats['complexity'] = {
                'mean': np.mean(complexities),
                'std': np.std(complexities),
                'min': np.min(complexities),
                'max': np.max(complexities),
                'median': np.median(complexities)
            }

        # Count subgenres if available
        if any('subgenre' in s for s in dataset):
            subgenre_counts = defaultdict(int)
            for sample in dataset:
                subgenre = sample.get('subgenre', 'unknown')
                subgenre_counts[subgenre] += 1
            stats['subgenres'] = dict(subgenre_counts)

        return stats

    def visualize_genre_balance(
        self,
        dataset: List[Dict[str, Any]]
    ) -> str:
        """
        Create ASCII visualization of genre balance.

        Args:
            dataset: Dataset to visualize

        Returns:
            String with bar chart
        """
        genre_counts = defaultdict(int)
        for sample in dataset:
            genre = sample.get('genre', 'unknown')
            genre_counts[genre] += 1

        max_count = max(genre_counts.values())
        max_bar_length = 50

        lines = ["Genre Distribution:"]
        lines.append("-" * 60)

        for genre, count in sorted(genre_counts.items()):
            proportion = count / max_count
            bar_length = int(proportion * max_bar_length)
            bar = "█" * bar_length
            lines.append(f"{genre:>12} | {bar} {count}")

        return "\n".join(lines)

    def generate_report(
        self,
        dataset: List[Dict[str, Any]]
    ) -> str:
        """
        Generate comprehensive statistics report.

        Args:
            dataset: Dataset to analyze

        Returns:
            String with formatted report
        """
        lines = ["=" * 60]
        lines.append("GENRE DATA STATISTICS REPORT")
        lines.append("=" * 60)

        # Overall stats
        overall = self.compute_genre_distribution(dataset)
        lines.append(f"\nTotal samples: {overall['total_samples']}")
        lines.append(f"Number of genres: {overall['num_genres']}")
        lines.append(f"Imbalance ratio: {overall['imbalance_ratio']:.2f}:1")

        # Genre distribution
        lines.append("\n" + self.visualize_genre_balance(dataset))

        # Per-genre parameter stats
        lines.append("\n\nPer-Genre Parameter Statistics:")
        lines.append("-" * 60)

        genre_param_stats = self.compute_parameter_distributions(dataset, by_genre=True)

        for genre, stats in genre_param_stats.items():
            lines.append(f"\n{genre.upper()}:")
            if 'tempo' in stats:
                lines.append(f"  Tempo: {stats['tempo']['mean']:.1f} ± {stats['tempo']['std']:.1f} BPM")
            if 'complexity' in stats:
                lines.append(f"  Complexity: {stats['complexity']['mean']:.2f} ± {stats['complexity']['std']:.2f}")
            if 'subgenres' in stats:
                lines.append(f"  Subgenres: {len(stats['subgenres'])}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create dummy data
    dummy_dataset = []
    genres = ['jazz', 'classical', 'rock', 'electronic', 'pop']
    counts = [105, 140, 70, 84, 126]

    file_id = 0
    for genre, count in zip(genres, counts):
        for _ in range(count):
            dummy_dataset.append({
                'file_id': f"{genre}_{file_id:03d}",
                'genre': genre,
                'subgenre': f"{genre}_sub_{np.random.randint(1, 4)}",
                'tempo_bpm': np.random.uniform(80, 160),
                'complexity': np.random.uniform(0, 1),
                'notes': [
                    {'pitch': 60, 'velocity': 80, 'start': 0.0, 'end': 0.5}
                ],
                'key': 'C'
            })
            file_id += 1

    # Test validator
    print("=== Testing AugmentationValidator ===")
    validator = AugmentationValidator()

    original = dummy_dataset[0]
    augmented = dummy_dataset[0].copy()
    augmented['notes'][0]['pitch'] = 65  # Transpose

    is_valid, issues = validator.validate_augmentation(original, augmented)
    print(f"Valid: {is_valid}, Issues: {issues}")

    # Test splitter
    print("\n=== Testing GenreValidationSplitter ===")
    splitter = GenreValidationSplitter()
    folds = splitter.k_fold_split(dummy_dataset[:100], k=5)
    print(f"Created {len(folds)} folds")
    print(f"Fold 0: {len(folds[0][0])} train, {len(folds[0][1])} val")

    # Test statistics
    print("\n=== Testing GenreDataStatistics ===")
    stats = GenreDataStatistics()
    report = stats.generate_report(dummy_dataset)
    print(report)
