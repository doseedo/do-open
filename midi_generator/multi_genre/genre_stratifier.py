"""
Genre Stratification Module
Handles stratified splitting of multi-genre MIDI datasets

Author: Agent 07
Date: November 20, 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class GenreStratifier:
    """
    Handles stratified splitting of multi-genre dataset.

    Ensures proportional representation across multiple dimensions:
    - Primary genre
    - Subgenre
    - Tempo range
    - Key
    - Complexity level

    Example:
        >>> stratifier = GenreStratifier(
        ...     stratify_by=['genre', 'subgenre', 'tempo_range']
        ... )
        >>> train, val, test = stratifier.split(
        ...     dataset,
        ...     train_ratio=0.7,
        ...     val_ratio=0.15,
        ...     test_ratio=0.15
        ... )
    """

    def __init__(
        self,
        stratify_by: List[str] = None,
        random_seed: int = 42
    ):
        """
        Initialize GenreStratifier.

        Args:
            stratify_by: List of fields to stratify by. Options:
                - 'genre': Primary genre
                - 'subgenre': Subgenre within genre
                - 'tempo_range': Tempo binning
                - 'key': Musical key
                - 'complexity': Complexity level
            random_seed: Random seed for reproducibility
        """
        self.stratify_by = stratify_by or ['genre', 'subgenre', 'tempo_range']
        self.random_seed = random_seed
        np.random.seed(random_seed)

        logger.info(f"GenreStratifier initialized with stratify_by={self.stratify_by}")

    def split(
        self,
        dataset: List[Dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset into train/val/test with stratification.

        Args:
            dataset: List of data samples, each a dict with:
                - 'file_id': Unique identifier
                - 'genre': Primary genre
                - 'subgenre': Subgenre (optional)
                - 'tempo_bpm': Tempo in BPM (optional)
                - 'complexity': Complexity score (optional)
                - ... other metadata
            train_ratio: Training set proportion (default 0.7)
            val_ratio: Validation set proportion (default 0.15)
            test_ratio: Test set proportion (default 0.15)

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        logger.info(f"Splitting {len(dataset)} samples into "
                   f"{train_ratio:.1%}/{val_ratio:.1%}/{test_ratio:.1%}")

        # Group samples by stratification keys
        strata = self._create_strata(dataset)

        logger.info(f"Created {len(strata)} strata")

        # Split each stratum
        train_data = []
        val_data = []
        test_data = []

        for stratum_key, samples in strata.items():
            n_samples = len(samples)

            # Shuffle samples within stratum
            indices = np.random.permutation(n_samples)
            shuffled_samples = [samples[i] for i in indices]

            # Calculate split points
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)

            # Split
            stratum_train = shuffled_samples[:n_train]
            stratum_val = shuffled_samples[n_train:n_train + n_val]
            stratum_test = shuffled_samples[n_train + n_val:]

            train_data.extend(stratum_train)
            val_data.extend(stratum_val)
            test_data.extend(stratum_test)

            logger.debug(f"Stratum {stratum_key}: {len(stratum_train)}/{len(stratum_val)}/{len(stratum_test)}")

        # Final shuffle
        np.random.shuffle(train_data)
        np.random.shuffle(val_data)
        np.random.shuffle(test_data)

        logger.info(f"Final split: {len(train_data)}/{len(val_data)}/{len(test_data)}")

        # Validate split
        self.validate_split(dataset, train_data, val_data, test_data)

        return train_data, val_data, test_data

    def _create_strata(
        self,
        dataset: List[Dict[str, Any]]
    ) -> Dict[Tuple, List[Dict]]:
        """
        Group samples into strata based on stratification keys.

        Args:
            dataset: List of data samples

        Returns:
            Dictionary mapping stratum_key -> list of samples
        """
        strata = defaultdict(list)

        for sample in dataset:
            # Build stratum key as tuple
            key_parts = []

            for field in self.stratify_by:
                if field == 'genre':
                    key_parts.append(sample.get('genre', 'unknown'))

                elif field == 'subgenre':
                    key_parts.append(sample.get('subgenre', 'unknown'))

                elif field == 'tempo_range':
                    tempo = sample.get('tempo_bpm', 120)
                    tempo_range = self._get_tempo_range(tempo)
                    key_parts.append(tempo_range)

                elif field == 'key':
                    key_parts.append(sample.get('key', 'unknown'))

                elif field == 'complexity':
                    complexity = sample.get('complexity', 0.5)
                    complexity_level = self._get_complexity_level(complexity)
                    key_parts.append(complexity_level)

                else:
                    logger.warning(f"Unknown stratification field: {field}")
                    key_parts.append('unknown')

            stratum_key = tuple(key_parts)
            strata[stratum_key].append(sample)

        return strata

    def _get_tempo_range(self, tempo: float) -> str:
        """Bin tempo into range categories."""
        if tempo < 80:
            return 'slow'
        elif tempo < 140:
            return 'medium'
        else:
            return 'fast'

    def _get_complexity_level(self, complexity: float) -> str:
        """Bin complexity into level categories."""
        if complexity < 0.33:
            return 'simple'
        elif complexity < 0.67:
            return 'medium'
        else:
            return 'complex'

    def validate_split(
        self,
        original_dataset: List[Dict],
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict]
    ) -> bool:
        """
        Validate that split maintains genre distribution.

        Args:
            original_dataset: Original full dataset
            train_data: Training split
            val_data: Validation split
            test_data: Test split

        Returns:
            True if validation passes

        Raises:
            AssertionError if validation fails
        """
        logger.info("Validating split...")

        # Check: No data leakage
        all_splits = train_data + val_data + test_data
        assert len(all_splits) == len(original_dataset), \
            f"Data loss: {len(all_splits)} != {len(original_dataset)}"

        # Check: No duplicates across splits
        train_ids = {s['file_id'] for s in train_data}
        val_ids = {s['file_id'] for s in val_data}
        test_ids = {s['file_id'] for s in test_data}

        assert len(train_ids & val_ids) == 0, "Data leakage: train-val overlap"
        assert len(train_ids & test_ids) == 0, "Data leakage: train-test overlap"
        assert len(val_ids & test_ids) == 0, "Data leakage: val-test overlap"

        # Check: Genre distribution preserved
        original_genre_dist = self._compute_genre_distribution(original_dataset)
        train_genre_dist = self._compute_genre_distribution(train_data)
        val_genre_dist = self._compute_genre_distribution(val_data)
        test_genre_dist = self._compute_genre_distribution(test_data)

        logger.info("Genre distributions:")
        logger.info(f"  Original: {original_genre_dist}")
        logger.info(f"  Train:    {train_genre_dist}")
        logger.info(f"  Val:      {val_genre_dist}")
        logger.info(f"  Test:     {test_genre_dist}")

        # Allow ±2% deviation
        tolerance = 0.02
        for genre in original_genre_dist.keys():
            original_prop = original_genre_dist[genre]
            train_prop = train_genre_dist.get(genre, 0.0)
            val_prop = val_genre_dist.get(genre, 0.0)
            test_prop = test_genre_dist.get(genre, 0.0)

            assert abs(train_prop - original_prop) < tolerance, \
                f"Train {genre} proportion {train_prop:.2%} deviates from {original_prop:.2%}"
            assert abs(val_prop - original_prop) < tolerance, \
                f"Val {genre} proportion {val_prop:.2%} deviates from {original_prop:.2%}"
            assert abs(test_prop - original_prop) < tolerance, \
                f"Test {genre} proportion {test_prop:.2%} deviates from {original_prop:.2%}"

        logger.info("✓ Split validation passed")
        return True

    def _compute_genre_distribution(
        self,
        dataset: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute genre distribution as proportions.

        Args:
            dataset: List of data samples

        Returns:
            Dictionary mapping genre -> proportion
        """
        genre_counts = defaultdict(int)
        for sample in dataset:
            genre = sample.get('genre', 'unknown')
            genre_counts[genre] += 1

        total = len(dataset)
        genre_dist = {
            genre: count / total
            for genre, count in genre_counts.items()
        }

        return genre_dist

    def get_genre_statistics(
        self,
        dataset: List[Dict]
    ) -> Dict[str, Any]:
        """
        Compute comprehensive genre statistics.

        Args:
            dataset: List of data samples

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_samples': len(dataset),
            'genre_counts': defaultdict(int),
            'subgenre_counts': defaultdict(lambda: defaultdict(int)),
            'tempo_ranges': defaultdict(int),
            'complexity_levels': defaultdict(int)
        }

        for sample in dataset:
            genre = sample.get('genre', 'unknown')
            subgenre = sample.get('subgenre', 'unknown')
            tempo = sample.get('tempo_bpm', 120)
            complexity = sample.get('complexity', 0.5)

            stats['genre_counts'][genre] += 1
            stats['subgenre_counts'][genre][subgenre] += 1
            stats['tempo_ranges'][self._get_tempo_range(tempo)] += 1
            stats['complexity_levels'][self._get_complexity_level(complexity)] += 1

        return dict(stats)


if __name__ == '__main__':
    # Example usage with dummy data
    logging.basicConfig(level=logging.INFO)

    # Create dummy dataset
    dummy_dataset = []
    genres = ['jazz', 'classical', 'rock', 'electronic', 'pop']
    subgenres = {
        'jazz': ['bebop', 'swing', 'modal', 'fusion'],
        'classical': ['baroque', 'romantic', 'contemporary'],
        'rock': ['classic', 'progressive', 'metal'],
        'electronic': ['ambient', 'techno', 'idm'],
        'pop': ['80s', '90s', '2000s']
    }

    file_id = 0
    for genre in genres:
        n_samples = np.random.randint(20, 50)  # Imbalanced
        for _ in range(n_samples):
            dummy_dataset.append({
                'file_id': f"{genre}_{file_id:03d}",
                'genre': genre,
                'subgenre': np.random.choice(subgenres[genre]),
                'tempo_bpm': np.random.uniform(60, 180),
                'complexity': np.random.uniform(0, 1)
            })
            file_id += 1

    print(f"Created dummy dataset with {len(dummy_dataset)} samples")

    # Test stratifier
    stratifier = GenreStratifier(
        stratify_by=['genre', 'subgenre', 'tempo_range']
    )

    train, val, test = stratifier.split(dummy_dataset)

    print(f"\nSplit results:")
    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples")

    # Get statistics
    stats = stratifier.get_genre_statistics(dummy_dataset)
    print(f"\nGenre counts: {dict(stats['genre_counts'])}")
    print(f"Tempo ranges: {dict(stats['tempo_ranges'])}")
    print(f"Complexity levels: {dict(stats['complexity_levels'])}")
