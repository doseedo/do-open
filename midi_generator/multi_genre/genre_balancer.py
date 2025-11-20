"""
Genre Balancing Module
Handles genre imbalance in multi-genre datasets

Author: Agent 07
Date: November 20, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging

from .augmentation import GenreAugmentationPipeline

logger = logging.getLogger(__name__)


class GenreBalancer:
    """
    Balance dataset across genres using multiple strategies:
    1. Over-sampling (data augmentation)
    2. Weighted loss functions
    3. Balanced batch sampling

    Example:
        >>> balancer = GenreBalancer(target_samples_per_genre=500)
        >>> balanced_dataset = balancer.balance(
        ...     train_dataset,
        ...     method='augmentation'
        ... )
        >>> class_weights = balancer.compute_class_weights(train_dataset)
    """

    def __init__(
        self,
        target_samples_per_genre: int = 500,
        augmentation_pipelines: Optional[Dict[str, GenreAugmentationPipeline]] = None
    ):
        """
        Initialize GenreBalancer.

        Args:
            target_samples_per_genre: Target number of training samples per genre
            augmentation_pipelines: Dict mapping genre -> GenreAugmentationPipeline
                                   (will create default if None)
        """
        self.target = target_samples_per_genre

        # Initialize augmentation pipelines
        if augmentation_pipelines is None:
            self.pipelines = {
                genre: GenreAugmentationPipeline(genre)
                for genre in ['jazz', 'classical', 'rock', 'electronic', 'pop']
            }
        else:
            self.pipelines = augmentation_pipelines

        logger.info(f"GenreBalancer initialized with target={target_samples_per_genre}")

    def balance(
        self,
        dataset: List[Dict[str, Any]],
        method: str = 'augmentation'
    ) -> List[Dict[str, Any]]:
        """
        Balance dataset across genres.

        Args:
            dataset: List of data samples
            method: Balancing method:
                - 'augmentation': Generate augmented samples
                - 'oversample': Duplicate existing samples
                - 'undersample': Remove samples from majority classes

        Returns:
            Balanced dataset
        """
        if method == 'augmentation':
            return self._balance_by_augmentation(dataset)
        elif method == 'oversample':
            return self._balance_by_oversampling(dataset)
        elif method == 'undersample':
            return self._balance_by_undersampling(dataset)
        else:
            raise ValueError(f"Unknown balancing method: {method}")

    def _balance_by_augmentation(
        self,
        dataset: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Balance by generating augmented versions.

        Args:
            dataset: Original dataset

        Returns:
            Balanced dataset with augmented samples
        """
        logger.info("Balancing dataset by augmentation...")

        # Group by genre
        genre_samples = self._group_by_genre(dataset)

        # Count current samples per genre
        genre_counts = {
            genre: len(samples)
            for genre, samples in genre_samples.items()
        }

        logger.info(f"Current genre counts: {genre_counts}")

        # Calculate augmentation multipliers
        multipliers = {}
        for genre, count in genre_counts.items():
            if count > 0:
                multipliers[genre] = self.target / count
            else:
                multipliers[genre] = 0

        logger.info(f"Augmentation multipliers: {multipliers}")

        # Generate augmented dataset
        balanced_dataset = []

        for genre, samples in genre_samples.items():
            # Add original samples
            balanced_dataset.extend(samples)

            # Calculate how many augmented versions to generate
            multiplier = multipliers[genre]
            num_augmented_per_sample = int(multiplier) - 1  # -1 because we keep original

            if num_augmented_per_sample > 0:
                pipeline = self.pipelines.get(genre)
                if pipeline is None:
                    logger.warning(f"No pipeline for {genre}, skipping augmentation")
                    continue

                logger.info(f"Generating {num_augmented_per_sample} augmented versions "
                          f"per {genre} sample...")

                for sample in samples:
                    # Generate augmented versions
                    augmented_samples = pipeline.augment(
                        sample,
                        num_variations=num_augmented_per_sample
                    )
                    balanced_dataset.extend(augmented_samples)

        logger.info(f"Balanced dataset size: {len(balanced_dataset)} "
                   f"(from {len(dataset)} original)")

        # Verify balance
        self._verify_balance(balanced_dataset)

        return balanced_dataset

    def _balance_by_oversampling(
        self,
        dataset: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Balance by duplicating minority class samples.

        Args:
            dataset: Original dataset

        Returns:
            Balanced dataset with duplicated samples
        """
        logger.info("Balancing dataset by oversampling...")

        genre_samples = self._group_by_genre(dataset)
        balanced_dataset = []

        for genre, samples in genre_samples.items():
            if len(samples) == 0:
                continue

            # Calculate how many copies needed
            num_copies = int(np.ceil(self.target / len(samples)))

            # Replicate samples
            for _ in range(num_copies):
                balanced_dataset.extend(samples)

            # Trim to exact target
            genre_subset = [s for s in balanced_dataset if s.get('genre') == genre]
            if len(genre_subset) > self.target:
                # Remove excess
                balanced_dataset = [
                    s for s in balanced_dataset
                    if s.get('genre') != genre
                ]
                balanced_dataset.extend(genre_subset[:self.target])

        logger.info(f"Balanced dataset size: {len(balanced_dataset)}")

        return balanced_dataset

    def _balance_by_undersampling(
        self,
        dataset: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Balance by removing majority class samples.

        Args:
            dataset: Original dataset

        Returns:
            Balanced dataset with removed samples
        """
        logger.info("Balancing dataset by undersampling...")

        genre_samples = self._group_by_genre(dataset)

        # Find minimum count
        min_count = min(len(samples) for samples in genre_samples.values())
        target = max(min_count, self.target // len(genre_samples))

        logger.info(f"Undersampling to {target} samples per genre")

        balanced_dataset = []

        for genre, samples in genre_samples.items():
            if len(samples) <= target:
                balanced_dataset.extend(samples)
            else:
                # Randomly sample target samples
                indices = np.random.choice(len(samples), target, replace=False)
                balanced_dataset.extend([samples[i] for i in indices])

        logger.info(f"Balanced dataset size: {len(balanced_dataset)}")

        return balanced_dataset

    def compute_class_weights(
        self,
        dataset: List[Dict[str, Any]],
        method: str = 'inverse_frequency'
    ) -> Dict[str, float]:
        """
        Compute class weights for weighted loss function.

        Args:
            dataset: Training dataset
            method: Weighting method:
                - 'inverse_frequency': weight = max_count / count
                - 'effective_samples': weight based on effective sample size

        Returns:
            Dictionary mapping genre -> weight
        """
        genre_samples = self._group_by_genre(dataset)
        genre_counts = {
            genre: len(samples)
            for genre, samples in genre_samples.items()
        }

        if method == 'inverse_frequency':
            max_count = max(genre_counts.values())
            weights = {
                genre: max_count / count if count > 0 else 0
                for genre, count in genre_counts.items()
            }

        elif method == 'effective_samples':
            # Based on Cui et al. 2019 "Class-Balanced Loss Based on Effective Number of Samples"
            beta = 0.9999
            effective_num = {
                genre: (1.0 - beta ** count) / (1.0 - beta) if count > 0 else 0
                for genre, count in genre_counts.items()
            }
            weights = {
                genre: 1.0 / eff_num if eff_num > 0 else 0
                for genre, eff_num in effective_num.items()
            }

        else:
            raise ValueError(f"Unknown weighting method: {method}")

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {
            genre: weight / total_weight * len(weights)
            for genre, weight in weights.items()
        }

        logger.info(f"Class weights ({method}): {weights}")

        return weights

    def create_balanced_sampler(
        self,
        dataset: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> 'BalancedGenreSampler':
        """
        Create a sampler that ensures balanced genre representation in batches.

        Args:
            dataset: Training dataset
            batch_size: Batch size

        Returns:
            BalancedGenreSampler instance
        """
        return BalancedGenreSampler(dataset, batch_size)

    def _group_by_genre(
        self,
        dataset: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict]]:
        """Group samples by genre."""
        genre_samples = defaultdict(list)
        for sample in dataset:
            genre = sample.get('genre', 'unknown')
            genre_samples[genre].append(sample)
        return dict(genre_samples)

    def _verify_balance(self, dataset: List[Dict[str, Any]]):
        """Verify that dataset is balanced."""
        genre_counts = defaultdict(int)
        for sample in dataset:
            genre = sample.get('genre', 'unknown')
            genre_counts[genre] += 1

        logger.info("Final genre distribution:")
        for genre, count in sorted(genre_counts.items()):
            logger.info(f"  {genre}: {count} samples")


class BalancedGenreSampler:
    """
    Sampler that ensures each batch has equal representation from all genres.

    Example:
        >>> sampler = BalancedGenreSampler(dataset, batch_size=32)
        >>> for batch in sampler:
        ...     # batch has ~6-7 samples from each of 5 genres
        ...     process(batch)
    """

    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        batch_size: int = 32
    ):
        """
        Initialize sampler.

        Args:
            dataset: Training dataset
            batch_size: Batch size
        """
        self.dataset = dataset
        self.batch_size = batch_size

        # Group by genre
        self.genre_samples = defaultdict(list)
        for i, sample in enumerate(dataset):
            genre = sample.get('genre', 'unknown')
            self.genre_samples[genre].append(i)

        self.genres = list(self.genre_samples.keys())
        self.samples_per_genre = batch_size // len(self.genres)

        # Shuffle indices within each genre
        for genre in self.genres:
            np.random.shuffle(self.genre_samples[genre])

        # Track current position in each genre
        self.genre_positions = {genre: 0 for genre in self.genres}

        logger.info(f"BalancedGenreSampler: {len(self.genres)} genres, "
                   f"{self.samples_per_genre} samples/genre per batch")

    def sample_batch(self) -> List[Dict[str, Any]]:
        """
        Sample a balanced batch.

        Returns:
            List of samples forming a balanced batch
        """
        batch = []

        for genre in self.genres:
            indices = self.genre_samples[genre]
            pos = self.genre_positions[genre]

            # Sample from this genre
            for _ in range(self.samples_per_genre):
                batch.append(self.dataset[indices[pos % len(indices)]])
                pos += 1

            self.genre_positions[genre] = pos

        # Shuffle batch
        np.random.shuffle(batch)

        return batch

    def __iter__(self):
        """Iterator interface."""
        while True:
            yield self.sample_batch()

    def __len__(self):
        """Approximate number of batches per epoch."""
        min_genre_samples = min(len(indices) for indices in self.genre_samples.values())
        return min_genre_samples // self.samples_per_genre


if __name__ == '__main__':
    # Example usage with dummy data
    logging.basicConfig(level=logging.INFO)

    # Create imbalanced dummy dataset
    dummy_dataset = []
    genre_counts = {'jazz': 105, 'classical': 140, 'rock': 70, 'electronic': 84, 'pop': 126}

    file_id = 0
    for genre, count in genre_counts.items():
        for _ in range(count):
            dummy_dataset.append({
                'file_id': f"{genre}_{file_id:03d}",
                'genre': genre,
                'notes': [
                    {'pitch': 60, 'velocity': 80, 'start': 0.0, 'end': 0.5}
                ],
                'tempo_bpm': 120,
                'key': 'C'
            })
            file_id += 1

    print(f"Original dataset: {len(dummy_dataset)} samples")
    print(f"Genre distribution: {genre_counts}")

    # Test balancer
    balancer = GenreBalancer(target_samples_per_genre=500)

    # Compute class weights
    weights = balancer.compute_class_weights(dummy_dataset)
    print(f"\nClass weights: {weights}")

    # Balance by augmentation
    print("\n--- Testing augmentation balancing ---")
    balanced = balancer.balance(dummy_dataset, method='augmentation')
    print(f"Balanced dataset: {len(balanced)} samples")

    # Test balanced sampler
    print("\n--- Testing balanced sampler ---")
    sampler = balancer.create_balanced_sampler(dummy_dataset, batch_size=25)
    batch = sampler.sample_batch()
    print(f"Batch size: {len(batch)}")

    batch_genres = defaultdict(int)
    for sample in batch:
        batch_genres[sample['genre']] += 1
    print(f"Batch genre distribution: {dict(batch_genres)}")
