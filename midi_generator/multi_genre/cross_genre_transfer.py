"""
Cross-Genre Transfer Learning Module
Utilities for cross-genre knowledge transfer

Author: Agent 07
Date: November 20, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Genre similarity matrix (based on musical properties)
GENRE_SIMILARITY = {
    'jazz': {
        'jazz': 1.00,
        'classical': 0.40,
        'rock': 0.30,
        'electronic': 0.20,
        'pop': 0.50
    },
    'classical': {
        'jazz': 0.40,
        'classical': 1.00,
        'rock': 0.30,
        'electronic': 0.20,
        'pop': 0.40
    },
    'rock': {
        'jazz': 0.30,
        'classical': 0.30,
        'rock': 1.00,
        'electronic': 0.40,
        'pop': 0.70
    },
    'electronic': {
        'jazz': 0.20,
        'classical': 0.20,
        'rock': 0.40,
        'electronic': 1.00,
        'pop': 0.60
    },
    'pop': {
        'jazz': 0.50,
        'classical': 0.40,
        'rock': 0.70,
        'electronic': 0.60,
        'pop': 1.00
    }
}


class CrossGenreTransfer:
    """
    Utilities for cross-genre transfer learning.

    Features:
    - Identify similar genres for transfer learning
    - Create mixed batches with similar genres
    - Compute transfer learning strategies
    - Domain adaptation utilities

    Example:
        >>> transfer = CrossGenreTransfer()
        >>> similar = transfer.get_transfer_genres('rock', top_k=2)
        >>> # similar = ['pop', 'electronic']
        >>> mixed_batch = transfer.create_mixed_batch(
        ...     dataset, batch_size=32, target_genre='rock'
        ... )
    """

    def __init__(
        self,
        similarity_matrix: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize cross-genre transfer utilities.

        Args:
            similarity_matrix: Custom similarity matrix (uses default if None)
        """
        self.similarity_matrix = similarity_matrix or GENRE_SIMILARITY
        logger.info("CrossGenreTransfer initialized")

    def get_transfer_genres(
        self,
        target_genre: str,
        top_k: int = 2
    ) -> List[str]:
        """
        Get most similar genres for transfer learning.

        Args:
            target_genre: Target genre to improve
            top_k: Number of similar genres to return

        Returns:
            List of similar genre names (excluding target genre itself)
        """
        if target_genre not in self.similarity_matrix:
            logger.warning(f"Unknown genre: {target_genre}")
            return []

        similarities = self.similarity_matrix[target_genre]

        # Sort by similarity (excluding self)
        sorted_genres = sorted(
            [(g, sim) for g, sim in similarities.items() if g != target_genre],
            key=lambda x: x[1],
            reverse=True
        )

        similar_genres = [g for g, sim in sorted_genres[:top_k]]

        logger.info(f"Most similar to {target_genre}: {similar_genres}")

        return similar_genres

    def get_similarity(self, genre1: str, genre2: str) -> float:
        """
        Get similarity score between two genres.

        Args:
            genre1: First genre
            genre2: Second genre

        Returns:
            Similarity score (0.0-1.0)
        """
        if genre1 not in self.similarity_matrix:
            return 0.0
        return self.similarity_matrix[genre1].get(genre2, 0.0)

    def create_mixed_batch(
        self,
        dataset: List[Dict],
        batch_size: int,
        target_genre: str,
        target_ratio: float = 0.7,
        top_k_similar: int = 2
    ) -> List[Dict]:
        """
        Create training batch mixing target genre with similar genres.

        Useful for transfer learning: include samples from similar genres
        to help model generalize.

        Args:
            dataset: Full dataset
            batch_size: Batch size
            target_genre: Primary genre for batch
            target_ratio: Ratio of target genre samples (e.g., 0.7 = 70%)
            top_k_similar: Number of similar genres to include

        Returns:
            Mixed batch
        """
        # Group by genre
        genre_samples = {}
        for sample in dataset:
            genre = sample.get('genre', 'unknown')
            if genre not in genre_samples:
                genre_samples[genre] = []
            genre_samples[genre].append(sample)

        # Get similar genres
        similar_genres = self.get_transfer_genres(target_genre, top_k=top_k_similar)

        # Calculate samples per genre
        target_samples = int(batch_size * target_ratio)
        similar_samples_total = batch_size - target_samples
        similar_samples_each = similar_samples_total // len(similar_genres) if similar_genres else 0

        batch = []

        # Add target genre samples
        if target_genre in genre_samples:
            target_pool = genre_samples[target_genre]
            indices = np.random.choice(len(target_pool), min(target_samples, len(target_pool)))
            batch.extend([target_pool[i] for i in indices])

        # Add similar genre samples
        for genre in similar_genres:
            if genre in genre_samples:
                genre_pool = genre_samples[genre]
                indices = np.random.choice(
                    len(genre_pool),
                    min(similar_samples_each, len(genre_pool))
                )
                batch.extend([genre_pool[i] for i in indices])

        # Shuffle
        np.random.shuffle(batch)

        logger.debug(f"Created mixed batch: {len(batch)} samples "
                    f"({target_genre} + {similar_genres})")

        return batch

    def compute_transfer_strategy(
        self,
        target_genre: str,
        dataset: List[Dict]
    ) -> Dict:
        """
        Compute recommended transfer learning strategy for target genre.

        Args:
            target_genre: Genre to improve
            dataset: Training dataset

        Returns:
            Dictionary with transfer learning strategy
        """
        # Count samples per genre
        genre_counts = {}
        for sample in dataset:
            genre = sample.get('genre', 'unknown')
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

        target_count = genre_counts.get(target_genre, 0)

        # Get similar genres
        similar_genres = self.get_transfer_genres(target_genre, top_k=3)

        # Compute strategy
        strategy = {
            'target_genre': target_genre,
            'target_samples': target_count,
            'is_minority': target_count < 100,  # Threshold for minority class
            'recommended_approach': None,
            'transfer_from': [],
            'pretrain_genres': [],
            'mixing_ratio': {}
        }

        if target_count < 50:
            # Very low samples - aggressive transfer learning
            strategy['recommended_approach'] = 'aggressive_transfer'
            strategy['transfer_from'] = similar_genres[:2]
            strategy['pretrain_genres'] = similar_genres
            strategy['mixing_ratio'] = {
                target_genre: 0.5,  # 50% target
                similar_genres[0]: 0.3,  # 30% most similar
                similar_genres[1]: 0.2   # 20% second most similar
            }

        elif target_count < 100:
            # Low samples - moderate transfer learning
            strategy['recommended_approach'] = 'moderate_transfer'
            strategy['transfer_from'] = similar_genres[:1]
            strategy['pretrain_genres'] = similar_genres[:2]
            strategy['mixing_ratio'] = {
                target_genre: 0.7,  # 70% target
                similar_genres[0]: 0.3  # 30% most similar
            }

        else:
            # Sufficient samples - light transfer learning
            strategy['recommended_approach'] = 'light_transfer'
            strategy['transfer_from'] = []
            strategy['pretrain_genres'] = ['all']  # Pretrain on all genres
            strategy['mixing_ratio'] = {
                target_genre: 1.0  # 100% target
            }

        logger.info(f"Transfer strategy for {target_genre}: "
                   f"{strategy['recommended_approach']}")

        return strategy

    def get_ensemble_weights(
        self,
        target_genre: str,
        top_k: int = 2
    ) -> Dict[str, float]:
        """
        Get ensemble weights for combining predictions from multiple genre models.

        Args:
            target_genre: Target genre
            top_k: Number of similar genre models to include

        Returns:
            Dictionary mapping genre -> ensemble weight
        """
        if target_genre not in self.similarity_matrix:
            return {target_genre: 1.0}

        # Get similarities
        similar_genres = self.get_transfer_genres(target_genre, top_k=top_k)
        similarities = [
            self.get_similarity(target_genre, g)
            for g in similar_genres
        ]

        # Create weights
        # Primary model gets higher weight (0.7), similar models share remaining (0.3)
        weights = {target_genre: 0.7}

        if similarities:
            total_similarity = sum(similarities)
            for genre, sim in zip(similar_genres, similarities):
                weights[genre] = 0.3 * (sim / total_similarity)

        logger.debug(f"Ensemble weights for {target_genre}: {weights}")

        return weights

    def visualize_similarity_matrix(self) -> str:
        """
        Create text visualization of genre similarity matrix.

        Returns:
            String representation of matrix
        """
        genres = sorted(self.similarity_matrix.keys())

        # Header
        lines = []
        header = "           " + "  ".join(f"{g[:4]:>4}" for g in genres)
        lines.append(header)
        lines.append("-" * len(header))

        # Rows
        for g1 in genres:
            row = f"{g1:>10} "
            for g2 in genres:
                sim = self.similarity_matrix[g1].get(g2, 0.0)
                row += f" {sim:>4.2f} "
            lines.append(row)

        return "\n".join(lines)


class DomainAdaptation:
    """
    Utilities for domain adaptation between genres.

    Domain adaptation reduces genre-specific biases in learned representations,
    making models more transferable across genres.
    """

    def __init__(self):
        """Initialize domain adaptation utilities."""
        logger.info("DomainAdaptation initialized")

    def compute_domain_confusion_loss(
        self,
        features: np.ndarray,
        genre_labels: np.ndarray
    ) -> float:
        """
        Compute domain confusion loss.

        Goal: Make features indistinguishable across genres.

        Args:
            features: Feature representations (N x D)
            genre_labels: Genre labels (N,)

        Returns:
            Domain confusion loss (lower = more genre-agnostic features)
        """
        # Placeholder implementation
        # Real implementation would train a genre discriminator
        # and compute adversarial loss

        # For now, compute genre separability using simple metric
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(features, genre_labels)
            accuracy = lda.score(features, genre_labels)

            # High accuracy = features are genre-specific = high loss
            # Low accuracy = features are genre-agnostic = low loss
            confusion_loss = accuracy

            logger.debug(f"Domain confusion loss: {confusion_loss:.3f}")
            return confusion_loss

        except Exception as e:
            logger.warning(f"Could not compute domain confusion loss: {e}")
            return 0.0

    def compute_genre_bias(
        self,
        predictions: Dict[str, np.ndarray],
        true_labels: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute genre bias in predictions.

        Args:
            predictions: Dict mapping genre -> predictions
            true_labels: Dict mapping genre -> true labels

        Returns:
            Dict mapping genre -> bias score (0 = no bias, 1 = high bias)
        """
        bias_scores = {}

        overall_accuracy = []
        for genre in predictions.keys():
            pred = predictions[genre]
            true = true_labels[genre]
            acc = np.mean(pred == true)
            overall_accuracy.append(acc)

        mean_acc = np.mean(overall_accuracy)

        for genre in predictions.keys():
            pred = predictions[genre]
            true = true_labels[genre]
            genre_acc = np.mean(pred == true)

            # Bias = deviation from mean accuracy
            bias = abs(genre_acc - mean_acc) / mean_acc if mean_acc > 0 else 0
            bias_scores[genre] = bias

        logger.info(f"Genre bias scores: {bias_scores}")

        return bias_scores


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    transfer = CrossGenreTransfer()

    # Test similarity queries
    print("=== Genre Similarity Matrix ===")
    print(transfer.visualize_similarity_matrix())

    # Test transfer genres
    print("\n=== Transfer Learning Recommendations ===")
    for genre in ['jazz', 'rock', 'classical']:
        similar = transfer.get_transfer_genres(genre, top_k=2)
        print(f"{genre} -> {similar}")

    # Test ensemble weights
    print("\n=== Ensemble Weights ===")
    for genre in ['jazz', 'rock']:
        weights = transfer.get_ensemble_weights(genre, top_k=2)
        print(f"{genre}: {weights}")

    # Test transfer strategy
    print("\n=== Transfer Strategy ===")
    dummy_dataset = [
        {'genre': 'jazz', 'file_id': f'jazz_{i}'} for i in range(105)
    ] + [
        {'genre': 'rock', 'file_id': f'rock_{i}'} for i in range(70)
    ] + [
        {'genre': 'classical', 'file_id': f'classical_{i}'} for i in range(140)
    ]

    for genre in ['jazz', 'rock', 'classical']:
        strategy = transfer.compute_transfer_strategy(genre, dummy_dataset)
        print(f"\n{genre}:")
        print(f"  Approach: {strategy['recommended_approach']}")
        print(f"  Transfer from: {strategy['transfer_from']}")
        print(f"  Mixing ratio: {strategy['mixing_ratio']}")
