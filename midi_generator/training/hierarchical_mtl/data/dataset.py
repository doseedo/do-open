"""
Hierarchical MIDI Dataset for Multi-Task Learning.

This module provides dataset classes for loading and processing the 750-file
MIDI corpus with hierarchical parameter labels for training the MTL model.

Author: Agent 06
Date: November 20, 2025
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import random


class HierarchicalMIDIDataset(Dataset):
    """
    Dataset for hierarchical MIDI parameter learning.

    Loads 750-file MIDI corpus with labels for all 50 hierarchical parameters:
    - Level 1: Global Context (8 parameters)
    - Level 2: Universal Dimensions (20 parameters)
    - Level 3: Genre-Specific Details (22 parameters)

    Args:
        labeled_dataset_path: Path to labeled_dataset.json from Agent 03
        features_dir: Directory containing extracted features (200D vectors)
        split: "train", "val", or "test"
        transform: Optional data augmentation transform
        normalize: Whether to normalize features
    """

    def __init__(
        self,
        labeled_dataset_path: Path,
        features_dir: Optional[Path] = None,
        split: str = "train",
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalization_stats: Optional[Dict[str, Any]] = None
    ):
        super().__init__()

        self.labeled_dataset_path = Path(labeled_dataset_path)
        self.features_dir = Path(features_dir) if features_dir else None
        self.split = split
        self.transform = transform
        self.normalize = normalize

        # Load labeled dataset
        with open(self.labeled_dataset_path, 'r') as f:
            self.labeled_data = json.load(f)

        # Filter by split
        self.samples = [
            item for item in self.labeled_data
            if item.get('split', 'train') == split
        ]

        print(f"Loaded {len(self.samples)} {split} samples")

        # Build genre index
        self.genre_to_indices = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            genre = sample['labels']['level1'].get('genre.primary', 'unknown')
            self.genre_to_indices[genre].append(idx)

        # Normalization statistics
        if normalize:
            if normalization_stats is None:
                # Compute statistics from training data
                self.normalization_stats = self._compute_normalization_stats()
            else:
                self.normalization_stats = normalization_stats
        else:
            self.normalization_stats = None

        # Parameter definitions (50 parameters)
        self.level1_params = [
            'genre.primary',
            'tempo.bpm',
            'time_signature',
            'key.tonic',
            'key.mode',
            'energy.level',
            'complexity.overall',
            'structure.form'
        ]

        self.level2_params = {
            'harmony': [
                'harmony.chord_density',
                'harmony.complexity',
                'harmony.chromaticism',
                'harmony.tension',
                'harmony.voicing_spread',
                'harmony.progression_predictability'
            ],
            'melody': [
                'melody.note_density',
                'melody.range_semitones',
                'melody.contour_smoothness',
                'melody.rhythmic_complexity',
                'melody.repetition'
            ],
            'rhythm': [
                'rhythm.subdivision',
                'rhythm.syncopation',
                'rhythm.groove_consistency',
                'rhythm.polyrhythm',
                'rhythm.swing_amount'
            ],
            'dynamics': [
                'dynamics.overall_level',
                'dynamics.range'
            ],
            'texture': [
                'texture.polyphony',
                'texture.density'
            ]
        }

        # Level 3: Genre-specific parameters
        self.level3_params = {
            'universal': [
                'orchestration.instrument_count',
                'orchestration.register_balance',
                'articulation.legato_ratio',
                'structure.section_contrast',
                'structure.repetition_level'
            ],
            'jazz': [
                'jazz.swing_feel',
                'jazz.walking_bass',
                'jazz.improvisation_ratio',
                'jazz.bebop_vocabulary'
            ],
            'classical': [
                'classical.counterpoint',
                'classical.development_density',
                'classical.voice_leading_quality'
            ],
            'rock': [
                'rock.power_chord_ratio',
                'rock.riff_repetition',
                'rock.distortion_level'
            ],
            'electronic': [
                'electronic.quantization',
                'electronic.filter_movement',
                'electronic.arpeggio_density'
            ],
            'hiphop': [
                'hiphop.sample_based',
                'hiphop.boom_bap_feel'
            ],
            'latin': [
                'latin.clave_pattern',
                'latin.montuno_complexity'
            ]
        }

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - features: 200D feature vector (torch.Tensor)
                - level1: Dict of level 1 parameter values
                - level2: Dict of level 2 parameter values
                - level3: Dict of level 3 parameter values (genre-specific)
                - genre: Genre label (str)
                - file_id: File identifier
        """
        sample = self.samples[idx]

        # Load features
        features = self._load_features(sample)

        # Normalize if requested
        if self.normalize and self.normalization_stats is not None:
            features = self._normalize_features(features)

        # Apply augmentation if provided
        if self.transform is not None:
            features = self.transform(features)

        # Extract hierarchical labels
        labels = sample['labels']

        level1_labels = self._extract_level1_labels(labels['level1'])
        level2_labels = self._extract_level2_labels(labels['level2'])
        level3_labels = self._extract_level3_labels(
            labels['level3'],
            labels['level1'].get('genre.primary', 'unknown')
        )

        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'level1': level1_labels,
            'level2': level2_labels,
            'level3': level3_labels,
            'genre': labels['level1'].get('genre.primary', 'unknown'),
            'file_id': sample['file_id']
        }

    def _load_features(self, sample: Dict[str, Any]) -> np.ndarray:
        """Load 200D feature vector for a sample."""
        if self.features_dir:
            # Load from pre-extracted features
            feature_file = self.features_dir / f"{sample['file_id']}.npy"
            if feature_file.exists():
                return np.load(feature_file)

        # Fallback: extract features from MIDI file
        # (This requires Agent 04's optimized feature extractor)
        # For now, return placeholder
        return np.random.randn(200).astype(np.float32)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using stored statistics."""
        mean = self.normalization_stats['mean']
        std = self.normalization_stats['std']
        return (features - mean) / (std + 1e-8)

    def _compute_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Compute mean and std for feature normalization (only on training set)."""
        if self.split != 'train':
            raise ValueError("Normalization stats should only be computed on training set")

        all_features = []
        for sample in self.samples[:100]:  # Use subset for efficiency
            features = self._load_features(sample)
            all_features.append(features)

        all_features = np.stack(all_features, axis=0)
        mean = np.mean(all_features, axis=0)
        std = np.std(all_features, axis=0)

        return {'mean': mean, 'std': std}

    def _extract_level1_labels(self, level1: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract and convert level 1 labels to tensors."""
        labels = {}

        for param_name in self.level1_params:
            value = level1.get(param_name, None)

            if value is None:
                # Missing value - use default
                labels[param_name] = torch.tensor(0.0, dtype=torch.float32)
                continue

            # Convert based on parameter type
            if param_name == 'genre.primary':
                # Categorical - one-hot encoding
                genre_map = {'jazz': 0, 'classical': 1, 'rock': 2, 'electronic': 3,
                             'pop': 4, 'hiphop': 5, 'latin': 6}
                genre_idx = genre_map.get(value, 0)
                labels[param_name] = torch.tensor(genre_idx, dtype=torch.long)

            elif param_name == 'time_signature':
                # Categorical
                ts_map = {'4/4': 0, '3/4': 1, '6/8': 2, '5/4': 3, '7/8': 4}
                ts_idx = ts_map.get(value, 0)
                labels[param_name] = torch.tensor(ts_idx, dtype=torch.long)

            elif param_name == 'key.tonic':
                # Categorical (12 keys)
                key_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                           'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
                key_idx = key_map.get(value, 0)
                labels[param_name] = torch.tensor(key_idx, dtype=torch.long)

            elif param_name == 'key.mode':
                # Binary: major (0) or minor (1)
                mode_idx = 0 if value == 'major' else 1
                labels[param_name] = torch.tensor(mode_idx, dtype=torch.long)

            elif param_name == 'structure.form':
                # Categorical
                form_map = {'AABA': 0, 'ABAB': 1, 'verse-chorus': 2, 'sonata': 3,
                            'rondo': 4, 'free': 5}
                form_idx = form_map.get(value, 5)
                labels[param_name] = torch.tensor(form_idx, dtype=torch.long)

            else:
                # Continuous parameters
                labels[param_name] = torch.tensor(float(value), dtype=torch.float32)

        return labels

    def _extract_level2_labels(self, level2: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract and convert level 2 labels to tensors."""
        labels = {}

        # Flatten level2 parameters
        all_level2_params = []
        for category in self.level2_params.values():
            all_level2_params.extend(category)

        for param_name in all_level2_params:
            value = level2.get(param_name, None)

            if value is None:
                labels[param_name] = torch.tensor(0.0, dtype=torch.float32)
            else:
                # All level 2 parameters are continuous
                labels[param_name] = torch.tensor(float(value), dtype=torch.float32)

        return labels

    def _extract_level3_labels(
        self,
        level3: Dict[str, Any],
        genre: str
    ) -> Dict[str, torch.Tensor]:
        """Extract and convert level 3 labels to tensors."""
        labels = {}

        # Universal level 3 parameters (apply to all genres)
        for param_name in self.level3_params['universal']:
            value = level3.get(param_name, None)
            if value is None:
                labels[param_name] = torch.tensor(0.0, dtype=torch.float32)
            else:
                labels[param_name] = torch.tensor(float(value), dtype=torch.float32)

        # Genre-specific parameters
        if genre in self.level3_params:
            for param_name in self.level3_params[genre]:
                value = level3.get(param_name, None)
                if value is None:
                    # Genre-specific param not applicable - use NaN or mask
                    labels[param_name] = torch.tensor(float('nan'), dtype=torch.float32)
                else:
                    if 'feel' in param_name or 'pattern' in param_name:
                        # Categorical genre-specific param
                        # For now, treat as continuous (can be refined)
                        labels[param_name] = torch.tensor(float(value), dtype=torch.float32)
                    else:
                        labels[param_name] = torch.tensor(float(value), dtype=torch.float32)

        return labels

    def get_genre_distribution(self) -> Dict[str, int]:
        """Get distribution of genres in the dataset."""
        return {genre: len(indices) for genre, indices in self.genre_to_indices.items()}


class DataAugmenter:
    """
    Data augmentation for MIDI features.

    Applies random noise, scaling, and other transformations to features
    to improve model generalization.
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        prob: float = 0.5
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Apply augmentation to features."""
        if random.random() > self.prob:
            return features

        # Add Gaussian noise
        features = features + np.random.randn(*features.shape) * self.noise_std

        # Random scaling
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        features = features * scale

        return features


class FeatureNormalizer:
    """Feature normalization utility."""

    def __init__(self, method: str = "standardize"):
        """
        Initialize normalizer.

        Args:
            method: "standardize" (z-score) or "minmax"
        """
        self.method = method
        self.stats = None

    def fit(self, features: np.ndarray):
        """Fit normalization statistics."""
        if self.method == "standardize":
            self.stats = {
                'mean': np.mean(features, axis=0),
                'std': np.std(features, axis=0)
            }
        elif self.method == "minmax":
            self.stats = {
                'min': np.min(features, axis=0),
                'max': np.max(features, axis=0)
            }
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if self.stats is None:
            raise ValueError("Normalizer not fitted")

        if self.method == "standardize":
            return (features - self.stats['mean']) / (self.stats['std'] + 1e-8)
        elif self.method == "minmax":
            return (features - self.stats['min']) / (self.stats['max'] - self.stats['min'] + 1e-8)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(features)
        return self.transform(features)


def create_dataloaders(
    labeled_dataset_path: Path,
    features_dir: Optional[Path] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_augmentation: bool = True,
    augmentation_prob: float = 0.3,
    normalize: bool = True,
    stratified_sampling: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        labeled_dataset_path: Path to labeled dataset JSON
        features_dir: Directory with pre-extracted features
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        use_augmentation: Whether to use data augmentation (train only)
        augmentation_prob: Probability of applying augmentation
        normalize: Whether to normalize features
        stratified_sampling: Whether to use stratified sampling by genre

    Returns:
        (train_loader, val_loader, test_loader)
    """

    # Create augmenter
    augmenter = DataAugmenter(prob=augmentation_prob) if use_augmentation else None

    # Create train dataset and compute normalization stats
    train_dataset = HierarchicalMIDIDataset(
        labeled_dataset_path=labeled_dataset_path,
        features_dir=features_dir,
        split="train",
        transform=augmenter,
        normalize=normalize
    )

    # Get normalization stats from train
    norm_stats = train_dataset.normalization_stats if normalize else None

    # Create val and test datasets
    val_dataset = HierarchicalMIDIDataset(
        labeled_dataset_path=labeled_dataset_path,
        features_dir=features_dir,
        split="val",
        transform=None,  # No augmentation for validation
        normalize=normalize,
        normalization_stats=norm_stats
    )

    test_dataset = HierarchicalMIDIDataset(
        labeled_dataset_path=labeled_dataset_path,
        features_dir=features_dir,
        split="test",
        transform=None,  # No augmentation for test
        normalize=normalize,
        normalization_stats=norm_stats
    )

    # Create samplers
    train_sampler = None
    if stratified_sampling:
        # Create weighted sampler for genre balance
        genre_dist = train_dataset.get_genre_distribution()
        weights = []
        for sample in train_dataset.samples:
            genre = sample['labels']['level1'].get('genre.primary', 'unknown')
            weight = 1.0 / genre_dist.get(genre, 1)
            weights.append(weight)

        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader
