"""
Complete Example Usage of Multi-Genre Data Specialist Module
Demonstrates all features with dummy data

Author: Agent 07
Date: November 20, 2025
"""

import numpy as np
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import all modules
from genre_stratifier import GenreStratifier
from augmentation import GenreAugmentationPipeline
from genre_balancer import GenreBalancer, BalancedGenreSampler
from cross_genre_transfer import CrossGenreTransfer
from validation import (
    AugmentationValidator,
    GenreValidationSplitter,
    GenreDataStatistics
)


def create_dummy_dataset(
    genre_counts: Dict[str, int] = None
) -> List[Dict[str, Any]]:
    """Create dummy MIDI dataset for testing."""

    if genre_counts is None:
        genre_counts = {
            'jazz': 105,
            'classical': 140,
            'rock': 70,
            'electronic': 84,
            'pop': 126
        }

    subgenres = {
        'jazz': ['bebop', 'swing', 'modal', 'fusion'],
        'classical': ['baroque', 'romantic', 'contemporary'],
        'rock': ['classic', 'progressive', 'metal'],
        'electronic': ['ambient', 'techno', 'idm'],
        'pop': ['80s', '90s', '2000s']
    }

    dataset = []
    file_id = 0

    for genre, count in genre_counts.items():
        for _ in range(count):
            dataset.append({
                'file_id': f"{genre}_{file_id:03d}",
                'genre': genre,
                'subgenre': np.random.choice(subgenres[genre]),
                'tempo_bpm': np.random.uniform(80, 180),
                'complexity': np.random.uniform(0, 1),
                'key': np.random.choice(['C', 'D', 'E', 'F', 'G', 'A', 'B']),
                'notes': [
                    {
                        'pitch': int(np.random.randint(48, 84)),
                        'velocity': int(np.random.randint(60, 100)),
                        'start': float(i * 0.5),
                        'end': float((i + 1) * 0.5)
                    }
                    for i in range(8)
                ]
            })
            file_id += 1

    logger.info(f"Created dummy dataset with {len(dataset)} samples")
    return dataset


def example_1_stratification():
    """Example 1: Stratified data splitting."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Stratified Data Splitting")
    print("="*70)

    # Create dataset
    dataset = create_dummy_dataset()

    # Create stratifier
    stratifier = GenreStratifier(
        stratify_by=['genre', 'subgenre', 'tempo_range']
    )

    # Split dataset
    train, val, test = stratifier.split(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    print(f"\nSplit results:")
    print(f"  Training:   {len(train)} samples")
    print(f"  Validation: {len(val)} samples")
    print(f"  Test:       {len(test)} samples")

    # Get statistics
    stats = stratifier.get_genre_statistics(dataset)
    print(f"\nGenre distribution:")
    for genre, count in stats['genre_counts'].items():
        print(f"  {genre}: {count} samples")

    return train, val, test


def example_2_augmentation():
    """Example 2: Genre-specific augmentation."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Genre-Specific Augmentation")
    print("="*70)

    # Create sample MIDI data
    midi_data = {
        'file_id': 'jazz_001',
        'genre': 'jazz',
        'notes': [
            {'pitch': 60, 'velocity': 80, 'start': 0.0, 'end': 0.5},
            {'pitch': 64, 'velocity': 85, 'start': 0.5, 'end': 1.0},
            {'pitch': 67, 'velocity': 90, 'start': 1.0, 'end': 1.5},
            {'pitch': 71, 'velocity': 85, 'start': 1.5, 'end': 2.0},
        ],
        'tempo_bpm': 180,
        'key': 'F'
    }

    print("\nOriginal MIDI:")
    print(f"  Genre: {midi_data['genre']}")
    print(f"  Tempo: {midi_data['tempo_bpm']} BPM")
    print(f"  Key: {midi_data['key']}")
    print(f"  Pitches: {[n['pitch'] for n in midi_data['notes']]}")

    # Test each genre's augmentation pipeline
    for genre in ['jazz', 'classical', 'rock', 'electronic', 'pop']:
        print(f"\n--- {genre.upper()} Augmentation ---")

        pipeline = GenreAugmentationPipeline(genre)

        # Generate 2 variations
        variations = pipeline.augment(midi_data, num_variations=2)

        for i, var in enumerate(variations):
            print(f"  Variation {i+1}:")
            print(f"    Tempo: {var['tempo_bpm']:.1f} BPM")
            print(f"    Key: {var['key']}")
            print(f"    Pitches: {[n['pitch'] for n in var['notes']]}")
            print(f"    Augmentations: {var.get('augmentation', {})}")

    return midi_data


def example_3_balancing():
    """Example 3: Genre balancing."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Genre Balancing")
    print("="*70)

    # Create imbalanced dataset
    dataset = create_dummy_dataset()

    print("\nOriginal genre distribution:")
    genre_counts = {}
    for sample in dataset:
        genre = sample['genre']
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

    for genre, count in sorted(genre_counts.items()):
        print(f"  {genre}: {count} samples")

    # Create balancer
    balancer = GenreBalancer(target_samples_per_genre=500)

    # Compute class weights
    print("\n--- Class Weights ---")
    weights = balancer.compute_class_weights(dataset)
    for genre, weight in sorted(weights.items()):
        print(f"  {genre}: {weight:.3f}")

    # Balance by augmentation
    print("\n--- Balancing by Augmentation ---")
    balanced = balancer.balance(dataset, method='augmentation')

    print(f"\nBalanced genre distribution:")
    balanced_counts = {}
    for sample in balanced:
        genre = sample['genre']
        balanced_counts[genre] = balanced_counts.get(genre, 0) + 1

    for genre, count in sorted(balanced_counts.items()):
        print(f"  {genre}: {count} samples")

    # Test balanced sampler
    print("\n--- Balanced Batch Sampler ---")
    sampler = balancer.create_balanced_sampler(dataset, batch_size=25)
    batch = sampler.sample_batch()

    batch_counts = {}
    for sample in batch:
        genre = sample['genre']
        batch_counts[genre] = batch_counts.get(genre, 0) + 1

    print(f"Batch size: {len(batch)}")
    print(f"Batch genre distribution: {batch_counts}")

    return balanced


def example_4_transfer_learning():
    """Example 4: Cross-genre transfer learning."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Cross-Genre Transfer Learning")
    print("="*70)

    dataset = create_dummy_dataset()

    transfer = CrossGenreTransfer()

    # Show similarity matrix
    print("\nGenre Similarity Matrix:")
    print(transfer.visualize_similarity_matrix())

    # Get similar genres for each
    print("\n--- Transfer Learning Recommendations ---")
    for genre in ['jazz', 'classical', 'rock', 'electronic', 'pop']:
        similar = transfer.get_transfer_genres(genre, top_k=2)
        print(f"{genre:>12} -> {similar}")

    # Compute transfer strategies
    print("\n--- Transfer Learning Strategies ---")
    for genre in ['jazz', 'rock']:
        strategy = transfer.compute_transfer_strategy(genre, dataset)
        print(f"\n{genre.upper()}:")
        print(f"  Approach: {strategy['recommended_approach']}")
        print(f"  Transfer from: {strategy['transfer_from']}")
        print(f"  Mixing ratio: {strategy['mixing_ratio']}")

    # Create mixed batch
    print("\n--- Mixed Training Batch ---")
    mixed_batch = transfer.create_mixed_batch(
        dataset,
        batch_size=32,
        target_genre='rock',
        target_ratio=0.7
    )

    batch_genres = {}
    for sample in mixed_batch:
        genre = sample['genre']
        batch_genres[genre] = batch_genres.get(genre, 0) + 1

    print(f"Mixed batch composition: {batch_genres}")

    # Get ensemble weights
    print("\n--- Ensemble Weights ---")
    for genre in ['jazz', 'rock']:
        weights = transfer.get_ensemble_weights(genre, top_k=2)
        print(f"{genre}: {weights}")


def example_5_validation():
    """Example 5: Validation and quality assurance."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Validation and Quality Assurance")
    print("="*70)

    dataset = create_dummy_dataset()

    # Test augmentation validator
    print("\n--- Augmentation Validation ---")
    validator = AugmentationValidator()

    original = dataset[0]
    augmented = original.copy()
    augmented['notes'] = original['notes'].copy()
    augmented['notes'][0] = original['notes'][0].copy()
    augmented['notes'][0]['pitch'] = original['notes'][0]['pitch'] + 5  # Transpose

    is_valid, issues = validator.validate_augmentation(original, augmented)
    print(f"Validation result: {'PASS' if is_valid else 'FAIL'}")
    if issues:
        print(f"Issues: {issues}")

    # Test invalid augmentation
    print("\n--- Testing Invalid Augmentation ---")
    invalid = original.copy()
    invalid['notes'] = original['notes'].copy()
    invalid['notes'][0] = original['notes'][0].copy()
    invalid['notes'][0]['pitch'] = 200  # Invalid pitch

    is_valid, issues = validator.validate_augmentation(original, invalid)
    print(f"Validation result: {'PASS' if is_valid else 'FAIL'}")
    if issues:
        print(f"Issues found: {issues}")

    # Test k-fold splitting
    print("\n--- K-Fold Cross-Validation ---")
    splitter = GenreValidationSplitter()
    folds = splitter.k_fold_split(dataset[:200], k=5)

    print(f"Created {len(folds)} folds:")
    for i, (train, val) in enumerate(folds):
        print(f"  Fold {i+1}: {len(train)} train, {len(val)} val")

    # Generate statistics report
    print("\n--- Dataset Statistics Report ---")
    stats = GenreDataStatistics()
    report = stats.generate_report(dataset)
    print(report)


def example_6_complete_pipeline():
    """Example 6: Complete training pipeline."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Complete Training Pipeline")
    print("="*70)

    # 1. Load dataset
    print("\n1. Loading dataset...")
    dataset = create_dummy_dataset()
    print(f"   Loaded {len(dataset)} samples")

    # 2. Stratified split
    print("\n2. Stratified splitting...")
    stratifier = GenreStratifier()
    train, val, test = stratifier.split(dataset)
    print(f"   Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # 3. Balance training set
    print("\n3. Balancing training set...")
    balancer = GenreBalancer(target_samples_per_genre=500)
    balanced_train = balancer.balance(train, method='augmentation')
    print(f"   Balanced train set: {len(balanced_train)} samples")

    # 4. Compute class weights
    print("\n4. Computing class weights...")
    class_weights = balancer.compute_class_weights(balanced_train)
    print(f"   Weights: {class_weights}")

    # 5. Create transfer learning strategies
    print("\n5. Planning transfer learning...")
    transfer = CrossGenreTransfer()
    for genre in ['rock', 'jazz']:
        strategy = transfer.compute_transfer_strategy(genre, balanced_train)
        print(f"   {genre}: {strategy['recommended_approach']}")

    # 6. Validate augmented samples
    print("\n6. Validating augmented samples...")
    validator = AugmentationValidator()
    validated = 0
    invalid = 0

    for sample in balanced_train[:50]:  # Check first 50
        if 'original_file_id' in sample:
            # It's augmented, validate it
            # (In real scenario, would load original)
            is_valid, _ = validator.validate_augmentation(sample, sample)
            if is_valid:
                validated += 1
            else:
                invalid += 1

    print(f"   Validated: {validated}, Invalid: {invalid}")

    # 7. Generate report
    print("\n7. Generating statistics...")
    stats = GenreDataStatistics()
    distribution = stats.compute_genre_distribution(balanced_train)
    print(f"   Total samples: {distribution['total_samples']}")
    print(f"   Genres: {distribution['num_genres']}")
    print(f"   Imbalance ratio: {distribution['imbalance_ratio']:.2f}")

    print("\n" + "="*70)
    print("COMPLETE PIPELINE EXECUTED SUCCESSFULLY")
    print("="*70)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MULTI-GENRE DATA SPECIALIST - COMPLETE EXAMPLES")
    print("Agent 07 - Dø MIDI Generator v2.0")
    print("="*70)

    try:
        # Run all examples
        example_1_stratification()
        example_2_augmentation()
        example_3_balancing()
        example_4_transfer_learning()
        example_5_validation()
        example_6_complete_pipeline()

        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
