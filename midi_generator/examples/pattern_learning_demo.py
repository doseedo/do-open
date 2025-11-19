#!/usr/bin/env python3
"""
Comprehensive Pattern Learning Demo
====================================

This example demonstrates all machine learning and pattern discovery
capabilities of the MIDI generator library.

Demonstrates:
1. Pattern extraction from sequences
2. N-gram analysis and prediction
3. Melodic clustering
4. Style learning from corpus
5. Motif extraction and database
6. Learned fitness functions

Usage:
    python pattern_learning_demo.py
"""

import sys
sys.path.insert(0, '/home/user/Do')

from midi_generator.learning import (
    PatternExtractor, NGramExtractor, MelodicClusterer,
    CorpusAnalyzer, StyleLearner, StyleClassifier,
    Motif, MotifExtractor, MotifDatabase,
)

from midi_generator.optimization import (
    LearnedFitnessFunction, PreferenceLearner, MultiObjectiveFitness,
    MelodyExample, melodic_smoothness, rhythmic_variety
)


def demo_pattern_extraction():
    """Demonstrate pattern extraction and n-gram analysis."""
    print("\n" + "=" * 70)
    print("DEMO 1: Pattern Extraction and N-gram Analysis")
    print("=" * 70)

    # Example melodic sequences (MIDI note numbers)
    sequences = [
        [60, 62, 64, 65, 67, 65, 64, 62, 60],  # C major scale up and down
        [64, 65, 67, 69, 67, 65, 64, 62, 64],  # Similar pattern
        [60, 62, 64, 65, 67, 69, 71, 72],      # Full octave
        [67, 65, 64, 62, 60, 62, 64, 65, 67],  # Variation
    ]

    durations = [
        [0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0],
        [0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
    ]

    # Extract patterns
    extractor = PatternExtractor()
    melodic_patterns = extractor.extract_melodic_patterns(
        sequences, durations, min_frequency=2
    )

    print(f"\n✓ Found {len(melodic_patterns)} recurring melodic patterns")
    print("\nTop 5 patterns by frequency:")
    for i, pattern in enumerate(melodic_patterns[:5], 1):
        print(f"\n  {i}. Frequency: {pattern.frequency}")
        print(f"     Notes: {pattern.notes}")
        print(f"     Intervals: {pattern.intervals}")
        print(f"     Contour: {pattern.contour}")

    # N-gram analysis
    print("\n" + "-" * 70)
    print("N-gram Analysis")
    print("-" * 70)

    extractor.ngram_extractor.build_ngram_model(sequences, 'interval')

    for n in [2, 3, 4]:
        top_ngrams = extractor.ngram_extractor.get_most_common_ngrams(
            n, 'interval', top_k=3
        )
        print(f"\nMost common {n}-grams (intervals):")
        for ngram, count in top_ngrams:
            print(f"  {ngram}: {count} occurrences")

    # Prediction
    print("\n\nPredictive modeling:")
    context = (2, 2)  # Two steps up
    predictions = extractor.ngram_extractor.predict_next(context, 'interval')
    print(f"Given context {context}, most likely next intervals:")
    for interval, prob in predictions[:3]:
        print(f"  {interval:+2d}: {prob:.2%}")


def demo_clustering():
    """Demonstrate melodic clustering."""
    print("\n" + "=" * 70)
    print("DEMO 2: Melodic Clustering")
    print("=" * 70)

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("\n⚠ Skipping: scikit-learn not installed")
        return

    from midi_generator.learning.pattern_extractor import MelodicPattern

    # Create diverse melodic patterns
    patterns = [
        # Stepwise patterns
        MelodicPattern([60, 62, 64, 65, 67], [2, 2, 1, 2], [1, 1, 1, 1], [0.5]*5),
        MelodicPattern([64, 66, 67, 69, 71], [2, 1, 2, 2], [1, 1, 1, 1], [0.5]*5),

        # Leaping patterns
        MelodicPattern([60, 67, 64, 71, 69], [7, -3, 7, -2], [1, -1, 1, -1], [1.0]*5),
        MelodicPattern([62, 69, 65, 72, 67], [7, -4, 7, -5], [1, -1, 1, -1], [1.0]*5),

        # Descending patterns
        MelodicPattern([72, 71, 69, 67, 65], [-1, -2, -2, -2], [-1, -1, -1, -1], [0.5]*5),
        MelodicPattern([76, 74, 72, 71, 69], [-2, -2, -1, -2], [-1, -1, -1, -1], [0.5]*5),
    ]

    clusterer = MelodicClusterer()
    for pattern in patterns:
        clusterer.add_pattern(pattern)

    # Cluster
    clusters = clusterer.cluster_hierarchical(n_clusters=3, similarity_metric='combined')

    print(f"\n✓ Clustered {len(patterns)} patterns into {len(clusters)} groups")

    stats = clusterer.get_cluster_statistics()
    for cluster_id, cluster_stats in stats.items():
        print(f"\n  Cluster {cluster_id}:")
        print(f"    Size: {cluster_stats['size']} patterns")
        print(f"    Avg length: {cluster_stats['avg_length']:.1f} notes")
        print(f"    Avg interval: {cluster_stats['avg_interval']:.2f} semitones")


def demo_style_learning():
    """Demonstrate learning styles from corpus."""
    print("\n" + "=" * 70)
    print("DEMO 3: Style Learning and Generation")
    print("=" * 70)

    # Simulate Bach-style sequences (stepwise, contrapuntal)
    bach_sequences = [
        [60, 62, 64, 65, 67, 65, 64, 62, 60],
        [64, 62, 60, 62, 64, 65, 67, 69, 67],
        [67, 65, 64, 62, 64, 65, 67, 65, 64],
        [62, 64, 65, 67, 65, 64, 62, 60, 62],
    ]

    # Simulate Mozart-style sequences (more arpeggios)
    mozart_sequences = [
        [60, 64, 67, 72, 67, 64, 60, 64, 67],
        [65, 69, 72, 69, 65, 69, 72, 76, 72],
        [62, 65, 69, 74, 69, 65, 62, 65, 69],
        [64, 67, 71, 76, 71, 67, 64, 67, 71],
    ]

    # Learn styles
    learner = StyleLearner()

    print("\n✓ Learning Bach style...")
    bach_model = learner.learn_style("Bach", bach_sequences)
    print(f"  Trained on {bach_model.num_pieces} pieces")
    print(f"  Average interval: {bach_model.statistics['avg_interval']:.2f}")
    print(f"  Stepwise ratio: {bach_model.statistics['stepwise_ratio']:.2%}")

    print("\n✓ Learning Mozart style...")
    mozart_model = learner.learn_style("Mozart", mozart_sequences)
    print(f"  Trained on {mozart_model.num_pieces} pieces")
    print(f"  Average interval: {mozart_model.statistics['avg_interval']:.2f}")
    print(f"  Stepwise ratio: {mozart_model.statistics['stepwise_ratio']:.2%}")

    # Compare styles
    print("\n" + "-" * 70)
    print("Style Comparison")
    print("-" * 70)

    comparison = learner.compare_styles("Bach", "Mozart")
    for metric, value in list(comparison.items())[:5]:
        print(f"  {metric}: {value:.4f}")

    # Generate in styles
    print("\n" + "-" * 70)
    print("Style-Based Generation")
    print("-" * 70)

    bach_gen = learner.generate_in_style("Bach", length=12, start_pitch=60)
    print(f"\nBach-style melody:\n  {bach_gen}")

    mozart_gen = learner.generate_in_style("Mozart", length=12, start_pitch=60)
    print(f"\nMozart-style melody:\n  {mozart_gen}")

    # Style interpolation
    hybrid = learner.interpolate_styles("Bach", "Mozart", alpha=0.5)
    print(f"\n✓ Created hybrid style: {hybrid.name}")
    print(f"  Average interval: {hybrid.statistics.get('avg_interval', 0):.2f}")


def demo_motif_library():
    """Demonstrate motif extraction and database."""
    print("\n" + "=" * 70)
    print("DEMO 4: Motif Library and Database")
    print("=" * 70)

    # Create famous motifs
    motifs = []

    # Beethoven's 5th Symphony opening
    beethoven_5th = Motif(
        id="beethoven_5th",
        notes=[67, 67, 67, 63],
        intervals=[0, 0, -4],
        rhythm=[0.5, 0.5, 0.5, 2.0],
        contour=[0, 0, -1],
        source="Symphony No. 5",
        composer="Beethoven",
        genre="classical",
        era="romantic",
        emotion_tags=["dramatic", "heroic"],
    )
    motifs.append(beethoven_5th)

    # Mozart's Eine kleine Nachtmusik
    mozart_nacht = Motif(
        id="mozart_nacht",
        notes=[67, 64, 67, 64, 67, 69, 71, 72],
        intervals=[-3, 3, -3, 3, 2, 2, 1],
        rhythm=[0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.5, 1.0],
        contour=[-1, 1, -1, 1, 1, 1, 1],
        source="Eine kleine Nachtmusik",
        composer="Mozart",
        genre="classical",
        era="classical",
        emotion_tags=["joyful", "playful"],
    )
    motifs.append(mozart_nacht)

    # Initialize database
    db = MotifDatabase("/tmp/demo_motif_library.json")

    for motif in motifs:
        db.add_motif(motif)
        print(f"\n✓ Added motif: {motif.id}")
        print(f"  Source: {motif.source} ({motif.composer})")
        print(f"  Contour: {motif.contour_type}")
        print(f"  Range: {motif.pitch_range} semitones")

    # Search by tags
    print("\n" + "-" * 70)
    print("Tag-based Search")
    print("-" * 70)

    dramatic = db.search_by_tags(emotion="dramatic")
    print(f"\nDramatic motifs: {len(dramatic)}")
    for motif in dramatic:
        print(f"  - {motif.id}: {motif.source}")

    classical = db.search_by_tags(genre="classical")
    print(f"\nClassical motifs: {len(classical)}")
    for motif in classical:
        print(f"  - {motif.id}: {motif.composer}")

    # Motif transformations
    print("\n" + "-" * 70)
    print("Motif Transformations")
    print("-" * 70)

    print(f"\nOriginal (Beethoven 5th): {beethoven_5th.notes}")
    print(f"Transposed (+2):          {beethoven_5th.transpose(2).notes}")
    print(f"Retrograde:               {beethoven_5th.retrograde().notes}")
    print(f"Inversion:                {beethoven_5th.inversion().notes}")

    # Database statistics
    print("\n" + "-" * 70)
    print("Database Statistics")
    print("-" * 70)

    stats = db.get_statistics()
    print(f"\nTotal motifs: {stats['total_motifs']}")
    print(f"Average length: {stats['avg_length']:.1f} notes")
    print(f"Average range: {stats['avg_range']:.1f} semitones")

    # Save
    db.save()
    print(f"\n✓ Database saved to {db.db_path}")


def demo_learned_fitness():
    """Demonstrate learned fitness functions."""
    print("\n" + "=" * 70)
    print("DEMO 5: Learned Fitness Functions")
    print("=" * 70)

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("\n⚠ Skipping: scikit-learn not installed")
        return

    # Create training data
    print("\n✓ Creating training examples...")

    good_melodies = [
        ([60, 62, 64, 65, 67, 65, 64, 62, 60], [0.5]*9, 3),  # Excellent
        ([64, 65, 67, 69, 67, 65, 64, 62, 64], [0.5]*9, 3),
        ([67, 69, 71, 72, 71, 69, 67, 65, 67], [0.5]*9, 2),  # Good
    ]

    bad_melodies = [
        ([60, 72, 48, 84, 36, 60, 72, 48], [1.0]*8, 0),  # Poor
        ([60, 60, 60, 60, 60, 60, 60, 60], [1.0]*8, 0),
        ([60, 61, 62, 63, 64, 65, 66, 67], [0.25]*8, 1),  # Fair
    ]

    examples = []
    for pitches, durations, label in good_melodies + bad_melodies:
        examples.append(MelodyExample(pitches, durations, label))

    # Train
    print("✓ Training fitness function...")
    fitness = LearnedFitnessFunction(model_type='random_forest', task='classification')
    fitness.train(examples)

    # Evaluate
    print("\n" + "-" * 70)
    print("Melody Evaluation")
    print("-" * 70)

    test_melodies = [
        ([60, 62, 64, 65, 64, 62, 60], "Stepwise melody"),
        ([60, 67, 55, 72, 48, 84, 36], "Random leaps"),
        ([64, 65, 67, 69, 67, 64, 62], "Good contour"),
        ([60, 60, 60, 60, 60, 60, 60], "Boring repetition"),
    ]

    for pitches, description in test_melodies:
        score = fitness.evaluate(pitches)
        rating = "⭐" * int(score * 4)
        print(f"\n{description}:")
        print(f"  Score: {score:.3f} {rating}")

    # Feature importance
    print("\n" + "-" * 70)
    print("Most Important Features")
    print("-" * 70)

    importance = fitness.get_feature_importance()
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8]

    for feature, imp in top_features:
        bar = "█" * int(imp * 50)
        print(f"  {feature:20s}: {bar} {imp:.3f}")

    # Multi-objective fitness
    print("\n" + "-" * 70)
    print("Multi-Objective Optimization")
    print("-" * 70)

    multi_fitness = MultiObjectiveFitness()
    multi_fitness.add_objective("smoothness", melodic_smoothness, weight=1.0)
    multi_fitness.add_objective("variety", rhythmic_variety, weight=0.5)

    test_melody = [60, 62, 64, 65, 67, 69, 71, 72]
    test_rhythm = [0.5, 0.5, 0.25, 0.25, 1.0, 0.5, 0.5, 2.0]

    overall = multi_fitness.evaluate(test_melody, test_rhythm)
    objectives = multi_fitness.evaluate_all(test_melody, test_rhythm)

    print(f"\nTest melody: {test_melody}")
    print(f"\nOverall score: {overall:.3f}")
    print("\nObjective breakdown:")
    for name, score in objectives.items():
        print(f"  {name:15s}: {score:.3f}")


def main():
    """Run all demonstrations."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 15 + "PATTERN LEARNING DEMO" + " " * 32 + "█")
    print("█" + " " * 10 + "Machine Learning for MIDI Generation" + " " * 22 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    try:
        demo_pattern_extraction()
        demo_clustering()
        demo_style_learning()
        demo_motif_library()
        demo_learned_fitness()

        print("\n" + "=" * 70)
        print("✓ All demonstrations completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
