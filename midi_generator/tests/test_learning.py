#!/usr/bin/env python3
"""
Test Suite for Learning Modules
================================

Comprehensive tests for pattern extraction, corpus learning,
motif library, and fitness learning modules.

Usage:
    python test_learning.py
    pytest test_learning.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
from midi_generator.learning import (
    PatternExtractor, NGramExtractor, MelodicClusterer,
    CorpusAnalyzer, StyleLearner, StyleClassifier,
    Motif, MotifExtractor, MotifDatabase,
    MelodicPattern,
)


class TestPatternExtractor(unittest.TestCase):
    """Test pattern extraction functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = PatternExtractor()

        self.sequences = [
            [60, 62, 64, 65, 67, 65, 64, 62],
            [64, 65, 67, 69, 67, 65, 64, 62],
            [60, 62, 64, 65, 67, 69, 71, 72],
        ]

        self.durations = [
            [0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0],
            [0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
        ]

    def test_extract_melodic_patterns(self):
        """Test melodic pattern extraction."""
        patterns = self.extractor.extract_melodic_patterns(
            self.sequences, self.durations, min_frequency=2
        )

        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)

        for pattern in patterns:
            self.assertIsInstance(pattern, MelodicPattern)
            self.assertGreaterEqual(pattern.frequency, 2)
            self.assertEqual(len(pattern.intervals), len(pattern.notes) - 1)

    def test_extract_harmonic_patterns(self):
        """Test harmonic pattern extraction."""
        chord_sequences = [
            ['C', 'F', 'G', 'C'],
            ['C', 'Am', 'F', 'G'],
            ['C', 'F', 'G', 'Am'],
        ]

        patterns = self.extractor.extract_harmonic_patterns(
            chord_sequences, min_frequency=2
        )

        self.assertIsInstance(patterns, list)
        # May or may not find patterns depending on sequences

    def test_extract_rhythmic_patterns(self):
        """Test rhythmic pattern extraction."""
        patterns = self.extractor.extract_rhythmic_patterns(
            self.durations, min_frequency=2
        )

        self.assertIsInstance(patterns, list)


class TestNGramExtractor(unittest.TestCase):
    """Test n-gram extraction and prediction."""

    def setUp(self):
        """Set up test fixtures."""
        self.ngram = NGramExtractor(min_n=2, max_n=4)
        self.sequences = [
            [60, 62, 64, 65, 67],
            [64, 65, 67, 69, 71],
            [60, 62, 64, 65, 67],
        ]

    def test_extract_pitch_ngrams(self):
        """Test pitch n-gram extraction."""
        ngrams = self.ngram.extract_pitch_ngrams([60, 62, 64, 65], 2)

        self.assertEqual(len(ngrams), 3)
        self.assertEqual(ngrams[0], (60, 62))
        self.assertEqual(ngrams[1], (62, 64))
        self.assertEqual(ngrams[2], (64, 65))

    def test_extract_interval_ngrams(self):
        """Test interval n-gram extraction."""
        ngrams = self.ngram.extract_interval_ngrams([60, 62, 64, 65], 2)

        self.assertEqual(len(ngrams), 2)
        self.assertEqual(ngrams[0], (2, 2))  # Intervals: +2, +2
        self.assertEqual(ngrams[1], (2, 1))  # Intervals: +2, +1

    def test_build_ngram_model(self):
        """Test building n-gram frequency model."""
        self.ngram.build_ngram_model(self.sequences, 'interval')

        self.assertIn(2, self.ngram.interval_ngrams)
        self.assertIn(3, self.ngram.interval_ngrams)
        self.assertGreater(len(self.ngram.interval_ngrams[2]), 0)

    def test_get_most_common_ngrams(self):
        """Test retrieving most common n-grams."""
        self.ngram.build_ngram_model(self.sequences, 'interval')

        common = self.ngram.get_most_common_ngrams(2, 'interval', top_k=3)

        self.assertIsInstance(common, list)
        self.assertLessEqual(len(common), 3)

        if common:
            ngram, count = common[0]
            self.assertIsInstance(ngram, tuple)
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)

    def test_compute_entropy(self):
        """Test n-gram entropy computation."""
        self.ngram.build_ngram_model(self.sequences, 'interval')

        entropy = self.ngram.compute_ngram_entropy(2, 'interval')

        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)


class TestStyleLearner(unittest.TestCase):
    """Test style learning functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.learner = StyleLearner()

        self.bach_sequences = [
            [60, 62, 64, 65, 67, 65, 64, 62],
            [64, 62, 60, 62, 64, 65, 67, 69],
            [67, 65, 64, 62, 64, 65, 67, 65],
        ]

        self.mozart_sequences = [
            [60, 64, 67, 72, 67, 64, 60, 64],
            [65, 69, 72, 69, 65, 69, 72, 76],
            [62, 65, 69, 74, 69, 65, 62, 65],
        ]

    def test_learn_style(self):
        """Test learning style from sequences."""
        model = self.learner.learn_style("Bach", self.bach_sequences)

        self.assertEqual(model.name, "Bach")
        self.assertEqual(model.num_pieces, len(self.bach_sequences))
        self.assertGreater(len(model.pitch_dist), 0)
        self.assertGreater(len(model.interval_dist), 0)
        self.assertIn('avg_pitch', model.statistics)

    def test_generate_in_style(self):
        """Test generating melody in learned style."""
        self.learner.learn_style("Bach", self.bach_sequences)

        generated = self.learner.generate_in_style("Bach", length=10, start_pitch=60)

        self.assertIsInstance(generated, list)
        self.assertEqual(len(generated), 10)
        self.assertTrue(all(isinstance(p, (int, np.integer)) for p in generated))

    def test_compare_styles(self):
        """Test comparing two learned styles."""
        self.learner.learn_style("Bach", self.bach_sequences)
        self.learner.learn_style("Mozart", self.mozart_sequences)

        comparison = self.learner.compare_styles("Bach", "Mozart")

        self.assertIsInstance(comparison, dict)
        self.assertIn('pitch_kl_div', comparison)
        self.assertIn('interval_kl_div', comparison)

    def test_interpolate_styles(self):
        """Test style interpolation."""
        self.learner.learn_style("Bach", self.bach_sequences)
        self.learner.learn_style("Mozart", self.mozart_sequences)

        hybrid = self.learner.interpolate_styles("Bach", "Mozart", alpha=0.5)

        self.assertIsNotNone(hybrid)
        self.assertIn("Bach", hybrid.name)
        self.assertIn("Mozart", hybrid.name)


class TestMotif(unittest.TestCase):
    """Test motif functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.motif = Motif(
            id="test_motif",
            notes=[60, 62, 64, 65],
            intervals=[2, 2, 1],
            rhythm=[0.5, 0.5, 0.5, 1.0],
            contour=[1, 1, 1],
        )

    def test_motif_creation(self):
        """Test creating a motif."""
        self.assertEqual(self.motif.length, 4)
        self.assertEqual(self.motif.pitch_range, 5)  # 65 - 60
        self.assertGreater(self.motif.avg_interval, 0)

    def test_transpose(self):
        """Test motif transposition."""
        transposed = self.motif.transpose(2)

        self.assertEqual(len(transposed.notes), len(self.motif.notes))
        self.assertEqual(transposed.notes[0], self.motif.notes[0] + 2)
        self.assertEqual(transposed.intervals, self.motif.intervals)

    def test_retrograde(self):
        """Test motif retrograde."""
        retro = self.motif.retrograde()

        self.assertEqual(retro.notes, list(reversed(self.motif.notes)))
        self.assertEqual(len(retro.intervals), len(self.motif.intervals))

    def test_inversion(self):
        """Test motif inversion."""
        inverted = self.motif.inversion()

        self.assertEqual(len(inverted.notes), len(self.motif.notes))
        self.assertEqual(inverted.notes[0], self.motif.notes[0])
        self.assertEqual(inverted.intervals, [-i for i in self.motif.intervals])

    def test_augmentation(self):
        """Test rhythmic augmentation."""
        augmented = self.motif.augmentation(factor=2.0)

        self.assertEqual(augmented.rhythm, [r * 2.0 for r in self.motif.rhythm])
        self.assertEqual(augmented.notes, self.motif.notes)


class TestMotifDatabase(unittest.TestCase):
    """Test motif database functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.db = MotifDatabase("/tmp/test_motif_db.json")

        self.motif1 = Motif(
            id="motif1",
            notes=[60, 62, 64, 65],
            intervals=[2, 2, 1],
            rhythm=[0.5]*4,
            contour=[1, 1, 1],
            genre="classical",
            emotion_tags=["joyful"],
        )

        self.motif2 = Motif(
            id="motif2",
            notes=[72, 71, 69, 67],
            intervals=[-1, -2, -2],
            rhythm=[0.5]*4,
            contour=[-1, -1, -1],
            genre="classical",
            emotion_tags=["sad"],
        )

    def test_add_motif(self):
        """Test adding motif to database."""
        self.db.add_motif(self.motif1)

        self.assertIn(self.motif1.id, self.db.motifs)
        self.assertEqual(self.db.motifs[self.motif1.id], self.motif1)

    def test_search_by_tags(self):
        """Test searching motifs by tags."""
        self.db.add_motif(self.motif1)
        self.db.add_motif(self.motif2)

        results = self.db.search_by_tags(emotion="joyful")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "motif1")

    def test_find_similar(self):
        """Test finding similar motifs."""
        self.db.add_motif(self.motif1)
        self.db.add_motif(self.motif2)

        similar = self.db.find_similar(self.motif1, top_k=1)

        self.assertEqual(len(similar), 1)
        motif, similarity = similar[0]
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

    def test_get_statistics(self):
        """Test database statistics."""
        self.db.add_motif(self.motif1)
        self.db.add_motif(self.motif2)

        stats = self.db.get_statistics()

        self.assertEqual(stats['total_motifs'], 2)
        self.assertIn('avg_length', stats)
        self.assertIn('avg_range', stats)


def run_tests():
    """Run all tests and print results."""
    print("\n" + "=" * 70)
    print("Running Learning Module Tests")
    print("=" * 70 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPatternExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestNGramExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestStyleLearner))
    suite.addTests(loader.loadTestsFromTestCase(TestMotif))
    suite.addTests(loader.loadTestsFromTestCase(TestMotifDatabase))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")

    return result.wasSuccessful()


if __name__ == "__main__":
    try:
        import numpy as np
    except ImportError:
        print("Warning: numpy not available, some tests may fail")

    success = run_tests()
    sys.exit(0 if success else 1)
