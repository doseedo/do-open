#!/usr/bin/env python3
"""
Unit Tests for Extended Harmony Module

Comprehensive test suite covering all features:
- Upper structure triads
- Polychords
- Cluster voicings
- Slash chords
- Altered dominants
- Multi-tonic analysis

Author: Agent 8
Date: 2025-11-19
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from advanced_modules.extended_harmony import (
    ExtendedHarmony,
    UpperStructureType,
    ClusterType,
    PolychordRelation,
    TonalCenter,
    Chord,
    Polychord,
    MultiTonicAnalysis
)


class TestUpperStructureTriads(unittest.TestCase):
    """Test upper structure triad generation"""

    def setUp(self):
        self.harmony = ExtendedHarmony()

    def test_maj_sharp11_structure(self):
        """Test major triad on #11 (Lydian dominant)"""
        chord = self.harmony.create_upper_structure(7, "maj_#11", octave=4)
        self.assertEqual(chord.root, 7)  # G
        self.assertEqual(chord.quality, "dom")
        self.assertIn("#11", chord.extensions)
        self.assertTrue(len(chord.voicing) > 0)

    def test_maj_flat9_structure(self):
        """Test major triad on b9"""
        chord = self.harmony.create_upper_structure(7, "maj_b9", octave=4)
        self.assertIn("b9", chord.extensions)
        self.assertIn("b13", chord.extensions)

    def test_maj_sharp9_structure(self):
        """Test major triad on #9"""
        chord = self.harmony.create_upper_structure(7, "maj_#9", octave=4)
        self.assertIn("#9", chord.extensions)

    def test_upper_structure_without_root(self):
        """Test upper structure without root note"""
        chord = self.harmony.create_upper_structure(7, "maj_#11", include_root=False)
        # Should still have voicing but possibly fewer notes
        self.assertTrue(len(chord.voicing) > 0)

    def test_invalid_structure_type(self):
        """Test invalid structure type raises error"""
        with self.assertRaises(ValueError):
            self.harmony.create_upper_structure(7, "invalid_type")

    def test_different_roots(self):
        """Test upper structures on different roots"""
        for root in [0, 2, 5, 7, 9]:
            chord = self.harmony.create_upper_structure(root, "maj_#11")
            self.assertEqual(chord.root, root)


class TestPolychords(unittest.TestCase):
    """Test polychord generation"""

    def setUp(self):
        self.harmony = ExtendedHarmony()

    def test_petrushka_chord(self):
        """Test Petrushka chord (C major + F# major)"""
        poly = self.harmony.create_polychord(0, "maj", 6, "maj")
        self.assertEqual(poly.upper_chord.root, 0)
        self.assertEqual(poly.lower_chord.root, 6)
        self.assertEqual(poly.relation, PolychordRelation.TRITONE)

    def test_neosoul_polychord(self):
        """Test neo-soul polychord (Cmaj7 over Dm)"""
        poly = self.harmony.create_polychord(0, "maj7", 2, "min")
        self.assertEqual(poly.upper_chord.quality, "maj7")
        self.assertEqual(poly.lower_chord.quality, "min")

    def test_polychord_combined_voicing(self):
        """Test combined voicing includes both chords"""
        poly = self.harmony.create_polychord(0, "maj", 6, "maj")
        # Should have notes from both triads
        self.assertTrue(len(poly.combined_voicing) >= 6)

    def test_parallel_polychord(self):
        """Test parallel polychord (same root, different quality)"""
        poly = self.harmony.create_polychord(0, "maj", 0, "min")
        self.assertEqual(poly.relation, PolychordRelation.PARALLEL)

    def test_chromatic_mediant_relation(self):
        """Test chromatic mediant relation"""
        poly = self.harmony.create_polychord(0, "maj", 1, "maj")
        self.assertEqual(poly.relation, PolychordRelation.CHROMATIC_MEDIANT)

    def test_polychord_spacing(self):
        """Test polychord spacing parameter"""
        poly1 = self.harmony.create_polychord(0, "maj", 5, "maj", spacing=12)
        poly2 = self.harmony.create_polychord(0, "maj", 5, "maj", spacing=24)
        # Second polychord should have wider spacing
        self.assertNotEqual(poly1.combined_voicing, poly2.combined_voicing)


class TestClusterVoicings(unittest.TestCase):
    """Test cluster voicing generation"""

    def setUp(self):
        self.harmony = ExtendedHarmony()

    def test_chromatic_cluster(self):
        """Test chromatic cluster (Ligeti style)"""
        cluster = self.harmony.create_cluster(60, ClusterType.CHROMATIC, 5)
        self.assertEqual(len(cluster), 5)
        # Should be adjacent semitones
        for i in range(len(cluster) - 1):
            self.assertEqual(cluster[i+1] - cluster[i], 1)

    def test_diatonic_cluster(self):
        """Test diatonic cluster (Bartók style)"""
        cluster = self.harmony.create_cluster(60, ClusterType.DIATONIC, 4)
        self.assertTrue(len(cluster) >= 4)
        # Should follow C major scale intervals

    def test_pentatonic_cluster(self):
        """Test pentatonic cluster"""
        cluster = self.harmony.create_cluster(60, ClusterType.PENTATONIC, 5)
        self.assertTrue(len(cluster) >= 5)

    def test_whole_tone_cluster(self):
        """Test whole-tone cluster"""
        cluster = self.harmony.create_cluster(60, ClusterType.WHOLE_TONE, 4)
        self.assertEqual(len(cluster), 4)
        # Should be whole steps
        for i in range(len(cluster) - 1):
            self.assertEqual(cluster[i+1] - cluster[i], 2)

    def test_quartal_cluster(self):
        """Test quartal cluster (McCoy Tyner style)"""
        cluster = self.harmony.create_cluster(60, ClusterType.QUARTAL, 4)
        # Should be stacked fourths (5 semitones)
        for i in range(len(cluster) - 1):
            self.assertEqual(cluster[i+1] - cluster[i], 5)

    def test_quintal_cluster(self):
        """Test quintal cluster (Hindemith style)"""
        cluster = self.harmony.create_cluster(60, ClusterType.QUINTAL, 3)
        # Should be stacked fifths (7 semitones)
        for i in range(len(cluster) - 1):
            self.assertEqual(cluster[i+1] - cluster[i], 7)

    def test_cluster_span_limit(self):
        """Test cluster respects span limit"""
        cluster = self.harmony.create_cluster(60, ClusterType.CHROMATIC, 20, span_semitones=12)
        # All notes should be within span
        for note in cluster:
            self.assertLess(note, 60 + 12)

    def test_empty_cluster(self):
        """Test cluster with 0 notes"""
        cluster = self.harmony.create_cluster(60, ClusterType.CHROMATIC, 0)
        self.assertEqual(len(cluster), 0)


class TestSlashChords(unittest.TestCase):
    """Test slash chord generation"""

    def setUp(self):
        self.harmony = ExtendedHarmony()

    def test_first_inversion(self):
        """Test Cmaj7/E (first inversion)"""
        chord = self.harmony.create_slash_chord(0, "maj7", 4)
        self.assertEqual(chord.root, 0)
        self.assertEqual(chord.bass_note, 4)
        self.assertTrue(52 in chord.voicing or 64 in chord.voicing)  # E in bass

    def test_folk_slash_chord(self):
        """Test D/F# (common in folk/pop)"""
        chord = self.harmony.create_slash_chord(2, "maj", 6)
        self.assertEqual(chord.root, 2)
        self.assertEqual(chord.bass_note, 6)

    def test_bass_motion_slash(self):
        """Test G7/D (for smooth bass motion)"""
        chord = self.harmony.create_slash_chord(7, "dom", 2)
        self.assertEqual(chord.bass_note, 2)

    def test_slash_chord_with_extensions(self):
        """Test slash chord with extensions"""
        chord = self.harmony.create_slash_chord(0, "maj7", 4, extensions=["9", "13"])
        self.assertIn("9", chord.extensions)
        self.assertIn("13", chord.extensions)

    def test_slash_chord_string_representation(self):
        """Test slash chord displays correctly"""
        chord = self.harmony.create_slash_chord(0, "maj7", 4)
        chord_str = str(chord)
        self.assertIn("/", chord_str)


class TestAlteredDominants(unittest.TestCase):
    """Test altered dominant generation"""

    def setUp(self):
        self.harmony = ExtendedHarmony()

    def test_g7alt_all_alterations(self):
        """Test G7alt with all alterations"""
        chord = self.harmony.create_altered_dominant(7, ["b9", "#9", "#11", "b13"])
        self.assertEqual(chord.root, 7)
        self.assertEqual(chord.quality, "dom")
        self.assertEqual(len(chord.extensions), 4)

    def test_lydian_dominant(self):
        """Test G7#11 (Lydian dominant)"""
        chord = self.harmony.create_altered_dominant(7, ["#11"])
        self.assertIn("#11", chord.extensions)

    def test_flat9_flat13(self):
        """Test G7b9b13"""
        chord = self.harmony.create_altered_dominant(7, ["b9", "b13"])
        self.assertIn("b9", chord.extensions)
        self.assertIn("b13", chord.extensions)

    def test_altered_5th(self):
        """Test altered fifth (#5 or b5)"""
        chord_sharp5 = self.harmony.create_altered_dominant(7, ["#5"])
        chord_flat5 = self.harmony.create_altered_dominant(7, ["b5"])
        self.assertIn("#5", chord_sharp5.extensions)
        self.assertIn("b5", chord_flat5.extensions)

    def test_tight_vs_spread_voicing(self):
        """Test tight vs spread voicing styles"""
        tight = self.harmony.create_altered_dominant(7, ["b9"], voicing_style="tight")
        spread = self.harmony.create_altered_dominant(7, ["b9"], voicing_style="spread")
        # Spread should have wider range
        tight_range = max(tight.voicing) - min(tight.voicing)
        spread_range = max(spread.voicing) - min(spread.voicing)
        self.assertGreater(spread_range, tight_range)

    def test_dominant_with_natural_13(self):
        """Test dominant with natural 13"""
        chord = self.harmony.create_altered_dominant(7, ["13"])
        self.assertIn("13", chord.extensions)


class TestMultiTonicAnalysis(unittest.TestCase):
    """Test multi-tonic system analysis"""

    def setUp(self):
        self.harmony = ExtendedHarmony()

    def test_single_tonic_clear(self):
        """Test progression with single clear tonic"""
        progression = [
            self.harmony.create_slash_chord(0, "maj", 0),
            self.harmony.create_slash_chord(0, "maj", 0),
            self.harmony.create_slash_chord(0, "maj", 0),
            self.harmony.create_slash_chord(0, "maj", 0),
        ]
        analysis = self.harmony.analyze_multitonic_system(progression)
        self.assertEqual(analysis.primary_key, 0)
        self.assertEqual(analysis.ambiguity_score, 0.0)

    def test_dual_tonic_ambiguous(self):
        """Test progression with competing tonal centers"""
        progression = [
            self.harmony.create_slash_chord(0, "maj", 0),
            self.harmony.create_slash_chord(0, "maj", 0),
            self.harmony.create_slash_chord(6, "maj", 6),
            self.harmony.create_slash_chord(6, "maj", 6),
        ]
        analysis = self.harmony.analyze_multitonic_system(progression)
        # Should detect ambiguity
        self.assertGreater(analysis.ambiguity_score, 0.5)

    def test_secondary_tonal_centers(self):
        """Test detection of secondary tonal centers"""
        progression = [
            self.harmony.create_slash_chord(0, "maj", 0),  # C
            self.harmony.create_slash_chord(0, "maj", 0),  # C
            self.harmony.create_slash_chord(0, "maj", 0),  # C
            self.harmony.create_slash_chord(5, "maj", 5),  # F
        ]
        analysis = self.harmony.analyze_multitonic_system(progression)
        self.assertEqual(analysis.primary_key, 0)
        self.assertIn(5, analysis.secondary_keys)

    def test_empty_progression(self):
        """Test analysis of empty progression"""
        analysis = self.harmony.analyze_multitonic_system([])
        self.assertEqual(analysis.ambiguity_score, 1.0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""

    def setUp(self):
        self.harmony = ExtendedHarmony()

    def test_chord_to_midi_notes(self):
        """Test extracting MIDI notes from chord"""
        chord = self.harmony.create_slash_chord(0, "maj", 0)
        notes = self.harmony.chord_to_midi_notes(chord)
        self.assertTrue(len(notes) > 0)
        self.assertTrue(all(isinstance(n, int) for n in notes))

    def test_get_chord_name(self):
        """Test getting chord name string"""
        chord = self.harmony.create_slash_chord(0, "maj7", 4)
        name = self.harmony.get_chord_name(chord)
        self.assertIsInstance(name, str)
        self.assertIn("/", name)

    def test_transpose_chord(self):
        """Test transposing chord"""
        chord = self.harmony.create_slash_chord(0, "maj", 0)
        transposed = self.harmony.transpose_chord(chord, 2)
        self.assertEqual(transposed.root, 2)  # C -> D
        # Voicing should be transposed
        for orig, trans in zip(chord.voicing, transposed.voicing):
            self.assertEqual(trans - orig, 2)

    def test_transpose_with_slash(self):
        """Test transposing slash chord"""
        chord = self.harmony.create_slash_chord(0, "maj7", 4)
        transposed = self.harmony.transpose_chord(chord, 3)
        self.assertEqual(transposed.root, 3)
        self.assertEqual(transposed.bass_note, 7)

    def test_note_names_display(self):
        """Test note names are correct"""
        self.assertEqual(self.harmony.NOTE_NAMES[0], "C")
        self.assertEqual(self.harmony.NOTE_NAMES[7], "G")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        self.harmony = ExtendedHarmony()

    def test_extreme_octaves(self):
        """Test extreme octave ranges"""
        chord_low = self.harmony.create_slash_chord(0, "maj", 0, octave=0)
        chord_high = self.harmony.create_slash_chord(0, "maj", 0, octave=8)
        self.assertTrue(all(n >= 0 for n in chord_low.voicing))
        self.assertTrue(all(n < 128 for n in chord_high.voicing))

    def test_pitch_class_wraparound(self):
        """Test pitch class wraparound (0-11)"""
        chord = self.harmony.create_upper_structure(11, "maj_#11")
        self.assertEqual(chord.root, 11)

    def test_large_cluster(self):
        """Test very large cluster"""
        cluster = self.harmony.create_cluster(60, ClusterType.CHROMATIC, 100, span_semitones=12)
        # Should be limited by span
        self.assertTrue(all(n < 60 + 12 for n in cluster))

    def test_chord_str_representation(self):
        """Test chord string representation"""
        chord = Chord(root=0, quality="maj7", extensions=["9", "#11"])
        chord_str = str(chord)
        self.assertIn("C", chord_str)


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_tests(verbosity=2):
    """Run all tests with specified verbosity"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestUpperStructureTriads))
    suite.addTests(loader.loadTestsFromTestCase(TestPolychords))
    suite.addTests(loader.loadTestsFromTestCase(TestClusterVoicings))
    suite.addTests(loader.loadTestsFromTestCase(TestSlashChords))
    suite.addTests(loader.loadTestsFromTestCase(TestAlteredDominants))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiTonicAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("EXTENDED HARMONY MODULE - Unit Tests")
    print("=" * 70)
    print()

    result = run_tests(verbosity=2)

    print()
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
