#!/usr/bin/env python3
"""
Comprehensive Tests for Film Scoring Engine

Tests cover:
1. Video analysis components
2. Music generation techniques
3. Leitmotif system
4. Tension mapping
5. SMPTE timecode
6. Integration with chord/melody modules
"""

import unittest
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from film_scoring_engine import (
    FilmScoringEngine,
    FilmScoringTechniques,
    LeitmotifEngine,
    Leitmotif,
    TensionArc,
    SMPTETimecode,
    VideoFeatures,
    MoodCategory,
    TensionLevel,
    ScoringSyncType,
)


class TestSMPTETimecode(unittest.TestCase):
    """Test SMPTE timecode conversions"""

    def test_timecode_creation(self):
        """Test creating timecode"""
        tc = SMPTETimecode(1, 30, 45, 12, framerate=24.0)
        self.assertEqual(tc.hours, 1)
        self.assertEqual(tc.minutes, 30)
        self.assertEqual(tc.seconds, 45)
        self.assertEqual(tc.frames, 12)

    def test_timecode_to_seconds(self):
        """Test converting timecode to seconds"""
        tc = SMPTETimecode(0, 1, 30, 12, framerate=24.0)
        seconds = tc.to_seconds()
        expected = 90.5  # 1*60 + 30 + 12/24
        self.assertAlmostEqual(seconds, expected, places=2)

    def test_timecode_from_seconds(self):
        """Test creating timecode from seconds"""
        tc = SMPTETimecode.from_seconds(95.5, framerate=24.0)
        self.assertEqual(tc.minutes, 1)
        self.assertEqual(tc.seconds, 35)
        self.assertAlmostEqual(tc.frames, 12, delta=1)

    def test_timecode_string_representation(self):
        """Test timecode string formatting"""
        tc = SMPTETimecode(1, 30, 45, 12)
        self.assertEqual(str(tc), "01:30:45:12")


class TestTensionArc(unittest.TestCase):
    """Test tension arc interpolation"""

    def test_tension_arc_creation(self):
        """Test creating tension arc"""
        arc = TensionArc(
            timestamps=[0.0, 10.0, 20.0],
            tension_values=[0.2, 0.8, 0.3]
        )
        self.assertEqual(len(arc.timestamps), 3)
        self.assertEqual(len(arc.tension_values), 3)

    def test_tension_interpolation(self):
        """Test tension value interpolation"""
        arc = TensionArc(
            timestamps=[0.0, 10.0, 20.0],
            tension_values=[0.0, 1.0, 0.0]
        )

        # Test exact points
        self.assertAlmostEqual(arc.get_tension_at(0.0), 0.0, places=2)
        self.assertAlmostEqual(arc.get_tension_at(10.0), 1.0, places=2)
        self.assertAlmostEqual(arc.get_tension_at(20.0), 0.0, places=2)

        # Test interpolated point (midway should be 0.5)
        self.assertAlmostEqual(arc.get_tension_at(5.0), 0.5, places=1)

    def test_tension_edge_cases(self):
        """Test tension arc edge cases"""
        arc = TensionArc(
            timestamps=[10.0, 20.0],
            tension_values=[0.5, 0.8]
        )

        # Before first timestamp
        self.assertEqual(arc.get_tension_at(5.0), 0.5)

        # After last timestamp
        self.assertEqual(arc.get_tension_at(30.0), 0.8)

    def test_empty_tension_arc(self):
        """Test empty tension arc"""
        arc = TensionArc(timestamps=[], tension_values=[])
        self.assertEqual(arc.get_tension_at(5.0), 0.5)  # Default


class TestFilmScoringTechniques(unittest.TestCase):
    """Test film scoring compositional techniques"""

    def setUp(self):
        self.techniques = FilmScoringTechniques()

    def test_tension_to_chord_complexity(self):
        """Test mapping tension to chord types"""
        # Low tension → simple chords
        self.assertEqual(
            self.techniques.tension_to_chord_complexity(0.1),
            "maj"
        )

        # High tension → complex chords
        self.assertIn(
            self.techniques.tension_to_chord_complexity(0.85),
            ["7b9", "dim"]
        )

    def test_chromatic_voice_leading(self):
        """Test chromatic voice leading generation"""
        chords = self.techniques.chromatic_voice_leading(
            start_chord="C",
            end_chord="E",
            steps=4
        )

        # Should have 5 chords (start + 4 steps)
        self.assertEqual(len(chords), 5)

        # Should all be minor (Zimmer style)
        for chord in chords:
            self.assertIn("m", chord)

    def test_ostinato_patterns(self):
        """Test ostinato pattern generation"""
        # Suspense ostinato
        suspense = self.techniques.ostinato_pattern("C", "suspense")
        self.assertIsInstance(suspense, dict)
        self.assertGreater(len(suspense), 0)

        # Action ostinato
        action = self.techniques.ostinato_pattern("D", "action")
        self.assertIsInstance(action, dict)

        # Mystery ostinato
        mystery = self.techniques.ostinato_pattern("F", "mystery")
        self.assertIsInstance(mystery, dict)

    def test_mood_to_scale_mapping(self):
        """Test visual mood to musical scale conversion"""
        # Bright moods → major
        scale = self.techniques.mood_to_scale_context(MoodCategory.WARM_BRIGHT)
        self.assertEqual(scale, "major")

        # Dark moods → minor/harmonic minor
        scale = self.techniques.mood_to_scale_context(MoodCategory.COOL_DARK)
        self.assertIn(scale, ["minor", "harmonic_minor"])

    def test_progression_morphing(self):
        """Test chord progression morphing"""
        original = {0: "C", 4: "F", 8: "G", 12: "C"}

        # Morph to dark/tense
        morphed_dark = self.techniques.morph_progression(
            original,
            target_mood=MoodCategory.COOL_DARK,
            tension=0.8
        )

        # Should have same number of chords
        self.assertEqual(len(morphed_dark), len(original))

        # Should have beats at same positions
        self.assertEqual(set(morphed_dark.keys()), set(original.keys()))

        # High tension + dark mood should produce minor chords
        for chord in morphed_dark.values():
            self.assertIn("m", chord.lower())


class TestLeitmotifEngine(unittest.TestCase):
    """Test leitmotif system"""

    def setUp(self):
        self.engine = LeitmotifEngine()

    def test_register_motif(self):
        """Test registering leitmotifs"""
        motif = Leitmotif(
            name="Hero",
            chord_progression={0: "C", 4: "G"},
            harmonic_character="major"
        )

        self.engine.register_motif(motif)
        self.assertIn("Hero", self.engine.motifs)
        self.assertEqual(self.engine.motifs["Hero"].name, "Hero")

    def test_get_variation_basic(self):
        """Test getting basic leitmotif variation"""
        motif = Leitmotif(
            name="Test",
            chord_progression={0: "C", 4: "F", 8: "G"},
            harmonic_character="major"
        )
        self.engine.register_motif(motif)

        variation = self.engine.get_variation("Test", tension=0.5)
        self.assertIsInstance(variation, dict)
        self.assertGreater(len(variation), 0)

    def test_get_variation_augmentation(self):
        """Test rhythmic augmentation (slower)"""
        motif = Leitmotif(
            name="Test",
            chord_progression={0: "C", 2: "F", 4: "G"},
            can_augment=True
        )
        self.engine.register_motif(motif)

        # High tension should trigger augmentation
        variation = self.engine.get_variation("Test", tension=0.9, tempo_factor=0.5)

        # Beats should be stretched
        # Original: 0, 2, 4 → Augmented: 0, 4, 8
        self.assertIn(0, variation)

    def test_get_variation_transposition(self):
        """Test transposing leitmotif"""
        motif = Leitmotif(
            name="Test",
            chord_progression={0: "C"},
            can_transpose=True
        )
        self.engine.register_motif(motif)

        # Transpose up 2 semitones (C → D)
        variation = self.engine.get_variation(
            "Test",
            tension=0.5,
            transpose_semitones=2
        )

        # Should contain D chord
        self.assertEqual(variation[0][0], "D")

    def test_nonexistent_motif(self):
        """Test requesting nonexistent leitmotif"""
        variation = self.engine.get_variation("NonExistent", tension=0.5)

        # Should return fallback
        self.assertEqual(variation, {0: "C"})


class TestVideoFeatures(unittest.TestCase):
    """Test video feature data structures"""

    def test_video_features_creation(self):
        """Test creating VideoFeatures object"""
        features = VideoFeatures(
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            avg_brightness=0.6,
            avg_saturation=0.7,
            mood=MoodCategory.WARM_BRIGHT,
            visual_tension=0.4
        )

        self.assertEqual(features.start_time, 0.0)
        self.assertEqual(features.end_time, 10.0)
        self.assertEqual(features.duration, 10.0)
        self.assertEqual(features.mood, MoodCategory.WARM_BRIGHT)
        self.assertAlmostEqual(features.visual_tension, 0.4)

    def test_video_features_defaults(self):
        """Test default values"""
        features = VideoFeatures(
            start_time=0.0,
            end_time=5.0,
            duration=5.0
        )

        # Should have default values
        self.assertGreaterEqual(features.avg_brightness, 0.0)
        self.assertLessEqual(features.avg_brightness, 1.0)
        self.assertIsInstance(features.mood, MoodCategory)


class TestFilmScoringEngine(unittest.TestCase):
    """Test main film scoring engine"""

    def test_engine_creation_without_video(self):
        """Test creating engine without video"""
        engine = FilmScoringEngine(video_path=None, bpm=120)

        self.assertEqual(engine.bpm, 120)
        self.assertIsNone(engine.video_analyzer)
        self.assertIsInstance(engine.leitmotif_engine, LeitmotifEngine)
        self.assertIsInstance(engine.techniques, FilmScoringTechniques)

    def test_engine_components(self):
        """Test engine has all components"""
        engine = FilmScoringEngine(video_path=None, bpm=120)

        # Check components exist
        self.assertIsNotNone(engine.leitmotif_engine)
        self.assertIsNotNone(engine.techniques)
        self.assertIsInstance(engine.video_features, list)
        self.assertIsNone(engine.tension_arc)  # Not yet analyzed

    def test_generate_progression_from_tension(self):
        """Test generating progression based on tension"""
        engine = FilmScoringEngine(video_path=None, bpm=120)

        # Low tension
        low_prog = engine._generate_progression_from_tension(0.2)
        self.assertIsInstance(low_prog, dict)
        self.assertGreater(len(low_prog), 0)

        # High tension
        high_prog = engine._generate_progression_from_tension(0.9)
        self.assertIsInstance(high_prog, dict)

        # High tension should use more dissonant chords
        # Check if any chord contains "dim" (diminished)
        has_dissonance = any("dim" in chord for chord in high_prog.values())
        self.assertTrue(has_dissonance)


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def test_full_workflow_without_video(self):
        """Test complete workflow without actual video file"""
        # Create engine
        engine = FilmScoringEngine(video_path=None, bpm=120)

        # Create manual video features (simulating analysis)
        engine.video_features = [
            VideoFeatures(
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                mood=MoodCategory.WARM_BRIGHT,
                visual_tension=0.3,
                is_scene_start=True,
                scene_id=0
            ),
            VideoFeatures(
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                mood=MoodCategory.COOL_DARK,
                visual_tension=0.8,
                is_scene_start=True,
                scene_id=1
            )
        ]

        # Create tension arc
        engine.tension_arc = TensionArc(
            timestamps=[0.0, 10.0, 20.0],
            tension_values=[0.3, 0.8, 0.4]
        )

        # Register leitmotif
        hero_theme = Leitmotif(
            name="Hero",
            chord_progression={0: "C", 4: "F", 8: "G", 12: "C"},
            harmonic_character="major"
        )
        engine.leitmotif_engine.register_motif(hero_theme)

        # Generate score
        # Note: This will create a MIDI file - we test it doesn't crash
        try:
            midi_path = engine.generate_score(
                base_progression=hero_theme.chord_progression,
                scoring_approach=ScoringSyncType.TENSION_ARC
            )

            # Should return a path
            self.assertIsInstance(midi_path, str)
            self.assertTrue(len(midi_path) > 0)

        except Exception as e:
            self.fail(f"Score generation failed: {e}")


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_tests(verbosity=2):
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSMPTETimecode))
    suite.addTests(loader.loadTestsFromTestCase(TestTensionArc))
    suite.addTests(loader.loadTestsFromTestCase(TestFilmScoringTechniques))
    suite.addTests(loader.loadTestsFromTestCase(TestLeitmotifEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestVideoFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestFilmScoringEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("="*70)
    print("FILM SCORING ENGINE - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print()

    result = run_tests(verbosity=2)

    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("="*70)

    sys.exit(0 if result.wasSuccessful() else 1)
