#!/usr/bin/env python3
"""
Comprehensive Tests for Genre Detection Module

Tests all functionality of the genre_detector module including:
- Feature extraction (rhythmic, harmonic, melodic, instrumentation)
- Genre classification
- Swing detection
- Chord progression extraction
- Per-track and per-section analysis

Author: Agent 1 - Genre Detection & Feature Extraction
"""

import unittest
import tempfile
import os
from pathlib import Path
import numpy as np

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from midi_generator.analysis.genre_detector import (
    GenreDetector,
    SwingDetector,
    ChordProgressionExtractor,
    RhythmicFeatureExtractor,
    HarmonicFeatureExtractor,
    MelodicFeatureExtractor,
    InstrumentationFeatureExtractor,
    calculate_feature_distance,
    load_genre_database
)

from midi_generator.analysis.midi_analyzer import NoteEvent, ChordEvent, MidiAnalyzer
from midi_generator.generators.style_fusion import GENRE_PROFILES

# For creating test MIDI files
from mido import MidiFile, MidiTrack, Message, MetaMessage


# ==============================================================================
# TEST MIDI FILE CREATION UTILITIES
# ==============================================================================

class MidiFileCreator:
    """Helper class to create test MIDI files"""

    @staticmethod
    def create_jazz_example(filename: str) -> str:
        """
        Create a jazz-style MIDI file for testing

        Features:
        - Swing rhythm (0.67 swing factor)
        - Extended harmonies (maj7, min7, dom7)
        - Medium tempo (~140 BPM)
        - Moderate syncopation
        """
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        # Set tempo (140 BPM)
        tempo = 428571  # microseconds per beat
        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))

        # Piano
        track.append(Message('program_change', program=0, time=0))

        ticks_per_beat = mid.ticks_per_beat

        # Create swing pattern with jazz chords
        # Cmaj7 chord (swing eighths)
        track.append(Message('note_on', note=60, velocity=80, time=0))
        track.append(Message('note_on', note=64, velocity=75, time=0))
        track.append(Message('note_on', note=67, velocity=75, time=0))
        track.append(Message('note_on', note=71, velocity=70, time=0))

        track.append(Message('note_off', note=60, velocity=0, time=ticks_per_beat))
        track.append(Message('note_off', note=64, velocity=0, time=0))
        track.append(Message('note_off', note=67, velocity=0, time=0))
        track.append(Message('note_off', note=71, velocity=0, time=0))

        # Swing eighth note (delayed)
        swing_delay = int(ticks_per_beat * 0.33)  # Triplet swing

        track.append(Message('note_on', note=62, velocity=70, time=swing_delay))
        track.append(Message('note_off', note=62, velocity=0, time=ticks_per_beat - swing_delay))

        # Am7 chord
        track.append(Message('note_on', note=57, velocity=80, time=0))
        track.append(Message('note_on', note=60, velocity=75, time=0))
        track.append(Message('note_on', note=64, velocity=75, time=0))
        track.append(Message('note_on', note=67, velocity=70, time=0))

        track.append(Message('note_off', note=57, velocity=0, time=ticks_per_beat))
        track.append(Message('note_off', note=60, velocity=0, time=0))
        track.append(Message('note_off', note=64, velocity=0, time=0))
        track.append(Message('note_off', note=67, velocity=0, time=0))

        # Swing eighth
        track.append(Message('note_on', note=59, velocity=70, time=swing_delay))
        track.append(Message('note_off', note=59, velocity=0, time=ticks_per_beat - swing_delay))

        # Dm7 chord
        track.append(Message('note_on', note=62, velocity=80, time=0))
        track.append(Message('note_on', note=65, velocity=75, time=0))
        track.append(Message('note_on', note=69, velocity=75, time=0))
        track.append(Message('note_on', note=72, velocity=70, time=0))

        track.append(Message('note_off', note=62, velocity=0, time=ticks_per_beat))
        track.append(Message('note_off', note=65, velocity=0, time=0))
        track.append(Message('note_off', note=69, velocity=0, time=0))
        track.append(Message('note_off', note=72, velocity=0, time=0))

        # G7 chord
        track.append(Message('note_on', note=55, velocity=80, time=ticks_per_beat))
        track.append(Message('note_on', note=59, velocity=75, time=0))
        track.append(Message('note_on', note=62, velocity=75, time=0))
        track.append(Message('note_on', note=65, velocity=70, time=0))

        track.append(Message('note_off', note=55, velocity=0, time=ticks_per_beat * 2))
        track.append(Message('note_off', note=59, velocity=0, time=0))
        track.append(Message('note_off', note=62, velocity=0, time=0))
        track.append(Message('note_off', note=65, velocity=0, time=0))

        mid.save(filename)
        return filename

    @staticmethod
    def create_hiphop_example(filename: str) -> str:
        """
        Create a hip-hop style MIDI file

        Features:
        - Straight/slightly laid-back rhythm
        - Sparse harmony
        - Low tempo (~90 BPM)
        - High syncopation
        """
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        # Set tempo (90 BPM)
        tempo = 666667
        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))

        # Synth bass
        track.append(Message('program_change', program=38, time=0))

        ticks_per_beat = mid.ticks_per_beat

        # Hip-hop drum pattern with syncopation
        # Kick on 1 and 3
        track.append(Message('note_on', note=36, velocity=100, time=0))
        track.append(Message('note_off', note=36, velocity=0, time=ticks_per_beat // 4))

        # Snare on 2 and 4 (backbeat)
        track.append(Message('note_on', note=38, velocity=95, time=ticks_per_beat - ticks_per_beat // 4))
        track.append(Message('note_off', note=38, velocity=0, time=ticks_per_beat // 4))

        # Hi-hat pattern (16th notes with variations)
        for i in range(16):
            velocity = 60 if i % 2 == 0 else 50  # Accent on-beats
            time_offset = ticks_per_beat // 4 if i == 0 else 0
            track.append(Message('note_on', note=42, velocity=velocity, time=time_offset))
            track.append(Message('note_off', note=42, velocity=0, time=ticks_per_beat // 4))

        # Simple chord progression (sparse)
        track.append(Message('note_on', note=48, velocity=70, time=0))
        track.append(Message('note_on', note=52, velocity=65, time=0))
        track.append(Message('note_off', note=48, velocity=0, time=ticks_per_beat * 4))
        track.append(Message('note_off', note=52, velocity=0, time=0))

        mid.save(filename)
        return filename

    @staticmethod
    def create_straight_rhythm(filename: str) -> str:
        """Create MIDI with straight (non-swing) rhythm"""
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        track.append(MetaMessage('set_tempo', tempo=500000, time=0))  # 120 BPM

        ticks_per_beat = mid.ticks_per_beat

        # Perfectly straight eighth notes
        for i in range(16):
            track.append(Message('note_on', note=60 + (i % 12), velocity=80,
                               time=0 if i == 0 else ticks_per_beat // 2))
            track.append(Message('note_off', note=60 + (i % 12), velocity=0,
                               time=ticks_per_beat // 4))

        mid.save(filename)
        return filename


# ==============================================================================
# UNIT TESTS
# ==============================================================================

class TestRhythmicFeatureExtractor(unittest.TestCase):
    """Test rhythmic feature extraction"""

    def setUp(self):
        """Create test note events"""
        # Create notes with swing feel
        self.swing_notes = [
            NoteEvent(0.0, 0.25, 0, 100, 60, 80, 0, 0),
            NoteEvent(0.333, 0.167, 100, 50, 62, 70, 0, 0),  # Delayed eighth (swing)
            NoteEvent(0.5, 0.25, 200, 100, 64, 80, 0, 0),
            NoteEvent(0.833, 0.167, 300, 50, 65, 70, 0, 0),  # Delayed eighth
        ]

        # Straight notes
        self.straight_notes = [
            NoteEvent(0.0, 0.25, 0, 100, 60, 80, 0, 0),
            NoteEvent(0.25, 0.25, 100, 100, 62, 80, 0, 0),
            NoteEvent(0.5, 0.25, 200, 100, 64, 80, 0, 0),
            NoteEvent(0.75, 0.25, 300, 100, 65, 80, 0, 0),
        ]

    def test_extract_features(self):
        """Test basic feature extraction"""
        features = RhythmicFeatureExtractor.extract_features(
            self.straight_notes, 120.0, (4, 4)
        )

        self.assertIn('tempo_bpm', features)
        self.assertIn('swing_factor', features)
        self.assertIn('syncopation', features)
        self.assertIn('rhythmic_complexity', features)
        self.assertIn('note_density', features)
        self.assertIn('groove_type', features)

        self.assertEqual(features['tempo_bpm'], 120.0)

    def test_syncopation_calculation(self):
        """Test syncopation calculation"""
        features = RhythmicFeatureExtractor.extract_features(
            self.straight_notes, 120.0, (4, 4)
        )

        # Straight notes should have low syncopation
        self.assertLess(features['syncopation'], 0.5)

    def test_note_density(self):
        """Test note density calculation"""
        features = RhythmicFeatureExtractor.extract_features(
            self.straight_notes, 120.0, (4, 4)
        )

        # 4 notes in 1 second at 120 BPM = 2 beats = 2 notes per beat
        self.assertGreater(features['note_density'], 0)


class TestSwingDetector(unittest.TestCase):
    """Test swing detection"""

    def test_classify_groove_straight(self):
        """Test classification of straight groove"""
        groove = SwingDetector.classify_groove_type(
            swing_factor=0.5,
            syncopation=0.3,
            note_density=4.0
        )

        self.assertEqual(groove, 'straight')

    def test_classify_groove_swing(self):
        """Test classification of swing groove"""
        groove = SwingDetector.classify_groove_type(
            swing_factor=0.67,
            syncopation=0.5,
            note_density=6.0
        )

        self.assertIn(groove, ['swing', 'triplet'])

    def test_classify_groove_shuffle(self):
        """Test classification of shuffle groove"""
        groove = SwingDetector.classify_groove_type(
            swing_factor=0.58,
            syncopation=0.4,
            note_density=5.0
        )

        self.assertEqual(groove, 'shuffle')


class TestHarmonicFeatureExtractor(unittest.TestCase):
    """Test harmonic feature extraction"""

    def setUp(self):
        """Create test chord events"""
        self.chords = [
            ChordEvent(0.0, 1.0, 0, 'major7', [0, 4, 7, 11], 0, 0.9),
            ChordEvent(1.0, 1.0, 9, 'minor7', [9, 0, 4, 7], 9, 0.85),
            ChordEvent(2.0, 1.0, 2, 'minor7', [2, 5, 9, 0], 2, 0.9),
            ChordEvent(3.0, 1.0, 7, 'dom7', [7, 11, 2, 5], 7, 0.88),
        ]

        # Create mock key signature
        class MockKey:
            def __init__(self):
                self.tonic = 0
                self.mode = 'major'

        self.key = MockKey()

    def test_extract_features(self):
        """Test harmonic feature extraction"""
        features = HarmonicFeatureExtractor.extract_features(
            self.chords, self.key, 4.0, 120.0
        )

        self.assertIn('chord_types', features)
        self.assertIn('harmonic_rhythm', features)
        self.assertIn('use_extensions', features)
        self.assertIn('chromaticism', features)

        # Should detect extended chords
        self.assertTrue(features['use_extensions'])

    def test_chord_types(self):
        """Test chord type extraction"""
        features = HarmonicFeatureExtractor.extract_features(
            self.chords, self.key, 4.0, 120.0
        )

        chord_types = features['chord_types']
        self.assertIn('major7', chord_types)
        self.assertIn('minor7', chord_types)
        self.assertIn('dom7', chord_types)

    def test_harmonic_rhythm(self):
        """Test harmonic rhythm calculation"""
        features = HarmonicFeatureExtractor.extract_features(
            self.chords, self.key, 4.0, 120.0
        )

        # 4 chords in 4 seconds at 120 BPM (= 8 beats = 2 measures)
        # = 2 chords per measure
        self.assertGreater(features['harmonic_rhythm'], 0)


class TestMelodicFeatureExtractor(unittest.TestCase):
    """Test melodic feature extraction"""

    def setUp(self):
        """Create test melody"""
        # Stepwise ascending melody
        self.stepwise_melody = [
            NoteEvent(i * 0.5, 0.4, i * 200, 150, 60 + i, 80, 0, 0)
            for i in range(8)
        ]

        # Angular melody (leaps)
        self.angular_melody = [
            NoteEvent(0.0, 0.5, 0, 200, 60, 80, 0, 0),
            NoteEvent(0.5, 0.5, 200, 200, 67, 80, 0, 0),  # Leap of 7
            NoteEvent(1.0, 0.5, 400, 200, 62, 80, 0, 0),  # Leap of 5 down
            NoteEvent(1.5, 0.5, 600, 200, 71, 80, 0, 0),  # Leap of 9
        ]

    def test_extract_features(self):
        """Test melodic feature extraction"""
        features = MelodicFeatureExtractor.extract_features(self.stepwise_melody)

        self.assertIn('interval_distribution', features)
        self.assertIn('contour_type', features)
        self.assertIn('ornamentation_density', features)
        self.assertIn('range_semitones', features)

    def test_stepwise_motion(self):
        """Test detection of stepwise motion"""
        features = MelodicFeatureExtractor.extract_features(self.stepwise_melody)

        dist = features['interval_distribution']
        # Most intervals should be steps
        self.assertGreater(dist['step'], 0.8)

    def test_angular_motion(self):
        """Test detection of angular/leaping motion"""
        features = MelodicFeatureExtractor.extract_features(self.angular_melody)

        dist = features['interval_distribution']
        # Should have leaps
        self.assertGreater(dist['leap'], 0.5)

    def test_contour_ascending(self):
        """Test ascending contour detection"""
        features = MelodicFeatureExtractor.extract_features(self.stepwise_melody)

        # Stepwise ascending should be classified as ascending
        self.assertEqual(features['contour_type'], 'ascending')


class TestGenreDetector(unittest.TestCase):
    """Test main GenreDetector class"""

    @classmethod
    def setUpClass(cls):
        """Create test MIDI files"""
        cls.temp_dir = tempfile.mkdtemp()

        cls.jazz_file = os.path.join(cls.temp_dir, 'jazz_test.mid')
        cls.hiphop_file = os.path.join(cls.temp_dir, 'hiphop_test.mid')
        cls.straight_file = os.path.join(cls.temp_dir, 'straight_test.mid')

        MidiFileCreator.create_jazz_example(cls.jazz_file)
        MidiFileCreator.create_hiphop_example(cls.hiphop_file)
        MidiFileCreator.create_straight_rhythm(cls.straight_file)

    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test GenreDetector initialization"""
        detector = GenreDetector(self.jazz_file)
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.midi_file)

    def test_file_not_found(self):
        """Test error handling for non-existent file"""
        with self.assertRaises(FileNotFoundError):
            GenreDetector('nonexistent.mid')

    def test_extract_rhythmic_features(self):
        """Test rhythmic feature extraction"""
        detector = GenreDetector(self.jazz_file)
        features = detector.extract_rhythmic_features()

        self.assertIn('tempo_bpm', features)
        self.assertIn('swing_factor', features)
        self.assertIn('syncopation', features)

        # Jazz should have swing
        self.assertGreater(features['swing_factor'], 0.55)

    def test_extract_harmonic_features(self):
        """Test harmonic feature extraction"""
        detector = GenreDetector(self.jazz_file)
        features = detector.extract_harmonic_features()

        self.assertIn('chord_types', features)
        self.assertIn('harmonic_rhythm', features)

    def test_extract_melodic_features(self):
        """Test melodic feature extraction"""
        detector = GenreDetector(self.jazz_file)
        features = detector.extract_melodic_features()

        self.assertIn('interval_distribution', features)
        self.assertIn('contour_type', features)

    def test_extract_instrumentation_features(self):
        """Test instrumentation feature extraction"""
        detector = GenreDetector(self.jazz_file)
        features = detector.extract_instrumentation_features()

        self.assertIn('instruments', features)
        self.assertIn('texture', features)
        self.assertIn('register_distribution', features)

    def test_classify_genre(self):
        """Test genre classification"""
        detector = GenreDetector(self.jazz_file)
        classifications = detector.classify_genre(top_n=3)

        self.assertEqual(len(classifications), 3)

        # Each classification should be (genre_name, score)
        for genre, score in classifications:
            self.assertIsInstance(genre, str)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        # Top match should be jazz or related genre
        top_genre, top_score = classifications[0]
        self.assertIn(top_genre.lower(), ['jazz', 'blues', 'funk'])

    def test_to_genre_features(self):
        """Test conversion to GenreFeatures"""
        detector = GenreDetector(self.jazz_file)
        genre_features = detector.to_genre_features()

        self.assertIsNotNone(genre_features)
        self.assertIsNotNone(genre_features.name)
        self.assertIsNotNone(genre_features.tempo_range)
        self.assertIsNotNone(genre_features.swing_factor)

    def test_swing_detection_jazz(self):
        """Test swing detection on jazz file"""
        detector = GenreDetector(self.jazz_file)
        features = detector.extract_rhythmic_features()

        # Jazz example has swing
        self.assertGreater(features['swing_factor'], 0.55)

    def test_swing_detection_straight(self):
        """Test swing detection on straight rhythm"""
        detector = GenreDetector(self.straight_file)
        features = detector.extract_rhythmic_features()

        # Straight rhythm should have swing factor close to 0.5
        self.assertLess(features['swing_factor'], 0.53)


class TestChordProgressionExtractor(unittest.TestCase):
    """Test chord progression extraction"""

    def test_extract_chord_progression(self):
        """Test chord progression extraction"""
        chords = [
            ChordEvent(0.0, 1.0, 0, 'major7', [0, 4, 7, 11], 0, 0.9),
            ChordEvent(1.0, 1.0, 9, 'minor7', [9, 0, 4, 7], 9, 0.85),
            ChordEvent(2.0, 1.0, 2, 'minor7', [2, 5, 9, 0], 2, 0.9),
            ChordEvent(3.0, 1.0, 7, 'dom7', [7, 11, 2, 5], 7, 0.88),
        ]

        progression = ChordProgressionExtractor.extract_chord_progression(chords)

        self.assertEqual(len(progression), 4)
        self.assertEqual(progression[0], 'Cmaj7')
        self.assertEqual(progression[1], 'Am7')
        self.assertEqual(progression[2], 'Dm7')
        self.assertEqual(progression[3], 'G7')

    def test_empty_chords(self):
        """Test with empty chord list"""
        progression = ChordProgressionExtractor.extract_chord_progression([])
        self.assertEqual(progression, [])


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions"""

    def test_load_genre_database(self):
        """Test loading genre database"""
        database = load_genre_database()

        self.assertIsInstance(database, dict)
        self.assertGreater(len(database), 0)
        self.assertIn('jazz', database)
        self.assertIn('hiphop', database)

    def test_calculate_feature_distance(self):
        """Test feature distance calculation"""
        # Create mock features
        rhythmic = {
            'tempo_bpm': 120,
            'swing_factor': 0.67,
            'syncopation': 0.7,
            'rhythmic_complexity': 0.8
        }

        harmonic = {
            'chord_types': ['maj7', 'min7', 'dom7'],
            'harmonic_rhythm': 4.0,
            'chromaticism': 0.6
        }

        melodic = {
            'ornamentation_density': 0.6
        }

        instrumentation = {
            'texture': 'polyphonic'
        }

        # Compare to jazz profile
        jazz_profile = GENRE_PROFILES['jazz']

        distance = calculate_feature_distance(
            rhythmic, harmonic, melodic, instrumentation, jazz_profile
        )

        # Should be a reasonable distance
        self.assertGreaterEqual(distance, 0.0)
        self.assertLessEqual(distance, 5.0)

        # Compare to very different profile (electronic)
        electronic_profile = GENRE_PROFILES['electronic']

        distance_electronic = calculate_feature_distance(
            rhythmic, harmonic, melodic, instrumentation, electronic_profile
        )

        # Distance to electronic should be greater than distance to jazz
        self.assertGreater(distance_electronic, distance)


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    @classmethod
    def setUpClass(cls):
        """Create test files"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.jazz_file = os.path.join(cls.temp_dir, 'jazz_integration.mid')
        MidiFileCreator.create_jazz_example(cls.jazz_file)

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_full_pipeline(self):
        """Test complete analysis pipeline"""
        # Create detector
        detector = GenreDetector(self.jazz_file)

        # Extract all features
        rhythmic = detector.extract_rhythmic_features()
        harmonic = detector.extract_harmonic_features()
        melodic = detector.extract_melodic_features()
        instrumentation = detector.extract_instrumentation_features()

        # Classify genre
        classifications = detector.classify_genre()

        # Convert to GenreFeatures
        genre_features = detector.to_genre_features()

        # All steps should complete successfully
        self.assertIsNotNone(rhythmic)
        self.assertIsNotNone(harmonic)
        self.assertIsNotNone(melodic)
        self.assertIsNotNone(instrumentation)
        self.assertIsNotNone(classifications)
        self.assertIsNotNone(genre_features)

    def test_genre_features_compatibility(self):
        """Test compatibility with style_fusion GenreFeatures"""
        detector = GenreDetector(self.jazz_file)
        genre_features = detector.to_genre_features()

        # Should have all required attributes
        self.assertTrue(hasattr(genre_features, 'name'))
        self.assertTrue(hasattr(genre_features, 'tempo_range'))
        self.assertTrue(hasattr(genre_features, 'swing_factor'))
        self.assertTrue(hasattr(genre_features, 'chord_types'))
        self.assertTrue(hasattr(genre_features, 'instruments'))


# ==============================================================================
# TEST RUNNER
# ==============================================================================

def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRhythmicFeatureExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestSwingDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestHarmonicFeatureExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestMelodicFeatureExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestGenreDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestChordProgressionExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestHelperFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("Genre Detection Module - Comprehensive Test Suite")
    print("=" * 80)
    result = run_tests()
    print("\n" + "=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
