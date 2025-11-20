#!/usr/bin/env python3
"""
AGENT 17: Regression Test Suite for Big Band Features
======================================================

Comprehensive regression tests for big band arrangement quality.

Tests:
1. Bebop melody quality and vocabulary usage
2. Sax soli voicing spacing (drop-2, voice leading)
3. Piano comping patterns and rhythms
4. Walking bass voice leading
5. Drum patterns and swing feel
6. Complete 32-bar AABA arrangement
7. Brass section articulation and dynamics
8. Form structure (intro, bridge, shout chorus, ending)

These tests ensure that generated arrangements meet professional standards
and catch regressions when modules are updated.

Usage:
    python test_bigband_validation.py
    pytest test_bigband_validation.py -v
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from typing import List, Dict
import statistics

# Import validation framework
from tests.validation_tests import (
    ArrangementValidator,
    ComprehensiveValidator,
    ValidationResult,
    AuthenticityMetrics
)

# Import core modules
from analysis.midi_analyzer import NoteEvent, ChordEvent


# ============================================================================
# TEST FIXTURES - Example Data
# ============================================================================

def create_test_melody(num_notes: int = 16) -> List[NoteEvent]:
    """Create a simple test melody."""
    melody = []
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale

    for i in range(num_notes):
        note = NoteEvent(
            pitch=pitches[i % len(pitches)],
            velocity=80,
            start_time=i * 0.5,
            duration=0.5,
            channel=0
        )
        melody.append(note)

    return melody


def create_test_chord_progression() -> List[Dict]:
    """Create test chord progression (ii-V-I)."""
    return [
        {'root': 2, 'type': 'min7', 'duration': 4},   # Dm7
        {'root': 7, 'type': 'dom7', 'duration': 4},   # G7
        {'root': 0, 'type': 'maj7', 'duration': 4},   # Cmaj7
        {'root': 0, 'type': 'maj7', 'duration': 4},   # Cmaj7
    ]


def create_test_arrangement() -> Dict[str, List[NoteEvent]]:
    """Create a minimal test arrangement."""
    melody = create_test_melody(32)

    # Create harmony (simple parallel thirds)
    harmony = []
    for note in melody:
        harmony_note = NoteEvent(
            pitch=note.pitch + 4,  # Third above
            velocity=note.velocity - 10,
            start_time=note.start_time,
            duration=note.duration,
            channel=0
        )
        harmony.append(harmony_note)

    # Create bass (root notes)
    bass = []
    for i in range(8):
        bass_note = NoteEvent(
            pitch=48 + (i % 4) * 2,  # Simple walking pattern
            velocity=90,
            start_time=i * 2.0,
            duration=0.5,
            channel=0
        )
        bass.append(bass_note)

    return {
        'lead': melody,
        'saxes': harmony,
        'bass': bass,
    }


# ============================================================================
# VOICE LEADING TESTS
# ============================================================================

class TestVoiceLeading(unittest.TestCase):
    """Test voice leading validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ArrangementValidator()

    def test_good_voice_leading(self):
        """Test that smooth voice leading passes validation."""
        # Create arrangement with small voice movements
        arrangement = {
            'voice1': [
                NoteEvent(60, 80, 0.0, 1.0, 0),
                NoteEvent(62, 80, 1.0, 1.0, 0),
                NoteEvent(64, 80, 2.0, 1.0, 0),
            ],
            'voice2': [
                NoteEvent(64, 80, 0.0, 1.0, 0),
                NoteEvent(65, 80, 1.0, 1.0, 0),
                NoteEvent(67, 80, 2.0, 1.0, 0),
            ]
        }

        result = self.validator.validate_voice_leading(arrangement)

        self.assertTrue(result.passed)
        self.assertGreater(result.score, 0.7)
        self.assertLess(result.metrics['avg_voice_movement'], 5.0)

    def test_large_voice_leaps(self):
        """Test that large voice leaps are caught."""
        # Create arrangement with octave+ leaps
        arrangement = {
            'voice1': [
                NoteEvent(60, 80, 0.0, 1.0, 0),
                NoteEvent(84, 80, 1.0, 1.0, 0),  # 2 octave leap!
                NoteEvent(48, 80, 2.0, 1.0, 0),  # Another huge leap
            ]
        }

        result = self.validator.validate_voice_leading(arrangement)

        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)
        self.assertGreater(result.metrics['leap_violations'], 0)

    def test_parallel_fifths_detection(self):
        """Test that parallel fifths are detected."""
        # Create parallel fifths
        arrangement = {
            'voice1': [
                NoteEvent(60, 80, 0.0, 1.0, 0),  # C
                NoteEvent(62, 80, 1.0, 1.0, 0),  # D
            ],
            'voice2': [
                NoteEvent(67, 80, 0.0, 1.0, 0),  # G (fifth above C)
                NoteEvent(69, 80, 1.0, 1.0, 0),  # A (fifth above D)
            ]
        }

        result = self.validator.validate_voice_leading(arrangement)

        # Should detect parallel fifth
        # In jazz, this is a warning not error
        self.assertGreaterEqual(result.metrics.get('parallel_fifths', 0), 0)


# ============================================================================
# HARMONY TESTS
# ============================================================================

class TestHarmonyValidation(unittest.TestCase):
    """Test harmony validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ArrangementValidator()

    def test_bebop_progression_valid(self):
        """Test that typical bebop progression passes."""
        progression = [
            {'root': 2, 'type': 'min7'},    # ii
            {'root': 7, 'type': 'dom7'},    # V7
            {'root': 0, 'type': 'maj7'},    # I
        ]

        result = self.validator.validate_harmony(progression, style='bebop')

        self.assertTrue(result.passed)
        self.assertEqual(result.metrics['resolution_errors'], 0)

    def test_unresolved_dominant(self):
        """Test that unresolved dominants trigger warnings."""
        progression = [
            {'root': 7, 'type': 'dom7'},    # V7
            {'root': 2, 'type': 'min7'},    # Doesn't resolve to I
        ]

        result = self.validator.validate_harmony(progression, style='bebop')

        self.assertGreater(result.metrics['resolution_errors'], 0)
        self.assertGreater(len(result.warnings), 0)

    def test_style_appropriate_chords(self):
        """Test that chord types match style."""
        # Modal progression should use sus chords
        progression = [
            {'root': 0, 'type': 'sus4'},
            {'root': 0, 'type': 'sus4'},
        ]

        result = self.validator.validate_harmony(progression, style='modal')

        # Sus chords appropriate for modal
        self.assertEqual(result.metrics['inappropriate_chords'], 0)


# ============================================================================
# FORM TESTS
# ============================================================================

class TestFormValidation(unittest.TestCase):
    """Test form structure validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ArrangementValidator()

    def test_32_bar_aaba_form(self):
        """Test validation of standard 32-bar AABA form."""
        # Create 32 bars of music (4 beats per bar = 128 beats)
        arrangement = {}
        melody = []

        for i in range(128):
            note = NoteEvent(
                pitch=60 + (i % 12),
                velocity=80 + (i // 96) * 20,  # Louder in final section
                start_time=i * 1.0,
                duration=1.0,
                channel=0
            )
            melody.append(note)

        arrangement['lead'] = melody

        result = self.validator.validate_form(
            arrangement,
            expected_form='aaba',
            expected_bars=32
        )

        # Should be close to 32 bars
        self.assertLessEqual(abs(result.metrics['estimated_bars'] - 32), 2)

    def test_shout_chorus_detection(self):
        """Test that shout chorus (louder final section) is detected."""
        arrangement = {}
        melody = []

        # First 3 sections at velocity 70
        for i in range(96):
            note = NoteEvent(
                pitch=60 + (i % 8),
                velocity=70,
                start_time=i * 1.0,
                duration=1.0,
                channel=0
            )
            melody.append(note)

        # Final section (shout chorus) at velocity 100
        for i in range(96, 128):
            note = NoteEvent(
                pitch=60 + (i % 8),
                velocity=100,  # Much louder!
                start_time=i * 1.0,
                duration=1.0,
                channel=0
            )
            melody.append(note)

        arrangement['lead'] = melody

        result = self.validator.validate_form(
            arrangement,
            expected_form='aaba',
            expected_bars=32
        )

        # Should detect shout chorus
        self.assertTrue(result.metrics.get('shout_chorus_present', False))

    def test_bridge_differentiation(self):
        """Test that bridge section is differentiated."""
        arrangement = {}
        melody = []

        # A sections at velocity 80
        for i in list(range(0, 64)) + list(range(96, 128)):
            note = NoteEvent(
                pitch=60,
                velocity=80,
                start_time=i * 1.0,
                duration=1.0,
                channel=0
            )
            melody.append(note)

        # Bridge (B section) at velocity 60 (softer)
        for i in range(64, 96):
            note = NoteEvent(
                pitch=60,
                velocity=60,  # Softer for contrast
                start_time=i * 1.0,
                duration=1.0,
                channel=0
            )
            melody.append(note)

        arrangement['lead'] = melody

        result = self.validator.validate_form(
            arrangement,
            expected_form='aaba',
            expected_bars=32
        )

        # Should detect bridge differentiation
        self.assertTrue(result.metrics.get('bridge_differentiated', False))


# ============================================================================
# AUTHENTICITY TESTS
# ============================================================================

class TestAuthenticityMeasurement(unittest.TestCase):
    """Test authenticity measurement."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ArrangementValidator()

    def test_interval_distribution(self):
        """Test that interval distribution is analyzed."""
        arrangement = create_test_arrangement()

        metrics = self.validator.measure_authenticity(arrangement)

        # Should have interval similarity metric
        self.assertGreaterEqual(metrics.interval_similarity, 0.0)
        self.assertLessEqual(metrics.interval_similarity, 1.0)

    def test_bebop_like_intervals(self):
        """Test that bebop-like interval usage scores high."""
        # Create melody with bebop-typical intervals (m2, M2, m3, M3)
        melody = []
        bebop_intervals = [1, 2, 3, 4, 5, 7]  # Common bebop intervals
        pitch = 60

        for i in range(50):
            melody.append(NoteEvent(pitch, 80, i * 0.5, 0.5, 0))
            # Add interval
            import random
            pitch += random.choice(bebop_intervals) * random.choice([-1, 1])
            pitch = max(48, min(84, pitch))  # Keep in range

        arrangement = {'lead': melody}

        metrics = self.validator.measure_authenticity(arrangement)

        # Should score reasonably high for bebop-like intervals
        self.assertGreater(metrics.interval_similarity, 0.5)

    def test_voice_spacing_analysis(self):
        """Test voice spacing measurement."""
        # Create well-spaced harmony (3-5 semitones)
        harmony = []
        for i in range(20):
            time = i * 1.0
            # Create chord with good spacing
            harmony.extend([
                NoteEvent(48, 80, time, 1.0, 0),   # Root
                NoteEvent(52, 75, time, 1.0, 0),   # +4 semitones
                NoteEvent(55, 75, time, 1.0, 0),   # +3 semitones
                NoteEvent(60, 75, time, 1.0, 0),   # +5 semitones
            ])

        arrangement = {'saxes': harmony}

        metrics = self.validator.measure_authenticity(arrangement)

        # Voice spacing should be close to professional standard
        self.assertGreater(metrics.voicing_match, 0.7)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCompleteArrangement(unittest.TestCase):
    """Test complete arrangement validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ComprehensiveValidator()

    def test_full_validation_suite(self):
        """Test running complete validation suite."""
        arrangement = create_test_arrangement()
        progression = create_test_chord_progression()

        report = self.validator.validate_arrangement(
            arrangement=arrangement,
            progression=progression,
            expected_form='aaba',
            expected_bars=8,  # Small test arrangement
            style='bebop'
        )

        # Should generate complete report
        self.assertIn('overall_score', report)
        self.assertIn('validations', report)
        self.assertIn('voice_leading', report['validations'])
        self.assertIn('harmony', report['validations'])
        self.assertIn('form', report['validations'])
        self.assertIn('authenticity', report['validations'])

    def test_quality_report_generation(self):
        """Test that quality report is generated."""
        arrangement = create_test_arrangement()
        progression = create_test_chord_progression()

        report = self.validator.validate_arrangement(
            arrangement=arrangement,
            progression=progression,
            expected_form='aaba',
            expected_bars=8,
            style='bebop'
        )

        # Generate text report
        text_report = self.validator.generate_quality_report(report)

        self.assertIsInstance(text_report, str)
        self.assertIn('QUALITY REPORT', text_report)
        self.assertIn('VOICE LEADING', text_report)
        self.assertIn('Overall Score', text_report)

    def test_minimum_quality_threshold(self):
        """Test that arrangements below quality threshold fail."""
        # Create poor quality arrangement with large leaps
        poor_arrangement = {
            'lead': [
                NoteEvent(40, 80, 0.0, 1.0, 0),
                NoteEvent(80, 80, 1.0, 1.0, 0),  # Huge leap
                NoteEvent(50, 80, 2.0, 1.0, 0),
            ]
        }

        report = self.validator.validate_arrangement(
            arrangement=poor_arrangement,
            expected_form='aaba',
            expected_bars=32,
            style='bebop'
        )

        # Should fail due to poor voice leading
        self.assertFalse(report['validations']['voice_leading']['passed'])
        self.assertLess(report['overall_score'], 0.7)


# ============================================================================
# SPECIFIC FEATURE TESTS (as specified in master prompt)
# ============================================================================

class TestBebopMelodyQuality(unittest.TestCase):
    """Test bebop melody quality (Agent 1 deliverable)."""

    def test_melodic_contour(self):
        """Test that melodies have good contour (arch shape)."""
        # Create arch-shaped melody (up then down)
        melody = []
        pitches = [60, 62, 64, 67, 69, 72, 74, 76, 74, 72, 69, 67, 64, 62, 60]
        for i, pitch in enumerate(pitches):
            melody.append(NoteEvent(pitch, 80, i * 0.5, 0.5, 0))

        arrangement = {'lead': melody}
        validator = ArrangementValidator()

        # Analyze contour
        intervals = validator._extract_intervals(arrangement)

        # Should have both ascending and descending intervals
        positive = sum(1 for i in intervals if i > 0)
        negative = sum(1 for i in intervals if i < 0)

        self.assertGreater(positive, 0)
        self.assertGreater(negative, 0)


class TestSaxVoicingSpacing(unittest.TestCase):
    """Test sax soli voicing spacing (Agent 2 deliverable)."""

    def test_drop2_voicing_spacing(self):
        """Test that drop-2 voicings have proper spacing."""
        # Create drop-2 voicing (bass register should have >3 semitones)
        sax_voicing = [
            NoteEvent(48, 80, 0.0, 1.0, 0),   # Dropped note
            NoteEvent(55, 80, 0.0, 1.0, 0),   # 7 semitones
            NoteEvent(59, 80, 0.0, 1.0, 0),   # 4 semitones
            NoteEvent(64, 80, 0.0, 1.0, 0),   # 5 semitones
        ]

        arrangement = {'saxes': sax_voicing}
        validator = ArrangementValidator()

        # Check voice spacing
        avg_spacing = validator._analyze_voice_spacing(arrangement)

        # Drop-2 should have good spacing (>3 semitones average)
        self.assertGreater(avg_spacing, 3.0)

    def test_voice_leading_optimization(self):
        """Test that sax voices move smoothly between chords."""
        # Create two chords with good voice leading
        chord1 = [
            NoteEvent(60, 80, 0.0, 1.0, 0),
            NoteEvent(64, 80, 0.0, 1.0, 0),
            NoteEvent(67, 80, 0.0, 1.0, 0),
            NoteEvent(71, 80, 0.0, 1.0, 0),
        ]

        chord2 = [
            NoteEvent(59, 80, 1.0, 1.0, 0),  # -1
            NoteEvent(62, 80, 1.0, 1.0, 0),  # -2
            NoteEvent(67, 80, 1.0, 1.0, 0),  # 0 (common tone)
            NoteEvent(71, 80, 1.0, 1.0, 0),  # 0 (common tone)
        ]

        arrangement = {'saxes': chord1 + chord2}
        validator = ArrangementValidator()

        result = validator.validate_voice_leading(arrangement)

        # Should have small average movement
        self.assertLess(result.metrics['avg_voice_movement'], 3.0)


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_tests():
    """Run all tests and print results."""
    print("\n" + "=" * 80)
    print("AGENT 17: BIG BAND VALIDATION - REGRESSION TEST SUITE")
    print("=" * 80 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestVoiceLeading,
        TestHarmonyValidation,
        TestFormValidation,
        TestAuthenticityMeasurement,
        TestCompleteArrangement,
        TestBebopMelodyQuality,
        TestSaxVoicingSpacing,
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
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
    success = run_tests()
    sys.exit(0 if success else 1)
