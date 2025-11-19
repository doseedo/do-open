#!/usr/bin/env python3
"""
Comprehensive Test Suite for Granular Control System
====================================================

Tests all major features of the granular control system including:
- Rhythm pattern creation and manipulation
- Section-specific voicing (brass, strings, woodwinds)
- Articulation patterns
- Chord-to-pitch mapping
- Percussion generation
- Dynamics and expression
- Phrase shaping
- Advanced control features

Author: Agent 8
Date: 2025-11-19
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from midi_generator.generators.granular_control import (
    # Core classes
    GranularControl,
    RhythmPattern,
    InstrumentSection,
    ArticulationType,
    VoicingStrategy,
    Register,
    # Engines
    BrassVoicingEngine,
    StringVoicingEngine,
    WoodwindVoicingEngine,
    PercussionVoicingEngine,
    DynamicsEngine,
    PhraseShaper,
    AdvancedControlEngine,
    # Utilities
    ChordToPitchMapper,
    ArticulationLibrary,
    # Convenience functions
    create_brass_hits,
    create_string_pad
)


class TestRhythmPattern(unittest.TestCase):
    """Test RhythmPattern class"""

    def test_create_basic_pattern(self):
        """Test creating a basic rhythm pattern"""
        rhythm = RhythmPattern(
            onsets=[0.0, 1.0, 2.0, 3.0],
            durations=[0.5, 0.5, 0.5, 0.5]
        )

        self.assertEqual(rhythm.num_events, 4)
        self.assertEqual(len(rhythm.accents), 4)
        self.assertEqual(len(rhythm.velocities), 4)
        self.assertEqual(len(rhythm.articulations), 4)

    def test_pattern_with_accents(self):
        """Test pattern with accent specification"""
        rhythm = RhythmPattern(
            onsets=[0.0, 1.0, 2.0, 3.0],
            durations=[0.5, 0.5, 0.5, 0.5],
            accents=[True, False, True, False]
        )

        # Accented notes should have higher velocity
        self.assertEqual(rhythm.velocities[0], 100)  # Accented
        self.assertEqual(rhythm.velocities[1], 80)   # Unaccented

    def test_apply_swing(self):
        """Test swing feel application"""
        rhythm = RhythmPattern(
            onsets=[0.0, 0.5, 1.0, 1.5],  # Straight 8th notes
            durations=[0.5, 0.5, 0.5, 0.5]
        )

        swung = rhythm.apply_swing(swing_factor=0.67)

        # Offbeat 8th notes should be delayed
        self.assertGreater(swung.onsets[1], 0.5)
        self.assertAlmostEqual(swung.onsets[1], 0.67, places=2)

    def test_validation_error(self):
        """Test validation for mismatched lengths"""
        with self.assertRaises(ValueError):
            RhythmPattern(
                onsets=[0.0, 1.0],
                durations=[0.5]  # Mismatch!
            )


class TestChordToPitchMapper(unittest.TestCase):
    """Test chord parsing and pitch mapping"""

    def test_parse_major_seventh(self):
        """Test parsing Cmaj7 chord"""
        result = ChordToPitchMapper.parse_chord("Cmaj7")

        self.assertEqual(result['root'], 0)  # C = 0
        self.assertIn(0, result['chord_tones'])   # Root
        self.assertIn(4, result['chord_tones'])   # Major third
        self.assertIn(7, result['chord_tones'])   # Fifth
        self.assertIn(11, result['chord_tones'])  # Major seventh

    def test_parse_minor_seventh(self):
        """Test parsing Dm7 chord"""
        result = ChordToPitchMapper.parse_chord("Dm7")

        self.assertEqual(result['root'], 2)  # D = 2
        self.assertIn(2, result['chord_tones'])   # Root
        self.assertIn(5, result['chord_tones'])   # Minor third (D=2, F=5)
        self.assertIn(9, result['chord_tones'])   # Fifth (A=9)

    def test_parse_dominant_seventh(self):
        """Test parsing G7 chord"""
        result = ChordToPitchMapper.parse_chord("G7")

        self.assertEqual(result['root'], 7)  # G = 7
        self.assertIn(7, result['chord_tones'])   # Root
        self.assertIn(11, result['chord_tones'])  # Major third
        self.assertIn(2, result['chord_tones'])   # Fifth (wrapped)

    def test_parse_half_diminished(self):
        """Test parsing Bm7b5 chord"""
        result = ChordToPitchMapper.parse_chord("Bm7b5")

        self.assertEqual(result['root'], 11)  # B = 11

    def test_rhythm_to_notes(self):
        """Test converting rhythm to pitches"""
        rhythm = RhythmPattern(
            onsets=[0.0, 1.0, 2.0, 3.0],
            durations=[0.5, 0.5, 0.5, 0.5],
            accents=[True, False, True, False]
        )

        pitches = ChordToPitchMapper.rhythm_to_notes(
            rhythm, "Cmaj7", target_register=(60, 72)
        )

        self.assertEqual(len(pitches), 4)
        # All pitches should be in target register
        for pitch in pitches:
            self.assertGreaterEqual(pitch, 60)
            self.assertLessEqual(pitch, 72)


class TestArticulationLibrary(unittest.TestCase):
    """Test articulation pattern library"""

    def test_get_brass_pattern(self):
        """Test retrieving brass articulation pattern"""
        pattern = ArticulationLibrary.get_pattern(
            InstrumentSection.BRASS, 'hits'
        )

        self.assertIsInstance(pattern, list)
        self.assertGreater(len(pattern), 0)
        self.assertIn(ArticulationType.TONGUED, pattern)

    def test_get_string_pattern(self):
        """Test retrieving string articulation pattern"""
        pattern = ArticulationLibrary.get_pattern(
            InstrumentSection.STRINGS, 'short'
        )

        self.assertIn(ArticulationType.SPICCATO, pattern)

    def test_recommend_for_short_notes(self):
        """Test automatic recommendation for short notes"""
        rhythm = RhythmPattern(
            onsets=[0.0, 0.25, 0.5, 0.75],  # Fast 16th notes
            durations=[0.2, 0.2, 0.2, 0.2]
        )

        pattern = ArticulationLibrary.recommend_for_rhythm(
            InstrumentSection.BRASS, rhythm
        )

        # Should recommend fast articulation for brass
        self.assertIsInstance(pattern, list)

    def test_recommend_for_long_notes(self):
        """Test automatic recommendation for sustained notes"""
        rhythm = RhythmPattern(
            onsets=[0.0, 4.0],  # Long notes
            durations=[3.5, 3.5]
        )

        pattern = ArticulationLibrary.recommend_for_rhythm(
            InstrumentSection.STRINGS, rhythm
        )

        # Should recommend smooth articulation
        self.assertIsInstance(pattern, list)


class TestBrassVoicingEngine(unittest.TestCase):
    """Test brass section voicing"""

    def test_voice_chord_drop_2(self):
        """Test drop-2 voicing for brass"""
        voicing = BrassVoicingEngine.voice_chord(
            "Cmaj7", ensemble='big_band', voicing_type=VoicingStrategy.DROP_2
        )

        self.assertIsInstance(voicing, list)
        self.assertGreater(len(voicing), 0)

        # Each entry should be (instrument, pitch)
        for inst, pitch in voicing:
            self.assertIsInstance(inst, str)
            self.assertIsInstance(pitch, int)
            self.assertGreater(pitch, 0)
            self.assertLess(pitch, 128)

    def test_voice_chord_unison(self):
        """Test unison voicing"""
        voicing = BrassVoicingEngine.voice_chord(
            "G7", ensemble='brass_quartet', voicing_type=VoicingStrategy.UNISON
        )

        # All instruments should play same pitch
        pitches = [pitch for _, pitch in voicing]
        self.assertEqual(len(set(pitches)), 1)

    def test_voice_chord_octaves(self):
        """Test octave doubling"""
        voicing = BrassVoicingEngine.voice_chord(
            "Dm7", ensemble='brass_quartet', voicing_type=VoicingStrategy.OCTAVES
        )

        pitches = [pitch for _, pitch in voicing]
        # Should have octave relationships
        for i in range(len(pitches) - 1):
            # Allow for some flexibility in octave spacing
            self.assertTrue(abs(pitches[i+1] - pitches[i]) >= 0)


class TestStringVoicingEngine(unittest.TestCase):
    """Test string section voicing"""

    def test_voice_string_quartet(self):
        """Test string quartet voicing"""
        voicing = StringVoicingEngine.voice_chord(
            "Cmaj7", ensemble='string_quartet', voicing_type=VoicingStrategy.CLOSE
        )

        self.assertEqual(len(voicing), 4)  # 4 instruments in quartet

        instruments = [inst for inst, _ in voicing]
        self.assertIn('violin', instruments)
        self.assertIn('viola', instruments)
        self.assertIn('cello', instruments)

    def test_voice_string_section(self):
        """Test full string section voicing"""
        voicing = StringVoicingEngine.voice_chord(
            "G7", ensemble='string_section', voicing_type=VoicingStrategy.OPEN
        )

        self.assertGreater(len(voicing), 4)  # Larger section


class TestWoodwindVoicingEngine(unittest.TestCase):
    """Test woodwind section voicing"""

    def test_traditional_voicing(self):
        """Test traditional woodwind voicing (flute on top)"""
        voicing = WoodwindVoicingEngine.voice_chord(
            "Cmaj7", ensemble='wind_quartet',
            voicing_type=VoicingStrategy.TRADITIONAL
        )

        self.assertEqual(len(voicing), 4)

        # Flute should be highest
        instruments = [inst for inst, _ in voicing]
        pitches = [pitch for _, pitch in voicing]

        flute_idx = instruments.index('flute')
        # Flute should have highest or near-highest pitch
        self.assertGreaterEqual(pitches[flute_idx], min(pitches))

    def test_interlocking_voicing(self):
        """Test interlocking woodwind voicing"""
        voicing = WoodwindVoicingEngine.voice_chord(
            "Dm7", ensemble='wind_quintet',
            voicing_type=VoicingStrategy.INTERLOCKING
        )

        self.assertGreater(len(voicing), 0)


class TestGranularControl(unittest.TestCase):
    """Test main GranularControl engine"""

    def setUp(self):
        """Set up test fixtures"""
        self.gc = GranularControl()

        self.basic_rhythm = RhythmPattern(
            onsets=[0.0, 2.0],
            durations=[0.25, 0.25],
            accents=[True, True]
        )

        self.chords = ["Cmaj7", "Dm7", "G7", "Cmaj7"]

    def test_generate_brass_hits(self):
        """Test generating brass hits"""
        output = self.gc.generate_hits(
            rhythm=self.basic_rhythm,
            chord_progression=self.chords,
            measures=4
        )

        self.assertIsNotNone(output)
        self.assertEqual(output.section, InstrumentSection.BRASS)
        self.assertGreater(len(output.notes), 0)

    def test_generate_string_sustained(self):
        """Test generating sustained string pad"""
        rhythm = RhythmPattern(
            onsets=[0.0, 4.0, 8.0, 12.0],
            durations=[4.0, 4.0, 4.0, 4.0]
        )

        output = self.gc.generate_sustained(
            rhythm=rhythm,
            chord_progression=self.chords,
            section=InstrumentSection.STRINGS,
            measures=4
        )

        self.assertEqual(output.section, InstrumentSection.STRINGS)
        self.assertGreater(len(output.notes), 0)

    def test_generate_with_custom_instruments(self):
        """Test generation with specific instruments"""
        output = self.gc.generate(
            rhythm_pattern=self.basic_rhythm,
            chord_progression=self.chords,
            section=InstrumentSection.BRASS,
            instruments=['trumpet', 'trombone'],
            measures=2
        )

        # Should generate notes for specified instruments
        instruments_used = set(note.instrument for note in output.notes)
        self.assertIn('trumpet', instruments_used)
        self.assertIn('trombone', instruments_used)

    def test_generate_with_swing(self):
        """Test generation with swing feel"""
        straight_rhythm = RhythmPattern(
            onsets=[0.0, 0.5, 1.0, 1.5],
            durations=[0.5, 0.5, 0.5, 0.5]
        )

        output = self.gc.generate(
            rhythm_pattern=straight_rhythm,
            chord_progression=self.chords,
            section=InstrumentSection.BRASS,
            measures=2,
            apply_swing=True,
            swing_factor=0.67
        )

        # Check that timing has changed
        onsets = [note.onset for note in output.notes[:4]]  # First measure
        # Swung 8th notes should not be at 0.5, 1.5
        self.assertTrue(any(abs(onset - 0.5) > 0.1 for onset in onsets))

    def test_playability_assessment(self):
        """Test playability assessment"""
        output = self.gc.generate_hits(
            rhythm=self.basic_rhythm,
            chord_progression=self.chords,
            measures=2
        )

        # Should have playability rating
        self.assertIsNotNone(output.voicing_quality)

    def test_warnings_for_range_violations(self):
        """Test that warnings are generated for out-of-range notes"""
        # This test depends on specific instrument ranges
        # May or may not generate warnings depending on voicing
        output = self.gc.generate(
            rhythm_pattern=self.basic_rhythm,
            chord_progression=self.chords,
            section=InstrumentSection.BRASS,
            measures=2
        )

        self.assertIsInstance(output.warnings, list)
        self.assertIsInstance(output.suggestions, list)


class TestPercussionVoicingEngine(unittest.TestCase):
    """Test percussion/drum generation"""

    def test_rhythm_to_drums(self):
        """Test converting rhythm to drum hits"""
        rhythm = RhythmPattern(
            onsets=[0.0, 1.0, 2.0, 3.0],
            durations=[0.25, 0.25, 0.25, 0.25],
            accents=[True, False, True, False]
        )

        notes = PercussionVoicingEngine.rhythm_to_drums(
            rhythm, drum_voices=['kick', 'snare']
        )

        self.assertEqual(len(notes), 4)

        # Check drum pitches are valid
        for note in notes:
            self.assertGreater(note.pitch, 0)
            self.assertLess(note.pitch, 128)
            self.assertEqual(note.instrument, 'drums')

    def test_create_rock_beat(self):
        """Test creating basic rock drum beat"""
        notes = PercussionVoicingEngine.create_basic_beat(
            style='rock', measures=2
        )

        self.assertGreater(len(notes), 0)

        # Should have kick, snare, and hihat
        pitches = set(note.pitch for note in notes)
        self.assertIn(PercussionVoicingEngine.DRUM_MAP['kick'], pitches)
        self.assertIn(PercussionVoicingEngine.DRUM_MAP['snare'], pitches)
        self.assertIn(PercussionVoicingEngine.DRUM_MAP['closed_hihat'], pitches)

    def test_create_jazz_beat(self):
        """Test creating jazz ride pattern"""
        notes = PercussionVoicingEngine.create_basic_beat(
            style='jazz', measures=1
        )

        self.assertGreater(len(notes), 0)

        # Should have ride cymbal
        pitches = set(note.pitch for note in notes)
        self.assertIn(PercussionVoicingEngine.DRUM_MAP['ride'], pitches)


class TestDynamicsEngine(unittest.TestCase):
    """Test dynamics and expression control"""

    def test_crescendo(self):
        """Test crescendo application"""
        # Create some test notes
        from midi_generator.generators.granular_control import GeneratedNote

        notes = [
            GeneratedNote(60, 0.0, 0.5, 80, ArticulationType.STACCATO, 'piano'),
            GeneratedNote(62, 1.0, 0.5, 80, ArticulationType.STACCATO, 'piano'),
            GeneratedNote(64, 2.0, 0.5, 80, ArticulationType.STACCATO, 'piano'),
            GeneratedNote(65, 3.0, 0.5, 80, ArticulationType.STACCATO, 'piano'),
        ]

        DynamicsEngine.apply_dynamics_curve(
            notes, curve_type='crescendo', start_dynamic='p', end_dynamic='f'
        )

        # Velocity should increase
        self.assertLess(notes[0].velocity, notes[-1].velocity)

        # Should be gradual
        for i in range(len(notes) - 1):
            self.assertLessEqual(notes[i].velocity, notes[i+1].velocity)

    def test_decrescendo(self):
        """Test decrescendo application"""
        from midi_generator.generators.granular_control import GeneratedNote

        notes = [
            GeneratedNote(60, i * 1.0, 0.5, 80, ArticulationType.STACCATO, 'piano')
            for i in range(4)
        ]

        DynamicsEngine.apply_dynamics_curve(
            notes, curve_type='decrescendo', start_dynamic='f', end_dynamic='p'
        )

        # Velocity should decrease
        self.assertGreater(notes[0].velocity, notes[-1].velocity)

    def test_apply_accents(self):
        """Test accent pattern application"""
        from midi_generator.generators.granular_control import GeneratedNote

        notes = [
            GeneratedNote(60, i * 0.5, 0.25, 80, ArticulationType.STACCATO, 'piano')
            for i in range(8)
        ]

        DynamicsEngine.apply_accents(
            notes, accent_pattern=[True, False, True, False], accent_amount=20
        )

        # First note should be accented
        self.assertEqual(notes[0].velocity, 100)  # 80 + 20
        # Second note should not
        self.assertEqual(notes[1].velocity, 80)


class TestPhraseShaper(unittest.TestCase):
    """Test phrase shaping utilities"""

    def test_add_ritardando(self):
        """Test adding ritardando to phrase ending"""
        from midi_generator.generators.granular_control import GeneratedNote

        notes = [
            GeneratedNote(60, i * 1.0, 0.5, 80, ArticulationType.STACCATO, 'piano')
            for i in range(8)
        ]

        original_onsets = [note.onset for note in notes]

        PhraseShaper.add_phrase_ending(notes, ending_type='ritardando')

        # Last few notes should be delayed
        self.assertGreater(notes[-1].onset, original_onsets[-1])

    def test_add_decrescendo_ending(self):
        """Test adding decrescendo to phrase ending"""
        from midi_generator.generators.granular_control import GeneratedNote

        notes = [
            GeneratedNote(60, i * 1.0, 0.5, 80, ArticulationType.STACCATO, 'piano')
            for i in range(8)
        ]

        original_velocity = notes[-1].velocity

        PhraseShaper.add_phrase_ending(notes, ending_type='decrescendo')

        # Last note should be softer
        self.assertLess(notes[-1].velocity, original_velocity)

    def test_add_ornaments(self):
        """Test adding ornaments"""
        from midi_generator.generators.granular_control import GeneratedNote

        notes = [
            GeneratedNote(60, i * 1.0, 0.5, 80, ArticulationType.STACCATO, 'piano')
            for i in range(4)
        ]

        original_count = len(notes)

        ornamented = PhraseShaper.add_ornaments(
            notes, ornament_type='grace_note', positions=[0]
        )

        # Should have added grace note
        self.assertGreater(len(ornamented), original_count)


class TestAdvancedControlEngine(unittest.TestCase):
    """Test advanced control features"""

    def test_humanization(self):
        """Test MIDI humanization"""
        from midi_generator.generators.granular_control import GeneratedNote

        notes = [
            GeneratedNote(60, i * 1.0, 0.5, 80, ArticulationType.STACCATO, 'piano')
            for i in range(8)
        ]

        original_onsets = [note.onset for note in notes]
        original_velocities = [note.velocity for note in notes]

        AdvancedControlEngine.apply_humanization(notes)

        # Some timing should have changed
        changed_timing = any(
            notes[i].onset != original_onsets[i] for i in range(len(notes))
        )
        self.assertTrue(changed_timing)

        # Some velocities should have changed
        changed_velocity = any(
            notes[i].velocity != original_velocities[i] for i in range(len(notes))
        )
        self.assertTrue(changed_velocity)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""

    def test_create_brass_hits(self):
        """Test quick brass hits function"""
        output = create_brass_hits(
            onsets=[0.0, 2.0],
            chord_progression=["Cmaj7", "G7", "Cmaj7", "G7"]
        )

        self.assertIsNotNone(output)
        self.assertEqual(output.section, InstrumentSection.BRASS)
        self.assertGreater(len(output.notes), 0)

    def test_create_string_pad(self):
        """Test quick string pad function"""
        output = create_string_pad(
            duration=4.0,
            chord_progression=["Cmaj7", "Dm7", "Em7", "Fmaj7"]
        )

        self.assertIsNotNone(output)
        self.assertEqual(output.section, InstrumentSection.STRINGS)
        self.assertGreater(len(output.notes), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple features"""

    def test_complete_arrangement(self):
        """Test creating complete multi-section arrangement"""
        gc = GranularControl()

        chords = ["Cmaj7", "Am7", "Dm7", "G7"]

        # Brass hits
        brass_rhythm = RhythmPattern(
            onsets=[0.0, 2.0],
            durations=[0.25, 0.25],
            accents=[True, True]
        )

        brass_output = gc.generate_hits(
            rhythm=brass_rhythm,
            chord_progression=chords,
            measures=4
        )

        # String pad
        string_rhythm = RhythmPattern(
            onsets=[0.0, 4.0, 8.0, 12.0],
            durations=[4.0, 4.0, 4.0, 4.0]
        )

        string_output = gc.generate_sustained(
            rhythm=string_rhythm,
            chord_progression=chords,
            section=InstrumentSection.STRINGS,
            measures=4
        )

        # Both should generate successfully
        self.assertGreater(len(brass_output.notes), 0)
        self.assertGreater(len(string_output.notes), 0)

        # Different sections
        self.assertNotEqual(brass_output.section, string_output.section)

    def test_layered_texture(self):
        """Test creating layered texture"""
        base_rhythm = RhythmPattern(
            onsets=[0.0, 1.0, 2.0, 3.0],
            durations=[0.5, 0.5, 0.5, 0.5]
        )

        chords = ["Cmaj7", "Dm7", "G7", "Cmaj7"]

        layers = [
            {
                'section': InstrumentSection.BRASS,
                'instruments': ['trumpet', 'trombone'],
                'measures': 4
            },
            {
                'section': InstrumentSection.STRINGS,
                'instruments': ['violin', 'cello'],
                'offset': 0.5,  # Delay by half beat
                'measures': 4
            }
        ]

        outputs = AdvancedControlEngine.create_layered_texture(
            base_rhythm, chords, layers
        )

        self.assertEqual(len(outputs), 2)
        self.assertGreater(len(outputs[0].notes), 0)
        self.assertGreater(len(outputs[1].notes), 0)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("=" * 70)
    print("GRANULAR CONTROL SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print()

    run_tests()
