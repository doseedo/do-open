#!/usr/bin/env python3
"""
Comprehensive Test Suite for Context-Aware Generator

Tests for:
- MIDI analysis
- Track generation (bass, harmony, melody, percussion)
- Inpainting with new chords
- Genre changes in sections
- Voice leading validation
- Boundary smoothing
- Smart orchestration suggestions
- Context extraction

Author: Agent 3 - Context-Aware Generation
Date: 2025
"""

import unittest
import sys
import os
from pathlib import Path
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from midi_generator.generators.context_aware_generator import (
    ContextAwareGenerator,
    TrackInpainter,
    SmartOrchestrator,
    ArrangementAnalysis,
    BoundaryContext,
    GenerationConstraints
)


class TestMIDICreation(unittest.TestCase):
    """Helper class to create test MIDI files"""

    @staticmethod
    def create_simple_midi(filename: str, tempo: int = 120, measures: int = 4):
        """
        Create simple test MIDI file with piano chords

        Args:
            filename: Output filename
            tempo: Tempo in BPM
            measures: Number of measures
        """
        mid = MidiFile(ticks_per_beat=480)
        track = MidiTrack()
        mid.tracks.append(track)

        # Set tempo
        track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))

        # Set time signature (4/4)
        track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))

        # Add program change (piano)
        track.append(Message('program_change', program=0, time=0))

        # Add simple chord progression: C - F - G - C
        chords = [
            [60, 64, 67],  # C major (C, E, G)
            [65, 69, 72],  # F major (F, A, C)
            [67, 71, 74],  # G major (G, B, D)
            [60, 64, 67],  # C major
        ]

        ticks_per_measure = 480 * 4  # 4 beats per measure

        for measure in range(min(measures, len(chords))):
            chord = chords[measure]

            # Note on for all chord tones
            for i, pitch in enumerate(chord):
                track.append(Message('note_on', note=pitch, velocity=80,
                                   time=ticks_per_measure if i == 0 else 0))

            # Note off for all chord tones
            for i, pitch in enumerate(chord):
                track.append(Message('note_off', note=pitch, velocity=0,
                                   time=ticks_per_measure if i == 0 else 0))

        mid.save(filename)
        return filename

    @staticmethod
    def create_piano_drums_midi(filename: str):
        """Create MIDI with piano and drums"""
        mid = MidiFile(ticks_per_beat=480)

        # Piano track
        piano_track = MidiTrack()
        mid.tracks.append(piano_track)
        piano_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
        piano_track.append(Message('program_change', program=0, time=0))

        # Add simple melody
        notes = [60, 62, 64, 65, 67]
        ticks_per_note = 480

        for note in notes:
            piano_track.append(Message('note_on', note=note, velocity=80, time=0))
            piano_track.append(Message('note_off', note=note, velocity=0, time=ticks_per_note))

        # Drum track
        drum_track = MidiTrack()
        mid.tracks.append(drum_track)
        drum_track.append(Message('program_change', program=0, channel=9, time=0))  # Channel 10 for drums

        # Add simple beat (kick)
        for i in range(4):
            drum_track.append(Message('note_on', note=36, velocity=90, channel=9, time=0))
            drum_track.append(Message('note_off', note=36, velocity=0, channel=9, time=480 * 4))

        mid.save(filename)
        return filename


class TestContextAwareGenerator(unittest.TestCase):
    """Test ContextAwareGenerator class"""

    def setUp(self):
        """Create temporary test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_midi = os.path.join(self.temp_dir, 'test.mid')
        TestMIDICreation.create_simple_midi(self.test_midi, tempo=120, measures=4)

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test generator initialization"""
        gen = ContextAwareGenerator(self.test_midi)
        self.assertIsNotNone(gen)
        self.assertEqual(gen.midi_file, self.test_midi)
        self.assertIsNone(gen.analysis)

    def test_analyze(self):
        """Test MIDI analysis"""
        gen = ContextAwareGenerator(self.test_midi)
        analysis = gen.analyze()

        self.assertIsNotNone(analysis)
        self.assertIsInstance(analysis, ArrangementAnalysis)
        self.assertEqual(analysis.tempo, 120)
        self.assertEqual(analysis.time_signature, (4, 4))
        self.assertGreater(analysis.length_measures, 0)

    def test_chord_extraction(self):
        """Test chord extraction from MIDI"""
        gen = ContextAwareGenerator(self.test_midi)
        analysis = gen.analyze()

        self.assertIsNotNone(analysis.chord_progression)
        self.assertGreater(len(analysis.chord_progression), 0)
        print(f"Detected chords: {analysis.chord_progression}")

    def test_density_calculation(self):
        """Test rhythmic density calculation"""
        gen = ContextAwareGenerator(self.test_midi)
        analysis = gen.analyze()

        self.assertIsNotNone(analysis.density_per_measure)
        self.assertGreater(len(analysis.density_per_measure), 0)

        # Should have positive density
        total_density = sum(analysis.density_per_measure)
        self.assertGreater(total_density, 0)

    def test_register_distribution(self):
        """Test register distribution analysis"""
        gen = ContextAwareGenerator(self.test_midi)
        analysis = gen.analyze()

        reg_dist = analysis.register_distribution
        self.assertIn('low', reg_dist)
        self.assertIn('mid', reg_dist)
        self.assertIn('high', reg_dist)

        # Sum should be approximately 1.0
        total = reg_dist['low'] + reg_dist['mid'] + reg_dist['high']
        self.assertAlmostEqual(total, 1.0, places=1)

    def test_texture_analysis(self):
        """Test texture analysis"""
        gen = ContextAwareGenerator(self.test_midi)
        analysis = gen.analyze()

        self.assertIn(analysis.texture, ['monophonic', 'homophonic', 'polyphonic'])

    def test_track_role_classification(self):
        """Test track role classification"""
        gen = ContextAwareGenerator(self.test_midi)
        analysis = gen.analyze()

        self.assertIsNotNone(analysis.track_roles)
        self.assertGreater(len(analysis.track_roles), 0)

        # Roles should be valid
        valid_roles = {'melody', 'harmony', 'bass', 'drums', 'unknown'}
        for role in analysis.track_roles.values():
            self.assertIn(role, valid_roles)


class TestTrackGeneration(unittest.TestCase):
    """Test track generation capabilities"""

    def setUp(self):
        """Create test MIDI"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_midi = os.path.join(self.temp_dir, 'test.mid')
        TestMIDICreation.create_simple_midi(self.test_midi, tempo=120, measures=8)

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_add_bass_track(self):
        """Test adding bass track"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        bass_notes = gen.add_track(
            instrument=33,  # Fingered bass
            track_type='bass'
        )

        self.assertIsNotNone(bass_notes)
        self.assertGreater(len(bass_notes), 0)

        # Bass should be in low register
        for pitch, _, _, _ in bass_notes:
            self.assertLess(pitch, 60)  # Below middle C

    def test_add_harmony_track(self):
        """Test adding harmony track"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        harmony_notes = gen.add_track(
            instrument=48,  # String ensemble
            track_type='harmony'
        )

        self.assertIsNotNone(harmony_notes)
        self.assertGreater(len(harmony_notes), 0)

    def test_add_melody_track(self):
        """Test adding melody track"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        melody_notes = gen.add_track(
            instrument=73,  # Flute
            track_type='melody'
        )

        self.assertIsNotNone(melody_notes)
        self.assertGreater(len(melody_notes), 0)

    def test_add_percussion_track(self):
        """Test adding percussion track"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        drum_notes = gen.add_track(
            instrument=128,  # Drums
            track_type='percussion'
        )

        self.assertIsNotNone(drum_notes)
        self.assertGreater(len(drum_notes), 0)

    def test_auto_track_type_inference(self):
        """Test automatic track type inference"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        # Bass instrument should infer bass type
        bass_notes = gen.add_track(instrument=33, track_type='auto')
        self.assertIsNotNone(bass_notes)

        # String should infer harmony or melody
        string_notes = gen.add_track(instrument=48, track_type='auto')
        self.assertIsNotNone(string_notes)

    def test_genre_specification(self):
        """Test generating with specific genre"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        # Generate funk bass
        funk_bass = gen.add_track(
            instrument=33,
            genre='funk',
            track_type='bass'
        )

        self.assertIsNotNone(funk_bass)
        self.assertGreater(len(funk_bass), 0)

    def test_add_section(self):
        """Test adding track to specific section"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        # Add bass to measures 2-4 only
        section_notes = gen.add_section(
            start_measure=2,
            end_measure=4,
            instrument=33,
            track_type='bass'
        )

        self.assertIsNotNone(section_notes)
        # Should have fewer notes than full track
        full_track = gen.add_track(instrument=33, track_type='bass')
        self.assertLess(len(section_notes), len(full_track))


class TestInpainting(unittest.TestCase):
    """Test inpainting capabilities"""

    def setUp(self):
        """Create test MIDI"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_midi = os.path.join(self.temp_dir, 'test.mid')
        TestMIDICreation.create_simple_midi(self.test_midi, tempo=120, measures=8)

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_regenerate_section(self):
        """Test regenerating section of existing track"""
        gen = ContextAwareGenerator(self.test_midi)
        analysis = gen.analyze()

        # Regenerate measures 2-4
        new_section = gen.regenerate_section(
            track_number=0,
            start_measure=2,
            end_measure=4
        )

        self.assertIsNotNone(new_section)
        self.assertGreater(len(new_section), 0)

    def test_regenerate_with_new_chords(self):
        """Test regenerating with new chord progression"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        new_chords = ['Dm7', 'G7', 'Cmaj7', 'A7']

        new_section = gen.regenerate_section(
            track_number=0,
            start_measure=0,
            end_measure=4,
            new_chords=new_chords
        )

        self.assertIsNotNone(new_section)
        self.assertGreater(len(new_section), 0)

    def test_regenerate_with_genre_change(self):
        """Test regenerating section in different genre"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        # Change to funk style
        new_section = gen.regenerate_section(
            track_number=0,
            start_measure=2,
            end_measure=4,
            new_genre='funk'
        )

        self.assertIsNotNone(new_section)
        self.assertGreater(len(new_section), 0)

    def test_inpainter_class(self):
        """Test TrackInpainter class"""
        inpainter = TrackInpainter(self.test_midi)

        self.assertIsNotNone(inpainter.generator)
        self.assertIsNotNone(inpainter.generator.analysis)

    def test_inpainter_measures(self):
        """Test inpainting specific measures"""
        inpainter = TrackInpainter(self.test_midi)

        new_section = inpainter.inpaint_measures(
            track=0,
            start=1,
            end=3,
            new_chords=['Am7', 'D7']
        )

        self.assertIsNotNone(new_section)

    def test_inpainter_genre_change(self):
        """Test inpainting with genre change"""
        inpainter = TrackInpainter(self.test_midi)

        new_section = inpainter.inpaint_with_genre_change(
            track=0,
            start=2,
            end=4,
            new_genre='edm'
        )

        self.assertIsNotNone(new_section)


class TestBoundaryContext(unittest.TestCase):
    """Test boundary context extraction"""

    def setUp(self):
        """Create test MIDI"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_midi = os.path.join(self.temp_dir, 'test.mid')
        TestMIDICreation.create_simple_midi(self.test_midi, tempo=120, measures=4)

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_extract_entry_context(self):
        """Test extracting entry context"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        if gen.analysis.tracks:
            track = gen.analysis.tracks[0]
            context = gen._extract_entry_context(track, 0)

            self.assertIsInstance(context, BoundaryContext)
            self.assertEqual(context.measure, 0)
            self.assertIn(context.voice_leading_tendency, ['ascending', 'descending', 'static'])
            self.assertIn(context.register, ['low', 'mid', 'high'])

    def test_extract_exit_context(self):
        """Test extracting exit context"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        if gen.analysis.tracks:
            track = gen.analysis.tracks[0]
            context = gen._extract_exit_context(track, 1)

            self.assertIsInstance(context, BoundaryContext)


class TestSmartOrchestrator(unittest.TestCase):
    """Test SmartOrchestrator class"""

    def setUp(self):
        """Create test MIDI with piano and drums"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_midi = os.path.join(self.temp_dir, 'piano_drums.mid')
        TestMIDICreation.create_piano_drums_midi(self.test_midi)

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = SmartOrchestrator(self.test_midi)

        self.assertIsNotNone(orchestrator)
        self.assertIsNotNone(orchestrator.analysis)
        self.assertEqual(len(orchestrator.added_tracks), 0)

    def test_suggest_additions(self):
        """Test suggesting track additions"""
        orchestrator = SmartOrchestrator(self.test_midi)
        suggestions = orchestrator.suggest_additions()

        self.assertIsNotNone(suggestions)
        self.assertIsInstance(suggestions, list)

        # Should suggest at least bass
        if suggestions:
            self.assertIn('instrument', suggestions[0])
            self.assertIn('reason', suggestions[0])
            self.assertIn('priority', suggestions[0])

    def test_add_suggested_track(self):
        """Test adding suggested track"""
        orchestrator = SmartOrchestrator(self.test_midi)
        suggestions = orchestrator.suggest_additions()

        if suggestions:
            notes = orchestrator.add_suggested_track(suggestions[0])

            self.assertIsNotNone(notes)
            self.assertGreater(len(notes), 0)
            self.assertEqual(len(orchestrator.added_tracks), 1)

    def test_analyze_orchestral_balance(self):
        """Test orchestral balance analysis"""
        orchestrator = SmartOrchestrator(self.test_midi)
        balance = orchestrator.analyze_orchestral_balance()

        self.assertIn('register_distribution', balance)
        self.assertIn('texture', balance)
        self.assertIn('has_bass', balance)
        self.assertIn('has_drums', balance)
        self.assertIsInstance(balance['has_bass'], bool)


class TestExport(unittest.TestCase):
    """Test MIDI export functionality"""

    def setUp(self):
        """Create test MIDI"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_midi = os.path.join(self.temp_dir, 'test.mid')
        TestMIDICreation.create_simple_midi(self.test_midi, tempo=120, measures=4)

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_export_with_new_track(self):
        """Test exporting MIDI with new track"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        bass_notes = gen.add_track(instrument=33, track_type='bass')

        output_file = os.path.join(self.temp_dir, 'output.mid')
        gen.export_with_new_track(bass_notes, output_file, instrument=33)

        # Verify file was created
        self.assertTrue(os.path.exists(output_file))

        # Verify it's a valid MIDI file
        midi = MidiFile(output_file)
        self.assertGreater(len(midi.tracks), 0)


class TestConstraints(unittest.TestCase):
    """Test generation constraints"""

    def test_constraints_creation(self):
        """Test creating generation constraints"""
        constraints = GenerationConstraints(
            follow_harmony=True,
            match_density=True,
            avoid_voice_leading_errors=True,
            max_voice_leading_distance=5
        )

        self.assertTrue(constraints.follow_harmony)
        self.assertTrue(constraints.match_density)
        self.assertEqual(constraints.max_voice_leading_distance, 5)

    def test_default_constraints(self):
        """Test default constraint values"""
        constraints = GenerationConstraints()

        self.assertTrue(constraints.follow_harmony)
        self.assertTrue(constraints.match_density)
        self.assertTrue(constraints.avoid_voice_leading_errors)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        """Create test MIDI"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_midi = os.path.join(self.temp_dir, 'test.mid')
        TestMIDICreation.create_simple_midi(self.test_midi, tempo=120, measures=2)

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_empty_genre(self):
        """Test with None genre (should use detected)"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        notes = gen.add_track(instrument=33, genre=None, track_type='bass')
        self.assertIsNotNone(notes)

    def test_invalid_track_number(self):
        """Test regenerating non-existent track"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        with self.assertRaises(ValueError):
            gen.regenerate_section(
                track_number=999,  # Invalid track
                start_measure=0,
                end_measure=2
            )

    def test_invalid_track_type(self):
        """Test invalid track type"""
        gen = ContextAwareGenerator(self.test_midi)
        gen.analyze()

        with self.assertRaises(ValueError):
            gen.add_track(instrument=0, track_type='invalid_type')


# ==============================================================================
# TEST RUNNER
# ==============================================================================

def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestContextAwareGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestTrackGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestInpainting))
    suite.addTests(loader.loadTestsFromTestCase(TestBoundaryContext))
    suite.addTests(loader.loadTestsFromTestCase(TestSmartOrchestrator))
    suite.addTests(loader.loadTestsFromTestCase(TestExport))
    suite.addTests(loader.loadTestsFromTestCase(TestConstraints))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
