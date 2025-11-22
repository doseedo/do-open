"""
Test Suite for Features→MIDI Conversion
========================================

Comprehensive tests for the Features→MIDI decoder system.

Tests:
- Base interface validation
- Rule-based converter functionality
- End-to-end reconstruction
- Parameter extraction accuracy
- MIDI quality metrics

Author: Agent 1 - MIDI Decoder Architecture Lead
Date: November 22, 2025
"""

import unittest
import numpy as np
import mido
from pathlib import Path
import tempfile
import shutil

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from midi_generator.models.features_to_midi import FeaturesToMIDI, validate_features
from midi_generator.models.rule_based_midi import RuleBasedFeaturesToMIDI


class TestValidateFeatures(unittest.TestCase):
    """Test feature validation utilities."""

    def test_valid_features_1d(self):
        """Test validation of 1D features."""
        features = np.random.randn(1150)
        self.assertTrue(validate_features(features))

    def test_valid_features_2d(self):
        """Test validation of 2D batch features."""
        features = np.random.randn(4, 1150)
        self.assertTrue(validate_features(features))

    def test_invalid_shape(self):
        """Test rejection of invalid shape."""
        features = np.random.randn(1000)  # Wrong size
        self.assertFalse(validate_features(features))

    def test_invalid_nan(self):
        """Test rejection of NaN features."""
        features = np.random.randn(1150)
        features[0] = np.nan
        self.assertFalse(validate_features(features))

    def test_invalid_inf(self):
        """Test rejection of infinite features."""
        features = np.random.randn(1150)
        features[0] = np.inf
        self.assertFalse(validate_features(features))


class TestRuleBasedConverter(unittest.TestCase):
    """Test rule-based Features→MIDI converter."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = RuleBasedFeaturesToMIDI(verbose=False)
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.temp_dir)

    def test_features_to_parameters(self):
        """Test parameter extraction from features."""
        features = np.random.randn(1150)
        params = self.converter.features_to_parameters(features)

        # Check required parameters exist
        required_params = [
            'key', 'mode', 'tempo_bpm', 'time_signature',
            'melodic_range', 'velocity_mean', 'voice_count',
            'num_bars'
        ]

        for param in required_params:
            self.assertIn(param, params)

        # Check parameter ranges
        self.assertTrue(60 <= params['tempo_bpm'] <= 200)
        self.assertTrue(1 <= params['velocity_mean'] <= 127)
        self.assertTrue(1 <= params['voice_count'] <= 8)
        self.assertTrue(4 <= params['num_bars'] <= 32)

    def test_features_to_midi_basic(self):
        """Test basic MIDI generation."""
        features = np.random.randn(1150)
        midi = self.converter.features_to_midi(features)

        # Check MIDI is valid
        self.assertIsInstance(midi, mido.MidiFile)
        self.assertTrue(self.converter.validate_output(midi))
        self.assertGreater(len(midi.tracks), 0)

    def test_features_to_midi_with_save(self):
        """Test MIDI generation with file save."""
        features = np.random.randn(1150)
        output_path = self.temp_dir / "test_output.mid"

        midi = self.converter.features_to_midi(features, output_path=output_path)

        # Check file was created
        self.assertTrue(output_path.exists())

        # Check can be loaded
        loaded_midi = mido.MidiFile(str(output_path))
        self.assertTrue(self.converter.validate_output(loaded_midi))

    def test_batch_features(self):
        """Test handling batch of features."""
        batch_features = np.random.randn(4, 1150)

        # Should handle first in batch
        midi = self.converter.features_to_midi(batch_features)
        self.assertTrue(self.converter.validate_output(midi))

    def test_extract_key(self):
        """Test key extraction."""
        features = np.random.randn(1150)
        params = self.converter.features_to_parameters(features)

        # Key should be one of the 12 notes
        self.assertIn(params['key'], self.converter.note_names)

    def test_extract_mode(self):
        """Test mode extraction."""
        features = np.random.randn(1150)
        params = self.converter.features_to_parameters(features)

        # Mode should be major or minor (for simple implementation)
        self.assertIn(params['mode'], ['major', 'minor'])

    def test_extract_tempo(self):
        """Test tempo extraction."""
        features = np.random.randn(1150)
        params = self.converter.features_to_parameters(features)

        # Tempo should be in reasonable range
        self.assertTrue(60 <= params['tempo_bpm'] <= 200)

    def test_extract_time_signature(self):
        """Test time signature extraction."""
        features = np.random.randn(1150)
        params = self.converter.features_to_parameters(features)

        # Should be one of common time signatures
        self.assertIn(params['time_signature'], ['4/4', '3/4', '6/8'])

    def test_midi_has_melody_track(self):
        """Test that generated MIDI has melody track."""
        features = np.random.randn(1150)
        midi = self.converter.features_to_midi(features)

        # Count note events
        note_count = 0
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on':
                    note_count += 1

        self.assertGreater(note_count, 0)

    def test_midi_has_tempo(self):
        """Test that generated MIDI has tempo meta message."""
        features = np.random.randn(1150)
        midi = self.converter.features_to_midi(features)

        # Check for tempo message
        has_tempo = False
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    has_tempo = True
                    break

        self.assertTrue(has_tempo)

    def test_different_features_produce_different_midi(self):
        """Test that different features produce different MIDI."""
        features1 = np.random.randn(1150)
        features2 = np.random.randn(1150)

        params1 = self.converter.features_to_parameters(features1)
        params2 = self.converter.features_to_parameters(features2)

        # Parameters should be different
        # (very unlikely to be identical for random features)
        different = False
        for key in params1:
            if params1[key] != params2[key]:
                different = True
                break

        self.assertTrue(different)


class TestFeatureSlicing(unittest.TestCase):
    """Test feature slicing utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = RuleBasedFeaturesToMIDI()
        self.features = np.random.randn(1150)

    def test_extract_harmony_slice(self):
        """Test harmony feature extraction."""
        harmony = self.converter.extract_feature_slice(self.features, 'harmony')
        self.assertEqual(harmony.shape, (250,))

    def test_extract_rhythm_slice(self):
        """Test rhythm feature extraction."""
        rhythm = self.converter.extract_feature_slice(self.features, 'rhythm')
        self.assertEqual(rhythm.shape, (250,))

    def test_extract_melody_slice(self):
        """Test melody feature extraction."""
        melody = self.converter.extract_feature_slice(self.features, 'melody')
        self.assertEqual(melody.shape, (200,))

    def test_extract_dynamics_slice(self):
        """Test dynamics feature extraction."""
        dynamics = self.converter.extract_feature_slice(self.features, 'dynamics')
        self.assertEqual(dynamics.shape, (150,))

    def test_extract_texture_slice(self):
        """Test texture feature extraction."""
        texture = self.converter.extract_feature_slice(self.features, 'texture')
        self.assertEqual(texture.shape, (100,))

    def test_extract_structure_slice(self):
        """Test structure feature extraction."""
        structure = self.converter.extract_feature_slice(self.features, 'structure')
        self.assertEqual(structure.shape, (50,))

    def test_extract_orchestration_slice(self):
        """Test orchestration feature extraction."""
        orch = self.converter.extract_feature_slice(self.features, 'orchestration')
        self.assertEqual(orch.shape, (150,))

    def test_invalid_category(self):
        """Test handling of invalid category."""
        with self.assertRaises(ValueError):
            self.converter.extract_feature_slice(self.features, 'invalid')


class TestMIDIValidation(unittest.TestCase):
    """Test MIDI validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = RuleBasedFeaturesToMIDI()

    def test_validate_valid_midi(self):
        """Test validation of valid MIDI."""
        # Generate valid MIDI
        features = np.random.randn(1150)
        midi = self.converter.features_to_midi(features)

        self.assertTrue(self.converter.validate_output(midi))

    def test_validate_empty_midi(self):
        """Test rejection of empty MIDI."""
        midi = mido.MidiFile()
        self.assertFalse(self.converter.validate_output(midi))

    def test_validate_midi_no_notes(self):
        """Test rejection of MIDI with no notes."""
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        midi.tracks.append(track)

        self.assertFalse(self.converter.validate_output(midi))


class TestReconstructionQuality(unittest.TestCase):
    """Test reconstruction quality metrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = RuleBasedFeaturesToMIDI()

    def create_simple_midi(self) -> mido.MidiFile:
        """Create a simple test MIDI file."""
        midi = mido.MidiFile(type=1, ticks_per_beat=480)
        track = mido.MidiTrack()
        midi.tracks.append(track)

        # Add tempo
        track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

        # Add a few notes
        for pitch in [60, 62, 64, 65]:
            track.append(mido.Message('note_on', note=pitch, velocity=80, time=0))
            track.append(mido.Message('note_off', note=pitch, velocity=0, time=480))

        track.append(mido.MetaMessage('end_of_track', time=0))
        return midi

    def test_compute_quality_identical(self):
        """Test quality metrics for identical MIDI."""
        midi1 = self.create_simple_midi()
        midi2 = self.create_simple_midi()

        quality = self.converter.compute_reconstruction_quality(midi1, midi2)

        # Should have perfect scores
        self.assertGreaterEqual(quality['note_precision'], 0.9)
        self.assertGreaterEqual(quality['note_recall'], 0.9)
        self.assertGreaterEqual(quality['note_f1'], 0.9)

    def test_compute_quality_different(self):
        """Test quality metrics for different MIDI."""
        midi1 = self.create_simple_midi()

        # Create different MIDI
        features = np.random.randn(1150)
        midi2 = self.converter.features_to_midi(features)

        quality = self.converter.compute_reconstruction_quality(midi1, midi2)

        # Check metrics are in valid range [0, 1]
        self.assertTrue(0 <= quality['note_precision'] <= 1)
        self.assertTrue(0 <= quality['note_recall'] <= 1)
        self.assertTrue(0 <= quality['note_f1'] <= 1)
        self.assertTrue(0 <= quality['overall_similarity'] <= 1)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test end-to-end workflow with real feature extractor."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = RuleBasedFeaturesToMIDI(verbose=False)
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.temp_dir)

    def test_roundtrip_workflow(self):
        """Test full roundtrip: MIDI → Features → MIDI."""
        # Note: This test assumes DeepFeatureExtractor exists
        try:
            from midi_generator.synthesis.deep_feature_extractor import DeepFeatureExtractor

            # Create simple test MIDI
            original_midi = self.converter.features_to_midi(np.random.randn(1150))
            midi_path = self.temp_dir / "original.mid"
            original_midi.save(str(midi_path))

            # Extract features
            extractor = DeepFeatureExtractor()
            features = extractor.extract(midi_path)

            # Convert back to MIDI
            reconstructed_midi = self.converter.features_to_midi(features)

            # Validate
            self.assertTrue(self.converter.validate_output(reconstructed_midi))

            # Compute quality
            quality = self.converter.compute_reconstruction_quality(
                original_midi, reconstructed_midi
            )

            # Should have some similarity (though won't be perfect)
            self.assertGreater(quality['overall_similarity'], 0.0)

        except ImportError:
            self.skipTest("DeepFeatureExtractor not available")


# ============================================================================
# Test Runner
# ============================================================================

def run_tests(verbose: bool = True):
    """
    Run all tests.

    Args:
        verbose: Print verbose test output
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestValidateFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestRuleBasedConverter))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureSlicing))
    suite.addTests(loader.loadTestsFromTestCase(TestMIDIValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestReconstructionQuality))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndWorkflow))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)

    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    print("="*70)
    print("Features→MIDI Decoder Test Suite")
    print("="*70)
    print()

    success = run_tests(verbose=True)

    print()
    print("="*70)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*70)

    sys.exit(0 if success else 1)
