"""
Unit Tests for Gap Dataset - Agent 4
=====================================

Comprehensive tests for:
1. ParameterMIDIGenerator
2. GapAnalyzer
3. GapCache
4. GapDataset

Author: Agent 4 - Gap Dataset Creation
License: MIT
"""

import unittest
import tempfile
import shutil
import json
import pickle
from pathlib import Path
import numpy as np

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from midi_generator.learning.gap_dataset import (
    ParameterMIDIGenerator,
    GapAnalyzer,
    GapCache,
    ReconstructionGap,
    CorpusGapStatistics
)

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

try:
    from midi_generator.learning.gap_dataset import GapDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestParameterMIDIGenerator(unittest.TestCase):
    """Test ParameterMIDIGenerator"""

    def setUp(self):
        """Set up test fixtures"""
        if not MIDO_AVAILABLE:
            self.skipTest("mido not available")

        self.generator = ParameterMIDIGenerator(verbose=False)
        self.temp_dir = tempfile.mkdtemp()

        # Create sample parameters
        self.sample_parameters = {
            'level1_global': {
                'tempo.bpm': 120.0,
                'time_signature.string': '4/4',
                'genre.primary': 'jazz',
                'key.root': 'C',
                'key.mode': 'major',
                'duration.beats': 16,
                'duration.seconds': 8.0,
                'complexity.overall': 0.7
            },
            'level2_universal': {
                'melody': {
                    'pitch_range.min': 60,
                    'pitch_range.max': 84,
                    'pitch_statistics.mean': 72.0,
                    'note_density.notes_per_beat': 2.0
                },
                'harmony': {
                    'chord_complexity.avg_notes': 3.0,
                    'chord_change_rate.changes_per_beat': 0.5
                },
                'rhythm': {
                    'note_density.notes_per_beat': 4.0,
                    'syncopation.degree': 0.3
                },
                'bass': {
                    'pitch_range.min': 28,
                    'pitch_range.max': 55,
                    'pitch_statistics.mean': 40.0
                }
            },
            'level3_genre_specific': {
                'harmony': {
                    'jazz.voicing_complexity': 0.8
                },
                'bass': {
                    'jazz.walking_pattern': True
                }
            }
        }

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def test_generator_initialization(self):
        """Test generator initializes correctly"""
        self.assertIsNotNone(self.generator)

    def test_generate_midi(self):
        """Test basic MIDI generation"""
        output_path = Path(self.temp_dir) / 'test.mid'
        midi = self.generator.generate(
            self.sample_parameters,
            output_path=output_path
        )

        self.assertIsNotNone(midi, "MIDI generation should succeed")
        self.assertTrue(output_path.exists(), "MIDI file should be created")
        self.assertGreater(output_path.stat().st_size, 0, "MIDI file should not be empty")

    def test_generated_midi_structure(self):
        """Test generated MIDI has correct structure"""
        midi = self.generator.generate(self.sample_parameters)

        self.assertIsNotNone(midi)
        self.assertGreater(len(midi.tracks), 0, "Should have at least one track")

        # Check for tempo track
        tempo_track = midi.tracks[0]
        has_tempo = any(msg.type == 'set_tempo' for msg in tempo_track)
        self.assertTrue(has_tempo, "Should have tempo message")

    def test_parse_time_signature(self):
        """Test time signature parsing"""
        numerator, denominator = self.generator._parse_time_signature('4/4')
        self.assertEqual(numerator, 4)
        self.assertEqual(denominator, 4)

        numerator, denominator = self.generator._parse_time_signature('3/4')
        self.assertEqual(numerator, 3)
        self.assertEqual(denominator, 4)

    def test_build_chord(self):
        """Test chord building"""
        chord = self.generator._build_chord(60, 3)  # C major triad
        self.assertEqual(len(chord), 3)
        self.assertIn(60, chord)  # Root
        self.assertIn(64, chord)  # Major 3rd
        self.assertIn(67, chord)  # Perfect 5th

    def test_generate_with_missing_parameters(self):
        """Test generation with incomplete parameters"""
        minimal_params = {
            'level1_global': {'tempo.bpm': 120.0},
            'level2_universal': {},
            'level3_genre_specific': {}
        }

        midi = self.generator.generate(minimal_params)
        self.assertIsNotNone(midi, "Should handle missing parameters gracefully")

    def test_generate_multiple_files(self):
        """Test generating multiple MIDI files"""
        for i in range(3):
            output_path = Path(self.temp_dir) / f'test_{i}.mid'
            midi = self.generator.generate(
                self.sample_parameters,
                output_path=output_path
            )
            self.assertIsNotNone(midi)
            self.assertTrue(output_path.exists())


class TestGapCache(unittest.TestCase):
    """Test GapCache"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = GapCache(
            cache_dir=Path(self.temp_dir) / 'cache',
            max_size_gb=0.1,  # Small cache for testing
            verbose=False
        )

        # Create sample gap
        self.sample_gap = ReconstructionGap(
            file_id='test_001',
            file_path='/tmp/test.mid',
            original_features=np.random.randn(200),
            original_parameters={'test': 1.0},
            reconstructed_features=np.random.randn(200),
            reconstructed_parameters={'test': 0.9},
            feature_gaps=np.random.rand(200),
            parameter_gaps={'test': 0.1},
            total_gap=0.05,
            max_gap_indices=[0, 1, 2],
            max_gap_values=[0.9, 0.8, 0.7],
            computation_time=1.0,
            success=True
        )

        # Create sample MIDI file
        self.test_midi_path = Path(self.temp_dir) / 'test.mid'
        self.test_midi_path.write_text("dummy midi content")

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def test_cache_initialization(self):
        """Test cache initializes correctly"""
        self.assertTrue(self.cache.cache_dir.exists())
        self.assertTrue(self.cache.metadata_file.exists())

    def test_put_and_get(self):
        """Test putting and getting from cache"""
        # Put gap in cache
        self.cache.put(self.test_midi_path, self.sample_gap)

        # Get gap from cache
        cached_gap = self.cache.get(self.test_midi_path)

        self.assertIsNotNone(cached_gap)
        self.assertEqual(cached_gap.file_id, self.sample_gap.file_id)
        self.assertEqual(cached_gap.total_gap, self.sample_gap.total_gap)

    def test_cache_miss(self):
        """Test cache miss returns None"""
        non_existent_path = Path(self.temp_dir) / 'nonexistent.mid'
        non_existent_path.write_text("other content")

        result = self.cache.get(non_existent_path)
        self.assertIsNone(result)

    def test_cache_stats(self):
        """Test cache statistics"""
        # Add items to cache
        self.cache.put(self.test_midi_path, self.sample_gap)

        # Get statistics
        stats = self.cache.get_stats()

        self.assertEqual(stats['entries'], 1)
        self.assertGreater(stats['total_size_gb'], 0)
        self.assertEqual(stats['miss_count'], 0)

    def test_cache_eviction(self):
        """Test cache eviction when full"""
        # Create multiple large gaps to fill cache
        for i in range(10):
            test_path = Path(self.temp_dir) / f'test_{i}.mid'
            test_path.write_text(f"dummy midi content {i}")

            gap = ReconstructionGap(
                file_id=f'test_{i}',
                file_path=str(test_path),
                original_features=np.random.randn(200),
                original_parameters={},
                reconstructed_features=np.random.randn(200),
                reconstructed_parameters={},
                feature_gaps=np.random.rand(200),
                parameter_gaps={},
                total_gap=0.05,
                max_gap_indices=[],
                max_gap_values=[],
                computation_time=1.0,
                success=True
            )

            self.cache.put(test_path, gap)

        # Check that cache size is within limits
        stats = self.cache.get_stats()
        self.assertLessEqual(stats['total_size_gb'], 0.1)

    def test_clear_cache(self):
        """Test clearing cache"""
        # Add item
        self.cache.put(self.test_midi_path, self.sample_gap)

        # Clear
        self.cache.clear()

        # Verify cleared
        stats = self.cache.get_stats()
        self.assertEqual(stats['entries'], 0)
        self.assertEqual(stats['total_size_gb'], 0)

    def test_compute_file_hash(self):
        """Test file hash computation"""
        hash1 = self.cache._compute_file_hash(self.test_midi_path)
        hash2 = self.cache._compute_file_hash(self.test_midi_path)

        self.assertEqual(hash1, hash2, "Hash should be deterministic")
        self.assertEqual(len(hash1), 64, "SHA256 hash should be 64 chars")


@unittest.skipIf(not MIDO_AVAILABLE, "mido not available")
class TestGapAnalyzer(unittest.TestCase):
    """Test GapAnalyzer (requires mido)"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

        # Create simple test MIDI file
        self.test_midi_path = self._create_test_midi()

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def _create_test_midi(self) -> Path:
        """Create a simple valid MIDI file for testing"""
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)

        # Add tempo
        track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

        # Add simple notes
        track.append(mido.Message('note_on', note=60, velocity=64, time=0))
        track.append(mido.Message('note_off', note=60, velocity=0, time=480))
        track.append(mido.Message('note_on', note=64, velocity=64, time=0))
        track.append(mido.Message('note_off', note=64, velocity=0, time=480))
        track.append(mido.Message('note_on', note=67, velocity=64, time=0))
        track.append(mido.Message('note_off', note=67, velocity=0, time=480))

        path = Path(self.temp_dir) / 'test.mid'
        midi.save(path)
        return path

    def test_reconstruction_gap_dataclass(self):
        """Test ReconstructionGap dataclass"""
        gap = ReconstructionGap(
            file_id='test',
            file_path='/tmp/test.mid',
            original_features=np.array([1.0, 2.0, 3.0]),
            original_parameters={},
            reconstructed_features=np.array([1.1, 2.1, 2.9]),
            reconstructed_parameters={},
            feature_gaps=np.array([0.1, 0.1, 0.1]),
            parameter_gaps={},
            total_gap=0.1,
            max_gap_indices=[0, 1, 2],
            max_gap_values=[0.1, 0.1, 0.1]
        )

        self.assertEqual(gap.file_id, 'test')
        self.assertEqual(len(gap.original_features), 3)

        # Test get_top_gaps
        top_gaps = gap.get_top_gaps(k=2)
        self.assertEqual(len(top_gaps), 2)

        # Test to_dict
        gap_dict = gap.to_dict()
        self.assertIn('file_id', gap_dict)
        self.assertIn('total_gap', gap_dict)

    def test_corpus_gap_statistics_dataclass(self):
        """Test CorpusGapStatistics dataclass"""
        stats = CorpusGapStatistics(
            n_files=100,
            total_files_processed=95,
            failed_files=5,
            mean_feature_gaps=np.random.rand(200),
            std_feature_gaps=np.random.rand(200),
            max_feature_gaps=np.random.rand(200),
            mean_parameter_gaps={'param1': 0.1},
            std_parameter_gaps={'param1': 0.05},
            max_parameter_gaps={'param1': 0.3},
            mean_total_gap=0.15,
            std_total_gap=0.05,
            percentiles={'p50': 0.15, 'p95': 0.25},
            top_gap_features=[(0, 0.5), (1, 0.4)]
        )

        self.assertEqual(stats.n_files, 100)
        self.assertEqual(stats.failed_files, 5)

        # Test to_dict
        stats_dict = stats.to_dict()
        self.assertIn('mean_total_gap', stats_dict)
        self.assertIn('top_gap_features', stats_dict)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
@unittest.skipIf(not MIDO_AVAILABLE, "mido not available")
class TestGapDataset(unittest.TestCase):
    """Test GapDataset (requires PyTorch and mido)"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

        # Create test MIDI files
        self.test_midi_files = []
        for i in range(3):
            path = self._create_test_midi(i)
            self.test_midi_files.append(path)

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def _create_test_midi(self, idx: int) -> Path:
        """Create a simple valid MIDI file for testing"""
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)

        # Add tempo
        track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

        # Add different notes for each file
        base_note = 60 + idx * 4
        track.append(mido.Message('note_on', note=base_note, velocity=64, time=0))
        track.append(mido.Message('note_off', note=base_note, velocity=0, time=480))

        path = Path(self.temp_dir) / f'test_{idx}.mid'
        midi.save(path)
        return path

    def test_dataset_flatten_parameters(self):
        """Test parameter flattening"""
        from midi_generator.learning.gap_dataset import GapDataset

        # Create mock dataset (without precomputing)
        class MockGapAnalyzer:
            pass

        dataset = GapDataset(
            midi_files=self.test_midi_files,
            gap_analyzer=MockGapAnalyzer(),
            cache=None,
            precompute=False,
            verbose=False
        )

        # Test parameter flattening
        test_params = {
            'level1_global': {
                'tempo.bpm': 120.0,
                'key.root': 'C'
            },
            'level2_universal': {
                'melody': {
                    'pitch_range.min': 60,
                    'pitch_range.max': 84
                }
            },
            'level3_genre_specific': {}
        }

        flat = dataset._flatten_parameters(test_params)
        self.assertEqual(len(flat), 50, "Should have exactly 50 parameters")
        self.assertTrue(np.all(np.isfinite(flat)), "All values should be finite")


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_workflow(self):
        """Test complete workflow (mock version)"""
        # This tests the data structures without requiring
        # the full extraction pipeline

        # Create sample gap
        gap = ReconstructionGap(
            file_id='test',
            file_path='/tmp/test.mid',
            original_features=np.random.randn(200),
            original_parameters={
                'level1_global': {'tempo.bpm': 120.0},
                'level2_universal': {},
                'level3_genre_specific': {}
            },
            reconstructed_features=np.random.randn(200),
            reconstructed_parameters={
                'level1_global': {'tempo.bpm': 118.0},
                'level2_universal': {},
                'level3_genre_specific': {}
            },
            feature_gaps=np.random.rand(200),
            parameter_gaps={'tempo.bpm': 2.0},
            total_gap=0.05,
            max_gap_indices=[0, 1, 2],
            max_gap_values=[0.9, 0.8, 0.7]
        )

        # Verify gap structure
        self.assertEqual(len(gap.original_features), 200)
        self.assertEqual(len(gap.feature_gaps), 200)
        self.assertIn('tempo.bpm', gap.parameter_gaps)

        # Test serialization
        gap_dict = gap.to_dict()
        self.assertIsInstance(gap_dict, dict)
        self.assertIn('total_gap', gap_dict)

    def test_cache_persistence(self):
        """Test cache persists across instances"""
        cache_dir = Path(self.temp_dir) / 'cache'

        # Create cache and add item
        cache1 = GapCache(cache_dir=cache_dir, verbose=False)

        test_path = Path(self.temp_dir) / 'test.mid'
        test_path.write_text("dummy content")

        gap = ReconstructionGap(
            file_id='test',
            file_path=str(test_path),
            original_features=np.random.randn(200),
            original_parameters={},
            reconstructed_features=np.random.randn(200),
            reconstructed_parameters={},
            feature_gaps=np.random.rand(200),
            parameter_gaps={},
            total_gap=0.05,
            max_gap_indices=[],
            max_gap_values=[]
        )

        cache1.put(test_path, gap)

        # Create new cache instance
        cache2 = GapCache(cache_dir=cache_dir, verbose=False)

        # Verify item exists
        cached_gap = cache2.get(test_path)
        self.assertIsNotNone(cached_gap)
        self.assertEqual(cached_gap.file_id, 'test')


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestParameterMIDIGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestGapCache))
    suite.addTests(loader.loadTestsFromTestCase(TestGapAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestGapDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
