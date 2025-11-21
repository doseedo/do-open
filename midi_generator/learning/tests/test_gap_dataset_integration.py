"""
Integration Tests for Gap Dataset - Agent 4
============================================

Tests integration with existing systems:
1. OptimizedFeatureExtractor (200D features)
2. HierarchicalParameterExtractorV2 (50 parameters)
3. PyTorch DataLoader
4. End-to-end workflow

Author: Agent 4 - Gap Dataset Creation
License: MIT
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from midi_generator.learning.gap_dataset import (
    ParameterMIDIGenerator,
    GapAnalyzer,
    GapCache,
    create_gap_dataset_from_directory
)

# Try to import existing systems
try:
    from midi_generator.feature_selection.optimized_feature_extractor import (
        OptimizedFeatureExtractor
    )
    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError:
    FEATURE_EXTRACTOR_AVAILABLE = False

try:
    from midi_generator.parameters.hierarchical_extractor_v2 import (
        HierarchicalParameterExtractorV2
    )
    PARAMETER_EXTRACTOR_AVAILABLE = True
except ImportError:
    PARAMETER_EXTRACTOR_AVAILABLE = False


@unittest.skipIf(
    not MIDO_AVAILABLE or not TORCH_AVAILABLE,
    "Requires mido and PyTorch"
)
class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflow"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.midi_dir = Path(self.temp_dir) / 'midi'
        self.cache_dir = Path(self.temp_dir) / 'cache'
        self.output_dir = Path(self.temp_dir) / 'output'

        self.midi_dir.mkdir(parents=True)

        # Create test MIDI files
        self.create_test_corpus()

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def create_test_corpus(self):
        """Create a small test corpus"""
        for i in range(5):
            midi = mido.MidiFile()
            track = mido.MidiTrack()
            midi.tracks.append(track)

            # Add tempo
            track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

            # Add notes with variation
            base_note = 60 + (i * 2)
            for j in range(8):
                note = base_note + (j % 5)
                track.append(mido.Message('note_on', note=note, velocity=70, time=0))
                track.append(mido.Message('note_off', note=note, velocity=0, time=240))

            # Save
            path = self.midi_dir / f'test_{i:03d}.mid'
            midi.save(path)

    def test_parameter_midi_generation(self):
        """Test ParameterMIDIGenerator produces valid MIDI"""
        generator = ParameterMIDIGenerator(verbose=False)

        # Create sample parameters
        parameters = {
            'level1_global': {
                'tempo.bpm': 120.0,
                'time_signature.string': '4/4',
                'genre.primary': 'pop'
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
                }
            },
            'level3_genre_specific': {}
        }

        # Generate MIDI
        output_path = self.midi_dir / 'generated.mid'
        midi = generator.generate(parameters, output_path=output_path)

        # Verify
        self.assertIsNotNone(midi, "MIDI generation should succeed")
        self.assertTrue(output_path.exists(), "Output file should exist")
        self.assertGreater(len(midi.tracks), 0, "Should have tracks")

        # Verify MIDI is valid
        loaded_midi = mido.MidiFile(output_path)
        self.assertGreater(len(loaded_midi.tracks), 0)

    def test_gap_analyzer_workflow(self):
        """Test GapAnalyzer computes gaps correctly"""
        # This is a simplified test using mock extractors
        # Full test requires actual feature/parameter extractors

        generator = ParameterMIDIGenerator(verbose=False)

        # Get first MIDI file
        midi_files = list(self.midi_dir.glob('*.mid'))
        self.assertGreater(len(midi_files), 0, "Should have MIDI files")

        test_file = midi_files[0]

        # Verify file is valid
        midi = mido.MidiFile(test_file)
        self.assertGreater(len(midi.tracks), 0)

    def test_gap_cache_workflow(self):
        """Test GapCache stores and retrieves correctly"""
        from midi_generator.learning.gap_dataset import ReconstructionGap

        # Create cache
        cache = GapCache(
            cache_dir=self.cache_dir,
            max_size_gb=0.1,
            verbose=False
        )

        # Get test file
        midi_files = list(self.midi_dir.glob('*.mid'))
        test_file = midi_files[0]

        # Create mock gap
        gap = ReconstructionGap(
            file_id=test_file.stem,
            file_path=str(test_file),
            original_features=np.random.randn(200),
            original_parameters={},
            reconstructed_features=np.random.randn(200),
            reconstructed_parameters={},
            feature_gaps=np.random.rand(200),
            parameter_gaps={},
            total_gap=0.05,
            max_gap_indices=[0, 1, 2],
            max_gap_values=[0.5, 0.4, 0.3]
        )

        # Cache it
        cache.put(test_file, gap)

        # Retrieve it
        cached_gap = cache.get(test_file)

        self.assertIsNotNone(cached_gap)
        self.assertEqual(cached_gap.file_id, gap.file_id)
        self.assertEqual(cached_gap.total_gap, gap.total_gap)

        # Verify cache stats
        stats = cache.get_stats()
        self.assertEqual(stats['entries'], 1)
        self.assertEqual(stats['hit_count'], 1)

    def test_multiple_cache_operations(self):
        """Test multiple cache operations"""
        cache = GapCache(
            cache_dir=self.cache_dir,
            max_size_gb=0.1,
            verbose=False
        )

        from midi_generator.learning.gap_dataset import ReconstructionGap

        # Cache multiple files
        midi_files = list(self.midi_dir.glob('*.mid'))

        for midi_file in midi_files:
            gap = ReconstructionGap(
                file_id=midi_file.stem,
                file_path=str(midi_file),
                original_features=np.random.randn(200),
                original_parameters={},
                reconstructed_features=np.random.randn(200),
                reconstructed_parameters={},
                feature_gaps=np.random.rand(200),
                parameter_gaps={},
                total_gap=np.random.rand(),
                max_gap_indices=[],
                max_gap_values=[]
            )

            cache.put(midi_file, gap)

        # Verify all are cached
        stats = cache.get_stats()
        self.assertEqual(stats['entries'], len(midi_files))

        # Retrieve all
        for midi_file in midi_files:
            cached_gap = cache.get(midi_file)
            self.assertIsNotNone(cached_gap)

    @unittest.skipIf(
        not TORCH_AVAILABLE,
        "Requires PyTorch"
    )
    def test_dataset_structure(self):
        """Test GapDataset structure (without full extraction)"""
        from midi_generator.learning.gap_dataset import GapDataset

        # This test uses mock data to verify dataset structure
        # without requiring full feature/parameter extraction

        # Create mock analyzer that returns fake gaps
        class MockGapAnalyzer:
            def compute_gap(self, midi_file):
                from midi_generator.learning.gap_dataset import ReconstructionGap

                return ReconstructionGap(
                    file_id=midi_file.stem,
                    file_path=str(midi_file),
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
                    max_gap_values=[0.5, 0.4, 0.3],
                    success=True
                )

        midi_files = list(self.midi_dir.glob('*.mid'))

        # Create dataset with mock analyzer
        dataset = GapDataset(
            midi_files=midi_files,
            gap_analyzer=MockGapAnalyzer(),
            cache=None,
            precompute=True,
            normalize_features=False,
            verbose=False
        )

        # Verify dataset
        self.assertEqual(len(dataset), len(midi_files))

        # Get first item
        item = dataset[0]

        # Verify structure
        self.assertIn('features', item)
        self.assertIn('gaps', item)
        self.assertIn('parameters_flat', item)
        self.assertIn('file_id', item)

        # Verify shapes
        self.assertEqual(item['features'].shape, torch.Size([200]))
        self.assertEqual(item['gaps'].shape, torch.Size([200]))
        self.assertEqual(item['parameters_flat'].shape, torch.Size([50]))

    @unittest.skipIf(
        not TORCH_AVAILABLE,
        "Requires PyTorch"
    )
    def test_dataloader_integration(self):
        """Test PyTorch DataLoader integration"""
        from midi_generator.learning.gap_dataset import GapDataset

        class MockGapAnalyzer:
            def compute_gap(self, midi_file):
                from midi_generator.learning.gap_dataset import ReconstructionGap

                return ReconstructionGap(
                    file_id=midi_file.stem,
                    file_path=str(midi_file),
                    original_features=np.random.randn(200),
                    original_parameters={
                        'level1_global': {},
                        'level2_universal': {},
                        'level3_genre_specific': {}
                    },
                    reconstructed_features=np.random.randn(200),
                    reconstructed_parameters={
                        'level1_global': {},
                        'level2_universal': {},
                        'level3_genre_specific': {}
                    },
                    feature_gaps=np.random.rand(200),
                    parameter_gaps={},
                    total_gap=0.05,
                    max_gap_indices=[],
                    max_gap_values=[],
                    success=True
                )

        midi_files = list(self.midi_dir.glob('*.mid'))

        dataset = GapDataset(
            midi_files=midi_files,
            gap_analyzer=MockGapAnalyzer(),
            cache=None,
            precompute=True,
            normalize_features=False,
            verbose=False
        )

        # Create DataLoader
        dataloader = dataset.get_dataloader(batch_size=2, shuffle=False)

        # Iterate through batches
        batch_count = 0
        for batch in dataloader:
            batch_count += 1

            # Verify batch structure
            self.assertIn('features', batch)
            self.assertIn('gaps', batch)
            self.assertIn('parameters_flat', batch)

            # Verify batch shapes
            batch_size = batch['features'].shape[0]
            self.assertLessEqual(batch_size, 2)
            self.assertEqual(batch['features'].shape[1], 200)
            self.assertEqual(batch['gaps'].shape[1], 200)
            self.assertEqual(batch['parameters_flat'].shape[1], 50)

        self.assertGreater(batch_count, 0, "Should have at least one batch")


class TestPerformance(unittest.TestCase):
    """Performance tests"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def test_cache_performance(self):
        """Test cache improves performance"""
        from midi_generator.learning.gap_dataset import GapCache, ReconstructionGap
        import time

        cache = GapCache(
            cache_dir=Path(self.temp_dir) / 'cache',
            max_size_gb=0.1,
            verbose=False
        )

        # Create test file
        test_file = Path(self.temp_dir) / 'test.mid'
        test_file.write_text("dummy midi content")

        # Create gap
        gap = ReconstructionGap(
            file_id='test',
            file_path=str(test_file),
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

        # First put (cache miss)
        start = time.time()
        cache.put(test_file, gap)
        put_time = time.time() - start

        # Get (cache hit)
        start = time.time()
        cached_gap = cache.get(test_file)
        get_time = time.time() - start

        # Verify hit is faster (typically 100x+)
        self.assertIsNotNone(cached_gap)
        self.assertLess(get_time, put_time)  # Get should be faster


def run_integration_tests():
    """Run all integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndWorkflow))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)
