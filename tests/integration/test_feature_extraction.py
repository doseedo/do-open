"""
Integration tests for Agent 8: Feature Extraction Pipeline

Tests the deep feature extractor's ability to extract 1000+ features
from MIDI files with proper validation and error handling.

Author: Agent 34 - Integration Testing Coordinator
"""

import pytest
import numpy as np
from pathlib import Path

try:
    from midi_generator.synthesis.deep_feature_extractor import DeepFeatureExtractor
    AGENT8_AVAILABLE = True
except ImportError:
    AGENT8_AVAILABLE = False

try:
    import mido
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False


@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary test data directory"""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    return test_dir


@pytest.fixture
def simple_midi(test_data_dir):
    """Create a simple test MIDI file"""
    if not HAS_MIDO:
        pytest.skip("mido not available")

    test_file = test_data_dir / "simple.mid"

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=500000))

    # C major scale
    notes = [60, 62, 64, 65, 67, 69, 71, 72]
    for note in notes:
        track.append(mido.Message('note_on', note=note, velocity=64, time=0))
        track.append(mido.Message('note_off', note=note, velocity=64, time=480))

    mid.save(test_file)
    return test_file


@pytest.fixture
def extractor():
    """Create feature extractor instance"""
    if not AGENT8_AVAILABLE:
        pytest.skip("Agent 8 not available")
    return DeepFeatureExtractor()


class TestFeatureExtraction:
    """Test suite for feature extraction"""

    def test_extractor_initialization(self, extractor):
        """Test that extractor can be initialized"""
        assert extractor is not None

    def test_extract_features(self, extractor, simple_midi):
        """Test basic feature extraction"""
        features = extractor.extract_features(str(simple_midi))
        assert features is not None
        assert len(features) > 0

    def test_feature_count(self, extractor, simple_midi):
        """Test that 1000+ features are extracted"""
        features = extractor.extract_features(str(simple_midi))
        assert len(features) >= 1000, f"Expected >=1000 features, got {len(features)}"

    def test_feature_names(self, extractor):
        """Test that feature names are available"""
        feature_names = extractor.get_feature_names()
        assert len(feature_names) >= 1000
        assert all(isinstance(name, str) for name in feature_names)

    def test_feature_values_valid(self, extractor, simple_midi):
        """Test that feature values are valid (no NaN/inf)"""
        features = extractor.extract_features(str(simple_midi))

        assert not np.any(np.isnan(features)), "Features contain NaN values"
        assert not np.any(np.isinf(features)), "Features contain infinite values"

    def test_feature_extraction_deterministic(self, extractor, simple_midi):
        """Test that feature extraction is deterministic"""
        features1 = extractor.extract_features(str(simple_midi))
        features2 = extractor.extract_features(str(simple_midi))

        np.testing.assert_array_equal(features1, features2,
                                     err_msg="Feature extraction is not deterministic")

    def test_feature_categories(self, extractor):
        """Test that all feature categories are present"""
        feature_names = extractor.get_feature_names()

        # Check for major categories
        categories = ['harmony', 'melody', 'rhythm', 'dynamics', 'texture', 'structure']
        for category in categories:
            matching = [name for name in feature_names if category in name.lower()]
            assert len(matching) > 0, f"No features found for category: {category}"


@pytest.mark.slow
class TestFeatureExtractionPerformance:
    """Performance tests for feature extraction"""

    def test_extraction_speed(self, extractor, simple_midi):
        """Test that extraction completes in reasonable time"""
        import time

        start = time.time()
        features = extractor.extract_features(str(simple_midi))
        duration = time.time() - start

        assert duration < 5.0, f"Extraction too slow: {duration:.2f}s"

    def test_batch_extraction(self, extractor, test_data_dir):
        """Test batch feature extraction"""
        if not HAS_MIDO:
            pytest.skip("mido not available")

        # Create multiple test files
        test_files = []
        for i in range(5):
            test_file = test_data_dir / f"test_{i}.mid"
            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)
            track.append(mido.MetaMessage('set_tempo', tempo=500000))
            track.append(mido.Message('note_on', note=60, velocity=64, time=0))
            track.append(mido.Message('note_off', note=60, velocity=64, time=480))
            mid.save(test_file)
            test_files.append(test_file)

        # Extract features from all
        import time
        start = time.time()

        feature_list = []
        for test_file in test_files:
            features = extractor.extract_features(str(test_file))
            feature_list.append(features)

        duration = time.time() - start

        assert len(feature_list) == 5
        assert duration < 10.0, f"Batch extraction too slow: {duration:.2f}s"


@pytest.mark.regression
class TestFeatureExtractionRegression:
    """Regression tests for known issues"""

    def test_empty_midi_handling(self, extractor, test_data_dir):
        """Test handling of empty MIDI files"""
        if not HAS_MIDO:
            pytest.skip("mido not available")

        # Create empty MIDI
        empty_file = test_data_dir / "empty.mid"
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        mid.save(empty_file)

        # Should handle gracefully
        try:
            features = extractor.extract_features(str(empty_file))
            assert features is not None
        except Exception as e:
            pytest.fail(f"Failed to handle empty MIDI: {e}")

    def test_malformed_midi(self, extractor, test_data_dir):
        """Test handling of malformed MIDI files"""
        # Create a non-MIDI file
        bad_file = test_data_dir / "bad.mid"
        bad_file.write_text("This is not a MIDI file")

        # Should handle gracefully
        with pytest.raises(Exception):
            extractor.extract_features(str(bad_file))
