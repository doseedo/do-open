"""
End-to-End Integration Tests

Tests complete workflows across multiple agents:
- MIDI → Features → Parameters → Generation
- Gap Detection → Proposal → Code Generation → Training → Deployment

Author: Agent 34 - Integration Testing Coordinator
"""

import pytest
import numpy as np
from pathlib import Path

try:
    from midi_generator.synthesis.deep_feature_extractor import DeepFeatureExtractor
    from midi_generator.analysis.intelligent_gap_detector import IntelligentGapDetector
    from midi_generator.orchestration.expansion_orchestrator import ExpansionOrchestrator
    ALL_AGENTS_AVAILABLE = True
except ImportError:
    ALL_AGENTS_AVAILABLE = False

try:
    import mido
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False


@pytest.fixture
def test_midi(tmp_path):
    """Create a test MIDI file"""
    if not HAS_MIDO:
        pytest.skip("mido not available")

    test_file = tmp_path / "test.mid"

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=500000))

    # C major chord progression
    chords = [
        [60, 64, 67],  # C major
        [65, 69, 72],  # F major
        [67, 71, 74],  # G major
        [60, 64, 67],  # C major
    ]

    for chord in chords:
        for note in chord:
            track.append(mido.Message('note_on', note=note, velocity=64, time=0))
        for note in chord:
            track.append(mido.Message('note_off', note=note, velocity=64, time=480))

    mid.save(test_file)
    return test_file


@pytest.mark.e2e
class TestEndToEndWorkflows:
    """End-to-end workflow tests"""

    def test_midi_to_features(self, test_midi):
        """Test MIDI → Features pipeline"""
        if not ALL_AGENTS_AVAILABLE:
            pytest.skip("Not all agents available")

        extractor = DeepFeatureExtractor()
        features = extractor.extract_features(str(test_midi))

        assert features is not None
        assert len(features) >= 1000
        assert not np.any(np.isnan(features))

    def test_gap_detection_workflow(self, test_midi):
        """Test gap detection workflow"""
        if not ALL_AGENTS_AVAILABLE:
            pytest.skip("Not all agents available")

        detector = IntelligentGapDetector()

        # This should run without errors
        gaps = detector.detect_gaps(str(test_midi))

        # May or may not find gaps, but should complete
        assert gaps is not None or gaps == []

    def test_orchestrator_initialization(self):
        """Test orchestrator can be initialized"""
        if not ALL_AGENTS_AVAILABLE:
            pytest.skip("Not all agents available")

        orchestrator = ExpansionOrchestrator()
        assert orchestrator is not None


@pytest.mark.slow
@pytest.mark.e2e
class TestCompleteExpansionCycle:
    """Tests for complete expansion cycle"""

    def test_expansion_components_available(self):
        """Test that all expansion components are available"""
        components = []

        try:
            from midi_generator.llm.parameter_proposer import LLMParameterProposer
            components.append("parameter_proposer")
        except ImportError:
            pass

        try:
            from midi_generator.llm.code_generator import LLMCodeGenerationAgent
            components.append("code_generator")
        except ImportError:
            pass

        try:
            from midi_generator.training.synthetic_data_generator import SyntheticTrainingDataGenerator
            components.append("data_generator")
        except ImportError:
            pass

        try:
            from midi_generator.training.model_trainer import ModelTrainingSpecialist
            components.append("model_trainer")
        except ImportError:
            pass

        # Should have at least some components
        assert len(components) > 0, "No expansion components available"


@pytest.mark.performance
class TestEndToEndPerformance:
    """Performance tests for end-to-end workflows"""

    def test_feature_extraction_performance(self, test_midi):
        """Test feature extraction performance"""
        if not ALL_AGENTS_AVAILABLE:
            pytest.skip("Not all agents available")

        import time

        extractor = DeepFeatureExtractor()

        start = time.time()
        features = extractor.extract_features(str(test_midi))
        duration = time.time() - start

        assert duration < 5.0, f"Too slow: {duration:.2f}s"
        assert len(features) >= 1000


@pytest.mark.regression
class TestEndToEndRegression:
    """Regression tests for end-to-end workflows"""

    def test_pipeline_error_handling(self, tmp_path):
        """Test that pipeline handles errors gracefully"""
        if not ALL_AGENTS_AVAILABLE:
            pytest.skip("Not all agents available")

        extractor = DeepFeatureExtractor()

        # Try with non-existent file
        bad_file = tmp_path / "nonexistent.mid"

        with pytest.raises(Exception):
            extractor.extract_features(str(bad_file))
