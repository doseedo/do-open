"""
Analysis Package - MIDI Analysis and Gap Detection

This package provides comprehensive MIDI analysis, feature extraction,
genre detection, and intelligent gap detection for self-expanding music generation.

Modules:
- midi_analyzer: Core MIDI analysis and feature extraction
- genre_detector: Genre detection and style classification
- dataset_analyzer: Dataset analysis and statistics
- intelligent_gap_detector: Gap detection and parameter suggestion (Agent 10)
"""

# Core analysis tools
from .midi_analyzer import MidiAnalyzer, NoteEvent, ChordEvent
from .genre_detector import GenreDetector, RhythmicFeatureExtractor
from .dataset_analyzer import DatasetAnalyzer

# Agent 10: Intelligent Gap Detector
from .intelligent_gap_detector import (
    # Core classes
    IntelligentGapDetector,
    FeatureToParameterMapper,
    FeatureParameterMapping,
    ParameterSuggestion,

    # Advanced analysis
    GapPredictor,
    GapTracker,
    AdvancedCorrelationAnalyzer,
    GapVisualizationHelper,
    XGBoostIntegration,

    # Convenience functions
    detect_gaps_from_errors,
    analyze_systematic_gaps,
    create_full_pipeline,
)

__all__ = [
    # Core analysis
    'MidiAnalyzer',
    'NoteEvent',
    'ChordEvent',
    'GenreDetector',
    'RhythmicFeatureExtractor',
    'DatasetAnalyzer',

    # Gap detection
    'IntelligentGapDetector',
    'FeatureToParameterMapper',
    'FeatureParameterMapping',
    'ParameterSuggestion',
    'GapPredictor',
    'GapTracker',
    'AdvancedCorrelationAnalyzer',
    'GapVisualizationHelper',
    'XGBoostIntegration',

    # Functions
    'detect_gaps_from_errors',
    'analyze_systematic_gaps',
    'create_full_pipeline',
]
