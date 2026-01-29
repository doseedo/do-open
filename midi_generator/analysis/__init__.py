"""
Analysis Package - MIDI Analysis and Gap Detection

This package provides comprehensive MIDI analysis, feature extraction,
genre detection, and intelligent gap detection for self-expanding music generation.

Modules:
- midi_analyzer: Core MIDI analysis and feature extraction
- genre_detector: Genre detection and style classification
- dataset_analyzer: Dataset analysis and statistics
- intelligent_gap_detector: Gap detection and parameter suggestion (Agent 10)
- feature_correlation_analyzer: Feature correlation analysis (Agent 25)
"""

# Core analysis tools - these are essential
try:
    from .midi_analyzer import MidiAnalyzer, NoteEvent, ChordEvent
except ImportError as e:
    print(f"Warning: midi_analyzer import failed: {e}")
    MidiAnalyzer = None
    NoteEvent = None
    ChordEvent = None

try:
    from .genre_detector import GenreDetector, RhythmicFeatureExtractor
except ImportError as e:
    print(f"Warning: genre_detector import failed: {e}")
    GenreDetector = None
    RhythmicFeatureExtractor = None

try:
    from .dataset_analyzer import DatasetAnalyzer
except ImportError as e:
    DatasetAnalyzer = None

# Agent 10: Intelligent Gap Detector (optional)
try:
    from .intelligent_gap_detector import (
        IntelligentGapDetector,
        FeatureToParameterMapper,
        FeatureParameterMapping,
        ParameterSuggestion,
        GapPredictor,
        GapTracker,
        AdvancedCorrelationAnalyzer,
        GapVisualizationHelper,
        XGBoostIntegration,
        detect_gaps_from_errors,
        analyze_systematic_gaps,
        create_full_pipeline,
    )
except ImportError as e:
    IntelligentGapDetector = None
    FeatureToParameterMapper = None
    FeatureParameterMapping = None
    ParameterSuggestion = None
    GapPredictor = None
    GapTracker = None
    AdvancedCorrelationAnalyzer = None
    GapVisualizationHelper = None
    XGBoostIntegration = None
    detect_gaps_from_errors = None
    analyze_systematic_gaps = None
    create_full_pipeline = None

# Agent 25: Feature Correlation Analyzer (optional)
try:
    from .feature_correlation_analyzer import (
        FeatureCorrelationAnalyzer,
        CorrelationResult,
        RedundantFeaturePair,
        FeatureSubset,
        FeatureInteraction,
        CorrelationAnalysisReport,
        quick_correlation_analysis,
        find_best_features_for_parameter,
    )
except ImportError as e:
    FeatureCorrelationAnalyzer = None
    CorrelationResult = None
    RedundantFeaturePair = None
    FeatureSubset = None
    FeatureInteraction = None
    CorrelationAnalysisReport = None
    quick_correlation_analysis = None
    find_best_features_for_parameter = None

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

    # Agent 25: Feature correlation
    'FeatureCorrelationAnalyzer',
    'CorrelationResult',
    'RedundantFeaturePair',
    'FeatureSubset',
    'FeatureInteraction',
    'CorrelationAnalysisReport',
    'quick_correlation_analysis',
    'find_best_features_for_parameter',
]
