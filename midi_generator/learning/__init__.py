"""
Learning Module - Musical Program Synthesis
============================================

This module contains machine learning components for the system:

- Pattern recognition and extraction
- Corpus learning from MIDI files
- Motif library management
- Natural language to parameter prediction
- Feature-parameter mapping (Agent 9)

Key Components:
- FeatureParameterMapper: Maps 1000 features to 515+ parameters (Agent 9)
- PatternExtractor: Extract musical patterns from MIDI
- CorpusLearner: Learn from MIDI corpus
- MotifLibrary: Manage and reuse motifs
- NaturalLanguagePredictor: Natural language interface

Author: Agents 2, 9, and others
License: MIT
"""

from .feature_parameter_mapper import (
    FeatureParameterMapper,
    FeatureImportance,
    MappingMetrics,
    TrainingExample,
    PredictionResult,
    FeatureSelector,
    create_mapper,
    train_from_midi_corpus
)

try:
    from .pattern_extractor import PatternExtractor
except ImportError:
    PatternExtractor = None

try:
    from .corpus_learner import CorpusLearner
except ImportError:
    CorpusLearner = None

try:
    from .motif_library import MotifLibrary
except ImportError:
    MotifLibrary = None

try:
    from .natural_language_predictor import NaturalLanguagePredictor
except ImportError:
    NaturalLanguagePredictor = None


__all__ = [
    # Agent 9: Feature-Parameter Mapping
    'FeatureParameterMapper',
    'FeatureImportance',
    'MappingMetrics',
    'TrainingExample',
    'PredictionResult',
    'FeatureSelector',
    'create_mapper',
    'train_from_midi_corpus',

    # Other learning components
    'PatternExtractor',
    'CorpusLearner',
    'MotifLibrary',
    'NaturalLanguagePredictor',
]
