"""
Learning Module - Musical Program Synthesis
============================================

This module contains machine learning components for the system:

- Pattern recognition and extraction
- Corpus learning from MIDI files
- Motif library management
- Natural language to parameter prediction
- Feature-parameter mapping (Agent 9)
- Musical locality functions (Agent 1)

Key Components:
- MusicalLocalityFunctions: 12 musical transformations for semantic discovery (Agent 1)
- FeatureParameterMapper: Maps 1000 features to 515+ parameters (Agent 9)
- PatternExtractor: Extract musical patterns from MIDI
- CorpusLearner: Learn from MIDI corpus
- MotifLibrary: Manage and reuse motifs
- NaturalLanguagePredictor: Natural language interface

Author: Agents 1, 2, 9, and others
License: MIT
"""

# Agent 1: Musical Locality Functions
try:
    from .musical_locality import (
        LocalityType,
        MusicalTransform,
        MusicalLocalityFunctions,
        create_random_transform
    )
except ImportError:
    LocalityType = None
    MusicalTransform = None
    MusicalLocalityFunctions = None
    create_random_transform = None

# Agent 9: Feature-Parameter Mapping
try:
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
except ImportError:
    FeatureParameterMapper = None
    FeatureImportance = None
    MappingMetrics = None
    TrainingExample = None
    PredictionResult = None
    FeatureSelector = None
    create_mapper = None
    train_from_midi_corpus = None

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
    # Agent 1: Musical Locality Functions
    'LocalityType',
    'MusicalTransform',
    'MusicalLocalityFunctions',
    'create_random_transform',

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
