"""
Learning Module - Musical Program Synthesis
============================================

This module contains machine learning components for the system:

- Pattern recognition and extraction
- Corpus learning from MIDI files
- Motif library management
- Natural language to parameter prediction
- Feature-parameter mapping (Agent 9)
- Semantic feature representations (Agent 2)
- Musical locality functions (Agent 1)

Key Components:
- FeatureParameterMapper: Maps 1000 features to 515+ parameters (Agent 9)
- SemanticFeature: Learned semantic features (Agent 2)
- SemanticFeatureBank: Manage semantic features (Agent 2)
- MusicalLocalityFunctions: Musical transformations (Agent 1)
- PatternExtractor: Extract musical patterns from MIDI
- CorpusLearner: Learn from MIDI corpus
- MotifLibrary: Manage and reuse motifs
- NaturalLanguagePredictor: Natural language interface

Author: Agents 1, 2, 9, and others
License: MIT
"""

# Agent 9: Feature-Parameter Mapping
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

# Agent 1: Musical Locality Functions
try:
    from .musical_locality import (
        LocalityType,
        MusicalTransform,
        MusicalLocalityFunctions,
        transpose_features,
        augment_features,
        retrograde_features
    )
    LOCALITY_AVAILABLE = True
except ImportError:
    LocalityType = None
    MusicalTransform = None
    MusicalLocalityFunctions = None
    transpose_features = None
    augment_features = None
    retrograde_features = None
    LOCALITY_AVAILABLE = False

# Agent 2: Semantic Feature Representations
try:
    from .semantic_features import (
        FeatureModality,
        SemanticFeature,
        SemanticFeatureBank,
        cosine_similarity,
        euclidean_distance,
        activation_correlation,
        find_similar_features,
        detect_redundant_features,
        create_semantic_feature
    )
    SEMANTIC_FEATURES_AVAILABLE = True
except ImportError:
    FeatureModality = None
    SemanticFeature = None
    SemanticFeatureBank = None
    cosine_similarity = None
    euclidean_distance = None
    activation_correlation = None
    find_similar_features = None
    detect_redundant_features = None
    create_semantic_feature = None
    SEMANTIC_FEATURES_AVAILABLE = False

# Other learning components
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

    # Agent 1: Musical Locality Functions
    'LocalityType',
    'MusicalTransform',
    'MusicalLocalityFunctions',
    'transpose_features',
    'augment_features',
    'retrograde_features',
    'LOCALITY_AVAILABLE',

    # Agent 2: Semantic Feature Representations
    'FeatureModality',
    'SemanticFeature',
    'SemanticFeatureBank',
    'cosine_similarity',
    'euclidean_distance',
    'activation_correlation',
    'find_similar_features',
    'detect_redundant_features',
    'create_semantic_feature',
    'SEMANTIC_FEATURES_AVAILABLE',

    # Other learning components
    'PatternExtractor',
    'CorpusLearner',
    'MotifLibrary',
    'NaturalLanguagePredictor',
]
