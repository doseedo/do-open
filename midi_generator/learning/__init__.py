"""
Learning Module - Musical Program Synthesis
============================================

This module contains machine learning components for the system:

- Pattern recognition and extraction
- Corpus learning from MIDI files
- Motif library management
- Natural language to parameter prediction
- Feature-parameter mapping (Agent 9)
- Semantic feature discovery (Agent 3)
- Texture semantic encoding (Agent 6)
- Texture analysis algorithms (Agent 6)

Key Components:
- SemanticFeatureEncoder: Neural architecture for discovering musical parameters (Agent 3)
- TextureSemanticEncoder: Specialized encoder for texture parameters (Agent 6)
- DetailedTextureAnalyzer: Comprehensive texture analysis (Agent 6)
- FeatureParameterMapper: Maps 1000 features to 515+ parameters (Agent 9)
- PatternExtractor: Extract musical patterns from MIDI
- CorpusLearner: Learn from MIDI corpus
- MotifLibrary: Manage and reuse motifs
- NaturalLanguagePredictor: Natural language interface

Author: Agents 2, 3, 6, 9, and others
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

# Agent 3: Semantic Feature Discovery
try:
    from .semantic_encoder import (
        SemanticFeatureEncoder,
        EncoderConfig,
        TrainingMetrics,
        EncoderNetwork,
        DecoderNetwork,
        LocalityPredictor,
        create_default_encoder,
        compute_reconstruction_quality,
        analyze_semantic_features
    )
    SEMANTIC_ENCODER_AVAILABLE = True
except ImportError:
    SemanticFeatureEncoder = None
    EncoderConfig = None
    TrainingMetrics = None
    EncoderNetwork = None
    DecoderNetwork = None
    LocalityPredictor = None
    create_default_encoder = None
    compute_reconstruction_quality = None
    analyze_semantic_features = None
    SEMANTIC_ENCODER_AVAILABLE = False

# Agent 6: Texture Semantic Encoding
try:
    from .texture_encoder import (
        TextureSemanticEncoder,
        TextureEncoderConfig,
        TextureAnalyzer,
        TextureLocalityType,
        create_default_texture_encoder,
        extract_texture_from_midi
    )
    TEXTURE_ENCODER_AVAILABLE = True
except ImportError:
    TextureSemanticEncoder = None
    TextureEncoderConfig = None
    TextureAnalyzer = None
    TextureLocalityType = None
    create_default_texture_encoder = None
    extract_texture_from_midi = None
    TEXTURE_ENCODER_AVAILABLE = False

# Agent 6: Detailed Texture Analysis
try:
    from .texture_analysis import (
        DetailedTextureAnalyzer,
        Note,
        TextureProfile
    )
    TEXTURE_ANALYSIS_AVAILABLE = True
except ImportError:
    DetailedTextureAnalyzer = None
    Note = None
    TextureProfile = None
    TEXTURE_ANALYSIS_AVAILABLE = False


__all__ = [
    # Agent 3: Semantic Feature Discovery
    'SemanticFeatureEncoder',
    'EncoderConfig',
    'TrainingMetrics',
    'EncoderNetwork',
    'DecoderNetwork',
    'LocalityPredictor',
    'create_default_encoder',
    'compute_reconstruction_quality',
    'analyze_semantic_features',
    'SEMANTIC_ENCODER_AVAILABLE',

    # Agent 6: Texture Semantic Encoding
    'TextureSemanticEncoder',
    'TextureEncoderConfig',
    'TextureAnalyzer',
    'TextureLocalityType',
    'create_default_texture_encoder',
    'extract_texture_from_midi',
    'TEXTURE_ENCODER_AVAILABLE',

    # Agent 6: Detailed Texture Analysis
    'DetailedTextureAnalyzer',
    'Note',
    'TextureProfile',
    'TEXTURE_ANALYSIS_AVAILABLE',

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
