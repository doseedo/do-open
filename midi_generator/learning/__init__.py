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
- Form/structure semantic encoding (Agent 4)

Key Components:
- SemanticFeatureEncoder: Neural architecture for discovering musical parameters (Agent 3)
- FormSemanticEncoder: Specialized encoder for form/structure parameters (Agent 4)
- FeatureParameterMapper: Maps 1000 features to 515+ parameters (Agent 9)
- PatternExtractor: Extract musical patterns from MIDI
- CorpusLearner: Learn from MIDI corpus
- MotifLibrary: Manage and reuse motifs
- NaturalLanguagePredictor: Natural language interface

Author: Agents 2, 3, 4, 9, and others
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

# Agent 4: Form/Structure Semantic Encoder
try:
    from .form_encoder import (
        FormSemanticEncoder,
        FormEncoderConfig,
        FormParameter,
        FormLocalityFunctions,
        create_form_encoder,
        PARAMETER_DESCRIPTIONS
    )
    FORM_ENCODER_AVAILABLE = True
except ImportError:
    FormSemanticEncoder = None
    FormEncoderConfig = None
    FormParameter = None
    FormLocalityFunctions = None
    create_form_encoder = None
    PARAMETER_DESCRIPTIONS = None
    FORM_ENCODER_AVAILABLE = False

# Agent 5: Orchestration Semantic Encoder (Modular Semantic Discovery)
try:
    from .orchestration_encoder import (
        OrchestrationSemanticEncoder,
        OrchestrationFeatureExtractor,
        VoiceIndependenceAnalyzer,
        create_orchestration_encoder,
        analyze_orchestration_from_midi,
        ORCHESTRATION_PARAMETERS
    )
    ORCHESTRATION_ENCODER_AVAILABLE = True
except ImportError:
    OrchestrationSemanticEncoder = None
    OrchestrationFeatureExtractor = None
    VoiceIndependenceAnalyzer = None
    create_orchestration_encoder = None
    analyze_orchestration_from_midi = None
    ORCHESTRATION_PARAMETERS = None
    ORCHESTRATION_ENCODER_AVAILABLE = False


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

    # Agent 4: Form/Structure Semantic Encoder
    'FormSemanticEncoder',
    'FormEncoderConfig',
    'FormParameter',
    'FormLocalityFunctions',
    'create_form_encoder',
    'PARAMETER_DESCRIPTIONS',
    'FORM_ENCODER_AVAILABLE',

    # Agent 5: Orchestration Semantic Encoder
    'OrchestrationSemanticEncoder',
    'OrchestrationFeatureExtractor',
    'VoiceIndependenceAnalyzer',
    'create_orchestration_encoder',
    'analyze_orchestration_from_midi',
    'ORCHESTRATION_PARAMETERS',
    'ORCHESTRATION_ENCODER_AVAILABLE',

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
