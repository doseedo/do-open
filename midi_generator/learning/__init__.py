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
- Cross-dimensional pattern discovery (Agent 7)

Key Components:

**Agent 3: Semantic Feature Discovery**
- SemanticFeatureEncoder: Neural architecture for discovering musical parameters
- Discovers interpretable semantic features from reconstruction gaps

**Agent 7: Cross-Dimensional Encoder (Modular Architecture)**
- CrossDimensionalEncoder: Discovers interaction patterns between musical dimensions
- Fuses 110 parameters from 5 dimension encoders (harmony, rhythm, form, orchestration, texture)
- Discovers 10 cross-dimensional parameters capturing musical coherence
- InteractionPatternDiscoverer: Finds statistical patterns between dimensions
- ParameterCouplingValidator: Validates musical coherence constraints

**Agent 9: Feature-Parameter Mapping**
- FeatureParameterMapper: Maps 1000 features to 515+ parameters
- Learns optimal feature-to-parameter relationships

**Other Components:**
- PatternExtractor: Extract musical patterns from MIDI
- CorpusLearner: Learn from MIDI corpus
- MotifLibrary: Manage and reuse motifs
- NaturalLanguagePredictor: Natural language interface

Author: Agents 2, 3, 7, 9, and others
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

# Agent 7: Cross-Dimensional Encoder (Modular Architecture)
try:
    from .cross_dimensional_encoder import (
        CrossDimensionalEncoder,
        CrossDimensionalConfig,
        CrossDimensionalParameters,
        FusionNetwork,
        CrossEncoderNetwork,
        ReconstructionNetwork,
        create_default_cross_encoder,
        analyze_interaction_patterns
    )
    CROSS_DIMENSIONAL_ENCODER_AVAILABLE = True
except ImportError:
    CrossDimensionalEncoder = None
    CrossDimensionalConfig = None
    CrossDimensionalParameters = None
    FusionNetwork = None
    CrossEncoderNetwork = None
    ReconstructionNetwork = None
    create_default_cross_encoder = None
    analyze_interaction_patterns = None
    CROSS_DIMENSIONAL_ENCODER_AVAILABLE = False

# Agent 7: Interaction Pattern Discovery
try:
    from .interaction_patterns import (
        InteractionPatternDiscoverer,
        InteractionPattern
    )
    INTERACTION_PATTERNS_AVAILABLE = True
except ImportError:
    InteractionPatternDiscoverer = None
    InteractionPattern = None
    INTERACTION_PATTERNS_AVAILABLE = False

# Agent 7: Parameter Coupling Validation
try:
    from .parameter_coupling import (
        ParameterCouplingValidator,
        CouplingConstraint,
        CouplingType
    )
    PARAMETER_COUPLING_AVAILABLE = True
except ImportError:
    ParameterCouplingValidator = None
    CouplingConstraint = None
    CouplingType = None
    PARAMETER_COUPLING_AVAILABLE = False


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

    # Agent 7: Cross-Dimensional Encoder (Modular Architecture)
    'CrossDimensionalEncoder',
    'CrossDimensionalConfig',
    'CrossDimensionalParameters',
    'FusionNetwork',
    'CrossEncoderNetwork',
    'ReconstructionNetwork',
    'create_default_cross_encoder',
    'analyze_interaction_patterns',
    'CROSS_DIMENSIONAL_ENCODER_AVAILABLE',

    # Agent 7: Interaction Pattern Discovery
    'InteractionPatternDiscoverer',
    'InteractionPattern',
    'INTERACTION_PATTERNS_AVAILABLE',

    # Agent 7: Parameter Coupling Validation
    'ParameterCouplingValidator',
    'CouplingConstraint',
    'CouplingType',
    'PARAMETER_COUPLING_AVAILABLE',

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
