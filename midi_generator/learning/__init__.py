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
- Orchestration semantic encoding (Agent 5)
- Texture semantic encoding (Agent 6)
- Texture analysis algorithms (Agent 6)
- Cross-dimensional pattern discovery (Agent 7)
- Sparse transform learning and gap-driven discovery (Agent 8)

Key Components:

**Agent 3: Semantic Feature Discovery**
- SemanticFeatureEncoder: Neural architecture for discovering musical parameters
- Discovers interpretable semantic features from reconstruction gaps

**Agent 4: Form/Structure Semantic Encoder**
- FormSemanticEncoder: Specialized encoder for form/structure parameters (15 params)

**Agent 5: Orchestration Semantic Encoder**
- OrchestrationSemanticEncoder: Orchestration and voice independence encoder (25 params)

**Agent 6: Texture Semantic Encoder**
- TextureSemanticEncoder: Specialized encoder for texture parameters (20 params)
- DetailedTextureAnalyzer: Comprehensive texture analysis

**Agent 7: Cross-Dimensional Encoder (Modular Architecture)**
- CrossDimensionalEncoder: Discovers interaction patterns between musical dimensions
- Fuses 110 parameters from 5 dimension encoders (harmony, rhythm, form, orchestration, texture)
- Discovers 10 cross-dimensional parameters capturing musical coherence
- InteractionPatternDiscoverer: Finds statistical patterns between dimensions
- ParameterCouplingValidator: Validates musical coherence constraints

**Agent 8: Sparse Transform Learning & Gap-Driven Discovery**
- SparseTransformLearner: Discover 120-140 data-driven transforms via sparse coding
- GapDrivenDiscovery: Iteratively discover transforms to fill reconstruction gaps
- MIDIFeatureExtractor: Extract 1,150D feature vectors from MIDI
- ResidualAnalyzer: Analyze reconstruction residuals
- GapClusterer: Cluster residuals to identify systematic gaps

**Agent 9: Feature-Parameter Mapping**
- FeatureParameterMapper: Maps 1000 features to 515+ parameters
- Learns optimal feature-to-parameter relationships

**Other Components:**
- PatternExtractor: Extract musical patterns from MIDI
- CorpusLearner: Learn from MIDI corpus
- MotifLibrary: Manage and reuse motifs
- NaturalLanguagePredictor: Natural language interface

Author: Agents 2, 3, 4, 5, 6, 7, 9, and others
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

# Agent 8: Sparse Transform Learning & Gap-Driven Discovery
try:
    from .sparse_transform_learning import (
        SparseTransformLearner,
        MIDIFeatureExtractor,
        LearnedComponent
    )
    from .gap_driven_discovery import (
        GapDrivenDiscovery,
        ResidualAnalyzer,
        GapClusterer,
        ReconstructionResult
    )
    SPARSE_LEARNING_AVAILABLE = True
except ImportError:
    SparseTransformLearner = None
    MIDIFeatureExtractor = None
    LearnedComponent = None
    GapDrivenDiscovery = None
    ResidualAnalyzer = None
    GapClusterer = None
    ReconstructionResult = None
    SPARSE_LEARNING_AVAILABLE = False


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

    # Agent 8: Sparse Transform Learning & Gap-Driven Discovery
    'SparseTransformLearner',
    'MIDIFeatureExtractor',
    'LearnedComponent',
    'GapDrivenDiscovery',
    'ResidualAnalyzer',
    'GapClusterer',
    'ReconstructionResult',
    'SPARSE_LEARNING_AVAILABLE',

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
