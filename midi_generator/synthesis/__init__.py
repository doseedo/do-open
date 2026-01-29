#!/usr/bin/env python3
"""
Musical Program Synthesis System
=================================

This package implements a complete Musical Program Synthesis system that:
1. Extracts deep features from MIDI files (1000+ dimensions)
2. Learns optimal parameters via XGBoost
3. Generates similar music with precise control

Modules:
- deep_feature_extractor: Extract 1000+ musical features from MIDI files (Agent 8) ✅
- inverse_analyzer: Inverse MIDI analysis (coming soon)
- gap_detector: Intelligent gap detection (Agent 10) ✅
- transform_dsl: Domain-Specific Language for MIDI transforms (Agent 8) ✅
- synthetic_dataset: Generate training data for neural program synthesis (Agent 8) ✅
- neural_synthesizer: Neural architecture for learning transforms (Agent 8) ✅
- program_synthesis_trainer: Training infrastructure (Agent 8) ✅

Neural Program Synthesis (Agent 8):
- Phase 1-2: DSL with constrained grammar + 100 transform templates ✅
- Phase 3: Neural architecture (Transformer encoder-decoder) ✅
- Phase 4: Training infrastructure with grammar-constrained generation ✅
- Complete system for learning transforms from MIDI examples

Author: Agent 10 - Integration & API
"""

from .deep_feature_extractor import DeepFeatureExtractor, FeatureVector

# Neural Program Synthesis (DSL + Dataset)
try:
    from .transform_dsl import (
        DSLProgram,
        Statement,
        ForEach,
        If,
        Operation,
        Aggregate,
        Value,
        IteratorType,
        FilterType,
        OperationType,
        AggregatorType,
        DSLVocabulary,
        create_example_programs
    )
    from .synthetic_dataset import (
        SyntheticDatasetGenerator,
        SyntheticExample,
        TransformTemplateLibrary
    )
    from .neural_synthesizer import (
        NeuralProgramSynthesizer,
        MIDIDifferenceEncoder,
        DSLProgramDecoder,
        DSLGrammarConstraints,
        midi_to_pianoroll
    )
    from .program_synthesis_trainer import (
        ProgramSynthesisTrainer,
        SyntheticMIDIDataset,
        create_dataloaders,
        train_neural_synthesizer
    )
    NEURAL_SYNTHESIS_AVAILABLE = True
except ImportError:
    DSLProgram = None
    Statement = None
    ForEach = None
    If = None
    Operation = None
    Aggregate = None
    Value = None
    IteratorType = None
    FilterType = None
    OperationType = None
    AggregatorType = None
    DSLVocabulary = None
    create_example_programs = None
    SyntheticDatasetGenerator = None
    SyntheticExample = None
    TransformTemplateLibrary = None
    NeuralProgramSynthesizer = None
    MIDIDifferenceEncoder = None
    DSLProgramDecoder = None
    DSLGrammarConstraints = None
    midi_to_pianoroll = None
    ProgramSynthesisTrainer = None
    SyntheticMIDIDataset = None
    create_dataloaders = None
    train_neural_synthesizer = None
    NEURAL_SYNTHESIS_AVAILABLE = False

__all__ = [
    # Feature extraction
    'DeepFeatureExtractor',
    'FeatureVector',

    # Neural Program Synthesis - DSL
    'DSLProgram',
    'Statement',
    'ForEach',
    'If',
    'Operation',
    'Aggregate',
    'Value',
    'IteratorType',
    'FilterType',
    'OperationType',
    'AggregatorType',
    'DSLVocabulary',
    'create_example_programs',

    # Neural Program Synthesis - Dataset
    'SyntheticDatasetGenerator',
    'SyntheticExample',
    'TransformTemplateLibrary',

    # Neural Program Synthesis - Neural Architecture
    'NeuralProgramSynthesizer',
    'MIDIDifferenceEncoder',
    'DSLProgramDecoder',
    'DSLGrammarConstraints',
    'midi_to_pianoroll',

    # Neural Program Synthesis - Training
    'ProgramSynthesisTrainer',
    'SyntheticMIDIDataset',
    'create_dataloaders',
    'train_neural_synthesizer',

    'NEURAL_SYNTHESIS_AVAILABLE',
]
