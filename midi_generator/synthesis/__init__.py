"""
Synthesis Module
================

This module contains inverse synthesis and analysis tools for Musical Program Synthesis.

Modules:
- deep_feature_extractor: Extract 1000+ musical features from MIDI files (Agent 8) ✅
- inverse_analyzer: Inverse MIDI analysis (coming soon)
- gap_detector: Intelligent gap detection (Agent 10) ✅
- transform_dsl: Domain-Specific Language for MIDI transforms (Agent 8) ✅
- synthetic_dataset: Generate training data for neural program synthesis (Agent 8) ✅

Neural Program Synthesis (Agent 8):
- DSL with constrained grammar for musical transformations
- 100 transform templates across 6 categories
- Synthetic dataset generator for 10,000 training examples
- Foundation for learning transforms from MIDI examples

Author: Musical Program Synthesis Team
License: MIT
"""

from .deep_feature_extractor import DeepFeatureExtractor, extract_features

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
    NEURAL_SYNTHESIS_AVAILABLE = False

__all__ = [
    # Feature extraction
    'DeepFeatureExtractor',
    'extract_features',

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
    'NEURAL_SYNTHESIS_AVAILABLE',
]
