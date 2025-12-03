"""
Grammar-based pattern discovery using SEQUITUR algorithm.

This module provides O(n) grammar induction with MDL pruning for
discovering hierarchical patterns in musical data.

Components:
- SequiturGrammar: O(n) grammar induction with digram uniqueness
- VocabularyOptimizer: MDL-based rule selection
- GPUTokenizer: GPU-accelerated preprocessing

Usage:
    from grammar import run_grammar_discovery_pipeline

    # Given list of FactoredObjects
    grammar, selected_rules, patterns = run_grammar_discovery_pipeline(
        objects, target_vocab_size=400
    )
"""

from .sequitur import (
    SequiturGrammar,
    Rule,
    Symbol,
    serialize_factored_object,
    serialize_track,
    serialize_corpus,
    build_grammar_from_corpus
)

from .mdl_optimizer import (
    VocabularyOptimizer,
    MDLScore,
    VocabularyStats,
    PatternCandidate,
    extract_patterns_from_grammar,
    run_grammar_discovery_pipeline,
    integrate_grammar_patterns_with_transforms
)

from .gpu_tokenizer import (
    GPUTokenizer,
    TokenizationConfig,
    BatchPitchExtractor,
    tokenize_factored_objects_gpu
)

__all__ = [
    # Core SEQUITUR
    'SequiturGrammar',
    'Rule',
    'Symbol',
    'serialize_factored_object',
    'serialize_track',
    'serialize_corpus',
    'build_grammar_from_corpus',

    # MDL Optimizer
    'VocabularyOptimizer',
    'MDLScore',
    'VocabularyStats',
    'PatternCandidate',
    'extract_patterns_from_grammar',
    'run_grammar_discovery_pipeline',
    'integrate_grammar_patterns_with_transforms',

    # GPU Tokenizer
    'GPUTokenizer',
    'TokenizationConfig',
    'BatchPitchExtractor',
    'tokenize_factored_objects_gpu'
]
