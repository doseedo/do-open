"""
MDL Vocabulary Optimization
===========================

Minimum Description Length (MDL) based vocabulary selection and optimization.

The MDL principle: The best vocabulary is one that minimizes:
    total_bits = bits_to_describe_vocabulary + bits_to_encode_corpus_using_vocabulary

Components:
- VocabularyOptimizer: Select top-k patterns by MDL score
- PatternScorer: Compute bits saved by each pattern
- VocabularyAssembler: Combine algebraic transforms, grammar patterns, and cross-track types

Target: <500 total elements
- Algebraic: 24 (D24) + 14 (Rhythm) = ~38
- Grammar patterns: 300-400 (from SEQUITUR)
- Cross-track: ~20
- Overhead: ~50

Author: Dosedo Architecture v2
"""

from .vocabulary_optimizer import (
    VocabularyOptimizer,
    PatternScore,
    compute_mdl_score,
    assemble_final_vocabulary
)

__all__ = [
    'VocabularyOptimizer',
    'PatternScore',
    'compute_mdl_score',
    'assemble_final_vocabulary'
]
