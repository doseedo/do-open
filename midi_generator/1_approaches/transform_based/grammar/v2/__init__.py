"""
Grammar V2: GPU-Optimized Grammar Induction for Musical Pattern Discovery
=========================================================================

This module contains improved grammar induction algorithms optimized for
A100 40GB GPU. Each algorithm addresses specific limitations of the
original SEQUITUR-based approach.

Modules:
--------
1. repair.py - RePair algorithm (replaces most frequent pair globally)
2. min_length_filter.py - Minimum pattern length constraint
3. longestfirst.py - LONGESTFIRST variant for phrase detection
4. boundary_detection.py - Self-similarity and Markov surprise boundaries
5. adaptor_grammar.py - Pitman-Yor prior for productive unit learning
6. hdp_hsmm.py - HDP-HSMM segmentation with explicit duration modeling

Quick Start:
------------
```python
from grammar.v2 import (
    build_repair_grammar,
    filter_grammar_by_length,
    build_longestfirst_grammar,
    detect_boundaries,
    build_adaptor_grammar,
    fit_hdp_hsmm,
)

# Build RePair grammar
grammar = build_repair_grammar(sequences, device='cuda')

# Apply minimum length filter
filtered = filter_grammar_by_length(grammar, min_length=4)

# Or use LONGESTFIRST directly
lf_grammar = build_longestfirst_grammar(
    pitch_sequences,
    duration_sequences,
    include_duration=True,
)

# Detect phrase boundaries
ssm_boundaries = detect_boundaries(sequence, method='ssm')
markov_boundaries = detect_boundaries(sequence, method='markov')

# Learn productive units with Adaptor Grammar
ag = build_adaptor_grammar(sequences, n_iterations=100)

# Segment with HDP-HSMM
result = fit_hdp_hsmm(sequences, n_iterations=100)
```

GPU Memory Requirements:
------------------------
- All algorithms optimized for A100 40GB
- For smaller GPUs, reduce batch sizes in individual modules
- CPU fallback available for all algorithms

References:
-----------
- RePair: Larsson & Moffat (1999)
- LONGESTFIRST: Pearce & Wiggins (2004)
- Adaptor Grammar: Johnson et al. (2007)
- HDP-HSMM: Van Gael et al. (2008)
"""

# Import main functions for convenient access
from .repair import (
    build_repair_grammar,
    build_repair_from_corpus,
    RePairGrammar,
    RePairRule,
)

# True GPU Re-Pair v2 (tensor-based, no Python dicts)
from .repair_gpu_v2 import (
    build_repair_grammar_v2,
    RePairGrammarV2,
    RePairGPUv2,
)

from .min_length_filter import (
    filter_grammar_by_length,
    FilteredGrammar,
    FilteredRule,
)

from .longestfirst import (
    build_longestfirst_grammar,
    build_longestfirst_from_corpus,
    LFGrammar,
    LFRule,
)

from .boundary_detection import (
    detect_boundaries,
    segment_sequences,
    BoundaryResult,
    Boundary,
)

from .adaptor_grammar import (
    build_adaptor_grammar,
    build_adaptor_grammar_from_corpus,
    AdaptorGrammar,
    AdaptedRule,
)

from .hdp_hsmm import (
    fit_hdp_hsmm,
    fit_hdp_hsmm_from_corpus,
    HDPHSMMResult,
    HSMMState,
)

__all__ = [
    # RePair
    'build_repair_grammar',
    'build_repair_from_corpus',
    'RePairGrammar',
    'RePairRule',
    # GPU Re-Pair v2 (tensor-based)
    'build_repair_grammar_v2',
    'RePairGrammarV2',
    'RePairGPUv2',
    # Min length filter
    'filter_grammar_by_length',
    'FilteredGrammar',
    'FilteredRule',
    # LONGESTFIRST
    'build_longestfirst_grammar',
    'build_longestfirst_from_corpus',
    'LFGrammar',
    'LFRule',
    # Boundary detection
    'detect_boundaries',
    'segment_sequences',
    'BoundaryResult',
    'Boundary',
    # Adaptor Grammar
    'build_adaptor_grammar',
    'build_adaptor_grammar_from_corpus',
    'AdaptorGrammar',
    'AdaptedRule',
    # HDP-HSMM
    'fit_hdp_hsmm',
    'fit_hdp_hsmm_from_corpus',
    'HDPHSMMResult',
    'HSMMState',
]
