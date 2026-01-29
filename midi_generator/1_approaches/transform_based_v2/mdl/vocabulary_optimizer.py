"""
MDL Vocabulary Optimizer
========================

Selects an optimal vocabulary using Minimum Description Length (MDL) principle.

MDL Score for a pattern:
    score = bits_saved - bits_to_describe

    bits_saved = usage_count × (pattern_length × bits_per_token - bits_for_rule_reference)
    bits_to_describe = pattern_length × bits_per_token + overhead

A pattern is beneficial if score > 0 (saves more bits than it costs).

The optimizer:
1. Scores all patterns from SEQUITUR grammar
2. Sorts by MDL score descending
3. Greedily selects until target vocabulary size
4. Verifies reconstruction accuracy

Author: Dosedo Architecture v2
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import math

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PatternScore:
    """MDL score for a single pattern."""
    pattern_id: int
    pattern_type: str  # 'rhythm', 'pitch_class', 'grammar', 'algebraic', 'cross_track'
    usage_count: int
    pattern_length: int
    bits_saved: float
    bits_to_describe: float
    mdl_score: float  # bits_saved - bits_to_describe
    depth: int = 1  # Hierarchical depth (1 = terminal-only)

    @property
    def is_beneficial(self) -> bool:
        return self.mdl_score > 0

    def __repr__(self):
        return f"PatternScore({self.pattern_type}:{self.pattern_id}, score={self.mdl_score:.1f}, usage={self.usage_count})"


@dataclass
class CorpusStats:
    """Statistics about the corpus for MDL calculation."""
    n_objects: int
    n_tokens: int
    n_unique_rhythm_patterns: int
    n_unique_pitch_patterns: int
    avg_pattern_length: float

    # Token frequency distribution
    rhythm_pattern_counts: Dict[int, int] = field(default_factory=dict)
    pitch_pattern_counts: Dict[int, int] = field(default_factory=dict)

    # Bits per token (log2 of vocabulary size)
    bits_per_rhythm_token: float = 10.0
    bits_per_pitch_token: float = 7.0  # log2(128) for MIDI pitches

    def __repr__(self):
        return f"CorpusStats(objects={self.n_objects}, tokens={self.n_tokens})"


@dataclass
class Vocabulary:
    """Final assembled vocabulary."""
    # Algebraic transforms
    d24_elements: List[int]  # Element indices 0-23
    rhythm_elements: List[int]  # Element indices 0-13

    # Grammar patterns (from SEQUITUR)
    grammar_rules: Dict[int, List]  # rule_id -> expansion

    # Cross-track relation types
    cross_track_types: List[int]  # CrossTrackRelationType indices

    # Combined vocabulary
    all_elements: List[Tuple[str, int]]  # (type, id) pairs

    @property
    def size(self) -> int:
        return len(self.all_elements)

    def __repr__(self):
        return (f"Vocabulary(size={self.size}, d24={len(self.d24_elements)}, "
                f"rhythm={len(self.rhythm_elements)}, grammar={len(self.grammar_rules)}, "
                f"cross_track={len(self.cross_track_types)})")


# =============================================================================
# MDL SCORING FUNCTIONS
# =============================================================================

def compute_bits_per_token(vocab_size: int) -> float:
    """Compute bits needed to encode one token from vocabulary."""
    if vocab_size <= 0:
        return 0.0
    return math.log2(vocab_size)


def compute_pattern_bits_saved(
    pattern_length: int,
    usage_count: int,
    bits_per_token: float,
    vocab_size: int
) -> float:
    """
    Compute bits saved by using a pattern.

    Without pattern: Each occurrence costs pattern_length × bits_per_token
    With pattern: Each occurrence costs log2(vocab_size + 1) (reference to rule)

    bits_saved = usage_count × (pattern_length × bits_per_token - reference_cost)
    """
    if usage_count <= 1:
        return 0.0

    # Cost to encode pattern inline
    inline_cost = pattern_length * bits_per_token

    # Cost to reference the pattern (new vocab item)
    reference_cost = compute_bits_per_token(vocab_size + 1)

    # Bits saved per usage
    savings_per_use = inline_cost - reference_cost

    # Total savings (only for usages beyond the first, which defines the pattern)
    return (usage_count - 1) * savings_per_use


def compute_pattern_bits_to_describe(
    pattern_length: int,
    bits_per_token: float,
    overhead: float = 8.0  # Bits for rule metadata
) -> float:
    """
    Compute bits needed to describe a pattern in the vocabulary.

    Cost = pattern_length × bits_per_token + overhead
    """
    return pattern_length * bits_per_token + overhead


def compute_mdl_score(
    pattern_length: int,
    usage_count: int,
    bits_per_token: float,
    vocab_size: int,
    overhead: float = 8.0
) -> float:
    """
    Compute MDL score for a pattern.

    score = bits_saved - bits_to_describe

    Positive score = pattern is beneficial
    Negative score = pattern costs more than it saves
    """
    bits_saved = compute_pattern_bits_saved(
        pattern_length, usage_count, bits_per_token, vocab_size
    )
    bits_to_describe = compute_pattern_bits_to_describe(
        pattern_length, bits_per_token, overhead
    )

    return bits_saved - bits_to_describe


# =============================================================================
# TRANSFORM-BASED DEDUPLICATION
# =============================================================================

def build_transform_equivalence_classes(
    d24_transform_table: np.ndarray,
    pattern_ids: List[int]
) -> Tuple[Dict[int, int], Dict[int, List[Tuple[int, int]]]]:
    """
    Build equivalence classes of patterns related by D24 transforms.

    If pattern_j = T_k(pattern_i), they belong to the same equivalence class.
    We keep the "canonical" pattern (lowest id) and record how to derive others.

    Args:
        d24_transform_table: NxN table where table[i,j] = transform_id that
                             maps pattern_i to pattern_j (-1 if none)
        pattern_ids: List of pattern IDs (corresponding to table indices)

    Returns:
        (canonical_map, derivations)
        - canonical_map: pattern_id -> canonical_pattern_id
        - derivations: canonical_id -> [(derived_id, transform_id), ...]
    """
    N = len(pattern_ids)
    if N == 0 or d24_transform_table is None or d24_transform_table.size == 0:
        # No deduplication possible
        return {pid: pid for pid in pattern_ids}, {}

    # Use Union-Find to build equivalence classes
    parent = list(range(N))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            # Keep smaller index as root (canonical)
            if px < py:
                parent[py] = px
            else:
                parent[px] = py

    # Union patterns that are related by transforms
    for i in range(N):
        for j in range(i + 1, N):
            if i < d24_transform_table.shape[0] and j < d24_transform_table.shape[1]:
                t_id = d24_transform_table[i, j]
                if t_id > 0:  # Non-identity transform exists
                    union(i, j)

    # Build canonical map and derivations
    canonical_map = {}
    derivations = defaultdict(list)

    for i, pid in enumerate(pattern_ids):
        root = find(i)
        canonical_pid = pattern_ids[root]
        canonical_map[pid] = canonical_pid

        if root != i:
            # This pattern is derived from canonical
            if root < d24_transform_table.shape[0] and i < d24_transform_table.shape[1]:
                t_id = d24_transform_table[root, i]
                if t_id > 0:
                    derivations[canonical_pid].append((pid, int(t_id)))

    return canonical_map, dict(derivations)


def deduplicate_patterns_via_transforms(
    pattern_scores: List['PatternScore'],
    d24_transform_table: np.ndarray = None,
    verbose: bool = False
) -> Tuple[List['PatternScore'], Dict[int, List[Tuple[int, int]]]]:
    """
    Deduplicate patterns using D24 transforms.

    Instead of keeping both Pattern_42 and Pattern_17 when Pattern_42 = T₇(Pattern_17),
    we keep only Pattern_17 (canonical) and record the derivation.

    Args:
        pattern_scores: List of scored patterns
        d24_transform_table: Transform table from discovery
        verbose: Print deduplication stats

    Returns:
        (deduplicated_scores, derivation_map)
        - deduplicated_scores: Patterns with duplicates removed
        - derivation_map: canonical_id -> [(derived_id, transform_id), ...]
    """
    if d24_transform_table is None or d24_transform_table.size == 0:
        return pattern_scores, {}

    # Get grammar pattern IDs (only grammar patterns use transform table)
    grammar_scores = [s for s in pattern_scores if s.pattern_type == 'grammar']
    other_scores = [s for s in pattern_scores if s.pattern_type != 'grammar']

    if not grammar_scores:
        return pattern_scores, {}

    # The transform table indices correspond to extracted grammar patterns
    # We need to map pattern_id to table index
    pattern_ids = [s.pattern_id for s in grammar_scores]

    # Build equivalence classes
    canonical_map, derivations = build_transform_equivalence_classes(
        d24_transform_table, pattern_ids
    )

    # Keep only canonical patterns
    seen_canonical = set()
    deduplicated = []

    for score in grammar_scores:
        canonical = canonical_map.get(score.pattern_id, score.pattern_id)
        if canonical not in seen_canonical:
            # This is the canonical representative
            seen_canonical.add(canonical)

            # If this pattern is a derived pattern, find its canonical score
            if canonical == score.pattern_id:
                deduplicated.append(score)
            else:
                # Find the canonical score
                for s in grammar_scores:
                    if s.pattern_id == canonical:
                        if canonical not in {d.pattern_id for d in deduplicated}:
                            deduplicated.append(s)
                        break

    if verbose:
        removed = len(grammar_scores) - len(deduplicated)
        n_classes = len(set(canonical_map.values()))
        print(f"  Transform deduplication: {len(grammar_scores)} -> {len(deduplicated)} "
              f"({removed} removed, {n_classes} equivalence classes)")
        if derivations:
            total_derivations = sum(len(v) for v in derivations.values())
            print(f"  Derivable patterns: {total_derivations} via transforms")

    # Combine with non-grammar scores
    result = other_scores + deduplicated
    result.sort(key=lambda s: s.mdl_score, reverse=True)

    return result, derivations


# =============================================================================
# VOCABULARY OPTIMIZER
# =============================================================================

class VocabularyOptimizer:
    """
    MDL-based vocabulary selection.

    Selects patterns that maximize compression (minimize total description length).

    Usage:
        optimizer = VocabularyOptimizer(target_size=500)

        # Score patterns from SEQUITUR grammar
        scores = optimizer.score_grammar_patterns(grammar, corpus_stats)

        # Get optimal vocabulary
        selected = optimizer.optimize(scores)
    """

    def __init__(self, target_size: int = 500):
        """
        Initialize optimizer.

        Args:
            target_size: Maximum vocabulary size to select
        """
        self.target_size = target_size

    def score_grammar_patterns(
        self,
        grammar,  # SequiturGrammar
        corpus_stats: CorpusStats
    ) -> List[PatternScore]:
        """
        Score all patterns from a SEQUITUR grammar.

        Args:
            grammar: SequiturGrammar with induced rules
            corpus_stats: Statistics about the corpus

        Returns:
            List of PatternScore, sorted by MDL score descending
        """
        scores = []

        # Get rule statistics from grammar
        rule_stats = grammar.get_rule_stats()

        # Current vocabulary size (for computing reference cost)
        vocab_size = len(rule_stats)

        for rule_id, stats in rule_stats.items():
            if rule_id == 0:  # Skip start rule
                continue

            # Get the rule expansion
            rule = grammar.rules.get(rule_id)
            if rule is None:
                continue

            expansion_length = stats['expansion_length']
            usage_count = stats['usage_count']
            depth = stats['depth']

            # Compute MDL score
            mdl_score = compute_mdl_score(
                pattern_length=expansion_length,
                usage_count=usage_count,
                bits_per_token=corpus_stats.bits_per_rhythm_token,
                vocab_size=vocab_size
            )

            scores.append(PatternScore(
                pattern_id=rule_id,
                pattern_type='grammar',
                usage_count=usage_count,
                pattern_length=expansion_length,
                bits_saved=compute_pattern_bits_saved(
                    expansion_length, usage_count,
                    corpus_stats.bits_per_rhythm_token, vocab_size
                ),
                bits_to_describe=compute_pattern_bits_to_describe(
                    expansion_length, corpus_stats.bits_per_rhythm_token
                ),
                mdl_score=mdl_score,
                depth=depth
            ))

        # Sort by MDL score descending
        scores.sort(key=lambda s: s.mdl_score, reverse=True)

        return scores

    def score_algebraic_elements(
        self,
        d24_usage: Dict[int, int],  # element -> usage count
        rhythm_usage: Dict[int, int],
        corpus_stats: CorpusStats
    ) -> List[PatternScore]:
        """
        Score algebraic transform elements.

        Algebraic elements have fixed "pattern length" of 1 (atomic operations).
        Their value comes from the compression they enable.
        """
        scores = []

        # D24 elements (always included - they're the basis)
        for elem_id, usage in d24_usage.items():
            # D24 elements are "free" - they enable compression without storage cost
            # Score based on usage (more used = more valuable)
            scores.append(PatternScore(
                pattern_id=elem_id,
                pattern_type='d24',
                usage_count=usage,
                pattern_length=1,
                bits_saved=usage * 4.0,  # Rough estimate: 4 bits per transposition
                bits_to_describe=0.0,  # Free (part of base vocabulary)
                mdl_score=usage * 4.0,
                depth=1
            ))

        # Rhythm elements
        for elem_id, usage in rhythm_usage.items():
            scores.append(PatternScore(
                pattern_id=elem_id,
                pattern_type='rhythm',
                usage_count=usage,
                pattern_length=1,
                bits_saved=usage * 3.0,
                bits_to_describe=0.0,
                mdl_score=usage * 3.0,
                depth=1
            ))

        return scores

    def score_cross_track_types(
        self,
        cross_track_usage: Dict[int, int],  # relation_type -> usage count
        corpus_stats: CorpusStats
    ) -> List[PatternScore]:
        """
        Score cross-track relation types.
        """
        scores = []

        for rel_type, usage in cross_track_usage.items():
            # Cross-track types are somewhat like patterns
            # They compress multi-track relationships
            scores.append(PatternScore(
                pattern_id=rel_type,
                pattern_type='cross_track',
                usage_count=usage,
                pattern_length=2,  # Binary relationship
                bits_saved=usage * 5.0,  # Compresses two track references
                bits_to_describe=4.0,  # Type identifier
                mdl_score=usage * 5.0 - 4.0,
                depth=1
            ))

        return scores

    def optimize(
        self,
        all_scores: List[PatternScore],
        min_score: float = 0.0
    ) -> List[PatternScore]:
        """
        Select optimal vocabulary using greedy MDL selection.

        Args:
            all_scores: All pattern scores (already sorted by MDL score)
            min_score: Minimum MDL score to include

        Returns:
            Selected patterns (up to target_size)
        """
        selected = []

        # Separate by type to ensure diversity
        by_type = defaultdict(list)
        for score in all_scores:
            if score.mdl_score >= min_score:
                by_type[score.pattern_type].append(score)

        # Always include all algebraic elements (they're "free")
        for elem in by_type.get('d24', []):
            selected.append(elem)
        for elem in by_type.get('rhythm', []):
            selected.append(elem)

        # Add cross-track types (usually small count)
        for elem in by_type.get('cross_track', []):
            if len(selected) < self.target_size:
                selected.append(elem)

        # Fill remaining with grammar patterns
        grammar_patterns = by_type.get('grammar', [])
        for pattern in grammar_patterns:
            if len(selected) >= self.target_size:
                break
            selected.append(pattern)

        return selected

    def verify_reconstruction(
        self,
        selected: List[PatternScore],
        test_sequences: List[List[int]],
        grammar  # SequiturGrammar
    ) -> float:
        """
        Verify reconstruction accuracy on held-out test sequences.

        Args:
            selected: Selected vocabulary patterns
            test_sequences: Test sequences for verification
            grammar: Grammar used for encoding/decoding

        Returns:
            Accuracy (0.0 to 1.0)
        """
        if not test_sequences:
            return 1.0

        # Get selected rule IDs
        selected_rules = {s.pattern_id for s in selected if s.pattern_type == 'grammar'}

        total_tokens = 0
        matched_tokens = 0

        for seq in test_sequences:
            # Encode with grammar
            encoded = grammar.encode(seq)

            # Decode back
            decoded = grammar.decode(encoded)

            # Compare
            total_tokens += len(seq)
            matched_tokens += sum(1 for a, b in zip(seq, decoded) if a == b)

        return matched_tokens / max(1, total_tokens)


# =============================================================================
# VOCABULARY ASSEMBLY
# =============================================================================

def assemble_final_vocabulary(
    d24_elements: List[int] = None,
    rhythm_elements: List[int] = None,
    grammar_patterns: Dict[int, List] = None,
    cross_track_types: List[int] = None
) -> Vocabulary:
    """
    Assemble final vocabulary from all components.

    Target: <500 elements
    - Algebraic: 24 (D24) + 14 (Rhythm) = 38
    - Grammar patterns: 300-400
    - Cross-track: ~20
    - Overhead: ~50

    Args:
        d24_elements: D24 group elements to include (default: all 24)
        rhythm_elements: Rhythm group elements to include (default: all 14)
        grammar_patterns: Grammar rules {rule_id: expansion}
        cross_track_types: Cross-track relation type indices

    Returns:
        Assembled Vocabulary
    """
    # Defaults
    if d24_elements is None:
        d24_elements = list(range(24))
    if rhythm_elements is None:
        rhythm_elements = list(range(14))
    if grammar_patterns is None:
        grammar_patterns = {}
    if cross_track_types is None:
        cross_track_types = list(range(20))

    # Build combined vocabulary
    all_elements = []

    # Add D24 elements
    for elem in d24_elements:
        all_elements.append(('d24', elem))

    # Add rhythm elements
    for elem in rhythm_elements:
        all_elements.append(('rhythm', elem))

    # Add grammar patterns
    for rule_id in grammar_patterns.keys():
        all_elements.append(('grammar', rule_id))

    # Add cross-track types
    for rel_type in cross_track_types:
        all_elements.append(('cross_track', rel_type))

    return Vocabulary(
        d24_elements=d24_elements,
        rhythm_elements=rhythm_elements,
        grammar_rules=grammar_patterns,
        cross_track_types=cross_track_types,
        all_elements=all_elements
    )


def compute_corpus_stats(objects: List) -> CorpusStats:
    """
    Compute corpus statistics for MDL calculation.

    Args:
        objects: List of FactoredObjectV2

    Returns:
        CorpusStats
    """
    rhythm_counts = defaultdict(int)
    pitch_counts = defaultdict(int)
    total_tokens = 0
    total_length = 0

    for obj in objects:
        if obj.num_notes == 0:
            continue

        # Count rhythm patterns
        rh = hash(obj.rhythm.tobytes())
        rhythm_counts[rh] += 1

        # Count pitch patterns
        ph = hash(obj.pitch_class.tobytes())
        pitch_counts[ph] += 1

        # Count tokens
        total_tokens += obj.num_notes * 3  # pitch_class, octave, duration per note
        total_length += len(obj.rhythm) + obj.num_notes * 3

    avg_length = total_length / max(1, len(objects))

    return CorpusStats(
        n_objects=len(objects),
        n_tokens=total_tokens,
        n_unique_rhythm_patterns=len(rhythm_counts),
        n_unique_pitch_patterns=len(pitch_counts),
        avg_pattern_length=avg_length,
        rhythm_pattern_counts=dict(rhythm_counts),
        pitch_pattern_counts=dict(pitch_counts),
        bits_per_rhythm_token=compute_bits_per_token(len(rhythm_counts)),
        bits_per_pitch_token=compute_bits_per_token(len(pitch_counts))
    )


def run_vocabulary_optimization(
    grammar,  # SequiturGrammar
    objects: List,
    target_size: int = 500,
    d24_transform_table: np.ndarray = None,
    verbose: bool = True
) -> Tuple[Vocabulary, List[PatternScore]]:
    """
    Run full vocabulary optimization pipeline.

    Args:
        grammar: SequiturGrammar with induced rules
        objects: List of FactoredObjectV2
        target_size: Maximum vocabulary size
        d24_transform_table: Optional D24 transform table for deduplication
        verbose: Print progress

    Returns:
        (Vocabulary, selected_scores)
    """
    if verbose:
        print(f"\n{'='*70}")
        print("MDL VOCABULARY OPTIMIZATION")
        print(f"{'='*70}")

    # Compute corpus stats
    stats = compute_corpus_stats(objects)
    if verbose:
        print(f"  Corpus: {stats.n_objects} objects, {stats.n_tokens} tokens")
        print(f"  Unique patterns: {stats.n_unique_rhythm_patterns} rhythm, "
              f"{stats.n_unique_pitch_patterns} pitch")

    # Initialize optimizer
    optimizer = VocabularyOptimizer(target_size=target_size)

    # Score grammar patterns
    grammar_scores = optimizer.score_grammar_patterns(grammar, stats)
    if verbose:
        print(f"  Scored {len(grammar_scores)} grammar patterns")
        if grammar_scores:
            top_3 = grammar_scores[:3]
            print(f"  Top 3: {[f'{s.pattern_id}:{s.mdl_score:.1f}' for s in top_3]}")

    # Score algebraic elements (with dummy usage for now)
    d24_usage = {i: 100 for i in range(24)}  # Assume uniform usage
    rhythm_usage = {i: 50 for i in range(14)}
    algebraic_scores = optimizer.score_algebraic_elements(d24_usage, rhythm_usage, stats)

    # Score cross-track types (with dummy usage)
    cross_track_usage = {i: 20 for i in range(20)}
    cross_track_scores = optimizer.score_cross_track_types(cross_track_usage, stats)

    # Combine all scores
    all_scores = grammar_scores + algebraic_scores + cross_track_scores
    all_scores.sort(key=lambda s: s.mdl_score, reverse=True)

    # === KEY: Deduplicate patterns via D24 transforms ===
    # If Pattern_A = T_k(Pattern_B), keep only one + transform
    derivation_map = {}
    if d24_transform_table is not None and d24_transform_table.size > 0:
        all_scores, derivation_map = deduplicate_patterns_via_transforms(
            all_scores, d24_transform_table, verbose=verbose
        )

    # Select optimal vocabulary
    selected = optimizer.optimize(all_scores)
    if verbose:
        print(f"  Selected {len(selected)} patterns")

        # Count by type
        by_type = defaultdict(int)
        for s in selected:
            by_type[s.pattern_type] += 1
        print(f"  By type: {dict(by_type)}")

        # Total MDL score
        total_score = sum(s.mdl_score for s in selected)
        print(f"  Total MDL score: {total_score:.1f} bits")

    # Assemble vocabulary
    grammar_rules = {
        s.pattern_id: grammar.rules[s.pattern_id].to_list()
        for s in selected if s.pattern_type == 'grammar' and s.pattern_id in grammar.rules
    }

    vocabulary = assemble_final_vocabulary(
        d24_elements=[s.pattern_id for s in selected if s.pattern_type == 'd24'],
        rhythm_elements=[s.pattern_id for s in selected if s.pattern_type == 'rhythm'],
        grammar_patterns=grammar_rules,
        cross_track_types=[s.pattern_id for s in selected if s.pattern_type == 'cross_track']
    )

    if verbose:
        print(f"\n  Final vocabulary: {vocabulary}")

    return vocabulary, selected
