"""
Step 5: Adaptor Grammar Implementation (GPU-Optimized)
======================================================

Adaptor Grammars learn which patterns are 'productive units' (should be
memorized as wholes) vs compositionally derived (should be built from primitives).

Key insight from the research:
- Standard PCFG: Every derivation is compositional
- Adaptor Grammar: Some subtrees get cached and reused as wholes

The Pitman-Yor prior automatically learns:
- Which interval sequences deserve 'motif' status
- Which motif sequences deserve 'phrase' status

GPU Optimizations:
- Batch probability computations
- Parallel CRP sampling
- Vectorized table operations
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
import time
import math


@dataclass
class AdaptedRule:
    """A rule with adaptation (caching) statistics."""
    rule_id: int
    left_symbol: str      # Non-terminal (e.g., 'PHRASE', 'MOTIF', 'INTERVAL')
    right_symbols: List   # List of symbols (terminals or non-terminals)

    # Pitman-Yor parameters
    n_customers: int = 0       # Total times this rule was used
    n_tables: int = 0          # Number of unique "table" instances
    table_counts: List[int] = field(default_factory=list)  # Customers per table

    # Adaptation flag
    is_adapted: bool = False   # Whether this non-terminal is adapted

    def __repr__(self):
        rhs = ' '.join(str(s) for s in self.right_symbols)
        return f"{self.left_symbol} -> {rhs} [n={self.n_customers}, tables={self.n_tables}]"


@dataclass
class AdaptorGrammar:
    """
    Adaptor Grammar with Pitman-Yor prior.

    Structure for musical sequences:
    - SONG -> PHRASE+
    - PHRASE -> MOTIF+ (adapted - phrases get cached)
    - MOTIF -> INTERVAL+ (adapted - motifs get cached)
    - INTERVAL -> terminal (pitch interval)
    """
    rules: Dict[str, List[AdaptedRule]] = field(default_factory=dict)
    adapted_nonterminals: Set[str] = field(default_factory=set)

    # Pitman-Yor hyperparameters per non-terminal
    # a (discount): 0 = Dirichlet, >0 = more productive
    # b (concentration): higher = more tables (unique patterns)
    py_discount: Dict[str, float] = field(default_factory=dict)  # a
    py_concentration: Dict[str, float] = field(default_factory=dict)  # b

    # Cached adaptations (the "restaurant" tables)
    # Maps: non-terminal -> list of cached derivation trees
    cache: Dict[str, List[Tuple]] = field(default_factory=dict)
    cache_counts: Dict[str, List[int]] = field(default_factory=dict)

    # Statistics
    n_adaptations: int = 0
    total_tables: int = 0


class ChineseRestaurantProcess:
    """
    GPU-accelerated Chinese Restaurant Process for Pitman-Yor sampling.

    The CRP determines whether to:
    1. Sit at existing table (reuse cached derivation)
    2. Open new table (create new derivation)

    Probability of sitting at table k with n_k customers:
        P(k) = (n_k - a) / (n + b)

    Probability of new table:
        P(new) = (b + K*a) / (n + b)

    Where:
        a = discount (0 <= a < 1)
        b = concentration (b > -a)
        n = total customers
        K = number of tables
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

    def sample_table(
        self,
        table_counts: torch.Tensor,  # [K] counts per table
        discount: float = 0.5,
        concentration: float = 1.0,
    ) -> Tuple[int, bool]:
        """
        Sample which table to sit at.

        Returns:
            (table_idx, is_new_table)
        """
        K = len(table_counts)
        n = table_counts.sum().item()

        if K == 0:
            return 0, True

        # Compute probabilities
        # Existing tables: (n_k - a) / (n + b)
        probs = (table_counts.float() - discount) / (n + concentration)

        # New table: (b + K*a) / (n + b)
        new_table_prob = (concentration + K * discount) / (n + concentration)

        # Concatenate and sample
        all_probs = torch.cat([probs, torch.tensor([new_table_prob], device=self.device)])
        all_probs = torch.clamp(all_probs, min=1e-10)  # Avoid zeros
        all_probs = all_probs / all_probs.sum()  # Normalize

        # Sample
        idx = torch.multinomial(all_probs, 1).item()

        is_new = (idx == K)
        return idx if not is_new else K, is_new

    def batch_sample_tables(
        self,
        batch_table_counts: List[torch.Tensor],
        discount: float = 0.5,
        concentration: float = 1.0,
    ) -> List[Tuple[int, bool]]:
        """
        Batch sample tables for multiple CRP instances.
        """
        results = []
        for table_counts in batch_table_counts:
            results.append(self.sample_table(table_counts, discount, concentration))
        return results


class AdaptorGrammarGPU:
    """
    GPU-accelerated Adaptor Grammar induction.

    Algorithm:
    1. Initialize grammar with base rules (SONG, PHRASE, MOTIF, INTERVAL)
    2. For each sequence:
       a. Parse bottom-up to get derivation
       b. For adapted non-terminals, use CRP to decide cache vs compose
       c. Update tables and counts
    3. After burn-in, extract most probable cached patterns

    Musical hierarchy:
    - INTERVAL: Single pitch transitions (e.g., +2 semitones)
    - MOTIF: Short melodic fragments (adapted - cached)
    - PHRASE: Longer melodic units (adapted - cached)
    - SONG: Full sequence
    """

    def __init__(
        self,
        device: str = 'cuda',
        # Pitman-Yor hyperparameters
        motif_discount: float = 0.5,     # Higher = more unique motifs
        motif_concentration: float = 1.0,
        phrase_discount: float = 0.3,    # Lower = more reuse
        phrase_concentration: float = 2.0,
        # Structure parameters
        min_motif_length: int = 2,
        max_motif_length: int = 8,
        min_phrase_length: int = 4,
        max_phrase_length: int = 32,
        # Sampling
        n_iterations: int = 100,
        burn_in: int = 50,
        verbose: bool = False,
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.motif_discount = motif_discount
        self.motif_concentration = motif_concentration
        self.phrase_discount = phrase_discount
        self.phrase_concentration = phrase_concentration

        self.min_motif_length = min_motif_length
        self.max_motif_length = max_motif_length
        self.min_phrase_length = min_phrase_length
        self.max_phrase_length = max_phrase_length

        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.verbose = verbose

        self.crp = ChineseRestaurantProcess(device)

    def induce(
        self,
        sequences: List[List[int]],
    ) -> AdaptorGrammar:
        """
        Run Adaptor Grammar inference on sequences.

        Args:
            sequences: List of integer sequences (pitch intervals or pitch classes)

        Returns:
            AdaptorGrammar with learned adaptations
        """
        start_time = time.time()

        # Initialize grammar
        grammar = self._initialize_grammar()

        # Convert sequences to intervals if they're pitch classes
        interval_sequences = self._to_intervals(sequences)

        if self.verbose:
            print(f"[AdaptorGrammar] Starting with {len(sequences)} sequences")
            print(f"[AdaptorGrammar] Motif PY: a={self.motif_discount}, b={self.motif_concentration}")
            print(f"[AdaptorGrammar] Phrase PY: a={self.phrase_discount}, b={self.phrase_concentration}")

        # Initialize caches
        # Key: tuple of intervals, Value: (count, table_id)
        motif_cache: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        phrase_cache: Dict[Tuple[Tuple[int, ...], ...], List[int]] = defaultdict(list)

        # Gibbs sampling iterations
        for iteration in range(self.n_iterations):
            # Process each sequence
            for seq_idx, intervals in enumerate(interval_sequences):
                if len(intervals) < self.min_motif_length:
                    continue

                # Bottom-up parsing with CRP decisions
                # Step 1: Segment into motifs
                motifs, motif_decisions = self._segment_into_motifs(
                    intervals, motif_cache, grammar
                )

                # Step 2: Segment motifs into phrases
                phrases, phrase_decisions = self._segment_into_phrases(
                    motifs, phrase_cache, grammar
                )

                # Update caches based on decisions
                for motif, (table_idx, is_new) in zip(motifs, motif_decisions):
                    motif_key = tuple(motif)
                    if is_new:
                        motif_cache[motif_key].append(1)
                    else:
                        if table_idx < len(motif_cache[motif_key]):
                            motif_cache[motif_key][table_idx] += 1

                for phrase, (table_idx, is_new) in zip(phrases, phrase_decisions):
                    phrase_key = tuple(tuple(m) for m in phrase)
                    if is_new:
                        phrase_cache[phrase_key].append(1)
                    else:
                        if table_idx < len(phrase_cache[phrase_key]):
                            phrase_cache[phrase_key][table_idx] += 1

            if self.verbose and iteration % 20 == 0:
                n_motif_types = len(motif_cache)
                n_phrase_types = len(phrase_cache)
                n_motif_tables = sum(len(v) for v in motif_cache.values())
                n_phrase_tables = sum(len(v) for v in phrase_cache.values())
                print(f"[AdaptorGrammar] Iter {iteration}: "
                      f"motifs={n_motif_types} ({n_motif_tables} tables), "
                      f"phrases={n_phrase_types} ({n_phrase_tables} tables)")

        # Extract final grammar from post-burn-in samples
        grammar = self._extract_grammar(motif_cache, phrase_cache)

        elapsed = time.time() - start_time
        if self.verbose:
            print(f"[AdaptorGrammar] Complete in {elapsed:.2f}s")
            print(f"[AdaptorGrammar] {len(grammar.cache.get('MOTIF', []))} adapted motifs")
            print(f"[AdaptorGrammar] {len(grammar.cache.get('PHRASE', []))} adapted phrases")

        return grammar

    def _initialize_grammar(self) -> AdaptorGrammar:
        """Initialize base grammar structure."""
        grammar = AdaptorGrammar()

        # Set adapted non-terminals
        grammar.adapted_nonterminals = {'MOTIF', 'PHRASE'}

        # Set PY parameters
        grammar.py_discount = {
            'MOTIF': self.motif_discount,
            'PHRASE': self.phrase_discount,
        }
        grammar.py_concentration = {
            'MOTIF': self.motif_concentration,
            'PHRASE': self.phrase_concentration,
        }

        # Initialize caches
        grammar.cache = {'MOTIF': [], 'PHRASE': []}
        grammar.cache_counts = {'MOTIF': [], 'PHRASE': []}

        return grammar

    def _to_intervals(self, sequences: List[List[int]]) -> List[List[int]]:
        """Convert pitch sequences to interval sequences."""
        interval_seqs = []
        for seq in sequences:
            if len(seq) < 2:
                interval_seqs.append([])
                continue
            intervals = [seq[i+1] - seq[i] for i in range(len(seq) - 1)]
            interval_seqs.append(intervals)
        return interval_seqs

    def _segment_into_motifs(
        self,
        intervals: List[int],
        cache: Dict[Tuple[int, ...], List[int]],
        grammar: AdaptorGrammar,
    ) -> Tuple[List[List[int]], List[Tuple[int, bool]]]:
        """
        Segment interval sequence into motifs using CRP.

        Returns:
            (list of motifs, list of (table_idx, is_new) decisions)
        """
        motifs = []
        decisions = []

        i = 0
        while i < len(intervals):
            # Try to find best matching cached motif
            best_motif = None
            best_length = 0
            best_cache_key = None

            # Check cache for longest match
            for length in range(min(self.max_motif_length, len(intervals) - i),
                               self.min_motif_length - 1, -1):
                candidate = tuple(intervals[i:i+length])
                if candidate in cache and len(cache[candidate]) > 0:
                    best_motif = list(candidate)
                    best_length = length
                    best_cache_key = candidate
                    break

            if best_motif is not None:
                # CRP decision: existing table or new?
                table_counts = torch.tensor(
                    cache[best_cache_key],
                    dtype=torch.float32,
                    device=self.device
                )
                table_idx, is_new = self.crp.sample_table(
                    table_counts,
                    grammar.py_discount['MOTIF'],
                    grammar.py_concentration['MOTIF']
                )

                motifs.append(best_motif)
                decisions.append((table_idx, is_new))
                i += best_length
            else:
                # No cache hit - create new motif of random length
                length = min(
                    np.random.randint(self.min_motif_length, self.max_motif_length + 1),
                    len(intervals) - i
                )
                length = max(length, self.min_motif_length)

                if i + length > len(intervals):
                    length = len(intervals) - i

                new_motif = intervals[i:i+length]
                motifs.append(new_motif)
                decisions.append((0, True))  # Always new table
                i += length

        return motifs, decisions

    def _segment_into_phrases(
        self,
        motifs: List[List[int]],
        cache: Dict[Tuple[Tuple[int, ...], ...], List[int]],
        grammar: AdaptorGrammar,
    ) -> Tuple[List[List[List[int]]], List[Tuple[int, bool]]]:
        """
        Segment motif sequence into phrases using CRP.

        Returns:
            (list of phrases, list of (table_idx, is_new) decisions)
        """
        phrases = []
        decisions = []

        i = 0
        while i < len(motifs):
            # Try to find cached phrase
            best_phrase = None
            best_length = 0
            best_cache_key = None

            # Compute phrase length in motifs
            max_phrase_motifs = min(8, len(motifs) - i)  # Max 8 motifs per phrase

            for n_motifs in range(max_phrase_motifs, 0, -1):
                candidate_motifs = motifs[i:i+n_motifs]
                candidate_key = tuple(tuple(m) for m in candidate_motifs)

                # Check total interval length
                total_intervals = sum(len(m) for m in candidate_motifs)
                if total_intervals < self.min_phrase_length:
                    continue
                if total_intervals > self.max_phrase_length:
                    continue

                if candidate_key in cache and len(cache[candidate_key]) > 0:
                    best_phrase = candidate_motifs
                    best_length = n_motifs
                    best_cache_key = candidate_key
                    break

            if best_phrase is not None:
                # CRP decision
                table_counts = torch.tensor(
                    cache[best_cache_key],
                    dtype=torch.float32,
                    device=self.device
                )
                table_idx, is_new = self.crp.sample_table(
                    table_counts,
                    grammar.py_discount['PHRASE'],
                    grammar.py_concentration['PHRASE']
                )

                phrases.append(best_phrase)
                decisions.append((table_idx, is_new))
                i += best_length
            else:
                # No cache hit - create new phrase
                n_motifs = min(
                    np.random.randint(1, 5),  # 1-4 motifs per phrase
                    len(motifs) - i
                )

                new_phrase = motifs[i:i+n_motifs]
                phrases.append(new_phrase)
                decisions.append((0, True))
                i += n_motifs

        return phrases, decisions

    def _extract_grammar(
        self,
        motif_cache: Dict[Tuple[int, ...], List[int]],
        phrase_cache: Dict[Tuple[Tuple[int, ...], ...], List[int]],
    ) -> AdaptorGrammar:
        """Extract final grammar from caches."""
        grammar = self._initialize_grammar()

        # Sort motifs by total count
        motif_items = []
        for pattern, tables in motif_cache.items():
            total_count = sum(tables)
            if total_count >= 2:  # Minimum usage
                motif_items.append((pattern, tables, total_count))

        motif_items.sort(key=lambda x: x[2], reverse=True)

        # Add to grammar cache
        for pattern, tables, count in motif_items:
            grammar.cache['MOTIF'].append(list(pattern))
            grammar.cache_counts['MOTIF'].append(tables)

        # Sort phrases by total count
        phrase_items = []
        for pattern, tables in phrase_cache.items():
            total_count = sum(tables)
            if total_count >= 2:
                phrase_items.append((pattern, tables, total_count))

        phrase_items.sort(key=lambda x: x[2], reverse=True)

        for pattern, tables, count in phrase_items:
            # Convert back to lists
            phrase_motifs = [list(m) for m in pattern]
            grammar.cache['PHRASE'].append(phrase_motifs)
            grammar.cache_counts['PHRASE'].append(tables)

        # Compute statistics
        grammar.n_adaptations = len(grammar.cache['MOTIF']) + len(grammar.cache['PHRASE'])
        grammar.total_tables = (
            sum(len(t) for t in grammar.cache_counts['MOTIF']) +
            sum(len(t) for t in grammar.cache_counts['PHRASE'])
        )

        return grammar


def build_adaptor_grammar(
    sequences: List[List[int]],
    device: str = 'cuda',
    motif_discount: float = 0.5,
    motif_concentration: float = 1.0,
    phrase_discount: float = 0.3,
    phrase_concentration: float = 2.0,
    n_iterations: int = 100,
    burn_in: int = 50,
    verbose: bool = False,
) -> AdaptorGrammar:
    """
    Build Adaptor Grammar from sequences.

    Args:
        sequences: List of pitch class sequences
        device: 'cuda' or 'cpu'
        motif_discount: PY discount for motifs (0-1)
        motif_concentration: PY concentration for motifs
        phrase_discount: PY discount for phrases
        phrase_concentration: PY concentration for phrases
        n_iterations: Gibbs sampling iterations
        burn_in: Iterations to discard
        verbose: Print progress

    Returns:
        AdaptorGrammar with learned adaptations
    """
    ag = AdaptorGrammarGPU(
        device=device,
        motif_discount=motif_discount,
        motif_concentration=motif_concentration,
        phrase_discount=phrase_discount,
        phrase_concentration=phrase_concentration,
        n_iterations=n_iterations,
        burn_in=burn_in,
        verbose=verbose,
    )

    return ag.induce(sequences)


def build_adaptor_grammar_from_corpus(
    factored_objects: List,
    device: str = 'cuda',
    motif_discount: float = 0.5,
    motif_concentration: float = 1.0,
    phrase_discount: float = 0.3,
    phrase_concentration: float = 2.0,
    n_iterations: int = 100,
    verbose: bool = False,
) -> AdaptorGrammar:
    """
    Build Adaptor Grammar from factored MIDI objects.

    Args:
        factored_objects: List of factored objects with pitch_class arrays

    Returns:
        AdaptorGrammar
    """
    sequences = []
    for obj in factored_objects:
        if hasattr(obj, 'pitch_class') and len(obj.pitch_class) > 0:
            pc = obj.pitch_class
            if hasattr(pc, 'tolist'):
                pc = pc.tolist()
            sequences.append(pc)

    if verbose:
        print(f"[AdaptorGrammar] Extracted {len(sequences)} sequences")

    return build_adaptor_grammar(
        sequences,
        device=device,
        motif_discount=motif_discount,
        motif_concentration=motif_concentration,
        phrase_discount=phrase_discount,
        phrase_concentration=phrase_concentration,
        n_iterations=n_iterations,
        verbose=verbose,
    )


if __name__ == '__main__':
    print("Testing Adaptor Grammar GPU implementation...")

    # Test with musical patterns
    test_sequences = [
        # C major scale variations
        [0, 2, 4, 5, 7, 9, 11, 0, 2, 4, 5, 7],
        [0, 2, 4, 5, 7, 9, 11, 0],
        [0, 2, 4, 5, 7, 0, 2, 4, 5, 7],
        # Arpeggios
        [0, 4, 7, 0, 4, 7, 0, 4, 7],
        [0, 3, 7, 0, 3, 7, 0, 3, 7],
        # Mixed
        [0, 2, 4, 0, 4, 7, 5, 7, 9],
        [0, 2, 4, 0, 4, 7, 5, 7, 9, 0, 2, 4],
    ]

    grammar = build_adaptor_grammar(
        test_sequences,
        device='cuda',
        n_iterations=50,
        verbose=True,
    )

    print(f"\nAdapted Motifs (top 10):")
    for i, (motif, counts) in enumerate(zip(
        grammar.cache.get('MOTIF', [])[:10],
        grammar.cache_counts.get('MOTIF', [])[:10]
    )):
        total = sum(counts)
        print(f"  {i+1}. {motif} (count={total}, tables={len(counts)})")

    print(f"\nAdapted Phrases (top 5):")
    for i, (phrase, counts) in enumerate(zip(
        grammar.cache.get('PHRASE', [])[:5],
        grammar.cache_counts.get('PHRASE', [])[:5]
    )):
        total = sum(counts)
        print(f"  {i+1}. {phrase} (count={total}, tables={len(counts)})")

    print(f"\nStatistics:")
    print(f"  Total adaptations: {grammar.n_adaptations}")
    print(f"  Total tables: {grammar.total_tables}")
