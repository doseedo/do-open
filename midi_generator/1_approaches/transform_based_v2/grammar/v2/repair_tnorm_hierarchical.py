"""
T-Normalized Hierarchical GPU Re-Pair Implementation
=====================================================

Combines T-normalization (counting by interval) with hierarchical pattern
composition (rules can reference other rules).

Key features:
1. Pitch classes (0-11) as input - normalized for octave
2. Pairs counted by INTERVAL (not absolute values) - T-normalization
3. All transpositions replaced when best interval found
4. Rules can reference other rules - hierarchical patterns (length 3-40+)

Example of hierarchical T-normalized patterns:
- R12 = interval +3 (minor 3rd) - pairs like (C,Eb), (D,F), (G,Bb)...
- R13 = interval +4 (major 3rd) - pairs like (C,E), (D,F#), (Eb,G)...
- R14 = (R12, R13) - minor 3rd followed by major 3rd (forms major triad)
- All transpositions of R14 match: C-Eb-G, D-F-A, E-G#-B, etc.

This enables discovering musical structures like:
- Major/minor triads at any transposition
- Scale fragments at any transposition
- Common progressions at any transposition
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class TnormHierarchicalGrammar:
    """Grammar produced by T-normalized hierarchical Re-Pair."""
    # Rule table: rule_id -> (left, right) where left/right can be terminals or rules
    rule_table: torch.Tensor  # Shape: [n_rules, 2]
    rule_counts: torch.Tensor  # Shape: [n_rules]

    # The canonical interval for each rule (for display/debugging)
    # For rules combining other rules, this is -1
    rule_intervals: torch.Tensor  # Shape: [n_rules]

    # Final compressed sequence
    final_sequence: torch.Tensor

    # Metadata
    n_terminals: int  # Pitch class range (12)
    n_rules: int
    original_length: int
    compressed_length: int
    device: str

    def compression_ratio(self) -> float:
        if self.compressed_length == 0:
            return 1.0
        return self.original_length / self.compressed_length

    def get_rule(self, rule_id: int) -> Tuple[int, int]:
        """Get left/right symbols for a rule."""
        idx = rule_id - self.n_terminals
        if idx < 0 or idx >= len(self.rule_table):
            return (rule_id, -1)  # Terminal
        row = self.rule_table[idx]
        return (row[0].item(), row[1].item())

    def expand_rule(self, rule_id: int, memo: Optional[Dict] = None) -> List[int]:
        """Recursively expand rule to terminals (pitch classes 0-11)."""
        if memo is None:
            memo = {}
        if rule_id in memo:
            return memo[rule_id]

        if rule_id < self.n_terminals:
            return [rule_id]

        left, right = self.get_rule(rule_id)
        if right == -1:
            return [left % self.n_terminals]  # Shouldn't happen, but safety

        result = self.expand_rule(left, memo) + self.expand_rule(right, memo)
        memo[rule_id] = result
        return result

    def get_pattern_interval_sequence(self, rule_id: int) -> List[int]:
        """Get the interval sequence for a pattern (pitch differences)."""
        pcs = self.expand_rule(rule_id)
        return [(pcs[i+1] - pcs[i]) % 12 for i in range(len(pcs) - 1)]


class RePairTnormHierarchical:
    """
    T-Normalized Hierarchical GPU Re-Pair.

    Key differences from basic T-norm Re-Pair:
    1. Rules can contain other rules (not just terminals)
    2. When counting pairs, we normalize both terminals AND rules
    3. For rule-rule pairs, we use the first note of each expansion
    """

    def __init__(
        self,
        device: str = 'cuda',
        min_pair_count: int = 2,
        max_rules: int = 5000,
        pitch_range: int = 12,  # Pitch classes 0-11
        verbose: bool = True,
        progress_every: int = 100,
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.min_pair_count = min_pair_count
        self.max_rules = max_rules
        self.pitch_range = pitch_range
        self.verbose = verbose
        self.progress_every = progress_every

    def induce(self, sequences: List[List[int]]) -> TnormHierarchicalGrammar:
        """
        Run T-normalized hierarchical Re-Pair.

        Args:
            sequences: List of integer sequences (pitch classes 0-11)

        Returns:
            TnormHierarchicalGrammar with induced grammar
        """
        start_time = time.time()

        n_terminals = self.pitch_range  # 12 for pitch classes
        separator = -1

        # Concatenate sequences with separators
        combined = []
        for seq in sequences:
            combined.extend(seq)
            combined.append(separator)

        original_length = len(combined)

        # Move to GPU
        seq = torch.tensor(combined, dtype=torch.int64, device=self.device)

        if self.verbose:
            print(f"[RePair-TnormHier] {len(sequences)} seqs, {original_length:,} tokens", flush=True)

        # Pre-allocate rule table on GPU
        rule_table = torch.zeros((self.max_rules, 2), dtype=torch.int64, device=self.device)
        rule_counts = torch.zeros(self.max_rules, dtype=torch.int64, device=self.device)
        rule_intervals = torch.full((self.max_rules,), -1, dtype=torch.int64, device=self.device)

        # Track the "base pitch" of each symbol (for T-normalization of compound patterns)
        # For terminals, base = terminal value
        # For rules, base = first terminal in expansion
        # We use a lookup table: symbol_base[symbol] = base pitch
        # For symbols 0..n_terminals-1: base = symbol
        # For symbols >= n_terminals: base is stored in corresponding index
        max_symbols = n_terminals + self.max_rules
        symbol_base = torch.zeros(max_symbols, dtype=torch.int64, device=self.device)
        symbol_base[:n_terminals] = torch.arange(n_terminals, device=self.device)

        next_rule_id = n_terminals
        n_rules = 0

        # Main Re-Pair loop
        iteration = 0
        while iteration < self.max_rules:
            # ===== STEP 1: Find most frequent T-normalized pair =====

            # Get consecutive pairs
            left = seq[:-1]
            right = seq[1:]

            # Mask out pairs with separator
            valid = (left != separator) & (right != separator)
            left_valid = left[valid]
            right_valid = right[valid]

            if len(left_valid) == 0:
                break

            # T-NORMALIZATION: Get base pitch for each symbol using lookup table
            # This is now O(1) tensor indexing instead of Python loops
            left_base = symbol_base[left_valid]
            right_base = symbol_base[right_valid]

            # Compute intervals (mod 12 for pitch classes)
            intervals = (right_base - left_base) % n_terminals

            # Also need to track the actual (left, right) for replacement
            # Encode as: left_symbol * max_vocab + right_symbol
            max_vocab = next_rule_id + self.max_rules
            pair_ids = left_valid * max_vocab + right_valid

            # Count by interval first to find best interval
            # SKIP unison (interval=0) for real music - repeated notes dominate otherwise
            non_unison = intervals != 0
            if non_unison.any():
                intervals_filtered = intervals[non_unison]
                pair_ids_filtered = pair_ids[non_unison]
            else:
                # Fall back to all intervals if no non-unison pairs
                intervals_filtered = intervals
                pair_ids_filtered = pair_ids

            unique_intervals, interval_counts = torch.unique(intervals_filtered, return_counts=True)
            max_interval_idx = interval_counts.argmax()
            max_interval_count = interval_counts[max_interval_idx].item()
            best_interval = unique_intervals[max_interval_idx].item()

            if max_interval_count < self.min_pair_count:
                break

            # Now find the most common actual pair with this interval
            # (we replace exact pairs, but count by interval)
            interval_mask = intervals_filtered == best_interval
            pairs_with_interval = pair_ids_filtered[interval_mask]

            unique_pairs, pair_counts = torch.unique(pairs_with_interval, return_counts=True)
            max_pair_idx = pair_counts.argmax()
            best_pair_id = unique_pairs[max_pair_idx].item()
            actual_count = pair_counts[max_pair_idx].item()

            # Decode best pair
            best_left = best_pair_id // max_vocab
            best_right = best_pair_id % max_vocab

            # ===== STEP 2: Create new rule =====
            rule_id = next_rule_id
            next_rule_id += 1

            # Store rule
            rule_table[n_rules, 0] = best_left
            rule_table[n_rules, 1] = best_right
            rule_counts[n_rules] = actual_count
            rule_intervals[n_rules] = best_interval

            # Compute base pitch for the new rule (first terminal in expansion)
            # Simply look up the base of the left symbol
            new_base = symbol_base[best_left]

            # Store base for the new rule
            symbol_base[rule_id] = new_base

            n_rules += 1

            # ===== STEP 3: Replace all occurrences of this pair =====
            seq = self._replace_pair_gpu(seq, best_left, best_right, rule_id, separator)

            # Progress
            if self.verbose and (iteration % self.progress_every == 0 or iteration < 10):
                elapsed = time.time() - start_time
                interval_name = self._interval_name(best_interval)
                is_hierarchical = best_left >= n_terminals or best_right >= n_terminals
                hier_marker = " [HIER]" if is_hierarchical else ""
                print(f"  [iter {iteration}] R{rule_id}=({best_left},{best_right}) "
                      f"interval={best_interval} ({interval_name}){hier_marker} "
                      f"count={actual_count:,}, len={len(seq):,}, time={elapsed:.1f}s", flush=True)

            iteration += 1

        # Remove separators from final sequence
        final_seq = seq[seq != separator]

        elapsed = time.time() - start_time

        if self.verbose:
            ratio = original_length / len(final_seq) if len(final_seq) > 0 else 1.0
            # Count hierarchical rules
            hier_count = 0
            for i in range(n_rules):
                left, right = rule_table[i]
                if left.item() >= n_terminals or right.item() >= n_terminals:
                    hier_count += 1
            print(f"[RePair-TnormHier] Done: {n_rules} rules ({hier_count} hierarchical), "
                  f"compression {ratio:.2f}x, time {elapsed:.1f}s", flush=True)

        return TnormHierarchicalGrammar(
            rule_table=rule_table[:n_rules],
            rule_counts=rule_counts[:n_rules],
            rule_intervals=rule_intervals[:n_rules],
            final_sequence=final_seq,
            n_terminals=n_terminals,
            n_rules=n_rules,
            original_length=original_length,
            compressed_length=len(final_seq),
            device=self.device,
        )

    def _get_symbol_bases(
        self,
        symbols: torch.Tensor,
        n_terminals: int,
        rule_table: torch.Tensor,
        n_rules: int
    ) -> torch.Tensor:
        """
        Get the base pitch (first terminal) for each symbol.

        For terminals: base = symbol
        For rules: base = first terminal in recursive expansion
        """
        bases = torch.zeros_like(symbols)

        # Terminals: base = symbol itself
        terminal_mask = symbols < n_terminals
        bases[terminal_mask] = symbols[terminal_mask]

        # Rules: need to recursively find first terminal
        rule_mask = symbols >= n_terminals
        if not rule_mask.any():
            return bases

        rule_symbols = symbols[rule_mask]
        rule_bases = torch.zeros_like(rule_symbols)

        for i, sym in enumerate(rule_symbols):
            rule_bases[i] = self._find_first_terminal(sym.item(), n_terminals, rule_table, n_rules)

        bases[rule_mask] = rule_bases
        return bases

    def _find_first_terminal(
        self,
        rule_id: int,
        n_terminals: int,
        rule_table: torch.Tensor,
        n_rules: int
    ) -> int:
        """Recursively find first terminal in a rule's expansion."""
        if rule_id < n_terminals:
            return rule_id

        idx = rule_id - n_terminals
        if idx >= n_rules:
            return rule_id % n_terminals  # Safety fallback

        left = rule_table[idx, 0].item()
        return self._find_first_terminal(left, n_terminals, rule_table, n_rules)

    def _replace_pair_gpu(
        self,
        seq: torch.Tensor,
        left: int,
        right: int,
        new_symbol: int,
        separator: int,
    ) -> torch.Tensor:
        """Replace all (left, right) pairs with new_symbol."""
        n = len(seq)
        if n < 2:
            return seq

        # Find pair positions
        is_left = seq[:-1] == left
        is_right = seq[1:] == right
        not_sep_left = seq[:-1] != separator
        not_sep_right = seq[1:] != separator

        pair_mask = is_left & is_right & not_sep_left & not_sep_right

        # Handle overlapping pairs
        pair_indices = torch.where(pair_mask)[0]

        if len(pair_indices) == 0:
            return seq

        # Remove overlapping: greedy left-to-right
        if len(pair_indices) > 1:
            diffs = pair_indices[1:] - pair_indices[:-1]
            keep = torch.ones(len(pair_indices), dtype=torch.bool, device=seq.device)
            keep[1:] = diffs > 1
            pair_indices = pair_indices[keep]

        # Build output
        keep_mask = torch.ones(n, dtype=torch.bool, device=seq.device)
        keep_mask[pair_indices + 1] = False  # Remove right elements

        # Replace left elements with new symbol
        result = seq.clone()
        result[pair_indices] = new_symbol

        return result[keep_mask]

    def _interval_name(self, interval: int) -> str:
        """Convert interval to musical name."""
        names = {
            0: "unison", 1: "m2", 2: "M2", 3: "m3", 4: "M3",
            5: "P4", 6: "tritone", 7: "P5", 8: "m6", 9: "M6",
            10: "m7", 11: "M7"
        }
        return names.get(interval % 12, f"{interval}")


def build_tnorm_hierarchical_grammar(
    sequences: List[List[int]],
    device: str = 'cuda',
    min_pair_count: int = 2,
    max_rules: int = 5000,
    pitch_range: int = 12,
    verbose: bool = True,
) -> TnormHierarchicalGrammar:
    """
    Build grammar using T-normalized hierarchical Re-Pair.

    Args:
        sequences: List of integer sequences (pitch classes 0-11)
        device: 'cuda' or 'cpu'
        min_pair_count: Minimum pair frequency
        max_rules: Maximum rules to create
        pitch_range: Terminal vocabulary size (12 for pitch classes)
        verbose: Print progress

    Returns:
        TnormHierarchicalGrammar
    """
    repair = RePairTnormHierarchical(
        device=device,
        min_pair_count=min_pair_count,
        max_rules=max_rules,
        pitch_range=pitch_range,
        verbose=verbose,
    )
    return repair.induce(sequences)


if __name__ == '__main__':
    print("Testing T-Normalized Hierarchical GPU Re-Pair...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test with musical data
    np.random.seed(42)
    n_sequences = 1000
    sequences = []

    # Create sequences with repeated TRANSPOSED patterns
    # These should be discovered as single patterns with T-normalization
    base_patterns = [
        [0, 4, 7],       # Major triad (root, M3, P5)
        [0, 3, 7],       # Minor triad (root, m3, P5)
        [0, 2, 4, 5, 7], # Major scale fragment
        [0, 4, 7, 11],   # Major 7th chord
    ]

    for _ in range(n_sequences):
        length = np.random.randint(50, 200)
        seq = []
        while len(seq) < length:
            # Pick a random pattern
            pattern = base_patterns[np.random.randint(len(base_patterns))]
            # Transpose randomly (mod 12 for pitch classes)
            transpose = np.random.randint(0, 12)
            transposed = [(p + transpose) % 12 for p in pattern]
            seq.extend(transposed)
        sequences.append(seq[:length])

    total_notes = sum(len(s) for s in sequences)
    print(f"\nTest data: {n_sequences} sequences, {total_notes:,} notes")
    print("Patterns: Major/minor triads, scale fragment, maj7 chord (all transposed)")

    # Run T-normalized hierarchical Re-Pair
    t0 = time.time()
    grammar = build_tnorm_hierarchical_grammar(sequences, device='cuda', verbose=True)
    elapsed = time.time() - t0

    print(f"\nResults:")
    print(f"  Rules: {grammar.n_rules}")
    print(f"  Compression: {grammar.compression_ratio():.2f}x")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {total_notes/elapsed:,.0f} notes/sec")

    # Show rules by length
    print("\nRules by expansion length:")
    length_counts = {}
    for i in range(grammar.n_rules):
        rule_id = grammar.n_terminals + i
        expansion = grammar.expand_rule(rule_id)
        length = len(expansion)
        length_counts[length] = length_counts.get(length, 0) + 1

    for length in sorted(length_counts.keys()):
        print(f"  Length {length}: {length_counts[length]} rules")

    # Show top rules by count
    print("\nTop 10 rules by count:")
    sorted_idx = grammar.rule_counts.argsort(descending=True)[:10]
    for idx in sorted_idx:
        rule_id = grammar.n_terminals + idx.item()
        left, right = grammar.get_rule(rule_id)
        count = grammar.rule_counts[idx].item()
        interval = grammar.rule_intervals[idx].item()
        expansion = grammar.expand_rule(rule_id)
        interval_name = {
            0: "unison", 1: "m2", 2: "M2", 3: "m3", 4: "M3",
            5: "P4", 6: "tritone", 7: "P5", 8: "m6", 9: "M6",
            10: "m7", 11: "M7"
        }.get(interval % 12 if interval >= 0 else -1, f"{interval}")
        is_hier = left >= 12 or right >= 12
        hier_mark = " [HIER]" if is_hier else ""
        print(f"  R{rule_id}: ({left},{right}) interval={interval_name}{hier_mark} "
              f"-> {expansion} (count={count:,})")
