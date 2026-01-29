"""
T-Normalized GPU Re-Pair Implementation
========================================

Re-Pair with T-normalization: pairs are normalized by interval so that
all transpositions of the same pattern are counted together.

Example:
  (C, E)  -> interval +4 -> canonical (0, 4)
  (Eb, G) -> interval +4 -> canonical (0, 4)
  (G, B)  -> interval +4 -> canonical (0, 4)

All three count as ONE canonical pair with 3 occurrences.

Key differences from regular Re-Pair:
1. Pairs are encoded as intervals, not absolute pitches
2. When replacing, ALL transpositions of the best pair are replaced
3. The rule stores the canonical form (0-based interval)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class TnormGrammar:
    """Grammar produced by T-normalized Re-Pair."""
    # Rule table: rule_id -> (left_interval, right_interval)
    # All rules are in canonical form where first note = 0
    rule_table: torch.Tensor  # Shape: [n_rules, 2]
    rule_counts: torch.Tensor  # Shape: [n_rules]

    # Final compressed sequence (still absolute pitches)
    final_sequence: torch.Tensor

    # Metadata
    n_terminals: int  # Pitch range (e.g., 128 for MIDI)
    n_rules: int
    original_length: int
    compressed_length: int
    device: str

    def compression_ratio(self) -> float:
        if self.compressed_length == 0:
            return 1.0
        return self.original_length / self.compressed_length

    def get_rule_canonical(self, rule_id: int) -> Tuple[int, int]:
        """Get canonical form of rule (intervals from 0)."""
        idx = rule_id - self.n_terminals
        if idx < 0 or idx >= len(self.rule_table):
            return (rule_id, -1)  # Terminal
        row = self.rule_table[idx]
        return (row[0].item(), row[1].item())


class RePairTnorm:
    """
    T-Normalized GPU Re-Pair.

    Pairs are normalized by interval:
    - (C, E) and (Eb, G) both become canonical (0, 4)
    - This dramatically increases pattern frequency
    """

    def __init__(
        self,
        device: str = 'cuda',
        min_pair_count: int = 2,
        max_rules: int = 5000,
        pitch_range: int = 128,  # MIDI pitch range
        verbose: bool = True,
        progress_every: int = 100,
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.min_pair_count = min_pair_count
        self.max_rules = max_rules
        self.pitch_range = pitch_range
        self.verbose = verbose
        self.progress_every = progress_every

    def induce(self, sequences: List[List[int]]) -> TnormGrammar:
        """
        Run T-normalized Re-Pair.

        Args:
            sequences: List of integer sequences (MIDI pitches 0-127)

        Returns:
            TnormGrammar with induced grammar
        """
        start_time = time.time()

        # For T-norm, we need the original values for interval calculation
        # but we'll use rule IDs starting from pitch_range
        n_terminals = self.pitch_range

        # Use separator that won't collide
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
            print(f"[RePair-Tnorm] {len(sequences)} seqs, {original_length:,} tokens", flush=True)

        # Pre-allocate rule table on GPU
        # For T-norm: stores canonical intervals (0, interval)
        rule_table = torch.zeros((self.max_rules, 2), dtype=torch.int64, device=self.device)
        rule_counts = torch.zeros(self.max_rules, dtype=torch.int64, device=self.device)

        next_rule_id = n_terminals
        n_rules = 0

        # Main Re-Pair loop
        iteration = 0
        while iteration < self.max_rules:
            # ===== STEP 1: Find most frequent T-normalized pair =====

            # Get consecutive pairs
            left = seq[:-1]
            right = seq[1:]

            # Mask out pairs with separator and non-terminal pairs
            # Only count pairs where both are terminals (pitches)
            valid = (left != separator) & (right != separator)
            valid = valid & (left < n_terminals) & (right < n_terminals)

            left_valid = left[valid]
            right_valid = right[valid]

            if len(left_valid) == 0:
                # No more terminal pairs, but there might be pairs involving rules
                # For simplicity, we'll also count mixed pairs
                valid = (left != separator) & (right != separator)
                left_valid = left[valid]
                right_valid = right[valid]

                if len(left_valid) == 0:
                    break

            # T-NORMALIZATION: Encode as interval
            # canonical_id = interval (right - left, modulo for wrap)
            # For music, we use signed interval (can be negative)
            intervals = right_valid - left_valid

            # For counting, we just need the interval
            # But we also need to know the base for reconstruction
            unique_intervals, counts = torch.unique(intervals, return_counts=True)
            max_idx = counts.argmax()
            max_count = counts[max_idx].item()
            best_interval = unique_intervals[max_idx].item()

            # Check termination
            if max_count < self.min_pair_count:
                break

            # ===== STEP 2: Create new rule =====
            rule_id = next_rule_id
            next_rule_id += 1

            # Store canonical form: (0, interval)
            rule_table[n_rules, 0] = 0  # Canonical left is always 0
            rule_table[n_rules, 1] = best_interval
            rule_counts[n_rules] = max_count
            n_rules += 1

            # ===== STEP 3: Replace ALL transpositions of this interval =====
            seq = self._replace_interval_gpu(seq, best_interval, rule_id, separator, n_terminals)

            # Progress
            if self.verbose and (iteration % self.progress_every == 0 or iteration < 10):
                elapsed = time.time() - start_time
                print(f"  [iter {iteration}] R{rule_id} interval={best_interval:+d} "
                      f"count={max_count:,}, len={len(seq):,}, time={elapsed:.1f}s", flush=True)

            iteration += 1

        # Remove separators from final sequence
        final_seq = seq[seq != separator]

        elapsed = time.time() - start_time

        if self.verbose:
            ratio = original_length / len(final_seq) if len(final_seq) > 0 else 1.0
            print(f"[RePair-Tnorm] Done: {n_rules} rules, "
                  f"compression {ratio:.2f}x, "
                  f"time {elapsed:.1f}s", flush=True)

        return TnormGrammar(
            rule_table=rule_table[:n_rules],
            rule_counts=rule_counts[:n_rules],
            final_sequence=final_seq,
            n_terminals=n_terminals,
            n_rules=n_rules,
            original_length=original_length,
            compressed_length=len(final_seq),
            device=self.device,
        )

    def _replace_interval_gpu(
        self,
        seq: torch.Tensor,
        interval: int,
        new_symbol: int,
        separator: int,
        n_terminals: int,
    ) -> torch.Tensor:
        """
        Replace ALL pairs with the given interval with new_symbol.

        This is the key T-normalization: (C,E), (D,F#), (Eb,G) all have
        interval +4 and ALL get replaced with the same rule symbol.

        The new_symbol replaces the pair, but we need to track transposition.
        For now, we replace with new_symbol and lose transposition info
        (can be recovered from context or stored separately).
        """
        n = len(seq)
        if n < 2:
            return seq

        left = seq[:-1]
        right = seq[1:]

        # Find all pairs with this interval (among terminals only for now)
        valid_pair = (left != separator) & (right != separator)
        valid_pair = valid_pair & (left < n_terminals) & (right < n_terminals)
        has_interval = (right - left) == interval

        pair_mask = valid_pair & has_interval

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

        # Replace left elements with new_symbol
        result = seq.clone()
        result[pair_indices] = new_symbol

        return result[keep_mask]


def build_tnorm_grammar(
    sequences: List[List[int]],
    device: str = 'cuda',
    min_pair_count: int = 2,
    max_rules: int = 5000,
    pitch_range: int = 128,
    verbose: bool = True,
) -> TnormGrammar:
    """
    Build grammar using T-normalized Re-Pair.

    Args:
        sequences: List of integer sequences (MIDI pitches)
        device: 'cuda' or 'cpu'
        min_pair_count: Minimum pair frequency
        max_rules: Maximum rules to create
        pitch_range: Terminal vocabulary size (12 for pitch classes, 128 for full MIDI)
        verbose: Print progress

    Returns:
        TnormGrammar
    """
    repair = RePairTnorm(
        device=device,
        min_pair_count=min_pair_count,
        max_rules=max_rules,
        pitch_range=pitch_range,
        verbose=verbose,
    )
    return repair.induce(sequences)


if __name__ == '__main__':
    print("Testing T-Normalized GPU Re-Pair...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test with musical data (intervals matter)
    np.random.seed(42)
    n_sequences = 1000
    sequences = []

    # Create sequences with repeated TRANSPOSED patterns
    # E.g., C-E-G at different octaves
    base_patterns = [
        [0, 4, 7],       # Major triad
        [0, 3, 7],       # Minor triad
        [0, 2, 4, 5, 7], # Major scale fragment
    ]

    for _ in range(n_sequences):
        length = np.random.randint(50, 200)
        seq = []
        while len(seq) < length:
            # Pick a random pattern
            pattern = base_patterns[np.random.randint(len(base_patterns))]
            # Transpose randomly within MIDI range
            transpose = np.random.randint(36, 84)  # C2 to C6
            transposed = [p + transpose for p in pattern]
            seq.extend(transposed)
        sequences.append(seq[:length])

    total_notes = sum(len(s) for s in sequences)
    print(f"\nTest data: {n_sequences} sequences, {total_notes:,} notes")
    print("Patterns: Major triad, Minor triad, Scale fragment (all transposed)")

    # Run T-normalized Re-Pair
    t0 = time.time()
    grammar = build_tnorm_grammar(sequences, device='cuda', verbose=True)
    elapsed = time.time() - t0

    print(f"\nResults:")
    print(f"  Rules: {grammar.n_rules}")
    print(f"  Compression: {grammar.compression_ratio():.2f}x")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {total_notes/elapsed:,.0f} notes/sec")

    # Show top rules
    print("\nTop 10 rules by count:")
    sorted_idx = grammar.rule_counts.argsort(descending=True)[:10]
    for idx in sorted_idx:
        rule_id = grammar.n_terminals + idx.item()
        left, right = grammar.get_rule_canonical(rule_id)
        count = grammar.rule_counts[idx].item()
        # Interpret interval musically
        interval = right - left
        interval_names = {
            0: "unison", 1: "m2", 2: "M2", 3: "m3", 4: "M3",
            5: "P4", 6: "tritone", 7: "P5", 8: "m6", 9: "M6",
            10: "m7", 11: "M7", 12: "octave"
        }
        name = interval_names.get(abs(interval) % 12, f"{interval:+d}")
        print(f"  R{rule_id}: interval {interval:+d} ({name}), count={count:,}")
