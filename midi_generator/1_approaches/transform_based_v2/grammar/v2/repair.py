"""
RePair Grammar Induction (GPU-Optimized for A100)
=================================================

RePair replaces the most frequent pair globally (not first-seen like SEQUITUR),
producing fewer but longer rules with higher coverage.

GPU Optimizations:
1. Pair counting via tensor operations on GPU
2. Batch replacement using scatter operations
3. Parallel bigram extraction
4. Memory-efficient for 40GB A100

Key differences from SEQUITUR:
- SEQUITUR: Replace first occurrence of digram when count > 1
- RePair: Replace ALL occurrences of most frequent pair globally
- Result: Longer rules, fewer rules, better coverage

Complexity: O(n log n) with GPU acceleration
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
import time


@dataclass
class RePairRule:
    """A grammar rule produced by RePair."""
    rule_id: int
    left: int   # Left symbol (terminal or rule ID)
    right: int  # Right symbol (terminal or rule ID)
    expansion: List[int] = field(default_factory=list)  # Full terminal expansion
    count: int = 0  # How many times this rule was used

    def to_list(self) -> List:
        """Convert to serializable list."""
        return [self.left, self.right]

    def __repr__(self):
        return f"R{self.rule_id} -> ({self.left}, {self.right}) [count={self.count}]"


@dataclass
class RePairGrammar:
    """Grammar produced by RePair algorithm."""
    rules: Dict[int, RePairRule] = field(default_factory=dict)
    start_sequence: List[int] = field(default_factory=list)  # Final encoded sequence
    terminal_count: int = 0  # Number of unique terminals

    # Statistics
    original_length: int = 0
    compressed_length: int = 0
    n_rules: int = 0

    def get_vocabulary_size(self) -> int:
        return len(self.rules)

    def get_rule_stats(self) -> Dict:
        """Get statistics about rules."""
        stats = {}
        for rid, rule in self.rules.items():
            expansion = self._expand_rule(rid)
            stats[str(rid)] = {
                'length': 2,  # RePair rules are always pairs
                'usage_count': rule.count,
                'expansion_length': len(expansion),
                'depth': self._rule_depth(rid),
            }
        return stats

    def _expand_rule(self, rule_id: int, memo: Optional[Dict] = None) -> List[int]:
        """Expand a rule to its terminal sequence."""
        if memo is None:
            memo = {}
        if rule_id in memo:
            return memo[rule_id]

        if rule_id not in self.rules:
            # Terminal
            return [rule_id]

        rule = self.rules[rule_id]
        left_exp = self._expand_rule(rule.left, memo)
        right_exp = self._expand_rule(rule.right, memo)
        result = left_exp + right_exp
        memo[rule_id] = result
        return result

    def _rule_depth(self, rule_id: int, memo: Optional[Dict] = None) -> int:
        """Compute nesting depth of a rule."""
        if memo is None:
            memo = {}
        if rule_id in memo:
            return memo[rule_id]

        if rule_id not in self.rules:
            return 0

        rule = self.rules[rule_id]
        left_depth = self._rule_depth(rule.left, memo)
        right_depth = self._rule_depth(rule.right, memo)
        result = 1 + max(left_depth, right_depth)
        memo[rule_id] = result
        return result

    def compression_ratio(self) -> float:
        if self.compressed_length == 0:
            return 1.0
        return self.original_length / self.compressed_length


class RePairGPU:
    """
    GPU-accelerated RePair implementation for A100.

    Algorithm:
    1. Count all bigram frequencies in parallel (GPU)
    2. Find most frequent bigram
    3. Replace ALL occurrences with new rule symbol
    4. Repeat until no bigram has count > 1

    GPU optimizations:
    - Bigram counting via tensor scatter_add
    - Parallel replacement via boolean masking
    - Batch processing for memory efficiency
    """

    def __init__(
        self,
        device: str = 'cuda',
        min_pair_count: int = 2,  # Minimum frequency to create rule
        max_rules: int = 10000,   # Maximum number of rules
        verbose: bool = False,
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.min_pair_count = min_pair_count
        self.max_rules = max_rules
        self.verbose = verbose

        # Rule counter (rules start after max terminal ID)
        self.next_rule_id = 0
        self.max_terminal = 0

    def induce(self, sequences: List[List[int]]) -> RePairGrammar:
        """
        Run RePair on multiple sequences.

        Args:
            sequences: List of integer sequences (token IDs)

        Returns:
            RePairGrammar with induced rules
        """
        start_time = time.time()

        # Concatenate all sequences with separators
        # Use a special separator token that won't form pairs
        separator = -1

        # Find max terminal ID
        self.max_terminal = max(max(seq) for seq in sequences if seq) + 1
        self.next_rule_id = self.max_terminal

        # Build combined sequence
        combined = []
        seq_boundaries = [0]
        for seq in sequences:
            combined.extend(seq)
            combined.append(separator)
            seq_boundaries.append(len(combined))

        # Convert to tensor
        seq_tensor = torch.tensor(combined, dtype=torch.int64, device=self.device)
        original_length = len(seq_tensor)

        if self.verbose:
            print(f"[RePair] Starting with {len(sequences)} sequences, {original_length} tokens")
            print(f"[RePair] Max terminal ID: {self.max_terminal}")

        # Run RePair iterations
        grammar = RePairGrammar()
        grammar.original_length = original_length
        grammar.terminal_count = self.max_terminal

        iteration = 0
        while iteration < self.max_rules:
            # Count bigram frequencies
            pair_counts, pair_to_idx, idx_to_pair = self._count_bigrams_gpu(seq_tensor, separator)

            if len(pair_counts) == 0:
                break

            # Find most frequent pair
            max_count = pair_counts.max().item()

            if max_count < self.min_pair_count:
                break

            max_idx = pair_counts.argmax().item()
            left, right = idx_to_pair[max_idx]

            # Create new rule
            rule_id = self.next_rule_id
            self.next_rule_id += 1

            rule = RePairRule(
                rule_id=rule_id,
                left=left,
                right=right,
                count=max_count,
            )
            grammar.rules[rule_id] = rule

            if self.verbose and iteration % 100 == 0:
                print(f"[RePair] Iteration {iteration}: Rule R{rule_id} = ({left}, {right}), count={max_count}")

            # Replace all occurrences
            seq_tensor = self._replace_pair_gpu(seq_tensor, left, right, rule_id, separator)

            iteration += 1

        # Extract final sequence (remove separators)
        final_seq = seq_tensor[seq_tensor != separator].cpu().tolist()
        grammar.start_sequence = final_seq
        grammar.compressed_length = len(final_seq)
        grammar.n_rules = len(grammar.rules)

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"[RePair] Complete: {grammar.n_rules} rules, "
                  f"compression {grammar.compression_ratio():.2f}x, "
                  f"time {elapsed:.2f}s")

        return grammar

    def _count_bigrams_gpu(
        self,
        seq: torch.Tensor,
        separator: int,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Count bigram frequencies using GPU.

        Returns:
            (counts tensor, pair->idx dict, idx->pair dict)
        """
        # Get bigrams (pairs of consecutive elements)
        left = seq[:-1]
        right = seq[1:]

        # Mask out pairs containing separator
        valid_mask = (left != separator) & (right != separator)
        left = left[valid_mask]
        right = right[valid_mask]

        if len(left) == 0:
            return torch.tensor([]), {}, {}

        # Create unique pair IDs
        # Use Cantor pairing function: (a + b) * (a + b + 1) / 2 + b
        # But simpler: a * MAX + b where MAX > max(b)
        max_val = max(seq.max().item(), self.next_rule_id) + 1
        pair_ids = left * max_val + right

        # Count unique pairs
        unique_pairs, inverse, counts = torch.unique(
            pair_ids, return_inverse=True, return_counts=True
        )

        # Build lookup dictionaries
        pair_to_idx = {}
        idx_to_pair = {}

        unique_pairs_cpu = unique_pairs.cpu().numpy()
        for idx, pair_id in enumerate(unique_pairs_cpu):
            l = int(pair_id // max_val)
            r = int(pair_id % max_val)
            pair_to_idx[(l, r)] = idx
            idx_to_pair[idx] = (l, r)

        return counts, pair_to_idx, idx_to_pair

    def _replace_pair_gpu(
        self,
        seq: torch.Tensor,
        left: int,
        right: int,
        new_symbol: int,
        separator: int,
    ) -> torch.Tensor:
        """
        Replace all occurrences of (left, right) with new_symbol.

        GPU-optimized using boolean masking.
        """
        # Find all positions where pair occurs
        is_left = seq[:-1] == left
        is_right = seq[1:] == right
        pair_starts = is_left & is_right

        # Also check not crossing separator
        not_sep_left = seq[:-1] != separator
        not_sep_right = seq[1:] != separator
        pair_starts = pair_starts & not_sep_left & not_sep_right

        # Build new sequence
        # Mark positions to keep vs skip
        n = len(seq)
        keep_mask = torch.ones(n, dtype=torch.bool, device=self.device)

        # The position after each pair start should be removed
        pair_start_indices = torch.where(pair_starts)[0]
        if len(pair_start_indices) == 0:
            return seq

        # Mark right elements of pairs for removal
        keep_mask[pair_start_indices + 1] = False

        # Replace left elements with new symbol
        seq = seq.clone()
        seq[pair_start_indices] = new_symbol

        # Filter to kept elements
        new_seq = seq[keep_mask]

        return new_seq


class RePairCPU:
    """
    CPU fallback for RePair when GPU not available.

    Uses optimized Python with numpy for pair counting.
    """

    def __init__(
        self,
        min_pair_count: int = 2,
        max_rules: int = 10000,
        verbose: bool = False,
    ):
        self.min_pair_count = min_pair_count
        self.max_rules = max_rules
        self.verbose = verbose
        self.next_rule_id = 0
        self.max_terminal = 0

    def induce(self, sequences: List[List[int]]) -> RePairGrammar:
        """Run RePair on CPU."""
        start_time = time.time()

        # Find max terminal
        self.max_terminal = max(max(seq) for seq in sequences if seq) + 1
        self.next_rule_id = self.max_terminal

        # Concatenate with separators
        separator = -1
        combined = []
        for seq in sequences:
            combined.extend(seq)
            combined.append(separator)

        original_length = len(combined)
        grammar = RePairGrammar()
        grammar.original_length = original_length
        grammar.terminal_count = self.max_terminal

        if self.verbose:
            print(f"[RePair-CPU] Starting with {len(sequences)} sequences, {original_length} tokens")

        # Main loop
        iteration = 0
        while iteration < self.max_rules:
            # Count pairs
            pair_counts = Counter()
            for i in range(len(combined) - 1):
                if combined[i] != separator and combined[i+1] != separator:
                    pair_counts[(combined[i], combined[i+1])] += 1

            if not pair_counts:
                break

            # Find most frequent
            (left, right), max_count = pair_counts.most_common(1)[0]

            if max_count < self.min_pair_count:
                break

            # Create rule
            rule_id = self.next_rule_id
            self.next_rule_id += 1

            rule = RePairRule(
                rule_id=rule_id,
                left=left,
                right=right,
                count=max_count,
            )
            grammar.rules[rule_id] = rule

            if self.verbose and iteration % 100 == 0:
                print(f"[RePair-CPU] Iteration {iteration}: R{rule_id} = ({left}, {right}), count={max_count}")

            # Replace all occurrences
            new_combined = []
            i = 0
            while i < len(combined):
                if (i < len(combined) - 1 and
                    combined[i] == left and
                    combined[i+1] == right and
                    combined[i] != separator and
                    combined[i+1] != separator):
                    new_combined.append(rule_id)
                    i += 2
                else:
                    new_combined.append(combined[i])
                    i += 1

            combined = new_combined
            iteration += 1

        # Remove separators
        grammar.start_sequence = [x for x in combined if x != separator]
        grammar.compressed_length = len(grammar.start_sequence)
        grammar.n_rules = len(grammar.rules)

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"[RePair-CPU] Complete: {grammar.n_rules} rules, "
                  f"compression {grammar.compression_ratio():.2f}x, "
                  f"time {elapsed:.2f}s")

        return grammar


def build_repair_grammar(
    sequences: List[List[int]],
    device: str = 'cuda',
    min_pair_count: int = 2,
    max_rules: int = 10000,
    verbose: bool = False,
) -> RePairGrammar:
    """
    Build grammar using RePair algorithm.

    Args:
        sequences: List of integer sequences
        device: 'cuda' or 'cpu'
        min_pair_count: Minimum pair frequency to create rule
        max_rules: Maximum number of rules to create
        verbose: Print progress

    Returns:
        RePairGrammar
    """
    if device == 'cuda' and torch.cuda.is_available():
        repair = RePairGPU(
            device=device,
            min_pair_count=min_pair_count,
            max_rules=max_rules,
            verbose=verbose,
        )
    else:
        repair = RePairCPU(
            min_pair_count=min_pair_count,
            max_rules=max_rules,
            verbose=verbose,
        )

    return repair.induce(sequences)


def build_repair_from_corpus(
    factored_objects: List,
    device: str = 'cuda',
    min_pair_count: int = 2,
    max_rules: int = 10000,
    verbose: bool = False,
) -> RePairGrammar:
    """
    Build RePair grammar from factored MIDI objects.

    Args:
        factored_objects: List of factored objects with pitch_class arrays
        device: 'cuda' or 'cpu'
        min_pair_count: Minimum frequency for rule creation
        max_rules: Maximum rules
        verbose: Print progress

    Returns:
        RePairGrammar
    """
    # Extract pitch class sequences
    sequences = []
    for obj in factored_objects:
        if hasattr(obj, 'pitch_class') and len(obj.pitch_class) > 0:
            # Convert to list of integers
            pc = obj.pitch_class
            if hasattr(pc, 'tolist'):
                pc = pc.tolist()
            sequences.append(pc)

    if verbose:
        print(f"[RePair] Extracted {len(sequences)} sequences from {len(factored_objects)} objects")

    return build_repair_grammar(
        sequences,
        device=device,
        min_pair_count=min_pair_count,
        max_rules=max_rules,
        verbose=verbose,
    )


if __name__ == '__main__':
    # Test RePair
    print("Testing RePair GPU implementation...")

    # Simple test sequences
    test_seqs = [
        [1, 2, 3, 1, 2, 3, 1, 2, 3],
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        [2, 3, 2, 3, 2, 3, 4, 5, 4, 5],
    ]

    grammar = build_repair_grammar(test_seqs, device='cuda', verbose=True)

    print(f"\nGrammar rules:")
    for rid, rule in grammar.rules.items():
        print(f"  {rule}")

    print(f"\nFinal sequence: {grammar.start_sequence}")
    print(f"Compression: {grammar.compression_ratio():.2f}x")
