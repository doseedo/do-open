"""
Step 2: Minimum Pattern Length Constraint
=========================================

Forces grammar rules to be musically meaningful minimum size.

How it works:
1. Post-process grammar: discard rules shorter than N notes (default N=4)
2. Re-encode using only surviving rules
3. Short patterns become literal sequences, long patterns become rules

Trade-off: Less compression, more musical meaning
"""

import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class FilteredGrammar:
    """Grammar with minimum length filter applied."""
    rules: Dict[int, 'FilteredRule'] = field(default_factory=dict)
    start_sequence: List[int] = field(default_factory=list)

    # Statistics
    original_rule_count: int = 0
    filtered_rule_count: int = 0
    min_length: int = 4

    # Mapping from old rule IDs to new
    rule_id_map: Dict[int, int] = field(default_factory=dict)

    def get_vocabulary_size(self) -> int:
        return len(self.rules)

    def get_rule_stats(self) -> Dict:
        stats = {}
        for rid, rule in self.rules.items():
            stats[str(rid)] = {
                'length': len(rule.expansion),
                'usage_count': rule.usage_count,
                'expansion_length': len(rule.expansion),
                'depth': 1,  # Filtered rules are flat
            }
        return stats

    def compression_ratio(self) -> float:
        if not self.start_sequence:
            return 1.0
        # Count total symbols including expansions
        total_terminals = sum(len(r.expansion) * r.usage_count for r in self.rules.values())
        return total_terminals / len(self.start_sequence) if self.start_sequence else 1.0


@dataclass
class FilteredRule:
    """A rule that passed the minimum length filter."""
    rule_id: int
    expansion: List[int]  # Terminal expansion
    usage_count: int = 0
    original_rule_id: int = -1

    def to_list(self) -> List[int]:
        return self.expansion

    def __repr__(self):
        exp_str = str(self.expansion[:8])
        if len(self.expansion) > 8:
            exp_str = exp_str[:-1] + ", ...]"
        return f"R{self.rule_id} -> {exp_str} (len={len(self.expansion)}, count={self.usage_count})"


class MinLengthFilter:
    """
    Filters grammar rules by minimum expansion length.

    Rules shorter than min_length are "inlined" - their expansions
    replace their occurrences in other rules and the start sequence.
    """

    def __init__(self, min_length: int = 4, verbose: bool = False):
        """
        Args:
            min_length: Minimum number of terminals in rule expansion
            verbose: Print filtering progress
        """
        self.min_length = min_length
        self.verbose = verbose

    def filter(self, grammar) -> FilteredGrammar:
        """
        Filter grammar to keep only rules >= min_length.

        Args:
            grammar: Input grammar (SEQUITUR or RePair)

        Returns:
            FilteredGrammar with only long rules
        """
        # First, expand all rules to their terminal sequences
        expansions = self._compute_expansions(grammar)

        # Identify rules that pass the length filter
        passing_rules = {}
        for rule_id, expansion in expansions.items():
            if len(expansion) >= self.min_length:
                passing_rules[rule_id] = expansion

        if self.verbose:
            print(f"[MinLength] Original rules: {len(expansions)}")
            print(f"[MinLength] Rules >= {self.min_length} notes: {len(passing_rules)}")

        # Re-encode the sequence using only passing rules
        # This requires finding occurrences of passing rule expansions

        # Get original terminal sequence
        if hasattr(grammar, 'start_sequence'):
            original_seq = self._expand_sequence(grammar.start_sequence, expansions)
        else:
            # For SEQUITUR, expand from start rule
            original_seq = self._expand_sequitur(grammar)

        if self.verbose:
            print(f"[MinLength] Original sequence length: {len(original_seq)}")

        # Build filtered grammar
        filtered = FilteredGrammar()
        filtered.original_rule_count = len(expansions)
        filtered.min_length = self.min_length

        # Create new rule IDs (contiguous starting from max_terminal + 1)
        max_terminal = max(original_seq) if original_seq else 0
        new_rule_id = max_terminal + 1

        old_to_new = {}
        for old_id, expansion in sorted(passing_rules.items()):
            old_to_new[old_id] = new_rule_id
            filtered.rules[new_rule_id] = FilteredRule(
                rule_id=new_rule_id,
                expansion=expansion,
                original_rule_id=old_id,
            )
            new_rule_id += 1

        filtered.rule_id_map = old_to_new
        filtered.filtered_rule_count = len(filtered.rules)

        # Re-encode sequence using passing rules
        filtered.start_sequence = self._encode_with_rules(
            original_seq, filtered.rules
        )

        if self.verbose:
            print(f"[MinLength] Filtered sequence length: {len(filtered.start_sequence)}")
            print(f"[MinLength] Compression: {filtered.compression_ratio():.2f}x")

        return filtered

    def _compute_expansions(self, grammar) -> Dict[int, List[int]]:
        """Compute terminal expansions for all rules."""
        expansions = {}

        if hasattr(grammar, 'rules'):
            for rule_id, rule in grammar.rules.items():
                expansion = self._expand_rule(rule_id, grammar, {})
                expansions[rule_id] = expansion

        return expansions

    def _expand_rule(
        self,
        rule_id: int,
        grammar,
        memo: Dict[int, List[int]]
    ) -> List[int]:
        """Recursively expand a rule to terminals."""
        if rule_id in memo:
            return memo[rule_id]

        if not hasattr(grammar, 'rules') or rule_id not in grammar.rules:
            # Terminal
            return [rule_id]

        rule = grammar.rules[rule_id]

        # Handle different rule formats
        if hasattr(rule, 'left') and hasattr(rule, 'right'):
            # RePair format
            left_exp = self._expand_rule(rule.left, grammar, memo)
            right_exp = self._expand_rule(rule.right, grammar, memo)
            result = left_exp + right_exp
        elif hasattr(rule, 'expansion'):
            # SEQUITUR format
            result = []
            for symbol in rule.expansion:
                if isinstance(symbol, tuple) and symbol[0] == 'R':
                    # Rule reference
                    sub_exp = self._expand_rule(symbol[1], grammar, memo)
                    result.extend(sub_exp)
                elif isinstance(symbol, int):
                    result.append(symbol)
        elif hasattr(rule, 'to_list'):
            # Generic format
            rule_list = rule.to_list()
            result = []
            for symbol in rule_list:
                if isinstance(symbol, list) and len(symbol) == 2 and symbol[0] == 'R':
                    sub_exp = self._expand_rule(symbol[1], grammar, memo)
                    result.extend(sub_exp)
                elif isinstance(symbol, int):
                    result.append(symbol)
                else:
                    result.append(symbol)
        else:
            result = [rule_id]

        memo[rule_id] = result
        return result

    def _expand_sequence(
        self,
        sequence: List,
        expansions: Dict[int, List[int]]
    ) -> List[int]:
        """Expand a sequence using precomputed expansions."""
        result = []
        for symbol in sequence:
            if symbol in expansions:
                result.extend(expansions[symbol])
            else:
                result.append(symbol)
        return result

    def _expand_sequitur(self, grammar) -> List[int]:
        """Expand SEQUITUR grammar to terminal sequence."""
        if not hasattr(grammar, 'rules') or 0 not in grammar.rules:
            return []

        memo = {}
        return self._expand_rule(0, grammar, memo)

    def _encode_with_rules(
        self,
        sequence: List[int],
        rules: Dict[int, FilteredRule]
    ) -> List[int]:
        """
        Re-encode sequence using only the filtered rules.

        Uses greedy longest-match algorithm:
        1. Sort rules by length (longest first)
        2. At each position, try to match longest rule
        3. If match, emit rule ID and skip matched positions
        4. If no match, emit terminal
        """
        if not rules:
            return sequence

        # Sort rules by expansion length (longest first)
        sorted_rules = sorted(
            rules.items(),
            key=lambda x: len(x[1].expansion),
            reverse=True
        )

        # Build lookup for fast matching
        # Group by first symbol and length for efficiency
        rule_lookup: Dict[Tuple[int, int], List[Tuple[int, FilteredRule]]] = {}
        for rule_id, rule in sorted_rules:
            if rule.expansion:
                key = (rule.expansion[0], len(rule.expansion))
                if key not in rule_lookup:
                    rule_lookup[key] = []
                rule_lookup[key].append((rule_id, rule))

        # Greedy encoding
        result = []
        i = 0
        seq_len = len(sequence)

        while i < seq_len:
            matched = False

            # Try rules of decreasing length
            for rule_id, rule in sorted_rules:
                exp_len = len(rule.expansion)
                if i + exp_len > seq_len:
                    continue

                # Quick check: first symbol must match
                if sequence[i] != rule.expansion[0]:
                    continue

                # Full match check
                if sequence[i:i+exp_len] == rule.expansion:
                    result.append(rule_id)
                    rule.usage_count += 1
                    i += exp_len
                    matched = True
                    break

            if not matched:
                result.append(sequence[i])
                i += 1

        return result


def filter_grammar_by_length(
    grammar,
    min_length: int = 4,
    verbose: bool = False,
) -> FilteredGrammar:
    """
    Convenience function to filter grammar by minimum length.

    Args:
        grammar: Input grammar (SEQUITUR, RePair, or any with .rules dict)
        min_length: Minimum terminal expansion length
        verbose: Print progress

    Returns:
        FilteredGrammar with only long rules
    """
    filter = MinLengthFilter(min_length=min_length, verbose=verbose)
    return filter.filter(grammar)


if __name__ == '__main__':
    # Test with RePair
    from repair import build_repair_grammar

    print("Testing minimum length filter...")

    # Create test sequences with repeated patterns
    test_seqs = [
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],  # Pattern of 5
        [1, 2, 3, 1, 2, 3, 1, 2, 3],  # Pattern of 3
        [6, 7, 8, 9, 6, 7, 8, 9],  # Pattern of 4
    ]

    # Build RePair grammar
    grammar = build_repair_grammar(test_seqs, device='cpu', verbose=True)

    print(f"\nOriginal grammar rules:")
    for rid, rule in grammar.rules.items():
        exp = grammar._expand_rule(rid)
        print(f"  R{rid}: {exp}")

    # Apply minimum length filter
    filtered = filter_grammar_by_length(grammar, min_length=4, verbose=True)

    print(f"\nFiltered grammar rules (min_length=4):")
    for rid, rule in filtered.rules.items():
        print(f"  {rule}")

    print(f"\nFiltered sequence: {filtered.start_sequence}")
