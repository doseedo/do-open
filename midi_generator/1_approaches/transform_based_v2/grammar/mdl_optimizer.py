"""
MDL Vocabulary Optimizer

Uses Minimum Description Length (MDL) principle to select the optimal
subset of grammar rules that minimizes total description length.

MDL calculation:
    Total cost = Model cost + Data cost
    Model cost = bits to describe the grammar rules
    Data cost = bits to encode the corpus using the grammar

A rule is beneficial if:
    bits_saved = usage_count * (pattern_length * bits_per_token - bits_for_rule_reference)
    bits_to_describe = pattern_length * bits_per_token + overhead
    benefit = bits_saved - bits_to_describe > 0

Author: MDL Vocabulary Optimizer for Musical Pattern Discovery
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .sequitur import SequiturGrammar, Rule


@dataclass
class MDLScore:
    """MDL score for a single grammar rule."""
    rule_id: int
    usage_count: int
    pattern_length: int  # Length in rule symbols
    expansion_length: int  # Length when fully expanded to terminals
    depth: int

    # Computed costs
    definition_cost: float = 0.0  # Bits to describe the rule
    usage_cost: float = 0.0  # Bits per usage (pointer into vocabulary)
    total_savings: float = 0.0  # Total bits saved by using this rule
    net_benefit: float = 0.0  # savings - definition_cost

    @property
    def compression_ratio(self) -> float:
        """Coverage per bit of description."""
        if self.definition_cost + self.usage_cost == 0:
            return 0.0
        return self.usage_count / (self.definition_cost + self.usage_cost * self.usage_count)


@dataclass
class VocabularyStats:
    """Statistics about the optimized vocabulary."""
    total_rules: int
    selected_rules: int
    total_tokens: int
    encoded_tokens: int
    compression_ratio: float
    total_bits_saved: float
    vocabulary_cost: float
    max_depth: int
    avg_usage: float


class VocabularyOptimizer:
    """
    MDL-based vocabulary selection from SEQUITUR grammar.

    Selects the subset of grammar rules that minimize total description length.
    """

    def __init__(self,
                 target_size: int = 400,
                 bits_per_terminal: float = 8.0,
                 rule_overhead_bits: float = 2.0,
                 min_usage: int = 2,
                 min_benefit: float = 0.0):
        """
        Initialize optimizer.

        Args:
            target_size: Target vocabulary size (aim for 300-400)
            bits_per_terminal: Bits to encode one terminal token (~8 for typical vocab)
            rule_overhead_bits: Overhead bits for rule structure
            min_usage: Minimum usage count to consider a rule
            min_benefit: Minimum net MDL benefit to include rule
        """
        self.target_size = target_size
        self.bits_per_terminal = bits_per_terminal
        self.rule_overhead_bits = rule_overhead_bits
        self.min_usage = min_usage
        self.min_benefit = min_benefit

    def score_rule(self,
                   rule: Rule,
                   vocab_size: int,
                   rule_stats: Dict) -> MDLScore:
        """
        Compute MDL score for a single rule.

        Args:
            rule: The grammar rule
            vocab_size: Current vocabulary size (for pointer cost)
            rule_stats: Statistics dict from grammar.get_rule_stats()

        Returns:
            MDLScore with computed costs and benefit
        """
        stats = rule_stats[rule.id]

        score = MDLScore(
            rule_id=rule.id,
            usage_count=stats['usage_count'],
            pattern_length=stats['length'],
            expansion_length=stats['expansion_length'],
            depth=stats['depth']
        )

        # Cost to define this rule
        # = length * bits_per_symbol + overhead
        # symbols can be terminals (bits_per_terminal) or rule refs (log2(vocab))
        bits_per_ref = math.log2(max(1, vocab_size))
        avg_bits_per_symbol = (self.bits_per_terminal + bits_per_ref) / 2
        score.definition_cost = (
            stats['length'] * avg_bits_per_symbol +
            self.rule_overhead_bits
        )

        # Cost to use this rule (pointer into vocabulary)
        score.usage_cost = math.log2(max(1, vocab_size))

        # Savings: each use saves (expansion_length - 1) * bits_per_terminal
        # because we replace expansion_length terminals with 1 rule reference
        bits_without = stats['expansion_length'] * self.bits_per_terminal
        bits_with = score.usage_cost
        score.total_savings = stats['usage_count'] * (bits_without - bits_with)

        # Net benefit = savings - cost of having rule in vocabulary
        score.net_benefit = score.total_savings - score.definition_cost

        return score

    def optimize(self,
                 grammar: SequiturGrammar,
                 verbose: bool = True) -> Tuple[Set[int], VocabularyStats]:
        """
        Select optimal rule subset using greedy MDL optimization.

        Algorithm:
        1. Score all rules
        2. Sort by net benefit descending
        3. Greedily add rules while benefit > 0 and under target size

        Args:
            grammar: SequiturGrammar with induced rules
            verbose: Print progress

        Returns:
            (selected_rule_ids, stats)
        """
        if verbose:
            print(f"\n{'='*70}")
            print("MDL VOCABULARY OPTIMIZATION")
            print(f"{'='*70}")
            print(f"  Total rules in grammar: {len(grammar.rules)}")
            print(f"  Target vocabulary size: {self.target_size}")

        rule_stats = grammar.get_rule_stats()

        # Score all rules (except start rule 0)
        scores = []
        initial_vocab_size = len(grammar.rules)

        for rule_id, rule in grammar.rules.items():
            if rule_id == 0:  # Skip start rule
                continue

            stats = rule_stats[rule_id]
            if stats['usage_count'] < self.min_usage:
                continue

            score = self.score_rule(rule, initial_vocab_size, rule_stats)
            scores.append(score)

        if verbose:
            print(f"  Rules meeting min_usage ({self.min_usage}): {len(scores)}")

        # Sort by net benefit descending
        scores.sort(key=lambda s: s.net_benefit, reverse=True)

        # Greedy selection
        selected = set()
        total_benefit = 0.0
        total_savings = 0.0

        for score in scores:
            if len(selected) >= self.target_size - 1:  # -1 for start rule
                break

            if score.net_benefit < self.min_benefit:
                break

            selected.add(score.rule_id)
            total_benefit += score.net_benefit
            total_savings += score.total_savings

        if verbose:
            print(f"  Selected rules: {len(selected)}")
            print(f"  Total net benefit: {total_benefit:.1f} bits")

            if scores:
                # Show top 10 selected
                print(f"\n  Top 10 rules by benefit:")
                for score in scores[:10]:
                    status = "+" if score.rule_id in selected else "-"
                    print(f"    {status} R{score.rule_id}: benefit={score.net_benefit:.1f}, "
                          f"usage={score.usage_count}, depth={score.depth}")

        # Compute stats
        stats = VocabularyStats(
            total_rules=len(grammar.rules),
            selected_rules=len(selected) + 1,  # +1 for start rule
            total_tokens=rule_stats[0]['expansion_length'] if 0 in rule_stats else 0,
            encoded_tokens=0,  # Would need to re-encode to compute
            compression_ratio=0.0,
            total_bits_saved=total_savings,
            vocabulary_cost=sum(s.definition_cost for s in scores if s.rule_id in selected),
            max_depth=max((s.depth for s in scores if s.rule_id in selected), default=0),
            avg_usage=np.mean([s.usage_count for s in scores if s.rule_id in selected]) if selected else 0
        )

        # Estimate compression ratio
        original_bits = stats.total_tokens * self.bits_per_terminal
        if original_bits > 0:
            stats.compression_ratio = (original_bits - total_savings) / original_bits

        return selected, stats

    def prune_grammar(self,
                      grammar: SequiturGrammar,
                      selected_rules: Set[int]) -> Dict[int, Rule]:
        """
        Create a pruned grammar containing only selected rules.

        Args:
            grammar: Original grammar
            selected_rules: Set of rule IDs to keep

        Returns:
            Dict mapping rule_id -> Rule for selected rules
        """
        pruned = {}

        # Always include start rule
        pruned[0] = grammar.rules[0]

        for rule_id in selected_rules:
            if rule_id in grammar.rules:
                pruned[rule_id] = grammar.rules[rule_id]

        return pruned


@dataclass
class PatternCandidate:
    """A candidate pattern from grammar rules."""
    rule_id: int
    expansion: List[int]  # Terminal sequence
    usage_count: int
    depth: int
    mdl_benefit: float

    @property
    def length(self) -> int:
        return len(self.expansion)


def extract_patterns_from_grammar(grammar: SequiturGrammar,
                                   selected_rules: Set[int],
                                   verbose: bool = True) -> List[PatternCandidate]:
    """
    Extract pattern candidates from grammar rules.

    These patterns can be integrated with the existing transform discovery.

    Args:
        grammar: SEQUITUR grammar
        selected_rules: MDL-selected rule IDs
        verbose: Print progress

    Returns:
        List of PatternCandidate representing discovered patterns
    """
    if verbose:
        print(f"\n  Extracting patterns from {len(selected_rules)} rules...")

    rule_stats = grammar.get_rule_stats()
    optimizer = VocabularyOptimizer()

    patterns = []
    for rule_id in selected_rules:
        if rule_id == 0:
            continue

        rule = grammar.rules[rule_id]
        stats = rule_stats[rule_id]
        score = optimizer.score_rule(rule, len(grammar.rules), rule_stats)

        pattern = PatternCandidate(
            rule_id=rule_id,
            expansion=rule.expand(),
            usage_count=stats['usage_count'],
            depth=stats['depth'],
            mdl_benefit=score.net_benefit
        )
        patterns.append(pattern)

    # Sort by MDL benefit
    patterns.sort(key=lambda p: p.mdl_benefit, reverse=True)

    if verbose:
        print(f"  Extracted {len(patterns)} patterns")
        if patterns:
            print(f"  Length range: {min(p.length for p in patterns)} - {max(p.length for p in patterns)}")
            print(f"  Depth range: {min(p.depth for p in patterns)} - {max(p.depth for p in patterns)}")

    return patterns


def run_grammar_discovery_pipeline(objects: List,
                                    target_vocab_size: int = 400,
                                    verbose: bool = True) -> Tuple[SequiturGrammar, Set[int], List[PatternCandidate]]:
    """
    Full pipeline: SEQUITUR induction + MDL optimization + pattern extraction.

    This replaces the hierarchical composition discovery phases with
    grammar-based pattern discovery.

    Args:
        objects: List of FactoredObjects
        target_vocab_size: Target number of patterns (300-400)
        verbose: Print progress

    Returns:
        (grammar, selected_rule_ids, patterns)
    """
    from .sequitur import build_grammar_from_corpus

    # Phase 1: Build grammar with SEQUITUR
    grammar = build_grammar_from_corpus(objects, verbose=verbose)

    # Phase 2: MDL optimization
    optimizer = VocabularyOptimizer(target_size=target_vocab_size)
    selected_rules, stats = optimizer.optimize(grammar, verbose=verbose)

    if verbose:
        print(f"\n  Vocabulary Stats:")
        print(f"    Total rules: {stats.total_rules}")
        print(f"    Selected: {stats.selected_rules}")
        print(f"    Compression ratio: {stats.compression_ratio:.2%}")
        print(f"    Max depth: {stats.max_depth}")

    # Phase 3: Extract patterns
    patterns = extract_patterns_from_grammar(grammar, selected_rules, verbose=verbose)

    return grammar, selected_rules, patterns


# =============================================================================
# INTEGRATION WITH EXISTING PIPELINE
# =============================================================================

def integrate_grammar_patterns_with_transforms(patterns: List[PatternCandidate],
                                                 existing_vocab: Dict,
                                                 verbose: bool = True) -> Dict:
    """
    Integrate grammar-discovered patterns with existing transform vocabulary.

    This allows combining:
    - Algebraic transforms (D24, rhythm group)
    - SEQUITUR-discovered repetition patterns
    - Cross-track relationships

    Args:
        patterns: Patterns from grammar discovery
        existing_vocab: Existing transform vocabulary dict
        verbose: Print progress

    Returns:
        Combined vocabulary
    """
    if verbose:
        print(f"\n{'='*70}")
        print("INTEGRATING GRAMMAR PATTERNS WITH TRANSFORMS")
        print(f"{'='*70}")
        print(f"  Grammar patterns: {len(patterns)}")
        print(f"  Existing vocabulary size: {len(existing_vocab)}")

    combined = dict(existing_vocab)

    # Add grammar patterns as new vocabulary entries
    pattern_offset = max(existing_vocab.keys(), default=-1) + 1

    for i, pattern in enumerate(patterns):
        vocab_id = pattern_offset + i
        combined[vocab_id] = {
            'type': 'grammar_pattern',
            'rule_id': pattern.rule_id,
            'expansion': pattern.expansion,
            'usage_count': pattern.usage_count,
            'depth': pattern.depth,
            'mdl_benefit': pattern.mdl_benefit
        }

    if verbose:
        print(f"  Combined vocabulary size: {len(combined)}")

    return combined
