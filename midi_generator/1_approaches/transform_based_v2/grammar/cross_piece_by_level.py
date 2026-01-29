"""
Cross-Piece Analysis by Hierarchy Level
========================================

Analyzes pattern sharing across pieces at each semantic level.

Hypothesis: Sharing might be HIGHER at motif/phrase level than at interval
level (intervals are too granular) or section level (too piece-specific).

Usage:
    python cross_piece_by_level.py checkpoint_v2.npz
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from grammar.level_api import GrammarHierarchy, SEMANTIC_LEVELS


@dataclass
class LevelSharingStats:
    """Sharing statistics for a semantic level."""
    level: str
    depth_range: Tuple[int, int]
    total_rules: int
    single_piece_rules: int  # Rules used in only 1 piece
    multi_piece_rules: int   # Rules used in 2+ pieces
    universal_rules: int     # Rules used in 50%+ pieces
    sharing_rate: float      # multi_piece / total
    universal_rate: float    # universal / total
    avg_pieces_per_rule: float


def analyze_cross_piece_by_level(checkpoint_path: str) -> Dict[str, LevelSharingStats]:
    """
    Analyze cross-piece pattern sharing at each hierarchy level.

    Uses grammar structure to estimate sharing via rule reference counts.
    Rules referenced by many different parent rules are more "shared".

    Returns:
        Dict mapping semantic level -> LevelSharingStats
    """
    # Load grammar
    grammar = GrammarHierarchy.from_checkpoint(checkpoint_path)

    # Always use grammar-based analysis (more reliable than encoding tokens)
    return _analyze_from_grammar_only(grammar)


def _extract_rules_from_tokens(tokens, piece_id: int, rule_to_pieces: Dict[str, Set[int]]):
    """Extract rule references from tokens."""
    if isinstance(tokens, list):
        for token in tokens:
            if isinstance(token, list) and len(token) >= 2:
                if token[0] == 'R' or token[0] == 'INTRO':
                    rule_id = str(token[1])
                    rule_to_pieces[rule_id].add(piece_id)
                elif token[0] in ('REPEAT', 'TRANSFORM'):
                    # These reference patterns, not rules directly
                    pass
            elif isinstance(token, dict):
                if 'rule' in token:
                    rule_to_pieces[str(token['rule'])].add(piece_id)
                if 'pattern' in token:
                    rule_to_pieces[str(token['pattern'])].add(piece_id)
    elif isinstance(tokens, dict):
        for key, value in tokens.items():
            _extract_rules_from_tokens(value, piece_id, rule_to_pieces)


def _analyze_from_grammar_only(grammar: GrammarHierarchy) -> Dict[str, LevelSharingStats]:
    """
    Analyze sharing based on grammar structure alone.
    Uses usage count proxies from rule references.
    """
    # Count how many times each rule is referenced by other rules
    rule_usage: Dict[str, int] = defaultdict(int)
    rule_parents: Dict[str, Set[str]] = defaultdict(set)

    for rule_id, rule in grammar.rules.items():
        for child_id in rule.children:
            rule_usage[child_id] += 1
            rule_parents[child_id].add(rule_id)

    # Estimate "pieces" by looking at top-level rules (high depth)
    max_depth = grammar.get_max_depth()
    top_rules = grammar.by_depth.get(max_depth, []) + grammar.by_depth.get(max_depth - 1, [])

    results = {}
    level_depths = defaultdict(list)
    for depth, level in SEMANTIC_LEVELS.items():
        level_depths[level].append(depth)

    for level, depths in level_depths.items():
        rules_at_level = []
        for d in depths:
            rules_at_level.extend(grammar.by_depth.get(d, []))

        if not rules_at_level:
            continue

        single_piece = 0
        multi_piece = 0
        universal = 0
        total_usage = 0

        for rule_id in rules_at_level:
            usage = rule_usage.get(rule_id, 0)
            total_usage += usage

            if usage <= 1:
                single_piece += 1
            elif usage >= 2:
                multi_piece += 1
                if usage >= 10:  # Proxy for "universal"
                    universal += 1

        total = len(rules_at_level)
        results[level] = LevelSharingStats(
            level=level,
            depth_range=(min(depths), max(depths)),
            total_rules=total,
            single_piece_rules=single_piece,
            multi_piece_rules=multi_piece,
            universal_rules=universal,
            sharing_rate=multi_piece / total if total > 0 else 0,
            universal_rate=universal / total if total > 0 else 0,
            avg_pieces_per_rule=total_usage / total if total > 0 else 0,
        )

    return results


def print_sharing_report(results: Dict[str, LevelSharingStats]):
    """Print formatted sharing report."""
    print("\n" + "=" * 70)
    print("CROSS-PIECE SHARING ANALYSIS BY HIERARCHY LEVEL")
    print("=" * 70)

    # Sort by sharing rate
    sorted_levels = sorted(results.items(), key=lambda x: x[1].sharing_rate, reverse=True)

    print(f"\n{'Level':<15} {'Depth':<8} {'Total':<8} {'Multi-Piece':<12} {'Share Rate':<12} {'Universal':<10}")
    print("-" * 70)

    for level, stats in sorted_levels:
        depth_str = f"{stats.depth_range[0]}-{stats.depth_range[1]}"
        print(f"{level:<15} {depth_str:<8} {stats.total_rules:<8} "
              f"{stats.multi_piece_rules:<12} {stats.sharing_rate*100:>6.1f}%      "
              f"{stats.universal_rules:<10}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best_level = max(results.items(), key=lambda x: x[1].sharing_rate)
    worst_level = min(results.items(), key=lambda x: x[1].sharing_rate)

    print(f"\nHighest sharing: {best_level[0].upper()} ({best_level[1].sharing_rate*100:.1f}%)")
    print(f"Lowest sharing:  {worst_level[0].upper()} ({worst_level[1].sharing_rate*100:.1f}%)")

    # Test hypothesis
    motif_stats = results.get('motif')
    phrase_stats = results.get('phrase')
    interval_stats = results.get('interval')
    section_stats = results.get('section')

    if motif_stats and interval_stats and interval_stats.sharing_rate > 0:
        print(f"\nMotif vs Interval sharing: {motif_stats.sharing_rate/interval_stats.sharing_rate:.2f}x")

    if phrase_stats and section_stats and section_stats.sharing_rate > 0:
        print(f"Phrase vs Section sharing: {phrase_stats.sharing_rate/section_stats.sharing_rate:.2f}x")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cross-piece analysis by hierarchy level')
    parser.add_argument('checkpoint', help='Path to checkpoint NPZ file')
    parser.add_argument('--output', '-o', help='Output JSON file')
    args = parser.parse_args()

    results = analyze_cross_piece_by_level(args.checkpoint)
    print_sharing_report(results)

    if args.output:
        output = {
            level: {
                'level': stats.level,
                'depth_range': stats.depth_range,
                'total_rules': stats.total_rules,
                'single_piece_rules': stats.single_piece_rules,
                'multi_piece_rules': stats.multi_piece_rules,
                'universal_rules': stats.universal_rules,
                'sharing_rate': stats.sharing_rate,
                'universal_rate': stats.universal_rate,
                'avg_pieces_per_rule': stats.avg_pieces_per_rule,
            }
            for level, stats in results.items()
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")
