"""
Grammar Hierarchy Analysis for V32
==================================

Analyzes the existing SEQUITUR grammar hierarchy in V32 checkpoint.
No new algorithms needed - just surfaces what's already there.
"""

import numpy as np
import json
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class HierarchyStats:
    """Statistics about grammar hierarchy."""
    max_depth: int
    n_rules: int
    rules_by_depth: Dict[int, int]  # depth -> count
    avg_rule_size: float
    avg_child_refs: float


def load_grammar_from_checkpoint(checkpoint_path: str) -> Dict:
    """Load grammar rules from checkpoint."""
    data = np.load(checkpoint_path, allow_pickle=True)
    return json.loads(data['grammar_rules_json'][0])


def compute_rule_depths(rules: Dict) -> Dict[str, int]:
    """
    Compute depth of each rule (longest path to terminal).

    Depth 1 = rule with only terminals
    Depth N = rule whose deepest child has depth N-1
    """
    depths = {}

    def get_depth(rule_id: str) -> int:
        if rule_id in depths:
            return depths[rule_id]

        if rule_id not in rules:
            # Terminal
            depths[rule_id] = 0
            return 0

        rule = rules[rule_id]

        # Find all child rule references
        child_depths = []
        for element in rule:
            if isinstance(element, list) and len(element) == 2 and element[0] == 'R':
                child_id = str(element[1])
                child_depths.append(get_depth(child_id))

        if child_depths:
            depths[rule_id] = max(child_depths) + 1
        else:
            depths[rule_id] = 1

        return depths[rule_id]

    for rule_id in rules:
        get_depth(rule_id)

    return depths


def group_rules_by_depth(rules: Dict, depths: Dict[str, int]) -> Dict[int, List[str]]:
    """Group rule IDs by their depth level."""
    groups = defaultdict(list)
    for rule_id, depth in depths.items():
        if rule_id in rules:  # Only actual rules
            groups[depth].append(rule_id)
    return dict(groups)


def compute_rule_stats(rule: List) -> Tuple[int, int]:
    """Compute (n_terminals, n_rule_refs) for a rule."""
    n_terminals = sum(1 for e in rule if not (isinstance(e, list) and e[0] == 'R'))
    n_rules = sum(1 for e in rule if isinstance(e, list) and e[0] == 'R')
    return n_terminals, n_rules


def expand_rule_to_terminals(rules: Dict, rule_id: str, memo: Dict = None) -> List[int]:
    """Recursively expand a rule to its terminal sequence."""
    if memo is None:
        memo = {}

    if rule_id in memo:
        return memo[rule_id]

    if rule_id not in rules:
        # Terminal
        try:
            return [int(rule_id)]
        except ValueError:
            return []

    rule = rules[rule_id]
    result = []

    for element in rule:
        if isinstance(element, list) and len(element) == 2 and element[0] == 'R':
            child_id = str(element[1])
            result.extend(expand_rule_to_terminals(rules, child_id, memo))
        elif isinstance(element, int):
            result.append(element)

    memo[rule_id] = result
    return result


def analyze_hierarchy(checkpoint_path: str, verbose: bool = True) -> HierarchyStats:
    """
    Analyze grammar hierarchy from checkpoint.

    Returns:
        HierarchyStats with depth distribution
    """
    rules = load_grammar_from_checkpoint(checkpoint_path)
    depths = compute_rule_depths(rules)
    depth_groups = group_rules_by_depth(rules, depths)

    # Compute stats
    total_terminals = 0
    total_refs = 0
    for rule_id, rule in rules.items():
        n_term, n_ref = compute_rule_stats(rule)
        total_terminals += n_term
        total_refs += n_ref

    stats = HierarchyStats(
        max_depth=max(depths.values()),
        n_rules=len(rules),
        rules_by_depth={d: len(r) for d, r in depth_groups.items()},
        avg_rule_size=total_terminals / len(rules),
        avg_child_refs=total_refs / len(rules),
    )

    if verbose:
        print(f"=== GRAMMAR HIERARCHY ANALYSIS ===")
        print(f"Total rules: {stats.n_rules}")
        print(f"Max depth: {stats.max_depth}")
        print(f"Avg terminals per rule: {stats.avg_rule_size:.2f}")
        print(f"Avg rule refs per rule: {stats.avg_child_refs:.2f}")
        print(f"\nRules by depth:")
        for depth in sorted(stats.rules_by_depth.keys()):
            count = stats.rules_by_depth[depth]
            pct = 100 * count / stats.n_rules
            bar = "█" * int(pct / 2)
            print(f"  Depth {depth:2d}: {count:5d} ({pct:5.1f}%) {bar}")

    return stats


def get_rules_at_depth(checkpoint_path: str, depth: int) -> List[Tuple[str, List]]:
    """Get all rules at a specific depth level."""
    rules = load_grammar_from_checkpoint(checkpoint_path)
    depths = compute_rule_depths(rules)

    return [(rule_id, rules[rule_id])
            for rule_id, d in depths.items()
            if d == depth and rule_id in rules]


def get_deepest_rules(checkpoint_path: str, n: int = 10) -> List[Tuple[str, int, List[int]]]:
    """
    Get the N deepest rules with their terminal expansions.

    Returns:
        List of (rule_id, depth, terminal_expansion)
    """
    rules = load_grammar_from_checkpoint(checkpoint_path)
    depths = compute_rule_depths(rules)

    # Sort by depth descending
    sorted_rules = sorted(
        [(rid, d) for rid, d in depths.items() if rid in rules],
        key=lambda x: x[1],
        reverse=True
    )[:n]

    memo = {}
    result = []
    for rule_id, depth in sorted_rules:
        expansion = expand_rule_to_terminals(rules, rule_id, memo)
        result.append((rule_id, depth, expansion))

    return result


def semantic_level_interpretation(depth: int, max_depth: int = 12) -> str:
    """
    Interpret depth level in musical terms.

    Based on typical musical hierarchy:
    - Depth 1-2: Intervals, small motifs (2-4 notes)
    - Depth 3-4: Motifs, short phrases (4-8 notes)
    - Depth 5-6: Phrases (8-16 notes)
    - Depth 7-8: Sections, periods (16-32 notes)
    - Depth 9+: Large-scale structure
    """
    if depth <= 2:
        return "interval/atomic"
    elif depth <= 4:
        return "motif"
    elif depth <= 6:
        return "phrase"
    elif depth <= 8:
        return "section"
    else:
        return "large-scale"


if __name__ == '__main__':
    import sys

    checkpoint = sys.argv[1] if len(sys.argv) > 1 else 'checkpoint_v2.npz'

    # Analyze hierarchy
    stats = analyze_hierarchy(checkpoint, verbose=True)

    # Show deepest rules
    print(f"\n=== DEEPEST RULES ===")
    deepest = get_deepest_rules(checkpoint, n=5)
    for rule_id, depth, expansion in deepest:
        level = semantic_level_interpretation(depth)
        print(f"R{rule_id} (depth {depth}, {level}): {len(expansion)} terminals")
        if len(expansion) <= 20:
            print(f"  Expansion: {expansion}")
        else:
            print(f"  Expansion: {expansion[:10]} ... {expansion[-10:]}")

    # Show interpretation
    print(f"\n=== SEMANTIC INTERPRETATION ===")
    for depth in sorted(stats.rules_by_depth.keys()):
        level = semantic_level_interpretation(depth)
        count = stats.rules_by_depth[depth]
        print(f"Depth {depth:2d} ({level:12s}): {count} rules")
