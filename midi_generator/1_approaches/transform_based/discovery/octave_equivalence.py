"""
Octave Equivalence Discovery (Phase 5g)
=======================================

MDL-based test to discover whether octave equivalence improves compression
for the current corpus.

Philosophy: Let the data decide.
- Re-Pair stores EXACT intervals (+2, +14 are different)
- This phase tests: "Does treating +2 ≈ +14 reduce description length?"
- If yes → octave equivalence is real for this corpus
- If no → distinct intervals are preferred

This is DISCOVERY, not prescription:
- We don't assume octave equivalence (like old mod-12)
- We test it and report the MDL benefit
- Different corpora may have different results:
  - Big band (lots of octave doublings) → likely benefits
  - Atonal music (register matters) → likely no benefit
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


def signed_mod12(interval: int) -> int:
    """
    Compute signed mod-12 that preserves direction.

    +14 → +2 (ascending major 2nd + octave → ascending major 2nd)
    -14 → -2 (descending major 2nd + octave → descending major 2nd)
    +12 → 0  (octave → unison)
    -12 → 0  (octave down → unison)
    """
    if interval == 0:
        return 0
    sign = 1 if interval > 0 else -1
    magnitude = abs(interval) % 12
    # Handle octave exactly
    if magnitude == 0:
        return 0  # Octaves become unisons
    return sign * magnitude


def compute_octave_offset(interval: int) -> int:
    """
    Compute the octave offset from the normalized interval.

    +14 → +1 octave offset
    +2  → 0 octave offset
    -14 → -1 octave offset
    +24 → +2 octave offset
    """
    if interval == 0:
        return 0
    sign = 1 if interval > 0 else -1
    octaves = abs(interval) // 12
    return sign * octaves


def run_octave_equivalence_discovery(
    patterns: List[Dict],
    grammar_intervals: Optional[List[int]] = None,
    min_benefit_bits: float = 100.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Test whether octave equivalence improves compression via MDL.

    Args:
        patterns: List of pattern dicts with 'pitch_intervals' and 'count'
        grammar_intervals: Optional list of all rule intervals from grammar
        min_benefit_bits: Minimum MDL benefit to declare equivalence useful
        verbose: Print progress

    Returns:
        Dict with:
        - 'octave_equivalence_useful': bool
        - 'mdl_benefit_bits': float (positive = equivalence helps)
        - 'distinct_patterns': int
        - 'equivalent_classes': int
        - 'octave_offset_bits': float (cost of storing offsets)
        - 'equivalence_classes': Dict mapping normalized tuple -> list of pattern indices
    """
    if verbose:
        print("=" * 60)
        print("OCTAVE EQUIVALENCE MDL TEST")
        print("=" * 60)
        print(f"Input: {len(patterns)} patterns")

    # Collect all intervals from patterns
    all_intervals = []
    pattern_intervals_list = []  # List of (pattern_idx, intervals_tuple)

    for i, p in enumerate(patterns):
        intervals = p.get('pitch_intervals', [])
        if intervals:
            intervals_tuple = tuple(intervals)
            pattern_intervals_list.append((i, intervals_tuple))
            all_intervals.extend(intervals)

    # Add grammar intervals if provided
    if grammar_intervals:
        all_intervals.extend(grammar_intervals)

    if not all_intervals:
        if verbose:
            print("  No intervals found, skipping test")
        return {
            'octave_equivalence_useful': False,
            'mdl_benefit_bits': 0.0,
            'distinct_patterns': 0,
            'equivalent_classes': 0,
            'octave_offset_bits': 0.0,
            'equivalence_classes': {},
        }

    # Count distinct intervals
    distinct_intervals = set(all_intervals)
    n_distinct = len(distinct_intervals)

    # Compute equivalence classes (mod-12)
    interval_to_class = {}
    class_members = defaultdict(list)

    for interval in distinct_intervals:
        normalized = signed_mod12(interval)
        interval_to_class[interval] = normalized
        class_members[normalized].append(interval)

    n_classes = len(class_members)

    if verbose:
        print(f"  Distinct intervals: {n_distinct}")
        print(f"  Equivalence classes: {n_classes}")
        print(f"  Reduction: {n_distinct} -> {n_classes} ({100*(1 - n_classes/n_distinct):.1f}% fewer)")

    # Group patterns by their normalized interval sequence
    pattern_equivalence = defaultdict(list)  # normalized_tuple -> [pattern_indices]

    for i, intervals_tuple in pattern_intervals_list:
        normalized_tuple = tuple(signed_mod12(iv) for iv in intervals_tuple)
        pattern_equivalence[normalized_tuple].append(i)

    n_distinct_patterns = len(pattern_intervals_list)
    n_pattern_classes = len(pattern_equivalence)

    if verbose:
        print(f"\n  Pattern interval sequences:")
        print(f"    Distinct: {n_distinct_patterns}")
        print(f"    After equivalence: {n_pattern_classes}")

    # === MDL Cost Comparison ===

    # Cost 1: Distinct intervals (current)
    # Each interval needs log2(n_distinct) bits to specify
    distinct_cost_per_interval = math.log2(n_distinct) if n_distinct > 1 else 0
    total_distinct_cost = distinct_cost_per_interval * len(all_intervals)

    # Cost 2: Equivalence classes + octave offsets
    # Each normalized interval needs log2(n_classes) bits
    # Plus each octave offset needs bits to specify
    equiv_cost_per_interval = math.log2(n_classes) if n_classes > 1 else 0

    # Octave offsets: compute actual distribution
    octave_offsets = [compute_octave_offset(iv) for iv in all_intervals]
    offset_counts = defaultdict(int)
    for offset in octave_offsets:
        offset_counts[offset] += 1

    # Entropy-based cost for octave offsets
    n_total = len(octave_offsets)
    offset_entropy = 0.0
    for count in offset_counts.values():
        if count > 0:
            p = count / n_total
            offset_entropy -= p * math.log2(p)

    # Total equivalence cost: class specification + offset specification
    total_equiv_cost = equiv_cost_per_interval * len(all_intervals)
    total_offset_cost = offset_entropy * len(all_intervals)
    total_with_equiv = total_equiv_cost + total_offset_cost

    # MDL benefit (positive = equivalence helps)
    mdl_benefit = total_distinct_cost - total_with_equiv

    if verbose:
        print(f"\n  MDL Analysis:")
        print(f"    Distinct cost: {total_distinct_cost:.1f} bits")
        print(f"      ({distinct_cost_per_interval:.2f} bits/interval x {len(all_intervals)} intervals)")
        print(f"    Equivalence cost: {total_with_equiv:.1f} bits")
        print(f"      Class: {total_equiv_cost:.1f} bits ({equiv_cost_per_interval:.2f} bits/interval)")
        print(f"      Octave offset: {total_offset_cost:.1f} bits ({offset_entropy:.2f} bits/interval)")
        print(f"    MDL benefit: {mdl_benefit:.1f} bits")

    # Report octave offset distribution
    if verbose:
        print(f"\n  Octave offset distribution:")
        for offset in sorted(offset_counts.keys()):
            count = offset_counts[offset]
            pct = 100 * count / n_total
            print(f"    {offset:+d} octave(s): {count:,} ({pct:.1f}%)")

    # Check multi-octave patterns
    multi_octave_patterns = []
    for i, intervals_tuple in pattern_intervals_list:
        has_multi = any(abs(iv) >= 12 for iv in intervals_tuple)
        if has_multi:
            multi_octave_patterns.append((i, intervals_tuple))

    if verbose and multi_octave_patterns:
        print(f"\n  Patterns with compound intervals (>= octave): {len(multi_octave_patterns)}")
        for i, intervals in multi_octave_patterns[:5]:
            print(f"    Pattern {i}: {list(intervals)}")
        if len(multi_octave_patterns) > 5:
            print(f"    ... and {len(multi_octave_patterns) - 5} more")

    # Report equivalence class groupings
    merged_classes = [(k, v) for k, v in pattern_equivalence.items() if len(v) > 1]
    merged_classes.sort(key=lambda x: -len(x[1]))

    if verbose and merged_classes:
        print(f"\n  Patterns merged by octave equivalence: {len(merged_classes)} groups")
        for normalized, indices in merged_classes[:5]:
            original_intervals = [pattern_intervals_list[idx][1] for idx in indices if idx < len(pattern_intervals_list)]
            print(f"    {list(normalized)}: {len(indices)} patterns merged")
            for orig in original_intervals[:3]:
                print(f"      <- {list(orig)}")
            if len(original_intervals) > 3:
                print(f"      ... and {len(original_intervals) - 3} more")
        if len(merged_classes) > 5:
            print(f"    ... and {len(merged_classes) - 5} more groups")

    # Decision
    is_useful = mdl_benefit >= min_benefit_bits

    if verbose:
        print(f"\n  Decision: Octave equivalence is {'USEFUL' if is_useful else 'NOT useful'}")
        print(f"    MDL benefit {mdl_benefit:.1f} {'>' if is_useful else '<'} threshold {min_benefit_bits}")

    return {
        'octave_equivalence_useful': is_useful,
        'mdl_benefit_bits': mdl_benefit,
        'distinct_patterns': n_distinct_patterns,
        'equivalent_classes': n_pattern_classes,
        'distinct_intervals': n_distinct,
        'interval_classes': n_classes,
        'octave_offset_bits': total_offset_cost,
        'octave_offset_entropy': offset_entropy,
        'octave_offset_distribution': dict(offset_counts),
        'equivalence_classes': {
            str(k): v for k, v in pattern_equivalence.items()
        },
        'multi_octave_pattern_count': len(multi_octave_patterns),
    }


def test_octave_equivalence_on_grammar(
    grammar,
    min_benefit_bits: float = 100.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Test octave equivalence directly on grammar rule intervals.

    Args:
        grammar: FactoredHierarchicalGrammar with rule_pitch_intervals
        min_benefit_bits: Minimum MDL benefit threshold
        verbose: Print progress

    Returns:
        Dict with MDL analysis results
    """
    # Extract all rule intervals
    intervals = grammar.rule_pitch_intervals.cpu().numpy().tolist()

    # Create pseudo-patterns for the test
    patterns = [
        {'pitch_intervals': [iv], 'count': grammar.rule_counts[i].item()}
        for i, iv in enumerate(intervals)
    ]

    return run_octave_equivalence_discovery(
        patterns=patterns,
        grammar_intervals=intervals,
        min_benefit_bits=min_benefit_bits,
        verbose=verbose,
    )
