#!/usr/bin/env python3
"""
Convert v28 T-norm Re-Pair to GenomeGraph format.

The T-norm Re-Pair output already encodes interval-based transforms:
- rule_table[i] = (0, interval) where interval is pitch difference
- This directly maps to T transforms: interval +3 → T3

This script:
1. Expands T-norm rules into full patterns (pitch_classes)
2. Discovers transform edges between patterns (T, I, R)
3. Builds GenomeGraph with temporal edges from occurrence data
4. Saves in genome graph format ready for gene editor

Usage:
    python scripts/convert_v28_to_genome.py checkpoint_v28_tnorm_repair.npz --output checkpoint_v28_graph.npz
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grammar.genome_graph import (
    GenomeGraph, Pattern, Edge,
    find_pitch_transform, quantize_time_delta,
)


@dataclass
class TnormRule:
    """A T-normalized Re-Pair rule."""
    id: int
    interval: int  # The canonical interval (0, interval)
    count: int  # Usage count
    expansion: List[int]  # Full terminal expansion (pitch classes)


def expand_tnorm_rules(
    rule_table: np.ndarray,
    rule_counts: np.ndarray,
    n_terminals: int,
    final_sequence: np.ndarray,
    verbose: bool = True
) -> Dict[int, TnormRule]:
    """
    Expand T-norm Re-Pair rules to get full terminal expansions.

    T-norm rules are stored as (0, interval) pairs.
    We need to trace through the final_sequence to get actual patterns.

    Args:
        rule_table: Shape [n_rules, 2], each row is (0, interval)
        rule_counts: Shape [n_rules], usage counts
        n_terminals: Number of terminal symbols (12 for pitch classes)
        final_sequence: Compressed sequence with rule IDs
        verbose: Print progress

    Returns:
        Dict of rule_id -> TnormRule with full expansions
    """
    n_rules = len(rule_table)

    if verbose:
        print(f"Expanding {n_rules} T-norm rules...")

    # Build rule expansion table
    # For T-norm, rules are binary: each rule replaces (left, left+interval)
    # But they're nested - we need recursive expansion

    rules = {}

    # First pass: create basic rules from table
    for i in range(n_rules):
        rule_id = n_terminals + i
        left, right = rule_table[i]
        interval = right - left  # Should equal right since left=0

        rules[rule_id] = TnormRule(
            id=rule_id,
            interval=int(interval),
            count=int(rule_counts[i]),
            expansion=[int(left), int(right)],  # Initial: just the pair
        )

    # Now expand rules recursively
    # In Re-Pair, a rule can contain other rules as symbols

    def expand_symbol(symbol: int, memo: Dict[int, List[int]]) -> List[int]:
        """Recursively expand a symbol to terminals."""
        if symbol in memo:
            return memo[symbol]

        if symbol < n_terminals:
            return [symbol]  # Terminal

        if symbol not in rules:
            return [symbol % n_terminals]  # Unknown rule, treat as terminal

        rule = rules[symbol]
        # The rule expansion contains left and right symbols
        # which may themselves be rules
        result = []
        for s in [rule.expansion[0], rule.expansion[1]]:
            if s < n_terminals:
                result.append(s)
            else:
                result.extend(expand_symbol(s, memo))

        memo[symbol] = result
        return result

    # Expand all rules
    memo = {}
    for rule_id in rules:
        rules[rule_id].expansion = expand_symbol(rule_id, memo)

    if verbose:
        # Show sample expansions
        sorted_rules = sorted(rules.values(), key=lambda r: r.count, reverse=True)[:10]
        print(f"Top 10 rules by count:")
        for r in sorted_rules:
            print(f"  R{r.id}: interval={r.interval:+d}, count={r.count:,}, len={len(r.expansion)}")

    return rules


def extract_patterns_from_sequence(
    final_sequence: np.ndarray,
    rules: Dict[int, TnormRule],
    n_terminals: int,
    file_names: np.ndarray,
    verbose: bool = True
) -> Tuple[Dict[int, Pattern], Dict[int, List[Dict]]]:
    """
    Extract patterns and their occurrences from the final sequence.

    The final_sequence contains interleaved track data with separators (-1).
    We extract:
    1. Unique patterns (rule expansions)
    2. Where each pattern occurs (piece, track, position)

    Args:
        final_sequence: Compressed sequence with rule IDs and terminals
        rules: Expanded rule definitions
        n_terminals: Terminal vocabulary size (12)
        file_names: Track/file names for occurrence tracking
        verbose: Print progress

    Returns:
        (patterns, occurrences)
        - patterns: Dict of pattern_id -> Pattern
        - occurrences: Dict of pattern_id -> list of occurrence dicts
    """
    if verbose:
        print(f"Extracting patterns from sequence of length {len(final_sequence):,}...")

    patterns = {}
    occurrences = defaultdict(list)

    # Track boundaries in the sequence (separated by -1)
    separator = -1
    track_boundaries = [-1] + list(np.where(final_sequence == separator)[0]) + [len(final_sequence)]

    if verbose:
        print(f"Found {len(track_boundaries)-1} tracks")

    # Process each track
    position = 0
    for track_idx in range(len(track_boundaries) - 1):
        start = track_boundaries[track_idx] + 1
        end = track_boundaries[track_idx + 1]

        if start >= end:
            continue

        track_seq = final_sequence[start:end]

        # Get file name for this track
        if track_idx < len(file_names):
            piece_id = str(file_names[track_idx])
        else:
            piece_id = f"track_{track_idx}"

        # Track position in original sequence
        local_pos = 0

        for symbol in track_seq:
            if symbol < 0:
                continue  # Skip separators

            if symbol >= n_terminals and symbol in rules:
                # This is a rule - record as pattern
                rule = rules[symbol]
                pattern_id = symbol

                if pattern_id not in patterns:
                    # Create pattern
                    patterns[pattern_id] = Pattern(
                        id=pattern_id,
                        pitch_classes=rule.expansion,
                        octaves=[4] * len(rule.expansion),  # Default octave
                        velocities=[80] * len(rule.expansion),  # Default velocity
                        durations=[480] * len(rule.expansion),  # Default duration (quarter note)
                        rhythm_ioi=[480] * max(0, len(rule.expansion) - 1),  # Default IOI
                    )

                # Record occurrence
                occurrences[pattern_id].append({
                    'piece_id': piece_id,
                    'track_id': track_idx,
                    'onset_time': local_pos * 480,  # Approximate onset
                })

                local_pos += len(rule.expansion)
            else:
                # Terminal symbol
                local_pos += 1

    if verbose:
        print(f"Extracted {len(patterns)} unique patterns")
        print(f"Total occurrences: {sum(len(v) for v in occurrences.values()):,}")

    return patterns, dict(occurrences)


def discover_transform_edges(
    patterns: Dict[int, Pattern],
    verbose: bool = True
) -> List[Dict]:
    """
    Discover transform edges between patterns.

    For each pair of patterns, check if one is a T, I, or R transform of the other.

    Args:
        patterns: Dict of pattern_id -> Pattern
        verbose: Print progress

    Returns:
        List of edge dicts {source, target, transform, edge_type}
    """
    if verbose:
        print(f"Discovering transforms between {len(patterns)} patterns...")

    edges = []
    pattern_list = list(patterns.values())
    n_patterns = len(pattern_list)

    # Group by length for efficient comparison
    by_length = defaultdict(list)
    for p in pattern_list:
        by_length[p.length].append(p)

    total_compared = 0
    found_transforms = 0

    for length, group in by_length.items():
        n = len(group)
        if n < 2:
            continue

        if verbose and n > 100:
            print(f"  Length {length}: {n} patterns, {n*(n-1)//2} pairs to check")

        for i in range(n):
            for j in range(i + 1, n):
                p1 = group[i]
                p2 = group[j]

                # Find transform from p1 to p2
                transform = find_pitch_transform(p1.pitch_classes, p2.pitch_classes)

                if transform and transform not in ('none', 'identity'):
                    edges.append({
                        'source': p1.id,
                        'target': p2.id,
                        'transform': transform,
                        'edge_type': 'derived',
                    })
                    found_transforms += 1

                total_compared += 1

    if verbose:
        print(f"  Compared {total_compared:,} pairs, found {found_transforms} transform edges")

    return edges


def extract_temporal_edges(
    patterns: Dict[int, Pattern],
    occurrences: Dict[int, List[Dict]],
    verbose: bool = True
) -> List[Dict]:
    """
    Extract temporal edges from occurrence data.

    For consecutive patterns in the same track, create τ edges.

    Args:
        patterns: Dict of pattern_id -> Pattern
        occurrences: Dict of pattern_id -> list of occurrence dicts
        verbose: Print progress

    Returns:
        List of edge dicts with temporal transforms
    """
    if verbose:
        print("Extracting temporal edges from occurrences...")

    # Flatten all occurrences with pattern IDs
    all_occurrences = []
    for pid, occs in occurrences.items():
        for occ in occs:
            all_occurrences.append({
                'pattern_id': pid,
                'piece_id': occ['piece_id'],
                'track_id': occ['track_id'],
                'onset_time': occ['onset_time'],
            })

    # Sort by (piece, track, time)
    all_occurrences.sort(key=lambda x: (x['piece_id'], x['track_id'], x['onset_time']))

    edges = []

    # Process consecutive pairs within same track
    for i in range(len(all_occurrences) - 1):
        curr = all_occurrences[i]
        next_occ = all_occurrences[i + 1]

        # Must be same piece and track
        if curr['piece_id'] != next_occ['piece_id'] or curr['track_id'] != next_occ['track_id']:
            continue

        # Compute time delta
        delta = next_occ['onset_time'] - curr['onset_time']
        if delta <= 0:
            continue

        # Quantize to standard grid
        delta_q = quantize_time_delta(delta)

        # Find pitch transform between patterns
        src_pattern = patterns.get(curr['pattern_id'])
        tgt_pattern = patterns.get(next_occ['pattern_id'])

        if not src_pattern or not tgt_pattern:
            continue

        pitch_t = find_pitch_transform(src_pattern.pitch_classes, tgt_pattern.pitch_classes)

        # Compose transform: pitch + time
        if pitch_t and pitch_t not in ('none', 'identity'):
            transform = f"{pitch_t}∘τ{delta_q}"
        else:
            transform = f"τ{delta_q}"

        edges.append({
            'source': curr['pattern_id'],
            'target': next_occ['pattern_id'],
            'transform': transform,
            'edge_type': 'temporal',
            'piece_id': curr['piece_id'],
            'track_id': curr['track_id'],
            'source_onset': curr['onset_time'],
        })

    if verbose:
        print(f"  Extracted {len(edges):,} temporal edges")

    return edges


def build_genome_graph(
    patterns: Dict[int, Pattern],
    occurrences: Dict[int, List[Dict]],
    transform_edges: List[Dict],
    temporal_edges: List[Dict],
    verbose: bool = True
) -> GenomeGraph:
    """
    Build GenomeGraph from extracted data.

    Args:
        patterns: Dict of pattern_id -> Pattern
        occurrences: Dict of pattern_id -> occurrence list
        transform_edges: Transform edges (T, I, R)
        temporal_edges: Temporal edges (τ)
        verbose: Print progress

    Returns:
        GenomeGraph instance
    """
    if verbose:
        print("Building genome graph...")

    graph = GenomeGraph()

    # Add patterns
    for pattern in patterns.values():
        graph.add_pattern(pattern)

    # Add transform edges
    for e in transform_edges:
        graph.add_edge(
            e['source'], e['target'], e['transform'],
            e['edge_type']
        )

    # Add temporal edges
    for e in temporal_edges:
        graph.add_edge(
            e['source'], e['target'], e['transform'],
            e['edge_type'], e.get('piece_id'), e.get('track_id'),
            e.get('source_onset')
        )

    # Store occurrences for reconstruction
    for pid, occ_list in occurrences.items():
        graph.occurrences[pid] = occ_list

    if verbose:
        stats = graph.stats()
        print(f"  Patterns: {stats['n_patterns']}")
        print(f"  Edges: {stats['n_edges']}")
        print(f"  Transform counts: {stats['transform_counts']}")

    return graph


def convert_v28_to_genome(
    checkpoint_path: str,
    output_path: str = None,
    verbose: bool = True
) -> GenomeGraph:
    """
    Convert v28 T-norm Re-Pair checkpoint to GenomeGraph.

    Args:
        checkpoint_path: Path to v28 checkpoint
        output_path: Optional output path (default: *_graph.npz)
        verbose: Print progress

    Returns:
        GenomeGraph
    """
    t0 = time.time()

    if verbose:
        print("=" * 70)
        print("V28 T-NORM TO GENOME GRAPH CONVERSION")
        print("=" * 70)

    # Load checkpoint
    ckpt = np.load(checkpoint_path, allow_pickle=True)

    rule_table = ckpt['rule_table']
    rule_counts = ckpt['rule_counts']
    n_terminals = int(ckpt['n_terminals'])
    final_sequence = ckpt['final_sequence']
    file_names = ckpt['file_names']

    if verbose:
        print(f"Loaded: {len(rule_table)} rules, {len(final_sequence):,} tokens")
        print(f"  n_terminals: {n_terminals}")
        print(f"  n_tracks: {len(file_names)}")

    # Step 1: Expand rules
    rules = expand_tnorm_rules(
        rule_table, rule_counts, n_terminals, final_sequence, verbose
    )

    # Step 2: Extract patterns from sequence
    patterns, occurrences = extract_patterns_from_sequence(
        final_sequence, rules, n_terminals, file_names, verbose
    )

    # Step 3: Discover transform edges
    transform_edges = discover_transform_edges(patterns, verbose)

    # Step 4: Extract temporal edges
    temporal_edges = extract_temporal_edges(patterns, occurrences, verbose)

    # Step 5: Build genome graph
    graph = build_genome_graph(
        patterns, occurrences, transform_edges, temporal_edges, verbose
    )

    # Save
    if output_path is None:
        output_path = checkpoint_path.replace('.npz', '_graph.npz')

    graph.to_checkpoint(output_path)

    elapsed = time.time() - t0

    if verbose:
        print()
        print("=" * 70)
        print(f"CONVERSION COMPLETE in {elapsed:.1f}s")
        print("=" * 70)
        print(f"Output: {output_path}")
        print(f"Size: {Path(output_path).stat().st_size / 1024 / 1024:.1f}MB")

    return graph


def main():
    parser = argparse.ArgumentParser(description='Convert v28 T-norm to GenomeGraph')
    parser.add_argument('checkpoint', help='Path to v28 T-norm Re-Pair checkpoint')
    parser.add_argument('--output', '-o', help='Output path (default: *_graph.npz)')
    args = parser.parse_args()

    convert_v28_to_genome(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
