#!/usr/bin/env python3
"""
Convert v30 T-norm Hierarchical Re-Pair to GenomeGraph format.

The v30 checkpoint contains hierarchical rules with T-normalization.
Rules are stored as (left, right) pairs where both can be terminals or other rules.

This script:
1. Loads the v30 checkpoint with hierarchical patterns
2. Creates Pattern objects from the expanded pitch_classes
3. Discovers transform edges (T, I, R) between patterns
4. Saves in GenomeGraph format for visualization

Usage:
    python scripts/convert_v30_to_genome.py checkpoint_v30_tnorm_hier.npz --output checkpoint_v30_graph.npz
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grammar.genome_graph import (
    GenomeGraph, Pattern,
    find_pitch_transform,
)


def convert_v30_to_genome(
    checkpoint_path: str,
    output_path: str = None,
    verbose: bool = True
) -> GenomeGraph:
    """
    Convert v30 T-norm Hierarchical Re-Pair checkpoint to GenomeGraph.

    Args:
        checkpoint_path: Path to v30 checkpoint
        output_path: Optional output path (default: *_graph.npz)
        verbose: Print progress

    Returns:
        GenomeGraph
    """
    t0 = time.time()

    if verbose:
        print("=" * 70)
        print("V30 T-NORM HIERARCHICAL TO GENOME GRAPH CONVERSION")
        print("=" * 70)

    # Load checkpoint
    ckpt = np.load(checkpoint_path, allow_pickle=True)

    n_terminals = int(ckpt['n_terminals'])
    n_rules = int(ckpt['n_rules'])
    rule_counts = ckpt['rule_counts']
    rule_intervals = ckpt['rule_intervals'] if 'rule_intervals' in ckpt else None
    final_sequence = ckpt['final_sequence'] if 'final_sequence' in ckpt else None
    file_names = ckpt['file_names'] if 'file_names' in ckpt else None

    if verbose:
        print(f"Loaded: {n_rules} rules, n_terminals={n_terminals}")
        if 'original_length' in ckpt and 'compressed_length' in ckpt:
            print(f"  Compression: {ckpt['original_length'] / ckpt['compressed_length']:.2f}x")
        if file_names is not None:
            unique_pieces = set(str(f) for f in file_names)
            print(f"  Pieces: {len(unique_pieces)}")

    # Load patterns from JSON (these have full expansions)
    patterns_path = checkpoint_path.replace('.npz', '_patterns.json')
    if os.path.exists(patterns_path):
        with open(patterns_path) as f:
            patterns_json = json.load(f)
        if verbose:
            print(f"Loaded {len(patterns_json)} patterns from {patterns_path}")
    else:
        if verbose:
            print(f"Warning: {patterns_path} not found, using rule table directly")
        patterns_json = {}

    # Build GenomeGraph
    graph = GenomeGraph()

    # Create Pattern objects from expanded patterns
    patterns = {}
    for pid_str, pdata in patterns_json.items():
        pid = int(pid_str)
        pc = pdata['pitch_classes']
        if len(pc) < 2:
            continue

        pattern = Pattern(
            id=pid,
            pitch_classes=pc,
            octaves=[4] * len(pc),
            velocities=[80] * len(pc),
            durations=[480] * len(pc),
            rhythm_ioi=[480] * max(0, len(pc) - 1),
        )
        patterns[pid] = pattern
        graph.add_pattern(pattern)

    if verbose:
        print(f"Created {len(patterns)} patterns in graph")

    # Extract temporal edges from final_sequence
    if final_sequence is not None and file_names is not None:
        if verbose:
            print("Extracting temporal edges from final sequence...")

        # Find track boundaries (separator = -1)
        separator = -1
        track_boundaries = [-1] + list(np.where(final_sequence == separator)[0]) + [len(final_sequence)]

        temporal_edges = 0
        occurrences_by_pattern = defaultdict(list)

        for track_idx in range(len(track_boundaries) - 1):
            start = track_boundaries[track_idx] + 1
            end = track_boundaries[track_idx + 1]

            if start >= end:
                continue

            track_seq = final_sequence[start:end]
            piece_id = str(file_names[track_idx]) if track_idx < len(file_names) else f"track_{track_idx}"

            # Find consecutive rule pairs in this track
            prev_rule = None
            prev_pos = 0
            local_pos = 0

            for symbol in track_seq:
                if symbol < 0:
                    continue

                if symbol >= n_terminals and symbol in patterns:
                    # Record occurrence
                    occurrences_by_pattern[symbol].append({
                        'piece_id': piece_id,
                        'track_id': track_idx,
                        'onset_time': local_pos * 480,
                    })

                    # Create temporal edge to previous pattern
                    if prev_rule is not None and prev_rule in patterns:
                        delta = (local_pos - prev_pos) * 480
                        transform = f"τ{delta}" if delta > 0 else "τ480"
                        graph.add_edge(
                            prev_rule, symbol, transform, 'temporal',
                            piece_id, track_idx, prev_pos * 480
                        )
                        temporal_edges += 1

                    prev_rule = symbol
                    prev_pos = local_pos
                    local_pos += patterns[symbol].length
                else:
                    local_pos += 1

        # Store occurrences in graph
        for pid, occs in occurrences_by_pattern.items():
            graph.occurrences[pid] = occs

        if verbose:
            print(f"  Found {temporal_edges} temporal edges")
            print(f"  Pieces with patterns: {len(set(o['piece_id'] for occs in occurrences_by_pattern.values() for o in occs))}")

    # Discover transform edges between patterns
    if verbose:
        print("Discovering transform edges...")

    # Group patterns by length for efficient comparison
    by_length = defaultdict(list)
    for p in patterns.values():
        by_length[p.length].append(p)

    edges_found = 0
    for length, group in by_length.items():
        n = len(group)
        if n < 2:
            continue

        for i in range(n):
            for j in range(i + 1, n):
                p1 = group[i]
                p2 = group[j]

                # Find transform
                transform = find_pitch_transform(p1.pitch_classes, p2.pitch_classes)

                if transform and transform not in ('none', 'identity'):
                    graph.add_edge(p1.id, p2.id, transform, 'derived')
                    edges_found += 1

    if verbose:
        print(f"  Found {edges_found} transform edges")

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
        stats = graph.stats()
        print(f"Patterns: {stats['n_patterns']}")
        print(f"Edges: {stats['n_edges']}")
        print(f"Pieces: {stats['n_pieces']}")
        print(f"Output: {output_path}")
        print(f"Size: {Path(output_path).stat().st_size / 1024 / 1024:.2f}MB")

    return graph


def test_coverage(
    graph: GenomeGraph,
    midi_path: str,
    verbose: bool = True
) -> Dict:
    """
    Test coverage of a MIDI file using the grammar patterns.

    For T-normalized patterns, we match by interval sequence rather than exact pitch.

    Args:
        graph: GenomeGraph with patterns
        midi_path: Path to MIDI file to test
        verbose: Print progress

    Returns:
        Coverage statistics
    """
    # Import MIDI loader
    from scripts.run_factored_pipeline import load_midi_factored

    if verbose:
        print(f"\nTesting coverage on: {midi_path}")

    # Load MIDI
    tracks = load_midi_factored(midi_path)
    if not tracks:
        return {'error': 'Failed to load MIDI'}

    # Extract pitch classes
    all_pc = []
    for track in tracks:
        pc = track.pitch_classes
        if hasattr(pc, 'tolist'):
            pc = pc.tolist()
        all_pc.extend(pc)

    total_notes = len(all_pc)
    if verbose:
        print(f"  Total notes: {total_notes}")
        print(f"  Tracks: {len(tracks)}")

    # Build pattern lookup by interval sequence
    # For T-normalized patterns, we convert to interval sequence for matching
    def to_intervals(pc_list):
        """Convert pitch classes to interval sequence (for T-normalization)."""
        if len(pc_list) < 2:
            return tuple()
        intervals = []
        for i in range(len(pc_list) - 1):
            interval = (pc_list[i + 1] - pc_list[i]) % 12
            intervals.append(interval)
        return tuple(intervals)

    # Build lookup: interval_seq -> (pattern_id, pitch_classes, count)
    pattern_lookup = {}
    for pid, pattern in graph.patterns.items():
        interval_seq = to_intervals(pattern.pitch_classes)
        if interval_seq:
            pattern_lookup[interval_seq] = (pid, pattern.pitch_classes, pattern.length)

    if verbose:
        print(f"  Patterns with intervals: {len(pattern_lookup)}")

    # Match patterns greedily
    covered_notes = 0
    matches = []
    lengths_matched = []

    i = 0
    while i < total_notes:
        best_match = None
        best_length = 0

        # Try to match longest pattern first
        for length in range(min(64, total_notes - i), 1, -1):
            window = all_pc[i:i+length]
            interval_seq = to_intervals(window)

            if interval_seq in pattern_lookup:
                pid, pc, plen = pattern_lookup[interval_seq]
                if length > best_length:
                    best_length = length
                    best_match = (pid, length, i)
                    break  # Longest first, so we can break

        if best_match:
            pid, length, pos = best_match
            matches.append(best_match)
            lengths_matched.append(length)
            covered_notes += length
            i += length
        else:
            i += 1  # Skip unmatched note

    # Compute coverage
    coverage = covered_notes / total_notes if total_notes > 0 else 0

    result = {
        'total_notes': total_notes,
        'covered_notes': covered_notes,
        'coverage': coverage,
        'n_matches': len(matches),
        'avg_match_length': sum(lengths_matched) / len(lengths_matched) if lengths_matched else 0,
        'max_match_length': max(lengths_matched) if lengths_matched else 0,
        'unique_patterns_used': len(set(m[0] for m in matches)),
    }

    if verbose:
        print(f"\n  === Coverage Results ===")
        print(f"  Total notes: {result['total_notes']:,}")
        print(f"  Covered notes: {result['covered_notes']:,}")
        print(f"  Coverage: {result['coverage']:.1%}")
        print(f"  Matches: {result['n_matches']}")
        print(f"  Avg match length: {result['avg_match_length']:.1f}")
        print(f"  Max match length: {result['max_match_length']}")
        print(f"  Unique patterns: {result['unique_patterns_used']}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Convert v30 T-norm Hierarchical to GenomeGraph')
    parser.add_argument('checkpoint', help='Path to v30 checkpoint')
    parser.add_argument('--output', '-o', help='Output path (default: *_graph.npz)')
    parser.add_argument('--test-coverage', '-t', help='MIDI file to test coverage on')
    args = parser.parse_args()

    graph = convert_v30_to_genome(args.checkpoint, args.output)

    if args.test_coverage:
        test_coverage(graph, args.test_coverage)


if __name__ == '__main__':
    main()
