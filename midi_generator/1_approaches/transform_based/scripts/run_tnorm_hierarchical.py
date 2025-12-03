#!/usr/bin/env python3
"""
Run T-normalized Hierarchical Re-Pair on MIDI corpus.

This combines:
1. T-normalization: pairs counted by interval (all transpositions count together)
2. Hierarchical composition: rules can reference rules (length 3-40+)

Usage:
    python scripts/run_tnorm_hierarchical.py /path/to/midi_corpus --max-files 500
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from scripts.run_factored_pipeline import load_midi_factored
from grammar.v2.repair_tnorm_hierarchical import build_tnorm_hierarchical_grammar, TnormHierarchicalGrammar


def main():
    parser = argparse.ArgumentParser(description='T-normalized Hierarchical Re-Pair extraction')
    parser.add_argument('corpus_path', help='Path to MIDI corpus')
    parser.add_argument('--max-files', '-n', type=int, default=500, help='Max files to process')
    parser.add_argument('--min-count', type=int, default=2, help='Minimum pair count')
    parser.add_argument('--max-rules', type=int, default=5000, help='Maximum rules')
    parser.add_argument('--output', '-o', default='checkpoint_v29_tnorm_hier.npz',
                        help='Output checkpoint file')
    args = parser.parse_args()

    print(f"T-Normalized Hierarchical Re-Pair Extraction")
    print(f"=============================================")
    print(f"Corpus: {args.corpus_path}")
    print(f"Max files: {args.max_files}")
    print(f"Min count: {args.min_count}")
    print(f"Max rules: {args.max_rules}")
    print()

    # Find MIDI files
    midi_folder = Path(args.corpus_path)
    midi_files = list(midi_folder.glob("**/*.mid")) + list(midi_folder.glob("**/*.midi"))
    if args.max_files:
        midi_files = midi_files[:args.max_files]

    print(f"Found {len(midi_files)} MIDI files")

    # Load all tracks
    all_tracks = []
    piece_info = []
    total_notes = 0
    n_files_processed = 0
    n_files_failed = 0

    for i, midi_path in enumerate(midi_files):
        try:
            tracks = load_midi_factored(str(midi_path))
            if tracks:
                for track in tracks:
                    if len(track.notes) >= 2:
                        all_tracks.append(track)
                        piece_info.append({
                            'piece_id': midi_path.stem,
                            'track_id': track.track_id,
                            'n_notes': len(track.notes),
                        })
                        total_notes += len(track.notes)
                n_files_processed += 1
        except Exception as e:
            n_files_failed += 1

        if (i + 1) % 50 == 0:
            print(f"  Loaded {i+1}/{len(midi_files)} files, {len(all_tracks)} tracks, {total_notes:,} notes")

    print(f"\nLoaded {len(all_tracks)} tracks from {n_files_processed} files ({n_files_failed} failed)")
    print(f"Total notes: {total_notes:,}")
    print()

    # Extract pitch class sequences from tracks
    # Using pitch classes (0-11) for T-normalization
    sequences = []
    file_names = []
    for track, info in zip(all_tracks, piece_info):
        # Get pitch classes (0-11)
        pc = track.pitch_classes
        if hasattr(pc, 'tolist'):
            pc = pc.tolist()
        if len(pc) >= 2:
            sequences.append(pc)
            file_names.append(info['piece_id'])

    print(f"Got {len(sequences)} sequences with {sum(len(s) for s in sequences):,} notes")
    print()

    # Run T-normalized Hierarchical Re-Pair
    print("=== Running T-Normalized Hierarchical Re-Pair ===")
    t0 = time.time()
    grammar = build_tnorm_hierarchical_grammar(
        sequences,
        device='cuda',
        min_pair_count=args.min_count,
        max_rules=args.max_rules,
        pitch_range=12,  # pitch_classes are 0-11
        verbose=True,
    )
    grammar_time = time.time() - t0

    print()
    print(f"Grammar extraction complete in {grammar_time:.1f}s")
    print(f"  Rules: {grammar.n_rules}")
    print(f"  Compression: {grammar.compression_ratio():.2f}x")
    print(f"  Speed: {total_notes/grammar_time:,.0f} notes/sec")

    # Analyze rules by expansion length
    print("\nAnalyzing rule expansion lengths...")
    length_counts = {}
    rule_lengths = []
    for i in range(grammar.n_rules):
        rule_id = grammar.n_terminals + i
        expansion = grammar.expand_rule(rule_id)
        length = len(expansion)
        length_counts[length] = length_counts.get(length, 0) + 1
        rule_lengths.append(length)

    print("\nRules by expansion length:")
    for length in sorted(length_counts.keys()):
        print(f"  Length {length}: {length_counts[length]} rules")

    # Show top rules by count
    print("\nTop 20 rules by frequency:")
    sorted_idx = grammar.rule_counts.argsort(descending=True)[:20]
    interval_names = {
        0: "unison", 1: "m2", 2: "M2", 3: "m3", 4: "M3",
        5: "P4", 6: "tritone", 7: "P5", 8: "m6", 9: "M6",
        10: "m7", 11: "M7"
    }
    for i, idx in enumerate(sorted_idx):
        rule_id = grammar.n_terminals + idx.item()
        left, right = grammar.get_rule(rule_id)
        count = grammar.rule_counts[idx].item()
        interval = grammar.rule_intervals[idx].item()
        name = interval_names.get(interval % 12 if interval >= 0 else -1, f"{interval}")
        expansion = grammar.expand_rule(rule_id)
        is_hier = left >= 12 or right >= 12
        hier_mark = " [HIER]" if is_hier else ""
        # Interval sequence
        intervals_seq = grammar.get_pattern_interval_sequence(rule_id)
        print(f"  {i+1:2d}. R{rule_id}: ({left},{right}) interval={name:>7s}{hier_mark:6s} "
              f"count={count:,}, len={len(expansion)}, intervals={intervals_seq}")

    # Save checkpoint
    print(f"\nSaving to {args.output}...")

    # Build expanded patterns for the checkpoint
    canonical_patterns = {}
    for i in range(grammar.n_rules):
        rule_id = grammar.n_terminals + i
        expansion = grammar.expand_rule(rule_id)
        if len(expansion) >= 3:  # Only save patterns of length 3+
            canonical_patterns[rule_id] = {
                'pitch_classes': expansion,
                'count': grammar.rule_counts[i].item(),
                'interval': grammar.rule_intervals[i].item(),
            }

    np.savez_compressed(
        args.output,
        # Grammar
        rule_table=grammar.rule_table.cpu().numpy(),
        rule_counts=grammar.rule_counts.cpu().numpy(),
        rule_intervals=grammar.rule_intervals.cpu().numpy(),
        final_sequence=grammar.final_sequence.cpu().numpy(),
        n_terminals=grammar.n_terminals,
        n_rules=grammar.n_rules,
        original_length=grammar.original_length,
        compressed_length=grammar.compressed_length,
        # Rule expansions
        rule_lengths=np.array(rule_lengths),
        # Metadata
        n_files=n_files_processed,
        n_tracks=len(all_tracks),
        total_notes=total_notes,
        file_names=np.array(file_names, dtype=object),
        # Patterns (for genome conversion)
        n_canonical=len(canonical_patterns),
    )

    # Save canonical patterns separately (they can be large)
    import json
    patterns_file = args.output.replace('.npz', '_patterns.json')
    with open(patterns_file, 'w') as f:
        json.dump(canonical_patterns, f)

    print(f"Saved {Path(args.output).stat().st_size / 1024 / 1024:.1f}MB")
    print(f"Saved {len(canonical_patterns)} canonical patterns (len>=3) to {patterns_file}")

    print("\nDone!")


if __name__ == '__main__':
    main()
