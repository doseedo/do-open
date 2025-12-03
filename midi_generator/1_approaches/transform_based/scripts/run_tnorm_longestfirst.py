#!/usr/bin/env python3
"""
Run T-normalized LONGESTFIRST grammar extraction on MIDI corpus.
This produces canonical patterns with transposition tracking.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_factored_pipeline import load_midi_factored
from grammar.v2.longestfirst import build_longestfirst_grammar

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("midi_folder", help="Path to MIDI files")
    parser.add_argument("--max-files", "-n", type=int, default=None)
    parser.add_argument("--output", "-o", default="checkpoint_v28_tnorm.npz")
    parser.add_argument("--min-length", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-rules", type=int, default=2000)
    args = parser.parse_args()

    start_time = time.time()
    
    # Find MIDI files
    midi_folder = Path(args.midi_folder)
    midi_files = list(midi_folder.glob("**/*.mid")) + list(midi_folder.glob("**/*.midi"))
    if args.max_files:
        midi_files = midi_files[:args.max_files]
    
    print(f"Found {len(midi_files)} MIDI files")
    
    # Load all tracks with channel 9 fix
    all_tracks = []
    piece_info = []  # Track which piece each track came from
    n_notes_total = 0
    n_files_processed = 0
    n_files_failed = 0
    
    for i, midi_path in enumerate(midi_files):
        try:
            tracks = load_midi_factored(str(midi_path))
            if tracks:
                for track in tracks:
                    if len(track.pitch_classes) >= args.min_length:
                        all_tracks.append(track)
                        piece_info.append({
                            'piece_id': midi_path.stem,
                            'track_id': track.track_id,
                            'n_notes': len(track.notes),
                        })
                        n_notes_total += len(track.notes)
                n_files_processed += 1
        except Exception as e:
            n_files_failed += 1
            if i < 10:
                print(f"  Failed: {midi_path.name}: {e}")
        
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(midi_files)} files, {len(all_tracks)} tracks, {n_notes_total:,} notes")
    
    print(f"\nLoaded {len(all_tracks)} tracks from {n_files_processed} files ({n_files_failed} failed)")
    print(f"Total notes: {n_notes_total:,}")
    
    # Extract pitch class sequences
    pitch_sequences = []
    for track in all_tracks:
        pc = track.pitch_classes
        if hasattr(pc, 'tolist'):
            pc = pc.tolist()
        pitch_sequences.append(pc)
    
    print(f"\n=== Building T-normalized LONGESTFIRST Grammar ===")
    print(f"Parameters: min_length={args.min_length}, max_length={args.max_length}, "
          f"min_freq={args.min_frequency}, max_rules={args.max_rules}")
    
    # Build grammar with T-normalization
    grammar = build_longestfirst_grammar(
        pitch_sequences,
        duration_sequences=None,  # Pitch-only for T-normalization
        device='cuda',
        min_length=args.min_length,
        max_length=args.max_length,
        min_frequency=args.min_frequency,
        max_rules=args.max_rules,
        include_duration=False,  # Must be False for T-norm
        t_normalize=True,        # THE KEY FLAG
        verbose=True,
    )
    
    # Analyze results
    print(f"\n=== Results ===")
    print(f"Canonical patterns: {len(grammar.rules)}")
    print(f"T-normalized: {grammar.t_normalized}")
    
    # Count total transposition variants
    total_variants = 0
    patterns_with_multiple_keys = 0
    for rule in grammar.rules.values():
        if rule.transposition_counts:
            n_keys = len(rule.transposition_counts)
            total_variants += n_keys
            if n_keys > 1:
                patterns_with_multiple_keys += 1
    
    print(f"Total T-variants: {total_variants} across {len(grammar.rules)} canonical patterns")
    print(f"Patterns appearing in multiple keys: {patterns_with_multiple_keys}")
    
    if len(grammar.rules) > 0:
        avg_keys = total_variants / len(grammar.rules)
        print(f"Avg keys per pattern: {avg_keys:.1f}")
    
    # Show top patterns by count
    print(f"\nTop 20 patterns by occurrence:")
    sorted_rules = sorted(grammar.rules.values(), key=lambda r: r.count, reverse=True)[:20]
    for rule in sorted_rules:
        pattern_str = str(list(rule.pattern[:8]))
        if len(rule.pattern) > 8:
            pattern_str = pattern_str[:-1] + ", ...]"
        keys_str = ""
        if rule.transposition_counts:
            keys = sorted(rule.transposition_counts.keys())
            keys_str = f" keys={keys}"
        print(f"  R{rule.rule_id}: len={len(rule.pattern)}, count={rule.count}{keys_str} {pattern_str}")
    
    # Save checkpoint
    print(f"\nSaving to {args.output}...")
    
    # Convert rules to JSON-serializable format
    rules_data = {}
    for rid, rule in grammar.rules.items():
        rules_data[str(rid)] = {
            'pattern': list(rule.pattern),
            'count': rule.count,
            'score': rule.score,
            'canonical_pitches': list(rule.canonical_pitches) if rule.canonical_pitches else None,
            'transposition_counts': dict(rule.transposition_counts) if rule.transposition_counts else None,
        }
    
    np.savez_compressed(
        args.output,
        version=np.array(['v28_tnorm_longestfirst']),
        stats_json=np.array([json.dumps({
            'n_files_processed': n_files_processed,
            'n_files_failed': n_files_failed,
            'n_tracks': len(all_tracks),
            'n_notes': n_notes_total,
            'n_patterns': len(grammar.rules),
            'n_transposition_variants': total_variants,
            'patterns_in_multiple_keys': patterns_with_multiple_keys,
            't_normalized': grammar.t_normalized,
        })]),
        grammar_rules_json=np.array([json.dumps(rules_data)]),
        piece_info_json=np.array([json.dumps(piece_info)]),
        start_sequence=np.array(grammar.start_sequence, dtype=np.int32),
    )
    
    elapsed = time.time() - start_time
    print(f"\nDone! {len(grammar.rules)} canonical patterns in {elapsed:.1f}s")
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
