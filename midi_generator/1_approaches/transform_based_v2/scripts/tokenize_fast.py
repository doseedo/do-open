#!/usr/bin/env python3
"""
Fast Tokenizer - Streaming Approach (v3 - Fixed Transpose)
===========================================================

Process piece-by-piece without loading all 2.6M occurrences into memory.
Adds track tokens for self-contained sequences.

V3 FIX: Compute transpose from canonical_pitches + octave_transform
instead of using broken pitch_offset (which is always 0).
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def get_role(gm_program: int) -> str:
    """Map GM program to musical role."""
    if gm_program < 8: return "piano"
    elif gm_program < 16: return "chromperc"
    elif gm_program < 24: return "organ"
    elif gm_program < 32: return "guitar"
    elif gm_program < 40: return "bass"
    elif gm_program < 48: return "strings"
    elif gm_program < 56: return "ensemble"
    elif gm_program < 64: return "brass"
    elif gm_program < 72: return "reed"
    elif gm_program < 80: return "pipe"
    elif gm_program < 88: return "synlead"
    elif gm_program < 96: return "synpad"
    elif gm_program < 104: return "synfx"
    elif gm_program < 112: return "ethnic"
    elif gm_program < 120: return "perc"
    else: return "sfx"


def delta_bucket(delta: int) -> int:
    if delta <= 0: return 0
    elif delta <= 120: return 1
    elif delta <= 240: return 2
    elif delta <= 480: return 3
    elif delta <= 960: return 4
    elif delta <= 1920: return 5
    elif delta <= 3840: return 6
    else: return 7


def tokenize_piece_interleaved(tracks: dict, ticks_per_bar: int = 1920):
    """Tokenize with all tracks interleaved."""
    all_events = []
    track_roles = {}

    for track_id, occs in tracks.items():
        if occs:
            track_roles[track_id] = occs[0]['role']
        for occ in occs:
            all_events.append({'track_id': track_id, **occ})

    all_events.sort(key=lambda x: (x['onset_time'], x['track_id']))

    if not all_events:
        return [], {}

    tokens = []
    prev_onset = 0
    prev_track = None
    current_bar = -1

    for evt in all_events:
        onset = evt['onset_time']
        track_id = evt['track_id']

        # Bar marker
        bar = onset // ticks_per_bar
        if bar > current_bar:
            current_bar = bar
            tokens.append("BAR")

        # Track switch
        if track_id != prev_track:
            tokens.append(f"TR:{track_id}")
            role = track_roles.get(track_id, "unknown")
            tokens.append(f"ROLE:{role}")
            prev_track = track_id

        # Delta time
        if onset > prev_onset:
            bucket = delta_bucket(onset - prev_onset)
            tokens.append(f"D{bucket}")
        else:
            tokens.append("D0")

        # Pattern, Transpose, Octave
        tokens.append(f"P{evt['rule_id']}")
        tokens.append(f"T{evt['transpose']}")  # V3: Use computed transpose
        tokens.append(f"O{max(-2, min(2, evt['octave']))}")

        prev_onset = onset

    return tokens, track_roles


def main():
    parser = argparse.ArgumentParser(description='Fast tokenizer with track tokens')
    parser.add_argument('patterns_json', help='Path to patterns JSON file')
    parser.add_argument('--output', '-o', default='corpus_tokens_v2.jsonl')
    parser.add_argument('--vocab', '-v', default='vocab_v2.json')
    parser.add_argument('--min-length', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--min-pattern-notes', type=int, default=4,
                        help='Minimum pattern length in notes (skip shorter patterns)')
    args = parser.parse_args()

    print("=" * 60)
    print("FAST TOKENIZER WITH TRACK TOKENS")
    print("=" * 60)

    print(f"\nLoading: {args.patterns_json}")
    with open(args.patterns_json) as f:
        rules = json.load(f)
    print(f"  Patterns: {len(rules)}")

    # Build pattern canonical pitch lookup for transpose computation
    # V3 FIX: Use canonical_pitches[0] + octave_transform to get actual transpose
    pattern_canonical_root = {}
    for rule_id, rule in rules.items():
        canonical = rule.get('canonical_pitches', [])
        if canonical:
            pattern_canonical_root[rule_id] = canonical[0]
        else:
            pattern_canonical_root[rule_id] = 60  # Default middle C
    print(f"  Built canonical root lookup for {len(pattern_canonical_root)} patterns")

    # First pass: collect occurrences by piece
    print("\nGrouping occurrences by piece...")
    piece_data = defaultdict(lambda: defaultdict(list))

    for i, (rule_id, rule) in enumerate(rules.items()):
        if i % 2000 == 0:
            print(f"  Rule {i}/{len(rules)}...")

        pattern_len = len(rule.get('pitch_classes', []))

        # V4: Skip patterns shorter than minimum
        if pattern_len < args.min_pattern_notes:
            continue

        # Get canonical root for this pattern
        canonical_root = pattern_canonical_root.get(rule_id, 60)

        for occ in rule.get('occurrences', []):
            piece_id = occ.get('piece_id', '')
            track_id = occ.get('track_id', 0)
            octave_transform = occ.get('octave_transform', 0)

            # V3 FIX: Compute actual transpose from canonical + octave_transform
            # actual_first_pitch = canonical_root + octave_transform
            # transpose = actual_first_pitch % 12
            actual_pitch = canonical_root + octave_transform
            transpose = actual_pitch % 12

            piece_data[piece_id][track_id].append({
                'rule_id': rule_id,
                'onset_time': occ.get('onset_time', 0),
                'transpose': transpose,  # V3: Actual key (0-11)
                'octave': octave_transform // 12,
                'pattern_len': pattern_len,
                'role': get_role(occ.get('gm_program', 0)),
            })

    print(f"  Found {len(piece_data)} pieces")

    # Second pass: tokenize each piece
    print("\nTokenizing pieces...")
    all_seqs = []
    vocab = set()

    for i, (piece_id, tracks) in enumerate(piece_data.items()):
        if i % 100 == 0:
            print(f"  Piece {i}/{len(piece_data)}...")

        # Sort and dedupe each track
        for track_id in tracks:
            occs = tracks[track_id]
            occs.sort(key=lambda x: x['onset_time'])

            # Keep longest at each onset
            by_onset = defaultdict(list)
            for occ in occs:
                by_onset[occ['onset_time']].append(occ)

            tracks[track_id] = [
                max(by_onset[t], key=lambda x: x['pattern_len'])
                for t in sorted(by_onset.keys())
            ]

        # Tokenize
        tokens, track_roles = tokenize_piece_interleaved(tracks)

        if args.min_length <= len(tokens):
            if len(tokens) > args.max_length:
                tokens = tokens[:args.max_length]

            vocab.update(tokens)
            all_seqs.append({
                'piece_id': piece_id,
                'n_tracks': len(tracks),
                'tokens': tokens,
            })

    print(f"  Tokenized {len(all_seqs)} sequences")

    # Build vocab
    print("\nBuilding vocabulary...")
    def sort_key(x):
        if x.startswith('TR:'): return (0, int(x[3:]))
        elif x.startswith('ROLE:'): return (1, x)
        elif x == 'BAR': return (2, 0)
        elif x.startswith('D'): return (3, int(x[1:]))
        elif x.startswith('P'): return (4, int(x[1:]))
        elif x.startswith('T'): return (5, int(x[1:]))
        elif x.startswith('O'): return (6, int(x[1:].replace('-', '')))
        return (7, x)

    vocab = sorted(vocab, key=sort_key)
    vocab = ['<pad>', '<bos>', '<eos>', '<unk>'] + vocab
    vocab_dict = {t: i for i, t in enumerate(vocab)}
    print(f"  Vocab size: {len(vocab_dict)}")

    # Stats
    total_tokens = sum(len(s['tokens']) for s in all_seqs)
    print(f"\nStats:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg length: {total_tokens/len(all_seqs):.1f}")

    # Save
    with open(args.output, 'w') as f:
        for seq in all_seqs:
            f.write(json.dumps(seq) + '\n')
    print(f"\nSaved: {args.output}")

    with open(args.vocab, 'w') as f:
        json.dump(vocab_dict, f, indent=2)
    print(f"Saved: {args.vocab}")

    # Sample
    print("\n" + "=" * 60)
    print("SAMPLE:")
    if all_seqs:
        s = all_seqs[0]
        print(f"Piece: {s['piece_id']}, Tracks: {s['n_tracks']}")
        print(f"Tokens: {' '.join(s['tokens'][:60])}...")


if __name__ == '__main__':
    main()
