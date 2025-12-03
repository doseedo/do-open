#!/usr/bin/env python3
"""
Tokenize Corpus V2 - Multi-Track Aware
=======================================

Key improvements over V1:
1. Track tokens (TR:0, TR:1, etc.) - track identity in sequence
2. Interleaved multi-track sequences - all tracks of a piece together
3. Bar markers (BAR) - explicit structure
4. Instrument role tokens (ROLE:bass, ROLE:melody, etc.)

This makes the token sequence SELF-CONTAINED - no checkpoint lookup needed
for reconstruction.

Token format:
  BAR         - Bar boundary
  TR:0        - Switch to track 0
  ROLE:bass   - Track role (derived from GM program)
  D3          - Delta time bucket
  P123        - Pattern ID
  T7          - Transpose (0-11)
  O0          - Octave offset
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# GM Program to role mapping
def get_role(gm_program: int) -> str:
    """Map GM program to musical role."""
    if gm_program in range(0, 8):
        return "piano"
    elif gm_program in range(8, 16):
        return "chromperc"  # chromatic percussion
    elif gm_program in range(16, 24):
        return "organ"
    elif gm_program in range(24, 32):
        return "guitar"
    elif gm_program in range(32, 40):
        return "bass"
    elif gm_program in range(40, 48):
        return "strings"
    elif gm_program in range(48, 56):
        return "ensemble"
    elif gm_program in range(56, 64):
        return "brass"
    elif gm_program in range(64, 72):
        return "reed"
    elif gm_program in range(72, 80):
        return "pipe"
    elif gm_program in range(80, 88):
        return "synlead"
    elif gm_program in range(88, 96):
        return "synpad"
    elif gm_program in range(96, 104):
        return "synfx"
    elif gm_program in range(104, 112):
        return "ethnic"
    elif gm_program in range(112, 120):
        return "perc"
    elif gm_program in range(120, 128):
        return "sfx"
    else:
        return "unknown"


def load_checkpoint(checkpoint_path: str):
    """Load checkpoint and return patterns."""
    ckpt = np.load(checkpoint_path, allow_pickle=True)

    patterns_file = ckpt.get('patterns_json_file', [None])[0]
    if patterns_file:
        patterns_path = Path(checkpoint_path).parent / patterns_file
        with open(patterns_path) as f:
            rules = json.load(f)
    else:
        rules = json.loads(str(ckpt['patterns_json'][0]))

    return rules


def build_piece_sequences(rules: dict):
    """Build sequences grouped by piece with all tracks."""
    # Collect all occurrences with track info
    piece_data = defaultdict(lambda: defaultdict(list))

    n_rules = len(rules)
    n_occs = 0

    for i, (rule_id, rule) in enumerate(rules.items()):
        if i % 1000 == 0:
            print(f"  Processing rule {i}/{n_rules}...")

        pattern_len = len(rule.get('pitch_classes', []))

        for occ in rule.get('occurrences', []):
            piece_id = occ.get('piece_id', '')
            track_id = occ.get('track_id', 0)
            onset = occ.get('onset_time', 0)
            gm_program = occ.get('gm_program', 0)

            piece_data[piece_id][track_id].append({
                'rule_id': rule_id,
                'onset_time': onset,
                'pitch_offset': occ.get('pitch_offset', 0),
                'octave': occ.get('octave_transform', 0) // 12,
                'pattern_len': pattern_len,
                'gm_program': gm_program,
                'role': get_role(gm_program),
            })
            n_occs += 1

    print(f"  Collected {n_occs:,} occurrences from {len(piece_data)} pieces")

    # Sort each track by onset, deduplicate
    print("  Sorting and deduplicating...")
    for piece_id in piece_data:
        for track_id in piece_data[piece_id]:
            occs = piece_data[piece_id][track_id]
            occs.sort(key=lambda x: x['onset_time'])

            # Deduplicate: keep longest at each onset
            by_onset = defaultdict(list)
            for occ in occs:
                by_onset[occ['onset_time']].append(occ)

            filtered = []
            for onset in sorted(by_onset.keys()):
                best = max(by_onset[onset], key=lambda x: x['pattern_len'])
                filtered.append(best)

            piece_data[piece_id][track_id] = filtered

    return piece_data


def delta_bucket(delta: int) -> int:
    """Convert delta time to bucket."""
    if delta <= 0:
        return 0
    elif delta <= 120:
        return 1
    elif delta <= 240:
        return 2
    elif delta <= 480:
        return 3
    elif delta <= 960:
        return 4
    elif delta <= 1920:
        return 5
    elif delta <= 3840:
        return 6
    else:
        return 7


def tokenize_piece_interleaved(piece_tracks: dict, ticks_per_bar: int = 1920):
    """Tokenize a piece with all tracks interleaved by time.

    Creates a sequence that interleaves all tracks, with track markers.
    """
    # Collect all events with their track
    all_events = []
    track_roles = {}

    for track_id, occs in piece_tracks.items():
        if occs:
            track_roles[track_id] = occs[0]['role']
        for occ in occs:
            all_events.append({
                'track_id': track_id,
                **occ
            })

    # Sort by onset time, then by track
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
            # Add role on first occurrence of track
            if prev_track is None or track_id not in [t for t in tokens if t.startswith("TR:")]:
                role = track_roles.get(track_id, "unknown")
                tokens.append(f"ROLE:{role}")
            prev_track = track_id

        # Delta time
        if onset > prev_onset:
            bucket = delta_bucket(onset - prev_onset)
            tokens.append(f"D{bucket}")
        else:
            tokens.append("D0")

        # Pattern
        tokens.append(f"P{evt['rule_id']}")

        # Transpose
        transpose = evt['pitch_offset'] % 12
        tokens.append(f"T{transpose}")

        # Octave
        octave = max(-2, min(2, evt['octave']))
        tokens.append(f"O{octave}")

        prev_onset = onset

    return tokens, track_roles


def tokenize_piece_by_track(piece_tracks: dict):
    """Tokenize each track separately (V1 style but with track tokens)."""
    sequences = []

    for track_id, occs in piece_tracks.items():
        if len(occs) < 8:
            continue

        tokens = []
        role = occs[0]['role'] if occs else "unknown"

        # Track header
        tokens.append(f"TR:{track_id}")
        tokens.append(f"ROLE:{role}")

        prev_onset = 0
        for i, occ in enumerate(occs):
            # Delta
            if i > 0:
                delta = occ['onset_time'] - prev_onset
                bucket = delta_bucket(delta)
                tokens.append(f"D{bucket}")

            # Pattern
            tokens.append(f"P{occ['rule_id']}")

            # Transpose
            transpose = occ['pitch_offset'] % 12
            tokens.append(f"T{transpose}")

            # Octave
            octave = max(-2, min(2, occ['octave']))
            tokens.append(f"O{octave}")

            prev_onset = occ['onset_time']

        sequences.append({
            'track_id': track_id,
            'role': role,
            'tokens': tokens
        })

    return sequences


def build_vocab(all_tokens: list):
    """Build vocabulary from all tokens."""
    vocab = set()
    for tokens in all_tokens:
        vocab.update(tokens)

    # Sort for consistency
    def sort_key(x):
        if x.startswith('TR:'):
            return (0, int(x[3:]))
        elif x.startswith('ROLE:'):
            return (1, x)
        elif x == 'BAR':
            return (2, 0)
        elif x.startswith('D'):
            return (3, int(x[1:]) if x[1:].lstrip('-').isdigit() else 0)
        elif x.startswith('P'):
            return (4, int(x[1:]) if x[1:].isdigit() else 0)
        elif x.startswith('T'):
            return (5, int(x[1:]) if x[1:].isdigit() else 0)
        elif x.startswith('O'):
            return (6, int(x[1:]) if x[1:].lstrip('-').isdigit() else 0)
        else:
            return (7, x)

    vocab = sorted(vocab, key=sort_key)

    # Add special tokens
    special = ['<pad>', '<bos>', '<eos>', '<unk>']
    vocab = special + vocab

    return {tok: i for i, tok in enumerate(vocab)}


def main():
    parser = argparse.ArgumentParser(description='Tokenize corpus V2 - multi-track aware')
    parser.add_argument('checkpoint', help='Path to checkpoint .npz file')
    parser.add_argument('--output', '-o', default='corpus_tokens_v2.jsonl', help='Output JSONL file')
    parser.add_argument('--vocab', '-v', default='vocab_v2.json', help='Output vocab file')
    parser.add_argument('--mode', '-m', choices=['interleaved', 'by_track'], default='interleaved',
                        help='Tokenization mode')
    parser.add_argument('--min-length', type=int, default=32, help='Minimum sequence length')
    parser.add_argument('--max-length', type=int, default=1024, help='Maximum sequence length')
    args = parser.parse_args()

    print("=" * 60)
    print("TOKENIZE CORPUS V2 - MULTI-TRACK AWARE")
    print("=" * 60)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    rules = load_checkpoint(args.checkpoint)
    print(f"  Loaded {len(rules)} patterns")

    # Build piece sequences
    print("\nBuilding piece sequences...")
    piece_data = build_piece_sequences(rules)
    print(f"  Found {len(piece_data)} pieces")

    # Tokenize
    print(f"\nTokenizing (mode: {args.mode})...")
    all_token_seqs = []

    for piece_id, tracks in piece_data.items():
        if args.mode == 'interleaved':
            tokens, track_roles = tokenize_piece_interleaved(tracks)

            if len(tokens) >= args.min_length:
                if len(tokens) > args.max_length:
                    tokens = tokens[:args.max_length]

                all_token_seqs.append({
                    'piece_id': piece_id,
                    'n_tracks': len(tracks),
                    'track_roles': track_roles,
                    'tokens': tokens,
                })
        else:  # by_track
            for seq in tokenize_piece_by_track(tracks):
                tokens = seq['tokens']
                if len(tokens) >= args.min_length:
                    if len(tokens) > args.max_length:
                        tokens = tokens[:args.max_length]

                    all_token_seqs.append({
                        'piece_id': piece_id,
                        'track_id': seq['track_id'],
                        'role': seq['role'],
                        'tokens': tokens,
                    })

    print(f"  Tokenized {len(all_token_seqs)} sequences")

    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = build_vocab([s['tokens'] for s in all_token_seqs])
    print(f"  Vocabulary size: {len(vocab)}")

    # Stats
    total_tokens = sum(len(s['tokens']) for s in all_token_seqs)
    avg_len = total_tokens / len(all_token_seqs) if all_token_seqs else 0

    # Count token types
    token_types = defaultdict(int)
    for s in all_token_seqs:
        for t in s['tokens']:
            if t.startswith('TR:'):
                token_types['TR'] += 1
            elif t.startswith('ROLE:'):
                token_types['ROLE'] += 1
            elif t == 'BAR':
                token_types['BAR'] += 1
            elif t.startswith('D'):
                token_types['D'] += 1
            elif t.startswith('P'):
                token_types['P'] += 1
            elif t.startswith('T'):
                token_types['T'] += 1
            elif t.startswith('O'):
                token_types['O'] += 1

    print(f"\nStats:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg sequence length: {avg_len:.1f}")
    print(f"  Token type distribution:")
    for tt, count in sorted(token_types.items()):
        print(f"    {tt}: {count:,}")

    # Unique roles
    roles = set()
    for s in all_token_seqs:
        for t in s['tokens']:
            if t.startswith('ROLE:'):
                roles.add(t[5:])
    print(f"  Unique roles: {sorted(roles)}")

    # Save tokens
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        for seq in all_token_seqs:
            f.write(json.dumps(seq) + '\n')
    print(f"\nSaved tokens to: {output_path}")

    # Save vocab
    vocab_path = Path(args.vocab)
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocab to: {vocab_path}")

    # Show sample
    print("\n" + "=" * 60)
    print("SAMPLE SEQUENCE")
    print("=" * 60)
    if all_token_seqs:
        sample = all_token_seqs[0]
        print(f"Piece: {sample['piece_id']}")
        if 'n_tracks' in sample:
            print(f"Tracks: {sample['n_tracks']}, Roles: {sample.get('track_roles', {})}")
        print(f"Tokens ({len(sample['tokens'])}): {' '.join(sample['tokens'][:50])}...")

    print("\nDone!")


if __name__ == '__main__':
    main()
