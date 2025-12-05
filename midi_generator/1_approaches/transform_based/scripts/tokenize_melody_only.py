#!/usr/bin/env python3
"""
MELODY-ONLY TOKENIZER (V5)

Strategy: Train transformer on melody/lead voices only.
At generation time, use orchestration rules to derive accompaniment.

This separates concerns:
- Transformer learns: melodic coherence, harmonic progressions
- Rules provide: voice leading, instrument relationships
"""

import json
import argparse
from collections import defaultdict

# GM program ranges for melody instruments
MELODY_PROGRAMS = set(range(0, 8))    # Piano
MELODY_PROGRAMS |= set(range(24, 32)) # Guitar
MELODY_PROGRAMS |= set(range(40, 48)) # Strings (violin, viola, cello)
MELODY_PROGRAMS |= set(range(56, 64)) # Brass
MELODY_PROGRAMS |= set(range(64, 72)) # Reed (sax, oboe, clarinet)
MELODY_PROGRAMS |= set(range(72, 80)) # Pipe (flute, piccolo)
MELODY_PROGRAMS |= set(range(80, 88)) # Synth lead

# Roles considered "lead/melody"
MELODY_ROLES = {'piano', 'guitar', 'strings', 'brass', 'reed', 'pipe', 'synlead'}

def get_role(gm_program):
    """Map GM program to role."""
    if gm_program < 8: return 'piano'
    elif gm_program < 16: return 'chromperc'
    elif gm_program < 24: return 'organ'
    elif gm_program < 32: return 'guitar'
    elif gm_program < 40: return 'bass'
    elif gm_program < 48: return 'strings'
    elif gm_program < 56: return 'ensemble'
    elif gm_program < 64: return 'brass'
    elif gm_program < 72: return 'reed'
    elif gm_program < 80: return 'pipe'
    elif gm_program < 88: return 'synlead'
    elif gm_program < 96: return 'synpad'
    elif gm_program < 104: return 'synfx'
    elif gm_program < 112: return 'ethnic'
    elif gm_program < 120: return 'perc'
    else: return 'sfx'

def is_melody_track(gm_program, role):
    """Check if this track should be considered melody/lead."""
    return role in MELODY_ROLES

def tokenize_piece_melody_only(tracks: dict, ticks_per_bar: int = 1920):
    """
    Tokenize only melody/lead tracks.
    Skip bass, drums, pads, etc.
    """
    melody_events = []

    for track_id, occs in tracks.items():
        if not occs:
            continue
        role = occs[0].get('role', 'unknown')

        # Only include melody tracks
        if not is_melody_track(occs[0].get('gm_program', 0), role):
            continue

        for occ in occs:
            melody_events.append({
                'track_id': track_id,
                'role': role,
                **occ
            })

    if not melody_events:
        return [], {}

    # Sort by onset time, then track
    melody_events.sort(key=lambda x: (x['onset_time'], x['track_id']))

    tokens = []
    prev_onset = 0
    prev_track = None
    current_bar = -1
    track_roles = {}

    for evt in melody_events:
        onset = evt['onset_time']
        track_id = evt['track_id']
        role = evt['role']
        track_roles[track_id] = role

        # Bar marker
        bar = onset // ticks_per_bar
        if bar > current_bar:
            current_bar = bar
            tokens.append("BAR")

        # Track switch
        if track_id != prev_track:
            tokens.append(f"TR:{track_id}")
            tokens.append(f"ROLE:{role}")
            prev_track = track_id

        # Rhythm delta (quantized)
        delta = onset - prev_onset
        delta_bucket = min(7, delta // 240)  # 240 ticks = 1/8 note
        tokens.append(f"D{delta_bucket}")
        prev_onset = onset

        # Pattern, transpose, octave
        tokens.append(f"P{evt['pattern_id']}")
        tokens.append(f"T{evt['transpose']}")
        tokens.append(f"O{evt['octave']}")

    return tokens, track_roles


def main():
    parser = argparse.ArgumentParser(description='Melody-only tokenizer')
    parser.add_argument('checkpoint', help='Patterns JSON file')
    parser.add_argument('--output', '-o', default='corpus_melody_v5.jsonl')
    parser.add_argument('--vocab', '-v', default='vocab_melody_v5.json')
    parser.add_argument('--min-pattern-notes', type=int, default=4)
    parser.add_argument('--min-length', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=1024)
    args = parser.parse_args()

    print("=" * 60)
    print("MELODY-ONLY TOKENIZER (V5)")
    print("=" * 60)

    # Load patterns
    print(f"\nLoading: {args.checkpoint}")
    with open(args.checkpoint) as f:
        patterns = json.load(f)
    print(f"  Patterns: {len(patterns)}")

    # Build canonical root lookup
    pattern_canonical_root = {}
    for rule_id, rule in patterns.items():
        canonical = rule.get('canonical_pitches', [60])
        pattern_canonical_root[rule_id] = canonical[0] if canonical else 60

    # Group occurrences by piece
    print("\nGrouping occurrences by piece...")
    pieces = defaultdict(lambda: defaultdict(list))

    melody_patterns = 0
    skipped_short = 0
    skipped_role = 0

    for i, (rule_id, rule) in enumerate(patterns.items()):
        if i % 2000 == 0:
            print(f"  Rule {i}/{len(patterns)}...")

        pattern_len = len(rule.get('pitch_classes', []))
        if pattern_len < args.min_pattern_notes:
            skipped_short += len(rule.get('occurrences', []))
            continue

        canonical_root = pattern_canonical_root.get(rule_id, 60)

        for occ in rule.get('occurrences', []):
            piece_id = occ['piece_id']
            track_id = occ['track_id']
            gm_program = occ.get('gm_program', 0)
            role = get_role(gm_program)

            # Filter to melody roles only
            if not is_melody_track(gm_program, role):
                skipped_role += 1
                continue

            melody_patterns += 1

            octave_transform = occ.get('octave_transform', 0)
            actual_pitch = canonical_root + octave_transform
            transpose = actual_pitch % 12

            pieces[piece_id][track_id].append({
                'pattern_id': rule_id,
                'onset_time': occ.get('onset_time', occ.get('position', 0)),
                'gm_program': gm_program,
                'role': role,
                'transpose': transpose,
                'octave': octave_transform // 12,
            })

    print(f"  Found {len(pieces)} pieces")
    print(f"  Melody patterns: {melody_patterns}")
    print(f"  Skipped (short): {skipped_short}")
    print(f"  Skipped (non-melody): {skipped_role}")

    # Tokenize
    print("\nTokenizing melody tracks...")
    sequences = []
    vocab = set()

    for i, (piece_id, tracks) in enumerate(pieces.items()):
        if i % 100 == 0:
            print(f"  Piece {i}/{len(pieces)}...")

        tokens, track_roles = tokenize_piece_melody_only(tracks)

        if len(tokens) < args.min_length:
            continue
        if len(tokens) > args.max_length:
            tokens = tokens[:args.max_length]

        vocab.update(tokens)
        sequences.append({
            'piece_id': piece_id,
            'tokens': tokens,
            'track_roles': track_roles
        })

    print(f"  Tokenized {len(sequences)} sequences")

    # Build vocab
    print("\nBuilding vocabulary...")
    vocab_list = sorted(vocab)
    vocab_map = {t: i for i, t in enumerate(vocab_list)}
    print(f"  Vocab size: {len(vocab_map)}")

    # Stats
    total_tokens = sum(len(s['tokens']) for s in sequences)
    print(f"\nStats:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg length: {total_tokens / len(sequences):.1f}")

    # Analyze role distribution
    role_counts = defaultdict(int)
    for seq in sequences:
        for t in seq['tokens']:
            if t.startswith('ROLE:'):
                role_counts[t[5:]] += 1

    print(f"\nRole distribution in melody corpus:")
    for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
        print(f"  {role:12s}: {count:6d}")

    # Save
    with open(args.output, 'w') as f:
        for seq in sequences:
            f.write(json.dumps(seq) + '\n')
    print(f"\nSaved: {args.output}")

    with open(args.vocab, 'w') as f:
        json.dump(vocab_map, f)
    print(f"Saved: {args.vocab}")

    # Sample
    print("\n" + "=" * 60)
    print("SAMPLE:")
    if sequences:
        sample = sequences[0]
        print(f"Piece: {sample['piece_id']}, Roles: {sample['track_roles']}")
        print(f"Tokens: {' '.join(sample['tokens'][:80])}...")

    return sequences, vocab_map


if __name__ == '__main__':
    main()
