#!/usr/bin/env python3
"""
RULE-BASED MULTI-TRACK GENERATOR

Takes melody from transformer and applies orchestration rules
to generate bass, chords, and other accompaniment.

Architecture:
1. Transformer generates melody sequence
2. Orchestration rules derive accompaniment:
   - melody → bass: typically T7 (fifth below) or T0 (root), O-2
   - melody → chords: T0, T4, T7 (triads)
   - melody → pad: T0, sustained

This separates concerns cleanly:
- Transformer: melodic coherence
- Rules: harmonic voice leading
"""

import json
import argparse
import random
from collections import defaultdict

# Default orchestration rules (can be loaded from checkpoint)
DEFAULT_RULES = {
    'melody_to_bass': [
        {'transpose_delta': 0, 'octave_delta': -2, 'weight': 0.4},   # Root
        {'transpose_delta': 7, 'octave_delta': -2, 'weight': 0.35},  # Fifth
        {'transpose_delta': 5, 'octave_delta': -2, 'weight': 0.25},  # Fourth
    ],
    'melody_to_chord': [
        {'transpose_deltas': [0, 4, 7], 'octave_delta': -1, 'weight': 0.5},   # Major triad
        {'transpose_deltas': [0, 3, 7], 'octave_delta': -1, 'weight': 0.3},   # Minor triad
        {'transpose_deltas': [0, 4, 7, 11], 'octave_delta': -1, 'weight': 0.2}, # Maj7
    ],
    'melody_to_pad': [
        {'transpose_delta': 0, 'octave_delta': 0, 'sustain': True, 'weight': 0.6},
        {'transpose_delta': 7, 'octave_delta': -1, 'sustain': True, 'weight': 0.4},
    ],
}


def load_orchestration_rules(checkpoint_path):
    """
    Load learned orchestration rules from checkpoint.
    Falls back to defaults if not available.
    """
    try:
        import numpy as np
        data = np.load(checkpoint_path, allow_pickle=True)
        if 'orchestration_rules' in data:
            rules = data['orchestration_rules'].item()
            print(f"Loaded {len(rules)} orchestration rules from checkpoint")
            return rules
    except:
        pass

    print("Using default orchestration rules")
    return DEFAULT_RULES


def parse_melody_tokens(tokens):
    """
    Parse melody tokens into structured events.
    Returns list of: {'bar': int, 'pattern': str, 'transpose': int, 'octave': int, 'delta': int}
    """
    events = []
    current_bar = 0
    current_delta = 0

    i = 0
    while i < len(tokens):
        t = tokens[i]

        if t == 'BAR':
            current_bar += 1
        elif t.startswith('D'):
            current_delta = int(t[1:])
        elif t.startswith('P'):
            # Look ahead for T and O
            pattern_id = t
            transpose = 0
            octave = 0

            if i + 1 < len(tokens) and tokens[i + 1].startswith('T'):
                transpose = int(tokens[i + 1][1:])
                i += 1
            if i + 1 < len(tokens) and tokens[i + 1].startswith('O'):
                octave = int(tokens[i + 1][1:])
                i += 1

            events.append({
                'bar': current_bar,
                'pattern': pattern_id,
                'transpose': transpose,
                'octave': octave,
                'delta': current_delta,
            })
        # Skip TR: and ROLE: tokens (melody-only, single track)
        i += 1

    return events


def apply_rule_weighted(rules):
    """Select a rule based on weights."""
    total = sum(r['weight'] for r in rules)
    r = random.random() * total
    cumsum = 0
    for rule in rules:
        cumsum += rule['weight']
        if r <= cumsum:
            return rule
    return rules[-1]


def derive_bass_line(melody_events, rules):
    """
    Generate bass line from melody using rules.
    Bass typically plays on downbeats, following harmonic rhythm.
    """
    bass_events = []
    bass_rules = rules.get('melody_to_bass', DEFAULT_RULES['melody_to_bass'])

    # Bass plays on bar boundaries (downbeats)
    bars_seen = set()
    for evt in melody_events:
        bar = evt['bar']
        if bar in bars_seen:
            continue
        bars_seen.add(bar)

        rule = apply_rule_weighted(bass_rules)
        bass_transpose = (evt['transpose'] + rule['transpose_delta']) % 12
        bass_octave = evt['octave'] + rule['octave_delta']

        bass_events.append({
            'bar': bar,
            'pattern': evt['pattern'],  # Same melodic shape, different register
            'transpose': bass_transpose,
            'octave': bass_octave,
            'delta': 0,  # Downbeat
            'role': 'bass',
        })

    return bass_events


def derive_chord_hits(melody_events, rules):
    """
    Generate chord stabs from melody.
    Chords hit on beats 1 and 3 (or follow melody rhythm).
    """
    chord_events = []
    chord_rules = rules.get('melody_to_chord', DEFAULT_RULES['melody_to_chord'])

    # Chord on first event of each bar
    bars_seen = set()
    for evt in melody_events:
        bar = evt['bar']
        if bar in bars_seen:
            continue
        bars_seen.add(bar)

        rule = apply_rule_weighted(chord_rules)

        # For chords, we emit multiple notes (or a chord pattern)
        for i, t_delta in enumerate(rule['transpose_deltas']):
            chord_transpose = (evt['transpose'] + t_delta) % 12
            chord_events.append({
                'bar': bar,
                'pattern': evt['pattern'],
                'transpose': chord_transpose,
                'octave': evt['octave'] + rule['octave_delta'],
                'delta': 0 if i == 0 else 0,  # Simultaneous
                'role': 'piano',
                'voice': i,
            })

    return chord_events


def derive_pad(melody_events, rules):
    """
    Generate sustained pad from melody.
    Pad holds notes across bars.
    """
    pad_events = []
    pad_rules = rules.get('melody_to_pad', DEFAULT_RULES['melody_to_pad'])

    # Pad changes every 2-4 bars
    change_interval = 2
    last_bar = -change_interval

    for evt in melody_events:
        bar = evt['bar']
        if bar - last_bar < change_interval:
            continue
        last_bar = bar

        rule = apply_rule_weighted(pad_rules)
        pad_transpose = (evt['transpose'] + rule['transpose_delta']) % 12

        pad_events.append({
            'bar': bar,
            'pattern': evt['pattern'],
            'transpose': pad_transpose,
            'octave': evt['octave'] + rule['octave_delta'],
            'delta': 0,
            'role': 'synpad',
            'sustain': rule.get('sustain', False),
        })

    return pad_events


def events_to_tokens(melody_events, bass_events, chord_events, pad_events):
    """
    Combine all events into interleaved token stream.
    Sort by bar, then by role priority (melody first).
    """
    all_events = []

    for evt in melody_events:
        all_events.append({**evt, 'role': 'melody', 'priority': 0})
    for evt in bass_events:
        all_events.append({**evt, 'priority': 1})
    for evt in chord_events:
        all_events.append({**evt, 'priority': 2})
    for evt in pad_events:
        all_events.append({**evt, 'priority': 3})

    # Sort by bar, then priority
    all_events.sort(key=lambda x: (x['bar'], x['priority'], x.get('voice', 0)))

    tokens = []
    current_bar = -1
    current_role = None

    for evt in all_events:
        # Bar marker
        if evt['bar'] > current_bar:
            current_bar = evt['bar']
            tokens.append('BAR')

        # Role change
        role = evt['role']
        if role != current_role:
            tokens.append(f"ROLE:{role}")
            current_role = role

        # Event tokens
        tokens.append(f"D{evt['delta']}")
        tokens.append(evt['pattern'])
        tokens.append(f"T{evt['transpose']}")
        tokens.append(f"O{evt['octave']}")

    return tokens


def generate_multitrack(melody_tokens, rules, include_bass=True, include_chords=True, include_pad=False):
    """
    Main generation function.
    Takes melody tokens, returns full multi-track token sequence.
    """
    # Parse melody
    melody_events = parse_melody_tokens(melody_tokens)

    if not melody_events:
        return melody_tokens  # Return as-is if parsing failed

    # Derive accompaniment
    bass_events = derive_bass_line(melody_events, rules) if include_bass else []
    chord_events = derive_chord_hits(melody_events, rules) if include_chords else []
    pad_events = derive_pad(melody_events, rules) if include_pad else []

    # Combine
    full_tokens = events_to_tokens(melody_events, bass_events, chord_events, pad_events)

    return full_tokens


def main():
    parser = argparse.ArgumentParser(description='Rule-based multi-track generator')
    parser.add_argument('melody_file', help='Melody tokens (one sequence per line, JSON)')
    parser.add_argument('--output', '-o', default='multitrack_output.jsonl')
    parser.add_argument('--checkpoint', help='Checkpoint with orchestration rules')
    parser.add_argument('--no-bass', action='store_true')
    parser.add_argument('--no-chords', action='store_true')
    parser.add_argument('--with-pad', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("RULE-BASED MULTI-TRACK GENERATOR")
    print("=" * 60)

    # Load rules
    if args.checkpoint:
        rules = load_orchestration_rules(args.checkpoint)
    else:
        rules = DEFAULT_RULES

    # Process melodies
    print(f"\nProcessing: {args.melody_file}")
    results = []

    with open(args.melody_file) as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            melody_tokens = data.get('tokens', data.get('generated', []))

            full_tokens = generate_multitrack(
                melody_tokens,
                rules,
                include_bass=not args.no_bass,
                include_chords=not args.no_chords,
                include_pad=args.with_pad,
            )

            results.append({
                'piece_id': data.get('piece_id', f'generated_{i}'),
                'tokens': full_tokens,
                'melody_length': len(melody_tokens),
                'full_length': len(full_tokens),
            })

            if i < 2:
                print(f"\nSample {i}:")
                print(f"  Melody: {len(melody_tokens)} tokens")
                print(f"  Full: {len(full_tokens)} tokens")
                print(f"  Preview: {' '.join(full_tokens[:50])}...")

    # Save
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\nSaved {len(results)} sequences to: {args.output}")


if __name__ == '__main__':
    main()
