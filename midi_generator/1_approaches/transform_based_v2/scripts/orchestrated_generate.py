#!/usr/bin/env python3
"""
Orchestration-Driven Music Generator
=====================================

Instead of generating random pattern soup, this:
1. Picks a "lead" track from real piece occurrences
2. Uses orchestration rules to derive other tracks
3. Maintains harmonic relationships (T5, T7 = 4ths, 5ths)

The key insight: orchestration rules encode REAL musical relationships.
"""

import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from midiutil import MIDIFile


def load_checkpoint(path):
    """Load patterns and occurrences from checkpoint."""
    # The npz just has metadata - actual data in JSON files
    base = Path(path).stem
    patterns_file = Path(path).parent / f"{base}_patterns.json"

    with open(patterns_file) as f:
        patterns_raw = json.load(f)

    # Convert to int keys and extract all occurrences
    patterns = {}
    all_occurrences = []

    for pid_str, pdata in patterns_raw.items():
        pid = int(pid_str)
        patterns[pid] = pdata
        patterns[pid]['id'] = pid

        # Extract occurrences from this pattern
        for occ in pdata.get('occurrences', []):
            occ['rule_id'] = pid
            all_occurrences.append(occ)

    return patterns, all_occurrences


def load_orchestration_rules(path):
    """Load orchestration rules."""
    with open(path) as f:
        rules = json.load(f)

    # Index by source instrument
    by_source = defaultdict(list)
    for r in rules:
        by_source[r['source_instrument']].append(r)

    return rules, by_source


def get_pattern_pitches(pattern, transpose=0, octave_shift=0):
    """Get absolute pitches from pattern."""
    if 'canonical_pitches' not in pattern:
        return []

    base = pattern.get('base_pitch', 60)
    pitches = []
    for rel_pitch in pattern['canonical_pitches']:
        p = base + rel_pitch + transpose + (octave_shift * 12)
        pitches.append(max(0, min(127, p)))

    return pitches


def apply_transform(pitches, transform):
    """Apply a transform to pitches."""
    if transform == 'identity':
        return pitches

    if transform.startswith('T'):
        # Transpose
        interval = int(transform[1:])
        return [p + interval for p in pitches]

    if transform.startswith('I'):
        # Inversion around axis
        axis = int(transform[1:])
        # Invert around middle C + axis
        center = 60 + axis
        return [2 * center - p for p in pitches]

    return pitches


def group_occurrences_by_piece(occurrences):
    """Group occurrences by piece and track."""
    pieces = defaultdict(lambda: defaultdict(list))

    for occ in occurrences:
        piece_id = occ.get('piece_id', 'unknown')
        track_id = occ.get('track_id', 0)
        pieces[piece_id][track_id].append(occ)

    # Sort each track by onset
    for piece_id in pieces:
        for track_id in pieces[piece_id]:
            pieces[piece_id][track_id].sort(key=lambda x: x.get('onset', 0))

    return pieces


def select_lead_track(piece_tracks, patterns):
    """Select the best lead track (most melodic content)."""
    best_track = None
    best_score = 0

    for track_id, occs in piece_tracks.items():
        # Score based on number of occurrences and pattern variety
        pattern_ids = set(o.get('rule_id', o.get('pattern_id', 0)) for o in occs)
        score = len(occs) * len(pattern_ids)

        if score > best_score:
            best_score = score
            best_track = track_id

    return best_track


def generate_orchestrated(patterns, occurrences, rules_by_source,
                          duration_bars=16, tempo=120):
    """Generate multi-track music using orchestration rules."""

    # Group occurrences by piece
    pieces = group_occurrences_by_piece(occurrences)

    # Pick a random source piece with multiple tracks
    multi_track_pieces = {k: v for k, v in pieces.items() if len(v) >= 2}
    if not multi_track_pieces:
        multi_track_pieces = pieces

    source_piece = random.choice(list(multi_track_pieces.keys()))
    piece_tracks = pieces[source_piece]

    print(f"Source piece: {source_piece}")
    print(f"  Tracks: {list(piece_tracks.keys())}")

    # Select lead track
    lead_track_id = select_lead_track(piece_tracks, patterns)
    lead_occs = piece_tracks[lead_track_id]

    print(f"  Lead track: {lead_track_id} ({len(lead_occs)} patterns)")

    # Create MIDI
    midi = MIDIFile(numTracks=5, deinterleave=False)
    ticks_per_beat = 480

    # Track assignments
    track_roles = {
        0: ("Lead", 0, 100),      # Piano
        1: ("Bass", 32, 90),      # Acoustic Bass
        2: ("Chords", 4, 80),     # Electric Piano
        3: ("Strings", 48, 70),   # Strings
        4: ("Counter", 73, 75),   # Flute
    }

    for track_idx, (role, program, velocity) in track_roles.items():
        midi.addTrackName(track_idx, 0, role)
        midi.addProgramChange(track_idx, track_idx, 0, program)
        midi.addTempo(track_idx, 0, tempo)

    # Orchestration transforms for derived tracks
    derived_transforms = {
        1: ('T5', -12),   # Bass: 4th down, octave down
        2: ('T7', 0),     # Chords: 5th up
        3: ('identity', 0),  # Strings: double lead
        4: ('T7', 12),    # Counter: 5th up, octave up
    }

    # Limit duration
    max_time = duration_bars * 4 * ticks_per_beat  # 4 beats per bar

    notes_added = defaultdict(int)

    for occ in lead_occs:
        onset = occ.get('onset', 0)
        if onset > max_time:
            break

        rule_id = occ.get('rule_id', occ.get('pattern_id', 0))
        if rule_id not in patterns:
            continue

        pattern = patterns[rule_id]
        transpose = occ.get('transpose', 0)
        octave = occ.get('octave_shift', 0)

        # Get lead pitches
        lead_pitches = get_pattern_pitches(pattern, transpose, octave)
        if not lead_pitches:
            continue

        # Get durations from pattern
        if 'durations' in pattern:
            durations = list(pattern['durations'])
        else:
            durations = [ticks_per_beat // 2] * len(lead_pitches)

        # Add lead notes
        time_beats = onset / ticks_per_beat
        for i, (pitch, dur) in enumerate(zip(lead_pitches, durations)):
            note_time = time_beats + (i * 0.25)  # Slight offset for each note
            dur_beats = max(0.1, dur / ticks_per_beat)
            midi.addNote(0, 0, pitch, note_time, dur_beats, track_roles[0][2])
            notes_added[0] += 1

        # Add derived tracks using transforms
        for track_idx, (transform, octave_offset) in derived_transforms.items():
            derived_pitches = apply_transform(lead_pitches, transform)
            derived_pitches = [p + octave_offset for p in derived_pitches]

            # Clamp to valid MIDI range
            derived_pitches = [max(0, min(127, p)) for p in derived_pitches]

            # Add with slight timing offset for each track
            timing_offset = track_idx * 0.02

            for i, (pitch, dur) in enumerate(zip(derived_pitches, durations)):
                note_time = time_beats + (i * 0.25) + timing_offset
                dur_beats = max(0.1, dur / ticks_per_beat)
                velocity = track_roles[track_idx][2]
                midi.addNote(track_idx, track_idx, pitch, note_time, dur_beats, velocity)
                notes_added[track_idx] += 1

    print(f"\nNotes per track:")
    for track_idx, count in notes_added.items():
        print(f"  {track_roles[track_idx][0]}: {count}")

    return midi


def main():
    parser = argparse.ArgumentParser(description='Orchestration-driven music generator')
    parser.add_argument('checkpoint', help='Path to checkpoint.npz')
    parser.add_argument('--orchestration', '-r', help='Path to orchestration rules JSON')
    parser.add_argument('--output', '-o', default='orchestrated.mid', help='Output MIDI file')
    parser.add_argument('--bars', '-b', type=int, default=16, help='Number of bars')
    parser.add_argument('--tempo', '-t', type=int, default=100, help='Tempo BPM')
    parser.add_argument('--seed', '-s', type=int, help='Random seed')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    print("=" * 60)
    print("ORCHESTRATION-DRIVEN GENERATOR")
    print("=" * 60)

    # Load data
    print(f"\nLoading checkpoint: {args.checkpoint}")
    patterns, occurrences = load_checkpoint(args.checkpoint)
    print(f"  Patterns: {len(patterns)}")
    print(f"  Occurrences: {len(occurrences)}")

    # Load orchestration rules if provided
    rules_by_source = {}
    if args.orchestration:
        print(f"\nLoading orchestration rules: {args.orchestration}")
        rules, rules_by_source = load_orchestration_rules(args.orchestration)
        print(f"  Rules: {len(rules)}")

    # Generate
    print(f"\nGenerating {args.bars} bars at {args.tempo} BPM...")
    midi = generate_orchestrated(
        patterns, occurrences, rules_by_source,
        duration_bars=args.bars, tempo=args.tempo
    )

    # Save
    with open(args.output, 'wb') as f:
        midi.writeFile(f)

    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
