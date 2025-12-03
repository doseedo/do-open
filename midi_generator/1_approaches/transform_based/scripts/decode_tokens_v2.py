#!/usr/bin/env python3
"""
Decode V2 Token Sequences to MIDI
==================================

Handles multi-track tokens with:
  BAR       - Bar boundary marker
  TR:n      - Switch to track n
  ROLE:xxx  - Track role (piano, bass, etc.)
  D{0-7}    - Delta time bucket
  P{id}     - Pattern ID
  T{0-11}   - Transpose
  O{-2..2}  - Octave offset

Creates proper multi-track MIDI output.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


# Delta time bucket to ticks mapping
DELTA_BUCKETS = {
    0: 0,
    1: 120,
    2: 240,
    3: 480,
    4: 960,
    5: 1920,
    6: 3840,
    7: 7680,
}

# Role to GM program mapping
ROLE_TO_PROGRAM = {
    'piano': 0,
    'chromperc': 11,  # Vibraphone
    'organ': 19,
    'guitar': 25,
    'bass': 33,
    'strings': 48,
    'ensemble': 48,
    'brass': 61,
    'reed': 66,
    'pipe': 73,
    'synlead': 80,
    'synpad': 88,
    'synfx': 96,
    'ethnic': 104,
    'perc': 114,
    'sfx': 120,
    'unknown': 0,
}


def load_checkpoint(checkpoint_path: str):
    """Load pattern codec from checkpoint."""
    ckpt = np.load(checkpoint_path, allow_pickle=True)

    patterns_file = ckpt.get('patterns_json_file', [None])[0]
    if patterns_file:
        patterns_path = Path(checkpoint_path).parent / patterns_file
        with open(patterns_path) as f:
            rules = json.load(f)
    else:
        rules = json.loads(str(ckpt['patterns_json'][0]))

    return rules


def parse_tokens_v2(token_string: str) -> List[Tuple[str, any]]:
    """Parse V2 token string into list of (type, value) tuples."""
    tokens = token_string.strip().split()
    parsed = []

    for tok in tokens:
        if tok in ['<pad>', '<bos>', '<eos>', '<unk>']:
            continue

        if tok == 'BAR':
            parsed.append(('BAR', None))
        elif tok.startswith('TR:'):
            try:
                track_id = int(tok[3:])
                parsed.append(('TR', track_id))
            except ValueError:
                continue
        elif tok.startswith('ROLE:'):
            role = tok[5:]
            parsed.append(('ROLE', role))
        elif tok.startswith('D') and len(tok) >= 2:
            try:
                parsed.append(('D', int(tok[1:])))
            except ValueError:
                continue
        elif tok.startswith('P') and len(tok) >= 2:
            try:
                parsed.append(('P', int(tok[1:])))
            except ValueError:
                continue
        elif tok.startswith('T') and len(tok) >= 2:
            try:
                parsed.append(('T', int(tok[1:])))
            except ValueError:
                continue
        elif tok.startswith('O') and len(tok) >= 2:
            try:
                parsed.append(('O', int(tok[1:])))
            except ValueError:
                continue

    return parsed


def decode_to_tracks(
    tokens: List[Tuple[str, any]],
    rules: dict,
    base_octave: int = 5,
    tau_offset: int = 480,
    velocity: int = 90,
) -> Dict[int, List[Dict]]:
    """Decode tokens to per-track note lists."""
    tracks = defaultdict(list)
    track_programs = {}

    current_time = 0
    current_track = 0
    current_role = 'piano'

    # State for current pattern
    pending_pattern = None
    pending_transpose = 0
    pending_octave = 0

    def emit_pending():
        nonlocal pending_pattern, pending_transpose, pending_octave
        if pending_pattern is not None:
            pattern_notes = pattern_to_notes(
                rules.get(str(pending_pattern), {}),
                transpose=pending_transpose,
                octave_offset=pending_octave,
                base_octave=base_octave,
                tau_offset=tau_offset,
                velocity=velocity,
                start_time=current_time,
            )
            tracks[current_track].extend(pattern_notes)
            pending_pattern = None
            pending_transpose = 0
            pending_octave = 0

    for tok_type, tok_value in tokens:
        if tok_type == 'BAR':
            # Bar marker - just a structural hint, no action needed
            pass

        elif tok_type == 'TR':
            # Track switch - emit pending, switch track
            emit_pending()
            current_track = tok_value

        elif tok_type == 'ROLE':
            # Track role - record program for this track
            current_role = tok_value
            track_programs[current_track] = ROLE_TO_PROGRAM.get(tok_value, 0)

        elif tok_type == 'D':
            # Delta time - emit pending pattern, advance clock
            emit_pending()
            current_time += DELTA_BUCKETS.get(tok_value, 480)

        elif tok_type == 'P':
            # Pattern token - store for emission
            emit_pending()
            pending_pattern = tok_value
            pending_transpose = 0
            pending_octave = 0

        elif tok_type == 'T':
            # Transpose token
            pending_transpose = tok_value % 12

        elif tok_type == 'O':
            # Octave token
            pending_octave = tok_value

    # Emit final pattern
    emit_pending()

    return dict(tracks), track_programs


def pattern_to_notes(
    pattern: dict,
    transpose: int = 0,
    octave_offset: int = 0,
    base_octave: int = 5,
    tau_offset: int = 480,
    velocity: int = 90,
    start_time: int = 0,
) -> List[Dict]:
    """Convert a single pattern to notes."""
    if not pattern:
        return []

    notes = []
    time = start_time

    # Use canonical_pitches if available (already has correct intervals)
    canonical_pitches = pattern.get('canonical_pitches', [])
    if canonical_pitches:
        # canonical_pitches are absolute relative to base 60
        for i, cp in enumerate(canonical_pitches):
            pitch = cp + transpose + (octave_offset * 12)

            # Clamp to MIDI range
            while pitch > 108:
                pitch -= 12
            while pitch < 24:
                pitch += 12

            # Get ratios
            duration_ratios = pattern.get('duration_ratios', [1.0] * len(canonical_pitches))
            velocity_ratios = pattern.get('velocity_ratios', [1.0] * len(canonical_pitches))
            rhythm_ratios = pattern.get('rhythm_ratios', [1.0])

            dur_ratio = duration_ratios[i] if i < len(duration_ratios) else 1.0
            dur_ratio = max(0.1, min(4.0, dur_ratio))
            duration = int(dur_ratio * tau_offset)
            duration = max(60, duration)

            vel_ratio = velocity_ratios[i] if i < len(velocity_ratios) else 1.0
            vel = int(vel_ratio * velocity)
            vel = min(127, max(1, vel))

            notes.append({
                'pitch': pitch,
                'velocity': vel,
                'time': time,
                'duration': duration,
            })

            # Advance time
            if i < len(rhythm_ratios):
                rr = rhythm_ratios[i]
                rr = max(0, min(4.0, rr))
                time += int(rr * tau_offset)

        return notes

    # Fallback to pitch_classes + intervals
    pitch_classes = pattern.get('pitch_classes', [])
    if not pitch_classes:
        return []

    rhythm_ratios = pattern.get('rhythm_ratios', [1.0])
    duration_ratios = pattern.get('duration_ratios', [1.0] * len(pitch_classes))
    velocity_ratios = pattern.get('velocity_ratios', [1.0] * len(pitch_classes))
    pitch_intervals = pattern.get('pitch_intervals', [])

    actual_octave = base_octave + octave_offset
    prev_pitch = actual_octave * 12 + ((pitch_classes[0] + transpose) % 12)

    for i, pc in enumerate(pitch_classes):
        if i == 0:
            pitch = actual_octave * 12 + ((pc + transpose) % 12)
        else:
            if i - 1 < len(pitch_intervals):
                pitch = prev_pitch + pitch_intervals[i - 1]
            else:
                # Heuristic
                prev_pc = (pitch_classes[i - 1] + transpose) % 12
                curr_pc = (pc + transpose) % 12
                diff = (curr_pc - prev_pc) % 12
                if diff > 6:
                    diff -= 12
                pitch = prev_pitch + diff

        while pitch > 108:
            pitch -= 12
        while pitch < 24:
            pitch += 12

        prev_pitch = pitch

        vel_ratio = velocity_ratios[i] if i < len(velocity_ratios) else 1.0
        vel = int(vel_ratio * velocity)
        vel = min(127, max(1, vel))

        dur_ratio = duration_ratios[i] if i < len(duration_ratios) else 1.0
        dur_ratio = max(0.1, min(4.0, dur_ratio))
        duration = int(dur_ratio * tau_offset)
        duration = max(60, duration)

        notes.append({
            'pitch': pitch,
            'velocity': vel,
            'time': time,
            'duration': duration,
        })

        if i < len(rhythm_ratios):
            rr = rhythm_ratios[i]
            rr = max(0, min(4.0, rr))
            time += int(rr * tau_offset)

    return notes


def tracks_to_midi(
    tracks: Dict[int, List[Dict]],
    track_programs: Dict[int, int],
    output_path: str,
    tempo_bpm: int = 120
):
    """Save multi-track notes to MIDI file."""
    from midiutil import MIDIFile

    # Get sorted track IDs
    track_ids = sorted(tracks.keys())
    n_tracks = len(track_ids)

    if n_tracks == 0:
        print("No tracks to save!")
        return

    midi = MIDIFile(numTracks=n_tracks, deinterleave=False)
    ticks_per_beat = 480

    for midi_track, track_id in enumerate(track_ids):
        notes = tracks[track_id]
        program = track_programs.get(track_id, 0)

        midi.addTrackName(midi_track, 0, f"Track {track_id}")
        midi.addTempo(midi_track, 0, tempo_bpm)
        midi.addProgramChange(midi_track, midi_track, 0, program)

        for note in notes:
            time_beats = note['time'] / ticks_per_beat
            dur_beats = note['duration'] / ticks_per_beat
            dur_beats = max(0.1, dur_beats)

            midi.addNote(
                track=midi_track,
                channel=midi_track % 16,
                pitch=note['pitch'],
                time=time_beats,
                duration=dur_beats,
                volume=note['velocity']
            )

    with open(output_path, 'wb') as f:
        midi.writeFile(f)

    print(f"Saved MIDI to: {output_path}")
    print(f"  Tracks: {n_tracks}")
    for midi_track, track_id in enumerate(track_ids):
        program = track_programs.get(track_id, 0)
        n_notes = len(tracks[track_id])
        print(f"    Track {track_id}: {n_notes} notes, program {program}")


def main():
    parser = argparse.ArgumentParser(description='Decode V2 tokens to multi-track MIDI')
    parser.add_argument('checkpoint', help='Path to checkpoint .npz file')
    parser.add_argument('tokens', help='Token file')
    parser.add_argument('--output', '-o', default='decoded_v2.mid', help='Output MIDI file')
    parser.add_argument('--tempo', '-t', type=int, default=120, help='Tempo BPM')
    parser.add_argument('--velocity', '-v', type=int, default=90, help='Base velocity')
    args = parser.parse_args()

    print("=" * 60)
    print("DECODE V2 TOKENS TO MULTI-TRACK MIDI")
    print("=" * 60)

    # Load codec
    print(f"\nLoading codec: {args.checkpoint}")
    rules = load_checkpoint(args.checkpoint)
    print(f"  Loaded {len(rules)} patterns")

    # Load tokens
    print(f"\nLoading tokens: {args.tokens}")
    with open(args.tokens) as f:
        content = f.read().strip()

    # Check if JSONL format
    if content.startswith('{'):
        first_line = content.split('\n')[0]
        data = json.loads(first_line)
        token_string = ' '.join(data['tokens'])
    else:
        token_string = content

    tokens = parse_tokens_v2(token_string)
    print(f"  Parsed {len(tokens)} tokens")

    # Count token types
    type_counts = defaultdict(int)
    for t, v in tokens:
        type_counts[t] += 1
    print(f"  Token types: {dict(type_counts)}")

    # Decode
    print("\nDecoding...")
    tracks, track_programs = decode_to_tracks(tokens, rules, velocity=args.velocity)

    total_notes = sum(len(notes) for notes in tracks.values())
    print(f"  Generated {total_notes} notes across {len(tracks)} tracks")

    # Save
    tracks_to_midi(tracks, track_programs, args.output, tempo_bpm=args.tempo)

    print("\nDone!")


if __name__ == '__main__':
    main()
