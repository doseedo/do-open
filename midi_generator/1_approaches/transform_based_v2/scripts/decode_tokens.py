#!/usr/bin/env python3
"""
Decode Token Sequences to MIDI
==============================

Converts transformer-generated token sequences back to MIDI
using the pattern codec.

Token format:
  D{bucket} - Delta time (0-7)
  P{id}     - Pattern ID
  T{offset} - Transpose (0-11)
  O{octave} - Octave offset (-2 to +2)

Usage:
  python decode_tokens.py checkpoint.npz tokens.txt -o output.mid
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


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


def parse_tokens(token_string: str) -> List[Tuple[str, int]]:
    """Parse token string into list of (type, value) tuples."""
    tokens = token_string.strip().split()
    parsed = []

    for tok in tokens:
        if tok in ['<pad>', '<bos>', '<eos>', '<unk>']:
            continue
        if len(tok) < 2:
            continue

        tok_type = tok[0]
        try:
            tok_value = int(tok[1:])
        except ValueError:
            continue

        parsed.append((tok_type, tok_value))

    return parsed


def decode_to_notes(
    tokens: List[Tuple[str, int]],
    rules: dict,
    base_octave: int = 5,
    tau_offset: int = 480,
    velocity: int = 90,
) -> List[Dict]:
    """Decode tokens to note list."""
    notes = []
    current_time = 0

    # State for current pattern
    pending_pattern = None
    pending_transpose = 0
    pending_octave = 0

    for tok_type, tok_value in tokens:
        if tok_type == 'D':
            # Delta time - advance clock
            current_time += DELTA_BUCKETS.get(tok_value, 480)

        elif tok_type == 'P':
            # Pattern token - emit previous pattern if any, store new
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
                notes.extend(pattern_notes)

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
        notes.extend(pattern_notes)

    return notes


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

    pitch_classes = pattern.get('pitch_classes', [])
    rhythm_ratios = pattern.get('rhythm_ratios', [1.0])
    duration_ratios = pattern.get('duration_ratios', [1.0] * len(pitch_classes))
    velocity_ratios = pattern.get('velocity_ratios', [1.0] * len(pitch_classes))

    # Get signed intervals if available
    pitch_intervals = pattern.get('pitch_intervals', [])
    use_intervals = len(pitch_intervals) == len(pitch_classes) - 1

    # Adjust octave
    actual_octave = base_octave + octave_offset
    prev_pitch = actual_octave * 12 + ((pitch_classes[0] + transpose) % 12) if pitch_classes else 60

    for i, pc in enumerate(pitch_classes):
        # Compute pitch
        if i == 0:
            pitch = actual_octave * 12 + ((pc + transpose) % 12)
        else:
            if use_intervals:
                pitch = prev_pitch + pitch_intervals[i - 1]
            else:
                # Heuristic: smallest interval
                prev_pc = (pitch_classes[i - 1] + transpose) % 12
                curr_pc = (pc + transpose) % 12
                diff = (curr_pc - prev_pc) % 12
                if diff > 6:
                    diff -= 12
                pitch = prev_pitch + diff

        # Clamp to MIDI range
        while pitch > 108:
            pitch -= 12
        while pitch < 24:
            pitch += 12

        prev_pitch = pitch

        # Velocity
        vel_ratio = velocity_ratios[i] if i < len(velocity_ratios) else 1.0
        vel = int(vel_ratio * velocity)
        vel = min(127, max(1, vel))

        # Duration
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

        # Advance time
        if i < len(rhythm_ratios):
            rr = rhythm_ratios[i]
            rr = max(0, min(4.0, rr))
            time += int(rr * tau_offset)

    return notes


def notes_to_midi(notes: List[Dict], output_path: str, tempo_bpm: int = 120):
    """Save notes to MIDI file."""
    try:
        import mido
    except ImportError:
        print("Error: mido not installed. Run: pip install mido")
        return

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    # Sort notes
    sorted_notes = sorted(notes, key=lambda n: (n['time'], n['pitch']))

    # Build events
    events = []
    for note in sorted_notes:
        events.append((note['time'], 'note_on', note['pitch'], note['velocity']))
        events.append((note['time'] + note['duration'], 'note_off', note['pitch'], 0))

    events.sort(key=lambda e: (e[0], 0 if e[1] == 'note_off' else 1))

    # Convert to delta times
    prev_time = 0
    for event in events:
        abs_time, msg_type, pitch, velocity = event
        delta = abs_time - prev_time

        if msg_type == 'note_on':
            track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=delta))
        else:
            track.append(mido.Message('note_off', note=pitch, velocity=0, time=delta))

        prev_time = abs_time

    track.append(mido.MetaMessage('end_of_track', time=0))

    mid.save(output_path)
    print(f"Saved MIDI to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Decode tokens to MIDI')
    parser.add_argument('checkpoint', help='Path to checkpoint .npz file')
    parser.add_argument('tokens', help='Token file (one sequence per line, or single sequence)')
    parser.add_argument('--output', '-o', default='decoded.mid', help='Output MIDI file')
    parser.add_argument('--tempo', '-t', type=int, default=120, help='Tempo BPM')
    parser.add_argument('--velocity', '-v', type=int, default=90, help='Base velocity')
    args = parser.parse_args()

    print("=" * 60)
    print("DECODE TOKENS TO MIDI")
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
        # JSONL - take first sequence
        first_line = content.split('\n')[0]
        data = json.loads(first_line)
        token_string = ' '.join(data['tokens'])
    else:
        token_string = content

    tokens = parse_tokens(token_string)
    print(f"  Parsed {len(tokens)} tokens")

    # Decode
    print("\nDecoding...")
    notes = decode_to_notes(tokens, rules, velocity=args.velocity)
    print(f"  Generated {len(notes)} notes")

    # Save
    notes_to_midi(notes, args.output, tempo_bpm=args.tempo)

    print("\nDone!")


if __name__ == '__main__':
    main()
