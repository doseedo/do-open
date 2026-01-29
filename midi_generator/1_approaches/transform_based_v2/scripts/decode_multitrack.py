#!/usr/bin/env python3
"""
Multitrack Token Decoder
========================

Properly decodes transformer-generated token sequences to multitrack MIDI,
respecting TR:X (track), ROLE:X (instrument), BAR, and pattern tokens.

Token format from corpus:
  BAR           - Bar marker
  TR:{id}       - Track ID
  ROLE:{name}   - Instrument role (brass, strings, keys, etc.)
  D{bucket}     - Delta time (0-7)
  P{id}         - Pattern ID
  T{offset}     - Transpose (0-11)
  O{octave}     - Octave offset (-2 to +2)

Usage:
  python decode_multitrack.py checkpoint.npz tokens.jsonl -o output.mid
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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
    'keys': 4,
    'organ': 19,
    'guitar': 25,
    'bass': 33,
    'strings': 48,
    'brass': 61,
    'winds': 73,
    'reed': 65,
    'pipe': 73,       # Pan flute / pipes
    'synth': 80,
    'lead': 81,
    'pad': 88,
    'chromperc': 8,   # Celesta/Glockenspiel
    'percussion': 0,  # Channel 10
    'drums': 0,       # Channel 10
    'voice': 52,
}

# Role to suggested octave offset
ROLE_OCTAVES = {
    'bass': -2,
    'piano': 0,
    'keys': 0,
    'guitar': 0,
    'strings': 0,
    'brass': 0,
    'winds': 0,
    'reed': 0,
    'synth': 0,
    'lead': 1,
    'pad': 0,
    'voice': 0,
}


def load_patterns(checkpoint_path: str) -> dict:
    """Load pattern codec from checkpoint."""
    ckpt = np.load(checkpoint_path, allow_pickle=True)

    patterns_file = ckpt.get('patterns_json_file', [None])[0]
    if patterns_file:
        patterns_path = Path(checkpoint_path).parent / patterns_file
        with open(patterns_path) as f:
            return json.load(f)
    else:
        return json.loads(str(ckpt['patterns_json'][0]))


def parse_tokens(tokens: List[str]) -> List[Tuple[str, any]]:
    """Parse token list into (type, value) tuples."""
    parsed = []

    for tok in tokens:
        if tok in ['<pad>', '<bos>', '<eos>', '<unk>']:
            continue

        if tok == 'BAR':
            parsed.append(('BAR', None))
        elif tok.startswith('TR:'):
            try:
                parsed.append(('TR', int(tok[3:])))
            except ValueError:
                pass
        elif tok.startswith('ROLE:'):
            parsed.append(('ROLE', tok[5:]))
        elif tok.startswith('D') and len(tok) >= 2:
            try:
                parsed.append(('D', int(tok[1:])))
            except ValueError:
                pass
        elif tok.startswith('P') and len(tok) >= 2:
            try:
                parsed.append(('P', int(tok[1:])))
            except ValueError:
                pass
        elif tok.startswith('T') and len(tok) >= 2:
            try:
                parsed.append(('T', int(tok[1:])))
            except ValueError:
                pass
        elif tok.startswith('O') and len(tok) >= 2:
            try:
                val = tok[1:]
                parsed.append(('O', int(val)))
            except ValueError:
                pass

    return parsed


def pattern_to_notes(
    pattern: dict,
    start_time: int,
    transpose: int = 0,
    octave_offset: int = 0,
    role_octave: int = 0,
    tau_offset: int = 480,
    velocity: int = 90,
) -> List[Dict]:
    """Convert a pattern to note list."""
    if not pattern:
        return []

    notes = []
    time = start_time

    pitch_classes = pattern.get('pitch_classes', [])
    if not pitch_classes:
        return []

    rhythm_ratios = pattern.get('rhythm_ratios', [1.0])
    duration_ratios = pattern.get('duration_ratios', [1.0] * len(pitch_classes))
    velocity_ratios = pattern.get('velocity_ratios', [1.0] * len(pitch_classes))
    pitch_intervals = pattern.get('pitch_intervals', [])

    # Base octave adjusted by role and explicit offset
    base_octave = 5 + role_octave + octave_offset
    use_intervals = len(pitch_intervals) == len(pitch_classes) - 1

    prev_pitch = base_octave * 12 + ((pitch_classes[0] + transpose) % 12)

    for i, pc in enumerate(pitch_classes):
        # Compute pitch
        if i == 0:
            pitch = base_octave * 12 + ((pc + transpose) % 12)
        else:
            if use_intervals:
                pitch = prev_pitch + pitch_intervals[i - 1]
            else:
                prev_pc = (pitch_classes[i - 1] + transpose) % 12
                curr_pc = (pc + transpose) % 12
                diff = (curr_pc - prev_pc) % 12
                if diff > 6:
                    diff -= 12
                pitch = prev_pitch + diff

        # Clamp to MIDI range
        pitch = max(21, min(108, pitch))
        prev_pitch = pitch

        # Velocity
        vel_ratio = velocity_ratios[i] if i < len(velocity_ratios) else 1.0
        vel = int(vel_ratio * velocity)
        vel = max(1, min(127, vel))

        # Duration
        dur_ratio = duration_ratios[i] if i < len(duration_ratios) else 1.0
        dur_ratio = max(0.1, min(4.0, dur_ratio))
        duration = max(60, int(dur_ratio * tau_offset))

        notes.append({
            'pitch': pitch,
            'velocity': vel,
            'time': time,
            'duration': duration,
        })

        # Advance time for next note in pattern
        if i < len(rhythm_ratios):
            rr = max(0, min(4.0, rhythm_ratios[i]))
            time += int(rr * tau_offset)

    return notes


def decode_multitrack(
    tokens: List[Tuple[str, any]],
    patterns: dict,
    ticks_per_bar: int = 1920,
    tau_offset: int = 480,
    base_velocity: int = 90,
) -> Dict[int, Dict]:
    """
    Decode tokens to multitrack structure.

    Returns: {track_id: {'role': str, 'notes': [...]}}
    """
    tracks = defaultdict(lambda: {'role': 'piano', 'notes': []})

    current_bar = 0
    current_track = 0
    current_role = 'piano'
    current_time = 0  # Time within current track context

    # Pending pattern state
    pending_pattern = None
    pending_transpose = 0
    pending_octave = 0

    def emit_pattern():
        nonlocal pending_pattern, pending_transpose, pending_octave
        if pending_pattern is None:
            return

        pattern_data = patterns.get(str(pending_pattern), {})
        role_octave = ROLE_OCTAVES.get(current_role, 0)

        notes = pattern_to_notes(
            pattern_data,
            start_time=current_bar * ticks_per_bar + current_time,
            transpose=pending_transpose,
            octave_offset=pending_octave,
            role_octave=role_octave,
            tau_offset=tau_offset,
            velocity=base_velocity,
        )

        tracks[current_track]['role'] = current_role
        tracks[current_track]['notes'].extend(notes)

        pending_pattern = None
        pending_transpose = 0
        pending_octave = 0

    for tok_type, tok_value in tokens:
        if tok_type == 'BAR':
            # Emit pending pattern before bar change
            emit_pattern()
            current_bar += 1
            current_time = 0

        elif tok_type == 'TR':
            # Emit pending pattern before track change
            emit_pattern()
            current_track = tok_value
            current_time = 0  # Reset time for new track

        elif tok_type == 'ROLE':
            current_role = tok_value
            tracks[current_track]['role'] = current_role

        elif tok_type == 'D':
            # Delta time - advance clock within current track
            current_time += DELTA_BUCKETS.get(tok_value, 0)

        elif tok_type == 'P':
            # Emit previous pattern, store new one
            emit_pattern()
            pending_pattern = tok_value

        elif tok_type == 'T':
            pending_transpose = tok_value % 12

        elif tok_type == 'O':
            pending_octave = tok_value

    # Emit final pattern
    emit_pattern()

    return dict(tracks)


def tracks_to_midi(tracks: Dict[int, Dict], output_path: str, tempo_bpm: int = 120):
    """Save multitrack structure to MIDI file."""
    try:
        import mido
    except ImportError:
        print("Error: mido not installed. Run: pip install mido")
        return

    mid = mido.MidiFile(ticks_per_beat=480)

    # Sort tracks by ID
    sorted_track_ids = sorted(tracks.keys())

    for track_id in sorted_track_ids:
        track_data = tracks[track_id]
        role = track_data['role']
        notes = track_data['notes']

        if not notes:
            continue

        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Set track name
        track.append(mido.MetaMessage('track_name', name=f"{role}_{track_id}", time=0))

        # Set tempo on first track
        if track_id == sorted_track_ids[0]:
            tempo = mido.bpm2tempo(tempo_bpm)
            track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

        # Set program (instrument)
        program = ROLE_TO_PROGRAM.get(role, 0)
        channel = 9 if role in ['drums', 'percussion'] else min(track_id % 16, 15)
        if channel == 9 and role not in ['drums', 'percussion']:
            channel = (track_id % 15)  # Skip channel 10 for non-drums

        track.append(mido.Message('program_change', program=program, channel=channel, time=0))

        # Build events
        events = []
        for note in notes:
            events.append((note['time'], 'note_on', note['pitch'], note['velocity'], channel))
            events.append((note['time'] + note['duration'], 'note_off', note['pitch'], 0, channel))

        events.sort(key=lambda e: (e[0], 0 if e[1] == 'note_off' else 1))

        # Convert to delta times
        prev_time = 0
        for event in events:
            abs_time, msg_type, pitch, velocity, ch = event
            delta = max(0, abs_time - prev_time)

            if msg_type == 'note_on':
                track.append(mido.Message('note_on', note=pitch, velocity=velocity, channel=ch, time=delta))
            else:
                track.append(mido.Message('note_off', note=pitch, velocity=0, channel=ch, time=delta))

            prev_time = abs_time

        track.append(mido.MetaMessage('end_of_track', time=0))

    mid.save(output_path)
    print(f"Saved multitrack MIDI to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Decode tokens to multitrack MIDI')
    parser.add_argument('checkpoint', help='Path to checkpoint .npz file')
    parser.add_argument('tokens', help='Token file (JSONL or space-separated)')
    parser.add_argument('--output', '-o', default='decoded_multitrack.mid', help='Output MIDI file')
    parser.add_argument('--tempo', '-t', type=int, default=120, help='Tempo BPM')
    parser.add_argument('--velocity', '-v', type=int, default=90, help='Base velocity')
    args = parser.parse_args()

    print("=" * 60)
    print("MULTITRACK TOKEN DECODER")
    print("=" * 60)

    # Load patterns
    print(f"\nLoading patterns: {args.checkpoint}")
    patterns = load_patterns(args.checkpoint)
    print(f"  Loaded {len(patterns)} patterns")

    # Load tokens
    print(f"\nLoading tokens: {args.tokens}")
    with open(args.tokens) as f:
        content = f.read().strip()

    # Parse tokens
    if content.startswith('{'):
        # JSONL format
        first_line = content.split('\n')[0]
        data = json.loads(first_line)
        token_list = data.get('tokens', [])
        if isinstance(token_list, str):
            token_list = token_list.split()
    else:
        # Space-separated
        token_list = content.split()

    parsed = parse_tokens(token_list)
    print(f"  Parsed {len(parsed)} tokens")

    # Count token types
    type_counts = defaultdict(int)
    for t, v in parsed:
        type_counts[t] += 1
    print(f"  Token breakdown: {dict(type_counts)}")

    # Decode
    print("\nDecoding to multitrack...")
    tracks = decode_multitrack(
        parsed,
        patterns,
        base_velocity=args.velocity
    )

    total_notes = sum(len(t['notes']) for t in tracks.values())
    print(f"  Generated {len(tracks)} tracks, {total_notes} total notes")

    for tid, tdata in sorted(tracks.items()):
        print(f"    Track {tid} ({tdata['role']}): {len(tdata['notes'])} notes")

    # Save
    tracks_to_midi(tracks, args.output, tempo_bpm=args.tempo)

    print("\nDone!")


if __name__ == '__main__':
    main()
