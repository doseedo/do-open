#!/usr/bin/env python3
"""
Convert generated arrangement tokens to MIDI file.
"""

import json
import argparse
from pathlib import Path
from midiutil import MIDIFile

# Role to GM program mapping
ROLE_TO_PROGRAM = {
    'piano': 0,      # Acoustic Grand Piano
    'guitar': 25,    # Acoustic Guitar (steel)
    'bass': 32,      # Acoustic Bass
    'strings': 48,   # String Ensemble 1
    'brass': 61,     # Brass Section
    'reed': 65,      # Alto Sax
    'pipe': 73,      # Flute
    'synlead': 80,   # Lead 1 (square)
    'synpad': 88,    # Pad 1 (new age)
    'synfx': 96,     # FX 1 (rain)
    'organ': 16,     # Drawbar Organ
    'chromperc': 8,  # Celesta
    'ethnic': 104,   # Sitar
    'percussive': 112, # Tinkle Bell
    'soundfx': 120,  # Guitar Fret Noise
    'melody': 65,    # Alto Sax (default melody)
}

def load_patterns(patterns_path):
    """Load pattern definitions."""
    with open(patterns_path) as f:
        return json.load(f)

def parse_tokens(tokens):
    """Parse token sequence into events."""
    events = []
    current_bar = -1
    current_track = 0
    current_role = 'piano'
    current_delta = 0

    i = 0
    while i < len(tokens):
        t = tokens[i]

        if t == 'BAR':
            current_bar += 1
        elif t.startswith('TR:'):
            try:
                current_track = int(t[3:])
            except:
                pass
        elif t.startswith('ROLE:'):
            current_role = t[5:]
        elif t.startswith('D') and t[1:].lstrip('-').isdigit():
            current_delta = int(t[1:])
        elif t.startswith('P') and t[1:].isdigit():
            pattern_id = t[1:]  # Remove 'P' prefix
            transpose = 0
            octave = 0

            # Look ahead for T and O
            if i + 1 < len(tokens):
                next_t = tokens[i + 1]
                if next_t.startswith('T') and not next_t.startswith('TR:'):
                    try:
                        transpose = int(next_t[1:])
                        i += 1
                    except:
                        pass
            if i + 1 < len(tokens):
                next_t = tokens[i + 1]
                if next_t.startswith('O'):
                    try:
                        octave = int(next_t[1:])
                        i += 1
                    except:
                        pass

            events.append({
                'bar': current_bar,
                'track': current_track,
                'role': current_role,
                'delta': current_delta,
                'pattern_id': pattern_id,
                'transpose': transpose,
                'octave': octave,
            })
        i += 1

    return events

def events_to_midi(events, patterns, output_path, tempo=120, ticks_per_beat=480):
    """Convert events to MIDI file."""

    # Group events by track
    tracks_data = {}
    for evt in events:
        track_key = (evt['track'], evt['role'])
        if track_key not in tracks_data:
            tracks_data[track_key] = []
        tracks_data[track_key].append(evt)

    # Create MIDI file (deinterleave=False to avoid overlapping note issues)
    num_tracks = len(tracks_data)
    midi = MIDIFile(num_tracks if num_tracks > 0 else 1, deinterleave=False)

    bar_duration = 4.0  # 4 beats per bar (4/4 time)

    for track_idx, ((track_num, role), track_events) in enumerate(tracks_data.items()):
        midi.addTrackName(track_idx, 0, f"{role}_{track_num}")
        midi.addTempo(track_idx, 0, tempo)

        # Set instrument
        program = ROLE_TO_PROGRAM.get(role, 0)
        midi.addProgramChange(track_idx, track_idx % 16, 0, program)

        for evt in track_events:
            pattern_id = evt['pattern_id']

            if pattern_id not in patterns:
                # Use a default note if pattern not found
                pitch = 60 + evt['transpose'] + (evt['octave'] * 12)
                time = evt['bar'] * bar_duration + (evt['delta'] / 8.0) * bar_duration
                midi.addNote(track_idx, track_idx % 16, pitch, time, 0.5, 80)
                continue

            pattern = patterns[pattern_id]

            # Calculate base time
            base_time = evt['bar'] * bar_duration + (evt['delta'] / 8.0) * bar_duration

            # Use canonical_pitches (preserves melodic contour) with transpose and octave offset
            # T# = pitch class transposition (0-11 semitones)
            # O# = octave offset (multiplied by 12 semitones)
            canonical_pitches = pattern.get('canonical_pitches', [])

            if canonical_pitches:
                # CORRECT: Use canonical pitches + transpose + octave offset
                # This preserves the melodic shape (intervals/direction)
                pitches = [p + evt['transpose'] + (evt['octave'] * 12) for p in canonical_pitches]
            else:
                # Fallback: Use pitch_intervals to reconstruct from first pitch class
                pitch_classes = pattern.get('pitch_classes', [0])
                pitch_intervals = pattern.get('pitch_intervals', [])

                # Determine base octave from role
                role_base_pitch = {
                    'bass': 36,      # C2
                    'piano': 60,     # C4
                    'guitar': 52,    # E3
                    'strings': 60,   # C4
                    'brass': 60,     # C4
                    'reed': 64,      # E4
                    'pipe': 72,      # C5
                    'synlead': 64,   # E4
                    'synpad': 60,    # C4
                    'organ': 60,     # C4
                }.get(role, 60)

                # Start from first pitch class + role base + transpose + octave
                first_pitch = role_base_pitch + pitch_classes[0] + evt['transpose'] + (evt['octave'] * 12)
                pitches = [first_pitch]

                # Apply signed intervals to get subsequent pitches
                for interval in pitch_intervals:
                    pitches.append(pitches[-1] + interval)

            # Clamp to valid MIDI range
            pitches = [max(0, min(127, p)) for p in pitches]

            # rhythm_ratios are inter-onset ratios (N-1 values for N notes)
            rhythm_ratios = pattern.get('rhythm_ratios', [1.0])
            base_ioi = 0.5  # Base inter-onset interval in beats

            cumulative_time = 0.0
            for j, pitch in enumerate(pitches):
                # Duration from pattern
                duration_ratios = pattern.get('duration_ratios', [1.0])
                dur_ratio = duration_ratios[j] if j < len(duration_ratios) else 1.0
                duration = min(max(dur_ratio * 0.25, 0.1), 1.0)

                note_time = base_time + cumulative_time
                velocity = 80
                midi.addNote(track_idx, track_idx % 16, pitch, note_time, duration, velocity)

                # Update cumulative time for next note
                if j < len(rhythm_ratios):
                    cumulative_time += rhythm_ratios[j] * base_ioi

    # Write MIDI file
    with open(output_path, 'wb') as f:
        midi.writeFile(f)

    return len(tracks_data)

def main():
    parser = argparse.ArgumentParser(description='Convert tokens to MIDI')
    parser.add_argument('--input', '-i', default='test_multitrack_output.jsonl', help='Input JSONL file')
    parser.add_argument('--patterns', '-p', default='checkpoint_v42_1000files_patterns.json', help='Patterns file')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory')
    parser.add_argument('--sample', '-s', type=int, default=None, help='Specific sample index (default: all)')
    parser.add_argument('--tempo', '-t', type=int, default=120, help='Tempo in BPM')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    input_path = base_dir / args.input
    patterns_path = base_dir / args.patterns
    output_dir = base_dir / args.output_dir

    print(f"Loading patterns from: {patterns_path}")
    patterns = load_patterns(patterns_path)
    print(f"Loaded {len(patterns)} patterns")

    print(f"Loading samples from: {input_path}")
    with open(input_path) as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} samples")

    # Process samples
    indices = [args.sample] if args.sample is not None else range(len(samples))

    for idx in indices:
        if idx >= len(samples):
            print(f"Sample {idx} not found")
            continue

        sample = samples[idx]
        tokens = sample['tokens']

        print(f"\nProcessing sample {idx}...")
        print(f"  Tokens: {len(tokens)}")
        print(f"  Bars: {sample.get('n_bars', '?')}")
        print(f"  Roles: {sample.get('roles', {})}")

        # Parse tokens
        events = parse_tokens(tokens)
        print(f"  Events: {len(events)}")

        if not events:
            print(f"  No events found, skipping")
            continue

        # Convert to MIDI
        output_path = output_dir / f"generated_sample_{idx}.mid"
        num_tracks = events_to_midi(events, patterns, output_path, tempo=args.tempo)

        print(f"  Saved: {output_path}")
        print(f"  Tracks: {num_tracks}")

if __name__ == '__main__':
    main()
