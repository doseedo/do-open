#!/usr/bin/env python3
"""
CHORD-CONDITIONED PATTERN GENERATION
====================================
Retrieval-based generation guided by chord progression.
No ML - just harmonically-indexed pattern lookup.
"""
import json
import random
import sys
import os
from collections import defaultdict
from midiutil import MIDIFile

os.chdir('/home/arlo/do-repo/midi_generator/1_approaches/transform_based')
sys.path.insert(0, '/home/arlo/do-repo/midi_generator/allharmony')

from midianal import CHORD_DEFINITIONS, PITCH_CLASSES

print("=" * 60)
print("CHORD-CONDITIONED PATTERN GENERATION")
print("=" * 60)

# Load patterns
with open('needed_patterns.json') as f:
    patterns = json.load(f)
print(f"Loaded {len(patterns)} patterns")

# GM program to instrument role
INSTRUMENT_ROLES = {
    0: 'piano', 24: 'guitar', 25: 'guitar', 27: 'guitar',
    32: 'bass', 33: 'bass',
    56: 'trumpet', 57: 'trombone', 60: 'horn',
    65: 'sax', 66: 'sax', 67: 'sax',
    72: 'flute', 73: 'flute',
    128: 'drums'
}

def get_chord_pitch_classes(chord_name):
    """Parse chord name and return pitch class set."""
    chord_name = chord_name.strip()
    if not chord_name:
        return set()

    # Extract root
    root = chord_name[0].upper()
    idx = 1
    if len(chord_name) > 1 and chord_name[1] in '#b':
        root = root + chord_name[1]
        idx = 2

    chord_type = chord_name[idx:] if idx < len(chord_name) else ''

    # Normalize chord type
    type_map = {
        '': 'maj', 'M': 'maj', 'maj': 'maj', 'major': 'maj',
        'm': 'min', 'min': 'min', 'minor': 'min', '-': 'min',
        '7': '7', 'dom7': '7',
        'm7': 'm7', 'min7': 'm7', '-7': 'm7',
        'maj7': 'maj7', 'M7': 'maj7', 'Δ7': 'maj7', 'Δ': 'maj7',
        'dim': 'dim', 'o': 'dim', 'dim7': 'dim',
        'm7b5': 'm7b5', 'ø': 'm7b5', 'ø7': 'm7b5',
        '9': '9', 'dom9': '9',
        'm9': 'm9', 'min9': 'm9',
        'maj9': 'maj9', 'M9': 'maj9',
        'sus4': 'sus4', 'sus': 'sus4',
        '13': '13', 'dom13': '13',
    }
    chord_type = type_map.get(chord_type, chord_type)

    # Get root pitch class
    root_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    root_pc = root_map.get(root, 0)

    # Get intervals
    if chord_type in CHORD_DEFINITIONS:
        intervals = CHORD_DEFINITIONS[chord_type]['intervals']
    else:
        intervals = {0, 4, 7}  # Default major

    return {(root_pc + i) % 12 for i in intervals}


def build_chord_index(patterns):
    """Index patterns by compatible chords."""
    # Group patterns by pitch class set
    by_pc_set = defaultdict(list)

    for name, p in patterns.items():
        pitches = p.get('canonical_pitches', [])
        if len(pitches) < 2:
            continue

        pcs = frozenset(pitch % 12 for pitch in pitches)
        gm = p.get('gm_program', 0)
        role = INSTRUMENT_ROLES.get(gm, 'other')

        by_pc_set[pcs].append({
            'name': name,
            'pattern': p,
            'pitch_classes': pcs,
            'gm_program': gm,
            'role': role,
            'n_notes': len(pitches)
        })

    return by_pc_set


def find_patterns_for_chord(chord_name, pc_index, role_filter=None, min_notes=2, max_notes=8):
    """Find patterns compatible with a chord."""
    chord_pcs = get_chord_pitch_classes(chord_name)
    if not chord_pcs:
        return []

    compatible = []
    for pc_set, pattern_list in pc_index.items():
        # Pattern pitch classes must be subset of chord
        if pc_set.issubset(chord_pcs):
            for p in pattern_list:
                if role_filter and p['role'] not in role_filter:
                    continue
                if p['n_notes'] < min_notes or p['n_notes'] > max_notes:
                    continue
                compatible.append(p)

    # Sort by note count (prefer fuller patterns)
    compatible.sort(key=lambda x: -x['n_notes'])
    return compatible


def decode_pattern(pattern_data, time_offset, transpose=0):
    """Decode pattern to note events."""
    p = pattern_data['pattern']
    pitches = p.get('canonical_pitches', [])
    rhythm = p.get('rhythm_ratios', [])
    durations = p.get('duration_ratios', [])
    velocities = p.get('velocity_ratios', [])
    gm = pattern_data['gm_program']

    notes = []
    current_time = time_offset

    for i, pitch in enumerate(pitches):
        actual_pitch = max(21, min(108, pitch + transpose))
        dur = durations[i] * 0.4 if i < len(durations) else 0.3
        dur = max(0.1, min(2.0, dur))
        vel = velocities[i] * 80 if i < len(velocities) else 80
        vel = int(max(50, min(110, vel)))

        notes.append({
            'pitch': actual_pitch,
            'time': current_time,
            'duration': dur,
            'velocity': vel,
            'gm_program': gm
        })

        if i < len(rhythm):
            current_time += max(0.15, rhythm[i] * 0.4)
        else:
            current_time += 0.25

    return notes, current_time


def generate_from_chords(chord_progression, pc_index, beats_per_chord=4):
    """Generate multi-track MIDI from chord progression."""
    all_events = []
    current_time = 0

    # Define instrument roles for arrangement
    arrangement = {
        'bass': {'min_notes': 2, 'max_notes': 4, 'count': 1},
        'piano': {'min_notes': 3, 'max_notes': 6, 'count': 1},
        'sax': {'min_notes': 2, 'max_notes': 5, 'count': 2},
        'trumpet': {'min_notes': 2, 'max_notes': 4, 'count': 1},
        'trombone': {'min_notes': 2, 'max_notes': 4, 'count': 1},
    }

    for chord_idx, chord in enumerate(chord_progression):
        chord_start = current_time
        chord_end = chord_start + beats_per_chord

        print(f"\n[Bar {chord_idx + 1}] {chord}")

        for role, params in arrangement.items():
            role_patterns = find_patterns_for_chord(
                chord, pc_index,
                role_filter=[role],
                min_notes=params['min_notes'],
                max_notes=params['max_notes']
            )

            if not role_patterns:
                continue

            # Select patterns for this role
            selected = random.sample(role_patterns, min(params['count'], len(role_patterns)))

            for pattern_data in selected:
                # Place pattern within chord duration
                pattern_time = chord_start + random.uniform(0, 0.5)

                # Transpose to fit chord better (simple: use pattern as-is)
                notes, end_time = decode_pattern(pattern_data, pattern_time)

                # Trim notes that extend past chord
                notes = [n for n in notes if n['time'] < chord_end]

                all_events.extend(notes)
                print(f"  {role}: {pattern_data['name']} ({len(notes)} notes)")

        current_time = chord_end

    return all_events, current_time


# Build index
print("\nBuilding chord index...")
pc_index = build_chord_index(patterns)
print(f"Indexed {len(pc_index)} unique pitch-class sets")

# Test chord progression (ii-V-I-vi in C)
progression = [
    'Dm7', 'G7', 'Cmaj7', 'Am7',
    'Dm7', 'G7', 'Cmaj7', 'Cmaj7',
    'Fmaj7', 'Fm7', 'Em7', 'A7',
    'Dm7', 'G7', 'Cmaj7', 'Am7'
]

print(f"\nGenerating from progression: {' | '.join(progression[:4])}...")
events, total_duration = generate_from_chords(progression, pc_index, beats_per_chord=2)

print(f"\n{'=' * 60}")
print(f"Generated {len(events)} notes over {total_duration} beats")

if not events:
    print("ERROR: No events generated!")
    sys.exit(1)

# Create MIDI
by_program = defaultdict(list)
for e in events:
    by_program[e['gm_program']].append(e)

print(f"Tracks: {len(by_program)}")

midi = MIDIFile(len(by_program), deinterleave=False)
tempo = 120

gm_names = {
    0: 'Piano', 32: 'Acoustic Bass', 33: 'Electric Bass',
    56: 'Trumpet', 57: 'Trombone', 60: 'French Horn',
    65: 'Alto Sax', 66: 'Tenor Sax', 67: 'Baritone Sax',
    72: 'Flute', 73: 'Flute'
}

for track_idx, (gm, track_events) in enumerate(sorted(by_program.items())):
    channel = track_idx % 9
    if channel >= 9:
        channel = (channel + 1) % 16

    midi.addTempo(track_idx, 0, tempo)
    midi.addProgramChange(track_idx, channel, 0, min(127, gm))

    for e in track_events:
        midi.addNote(track_idx, channel, e['pitch'], e['time'], e['duration'], e['velocity'])

    name = gm_names.get(gm, f'GM{gm}')
    print(f"  {name}: {len(track_events)} notes")

with open('generated_chordal.mid', 'wb') as f:
    midi.writeFile(f)
print(f"\nSaved: generated_chordal.mid")

# Evaluation
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

# Sort events by time
events.sort(key=lambda x: x['time'])

# 1. Temporal analysis
total_dur = events[-1]['time'] - events[0]['time'] if events else 0
density = len(events) / max(1, total_dur)
print(f"Duration: {total_dur:.1f} beats ({total_dur/2:.1f} seconds at 120bpm)")
print(f"Note density: {density:.2f} notes/beat")

# 2. Polyphony
from collections import Counter
time_buckets = Counter(round(e['time'], 1) for e in events)
max_poly = max(time_buckets.values()) if time_buckets else 0
avg_poly = sum(time_buckets.values()) / len(time_buckets) if time_buckets else 0
print(f"Max polyphony: {max_poly}")
print(f"Avg polyphony: {avg_poly:.1f}")

# 3. Pitch analysis
all_pitches = [e['pitch'] for e in events]
print(f"Pitch range: {min(all_pitches)} - {max(all_pitches)}")
print(f"Unique pitches: {len(set(all_pitches))}")

# 4. Harmonic coherence - check if notes fit chord at each time
print("\nHarmonic coherence check:")
chord_times = [(i * 2, progression[i]) for i in range(len(progression))]

correct = 0
total_checked = 0
for e in events:
    # Find which chord this note belongs to
    chord_idx = int(e['time'] / 2)
    if chord_idx >= len(progression):
        continue

    chord = progression[chord_idx]
    chord_pcs = get_chord_pitch_classes(chord)
    note_pc = e['pitch'] % 12

    if note_pc in chord_pcs:
        correct += 1
    total_checked += 1

if total_checked > 0:
    coherence = correct / total_checked
    print(f"  Notes fitting chord: {correct}/{total_checked} ({coherence:.1%})")

# 5. Instrument balance
print("\nInstrument balance:")
for gm, track_events in sorted(by_program.items()):
    pct = len(track_events) / len(events) * 100
    name = gm_names.get(gm, f'GM{gm}')
    print(f"  {name}: {pct:.0f}%")

# 6. Melodic analysis per track
print("\nMelodic intervals per track:")
for gm, track_events in sorted(by_program.items()):
    track_events.sort(key=lambda x: x['time'])
    pitches = [e['pitch'] for e in track_events]
    intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
    if intervals:
        avg_int = sum(intervals) / len(intervals)
        big_jumps = sum(1 for i in intervals if i > 12)
        name = gm_names.get(gm, f'GM{gm}')
        print(f"  {name}: avg interval {avg_int:.1f}, big jumps {big_jumps}/{len(intervals)}")

print("\n" + "=" * 60)
