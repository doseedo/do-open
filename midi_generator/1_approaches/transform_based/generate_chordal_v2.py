#!/usr/bin/env python3
"""
CHORD-CONDITIONED GENERATION v2
===============================
Uses pattern characteristics to assign proper musical roles:
- Bass: walking patterns (even rhythm, stepwise)
- Piano: chord voicings (simultaneous notes)
- Horns/Sax: melodic patterns (varied rhythm)
"""
import json
import random
import sys
import os
from collections import defaultdict
from midiutil import MIDIFile

os.chdir('/home/arlo/do-repo/midi_generator/1_approaches/transform_based')
sys.path.insert(0, '/home/arlo/do-repo/midi_generator/allharmony')

from midianal import CHORD_DEFINITIONS

print("=" * 60)
print("CHORD-CONDITIONED GENERATION v2 - Role-Aware")
print("=" * 60)

# Load patterns
with open('needed_patterns.json') as f:
    patterns = json.load(f)
print(f"Loaded {len(patterns)} patterns")


def classify_pattern(p):
    """Classify pattern by musical role based on its characteristics."""
    rhythm = p.get('rhythm_ratios', [])
    pitches = p.get('canonical_pitches', [])
    intervals = p.get('pitch_intervals', [])

    if len(pitches) < 2:
        return 'fragment'

    # Check for simultaneous notes (chord voicing)
    has_simultaneous = any(abs(r) < 0.1 for r in rhythm) if rhythm else False

    # Check for even rhythm (walking/comping)
    all_even = all(0.8 <= r <= 1.2 for r in rhythm) if rhythm and len(rhythm) > 0 else False

    # Check for stepwise motion
    avg_interval = sum(abs(i) for i in intervals) / len(intervals) if intervals else 0
    is_stepwise = avg_interval <= 3

    # Classify
    if has_simultaneous and len(pitches) >= 3:
        return 'chord'  # Chord voicing
    elif all_even and is_stepwise and len(pitches) >= 3:
        return 'walking'  # Walking bass line
    elif all_even and len(pitches) >= 2:
        return 'comping'  # Even rhythm but not stepwise
    else:
        return 'melodic'  # Varied rhythm = melodic line


def get_chord_pitch_classes(chord_name):
    """Parse chord name and return pitch class set."""
    chord_name = chord_name.strip()
    if not chord_name:
        return set()

    root = chord_name[0].upper()
    idx = 1
    if len(chord_name) > 1 and chord_name[1] in '#b':
        root = root + chord_name[1]
        idx = 2

    chord_type = chord_name[idx:] if idx < len(chord_name) else ''

    type_map = {
        '': 'maj', 'M': 'maj', 'maj': 'maj',
        'm': 'min', 'min': 'min', '-': 'min',
        '7': '7', 'm7': 'm7', 'min7': 'm7', '-7': 'm7',
        'maj7': 'maj7', 'M7': 'maj7', 'Δ7': 'maj7', 'Δ': 'maj7',
        'dim': 'dim', 'dim7': 'dim', 'm7b5': 'm7b5', 'ø': 'm7b5',
        '9': '9', 'm9': 'm9', 'maj9': 'maj9',
        'sus4': 'sus4', '13': '13',
    }
    chord_type = type_map.get(chord_type, chord_type)

    root_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    root_pc = root_map.get(root, 0)

    if chord_type in CHORD_DEFINITIONS:
        intervals = CHORD_DEFINITIONS[chord_type]['intervals']
    else:
        intervals = {0, 4, 7}

    return {(root_pc + i) % 12 for i in intervals}, root_pc


# Build role-aware index
print("\nClassifying patterns by musical role...")
role_index = defaultdict(list)

for name, p in patterns.items():
    pitches = p.get('canonical_pitches', [])
    if len(pitches) < 2:
        continue

    role = classify_pattern(p)
    pcs = frozenset(pitch % 12 for pitch in pitches)
    gm = p.get('gm_program', 0)

    # Store with role
    role_index[(role, pcs)].append({
        'name': name,
        'pattern': p,
        'gm_program': gm,
        'role': role,
        'n_notes': len(pitches),
        'pitch_range': (min(pitches), max(pitches))
    })

# Count by role
role_counts = defaultdict(int)
for (role, pcs), plist in role_index.items():
    role_counts[role] += len(plist)
print(f"  chord voicings: {role_counts['chord']}")
print(f"  walking lines: {role_counts['walking']}")
print(f"  comping: {role_counts['comping']}")
print(f"  melodic: {role_counts['melodic']}")


def find_patterns_by_role(chord_name, role, role_index, register=None):
    """Find patterns matching chord AND musical role."""
    chord_pcs, root_pc = get_chord_pitch_classes(chord_name)
    if not chord_pcs:
        return []

    compatible = []
    for (pat_role, pat_pcs), pattern_list in role_index.items():
        if pat_role != role:
            continue
        if not pat_pcs.issubset(chord_pcs):
            continue

        for p in pattern_list:
            # Filter by register if specified
            if register:
                low, high = p['pitch_range']
                if register == 'bass' and low > 55:
                    continue
                elif register == 'treble' and high < 55:
                    continue
            compatible.append(p)

    return compatible


def decode_pattern_with_role(pattern_data, time_offset, role):
    """Decode pattern with role-appropriate timing."""
    p = pattern_data['pattern']
    pitches = p.get('canonical_pitches', [])
    rhythm = p.get('rhythm_ratios', [])
    durations = p.get('duration_ratios', [])
    velocities = p.get('velocity_ratios', [])
    gm = pattern_data['gm_program']

    notes = []
    current_time = time_offset

    for i, pitch in enumerate(pitches):
        # Keep pitch as-is (patterns are already in correct register)
        actual_pitch = max(21, min(108, pitch))

        # Role-specific timing
        if role == 'chord':
            # Chords: use rhythm=0 for simultaneous, hold longer
            r = rhythm[i] if i < len(rhythm) else 0
            if abs(r) < 0.1:  # Simultaneous
                time_advance = 0
            else:
                time_advance = 0.5
            dur = 1.5  # Hold chord voicings
        elif role == 'walking':
            # Walking bass: even quarter notes
            time_advance = 0.5  # Quarter note at 120bpm
            dur = 0.45  # Slightly detached
        else:
            # Melodic: use actual rhythm
            r = rhythm[i] if i < len(rhythm) else 0.5
            time_advance = max(0.1, r * 0.5)
            dur = durations[i] * 0.4 if i < len(durations) else 0.3
            dur = max(0.1, min(1.0, dur))

        vel = velocities[i] * 80 if i < len(velocities) else 80
        vel = int(max(50, min(110, vel)))

        # Adjust velocity by role
        if role == 'chord':
            vel = int(vel * 0.8)  # Comping softer
        elif role == 'walking':
            vel = int(vel * 0.9)  # Bass steady

        notes.append({
            'pitch': actual_pitch,
            'time': current_time,
            'duration': dur,
            'velocity': vel,
            'gm_program': gm
        })

        current_time += time_advance

    return notes, current_time


def generate_bar(chord_name, bar_start, bar_duration, role_index):
    """Generate one bar with proper role assignment."""
    events = []

    # 1. BASS - walking pattern
    bass_patterns = find_patterns_by_role(chord_name, 'walking', role_index, register='bass')
    if not bass_patterns:
        # Fallback to any bass-register pattern
        bass_patterns = find_patterns_by_role(chord_name, 'comping', role_index, register='bass')

    if bass_patterns:
        # Pick pattern that fits bar duration
        candidates = [p for p in bass_patterns if p['n_notes'] <= bar_duration * 2 + 1]
        if candidates:
            bp = random.choice(candidates[:10])  # From top matches
            notes, _ = decode_pattern_with_role(bp, bar_start, 'walking')
            # Force bass to GM32/33 and low register
            for n in notes:
                n['gm_program'] = 32
                if n['pitch'] > 55:
                    n['pitch'] -= 12
                if n['pitch'] > 55:
                    n['pitch'] -= 12
            events.extend(notes)

    # 2. PIANO - chord voicing
    chord_patterns = find_patterns_by_role(chord_name, 'chord', role_index)
    if chord_patterns:
        # Pick a voicing
        cp = random.choice(chord_patterns[:10])
        notes, _ = decode_pattern_with_role(cp, bar_start + 0.05, 'chord')  # Slight offset
        for n in notes:
            n['gm_program'] = 0  # Piano
        events.extend(notes)

        # Maybe add second voicing later in bar
        if bar_duration >= 2 and random.random() > 0.5:
            cp2 = random.choice(chord_patterns[:10])
            notes2, _ = decode_pattern_with_role(cp2, bar_start + bar_duration/2, 'chord')
            for n in notes2:
                n['gm_program'] = 0
            events.extend(notes2)

    # 3. HORNS - melodic lines (select 1-2 horns)
    melodic_patterns = find_patterns_by_role(chord_name, 'melodic', role_index, register='treble')
    if melodic_patterns:
        # Alto sax melody
        horn_gms = [65, 66, 56, 57]  # Alto, Tenor, Trumpet, Trombone
        selected_horns = random.sample(horn_gms, min(2, len(horn_gms)))

        for gm in selected_horns:
            mp = random.choice(melodic_patterns[:15])
            start_offset = random.uniform(0, 0.3)
            notes, _ = decode_pattern_with_role(mp, bar_start + start_offset, 'melodic')
            for n in notes:
                n['gm_program'] = gm
            events.extend(notes)

    return events


# Generate from chord progression
progression = [
    'Dm7', 'G7', 'Cmaj7', 'Am7',
    'Dm7', 'G7', 'Cmaj7', 'Cmaj7',
    'Fmaj7', 'Fm7', 'Em7', 'A7',
    'Dm7', 'G7', 'Cmaj7', 'Am7'
]

beats_per_chord = 2
print(f"\nGenerating 16-bar progression...")

all_events = []
for i, chord in enumerate(progression):
    bar_start = i * beats_per_chord
    bar_events = generate_bar(chord, bar_start, beats_per_chord, role_index)
    all_events.extend(bar_events)
    print(f"  Bar {i+1}: {chord} -> {len(bar_events)} notes")

print(f"\nTotal: {len(all_events)} notes")

# Create MIDI
by_program = defaultdict(list)
for e in all_events:
    by_program[e['gm_program']].append(e)

gm_names = {
    0: 'Piano', 32: 'Acoustic Bass', 33: 'Electric Bass',
    56: 'Trumpet', 57: 'Trombone', 65: 'Alto Sax', 66: 'Tenor Sax'
}

midi = MIDIFile(len(by_program), deinterleave=False)
tempo = 120

for track_idx, (gm, track_events) in enumerate(sorted(by_program.items())):
    channel = track_idx % 9
    midi.addTempo(track_idx, 0, tempo)
    midi.addProgramChange(track_idx, channel, 0, min(127, gm))

    for e in track_events:
        midi.addNote(track_idx, channel, e['pitch'], e['time'], e['duration'], e['velocity'])

    name = gm_names.get(gm, f'GM{gm}')
    print(f"  {name}: {len(track_events)} notes")

with open('generated_chordal_v2.mid', 'wb') as f:
    midi.writeFile(f)
print(f"\nSaved: generated_chordal_v2.mid")

# Evaluation
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

all_events.sort(key=lambda x: x['time'])
total_dur = all_events[-1]['time'] if all_events else 0

# Harmonic coherence
chord_times = [(i * beats_per_chord, progression[i]) for i in range(len(progression))]
correct = 0
total = 0
for e in all_events:
    chord_idx = min(int(e['time'] / beats_per_chord), len(progression) - 1)
    chord_pcs, _ = get_chord_pitch_classes(progression[chord_idx])
    if e['pitch'] % 12 in chord_pcs:
        correct += 1
    total += 1
print(f"Harmonic coherence: {correct}/{total} ({100*correct/total:.0f}%)")

# Role analysis
print("\nRole analysis:")
for gm, track_events in sorted(by_program.items()):
    name = gm_names.get(gm, f'GM{gm}')
    track_events.sort(key=lambda x: x['time'])

    pitches = [e['pitch'] for e in track_events]
    times = [e['time'] for e in track_events]

    # Check rhythm evenness
    if len(times) > 1:
        deltas = [times[i+1] - times[i] for i in range(len(times)-1)]
        avg_delta = sum(deltas) / len(deltas)
        rhythm_variance = sum((d - avg_delta)**2 for d in deltas) / len(deltas)
    else:
        rhythm_variance = 0

    # Check intervals
    if len(pitches) > 1:
        intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
        avg_interval = sum(intervals) / len(intervals)
    else:
        avg_interval = 0

    pitch_range = f"{min(pitches)}-{max(pitches)}"

    print(f"  {name}: range={pitch_range}, avg_interval={avg_interval:.1f}, rhythm_var={rhythm_variance:.2f}")

print("\n" + "=" * 60)
