#!/usr/bin/env python3
"""
CHORD-CONDITIONED GENERATION v3 - Relationship-Aware
=====================================================
Uses co-occurrence (cross-track) and transition (temporal) data
to select patterns that actually played together in real pieces.
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
print("CHORD-CONDITIONED GENERATION v3 - Relationship-Aware")
print("=" * 60)

# Load all data
print("Loading patterns and relationships...")
with open('patterns_with_occurrences.json') as f:
    patterns = json.load(f)
print(f"  Patterns: {len(patterns)}")

with open('pattern_cooccurrence.json') as f:
    cooccurrence = json.load(f)
print(f"  Co-occurrence pairs: {sum(len(v) for v in cooccurrence.values())//2}")

with open('pattern_transitions.json') as f:
    transitions = json.load(f)
print(f"  Transitions: {sum(len(v) for v in transitions.values())}")


def classify_pattern(p):
    """Classify pattern by musical role."""
    rhythm = p.get('rhythm_ratios', [])
    pitches = p.get('canonical_pitches', [])
    intervals = p.get('pitch_intervals', [])

    if len(pitches) < 2:
        return 'fragment'

    has_simultaneous = any(abs(r) < 0.1 for r in rhythm) if rhythm else False
    all_even = all(0.8 <= r <= 1.2 for r in rhythm) if rhythm and len(rhythm) > 0 else False
    avg_interval = sum(abs(i) for i in intervals) / len(intervals) if intervals else 0
    is_stepwise = avg_interval <= 3

    if has_simultaneous and len(pitches) >= 3:
        return 'chord'
    elif all_even and is_stepwise and len(pitches) >= 3:
        return 'walking'
    elif all_even and len(pitches) >= 2:
        return 'comping'
    else:
        return 'melodic'


def get_chord_pitch_classes(chord_name):
    """Parse chord name and return pitch class set."""
    chord_name = chord_name.strip()
    if not chord_name:
        return set(), 0

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


# Build role-aware index with relationships
print("\nBuilding role index...")
role_index = defaultdict(list)

for name, p in patterns.items():
    pitches = p.get('canonical_pitches', [])
    if len(pitches) < 2:
        continue

    role = classify_pattern(p)
    pcs = frozenset(pitch % 12 for pitch in pitches)
    gm = p.get('gm_program', 0)

    role_index[(role, pcs)].append({
        'name': name,
        'pattern': p,
        'gm_program': gm,
        'role': role,
        'n_notes': len(pitches),
        'pitch_range': (min(pitches), max(pitches)),
        'cooccurs_with': cooccurrence.get(name, {}),
        'transitions_to': transitions.get(name, {})
    })

role_counts = defaultdict(int)
for (role, pcs), plist in role_index.items():
    role_counts[role] += len(plist)
print(f"  chord: {role_counts['chord']}, walking: {role_counts['walking']}, melodic: {role_counts['melodic']}")


def find_compatible_patterns(chord_name, role, register=None):
    """Find patterns matching chord AND role."""
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
            if register:
                low, high = p['pitch_range']
                if register == 'bass' and low > 55:
                    continue
                elif register == 'treble' and high < 55:
                    continue
            compatible.append(p)

    return compatible


def select_with_cooccurrence(patterns, anchor_pattern=None, top_n=20):
    """Select pattern preferring those that co-occurred with anchor."""
    if not patterns:
        return None

    if anchor_pattern is None:
        return random.choice(patterns[:top_n])

    # Score by co-occurrence with anchor
    anchor_name = anchor_pattern['name']
    scored = []
    for p in patterns:
        cooc_count = p['cooccurs_with'].get(anchor_name, 0)
        scored.append((p, cooc_count))

    # Sort by co-occurrence count
    scored.sort(key=lambda x: -x[1])

    # Weighted random from top candidates
    top = scored[:top_n]
    if top[0][1] > 0:  # Has co-occurrences
        # Weight by co-occurrence count + 1
        weights = [count + 1 for _, count in top]
        total = sum(weights)
        r = random.random() * total
        cumsum = 0
        for p, count in top:
            cumsum += count + 1
            if r <= cumsum:
                return p

    return random.choice([p for p, _ in top])


def select_with_transition(patterns, prev_pattern=None, top_n=20):
    """Select pattern preferring those that followed prev_pattern."""
    if not patterns:
        return None

    if prev_pattern is None:
        return random.choice(patterns[:top_n])

    prev_name = prev_pattern['name']
    prev_trans = transitions.get(prev_name, {})

    scored = []
    for p in patterns:
        trans_count = prev_trans.get(p['name'], 0)
        scored.append((p, trans_count))

    scored.sort(key=lambda x: -x[1])

    top = scored[:top_n]
    if top[0][1] > 0:
        weights = [count + 1 for _, count in top]
        total = sum(weights)
        r = random.random() * total
        cumsum = 0
        for p, count in top:
            cumsum += count + 1
            if r <= cumsum:
                return p

    return random.choice([p for p, _ in top])


def decode_pattern(pattern_data, time_offset, role):
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
        actual_pitch = max(21, min(108, pitch))

        if role == 'chord':
            r = rhythm[i] if i < len(rhythm) else 0
            time_advance = 0 if abs(r) < 0.1 else 0.5
            dur = 1.5
        elif role == 'walking':
            time_advance = 0.5
            dur = 0.45
        else:
            r = rhythm[i] if i < len(rhythm) else 0.5
            time_advance = max(0.1, r * 0.5)
            dur = durations[i] * 0.4 if i < len(durations) else 0.3
            dur = max(0.1, min(1.0, dur))

        vel = velocities[i] * 80 if i < len(velocities) else 80
        vel = int(max(50, min(110, vel)))

        if role == 'chord':
            vel = int(vel * 0.8)
        elif role == 'walking':
            vel = int(vel * 0.9)

        notes.append({
            'pitch': actual_pitch,
            'time': current_time,
            'duration': dur,
            'velocity': vel,
            'gm_program': gm
        })

        current_time += time_advance

    return notes, current_time


# Track state for relationship-aware generation
class TrackState:
    def __init__(self, role, gm_program, register=None):
        self.role = role
        self.gm_program = gm_program
        self.register = register
        self.prev_pattern = None
        self.notes = []


def generate_with_relationships(progression, beats_per_chord=2):
    """Generate using co-occurrence and transition relationships."""

    # Initialize track states
    tracks = {
        'bass': TrackState('walking', 32, 'bass'),
        'piano': TrackState('chord', 0, None),
        'alto_sax': TrackState('melodic', 65, 'treble'),
        'tenor_sax': TrackState('melodic', 66, 'treble'),
    }

    # Store bass patterns per bar for proper anchoring
    bass_per_bar = {}

    # First pass: select bass patterns as anchors (foundational)
    for i, chord in enumerate(progression):
        bar_start = i * beats_per_chord

        bass_patterns = find_compatible_patterns(chord, 'walking', 'bass')
        if not bass_patterns:
            bass_patterns = find_compatible_patterns(chord, 'comping', 'bass')

        if bass_patterns:
            bp = select_with_transition(bass_patterns, tracks['bass'].prev_pattern)
            if bp:
                notes, _ = decode_pattern(bp, bar_start, 'walking')
                for n in notes:
                    n['gm_program'] = 32
                    if n['pitch'] > 55:
                        n['pitch'] -= 12
                    if n['pitch'] > 55:
                        n['pitch'] -= 12
                tracks['bass'].notes.extend(notes)
                tracks['bass'].prev_pattern = bp
                bass_per_bar[i] = bp

    # Second pass: select other parts using co-occurrence with THIS bar's bass
    for i, chord in enumerate(progression):
        bar_start = i * beats_per_chord
        bass_anchor = bass_per_bar.get(i)

        # Piano chords - co-occur with this bar's bass
        chord_patterns = find_compatible_patterns(chord, 'chord')
        if chord_patterns:
            cp = select_with_cooccurrence(chord_patterns, bass_anchor)
            if cp:
                # Also prefer transition from previous piano pattern
                if tracks['piano'].prev_pattern:
                    trans = tracks['piano'].prev_pattern.get('transitions_to', {})
                    if cp['name'] not in trans and random.random() > 0.5:
                        cp2 = select_with_transition(chord_patterns, tracks['piano'].prev_pattern)
                        if cp2 and cp2['name'] in (bass_anchor['cooccurs_with'] if bass_anchor else {}):
                            cp = cp2

                notes, _ = decode_pattern(cp, bar_start + 0.05, 'chord')
                for n in notes:
                    n['gm_program'] = 0
                tracks['piano'].notes.extend(notes)
                tracks['piano'].prev_pattern = cp

        # Horns - MUST co-occur with bass, prefer transition from previous
        for horn_name in ['alto_sax', 'tenor_sax']:
            track = tracks[horn_name]
            mel_patterns = find_compatible_patterns(chord, 'melodic', 'treble')

            if mel_patterns and bass_anchor:
                # Filter to only patterns that co-occurred with bass
                bass_coocs = set(bass_anchor['cooccurs_with'].keys())
                cooccurred = [p for p in mel_patterns if p['name'] in bass_coocs]

                if cooccurred:
                    # From co-occurred patterns, prefer good transitions
                    mp = select_with_transition(cooccurred, track.prev_pattern)
                else:
                    # Fallback to co-occurrence scoring
                    mp = select_with_cooccurrence(mel_patterns, bass_anchor)

                if mp:
                    offset = random.uniform(0, 0.15)
                    notes, _ = decode_pattern(mp, bar_start + offset, 'melodic')
                    for n in notes:
                        n['gm_program'] = track.gm_program
                    track.notes.extend(notes)
                    track.prev_pattern = mp

    # Collect all events
    all_events = []
    for track in tracks.values():
        all_events.extend(track.notes)

    return all_events, tracks


# Generate
progression = [
    'Dm7', 'G7', 'Cmaj7', 'Am7',
    'Dm7', 'G7', 'Cmaj7', 'Cmaj7',
    'Fmaj7', 'Fm7', 'Em7', 'A7',
    'Dm7', 'G7', 'Cmaj7', 'Am7'
]

beats_per_chord = 2
print(f"\nGenerating {len(progression)}-bar progression with relationships...")

all_events, track_states = generate_with_relationships(progression, beats_per_chord)
print(f"Total: {len(all_events)} notes")

# Detailed relationship stats
print("\nRelationship utilization per bar:")
for i in range(min(4, len(progression))):
    chord = progression[i]
    print(f"  Bar {i+1} ({chord}):")

# Create MIDI
by_program = defaultdict(list)
for e in all_events:
    by_program[e['gm_program']].append(e)

gm_names = {
    0: 'Piano', 32: 'Acoustic Bass',
    65: 'Alto Sax', 66: 'Tenor Sax'
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

with open('generated_chordal_v3.mid', 'wb') as f:
    midi.writeFile(f)
print(f"\nSaved: generated_chordal_v3.mid")

# Evaluation
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

all_events.sort(key=lambda x: x['time'])

# Harmonic coherence
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

    if len(times) > 1:
        deltas = [times[i+1] - times[i] for i in range(len(times)-1)]
        avg_delta = sum(deltas) / len(deltas)
        rhythm_variance = sum((d - avg_delta)**2 for d in deltas) / len(deltas)
    else:
        rhythm_variance = 0

    if len(pitches) > 1:
        intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
        avg_interval = sum(intervals) / len(intervals)
    else:
        avg_interval = 0

    pitch_range = f"{min(pitches)}-{max(pitches)}"
    print(f"  {name}: range={pitch_range}, avg_interval={avg_interval:.1f}, rhythm_var={rhythm_variance:.2f}")

print("\n" + "=" * 60)
