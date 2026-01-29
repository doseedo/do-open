#!/usr/bin/env python3
"""
CHORD-CONDITIONED GENERATION v4 - Phrase-Level
===============================================
Extracts and uses multi-bar PHRASE CHAINS from real pieces,
not individual patterns. Preserves continuity because phrases
are actual consecutive patterns from real music.

Unit of generation: phrase (4-8 bar sequence from a real piece)
"""
import json
import random
import sys
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from midiutil import MIDIFile

os.chdir('/home/arlo/do-repo/midi_generator/1_approaches/transform_based')
sys.path.insert(0, '/home/arlo/do-repo/midi_generator/allharmony')

from midianal import CHORD_DEFINITIONS

print("=" * 60)
print("CHORD-CONDITIONED GENERATION v4 - Phrase-Level")
print("=" * 60)

# Load patterns with occurrences
with open('patterns_with_occurrences.json') as f:
    patterns = json.load(f)
print(f"Loaded {len(patterns)} patterns")


@dataclass
class PhraseChain:
    """A sequence of consecutive patterns from a real piece."""
    patterns: List[str]  # Pattern names in order
    piece_id: str
    track_id: int
    gm_program: int
    start_position: int
    end_position: int
    onset_times: List[int]  # Original onset times
    pitch_classes: set  # All pitch classes in phrase

    @property
    def length(self) -> int:
        return len(self.patterns)

    @property
    def duration_positions(self) -> int:
        return self.end_position - self.start_position


def extract_phrase_chains(patterns: Dict, min_length: int = 3, max_length: int = 12, max_gap: int = 2) -> List[PhraseChain]:
    """
    Extract phrase chains - consecutive patterns from same track in same piece.

    IMPORTANT: Limit to max_length to get actual phrases, not entire tracks.

    A phrase chain is patterns where:
    - Same piece_id
    - Same track_id
    - Positions are consecutive (or within max_gap)
    - Length between min_length and max_length
    """
    print("\nExtracting phrase chains...")

    # Build index: (piece_id, track_id) -> [(position, pattern_name, occurrence)]
    track_patterns = defaultdict(list)

    for name, p in patterns.items():
        for occ in p.get('occurrences', []):
            key = (occ['piece_id'], occ['track_id'])
            track_patterns[key].append({
                'position': occ['position'],
                'last_position': occ.get('last_position', occ['position'] + 1),
                'pattern': name,
                'onset_time': occ['onset_time'],
                'gm_program': occ['gm_program'],
                'pitches': p.get('canonical_pitches', [])
            })

    print(f"  Found {len(track_patterns)} unique (piece, track) combinations")

    # Sort each track's patterns by position
    for key in track_patterns:
        track_patterns[key].sort(key=lambda x: x['position'])

    # Extract non-overlapping phrase chunks (fast approach)
    chains = []

    for (piece_id, track_id), pats in track_patterns.items():
        if len(pats) < min_length:
            continue

        # Extract non-overlapping chunks of target size
        target_length = (min_length + max_length) // 2  # Middle of range

        i = 0
        while i < len(pats) - min_length + 1:
            # Try to extract a chunk of target_length
            chunk_end = min(i + target_length, len(pats))
            chain_patterns = pats[i:chunk_end]

            # Verify consecutiveness and find actual end
            actual_end = i + 1
            for k in range(len(chain_patterns) - 1):
                prev_end = chain_patterns[k]['last_position']
                next_start = chain_patterns[k + 1]['position']
                if next_start - prev_end > max_gap:
                    break
                actual_end = i + k + 2

            chain_patterns = pats[i:actual_end]

            if len(chain_patterns) >= min_length:
                all_pitches = set()
                for cp in chain_patterns:
                    all_pitches.update(pitch % 12 for pitch in cp['pitches'])

                chain = PhraseChain(
                    patterns=[cp['pattern'] for cp in chain_patterns],
                    piece_id=piece_id,
                    track_id=track_id,
                    gm_program=chain_patterns[0]['gm_program'],
                    start_position=chain_patterns[0]['position'],
                    end_position=chain_patterns[-1]['last_position'],
                    onset_times=[cp['onset_time'] for cp in chain_patterns],
                    pitch_classes=all_pitches
                )
                chains.append(chain)

            i = actual_end  # Move past this chunk

    print(f"  Extracted {len(chains)} phrase chains")

    # Stats
    lengths = [c.length for c in chains]
    print(f"  Chain lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

    return chains


def classify_chain_role(chain: PhraseChain, patterns: Dict) -> str:
    """Classify phrase chain by musical role."""
    gm = chain.gm_program

    # Check rhythm characteristics of patterns in chain
    all_even = True
    has_simultaneous = False
    total_intervals = []

    for pname in chain.patterns:
        p = patterns.get(pname, {})
        rhythm = p.get('rhythm_ratios', [])
        intervals = p.get('pitch_intervals', [])

        if rhythm:
            if any(abs(r) < 0.1 for r in rhythm):
                has_simultaneous = True
            if not all(0.8 <= r <= 1.2 for r in rhythm):
                all_even = False

        total_intervals.extend(intervals)

    avg_interval = sum(abs(i) for i in total_intervals) / len(total_intervals) if total_intervals else 0
    is_stepwise = avg_interval <= 3

    # Classify
    if gm in [32, 33, 34, 35, 36, 37, 38, 39]:  # Bass instruments
        return 'bass'
    elif gm == 0 or gm in range(1, 8):  # Piano/keys
        if has_simultaneous:
            return 'chord'
        return 'piano'
    elif gm in [65, 66, 67, 56, 57, 60, 72, 73]:  # Horns/winds
        return 'melodic'
    elif has_simultaneous:
        return 'chord'
    elif all_even and is_stepwise:
        return 'walking'
    else:
        return 'melodic'


def get_chord_pitch_classes(chord_name: str) -> Tuple[set, int]:
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
        'maj7': 'maj7', 'M7': 'maj7',
        'dim': 'dim', 'dim7': 'dim', 'm7b5': 'm7b5',
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


def find_compatible_chains(chains: List[PhraseChain], chord_progression: List[str],
                          role: str, min_length: int = 2,
                          max_candidates: int = 500) -> List[Tuple[PhraseChain, int, float]]:
    """Find phrase chains compatible with a chord progression segment.

    Returns: List of (chain, best_transpose, score) tuples.
    Expects chains to already be role-filtered.
    Optimized: samples up to max_candidates for efficiency.
    """

    # Get pitch classes for each chord in progression
    all_chord_pcs = set()
    first_chord_pcs, root = get_chord_pitch_classes(chord_progression[0])
    all_chord_pcs.update(first_chord_pcs)
    for chord in chord_progression[1:]:
        pcs, _ = get_chord_pitch_classes(chord)
        all_chord_pcs.update(pcs)

    # Scale tones (major scale based on first chord root)
    scale_intervals = [0, 2, 4, 5, 7, 9, 11]
    scale_pcs = {(root + i) % 12 for i in scale_intervals}

    # Filter by min_length
    length_filtered = [c for c in chains if c.length >= min_length]

    # Sample if too many candidates
    if len(length_filtered) > max_candidates:
        length_filtered = random.sample(length_filtered, max_candidates)

    compatible = []
    for chain in length_filtered:
        if not chain.pitch_classes:
            compatible.append((chain, 0, 0.5))
            continue

        # Find best transposition and score
        best_t = 0
        best_score = 0

        for t in range(12):
            transposed_pcs = {(pc + t) % 12 for pc in chain.pitch_classes}

            # Score: chord tones + scale tones (weighted)
            chord_overlap = len(transposed_pcs & all_chord_pcs)
            scale_overlap = len(transposed_pcs & scale_pcs)
            total = len(chain.pitch_classes)

            # Weighted score: chord tones worth more
            score = (chord_overlap * 2 + scale_overlap) / (total * 2)

            if score > best_score:
                best_score = score
                best_t = t

        # Role-based threshold (after optimal transposition)
        if role == 'bass':
            threshold = 0.7
        elif role == 'melodic':
            threshold = 0.55
        else:
            threshold = 0.6

        if best_score >= threshold:
            compatible.append((chain, best_t, best_score))

    # Sort by score (higher first)
    compatible.sort(key=lambda x: -x[2])

    return compatible


def decode_phrase_chain(chain: PhraseChain, patterns: Dict, start_time: float,
                       time_scale: float = 1.0, transpose: int = 0) -> List[Dict]:
    """Decode a phrase chain into note events, with optional transposition."""
    notes = []

    # Calculate time offset between patterns based on original onset times
    if len(chain.onset_times) > 1:
        original_deltas = [chain.onset_times[i+1] - chain.onset_times[i]
                         for i in range(len(chain.onset_times)-1)]
        # Normalize to beats (assume 480 ticks per beat)
        pattern_deltas = [d / 480.0 * time_scale for d in original_deltas]
    else:
        pattern_deltas = []

    current_time = start_time

    for i, pname in enumerate(chain.patterns):
        p = patterns.get(pname, {})
        pitches = p.get('canonical_pitches', [])
        rhythm = p.get('rhythm_ratios', [])
        durations = p.get('duration_ratios', [])
        velocities = p.get('velocity_ratios', [])

        pattern_time = current_time

        for j, pitch in enumerate(pitches):
            # Apply transposition
            actual_pitch = max(21, min(108, pitch + transpose))

            # Timing within pattern
            if j > 0 and j - 1 < len(rhythm):
                r = rhythm[j - 1]
                pattern_time += max(0.1, r * 0.5) * time_scale

            dur = durations[j] * 0.4 if j < len(durations) else 0.3
            dur = max(0.1, min(1.0, dur)) * time_scale

            vel = velocities[j] * 80 if j < len(velocities) else 80
            vel = int(max(50, min(110, vel)))

            notes.append({
                'pitch': actual_pitch,
                'time': pattern_time,
                'duration': dur,
                'velocity': vel,
                'gm_program': chain.gm_program
            })

        # Move to next pattern position
        if i < len(pattern_deltas):
            current_time += pattern_deltas[i]
        else:
            current_time += 1.0 * time_scale  # Default 1 beat between patterns

    return notes


def find_best_transposition(chain: PhraseChain, target_chord_pcs: set, root: int = 0) -> int:
    """Find transposition that maximizes chord tone overlap.

    Prefers transpositions that:
    1. Maximize overlap with chord tones
    2. Place the chain's "root" (lowest pitch class) on a chord tone
    """
    if not chain.pitch_classes or not target_chord_pcs:
        return 0

    best_transpose = 0
    best_score = -1

    # Find the lowest pitch class in the chain (likely the "root")
    chain_root = min(chain.pitch_classes) if chain.pitch_classes else 0

    # Try all 12 transpositions
    for t in range(12):
        transposed_pcs = {(pc + t) % 12 for pc in chain.pitch_classes}

        # Primary score: chord tone overlap
        overlap = len(transposed_pcs & target_chord_pcs)

        # Bonus if chain root lands on a chord tone
        transposed_root = (chain_root + t) % 12
        root_bonus = 2 if transposed_root in target_chord_pcs else 0

        # Bonus if it lands on the target root
        root_match_bonus = 3 if transposed_root == root else 0

        score = overlap * 10 + root_bonus + root_match_bonus

        if score > best_score:
            best_score = score
            best_transpose = t

    return best_transpose


# Extract phrase chains (limit to 4-8 patterns = ~2-4 bars)
chains = extract_phrase_chains(patterns, min_length=4, max_length=8, max_gap=2)

# Index by role with caching
print("\nIndexing chains by role...")
chains_by_role = defaultdict(list)
chain_role_cache = {}  # Cache role for each chain
for chain in chains:
    role = classify_chain_role(chain, patterns)
    chains_by_role[role].append(chain)
    chain_role_cache[id(chain)] = role

for role, role_chains in chains_by_role.items():
    print(f"  {role}: {len(role_chains)} chains")


def generate_phrase_based(progression: List[str], beats_per_chord: float = 2.0) -> List[Dict]:
    """Generate using phrase chains instead of individual patterns."""
    all_events = []
    total_duration = len(progression) * beats_per_chord

    # Determine phrase length in chords (how many chords per phrase)
    phrase_length = 4  # 4 chords = 1 phrase

    print(f"\nGenerating {len(progression)} chords, phrase length = {phrase_length}")

    # Track recently used pieces to avoid repetition
    recently_used = {
        'bass': set(),
        'piano': set(),
        'melodic': set(),
    }

    # Generate in phrase-sized chunks
    for phrase_idx in range(0, len(progression), phrase_length):
        phrase_chords = progression[phrase_idx:phrase_idx + phrase_length]
        phrase_start = phrase_idx * beats_per_chord
        phrase_duration = len(phrase_chords) * beats_per_chord

        print(f"\n  Phrase {phrase_idx // phrase_length + 1}: {' | '.join(phrase_chords)}")

        # For each role, find a phrase chain that fits this chord progression
        for role in ['bass', 'piano', 'melodic']:
            # Find compatible chains
            compatible = find_compatible_chains(
                chains_by_role.get(role, []) + chains_by_role.get('chord' if role == 'piano' else role, []),
                phrase_chords,
                role if role != 'piano' else None,
                min_length=2
            )

            if not compatible:
                print(f"    {role}: no compatible chains found")
                continue

            # Filter out recently used pieces for variety
            fresh_compatible = [(c, t, s) for c, t, s in compatible if c.piece_id not in recently_used[role]]
            if not fresh_compatible:
                # Reset if we've exhausted options
                recently_used[role].clear()
                fresh_compatible = compatible

            compatible = fresh_compatible

            # Weighted random selection - higher score = better, plus variety
            weights = []
            for c, t, s in compatible:
                # Score bonus + length bonus + random factor
                weight = s * 10 + c.length * 2 + random.uniform(0, 3)
                weights.append(weight)

            # Normalize and select
            total_weight = sum(weights)
            if total_weight > 0:
                probs = [w / total_weight for w in weights]
                chain, transpose, score = random.choices(compatible, weights=probs, k=1)[0]
            else:
                chain, transpose, score = random.choice(compatible)

            # Time scale to fit phrase duration
            chain_duration_estimate = chain.length * 0.5  # Rough estimate
            time_scale = phrase_duration / max(chain_duration_estimate, 1)
            time_scale = max(0.5, min(2.0, time_scale))  # Clamp

            # For bass, try harder to land on roots
            first_chord_pcs, first_root = get_chord_pitch_classes(phrase_chords[0])
            if role == 'bass' and chain.pitch_classes:
                # Find transposition that puts lowest pitch class on root
                chain_bass = min(chain.pitch_classes)
                root_transpose = (first_root - chain_bass) % 12
                # Use root transpose if it still has decent chord tone overlap
                all_chord_pcs = set(first_chord_pcs)
                for chord in phrase_chords[1:]:
                    pcs, _ = get_chord_pitch_classes(chord)
                    all_chord_pcs.update(pcs)
                transposed_pcs = {(pc + root_transpose) % 12 for pc in chain.pitch_classes}
                if len(transposed_pcs & all_chord_pcs) >= len(chain.pitch_classes) * 0.4:
                    transpose = root_transpose

            # Decode chain with transposition
            notes = decode_phrase_chain(chain, patterns, phrase_start, time_scale, transpose)

            # Adjust instruments
            for n in notes:
                if role == 'bass':
                    n['gm_program'] = 32
                    # Ensure bass register
                    while n['pitch'] > 55:
                        n['pitch'] -= 12
                elif role == 'piano':
                    n['gm_program'] = 0
                elif role == 'melodic':
                    # Assign to sax/brass
                    n['gm_program'] = random.choice([65, 66, 56, 57])

            all_events.extend(notes)
            recently_used[role].add(chain.piece_id)
            t_str = f", T+{transpose}" if transpose else ""
            print(f"    {role}: {chain.piece_id[:30]}... ({chain.length}p, {len(notes)}n{t_str}, s={score:.2f})")

    return all_events


# Generate
progression = [
    'Dm7', 'G7', 'Cmaj7', 'Am7',
    'Dm7', 'G7', 'Cmaj7', 'Cmaj7',
    'Fmaj7', 'Fm7', 'Em7', 'A7',
    'Dm7', 'G7', 'Cmaj7', 'Am7'
]

beats_per_chord = 2
all_events = generate_phrase_based(progression, beats_per_chord)

print(f"\nTotal: {len(all_events)} notes")

# Create MIDI
by_program = defaultdict(list)
for e in all_events:
    by_program[e['gm_program']].append(e)

gm_names = {
    0: 'Piano', 32: 'Acoustic Bass',
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

with open('generated_chordal_v4.mid', 'wb') as f:
    midi.writeFile(f)
print(f"\nSaved: generated_chordal_v4.mid")

# Evaluation
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

all_events.sort(key=lambda x: x['time'])

# REAL validation metrics
print("\n--- HARMONIC ANALYSIS ---")

# 1. Strong beat chord tones (beat 1 and 3 should be chord tones)
strong_beat_chord = 0
strong_beat_total = 0
weak_beat_chord = 0
weak_beat_total = 0

for e in all_events:
    chord_idx = min(int(e['time'] / beats_per_chord), len(progression) - 1)
    chord_pcs, root = get_chord_pitch_classes(progression[chord_idx])
    pitch_pc = e['pitch'] % 12

    # Check if on strong beat (0, 2 within a 4-beat bar)
    beat_in_bar = e['time'] % 4
    is_strong = beat_in_bar < 0.25 or (1.75 < beat_in_bar < 2.25)

    if is_strong:
        strong_beat_total += 1
        if pitch_pc in chord_pcs:
            strong_beat_chord += 1
    else:
        weak_beat_total += 1
        if pitch_pc in chord_pcs:
            weak_beat_chord += 1

if strong_beat_total > 0:
    print(f"Strong beat chord tones: {strong_beat_chord}/{strong_beat_total} ({100*strong_beat_chord/strong_beat_total:.0f}%)")
if weak_beat_total > 0:
    print(f"Weak beat chord tones: {weak_beat_chord}/{weak_beat_total} ({100*weak_beat_chord/weak_beat_total:.0f}%)")

# 2. Bass on root on beat 1
print("\n--- BASS ANALYSIS ---")
bass_events = [e for e in all_events if e['gm_program'] == 32]
bass_events.sort(key=lambda x: x['time'])

root_on_downbeat = 0
downbeat_count = 0
for e in bass_events:
    beat_in_bar = e['time'] % 4
    if beat_in_bar < 0.25:  # On beat 1
        downbeat_count += 1
        chord_idx = min(int(e['time'] / beats_per_chord), len(progression) - 1)
        _, root = get_chord_pitch_classes(progression[chord_idx])
        if e['pitch'] % 12 == root:
            root_on_downbeat += 1

if downbeat_count > 0:
    print(f"Bass root on downbeat: {root_on_downbeat}/{downbeat_count} ({100*root_on_downbeat/downbeat_count:.0f}%)")
else:
    print("No bass downbeats found")

# 3. Dissonance check - are there clashing notes?
print("\n--- DISSONANCE CHECK ---")
# Group notes by time window
from collections import defaultdict
time_buckets = defaultdict(list)
for e in all_events:
    bucket = int(e['time'] * 4)  # 16th note buckets
    time_buckets[bucket].append(e['pitch'] % 12)

clash_count = 0
total_buckets = 0
for bucket, pcs in time_buckets.items():
    if len(pcs) > 1:
        total_buckets += 1
        # Check for minor 2nds (1 semitone apart)
        pcs_set = set(pcs)
        for pc in pcs_set:
            if (pc + 1) % 12 in pcs_set:
                clash_count += 1
                break

if total_buckets > 0:
    print(f"Buckets with minor 2nd clashes: {clash_count}/{total_buckets} ({100*clash_count/total_buckets:.0f}%)")

# Phrase continuity - measure how many consecutive notes are from same source
print("\nPhrase continuity:")
for gm, track_events in sorted(by_program.items()):
    name = gm_names.get(gm, f'GM{gm}')
    track_events.sort(key=lambda x: x['time'])

    if len(track_events) < 2:
        continue

    pitches = [e['pitch'] for e in track_events]
    times = [e['time'] for e in track_events]

    # Interval smoothness
    intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
    avg_interval = sum(intervals) / len(intervals)
    big_jumps = sum(1 for i in intervals if i > 12)

    print(f"  {name}: avg_interval={avg_interval:.1f}, big_jumps={big_jumps}/{len(intervals)}")

print("\n" + "=" * 60)
