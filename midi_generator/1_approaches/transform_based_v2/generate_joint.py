#!/usr/bin/env python3
"""
Joint Type + Root Generator.

Learns transitions as (current_type, next_type, root_movement) triples.
No music theory labels - purely pattern-based from observed data.

The key insight: type transitions and root movements are CORRELATED.
When you see type A → type B, certain root movements are more likely.
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import numpy as np
import random

sys.path.insert(0, str(Path(__file__).parent))

import mido
from mido import Message, MidiFile, MidiTrack
from grammar.sequitur import SequiturGrammar


def load_midi_chords(midi_path: str, collapse_repeats: bool = True) -> List[frozenset]:
    try:
        midi = mido.MidiFile(midi_path)
    except:
        return []

    events = []
    for track in midi.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                events.append((abs_time, msg.note))

    if not events:
        return []

    events.sort(key=lambda x: x[0])
    tolerance = 10
    chords = []
    current_time = events[0][0]
    current_notes = set()

    for time, note in events:
        if time - current_time <= tolerance:
            current_notes.add(note % 12)
        else:
            if current_notes:
                chord = frozenset(current_notes)
                # Collapse repeated chords - only add if different from last
                if collapse_repeats:
                    if not chords or chord != chords[-1]:
                        chords.append(chord)
                else:
                    chords.append(chord)
            current_time = time
            current_notes = {note % 12}

    if current_notes:
        chord = frozenset(current_notes)
        if collapse_repeats:
            if not chords or chord != chords[-1]:
                chords.append(chord)
        else:
            chords.append(chord)

    return chords


def get_interval_pattern(chord: frozenset) -> Tuple[int, ...]:
    """Interval pattern from lowest note."""
    if not chord:
        return ()
    pcs = sorted(chord)
    root = pcs[0]
    return tuple((pc - root) % 12 for pc in pcs)


def get_root(chord: frozenset) -> int:
    return min(chord) if chord else 0


def realize_chord(pattern: Tuple[int, ...], root: int) -> frozenset:
    return frozenset((root + interval) % 12 for interval in pattern)


ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def chord_str(chord: frozenset) -> str:
    if not chord:
        return "rest"
    return "{" + ",".join(ROOT_NAMES[pc] for pc in sorted(chord)) + "}"


class JointGenerator:
    """
    Learns and generates using joint (type, root_movement) transitions.
    Purely pattern-based - no music theory assumptions.
    """

    def __init__(self):
        # Pattern vocabulary
        self.pattern_to_id: Dict[Tuple, int] = {}
        self.id_to_pattern: Dict[int, Tuple] = {}

        # Joint transitions: current_type → Counter of (next_type, root_movement)
        self.joint_transitions: Dict[int, Counter] = defaultdict(Counter)

        # Starting states: (type, root) pairs
        self.start_states: Counter = Counter()

        # Grammar on full (type, root_movement) sequences
        self.grammar: Optional[SequiturGrammar] = None

    def _get_pattern_id(self, pattern: Tuple) -> int:
        if pattern not in self.pattern_to_id:
            pid = len(self.pattern_to_id)
            self.pattern_to_id[pattern] = pid
            self.id_to_pattern[pid] = pattern
        return self.pattern_to_id[pattern]

    def learn(self, midi_dir: str, max_files: int = 500):
        print("=" * 60)
        print("LEARNING JOINT TYPE + ROOT TRANSITIONS")
        print("=" * 60)

        midi_files = list(Path(midi_dir).rglob("*.mid"))[:max_files]
        print(f"Loading {len(midi_files)} files...")

        # Collect all transitions as (type, root) sequences
        all_factored = []  # List of [(pattern_id, root), ...]

        for midi_path in midi_files:
            chords = load_midi_chords(str(midi_path))
            if len(chords) >= 2:
                factored = []
                for chord in chords:
                    pattern = get_interval_pattern(chord)
                    root = get_root(chord)
                    pid = self._get_pattern_id(pattern)
                    factored.append((pid, root))
                all_factored.append(factored)

        print(f"Loaded {len(all_factored)} progressions")
        print(f"Found {len(self.pattern_to_id)} unique interval patterns")

        # Learn JOINT transitions
        print("\nLearning joint (type, root_movement) transitions...")

        for seq in all_factored:
            if seq:
                # Starting state
                self.start_states[(seq[0][0], seq[0][1])] += 1

            for i in range(len(seq) - 1):
                type1, root1 = seq[i]
                type2, root2 = seq[i + 1]
                root_movement = (root2 - root1) % 12

                # Key: current type
                # Value: (next_type, root_movement) pair
                self.joint_transitions[type1][(type2, root_movement)] += 1

        # Build grammar on transition sequences
        print("Building grammar on transition patterns...")

        # Encode each transition as a single token: type2 * 12 + root_movement
        transition_sequences = []
        for seq in all_factored:
            trans_seq = []
            for i in range(len(seq) - 1):
                type1, root1 = seq[i]
                type2, root2 = seq[i + 1]
                root_movement = (root2 - root1) % 12
                # Encode as single token
                token = type2 * 12 + root_movement
                trans_seq.append(token)
            if trans_seq:
                transition_sequences.append(trans_seq)

        self.grammar = SequiturGrammar()
        for i, seq in enumerate(transition_sequences):
            self.grammar.ingest(seq)
            if i < len(transition_sequences) - 1:
                self.grammar.ingest([-1])

        print(f"Grammar rules on transitions: {self.grammar.get_vocabulary_size()}")

        # Show learned patterns
        print(f"\nMost common joint transitions:")
        all_trans = Counter()
        for type1, trans in self.joint_transitions.items():
            for (type2, root_mv), count in trans.items():
                all_trans[(type1, type2, root_mv)] += count

        for (t1, t2, rm), count in all_trans.most_common(15):
            p1 = self.id_to_pattern[t1]
            p2 = self.id_to_pattern[t2]
            print(f"  {str(p1):18} → {str(p2):18} root+{rm:2}: {count}")

        print("\nLearning complete!")

    def generate(
        self,
        length: int = 8,
        start_root: Optional[int] = None,
        temperature: float = 1.0,
        repetition_penalty: float = 0.3
    ) -> List[frozenset]:
        """Generate using joint sampling."""

        # Sample starting state
        if start_root is None:
            states = list(self.start_states.keys())
            weights = [self.start_states[s] for s in states]
            start_type, start_root = random.choices(states, weights=weights)[0]
        else:
            # Pick most common type for this root (or any)
            types_for_root = [(t, r) for (t, r) in self.start_states.keys() if r == start_root]
            if types_for_root:
                start_type = random.choice(types_for_root)[0]
            else:
                start_type = random.choice(list(self.id_to_pattern.keys()))

        # Build progression
        current_type = start_type
        current_root = start_root

        chords = [realize_chord(self.id_to_pattern[current_type], current_root)]
        recent_types = [current_type]

        for _ in range(length - 1):
            if current_type not in self.joint_transitions:
                # Fallback: random
                next_type = random.choice(list(self.id_to_pattern.keys()))
                root_movement = random.choice([0, 2, 5, 7, 10])
            else:
                trans = self.joint_transitions[current_type]
                candidates = list(trans.keys())
                weights = np.array([trans[c] for c in candidates], dtype=float)

                # Repetition penalty on type
                for i, (next_t, _) in enumerate(candidates):
                    if next_t in recent_types[-3:]:
                        weights[i] *= repetition_penalty

                # Temperature
                if temperature != 1.0:
                    weights = np.power(weights + 1e-10, 1.0 / temperature)

                weights /= weights.sum()
                idx = np.random.choice(len(candidates), p=weights)
                next_type, root_movement = candidates[idx]

            # Apply
            current_type = next_type
            current_root = (current_root + root_movement) % 12

            chord = realize_chord(self.id_to_pattern[current_type], current_root)
            chords.append(chord)
            recent_types.append(current_type)

        return chords

    def generate_transposed(
        self,
        length: int = 8,
        temperature: float = 0.8
    ) -> Dict[int, List[frozenset]]:
        """
        Generate ONE progression pattern, transpose to all 12 keys.

        This samples the TYPE + ROOT_MOVEMENT sequence once,
        then shifts all roots by each key offset.
        """
        # Sample starting type (root doesn't matter for pattern)
        types = list(set(t for t, r in self.start_states.keys()))
        type_counts = Counter()
        for (t, r), c in self.start_states.items():
            type_counts[t] += c
        weights = [type_counts[t] for t in types]
        start_type = random.choices(types, weights=weights)[0]

        # Generate sequence of (type, root_movement) - root movements relative
        current_type = start_type
        type_sequence = [current_type]
        movement_sequence = [0]  # First chord has no movement

        recent_types = [current_type]

        for _ in range(length - 1):
            if current_type not in self.joint_transitions:
                next_type = random.choice(list(self.id_to_pattern.keys()))
                root_movement = random.choice([0, 2, 5, 7, 10])
            else:
                trans = self.joint_transitions[current_type]
                candidates = list(trans.keys())
                weights = np.array([trans[c] for c in candidates], dtype=float)

                for i, (next_t, _) in enumerate(candidates):
                    if next_t in recent_types[-3:]:
                        weights[i] *= 0.3

                if temperature != 1.0:
                    weights = np.power(weights + 1e-10, 1.0 / temperature)

                weights /= weights.sum()
                idx = np.random.choice(len(candidates), p=weights)
                next_type, root_movement = candidates[idx]

            type_sequence.append(next_type)
            movement_sequence.append(root_movement)
            current_type = next_type
            recent_types.append(current_type)

        # Now realize in all 12 keys
        results = {}
        for key in range(12):
            chords = []
            current_root = key
            for i, (tid, movement) in enumerate(zip(type_sequence, movement_sequence)):
                if i > 0:
                    current_root = (current_root + movement) % 12
                pattern = self.id_to_pattern[tid]
                chord = realize_chord(pattern, current_root)
                chords.append(chord)
            results[key] = chords

        return results


def progression_to_midi(progression: List[frozenset], output_path: str,
                        chord_duration: int = 480, velocity: int = 80, octave: int = 4):
    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    track.append(Message('program_change', program=0, time=0))

    for i, chord in enumerate(progression):
        if not chord:
            continue
        notes = [pc + (octave * 12) for pc in sorted(chord)]
        for j, note in enumerate(notes):
            track.append(Message('note_on', note=note, velocity=velocity,
                                time=0 if j > 0 else (0 if i == 0 else chord_duration)))
        for j, note in enumerate(notes):
            track.append(Message('note_off', note=note, velocity=0,
                                time=chord_duration if j == 0 else 0))

    mid.save(output_path)


def main():
    print("=" * 70)
    print("JOINT TYPE + ROOT GENERATOR")
    print("=" * 70)
    print("\nLearns (type_transition, root_movement) as joint patterns.")
    print("No music theory - purely from observed data.\n")

    output_dir = Path(__file__).parent / "generated_chords"
    output_dir.mkdir(exist_ok=True)

    # Clear old files
    for f in output_dir.glob("*.mid"):
        f.unlink()

    generator = JointGenerator()
    generator.learn("/home/arlo/free-midi-chords/output", max_files=500)

    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING PROGRESSIONS")
    print("=" * 60)

    print("\n--- Individual Progressions ---")
    for i in range(5):
        prog = generator.generate(length=8, temperature=0.7, repetition_penalty=0.2)
        print(f"\nProgression {i+1}:")
        print(f"  {' | '.join(chord_str(c) for c in prog)}")

        # Show the pattern (types + movements)
        roots = [get_root(c) for c in prog]
        movements = [0] + [(roots[i+1] - roots[i]) % 12 for i in range(len(roots)-1)]

        pattern_str = []
        for c, m in zip(prog, movements):
            p = get_interval_pattern(c)
            pattern_str.append(f"{p}+{m}")
        print(f"  Pattern: {' → '.join(pattern_str)}")

        progression_to_midi(prog, str(output_dir / f"joint_{i+1}.mid"))

    # =========================================================================
    print("\n--- Same Pattern in All Keys ---")

    all_keys = generator.generate_transposed(length=6, temperature=0.7)

    # Show the pattern (just display chords, skip pattern analysis to avoid KeyError)
    print(f"\nRealized in each key:")
    for key in range(12):
        prog = all_keys[key]
        print(f"  {ROOT_NAMES[key]:2}: {' | '.join(chord_str(c) for c in prog)}")
        progression_to_midi(prog, str(output_dir / f"pattern_key_{ROOT_NAMES[key].replace('#','s')}.mid"))

    # =========================================================================
    print("\n--- Longer Progression ---")

    long_prog = generator.generate(length=16, temperature=0.6, repetition_penalty=0.15)
    print(f"\n16-chord progression:")
    print(f"  {' | '.join(chord_str(c) for c in long_prog[:8])}")
    print(f"  {' | '.join(chord_str(c) for c in long_prog[8:])}")

    progression_to_midi(long_prog, str(output_dir / "long_progression.mid"))

    # =========================================================================
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nGenerated {len(list(output_dir.glob('*.mid')))} MIDI files in {output_dir}")


if __name__ == "__main__":
    main()
