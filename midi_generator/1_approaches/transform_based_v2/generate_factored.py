#!/usr/bin/env python3
"""
Factored Chord Progression Generator.

This generator PROPERLY uses the transform-based architecture:
1. Factor chords into (interval_pattern, root)
2. Learn patterns over chord TYPES (transposition-invariant)
3. Generate chord type sequences
4. Apply transposition to realize in any key

This is what the architecture was designed to do.
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
from core.groups.d24_group import D24Group


# =============================================================================
# Data Loading & Factorization
# =============================================================================

def load_midi_chords(midi_path: str) -> List[frozenset]:
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
                chords.append(frozenset(current_notes))
            current_time = time
            current_notes = {note % 12}

    if current_notes:
        chords.append(frozenset(current_notes))

    return chords


def get_interval_pattern(chord: frozenset) -> Tuple[int, ...]:
    """Extract interval pattern (transposition-invariant)."""
    if not chord:
        return ()
    pcs = sorted(chord)
    root = pcs[0]
    return tuple((pc - root) % 12 for pc in pcs)


def get_root(chord: frozenset) -> int:
    """Get root (lowest pitch class)."""
    return min(chord) if chord else 0


def realize_chord(interval_pattern: Tuple[int, ...], root: int) -> frozenset:
    """Create a chord from interval pattern + root."""
    return frozenset((root + interval) % 12 for interval in interval_pattern)


PATTERN_NAMES = {
    (0, 4, 7): "maj",
    (0, 3, 7): "min",
    (0, 3, 6): "dim",
    (0, 4, 8): "aug",
    (0, 5, 7): "sus4",
    (0, 2, 7): "sus2",
    (0, 4, 7, 10): "7",
    (0, 4, 7, 11): "maj7",
    (0, 3, 7, 10): "min7",
    (0, 5, 9): "min(inv)",  # First inversion minor
    (0, 3, 8): "maj(inv)",  # First inversion major
    (0, 5, 8): "min(inv2)", # Second inversion minor
    (0, 4, 9): "maj(inv2)", # Second inversion major
}

ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def chord_name(chord: frozenset) -> str:
    """Get readable chord name."""
    if not chord:
        return "rest"
    pattern = get_interval_pattern(chord)
    root = get_root(chord)
    pname = PATTERN_NAMES.get(pattern, f"?{pattern}")
    return f"{ROOT_NAMES[root]}{pname}"


def pattern_name(pattern: Tuple[int, ...]) -> str:
    """Get pattern type name (without root)."""
    return PATTERN_NAMES.get(pattern, f"{pattern}")


# =============================================================================
# Factored Generator
# =============================================================================

class FactoredChordGenerator:
    """
    Generates chord progressions using factored representation.

    Key insight: Separate WHAT (chord type) from WHERE (transposition).
    Learn patterns over chord types, then transpose to any key.
    """

    def __init__(self):
        self.grammar: Optional[SequiturGrammar] = None
        self.d24 = D24Group()

        # Chord type vocabulary
        self.pattern_to_id: Dict[Tuple, int] = {}
        self.id_to_pattern: Dict[int, Tuple] = {}

        # Learned transitions (over chord TYPES, not specific chords)
        self.type_transitions: Dict[int, Counter] = defaultdict(Counter)

        # Root movement patterns
        self.root_movements: Counter = Counter()

        # Common starting types
        self.start_types: Counter = Counter()

    def learn(self, midi_dir: str, max_files: int = 500):
        """Learn from chord progressions, factoring into types + roots."""
        print("=" * 60)
        print("LEARNING FACTORED REPRESENTATION")
        print("=" * 60)

        midi_files = list(Path(midi_dir).rglob("*.mid"))[:max_files]
        print(f"Loading {len(midi_files)} files...")

        all_sequences = []  # List of (pattern_id, root) sequences
        all_type_sequences = []  # Just pattern_ids

        for midi_path in midi_files:
            chords = load_midi_chords(str(midi_path))
            if len(chords) >= 2:
                factored = []
                for chord in chords:
                    pattern = get_interval_pattern(chord)
                    root = get_root(chord)

                    # Build vocabulary
                    if pattern not in self.pattern_to_id:
                        pid = len(self.pattern_to_id)
                        self.pattern_to_id[pattern] = pid
                        self.id_to_pattern[pid] = pattern

                    factored.append((self.pattern_to_id[pattern], root))

                all_sequences.append(factored)
                all_type_sequences.append([f[0] for f in factored])

        print(f"Loaded {len(all_sequences)} progressions")
        print(f"Found {len(self.pattern_to_id)} unique chord types")

        # Learn TYPE transitions (transposition-invariant)
        print("\nLearning chord type transitions...")
        for seq in all_type_sequences:
            if seq:
                self.start_types[seq[0]] += 1
            for i in range(len(seq) - 1):
                self.type_transitions[seq[i]][seq[i + 1]] += 1

        # Learn ROOT movements
        print("Learning root movement patterns...")
        for seq in all_sequences:
            for i in range(len(seq) - 1):
                root1 = seq[i][1]
                root2 = seq[i + 1][1]
                movement = (root2 - root1) % 12
                self.root_movements[movement] += 1

        # Build grammar on TYPE sequences
        print("Building grammar on chord types...")
        self.grammar = SequiturGrammar()
        for i, seq in enumerate(all_type_sequences):
            self.grammar.ingest(seq)
            if i < len(all_type_sequences) - 1:
                self.grammar.ingest([-1])

        print(f"Grammar rules: {self.grammar.get_vocabulary_size()}")

        # Show what was learned
        print(f"\nMost common chord types:")
        type_counts = Counter()
        for seq in all_type_sequences:
            type_counts.update(seq)

        for pid, count in type_counts.most_common(10):
            pattern = self.id_to_pattern[pid]
            print(f"  {pattern_name(pattern):15} {str(pattern):20} count: {count}")

        print(f"\nMost common root movements:")
        for movement, count in self.root_movements.most_common(8):
            interval_name = ['unison', 'm2', 'M2', 'm3', 'M3', 'P4',
                           'tritone', 'P5', 'm6', 'M6', 'm7', 'M7'][movement]
            print(f"  +{movement:2} ({interval_name:7}): {count}")

        print("\nLearning complete!")

    def sample_type_sequence(
        self,
        length: int = 8,
        temperature: float = 1.0,
        repetition_penalty: float = 0.3
    ) -> List[int]:
        """Sample a sequence of chord TYPE ids."""
        # Start with common starting type
        types = list(self.start_types.keys())
        weights = [self.start_types[t] for t in types]
        current = random.choices(types, weights=weights)[0]

        sequence = [current]

        for _ in range(length - 1):
            if current not in self.type_transitions:
                # Fallback to any type
                next_type = random.choice(list(self.id_to_pattern.keys()))
            else:
                trans = self.type_transitions[current]
                candidates = list(trans.keys())
                weights = np.array([trans[c] for c in candidates], dtype=float)

                # Repetition penalty
                recent = sequence[-3:] if len(sequence) >= 3 else sequence
                for i, c in enumerate(candidates):
                    if c in recent:
                        weights[i] *= repetition_penalty

                # Temperature
                if temperature != 1.0:
                    weights = np.power(weights + 1e-10, 1.0 / temperature)

                weights /= weights.sum()
                next_type = np.random.choice(candidates, p=weights)

            sequence.append(next_type)
            current = next_type

        return sequence

    def sample_root_sequence(
        self,
        length: int,
        start_root: int = 0,
        use_learned_movements: bool = True
    ) -> List[int]:
        """Sample root movements to create a root sequence."""
        roots = [start_root]

        for _ in range(length - 1):
            if use_learned_movements:
                # Sample from learned root movements
                movements = list(self.root_movements.keys())
                weights = [self.root_movements[m] for m in movements]
                movement = random.choices(movements, weights=weights)[0]
            else:
                # Common movements: 5 (P4), 7 (P5), 0 (repeat)
                movement = random.choice([0, 5, 7, 2, 10])

            roots.append((roots[-1] + movement) % 12)

        return roots

    def generate(
        self,
        length: int = 8,
        key: Optional[int] = None,
        temperature: float = 0.8,
        repetition_penalty: float = 0.2
    ) -> List[frozenset]:
        """
        Generate a chord progression.

        Args:
            length: Number of chords
            key: Root of first chord (0-11), None = random
            temperature: Sampling temperature
            repetition_penalty: Penalty for repeating chord types

        Returns:
            List of chords (as pitch-class sets)
        """
        # 1. Sample chord TYPE sequence (transposition-invariant)
        type_sequence = self.sample_type_sequence(
            length=length,
            temperature=temperature,
            repetition_penalty=repetition_penalty
        )

        # 2. Sample ROOT sequence (or derive from key)
        if key is None:
            key = random.randint(0, 11)

        root_sequence = self.sample_root_sequence(length, start_root=key)

        # 3. Realize: combine types + roots into actual chords
        chords = []
        for type_id, root in zip(type_sequence, root_sequence):
            pattern = self.id_to_pattern[type_id]
            chord = realize_chord(pattern, root)
            chords.append(chord)

        return chords

    def generate_in_all_keys(
        self,
        length: int = 8,
        temperature: float = 0.8
    ) -> Dict[int, List[frozenset]]:
        """
        Generate ONE type sequence, realize in ALL 12 keys.

        This demonstrates the factorization - same progression, different keys.
        """
        # Sample type sequence ONCE
        type_sequence = self.sample_type_sequence(length, temperature)

        # Sample root movements ONCE
        base_roots = self.sample_root_sequence(length, start_root=0)

        # Realize in all 12 keys
        results = {}
        for key in range(12):
            transposed_roots = [(r + key) % 12 for r in base_roots]
            chords = []
            for type_id, root in zip(type_sequence, transposed_roots):
                pattern = self.id_to_pattern[type_id]
                chord = realize_chord(pattern, root)
                chords.append(chord)
            results[key] = chords

        return results

    def generate_with_grammar_rule(
        self,
        rule_id: int,
        key: int = 0
    ) -> Optional[List[frozenset]]:
        """Generate by expanding a specific grammar rule."""
        if rule_id not in self.grammar.rules:
            return None

        rule = self.grammar.rules[rule_id]
        type_sequence = [t for t in rule.expand() if t >= 0 and t in self.id_to_pattern]

        if not type_sequence:
            return None

        root_sequence = self.sample_root_sequence(len(type_sequence), start_root=key)

        chords = []
        for type_id, root in zip(type_sequence, root_sequence):
            pattern = self.id_to_pattern[type_id]
            chord = realize_chord(pattern, root)
            chords.append(chord)

        return chords


# =============================================================================
# MIDI Output
# =============================================================================

def progression_to_midi(
    progression: List[frozenset],
    output_path: str,
    chord_duration: int = 480,
    velocity: int = 80,
    octave: int = 4
):
    """Write progression to MIDI file."""
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


def print_progression(chords: List[frozenset]) -> str:
    """Format progression for display."""
    return " | ".join(chord_name(c) for c in chords)


def print_type_sequence(type_ids: List[int], id_to_pattern: Dict) -> str:
    """Format type sequence for display."""
    return " → ".join(pattern_name(id_to_pattern[t]) for t in type_ids)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("FACTORED CHORD GENERATOR")
    print("=" * 70)
    print("\nThis generator separates CHORD TYPE from TRANSPOSITION.")
    print("It learns patterns over types, then realizes in any key.\n")

    output_dir = Path(__file__).parent / "generated_chords"
    output_dir.mkdir(exist_ok=True)

    # Learn
    generator = FactoredChordGenerator()
    generator.learn("/home/arlo/free-midi-chords/output", max_files=500)

    # =========================================================================
    # Demo 1: Same progression in multiple keys
    # =========================================================================
    print("\n" + "=" * 60)
    print("DEMO 1: SAME PROGRESSION IN ALL 12 KEYS")
    print("=" * 60)
    print("\nGenerating ONE chord type sequence, realizing in all keys...")

    all_keys = generator.generate_in_all_keys(length=4, temperature=0.7)

    # Show the type sequence
    first_prog = all_keys[0]
    type_seq = [generator.pattern_to_id[get_interval_pattern(c)] for c in first_prog]
    print(f"\nChord type sequence: {print_type_sequence(type_seq, generator.id_to_pattern)}")

    print(f"\nRealized in each key:")
    for key in range(12):
        prog = all_keys[key]
        print(f"  Key of {ROOT_NAMES[key]:2}: {print_progression(prog)}")

        filepath = output_dir / f"same_prog_key_{ROOT_NAMES[key].replace('#', 's')}.mid"
        progression_to_midi(prog, str(filepath))

    # =========================================================================
    # Demo 2: Different progressions, same structure
    # =========================================================================
    print("\n" + "=" * 60)
    print("DEMO 2: GENERATING VARIED PROGRESSIONS")
    print("=" * 60)

    for i in range(5):
        prog = generator.generate(length=8, temperature=0.7, repetition_penalty=0.2)

        # Show both the types and the realized chords
        type_seq = [generator.pattern_to_id[get_interval_pattern(c)] for c in prog]

        print(f"\nProgression {i + 1}:")
        print(f"  Types:  {print_type_sequence(type_seq, generator.id_to_pattern)}")
        print(f"  Chords: {print_progression(prog)}")

        filepath = output_dir / f"factored_prog_{i + 1}.mid"
        progression_to_midi(prog, str(filepath))

    # =========================================================================
    # Demo 3: Grammar-based generation
    # =========================================================================
    print("\n" + "=" * 60)
    print("DEMO 3: GRAMMAR RULE EXPANSION")
    print("=" * 60)
    print("\nFinding interesting grammar rules and expanding them...")

    # Find rules with 3+ types, used multiple times
    interesting_rules = []
    for rule_id, rule in generator.grammar.rules.items():
        if rule_id == 0:
            continue
        expansion = [t for t in rule.expand() if t >= 0 and t in generator.id_to_pattern]
        if len(expansion) >= 3 and rule.usage_count >= 2:
            interesting_rules.append((rule_id, expansion, rule.usage_count))

    interesting_rules.sort(key=lambda x: -x[2])

    for rule_id, expansion, usage in interesting_rules[:5]:
        print(f"\nRule R{rule_id} (used {usage}x):")
        print(f"  Pattern: {print_type_sequence(expansion[:8], generator.id_to_pattern)}")

        # Realize in C and G
        for key, key_name in [(0, 'C'), (7, 'G')]:
            prog = generator.generate_with_grammar_rule(rule_id, key=key)
            if prog:
                print(f"  In {key_name}: {print_progression(prog[:8])}")

                filepath = output_dir / f"rule_{rule_id}_key_{key_name}.mid"
                progression_to_midi(prog, str(filepath))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    files = list(output_dir.glob("*.mid"))
    print(f"\nGenerated {len(files)} MIDI files in {output_dir}")

    print(f"""
    Key insight demonstrated:

    1. Chord types (maj, min, etc.) are learned SEPARATELY from roots
    2. Patterns over types generalize across all 12 keys
    3. Grammar rules capture common TYPE sequences (not tied to key)
    4. Generation: sample types → choose key → realize chords

    This is the PROPER use of the transform-based architecture.
    """)


if __name__ == "__main__":
    main()
