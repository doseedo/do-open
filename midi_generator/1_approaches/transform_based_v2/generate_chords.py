#!/usr/bin/env python3
"""
Chord Progression Generator using Transform-Based V2.

Generation approach:
1. Learn grammar from corpus (SEQUITUR)
2. Build Markov chain on grammar rule transitions
3. Sample new progressions using the Markov model
4. Optionally apply D24 transforms for variation
5. Output MIDI files

This validates that the transform-based architecture can generate
coherent musical output on simplified (single-track chord) data.
"""

import sys
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import random

sys.path.insert(0, str(Path(__file__).parent))

try:
    import mido
    from mido import Message, MidiFile, MidiTrack
except ImportError:
    print("Installing mido...")
    os.system("pip install mido")
    import mido
    from mido import Message, MidiFile, MidiTrack

from grammar.sequitur import SequiturGrammar
from core.groups.d24_group import D24Group


# =============================================================================
# MIDI Loading (from test script)
# =============================================================================

def load_midi_chords(midi_path: str) -> List[Tuple[int, frozenset]]:
    """Load MIDI and extract chord events as (time, pitch_class_set)."""
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
                chords.append((current_time, frozenset(current_notes)))
            current_time = time
            current_notes = {note % 12}

    if current_notes:
        chords.append((current_time, frozenset(current_notes)))

    return chords


def chord_to_token(chord: frozenset) -> int:
    """Convert pitch-class set to unique token."""
    return sum(1 << pc for pc in chord)


def token_to_chord(token: int) -> frozenset:
    """Convert token back to pitch-class set."""
    return frozenset(pc for pc in range(12) if token & (1 << pc))


def chord_name(pcs: frozenset) -> str:
    """Get readable name for a chord."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if not pcs:
        return "rest"
    pcs_list = sorted(pcs)
    if len(pcs_list) == 3:
        root = pcs_list[0]
        intervals = [(p - root) % 12 for p in pcs_list]
        if intervals == [0, 4, 7]:
            return f"{note_names[root]}maj"
        elif intervals == [0, 3, 7]:
            return f"{note_names[root]}min"
        elif intervals == [0, 3, 6]:
            return f"{note_names[root]}dim"
        elif intervals == [0, 4, 8]:
            return f"{note_names[root]}aug"
    return "{" + ",".join(note_names[pc] for pc in pcs_list) + "}"


# =============================================================================
# Markov Chain Generator
# =============================================================================

class MarkovChordGenerator:
    """
    Generates chord progressions using a Markov chain learned from grammar rules.

    Two-level generation:
    1. Rule-level Markov: Learn transitions between grammar rules
    2. Token-level Markov: Learn transitions between individual chords

    The grammar captures common patterns (I-IV-V-I becomes a rule),
    and Markov captures how these patterns connect.
    """

    def __init__(self):
        self.grammar: Optional[SequiturGrammar] = None
        self.chord_vocab: Dict[int, frozenset] = {}
        self.d24 = D24Group()

        # Token-level transitions: P(next_token | current_token)
        self.token_transitions: Dict[int, Counter] = defaultdict(Counter)

        # Rule-level: which tokens commonly start/end rules
        self.rule_start_tokens: Counter = Counter()
        self.rule_end_tokens: Counter = Counter()

        # Chord frequency for sampling
        self.chord_frequency: Counter = Counter()

    def learn(self, midi_dir: str, max_files: int = 500):
        """Learn from a directory of chord MIDI files."""
        print("=" * 60)
        print("LEARNING FROM CORPUS")
        print("=" * 60)

        midi_files = list(Path(midi_dir).rglob("*.mid"))[:max_files]
        print(f"Found {len(midi_files)} MIDI files")

        all_sequences = []
        all_tokens = []

        for midi_path in midi_files:
            chords = load_midi_chords(str(midi_path))
            if len(chords) >= 2:
                tokens = [chord_to_token(c[1]) for c in chords]
                all_sequences.append(tokens)
                all_tokens.extend(tokens)

                for _, chord in chords:
                    token = chord_to_token(chord)
                    self.chord_vocab[token] = chord

        print(f"Loaded {len(all_sequences)} progressions, {len(all_tokens)} chords")
        print(f"Unique chords: {len(self.chord_vocab)}")

        # Build token-level Markov chain
        print("\nBuilding token-level Markov chain...")
        for seq in all_sequences:
            for i in range(len(seq) - 1):
                self.token_transitions[seq[i]][seq[i + 1]] += 1
            self.chord_frequency.update(seq)

        # Build grammar
        print("\nBuilding SEQUITUR grammar...")
        self.grammar = SequiturGrammar()
        for i, seq in enumerate(all_sequences):
            self.grammar.ingest(seq)
            if i < len(all_sequences) - 1:
                self.grammar.ingest([-1])

        print(f"Grammar rules: {self.grammar.get_vocabulary_size()}")

        # Analyze rule boundaries
        for rule_id, rule in self.grammar.rules.items():
            if rule_id == 0:
                continue
            expansion = rule.expand()
            valid = [t for t in expansion if t in self.chord_vocab]
            if valid:
                self.rule_start_tokens[valid[0]] += rule.usage_count
                self.rule_end_tokens[valid[-1]] += rule.usage_count

        print("Learning complete!")

    def sample_next_token(
        self,
        current_token: int,
        temperature: float = 1.0,
        repetition_penalty: float = 0.3,
        recent_tokens: List[int] = None
    ) -> int:
        """
        Sample next chord token given current, using Markov transitions.

        Args:
            current_token: Current chord token
            temperature: Sampling temperature (lower = more deterministic)
            repetition_penalty: Factor to reduce probability of recent tokens (0-1)
            recent_tokens: List of recently generated tokens to penalize
        """
        if current_token not in self.token_transitions:
            # Fallback to frequency-weighted sampling
            tokens = list(self.chord_frequency.keys())
            weights = np.array([self.chord_frequency[t] for t in tokens], dtype=float)
        else:
            transitions = self.token_transitions[current_token]
            tokens = list(transitions.keys())
            weights = np.array([transitions[t] for t in tokens], dtype=float)

        # Apply repetition penalty
        if recent_tokens and repetition_penalty < 1.0:
            for i, tok in enumerate(tokens):
                if tok in recent_tokens:
                    # Stronger penalty for more recent tokens
                    recency = recent_tokens[::-1].index(tok) if tok in recent_tokens else len(recent_tokens)
                    penalty = repetition_penalty ** (1 + (len(recent_tokens) - recency) / len(recent_tokens))
                    weights[i] *= penalty

        # Apply temperature
        if temperature != 1.0:
            weights = np.power(weights + 1e-10, 1.0 / temperature)

        probs = weights / weights.sum()
        return np.random.choice(tokens, p=probs)

    def generate_progression(
        self,
        length: int = 8,
        start_token: Optional[int] = None,
        temperature: float = 1.0,
        repetition_penalty: float = 0.2,
        use_grammar: bool = True
    ) -> List[int]:
        """
        Generate a chord progression.

        Args:
            length: Number of chords to generate
            start_token: Starting chord (None = sample from common starts)
            temperature: Sampling temperature (lower = more deterministic)
            repetition_penalty: Penalty for repeating recent chords (0-1, lower = more penalty)
            use_grammar: If True, prefer grammar rule boundaries

        Returns:
            List of chord tokens
        """
        if start_token is None:
            # Sample from common progression starts
            if self.rule_start_tokens:
                tokens = list(self.rule_start_tokens.keys())
                weights = [self.rule_start_tokens[t] for t in tokens]
                start_token = random.choices(tokens, weights=weights)[0]
            else:
                start_token = random.choice(list(self.chord_vocab.keys()))

        progression = [start_token]
        current = start_token

        for _ in range(length - 1):
            # Pass recent tokens for repetition penalty
            recent = progression[-4:] if len(progression) >= 4 else progression
            next_token = self.sample_next_token(
                current,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                recent_tokens=recent
            )
            progression.append(next_token)
            current = next_token

        return progression

    def generate_with_transform(
        self,
        base_progression: List[int],
        transform: int
    ) -> List[int]:
        """
        Apply a D24 transform to an entire progression.

        Args:
            base_progression: List of chord tokens
            transform: D24 element (0-23)

        Returns:
            Transformed progression
        """
        result = []
        for token in base_progression:
            chord = token_to_chord(token)
            pcs = np.array(list(chord))
            transformed_pcs = self.d24.apply_to_pitch_class_array(transform, pcs)
            new_chord = frozenset(int(pc) for pc in transformed_pcs)
            result.append(chord_to_token(new_chord))
        return result

    def generate_variation_set(
        self,
        length: int = 8,
        num_variations: int = 4,
        temperature: float = 0.8,
        repetition_penalty: float = 0.15
    ) -> List[Tuple[str, List[int]]]:
        """
        Generate a base progression and variations via D24 transforms.

        Returns list of (description, progression) tuples.
        """
        base = self.generate_progression(
            length,
            temperature=temperature,
            repetition_penalty=repetition_penalty
        )

        variations = [("Original", base)]

        # Common musical transforms
        transforms = [
            (5, "Up 4th (subdominant)"),
            (7, "Up 5th (dominant)"),
            (2, "Up whole step"),
            (10, "Down whole step"),
        ]

        for t, desc in transforms[:num_variations - 1]:
            varied = self.generate_with_transform(base, t)
            variations.append((f"T{t}: {desc}", varied))

        return variations


# =============================================================================
# MIDI Output
# =============================================================================

def progression_to_midi(
    progression: List[int],
    output_path: str,
    chord_duration: int = 480,  # ticks (1 beat at 480 ticks/beat)
    velocity: int = 80,
    octave: int = 4
):
    """
    Write a chord progression to a MIDI file.

    Args:
        progression: List of chord tokens
        output_path: Path to write MIDI file
        chord_duration: Duration of each chord in ticks
        velocity: Note velocity
        octave: Base octave for chords
    """
    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)

    # Set tempo (120 BPM)
    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

    # Set instrument (piano)
    track.append(Message('program_change', program=0, time=0))

    for i, token in enumerate(progression):
        chord = token_to_chord(token)
        if not chord:
            continue

        # Convert pitch classes to MIDI notes
        notes = [pc + (octave * 12) for pc in sorted(chord)]

        # Note on (first note has delta time, rest have 0)
        for j, note in enumerate(notes):
            track.append(Message('note_on', note=note, velocity=velocity,
                                time=0 if j > 0 else (0 if i == 0 else chord_duration)))

        # Note off
        for j, note in enumerate(notes):
            track.append(Message('note_off', note=note, velocity=0,
                                time=chord_duration if j == 0 else 0))

    mid.save(output_path)


def print_progression(progression: List[int], chord_vocab: Dict[int, frozenset]):
    """Print a progression in readable format."""
    names = []
    for token in progression:
        if token in chord_vocab:
            names.append(chord_name(chord_vocab[token]))
        else:
            chord = token_to_chord(token)
            names.append(chord_name(chord))
    return " | ".join(names)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("CHORD PROGRESSION GENERATOR - Transform Based V2")
    print("=" * 70)

    # Configuration
    midi_dir = "/home/arlo/free-midi-chords/output"
    output_dir = Path(__file__).parent / "generated_chords"
    output_dir.mkdir(exist_ok=True)

    max_files = 500
    num_samples = 5
    progression_length = 8

    # Learn from corpus
    generator = MarkovChordGenerator()
    generator.learn(midi_dir, max_files)

    # Generate samples
    print("\n" + "=" * 60)
    print("GENERATING CHORD PROGRESSIONS")
    print("=" * 60)

    all_generated = []

    for i in range(num_samples):
        print(f"\n--- Sample {i + 1} ---")

        # Generate with variations
        variations = generator.generate_variation_set(
            length=progression_length,
            num_variations=4,
            temperature=0.8
        )

        for desc, progression in variations:
            prog_str = print_progression(progression, generator.chord_vocab)
            print(f"  {desc:25} {prog_str}")

            # Save MIDI
            filename = f"sample_{i + 1}_{desc.split(':')[0].replace(' ', '_').lower()}.mid"
            filepath = output_dir / filename
            progression_to_midi(progression, str(filepath))
            all_generated.append((desc, progression, filepath))

    # Generate longer progressions
    print("\n--- Extended Progressions ---")
    for i in range(3):
        long_prog = generator.generate_progression(
            length=16,
            temperature=0.7,
            repetition_penalty=0.1  # Strong penalty for longer progressions
        )
        prog_str = print_progression(long_prog, generator.chord_vocab)
        print(f"  Extended {i + 1}: {prog_str}")

        filepath = output_dir / f"extended_{i + 1}.mid"
        progression_to_midi(long_prog, str(filepath), chord_duration=480)

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\n  Output directory: {output_dir}")
    print(f"  Files generated: {len(list(output_dir.glob('*.mid')))}")

    # Analyze generated vs training distribution
    print("\n  Analyzing generated chord distribution...")
    generated_tokens = []
    for _, prog, _ in all_generated:
        generated_tokens.extend(prog)

    gen_counter = Counter(generated_tokens)
    train_counter = generator.chord_frequency

    print(f"\n  Top generated chords:")
    for token, count in gen_counter.most_common(5):
        chord = token_to_chord(token)
        train_pct = train_counter[token] / sum(train_counter.values()) * 100
        gen_pct = count / len(generated_tokens) * 100
        print(f"    {chord_name(chord):12} gen:{gen_pct:5.1f}%  train:{train_pct:5.1f}%")

    # Test D24 transform coherence
    print("\n  D24 Transform Analysis on Generated:")
    d24 = generator.d24
    transform_counts = Counter()

    for _, prog, _ in all_generated:
        for j in range(len(prog) - 1):
            c1 = token_to_chord(prog[j])
            c2 = token_to_chord(prog[j + 1])
            if len(c1) == len(c2) and c1 and c2:
                pcs1 = np.array(sorted(c1))
                for g in range(24):
                    if set(d24.apply_to_pitch_class_array(g, pcs1)) == c2:
                        transform_counts[g] += 1
                        break

    print(f"    Transform distribution in generated:")
    for t, count in transform_counts.most_common(8):
        print(f"      {d24.element_name(t):4}: {count}")


if __name__ == "__main__":
    main()
