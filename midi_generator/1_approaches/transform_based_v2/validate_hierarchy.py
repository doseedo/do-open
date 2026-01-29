#!/usr/bin/env python3
"""
Validate the sophisticated claims about transform_based:

1. Does SEQUITUR find PHRASE-level patterns (not just chord pairs)?
2. Are any phrases related by D24 transforms (transposed versions of each other)?
3. Can we generate at the phrase level instead of chord-by-chord?

This is the honest test of whether the architecture does what it claims.
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import mido
from grammar.sequitur import SequiturGrammar
from core.groups.d24_group import D24Group


def load_midi_chords(midi_path: str) -> List[frozenset]:
    """Load MIDI and return list of pitch-class sets."""
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


def chord_to_token(chord: frozenset) -> int:
    return sum(1 << pc for pc in chord)


def token_to_chord(token: int) -> frozenset:
    return frozenset(pc for pc in range(12) if token & (1 << pc))


def chord_name(pcs: frozenset) -> str:
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if not pcs:
        return "rest"
    pcs_list = sorted(pcs)
    if len(pcs_list) == 3:
        root = pcs_list[0]
        intervals = [(p - root) % 12 for p in pcs_list]
        if intervals == [0, 4, 7]:
            return f"{note_names[root]}"
        elif intervals == [0, 3, 7]:
            return f"{note_names[root]}m"
        elif intervals == [0, 3, 6]:
            return f"{note_names[root]}dim"
    return "{" + ",".join(note_names[pc] for pc in pcs_list) + "}"


def check_phrase_transform(phrase1: List[int], phrase2: List[int], d24: D24Group) -> int:
    """
    Check if phrase2 is a D24 transform of phrase1.
    Returns transform ID (0-23) or -1 if not related.
    """
    if len(phrase1) != len(phrase2):
        return -1

    # Try each of the 24 transforms
    for g in range(24):
        match = True
        for t1, t2 in zip(phrase1, phrase2):
            c1 = token_to_chord(t1)
            c2 = token_to_chord(t2)

            if len(c1) != len(c2):
                match = False
                break

            pcs1 = np.array(sorted(c1))
            transformed = set(d24.apply_to_pitch_class_array(g, pcs1))

            if transformed != c2:
                match = False
                break

        if match:
            return g

    return -1


def main():
    print("=" * 70)
    print("VALIDATING HIERARCHICAL PATTERNS & TRANSFORM RELATIONSHIPS")
    print("=" * 70)

    # Load data
    midi_dir = "/home/arlo/free-midi-chords/output"
    midi_files = list(Path(midi_dir).rglob("*.mid"))[:500]

    all_sequences = []
    chord_vocab = {}

    for midi_path in midi_files:
        chords = load_midi_chords(str(midi_path))
        if len(chords) >= 2:
            tokens = [chord_to_token(c) for c in chords]
            all_sequences.append(tokens)
            for c in chords:
                chord_vocab[chord_to_token(c)] = c

    print(f"\nLoaded {len(all_sequences)} progressions")

    # Build grammar
    grammar = SequiturGrammar()
    for i, seq in enumerate(all_sequences):
        grammar.ingest(seq)
        if i < len(all_sequences) - 1:
            grammar.ingest([-1])

    # =========================================================================
    # TEST 1: Does SEQUITUR find PHRASE-level patterns?
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: PHRASE-LEVEL PATTERNS")
    print("=" * 70)
    print("\nQuestion: Did SEQUITUR find patterns longer than 2 chords?")

    rules_by_length = defaultdict(list)

    for rule_id, rule in grammar.rules.items():
        if rule_id == 0:
            continue
        expansion = rule.expand()
        # Filter out separators
        chord_expansion = [t for t in expansion if t in chord_vocab]
        if len(chord_expansion) >= 2:
            rules_by_length[len(chord_expansion)].append((rule_id, chord_expansion, rule.usage_count))

    print(f"\nRules by expansion length:")
    for length in sorted(rules_by_length.keys()):
        rules = rules_by_length[length]
        total_usage = sum(r[2] for r in rules)
        print(f"  {length}-chord patterns: {len(rules)} rules, used {total_usage} total times")

    # Show examples of longer patterns
    print(f"\nExample 3+ chord patterns (phrases):")
    long_patterns = []
    for length in sorted(rules_by_length.keys(), reverse=True):
        if length >= 3:
            for rule_id, expansion, usage in rules_by_length[length][:3]:
                names = [chord_name(chord_vocab.get(t, frozenset())) for t in expansion[:8]]
                suffix = "..." if len(expansion) > 8 else ""
                print(f"  R{rule_id} (used {usage}x): {' | '.join(names)}{suffix}")
                long_patterns.append((rule_id, expansion, usage))
        if len(long_patterns) >= 10:
            break

    has_phrases = any(length >= 3 for length in rules_by_length.keys())
    print(f"\n>>> RESULT: {'YES' if has_phrases else 'NO'} - SEQUITUR {'found' if has_phrases else 'did NOT find'} phrase-level patterns")

    # =========================================================================
    # TEST 2: Are any phrases TRANSFORMS of each other?
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: TRANSFORM RELATIONSHIPS BETWEEN PHRASES")
    print("=" * 70)
    print("\nQuestion: Are any discovered phrases transpositions of each other?")

    d24 = D24Group()

    # Get all phrases (3+ chords)
    phrases = []
    for length in rules_by_length:
        if length >= 3:
            for rule_id, expansion, usage in rules_by_length[length]:
                phrases.append((rule_id, expansion, usage))

    # Check all pairs for transform relationships
    transform_pairs = []

    for i, (id1, exp1, usage1) in enumerate(phrases):
        for j, (id2, exp2, usage2) in enumerate(phrases):
            if i >= j:
                continue

            transform = check_phrase_transform(exp1, exp2, d24)
            if transform > 0:  # Exclude identity (T0)
                transform_pairs.append((id1, id2, transform, exp1, exp2))

    if transform_pairs:
        print(f"\nFound {len(transform_pairs)} phrase pairs related by transforms:")
        for id1, id2, t, exp1, exp2 in transform_pairs[:10]:
            names1 = [chord_name(chord_vocab.get(tok, frozenset())) for tok in exp1[:4]]
            names2 = [chord_name(chord_vocab.get(tok, frozenset())) for tok in exp2[:4]]
            print(f"\n  R{id1}: {' | '.join(names1)}")
            print(f"  R{id2}: {' | '.join(names2)}")
            print(f"  Relationship: {d24.element_name(t)} (transpose by {t if t < 12 else 'inversion'})")
    else:
        print(f"\nNo transform relationships found between {len(phrases)} phrases.")

    print(f"\n>>> RESULT: {'YES' if transform_pairs else 'NO'} - {'Found' if transform_pairs else 'Did NOT find'} phrase-level transform relationships")

    # =========================================================================
    # TEST 3: Can we find "chorus = verse + transposition" patterns?
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: SECTION-LEVEL STRUCTURE")
    print("=" * 70)
    print("\nQuestion: Do progressions contain repeated/transposed sections?")

    # Look within individual progressions for repeated structure
    section_patterns = []

    for seq in all_sequences:
        if len(seq) < 8:
            continue

        # Check if second half is transform of first half
        mid = len(seq) // 2
        first_half = seq[:mid]
        second_half = seq[mid:mid + len(first_half)]

        if len(first_half) == len(second_half):
            transform = check_phrase_transform(first_half, second_half, d24)
            if transform >= 0:
                section_patterns.append((first_half, second_half, transform))

    if section_patterns:
        print(f"\nFound {len(section_patterns)} progressions with structural repetition:")

        # Count by transform type
        transform_counts = Counter(p[2] for p in section_patterns)
        print(f"\n  By transform type:")
        for t, count in transform_counts.most_common(10):
            name = d24.element_name(t)
            desc = "exact repeat" if t == 0 else f"transposed ({name})"
            print(f"    {name}: {count} progressions ({desc})")

        # Show examples of non-identity transforms
        print(f"\n  Examples of transposed sections:")
        shown = 0
        for first, second, t in section_patterns:
            if t > 0 and shown < 3:
                names1 = [chord_name(chord_vocab.get(tok, frozenset())) for tok in first[:4]]
                names2 = [chord_name(chord_vocab.get(tok, frozenset())) for tok in second[:4]]
                print(f"\n    Section A: {' | '.join(names1)}")
                print(f"    Section B: {' | '.join(names2)}")
                print(f"    Relation: B = {d24.element_name(t)}(A)")
                shown += 1
    else:
        print(f"\nNo section-level repetition patterns found.")

    has_sections = len(section_patterns) > 0
    has_transposed_sections = any(p[2] > 0 for p in section_patterns)

    print(f"\n>>> RESULT: {'YES' if has_transposed_sections else 'NO'} - {'Found' if has_transposed_sections else 'Did NOT find'} transposed sections")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(f"""
    Claim 1: "Hierarchical patterns (phrase-level)"
    Result:  {'VALIDATED' if has_phrases else 'NOT FOUND'} - SEQUITUR {'found' if has_phrases else 'did not find'} 3+ chord patterns

    Claim 2: "Transform relationships between phrases"
    Result:  {'VALIDATED' if transform_pairs else 'NOT FOUND'} - {'Found' if transform_pairs else 'No'} phrases that are transpositions of each other

    Claim 3: "Section-level structure (chorus = transposed verse)"
    Result:  {'VALIDATED' if has_transposed_sections else 'NOT FOUND'} - {'Found' if has_transposed_sections else 'No'} transposed sections in progressions
    """)

    if has_phrases and (transform_pairs or has_transposed_sections):
        print("    CONCLUSION: The architecture CAN capture hierarchical + transform structure.")
        print("                The simple Markov generator didn't USE this - but it exists in the data.")
    else:
        print("    CONCLUSION: The chord data may be too simple to validate these claims.")
        print("                Need more complex music (full songs with verses/choruses).")


if __name__ == "__main__":
    main()
