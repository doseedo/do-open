#!/usr/bin/env python3
"""
Extended Harmony - Usage Examples

Comprehensive examples demonstrating all features of the Extended Harmony module
in real musical contexts.

Author: Agent 8
Date: 2025-11-19
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from advanced_modules.extended_harmony import (
    ExtendedHarmony,
    ClusterType,
    Chord
)


def example_jazz_reharmonization():
    """
    Example 1: Jazz Reharmonization with Upper Structures

    Transform a basic ii-V-I progression into sophisticated jazz harmony
    using upper structure triads.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Jazz Reharmonization with Upper Structures")
    print("=" * 70)

    harmony = ExtendedHarmony()

    print("\nBasic ii-V-I in C major:")
    print("  Dm7 - G7 - Cmaj7")

    print("\nReharmonized with upper structures:")

    # Dm7 stays the same
    print("\n1. Dm7 (no reharmonization needed)")

    # G7 with various upper structures
    print("\n2. G7 → Multiple upper structure options:")

    structures = [
        ("maj_#11", "G7#11 (Lydian dominant - bright, modal jazz)"),
        ("maj_b9", "G7b9 (Diminished scale - dark, tense)"),
        ("maj_#9", "G7#9 (Altered - modern jazz)"),
    ]

    for structure_type, description in structures:
        chord = harmony.create_upper_structure(7, structure_type, octave=4)
        print(f"   {chord} - {description}")
        print(f"   Voicing: {chord.voicing}")

    # Cmaj7 with upper structure for color
    print("\n3. Cmaj7 (tonic)")

    print("\nComplete reharmonized progression:")
    print("  Dm7 - G7#11 - Cmaj7")
    print("  (Modal, bright sound - McCoy Tyner style)")


def example_contemporary_classical():
    """
    Example 2: Contemporary Classical - Polychords and Clusters

    Create atmospheric textures using polychords and clusters
    (Bartók, Ligeti, Messiaen style).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Contemporary Classical Textures")
    print("=" * 70)

    harmony = ExtendedHarmony()

    print("\n1. POLYCHORDS (Bitonality)")
    print("-" * 70)

    # Petrushka chord (Stravinsky)
    petrushka = harmony.create_polychord(0, "maj", 6, "maj")
    print(f"\nPetrushka Chord (Stravinsky): {petrushka}")
    print(f"  C major over F# major (tritone relation)")
    print(f"  Combined voicing: {petrushka.combined_voicing}")
    print(f"  Use: Ballet music, dramatic scenes")

    # Bartók-style polychord
    bartok = harmony.create_polychord(0, "maj", 3, "min")
    print(f"\nBartók-style polychord: {bartok}")
    print(f"  Use: Dark, mysterious atmosphere")

    print("\n2. CLUSTER VOICINGS")
    print("-" * 70)

    # Ligeti-style chromatic cluster
    chromatic = harmony.create_cluster(60, ClusterType.CHROMATIC, 6)
    print(f"\nChromatic cluster (Ligeti): {chromatic}")
    print(f"  Use: Dense atmospheric textures, suspense")

    # Bartók-style diatonic cluster
    diatonic = harmony.create_cluster(60, ClusterType.DIATONIC, 5)
    print(f"\nDiatonic cluster (Bartók): {diatonic}")
    print(f"  Use: Folk-inspired modernism")

    # Messiaen-style whole-tone cluster
    wholetone = harmony.create_cluster(60, ClusterType.WHOLE_TONE, 4)
    print(f"\nWhole-tone cluster (Messiaen): {wholetone}")
    print(f"  Use: Dreamlike, floating quality")


def example_neo_soul_harmony():
    """
    Example 3: Neo-Soul Chord Progressions

    Create lush neo-soul harmony using polychords, slash chords,
    and extended voicings (D'Angelo, Robert Glasper, Erykah Badu style).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Neo-Soul Harmony")
    print("=" * 70)

    harmony = ExtendedHarmony()

    print("\nNeo-soul progression with rich voicings:")
    print("-" * 70)

    # 1. Cmaj7/E (slash chord for smooth bass)
    cmaj7_e = harmony.create_slash_chord(0, "maj7", 4, octave=3)
    print(f"\n1. {cmaj7_e}")
    print(f"   Voicing: {cmaj7_e.voicing}")
    print(f"   Use: Smooth first inversion, E in bass")

    # 2. Polychord: Dm9 over Cmaj7 sound
    poly = harmony.create_polychord(2, "min7", 0, "maj7", octave=3)
    print(f"\n2. {poly}")
    print(f"   Use: Lush, ambiguous harmony")

    # 3. Altered dominant
    g7alt = harmony.create_altered_dominant(7, ["b9", "#11"], octave=3)
    print(f"\n3. {g7alt}")
    print(f"   Voicing: {g7alt.voicing}")
    print(f"   Use: Tension before resolution")

    # 4. Slash chord resolution
    cmaj9_g = harmony.create_slash_chord(0, "maj7", 7, octave=3, extensions=["9"])
    print(f"\n4. {cmaj9_g}")
    print(f"   Use: Suspended resolution, G pedal tone")

    print("\nComplete progression:")
    print("  Cmaj7/E - Dm/C - G7#11 - Cmaj9/G")
    print("  (D'Angelo/Robert Glasper style)")


def example_modal_jazz():
    """
    Example 4: Modal Jazz - Quartal Voicings

    Create McCoy Tyner-style quartal voicings for modal jazz.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Modal Jazz - Quartal Voicings")
    print("=" * 70)

    harmony = ExtendedHarmony()

    print("\nMcCoy Tyner-style quartal clusters:")
    print("-" * 70)

    # Quartal voicings in different positions
    positions = [60, 62, 64, 65, 67]  # C, D, E, F, G
    note_names = ["C", "D", "E", "F", "G"]

    for pos, name in zip(positions, note_names):
        cluster = harmony.create_cluster(pos, ClusterType.QUARTAL, 4)
        print(f"\n{name} Dorian quartal voicing: {cluster}")

    print("\nUse over modal vamps:")
    print("  Dm7 vamp: Use D, E, F, G quartal voicings")
    print("  Creates open, spacious sound characteristic of modal jazz")


def example_film_scoring():
    """
    Example 5: Film Scoring - Creating Mood with Extended Harmony

    Use extended harmony techniques to create specific moods for film.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Film Scoring - Extended Harmony for Mood")
    print("=" * 70)

    harmony = ExtendedHarmony()

    print("\n1. SUSPENSE / TENSION")
    print("-" * 70)

    # Chromatic cluster for suspense
    suspense = harmony.create_cluster(48, ClusterType.CHROMATIC, 7, span_semitones=10)
    print(f"Low chromatic cluster: {suspense}")
    print("Use: Building tension, horror, uncertainty")

    # Tritone polychord
    tritone_poly = harmony.create_polychord(0, "min", 6, "min")
    print(f"\nTritone polychord: {tritone_poly}")
    print("Use: Maximum dissonance, dramatic tension")

    print("\n2. MYSTERY / AMBIGUITY")
    print("-" * 70)

    # Whole-tone cluster
    mystery = harmony.create_cluster(60, ClusterType.WHOLE_TONE, 5)
    print(f"Whole-tone cluster: {mystery}")
    print("Use: Dreamlike, mysterious, floating quality")

    # Multi-tonic progression
    progression = [
        harmony.create_slash_chord(0, "min", 0),
        harmony.create_slash_chord(6, "maj", 6),
        harmony.create_slash_chord(3, "min", 3),
        harmony.create_slash_chord(9, "maj", 9),
    ]
    analysis = harmony.analyze_multitonic_system(progression)
    print(f"\nAmbiguous tonal progression:")
    print(f"  Ambiguity score: {analysis.ambiguity_score:.2f}")
    print("Use: Uncertainty, shifting perspectives")

    print("\n3. BEAUTY / LUSH HARMONY")
    print("-" * 70)

    # Upper structure for lush sound
    lush = harmony.create_upper_structure(7, "maj_#11", octave=4)
    print(f"Lydian dominant (lush): {lush}")
    print("Use: Beautiful, open, modern romantic sound")

    # Pentatonic cluster
    beauty = harmony.create_cluster(64, ClusterType.PENTATONIC, 5)
    print(f"\nPentatonic cluster: {beauty}")
    print("Use: Consonant density, emotional warmth")


def example_jazz_standards_reharmonization():
    """
    Example 6: Reharmonizing Jazz Standards

    Take a standard progression and add sophisticated reharmonization.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Reharmonizing Jazz Standards")
    print("=" * 70)

    harmony = ExtendedHarmony()

    print("\nOriginal: 'Autumn Leaves' excerpt (Am - D7 - Gmaj7)")
    print("-" * 70)

    print("\nBasic voicings:")
    print("  Am7 - D7 - Gmaj7")

    print("\nAdvanced reharmonization options:")
    print("-" * 70)

    print("\nOption 1: Upper structure triads")
    d7_sharp11 = harmony.create_upper_structure(2, "maj_#11")
    print(f"  Am7 - {d7_sharp11} - Gmaj7")
    print(f"  (Bright, modal sound)")

    print("\nOption 2: Altered dominant")
    d7alt = harmony.create_altered_dominant(2, ["b9", "#9", "b13"])
    print(f"  Am7 - {d7alt} - Gmaj7")
    print(f"  (Darker, more tension)")

    print("\nOption 3: Tritone substitution with upper structure")
    ab7 = harmony.create_upper_structure(8, "maj_#11")  # Ab7 (tritone sub for D7)
    print(f"  Am7 - {ab7} - Gmaj7")
    print(f"  (Sophisticated, chromatic bass motion: A-Ab-G)")

    print("\nOption 4: Polychord approach")
    poly = harmony.create_polychord(0, "maj", 2, "dom", octave=3)
    print(f"  Am7 - {poly} - Gmaj7")
    print(f"  (Modern, complex harmony)")


def example_integration_with_midi():
    """
    Example 7: Integration with MIDI - Exporting Voicings

    Show how to extract MIDI notes for use in MIDI files.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: MIDI Integration")
    print("=" * 70)

    harmony = ExtendedHarmony()

    print("\nExtracting MIDI notes for file export:")
    print("-" * 70)

    # Create various chords
    chords = [
        ("Upper Structure", harmony.create_upper_structure(7, "maj_#11")),
        ("Polychord", harmony.create_polychord(0, "maj7", 2, "min").upper_chord),
        ("Slash Chord", harmony.create_slash_chord(0, "maj7", 4)),
        ("Altered Dominant", harmony.create_altered_dominant(7, ["b9", "#11"])),
    ]

    for name, chord in chords:
        midi_notes = harmony.chord_to_midi_notes(chord)
        chord_name = harmony.get_chord_name(chord)

        print(f"\n{name}: {chord_name}")
        print(f"  MIDI notes: {midi_notes}")
        print(f"  Note range: {min(midi_notes)} - {max(midi_notes)}")
        print(f"  Number of voices: {len(midi_notes)}")

    print("\n  → These MIDI note numbers can be used with:")
    print("     - mido library for MIDI file creation")
    print("     - pretty_midi for music analysis")
    print("     - Direct DAW import via MIDI")


def example_transposition():
    """
    Example 8: Transposition - Moving Harmony to Different Keys

    Demonstrate transposing complex harmonies to different keys.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Transposition to Different Keys")
    print("=" * 70)

    harmony = ExtendedHarmony()

    # Create a complex chord
    original = harmony.create_upper_structure(7, "maj_#11", octave=4)
    print(f"\nOriginal chord: {original}")
    print(f"Voicing: {original.voicing}")

    print("\nTransposed to different keys:")
    print("-" * 70)

    transpositions = [
        (0, "Same key (G)"),
        (1, "Half step up (Ab)"),
        (2, "Whole step up (A)"),
        (5, "Perfect fourth up (C)"),
        (-2, "Whole step down (F)"),
    ]

    for semitones, description in transpositions:
        transposed = harmony.transpose_chord(original, semitones)
        print(f"\n{description}: {transposed}")
        print(f"  Voicing: {transposed.voicing}")


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_all_examples():
    """Run all usage examples"""
    print("=" * 70)
    print("EXTENDED HARMONY MODULE - Comprehensive Usage Examples")
    print("=" * 70)
    print("\nDemonstrating all features in real musical contexts:")
    print("1. Jazz Reharmonization")
    print("2. Contemporary Classical")
    print("3. Neo-Soul Harmony")
    print("4. Modal Jazz")
    print("5. Film Scoring")
    print("6. Jazz Standards Reharmonization")
    print("7. MIDI Integration")
    print("8. Transposition")

    example_jazz_reharmonization()
    example_contemporary_classical()
    example_neo_soul_harmony()
    example_modal_jazz()
    example_film_scoring()
    example_jazz_standards_reharmonization()
    example_integration_with_midi()
    example_transposition()

    print("\n" + "=" * 70)
    print("All examples complete!")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - Module: advanced_modules/extended_harmony.py")
    print("  - Tests: advanced_modules/test_extended_harmony.py")
    print("  - Documentation: In module docstrings")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
