#!/usr/bin/env python3
"""
Classic Rock Generator - Comprehensive Demo

This demo showcases the capabilities of the classic_rock.py module,
demonstrating various rock sub-genres and musical features.

Demonstrates:
1. Classic Rock (60s-70s style)
2. Punk Rock (fast, simple, raw)
3. Alternative Rock (grunge, half-time feel)
4. Indie Rock (modern indie aesthetics)
5. Garage Rock (lo-fi, minimalist)
6. Post-Punk (angular, experimental)

Features shown:
- Chord progressions (I-IV-V, blues-based, modal)
- Power chord riffs
- Guitar techniques (bends, slides, vibrato)
- Drum patterns (rock beat, punk, half-time)
- Bass lines (root-fifth, walking, syncopated)
- Song structures (verse-chorus-solo-bridge)

Author: Agent 44 - Classic Rock Module
Date: 2025
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genres.classic_rock import (
    ClassicRockGenerator,
    RockStyle,
    RockScales,
    RockProgressions,
    GuitarLicks,
    RockDrums,
    RockBass,
    DrumStyle,
    BassPattern,
    create_classic_rock_song,
    SongStructure,
    GuitarTechnique
)


def demo_progressions():
    """Demo: Classic rock chord progressions"""
    print("\n" + "=" * 70)
    print("DEMO 1: Classic Rock Chord Progressions")
    print("=" * 70)

    progressions = {
        'I-IV-V (Classic)': 'i_iv_v',
        'I-bVII-IV (Sweet Child)': 'i_bvii_iv',
        'I-V-vi-IV (Ballad)': 'i_v_vi_iv',
        '12-Bar Blues': 'twelve_bar_blues',
        'Punk Three-Chord': 'punk_three_chord',
        'Modal Vamp (Post-Punk)': 'modal_vamp',
    }

    key_root = 69  # A
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    for name, prog_type in progressions.items():
        print(f"\n{name}:")
        progression = RockProgressions.get_progression(key_root, prog_type, 4)
        chord_names = []
        for root, quality, duration in progression:
            note_name = note_names[root % 12]
            chord_names.append(f"{note_name}{quality}")
        print(f"  {' - '.join(chord_names[:4])}")


def demo_scales():
    """Demo: Rock scales and modes"""
    print("\n" + "=" * 70)
    print("DEMO 2: Rock Scales and Modes")
    print("=" * 70)

    scales = {
        'Minor Pentatonic': 'minor_pentatonic',
        'Major Pentatonic': 'major_pentatonic',
        'Blues Scale': 'blues_scale',
        'Mixolydian (Classic Rock)': 'mixolydian',
        'Dorian (Alternative)': 'dorian',
        'Aeolian (Natural Minor)': 'aeolian',
    }

    root = 64  # E
    for name, scale_type in scales.items():
        notes = RockScales.get_notes(root, scale_type, octaves=1)
        intervals = [n - root for n in notes]
        print(f"\n{name}:")
        print(f"  Intervals: {intervals}")
        print(f"  Notes (MIDI): {notes}")


def demo_drum_patterns():
    """Demo: Drum patterns for different rock styles"""
    print("\n" + "=" * 70)
    print("DEMO 3: Drum Patterns")
    print("=" * 70)

    styles = [
        (DrumStyle.BASIC_ROCK, "Basic Rock Beat"),
        (DrumStyle.PUNK_BEAT, "Punk (Fast & Aggressive)"),
        (DrumStyle.HALF_TIME, "Half-Time (Grunge/Alt)"),
        (DrumStyle.FOUR_ON_FLOOR, "Four-on-Floor (Dance Rock)"),
        (DrumStyle.POST_PUNK, "Post-Punk (Angular)"),
        (DrumStyle.INDIE_SHUFFLE, "Indie Shuffle"),
    ]

    for drum_style, description in styles:
        pattern = RockDrums.generate_pattern(drum_style, bars=4)
        notes = RockDrums.pattern_to_notes(pattern)
        print(f"\n{description}:")
        print(f"  Total events: {len(notes)}")
        print(f"  Kick: {len(pattern.kick)}, Snare: {len(pattern.snare)}, Hi-hat: {len(pattern.hihat_closed)}")


def demo_bass_lines():
    """Demo: Bass line patterns"""
    print("\n" + "=" * 70)
    print("DEMO 4: Bass Line Patterns")
    print("=" * 70)

    # Test progression (A-D-E)
    progression = [
        (69, 'maj', 4.0),  # A
        (74, 'maj', 4.0),  # D
        (76, 'maj', 4.0),  # E
        (69, 'maj', 4.0),  # A
    ]

    patterns = [
        (BassPattern.ROOT_FIFTH, "Root-Fifth (Most Common)"),
        (BassPattern.OCTAVE_JUMP, "Octave Jumps (Energetic)"),
        (BassPattern.WALKING, "Walking Bass (Blues)"),
        (BassPattern.PEDAL_TONE, "Pedal Tone (Sustained)"),
        (BassPattern.SYNCOPATED, "Syncopated (Punk)"),
        (BassPattern.PENTATONIC_RIFF, "Pentatonic Riff (Melodic)"),
    ]

    for pattern_type, description in patterns:
        bass_line = RockBass.generate_bass_line(progression, pattern_type)
        print(f"\n{description}:")
        print(f"  Notes generated: {len(bass_line)}")
        print(f"  Pitch range: {min(n.pitch for n in bass_line)} - {max(n.pitch for n in bass_line)}")


def demo_guitar_techniques():
    """Demo: Guitar techniques and licks"""
    print("\n" + "=" * 70)
    print("DEMO 5: Guitar Techniques")
    print("=" * 70)

    print("\nPentatonic Lick (with bends and vibrato):")
    lick = GuitarLicks.generate_pentatonic_lick(64, 'minor_pentatonic', 8)
    print(f"  Total notes: {len(lick)}")
    bent_notes = [n for n in lick if n.bend_amount > 0]
    vibrato_notes = [n for n in lick if n.vibrato_depth > 0]
    print(f"  Notes with bends: {len(bent_notes)}")
    print(f"  Notes with vibrato: {len(vibrato_notes)}")

    if bent_notes:
        print(f"  Bend amounts: {[f'{n.bend_amount:.2f} semitones' for n in bent_notes[:3]]}")

    print("\nPower Chord Riff:")
    rhythm = [0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5]  # 4 beats
    power_riff = GuitarLicks.generate_power_chord_riff(57, rhythm, palm_mute=False)
    print(f"  Notes in riff: {len(power_riff)}")
    print(f"  Technique: Power Chords (root + fifth + octave)")

    print("\nPalm-Muted Power Chord Riff (Punk style):")
    palm_muted = GuitarLicks.generate_power_chord_riff(57, rhythm, palm_mute=True)
    print(f"  Notes in riff: {len(palm_muted)}")
    print(f"  Technique: Palm-muted power chords")
    print(f"  Avg velocity (muted): {sum(n.velocity for n in palm_muted) // len(palm_muted)}")


def demo_song_structures():
    """Demo: Song structures for different styles"""
    print("\n" + "=" * 70)
    print("DEMO 6: Song Structures")
    print("=" * 70)

    styles = [
        RockStyle.CLASSIC_ROCK,
        RockStyle.PUNK,
        RockStyle.ALTERNATIVE,
        RockStyle.INDIE,
        RockStyle.POST_PUNK,
    ]

    for style in styles:
        structure = SongStructure.generate_structure(style)
        total_bars = sum(s[1] for s in structure)
        sections = [f"{s[0]}({s[1]})" for s in structure]

        print(f"\n{style.value.replace('_', ' ').title()}:")
        print(f"  Sections: {' -> '.join(sections)}")
        print(f"  Total bars: {total_bars} (~{total_bars // 4} measures)")


def demo_complete_compositions():
    """Demo: Complete compositions in different styles"""
    print("\n" + "=" * 70)
    print("DEMO 7: Complete Compositions")
    print("=" * 70)

    compositions = [
        (RockStyle.CLASSIC_ROCK, 69, 120, "Classic Rock in A (120 BPM)"),
        (RockStyle.PUNK, 64, 180, "Punk Rock in E (180 BPM - Fast!)"),
        (RockStyle.ALTERNATIVE, 62, 95, "Alternative Rock in D (95 BPM - Grunge)"),
        (RockStyle.INDIE, 67, 110, "Indie Rock in G (110 BPM)"),
        (RockStyle.GARAGE, 64, 140, "Garage Rock in E (140 BPM - Raw)"),
        (RockStyle.POST_PUNK, 60, 105, "Post-Punk in C (105 BPM - Dark)"),
    ]

    for style, key_root, tempo, description in compositions:
        print(f"\n{description}:")
        gen = ClassicRockGenerator(style, key_root, tempo)
        comp = gen.generate_composition(length_bars=32, include_solo=(style == RockStyle.CLASSIC_ROCK))

        print(f"  Style: {comp['style']}")
        print(f"  Tempo: {comp['tempo']} BPM")
        print(f"  Key (MIDI): {comp['key']}")
        print(f"  Chords: {len(comp['chords'])}")
        print(f"  Drum events: {len(comp['drums'])}")
        print(f"  Bass notes: {len(comp['bass'])}")
        print(f"  Structure sections: {len(comp['structure'])}")
        print(f"  Has guitar solo: {'guitar_lead' in comp and len(comp.get('guitar_lead', [])) > 0}")


def demo_convenience_functions():
    """Demo: Convenience functions for quick song creation"""
    print("\n" + "=" * 70)
    print("DEMO 8: Convenience Functions")
    print("=" * 70)

    print("\nQuick song creation examples:")

    # Create various songs with convenience function
    songs = [
        ('A', 120, 'classic_rock', "Classic Rock Song in A"),
        ('E', 170, 'punk', "Punk Song in E (Fast!)"),
        ('D', 100, 'alternative', "Alternative Song in D"),
        ('G', 115, 'indie', "Indie Song in G"),
    ]

    for key, tempo, style, description in songs:
        song = create_classic_rock_song(key, tempo, style)
        print(f"\n{description}:")
        print(f"  Generated: {len(song['chords'])} chords, {len(song['drums'])} drum events")
        print(f"  Structure: {[s[0] for s in song['structure'][:4]]}...")


def demo_style_comparison():
    """Demo: Side-by-side comparison of different rock styles"""
    print("\n" + "=" * 70)
    print("DEMO 9: Style Comparison")
    print("=" * 70)

    print("\nComparing 6 rock sub-genres:")
    print("\nFeature            | Classic  | Punk     | Alt      | Indie    | Garage   | Post-Punk")
    print("-" * 90)

    styles_data = [
        ("Classic Rock", RockStyle.CLASSIC_ROCK, 120),
        ("Punk", RockStyle.PUNK, 180),
        ("Alternative", RockStyle.ALTERNATIVE, 95),
        ("Indie", RockStyle.INDIE, 110),
        ("Garage", RockStyle.GARAGE, 140),
        ("Post-Punk", RockStyle.POST_PUNK, 105),
    ]

    # Compare tempo
    tempos = []
    for name, style, default_tempo in styles_data:
        gen = ClassicRockGenerator(style, 60, default_tempo)
        tempos.append(f"{gen.tempo:>7}")
    print(f"Tempo (BPM)        | {' | '.join(tempos)}")

    # Compare structure length
    structure_bars = []
    for name, style, tempo in styles_data:
        structure = SongStructure.generate_structure(style)
        total = sum(s[1] for s in structure)
        structure_bars.append(f"{total:>7}")
    print(f"Structure (bars)   | {' | '.join(structure_bars)}")

    # Compare section count
    section_counts = []
    for name, style, tempo in styles_data:
        structure = SongStructure.generate_structure(style)
        section_counts.append(f"{len(structure):>7}")
    print(f"Sections (#)       | {' | '.join(section_counts)}")

    print("\nStyle characteristics:")
    characteristics = {
        "Classic Rock": "Extended solos, blues-based, dynamic",
        "Punk": "Fast tempo, 3-chord, short songs, raw energy",
        "Alternative": "Half-time feel, grunge aesthetic, introspective",
        "Indie": "Modern production, varied structures, experimental",
        "Garage": "Lo-fi, minimalist, back-to-basics approach",
        "Post-Punk": "Angular rhythms, unconventional structures, dark tone",
    }

    for style_name, char in characteristics.items():
        print(f"  {style_name:15s}: {char}")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("CLASSIC ROCK GENERATOR - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("\nShowcasing 6 rock sub-genres:")
    print("  1. Classic Rock (60s-70s: Led Zeppelin, Rolling Stones)")
    print("  2. Punk Rock (Ramones, Sex Pistols, The Clash)")
    print("  3. Alternative Rock (Nirvana, Radiohead, Pearl Jam)")
    print("  4. Indie Rock (Arctic Monkeys, The Strokes, Tame Impala)")
    print("  5. Garage Rock (The Black Keys, White Stripes)")
    print("  6. Post-Punk (Joy Division, The Cure, Talking Heads)")

    # Run all demos
    demo_progressions()
    demo_scales()
    demo_drum_patterns()
    demo_bass_lines()
    demo_guitar_techniques()
    demo_song_structures()
    demo_complete_compositions()
    demo_convenience_functions()
    demo_style_comparison()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nThe Classic Rock Generator provides comprehensive coverage of")
    print("non-metal rock genres with research-backed implementations.")
    print("\nKey features:")
    print("  - 6 distinct rock sub-genres")
    print("  - 6 chord progression types")
    print("  - 6 scale/mode systems")
    print("  - 6 drum pattern styles")
    print("  - 6 bass line patterns")
    print("  - Guitar techniques: bends, slides, vibrato, power chords")
    print("  - Complete song structures with verse-chorus-bridge-solo")
    print("\nAll implementations are based on research from:")
    print("  - 'The Guitar Handbook' - Ralph Denyer")
    print("  - 'How the Beatles Destroyed Rock 'n' Roll' - Elijah Wald")
    print("  - 'Our Band Could Be Your Life' - Michael Azerrad")
    print("  - 'Rip It Up and Start Again' - Simon Reynolds")
    print("  - Rolling Stone analysis of legendary guitarists")


if __name__ == "__main__":
    main()
