#!/usr/bin/env python3
"""
Count Basie Style Example
==========================

This example demonstrates how to use the Basie style arranger to create
authentic Count Basie-style big band arrangements.

**What This Demo Does:**
1. Creates a simple 12-bar blues progression
2. Generates a blues melody
3. Arranges it in Basie style
4. Exports to MIDI file

**Expected Output:**
- Sparse piano comping (Basie minimalism)
- Punchy brass and sax hits
- Walking bass
- Swing drums with feathered kick
- Freddie Green style rhythm guitar
- Blues-based riffs

Author: Agent 14 - Count Basie Style Analyzer
Date: 2025
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.midi_analyzer import NoteEvent, ChordEvent
from styles.basie_arranger import BasieArranger, BasieRiffGenerator
from styles.basie_profile import (
    BASIE_STYLE,
    BASIE_EARLY_KANSAS_CITY,
    BASIE_1950s_ATOMIC,
    BASIE_BALLAD,
    get_basie_style_for_context
)


def create_12_bar_blues_progression() -> list[ChordEvent]:
    """
    Create a 12-bar blues progression in F.

    Standard blues:
    | F7  | F7  | F7  | F7  |
    | Bb7 | Bb7 | F7  | F7  |
    | C7  | Bb7 | F7  | C7  |
    """
    chords = []
    bar_duration = 4.0  # 4 beats per bar

    # Bars 1-4: F7
    for bar in range(4):
        chords.append(ChordEvent(
            start_time=bar * bar_duration,
            duration=bar_duration,
            root=5,  # F
            quality='dom7',
            pitches=[5, 9, 0, 3],  # F A C Eb
            bass_note=5,
            confidence=1.0
        ))

    # Bars 5-6: Bb7
    for bar in range(4, 6):
        chords.append(ChordEvent(
            start_time=bar * bar_duration,
            duration=bar_duration,
            root=10,  # Bb
            quality='dom7',
            pitches=[10, 2, 5, 8],  # Bb D F Ab
            bass_note=10,
            confidence=1.0
        ))

    # Bars 7-8: F7
    for bar in range(6, 8):
        chords.append(ChordEvent(
            start_time=bar * bar_duration,
            duration=bar_duration,
            root=5,  # F
            quality='dom7',
            pitches=[5, 9, 0, 3],
            bass_note=5,
            confidence=1.0
        ))

    # Bar 9: C7
    chords.append(ChordEvent(
        start_time=8 * bar_duration,
        duration=bar_duration,
        root=0,  # C
        quality='dom7',
        pitches=[0, 4, 7, 10],  # C E G Bb
        bass_note=0,
        confidence=1.0
    ))

    # Bar 10: Bb7
    chords.append(ChordEvent(
        start_time=9 * bar_duration,
        duration=bar_duration,
        root=10,  # Bb
        quality='dom7',
        pitches=[10, 2, 5, 8],
        bass_note=10,
        confidence=1.0
    ))

    # Bar 11: F7
    chords.append(ChordEvent(
        start_time=10 * bar_duration,
        duration=bar_duration,
        root=5,  # F
        quality='dom7',
        pitches=[5, 9, 0, 3],
        bass_note=5,
        confidence=1.0
    ))

    # Bar 12: C7 (turnaround)
    chords.append(ChordEvent(
        start_time=11 * bar_duration,
        duration=bar_duration,
        root=0,  # C
        quality='dom7',
        pitches=[0, 4, 7, 10],
        bass_note=0,
        confidence=1.0
    ))

    return chords


def create_simple_blues_melody() -> list[NoteEvent]:
    """Create a simple blues melody over 12 bars."""
    melody = []

    # Simple blues lick pattern
    notes_pattern = [
        (0, 65, 1.0),    # F
        (1, 68, 0.5),    # Ab (blue note)
        (1.5, 70, 0.5),  # Bb
        (2, 72, 1.0),    # C
        (3, 70, 1.0),    # Bb
        (4, 65, 2.0),    # F (held)
        # Repeat with variation
        (6, 68, 0.5),
        (6.5, 70, 0.5),
        (7, 72, 1.0),
        (8, 73, 0.5),
        (8.5, 72, 0.5),
        (9, 70, 1.0),
        (10, 68, 1.0),
        (11, 65, 1.0),
    ]

    for start_beat, pitch, duration in notes_pattern:
        melody.append(NoteEvent(
            start_time=start_beat,
            duration=duration,
            start_tick=int(start_beat * 480),
            duration_ticks=int(duration * 480),
            pitch=pitch,
            velocity=85,
            channel=0,
            track_idx=0
        ))

    return melody


def demo_basie_arrangement(style_name: str = "standard"):
    """
    Demonstrate Basie-style arrangement.

    Args:
        style_name: 'standard', 'early', 'atomic', 'ballad'
    """
    print("=" * 80)
    print(f"COUNT BASIE STYLE ARRANGEMENT DEMO - {style_name.upper()}")
    print("=" * 80)
    print()

    # Create blues progression
    print("Creating 12-bar blues progression in F...")
    chords = create_12_bar_blues_progression()
    print(f"✓ Created {len(chords)} chords")
    print()

    # Create melody
    print("Creating blues melody...")
    melody = create_simple_blues_melody()
    print(f"✓ Created {len(melody)} melody notes")
    print()

    # Get appropriate style
    print(f"Loading Basie style profile: {style_name}...")
    style_config = get_basie_style_for_context(style_name)
    print(f"✓ Style loaded")
    print(f"  - Piano density: {style_config.piano_density * 100}%")
    print(f"  - Section hits: {style_config.use_section_hits * 100}%")
    print(f"  - Riff-based: {style_config.use_riffs * 100}%")
    print(f"  - Blues influence: {style_config.blues_influence * 100}%")
    print()

    # Create arrangement
    print("Arranging in Basie style...")
    arrangement = BasieArranger.arrange_in_basie_style(
        melody=melody,
        chords=chords,
        style_config=style_config
    )
    print(f"✓ Arrangement complete!")
    print()

    # Show arrangement details
    print("ARRANGEMENT SECTIONS:")
    print("-" * 80)
    for section, notes in arrangement.items():
        print(f"  {section:20s}: {len(notes):4d} notes")
    print()

    # Analyze Basie characteristics
    print("BASIE CHARACTERISTICS:")
    print("-" * 80)

    piano_notes = len(arrangement.get('piano', []))
    print(f"  Piano sparseness: {piano_notes} notes (Basie minimalism)")

    brass_hits = len([n for n in arrangement.get('brass_section', [])
                     if n.duration < 0.3])
    print(f"  Brass hits: {brass_hits} (punchy section hits)")

    has_guitar = 'guitar' in arrangement and len(arrangement['guitar']) > 0
    print(f"  Freddie Green guitar: {'Yes' if has_guitar else 'No'}")

    # Check for feathered kick in drums
    feathered_kicks = len([n for n in arrangement.get('drums', [])
                          if n.pitch == 36 and n.velocity < 50])
    print(f"  Feathered bass drum: {feathered_kicks} soft kick hits")

    print()
    print("=" * 80)
    print(f"Basie '{style_name}' arrangement demonstration complete!")
    print("=" * 80)
    print()

    return arrangement


def demo_basie_riff_generator():
    """Demonstrate the Basie riff generator."""
    print("=" * 80)
    print("BASIE RIFF GENERATOR DEMO")
    print("=" * 80)
    print()

    # Create a chord
    chord = ChordEvent(
        start_time=0.0,
        duration=8.0,
        root=5,  # F
        quality='dom7',
        pitches=[5, 9, 0, 3],
        bass_note=5,
        confidence=1.0
    )

    # Generate different riff types
    riff_types = ["blues", "call_response", "shout"]

    for riff_type in riff_types:
        print(f"Generating {riff_type} riff...")
        riff = BasieRiffGenerator.generate_basie_riff(
            chord=chord,
            bars=2,
            riff_style=riff_type,
            section="brass"
        )
        print(f"  ✓ Generated {len(riff)} notes")
        print(f"    Rhythm pattern: {[n.start_time for n in riff[:6]]}")
        print()

    # Generate section hit
    print("Generating punchy section hit...")
    hit = BasieRiffGenerator.generate_section_hit(
        chord=chord,
        time=0.0,
        voicing="basic_chord",
        duration=0.25
    )
    print(f"  ✓ Generated {len(hit)} note hit")
    print()

    print("=" * 80)
    print()


def demo_style_comparison():
    """Compare different Basie style variants."""
    print("=" * 80)
    print("BASIE STYLE VARIANTS COMPARISON")
    print("=" * 80)
    print()

    variants = {
        'standard': BASIE_STYLE,
        'early': BASIE_EARLY_KANSAS_CITY,
        'atomic': BASIE_1950s_ATOMIC,
        'ballad': BASIE_BALLAD,
    }

    print(f"{'Variant':<15} {'Piano':<10} {'Hits':<10} {'Blues':<10} {'Tempo':<15}")
    print("-" * 80)

    for name, style in variants.items():
        print(f"{name:<15} "
              f"{style.piano_density*100:>5.0f}%     "
              f"{style.use_section_hits*100:>5.0f}%     "
              f"{style.blues_influence*100:>5.0f}%     "
              f"{style.tempo_preference:<15}")

    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    print()
    print("COUNT BASIE STYLE ARRANGER - EXAMPLES")
    print("=" * 80)
    print()
    print("This demonstrates Agent 14's implementation of Count Basie's")
    print("legendary big band arranging style.")
    print()
    print("=" * 80)
    print()

    # Demo 1: Standard Basie arrangement
    demo_basie_arrangement("standard")

    # Demo 2: Riff generator
    demo_basie_riff_generator()

    # Demo 3: Style comparison
    demo_style_comparison()

    print()
    print("=" * 80)
    print("All demonstrations complete!")
    print()
    print("To create your own Basie-style arrangement:")
    print()
    print("  from styles.basie_arranger import BasieArranger")
    print("  from styles.basie_profile import BASIE_STYLE")
    print()
    print("  arrangement = BasieArranger.arrange_in_basie_style(")
    print("      melody=your_melody,")
    print("      chords=your_chords,")
    print("      style_config=BASIE_STYLE")
    print("  )")
    print()
    print("=" * 80)
