#!/usr/bin/env python3
"""
Big Band Articulation Demo - Agent 8
====================================

Demonstrates the Big Band Articulation Engine with pitch bend encoding.

Examples:
---------
1. All articulation types with MIDI export
2. Style-specific profiles (Ellington vs Basie vs Modern)
3. Automatic articulation suggestion
4. Integration with big band arrangement

Usage:
------
    python big_band_articulation_demo.py

Output:
-------
- MIDI files with different articulations
- Comparison of Ellington vs Basie styles
- Validation metrics

Author: Agent 8
Date: 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataclasses import dataclass
from typing import List

# Import articulation modules
from midi_generator.transformation.big_band_articulation import (
    BigBandArticulationEngine,
    BigBandArticulationType,
    ELLINGTON_PROFILE,
    BASIE_PROFILE,
    MODERN_PROFILE,
    STYLE_PROFILES
)

from midi_generator.transformation.articulation_midi_export import (
    ArticulationMIDIExporter,
    apply_style_articulations
)


# Simple JazzNote for testing
@dataclass
class JazzNote:
    """Simplified JazzNote for demo."""
    pitch: int
    velocity: int
    start_time: float
    duration: float
    articulation: str = "normal"
    channel: int = 0


def demo_1_all_articulations():
    """Demo 1: Test all pitch bend articulations."""
    print("\n" + "=" * 80)
    print("DEMO 1: All Pitch Bend Articulations")
    print("=" * 80)

    # Create test phrase with different articulations
    articulation_types = [
        ("normal", BigBandArticulationType.NORMAL),
        ("fall_short", BigBandArticulationType.FALL_SHORT),
        ("fall_long", BigBandArticulationType.FALL_LONG),
        ("doit", BigBandArticulationType.DOIT),
        ("rip", BigBandArticulationType.RIP),
        ("scoop", BigBandArticulationType.SCOOP),
        ("shake", BigBandArticulationType.SHAKE),
        ("growl", BigBandArticulationType.GROWL),
    ]

    engine = BigBandArticulationEngine(ticks_per_beat=480, tempo_bpm=120)

    print("\nTesting each articulation type:")
    print("-" * 80)

    for artic_name, artic_type in articulation_types:
        # Single note
        notes = [60]
        durations = [2.0]
        velocities = [80]
        start_times = [0.0]

        result = engine.apply_articulation(
            notes, durations, velocities, start_times, artic_type
        )

        print(f"\n{artic_name.upper()}:")
        print(f"  Duration: {result.durations[0]:.2f} beats (original: 2.00)")
        print(f"  Velocity: {result.velocities[0]} (original: 80)")
        print(f"  Pitch bends: {len(result.pitch_bends)} messages")

        if result.pitch_bends:
            print(f"  First bend: {result.pitch_bends[0].to_cents():.1f} cents @ tick {result.pitch_bends[0].time_ticks}")
            print(f"  Last bend: {result.pitch_bends[-1].to_cents():.1f} cents @ tick {result.pitch_bends[-1].time_ticks}")

        if result.cc_messages:
            print(f"  CC messages: {len(result.cc_messages)}")

    # Export all to MIDI
    print("\n" + "-" * 80)
    print("Exporting to MIDI files...")

    exporter = ArticulationMIDIExporter(tempo_bpm=120)

    for artic_name, artic_type in articulation_types:
        notes = [
            JazzNote(60, 80, 0.0, 2.0, articulation=artic_name),
            JazzNote(64, 85, 2.5, 2.0, articulation=artic_name),
            JazzNote(67, 90, 5.0, 2.0, articulation=artic_name),
            JazzNote(72, 95, 7.5, 3.0, articulation=artic_name),
        ]

        filename = f"articulation_{artic_name}.mid"
        exporter.export_jazz_notes_to_midi(notes, filename, f"Brass - {artic_name}")

    print("\n✓ Created MIDI files for all articulation types")


def demo_2_style_comparison():
    """Demo 2: Compare Ellington, Basie, and Modern styles."""
    print("\n" + "=" * 80)
    print("DEMO 2: Style Comparison - Ellington vs Basie vs Modern")
    print("=" * 80)

    # Same phrase with different styles
    base_notes = [
        JazzNote(60, 80, 0.0, 1.0),
        JazzNote(64, 85, 1.0, 1.0),
        JazzNote(67, 90, 2.0, 1.0),
        JazzNote(72, 95, 3.0, 1.0),  # Phrase ending
        JazzNote(71, 90, 4.0, 2.0),  # Sustained
        JazzNote(69, 85, 6.0, 0.5),
        JazzNote(67, 80, 6.5, 0.5),
        JazzNote(64, 75, 7.0, 1.0),  # Final note
    ]

    styles = ["ellington", "basie", "modern"]
    exporter = ArticulationMIDIExporter(tempo_bpm=140)

    for style in styles:
        print(f"\n{style.upper()} STYLE:")
        print("-" * 80)

        # Copy notes
        import copy
        notes = copy.deepcopy(base_notes)

        # Apply style
        notes = apply_style_articulations(notes, style=style)

        # Show articulations
        print("Articulation assignments:")
        for i, note in enumerate(notes):
            print(f"  Note {i+1} (pitch {note.pitch}, dur {note.duration:.1f}): {note.articulation}")

        # Export
        filename = f"style_{style}.mid"
        exporter.export_jazz_notes_to_midi(notes, filename, f"{style.capitalize()} Style")

    # Print style profiles
    print("\n" + "=" * 80)
    print("STYLE PROFILES:")
    print("=" * 80)

    for style_name, profile in STYLE_PROFILES.items():
        print(f"\n{profile.style_name}:")
        print(f"  {profile.description}")
        print(f"  Fall probability: {profile.fall_probability:.0%}")
        print(f"  Shake probability: {profile.shake_probability:.0%}")
        print(f"  Growl probability: {profile.growl_probability:.0%}")
        print(f"  Staccato probability: {profile.staccato_probability:.0%}")
        print(f"  Phrase ending: {profile.phrase_ending_articulation.value}")
        print(f"  Sustained notes: {profile.sustained_note_articulation.value}")

    print("\n✓ Created style comparison MIDI files")


def demo_3_articulation_suggestions():
    """Demo 3: Automatic articulation suggestions based on context."""
    print("\n" + "=" * 80)
    print("DEMO 3: Automatic Articulation Suggestions")
    print("=" * 80)

    engine = BigBandArticulationEngine()

    contexts = [
        "phrase_ending",
        "sustained",
        "section_hit",
        "background",
        "shout_chorus"
    ]

    styles = ["ellington", "basie", "modern"]

    print("\nArticulation suggestions by context and style:")
    print("-" * 80)
    print(f"{'Context':<20} {'Ellington':<20} {'Basie':<20} {'Modern':<20}")
    print("-" * 80)

    for context in contexts:
        row = [context]
        for style in styles:
            suggested = engine.suggest_articulation(context, style)
            row.append(suggested.value)
        print(f"{row[0]:<20} {row[1]:<20} {row[2]:<20} {row[3]:<20}")

    print()


def demo_4_phrase_with_dynamics():
    """Demo 4: Complete phrase with articulations and dynamics."""
    print("\n" + "=" * 80)
    print("DEMO 4: Complete Phrase with Articulations and Dynamics")
    print("=" * 80)

    # 8-bar phrase in Ellington style
    notes = [
        # Bar 1-2: Opening phrase
        JazzNote(60, 70, 0.0, 1.0, "normal"),
        JazzNote(62, 73, 1.0, 1.0, "normal"),
        JazzNote(64, 76, 2.0, 1.0, "accent"),
        JazzNote(65, 79, 3.0, 1.0, "fall_short"),

        # Bar 3-4: Build
        JazzNote(67, 82, 4.0, 1.0, "normal"),
        JazzNote(69, 85, 5.0, 1.0, "scoop"),
        JazzNote(71, 88, 6.0, 1.0, "accent"),
        JazzNote(72, 91, 7.0, 1.0, "shake"),  # Sustained high note

        # Bar 5-6: Peak
        JazzNote(74, 94, 8.0, 1.0, "rip"),  # Rip into climax
        JazzNote(76, 97, 9.0, 2.0, "shake"),  # Long shake
        JazzNote(74, 94, 11.0, 1.0, "normal"),

        # Bar 7-8: Resolution
        JazzNote(72, 91, 12.0, 1.0, "normal"),
        JazzNote(69, 85, 13.0, 1.0, "normal"),
        JazzNote(67, 79, 14.0, 1.0, "normal"),
        JazzNote(64, 70, 15.0, 1.0, "fall_long"),  # Final fall
    ]

    exporter = ArticulationMIDIExporter(tempo_bpm=120)
    exporter.export_jazz_notes_to_midi(
        notes,
        "complete_phrase_ellington.mid",
        "Ellington Phrase"
    )

    print("\n✓ Created complete 8-bar phrase with Ellington-style articulations")
    print("  - Opening: normal, light articulations")
    print("  - Build: scoops, accents")
    print("  - Peak: rip entry, shakes on sustained notes")
    print("  - Resolution: final fall")


def demo_5_validation():
    """Demo 5: Validation metrics."""
    print("\n" + "=" * 80)
    print("DEMO 5: Validation Metrics")
    print("=" * 80)

    engine = BigBandArticulationEngine(ticks_per_beat=480, tempo_bpm=120)

    # Test pitch bend accuracy
    print("\nPitch Bend Accuracy Test:")
    print("-" * 80)

    test_cases = [
        (BigBandArticulationType.FALL_SHORT, -200, 300),  # Target: -200 cents, 300ms
        (BigBandArticulationType.FALL_LONG, -400, 600),   # Target: -400 cents, 600ms
        (BigBandArticulationType.DOIT, 200, 200),         # Target: +200 cents, 200ms
        (BigBandArticulationType.RIP, -1200, 400),        # Target: -1200→0 cents, 400ms
    ]

    for artic, expected_cents, expected_duration_ms in test_cases:
        result = engine.apply_articulation(
            [60], [2.0], [80], [0.0], artic
        )

        if result.pitch_bends:
            first_bend = result.pitch_bends[0].to_cents()
            last_bend = result.pitch_bends[-1].to_cents()

            # For falls/doits, check if we reach target
            if artic == BigBandArticulationType.RIP:
                actual_range = abs(first_bend - last_bend)
                target_range = 1200  # One octave
                accuracy = (1 - abs(actual_range - target_range) / target_range) * 100
            else:
                actual_cents = last_bend
                accuracy = (1 - abs(actual_cents - expected_cents) / abs(expected_cents)) * 100 if expected_cents != 0 else 100

            print(f"\n{artic.value}:")
            print(f"  Expected: {expected_cents} cents")
            print(f"  Actual: {last_bend:.1f} cents")
            print(f"  Accuracy: {accuracy:.1f}%")
            print(f"  Messages: {len(result.pitch_bends)}")

    # Test duration modifications
    print("\n" + "-" * 80)
    print("Duration Modification Test:")
    print("-" * 80)

    for artic_type in [
        BigBandArticulationType.STACCATO,
        BigBandArticulationType.FALL_SHORT,
        BigBandArticulationType.RIP,
        BigBandArticulationType.SHAKE
    ]:
        result = engine.apply_articulation(
            [60], [2.0], [80], [0.0], artic_type
        )
        multiplier = result.durations[0] / 2.0
        print(f"  {artic_type.value}: {multiplier:.2f}x (duration: {result.durations[0]:.2f})")

    # Test velocity modifications
    print("\n" + "-" * 80)
    print("Velocity Modification Test:")
    print("-" * 80)

    for artic_type in [
        BigBandArticulationType.NORMAL,
        BigBandArticulationType.ACCENT,
        BigBandArticulationType.GHOST,
        BigBandArticulationType.RIP
    ]:
        result = engine.apply_articulation(
            [60], [2.0], [80], [0.0], artic_type
        )
        change = result.velocities[0] - 80
        print(f"  {artic_type.value}: {result.velocities[0]} ({change:+d} from base)")

    print("\n✓ Validation metrics completed")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("BIG BAND ARTICULATION ENGINE - COMPREHENSIVE DEMO")
    print("Agent 8: Articulation & Expression Engine")
    print("=" * 80)

    try:
        demo_1_all_articulations()
        demo_2_style_comparison()
        demo_3_articulation_suggestions()
        demo_4_phrase_with_dynamics()
        demo_5_validation()

        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nMIDI files created:")
        print("  - articulation_*.mid (all articulation types)")
        print("  - style_*.mid (Ellington, Basie, Modern comparison)")
        print("  - complete_phrase_ellington.mid (8-bar phrase)")
        print("\nOpen these files in your DAW to hear the results!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
