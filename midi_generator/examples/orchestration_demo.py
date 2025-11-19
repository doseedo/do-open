#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Orchestration Demo

Demonstrates the integration of all orchestration modules:
- Instrument Library
- Orchestrator (intelligent instrument selection)
- Texture Generator (accompaniment patterns)
- Articulation Engine (realistic articulations)

This example creates a complete orchestrated piece with:
- Melody (strings)
- Harmony (woodwinds)
- Bass line (cellos and basses)
- Accompaniment patterns (various textures)
- Realistic articulations

Author: Claude (Sonnet 4.5)
Created: 2025
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.instrument_library import (
    get_instrument, get_instruments_by_family, InstrumentFamily,
    midi_to_note_name
)
from generators.orchestrator import (
    Orchestrator, OrchestrationStyle, VoicePart, TextureType,
    create_string_section_voicing, get_orchestration_template
)
from generators.texture_generator import (
    TextureGenerator, AccompanimentPattern
)
from midi.articulation_engine import (
    ArticulationEngine, ArticulationType, create_expressive_phrase
)


def demo_1_simple_melody_orchestration():
    """Demo 1: Orchestrate a simple melody for strings"""
    print("\n" + "=" * 80)
    print("DEMO 1: Simple Melody Orchestration")
    print("=" * 80)

    # Create a simple melody (C major scale ascending)
    melody_notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
    melody_durations = [1.0] * 8
    melody_start_times = [i * 1.0 for i in range(8)]
    melody_velocities = [75] * 8

    # Create voice part
    melody_voice = VoicePart(
        notes=melody_notes,
        durations=melody_durations,
        start_times=melody_start_times,
        velocities=melody_velocities,
        texture_type=TextureType.MELODY,
        dynamic_level='mf'
    )

    # Create orchestrator (Romantic style)
    orchestrator = Orchestrator(style=OrchestrationStyle.ROMANTIC)

    # Orchestrate
    voicings = orchestrator.orchestrate([melody_voice])

    print(f"\nMelody orchestrated to {len(voicings)} instrument(s):")
    for voicing in voicings:
        inst = get_instrument(voicing.instrument_name)
        print(f"\n  {voicing.instrument_name}:")
        print(f"    Notes: {voicing.notes[:4]}... (first 4 of {len(voicing.notes)})")
        print(f"    Velocities: {voicing.velocities[:4]}...")
        print(f"    Family: {inst.family.value}")


def demo_2_texture_patterns():
    """Demo 2: Generate various accompaniment patterns"""
    print("\n" + "=" * 80)
    print("DEMO 2: Texture and Accompaniment Patterns")
    print("=" * 80)

    # Create texture generator
    generator = TextureGenerator(beats_per_bar=4, subdivision=4)

    # Chord: C major
    c_major = [60, 64, 67]
    bass_c = 48

    patterns = []

    # 1. Alberti bass
    print("\n1. Alberti Bass Pattern:")
    alberti = generator.generate_alberti_bass(c_major, num_bars=2)
    patterns.append(("Alberti Bass", alberti))
    print(f"   Generated {len(alberti.notes)} notes")
    print(f"   Pattern: {alberti.notes[:8]}")

    # 2. Arpeggiated
    print("\n2. Arpeggiated Pattern:")
    arp = generator.generate_arpeggiated(c_major, num_bars=1, notes_per_bar=8, direction="up")
    patterns.append(("Arpeggio", arp))
    print(f"   Generated {len(arp.notes)} notes")
    print(f"   Pattern: {arp.notes}")

    # 3. Waltz
    print("\n3. Waltz Pattern (3/4):")
    gen_3_4 = TextureGenerator(beats_per_bar=3, subdivision=4)
    waltz = gen_3_4.generate_waltz(c_major, bass_c, num_bars=2)
    patterns.append(("Waltz", waltz))
    print(f"   Generated {len(waltz.notes)} notes")

    # 4. Stride Piano
    print("\n4. Stride Piano:")
    stride = generator.generate_stride_piano(c_major, bass_c, num_bars=1)
    patterns.append(("Stride", stride))
    print(f"   Generated {len(stride.notes)} notes")

    # 5. Walking Bass
    print("\n5. Walking Bass:")
    c_major_scale = [0, 2, 4, 5, 7, 9, 11]
    walking = generator.generate_walking_bass(48, c_major_scale, num_bars=2)
    patterns.append(("Walking Bass", walking))
    print(f"   Generated {len(walking.notes)} notes")
    print(f"   Pattern: {walking.notes}")

    return patterns


def demo_3_articulations():
    """Demo 3: Apply articulations to notes"""
    print("\n" + "=" * 80)
    print("DEMO 3: Articulation Application")
    print("=" * 80)

    engine = ArticulationEngine()

    # Test notes
    notes = [60, 62, 64, 65, 67, 69, 71, 72]
    durations = [1.0] * 8
    velocities = [80] * 8

    print("\nOriginal:")
    print(f"  Durations: {durations[:4]}...")
    print(f"  Velocities: {velocities[:4]}...")

    # Test various articulations
    articulations = [
        ArticulationType.LEGATO,
        ArticulationType.STACCATO,
        ArticulationType.MARCATO,
        ArticulationType.PIZZICATO
    ]

    for artic in articulations:
        _, new_durs, new_vels = engine.apply_articulation(
            notes.copy(), durations.copy(), velocities.copy(), artic
        )
        print(f"\n{artic.value.upper()}:")
        print(f"  Durations: {[f'{d:.2f}' for d in new_durs[:4]]}...")
        print(f"  Velocities: {new_vels[:4]}...")

        # Get keyswitch if available
        ks = engine.get_keyswitch_note(artic)
        if ks:
            print(f"  Keyswitch: MIDI note {ks}")

    # Test expressive shaping
    print("\n\nExpressive Phrase Shaping:")
    phrase_types = ["crescendo", "diminuendo", "arch"]
    for phrase_type in phrase_types:
        _, _, shaped_vels = create_expressive_phrase(
            notes.copy(), durations.copy(), velocities.copy(), phrase_type
        )
        print(f"  {phrase_type.capitalize()}: {shaped_vels}")


def demo_4_full_orchestration():
    """Demo 4: Complete orchestration with all components"""
    print("\n" + "=" * 80)
    print("DEMO 4: Full Orchestration (Melody + Harmony + Bass)")
    print("=" * 80)

    # Create melody
    melody = [72, 74, 76, 77, 79, 77, 76, 74, 72]  # Simple descending melody
    melody_durations = [1.0] * len(melody)
    melody_times = [i * 1.0 for i in range(len(melody))]
    melody_velocities = [85] * len(melody)

    # Create harmony (chords)
    chords = [
        [60, 64, 67],  # C major
        [62, 65, 69],  # D minor
        [64, 67, 71],  # E minor
        [65, 69, 72],  # F major
        [67, 71, 74],  # G major
        [65, 69, 72],  # F major
        [64, 67, 71],  # E minor
        [62, 65, 69],  # D minor
        [60, 64, 67],  # C major
    ]

    # Create bass line
    bass = [48, 50, 52, 53, 55, 53, 52, 50, 48]  # Following chord roots

    # Create orchestrator
    orchestrator = Orchestrator(style=OrchestrationStyle.ROMANTIC)

    # Auto-arrange
    print("\nArranging melody, chords, and bass...")
    voicings = orchestrator.auto_arrange(
        melody=melody,
        chords=chords,
        bass=bass
    )

    print(f"\nCreated {len(voicings)} orchestral parts:")
    for voicing in voicings:
        inst = get_instrument(voicing.instrument_name)
        if inst:
            print(f"\n  {voicing.instrument_name} ({inst.family.value}):")
            print(f"    Notes: {len(voicing.notes)}")
            print(f"    Range: {min(voicing.notes)} - {max(voicing.notes)}")
            print(f"    Avg velocity: {sum(voicing.velocities) // len(voicing.velocities)}")

    # Apply articulations to each voicing
    print("\n\nApplying articulations:")
    engine = ArticulationEngine()

    for voicing in voicings:
        inst = get_instrument(voicing.instrument_name)
        if inst and inst.family == InstrumentFamily.STRINGS:
            # Strings: use legato for melody, spiccato for harmony
            artic = ArticulationType.LEGATO
            _, new_durs, new_vels = engine.apply_articulation(
                voicing.notes, voicing.durations, voicing.velocities, artic
            )
            voicing.durations = new_durs
            voicing.velocities = new_vels
            print(f"  {voicing.instrument_name}: {artic.value}")


def demo_5_string_section_voicing():
    """Demo 5: Create string section chord voicings"""
    print("\n" + "=" * 80)
    print("DEMO 5: String Section Chord Voicing")
    print("=" * 80)

    # Test different chords
    chords = [
        ([60, 64, 67], "C major"),
        ([60, 63, 67], "C minor"),
        ([62, 66, 69, 73], "D major 7"),
    ]

    styles = ["classical", "romantic", "modern"]

    for chord, chord_name in chords:
        print(f"\n{chord_name}: {[midi_to_note_name(n) for n in chord]}")

        for style in styles:
            voicing = create_string_section_voicing(chord, style=style)
            print(f"\n  {style.capitalize()} style:")
            for instrument, notes in voicing.items():
                note_names = [midi_to_note_name(n) for n in notes]
                print(f"    {instrument}: {note_names}")


def demo_6_instrument_database():
    """Demo 6: Explore instrument database"""
    print("\n" + "=" * 80)
    print("DEMO 6: Instrument Database")
    print("=" * 80)

    # Show all families
    print("\nInstrument Families:")
    for family in InstrumentFamily:
        instruments = get_instruments_by_family(family)
        if instruments:
            print(f"\n  {family.value.upper()}: {len(instruments)} instruments")
            for inst in instruments[:3]:  # Show first 3
                range_str = f"{midi_to_note_name(inst.range.lowest_note)} - " \
                           f"{midi_to_note_name(inst.range.highest_note)}"
                print(f"    • {inst.name}: {range_str}")

    # Show detailed info for Violin
    print("\n" + "-" * 80)
    print("Detailed: Violin")
    print("-" * 80)
    violin = get_instrument("Violin")
    if violin:
        print(f"Family: {violin.family.value}")
        print(f"MIDI Program: {violin.midi_program}")
        print(f"Range: {midi_to_note_name(violin.range.lowest_note)} - "
              f"{midi_to_note_name(violin.range.highest_note)}")
        print(f"Comfortable: {midi_to_note_name(violin.range.comfortable_low)} - "
              f"{midi_to_note_name(violin.range.comfortable_high)}")
        print(f"Optimal: {midi_to_note_name(violin.range.optimal_low)} - "
              f"{midi_to_note_name(violin.range.optimal_high)}")
        print(f"Max speed: {violin.max_speed} notes/sec")
        print(f"Polyphonic: {violin.polyphonic}")
        print(f"Articulations: {len(violin.articulations)}")
        print(f"Timbre: {', '.join(violin.timbre_descriptors)}")
        print(f"\nBlends well with: {', '.join(violin.blends_well_with)}")


def demo_7_orchestration_templates():
    """Demo 7: Show orchestration templates"""
    print("\n" + "=" * 80)
    print("DEMO 7: Orchestration Templates")
    print("=" * 80)

    templates = [
        "symphony_orchestra",
        "chamber_orchestra",
        "string_quartet",
        "wind_quintet",
        "brass_quintet",
        "piano_trio"
    ]

    for template_name in templates:
        instruments = get_orchestration_template(template_name)
        print(f"\n{template_name.replace('_', ' ').title()}:")
        print(f"  {len(instruments)} instruments:")
        for inst_name in instruments:
            inst = get_instrument(inst_name)
            if inst:
                print(f"    • {inst_name} ({inst.family.value})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all demos"""
    print("=" * 80)
    print("ORCHESTRATION & TIMBRE ENGINE - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("\nThis demo showcases the integration of:")
    print("  • Instrument Library (850+ lines)")
    print("  • Orchestrator (950+ lines)")
    print("  • Texture Generator (650+ lines)")
    print("  • Articulation Engine (550+ lines)")
    print("\nTotal: 3000+ lines of professional orchestration code")

    try:
        demo_1_simple_melody_orchestration()
        demo_2_texture_patterns()
        demo_3_articulations()
        demo_4_full_orchestration()
        demo_5_string_section_voicing()
        demo_6_instrument_database()
        demo_7_orchestration_templates()

        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nThe orchestration system is ready for:")
        print("  ✓ Intelligent instrument selection")
        print("  ✓ Professional orchestral arrangements")
        print("  ✓ Various accompaniment patterns")
        print("  ✓ Realistic articulations")
        print("  ✓ Complete MIDI generation")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
