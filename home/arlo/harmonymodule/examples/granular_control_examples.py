#!/usr/bin/env python3
"""
Granular Control System - Comprehensive Examples
================================================

This file contains practical examples demonstrating all major features
of the Granular Control System.

Examples included:
1. Basic brass hits
2. String pads
3. Funk brass section
4. Jazz big band
5. Film score textures
6. Layered arrangements
7. Percussion patterns
8. Dynamic shaping
9. Phrase endings
10. Complete multi-section arrangement

Author: Agent 8
Date: 2025-11-19
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from midi_generator.generators.granular_control import (
    # Core
    GranularControl,
    RhythmPattern,
    InstrumentSection,
    ArticulationType,
    VoicingStrategy,
    Register,
    # Engines
    BrassVoicingEngine,
    StringVoicingEngine,
    WoodwindVoicingEngine,
    PercussionVoicingEngine,
    DynamicsEngine,
    PhraseShaper,
    AdvancedControlEngine,
    # Convenience
    create_brass_hits,
    create_string_pad
)


def example_01_basic_brass_hits():
    """
    Example 1: Basic Brass Hits on Beats 1 and 3

    Creates simple brass hits for a typical pop/rock arrangement.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Brass Hits")
    print("=" * 70)

    # Use convenience function
    output = create_brass_hits(
        onsets=[0.0, 2.0],  # Beats 1 and 3
        chord_progression=["Cmaj7", "Dm7", "G7", "Cmaj7"],
        measures=4
    )

    print(f"\n✓ Generated {len(output.notes)} notes")
    print(f"✓ Playability: {output.voicing_quality.name}")
    print(f"✓ Section: {output.section.value}")

    # Show first 4 notes
    print("\nFirst 4 notes:")
    for note in output.notes[:4]:
        print(f"  {note.instrument:12} | Pitch: {note.pitch:3} | "
              f"Onset: {note.onset:5.2f} | Vel: {note.velocity:3} | "
              f"Artic: {note.articulation.value}")

    # Export
    gc = GranularControl()
    gc.to_midi(output, "output/example_01_brass_hits.mid", tempo=120)
    print("\n✓ Exported to: output/example_01_brass_hits.mid")

    return output


def example_02_string_pad():
    """
    Example 2: Sustained String Pad

    Creates a lush string pad with whole notes.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: String Pad")
    print("=" * 70)

    output = create_string_pad(
        duration=4.0,
        chord_progression=["Cmaj7", "Am7", "Fmaj7", "G7"],
        measures=4
    )

    print(f"\n✓ Generated {len(output.notes)} notes")
    print(f"✓ Section: {output.section.value}")

    # Apply crescendo
    DynamicsEngine.apply_dynamics_curve(
        output.notes,
        curve_type='arch',  # Swell effect
        start_dynamic='p',
        end_dynamic='mf'
    )

    print("✓ Applied arch dynamic curve (p → mf → p)")

    gc = GranularControl()
    gc.to_midi(output, "output/example_02_string_pad.mid", tempo=72)
    print("\n✓ Exported to: output/example_02_string_pad.mid")

    return output


def example_03_funk_brass():
    """
    Example 3: Syncopated Funk Brass Section

    Creates a tight, syncopated funk brass pattern.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Funk Brass Section")
    print("=" * 70)

    # Syncopated funk rhythm
    rhythm = RhythmPattern(
        onsets=[0.0, 0.75, 1.5, 2.5, 3.25],
        durations=[0.25, 0.25, 0.5, 0.25, 0.5],
        accents=[True, False, True, False, True]
    )

    gc = GranularControl()
    output = gc.generate(
        rhythm_pattern=rhythm,
        chord_progression=["Em7", "A7", "Dm7", "G7"],
        section=InstrumentSection.BRASS,
        articulation_style='hits',
        voicing_strategy=VoicingStrategy.DROP_2,
        measures=4
    )

    print(f"\n✓ Generated {len(output.notes)} notes")
    print(f"✓ Voicing: {VoicingStrategy.DROP_2.value}")
    print(f"✓ Articulation style: hits (tight, staccato)")

    # Humanize
    AdvancedControlEngine.apply_humanization(
        output.notes,
        timing_variance=0.01,
        velocity_variance=5
    )

    print("✓ Applied humanization")

    gc.to_midi(output, "output/example_03_funk_brass.mid", tempo=110)
    print("\n✓ Exported to: output/example_03_funk_brass.mid")

    return output


def example_04_jazz_big_band():
    """
    Example 4: Jazz Big Band with Swing

    Creates a classic jazz big band arrangement with swing feel.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Jazz Big Band with Swing")
    print("=" * 70)

    # Jazz rhythm (will be swung)
    rhythm = RhythmPattern(
        onsets=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],  # Straight 8ths
        durations=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        accents=[True, False, False, True, True, False, False, True]
    )

    gc = GranularControl()

    # Generate with swing
    output = gc.generate(
        rhythm_pattern=rhythm,
        chord_progression=["Cmaj7", "Am7", "Dm7", "G7"],
        section=InstrumentSection.BRASS,
        voicing_strategy=VoicingStrategy.DROP_2,
        articulation_style='jazz_articulation',
        measures=4,
        apply_swing=True,
        swing_factor=0.67  # Triplet swing
    )

    print(f"\n✓ Generated {len(output.notes)} notes")
    print(f"✓ Applied swing feel (factor: 0.67)")
    print(f"✓ Jazz articulation (tongued, accents, fall-offs)")

    # Add jazz drums
    drums = PercussionVoicingEngine.create_basic_beat(
        style='jazz',
        measures=4
    )

    print(f"✓ Added {len(drums)} drum notes (jazz ride pattern)")

    gc.to_midi(output, "output/example_04_jazz_bigband.mid", tempo=140)
    print("\n✓ Exported to: output/example_04_jazz_bigband.mid")

    return output, drums


def example_05_film_score_crescendo():
    """
    Example 5: Film Score Dramatic Crescendo

    Creates a dramatic orchestral build-up.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Film Score Dramatic Crescendo")
    print("=" * 70)

    # Long sustained notes
    rhythm = RhythmPattern(
        onsets=[0.0, 8.0, 16.0],
        durations=[8.0, 8.0, 8.0]
    )

    gc = GranularControl()

    # Strings
    strings = gc.generate_sustained(
        rhythm=rhythm,
        chord_progression=["Cm", "Ab", "Fm"],
        section=InstrumentSection.STRINGS,
        voicing_strategy=VoicingStrategy.OPEN,
        measures=24,
        beats_per_measure=4
    )

    # Apply dramatic crescendo
    DynamicsEngine.apply_dynamics_curve(
        strings.notes,
        curve_type='crescendo',
        start_dynamic='pp',
        end_dynamic='fff'
    )

    print(f"\n✓ Generated {len(strings.notes)} string notes")
    print(f"✓ Applied crescendo (pp → fff)")

    # Add brass for climax (last 8 measures)
    brass_rhythm = RhythmPattern(
        onsets=[0.0, 2.0, 4.0, 6.0],
        durations=[2.0, 2.0, 2.0, 2.0],
        accents=[False, False, True, True]
    )

    brass = gc.generate(
        rhythm_pattern=brass_rhythm,
        chord_progression=["Fm", "Fm"],
        section=InstrumentSection.BRASS,
        voicing_strategy=VoicingStrategy.OCTAVES,
        measures=8
    )

    # Shift brass to start at measure 17
    for note in brass.notes:
        note.onset += 64.0  # 16 measures * 4 beats

    print(f"✓ Added {len(brass.notes)} brass notes for climax")

    # Combine
    from midi_generator.generators.granular_control import SectionOutput

    all_notes = strings.notes + brass.notes

    combined = SectionOutput(
        section=InstrumentSection.STRINGS,
        notes=all_notes,
        voicing_quality=strings.voicing_quality,
        warnings=[],
        suggestions=[]
    )

    gc.to_midi(combined, "output/example_05_film_crescendo.mid", tempo=60)
    print("\n✓ Exported to: output/example_05_film_crescendo.mid")

    return combined


def example_06_layered_arrangement():
    """
    Example 6: Multi-Layer Textured Arrangement

    Creates a complex arrangement with multiple layers.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Layered Multi-Section Arrangement")
    print("=" * 70)

    base_rhythm = RhythmPattern(
        onsets=[0.0, 1.0, 2.0, 3.0],
        durations=[0.5, 0.5, 0.5, 0.5]
    )

    chords = ["Cmaj7", "Am7", "Fmaj7", "G7"]

    layers = [
        {
            'section': InstrumentSection.BRASS,
            'instruments': ['trumpet', 'trumpet', 'trombone'],
            'voicing_strategy': VoicingStrategy.DROP_2,
            'measures': 4
        },
        {
            'section': InstrumentSection.STRINGS,
            'instruments': ['violin', 'viola', 'cello'],
            'voicing_strategy': VoicingStrategy.CLOSE,
            'offset': 0.5,  # Delay by half beat
            'duration_multiplier': 2.0,  # Longer notes
            'measures': 4
        },
        {
            'section': InstrumentSection.WOODWINDS,
            'instruments': ['flute', 'oboe', 'clarinet'],
            'voicing_strategy': VoicingStrategy.TRADITIONAL,
            'offset': 0.25,  # Slight delay
            'measures': 4
        }
    ]

    outputs = AdvancedControlEngine.create_layered_texture(
        base_rhythm,
        chord_progression=chords,
        layers=layers
    )

    total_notes = sum(len(output.notes) for output in outputs)

    print(f"\n✓ Generated {len(layers)} layers")
    print(f"✓ Total notes: {total_notes}")
    print(f"  - Brass: {len(outputs[0].notes)} notes")
    print(f"  - Strings: {len(outputs[1].notes)} notes")
    print(f"  - Woodwinds: {len(outputs[2].notes)} notes")

    # Combine and export
    from midi_generator.generators.granular_control import SectionOutput

    all_notes = []
    for output in outputs:
        all_notes.extend(output.notes)

    combined = SectionOutput(
        section=InstrumentSection.BRASS,
        notes=all_notes,
        voicing_quality=outputs[0].voicing_quality,
        warnings=[],
        suggestions=[]
    )

    gc = GranularControl()
    gc.to_midi(combined, "output/example_06_layered.mid", tempo=100)
    print("\n✓ Exported to: output/example_06_layered.mid")

    return outputs


def example_07_percussion_patterns():
    """
    Example 7: Various Percussion Patterns

    Demonstrates different drum styles and custom patterns.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Percussion Patterns")
    print("=" * 70)

    # Rock beat
    rock = PercussionVoicingEngine.create_basic_beat(
        style='rock',
        measures=2
    )

    print(f"\n✓ Rock beat: {len(rock)} notes")

    # Jazz ride
    jazz = PercussionVoicingEngine.create_basic_beat(
        style='jazz',
        measures=2
    )

    print(f"✓ Jazz ride: {len(jazz)} notes")

    # Custom pattern
    custom_rhythm = RhythmPattern(
        onsets=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        durations=[0.25] * 8,
        accents=[True, False, True, False, True, False, True, False]
    )

    custom_drums = PercussionVoicingEngine.rhythm_to_drums(
        custom_rhythm,
        drum_voices=['kick', 'hihat', 'snare', 'hihat',
                    'kick', 'hihat', 'snare', 'hihat']
    )

    print(f"✓ Custom pattern: {len(custom_drums)} notes")

    # Export
    from midi_generator.generators.granular_control import SectionOutput, GeneratedNote

    all_drums = rock + jazz + [
        GeneratedNote(
            pitch=note.pitch,
            onset=note.onset + 16.0,  # Offset for third section
            duration=note.duration,
            velocity=note.velocity,
            articulation=note.articulation,
            instrument='drums'
        ) for note in custom_drums
    ]

    combined = SectionOutput(
        section=InstrumentSection.PERCUSSION,
        notes=all_drums,
        voicing_quality=None,
        warnings=[],
        suggestions=[]
    )

    gc = GranularControl()
    gc.to_midi(combined, "output/example_07_percussion.mid", tempo=120)
    print("\n✓ Exported to: output/example_07_percussion.mid")

    return all_drums


def example_08_dynamic_shaping():
    """
    Example 8: Dynamic Curve Shaping

    Demonstrates various dynamic curves and accent patterns.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Dynamic Shaping")
    print("=" * 70)

    rhythm = RhythmPattern(
        onsets=[i * 0.5 for i in range(16)],  # 16 8th notes
        durations=[0.5] * 16
    )

    gc = GranularControl()

    # Crescendo
    output1 = gc.generate(
        rhythm_pattern=rhythm,
        chord_progression=["Cmaj7"],
        section=InstrumentSection.STRINGS,
        measures=2
    )

    DynamicsEngine.apply_dynamics_curve(
        output1.notes,
        curve_type='crescendo',
        start_dynamic='p',
        end_dynamic='ff'
    )

    print("\n✓ Applied crescendo (p → ff)")

    # Arch (swell)
    output2 = gc.generate(
        rhythm_pattern=rhythm,
        chord_progression=["Dm7"],
        section=InstrumentSection.STRINGS,
        measures=2
    )

    # Shift onset
    for note in output2.notes:
        note.onset += 8.0

    DynamicsEngine.apply_dynamics_curve(
        output2.notes,
        curve_type='arch',
        start_dynamic='p',
        end_dynamic='f'
    )

    print("✓ Applied arch curve (swell)")

    # Accent pattern
    output3 = gc.generate(
        rhythm_pattern=rhythm,
        chord_progression=["Em7"],
        section=InstrumentSection.BRASS,
        measures=2
    )

    # Shift onset
    for note in output3.notes:
        note.onset += 16.0

    DynamicsEngine.apply_accents(
        output3.notes,
        accent_pattern=[True, False, False, True],
        accent_amount=25
    )

    print("✓ Applied accent pattern")

    # Combine
    from midi_generator.generators.granular_control import SectionOutput

    all_notes = output1.notes + output2.notes + output3.notes

    combined = SectionOutput(
        section=InstrumentSection.STRINGS,
        notes=all_notes,
        voicing_quality=output1.voicing_quality,
        warnings=[],
        suggestions=[]
    )

    gc.to_midi(combined, "output/example_08_dynamics.mid", tempo=90)
    print("\n✓ Exported to: output/example_08_dynamics.mid")

    return combined


def example_09_phrase_endings():
    """
    Example 9: Musical Phrase Endings

    Demonstrates ritardando, breath marks, and ornaments.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 9: Phrase Endings and Ornaments")
    print("=" * 70)

    rhythm = RhythmPattern(
        onsets=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        durations=[1.0] * 8
    )

    gc = GranularControl()

    # Phrase 1: Ritardando
    phrase1 = gc.generate(
        rhythm_pattern=rhythm,
        chord_progression=["Cmaj7", "Dm7"],
        section=InstrumentSection.WOODWINDS,
        measures=2
    )

    PhraseShaper.add_phrase_ending(phrase1.notes, ending_type='ritardando')
    print("\n✓ Phrase 1: Ritardando ending")

    # Phrase 2: Decrescendo
    phrase2 = gc.generate(
        rhythm_pattern=rhythm,
        chord_progression=["Em7", "Fmaj7"],
        section=InstrumentSection.WOODWINDS,
        measures=2
    )

    for note in phrase2.notes:
        note.onset += 8.0

    PhraseShaper.add_phrase_ending(phrase2.notes, ending_type='decrescendo')
    print("✓ Phrase 2: Decrescendo ending")

    # Phrase 3: With ornaments
    phrase3 = gc.generate(
        rhythm_pattern=rhythm,
        chord_progression=["G7", "Cmaj7"],
        section=InstrumentSection.WOODWINDS,
        measures=2
    )

    for note in phrase3.notes:
        note.onset += 16.0

    phrase3.notes = PhraseShaper.add_ornaments(
        phrase3.notes,
        ornament_type='grace_note',
        positions=[0, -1]
    )

    print("✓ Phrase 3: Grace notes added")

    # Combine
    from midi_generator.generators.granular_control import SectionOutput

    all_notes = phrase1.notes + phrase2.notes + phrase3.notes

    combined = SectionOutput(
        section=InstrumentSection.WOODWINDS,
        notes=all_notes,
        voicing_quality=phrase1.voicing_quality,
        warnings=[],
        suggestions=[]
    )

    gc.to_midi(combined, "output/example_09_phrases.mid", tempo=80)
    print("\n✓ Exported to: output/example_09_phrases.mid")

    return combined


def example_10_complete_arrangement():
    """
    Example 10: Complete Multi-Section Arrangement

    Creates a full arrangement combining all techniques.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 10: Complete Arrangement")
    print("=" * 70)

    gc = GranularControl()
    chords = ["Cmaj7", "Am7", "Dm7", "G7", "Cmaj7", "Fmaj7", "Em7", "Am7"]

    # 1. Drums (8 measures)
    drums = PercussionVoicingEngine.create_basic_beat(style='rock', measures=8)
    print(f"\n✓ Drums: {len(drums)} notes")

    # 2. String pad (sustained)
    string_rhythm = RhythmPattern(
        onsets=[i * 4.0 for i in range(8)],
        durations=[4.0] * 8
    )

    strings = gc.generate_sustained(
        rhythm=string_rhythm,
        chord_progression=chords,
        section=InstrumentSection.STRINGS,
        measures=8
    )

    DynamicsEngine.apply_dynamics_curve(
        strings.notes,
        curve_type='arch',
        start_dynamic='p',
        end_dynamic='mf'
    )

    print(f"✓ Strings: {len(strings.notes)} notes (with arch dynamic)")

    # 3. Brass hits (on 1 and 3, starting measure 5)
    brass_rhythm = RhythmPattern(
        onsets=[0.0, 2.0],
        durations=[0.25, 0.25],
        accents=[True, True]
    )

    brass = gc.generate_hits(
        rhythm=brass_rhythm,
        chord_progression=chords[4:],  # Last 4 chords
        measures=4
    )

    # Shift brass to start at measure 5
    for note in brass.notes:
        note.onset += 16.0  # 4 measures * 4 beats

    print(f"✓ Brass: {len(brass.notes)} notes (measures 5-8)")

    # 4. Woodwind melody (measures 5-8)
    melody_rhythm = RhythmPattern(
        onsets=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5],
        durations=[0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5],
        accents=[True, False, False, True, True, False, True]
    )

    woodwinds = gc.generate(
        rhythm_pattern=melody_rhythm,
        chord_progression=chords[4:],
        section=InstrumentSection.WOODWINDS,
        instruments=['flute', 'oboe'],
        voicing_strategy=VoicingStrategy.TRADITIONAL,
        measures=4
    )

    # Shift to measure 5
    for note in woodwinds.notes:
        note.onset += 16.0

    # Add phrase ending
    PhraseShaper.add_phrase_ending(woodwinds.notes, ending_type='ritardando')

    print(f"✓ Woodwinds: {len(woodwinds.notes)} notes (melody, measures 5-8)")

    # 5. Combine all sections
    from midi_generator.generators.granular_control import SectionOutput, GeneratedNote

    all_notes = strings.notes + brass.notes + woodwinds.notes

    # Add drums
    all_notes.extend([
        GeneratedNote(
            pitch=note.pitch,
            onset=note.onset,
            duration=note.duration,
            velocity=note.velocity,
            articulation=note.articulation,
            instrument='drums'
        ) for note in drums
    ])

    combined = SectionOutput(
        section=InstrumentSection.STRINGS,
        notes=all_notes,
        voicing_quality=strings.voicing_quality,
        warnings=[],
        suggestions=[]
    )

    # Apply humanization
    AdvancedControlEngine.apply_humanization(
        combined.notes,
        timing_variance=0.015,
        velocity_variance=6
    )

    print(f"✓ Humanization applied")

    total = len(combined.notes)
    print(f"\n✓ TOTAL: {total} notes across all sections")

    gc.to_midi(combined, "output/example_10_complete.mid", tempo=110)
    print("\n✓ Exported to: output/example_10_complete.mid")

    return combined


def run_all_examples():
    """Run all examples and export MIDI files"""
    print("\n" + "=" * 70)
    print("GRANULAR CONTROL SYSTEM - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    print("\nRunning all examples...")

    # Create output directory
    import os
    os.makedirs("output", exist_ok=True)

    # Run examples
    example_01_basic_brass_hits()
    example_02_string_pad()
    example_03_funk_brass()
    example_04_jazz_big_band()
    example_05_film_score_crescendo()
    example_06_layered_arrangement()
    example_07_percussion_patterns()
    example_08_dynamic_shaping()
    example_09_phrase_endings()
    example_10_complete_arrangement()

    print("\n" + "=" * 70)
    print("✅ ALL EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nMIDI files saved in: output/")
    print("\nExamples included:")
    print("  1. Basic brass hits")
    print("  2. String pads")
    print("  3. Funk brass")
    print("  4. Jazz big band with swing")
    print("  5. Film score crescendo")
    print("  6. Layered multi-section arrangement")
    print("  7. Percussion patterns")
    print("  8. Dynamic shaping")
    print("  9. Phrase endings and ornaments")
    print(" 10. Complete arrangement")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_all_examples()
