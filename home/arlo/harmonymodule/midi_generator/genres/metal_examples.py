#!/usr/bin/env python3
"""
Metal Generator - Usage Examples and Integration Demonstrations

This file demonstrates:
1. Basic usage of MetalGenerator
2. Integration with existing harmony/melody modules
3. MIDI file export
4. Advanced composition techniques
5. Complete song structure generation

Author: Agent 11
Date: 2025
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genres.metal import (
    MetalGenerator,
    MetalSubgenre,
    DropTuning,
    BlastBeatType,
    RiffTechnique,
    MetalScales,
    convert_to_midi_events,
    convert_drums_to_midi_events
)


# ============================================================================
# Example 1: Simple Thrash Metal Riff
# ============================================================================

def example_1_simple_thrash():
    """Generate a simple thrash metal riff"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Thrash Metal Riff (Metallica Style)")
    print("="*70)

    generator = MetalGenerator()

    # Generate a thrash riff in Drop D tuning
    riff = generator.generate_riff(
        subgenre=MetalSubgenre.THRASH,
        key=38,  # D (low D in Drop D)
        tuning=DropTuning.DROP_D,
        measures=4
    )

    print(f"\nGenerated thrash riff:")
    print(f"  - Notes: {len(riff.notes)} total")
    print(f"  - Technique: {riff.technique.value}")
    print(f"  - Palm mute intensity: {riff.palm_mute_intensity}")
    print(f"  - Tuning: {riff.tuning.value}")
    print(f"  - First 16 notes: {riff.notes[:16]}")
    print(f"  - Note range: {min(riff.notes)} - {max(riff.notes)}")

    return riff


# ============================================================================
# Example 2: Death Metal with Blast Beats
# ============================================================================

def example_2_death_metal_blast():
    """Generate death metal section with blast beats"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Death Metal with Blast Beats")
    print("="*70)

    generator = MetalGenerator()

    # Generate death metal riff
    riff = generator.generate_riff(
        subgenre=MetalSubgenre.DEATH,
        key=36,  # Low C
        scale='harmonic_minor',
        measures=4
    )

    # Generate blast beat pattern
    drums = generator.generate_drums(
        subgenre=MetalSubgenre.DEATH,
        blast_type=BlastBeatType.STANDARD,
        measures=4
    )

    print(f"\nDeath metal section:")
    print(f"  Riff:")
    print(f"    - {len(riff.notes)} notes")
    print(f"    - Technique: {riff.technique.value}")
    print(f"  Drums:")
    print(f"    - Kick hits: {len(drums.kick)}")
    print(f"    - Snare hits: {len(drums.snare)}")
    print(f"    - Ride cymbal: {len(drums.ride)} hits")
    print(f"    - BPM suggestion: 200+ for proper blast beat feel")

    return riff, drums


# ============================================================================
# Example 3: Djent Polyrhythmic Composition
# ============================================================================

def example_3_djent_polyrhythm():
    """Generate djent/Meshuggah-style polyrhythmic section"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Djent Polyrhythmic Riff (Meshuggah Style)")
    print("="*70)

    generator = MetalGenerator()

    # Generate 4:3 polyrhythm
    riff_4_3 = generator.generate_riff(
        subgenre=MetalSubgenre.DJENT,
        polymeter=(4, 3),
        measures=6,
        syncopation=0.8
    )

    # Generate 5:4 polyrhythm
    riff_5_4 = generator.generate_riff(
        subgenre=MetalSubgenre.DJENT,
        polymeter=(5, 4),
        measures=6,
        syncopation=0.7
    )

    print(f"\nDjent polyrhythmic riffs:")
    print(f"  4:3 Polymeter:")
    print(f"    - {len(riff_4_3.notes)} notes")
    print(f"    - Tuning: {riff_4_3.tuning.value}")
    print(f"    - Palm mute: {riff_4_3.palm_mute_intensity}")
    print(f"  5:4 Polymeter:")
    print(f"    - {len(riff_5_4.notes)} notes")
    print(f"    - Creates complex cross-rhythm against 4/4 drums")

    return riff_4_3, riff_5_4


# ============================================================================
# Example 4: Iron Maiden Gallop
# ============================================================================

def example_4_iron_maiden_gallop():
    """Generate classic Iron Maiden gallop pattern"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Iron Maiden Gallop Rhythm")
    print("="*70)

    generator = MetalGenerator()

    # Generate gallop pattern
    gallop = generator.riff_generator.generate_gallop_pattern(
        root_note=40,  # E
        measures=4,
        tuning=DropTuning.STANDARD
    )

    print(f"\nGallop pattern analysis:")
    print(f"  - Total notes: {len(gallop.notes)}")
    print(f"  - Pattern: 8th + two 16ths (LONG-short-short)")
    print(f"  - Durations (first 12): {gallop.durations[:12]}")
    print(f"  - Velocities (first 12): {gallop.velocities[:12]}")
    print(f"  - Classic songs: Run to the Hills, The Trooper")

    # Verify gallop pattern (should be 2,1,1 repeating)
    pattern_correct = True
    for i in range(0, min(12, len(gallop.durations)), 3):
        if gallop.durations[i:i+3] != [2, 1, 1]:
            pattern_correct = False
            break

    print(f"  - Pattern verification: {'✓ CORRECT' if pattern_correct else '✗ ERROR'}")

    return gallop


# ============================================================================
# Example 5: Neoclassical Sweep Picking
# ============================================================================

def example_5_neoclassical_sweep():
    """Generate Yngwie Malmsteen-style sweep picking"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Neoclassical Sweep Picking Arpeggios")
    print("="*70)

    generator = MetalGenerator()

    # Generate minor arpeggio sweep
    sweep_minor = generator.riff_generator.generate_sweep_arpeggio(
        root=57,  # A
        chord_type='minor',
        direction='both'
    )

    # Generate diminished arpeggio
    sweep_dim = generator.riff_generator.generate_sweep_arpeggio(
        root=60,  # C
        chord_type='diminished',
        direction='ascending'
    )

    print(f"\nNeoclassical sweep arpeggios:")
    print(f"  A minor arpeggio (both directions):")
    print(f"    - Notes: {sweep_minor.notes}")
    print(f"    - Length: {len(sweep_minor.notes)} notes")
    print(f"  C diminished arpeggio:")
    print(f"    - Notes: {sweep_dim.notes}")
    print(f"    - Interval pattern: 1-b3-b5-8 (diminished triad)")

    return sweep_minor, sweep_dim


# ============================================================================
# Example 6: Complete Metal Song Structure
# ============================================================================

def example_6_full_song_structure():
    """Generate a complete metal song structure"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Complete Metal Song Structure")
    print("="*70)

    generator = MetalGenerator()

    # Song structure: Intro - Verse - Chorus - Verse - Chorus - Bridge - Solo - Chorus - Outro

    structure = {}

    # 1. Intro (atmospheric, slow build)
    print("\n  Generating INTRO...")
    structure['intro'] = generator.generate_full_section(
        subgenre=MetalSubgenre.PROGRESSIVE,
        key=40,
        tuning=DropTuning.DROP_D,
        measures=8
    )

    # 2. Verse (thrash riff with double bass)
    print("  Generating VERSE...")
    structure['verse'] = generator.generate_full_section(
        subgenre=MetalSubgenre.THRASH,
        key=38,
        tuning=DropTuning.DROP_D,
        measures=8
    )

    # 3. Chorus (heavy, gallop rhythm)
    print("  Generating CHORUS...")
    structure['chorus'] = {
        'riff': generator.riff_generator.generate_gallop_pattern(
            root_note=38,
            measures=8,
            tuning=DropTuning.DROP_D
        ),
        'drums': generator.drum_generator.generate_double_bass_pattern(
            measures=8,
            pattern_type='sixteenths'
        )
    }

    # 4. Bridge (djent polyrhythm)
    print("  Generating BRIDGE...")
    structure['bridge'] = generator.generate_full_section(
        subgenre=MetalSubgenre.DJENT,
        key=33,
        tuning=DropTuning.DROP_A,
        measures=8
    )

    # 5. Solo section (neoclassical sweep picking)
    print("  Generating SOLO...")
    structure['solo'] = {
        'lead': generator.riff_generator.generate_sweep_arpeggio(
            root=60,
            chord_type='minor',
            direction='both'
        ),
        'rhythm': generator.generate_riff(
            subgenre=MetalSubgenre.THRASH,
            key=38,
            measures=8
        )
    }

    # 6. Breakdown (metalcore style)
    print("  Generating BREAKDOWN...")
    structure['breakdown'] = {
        'riff': generator.generate_riff(
            subgenre=MetalSubgenre.METALCORE,
            key=36,
            measures=4
        ),
        'drums': generator.drum_generator.generate_breakdown_pattern(
            measures=4,
            syncopation=0.8
        )
    }

    # Print song structure summary
    print("\n  Song Structure Complete!")
    print("  " + "-"*66)
    print(f"  Total sections: {len(structure)}")
    for section_name in structure.keys():
        print(f"    - {section_name.upper()}")
    print("  " + "-"*66)

    # Calculate total length
    total_measures = 8 + 8 + 8 + 8 + 8 + 4  # Sum of all section measures
    print(f"\n  Total measures: {total_measures}")
    print(f"  Estimated duration at 160 BPM: {total_measures * 4 * 60 / 160:.1f} seconds")

    return structure


# ============================================================================
# Example 7: All Blast Beat Types Comparison
# ============================================================================

def example_7_blast_beat_comparison():
    """Compare all blast beat types"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Blast Beat Types Comparison")
    print("="*70)

    generator = MetalGenerator()

    blast_types = [
        BlastBeatType.STANDARD,
        BlastBeatType.HAMMER,
        BlastBeatType.GRAVITY,
        BlastBeatType.BOMB,
        BlastBeatType.HYPER
    ]

    print("\n  Generating all blast beat variations...")
    print("  " + "-"*66)

    for blast_type in blast_types:
        pattern = generator.drum_generator.generate_blast_beat(
            blast_type=blast_type,
            measures=2,
            bpm=200
        )

        print(f"\n  {blast_type.value.upper()}:")
        print(f"    - Kick hits: {len(pattern.kick):2d}")
        print(f"    - Snare hits: {len(pattern.snare):2d}")
        print(f"    - Hi-hat hits: {len(pattern.hihat):2d}")
        print(f"    - Crash hits: {len(pattern.crash):2d}")

        # Describe characteristics
        if blast_type == BlastBeatType.STANDARD:
            print(f"    - Character: Traditional Euro blast, most common")
        elif blast_type == BlastBeatType.HAMMER:
            print(f"    - Character: Simultaneous kick+snare, very powerful")
        elif blast_type == BlastBeatType.GRAVITY:
            print(f"    - Character: Double snare technique, faster feel")
        elif blast_type == BlastBeatType.BOMB:
            print(f"    - Character: Everything together, maximum intensity")
        elif blast_type == BlastBeatType.HYPER:
            print(f"    - Character: Ultra-fast, extreme metal")

    print("  " + "-"*66)


# ============================================================================
# Example 8: Metal Scales Demonstration
# ============================================================================

def example_8_metal_scales():
    """Demonstrate all metal scales"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Metal Scales and Modes")
    print("="*70)

    scales = [
        ('harmonic_minor', 'Neoclassical, Dark'),
        ('phrygian_dominant', 'Eastern, Exotic'),
        ('phrygian', 'Dark, Spanish'),
        ('locrian', 'Diminished, Unstable'),
        ('octatonic_hw', 'Symmetrical, Meshuggah'),
        ('diminished', 'Dissonant, Complex')
    ]

    print("\n  Metal scale analysis (root = C / MIDI 60):")
    print("  " + "-"*66)

    for scale_name, description in scales:
        notes = MetalScales.get_notes(60, scale_name, octaves=1)
        intervals = [note - 60 for note in notes if note < 72]

        print(f"\n  {scale_name.upper()}:")
        print(f"    Description: {description}")
        print(f"    Intervals: {intervals}")
        print(f"    MIDI notes: {notes[:8]}")

        # Show which sub-genres use this scale
        if scale_name == 'harmonic_minor':
            print(f"    Used in: Neoclassical, Power Metal")
        elif scale_name == 'phrygian_dominant':
            print(f"    Used in: Death Metal, Black Metal")
        elif scale_name == 'octatonic_hw':
            print(f"    Used in: Djent, Progressive Metal")

    print("  " + "-"*66)


# ============================================================================
# Example 9: Drop Tuning Comparison
# ============================================================================

def example_9_drop_tunings():
    """Compare all drop tunings"""
    print("\n" + "="*70)
    print("EXAMPLE 9: Drop Tuning Systems")
    print("="*70)

    from genres.metal import TuningSystem

    tunings = [
        (DropTuning.STANDARD, 'Classic metal'),
        (DropTuning.DROP_D, 'Nu-metal, Modern rock'),
        (DropTuning.DROP_C, 'Metalcore, Deathcore'),
        (DropTuning.DROP_B, 'Extreme metal'),
        (DropTuning.DROP_A, 'Djent, Modern prog'),
        (DropTuning.SEVEN_STRING, '7-string standard'),
    ]

    print("\n  Tuning comparison (MIDI note numbers):")
    print("  " + "-"*66)

    for tuning, description in tunings:
        notes = TuningSystem.get_tuning(tuning)
        # Convert MIDI to note names
        note_names = []
        note_map = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for midi_note in notes:
            octave = (midi_note // 12) - 1
            note_name = note_map[midi_note % 12]
            note_names.append(f"{note_name}{octave}")

        print(f"\n  {tuning.value.upper()}:")
        print(f"    Notes: {' '.join(note_names)}")
        print(f"    MIDI: {notes}")
        print(f"    Style: {description}")
        print(f"    Lowest note: {note_names[0]} (MIDI {notes[0]})")

    print("  " + "-"*66)


# ============================================================================
# Example 10: MIDI Export
# ============================================================================

def example_10_midi_export():
    """Demonstrate MIDI event conversion"""
    print("\n" + "="*70)
    print("EXAMPLE 10: MIDI Event Export")
    print("="*70)

    generator = MetalGenerator()

    # Generate a complete section
    section = generator.generate_full_section(
        subgenre=MetalSubgenre.THRASH,
        key=38,
        tuning=DropTuning.DROP_D,
        measures=4
    )

    # Convert to MIDI events
    riff_events = convert_to_midi_events(
        section['riff'],
        start_tick=0,
        ppqn=480
    )

    drum_events = convert_drums_to_midi_events(
        section['drums'],
        start_tick=0,
        ppqn=480
    )

    print(f"\n  MIDI Conversion Results:")
    print(f"    - Riff events: {len(riff_events)}")
    print(f"    - Drum events: {len(drum_events)}")
    print(f"    - Total events: {len(riff_events) + len(drum_events)}")
    print(f"    - PPQN (resolution): 480")

    print(f"\n  Sample riff events (first 4):")
    for i, event in enumerate(riff_events[:4]):
        print(f"    {i+1}. {event}")

    print(f"\n  Sample drum events (first 4):")
    for i, event in enumerate(drum_events[:4]):
        print(f"    {i+1}. {event}")

    print(f"\n  These events can be written to .mid files using libraries like:")
    print(f"    - mido")
    print(f"    - pretty_midi")
    print(f"    - music21")

    return riff_events, drum_events


# ============================================================================
# Main Execution
# ============================================================================

def run_all_examples():
    """Run all examples"""
    print("\n" + "="*70)
    print("METAL GENERATOR - COMPREHENSIVE EXAMPLES")
    print("Agent 11: Metal & Heavy Music")
    print("="*70)

    try:
        example_1_simple_thrash()
        example_2_death_metal_blast()
        example_3_djent_polyrhythm()
        example_4_iron_maiden_gallop()
        example_5_neoclassical_sweep()
        example_6_full_song_structure()
        example_7_blast_beat_comparison()
        example_8_metal_scales()
        example_9_drop_tunings()
        example_10_midi_export()

        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nKey Features Demonstrated:")
        print("  ✓ 10 metal sub-genres")
        print("  ✓ 7 tuning systems")
        print("  ✓ 5 blast beat types")
        print("  ✓ Multiple guitar techniques")
        print("  ✓ Scale systems (harmonic minor, Phrygian dominant, octatonic)")
        print("  ✓ Complete song structure generation")
        print("  ✓ MIDI export capability")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
