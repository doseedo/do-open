#!/usr/bin/env python3
"""
Harmonic Rhythm Integration Example

Demonstrates integration of the HarmonicRhythm module with
the existing harmonymodule ecosystem.

This example shows:
1. Standalone usage of HarmonicRhythm
2. Integration with chord progressions
3. MIDI-ready output format
4. Real-world composition workflows

Author: Agent 19
Date: 2025
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from advanced_modules.harmonic_rhythm import (
    HarmonicRhythm,
    HarmonicDensity,
    GenreStyle,
    TensionCurveType,
    print_harmonic_rhythm_pattern
)


def example_1_basic_usage():
    """Example 1: Basic harmonic rhythm generation"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Harmonic Rhythm Generation")
    print("="*80)

    hr = HarmonicRhythm(beats_per_measure=4)

    # Generate a simple 8-bar pattern
    pattern = hr.generate_harmonic_rhythm(
        density="medium",
        total_measures=8,
        variation=0.2,
        prefer_downbeats=True
    )

    print(f"\nGenerated {len(pattern.chord_durations)} chords over {pattern.total_measures} measures")
    print(f"Average density: {pattern.average_density:.2f} chords per measure")

    # Show first 5 chord timings
    print("\nFirst 5 chord timings:")
    for cd in pattern.chord_durations[:5]:
        print(f"  Chord {cd.chord_index}: Measure {cd.measure}, "
              f"Beat {cd.beat_in_measure:.2f}, Duration {cd.duration_beats:.2f} beats")


def example_2_genre_specific():
    """Example 2: Genre-specific patterns"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Genre-Specific Harmonic Rhythms")
    print("="*80)

    hr = HarmonicRhythm()

    genres = ["pop", "bebop", "electronic", "blues"]

    for genre in genres:
        pattern = hr.create_genre_appropriate_rhythm(
            genre=genre,
            measures=8
        )
        print(f"\n{genre.upper()}: {pattern.average_density:.2f} chords/measure, "
              f"{len(pattern.chord_durations)} total chords")


def example_3_with_chord_progression():
    """Example 3: Map harmonic rhythm to chord progression"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Harmonic Rhythm + Chord Progression")
    print("="*80)

    hr = HarmonicRhythm()

    # Define a chord progression (typical pop: I-V-vi-IV)
    chords = ["C", "G", "Am", "F"]

    # Generate harmonic rhythm for 8 measures (pop style)
    rhythm_pattern = hr.create_genre_appropriate_rhythm(
        genre="pop",
        measures=8
    )

    # Map chords to rhythm pattern
    timed_chords = []
    for chord_dur in rhythm_pattern.chord_durations:
        # Cycle through chord progression
        chord_idx = chord_dur.chord_index % len(chords)

        timed_chords.append({
            "chord": chords[chord_idx],
            "measure": chord_dur.measure,
            "beat": chord_dur.beat_in_measure,
            "duration_beats": chord_dur.duration_beats,
            "start_beat_global": chord_dur.start_beat
        })

    print(f"\nMapped {len(chords)} chords to {len(timed_chords)} timed positions")
    print("\nTimed Chord Progression:")
    for tc in timed_chords[:8]:  # Show first 8
        print(f"  {tc['chord']:4s} | Measure {tc['measure']}, "
              f"Beat {tc['beat']:.1f}, Duration {tc['duration_beats']:.1f} beats")


def example_4_cadential_acceleration():
    """Example 4: Classical cadential acceleration"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Cadential Acceleration (Classical Technique)")
    print("="*80)

    hr = HarmonicRhythm()

    # Create pattern that accelerates toward cadence
    pattern = hr.apply_harmonic_acceleration(
        progression_length=12,
        start_density=1.0,   # Slow: 1 chord per measure
        end_density=4.0,     # Fast: 4 chords per measure
        curve="exponential",
        cadence_measures=3   # Last 3 measures accelerate
    )

    print(f"\nGenerated {len(pattern.chord_durations)} chords over 12 measures")

    # Analyze density by section
    early_chords = [cd for cd in pattern.chord_durations if cd.measure < 9]
    cadence_chords = [cd for cd in pattern.chord_durations if cd.measure >= 9]

    print(f"Measures 0-8 (pre-cadence): {len(early_chords)} chords")
    print(f"Measures 9-11 (cadence): {len(cadence_chords)} chords")
    print(f"\nAcceleration creates forward momentum toward cadence!")


def example_5_tension_based_pacing():
    """Example 5: Tension curve mapping"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Tension-Based Harmonic Pacing")
    print("="*80)

    hr = HarmonicRhythm()

    # Create arch-shaped tension: low → high → low
    pattern = hr.apply_tension_pacing(
        progression_length=16,
        tension_curve=TensionCurveType.ARCH,
        base_density=2.0,
        density_range=(1.0, 4.0)
    )

    print(f"\nGenerated {len(pattern.chord_durations)} chords")
    print("\nTension and density by measure:")

    for measure_idx in range(16):
        chords_in_measure = [cd for cd in pattern.chord_durations
                            if cd.measure == measure_idx]
        tension = pattern.tension_curve[measure_idx] if pattern.tension_curve else 0.5

        print(f"  Measure {measure_idx:2d}: Tension={tension:.2f}, "
              f"Chords={len(chords_in_measure)}")


def example_6_combine_sections():
    """Example 6: Combine different patterns for song form"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Song Form with Combined Patterns (Verse-Chorus-Bridge)")
    print("="*80)

    hr = HarmonicRhythm()

    # Verse: Sparse, calm (8 measures)
    verse = hr.generate_harmonic_rhythm(
        density=1.0,
        total_measures=8,
        variation=0.1
    )

    # Chorus: Dense, energetic (8 measures)
    chorus = hr.generate_harmonic_rhythm(
        density=2.5,
        total_measures=8,
        variation=0.2
    )

    # Bridge: Build tension (4 measures)
    bridge = hr.apply_harmonic_acceleration(
        progression_length=4,
        start_density=1.5,
        end_density=3.0,
        curve="exponential"
    )

    # Combine all sections
    song = hr.combine_patterns([verse, chorus, verse, chorus, bridge, chorus])

    print(f"\nComplete song structure:")
    print(f"  Total measures: {song.total_measures}")
    print(f"  Total chords: {len(song.chord_durations)}")
    print(f"  Average density: {song.average_density:.2f} chords/measure")

    # Analyze by section
    sections = [
        ("Verse 1", 0, 8),
        ("Chorus 1", 8, 16),
        ("Verse 2", 16, 24),
        ("Chorus 2", 24, 32),
        ("Bridge", 32, 36),
        ("Chorus 3", 36, 44)
    ]

    print("\nChords per section:")
    for name, start, end in sections:
        section_chords = [cd for cd in song.chord_durations
                         if start <= cd.measure < end]
        print(f"  {name:12s}: {len(section_chords):2d} chords")


def example_7_suspensions_and_anticipations():
    """Example 7: Add suspensions and anticipations"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Suspensions and Anticipations")
    print("="*80)

    hr = HarmonicRhythm()

    # Base pattern
    base = hr.generate_harmonic_rhythm(density=2.0, total_measures=8)

    # Add rhythmic interest
    enhanced = hr.add_suspensions(
        base,
        suspension_rate=0.3,
        anticipation_rate=0.2
    )

    suspensions = sum(1 for cd in enhanced.chord_durations if cd.is_suspension)
    anticipations = sum(1 for cd in enhanced.chord_durations if cd.is_anticipation)

    print(f"\nBase pattern: {len(base.chord_durations)} chords")
    print(f"Enhanced pattern: {len(enhanced.chord_durations)} chords")
    print(f"  Suspensions: {suspensions}")
    print(f"  Anticipations: {anticipations}")
    print(f"\nSuspensions delay chord changes; anticipations arrive early.")
    print("Both create rhythmic sophistication and forward motion.")


def example_8_analyze_existing():
    """Example 8: Analyze existing progression"""
    print("\n" + "="*80)
    print("EXAMPLE 8: Analyze Existing Chord Progression")
    print("="*80)

    hr = HarmonicRhythm()

    # Analyze a typical 12-bar blues progression
    blues_chords = [
        "C7", "C7", "C7", "C7",  # I (4 bars)
        "F7", "F7", "C7", "C7",  # IV-I (4 bars)
        "G7", "F7", "C7", "G7"   # V-IV-I-V (4 bars)
    ]
    blues_durations = [4.0] * 12  # 1 chord per measure (4 beats each)

    analysis = hr.analyze_chord_density(blues_chords, blues_durations)

    print("\n12-Bar Blues Analysis:")
    print(f"  Total chords: {analysis['total_chords']}")
    print(f"  Total measures: {analysis['total_measures']:.1f}")
    print(f"  Chords per measure: {analysis['chords_per_measure']:.2f}")
    print(f"  Density classification: {analysis['density_level']}")
    print(f"  Min/Max duration: {analysis['min_duration']:.1f} / {analysis['max_duration']:.1f} beats")


def example_9_midi_ready_output():
    """Example 9: MIDI-ready timing information"""
    print("\n" + "="*80)
    print("EXAMPLE 9: MIDI-Ready Timing Information")
    print("="*80)

    hr = HarmonicRhythm()

    pattern = hr.generate_harmonic_rhythm(density=2.0, total_measures=4)

    print(f"\nMIDI-Ready Chord Timing (BPM=120, 480 ticks/beat):")
    print(f"{'Chord':<8} {'Start Tick':<12} {'Duration Ticks':<15} {'Measure':<10}")
    print("-" * 50)

    TICKS_PER_BEAT = 480
    for cd in pattern.chord_durations:
        start_tick = int(cd.start_beat * TICKS_PER_BEAT)
        duration_ticks = int(cd.duration_beats * TICKS_PER_BEAT)

        print(f"{cd.chord_index:<8} {start_tick:<12} {duration_ticks:<15} {cd.measure:<10}")

    print("\nThis timing can be directly used for MIDI file generation.")


def example_10_detailed_pattern_display():
    """Example 10: Visual pattern display"""
    print("\n" + "="*80)
    print("EXAMPLE 10: Detailed Pattern Visualization")
    print("="*80)

    hr = HarmonicRhythm()

    # Create an interesting pattern
    pattern = hr.apply_harmonic_acceleration(
        progression_length=8,
        start_density=1.0,
        end_density=4.0,
        cadence_measures=2
    )

    # Add some suspensions
    pattern = hr.add_suspensions(pattern, suspension_rate=0.2, anticipation_rate=0.1)

    # Display using built-in formatter
    print_harmonic_rhythm_pattern(pattern)


def main():
    """Run all integration examples"""
    print("\n" + "="*80)
    print("HARMONIC RHYTHM MODULE - INTEGRATION EXAMPLES")
    print("Demonstrating real-world usage and integration patterns")
    print("="*80)

    examples = [
        example_1_basic_usage,
        example_2_genre_specific,
        example_3_with_chord_progression,
        example_4_cadential_acceleration,
        example_5_tension_based_pacing,
        example_6_combine_sections,
        example_7_suspensions_and_anticipations,
        example_8_analyze_existing,
        example_9_midi_ready_output,
        example_10_detailed_pattern_display
    ]

    for example_fn in examples:
        try:
            example_fn()
            print("\n✓ Example completed successfully")
        except Exception as e:
            print(f"\n✗ Example failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("ALL INTEGRATION EXAMPLES COMPLETED")
    print("="*80)
    print("\nNext steps:")
    print("  1. Integrate with chord progression generators")
    print("  2. Export to MIDI files with proper timing")
    print("  3. Combine with melody/bass generators")
    print("  4. Use in real-time composition systems")
    print("\nSee docs/HARMONIC_RHYTHM_README.md for full documentation.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
