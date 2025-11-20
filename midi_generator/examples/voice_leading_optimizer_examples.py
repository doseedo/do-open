#!/usr/bin/env python3
"""
Voice Leading Optimizer - Practical Integration Examples
=========================================================

Demonstrates how to use the Voice Leading Optimizer in real-world scenarios:
1. Big band sax soli
2. Jazz piano comping
3. String quartet
4. Brass section
5. Integration with existing arrangement engine

Author: Agent 11 - Voice Leading Optimization Engine
"""

import sys
from pathlib import Path
import importlib.util

# Import voice_leading_optimizer directly to avoid __init__.py imports
optimizer_path = Path(__file__).parent.parent / 'transformation' / 'voice_leading_optimizer.py'
spec = importlib.util.spec_from_file_location("voice_leading_optimizer", optimizer_path)
vlo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vlo)

# Extract classes
VoiceLeadingOptimizer = vlo.VoiceLeadingOptimizer
VoiceRange = vlo.VoiceRange
VoicingType = vlo.VoicingType
MinimizationStrategy = vlo.MinimizationStrategy


# ============================================================================
# EXAMPLE 1: BIG BAND SAX SOLI
# ============================================================================

def example_big_band_sax_soli():
    """
    Create professional sax soli voicing for a jazz standard.

    Uses 5-part drop-2 voicings (industry standard).
    Target: Average motion < 3 semitones
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: BIG BAND SAX SOLI (5-part Drop-2)")
    print("=" * 80)

    # Classic "Rhythm Changes" A section (I-VI-ii-V)
    chords = [
        {'root': 10, 'quality': 'maj7'},   # Bbmaj7
        {'root': 7, 'quality': 'dom7'},    # G7
        {'root': 0, 'quality': 'min7'},    # Cm7
        {'root': 5, 'quality': 'dom7'},    # F7
    ]

    # Sax section ranges (Alto 1, Alto 2, Tenor 1, Tenor 2, Bari)
    sax_ranges = [
        VoiceRange(46, 67, 49, 64),   # Bari sax (E2-G4, comfortable F2-E4)
        VoiceRange(47, 76, 50, 70),   # Tenor 2 (B2-E5, comfortable D3-Bb4)
        VoiceRange(47, 76, 50, 70),   # Tenor 1
        VoiceRange(52, 81, 55, 76),   # Alto 2 (E3-A5, comfortable G3-E5)
        VoiceRange(52, 81, 55, 76),   # Alto 1
    ]

    result = VoiceLeadingOptimizer.optimize_chord_sequence(
        chords=chords,
        num_voices=5,
        voice_ranges=sax_ranges,
        voicing_types=[VoicingType.DROP_2],  # Industry standard
        minimize=MinimizationStrategy.TOTAL_MOTION
    )

    # Print results
    chord_names = ['Bbmaj7', 'G7', 'Cm7', 'F7']
    sax_names = ['Bari', 'Tenor 2', 'Tenor 1', 'Alto 2', 'Alto 1']

    print("\nOptimized Sax Soli Voicings:")
    print("-" * 80)
    for i, voicing in enumerate(result.voicings):
        print(f"\n{chord_names[i]}:")
        for j, pitch in enumerate(voicing.pitches):
            note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][pitch % 12]
            octave = (pitch // 12) - 1
            print(f"  {sax_names[j]:>8}: {note_name}{octave} (MIDI {pitch})")

    print(f"\n📊 Voice Leading Quality:")
    print(f"  Average motion: {result.avg_motion:.2f} semitones {'✅' if result.avg_motion < 3.5 else '⚠️'}")
    print(f"  Total motion: {result.total_motion} semitones")
    print(f"  Maximum leap: {result.max_leap} semitones")
    print(f"  Common tones retained: {result.common_tones_retained}")

    return result


# ============================================================================
# EXAMPLE 2: JAZZ PIANO COMPING (BILL EVANS STYLE)
# ============================================================================

def example_jazz_piano_comping():
    """
    Create Bill Evans-style rootless voicings for jazz piano.

    Uses close position voicings (3-5-7-9 or 7-9-3-5 patterns).
    Target: Very smooth voice leading (< 3 semitones)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: JAZZ PIANO COMPING (Bill Evans Rootless Voicings)")
    print("=" * 80)

    # Classic ii-V-I in C major
    chords = [
        {'root': 2, 'quality': 'min7'},    # Dm7
        {'root': 7, 'quality': 'dom7'},    # G7
        {'root': 0, 'quality': 'maj7'},    # Cmaj7
    ]

    # Piano left-hand comping range
    piano_ranges = [
        VoiceRange(48, 72, 52, 68),   # ~C3 to C5, comfortable E3 to G#4
        VoiceRange(52, 76, 55, 72),
        VoiceRange(55, 79, 60, 75),
        VoiceRange(60, 84, 64, 80),
    ]

    result = VoiceLeadingOptimizer.optimize_chord_sequence(
        chords=chords,
        num_voices=4,
        voice_ranges=piano_ranges,
        voicing_types=[VoicingType.CLOSE],  # Rootless work well as close
        minimize=MinimizationStrategy.TOTAL_MOTION
    )

    # Print results
    chord_names = ['Dm7', 'G7', 'Cmaj7']

    print("\nRootless Voicings (Left Hand):")
    print("-" * 80)
    for i, voicing in enumerate(result.voicings):
        notes = []
        for pitch in voicing.pitches:
            note_name = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'][pitch % 12]
            octave = (pitch // 12) - 1
            notes.append(f"{note_name}{octave}")
        print(f"{chord_names[i]:>6}: [{', '.join(notes)}]")

    print(f"\n📊 Voice Leading Quality (Bill Evans standard):")
    print(f"  Average motion: {result.avg_motion:.2f} semitones {'✅ Excellent!' if result.avg_motion < 2.5 else '✅ Good' if result.avg_motion < 3.5 else '⚠️'}")

    return result


# ============================================================================
# EXAMPLE 3: STRING QUARTET (CLASSICAL STYLE)
# ============================================================================

def example_string_quartet():
    """
    Create string quartet voicing with classical voice leading.

    Uses close/open position with emphasis on outer voices.
    Target: Minimal voice movement, smooth outer voices
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: STRING QUARTET (Classical Voice Leading)")
    print("=" * 80)

    # I-IV-V-I in F major
    chords = [
        {'root': 5, 'quality': 'major'},   # F major
        {'root': 10, 'quality': 'major'},  # Bb major
        {'root': 0, 'quality': 'major'},   # C major
        {'root': 5, 'quality': 'major'},   # F major
    ]

    # String quartet ranges
    string_ranges = [
        VoiceRange(36, 72, 48, 60),   # Cello (C2-C5, comfortable C3-C4)
        VoiceRange(48, 84, 55, 72),   # Viola (C3-C6, comfortable G3-C5)
        VoiceRange(55, 91, 60, 84),   # Violin II (G3-G6, comfortable C4-C6)
        VoiceRange(55, 96, 64, 88),   # Violin I (G3-C7, comfortable E4-E6)
    ]

    result = VoiceLeadingOptimizer.optimize_chord_sequence(
        chords=chords,
        num_voices=4,
        voice_ranges=string_ranges,
        voicing_types=[VoicingType.CLOSE, VoicingType.OPEN],
        minimize=MinimizationStrategy.WEIGHTED  # Emphasize outer voices
    )

    # Print results
    chord_names = ['F', 'Bb', 'C', 'F']
    instrument_names = ['Cello', 'Viola', 'Violin II', 'Violin I']

    print("\nString Quartet Voicings:")
    print("-" * 80)
    for i, voicing in enumerate(result.voicings):
        print(f"\n{chord_names[i]}:")
        for j, pitch in enumerate(voicing.pitches):
            note_name = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'][pitch % 12]
            octave = (pitch // 12) - 1
            print(f"  {instrument_names[j]:>10}: {note_name}{octave}")

    print(f"\n📊 Voice Leading Quality:")
    print(f"  Average motion: {result.avg_motion:.2f} semitones")
    print(f"  Outer voice emphasis: ✅ (using WEIGHTED strategy)")

    return result


# ============================================================================
# EXAMPLE 4: COMPARING VOICING TYPES
# ============================================================================

def example_compare_voicing_types():
    """
    Compare different voicing types on the same progression.

    Shows how voicing type affects sound and voice leading.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: COMPARING VOICING TYPES (Same Progression)")
    print("=" * 80)

    # Simple progression (I-IV-V-I)
    chords = [
        {'root': 0, 'quality': 'maj7'},
        {'root': 5, 'quality': 'maj7'},
        {'root': 7, 'quality': 'dom7'},
        {'root': 0, 'quality': 'maj7'},
    ]

    # Generic 4-voice ranges
    ranges = [
        VoiceRange(48, 72, 52, 68),
        VoiceRange(52, 76, 55, 72),
        VoiceRange(60, 84, 64, 80),
        VoiceRange(64, 88, 67, 84),
    ]

    voicing_types = [
        VoicingType.CLOSE,
        VoicingType.DROP_2,
        VoicingType.SPREAD,
    ]

    print("\nComparing voicing types:")
    print("-" * 80)

    for v_type in voicing_types:
        result = VoiceLeadingOptimizer.optimize_chord_sequence(
            chords=chords,
            num_voices=4,
            voice_ranges=ranges,
            voicing_types=[v_type],
            minimize=MinimizationStrategy.TOTAL_MOTION
        )

        print(f"\n{v_type.value.upper()}:")
        print(f"  Average motion: {result.avg_motion:.2f} semitones")
        print(f"  Example voicing (Cmaj7): {result.voicings[0].pitches}")
        print(f"  Spacing: {result.voicings[0].pitches[-1] - result.voicings[0].pitches[0]} semitones")


# ============================================================================
# EXAMPLE 5: LONG PROGRESSION (CIRCLE OF FIFTHS)
# ============================================================================

def example_long_progression():
    """
    Optimize a longer progression through the circle of fifths.

    Demonstrates scalability and quality on complex progressions.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: LONG PROGRESSION (Circle of Fifths)")
    print("=" * 80)

    # Circle of fifths: Dm7-G7-Cmaj7-Am7-Dm7-G7-Cmaj7
    chords = [
        {'root': 2, 'quality': 'min7'},    # Dm7
        {'root': 7, 'quality': 'dom7'},    # G7
        {'root': 0, 'quality': 'maj7'},    # Cmaj7
        {'root': 9, 'quality': 'min7'},    # Am7
        {'root': 2, 'quality': 'min7'},    # Dm7
        {'root': 7, 'quality': 'dom7'},    # G7
        {'root': 0, 'quality': 'maj7'},    # Cmaj7
    ]

    ranges = [
        VoiceRange(48, 72, 52, 68),
        VoiceRange(52, 76, 55, 72),
        VoiceRange(60, 84, 64, 80),
        VoiceRange(64, 88, 67, 84),
    ]

    result = VoiceLeadingOptimizer.optimize_chord_sequence(
        chords=chords,
        num_voices=4,
        voice_ranges=ranges,
        voicing_types=[VoicingType.DROP_2],
        minimize=MinimizationStrategy.TOTAL_MOTION
    )

    chord_names = ['Dm7', 'G7', 'Cmaj7', 'Am7', 'Dm7', 'G7', 'Cmaj7']

    print("\nVoicing Sequence:")
    print("-" * 80)
    for i, (name, voicing) in enumerate(zip(chord_names, result.voicings)):
        motion = f"({result.motion_per_step[i-1]:.1f})" if i > 0 else ""
        print(f"{i+1}. {name:>6} {motion:>6}: {voicing.pitches}")

    print(f"\n📊 Overall Quality:")
    print(f"  Total motion: {result.total_motion} semitones across 6 changes")
    print(f"  Average motion: {result.avg_motion:.2f} semitones")
    print(f"  Maximum leap: {result.max_leap} semitones")
    print(f"  Quality: {'✅ Excellent' if result.avg_motion < 3 else '✅ Good' if result.avg_motion < 4 else '⚠️ Acceptable'}")

    return result


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

def run_all_examples():
    """Run all integration examples"""
    print("\n" + "=" * 80)
    print("VOICE LEADING OPTIMIZER - PRACTICAL INTEGRATION EXAMPLES")
    print("=" * 80)

    examples = [
        example_big_band_sax_soli,
        example_jazz_piano_comping,
        example_string_quartet,
        example_compare_voicing_types,
        example_long_progression,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n❌ Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ All examples completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review the generated voicings above")
    print("  2. Integrate into your arrangement engine")
    print("  3. Export to MIDI and listen to results")
    print("  4. Adjust parameters for your specific needs")
    print("\nDocumentation: docs/VOICE_LEADING_OPTIMIZER_GUIDE.md")
    print("=" * 80)


if __name__ == "__main__":
    run_all_examples()
