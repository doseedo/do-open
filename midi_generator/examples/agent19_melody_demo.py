"""
Agent 19: Melody Specialist Demo
=================================

Demonstrates the capabilities of the Melody Specialist:
1. Motif development with multiple transformation techniques
2. Melodic sequence generation
3. Contour optimization
4. Ornamentation system
5. Phrase structure analysis
6. Comprehensive melodic analysis

Author: Agent 19 Demo
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experts.melody_specialist import (
    MelodySpecialist,
    Note, Motif, ContourShape,
    MotifTransformation, SequenceType, OrnamentType
)


def create_test_melody() -> list:
    """Create a test melody"""
    # C major scale ascending and descending
    pitches = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60]
    notes = []
    current_time = 0.0

    for i, pitch in enumerate(pitches):
        duration = 0.5 if i % 2 == 0 else 0.5
        velocity = 80 + (i % 3) * 10  # Vary velocity
        notes.append(Note(
            pitch=pitch,
            duration=duration,
            velocity=velocity,
            start_time=current_time
        ))
        current_time += duration

    return notes


def create_jazz_motif() -> Motif:
    """Create a jazz motif for development"""
    # Classic bebop lick
    pitches = [60, 64, 67, 65, 62, 69, 67]
    notes = []
    current_time = 0.0

    for pitch in pitches:
        notes.append(Note(
            pitch=pitch,
            duration=0.25,
            velocity=90,
            start_time=current_time
        ))
        current_time += 0.25

    return Motif(notes=notes, name="bebop_lick", category="jazz")


def demo_motif_development():
    """Demonstrate motif development techniques"""
    print("\n" + "="*70)
    print("DEMO 1: MOTIF DEVELOPMENT")
    print("="*70)

    specialist = MelodySpecialist(key="C", mode="major")
    motif = create_jazz_motif()

    print(f"\nOriginal Motif: {len(motif)} notes")
    print(f"  Pitches: {[n.pitch for n in motif.notes]}")
    print(f"  Intervals: {motif.pitch_intervals()}")
    print(f"  Contour: {motif.contour()}")

    # Test all transformation techniques
    techniques = [
        (MotifTransformation.INVERSION, "Inversion (mirror)"),
        (MotifTransformation.RETROGRADE, "Retrograde (reverse)"),
        (MotifTransformation.RETROGRADE_INVERSION, "Retrograde Inversion"),
        (MotifTransformation.AUGMENTATION, "Augmentation (2x duration)"),
        (MotifTransformation.DIMINUTION, "Diminution (0.5x duration)"),
        (MotifTransformation.TRANSPOSITION, "Transposition (+5 semitones)"),
        (MotifTransformation.INTERVALLIC_EXPANSION, "Intervallic Expansion (1.5x)"),
        (MotifTransformation.INTERVALLIC_CONTRACTION, "Intervallic Contraction (0.5x)"),
    ]

    for technique, description in techniques:
        variations = specialist.develop_motif(motif, [technique], n_variations=1)
        if variations:
            var = variations[0]
            print(f"\n{description}:")
            print(f"  Pitches: {[n.pitch for n in var.notes]}")
            if len(var.notes) > 1:
                print(f"  Intervals: {var.pitch_intervals()}")


def demo_sequence_generation():
    """Demonstrate melodic sequence generation"""
    print("\n" + "="*70)
    print("DEMO 2: SEQUENCE GENERATION")
    print("="*70)

    specialist = MelodySpecialist(key="G", mode="major")

    # Create simple pattern
    pattern_notes = [
        Note(67, 0.5, 85, 0.0),
        Note(69, 0.5, 80, 0.5),
        Note(71, 0.5, 90, 1.0),
        Note(72, 1.0, 85, 1.5)
    ]
    pattern = Motif(notes=pattern_notes, name="sequence_pattern")

    print(f"\nPattern: {[n.pitch for n in pattern.notes]}")

    # Generate different sequence types
    sequence_types = [
        (SequenceType.ASCENDING, "Ascending by 2 semitones"),
        (SequenceType.DESCENDING, "Descending by 2 semitones"),
        (SequenceType.TONAL, "Tonal (by scale degrees)"),
        (SequenceType.CHROMATIC, "Chromatic (by semitones)"),
    ]

    for seq_type, description in sequence_types:
        sequence = specialist.generate_sequence(
            pattern,
            seq_type,
            repetitions=3,
            interval=2
        )
        print(f"\n{description}:")
        print(f"  Generated {len(sequence)} notes")
        pitches = [n.pitch for n in sequence]
        print(f"  Pitches: {pitches[:12]}... (showing first 12)")


def demo_contour_optimization():
    """Demonstrate contour optimization"""
    print("\n" + "="*70)
    print("DEMO 3: CONTOUR OPTIMIZATION")
    print("="*70)

    specialist = MelodySpecialist()
    melody = create_test_melody()

    print(f"\nOriginal Melody: {len(melody)} notes")
    pitches = [n.pitch for n in melody]
    print(f"  Pitches: {pitches}")

    # Analyze original contour
    original_analysis = specialist.analyze_contour(melody)
    print(f"\nOriginal Contour Analysis:")
    print(f"  Shape: {original_analysis['shape'].value}")
    print(f"  Direction Changes: {original_analysis['direction_changes']}")
    print(f"  Apex Position: {original_analysis['apex_position']:.2f}")

    # Optimize to different shapes
    shapes = [
        ContourShape.ARCH,
        ContourShape.INVERTED_ARCH,
        ContourShape.ASCENDING,
        ContourShape.WAVE
    ]

    for shape in shapes:
        optimized = specialist.optimize_contour(melody, shape, smoothing=0.7)
        opt_pitches = [n.pitch for n in optimized]
        print(f"\nOptimized to {shape.value}:")
        print(f"  Pitches: {opt_pitches}")


def demo_ornamentation():
    """Demonstrate ornamentation system"""
    print("\n" + "="*70)
    print("DEMO 4: ORNAMENTATION")
    print("="*70)

    specialist = MelodySpecialist()

    # Create simple melody for ornamentation
    simple_melody = [
        Note(60, 2.0, 80, 0.0),
        Note(64, 2.0, 85, 2.0),
        Note(67, 2.0, 90, 4.0),
        Note(72, 4.0, 85, 6.0)
    ]

    print(f"\nOriginal Melody: {len(simple_melody)} notes")
    print(f"  Pitches: {[n.pitch for n in simple_melody]}")
    print(f"  Durations: {[n.duration for n in simple_melody]}")

    # Apply different ornaments
    ornament_sets = [
        ([OrnamentType.TRILL], "Trills"),
        ([OrnamentType.TURN], "Turns"),
        ([OrnamentType.MORDENT], "Mordents"),
        ([OrnamentType.APPOGGIATURA], "Appoggiaturas"),
        ([OrnamentType.GRACE_NOTE], "Grace Notes"),
    ]

    for ornaments, description in ornament_sets:
        ornamented = specialist.add_ornamentation(
            simple_melody,
            ornaments,
            density=0.75
        )
        print(f"\n{description}:")
        print(f"  Original notes: {len(simple_melody)}")
        print(f"  Ornamented notes: {len(ornamented)}")
        if len(ornamented) <= 20:
            print(f"  Pitches: {[n.pitch for n in ornamented]}")


def demo_phrase_analysis():
    """Demonstrate phrase structure analysis"""
    print("\n" + "="*70)
    print("DEMO 5: PHRASE STRUCTURE ANALYSIS")
    print("="*70)

    specialist = MelodySpecialist()

    # Create melody with multiple phrases
    melody = create_test_melody() * 2  # Repeat for longer melody
    print(f"\nMelody: {len(melody)} notes")

    phrases = specialist.analyze_phrase_structure(melody, phrase_length=4.0)

    print(f"\nDetected {len(phrases)} phrases:")
    for i, phrase in enumerate(phrases):
        duration = phrase.end_time - phrase.start_time
        note_count = len(phrase.all_notes())
        print(f"\n  Phrase {i+1}:")
        print(f"    Time: {phrase.start_time:.1f} - {phrase.end_time:.1f} (duration: {duration:.1f})")
        print(f"    Notes: {note_count}")
        print(f"    Motifs: {len(phrase.motifs)}")
        print(f"    Cadence: {phrase.cadence_type}")


def demo_comprehensive_analysis():
    """Demonstrate comprehensive melodic analysis"""
    print("\n" + "="*70)
    print("DEMO 6: COMPREHENSIVE MELODIC ANALYSIS")
    print("="*70)

    specialist = MelodySpecialist()
    melody = create_test_melody()

    analysis = specialist.analyze_melody(melody)

    print(f"\nMelodic Analysis Report:")
    print(f"  Contour Shape: {analysis.contour_shape.value}")
    print(f"  Range: {analysis.range_semitones} semitones")
    print(f"  Tessitura (average pitch): MIDI {analysis.tessitura}")
    print(f"  Climax Position: {analysis.climax_position:.1%} through melody")
    print(f"  Stepwise Motion: {analysis.stepwise_motion_ratio:.1%}")
    print(f"  Chromaticism Score: {analysis.chromaticism_score:.1%}")
    print(f"  Direction Changes: {analysis.direction_changes}")
    print(f"  Leap Count: {analysis.leap_count}")
    print(f"  Sequence Detected: {'Yes' if analysis.sequence_detected else 'No'}")

    print(f"\n  Intervallic Profile:")
    for interval, count in sorted(analysis.intervallic_profile.items())[:10]:
        print(f"    {interval:+3d} semitones: {count} times")

    print(f"\n  Phrase Structure:")
    print(f"    Total Phrases: {len(analysis.phrases)}")
    print(f"    Total Motifs: {len(analysis.motifs)}")

    if analysis.phrases:
        print(f"\n  Phrase Details:")
        for i, phrase in enumerate(analysis.phrases[:3]):  # Show first 3
            print(f"    Phrase {i+1}: {len(phrase.motifs)} motifs, cadence={phrase.cadence_type}")


def demo_statistics():
    """Show processing statistics"""
    print("\n" + "="*70)
    print("DEMO 7: PROCESSING STATISTICS")
    print("="*70)

    specialist = MelodySpecialist()

    # Run various operations
    motif = create_jazz_motif()
    specialist.develop_motif(motif, [MotifTransformation.INVERSION], n_variations=5)
    specialist.generate_sequence(motif, SequenceType.ASCENDING, repetitions=4)

    melody = create_test_melody()
    specialist.add_ornamentation(
        melody,
        [OrnamentType.TRILL, OrnamentType.TURN],
        density=0.5
    )
    specialist.analyze_phrase_structure(melody)
    specialist.analyze_melody(melody)

    stats = specialist.get_statistics()

    print("\nProcessing Statistics:")
    print(f"  Motifs Developed: {stats['motifs_developed']}")
    print(f"  Sequences Generated: {stats['sequences_generated']}")
    print(f"  Ornaments Added: {stats['ornaments_added']}")
    print(f"  Phrases Analyzed: {stats['phrases_analyzed']}")


def main():
    """Run all demos"""
    print("="*70)
    print("AGENT 19: MELODY SPECIALIST - COMPREHENSIVE DEMO")
    print("="*70)
    print("\nThis demo showcases all capabilities of the Melody Specialist:")
    print("  1. Motif Development (10+ transformation techniques)")
    print("  2. Sequence Generation (6 sequence types)")
    print("  3. Contour Optimization (7 contour shapes)")
    print("  4. Ornamentation (10 ornament types)")
    print("  5. Phrase Structure Analysis")
    print("  6. Comprehensive Melodic Analysis")
    print("  7. Processing Statistics")

    try:
        demo_motif_development()
        demo_sequence_generation()
        demo_contour_optimization()
        demo_ornamentation()
        demo_phrase_analysis()
        demo_comprehensive_analysis()
        demo_statistics()

        print("\n" + "="*70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nThe Melody Specialist provides:")
        print("  ✓ 50+ specialized melody parameters")
        print("  ✓ 10 motif transformation techniques")
        print("  ✓ 6 types of melodic sequences")
        print("  ✓ 10 ornament types")
        print("  ✓ 7 contour shapes")
        print("  ✓ Comprehensive phrase analysis")
        print("  ✓ Full integration with Agent 4's melody parameters")
        print()

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
