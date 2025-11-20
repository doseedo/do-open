"""
AGENT 20: Rhythm Specialist - Example Usage
============================================

This demo shows how to use the RhythmSpecialist for advanced
rhythm generation and analysis.

Features demonstrated:
1. Polyrhythm generation (3:2, 4:3, 5:4, etc.)
2. Swing and groove application
3. Syncopation patterns
4. World rhythm patterns (clave, bell patterns)
5. Metric modulation
6. Odd meter patterns
7. Rhythmic tension curves
8. Complexity analysis

Author: Agent 20 Demo
"""

from pathlib import Path
from midi_generator.experts import RhythmSpecialist, get_rhythm_specialist
from midi_generator.experts.rhythm_specialist import RhythmicEvent, RhythmicEventType


def demo_polyrhythm():
    """Demonstrate polyrhythm generation"""
    print("\n" + "=" * 80)
    print("DEMO 1: POLYRHYTHM GENERATION")
    print("=" * 80)

    specialist = get_rhythm_specialist()

    # Generate various polyrhythms
    polyrhythms = [
        (3, 2),  # Triplets vs duplets (classic jazz)
        (4, 3),  # 4 against 3
        (5, 4),  # 5 against 4
        (7, 5),  # 7 against 5 (complex!)
    ]

    for ratio in polyrhythms:
        print(f"\n{ratio[0]}:{ratio[1]} Polyrhythm:")
        pattern = specialist.generate_polyrhythm(ratio, length_beats=4.0)

        print(f"  Voices: {len(pattern.voices)}")
        print(f"  Voice 1: {len(pattern.voices[0])} events")
        print(f"  Voice 2: {len(pattern.voices[1])} events")
        print(f"  Tension: {pattern.tension_level:.2f}")
        print(f"  Complexity: {'Simple' if pattern.tension_level < 0.3 else 'Moderate' if pattern.tension_level < 0.7 else 'Complex'}")


def demo_swing_and_groove():
    """Demonstrate swing and groove application"""
    print("\n" + "=" * 80)
    print("DEMO 2: SWING AND GROOVE")
    print("=" * 80)

    specialist = get_rhythm_specialist()

    # Create a simple straight pattern
    straight_pattern = [
        RhythmicEvent(onset_time=i * 0.5, duration=0.4, velocity=100)
        for i in range(8)
    ]

    print("\nOriginal straight pattern:")
    print(f"  Events: {len(straight_pattern)}")
    print(f"  Onsets: {[f'{e.onset_time:.2f}' for e in straight_pattern[:4]]}")

    # Apply different swing amounts
    for swing_amount in [0.0, 0.5, 1.0]:
        swung = specialist.apply_swing(straight_pattern, swing_amount=swing_amount)
        print(f"\nSwing amount {swing_amount:.1f}:")
        print(f"  Onsets: {[f'{e.onset_time:.2f}' for e in swung[:4]]}")

    # Apply groove templates
    print("\n\nGroove Templates:")
    groove_names = ['jazz_swing', 'shuffle', 'laid_back', 'pushed', 'drunk']

    for groove in groove_names:
        grooved = specialist.apply_groove_template(straight_pattern, groove)
        print(f"\n{groove}:")
        print(f"  Timing: {[f'{e.onset_time:.2f}' for e in grooved[:4]]}")
        print(f"  Velocities: {[e.velocity for e in grooved[:4]]}")


def demo_syncopation():
    """Demonstrate syncopation generation"""
    print("\n" + "=" * 80)
    print("DEMO 3: SYNCOPATION")
    print("=" * 80)

    specialist = get_rhythm_specialist()

    # Generate syncopated patterns with varying complexity
    for complexity in [0.3, 0.6, 0.9]:
        print(f"\nComplexity {complexity:.1f}:")
        pattern = specialist.generate_syncopation_pattern(
            length_beats=4.0,
            density=0.6,
            complexity=complexity
        )

        print(f"  Events: {len(pattern)}")
        print(f"  Accents: {sum(1 for e in pattern if e.event_type == RhythmicEventType.ACCENT)}")

        # Analyze complexity
        analysis = specialist.analyze_rhythmic_complexity(pattern)
        print(f"  Measured complexity: {analysis['complexity']:.2f}")
        print(f"  Syncopation score: {analysis['syncopation_score']:.2f}")


def demo_world_rhythms():
    """Demonstrate world rhythm patterns (clave)"""
    print("\n" + "=" * 80)
    print("DEMO 4: WORLD RHYTHM PATTERNS")
    print("=" * 80)

    specialist = get_rhythm_specialist()

    print("\nAvailable Clave Patterns:")
    available = specialist.get_available_claves()
    print(f"  {', '.join(available)}")

    print("\n" + "-" * 80)

    # Generate each clave pattern
    for clave_type in available:
        print(f"\n{clave_type.upper()}:")
        pattern = specialist.generate_clave(clave_type)

        print(f"  Events: {len(pattern)}")
        print(f"  Onsets: {[f'{e.onset_time:.2f}' for e in pattern]}")

        # Get pattern info
        clave_info = specialist.clave_patterns[clave_type]
        print(f"  Origin: {clave_info.origin}")
        print(f"  Feeling: {clave_info.feeling}")
        print(f"  Length: {clave_info.length_beats} beats")


def demo_metric_modulation():
    """Demonstrate metric modulation"""
    print("\n" + "=" * 80)
    print("DEMO 5: METRIC MODULATION")
    print("=" * 80)

    specialist = get_rhythm_specialist()

    # Create a simple pattern
    pattern = [
        RhythmicEvent(onset_time=i * 0.5, duration=0.4)
        for i in range(16)
    ]

    print("\nOriginal pattern (16 events):")
    print(f"  Onsets: {[f'{e.onset_time:.1f}' for e in pattern[:8]]}")

    # Apply metric modulation at beat 4
    modulated = specialist.apply_metric_modulation(
        pattern,
        ratio=(3, 2),  # New tempo is 3/2 of old (tempo increases)
        modulation_point=4.0
    )

    print("\nAfter 3:2 modulation at beat 4:")
    print(f"  Onsets: {[f'{e.onset_time:.1f}' for e in modulated[:8]]}")
    print(f"  Notice: Events after beat 4 are closer together (faster tempo)")


def demo_odd_meters():
    """Demonstrate odd meter patterns"""
    print("\n" + "=" * 80)
    print("DEMO 6: ODD METER PATTERNS")
    print("=" * 80)

    specialist = get_rhythm_specialist()

    # Generate patterns in various odd meters
    odd_meters = [
        ((5, 4), [3, 2], "3+2"),
        ((5, 4), [2, 3], "2+3"),
        ((7, 8), [3, 2, 2], "3+2+2"),
        ((7, 8), [2, 2, 3], "2+2+3"),
        ((11, 8), [3, 3, 3, 2], "3+3+3+2"),
    ]

    for time_sig, grouping, description in odd_meters:
        print(f"\n{time_sig[0]}/{time_sig[1]} ({description}):")
        pattern = specialist.generate_odd_meter_pattern(
            time_signature=time_sig,
            length_bars=2,
            grouping=grouping
        )

        print(f"  Events: {len(pattern)}")
        print(f"  Grouping: {grouping}")
        print(f"  Onsets: {[f'{e.onset_time:.2f}' for e in pattern[:6]]}")


def demo_tension_curves():
    """Demonstrate rhythmic tension curves"""
    print("\n" + "=" * 80)
    print("DEMO 7: RHYTHMIC TENSION CURVES")
    print("=" * 80)

    specialist = get_rhythm_specialist()

    # Create base pattern
    base_pattern = [
        RhythmicEvent(onset_time=i * 0.5, duration=0.4, velocity=80)
        for i in range(16)
    ]

    curve_types = ['buildup', 'breakdown', 'arc', 'valley']

    for curve_type in curve_types:
        print(f"\n{curve_type.upper()}:")
        modified = specialist.apply_tension_curve(base_pattern, curve_type)

        # Show velocity progression
        velocities = [e.velocity for e in modified]
        print(f"  Velocities: {velocities[:8]} ...")
        print(f"  Min velocity: {min(velocities)}")
        print(f"  Max velocity: {max(velocities)}")


def demo_humanization():
    """Demonstrate timing humanization"""
    print("\n" + "=" * 80)
    print("DEMO 8: HUMANIZATION")
    print("=" * 80)

    specialist = get_rhythm_specialist()

    # Create mechanical pattern
    mechanical = [
        RhythmicEvent(onset_time=i * 0.25, duration=0.2, velocity=100)
        for i in range(16)
    ]

    print("\nMechanical pattern:")
    print(f"  Onsets: {[f'{e.onset_time:.3f}' for e in mechanical[:8]]}")
    print(f"  Velocities: {[e.velocity for e in mechanical[:8]]}")

    # Apply humanization
    humanized = specialist.humanize_timing(mechanical, amount=0.7, randomness=0.5)

    print("\nHumanized pattern (amount=0.7, randomness=0.5):")
    print(f"  Onsets: {[f'{e.onset_time:.3f}' for e in humanized[:8]]}")
    print(f"  Velocities: {[e.velocity for e in humanized[:8]]}")
    print(f"  Notice: Slight timing variations and velocity changes")


def demo_export_to_midi():
    """Demonstrate MIDI export"""
    print("\n" + "=" * 80)
    print("DEMO 9: EXPORT TO MIDI")
    print("=" * 80)

    specialist = get_rhythm_specialist()
    output_dir = Path("output/agent20_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and export various patterns
    examples = [
        ('polyrhythm_3_2', specialist.generate_polyrhythm((3, 2), 8.0).voices[0]),
        ('son_clave', specialist.generate_clave('son')),
        ('syncopated', specialist.generate_syncopation_pattern(8.0, 0.6, 0.7)),
        ('7_8_pattern', specialist.generate_odd_meter_pattern((7, 8), 4)),
    ]

    for name, events in examples:
        output_path = output_dir / f"{name}.mid"
        specialist.events_to_midi(events, output_path, tempo=120, pitch=60)
        print(f"  Exported: {output_path}")


def demo_analysis():
    """Demonstrate rhythm analysis"""
    print("\n" + "=" * 80)
    print("DEMO 10: RHYTHM ANALYSIS")
    print("=" * 80)

    specialist = get_rhythm_specialist()

    # Create test patterns
    patterns = {
        'simple': [
            RhythmicEvent(onset_time=i, duration=0.8)
            for i in range(8)
        ],
        'complex': specialist.generate_syncopation_pattern(8.0, 0.7, 0.9),
    }

    for name, pattern in patterns.items():
        print(f"\n{name.upper()} pattern:")
        analysis = specialist.analyze_rhythmic_complexity(pattern)

        print(f"  Complexity: {analysis['complexity']:.3f}")
        print(f"  IOI variability: {analysis['ioi_variability']:.3f}")
        print(f"  Velocity variability: {analysis['velocity_variability']:.3f}")
        print(f"  Syncopation score: {analysis['syncopation_score']:.3f}")
        print(f"  Density: {analysis['density']} events")


def main():
    """Run all demos"""
    print("=" * 80)
    print("AGENT 20: RHYTHM SPECIALIST - COMPREHENSIVE DEMO")
    print("=" * 80)

    demo_polyrhythm()
    demo_swing_and_groove()
    demo_syncopation()
    demo_world_rhythms()
    demo_metric_modulation()
    demo_odd_meters()
    demo_tension_curves()
    demo_humanization()
    demo_export_to_midi()
    demo_analysis()

    print("\n" + "=" * 80)
    print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nThe Rhythm Specialist provides:")
    print("  • Polyrhythm generation and analysis")
    print("  • Swing and groove quantization")
    print("  • Syncopation pattern creation")
    print("  • World rhythm patterns (clave, bell patterns)")
    print("  • Metric modulation")
    print("  • Odd meter pattern generation")
    print("  • Rhythmic tension curves")
    print("  • Timing humanization")
    print("  • Comprehensive rhythm analysis")
    print("\nCheck the output/agent20_examples/ directory for MIDI files!")
    print("=" * 80)


if __name__ == '__main__':
    main()
