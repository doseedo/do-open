#!/usr/bin/env python3
"""
Agent 22: Dynamics Specialist - Demonstration
==============================================

Comprehensive demonstration of the Dynamics Specialist capabilities:
1. ADSR envelope generation and application
2. Dynamic curves (crescendo, diminuendo, custom)
3. Humanization (velocity and timing)
4. Voice balancing
5. Articulation-dynamics coupling
6. Inverse dynamics analysis

Usage:
    python agent22_dynamics_demo.py

Author: Agent 22 - Dynamics Specialist
License: MIT
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experts.dynamics_specialist import (
    DynamicsSpecialist,
    ADSREnvelope,
    DynamicCurve,
    DynamicCurveType,
    DynamicDirection,
    DynamicsProfile,
    Note,
    ArticulationType,
    create_default_profile,
    create_expressive_profile,
    create_mechanical_profile,
)


def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_adsr_envelopes():
    """Demonstrate ADSR envelope generation"""
    print_section("1. ADSR ENVELOPE DEMONSTRATION")

    specialist = DynamicsSpecialist(seed=42)

    # Create different ADSR envelopes
    envelopes = {
        "Piano (fast attack)": ADSREnvelope(
            attack_time=0.01,
            decay_time=0.2,
            sustain_level=0.6,
            release_time=0.3
        ),
        "Strings (slow attack)": ADSREnvelope(
            attack_time=0.3,
            decay_time=0.2,
            sustain_level=0.8,
            release_time=0.5
        ),
        "Organ (sustained)": ADSREnvelope(
            attack_time=0.05,
            decay_time=0.05,
            sustain_level=0.9,
            release_time=0.1
        ),
        "Plucked (percussive)": ADSREnvelope(
            attack_time=0.005,
            decay_time=0.5,
            sustain_level=0.2,
            release_time=0.1
        ),
    }

    for name, envelope in envelopes.items():
        curve = specialist.generate_adsr_envelope(envelope, note_duration=2.0)
        print(f"  {name}")
        print(f"    Attack:  {envelope.attack_time:.3f}s")
        print(f"    Decay:   {envelope.decay_time:.3f}s")
        print(f"    Sustain: {envelope.sustain_level:.2f}")
        print(f"    Release: {envelope.release_time:.3f}s")
        print(f"    Envelope curve: {len(curve)} samples, peak={curve.max():.2f}")
        print()

    # Apply ADSR to notes
    test_note = Note(
        pitch=60,
        velocity=100,
        start_time=0.0,
        end_time=2.0,
        duration=2.0
    )

    print("  Applying ADSR to test note (C4, vel=100, dur=2.0s):")
    for name, envelope in list(envelopes.items())[:2]:
        modified = specialist.apply_adsr_to_note(test_note, envelope)
        print(f"    {name}: velocity {test_note.velocity} → {modified.velocity}")

    print()
    print("✅ ADSR demonstration complete")


def demo_dynamic_curves():
    """Demonstrate dynamic curve generation"""
    print_section("2. DYNAMIC CURVE DEMONSTRATION")

    specialist = DynamicsSpecialist(seed=42)

    # Test different curve types
    curve_types = [
        (DynamicCurveType.LINEAR, "Linear"),
        (DynamicCurveType.EXPONENTIAL, "Exponential"),
        (DynamicCurveType.LOGARITHMIC, "Logarithmic"),
        (DynamicCurveType.SIGMOID, "Sigmoid (S-curve)"),
        (DynamicCurveType.PARABOLIC, "Parabolic"),
    ]

    print("  Crescendo Curves (0.3 → 1.0):")
    for curve_type, name in curve_types:
        curve_spec = DynamicCurve(
            curve_type=curve_type,
            direction=DynamicDirection.CRESCENDO,
            start_level=0.3,
            end_level=1.0,
            duration=4.0,
            shape_factor=2.0
        )
        curve_values = specialist.generate_dynamic_curve(curve_spec, num_points=50)
        print(f"    {name:20s}: start={curve_values[0]:.2f}, mid={curve_values[25]:.2f}, end={curve_values[-1]:.2f}")

    print()
    print("  Diminuendo Curves (1.0 → 0.3):")
    for curve_type, name in curve_types:
        curve_spec = DynamicCurve(
            curve_type=curve_type,
            direction=DynamicDirection.DIMINUENDO,
            start_level=1.0,
            end_level=0.3,
            duration=4.0,
            shape_factor=2.0
        )
        curve_values = specialist.generate_dynamic_curve(curve_spec, num_points=50)
        print(f"    {name:20s}: start={curve_values[0]:.2f}, mid={curve_values[25]:.2f}, end={curve_values[-1]:.2f}")

    # Apply to notes
    print()
    print("  Applying crescendo to note sequence:")
    notes = [
        Note(pitch=60 + i*2, velocity=70, start_time=i*0.5, end_time=(i+1)*0.5, duration=0.5)
        for i in range(8)
    ]

    modified = specialist.apply_crescendo(notes, start_level=0.5, end_level=1.0)
    print(f"    Original velocities: {[n.velocity for n in notes]}")
    print(f"    After crescendo:     {[n.velocity for n in modified]}")

    print()
    print("✅ Dynamic curve demonstration complete")


def demo_humanization():
    """Demonstrate humanization features"""
    print_section("3. HUMANIZATION DEMONSTRATION")

    specialist = DynamicsSpecialist(seed=42)

    # Create mechanical sequence
    mechanical_notes = [
        Note(pitch=60, velocity=80, start_time=i*0.25, end_time=(i+1)*0.25, duration=0.25)
        for i in range(16)
    ]

    print("  Original (mechanical) velocities:")
    print(f"    {[n.velocity for n in mechanical_notes[:8]]}")
    print()

    # Test different humanization levels
    humanization_levels = [0.1, 0.3, 0.5, 0.7]

    for level in humanization_levels:
        humanized = specialist.humanize_velocities(
            mechanical_notes.copy(),
            amount=level,
            preserve_accents=True
        )
        vels = [n.velocity for n in humanized[:8]]
        variance = np.std(vels)
        print(f"  Humanization {level:.1f}: velocities={vels}")
        print(f"                  variance={variance:.1f}")
        print()

    # Micro-dynamics
    print("  Adding micro-dynamics (phrase-level shaping):")
    micro_notes = specialist.add_micro_dynamics(mechanical_notes, variance=0.2, phrase_length=4)
    print(f"    {[n.velocity for n in micro_notes[:8]]}")
    print()

    # Timing humanization
    print("  Timing humanization:")
    timing_humanized = specialist.humanize_timing(mechanical_notes, variance=0.02)
    original_times = [f"{n.start_time:.3f}" for n in mechanical_notes[:4]]
    humanized_times = [f"{n.start_time:.3f}" for n in timing_humanized[:4]]
    print(f"    Original: {original_times}")
    print(f"    Humanized: {humanized_times}")

    print()
    print("✅ Humanization demonstration complete")


def demo_voice_balancing():
    """Demonstrate voice balancing"""
    print_section("4. VOICE BALANCING DEMONSTRATION")

    specialist = DynamicsSpecialist()

    # Create multiple voices
    voices = {
        0: [Note(pitch=72, velocity=90, start_time=i*0.5, end_time=(i+1)*0.5, duration=0.5) for i in range(4)],  # Melody
        1: [Note(pitch=64, velocity=80, start_time=i*0.5, end_time=(i+1)*0.5, duration=0.5) for i in range(4)],  # Harmony
        2: [Note(pitch=48, velocity=85, start_time=i*0.5, end_time=(i+1)*0.5, duration=0.5) for i in range(4)],  # Bass
        3: [Note(pitch=36, velocity=75, start_time=i*0.5, end_time=(i+1)*0.5, duration=0.5) for i in range(4)],  # Sub-bass
    }

    print("  Original voice velocities:")
    for voice_id, notes in voices.items():
        print(f"    Voice {voice_id}: {[n.velocity for n in notes]}")

    # Apply balance ratios
    balance_ratios = [1.2, 0.7, 0.9, 0.6]  # Emphasize melody, reduce harmony
    print(f"\n  Applying balance ratios: {balance_ratios}")

    balanced = specialist.balance_voices(voices, balance_ratios)

    print("\n  Balanced voice velocities:")
    for voice_id, notes in balanced.items():
        print(f"    Voice {voice_id}: {[n.velocity for n in notes]}")

    # Emphasize melody
    print("\n  Emphasizing melody (range 60-84):")
    all_notes = []
    for notes in voices.values():
        all_notes.extend(notes)

    emphasized = specialist.emphasize_melody(all_notes, emphasis=0.3)
    melody_notes = [n for n in emphasized if 60 <= n.pitch <= 84]
    other_notes = [n for n in emphasized if n.pitch < 60 or n.pitch > 84]

    print(f"    Melody velocity (avg): {np.mean([n.velocity for n in melody_notes]):.1f}")
    print(f"    Other velocity (avg):  {np.mean([n.velocity for n in other_notes]):.1f}")

    print()
    print("✅ Voice balancing demonstration complete")


def demo_articulation_coupling():
    """Demonstrate articulation-dynamics coupling"""
    print_section("5. ARTICULATION-DYNAMICS COUPLING")

    specialist = DynamicsSpecialist()

    # Create notes with different articulations
    articulations = {
        "Legato": ArticulationType.LEGATO,
        "Staccato": ArticulationType.STACCATO,
        "Marcato": ArticulationType.MARCATO,
        "Tenuto": ArticulationType.TENUTO,
        "Accent": ArticulationType.ACCENT,
        "Sforzando": ArticulationType.SFORZANDO,
    }

    base_velocity = 80
    print(f"  Base velocity: {base_velocity}")
    print()

    for name, articulation in articulations.items():
        note = Note(
            pitch=60,
            velocity=base_velocity,
            start_time=0.0,
            end_time=0.5,
            duration=0.5,
            articulation=articulation
        )

        modified = specialist.apply_articulation_dynamics([note])[0]
        modifier = specialist.articulation_velocity_modifiers[articulation]

        print(f"  {name:12s}: modifier={modifier:.2f}x → velocity={modified.velocity}")

    print()
    print("✅ Articulation coupling demonstration complete")


def demo_complete_profile():
    """Demonstrate complete dynamics profiles"""
    print_section("6. COMPLETE DYNAMICS PROFILE")

    specialist = DynamicsSpecialist(seed=42)

    # Create test notes
    notes = [
        Note(pitch=60 + (i % 12), velocity=75, start_time=i*0.25, end_time=(i+1)*0.25, duration=0.25)
        for i in range(16)
    ]

    profiles = {
        "Default": create_default_profile(),
        "Expressive": create_expressive_profile(),
        "Mechanical": create_mechanical_profile(),
    }

    print("  Applying different dynamics profiles to note sequence:")
    print()

    for profile_name, profile in profiles.items():
        modified = specialist.apply_dynamics_profile(notes.copy(), profile)
        vels = [n.velocity for n in modified[:8]]
        vel_mean = np.mean(vels)
        vel_std = np.std(vels)

        print(f"  {profile_name} Profile:")
        print(f"    Overall level: {profile.overall_level:.2f}")
        print(f"    Dynamic range: {profile.dynamic_range:.2f}")
        print(f"    Humanization:  {profile.humanization_amount:.2f}")
        print(f"    Result velocities (first 8): {vels}")
        print(f"    Mean: {vel_mean:.1f}, Std: {vel_std:.1f}")
        print()

    print("✅ Complete profile demonstration complete")


def demo_dynamics_analysis():
    """Demonstrate dynamics analysis"""
    print_section("7. DYNAMICS ANALYSIS (Inverse Learning)")

    specialist = DynamicsSpecialist()

    # Create notes with varied dynamics
    notes = [
        Note(pitch=60, velocity=40 + int(30 * np.sin(i * 0.3)), start_time=i*0.25, end_time=(i+1)*0.25, duration=0.25)
        for i in range(24)
    ]

    print("  Analyzing dynamics of note sequence:")
    print(f"  Velocities: {[n.velocity for n in notes[:12]]}")
    print()

    analysis = specialist.analyze_dynamics(notes)

    print("  Analysis Results:")
    print(f"    Mean velocity:        {analysis.mean_velocity:.1f}")
    print(f"    Velocity std:         {analysis.std_velocity:.1f}")
    print(f"    Velocity range:       {analysis.velocity_range}")
    print(f"    Dynamic contrast:     {analysis.dynamic_contrast:.2f}")
    print(f"    Crescendo count:      {analysis.crescendo_count}")
    print(f"    Diminuendo count:     {analysis.diminuendo_count}")
    print(f"    Accent frequency:     {analysis.accent_frequency:.2%}")
    print(f"    Ghost note frequency: {analysis.ghost_note_frequency:.2%}")
    print(f"    Velocity consistency: {analysis.velocity_consistency:.2f} (0=human, 1=mechanical)")
    print(f"    Natural variation:    {analysis.natural_variation_score:.2f}")
    print()

    print("✅ Dynamics analysis demonstration complete")


def main():
    """Run all demonstrations"""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "AGENT 22: DYNAMICS SPECIALIST" + " " * 29 + "║")
    print("║" + " " * 25 + "COMPREHENSIVE DEMO" + " " * 35 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    print("This demonstration showcases all capabilities of the Dynamics Specialist:")
    print("  • ADSR Envelope Control")
    print("  • Dynamic Curves (Crescendo, Diminuendo, Custom)")
    print("  • Humanization (Velocity & Timing)")
    print("  • Voice Balancing")
    print("  • Articulation-Dynamics Coupling")
    print("  • Inverse Dynamics Analysis")
    print()

    try:
        demo_adsr_envelopes()
        demo_dynamic_curves()
        demo_humanization()
        demo_voice_balancing()
        demo_articulation_coupling()
        demo_complete_profile()
        demo_dynamics_analysis()

        print()
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + " " * 25 + "🎉 ALL DEMOS COMPLETE! 🎉" + " " * 28 + "║")
        print("║" + " " * 78 + "║")
        print("║" + " " * 15 + "Agent 22 is ready for production use" + " " * 26 + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "═" * 78 + "╝")
        print()

        return 0

    except Exception as e:
        print()
        print("❌ ERROR during demonstration:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
