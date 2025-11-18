"""
AGENT 6 Comprehensive Demo: MIDI Expression & Performance

Demonstrates all AGENT 6 modules working together:
- CC Automation Engine
- Performance Engine (Piano, Strings, Brass, Guitar)
- MPE Support
- Velocity Modeling

This example creates a complete musical performance with realistic
expression and humanization.

Author: AGENT 6 - MIDI Expression & Performance
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from midi.cc_automation import (
    CCAutomationEngine, AutomationCurve, PhraseShaper,
    LFOModulator, CurveType, CCType
)
from midi.performance_engine import (
    PianoPerformer, StringPerformer, BrassPerformer,
    GuitarPerformer, Note, PitchBend
)
from midi.mpe_support import (
    MPEPerformance, MPEZoneLayout
)
from midi.velocity_modeling import (
    VelocityCurve, VelocityCurveType, InstrumentVelocityProfile,
    InstrumentType, AccentPattern, VelocityHumanizer
)


def demo_piano_performance():
    """Demonstrate realistic piano performance."""
    print("\n" + "=" * 70)
    print("DEMO 1: Realistic Piano Performance")
    print("=" * 70)

    # Create a simple melody (C major scale up and down)
    notes = [
        Note(60, 80, 0, 480, 0),      # C
        Note(62, 78, 480, 480, 0),    # D
        Note(64, 82, 960, 480, 0),    # E
        Note(65, 76, 1440, 480, 0),   # F
        Note(67, 85, 1920, 480, 0),   # G
        Note(69, 79, 2400, 480, 0),   # A
        Note(71, 88, 2880, 480, 0),   # B
        Note(72, 90, 3360, 960, 0),   # C (longer)
        # Descending
        Note(71, 75, 4320, 480, 0),   # B
        Note(69, 73, 4800, 480, 0),   # A
        Note(67, 78, 5280, 480, 0),   # G
        Note(65, 70, 5760, 480, 0),   # F
        Note(64, 76, 6240, 480, 0),   # E
        Note(62, 72, 6720, 480, 0),   # D
        Note(60, 80, 7200, 960, 0),   # C (final)
    ]

    print(f"\n1. Original melody: {len(notes)} notes")

    # Apply velocity curve for piano
    print("\n2. Applying piano velocity profile...")
    piano_profile = InstrumentVelocityProfile.get_profile(InstrumentType.PIANO)
    notes = piano_profile.apply_to_notes(notes)

    # Apply metric accents
    print("3. Applying metric accents (4/4)...")
    notes = AccentPattern.apply_metric_accents(notes, time_signature=(4, 4))

    # Humanize velocities
    print("4. Humanizing velocities...")
    notes = VelocityHumanizer.humanize(notes, variation_amount=0.12)

    # Apply piano performance techniques
    print("5. Applying piano performance model...")
    piano = PianoPerformer(ticks_per_quarter=480)
    notes, pedal_events = piano.apply_piano_performance(
        notes,
        enable_voicing=True,
        enable_spreading=True,
        enable_rubato=True,
        enable_pedal=True
    )

    print(f"   - Processed {len(notes)} notes")
    print(f"   - Generated {len(pedal_events)} pedal events")

    # Add expression automation
    print("6. Adding expression (CC11) automation...")
    automation = CCAutomationEngine()

    # Dynamic arc over the phrase
    expr_curve = PhraseShaper.create_dynamic_arc(
        start_time=0,
        peak_time=3360,  # Peak at high C
        end_time=7680,
        start_value=75,
        peak_value=105,
        end_value=70,
        cc_number=11  # Expression
    )
    automation.add_curve('expression', expr_curve)

    cc_events = automation.generate_all(0, 7680)
    print(f"   - Generated {len(cc_events)} CC events")

    print("\n✓ Piano performance complete!")
    print(f"   Total notes: {len(notes)}")
    print(f"   Velocity range: {min(n.velocity for n in notes)}-{max(n.velocity for n in notes)}")

    return notes, pedal_events, cc_events


def demo_string_section():
    """Demonstrate string section performance."""
    print("\n" + "=" * 70)
    print("DEMO 2: Realistic String Section Performance")
    print("=" * 70)

    # Create a sustained chord progression
    notes = [
        # Am chord
        Note(57, 75, 0, 1920, 0),     # A
        Note(60, 70, 0, 1920, 0),     # C
        Note(64, 72, 0, 1920, 0),     # E
        Note(69, 78, 0, 1920, 0),     # A
        # F chord
        Note(53, 73, 1920, 1920, 0),  # F
        Note(57, 68, 1920, 1920, 0),  # A
        Note(60, 70, 1920, 1920, 0),  # C
        Note(65, 75, 1920, 1920, 0),  # F
        # G chord
        Note(55, 77, 3840, 1920, 0),  # G
        Note(59, 72, 3840, 1920, 0),  # B
        Note(62, 74, 3840, 1920, 0),  # D
        Note(67, 80, 3840, 1920, 0),  # G
    ]

    print(f"\n1. Original chord progression: {len(notes)} notes")

    # Apply string velocity profile
    print("\n2. Applying string ensemble profile...")
    string_profile = InstrumentVelocityProfile.get_profile(InstrumentType.STRINGS)
    notes = string_profile.apply_to_notes(notes)

    # Apply string performance techniques
    print("3. Applying string section techniques...")
    strings = StringPerformer(section_size=8, ticks_per_quarter=480)

    # Section spread (musicians not perfectly together)
    notes = strings.apply_section_spread(notes, max_spread=12)
    print("   - Applied section spread (asynchrony)")

    # Bow changes
    notes = strings.apply_bow_changes(notes, bow_length=1920)
    print("   - Applied bow change dynamics")

    # Add vibrato to longer notes
    vibrato_events = strings.add_vibrato(notes, rate_hz=5.5, depth_cents=45)
    print(f"   - Generated {len(vibrato_events)} vibrato pitch bend events")

    # Add expression automation (swell on each chord)
    print("\n4. Adding expression swells...")
    automation = CCAutomationEngine()

    for i in range(3):
        swell = PhraseShaper.create_swell(
            start_time=i * 1920,
            end_time=(i + 1) * 1920,
            min_value=60,
            max_value=95,
            cc_number=11
        )
        automation.add_curve(f'swell_{i}', swell)

    cc_events = automation.generate_all(0, 5760)
    print(f"   - Generated {len(cc_events)} expression events")

    print("\n✓ String section performance complete!")
    print(f"   Total notes: {len(notes)}")
    print(f"   Vibrato events: {len(vibrato_events)}")

    return notes, vibrato_events, cc_events


def demo_mpe_performance():
    """Demonstrate MPE (MIDI Polyphonic Expression)."""
    print("\n" + "=" * 70)
    print("DEMO 3: MPE (MIDI Polyphonic Expression) Performance")
    print("=" * 70)

    # Create a melody with sustained notes
    notes = [
        Note(60, 85, 0, 1920, 0),
        Note(64, 80, 1920, 1920, 0),
        Note(67, 88, 3840, 1920, 0),
        Note(72, 90, 5760, 2400, 0),
    ]

    print(f"\n1. Original melody: {len(notes)} notes")

    # Convert to MPE with expressive gestures
    print("\n2. Converting to MPE with expressive gestures...")
    mpe = MPEPerformance(
        zone=MPEZoneLayout.LOWER_ZONE,
        num_channels=15,
        ticks_per_quarter=480
    )

    mpe_notes, events = mpe.convert_to_mpe(
        notes,
        add_vibrato=True,
        add_pressure=True,
        add_timbre=True,
        gesture_type='expressive'
    )

    print(f"   - MPE notes: {len(mpe_notes)}")
    print(f"   - Pitch bend events: {len(events['pitch_bends'])}")
    print(f"   - Pressure events: {len(events['pressure'])}")
    print(f"   - Timbre events: {len(events['timbre'])}")

    # Get MPE configuration
    config = mpe.get_configuration_messages()
    print(f"\n3. MPE configuration messages: {len(config)}")

    print("\n✓ MPE performance complete!")
    print("   Each note has independent pitch, pressure, and timbre control")

    return mpe_notes, events


def demo_brass_section():
    """Demonstrate brass section performance."""
    print("\n" + "=" * 70)
    print("DEMO 4: Brass Section Performance with Fall-offs")
    print("=" * 70)

    # Create a jazz-style brass phrase
    notes = [
        Note(67, 90, 0, 480, 0),      # G
        Note(69, 85, 480, 480, 0),    # A
        Note(71, 92, 960, 960, 0),    # B (longer, with fall-off)
        Note(69, 83, 1920, 480, 0),   # A
        Note(67, 88, 2400, 480, 0),   # G
        Note(65, 85, 2880, 960, 0),   # F (longer, with fall-off)
        Note(64, 90, 3840, 1920, 0),  # E (final note, with fall-off)
    ]

    print(f"\n1. Original brass phrase: {len(notes)} notes")

    # Apply brass velocity profile
    print("\n2. Applying brass profile...")
    brass_profile = InstrumentVelocityProfile.get_profile(InstrumentType.BRASS)
    notes = brass_profile.apply_to_notes(notes)

    # Apply brass performance techniques
    print("3. Applying brass articulations...")
    brass = BrassPerformer(ticks_per_quarter=480)

    # Staccato tonguing
    notes = brass.apply_tonguing(notes, tonguing_style='normal')
    print("   - Applied tonguing articulation")

    # Add fall-offs to ending notes
    fall_offs = brass.add_fall_offs(notes, fall_off_probability=0.6)
    print(f"   - Generated {len(fall_offs)} fall-off pitch bends")

    # Add dynamic accents
    print("\n4. Adding dynamic shaping...")
    notes = AccentPattern.apply_metric_accents(notes, time_signature=(4, 4))

    print("\n✓ Brass section performance complete!")
    print(f"   Total notes: {len(notes)}")
    print(f"   Fall-offs: {len(fall_offs)}")

    return notes, fall_offs


def demo_complete_ensemble():
    """Demonstrate complete ensemble with all techniques."""
    print("\n" + "=" * 70)
    print("DEMO 5: Complete Ensemble Performance")
    print("=" * 70)
    print("Combining Piano, Strings, Brass with full automation")

    # This would combine all previous demos into a full arrangement
    print("\n1. Creating piano part with expression...")
    piano_notes, piano_pedal, piano_cc = demo_piano_performance()

    print("\n2. Creating string section with vibrato...")
    string_notes, string_vibrato, string_cc = demo_string_section()

    print("\n3. Creating brass section with fall-offs...")
    brass_notes, brass_fall_offs = demo_brass_section()

    print("\n" + "=" * 70)
    print("ENSEMBLE SUMMARY")
    print("=" * 70)
    print(f"Piano:   {len(piano_notes)} notes, {len(piano_pedal)} pedal events")
    print(f"Strings: {len(string_notes)} notes, {len(string_vibrato)} pitch bends")
    print(f"Brass:   {len(brass_notes)} notes, {len(brass_fall_offs)} fall-offs")
    print(f"\nTotal performance events: {len(piano_notes) + len(string_notes) + len(brass_notes)}")

    print("\n✓ Complete ensemble performance generated!")

    return {
        'piano': (piano_notes, piano_pedal, piano_cc),
        'strings': (string_notes, string_vibrato, string_cc),
        'brass': (brass_notes, brass_fall_offs)
    }


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("AGENT 6: MIDI Expression & Performance - Comprehensive Demo")
    print("=" * 70)
    print("\nThis demo showcases all AGENT 6 modules:")
    print("  • CC Automation Engine (crescendos, LFOs, phrase shaping)")
    print("  • Performance Engine (piano, strings, brass, guitar)")
    print("  • MPE Support (per-note expression)")
    print("  • Velocity Modeling (curves, accents, humanization)")

    # Run individual demos
    demo_piano_performance()
    demo_string_section()
    demo_mpe_performance()
    demo_brass_section()

    # Run complete ensemble demo
    print("\n\n")
    ensemble = demo_complete_ensemble()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Integrate with MIDI file I/O library (mido, music21)")
    print("  2. Export to actual MIDI files")
    print("  3. Combine with AGENT 1-10 modules for complete generation")
    print("  4. Add real-time MIDI output capabilities")
    print("\n")


if __name__ == "__main__":
    main()
