"""
Export AGENT 6 Performance to MIDI File

This example shows how to export performances created with AGENT 6
modules to actual MIDI files using the mido library.

Note: Requires `mido` library: pip install mido

Author: AGENT 6 - MIDI Expression & Performance
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from midi.cc_automation import PhraseShaper, CCAutomationEngine, CCType
from midi.performance_engine import PianoPerformer, Note
from midi.velocity_modeling import (
    InstrumentVelocityProfile,
    InstrumentType,
    AccentPattern,
    VelocityHumanizer
)

# Check if mido is available
try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("WARNING: mido library not installed. Install with: pip install mido")
    print("This example will demonstrate the API but not create actual files.\n")


def create_midi_file(notes, cc_events, pedal_events, filename="output.mid",
                    ticks_per_beat=480, tempo=500000):
    """Create MIDI file from performance data.

    Args:
        notes: List of Note objects
        cc_events: List of CCEvent objects
        pedal_events: List of (time, cc_number, value) tuples
        filename: Output filename
        ticks_per_beat: MIDI ticks per quarter note
        tempo: Tempo in microseconds per beat (500000 = 120 BPM)

    Returns:
        MidiFile object (or None if mido not available)
    """
    if not MIDO_AVAILABLE:
        print(f"[SIMULATION] Would create MIDI file: {filename}")
        print(f"  - {len(notes)} notes")
        print(f"  - {len(cc_events)} CC events")
        print(f"  - {len(pedal_events)} pedal events")
        return None

    # Create MIDI file
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    # Add tempo
    track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

    # Add track name
    track.append(MetaMessage('track_name', name='Piano Performance', time=0))

    # Collect all events with absolute times
    all_events = []

    # Add note events
    for note in notes:
        # Note on
        all_events.append({
            'type': 'note_on',
            'time': note.start_time,
            'note': note.pitch,
            'velocity': note.velocity,
            'channel': note.channel
        })
        # Note off
        all_events.append({
            'type': 'note_off',
            'time': note.start_time + note.duration,
            'note': note.pitch,
            'velocity': 0,
            'channel': note.channel
        })

    # Add CC events
    for cc_event in cc_events:
        all_events.append({
            'type': 'control_change',
            'time': cc_event.time,
            'control': cc_event.cc_number,
            'value': cc_event.value,
            'channel': cc_event.channel
        })

    # Add pedal events
    for time, cc_num, value in pedal_events:
        all_events.append({
            'type': 'control_change',
            'time': time,
            'control': cc_num,
            'value': value,
            'channel': 0
        })

    # Sort by time
    all_events.sort(key=lambda e: e['time'])

    # Convert to delta times and add to track
    last_time = 0
    for event in all_events:
        delta_time = event['time'] - last_time
        last_time = event['time']

        if event['type'] == 'note_on':
            track.append(Message('note_on',
                               note=event['note'],
                               velocity=event['velocity'],
                               time=delta_time,
                               channel=event['channel']))
        elif event['type'] == 'note_off':
            track.append(Message('note_off',
                               note=event['note'],
                               velocity=event['velocity'],
                               time=delta_time,
                               channel=event['channel']))
        elif event['type'] == 'control_change':
            track.append(Message('control_change',
                               control=event['control'],
                               value=event['value'],
                               time=delta_time,
                               channel=event['channel']))

    # End of track
    track.append(MetaMessage('end_of_track', time=0))

    # Save file
    mid.save(filename)
    print(f"✓ MIDI file created: {filename}")
    print(f"  - Duration: {mid.length:.2f} seconds")
    print(f"  - Total events: {len(all_events)}")

    return mid


def example_1_simple_piano():
    """Example 1: Simple piano melody with expression."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Simple Piano Melody with Expression")
    print("=" * 70)

    # Create a simple C major scale
    notes = [
        Note(60, 80, 0, 480, 0),
        Note(62, 78, 480, 480, 0),
        Note(64, 82, 960, 480, 0),
        Note(65, 76, 1440, 480, 0),
        Note(67, 85, 1920, 480, 0),
        Note(69, 79, 2400, 480, 0),
        Note(71, 88, 2880, 480, 0),
        Note(72, 90, 3360, 960, 0),
    ]

    # Apply piano performance
    piano = PianoPerformer(ticks_per_quarter=480)
    notes, pedal_events = piano.apply_piano_performance(notes)

    # Add expression
    automation = CCAutomationEngine()
    expr_curve = PhraseShaper.create_crescendo(
        start_time=0,
        end_time=3840,
        start_value=70,
        end_value=110,
        cc_number=11
    )
    automation.add_curve('expression', expr_curve)
    cc_events = automation.generate_all(0, 4320)

    # Create MIDI file
    create_midi_file(notes, cc_events, pedal_events,
                    filename="piano_scale.mid")


def example_2_expressive_phrase():
    """Example 2: Expressive phrase with all techniques."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Expressive Phrase with All Techniques")
    print("=" * 70)

    # Create a melodic phrase
    notes = [
        # Phrase 1
        Note(60, 75, 0, 480, 0),
        Note(64, 78, 480, 480, 0),
        Note(67, 80, 960, 480, 0),
        Note(72, 85, 1440, 960, 0),
        # Phrase 2
        Note(71, 80, 2400, 480, 0),
        Note(69, 76, 2880, 480, 0),
        Note(67, 78, 3360, 480, 0),
        Note(65, 75, 3840, 480, 0),
        Note(64, 82, 4320, 960, 0),
    ]

    print("\n1. Applying velocity modeling...")
    # Instrument profile
    profile = InstrumentVelocityProfile.get_profile(InstrumentType.PIANO)
    notes = profile.apply_to_notes(notes)

    # Metric accents
    notes = AccentPattern.apply_metric_accents(notes, time_signature=(4, 4))

    # Humanize
    notes = VelocityHumanizer.humanize(notes, variation_amount=0.12)

    print("2. Applying piano performance...")
    # Piano performance
    piano = PianoPerformer(ticks_per_quarter=480)
    notes, pedal_events = piano.apply_piano_performance(notes)

    print("3. Creating expression automation...")
    # Expression with two dynamic arcs
    automation = CCAutomationEngine()

    # First phrase arc
    arc1 = PhraseShaper.create_dynamic_arc(
        start_time=0,
        peak_time=1440,
        end_time=2400,
        start_value=70,
        peak_value=105,
        end_value=75,
        cc_number=11
    )
    automation.add_curve('arc1', arc1)

    # Second phrase arc
    arc2 = PhraseShaper.create_dynamic_arc(
        start_time=2400,
        peak_time=4320,
        end_time=5280,
        start_value=75,
        peak_value=95,
        end_value=65,
        cc_number=11
    )
    automation.add_curve('arc2', arc2)

    cc_events = automation.generate_all(0, 5280)

    # Create MIDI file
    create_midi_file(notes, cc_events, pedal_events,
                    filename="expressive_phrase.mid")


def example_3_velocity_showcase():
    """Example 3: Showcase different velocity curves."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Velocity Curve Showcase")
    print("=" * 70)

    from midi.velocity_modeling import VelocityCurve, VelocityCurveType

    # Create same melody with different velocity curves
    base_notes = [
        Note(60, 50, 0, 480, 0),
        Note(64, 70, 480, 480, 0),
        Note(67, 90, 960, 480, 0),
        Note(72, 110, 1440, 960, 0),
    ]

    curves = [
        ('linear', VelocityCurveType.LINEAR),
        ('logarithmic', VelocityCurveType.LOGARITHMIC),
        ('exponential', VelocityCurveType.EXPONENTIAL),
        ('s_curve', VelocityCurveType.S_CURVE),
    ]

    for name, curve_type in curves:
        print(f"\nCreating {name} curve version...")
        curve = VelocityCurve(curve_type=curve_type)
        notes = curve.apply_to_notes(base_notes)

        # Simple CC for reference
        cc_events = []

        # No pedal
        pedal_events = []

        create_midi_file(notes, cc_events, pedal_events,
                        filename=f"velocity_{name}.mid")


def main():
    """Run all export examples."""
    print("\n" + "=" * 70)
    print("AGENT 6: Export to MIDI File Examples")
    print("=" * 70)

    if MIDO_AVAILABLE:
        print("\n✓ mido library found - will create actual MIDI files")
    else:
        print("\n⚠ mido library not found - running in simulation mode")
        print("Install with: pip install mido\n")

    # Run examples
    example_1_simple_piano()
    example_2_expressive_phrase()
    example_3_velocity_showcase()

    print("\n" + "=" * 70)
    print("EXPORT EXAMPLES COMPLETE")
    print("=" * 70)

    if MIDO_AVAILABLE:
        print("\nMIDI files created:")
        print("  - piano_scale.mid")
        print("  - expressive_phrase.mid")
        print("  - velocity_linear.mid")
        print("  - velocity_logarithmic.mid")
        print("  - velocity_exponential.mid")
        print("  - velocity_s_curve.mid")
        print("\nOpen these files in your DAW to hear the results!")
    else:
        print("\nTo create actual MIDI files, install mido:")
        print("  pip install mido")

    print()


if __name__ == "__main__":
    main()
