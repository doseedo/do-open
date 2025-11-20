#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Articulation MIDI Export Integration - Agent 8
==============================================

Integrates big band articulations with MIDI export, properly encoding
pitch bend messages and CC automation for authentic performance.

Features:
---------
- Converts JazzNote objects with articulations to MIDI messages
- Handles pitch bend automation for falls, doits, rips, shakes
- Merges articulation messages with note events
- Supports multiple tracks and channels
- Compatible with mido library

Integration Points:
-------------------
- BigBandArticulationEngine -> MIDI file export
- JazzNote with articulation field -> MIDI messages
- BigBandArranger output -> Professional MIDI export

Usage Example:
--------------
```python
from transformation.big_band_articulation import (
    BigBandArticulationEngine,
    BigBandArticulationType
)
from transformation.articulation_midi_export import ArticulationMIDIExporter
from genres.jazz import JazzNote

# Create notes with articulations
notes = [
    JazzNote(60, 80, 0.0, 2.0, articulation="fall_short"),
    JazzNote(64, 85, 2.0, 2.0, articulation="shake"),
    JazzNote(67, 90, 4.0, 2.0, articulation="normal"),
]

# Export to MIDI with articulations
exporter = ArticulationMIDIExporter(tempo_bpm=120)
midi_file = exporter.export_to_midi(notes, filename="with_articulations.mid")
```

Author: Agent 8 - Articulation & Expression Engine
Date: 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformation.big_band_articulation import (
    BigBandArticulationEngine,
    BigBandArticulationType,
    MIDIArticulationResult,
    PitchBendMessage
)

# Try to import mido
try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("WARNING: mido library not installed. Install with: pip install mido")


# ============================================================================
# JAZZ NOTE TO ARTICULATION TYPE MAPPING
# ============================================================================

ARTICULATION_STRING_TO_ENUM = {
    # Standard
    "normal": BigBandArticulationType.NORMAL,
    "staccato": BigBandArticulationType.STACCATO,
    "accent": BigBandArticulationType.ACCENT,
    "legato": BigBandArticulationType.LEGATO,
    "tenuto": BigBandArticulationType.TENUTO,
    "marcato": BigBandArticulationType.MARCATO,

    # Jazz-specific
    "ghost": BigBandArticulationType.GHOST,
    "swell": BigBandArticulationType.SWELL,

    # Pitch bend articulations
    "fall_short": BigBandArticulationType.FALL_SHORT,
    "fall_long": BigBandArticulationType.FALL_LONG,
    "fall": BigBandArticulationType.FALL_SHORT,  # Alias
    "doit": BigBandArticulationType.DOIT,
    "rip": BigBandArticulationType.RIP,
    "shake": BigBandArticulationType.SHAKE,
    "scoop": BigBandArticulationType.SCOOP,
    "growl": BigBandArticulationType.GROWL,
    "plunger": BigBandArticulationType.PLUNGER,

    # Mutes
    "cup_mute": BigBandArticulationType.CUP_MUTE,
    "harmon_mute": BigBandArticulationType.HARMON_MUTE,
    "straight_mute": BigBandArticulationType.STRAIGHT_MUTE,
}


def string_to_articulation(artic_str: str) -> BigBandArticulationType:
    """Convert string articulation to BigBandArticulationType enum.

    Args:
        artic_str: Articulation string (e.g., "fall_short", "shake", "normal")

    Returns:
        BigBandArticulationType enum value
    """
    return ARTICULATION_STRING_TO_ENUM.get(
        artic_str.lower(),
        BigBandArticulationType.NORMAL
    )


# ============================================================================
# ARTICULATION MIDI EXPORTER
# ============================================================================

class ArticulationMIDIExporter:
    """
    Export notes with articulations to MIDI files with pitch bend automation.

    Handles conversion of BigBandArticulationType to MIDI messages including:
    - Note on/off events
    - Pitch bend automation
    - CC messages (for growls, mutes, etc.)
    - Proper timing and channel assignment
    """

    def __init__(self, tempo_bpm: int = 120, ticks_per_beat: int = 480):
        """Initialize articulation MIDI exporter.

        Args:
            tempo_bpm: Tempo in beats per minute
            ticks_per_beat: MIDI ticks per quarter note (PPQ)
        """
        self.tempo_bpm = tempo_bpm
        self.ticks_per_beat = ticks_per_beat
        self.articulation_engine = BigBandArticulationEngine(
            ticks_per_beat=ticks_per_beat,
            tempo_bpm=tempo_bpm
        )

    def export_jazz_notes_to_midi(
        self,
        notes,  # List[JazzNote]
        filename: str = "output.mid",
        track_name: str = "Big Band"
    ) -> Optional[object]:
        """Export JazzNote objects with articulations to MIDI file.

        Args:
            notes: List of JazzNote objects (with articulation field)
            filename: Output MIDI filename
            track_name: Track name for MIDI file

        Returns:
            MidiFile object if successful, None otherwise
        """
        if not MIDO_AVAILABLE:
            print(f"[SIMULATION] Would create MIDI file: {filename}")
            print(f"  - {len(notes)} notes with articulations")
            return None

        # Group notes by articulation type
        artic_groups = self._group_notes_by_articulation(notes)

        # Process each group
        all_midi_events = []

        for artic_type, note_group in artic_groups.items():
            # Extract note data
            pitches = [n.pitch for n in note_group]
            durations = [n.duration for n in note_group]
            velocities = [n.velocity for n in note_group]
            start_times = [n.start_time for n in note_group]
            channels = [n.channel if hasattr(n, 'channel') else 0 for n in note_group]

            # Use first note's channel for group
            channel = channels[0] if channels else 0

            # Apply articulation
            result = self.articulation_engine.apply_articulation(
                pitches,
                durations,
                velocities,
                start_times,
                artic_type,
                channel=channel
            )

            # Convert to MIDI events
            events = self._articulation_result_to_midi_events(
                result, pitches, start_times, channel
            )
            all_midi_events.extend(events)

        # Create MIDI file
        midi_file = self._create_midi_file(all_midi_events, track_name)
        midi_file.save(filename)

        print(f"✓ MIDI file created: {filename}")
        print(f"  - {len(notes)} notes")
        print(f"  - {len([e for e in all_midi_events if e['type'] == 'pitch_bend'])} pitch bend messages")
        print(f"  - {len([e for e in all_midi_events if e['type'] == 'control_change'])} CC messages")

        return midi_file

    def _group_notes_by_articulation(self, notes) -> Dict[BigBandArticulationType, List]:
        """Group notes by articulation type for batch processing.

        Args:
            notes: List of JazzNote objects

        Returns:
            Dict mapping articulation type to list of notes
        """
        groups = {}

        for note in notes:
            # Get articulation string (default to "normal")
            artic_str = note.articulation if hasattr(note, 'articulation') else "normal"
            artic_type = string_to_articulation(artic_str)

            if artic_type not in groups:
                groups[artic_type] = []
            groups[artic_type].append(note)

        return groups

    def _articulation_result_to_midi_events(
        self,
        result: MIDIArticulationResult,
        pitches: List[int],
        start_times: List[float],
        channel: int
    ) -> List[Dict]:
        """Convert articulation result to MIDI event dictionaries.

        Args:
            result: MIDIArticulationResult from articulation engine
            pitches: Original MIDI pitches
            start_times: Note start times in beats
            channel: MIDI channel

        Returns:
            List of MIDI event dictionaries
        """
        events = []

        # Add note events
        for i, (pitch, start_time, duration, velocity) in enumerate(zip(
            result.notes, start_times, result.durations, result.velocities
        )):
            # Note on
            events.append({
                'type': 'note_on',
                'time': start_time,
                'note': pitch,
                'velocity': velocity,
                'channel': channel
            })

            # Note off
            events.append({
                'type': 'note_off',
                'time': start_time + duration,
                'note': pitch,
                'velocity': 0,
                'channel': channel
            })

        # Add pitch bend messages
        for pb in result.pitch_bends:
            events.append({
                'type': 'pitch_bend',
                'time': pb.time_ticks / self.ticks_per_beat,  # Convert to beats
                'pitch_bend': pb.pitch_bend_value,
                'channel': pb.channel
            })

        # Add CC messages
        for cc in result.cc_messages:
            events.append({
                'type': 'control_change',
                'time': cc['time_ticks'] / self.ticks_per_beat,  # Convert to beats
                'control': cc['cc_number'],
                'value': cc['value'],
                'channel': cc['channel']
            })

        return events

    def _create_midi_file(self, events: List[Dict], track_name: str) -> object:
        """Create MIDI file from events.

        Args:
            events: List of MIDI event dictionaries
            track_name: Name for MIDI track

        Returns:
            MidiFile object
        """
        # Create MIDI file
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)

        # Add tempo
        tempo_us = int(60000000 / self.tempo_bpm)
        track.append(MetaMessage('set_tempo', tempo=tempo_us, time=0))

        # Add track name
        track.append(MetaMessage('track_name', name=track_name, time=0))

        # Set pitch bend range to ±2 semitones (200 cents)
        # This is done via RPN (Registered Parameter Number)
        # RPN 0 = Pitch Bend Sensitivity
        track.append(Message('control_change', control=101, value=0, time=0, channel=0))  # RPN MSB
        track.append(Message('control_change', control=100, value=0, time=0, channel=0))  # RPN LSB
        track.append(Message('control_change', control=6, value=2, time=0, channel=0))    # Data Entry MSB (2 semitones)
        track.append(Message('control_change', control=38, value=0, time=0, channel=0))   # Data Entry LSB

        # Sort events by time
        events_sorted = sorted(events, key=lambda e: e['time'])

        # Convert to delta times and add to track
        last_time = 0.0
        for event in events_sorted:
            delta_time_beats = event['time'] - last_time
            delta_time_ticks = int(delta_time_beats * self.ticks_per_beat)
            last_time = event['time']

            if event['type'] == 'note_on':
                track.append(Message('note_on',
                                   note=event['note'],
                                   velocity=event['velocity'],
                                   time=delta_time_ticks,
                                   channel=event['channel']))

            elif event['type'] == 'note_off':
                track.append(Message('note_off',
                                   note=event['note'],
                                   velocity=event['velocity'],
                                   time=delta_time_ticks,
                                   channel=event['channel']))

            elif event['type'] == 'pitch_bend':
                track.append(Message('pitchwheel',
                                   pitch=event['pitch_bend'],
                                   time=delta_time_ticks,
                                   channel=event['channel']))

            elif event['type'] == 'control_change':
                track.append(Message('control_change',
                                   control=event['control'],
                                   value=event['value'],
                                   time=delta_time_ticks,
                                   channel=event['channel']))

        # End of track
        track.append(MetaMessage('end_of_track', time=0))

        return mid


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def apply_style_articulations(
    notes,  # List[JazzNote]
    style: str = "basie",
    phrase_positions: Optional[List[str]] = None
) -> List:
    """Apply style-specific articulations to notes automatically.

    Args:
        notes: List of JazzNote objects
        style: Big band style ("ellington", "basie", "thad_jones", "modern")
        phrase_positions: Optional list of positions ("start", "middle", "end", "climax")
                         If None, auto-detect based on note positions

    Returns:
        Modified list of JazzNote objects with articulations assigned
    """
    engine = BigBandArticulationEngine()

    # Auto-detect phrase positions if not provided
    if phrase_positions is None:
        phrase_positions = _detect_phrase_positions(notes)

    # Apply articulations
    for i, (note, position) in enumerate(zip(notes, phrase_positions)):
        # Determine context
        if position == "end":
            context = "phrase_ending"
        elif note.duration >= 2.0:
            context = "sustained"
        else:
            context = "normal"

        # Get suggested articulation
        suggested_artic = engine.suggest_articulation(context, style, position)

        # Update note's articulation field
        note.articulation = suggested_artic.value

    return notes


def _detect_phrase_positions(notes) -> List[str]:
    """Detect phrase positions (start, middle, end) based on note timing.

    Args:
        notes: List of JazzNote objects

    Returns:
        List of position strings
    """
    positions = []

    # Simple heuristic: detect phrase boundaries based on long notes or rests
    phrase_length = 4.0  # 4 beats = typical phrase

    for i, note in enumerate(notes):
        time_in_phrase = note.start_time % (phrase_length * 8)  # 8-bar phrases

        if time_in_phrase < phrase_length:
            positions.append("start")
        elif time_in_phrase >= phrase_length * 7:
            positions.append("end")
        else:
            positions.append("middle")

    return positions


# ============================================================================
# MAIN (EXAMPLES/TESTS)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ARTICULATION MIDI EXPORT - AGENT 8")
    print("Integration with MIDI Export Pipeline")
    print("=" * 80)

    if not MIDO_AVAILABLE:
        print("\n⚠ WARNING: mido library not available")
        print("Install with: pip install mido")
        print("Running in simulation mode...\n")

    # Create example JazzNote objects
    from dataclasses import dataclass

    @dataclass
    class JazzNote:
        """Simplified JazzNote for testing."""
        pitch: int
        velocity: int
        start_time: float
        duration: float
        articulation: str = "normal"
        channel: int = 0

    # Example 1: Manual articulation assignment
    print("\nExample 1: Manual articulation assignment")
    notes = [
        JazzNote(60, 80, 0.0, 2.0, articulation="normal"),
        JazzNote(64, 85, 2.0, 2.0, articulation="fall_short"),
        JazzNote(67, 90, 4.0, 2.0, articulation="shake"),
        JazzNote(72, 95, 6.0, 4.0, articulation="fall_long"),
    ]

    exporter = ArticulationMIDIExporter(tempo_bpm=120)
    exporter.export_jazz_notes_to_midi(notes, "test_articulations.mid", "Brass Section")

    # Example 2: Automatic style-based articulation
    print("\nExample 2: Automatic Ellington-style articulations")
    notes2 = [
        JazzNote(60, 80, 0.0, 1.0),
        JazzNote(64, 85, 1.0, 1.0),
        JazzNote(67, 90, 2.0, 1.0),
        JazzNote(72, 95, 3.0, 1.0),  # End of phrase
    ]

    notes2 = apply_style_articulations(notes2, style="ellington")
    print("Applied articulations:")
    for i, note in enumerate(notes2):
        print(f"  Note {i+1}: {note.articulation}")

    exporter.export_jazz_notes_to_midi(notes2, "ellington_style.mid", "Ellington Brass")

    # Example 3: Automatic Basie-style articulation
    print("\nExample 3: Automatic Basie-style articulations")
    notes3 = [
        JazzNote(60, 80, 0.0, 0.5),
        JazzNote(64, 85, 0.5, 0.5),
        JazzNote(67, 90, 1.0, 0.5),
        JazzNote(72, 95, 1.5, 0.5),
    ]

    notes3 = apply_style_articulations(notes3, style="basie")
    print("Applied articulations:")
    for i, note in enumerate(notes3):
        print(f"  Note {i+1}: {note.articulation}")

    exporter.export_jazz_notes_to_midi(notes3, "basie_style.mid", "Basie Brass")

    print("\n" + "=" * 80)
    print("MIDI EXPORT INTEGRATION COMPLETE!")
    print("Created files:")
    print("  - test_articulations.mid (manual articulations)")
    print("  - ellington_style.mid (auto Ellington style)")
    print("  - basie_style.mid (auto Basie style)")
    print("=" * 80)
