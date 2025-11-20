#!/usr/bin/env python3
"""
Professional Big Band Generator - Using Proven Library Modules
===============================================================

This script generates professional big band arrangements by:
1. Creating a jazz melody over a 12-bar blues progression
2. Using the EXISTING professional ArrangementEngine module
3. Guarantees authentic results from proven, researched code

The ArrangementEngine was built by Agent 8 with extensive research:
- Rimsky-Korsakov: Principles of Orchestration
- Walter Piston: Orchestration
- Duke Ellington: Big band arranging
- George Russell: Jazz concepts

Usage:
    python generate_big_band_proper.py [output_name] [tempo] [key]

Examples:
    python generate_big_band_proper.py swing 140 0     # C major, 140 BPM
    python generate_big_band_proper.py bebop 200 10    # Bb major, 200 BPM
    python generate_big_band_proper.py ballad 80 5     # F major ballad
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: mido library not installed.")
    print("Please install with: pip3 install mido")
    sys.exit(1)

try:
    from genres.jazz import (
        JazzGenerator, JazzStyle, JazzForm, SwingFeel,
        JazzNote, JazzChord, JazzProgressions,
        BebopMelodyGenerator
    )
    from transformation.arrangement_engine import ArrangementEngine
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Make sure you're running from the midi_generator directory")
    sys.exit(1)


def generate_jazz_lead_sheet(tempo: int = 140, key: int = 0, output_file: str = "leadsheet.mid"):
    """
    Generate a jazz lead sheet (melody over chord changes).

    This creates a simple bebop melody over a 12-bar jazz blues progression.
    The melody is intentionally simple - the ArrangementEngine will transform it.

    Args:
        tempo: Tempo in BPM
        key: Key as pitch class (0=C, 1=C#, etc.)
        output_file: Output MIDI filename

    Returns:
        Path to generated lead sheet
    """
    print("=" * 70)
    print("STEP 1: GENERATING JAZZ LEAD SHEET")
    print("=" * 70)
    print(f"Tempo: {tempo} BPM")
    print(f"Key: {get_key_name(key)}")
    print(f"Form: 12-bar jazz blues")
    print()

    # Create melody generator
    melody_gen = BebopMelodyGenerator()

    # Generate 12-bar jazz blues progression
    progression = JazzProgressions.jazz_blues(key)
    print(f"✓ Generated chord progression: {len(progression)} chords")

    # Generate bebop melody over the changes
    # Each chord gets 4 beats (1 bar in 4/4 time)
    melody_notes = []
    current_beat = 0.0

    for i, chord in enumerate(progression):
        # Generate 4 beats of melody over this chord
        phrase = melody_gen.generate_phrase(
            chord,
            length_beats=4,  # 1 bar
            density=0.6  # Medium density - not too busy
        )

        # Adjust timing to absolute beats
        for note in phrase:
            note.start_time += current_beat

        melody_notes.extend(phrase)
        current_beat += 4.0

    print(f"✓ Generated melody: {len(melody_notes)} notes over {int(current_beat)} beats ({int(current_beat/4)} bars)")

    # Create MIDI file
    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    tempo_microseconds = int(60_000_000 / tempo)
    track.append(MetaMessage('set_tempo', tempo=tempo_microseconds, time=0))
    track.append(MetaMessage('track_name', name='Melody', time=0))

    # Add melody notes
    events = []
    for note in melody_notes:
        events.append({
            'type': 'note_on',
            'time': note.start_time,
            'note': note.pitch,
            'velocity': note.velocity
        })
        events.append({
            'type': 'note_off',
            'time': note.start_time + note.duration,
            'note': note.pitch
        })

    events.sort(key=lambda e: e['time'])

    # Convert to delta times
    last_time = 0.0
    for event in events:
        delta_beats = event['time'] - last_time
        delta_ticks = int(delta_beats * 480)

        if event['type'] == 'note_on':
            track.append(Message('note_on', note=event['note'],
                               velocity=event['velocity'], time=delta_ticks))
        else:
            track.append(Message('note_off', note=event['note'],
                               velocity=0, time=delta_ticks))

        last_time = event['time']

    track.append(MetaMessage('end_of_track', time=0))

    # Save lead sheet
    mid.save(output_file)
    print(f"✓ Saved lead sheet: {output_file}")
    print()

    return output_file


def arrange_for_big_band(leadsheet_file: str, output_name: str = "arrangement"):
    """
    Arrange the lead sheet for big band using the professional ArrangementEngine.

    This uses the proven module that implements:
    - Duke Ellington arranging principles
    - Proper sax soli voicing (5-part close harmony)
    - Brass background figures (punches and stabs)
    - Walking bass lines
    - Piano comping
    - Swing drums

    Args:
        leadsheet_file: Path to lead sheet MIDI
        output_name: Base name for output file

    Returns:
        Path to arranged MIDI file
    """
    print("=" * 70)
    print("STEP 2: ARRANGING FOR BIG BAND")
    print("=" * 70)
    print("Using professional ArrangementEngine module...")
    print("Research: Ellington, Basie, Rimsky-Korsakov, Walter Piston")
    print()

    # Create arrangement engine
    engine = ArrangementEngine(leadsheet_file)

    # Arrange for big band
    output_path = engine.arrange('big_band')

    return output_path


def get_key_name(key: int) -> str:
    """Get key name from pitch class."""
    key_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    return key_names[key % 12]


def main():
    """Main entry point."""
    print()
    print("=" * 70)
    print("PROFESSIONAL BIG BAND GENERATOR")
    print("=" * 70)
    print("Using proven library modules with extensive research")
    print()

    # Parse arguments
    output_name = sys.argv[1] if len(sys.argv) > 1 else "big_band"
    tempo = int(sys.argv[2]) if len(sys.argv) > 2 else 140
    key = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    # Generate lead sheet
    leadsheet_file = f"{output_name}_leadsheet.mid"
    leadsheet_path = generate_jazz_lead_sheet(tempo, key, leadsheet_file)

    # Arrange for big band
    arrangement_path = arrange_for_big_band(leadsheet_path, output_name)

    print()
    print("=" * 70)
    print("✅ BIG BAND ARRANGEMENT COMPLETE!")
    print("=" * 70)
    print()
    print(f"Generated files:")
    print(f"  1. Lead sheet:  {leadsheet_file}")
    print(f"  2. Arrangement: {arrangement_path}")
    print()
    print("Instrumentation:")
    print("  • Sax section (5 saxes) - Close harmony soli")
    print("  • Brass section (8 brass) - Background figures")
    print("  • Rhythm section - Piano, bass, drums")
    print()
    print("Features:")
    print("  ✓ Duke Ellington arranging principles")
    print("  ✓ Proper voice leading and voicings")
    print("  ✓ Authentic swing feel")
    print("  ✓ Professional orchestration")
    print()
    print("Open in your DAW or notation software!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
