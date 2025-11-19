#!/usr/bin/env python3
"""
Create a test MIDI file with a simple chord progression for testing MIDI-only generation.
"""

import mido
import os

def create_test_midi_file():
    """Create a simple test MIDI file with a chord progression."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Add tempo (120 BPM)
    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

    # Chord progression: C - Am - F - G
    chords = [
        [60, 64, 67, 72],  # C major (C-E-G-C)
        [57, 60, 64, 69],  # A minor (A-C-E-A)
        [53, 57, 60, 65],  # F major (F-A-C-F)
        [55, 59, 62, 67],  # G major (G-B-D-G)
    ]

    chord_duration = 480 * 2  # 2 beats per chord

    for chord_idx, chord in enumerate(chords):
        # Note on events (all at once)
        for note in chord:
            track.append(mido.Message('note_on', note=note, velocity=80, time=0))

        # Wait for chord duration
        track.append(mido.Message('note_on', note=60, velocity=0, time=chord_duration))  # Dummy message for timing

        # Note off events (all at once)
        for note in chord:
            track.append(mido.Message('note_off', note=note, velocity=80, time=0))

    # Save the file
    output_path = "/home/arlo/Data/test_chord_progression.mid"
    mid.save(output_path)
    print(f"✅ Created test MIDI file: {output_path}")
    return output_path

if __name__ == "__main__":
    create_test_midi_file()