#!/usr/bin/env python3
"""
Debug piano roll extraction to see why notes are missing
"""
import pretty_midi
import numpy as np
import sys
import os

def analyze_piano_roll_extraction(midi_file_path):
    """Debug the piano roll extraction process"""

    print(f"Analyzing piano roll extraction from: {os.path.basename(midi_file_path)}")
    print("=" * 60)

    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # Get the piano roll (using the same method as the main script)
    piano_roll = midi_data.get_piano_roll(fs=43.066)

    print(f"Piano roll shape: {piano_roll.shape}")
    print(f"Time steps: {piano_roll.shape[1]}")
    print(f"Pitch range: 0-{piano_roll.shape[0]-1}")

    # Find all time frames with notes using the same logic as the main script
    time_frames = {}
    for t in range(piano_roll.shape[1]):
        active_pitches = np.where(piano_roll[:, t] > 0.1)[0]
        if len(active_pitches) > 0:
            time_frames[t] = list(active_pitches)

    print(f"Time frames found: {len(time_frames)}")

    # Extract all unique pitches found in time_frames
    time_frame_pitches = set()
    for pitches in time_frames.values():
        time_frame_pitches.update(pitches)

    # Extract all notes directly from MIDI instruments
    midi_notes = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                midi_notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity
                })

    midi_pitches = {note['pitch'] for note in midi_notes}

    print(f"\nPitch comparison:")
    print(f"  MIDI notes: {len(midi_notes)} notes, {len(midi_pitches)} unique pitches")
    print(f"  Time frames: {len(time_frame_pitches)} unique pitches")

    missing_from_time_frames = midi_pitches - time_frame_pitches
    extra_in_time_frames = time_frame_pitches - midi_pitches

    print(f"\nMissing from time frames: {len(missing_from_time_frames)}")
    if missing_from_time_frames:
        for pitch in sorted(missing_from_time_frames):
            pitch_name = pretty_midi.note_number_to_name(pitch)
            # Find the notes with this pitch
            notes_with_pitch = [n for n in midi_notes if n['pitch'] == pitch]
            print(f"  {pitch_name:4} ({pitch:2d}): {len(notes_with_pitch)} notes")
            for note in notes_with_pitch:
                print(f"    {note['start']:6.3f}-{note['end']:6.3f} vel={note['velocity']:3d}")

    print(f"\nExtra in time frames: {len(extra_in_time_frames)}")
    if extra_in_time_frames:
        for pitch in sorted(extra_in_time_frames):
            pitch_name = pretty_midi.note_number_to_name(pitch)
            print(f"  {pitch_name:4} ({pitch:2d})")

    # Check for low-velocity notes
    low_velocity_notes = [n for n in midi_notes if n['velocity'] < 50]
    print(f"\nLow velocity notes (vel < 50): {len(low_velocity_notes)}")

    # Check piano roll values for missing pitches
    if missing_from_time_frames:
        print(f"\nPiano roll values for missing pitches:")
        for pitch in sorted(missing_from_time_frames):
            pitch_name = pretty_midi.note_number_to_name(pitch)
            row = piano_roll[pitch, :]
            non_zero_values = row[row > 0]
            max_value = np.max(row) if len(row) > 0 else 0
            mean_value = np.mean(non_zero_values) if len(non_zero_values) > 0 else 0
            print(f"  {pitch_name:4} ({pitch:2d}): max={max_value:.3f}, mean={mean_value:.3f}, nonzero_count={len(non_zero_values)}")

if __name__ == "__main__":
    midi_file = "/home/arlo/Data/miditest/ChordProg3_basicpitch (2).mid"

    if os.path.exists(midi_file):
        analyze_piano_roll_extraction(midi_file)
    else:
        print(f"File not found: {midi_file}")