#!/usr/bin/env python3
"""
Test the voice leading fix on the problematic MIDI file
"""
import sys
sys.path.append('/home/arlo/Data')

import numpy as np
import mido
from pathlib import Path

# Import the voice separation functions from genfrominterface
from genfrominterface import (
    extract_note_events_from_piano_roll,
    group_notes_by_chord_changes,
    separate_piano_roll_voices_new,
    piano_roll_to_midi
)

def midi_to_piano_roll(midi_path, fps=10.766):
    """Convert MIDI file to piano roll representation"""
    midi = mido.MidiFile(midi_path)

    # Calculate total duration in ticks
    max_ticks = 0
    for track in midi.tracks:
        current_ticks = 0
        for msg in track:
            current_ticks += msg.time
            if msg.type in ['note_on', 'note_off']:
                max_ticks = max(max_ticks, current_ticks)

    # Convert ticks to seconds
    tempo = 500000  # default microseconds per beat
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

    ticks_per_beat = midi.ticks_per_beat
    seconds_per_tick = (tempo / 1000000.0) / ticks_per_beat
    duration = max_ticks * seconds_per_tick

    # Create piano roll
    num_frames = int(duration * fps) + 10
    piano_roll = np.zeros((128, num_frames))

    # Fill piano roll with notes
    for track in midi.tracks:
        current_ticks = 0
        active_notes = {}

        for msg in track:
            current_ticks += msg.time
            current_time = current_ticks * seconds_per_tick
            current_frame = int(current_time * fps)

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = current_frame
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_frame = active_notes[msg.note]
                    end_frame = current_frame
                    # Fill the piano roll for this note's duration
                    for frame in range(start_frame, min(end_frame, num_frames)):
                        piano_roll[msg.note, frame] = 1.0
                    del active_notes[msg.note]

    return piano_roll, fps

def analyze_voice_leading(voice_midi_paths):
    """Analyze the voice leading in separated MIDI files"""
    print("\n" + "="*80)
    print("VOICE LEADING ANALYSIS")
    print("="*80)

    for i, midi_path in enumerate(voice_midi_paths):
        print(f"\nVoice {i+1}: {Path(midi_path).name}")
        midi = mido.MidiFile(midi_path)

        notes = []
        for track in midi.tracks:
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append((abs_time, msg.note))

        print(f"  Notes: {len(notes)}")
        for j, (time, pitch) in enumerate(notes):
            if j > 0:
                prev_pitch = notes[j-1][1]
                interval = pitch - prev_pitch
                interval_str = f"{'+' if interval > 0 else ''}{interval}"
                print(f"    Note {j}: time={time}, pitch={pitch} (interval: {interval_str})")
            else:
                print(f"    Note {j}: time={time}, pitch={pitch}")

    # Check if parallel motion is preserved
    print("\n" + "="*80)
    print("PARALLEL MOTION CHECK")
    print("="*80)

    intervals = []
    for midi_path in voice_midi_paths:
        midi = mido.MidiFile(midi_path)
        notes = []
        for track in midi.tracks:
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append((abs_time, msg.note))

        if len(notes) >= 2:
            interval = notes[1][1] - notes[0][1]
            intervals.append(interval)

    if len(set(intervals)) == 1:
        print(f"✅ PASS: All voices move in parallel by {intervals[0]} semitones")
        return True
    else:
        print(f"❌ FAIL: Voices have different intervals: {intervals}")
        return False

def main():
    input_midi = "/home/arlo/Data/test/composition-1762452541395.mid"
    output_dir = Path("/home/arlo/Data/test/voice_leading_test")
    output_dir.mkdir(exist_ok=True)

    print(f"Testing voice separation on: {input_midi}")
    print(f"Output directory: {output_dir}")

    # Convert MIDI to piano roll
    print("\n1. Converting MIDI to piano roll...")
    piano_roll, fps = midi_to_piano_roll(input_midi)
    print(f"   Piano roll shape: {piano_roll.shape}")
    print(f"   FPS: {fps}")

    # Separate voices
    print("\n2. Separating voices...")
    voices = separate_piano_roll_voices_new(piano_roll)
    print(f"   Separated into {len(voices)} voices")

    # Save voice MIDI files
    print("\n3. Saving voice MIDI files...")
    voice_midi_paths = []
    for i, voice_pr in enumerate(voices):
        voice_path = output_dir / f"voice_{i+1}_fixed.mid"
        piano_roll_to_midi(voice_pr, voice_path, fps=fps, tempo=120.0)
        voice_midi_paths.append(str(voice_path))
        print(f"   Saved: {voice_path.name}")

    # Analyze voice leading
    success = analyze_voice_leading(voice_midi_paths)

    if success:
        print("\n" + "="*80)
        print("✅ VOICE LEADING FIX SUCCESSFUL!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ VOICE LEADING FIX FAILED - NEEDS MORE WORK")
        print("="*80)

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
