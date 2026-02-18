#!/usr/bin/env python3
"""
Debug script to test voice separation logic directly
"""
import numpy as np
import pretty_midi
import sys
sys.path.append('/home/arlo/Data')

from genfromweb5 import separate_piano_roll_voices, piano_roll_to_midi

def midi_to_piano_roll(midi_path, fps=43.066):
    """Convert MIDI file to piano roll for testing"""
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    if not midi_data.instruments:
        return np.zeros((128, 100))

    # Get the duration
    duration = midi_data.get_end_time()
    time_steps = int(duration * fps) + 1

    # Create piano roll
    piano_roll = np.zeros((128, time_steps))

    for note in midi_data.instruments[0].notes:
        start_frame = int(note.start * fps)
        end_frame = int(note.end * fps)
        piano_roll[note.pitch, start_frame:end_frame] = 1.0

    return piano_roll

def main():
    # Load original MIDI and convert to piano roll
    original_midi_path = "/home/arlo/Data/miditest/ChordProg3_basicpitch (2).mid"
    print("Loading original MIDI and converting to piano roll...")

    piano_roll = midi_to_piano_roll(original_midi_path)
    print(f"Piano roll shape: {piano_roll.shape}")

    # Count original notes
    original_note_count = np.sum(piano_roll > 0.1)
    print(f"Original note events in piano roll: {original_note_count}")

    # Test voice separation
    print("\n" + "="*60)
    print("TESTING VOICE SEPARATION")
    print("="*60)

    voices = separate_piano_roll_voices(piano_roll)

    print(f"\nSeparated into {len(voices)} voices")

    # Count notes in each voice
    total_separated_notes = 0
    for i, voice in enumerate(voices):
        note_count = np.sum(voice > 0.1)
        total_separated_notes += note_count
        print(f"Voice {i+1}: {note_count} note events")

    print(f"\nTotal original note events: {original_note_count}")
    print(f"Total separated note events: {total_separated_notes}")

    if total_separated_notes < original_note_count:
        print(f"❌ LOST {original_note_count - total_separated_notes} note events!")
    elif total_separated_notes > original_note_count:
        print(f"⚠️ DUPLICATED {total_separated_notes - original_note_count} note events!")
    else:
        print("✅ All note events preserved!")

    # Save each voice as MIDI for inspection
    output_dir = "/home/arlo/Data/miditest/debug_voices"
    import os
    os.makedirs(output_dir, exist_ok=True)

    for i, voice in enumerate(voices):
        voice_path = f"{output_dir}/debug_voice_{i+1}.mid"
        piano_roll_to_midi(voice, voice_path)
        print(f"Saved voice {i+1} to: {voice_path}")

    # Create combined MIDI
    combined_midi = pretty_midi.PrettyMIDI()
    for i, voice in enumerate(voices):
        # Convert voice to MIDI notes
        instrument = pretty_midi.Instrument(program=i, name=f"Voice {i+1}")

        fps = 43.066
        for pitch in range(128):
            # Find note onsets and offsets
            note_events = voice[pitch] > 0.1
            if not np.any(note_events):
                continue

            # Find transitions
            diff = np.diff(np.concatenate(([False], note_events, [False])).astype(int))
            onsets = np.where(diff == 1)[0]
            offsets = np.where(diff == -1)[0]

            # Create notes
            for onset, offset in zip(onsets, offsets):
                start_time = onset / fps
                end_time = offset / fps

                if end_time - start_time >= 0.05:  # Minimum duration
                    note = pretty_midi.Note(
                        velocity=80,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                    instrument.notes.append(note)

        combined_midi.instruments.append(instrument)

    combined_path = f"{output_dir}/debug_combined.mid"
    combined_midi.write(combined_path)
    print(f"Saved combined debug MIDI to: {combined_path}")

    # Final verification
    debug_midi = pretty_midi.PrettyMIDI(combined_path)
    debug_notes = []
    for inst in debug_midi.instruments:
        for note in inst.notes:
            debug_notes.append((note.pitch, round(note.start, 3), round(note.end, 3)))

    original_midi = pretty_midi.PrettyMIDI(original_midi_path)
    original_notes = []
    for note in original_midi.instruments[0].notes:
        original_notes.append((note.pitch, round(note.start, 3), round(note.end, 3)))

    print(f"\nFINAL VERIFICATION:")
    print(f"Original MIDI notes: {len(original_notes)}")
    print(f"Debug output notes: {len(debug_notes)}")

    missing = set(original_notes) - set(debug_notes)
    extra = set(debug_notes) - set(original_notes)

    if missing:
        print(f"❌ Still missing {len(missing)} notes:")
        for pitch, start, end in sorted(missing):
            note_name = pretty_midi.note_number_to_name(pitch)
            print(f"   {note_name} ({pitch}): {start}s - {end}s")

    if extra:
        print(f"⚠️ Extra {len(extra)} notes:")
        for pitch, start, end in sorted(extra):
            note_name = pretty_midi.note_number_to_name(pitch)
            print(f"   {note_name} ({pitch}): {start}s - {end}s")

if __name__ == "__main__":
    main()