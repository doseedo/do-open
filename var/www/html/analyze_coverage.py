#!/usr/bin/env python3
"""
Analyze note coverage between original and voice-separated MIDI files
"""
import pretty_midi
import numpy as np
import os

def analyze_midi_coverage(original_file, voices_file):
    """Compare note coverage between original and voice-separated files"""

    print(f"Analyzing coverage:")
    print(f"  Original: {os.path.basename(original_file)}")
    print(f"  Voices:   {os.path.basename(voices_file)}")
    print("=" * 60)

    # Load original file
    try:
        original_midi = pretty_midi.PrettyMIDI(original_file)
    except:
        print(f"Error: Could not load original file {original_file}")
        return

    # Load voices file
    try:
        voices_midi = pretty_midi.PrettyMIDI(voices_file)
    except:
        print(f"Error: Could not load voices file {voices_file}")
        return

    # Extract all notes from original
    original_notes = []
    for instrument in original_midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                original_notes.append({
                    'pitch': note.pitch,
                    'start': round(note.start, 3),
                    'end': round(note.end, 3),
                    'velocity': note.velocity
                })

    # Extract all notes from voices
    voice_notes = []
    for i, instrument in enumerate(voices_midi.instruments):
        if not instrument.is_drum:
            for note in instrument.notes:
                voice_notes.append({
                    'pitch': note.pitch,
                    'start': round(note.start, 3),
                    'end': round(note.end, 3),
                    'velocity': note.velocity,
                    'voice': i + 1
                })

    print(f"Original notes: {len(original_notes)}")
    print(f"Voice notes:    {len(voice_notes)}")

    # Create sets for comparison
    def note_key(note):
        return (note['pitch'], note['start'], note['end'])

    original_set = {note_key(note) for note in original_notes}
    voice_set = {note_key(note) for note in voice_notes}

    # Find missing and extra notes
    missing_notes = original_set - voice_set
    extra_notes = voice_set - original_set

    print(f"\nCoverage Analysis:")
    print(f"  Notes in original: {len(original_set)}")
    print(f"  Notes in voices:   {len(voice_set)}")
    print(f"  Missing notes:     {len(missing_notes)}")
    print(f"  Extra notes:       {len(extra_notes)}")

    if missing_notes:
        print(f"\nMissing notes (pitch, start, end):")
        for note in sorted(missing_notes):
            pitch_name = pretty_midi.note_number_to_name(note[0])
            print(f"  {pitch_name:4} ({note[0]:2d}) at {note[1]:5.3f}-{note[2]:5.3f}")

    if extra_notes:
        print(f"\nExtra notes (pitch, start, end):")
        for note in sorted(extra_notes):
            pitch_name = pretty_midi.note_number_to_name(note[0])
            print(f"  {pitch_name:4} ({note[0]:2d}) at {note[1]:5.3f}-{note[2]:5.3f}")

    # Coverage percentage
    coverage = (len(original_set & voice_set) / len(original_set)) * 100 if original_set else 0
    print(f"\nCoverage: {coverage:.1f}%")

    # Voice distribution
    print(f"\nVoice distribution:")
    voice_counts = {}
    for note in voice_notes:
        voice_num = note['voice']
        if voice_num not in voice_counts:
            voice_counts[voice_num] = []
        voice_counts[voice_num].append(note['pitch'])

    for voice_num in sorted(voice_counts.keys()):
        pitches = voice_counts[voice_num]
        if pitches:
            pitch_range = f"{min(pitches)}-{max(pitches)}"
            print(f"  Voice {voice_num}: {len(pitches)} notes, range {pitch_range}")

    return coverage >= 95.0  # Consider 95%+ coverage as good

if __name__ == "__main__":
    # Test files
    original_file = "/home/arlo/Data/miditest/ChordProg3_basicpitch (2).mid"
    voices_file = "/home/arlo/Data/miditest/ChordProg3_combined_voices_20250925-201756 (1).mid"

    if os.path.exists(original_file) and os.path.exists(voices_file):
        success = analyze_midi_coverage(original_file, voices_file)
        print(f"\n{'✅ GOOD' if success else '❌ NEEDS IMPROVEMENT'}: Note coverage")
    else:
        print(f"Files not found:")
        print(f"  Original: {original_file} ({'exists' if os.path.exists(original_file) else 'missing'})")
        print(f"  Voices:   {voices_file} ({'exists' if os.path.exists(voices_file) else 'missing'})")