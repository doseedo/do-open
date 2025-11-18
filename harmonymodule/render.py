#!/usr/bin/env python3
"""
render.py - MIDI Chord Progression Renderer

Parses chord symbols with repeat brackets and renders MIDI chord progressions in C major.

Usage:
    python render.py "||: I | V | iv | IV :||"
    python render.py "||: IV | V | I | I :||"

Chord Symbol Format:
- Roman numerals: I, ii, iii, IV, V, vi, vii
- Minor chords: lowercase (ii, iii, vi, vii) or with dash (ii-, vi-)
- Major chords: uppercase (I, IV, V)
- Repeat brackets: ||: chord | chord :||
- Bar separators: |
"""

import sys
import re
import mido
from typing import List, Tuple, Dict
import argparse
import os

# C Major scale degrees and their MIDI notes (C4 = 60)
C_MAJOR_SCALE = {
    'I': [60, 64, 67],      # C major (C-E-G)
    'i': [60, 63, 67],      # C minor (C-Eb-G)
    'i-': [60, 63, 67],     # C minor (C-Eb-G)
    'ii': [62, 65, 69],     # D minor (D-F-A)
    'ii-': [62, 65, 69],    # D minor (D-F-A)
    'II': [62, 66, 69],     # D major (D-F#-A)
    'iii': [64, 67, 71],    # E minor (E-G-B)
    'iii-': [64, 67, 71],   # E minor (E-G-B)
    'III': [64, 68, 71],    # E major (E-G#-B)
    'IV': [65, 69, 72],     # F major (F-A-C)
    'iv': [65, 68, 72],     # F minor (F-Ab-C) - borrowed from parallel minor
    'iv-': [65, 68, 72],    # F minor (F-Ab-C) - borrowed from parallel minor
    'V': [67, 71, 74],      # G major (G-B-D)
    'V7': [67, 71, 74, 77], # G dominant 7 (G-B-D-F)
    'v': [67, 70, 74],      # G minor (G-Bb-D)
    'v-': [67, 70, 74],     # G minor (G-Bb-D)
    'vi': [69, 72, 76],     # A minor (A-C-E)
    'vi-': [69, 72, 76],    # A minor (A-C-E)
    'VI': [69, 73, 76],     # A major (A-C#-E)
    'vii': [71, 74, 77],    # B diminished (B-D-F)
    'vii-': [71, 74, 77],   # B diminished (B-D-F)
    'VII': [71, 75, 78],    # B major (B-D#-F#)
    'bII': [61, 65, 68],    # Db major (Db-F-Ab)
    'bIII': [63, 67, 70],   # Eb major (Eb-G-Bb)
    'biii-': [63, 66, 70], # Eb minor (Eb-Gb-Bb)
    'bVI': [68, 72, 75],    # Ab major (Ab-C-Eb)
    'bVII': [70, 74, 77],   # Bb major (Bb-D-F)
    'i-7': [60, 63, 67, 70], # C minor 7 (C-Eb-G-Bb)
    'V7/VI': [64, 68, 71, 74], # E dominant 7 (E-G#-B-D) - V7 of vi
    '%': [],                # Repeat previous chord (handled in logic)
}

class ChordProgressionRenderer:
    def __init__(self, tempo_bpm: int = 120, beats_per_chord: int = 4):
        self.tempo_bpm = tempo_bpm
        self.beats_per_chord = beats_per_chord
        self.ticks_per_beat = 480
        self.chord_duration_ticks = self.ticks_per_beat * beats_per_chord

    def parse_chord_symbols(self, chord_string: str) -> List[str]:
        """Parse chord symbols with repeat brackets"""
        # Remove extra whitespace
        chord_string = chord_string.strip()

        # Check for repeat brackets ||: ... :||
        repeat_pattern = r'\|\|:\s*(.*?)\s*:\|\|'
        match = re.search(repeat_pattern, chord_string)

        if match:
            # Extract content inside repeat brackets
            repeat_content = match.group(1)
            chords = self.extract_chords_from_bars(repeat_content)
            # Repeat twice for ||: :|| notation
            return chords + chords
        else:
            # No repeat brackets, just parse as regular bar-separated chords
            return self.extract_chords_from_bars(chord_string)

    def extract_chords_from_bars(self, content: str) -> List[str]:
        """Extract chord symbols separated by | bars"""
        # Split by | and clean up each chord
        chord_parts = content.split('|')
        chords = []

        for part in chord_parts:
            chord = part.strip()
            if chord and chord not in ['||:', ':||']:
                chords.append(chord)

        return chords

    def chord_to_midi_notes(self, chord_symbol: str) -> List[int]:
        """Convert chord symbol to MIDI note numbers"""
        # Clean up the chord symbol
        chord = chord_symbol.strip()

        # Handle compound chords like "bVII – III" by taking the first chord
        if '–' in chord or '-' in chord and not chord.endswith('-'):
            # Split on em dash or regular dash and take first part
            if '–' in chord:
                chord = chord.split('–')[0].strip()
            elif ' -' in chord:
                chord = chord.split(' -')[0].strip()

        # Handle minor chords with dash notation (but not flat chords)
        if chord.endswith('-') and not chord.startswith('b'):
            chord = chord[:-1].lower()

        # Convert to standard case - preserve flat notation
        if chord.lower() in ['ii', 'iii', 'vi', 'vii', 'iv', 'i', 'v'] or chord.lower().startswith(('i-', 'ii-', 'iii-', 'iv-', 'v-', 'vi-', 'vii-')):
            chord = chord.lower()
        elif chord.startswith('b') and len(chord) > 1:
            # Keep flat chords as-is but ensure consistent case
            chord = chord
        else:
            chord = chord.upper()

        if chord in C_MAJOR_SCALE:
            return C_MAJOR_SCALE[chord]
        else:
            print(f"Warning: Unknown chord symbol '{chord_symbol}', using C major")
            return C_MAJOR_SCALE['I']

    def create_midi_file(self, chords: List[str], output_filename: str = "chord_progression.mid"):
        """Create MIDI file from chord progression"""
        # Create MIDI file
        mid = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Set tempo
        tempo_microseconds = int(60000000 / self.tempo_bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo_microseconds, time=0))

        # Process chords and handle '%' repeat symbols
        processed_chords = []
        for chord_symbol in chords:
            if chord_symbol == '%':
                if processed_chords:
                    processed_chords.append(processed_chords[-1])  # Repeat previous chord
                else:
                    processed_chords.append('I')  # Default to I if no previous chord
            else:
                processed_chords.append(chord_symbol)

        # Track currently playing notes for proper note-off messages
        current_notes = []

        for chord_symbol in processed_chords:
            # Get MIDI notes for this chord
            notes = self.chord_to_midi_notes(chord_symbol)

            # Turn off previous chord notes at the start of this chord (sustain until next chord)
            if current_notes:
                for j, note in enumerate(current_notes):
                    # First note gets timing for chord transition, others get time=0
                    time = self.chord_duration_ticks if j == 0 else 0
                    track.append(mido.Message('note_off', channel=0, note=note, velocity=64, time=time))
                current_notes = []

            # Turn on new chord notes immediately after turning off previous ones
            for j, note in enumerate(notes):
                # All notes start simultaneously (time=0)
                track.append(mido.Message('note_on', channel=0, note=note, velocity=80, time=0))
                current_notes.append(note)

        # Turn off final chord notes
        if current_notes:
            for j, note in enumerate(current_notes):
                time = self.chord_duration_ticks if j == 0 else 0
                track.append(mido.Message('note_off', channel=0, note=note, velocity=64, time=time))

        # Save MIDI file
        mid.save(output_filename)
        print(f"MIDI file saved as: {output_filename}")

        return mid

    def render_progression(self, chord_string: str, output_filename: str = None):
        """Main function to render chord progression"""
        print(f"Parsing chord progression: {chord_string}")

        # Parse chords
        chords = self.parse_chord_symbols(chord_string)
        print(f"Parsed chords: {chords}")

        # Convert to MIDI notes and display
        print("\nChord progression in C major:")
        for i, chord in enumerate(chords):
            notes = self.chord_to_midi_notes(chord)
            note_names = [self.midi_to_note_name(note) for note in notes]
            print(f"  {i+1:2d}. {chord:4s} -> {note_names}")

        # Generate output filename if not provided
        if output_filename is None:
            safe_name = re.sub(r'[^\w\-_]', '_', chord_string[:30])
            output_filename = f"progression_{safe_name}.mid"

        # Create MIDI file
        midi_file = self.create_midi_file(chords, output_filename)

        return midi_file, chords

    def midi_to_note_name(self, midi_note: int) -> str:
        """Convert MIDI note number to note name"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_note // 12) - 1
        note = notes[midi_note % 12]
        return f"{note}{octave}"

def process_progs_file(filename: str, tempo: int = 120, beats: int = 4):
    """Process all chord progressions from progs.txt file"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Create output folder
        output_folder = "chord_progressions"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")

        renderer = ChordProgressionRenderer(tempo_bpm=tempo, beats_per_chord=beats)

        progression_count = 0
        for i, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Process chord progressions (lines that contain chord symbols)
            if '||:' in line or any(chord in line for chord in ['I', 'V', 'vi', 'IV', 'ii']):
                progression_count += 1
                output_filename = os.path.join(output_folder, f"progression_{progression_count:02d}.mid")

                print(f"\n--- Processing progression {progression_count} ---")
                print(f"Line {i+1}: {line}")

                try:
                    renderer.render_progression(line, output_filename)
                    print(f"✓ Saved as: {output_filename}")
                except Exception as e:
                    print(f"✗ Error processing line {i+1}: {e}")
                    continue

        print(f"\n=== Processed {progression_count} progressions ===")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Render MIDI chord progressions from Roman numeral notation')
    parser.add_argument('progression', nargs='?', help='Chord progression string (e.g. "||: I | V | vi- | IV :||") or filename ending in .txt')
    parser.add_argument('-o', '--output', help='Output MIDI filename')
    parser.add_argument('-t', '--tempo', type=int, default=120, help='Tempo in BPM (default: 120)')
    parser.add_argument('-b', '--beats', type=int, default=4, help='Beats per chord (default: 4)')
    parser.add_argument('-f', '--file', help='Process chord progressions from a text file')

    args = parser.parse_args()

    # Check if processing a file
    if args.file or (args.progression and args.progression.endswith('.txt')):
        filename = args.file or args.progression
        process_progs_file(filename, args.tempo, args.beats)
        return

    # Check if progression argument is provided
    if not args.progression:
        print("Error: Please provide a chord progression string or use -f to specify a file")
        parser.print_help()
        sys.exit(1)

    # Create renderer for single progression
    renderer = ChordProgressionRenderer(tempo_bpm=args.tempo, beats_per_chord=args.beats)

    # Render progression
    try:
        renderer.render_progression(args.progression, args.output)
        print(f"\nSuccessfully rendered chord progression at {args.tempo} BPM")
        print(f"Each chord plays for {args.beats} beats")

    except Exception as e:
        print(f"Error rendering progression: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()