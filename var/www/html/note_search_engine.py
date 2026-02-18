#!/usr/bin/env python3
"""
Note-based MIDI Search Engine

Searches for MIDI files containing specific note combinations
and creates audio snippets of those sections.
"""

import os
import pandas as pd
import pretty_midi
import soundfile as sf
from pathlib import Path
import numpy as np
import json
import subprocess
import random
from collections import defaultdict

class NoteMIDISearchEngine:
    def __init__(self, csv_file="/home/arlo/Data/midi_analysis/chord_summary.csv",
                 output_dir="/home/arlo/Data/note_search_results"):
        self.csv_file = csv_file
        self.output_dir = Path(output_dir)
        self.df = self.load_data()
        self.output_dir.mkdir(exist_ok=True)

        # MIDI note number to note name mapping
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def load_data(self):
        """Load MIDI file paths from CSV"""
        try:
            df = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(df)} MIDI file paths")
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return pd.DataFrame()

    def note_to_midi_numbers(self, note_name):
        """Convert note name to all possible MIDI numbers (all octaves)"""
        note_name = note_name.upper()
        if note_name not in self.note_names:
            return []

        base_note = self.note_names.index(note_name)
        # Return MIDI numbers for all octaves (0-10)
        return [base_note + 12 * octave for octave in range(11) if base_note + 12 * octave <= 127]

    def search_files_with_notes(self, target_notes, max_files=10, strict_mode=False, exclude_extra_notes=False, randomize=True):
        """Search for MIDI files containing specific note combinations"""
        if self.df.empty:
            return []

        # Convert note names to MIDI numbers
        target_midi_notes = {}
        for note in target_notes:
            target_midi_notes[note] = self.note_to_midi_numbers(note)

        # Randomize file order if requested
        file_rows = self.df.sample(frac=1).reset_index(drop=True) if randomize else self.df

        matching_files = []
        processed_count = 0

        print(f"🎵 Searching for notes: {', '.join(target_notes)}")
        if strict_mode:
            print("🎯 Strict mode: Notes must be simultaneous")
        if exclude_extra_notes:
            print("🚫 Excluding sections with extra notes")

        for idx, row in file_rows.iterrows():
            if len(matching_files) >= max_files:  # Stop when we have enough matches
                break

            if processed_count >= max_files * 10:  # Safety limit
                break

            file_path = row['file_path']
            if not Path(file_path).exists():
                continue

            try:
                # Analyze MIDI file for note content
                note_analysis = self.analyze_midi_notes(
                    file_path, target_midi_notes, strict_mode, exclude_extra_notes
                )
                if note_analysis and note_analysis['matching_sections']:
                    note_analysis['file_info'] = row
                    matching_files.append(note_analysis)
                    print(f"✅ Found notes in: {row['filename']} ({len(note_analysis['matching_sections'])} sections)")

                processed_count += 1

            except Exception as e:
                print(f"⚠️  Error analyzing {row['filename']}: {e}")
                continue

        # Sort by number of matching sections but keep some randomization
        if randomize:
            # Group by quality, then shuffle within groups
            high_quality = [f for f in matching_files if len(f['matching_sections']) >= 3]
            medium_quality = [f for f in matching_files if 1 <= len(f['matching_sections']) < 3]

            random.shuffle(high_quality)
            random.shuffle(medium_quality)

            matching_files = high_quality + medium_quality
        else:
            matching_files.sort(key=lambda x: len(x['matching_sections']), reverse=True)

        return matching_files[:max_files]

    def analyze_midi_notes(self, file_path, target_midi_notes, strict_mode=False, exclude_extra_notes=False):
        """Analyze MIDI file to find sections with target notes"""
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path)
            total_duration = midi_data.get_end_time()

            if total_duration == 0:
                return None

            # Find time windows where target notes are present
            matching_sections = []
            window_size = 2.0 if not strict_mode else 0.5  # Smaller windows for strict mode

            for start_time in np.arange(0, total_duration - window_size, window_size / 4):
                end_time = min(start_time + window_size, total_duration)

                # Get notes active in this window
                if strict_mode:
                    active_notes = self.get_simultaneous_notes(midi_data, start_time, end_time)
                else:
                    active_notes = self.get_notes_in_window(midi_data, start_time, end_time)

                if not active_notes:
                    continue

                # Check if target notes are present
                target_note_numbers = set()
                for note_name, midi_numbers in target_midi_notes.items():
                    target_note_numbers.update(midi_numbers)

                notes_found = {}
                for note_name, midi_numbers in target_midi_notes.items():
                    notes_found[note_name] = any(note in active_notes for note in midi_numbers)

                # If all target notes are present
                if all(notes_found.values()):
                    # Filter out extra notes if requested
                    if exclude_extra_notes:
                        # Check if there are notes present that aren't in our target
                        note_names_present = [self.midi_to_note_name(note) for note in active_notes]
                        target_note_names = set(target_midi_notes.keys())
                        extra_notes = set(note_names_present) - target_note_names

                        if extra_notes:
                            continue  # Skip this section - has extra notes

                    matching_sections.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'notes_present': list(active_notes),
                        'note_names_present': [self.midi_to_note_name(note) for note in active_notes]
                    })

            if not matching_sections:
                return None

            # Merge overlapping sections
            merged_sections = self.merge_overlapping_sections(matching_sections)

            return {
                'file_path': file_path,
                'total_duration': total_duration,
                'matching_sections': merged_sections[:5]  # Limit to 5 best sections per file
            }

        except Exception as e:
            return None

    def get_notes_in_window(self, midi_data, start_time, end_time):
        """Get all notes active in a time window"""
        active_notes = set()

        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                # Check if note overlaps with time window
                if note.start < end_time and note.end > start_time:
                    active_notes.add(note.pitch)

        return active_notes

    def get_simultaneous_notes(self, midi_data, start_time, end_time):
        """Get notes that are playing simultaneously (stricter than window overlap)"""
        # Find the midpoint of the window
        mid_time = (start_time + end_time) / 2
        tolerance = 0.1  # 100ms tolerance for "simultaneous"

        active_notes = set()

        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                # Check if note is active at the midpoint
                if note.start <= mid_time + tolerance and note.end >= mid_time - tolerance:
                    active_notes.add(note.pitch)

        return active_notes

    def midi_to_note_name(self, midi_number):
        """Convert MIDI number to note name"""
        return self.note_names[midi_number % 12]

    def merge_overlapping_sections(self, sections):
        """Merge overlapping time sections"""
        if not sections:
            return []

        sections.sort(key=lambda x: x['start_time'])
        merged = [sections[0]]

        for current in sections[1:]:
            last = merged[-1]
            if current['start_time'] <= last['end_time'] + 0.5:  # 0.5s overlap tolerance
                # Merge sections
                last['end_time'] = max(last['end_time'], current['end_time'])
                # Combine notes
                last['notes_present'] = list(set(last['notes_present'] + current['notes_present']))
                last['note_names_present'] = list(set(last['note_names_present'] + current['note_names_present']))
            else:
                merged.append(current)

        return merged

    def create_audio_snippets(self, search_results, search_notes, strict_mode=False, exclude_extra_notes=False):
        """Create audio snippets for matching sections"""
        search_name = "_".join(search_notes).replace('#', 'sharp')
        if strict_mode:
            search_name += "_strict"
        if exclude_extra_notes:
            search_name += "_exact"

        result_dir = self.output_dir / search_name
        result_dir.mkdir(exist_ok=True)

        snippets_created = []

        for file_result in search_results:
            file_path = file_result['file_path']
            file_info = file_result['file_info']
            filename = Path(file_path).stem

            try:
                midi_data = pretty_midi.PrettyMIDI(file_path)

                for i, section in enumerate(file_result['matching_sections']):
                    # Extract slightly longer snippet for context
                    snippet_start = max(0, section['start_time'] - 0.5)
                    snippet_end = min(file_result['total_duration'], section['end_time'] + 0.5)

                    # Create snippet
                    snippet_midi = self.extract_midi_section(midi_data, snippet_start, snippet_end)

                    if not snippet_midi.instruments:
                        continue

                    # Generate filenames
                    snippet_name = f"{filename}_section{i+1:02d}_{snippet_start:.1f}s"
                    midi_path = result_dir / f"{snippet_name}.mid"
                    audio_path = result_dir / f"{snippet_name}.wav"

                    # Save MIDI snippet
                    snippet_midi.write(str(midi_path))

                    # Render to audio
                    self.render_midi_to_audio(midi_path, audio_path)

                    # Create metadata
                    snippet_info = {
                        'snippet_name': snippet_name,
                        'original_file': file_info['filename'],
                        'start_time': snippet_start,
                        'end_time': snippet_end,
                        'duration': snippet_end - snippet_start,
                        'notes_found': section['note_names_present'],
                        'search_notes': search_notes,
                        'audio_file': f"{snippet_name}.wav",
                        'midi_file': f"{snippet_name}.mid",
                        'session': file_info['session_name'],
                        'instrument': file_info['instrument']
                    }

                    # Save metadata
                    json_path = result_dir / f"{snippet_name}.json"
                    with open(json_path, 'w') as f:
                        json.dump(snippet_info, f, indent=2)

                    snippets_created.append(snippet_info)

            except Exception as e:
                print(f"❌ Error creating snippets for {filename}: {e}")

        # Create playlist
        playlist_data = {
            'search_notes': search_notes,
            'total_snippets': len(snippets_created),
            'snippets': snippets_created
        }

        playlist_path = result_dir / 'playlist.json'
        with open(playlist_path, 'w') as f:
            json.dump(playlist_data, f, indent=2)

        return snippets_created, result_dir

    def extract_midi_section(self, midi_data, start_time, end_time):
        """Extract a section of MIDI data"""
        section_midi = pretty_midi.PrettyMIDI()

        for instrument in midi_data.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )

            for note in instrument.notes:
                if note.start < end_time and note.end > start_time:
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=max(0, note.start - start_time),
                        end=min(note.end - start_time, end_time - start_time)
                    )
                    new_instrument.notes.append(new_note)

            if new_instrument.notes:
                section_midi.instruments.append(new_instrument)

        return section_midi

    def render_midi_to_audio(self, midi_path, audio_path):
        """Render MIDI to audio"""
        try:
            # Try FluidSynth first
            soundfont_paths = [
                '/usr/share/sounds/sf2/FluidR3_GM.sf2',
                '/usr/share/sounds/sf2/default.sf2'
            ]

            for soundfont in soundfont_paths:
                if Path(soundfont).exists():
                    cmd = [
                        'fluidsynth', '-ni', soundfont, str(midi_path),
                        '-F', str(audio_path), '-r', '44100'
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        return

            # Fallback to pretty_midi
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            audio = midi_data.synthesize(fs=44100)

            if len(audio) > 0:
                audio = audio / (np.max(np.abs(audio)) + 1e-7)
                sf.write(str(audio_path), audio, 44100)

        except Exception as e:
            print(f"⚠️ Audio rendering failed: {e}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Search for MIDI files with specific notes')
    parser.add_argument('notes', nargs='+', help='Notes to search for (e.g., C E G)')
    parser.add_argument('--max-files', type=int, default=5, help='Maximum files to process')
    parser.add_argument('--strict', action='store_true', help='Strict mode: notes must be simultaneous')
    parser.add_argument('--exact', action='store_true', help='Exclude sections with extra notes')
    parser.add_argument('--no-randomize', action='store_true', help='Disable randomization of results')

    args = parser.parse_args()

    # Validate notes
    valid_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for note in args.notes:
        if note.upper() not in valid_notes:
            print(f"❌ Invalid note: {note}. Valid notes: {', '.join(valid_notes)}")
            return

    search_engine = NoteMIDISearchEngine()

    print(f"🎵 Searching for notes: {', '.join(args.notes)}")
    results = search_engine.search_files_with_notes(
        args.notes,
        max_files=args.max_files,
        strict_mode=args.strict,
        exclude_extra_notes=args.exact,
        randomize=not args.no_randomize
    )

    if not results:
        print("❌ No files found with the specified notes")
        return

    print(f"✅ Found {len(results)} files with matching notes")

    # Create audio snippets
    snippets, output_dir = search_engine.create_audio_snippets(
        results, args.notes, strict_mode=args.strict, exclude_extra_notes=args.exact
    )

    print(f"🎵 Created {len(snippets)} audio snippets")
    print(f"📁 Saved to: {output_dir}")
    print(f"📄 Check playlist.json for details")

if __name__ == "__main__":
    main()