#!/usr/bin/env python3
"""
MIDI Chord Extractor

Extracts specific chord sections from MIDI files and creates organized folder structure
with audio renders for each detected chord section.
"""

import os
import pandas as pd
import pretty_midi
import soundfile as sf
from pathlib import Path
import numpy as np
from collections import defaultdict
import json
import subprocess
import tempfile

class MIDIChordExtractor:
    def __init__(self, csv_file="/home/arlo/Data/midi_analysis/chord_summary.csv",
                 output_dir="/home/arlo/Data/chord_extracts"):
        self.csv_file = csv_file
        self.output_dir = Path(output_dir)
        self.df = self.load_data()

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Load MIDI analysis data from CSV"""
        try:
            df = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(df)} MIDI files from CSV")
            return df
        except FileNotFoundError:
            print(f"❌ {self.csv_file} not found!")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return pd.DataFrame()

    def parse_chord_progression(self, progression_str, target_chord):
        """Parse chord progression string and find positions of target chord"""
        if not progression_str or pd.isna(progression_str):
            return []

        chords = [chord.strip() for chord in str(progression_str).split(' -> ')]
        matches = []

        for i, chord in enumerate(chords):
            if target_chord.lower() in chord.lower():
                matches.append({
                    'position': i,
                    'chord': chord,
                    'context_start': max(0, i-2),
                    'context_end': min(len(chords), i+3)
                })

        return matches

    def extract_chord_sections(self, target_chord, max_files=50):
        """Extract chord sections for a specific chord type"""
        if self.df.empty:
            return []

        results = []
        chord_name = target_chord.replace(' ', '_').replace('#', 'sharp').replace('b', 'flat')
        chord_dir = self.output_dir / chord_name
        chord_dir.mkdir(exist_ok=True)

        # Search for files containing the target chord
        matching_files = []
        for idx, row in self.df.iterrows():
            progression = str(row['chord_progression']).lower()
            if target_chord.lower() in progression:
                chord_matches = self.parse_chord_progression(row['chord_progression'], target_chord)
                if chord_matches:
                    matching_files.append({
                        'row': row,
                        'matches': chord_matches,
                        'file_path': row['file_path'],
                        'filename': row['filename']
                    })

        # Sort by number of matches and limit results
        matching_files.sort(key=lambda x: len(x['matches']), reverse=True)
        matching_files = matching_files[:max_files]

        print(f"🎵 Processing {len(matching_files)} files for chord '{target_chord}'...")

        for file_info in matching_files:
            try:
                self.process_file_for_chord(file_info, target_chord, chord_dir)
                results.append(file_info)
            except Exception as e:
                print(f"❌ Error processing {file_info['filename']}: {e}")

        return results

    def process_file_for_chord(self, file_info, target_chord, output_dir):
        """Process a single MIDI file and extract chord sections"""
        file_path = file_info['file_path']
        filename = file_info['filename']
        matches = file_info['matches']

        if not Path(file_path).exists():
            print(f"⚠️  File not found: {file_path}")
            return

        # Create subdirectory for this file
        file_dir = output_dir / Path(filename).stem
        file_dir.mkdir(exist_ok=True)

        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(file_path)
            total_duration = midi_data.get_end_time()

            # Estimate chord timing (simple approach - divide by number of chords)
            total_chords = len(str(file_info['row']['chord_progression']).split(' -> '))
            chord_duration = total_duration / max(total_chords, 1)

            for match_idx, match in enumerate(matches):
                # Calculate approximate timing for this chord
                start_time = match['position'] * chord_duration
                end_time = min(start_time + chord_duration * 2, total_duration)  # Include some context

                # Extract MIDI section
                section_midi = self.extract_midi_section(midi_data, start_time, end_time)

                # Save MIDI section
                section_filename = f"{match_idx+1:02d}_{match['chord']}_{match['position']:03d}.mid"
                section_path = file_dir / section_filename
                section_midi.write(str(section_path))

                # Render to audio using FluidSynth if available
                try:
                    self.render_midi_to_audio(section_path, section_path.with_suffix('.wav'))
                except Exception as e:
                    print(f"⚠️  Could not render audio for {section_filename}: {e}")

                # Create metadata file
                metadata = {
                    'original_file': file_path,
                    'chord': match['chord'],
                    'position': match['position'],
                    'start_time': start_time,
                    'end_time': end_time,
                    'context_chords': str(file_info['row']['chord_progression']).split(' -> ')[match['context_start']:match['context_end']],
                    'session': file_info['row']['session_name'],
                    'instrument': file_info['row']['instrument'],
                    'date': file_info['row']['date']
                }

                metadata_path = file_dir / f"{section_filename}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

        except Exception as e:
            print(f"❌ Error processing MIDI file {filename}: {e}")

    def extract_midi_section(self, midi_data, start_time, end_time):
        """Extract a section of MIDI data between start and end times"""
        # Create new MIDI object
        section_midi = pretty_midi.PrettyMIDI()

        for instrument in midi_data.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )

            # Copy notes within time range
            for note in instrument.notes:
                if note.start >= start_time and note.start <= end_time:
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start - start_time,  # Adjust timing to start from 0
                        end=min(note.end - start_time, end_time - start_time)
                    )
                    new_instrument.notes.append(new_note)

            if new_instrument.notes:  # Only add if there are notes
                section_midi.instruments.append(new_instrument)

        return section_midi

    def render_midi_to_audio(self, midi_path, audio_path):
        """Render MIDI to audio using FluidSynth"""
        # Check if FluidSynth is available
        try:
            result = subprocess.run(['which', 'fluidsynth'], capture_output=True, text=True)
            if result.returncode != 0:
                raise FileNotFoundError("FluidSynth not found")

            # Look for a soundfont
            soundfont_paths = [
                '/usr/share/sounds/sf2/FluidR3_GM.sf2',
                '/usr/share/sounds/sf2/default.sf2',
                '/usr/share/soundfonts/FluidR3_GM.sf2',
                '/usr/share/soundfonts/default.sf2'
            ]

            soundfont = None
            for sf_path in soundfont_paths:
                if Path(sf_path).exists():
                    soundfont = sf_path
                    break

            if not soundfont:
                raise FileNotFoundError("No soundfont found")

            # Render using FluidSynth
            cmd = [
                'fluidsynth',
                '-ni',
                soundfont,
                str(midi_path),
                '-F', str(audio_path),
                '-r', '44100'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"FluidSynth failed: {result.stderr}")

        except Exception as e:
            # Fallback: use pretty_midi to synthesize
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            audio = midi_data.synthesize(fs=44100)
            sf.write(str(audio_path), audio, 44100)

    def create_chord_browser(self, target_chord):
        """Create a simple HTML browser for the extracted chord sections"""
        chord_name = target_chord.replace(' ', '_').replace('#', 'sharp').replace('b', 'flat')
        chord_dir = self.output_dir / chord_name

        if not chord_dir.exists():
            print(f"❌ No extracts found for chord '{target_chord}'")
            return

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Chord Extracts: {target_chord}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .file-section {{ border: 1px solid #ccc; margin: 10px 0; padding: 15px; }}
        .chord-item {{ margin: 10px 0; padding: 10px; background: #f5f5f5; }}
        .metadata {{ font-size: 0.9em; color: #666; }}
        audio {{ width: 100%; margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>Chord Extracts: {target_chord}</h1>
    <p>Found {len(list(chord_dir.iterdir()))} files containing '{target_chord}' chord</p>
"""

        for file_dir in sorted(chord_dir.iterdir()):
            if file_dir.is_dir():
                html_content += f"""
    <div class="file-section">
        <h2>{file_dir.name}</h2>
"""

                # Find all chord sections in this file
                for midi_file in sorted(file_dir.glob("*.mid")):
                    audio_file = midi_file.with_suffix('.wav')
                    json_file = Path(str(midi_file) + '.json')

                    html_content += f"""
        <div class="chord-item">
            <h3>{midi_file.stem}</h3>
"""

                    if audio_file.exists():
                        html_content += f"""
            <audio controls>
                <source src="{audio_file.relative_to(chord_dir)}" type="audio/wav">
                Your browser does not support audio playback.
            </audio>
"""

                    if json_file.exists():
                        try:
                            with open(json_file) as f:
                                metadata = json.load(f)
                            html_content += f"""
            <div class="metadata">
                <strong>Chord:</strong> {metadata.get('chord', 'N/A')}<br>
                <strong>Position:</strong> {metadata.get('position', 'N/A')}<br>
                <strong>Session:</strong> {metadata.get('session', 'N/A')}<br>
                <strong>Instrument:</strong> {metadata.get('instrument', 'N/A')}<br>
                <strong>Context:</strong> {' -> '.join(metadata.get('context_chords', []))}<br>
            </div>
"""
                        except Exception as e:
                            html_content += f'<div class="metadata">Metadata error: {e}</div>'

                    html_content += """
        </div>
"""

                html_content += """
    </div>
"""

        html_content += """
</body>
</html>
"""

        # Save HTML file
        html_path = chord_dir / f"{chord_name}_browser.html"
        with open(html_path, 'w') as f:
            f.write(html_content)

        print(f"✅ Created chord browser: {html_path}")
        return html_path

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract chord sections from MIDI files')
    parser.add_argument('chord', help='Chord to search for (e.g., "major", "Cmajor", "minor")')
    parser.add_argument('--max-files', type=int, default=20, help='Maximum number of files to process')
    parser.add_argument('--output-dir', default='/home/arlo/Data/chord_extracts', help='Output directory')

    args = parser.parse_args()

    extractor = MIDIChordExtractor(output_dir=args.output_dir)

    print(f"🎵 Extracting chord sections for: {args.chord}")
    results = extractor.extract_chord_sections(args.chord, max_files=args.max_files)

    print(f"✅ Processed {len(results)} files")

    # Create browser
    browser_path = extractor.create_chord_browser(args.chord)
    print(f"🌐 Open browser at: file://{browser_path}")

if __name__ == "__main__":
    main()