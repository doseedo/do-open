#!/usr/bin/env python3
"""
MIDI Chord Audio Extractor

Extracts chord sections from MIDI files and renders them as audio snippets
for web interface playback.
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

class ChordAudioExtractor:
    def __init__(self, csv_file="/home/arlo/Data/midi_analysis/chord_summary.csv",
                 output_dir="/home/arlo/Data/chord_audio_snippets"):
        self.csv_file = csv_file
        self.output_dir = Path(output_dir)
        self.df = self.load_data()
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Load MIDI analysis data from CSV"""
        try:
            df = pd.read_csv(self.csv_file)
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return pd.DataFrame()

    def extract_chord_audio_snippets(self, target_chord, max_files=10):
        """Extract audio snippets for specific chord occurrences"""
        if self.df.empty:
            return []

        # Clean chord name for directory
        chord_name = target_chord.replace(' ', '_').replace('#', 'sharp').replace('b', 'flat')
        chord_dir = self.output_dir / chord_name
        chord_dir.mkdir(exist_ok=True)

        # Find matching files
        matching_files = []
        for idx, row in self.df.iterrows():
            progression = str(row['chord_progression']).lower()
            if target_chord.lower() in progression:
                matching_files.append(row)

        # Limit and sort by number of chords (more chords = more likely to have good examples)
        matching_files = matching_files[:max_files]

        snippets_created = []

        for file_info in matching_files:
            try:
                snippets = self.create_audio_snippets_for_file(file_info, target_chord, chord_dir)
                snippets_created.extend(snippets)
            except Exception as e:
                print(f"❌ Error processing {file_info['filename']}: {e}")
                continue

        # Create playlist/index for web interface
        self.create_web_playlist(chord_dir, target_chord, snippets_created)

        return snippets_created

    def create_audio_snippets_for_file(self, file_info, target_chord, output_dir):
        """Create audio snippets for chord occurrences in a single file"""
        file_path = file_info['file_path']
        filename = file_info['filename']

        if not Path(file_path).exists():
            return []

        # Parse chord progression to find target chord positions
        chord_positions = self.find_chord_positions(file_info['chord_progression'], target_chord)

        if not chord_positions:
            return []

        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(file_path)
            total_duration = midi_data.get_end_time()

            # Estimate timing for each chord
            total_chords = len(str(file_info['chord_progression']).split(' -> '))
            chord_duration = total_duration / max(total_chords, 1)

            snippets_created = []

            for i, pos_info in enumerate(chord_positions[:3]):  # Limit to first 3 occurrences per file
                position = pos_info['position']
                chord_name = pos_info['chord']

                # Calculate snippet timing (include some context)
                start_time = max(0, position * chord_duration - 1.0)  # 1 second before
                end_time = min(total_duration, (position + 2) * chord_duration + 1.0)  # Include next chord + 1 sec

                # Extract MIDI section
                snippet_midi = self.extract_midi_section(midi_data, start_time, end_time)

                if not snippet_midi.instruments:
                    continue

                # Create snippet filename
                safe_filename = filename.replace('.mid', '').replace(' ', '_')
                snippet_name = f"{safe_filename}_pos{position:03d}_{chord_name.replace(' ', '')}"

                # Save MIDI snippet
                midi_path = output_dir / f"{snippet_name}.mid"
                snippet_midi.write(str(midi_path))

                # Render to audio
                audio_path = output_dir / f"{snippet_name}.wav"
                self.render_midi_to_audio(midi_path, audio_path)

                # Create metadata
                snippet_info = {
                    'snippet_name': snippet_name,
                    'original_file': filename,
                    'chord': chord_name,
                    'position': position,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'audio_file': f"{snippet_name}.wav",
                    'midi_file': f"{snippet_name}.mid",
                    'session': file_info['session_name'],
                    'instrument': file_info['instrument'],
                    'percentage_through_song': (position / total_chords) * 100 if total_chords > 0 else 0
                }

                # Save metadata
                json_path = output_dir / f"{snippet_name}.json"
                with open(json_path, 'w') as f:
                    json.dump(snippet_info, f, indent=2)

                snippets_created.append(snippet_info)

        except Exception as e:
            print(f"❌ Error creating snippets for {filename}: {e}")

        return snippets_created

    def find_chord_positions(self, progression_str, target_chord):
        """Find positions where target chord appears"""
        if not progression_str or pd.isna(progression_str):
            return []

        chords = [chord.strip() for chord in str(progression_str).split(' -> ')]
        positions = []

        for i, chord in enumerate(chords):
            if target_chord.lower() in chord.lower():
                positions.append({
                    'position': i,
                    'chord': chord
                })

        return positions

    def extract_midi_section(self, midi_data, start_time, end_time):
        """Extract a section of MIDI data between start and end times"""
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
                        start=note.start - start_time,  # Adjust to start from 0
                        end=min(note.end - start_time, end_time - start_time)
                    )
                    new_instrument.notes.append(new_note)

            if new_instrument.notes:
                section_midi.instruments.append(new_instrument)

        return section_midi

    def render_midi_to_audio(self, midi_path, audio_path):
        """Render MIDI to audio using pretty_midi synthesis"""
        try:
            # Try FluidSynth first
            soundfont_paths = [
                '/usr/share/sounds/sf2/FluidR3_GM.sf2',
                '/usr/share/sounds/sf2/default.sf2'
            ]

            soundfont = None
            for sf_path in soundfont_paths:
                if Path(sf_path).exists():
                    soundfont = sf_path
                    break

            if soundfont:
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

            # Normalize audio to prevent clipping
            if len(audio) > 0:
                audio = audio / (np.max(np.abs(audio)) + 1e-7)

            sf.write(str(audio_path), audio, 44100)

        except Exception as e:
            print(f"⚠️  Audio rendering failed for {midi_path}: {e}")

    def create_web_playlist(self, chord_dir, target_chord, snippets):
        """Create web-friendly playlist file"""
        playlist_data = {
            'chord': target_chord,
            'total_snippets': len(snippets),
            'snippets': snippets
        }

        # Save playlist JSON
        playlist_path = chord_dir / 'playlist.json'
        with open(playlist_path, 'w') as f:
            json.dump(playlist_data, f, indent=2)

        # Create simple HTML player
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Chord Snippets: {target_chord}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .snippet {{ border: 1px solid #ccc; margin: 10px 0; padding: 15px; }}
        .snippet-header {{ font-weight: bold; margin-bottom: 10px; }}
        audio {{ width: 100%; margin: 10px 0; }}
        .metadata {{ font-size: 0.9em; color: #666; }}
    </style>
</head>
<body>
    <h1>🎵 Chord Snippets: {target_chord}</h1>
    <p>Found {len(snippets)} audio snippets</p>
"""

        for snippet in snippets:
            html_content += f"""
    <div class="snippet">
        <div class="snippet-header">
            🎹 {snippet['original_file']} - {snippet['chord']}
        </div>
        <audio controls>
            <source src="{snippet['audio_file']}" type="audio/wav">
            Your browser does not support audio playback.
        </audio>
        <div class="metadata">
            <strong>Instrument:</strong> {snippet['instrument']}<br>
            <strong>Session:</strong> {snippet['session']}<br>
            <strong>Position:</strong> {snippet['position']} ({snippet['percentage_through_song']:.1f}% through song)<br>
            <strong>Duration:</strong> {snippet['duration']:.1f}s
        </div>
    </div>
"""

        html_content += """
</body>
</html>
"""

        html_path = chord_dir / f"{target_chord}_player.html"
        with open(html_path, 'w') as f:
            f.write(html_content)

        print(f"✅ Created web player: {html_path}")
        return html_path

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract audio snippets for chord occurrences')
    parser.add_argument('chord', help='Chord to extract (e.g., "major", "Cmajor")')
    parser.add_argument('--max-files', type=int, default=10, help='Maximum files to process')

    args = parser.parse_args()

    extractor = ChordAudioExtractor()

    print(f"🎵 Extracting audio snippets for: {args.chord}")
    snippets = extractor.extract_chord_audio_snippets(args.chord, max_files=args.max_files)

    chord_name = args.chord.replace(' ', '_').replace('#', 'sharp').replace('b', 'flat')
    output_path = Path('/home/arlo/Data/chord_audio_snippets') / chord_name

    print(f"✅ Created {len(snippets)} audio snippets")
    print(f"📁 Saved to: {output_path}")
    print(f"🌐 Open player: {output_path}/{args.chord}_player.html")

if __name__ == "__main__":
    main()