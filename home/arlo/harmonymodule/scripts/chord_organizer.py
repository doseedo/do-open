#!/usr/bin/env python3
"""
Simple MIDI Chord Organizer

Creates organized folder structure for chord search results without heavy processing.
"""

import os
import pandas as pd
import shutil
from pathlib import Path
import json
from collections import defaultdict

class ChordOrganizer:
    def __init__(self, csv_file="/home/arlo/Data/midi_analysis/chord_summary.csv",
                 output_dir="/home/arlo/Data/chord_organized"):
        self.csv_file = csv_file
        self.output_dir = Path(output_dir)
        self.df = self.load_data()
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Load MIDI analysis data from CSV"""
        try:
            df = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(df)} MIDI files from CSV")
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return pd.DataFrame()

    def organize_chord_results(self, target_chord, max_files=20):
        """Organize files containing specific chord into manageable folders"""
        if self.df.empty:
            return []

        # Clean chord name for folder
        chord_name = target_chord.replace(' ', '_').replace('#', 'sharp').replace('b', 'flat')
        chord_dir = self.output_dir / chord_name
        chord_dir.mkdir(exist_ok=True)

        # Find matching files
        matching_files = []
        for idx, row in self.df.iterrows():
            progression = str(row['chord_progression']).lower()
            if target_chord.lower() in progression:
                matching_files.append(row)

        # Limit results
        matching_files = matching_files[:max_files]
        print(f"🎵 Organizing {len(matching_files)} files for chord '{target_chord}'...")

        # Group by session for better organization
        sessions = defaultdict(list)
        for file_info in matching_files:
            session = file_info['session_name']
            sessions[session].append(file_info)

        organized_info = []

        for session_name, files in sessions.items():
            session_dir = chord_dir / f"session_{session_name}"
            session_dir.mkdir(exist_ok=True)

            for file_info in files:
                try:
                    # Extract chord positions
                    chord_positions = self.find_chord_positions(file_info['chord_progression'], target_chord)

                    # Create file info
                    file_data = {
                        'original_path': file_info['file_path'],
                        'filename': file_info['filename'],
                        'session': session_name,
                        'instrument': file_info['instrument'],
                        'chord_positions': chord_positions,
                        'total_chords': file_info['num_chords'],
                        'duration': file_info['total_duration'],
                        'full_progression': file_info['chord_progression']
                    }

                    # Copy MIDI file to organized location
                    if Path(file_info['file_path']).exists():
                        dest_file = session_dir / file_info['filename']
                        if not dest_file.exists():
                            shutil.copy2(file_info['file_path'], dest_file)

                    # Create metadata file
                    metadata_file = session_dir / f"{Path(file_info['filename']).stem}_info.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(file_data, f, indent=2)

                    # Create readable summary
                    summary_file = session_dir / f"{Path(file_info['filename']).stem}_summary.txt"
                    self.create_summary_file(file_data, summary_file, target_chord)

                    organized_info.append(file_data)

                except Exception as e:
                    print(f"❌ Error organizing {file_info['filename']}: {e}")

        # Create index file
        self.create_index_file(chord_dir, target_chord, organized_info)

        return organized_info

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
                    'chord': chord,
                    'percentage': (i / len(chords)) * 100 if len(chords) > 0 else 0
                })

        return positions

    def create_summary_file(self, file_data, summary_path, target_chord):
        """Create human-readable summary file"""
        with open(summary_path, 'w') as f:
            f.write(f"CHORD ANALYSIS SUMMARY\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Target Chord: {target_chord}\n")
            f.write(f"File: {file_data['filename']}\n")
            f.write(f"Session: {file_data['session']}\n")
            f.write(f"Instrument: {file_data['instrument']}\n")
            f.write(f"Duration: {file_data['duration']:.2f} seconds\n")
            f.write(f"Total Chords: {file_data['total_chords']}\n\n")

            f.write(f"CHORD OCCURRENCES:\n")
            f.write(f"-" * 30 + "\n")
            for pos in file_data['chord_positions']:
                f.write(f"Position {pos['position']:3d}: {pos['chord']} ({pos['percentage']:.1f}% through song)\n")

            f.write(f"\nFULL PROGRESSION (first 100 chords):\n")
            f.write(f"-" * 40 + "\n")
            chords = file_data['full_progression'].split(' -> ')

            # Show progression in chunks of 10 chords per line
            for i in range(0, min(100, len(chords)), 10):
                chunk = ' -> '.join(chords[i:i+10])
                f.write(f"{i:3d}-{min(i+9, len(chords)-1):3d}: {chunk}\n")

            if len(chords) > 100:
                f.write(f"... and {len(chords) - 100} more chords\n")

    def create_index_file(self, chord_dir, target_chord, organized_info):
        """Create an index file listing all organized files"""
        index_path = chord_dir / "INDEX.txt"

        with open(index_path, 'w') as f:
            f.write(f"CHORD SEARCH RESULTS INDEX\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Search Term: {target_chord}\n")
            f.write(f"Total Files: {len(organized_info)}\n")
            f.write(f"Organization Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Group by session
            sessions = defaultdict(list)
            for info in organized_info:
                sessions[info['session']].append(info)

            for session_name, files in sessions.items():
                f.write(f"SESSION: {session_name}\n")
                f.write(f"-" * 30 + "\n")

                for info in files:
                    f.write(f"  📁 {info['filename']}\n")
                    f.write(f"     Instrument: {info['instrument']}\n")
                    f.write(f"     Duration: {info['duration']:.2f}s\n")
                    f.write(f"     Chord occurrences: {len(info['chord_positions'])}\n")
                    f.write(f"     Path: session_{session_name}/\n\n")

            f.write(f"\nDIRECTORY STRUCTURE:\n")
            f.write(f"-" * 30 + "\n")
            f.write(f"chord_organized/{target_chord.replace(' ', '_').replace('#', 'sharp').replace('b', 'flat')}/\n")
            for session_name in sessions.keys():
                f.write(f"├── session_{session_name}/\n")
                session_files = sessions[session_name]
                for i, info in enumerate(session_files):
                    prefix = "├── " if i < len(session_files) - 1 else "└── "
                    f.write(f"│   {prefix}{info['filename']}\n")
                    f.write(f"│   {prefix}{Path(info['filename']).stem}_info.json\n")
                    f.write(f"│   {prefix}{Path(info['filename']).stem}_summary.txt\n")

        print(f"✅ Created index file: {index_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Organize chord search results')
    parser.add_argument('chord', help='Chord to search for (e.g., "major", "Cmajor", "minor")')
    parser.add_argument('--max-files', type=int, default=20, help='Maximum number of files to organize')
    parser.add_argument('--output-dir', default='/home/arlo/Data/chord_organized', help='Output directory')

    args = parser.parse_args()

    organizer = ChordOrganizer(output_dir=args.output_dir)

    print(f"🎵 Organizing chord results for: {args.chord}")
    results = organizer.organize_chord_results(args.chord, max_files=args.max_files)

    chord_name = args.chord.replace(' ', '_').replace('#', 'sharp').replace('b', 'flat')
    output_path = Path(args.output_dir) / chord_name

    print(f"✅ Organized {len(results)} files")
    print(f"📁 Results saved to: {output_path}")
    print(f"📄 Check INDEX.txt for overview")

if __name__ == "__main__":
    main()