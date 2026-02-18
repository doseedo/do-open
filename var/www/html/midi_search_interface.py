#!/usr/bin/env python3
"""
MIDI Analysis Search Interface

A Gradio web interface to search and analyze MIDI data from midianal.py output.
Supports searching by chord, note, key, and various musical criteria.
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import ast

class MIDISearchEngine:
    def __init__(self, csv_file="/home/arlo/Data/midi_analysis/chord_summary.csv"):
        self.csv_file = csv_file
        self.df = self.load_data()

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

    def search_by_note(self, note_name, exact_match=True):
        """Search for pieces containing a specific note"""
        if self.df.empty:
            return "No data available"

        results = []
        note_name = note_name.strip().upper()

        for idx, row in self.df.iterrows():
            # Check root notes
            root_notes = str(row['root_notes']).upper()
            if note_name in root_notes:
                results.append({
                    'filename': row['filename'],
                    'file_path': row['file_path'],
                    'chord_progression': row['chord_progression'],
                    'root_notes': row['root_notes'],
                    'num_chords': row['num_chords'],
                    'duration': row['total_duration'],
                    'session_name': row['session_name'],
                    'instrument': row['instrument']
                })

        return sorted(results, key=lambda x: x['num_chords'], reverse=True)

    def search_by_chord(self, chord_name):
        """Search for pieces containing a specific chord"""
        if self.df.empty:
            return "No data available"

        results = []
        chord_name = chord_name.strip().lower()

        for idx, row in self.df.iterrows():
            # Check chord progression
            progression = str(row['chord_progression']).lower()
            if chord_name in progression:
                results.append({
                    'filename': row['filename'],
                    'file_path': row['file_path'],
                    'chord_progression': row['chord_progression'],
                    'unique_chords': row['unique_chords'],
                    'num_chords': row['num_chords'],
                    'duration': row['total_duration'],
                    'session_name': row['session_name'],
                    'instrument': row['instrument']
                })

        return sorted(results, key=lambda x: x['num_chords'], reverse=True)

    def search_by_key(self, key_name):
        """Search for pieces in a specific key (by root notes)"""
        if self.df.empty:
            return "No data available"

        results = []
        key_name = key_name.strip().upper()

        # Remove 'major' or 'minor' if present and get root note
        if 'MAJOR' in key_name:
            root_note = key_name.replace('MAJOR', '').strip()
        elif 'MINOR' in key_name:
            root_note = key_name.replace('MINOR', '').strip()
        else:
            root_note = key_name

        for idx, row in self.df.iterrows():
            root_notes = str(row['root_notes']).upper()
            if root_note in root_notes:
                results.append({
                    'filename': row['filename'],
                    'file_path': row['file_path'],
                    'chord_progression': row['chord_progression'],
                    'root_notes': row['root_notes'],
                    'num_chords': row['num_chords'],
                    'duration': row['total_duration'],
                    'session_name': row['session_name'],
                    'instrument': row['instrument']
                })

        return sorted(results, key=lambda x: x['num_chords'], reverse=True)

    def search_by_session(self, session_name):
        """Search for pieces from a specific session"""
        if self.df.empty:
            return "No data available"

        results = []
        session_name = session_name.strip()

        filtered_df = self.df[self.df['session_name'].str.contains(session_name, case=False, na=False)]

        for idx, row in filtered_df.iterrows():
            results.append({
                'filename': row['filename'],
                'file_path': row['file_path'],
                'chord_progression': row['chord_progression'],
                'num_chords': row['num_chords'],
                'duration': row['total_duration'],
                'session_name': row['session_name'],
                'instrument': row['instrument'],
                'date': row['date']
            })

        return sorted(results, key=lambda x: x['duration'], reverse=True)

    def search_by_chord_count(self, min_chords=1, max_chords=100):
        """Search for pieces with a specific number of chords"""
        if self.df.empty:
            return "No data available"

        filtered_df = self.df[
            (self.df['num_chords'] >= min_chords) &
            (self.df['num_chords'] <= max_chords)
        ]

        results = []
        for idx, row in filtered_df.iterrows():
            results.append({
                'filename': row['filename'],
                'file_path': row['file_path'],
                'chord_progression': row['chord_progression'],
                'num_chords': row['num_chords'],
                'duration': row['total_duration'],
                'session_name': row['session_name'],
                'instrument': row['instrument']
            })

        return sorted(results, key=lambda x: x['num_chords'], reverse=True)

    def get_statistics(self):
        """Get overall statistics about the MIDI collection"""
        if self.df.empty:
            return "No data available"

        # Basic stats
        total_files = len(self.df)
        total_chords = self.df['num_chords'].sum()
        avg_chords = self.df['num_chords'].mean()
        total_duration = self.df['total_duration'].sum()

        # Top sessions
        session_counts = self.df['session_name'].value_counts().head(10)

        # Top instruments
        instrument_counts = self.df['instrument'].value_counts().head(10)

        # Most common root notes
        all_root_notes = []
        for notes in self.df['root_notes'].dropna():
            if ', ' in str(notes):
                all_root_notes.extend(str(notes).split(', '))
            else:
                all_root_notes.append(str(notes))
        note_counts = Counter(all_root_notes).most_common(12)

        # Date distribution
        date_counts = self.df['date'].value_counts().head(10)

        return {
            'total_files': total_files,
            'total_chords': total_chords,
            'avg_chords': avg_chords,
            'total_duration': total_duration,
            'top_sessions': session_counts.to_dict(),
            'top_instruments': instrument_counts.to_dict(),
            'top_notes': note_counts,
            'top_dates': date_counts.to_dict()
        }

# Initialize search engine
search_engine = MIDISearchEngine()

def format_results(results, title):
    """Format search results for display"""
    if isinstance(results, str):
        return results

    if not results:
        return f"No results found for {title}"

    output = f"Found {len(results)} files for {title}:\n\n"

    for i, result in enumerate(results[:20]):  # Limit to top 20
        output += f"{i+1}. {result['filename']}\n"
        output += f"   Session: {result['session_name']}\n"
        output += f"   Instrument: {result['instrument']}\n"
        output += f"   Chords: {result['num_chords']} ({result.get('chord_progression', 'N/A')})\n"
        output += f"   Duration: {result['duration']:.2f}s\n"
        if 'root_notes' in result:
            output += f"   Root Notes: {result['root_notes']}\n"
        output += f"   Path: {result['file_path']}\n\n"

    return output

def create_note_search_interface():
    """Create note search interface"""
    def search_notes(note_name, exact_match):
        if not note_name:
            return "Please enter a note name (e.g., C, F#, Bb)"

        results = search_engine.search_by_note(note_name, exact_match)
        return format_results(results, f"note '{note_name}'")

    return search_notes

def create_chord_search_interface():
    """Create chord search interface"""
    def search_chords(chord_name, extract_sections=False):
        if not chord_name:
            return "Please enter a chord name (e.g., C, Am, major, minor)"

        results = search_engine.search_by_chord(chord_name)

        if extract_sections and results and isinstance(results, list):
            # Run chord extractor to create organized folders with audio
            try:
                import subprocess
                import sys

                # Run the chord organizer
                cmd = [sys.executable, "/home/arlo/Data/chord_organizer.py", chord_name, "--max-files", "20"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                if result.returncode == 0:
                    output_msg = f"✅ Created organized chord results for '{chord_name}'\n"
                    output_msg += f"📁 Check: /home/arlo/Data/chord_organized/{chord_name.replace(' ', '_').replace('#', 'sharp').replace('b', 'flat')}/\n\n"
                    output_msg += result.stdout + "\n\n"
                    output_msg += "=" * 50 + "\n"
                    output_msg += format_results(results[:10], f"chord '{chord_name}' (showing first 10)")
                    return output_msg
                else:
                    return f"❌ Error extracting chord sections: {result.stderr}\n\n" + format_results(results, f"chord '{chord_name}'")

            except Exception as e:
                return f"❌ Error running chord extractor: {e}\n\n" + format_results(results, f"chord '{chord_name}'")

        return format_results(results, f"chord '{chord_name}'")

    return search_chords

def create_key_search_interface():
    """Create key search interface"""
    def search_keys(key_name):
        if not key_name:
            return "Please enter a key (e.g., C, Am, F# major, Bb minor)"

        results = search_engine.search_by_key(key_name)
        return format_results(results, f"key '{key_name}'")

    return search_keys

def create_session_search_interface():
    """Create session search interface"""
    def search_sessions(session_name):
        if not session_name:
            return "Please enter a session name"

        results = search_engine.search_by_session(session_name)
        return format_results(results, f"session '{session_name}'")

    return search_sessions

def create_chord_count_interface():
    """Create chord count search interface"""
    def search_chord_count(min_chords, max_chords):
        results = search_engine.search_by_chord_count(min_chords, max_chords)
        return format_results(results, f"chord count {min_chords}-{max_chords}")

    return search_chord_count

def create_statistics_interface():
    """Create statistics interface"""
    def show_statistics():
        stats = search_engine.get_statistics()

        if isinstance(stats, str):
            return stats

        output = "📊 MIDI Collection Statistics\n"
        output += "=" * 40 + "\n\n"
        output += f"Total Files: {stats['total_files']:,}\n"
        output += f"Total Chords: {stats['total_chords']:,}\n"
        output += f"Average Chords per File: {stats['avg_chords']:.1f}\n"
        output += f"Total Duration: {stats['total_duration']:.1f} seconds\n\n"

        output += "🎵 Top Sessions:\n"
        for session, count in list(stats['top_sessions'].items())[:10]:
            output += f"   {session}: {count} files\n"

        output += "\n🎼 Top Instruments:\n"
        for instrument, count in list(stats['top_instruments'].items())[:10]:
            output += f"   {instrument}: {count} files\n"

        output += "\n🎶 Most Common Root Notes:\n"
        for note, count in stats['top_notes'][:12]:
            output += f"   {note}: {count} occurrences\n"

        output += "\n📅 Top Recording Dates:\n"
        for date, count in list(stats['top_dates'].items())[:10]:
            output += f"   {date}: {count} files\n"

        return output

    return show_statistics

# Create Gradio interface
def create_interface():
    """Create the main Gradio interface"""

    with gr.Blocks(title="MIDI Chord Analysis Search", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎹 MIDI Chord Analysis Search Interface

        Search and analyze your MIDI collection by chord, note, key, session, and more!

        **Data Source:** `/home/arlo/Data/midi_analysis/chord_summary.csv`

        ## 🔊 NEW: Audio Search Interface Available!
        **For audio snippet playback of chord sections:**
        - Launch: `python /home/arlo/Data/launch_audio_interface.py`
        - Access: http://localhost:7862
        """)

        with gr.Tabs():
            # Note Search Tab
            with gr.TabItem("🎵 Search by Note"):
                gr.Markdown("Find MIDI files containing specific root notes")

                with gr.Row():
                    note_input = gr.Textbox(
                        label="Note Name",
                        placeholder="Enter note (e.g., C, F#, Bb)",
                        value="C"
                    )
                    exact_match = gr.Checkbox(label="Exact Match", value=True)

                note_search_btn = gr.Button("Search Notes", variant="primary")
                note_results = gr.Textbox(label="Results", lines=20, max_lines=25)

                note_search_btn.click(
                    create_note_search_interface(),
                    inputs=[note_input, exact_match],
                    outputs=note_results
                )

            # Chord Search Tab
            with gr.TabItem("🎶 Search by Chord"):
                gr.Markdown("Find MIDI files containing specific chord types")

                chord_input = gr.Textbox(
                    label="Chord Name/Type",
                    placeholder="Enter chord (e.g., major, minor, C, Am)",
                    value="major"
                )

                extract_checkbox = gr.Checkbox(
                    label="Organize Results (creates manageable folder structure by session)",
                    value=False
                )

                with gr.Row():
                    chord_search_btn = gr.Button("Search Chords", variant="primary")
                    extract_btn = gr.Button("Search & Organize Files", variant="secondary")

                chord_results = gr.Textbox(label="Results", lines=20, max_lines=25)

                chord_search_btn.click(
                    create_chord_search_interface(),
                    inputs=[chord_input, gr.State(False)],
                    outputs=chord_results
                )

                extract_btn.click(
                    create_chord_search_interface(),
                    inputs=[chord_input, gr.State(True)],
                    outputs=chord_results
                )

            # Key Search Tab
            with gr.TabItem("🎼 Search by Key"):
                gr.Markdown("Find MIDI files by root note/key")

                key_input = gr.Textbox(
                    label="Key Name",
                    placeholder="Enter key (e.g., C, Am, F# major)",
                    value="C"
                )

                key_search_btn = gr.Button("Search Keys", variant="primary")
                key_results = gr.Textbox(label="Results", lines=20, max_lines=25)

                key_search_btn.click(
                    create_key_search_interface(),
                    inputs=key_input,
                    outputs=key_results
                )

            # Session Search Tab
            with gr.TabItem("🎤 Search by Session"):
                gr.Markdown("Find MIDI files from specific recording sessions")

                session_input = gr.Textbox(
                    label="Session Name",
                    placeholder="Enter session name (e.g., BET, RamonaSession)",
                    value=""
                )

                session_search_btn = gr.Button("Search Sessions", variant="primary")
                session_results = gr.Textbox(label="Results", lines=20, max_lines=25)

                session_search_btn.click(
                    create_session_search_interface(),
                    inputs=session_input,
                    outputs=session_results
                )

            # Chord Count Tab
            with gr.TabItem("🔢 Search by Chord Count"):
                gr.Markdown("Find MIDI files with specific numbers of chords")

                with gr.Row():
                    min_chords = gr.Slider(
                        minimum=1, maximum=50, value=1, step=1,
                        label="Minimum Chords"
                    )
                    max_chords = gr.Slider(
                        minimum=1, maximum=50, value=10, step=1,
                        label="Maximum Chords"
                    )

                count_search_btn = gr.Button("Search by Chord Count", variant="primary")
                count_results = gr.Textbox(label="Results", lines=20, max_lines=25)

                count_search_btn.click(
                    create_chord_count_interface(),
                    inputs=[min_chords, max_chords],
                    outputs=count_results
                )

            # Statistics Tab
            with gr.TabItem("📊 Statistics"):
                gr.Markdown("View statistics about your MIDI collection")

                stats_btn = gr.Button("Show Statistics", variant="primary")
                stats_results = gr.Textbox(label="Collection Statistics", lines=25, max_lines=30)

                stats_btn.click(
                    create_statistics_interface(),
                    inputs=None,
                    outputs=stats_results
                )

        # Footer
        gr.Markdown("""
        ---
        **Instructions:**
        - **Notes**: Enter note names like C, F#, Bb, etc.
        - **Chords**: Enter chord types like "major", "minor", or specific chords like "C", "Am"
        - **Keys**: Enter key signatures like "C", "Am", "F# major", etc.
        - **Sessions**: Enter session names from your recordings
        - **Chord Count**: Use sliders to find files with specific numbers of chords

        Data source: `/home/arlo/Data/midi_analysis/chord_summary.csv` (generated by midianal.py)
        """)

    return interface

if __name__ == "__main__":
    # Check if analysis file exists
    analysis_path = "/home/arlo/Data/midi_analysis/chord_summary.csv"
    if not Path(analysis_path).exists():
        print(f"❌ {analysis_path} not found!")
        print("Please run: python /home/arlo/Data/midianal.py")
        exit(1)

    print("🎹 Starting MIDI Chord Analysis Search Interface...")
    print(f"📁 Using data from: {analysis_path}")

    if not search_engine.df.empty:
        print(f"📊 Loaded {len(search_engine.df)} MIDI files")
        print(f"🎼 Total chords: {search_engine.df['num_chords'].sum():,}")

    # Launch interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )