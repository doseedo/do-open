#!/usr/bin/env python3
"""
Note Search Interface

Simple interface to search for MIDI files containing specific note combinations
and listen to audio snippets.
"""

import gradio as gr
import pandas as pd
from pathlib import Path
import subprocess
import sys
import json
import os

class NoteSearchInterface:
    def __init__(self):
        self.valid_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def search_notes(self, selected_notes, max_files):
        """Search for files containing selected notes"""
        if not selected_notes:
            return None, "Please select at least one note", []

        # Convert to list if single string
        if isinstance(selected_notes, str):
            selected_notes = [selected_notes]

        try:
            # Run note search engine
            cmd = [
                sys.executable, "/home/arlo/Data/note_search_engine.py"
            ] + selected_notes + ["--max-files", str(max_files)]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                return None, f"Search failed: {result.stderr}", []

            # Load results
            search_name = "_".join(selected_notes).replace('#', 'sharp')
            results_dir = Path('/home/arlo/Data/note_search_results') / search_name
            playlist_file = results_dir / 'playlist.json'

            if not playlist_file.exists():
                return None, "No matching files found", []

            with open(playlist_file, 'r') as f:
                playlist_data = json.load(f)

            snippets = playlist_data.get('snippets', [])

            if not snippets:
                return None, "No audio snippets created", []

            # Prepare audio files and descriptions
            audio_files = []
            descriptions = []

            for snippet in snippets:
                audio_path = results_dir / snippet['audio_file']
                if audio_path.exists():
                    audio_files.append(str(audio_path))

                    desc = f"🎹 {snippet['original_file']}\n"
                    desc += f"🎵 Notes found: {', '.join(snippet['notes_found'])}\n"
                    desc += f"🎼 {snippet['instrument']} - {snippet['session']}\n"
                    desc += f"⏱️ {snippet['duration']:.1f}s ({snippet['start_time']:.1f}s - {snippet['end_time']:.1f}s)"
                    descriptions.append(desc)

            summary = f"🎵 Found {len(audio_files)} audio snippets with notes: {', '.join(selected_notes)}\n\n"
            summary += result.stdout

            return audio_files[0] if audio_files else None, summary, list(zip(audio_files, descriptions))

        except Exception as e:
            return None, f"Error: {e}", []

def create_note_search_interface():
    """Create the note search interface"""

    search_interface = NoteSearchInterface()

    def search_and_play(selected_notes, max_files):
        audio_file, summary, audio_list = search_interface.search_notes(selected_notes, max_files)

        # Update playlist choices
        if audio_list:
            playlist_choices = [f"{i+1}. {desc.split('🎹')[1].split('🎵')[0].strip()}"
                              for i, (_, desc) in enumerate(audio_list)]
        else:
            playlist_choices = []

        return (
            audio_file,
            summary,
            gr.update(choices=playlist_choices, value=playlist_choices[0] if playlist_choices else None),
            audio_list
        )

    def change_audio(playlist_selection, audio_list_state):
        """Change audio when playlist selection changes"""
        if not playlist_selection or not audio_list_state:
            return None, "No audio available"

        try:
            index = int(playlist_selection.split('.')[0]) - 1
            if 0 <= index < len(audio_list_state):
                audio_path, description = audio_list_state[index]
                return audio_path, description
        except:
            pass

        return None, "Error loading audio"

    # Create interface
    with gr.Blocks(title="MIDI Note Search", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎵 MIDI Note Search Interface

        Search for MIDI files containing specific note combinations (e.g., C + E + G)
        and listen to audio snippets where those notes are actually present!

        **This searches for actual notes in the MIDI data, not chord labels.**
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Note selection
                note_selector = gr.CheckboxGroup(
                    choices=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
                    label="🎹 Select Notes to Search For",
                    value=['C', 'E', 'G']  # Default C major triad
                )

                max_files_slider = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="Max Files to Search"
                )

                search_btn = gr.Button("🔍 Search for Note Combinations", variant="primary", size="lg")

            with gr.Column(scale=3):
                # Audio player
                audio_player = gr.Audio(
                    label="🎵 Audio Snippet",
                    autoplay=False
                )

                # Playlist dropdown
                playlist_dropdown = gr.Dropdown(
                    label="📋 Found Snippets",
                    choices=[],
                    interactive=True
                )

        # Results display
        with gr.Row():
            with gr.Column():
                search_results = gr.Textbox(
                    label="Search Results",
                    lines=8,
                    max_lines=10
                )

            with gr.Column():
                current_snippet_info = gr.Textbox(
                    label="Current Snippet Details",
                    lines=8,
                    max_lines=10
                )

        # Hidden state
        audio_list_state = gr.State([])

        # Event handlers
        search_btn.click(
            search_and_play,
            inputs=[note_selector, max_files_slider],
            outputs=[audio_player, search_results, playlist_dropdown, audio_list_state]
        )

        playlist_dropdown.change(
            change_audio,
            inputs=[playlist_dropdown, audio_list_state],
            outputs=[audio_player, current_snippet_info]
        )

        # Instructions and examples
        gr.Markdown("""
        ## 🎯 How to Use

        1. **Select notes** you want to find (e.g., C, E, G for C major chord)
        2. **Set max files** to search through
        3. **Click search** to find MIDI files containing those notes
        4. **Listen to snippets** where the notes actually occur together
        5. **Browse different examples** using the playlist dropdown

        ## 🎵 Example Searches

        - **C Major Chord**: C, E, G
        - **D Minor Chord**: D, F, A
        - **G7 Chord**: G, B, D, F
        - **Single Note**: Just C (finds all files with C notes)
        - **Interval**: C, G (perfect fifth)

        ## ✨ What Makes This Better

        - ✅ **Searches actual MIDI note data** (not inaccurate chord labels)
        - ✅ **Finds real note combinations** playing simultaneously
        - ✅ **Creates audio snippets** of the exact moments notes occur
        - ✅ **Shows timing information** within the original files
        - ✅ **Easy playback** with web audio player

        Results saved to: `/home/arlo/Data/note_search_results/`
        """)

    return interface

def main():
    # Check requirements
    analysis_path = "/home/arlo/Data/midi_analysis/chord_summary.csv"
    if not Path(analysis_path).exists():
        print(f"❌ {analysis_path} not found!")
        print("Please run: python /home/arlo/Data/midianal.py")
        exit(1)

    print("🎵 Starting MIDI Note Search Interface...")
    print("📁 This interface searches for actual notes in MIDI files")
    print("🔊 Creates audio snippets where note combinations are found")
    print("📱 Interface will open at: http://localhost:7863")

    # Create output directory
    output_dir = Path("/home/arlo/Data/note_search_results")
    output_dir.mkdir(exist_ok=True)

    # Launch interface
    interface = create_note_search_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()