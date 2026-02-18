#!/usr/bin/env python3
"""
MIDI Audio Search Interface

Enhanced interface with audio snippet playback for chord search results.
"""

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys
import json
import os
from collections import defaultdict

class MIDIAudioSearchEngine:
    def __init__(self, csv_file="/home/arlo/Data/midi_analysis/chord_summary.csv"):
        self.csv_file = csv_file
        self.df = self.load_data()

    def load_data(self):
        """Load MIDI analysis data from CSV"""
        try:
            df = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(df)} MIDI files from CSV")
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return pd.DataFrame()

    def search_with_audio(self, chord_name, max_files=5):
        """Search for chord and create audio snippets"""
        if self.df.empty:
            return None, "No data available", []

        if not chord_name:
            return None, "Please enter a chord name", []

        # Run audio extractor
        try:
            cmd = [
                sys.executable, "/home/arlo/Data/chord_audio_extractor.py",
                chord_name, "--max-files", str(max_files)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                return None, f"Error creating audio snippets: {result.stderr}", []

            # Load created snippets
            chord_name_clean = chord_name.replace(' ', '_').replace('#', 'sharp').replace('b', 'flat')
            snippets_dir = Path('/home/arlo/Data/chord_audio_snippets') / chord_name_clean
            playlist_file = snippets_dir / 'playlist.json'

            if not playlist_file.exists():
                return None, "No audio snippets were created", []

            with open(playlist_file, 'r') as f:
                playlist_data = json.load(f)

            snippets = playlist_data.get('snippets', [])

            # Prepare audio files for Gradio
            audio_files = []
            descriptions = []

            for snippet in snippets:
                audio_path = snippets_dir / snippet['audio_file']
                if audio_path.exists():
                    audio_files.append(str(audio_path))
                    desc = f"🎹 {snippet['original_file']} - {snippet['chord']}\n"
                    desc += f"📍 Position {snippet['position']} ({snippet['percentage_through_song']:.1f}% through song)\n"
                    desc += f"🎼 {snippet['instrument']} - {snippet['session']}\n"
                    desc += f"⏱️ {snippet['duration']:.1f}s"
                    descriptions.append(desc)

            summary = f"🎵 Found {len(audio_files)} audio snippets for '{chord_name}'\n\n"
            summary += result.stdout

            return audio_files[0] if audio_files else None, summary, list(zip(audio_files, descriptions))

        except Exception as e:
            return None, f"Error: {e}", []

# Initialize search engine
search_engine = MIDIAudioSearchEngine()

def create_audio_search_interface():
    """Create the audio search interface"""
    def search_and_play(chord_name, max_files):
        audio_file, summary, audio_list = search_engine.search_with_audio(chord_name, max_files)

        # Update playlist
        if audio_list:
            playlist_choices = [f"{i+1}. {desc.split('🎹')[1].split(' - ')[0]} - {desc.split(' - ')[1].split('📍')[0]}"
                              for i, (_, desc) in enumerate(audio_list)]
        else:
            playlist_choices = []

        return (
            audio_file,  # First audio file
            summary,     # Text summary
            gr.update(choices=playlist_choices, value=playlist_choices[0] if playlist_choices else None),  # Playlist dropdown
            audio_list   # Store full audio list in state
        )

    def change_audio(playlist_selection, audio_list_state):
        """Change audio when playlist selection changes"""
        if not playlist_selection or not audio_list_state:
            return None, "No audio available"

        # Extract index from selection
        try:
            index = int(playlist_selection.split('.')[0]) - 1
            if 0 <= index < len(audio_list_state):
                audio_path, description = audio_list_state[index]
                return audio_path, description
        except:
            pass

        return None, "Error loading audio"

    # Create interface
    with gr.Blocks(title="MIDI Chord Audio Search", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎹 MIDI Chord Audio Search Interface

        Search for chord occurrences and listen to audio snippets of the detected sections!
        """)

        with gr.Row():
            with gr.Column(scale=2):
                chord_input = gr.Textbox(
                    label="Chord Name",
                    placeholder="Enter chord (e.g., major, minor, Cmajor)",
                    value="major"
                )
                max_files_slider = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="Max Files to Process"
                )
                search_btn = gr.Button("🔍 Search & Create Audio Snippets", variant="primary")

            with gr.Column(scale=3):
                # Audio player
                audio_player = gr.Audio(
                    label="🎵 Audio Snippet",
                    autoplay=False
                )

                # Playlist dropdown
                playlist_dropdown = gr.Dropdown(
                    label="📋 Snippet Playlist",
                    choices=[],
                    interactive=True
                )

        # Results and current snippet info
        with gr.Row():
            with gr.Column():
                search_results = gr.Textbox(
                    label="Search Results",
                    lines=8,
                    max_lines=10
                )

            with gr.Column():
                current_snippet_info = gr.Textbox(
                    label="Current Snippet Info",
                    lines=8,
                    max_lines=10
                )

        # Hidden state to store audio list
        audio_list_state = gr.State([])

        # Event handlers
        search_btn.click(
            search_and_play,
            inputs=[chord_input, max_files_slider],
            outputs=[audio_player, search_results, playlist_dropdown, audio_list_state]
        )

        playlist_dropdown.change(
            change_audio,
            inputs=[playlist_dropdown, audio_list_state],
            outputs=[audio_player, current_snippet_info]
        )

        # Instructions
        gr.Markdown("""
        ## 🎯 How to Use

        1. **Enter a chord name** (e.g., "major", "minor", "Cmajor", "Am")
        2. **Set max files** to process (more files = more examples, but slower)
        3. **Click search** to find and create audio snippets
        4. **Listen to snippets** using the audio player
        5. **Switch between snippets** using the playlist dropdown

        ## 🔊 What You Get

        - **Audio snippets** of detected chord sections (2-4 seconds each)
        - **Context information** showing position in song and surrounding details
        - **Organized by instrument and session** for easy browsing
        - **Direct playback** in the web interface

        Audio files are saved to: `/home/arlo/Data/chord_audio_snippets/`
        """)

    return interface

def main():
    # Check if analysis file exists
    analysis_path = "/home/arlo/Data/midi_analysis/chord_summary.csv"
    if not Path(analysis_path).exists():
        print(f"❌ {analysis_path} not found!")
        print("Please run: python /home/arlo/Data/midianal.py")
        exit(1)

    print("🎹 Starting MIDI Audio Search Interface...")
    print(f"📁 Using data from: {analysis_path}")

    if not search_engine.df.empty:
        print(f"📊 Loaded {len(search_engine.df)} MIDI files")

    # Create output directory
    output_dir = Path("/home/arlo/Data/chord_audio_snippets")
    output_dir.mkdir(exist_ok=True)

    # Launch interface
    interface = create_audio_search_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()