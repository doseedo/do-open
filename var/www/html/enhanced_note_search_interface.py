#!/usr/bin/env python3
"""
Enhanced Note Search Interface

With audio file preview, MIDI rendering, and instrument grouping.
"""

import gradio as gr
import pandas as pd
from pathlib import Path
import subprocess
import sys
import json
import os

class EnhancedNoteSearchInterface:
    def __init__(self):
        self.valid_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.training_manifest = self.load_training_manifest()

    def load_training_manifest(self):
        """Load training manifest to get audio file paths"""
        try:
            manifest_path = "/home/arlo/Data/final_training_manifest_final.json"
            with open(manifest_path, 'r') as f:
                data = json.load(f)

            # Create mapping from MIDI filename to audio path
            audio_mapping = {}
            for item in data:
                audio_path = item.get('audio_path', '')
                if audio_path:
                    # Extract filename and create mapping
                    filename = Path(audio_path).stem
                    audio_mapping[filename] = audio_path

            print(f"✅ Loaded {len(audio_mapping)} audio file mappings")
            return audio_mapping
        except Exception as e:
            print(f"⚠️ Could not load training manifest: {e}")
            return {}

    def search_notes(self, selected_notes, max_files, sort_by_instrument, strict_mode, exclude_extra_notes):
        """Search for files containing selected notes with enhanced features"""
        if not selected_notes:
            return None, "Please select at least one note", [], []

        # Convert to list if single string
        if isinstance(selected_notes, str):
            selected_notes = [selected_notes]

        try:
            # Run note search engine
            cmd = [
                sys.executable, "/home/arlo/Data/note_search_engine.py"
            ] + selected_notes + ["--max-files", str(max_files)]

            # Add filtering options
            if strict_mode:
                cmd.append("--strict")
            if exclude_extra_notes:
                cmd.append("--exact")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                return None, f"Search failed: {result.stderr}", [], []

            # Load results
            search_name = "_".join(selected_notes).replace('#', 'sharp')
            if strict_mode:
                search_name += "_strict"
            if exclude_extra_notes:
                search_name += "_exact"
            results_dir = Path('/home/arlo/Data/note_search_results') / search_name
            playlist_file = results_dir / 'playlist.json'

            if not playlist_file.exists():
                return None, "No matching files found", [], []

            with open(playlist_file, 'r') as f:
                playlist_data = json.load(f)

            snippets = playlist_data.get('snippets', [])

            if not snippets:
                return None, "No audio snippets created", [], []

            # Enhance snippets with audio file paths and grouping
            enhanced_snippets = self.enhance_snippets(snippets, results_dir)

            # Sort by instrument if requested
            if sort_by_instrument:
                enhanced_snippets.sort(key=lambda x: (x['instrument_group'], x['instrument']))

            # Prepare audio files and descriptions
            audio_files = []
            descriptions = []
            preview_files = []

            for snippet in enhanced_snippets:
                # Generated MIDI audio
                audio_path = results_dir / snippet['audio_file']
                if audio_path.exists():
                    audio_files.append(str(audio_path))

                    # Description with enhanced info
                    desc = f"🎹 {snippet['original_file']}\n"
                    desc += f"🎵 Notes: {', '.join(snippet['notes_found'])}\n"
                    desc += f"🎼 {snippet['instrument']} ({snippet['instrument_group']}) - {snippet['session']}\n"
                    desc += f"⏱️ {snippet['duration']:.1f}s ({snippet['start_time']:.1f}s - {snippet['end_time']:.1f}s)\n"

                    if snippet['original_audio_path']:
                        desc += f"🔊 Original: {Path(snippet['original_audio_path']).name}"
                    else:
                        desc += f"🔊 Original audio: Not found"

                    descriptions.append(desc)

                    # Preview files (original audio + MIDI render)
                    preview_files.append({
                        'original_audio': snippet['original_audio_path'],
                        'midi_render': str(audio_path),
                        'midi_file': str(results_dir / snippet['midi_file']),
                        'snippet_info': snippet
                    })

            summary = f"🎵 Found {len(audio_files)} audio snippets with notes: {', '.join(selected_notes)}\n"
            if sort_by_instrument:
                summary += "📊 Sorted by instrument groups\n"
            summary += f"\n{result.stdout}"

            return (
                audio_files[0] if audio_files else None,
                summary,
                list(zip(audio_files, descriptions)),
                preview_files
            )

        except Exception as e:
            return None, f"Error: {e}", [], []

    def enhance_snippets(self, snippets, results_dir):
        """Enhance snippets with additional metadata"""
        enhanced = []

        for snippet in snippets:
            # Find original audio file
            filename_base = Path(snippet['original_file']).stem
            original_audio = self.training_manifest.get(filename_base, '')

            # Determine instrument group
            instrument = snippet['instrument']
            instrument_group = self.categorize_instrument(instrument)

            enhanced_snippet = snippet.copy()
            enhanced_snippet['original_audio_path'] = original_audio
            enhanced_snippet['instrument_group'] = instrument_group

            enhanced.append(enhanced_snippet)

        return enhanced

    def categorize_instrument(self, instrument_name):
        """Categorize instruments into groups"""
        instrument = instrument_name.upper()

        if any(x in instrument for x in ['GTR', 'GUITAR', 'AMP']):
            return '🎸 Guitar'
        elif any(x in instrument for x in ['BASS', 'BASS.']):
            return '🎸 Bass'
        elif any(x in instrument for x in ['DRUM', 'KICK', 'SNARE', 'HAT', 'CYMBAL']):
            return '🥁 Drums'
        elif any(x in instrument for x in ['VOC', 'VOCAL', 'VOICE']):
            return '🎤 Vocals'
        elif any(x in instrument for x in ['PIANO', 'KEYS', 'SYNTH', 'PAD']):
            return '🎹 Keys'
        elif any(x in instrument for x in ['GHOST', 'GUIDE']):
            return '👻 Guide'
        elif instrument.isdigit() or any(x in instrument for x in ['121', '414', 'U89']):
            return '🎛️ Track'
        else:
            return '🎵 Other'

def create_enhanced_interface():
    """Create the enhanced note search interface"""

    search_interface = EnhancedNoteSearchInterface()

    def search_and_play(selected_notes, max_files, sort_by_instrument, strict_mode, exclude_extra_notes):
        audio_file, summary, audio_list, preview_list = search_interface.search_notes(
            selected_notes, max_files, sort_by_instrument, strict_mode, exclude_extra_notes
        )

        # Update playlist choices with instrument groups
        if audio_list:
            playlist_choices = []
            for i, (_, desc) in enumerate(audio_list):
                lines = desc.split('\n')
                filename = lines[0].replace('🎹 ', '')
                instrument_info = lines[2].replace('🎼 ', '') if len(lines) > 2 else 'Unknown'
                playlist_choices.append(f"{i+1}. {filename} - {instrument_info.split(' - ')[0]}")
        else:
            playlist_choices = []

        return (
            audio_file,
            summary,
            gr.update(choices=playlist_choices, value=playlist_choices[0] if playlist_choices else None),
            audio_list,
            preview_list
        )

    def change_audio(playlist_selection, audio_list_state, preview_list_state):
        """Change audio when playlist selection changes"""
        if not playlist_selection or not audio_list_state or not preview_list_state:
            return None, "No audio available", None

        try:
            index = int(playlist_selection.split('.')[0]) - 1
            if 0 <= index < len(audio_list_state):
                audio_path, description = audio_list_state[index]
                preview_info = preview_list_state[index]

                # Prepare preview info
                preview_text = f"🎵 MIDI Render: {Path(preview_info['midi_render']).name}\n"
                if preview_info['original_audio'] and Path(preview_info['original_audio']).exists():
                    preview_text += f"🔊 Original Audio: {Path(preview_info['original_audio']).name}\n"
                    preview_text += f"📁 Path: {preview_info['original_audio']}\n"
                else:
                    preview_text += f"🔊 Original Audio: Not found\n"

                preview_text += f"🎼 MIDI File: {Path(preview_info['midi_file']).name}\n"
                preview_text += f"📊 Instrument Group: {preview_info['snippet_info']['instrument_group']}\n"
                preview_text += f"\n{description}"

                # Return original audio if available, otherwise MIDI render
                if preview_info['original_audio'] and Path(preview_info['original_audio']).exists():
                    return preview_info['original_audio'], preview_text, audio_path
                else:
                    return audio_path, preview_text, None

        except Exception as e:
            return None, f"Error loading audio: {e}", None

        return None, "Error loading audio", None

    # Create interface
    with gr.Blocks(title="Enhanced MIDI Note Search", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎵 Enhanced MIDI Note Search Interface

        Search for MIDI files containing specific note combinations with:
        - ✅ **Original audio preview** from training manifest
        - ✅ **MIDI renders** of detected sections
        - ✅ **Instrument grouping** and sorting
        - ✅ **Precise timing** information

        **This searches actual notes in MIDI data, not chord labels.**
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Note selection
                note_selector = gr.CheckboxGroup(
                    choices=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
                    label="🎹 Select Notes to Search For",
                    value=['C', 'E', 'G']
                )

                with gr.Row():
                    max_files_slider = gr.Slider(
                        minimum=1, maximum=20, value=10, step=1,
                        label="Max Files to Search"
                    )
                    sort_by_instrument = gr.Checkbox(
                        label="📊 Sort by Instrument Groups",
                        value=True
                    )

                with gr.Row():
                    strict_mode = gr.Checkbox(
                        label="🎯 Strict Mode (notes must be simultaneous)",
                        value=False
                    )
                    exclude_extra_notes = gr.Checkbox(
                        label="🚫 Exact Match (exclude sections with extra notes)",
                        value=False
                    )

                search_btn = gr.Button("🔍 Search for Note Combinations", variant="primary", size="lg")

            with gr.Column(scale=3):
                # Audio players
                with gr.Tabs():
                    with gr.TabItem("🔊 Original Audio"):
                        original_audio_player = gr.Audio(
                            label="Original Audio File",
                            autoplay=False
                        )

                    with gr.TabItem("🎼 MIDI Render"):
                        midi_audio_player = gr.Audio(
                            label="Generated MIDI Audio",
                            autoplay=False
                        )

                # Playlist dropdown
                playlist_dropdown = gr.Dropdown(
                    label="📋 Found Snippets (by instrument group)",
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
                file_preview_info = gr.Textbox(
                    label="File Preview & Details",
                    lines=8,
                    max_lines=10
                )

        # Hidden states
        audio_list_state = gr.State([])
        preview_list_state = gr.State([])

        # Event handlers
        search_btn.click(
            search_and_play,
            inputs=[note_selector, max_files_slider, sort_by_instrument, strict_mode, exclude_extra_notes],
            outputs=[midi_audio_player, search_results, playlist_dropdown, audio_list_state, preview_list_state]
        )

        playlist_dropdown.change(
            change_audio,
            inputs=[playlist_dropdown, audio_list_state, preview_list_state],
            outputs=[original_audio_player, file_preview_info, midi_audio_player]
        )

        # Instructions
        gr.Markdown("""
        ## 🎯 How to Use

        1. **Select notes** to find (e.g., C, E, G for C major)
        2. **Choose search options**:
           - Max files to search
           - Sort by instrument groups
           - 🎯 **Strict Mode**: Notes must play simultaneously (more precise)
           - 🚫 **Exact Match**: Exclude sections with extra notes (cleaner results)
        3. **Click search** to find matching MIDI sections (randomized each time)
        4. **Browse results** grouped by instrument type
        5. **Listen to both**:
           - 🔊 **Original Audio**: Full recording from training data
           - 🎼 **MIDI Render**: Generated audio of detected section

        ## 🎸 Instrument Groups

        - **🎸 Guitar/Bass**: GTR, AMP, BASS instruments
        - **🥁 Drums**: DRUM, KICK, SNARE, percussion
        - **🎤 Vocals**: VOC, VOCAL, VOICE tracks
        - **🎹 Keys**: PIANO, KEYS, SYNTH, PAD
        - **👻 Guide**: GHOST, guide tracks
        - **🎛️ Track**: Numbered tracks (121, 414, U89)

        ## 🎵 Example Searches

        - **Major chords**: C, E, G
        - **Minor chords**: D, F, A
        - **7th chords**: G, B, D, F
        - **Power chords**: C, G
        - **Single notes**: Just C (all C occurrences)

        Results: `/home/arlo/Data/note_search_results/`
        """)

    return interface

def main():
    # Check requirements
    analysis_path = "/home/arlo/Data/midi_analysis/chord_summary.csv"
    if not Path(analysis_path).exists():
        print(f"❌ {analysis_path} not found!")
        exit(1)

    print("🎵 Starting Enhanced MIDI Note Search Interface...")
    print("🔊 With original audio preview and instrument grouping")
    print("📱 Interface will open at: http://localhost:7864")

    # Launch interface
    interface = create_enhanced_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()