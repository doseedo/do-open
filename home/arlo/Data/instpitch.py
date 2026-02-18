#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instrument Pitch Coverage Visualizer

Combines instrument groups from instsort.py and pitch data from midianal.py
to create pitch coverage density graphs for each instrument group/subgroup.

Shows the frequency distribution of pitches (C-2 to C8) across all MIDI files
for each instrument category, visualized as color density heatmaps.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import pretty_midi

# MIDI note number to note name mapping
def midi_to_note_name(midi_note):
    """Convert MIDI note number to note name (e.g., 60 -> C4)"""
    if midi_note < 0 or midi_note > 127:
        return f"N{midi_note}"

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 2  # C4 = 60, so octave calculation
    note = note_names[midi_note % 12]
    return f"{note}{octave}"

def load_instrument_groups(instsort_file="instrument_groups.json"):
    """Load instrument groups from instsort.py output"""
    if not Path(instsort_file).exists():
        print(f"Error: Instrument groups file not found: {instsort_file}")
        print("   Run instsort.py first to generate instrument groups")
        return {}

    try:
        with open(instsort_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded instrument groups: {len(data)} groups")
        return data
    except Exception as e:
        print(f"Error loading instrument groups: {e}")
        return {}

def load_pitch_data(midianal_file="/home/arlo/Data/midi_analysis/chord_database.json"):
    """Load pitch analysis from midianal.py output"""
    if not Path(midianal_file).exists():
        print(f"Error: MIDI analysis file not found: {midianal_file}")
        print("   Run midianal.py first to extract pitch data")
        return {}

    try:
        with open(midianal_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded pitch data for {len(data)} MIDI files")
        return data
    except Exception as e:
        print(f"Error loading pitch data: {e}")
        return {}

def extract_pitch_frequencies(pitch_data, file_paths=None):
    """
    Extract pitch frequency data from MIDI analysis
    Returns: dict mapping file_path -> Counter of pitch frequencies
    """
    pitch_frequencies = {}

    # Handle both dict and list formats
    if isinstance(pitch_data, list):
        # Convert list to dict format for processing
        data_items = [(item.get('file_path', f'item_{i}'), item) for i, item in enumerate(pitch_data)]
    else:
        data_items = pitch_data.items()

    for file_path, analysis in data_items:
        if file_paths and file_path not in file_paths:
            continue

        pitch_counter = Counter()

        # Extract pitches from different analysis formats
        if 'instruments' in analysis:
            # Format from midianal.py
            for instrument in analysis['instruments']:
                if 'notes' in instrument:
                    for note in instrument['notes']:
                        pitch = note.get('pitch', note.get('note', 0))
                        duration = note.get('duration', note.get('end', 0) - note.get('start', 0))
                        # Weight by duration (longer notes count more)
                        pitch_counter[pitch] += max(duration, 0.1)

        elif 'chords' in analysis:
            # Format from chord_database.json
            for chord in analysis['chords']:
                if 'midi_notes' in chord:
                    duration = chord.get('duration', 1.0)
                    for midi_note in chord['midi_notes']:
                        pitch_counter[midi_note] += duration

        elif 'pitches' in analysis:
            # Simple pitch list format
            for pitch in analysis['pitches']:
                pitch_counter[pitch] += 1

        # Also try to extract directly from MIDI file if available
        try:
            midi_path = Path(file_path)
            if midi_path.exists() and midi_path.suffix.lower() in ['.mid', '.midi']:
                midi_data = pretty_midi.PrettyMIDI(str(midi_path))
                for instrument in midi_data.instruments:
                    if instrument.is_drum:
                        continue  # Skip drum tracks
                    for note in instrument.notes:
                        duration = note.end - note.start
                        pitch_counter[note.pitch] += duration
        except:
            pass  # Fallback to analysis data only

        pitch_frequencies[file_path] = pitch_counter

    return pitch_frequencies

def create_pitch_coverage_graph(group_name, subgroup_name, pitch_data, output_dir="pitch_coverage"):
    """
    Create a pitch coverage density graph for a specific group/subgroup
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Combine all pitch data for this group/subgroup
    combined_pitches = Counter()
    total_files = len(pitch_data)

    for file_pitch_data in pitch_data.values():
        for pitch, count in file_pitch_data.items():
            combined_pitches[pitch] += count

    if not combined_pitches:
        print(f"Warning: No pitch data found for {group_name}/{subgroup_name}")
        return None

    # Create pitch range (C-2 to C8 = MIDI 0 to 108)
    pitch_range = list(range(0, 109))  # 0-108 inclusive
    pitch_frequencies = [combined_pitches.get(pitch, 0) for pitch in pitch_range]

    # Normalize frequencies for color mapping
    max_freq = max(pitch_frequencies) if max(pitch_frequencies) > 0 else 1
    normalized_frequencies = [freq / max_freq for freq in pitch_frequencies]

    # Create the visualization
    fig, ax = plt.subplots(figsize=(16, 12))

    # Create heatmap-style visualization
    # Each pitch gets a horizontal bar colored by frequency
    colors = plt.cm.plasma(normalized_frequencies)  # Use plasma colormap

    # Create bars
    y_positions = pitch_range
    bar_heights = [0.8] * len(pitch_range)  # Consistent bar height

    bars = ax.barh(y_positions, [1] * len(pitch_range), height=bar_heights,
                   color=colors, edgecolor='none', alpha=0.8)

    # Customize the plot
    title = f"Pitch Coverage: {group_name}"
    if subgroup_name and subgroup_name != group_name:
        title += f" - {subgroup_name}"
    title += f" ({total_files} files)"

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Pitch Frequency (Normalized)', fontsize=12)
    ax.set_ylabel('MIDI Pitch (Note)', fontsize=12)

    # Set up Y-axis with note names
    # Show every octave + some key notes
    note_ticks = []
    note_labels = []

    for octave in range(-2, 9):  # C-2 to C8
        c_note = (octave + 2) * 12  # C of this octave
        if 0 <= c_note <= 108:
            note_ticks.append(c_note)
            note_labels.append(f"C{octave}")

    # Add some additional key notes for reference
    key_notes = [21, 33, 45, 57, 69, 81, 93, 105]  # A0, A1, A2, A3, A4, A5, A6, A7
    for note in key_notes:
        if note not in note_ticks and 0 <= note <= 108:
            note_ticks.append(note)
            note_labels.append(midi_to_note_name(note))

    # Sort ticks
    tick_pairs = list(zip(note_ticks, note_labels))
    tick_pairs.sort()
    note_ticks, note_labels = zip(*tick_pairs)

    ax.set_yticks(note_ticks)
    ax.set_yticklabels(note_labels, fontsize=10)
    ax.set_ylim(-1, 109)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=max_freq))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Pitch Frequency (Total Duration)', rotation=270, labelpad=20, fontsize=11)

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add statistics text
    most_common_pitches = combined_pitches.most_common(5)
    stats_text = f"Range: {midi_to_note_name(min(combined_pitches.keys()))} - {midi_to_note_name(max(combined_pitches.keys()))}\n"
    stats_text += "Most common:\n"
    for pitch, freq in most_common_pitches:
        stats_text += f"  {midi_to_note_name(pitch)}: {freq:.1f}\n"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    safe_name = f"{group_name}_{subgroup_name}".replace('/', '_').replace(' ', '_')
    output_path = output_dir / f"pitch_coverage_{safe_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created pitch coverage graph: {output_path}")
    return str(output_path)

def analyze_group_pitch_coverage_from_manifest(pitch_data, manifest_file="/home/arlo/Data/final_training_manifest_final.json"):
    """
    Analyze pitch coverage for all groups and subgroups using the training manifest
    """
    output_dir = Path("pitch_coverage")
    output_dir.mkdir(exist_ok=True)

    print(f"\nCreating pitch coverage visualizations...")

    # Load training manifest
    try:
        with open(manifest_file, 'r') as f:
            manifest_data = json.load(f)
        print(f"Loaded training manifest with {len(manifest_data)} entries")
    except Exception as e:
        print(f"Error loading manifest: {e}")
        return []

    # Create file path to group mappings
    audio_to_midi_map = {}
    group_files = {}
    subgroup_files = {}

    for entry in manifest_data:
        audio_path = entry.get('audio_path')
        group = entry.get('group')
        subgroup = entry.get('sub_group')

        if audio_path and group:
            # Try to find corresponding MIDI analysis in chord database
            # The chord database has BasicPitch MIDI files, so we need to map audio to MIDI
            audio_basename = Path(audio_path).stem

            # Look for matching entries in pitch_data
            matching_entries = [
                (path, data) for path, data in pitch_data.items()
                if audio_basename in path or Path(path).stem == audio_basename
            ]

            if matching_entries:
                # Use the first match
                midi_path, midi_data = matching_entries[0]

                # Add to group files
                if group not in group_files:
                    group_files[group] = {}
                if midi_path not in group_files[group]:
                    group_files[group][midi_path] = midi_data

                # Add to subgroup files
                if subgroup:
                    subgroup_key = f"{group}_{subgroup}"
                    if subgroup_key not in subgroup_files:
                        subgroup_files[subgroup_key] = {}
                    subgroup_files[subgroup_key][midi_path] = midi_data

    created_graphs = []

    # Process each group
    for group_name, group_pitch_data in group_files.items():
        print(f"\nProcessing group: {group_name}")
        print(f"  Found {len(group_pitch_data)} files with pitch data")

        if not group_pitch_data:
            print(f"   Warning: No pitch data found for group {group_name}")
            continue

        # Create group-level graph
        graph_path = create_pitch_coverage_graph(group_name, group_name, group_pitch_data, output_dir)
        if graph_path:
            created_graphs.append(graph_path)

    # Process subgroups
    print(f"\nProcessing subgroups...")
    for subgroup_key, subgroup_pitch_data in subgroup_files.items():
        group_name, subgroup_name = subgroup_key.split('_', 1)
        print(f"  Processing {group_name}/{subgroup_name}: {len(subgroup_pitch_data)} files")

        if len(subgroup_pitch_data) >= 2:  # Only if enough data
            graph_path = create_pitch_coverage_graph(group_name, subgroup_name, subgroup_pitch_data, output_dir)
            if graph_path:
                created_graphs.append(graph_path)
        else:
            print(f"   Warning: Insufficient data for subgroup {subgroup_name} ({len(subgroup_pitch_data)} files)")

    # Create overview comparison
    print(f"\nCreating group comparison...")
    comparison_graph = create_overview_comparison_from_groups(group_files, output_dir)
    if comparison_graph:
        created_graphs.append(comparison_graph)

    return created_graphs

def create_overview_comparison_from_groups(group_files, output_dir="pitch_coverage"):
    """
    Create an overview comparison showing all groups side by side
    """
    output_dir = Path(output_dir)

    # Calculate group ranges for comparison
    group_ranges = {}

    for group_name, group_pitch_data in group_files.items():
        # Combine pitch data
        combined_pitches = Counter()
        for file_pitch_data in group_pitch_data.values():
            for pitch, count in file_pitch_data.items():
                combined_pitches[pitch] += count

        if combined_pitches:
            group_ranges[group_name] = {
                'min_pitch': min(combined_pitches.keys()),
                'max_pitch': max(combined_pitches.keys()),
                'most_common': combined_pitches.most_common(1)[0][0],
                'total_files': len(group_pitch_data),
                'pitch_distribution': combined_pitches
            }

    # Create comparison visualization
    fig, ax = plt.subplots(figsize=(18, 12))

    colors = plt.cm.Set3(np.linspace(0, 1, len(group_ranges)))
    y_offset = 0

    for i, (group_name, data) in enumerate(group_ranges.items()):
        # Draw range bar
        min_pitch = data['min_pitch']
        max_pitch = data['max_pitch']
        most_common = data['most_common']

        # Main range bar
        ax.barh(y_offset, max_pitch - min_pitch, left=min_pitch, height=0.6,
                color=colors[i], alpha=0.7, label=group_name)

        # Mark most common pitch
        ax.plot(most_common, y_offset, 'o', color='red', markersize=8, markeredgecolor='black')

        # Add group label
        ax.text(most_common + 2, y_offset, f"{group_name} ({data['total_files']} files)",
                verticalalignment='center', fontsize=10, fontweight='bold')

        y_offset += 1

    # Format plot
    ax.set_xlabel('MIDI Pitch', fontsize=12)
    ax.set_ylabel('Instrument Groups', fontsize=12)
    ax.set_title('Instrument Group Pitch Range Comparison', fontsize=16, fontweight='bold', pad=20)

    # Set up X-axis with note names
    note_ticks = list(range(0, 128, 12))  # Every octave
    note_labels = [midi_to_note_name(note) for note in note_ticks]
    ax.set_xticks(note_ticks)
    ax.set_xticklabels(note_labels, rotation=45)

    ax.set_yticks(range(len(group_ranges)))
    ax.set_yticklabels([])  # Remove y-axis labels (already shown on bars)

    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-5, 128)

    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save comparison plot
    output_path = output_dir / "group_range_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created group comparison: {output_path}")
    return str(output_path)

def create_overview_comparison(instrument_groups, pitch_data, output_dir="pitch_coverage"):
    """
    Create an overview comparison showing all groups side by side
    """
    output_dir = Path(output_dir)

    # Calculate group ranges for comparison
    group_ranges = {}

    for group_name, group_data in instrument_groups.items():
        # Get all files for this group
        group_files = set()
        if 'files' in group_data:
            group_files.update(group_data['files'])
        if 'subgroups' in group_data:
            for subgroup_data in group_data['subgroups'].values():
                if 'files' in subgroup_data:
                    group_files.update(subgroup_data['files'])

        # Combine pitch data
        combined_pitches = Counter()
        for file_path in group_files:
            if file_path in pitch_data:
                for pitch, count in pitch_data[file_path].items():
                    combined_pitches[pitch] += count

        if combined_pitches:
            group_ranges[group_name] = {
                'min_pitch': min(combined_pitches.keys()),
                'max_pitch': max(combined_pitches.keys()),
                'most_common': combined_pitches.most_common(1)[0][0],
                'total_files': len(group_files),
                'pitch_distribution': combined_pitches
            }

    # Create comparison visualization
    fig, ax = plt.subplots(figsize=(18, 12))

    colors = plt.cm.Set3(np.linspace(0, 1, len(group_ranges)))
    y_offset = 0

    for i, (group_name, data) in enumerate(group_ranges.items()):
        # Draw range bar
        min_pitch = data['min_pitch']
        max_pitch = data['max_pitch']
        most_common = data['most_common']

        # Main range bar
        ax.barh(y_offset, max_pitch - min_pitch, left=min_pitch, height=0.6,
                color=colors[i], alpha=0.7, label=group_name)

        # Mark most common pitch
        ax.plot(most_common, y_offset, 'o', color='red', markersize=8, markeredgecolor='black')

        # Add group label
        ax.text(most_common + 2, y_offset, f"{group_name} ({data['total_files']} files)",
                verticalalignment='center', fontsize=10, fontweight='bold')

        y_offset += 1

    # Format plot
    ax.set_xlabel('MIDI Pitch', fontsize=12)
    ax.set_ylabel('Instrument Groups', fontsize=12)
    ax.set_title('Instrument Group Pitch Range Comparison', fontsize=16, fontweight='bold', pad=20)

    # Set up X-axis with note names
    note_ticks = list(range(0, 128, 12))  # Every octave
    note_labels = [midi_to_note_name(note) for note in note_ticks]
    ax.set_xticks(note_ticks)
    ax.set_xticklabels(note_labels, rotation=45)

    ax.set_yticks(range(len(group_ranges)))
    ax.set_yticklabels([])  # Remove y-axis labels (already shown on bars)

    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-5, 128)

    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save comparison plot
    output_path = output_dir / "group_range_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created group comparison: {output_path}")
    return str(output_path)

def main():
    """Main function to create pitch coverage visualizations"""
    print("Instrument Pitch Coverage Visualizer")
    print("=" * 50)

    # Load pitch data from chord database (this will be large - 4.6GB)
    print("Loading pitch analysis...")
    midi_analysis = load_pitch_data("/home/arlo/Data/midi_analysis/chord_database.json")
    if not midi_analysis:
        return

    # Extract pitch frequencies for all files
    print("Extracting pitch frequencies...")
    pitch_data = extract_pitch_frequencies(midi_analysis)

    if not pitch_data:
        print("Error: No pitch frequency data extracted")
        return

    print(f"Processed pitch data for {len(pitch_data)} files")

    # Create pitch coverage visualizations using the training manifest
    print("\nCreating visualizations...")
    created_graphs = analyze_group_pitch_coverage_from_manifest(pitch_data)

    print(f"\nCreated {len(created_graphs)} visualizations:")
    for graph in created_graphs:
        print(f"   {graph}")

    print(f"\nAll pitch coverage visualizations saved to ./pitch_coverage/")

if __name__ == "__main__":
    main()