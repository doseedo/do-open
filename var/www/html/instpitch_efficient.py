#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficient Instrument Pitch Coverage Visualizer

Memory-efficient version that processes large datasets using streaming and chunking.
Creates pitch coverage density graphs from the training manifest and chord database.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter

# MIDI note number to note name mapping
def midi_to_note_name(midi_note):
    """Convert MIDI note number to note name (e.g., 60 -> C4)"""
    if midi_note < 0 or midi_note > 127:
        return f"N{midi_note}"

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 2  # C4 = 60, so octave calculation
    note = note_names[midi_note % 12]
    return f"{note}{octave}"

def process_chord_database_by_groups(chord_db_file, manifest_file, output_dir="pitch_coverage"):
    """Process the chord database efficiently by streaming and grouping by instrument"""

    print("Loading training manifest...")
    with open(manifest_file, 'r') as f:
        manifest_data = json.load(f)

    # Create mappings from audio filename to group info
    audio_basename_to_group = {}
    group_counts = Counter()
    subgroup_counts = Counter()

    for entry in manifest_data:
        audio_path = entry.get('audio_path', '')
        group = entry.get('group')
        subgroup = entry.get('sub_group')

        if audio_path and group:
            # Extract basename for matching
            basename = Path(audio_path).stem
            # Remove common patterns that might differ between audio and MIDI
            clean_basename = basename.replace('.wav', '').replace('.mp3', '').replace('.aif', '')

            audio_basename_to_group[clean_basename] = {
                'group': group,
                'subgroup': subgroup
            }
            group_counts[group] += 1
            if subgroup:
                subgroup_counts[f"{group}_{subgroup}"] += 1

    print(f"Mapped {len(audio_basename_to_group)} audio files to groups")
    print(f"Groups found: {dict(group_counts)}")

    # Initialize pitch counters for each group
    group_pitches = defaultdict(Counter)
    subgroup_pitches = defaultdict(Counter)
    matched_files = 0
    total_processed = 0

    print("\nProcessing chord database (streaming)...")

    # Stream through the chord database
    with open(chord_db_file, 'r') as f:
        # Read opening bracket
        f.read(1)  # Skip '['

        chord_entry = ""
        in_string = False
        bracket_depth = 0

        while True:
            char = f.read(1)
            if not char:
                break

            if char == '"' and (not chord_entry or chord_entry[-1] != '\\'):
                in_string = not in_string

            if not in_string:
                if char == '{':
                    bracket_depth += 1
                elif char == '}':
                    bracket_depth -= 1

                if bracket_depth == 0 and char in ',]':
                    # We have a complete entry
                    if chord_entry.strip():
                        try:
                            entry = json.loads('{' + chord_entry + '}')
                            total_processed += 1

                            if total_processed % 10000 == 0:
                                print(f"  Processed {total_processed} entries, matched {matched_files}")

                            # Extract pitch data
                            file_path = entry.get('file_path', '')
                            filename = entry.get('filename', '')

                            # Try to match with manifest
                            basename = Path(filename).stem if filename else Path(file_path).stem
                            clean_basename = basename.replace('.mid', '').replace('.midi', '')

                            group_info = None
                            # Try exact match first
                            if clean_basename in audio_basename_to_group:
                                group_info = audio_basename_to_group[clean_basename]
                            else:
                                # Try partial matches
                                for audio_base, info in audio_basename_to_group.items():
                                    if audio_base in clean_basename or clean_basename in audio_base:
                                        group_info = info
                                        break

                            if group_info:
                                matched_files += 1
                                group = group_info['group']
                                subgroup = group_info['subgroup']

                                # Extract pitches from chords
                                chords = entry.get('chords', [])
                                for chord in chords:
                                    midi_notes = chord.get('midi_notes', [])
                                    duration = chord.get('duration', 1.0)

                                    for midi_note in midi_notes:
                                        group_pitches[group][midi_note] += duration
                                        if subgroup:
                                            subgroup_key = f"{group}_{subgroup}"
                                            subgroup_pitches[subgroup_key][midi_note] += duration

                        except json.JSONDecodeError:
                            pass  # Skip malformed entries

                    chord_entry = ""
                    if char == ']':
                        break
                    continue

            if bracket_depth > 0:
                chord_entry += char

    print(f"\nCompleted processing: {total_processed} total entries, {matched_files} matched to groups")

    # Create visualizations
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    created_graphs = []

    # Create group visualizations
    for group, pitch_counter in group_pitches.items():
        if len(pitch_counter) > 0:
            print(f"Creating visualization for {group}: {len(pitch_counter)} unique pitches")
            graph_path = create_pitch_coverage_graph(group, group, pitch_counter, output_dir)
            if graph_path:
                created_graphs.append(graph_path)

    # Create subgroup visualizations
    for subgroup_key, pitch_counter in subgroup_pitches.items():
        if len(pitch_counter) > 10:  # Only create if substantial data
            group, subgroup = subgroup_key.split('_', 1)
            print(f"Creating visualization for {group}/{subgroup}: {len(pitch_counter)} unique pitches")
            graph_path = create_pitch_coverage_graph(group, subgroup, pitch_counter, output_dir)
            if graph_path:
                created_graphs.append(graph_path)

    # Create overview comparison
    comparison_path = create_group_comparison(group_pitches, output_dir)
    if comparison_path:
        created_graphs.append(comparison_path)

    return created_graphs

def create_pitch_coverage_graph(group_name, subgroup_name, pitch_counter, output_dir):
    """Create a pitch coverage density graph"""

    if not pitch_counter:
        return None

    # Create pitch range (C-2 to C8 = MIDI 0 to 108)
    pitch_range = list(range(0, 109))
    pitch_frequencies = [pitch_counter.get(pitch, 0) for pitch in pitch_range]

    # Normalize frequencies for color mapping
    max_freq = max(pitch_frequencies) if max(pitch_frequencies) > 0 else 1
    normalized_frequencies = [freq / max_freq for freq in pitch_frequencies]

    # Create the visualization
    fig, ax = plt.subplots(figsize=(16, 12))

    # Create heatmap-style visualization
    colors = plt.cm.plasma(normalized_frequencies)

    # Create bars
    y_positions = pitch_range
    bar_heights = [0.8] * len(pitch_range)

    bars = ax.barh(y_positions, [1] * len(pitch_range), height=bar_heights,
                   color=colors, edgecolor='none', alpha=0.8)

    # Customize the plot
    title = f"Pitch Coverage: {group_name}"
    if subgroup_name and subgroup_name != group_name:
        title += f" - {subgroup_name}"
    title += f" ({sum(pitch_frequencies):.0f} total occurrences)"

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Pitch Frequency (Normalized)', fontsize=12)
    ax.set_ylabel('MIDI Pitch (Note)', fontsize=12)

    # Set up Y-axis with note names
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
    note_ticks, note_labels = zip(*tick_pairs) if tick_pairs else ([], [])

    ax.set_yticks(note_ticks)
    ax.set_yticklabels(note_labels, fontsize=10)
    ax.set_ylim(-1, 109)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=max_freq))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Pitch Frequency', rotation=270, labelpad=20, fontsize=11)

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add statistics text
    most_common = pitch_counter.most_common(5)
    stats_text = f"Range: {midi_to_note_name(min(pitch_counter.keys()))} - {midi_to_note_name(max(pitch_counter.keys()))}\n"
    stats_text += "Most common:\n"
    for pitch, freq in most_common:
        stats_text += f"  {midi_to_note_name(pitch)}: {freq:.1f}\n"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    safe_name = f"{group_name}_{subgroup_name}".replace('/', '_').replace(' ', '_')
    output_path = output_dir / f"pitch_coverage_{safe_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Created: {output_path}")
    return str(output_path)

def create_group_comparison(group_pitches, output_dir):
    """Create an overview comparison of all groups"""

    fig, ax = plt.subplots(figsize=(18, 12))

    colors = plt.cm.Set3(np.linspace(0, 1, len(group_pitches)))
    y_offset = 0

    for i, (group_name, pitch_counter) in enumerate(group_pitches.items()):
        if not pitch_counter:
            continue

        min_pitch = min(pitch_counter.keys())
        max_pitch = max(pitch_counter.keys())
        most_common = pitch_counter.most_common(1)[0][0]
        total_occurrences = sum(pitch_counter.values())

        # Main range bar
        ax.barh(y_offset, max_pitch - min_pitch, left=min_pitch, height=0.6,
                color=colors[i], alpha=0.7, label=group_name)

        # Mark most common pitch
        ax.plot(most_common, y_offset, 'o', color='red', markersize=8, markeredgecolor='black')

        # Add group label
        ax.text(most_common + 2, y_offset, f"{group_name} ({total_occurrences:.0f} occurrences)",
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

    ax.set_yticks(range(len(group_pitches)))
    ax.set_yticklabels([])

    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-5, 128)

    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save comparison plot
    output_path = output_dir / "group_range_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Created: {output_path}")
    return str(output_path)

def main():
    """Main function"""
    print("Efficient Instrument Pitch Coverage Visualizer")
    print("=" * 60)

    chord_db_file = "/home/arlo/Data/midi_analysis/chord_database.json"
    manifest_file = "/home/arlo/Data/final_training_manifest_final.json"

    # Check if files exist
    if not Path(chord_db_file).exists():
        print(f"Error: Chord database not found: {chord_db_file}")
        return

    if not Path(manifest_file).exists():
        print(f"Error: Training manifest not found: {manifest_file}")
        return

    # Process data and create visualizations
    created_graphs = process_chord_database_by_groups(chord_db_file, manifest_file)

    print(f"\nCreated {len(created_graphs)} visualizations:")
    for graph in created_graphs:
        print(f"   {graph}")

    print(f"\nAll pitch coverage visualizations saved to ./pitch_coverage/")

if __name__ == "__main__":
    main()