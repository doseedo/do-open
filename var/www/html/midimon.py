#!/usr/bin/env python3
"""
MIDI Monitor (midimon.py)
Converts madmom onset detection results to multitrack MIDI files for drum analysis.

This script processes the onset data from drum groups and creates MIDI files
with separate tracks for each microphone/drum element.
"""

import os
import json
import numpy as np
from pathlib import Path
import mido
from multiprocessing import Pool, cpu_count

# === CONFIG ===
DRUM_GROUPS_JSON = Path("/mnt/msdd/drum_groups.json")
MIDI_OUTPUT_DIR = Path("/home/arlo/gcs-bucket/drum_midi")
SAMPLE_RATE = 44100
MAX_WORKERS = min(8, cpu_count())  # Use up to 8 cores for MIDI processing

# === MIDI CONFIG ===
# General MIDI drum kit note mappings
DRUM_NOTE_MAP = {
    "kick": 36,      # C2 - Bass Drum 1
    "snare": 38,     # D2 - Acoustic Snare
    "hihat": 42,     # F#2 - Closed Hi-Hat
    "tom": 47,       # B2 - Low-Mid Tom (will adjust per tom)
    "ride": 51,      # D#3 - Ride Cymbal 1
    "crash": 49,     # C#3 - Crash Cymbal 1
    "oh": 46,        # A#2 - Open Hi-Hat (for overhead)
    "room": 46,      # A#2 - Open Hi-Hat (for room)
    "other": 37      # C#2 - Side Stick
}

# === HELPER FUNCTIONS ===
def role_from_name(name: str) -> str:
    """Determine drum role from filename"""
    n = name.lower()
    if any(k in n for k in ["bd", "kik", "kick", "k in", "kout", "kk"]): return "kick"
    if any(k in n for k in ["snare", "snr", "sntop", "snrtop", "snbot", "snrbottom", "rim"]): return "snare"
    if any(k in n for k in ["hihat", "hh", "hat", "chh", "ohh", "closedhat", "openhat"]): return "hihat"
    if any(k in n for k in ["tom", "rtom", "ftom", "racktom", "floortom"]): return "tom"
    if any(k in n for k in ["ride"]): return "ride"
    if any(k in n for k in ["crash", "china", "splash", "stack"]): return "crash"
    if any(k in n for k in ["overhead", "ohl", "ohr", "ovh"]): return "oh"
    if any(k in n for k in ["room"]): return "room"
    if any(k in n for k in ["kit", "drum", "drums"]): return "kit"
    return "other"

def get_midi_note_for_role(role: str) -> int:
    """Get MIDI note number for drum role"""
    return DRUM_NOTE_MAP.get(role, DRUM_NOTE_MAP["other"])

def get_onsets_from_file(onset_file_path: Path) -> np.ndarray:
    """Load onsets from saved .onsets.npy file"""
    try:
        onset_times = np.load(onset_file_path)
        return onset_times
    except Exception as e:
        print(f"❌ Error loading {onset_file_path}: {str(e)}")
        return np.array([])

def convert_group_to_midi(group_data) -> tuple[str, int]:
    """Convert a single drum group to a MIDI file with exact stem filenames"""
    session_name, group_idx, group = group_data

    try:
        files = group.get('files', [])
        if not files:
            return f"{session_name}_group_{group_idx+1:02d}", 0

        # Find the earliest onset time across all files in this group to normalize timing
        earliest_time = float('inf')
        file_onsets = {}

        for file_info in files:
            filename = file_info.get('filename', '')
            onsets_file_path = file_info.get('onsets_file')

            if not onsets_file_path or not Path(onsets_file_path).exists():
                continue

            onset_times = get_onsets_from_file(Path(onsets_file_path))
            if len(onset_times) > 0:
                file_onsets[filename] = onset_times
                earliest_time = min(earliest_time, onset_times.min())

        if not file_onsets:
            return f"{session_name}_group_{group_idx+1:02d}", 0

        print(f"🎯 {session_name} Group {group_idx+1}: {len(file_onsets)} stems (earliest: {earliest_time:.2f}s)")

        # Create MIDI file with multiple tracks
        midi_file = mido.MidiFile(ticks_per_beat=480)

        # Track 0: Tempo/meta information
        tempo_track = mido.MidiTrack()
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))
        tempo_track.append(mido.MetaMessage('track_name', name=f"Tempo - {session_name} Group {group_idx+1}"))
        midi_file.tracks.append(tempo_track)

        processed_files = 0

        for filename, onset_times in file_onsets.items():
            # Normalize onset times to start from 0
            normalized_onsets = onset_times - earliest_time

            # Determine drum role and MIDI note
            role = role_from_name(filename)
            midi_note = get_midi_note_for_role(role)

            # Create track with exact filename
            track = mido.MidiTrack()
            track.append(mido.MetaMessage('track_name', name=filename))  # Use exact filename
            track.append(mido.Message('program_change', channel=9, program=0))  # Channel 10 for drums

            # Convert onset times to MIDI ticks and add note events
            current_time = 0

            for onset_time in normalized_onsets:
                # Convert seconds to MIDI ticks (assuming 120 BPM)
                tick_time = int(onset_time * midi_file.ticks_per_beat * 2)  # 2 beats per second at 120 BPM
                delta_time = max(0, tick_time - current_time)

                # Add note on
                track.append(mido.Message('note_on', channel=9, note=midi_note, velocity=100, time=delta_time))

                # Add note off (short duration - 1/16 note at 120 BPM)
                note_duration = midi_file.ticks_per_beat // 4
                track.append(mido.Message('note_off', channel=9, note=midi_note, velocity=0, time=note_duration))

                current_time = tick_time + note_duration

            midi_file.tracks.append(track)
            processed_files += 1

        if processed_files > 0:
            # Save MIDI file per group
            midi_dir = MIDI_OUTPUT_DIR / session_name
            midi_dir.mkdir(parents=True, exist_ok=True)
            midi_path = midi_dir / f"group_{group_idx+1:02d}_stems.mid"

            midi_file.save(midi_path)
            print(f"  💾 {processed_files} stems -> {midi_path.name}")
            return f"{session_name}_group_{group_idx+1:02d}", 1

        return f"{session_name}_group_{group_idx+1:02d}", 0

    except Exception as e:
        print(f"❌ Error processing {session_name} Group {group_idx+1}: {str(e)}")
        return f"{session_name}_group_{group_idx+1:02d}", 0

def main():
    # Load drum groups data
    if not DRUM_GROUPS_JSON.exists():
        print(f"❌ Drum groups file not found: {DRUM_GROUPS_JSON}")
        print("   Run madmoncomp.py first to generate drum groups.")
        return

    print(f"📖 Loading drum groups from: {DRUM_GROUPS_JSON}")
    with open(DRUM_GROUPS_JSON, 'r') as f:
        drum_groups = json.load(f)

    if not drum_groups:
        print("❌ No drum groups found in the data file.")
        return

    # Prepare group data for parallel processing
    group_tasks = []
    total_groups = 0

    for session_name, groups in drum_groups.items():
        for group_idx, group in enumerate(groups):
            group_tasks.append((session_name, group_idx, group))
            total_groups += 1

    print(f"🎵 Found {total_groups} drum groups across {len(drum_groups)} sessions")
    print(f"🚀 Converting to MIDI using {MAX_WORKERS} CPU cores...")

    total_converted = 0

    # Process groups in parallel
    with Pool(processes=MAX_WORKERS) as pool:
        results = pool.map(convert_group_to_midi, group_tasks)

        for group_name, success_count in results:
            total_converted += success_count

    print(f"\n✅ Complete! Converted {total_converted} drum groups to MIDI files")
    print(f"📁 MIDI files saved to: {MIDI_OUTPUT_DIR}")

if __name__ == "__main__":
    main()