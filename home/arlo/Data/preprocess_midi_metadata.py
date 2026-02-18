#!/usr/bin/env python3
"""
MIDI Metadata Preprocessing Script
Analyzes MIDI files and extracts track/instrument information for the search interface
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import pretty_midi

# General MIDI instrument categories
GM_INSTRUMENT_CATEGORIES = {
    # Piano (0-7)
    range(0, 8): "Piano",
    # Chromatic Percussion (8-15)
    range(8, 16): "Chromatic Percussion",
    # Organ (16-23)
    range(16, 24): "Organ",
    # Guitar (24-31)
    range(24, 32): "Guitar",
    # Bass (32-39)
    range(32, 40): "Bass",
    # Strings (40-47)
    range(40, 48): "Strings",
    # Ensemble (48-55)
    range(48, 56): "Ensemble",
    # Brass (56-63)
    range(56, 64): "Brass",
    # Reed (64-71)
    range(64, 72): "Reed",
    # Pipe (72-79)
    range(72, 80): "Pipe",
    # Synth Lead (80-87)
    range(80, 88): "Synth Lead",
    # Synth Pad (88-95)
    range(88, 96): "Synth Pad",
    # Synth Effects (96-103)
    range(96, 104): "Synth Effects",
    # Ethnic (104-111)
    range(104, 112): "Ethnic",
    # Percussive (112-119)
    range(112, 120): "Percussive",
    # Sound Effects (120-127)
    range(120, 128): "Sound Effects",
}

def get_instrument_category(program: int) -> str:
    """Get instrument category from MIDI program number"""
    for range_obj, category in GM_INSTRUMENT_CATEGORIES.items():
        if program in range_obj:
            return category
    return "Unknown"

def analyze_midi_file(midi_path: str) -> Dict:
    """
    Analyze a MIDI file and extract comprehensive metadata

    Args:
        midi_path: Path to MIDI file

    Returns:
        Dictionary containing MIDI metadata including tracks and instruments
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        # Basic file info
        metadata = {
            "filename": Path(midi_path).name,
            "duration": midi_data.get_end_time(),
            "tempo": midi_data.estimate_tempo(),
            "time_signature_changes": len(midi_data.time_signature_changes),
            "key_signature_changes": len(midi_data.key_signature_changes),
            "total_tracks": len(midi_data.instruments),
            "tracks": []
        }

        # Analyze each track/instrument
        for idx, instrument in enumerate(midi_data.instruments):
            track_info = {
                "track_number": idx + 1,
                "name": instrument.name if instrument.name else f"Track {idx + 1}",
                "is_drum": instrument.is_drum,
                "program": instrument.program if not instrument.is_drum else 128,  # 128 = drums
                "instrument_name": pretty_midi.program_to_instrument_name(instrument.program) if not instrument.is_drum else "Drums",
                "category": "Drums" if instrument.is_drum else get_instrument_category(instrument.program),
                "note_count": len(instrument.notes),
                "pitch_range": {
                    "min": min([note.pitch for note in instrument.notes]) if instrument.notes else 0,
                    "max": max([note.pitch for note in instrument.notes]) if instrument.notes else 0
                } if instrument.notes else None,
                "control_changes": len(instrument.control_changes),
                "pitch_bends": len(instrument.pitch_bends)
            }

            metadata["tracks"].append(track_info)

        # Summary statistics
        metadata["summary"] = {
            "total_notes": sum(len(inst.notes) for inst in midi_data.instruments),
            "drum_tracks": sum(1 for inst in midi_data.instruments if inst.is_drum),
            "melodic_tracks": sum(1 for inst in midi_data.instruments if not inst.is_drum),
            "categories_used": list(set(
                "Drums" if inst.is_drum else get_instrument_category(inst.program)
                for inst in midi_data.instruments
            ))
        }

        return metadata

    except Exception as e:
        return {
            "filename": Path(midi_path).name,
            "error": str(e),
            "tracks": []
        }

def preprocess_midi_directory(midi_dir: str, output_json: str = None, verbose: bool = False):
    """
    Preprocess all MIDI files in a directory and save metadata

    Args:
        midi_dir: Path to directory containing MIDI files
        output_json: Path to save JSON metadata (optional)
        verbose: Show detailed track/instrument information (default: False)
    """
    midi_path = Path(midi_dir)

    if not midi_path.exists():
        print(f"❌ Directory not found: {midi_dir}")
        return {}

    metadata_db = {}
    midi_files = list(midi_path.glob("*.mid")) + list(midi_path.glob("*.midi"))

    print(f"📁 Found {len(midi_files)} MIDI files in {midi_dir}")
    print(f"🔍 Analyzing MIDI files...")

    for idx, midi_file in enumerate(midi_files, 1):
        print(f"   [{idx}/{len(midi_files)}] {midi_file.name}...", end=" ")
        metadata = analyze_midi_file(str(midi_file))
        metadata_db[midi_file.name] = metadata

        if "error" in metadata:
            print(f"❌ Error: {metadata['error']}")
        else:
            print(f"✅ {metadata['total_tracks']} tracks")

            # Show detailed track info if verbose
            if verbose and metadata.get('tracks'):
                for track in metadata['tracks']:
                    icon = "🥁" if track['is_drum'] else "🎹"
                    print(f"      {icon} Track {track['track_number']}: {track['instrument_name']} (program {track['program']}) - {track['note_count']} notes")

    # Save to JSON if output path provided
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(metadata_db, f, indent=2)
        print(f"\n✅ Saved metadata to: {output_json}")

    return metadata_db

def get_midi_info(midi_path: str) -> Dict:
    """
    Get MIDI info for a single file (used by API endpoint)

    Args:
        midi_path: Path to MIDI file

    Returns:
        Dictionary containing MIDI metadata
    """
    return analyze_midi_file(midi_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess MIDI files and extract metadata")
    parser.add_argument("--midi_dir", default="/home/arlo/free-midi-chords/MIDIS",
                       help="Directory containing MIDI files")
    parser.add_argument("--output", default="/home/arlo/Data/midi_metadata.json",
                       help="Output JSON file for metadata")
    parser.add_argument("--single_file", help="Analyze a single MIDI file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed track/instrument information")

    args = parser.parse_args()

    if args.single_file:
        # Analyze single file
        metadata = get_midi_info(args.single_file)
        print(json.dumps(metadata, indent=2))
    else:
        # Preprocess entire directory
        preprocess_midi_directory(args.midi_dir, args.output, verbose=args.verbose)
