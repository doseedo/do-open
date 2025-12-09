#!/usr/bin/env python3
"""
Fix Decode Duplicates - Diagnose and fix overlapping pattern expansion.

The codec stores hierarchical patterns:
- Short 3-note pattern
- 7-note pattern (includes 3-note as prefix)
- 19-note pattern (includes both)

The bug: All occurrences at the same (piece, beat, instrument) are expanded,
causing 3x duplication of notes.

The fix: At each (piece, beat, instrument), only expand the LONGEST pattern.
This is the "greedy covering" approach - use the most specific pattern.
"""

import orjson
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


def diagnose_duplicates(patterns: dict, verbose: bool = True) -> dict:
    """Diagnose duplicate occurrence issue.

    Returns stats about how many locations have overlapping patterns.
    """
    # Group occurrences by (piece, onset, gm)
    location_patterns = defaultdict(list)

    for pid, p in patterns.items():
        gm = p.get('gm_program', 0)
        n_notes = len(p.get('pitch_intervals', [])) + 1

        for occ in p.get('occurrences', []):
            piece = occ.get('piece_id', 'unknown')
            onset = occ.get('onset_time', 0)

            location_patterns[(piece, onset, gm)].append({
                'pattern_id': pid,
                'n_notes': n_notes,
                'first_pitch': occ.get('first_pitch', 60),
                'tau': occ.get('tau_offset', 480)
            })

    # Count locations with multiple patterns
    single_locations = 0
    multi_locations = 0
    total_extra_occurrences = 0

    multi_examples = []

    for loc, items in location_patterns.items():
        if len(items) == 1:
            single_locations += 1
        else:
            multi_locations += 1
            total_extra_occurrences += len(items) - 1
            if len(multi_examples) < 5:
                multi_examples.append((loc, items))

    if verbose:
        print("=" * 60)
        print("DUPLICATE OCCURRENCE DIAGNOSIS")
        print("=" * 60)
        print(f"\nTotal unique locations (piece, onset, gm): {len(location_patterns)}")
        print(f"Locations with single pattern: {single_locations}")
        print(f"Locations with MULTIPLE patterns: {multi_locations}")
        print(f"Extra occurrences (would cause duplicates): {total_extra_occurrences}")

        if multi_examples:
            print("\nExamples of multi-pattern locations:")
            for loc, items in multi_examples[:3]:
                piece, onset, gm = loc
                print(f"\n  Location: piece={piece[:30]}, onset={onset}, GM={gm}")
                print(f"  Patterns at this location:")
                for item in sorted(items, key=lambda x: -x['n_notes']):
                    print(f"    - {item['pattern_id']}: {item['n_notes']} notes, pitch={item['first_pitch']}")

    return {
        'total_locations': len(location_patterns),
        'single_pattern_locations': single_locations,
        'multi_pattern_locations': multi_locations,
        'extra_occurrences': total_extra_occurrences,
        'location_patterns': location_patterns
    }


def build_deduplicated_index(patterns: dict) -> Dict[Tuple, dict]:
    """Build deduplicated index - only keep longest pattern at each location.

    Returns: {(piece, onset, gm): {pattern_info}}
    """
    # First get all occurrences
    diagnosis = diagnose_duplicates(patterns, verbose=False)
    location_patterns = diagnosis['location_patterns']

    # At each location, keep only the longest pattern
    deduped = {}
    for loc, items in location_patterns.items():
        # Sort by n_notes descending, pick first (longest)
        sorted_items = sorted(items, key=lambda x: -x['n_notes'])
        best = sorted_items[0]

        # Get full pattern info
        pid = best['pattern_id']
        p = patterns[pid]

        deduped[loc] = {
            'pattern_id': pid,
            'pattern': p,
            'first_pitch': best['first_pitch'],
            'tau': best['tau'],
            'n_notes': best['n_notes']
        }

    return deduped


def extract_corpus_excerpt_deduped(
    patterns: dict,
    instruments: List[int],
    min_length: int = 32,
    output_path: str = '/tmp/corpus_excerpt_deduped.mid'
) -> str:
    """Extract corpus excerpt with deduplication fix.

    This is the corrected version of extract_corpus_excerpt.py
    """
    print(f"Looking for segments with instruments: {instruments}")
    print(f"Minimum length: {min_length} beats")

    # Build deduplicated index
    print("\nBuilding deduplicated index...")
    deduped = build_deduplicated_index(patterns)
    print(f"  Total deduplicated locations: {len(deduped)}")

    # Filter to requested instruments
    filtered = {k: v for k, v in deduped.items() if k[2] in instruments}
    print(f"  Locations for target instruments: {len(filtered)}")

    # Group by (piece, beat)
    piece_beat_data = defaultdict(lambda: defaultdict(list))

    for (piece, onset, gm), info in filtered.items():
        beat = onset // 480
        piece_beat_data[piece][beat].append({
            'gm': gm,
            'onset': onset,
            **info
        })

    # Find pieces with good coverage
    candidates = []

    for piece, beat_data in piece_beat_data.items():
        sorted_beats = sorted(beat_data.keys())
        if not sorted_beats:
            continue

        # Find runs where at least 2 instruments present
        current_run_start = None
        current_run_length = 0
        best_run_start = None
        best_run_length = 0

        for beat in sorted_beats:
            gms_present = set(d['gm'] for d in beat_data[beat])
            has_enough = len(gms_present & set(instruments)) >= 2

            if has_enough:
                if current_run_start is None:
                    current_run_start = beat
                    current_run_length = 1
                else:
                    current_run_length += 1

                if current_run_length > best_run_length:
                    best_run_length = current_run_length
                    best_run_start = current_run_start
            else:
                current_run_start = None
                current_run_length = 0

        if best_run_length >= min_length:
            candidates.append((piece, best_run_start, best_run_length))

    if not candidates:
        print("No suitable segments found!")
        return None

    candidates.sort(key=lambda x: -x[2])

    print(f"\nFound {len(candidates)} candidate segments:")
    for piece, start, length in candidates[:5]:
        print(f"  {piece[:40]}: beat {start}-{start+length} ({length} beats)")

    # Extract best segment
    piece, start_beat, run_length = candidates[0]
    end_beat = start_beat + min(run_length, 64)

    print(f"\nExtracting: {piece[:40]} beats {start_beat}-{end_beat}")

    # Collect notes
    notes_by_gm = defaultdict(list)
    beat_data = piece_beat_data[piece]

    for beat in range(start_beat, end_beat):
        if beat not in beat_data:
            continue

        for item in beat_data[beat]:
            gm = item['gm']
            base_pitch = item['first_pitch']
            base_onset = item['onset']

            p = item['pattern']
            intervals = p.get('pitch_intervals', [0])
            rhythm_ratios = p.get('rhythm_ratios', [1.0])
            duration_ratios = p.get('duration_ratios', [0.9])
            tau = item['tau']

            # Expand pattern
            pitch = base_pitch
            time = base_onset - (start_beat * 480)

            for i, interval in enumerate([0] + list(intervals)):
                if i > 0:
                    pitch += interval

                ioi = int(tau * rhythm_ratios[i]) if i < len(rhythm_ratios) else tau
                dur = int(ioi * duration_ratios[i] * 0.9) if i < len(duration_ratios) else int(ioi * 0.9)

                notes_by_gm[gm].append({
                    'pitch': max(0, min(127, pitch)),
                    'onset': max(0, time),
                    'duration': max(1, dur),
                    'velocity': 80
                })

                time += ioi

    # Dedupe notes (same pitch, same onset within same track)
    for gm in notes_by_gm:
        seen = set()
        deduped_notes = []
        for n in notes_by_gm[gm]:
            key = (n['pitch'], n['onset'])
            if key not in seen:
                seen.add(key)
                deduped_notes.append(n)
        notes_by_gm[gm] = deduped_notes

    # Build MIDI
    mid = MidiFile(ticks_per_beat=480, type=1)

    tempo_track = MidiTrack()
    mid.tracks.append(tempo_track)
    tempo_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
    tempo_track.append(MetaMessage('end_of_track', time=0))

    channel_map = {}
    next_channel = 0

    gm_names = {
        32: "Acoustic Bass",
        65: "Alto Sax",
        66: "Tenor Sax",
        67: "Baritone Sax"
    }

    for gm in sorted(notes_by_gm.keys()):
        if gm not in channel_map:
            channel_map[gm] = next_channel
            next_channel += 1
            if next_channel == 9:
                next_channel = 10

        channel = channel_map[gm]
        track = MidiTrack()
        mid.tracks.append(track)

        name = gm_names.get(gm, f"GM {gm}")
        track.append(MetaMessage('track_name', name=name, time=0))
        track.append(Message('program_change', program=gm % 128, channel=channel, time=0))

        notes = sorted(notes_by_gm[gm], key=lambda n: n['onset'])

        events = []
        for n in notes:
            events.append((n['onset'], 'on', n['pitch'], n['velocity']))
            events.append((n['onset'] + n['duration'], 'off', n['pitch'], 0))

        events.sort(key=lambda x: (x[0], x[1] == 'on'))

        last_time = 0
        for event_time, event_type, pitch, vel in events:
            delta = max(0, event_time - last_time)
            if event_type == 'on':
                track.append(Message('note_on', note=pitch, velocity=vel, channel=channel, time=delta))
            else:
                track.append(Message('note_off', note=pitch, velocity=0, channel=channel, time=delta))
            last_time = event_time

        track.append(MetaMessage('end_of_track', time=0))

    mid.save(output_path)

    total_notes = sum(len(n) for n in notes_by_gm.values())
    print(f"\nSaved to: {output_path}")
    print(f"  Piece: {piece}")
    print(f"  Beats: {start_beat}-{end_beat} ({end_beat - start_beat} beats)")
    print(f"  Tracks: {len(mid.tracks) - 1}")  # -1 for tempo track
    print(f"  Notes: {total_notes}")
    print(f"  Instruments: {list(notes_by_gm.keys())}")

    return piece


def compare_with_without_dedupe(patterns: dict, instruments: List[int]):
    """Compare note counts with and without deduplication."""
    print("\n" + "=" * 60)
    print("COMPARING WITH/WITHOUT DEDUPLICATION")
    print("=" * 60)

    # Extract with original method (has dupes)
    print("\nExtracting WITHOUT deduplication...")
    extract_corpus_excerpt_original(patterns, instruments, output_path='/tmp/corpus_with_dupes.mid')
    mid1 = MidiFile('/tmp/corpus_with_dupes.mid')
    notes1 = count_midi_notes(mid1)

    # Extract with deduplication
    print("\nExtracting WITH deduplication...")
    extract_corpus_excerpt_deduped(patterns, instruments, output_path='/tmp/corpus_deduped.mid')
    mid2 = MidiFile('/tmp/corpus_deduped.mid')
    notes2 = count_midi_notes(mid2)

    print("\n" + "-" * 40)
    print(f"Notes WITHOUT deduplication: {notes1}")
    print(f"Notes WITH deduplication:    {notes2}")
    print(f"Duplicate ratio:             {notes1/notes2:.2f}x")
    print("-" * 40)


def extract_corpus_excerpt_original(
    patterns: dict,
    instruments: List[int],
    min_length: int = 32,
    output_path: str = '/tmp/corpus_original.mid'
):
    """Original extraction (with duplicates for comparison)."""
    piece_beat_data = defaultdict(lambda: defaultdict(list))

    for pid, p in patterns.items():
        gm = p.get('gm_program', 0)
        if gm not in instruments:
            continue

        for occ in p.get('occurrences', []):
            piece = occ.get('piece_id', 'unknown')
            onset = occ.get('onset_time', 0)
            pitch = occ.get('first_pitch', 60)
            beat = onset // 480

            piece_beat_data[piece][beat].append({
                'gm': gm,
                'pitch': pitch,
                'onset': onset,
                'intervals': p.get('pitch_intervals', [0]),
                'rhythm_ratios': p.get('rhythm_ratios', [1.0]),
                'duration_ratios': p.get('duration_ratios', [0.9]),
                'tau': occ.get('tau_offset', 480)
            })

    # Find best segment
    candidates = []
    for piece, beat_data in piece_beat_data.items():
        sorted_beats = sorted(beat_data.keys())
        if len(sorted_beats) >= min_length:
            candidates.append((piece, sorted_beats[0], len(sorted_beats)))

    if not candidates:
        return

    candidates.sort(key=lambda x: -x[2])
    piece, start_beat, _ = candidates[0]
    end_beat = start_beat + 64

    # Collect ALL notes (with duplicates)
    notes_by_gm = defaultdict(list)
    beat_data = piece_beat_data[piece]

    for beat in range(start_beat, end_beat):
        if beat not in beat_data:
            continue

        for item in beat_data[beat]:
            gm = item['gm']
            base_pitch = item['pitch']
            base_onset = item['onset']
            intervals = item['intervals']
            rhythm_ratios = item['rhythm_ratios']
            duration_ratios = item['duration_ratios']
            tau = item['tau']

            pitch = base_pitch
            time = base_onset - (start_beat * 480)

            for i, interval in enumerate([0] + list(intervals)):
                if i > 0:
                    pitch += interval

                ioi = int(tau * rhythm_ratios[i]) if i < len(rhythm_ratios) else tau
                dur = int(ioi * duration_ratios[i] * 0.9) if i < len(duration_ratios) else int(ioi * 0.9)

                notes_by_gm[gm].append({
                    'pitch': max(0, min(127, pitch)),
                    'onset': max(0, time),
                    'duration': max(1, dur),
                    'velocity': 80
                })
                time += ioi

    # Build MIDI (no deduplication)
    mid = MidiFile(ticks_per_beat=480, type=1)

    tempo_track = MidiTrack()
    mid.tracks.append(tempo_track)
    tempo_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
    tempo_track.append(MetaMessage('end_of_track', time=0))

    next_channel = 0
    for gm in sorted(notes_by_gm.keys()):
        channel = next_channel
        next_channel += 1
        if next_channel == 9:
            next_channel = 10

        track = MidiTrack()
        mid.tracks.append(track)
        track.append(Message('program_change', program=gm % 128, channel=channel, time=0))

        notes = sorted(notes_by_gm[gm], key=lambda n: n['onset'])

        events = []
        for n in notes:
            events.append((n['onset'], 'on', n['pitch'], n['velocity']))
            events.append((n['onset'] + n['duration'], 'off', n['pitch'], 0))

        events.sort(key=lambda x: (x[0], x[1] == 'on'))

        last_time = 0
        for event_time, event_type, pitch, vel in events:
            delta = max(0, event_time - last_time)
            if event_type == 'on':
                track.append(Message('note_on', note=pitch, velocity=vel, channel=channel, time=delta))
            else:
                track.append(Message('note_off', note=pitch, velocity=0, channel=channel, time=delta))
            last_time = event_time

        track.append(MetaMessage('end_of_track', time=0))

    mid.save(output_path)


def count_midi_notes(mid: MidiFile) -> int:
    """Count total notes in MIDI file."""
    count = 0
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                count += 1
    return count


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fix_decode_duplicates.py <checkpoint.npz> [--diagnose] [--extract] [--compare]")
        print("  --diagnose: Show duplicate occurrence statistics")
        print("  --extract:  Extract deduplicated corpus excerpt")
        print("  --compare:  Compare with/without deduplication")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    do_diagnose = '--diagnose' in sys.argv or len(sys.argv) == 2
    do_extract = '--extract' in sys.argv
    do_compare = '--compare' in sys.argv

    # Load patterns
    print(f"Loading checkpoint: {checkpoint_path}")
    data = np.load(checkpoint_path, allow_pickle=True)
    patterns_file = str(data['patterns_json_file'][0])

    import os
    base_dir = os.path.dirname(checkpoint_path)
    json_path = os.path.join(base_dir, patterns_file) if base_dir else patterns_file

    print(f"Loading patterns from: {json_path}")
    with open(json_path, 'rb') as f:
        patterns = orjson.loads(f.read())

    print(f"Loaded {len(patterns)} patterns")

    instruments = [32, 65, 66, 67]  # Bass + Sax section

    if do_diagnose:
        diagnose_duplicates(patterns)

    if do_extract:
        extract_corpus_excerpt_deduped(patterns, instruments)

    if do_compare:
        compare_with_without_dedupe(patterns, instruments)


if __name__ == '__main__':
    main()
