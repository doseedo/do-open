#!/usr/bin/env python3
"""
Extract a real corpus excerpt for A/B comparison with generated output.

Finds a segment where the target instruments (e.g., Bass + Sax section)
actually play together and exports it as MIDI.
"""

import orjson
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


def extract_corpus_excerpt(
    patterns: dict,
    instruments: List[int],
    min_length: int = 32,
    output_path: str = '/tmp/corpus_excerpt.mid'
) -> str:
    """Extract a real excerpt where target instruments play together.

    Args:
        patterns: Pattern dict from checkpoint
        instruments: GM programs to include
        min_length: Minimum number of beats with all instruments
        output_path: Where to save the MIDI

    Returns:
        piece_id of the extracted segment
    """
    print(f"Looking for segments with instruments: {instruments}")
    print(f"Minimum length: {min_length} beats")

    # Group occurrences by (piece, beat)
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

            # Get pattern details for reconstruction
            intervals = p.get('pitch_intervals', [0])
            rhythm_ratios = p.get('rhythm_ratios', [1.0])
            duration_ratios = p.get('duration_ratios', [0.9])
            tau = occ.get('tau_offset', 480)

            piece_beat_data[piece][beat].append({
                'gm': gm,
                'pitch': pitch,
                'onset': onset,
                'intervals': intervals,
                'rhythm_ratios': rhythm_ratios,
                'duration_ratios': duration_ratios,
                'tau': tau
            })

    # Find pieces with good coverage of target instruments
    candidates = []

    for piece, beat_data in piece_beat_data.items():
        # Find runs where all instruments are present
        sorted_beats = sorted(beat_data.keys())
        if not sorted_beats:
            continue

        current_run_start = None
        current_run_length = 0
        best_run_start = None
        best_run_length = 0

        for beat in sorted_beats:
            gms_present = set(d['gm'] for d in beat_data[beat])
            has_all = all(gm in gms_present for gm in instruments)

            if has_all:
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
        print(f"No segments found with all instruments for {min_length}+ beats")
        print("Trying with relaxed criteria...")

        # Relax: find best coverage even if not all instruments every beat
        for piece, beat_data in piece_beat_data.items():
            sorted_beats = sorted(beat_data.keys())
            if len(sorted_beats) >= min_length:
                # Count how many beats have at least 2 of our instruments
                good_beats = sum(1 for b in sorted_beats
                                if len(set(d['gm'] for d in beat_data[b]) & set(instruments)) >= 2)
                if good_beats >= min_length:
                    candidates.append((piece, sorted_beats[0], len(sorted_beats)))

    if not candidates:
        print("Still no candidates found!")
        return None

    # Sort by run length
    candidates.sort(key=lambda x: -x[2])

    print(f"\nFound {len(candidates)} candidate segments:")
    for piece, start, length in candidates[:5]:
        print(f"  {piece}: beat {start}-{start+length} ({length} beats)")

    # Pick the best one
    piece, start_beat, run_length = candidates[0]
    end_beat = start_beat + min(run_length, 64)  # Cap at 64 beats

    print(f"\nExtracting: {piece} beats {start_beat}-{end_beat}")

    # Collect notes for this segment
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

            # Expand pattern to notes
            pitch = base_pitch
            time = base_onset - (start_beat * 480)  # Normalize to start at 0

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

    # Build MIDI
    mid = MidiFile(ticks_per_beat=480, type=1)

    # Tempo track
    tempo_track = MidiTrack()
    mid.tracks.append(tempo_track)
    tempo_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
    tempo_track.append(MetaMessage('end_of_track', time=0))

    # Channel mapping
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

        # Track name
        name = gm_names.get(gm, f"GM {gm}")
        track.append(MetaMessage('track_name', name=name, time=0))
        track.append(Message('program_change', program=gm % 128, channel=channel, time=0))

        # Sort and dedupe notes
        notes = notes_by_gm[gm]
        notes.sort(key=lambda n: n['onset'])

        # Build events
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
    print(f"  Tracks: {len(mid.tracks)}")
    print(f"  Notes: {total_notes}")
    print(f"  Instruments: {list(notes_by_gm.keys())}")

    return piece


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_corpus_excerpt.py <checkpoint.npz> [-o output.mid] [--instruments GM1 GM2 ...]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_path = '/tmp/corpus_excerpt.mid'
    instruments = [32, 65, 66, 67]  # Bass + Sax section

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '-o' and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--instruments':
            instruments = []
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith('-'):
                instruments.append(int(sys.argv[i]))
                i += 1
        else:
            i += 1

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

    # Extract
    extract_corpus_excerpt(patterns, instruments, min_length=32, output_path=output_path)


if __name__ == '__main__':
    main()
