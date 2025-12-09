#!/usr/bin/env python3
"""
Proper Reconstruction - Use covering algorithm to avoid overlapping patterns.

The Re-Pair codec stores overlapping patterns:
- A 14-note pattern covering beats 255-262
- A 3-note pattern starting at beat 257 (WITHIN the 14-note coverage)
- etc.

For proper reconstruction, we need to pick NON-OVERLAPPING patterns
that together cover all original notes exactly once.

Algorithm:
1. Sort all occurrences by onset time
2. Greedy covering: pick longest pattern that starts at/after current position
3. Mark its end time as "covered up to"
4. Skip any patterns that start before "covered up to"
5. Continue until end of piece
"""

import orjson
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


def compute_pattern_end_time(onset: int, tau: int, rhythm_ratios: List[float], n_notes: int) -> int:
    """Compute the end time of a pattern (onset of last note + some duration)."""
    time = onset
    for i in range(n_notes - 1):
        if i < len(rhythm_ratios):
            time += int(tau * rhythm_ratios[i])
        else:
            time += tau
    # Add duration of last note
    time += tau  # Approximate
    return time


def extract_with_covering(
    patterns: dict,
    piece_id: str,
    gm: int,
    start_beat: int,
    end_beat: int
) -> List[dict]:
    """Extract notes using covering algorithm.

    Returns list of notes with {pitch, onset, duration, velocity}
    """
    start_tick = start_beat * 480
    end_tick = end_beat * 480

    # Collect all occurrences for this piece/gm in range
    occurrences = []

    for pid, p in patterns.items():
        if p.get('gm_program') != gm:
            continue

        n_notes = len(p.get('pitch_intervals', [])) + 1

        for occ in p.get('occurrences', []):
            if occ.get('piece_id', '') != piece_id:
                continue

            onset = occ.get('onset_time', 0)
            if not (start_tick <= onset <= end_tick):
                continue

            tau = occ.get('tau_offset', 480)
            rhythm_ratios = p.get('rhythm_ratios', [1.0] * n_notes)

            # Compute end time
            end_time = compute_pattern_end_time(onset, tau, rhythm_ratios, n_notes)

            occurrences.append({
                'pattern_id': pid,
                'pattern': p,
                'onset': onset,
                'end_time': end_time,
                'n_notes': n_notes,
                'first_pitch': occ.get('first_pitch', 60),
                'tau': tau,
            })

    # Sort by onset, then by n_notes descending (prefer longer patterns)
    occurrences.sort(key=lambda x: (x['onset'], -x['n_notes']))

    # Greedy covering
    covered_up_to = start_tick
    selected = []

    for occ in occurrences:
        # Skip if this pattern starts before our coverage point
        if occ['onset'] < covered_up_to:
            continue

        # Select this pattern
        selected.append(occ)

        # Update coverage (but don't go backwards if there's a gap)
        covered_up_to = max(covered_up_to, occ['end_time'])

    # Expand selected patterns to notes
    notes = []
    for occ in selected:
        p = occ['pattern']
        pitch = occ['first_pitch']
        time = occ['onset'] - start_tick  # Normalize to 0

        intervals = [0] + list(p.get('pitch_intervals', []))
        rhythm_ratios = p.get('rhythm_ratios', [])
        duration_ratios = p.get('duration_ratios', [0.9] * occ['n_notes'])
        tau = occ['tau']

        # Build IOIs from successive ratios: IOI[i+1] = IOI[i] * ratio[i]
        n_notes = occ['n_notes']
        iois = [tau]  # First note uses tau
        for r in rhythm_ratios[:n_notes - 1]:
            iois.append(int(iois[-1] * r))

        for i in range(n_notes):
            if i > 0:
                pitch += intervals[i]

            ioi = iois[i] if i < len(iois) else tau
            dur = int(ioi * duration_ratios[i] * 0.9) if i < len(duration_ratios) else int(ioi * 0.9)

            notes.append({
                'pitch': max(0, min(127, pitch)),
                'onset': max(0, time),
                'duration': max(1, dur),
                'velocity': 80
            })

            time += ioi

    return notes


def reconstruct_piece_segment(
    patterns: dict,
    piece_id: str,
    instruments: List[int],
    start_beat: int,
    end_beat: int,
    output_path: str
):
    """Reconstruct a piece segment using covering algorithm."""
    print(f"Reconstructing {piece_id}")
    print(f"  Beats: {start_beat}-{end_beat}")
    print(f"  Instruments: {instruments}")

    mid = MidiFile(ticks_per_beat=480, type=1)

    # Tempo track
    tempo_track = MidiTrack()
    mid.tracks.append(tempo_track)
    tempo_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
    tempo_track.append(MetaMessage('end_of_track', time=0))

    gm_names = {
        32: "Acoustic Bass",
        65: "Alto Sax",
        66: "Tenor Sax",
        67: "Baritone Sax"
    }

    total_notes = 0
    channel = 0

    for gm in instruments:
        notes = extract_with_covering(patterns, piece_id, gm, start_beat, end_beat)

        if not notes:
            print(f"  GM {gm}: 0 notes")
            continue

        print(f"  GM {gm}: {len(notes)} notes")
        total_notes += len(notes)

        track = MidiTrack()
        mid.tracks.append(track)

        name = gm_names.get(gm, f"GM {gm}")
        track.append(MetaMessage('track_name', name=name, time=0))
        track.append(Message('program_change', program=gm, channel=channel, time=0))

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
        channel += 1
        if channel == 9:
            channel = 10

    mid.save(output_path)
    print(f"\nSaved to: {output_path}")
    print(f"Total notes: {total_notes}")

    return total_notes


def main():
    import sys

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoint_v55_pure_contour_1000files.npz'

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

    # Reconstruct the test segment
    piece_id = "The_Electro_Suite__Hans_Zimmer"
    instruments = [65, 66, 67]
    start_beat = 255
    end_beat = 319
    output_path = '/tmp/proper_reconstruction.mid'

    reconstruct_piece_segment(
        patterns, piece_id, instruments, start_beat, end_beat, output_path
    )

    # Compare with ground truth
    print("\n=== COMPARISON ===")

    def count_notes(path):
        mid = MidiFile(path)
        count = 0
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    count += 1
        return count

    ground_truth = count_notes('/tmp/ground_truth_extract.mid')
    reconstruction = count_notes(output_path)
    naive_recon = count_notes('/tmp/corpus_excerpt_deduped.mid')

    print(f"Ground truth:           {ground_truth} notes")
    print(f"Proper reconstruction:  {reconstruction} notes")
    print(f"Naive reconstruction:   {naive_recon} notes")
    print(f"\nProper vs ground truth: {reconstruction - ground_truth:+d} ({100*(reconstruction-ground_truth)/ground_truth:+.1f}%)")
    print(f"Naive vs ground truth:  {naive_recon - ground_truth:+d} ({100*(naive_recon-ground_truth)/ground_truth:+.1f}%)")


if __name__ == '__main__':
    main()
