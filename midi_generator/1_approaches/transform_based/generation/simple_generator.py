"""Simple generator: Generation = Reconstruction with sampled occurrences."""

import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False


@dataclass
class Note:
    """A single MIDI note."""
    pitch: int
    onset: int  # in ticks
    duration: int
    velocity: int
    instrument: int  # GM program


def load_patterns(patterns_json_path: str) -> Tuple[Dict, Dict]:
    """Load patterns with their occurrence distributions.

    Returns:
        (patterns_dict, instrument_patterns) where instrument_patterns maps
        gm_program -> list of (pattern, weight) tuples for that instrument.
    """
    with open(patterns_json_path) as f:
        raw = json.load(f)

    patterns = {}
    # Build per-instrument pattern indexes
    instrument_patterns = defaultdict(list)  # gm -> [(pattern, weight), ...]

    for pid_str, data in raw.items():
        pid = int(pid_str)

        # Build distributions from occurrences
        first_pitch_counts = defaultdict(int)
        instrument_counts = defaultdict(int)
        instrument_pitch_counts = defaultdict(lambda: defaultdict(int))

        for occ in data.get('occurrences', []):
            first_pitch = occ.get('first_pitch', 60)
            gm_program = occ.get('gm_program', 0)
            first_pitch_counts[first_pitch] += 1
            instrument_counts[gm_program] += 1
            instrument_pitch_counts[gm_program][first_pitch] += 1

        pattern = {
            'id': pid,
            'intervals': data.get('pitch_intervals', []),
            'rhythm_bucket': data.get('rhythm_bucket', 8),
            'velocity_bucket': data.get('velocity_bucket', 3),
            'count': data.get('count', 1),
            'first_pitch_dist': dict(first_pitch_counts),
            'instrument_dist': dict(instrument_counts),
            'instrument_pitch_dist': {k: dict(v) for k, v in instrument_pitch_counts.items()},
        }
        patterns[pid] = pattern

        # Index patterns by instrument with their count for that instrument
        for gm, count in instrument_counts.items():
            instrument_patterns[gm].append((pattern, count))

    return patterns, dict(instrument_patterns)


def sample_from_dist(dist: Dict[int, int]) -> int:
    """Sample from a frequency distribution."""
    if not dist:
        return 0
    items = list(dist.keys())
    weights = list(dist.values())
    return random.choices(items, weights=weights, k=1)[0]


def sample_pattern(patterns: Dict, weighted: bool = True) -> Dict:
    """Sample a pattern weighted by corpus frequency."""
    pattern_list = list(patterns.values())
    if weighted:
        weights = [p['count'] for p in pattern_list]
        return random.choices(pattern_list, weights=weights, k=1)[0]
    return random.choice(pattern_list)


def sample_pattern_for_instrument(instrument_patterns: Dict, gm_program: int) -> Optional[Dict]:
    """Sample a pattern specifically for this instrument, weighted by how often it appears."""
    patterns_weights = instrument_patterns.get(gm_program, [])
    if not patterns_weights:
        return None
    patterns = [p for p, w in patterns_weights]
    weights = [w for p, w in patterns_weights]
    return random.choices(patterns, weights=weights, k=1)[0]


def bucket_to_ioi(rhythm_bucket: int, ticks_per_beat: int = 480) -> int:
    """Convert rhythm bucket to inter-onset interval."""
    # Bucket 8 = quarter note, others scale accordingly
    base_ioi = ticks_per_beat  # quarter note
    bucket_map = {
        4: base_ioi // 4,   # 16th
        5: base_ioi // 3,   # triplet 8th
        6: base_ioi // 2,   # 8th
        7: (base_ioi * 2) // 3,  # dotted 8th
        8: base_ioi,        # quarter
        9: (base_ioi * 3) // 2,  # dotted quarter
        10: base_ioi * 2,   # half
    }
    return bucket_map.get(rhythm_bucket, base_ioi)


def bucket_to_velocity(velocity_bucket: int) -> int:
    """Convert velocity bucket to MIDI velocity."""
    velocity_map = {0: 40, 1: 55, 2: 70, 3: 85, 4: 100, 5: 115}
    return velocity_map.get(velocity_bucket, 80)


def expand_pattern(pattern: Dict, first_pitch: int, start_time: int,
                   ticks_per_beat: int = 480) -> Tuple[List[Note], int]:
    """Expand pattern to notes - SAME AS RECONSTRUCTION.

    Args:
        pattern: The pattern dict with intervals, rhythm_bucket, etc.
        first_pitch: The actual MIDI pitch for the first note (with octave info!)
        start_time: Start time in ticks
        ticks_per_beat: Ticks per beat (default 480)

    Returns:
        (notes, end_time)
    """
    intervals = pattern['intervals']
    ioi = bucket_to_ioi(pattern['rhythm_bucket'], ticks_per_beat)
    velocity = bucket_to_velocity(pattern['velocity_bucket'])
    duration = int(ioi * 0.9)  # 90% of IOI

    notes = []
    current_time = start_time
    current_pitch = first_pitch  # Use first_pitch directly - has octave info!

    # First note
    notes.append(Note(
        pitch=max(0, min(127, current_pitch)),
        onset=current_time,
        duration=duration,
        velocity=velocity,
        instrument=0  # filled in later
    ))
    current_time += ioi

    # Subsequent notes from intervals
    for interval in intervals:
        current_pitch += interval
        notes.append(Note(
            pitch=max(0, min(127, current_pitch)),
            onset=current_time,
            duration=duration,
            velocity=velocity,
            instrument=0
        ))
        current_time += ioi

    return notes, current_time


def generate_track(instrument_patterns: Dict, gm_program: int,
                   n_patterns: int, ticks_per_beat: int = 480) -> List[Note]:
    """Generate a single instrument track by sampling patterns for that instrument.

    Each instrument gets its own sequence of patterns, all starting from time 0.
    """
    notes = []
    current_time = 0

    for _ in range(n_patterns):
        pattern = sample_pattern_for_instrument(instrument_patterns, gm_program)
        if pattern is None:
            continue

        # Sample first_pitch from this instrument's distribution for this pattern
        inst_pitch_dist = pattern.get('instrument_pitch_dist', {}).get(gm_program, {})
        if inst_pitch_dist:
            first_pitch = sample_from_dist(inst_pitch_dist)
        else:
            first_pitch = sample_from_dist(pattern.get('first_pitch_dist', {60: 1}))

        # Expand pattern
        pattern_notes, end_time = expand_pattern(pattern, first_pitch, current_time,
                                                  ticks_per_beat)

        # Assign instrument
        for note in pattern_notes:
            note.instrument = gm_program

        notes.extend(pattern_notes)
        current_time = end_time

    return notes


def generate(patterns: Dict, instrument_patterns: Dict,
             n_patterns_per_instrument: int = 32,
             instruments: List[int] = None,
             ticks_per_beat: int = 480) -> List[Note]:
    """Generate polyphonic music with multiple instruments playing in parallel.

    Each instrument gets its own track, all running simultaneously from time 0.
    This matches how real arrangements work - bass, piano, horns all play together.

    Args:
        patterns: All patterns dict
        instrument_patterns: Dict mapping gm_program -> [(pattern, weight), ...]
        n_patterns_per_instrument: How many patterns each instrument plays
        instruments: Which instruments to include (default: top 6 by pattern count)
        ticks_per_beat: MIDI resolution

    Returns:
        List of all notes from all instruments
    """
    # Default: use instruments with most patterns (typical big band)
    if instruments is None:
        # Sort instruments by total pattern count
        inst_counts = [(gm, sum(w for p, w in pats))
                       for gm, pats in instrument_patterns.items()]
        inst_counts.sort(key=lambda x: -x[1])
        # Take top 6 instruments
        instruments = [gm for gm, count in inst_counts[:6]]

    all_notes = []

    for gm_program in instruments:
        track_notes = generate_track(instrument_patterns, gm_program,
                                     n_patterns_per_instrument, ticks_per_beat)
        all_notes.extend(track_notes)

    return all_notes


def notes_to_midi(notes: List[Note], output_path: str,
                  tempo: int = 120, ticks_per_beat: int = 480):
    """Convert notes to MIDI file."""
    if not MIDO_AVAILABLE:
        raise ImportError("mido library required")

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    # Tempo track
    tempo_track = mido.MidiTrack()
    mid.tracks.append(tempo_track)
    tempo_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))

    # Group notes by instrument
    by_instrument = defaultdict(list)
    for note in notes:
        by_instrument[note.instrument].append(note)

    # GM program names
    gm_names = {
        0: 'Piano', 32: 'Acoustic Bass', 33: 'Electric Bass',
        56: 'Trumpet', 57: 'Trombone', 65: 'Alto Sax',
        66: 'Tenor Sax', 67: 'Baritone Sax'
    }

    # Create track for each instrument
    for gm_program, inst_notes in sorted(by_instrument.items()):
        track = mido.MidiTrack()
        mid.tracks.append(track)

        name = gm_names.get(gm_program, f'GM {gm_program}')
        track.append(mido.MetaMessage('track_name', name=name, time=0))
        track.append(mido.Message('program_change', program=gm_program, time=0))

        # Build events
        events = []
        for note in inst_notes:
            events.append(('on', note.onset, note.pitch, note.velocity))
            events.append(('off', note.onset + note.duration, note.pitch, 0))

        events.sort(key=lambda x: (x[1], x[0] != 'on'))

        # Convert to delta times
        last_time = 0
        for event_type, time, pitch, vel in events:
            delta = time - last_time
            if event_type == 'on':
                track.append(mido.Message('note_on', note=pitch, velocity=vel, time=delta))
            else:
                track.append(mido.Message('note_off', note=pitch, velocity=0, time=delta))
            last_time = time

    mid.save(output_path)
    return output_path


def main():
    """Simple generation from command line."""
    import argparse
    import numpy as np
    import os

    parser = argparse.ArgumentParser(description='Simple pattern-based generation')
    parser.add_argument('--checkpoint', '-c', required=True, help='Checkpoint .npz file')
    parser.add_argument('--output', '-o', required=True, help='Output MIDI file')
    parser.add_argument('--patterns', '-n', type=int, default=32,
                        help='Number of patterns per instrument')
    parser.add_argument('--tempo', type=int, default=120, help='Tempo in BPM')
    parser.add_argument('--instruments', type=int, default=6,
                        help='Number of instruments to include')
    args = parser.parse_args()

    # Load checkpoint
    cp = np.load(args.checkpoint, allow_pickle=True)
    checkpoint_dir = os.path.dirname(args.checkpoint)

    patterns_file = str(cp['patterns_json_file'].item())
    if not os.path.isabs(patterns_file):
        patterns_file = os.path.join(checkpoint_dir, patterns_file)

    print(f'Loading patterns from {patterns_file}')
    patterns, instrument_patterns = load_patterns(patterns_file)
    print(f'Loaded {len(patterns)} patterns')

    # Show available instruments
    inst_counts = [(gm, sum(w for p, w in pats))
                   for gm, pats in instrument_patterns.items()]
    inst_counts.sort(key=lambda x: -x[1])
    print(f'Top instruments by pattern count:')
    gm_names = {
        0: 'Piano', 32: 'Acoustic Bass', 33: 'Electric Bass',
        56: 'Trumpet', 57: 'Trombone', 65: 'Alto Sax',
        66: 'Tenor Sax', 67: 'Baritone Sax'
    }
    for gm, count in inst_counts[:10]:
        name = gm_names.get(gm, f'GM {gm}')
        print(f'  {name:20} (GM {gm:2}): {count:5} patterns')

    # Select top N instruments
    selected = [gm for gm, count in inst_counts[:args.instruments]]
    print(f'\nGenerating {args.patterns} patterns per instrument for {len(selected)} instruments...')

    notes = generate(patterns, instrument_patterns,
                     n_patterns_per_instrument=args.patterns,
                     instruments=selected)
    print(f'Generated {len(notes)} notes total')

    print(f'Saving to {args.output}')
    notes_to_midi(notes, args.output, tempo=args.tempo)
    print('Done!')


if __name__ == '__main__':
    main()
