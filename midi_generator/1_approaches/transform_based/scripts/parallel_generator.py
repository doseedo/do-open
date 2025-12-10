#!/usr/bin/env python3
"""
Parallel Multitrack Generator (v53 Checkpoint)
==============================================

CORRECT APPROACH: Each instrument has its own pattern vocabulary.
Tracks run in PARALLEL, not sequentially.

Key insight: The occurrence data tells us which instrument plays which pattern.
Build per-instrument vocabularies and sample from them independently.

Usage:
    python scripts/parallel_generator.py checkpoint.npz -o output.mid --bars 32
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# GM instrument ranges for proper octave placement
GM_RANGES = {
    # Bass instruments (low register)
    32: (28, 55),   # Acoustic Bass
    33: (28, 55),   # Electric Bass (finger)
    34: (28, 55),   # Electric Bass (pick)
    35: (28, 55),   # Fretless Bass
    36: (28, 55),   # Slap Bass 1
    37: (28, 55),   # Slap Bass 2
    38: (28, 55),   # Synth Bass 1
    39: (28, 55),   # Synth Bass 2
    43: (36, 60),   # Contrabass

    # Piano/keys (wide range)
    0: (36, 96),    # Acoustic Grand Piano
    1: (36, 96),    # Bright Acoustic Piano
    2: (36, 96),    # Electric Grand Piano
    3: (36, 96),    # Honky-tonk Piano
    4: (36, 96),    # Electric Piano 1
    5: (36, 96),    # Electric Piano 2

    # Guitar (mid range)
    24: (40, 84),   # Acoustic Guitar (nylon)
    25: (40, 84),   # Acoustic Guitar (steel)
    26: (40, 84),   # Electric Guitar (jazz)
    27: (40, 84),   # Electric Guitar (clean)

    # Brass (trumpet range)
    56: (52, 84),   # Trumpet
    57: (40, 72),   # Trombone
    58: (36, 60),   # Tuba
    59: (36, 72),   # Muted Trumpet
    60: (40, 72),   # French Horn
    61: (40, 72),   # Brass Section

    # Saxophones
    64: (44, 80),   # Soprano Sax
    65: (49, 81),   # Alto Sax
    66: (42, 75),   # Tenor Sax
    67: (36, 69),   # Baritone Sax

    # Woodwinds
    71: (60, 96),   # Clarinet
    72: (57, 96),   # Piccolo
    73: (60, 96),   # Flute
    74: (58, 91),   # Recorder
    75: (58, 80),   # Pan Flute
    68: (58, 91),   # Oboe
    69: (34, 75),   # English Horn
    70: (34, 67),   # Bassoon
}

# Default range for unknown instruments
DEFAULT_RANGE = (48, 84)


@dataclass
class InstrumentVocabulary:
    """Pattern vocabulary for a specific instrument."""
    gm_program: int
    patterns: List[Dict] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    pitch_distribution: Dict[int, int] = field(default_factory=dict)  # pitch -> count
    tau_distribution: Dict[int, int] = field(default_factory=dict)    # IOI -> count
    avg_pitch: float = 60.0
    avg_tau: float = 480.0  # Average IOI (quarter note default)
    pitch_range: Tuple[int, int] = (48, 84)

    def sample_pattern(self) -> Optional[Dict]:
        """Sample a pattern weighted by corpus frequency."""
        if not self.patterns:
            return None
        return random.choices(self.patterns, weights=self.weights)[0]

    def sample_pitch(self) -> int:
        """Sample a starting pitch from the instrument's distribution."""
        if not self.pitch_distribution:
            return int(self.avg_pitch)
        pitches = list(self.pitch_distribution.keys())
        weights = list(self.pitch_distribution.values())
        return random.choices(pitches, weights=weights)[0]

    def sample_tau(self) -> int:
        """Sample an IOI (timing) from the instrument's distribution."""
        if not self.tau_distribution:
            return int(self.avg_tau)
        taus = list(self.tau_distribution.keys())
        weights = list(self.tau_distribution.values())
        return random.choices(taus, weights=weights)[0]


class ParallelGenerator:
    """Generate multitrack music with per-instrument pattern vocabularies."""

    def __init__(self, checkpoint_path: str, verbose: bool = True):
        self.verbose = verbose
        self.load_checkpoint(checkpoint_path)
        self.build_instrument_vocabularies()

    def load_checkpoint(self, checkpoint_path: str):
        """Load v53 checkpoint with patterns."""
        if self.verbose:
            print(f"Loading checkpoint: {checkpoint_path}")

        ckpt = np.load(checkpoint_path, allow_pickle=True)
        base_path = checkpoint_path.replace('.npz', '')

        # Load patterns from external JSON
        patterns_file = ckpt.get('patterns_json_file', [None])[0]
        if patterns_file:
            patterns_path = os.path.join(os.path.dirname(checkpoint_path), patterns_file)
            with open(patterns_path) as f:
                self.patterns = json.load(f)
        else:
            raise ValueError("No patterns_json_file in checkpoint")

        # Load orchestration rules if available
        self.orchestration_rules = {}
        orch_file = ckpt.get('meta_patterns_json_file', [None])[0]
        if orch_file:
            orch_path = os.path.join(os.path.dirname(checkpoint_path), orch_file)
            if os.path.exists(orch_path):
                with open(orch_path) as f:
                    meta_data = json.load(f)
                    self.orchestration_rules = meta_data.get('orchestration_rules', {})

        if self.verbose:
            print(f"  Loaded {len(self.patterns)} patterns")

    def build_instrument_vocabularies(self):
        """Build per-instrument pattern vocabularies from occurrence data."""
        if self.verbose:
            print("Building per-instrument vocabularies...")

        # Group patterns by which instruments play them
        instrument_patterns = defaultdict(lambda: {
            'patterns': [],
            'counts': [],
            'pitches': defaultdict(int),
            'taus': defaultdict(int),  # IOI distribution from tau_offset
        })

        for pattern_id, pattern in self.patterns.items():
            occurrences = pattern.get('occurrences', [])
            if not occurrences:
                continue

            # Group by instrument
            by_instrument = defaultdict(list)
            for occ in occurrences:
                gm = occ.get('gm_program', 0)
                by_instrument[gm].append(occ)

            # Add pattern to each instrument's vocabulary
            for gm, occs in by_instrument.items():
                count = len(occs)
                instrument_patterns[gm]['patterns'].append({
                    'id': pattern_id,
                    'pitch_intervals': pattern.get('pitch_intervals', []),
                    'pitch_classes': pattern.get('pitch_classes', []),
                    'canonical_pitches': pattern.get('canonical_pitches', []),
                    'rhythm_bucket': pattern.get('rhythm_bucket', 8),
                    'velocity_bucket': pattern.get('velocity_bucket', 4),
                    'is_hierarchical': pattern.get('is_hierarchical', False),
                    'count': count,
                })
                instrument_patterns[gm]['counts'].append(count)

                # Collect pitch and tau distributions
                for occ in occs:
                    first_pitch = occ.get('first_pitch', 60)
                    instrument_patterns[gm]['pitches'][first_pitch] += 1
                    # Collect tau_offset for timing distribution
                    tau = occ.get('tau_offset', 480)
                    if tau > 0:
                        # Quantize tau to reasonable buckets (30, 60, 120, 240, 480, 960, etc.)
                        quantized_tau = max(30, min(1920, tau))
                        instrument_patterns[gm]['taus'][quantized_tau] += 1

        # Create InstrumentVocabulary objects
        self.vocabularies = {}
        for gm, data in instrument_patterns.items():
            if not data['patterns']:
                continue

            # Normalize weights
            total = sum(data['counts'])
            weights = [c / total for c in data['counts']]

            # Compute average pitch
            total_pitch = sum(p * c for p, c in data['pitches'].items())
            total_count = sum(data['pitches'].values())
            avg_pitch = total_pitch / total_count if total_count > 0 else 60

            # Compute average tau (IOI)
            total_tau = sum(t * c for t, c in data['taus'].items())
            tau_count = sum(data['taus'].values())
            avg_tau = total_tau / tau_count if tau_count > 0 else 480

            # Get instrument range
            pitch_range = GM_RANGES.get(gm, DEFAULT_RANGE)

            vocab = InstrumentVocabulary(
                gm_program=gm,
                patterns=data['patterns'],
                weights=weights,
                pitch_distribution=dict(data['pitches']),
                tau_distribution=dict(data['taus']),
                avg_pitch=avg_pitch,
                avg_tau=avg_tau,
                pitch_range=pitch_range,
            )
            self.vocabularies[gm] = vocab

        if self.verbose:
            print(f"  Built vocabularies for {len(self.vocabularies)} instruments:")
            for gm, vocab in sorted(self.vocabularies.items(),
                                    key=lambda x: len(x[1].patterns), reverse=True)[:10]:
                print(f"    GM {gm}: {len(vocab.patterns)} patterns, avg_pitch={vocab.avg_pitch:.0f}")

    def get_available_instruments(self) -> List[int]:
        """Get list of instruments with patterns."""
        return sorted(self.vocabularies.keys(),
                      key=lambda x: len(self.vocabularies[x].patterns), reverse=True)

    def rhythm_bucket_to_ioi(self, bucket: int, ticks_per_beat: int = 480) -> int:
        """Convert rhythm bucket to inter-onset interval."""
        # Bucket values roughly correspond to 16th note divisions
        # 0-3: 32nd notes, 4-7: 16th notes, 8-11: 8th notes, 12-15: quarter+
        if bucket < 4:
            return ticks_per_beat // 8  # 32nd note
        elif bucket < 8:
            return ticks_per_beat // 4  # 16th note
        elif bucket < 12:
            return ticks_per_beat // 2  # 8th note
        else:
            return ticks_per_beat       # quarter note

    def expand_pattern(
        self,
        pattern: Dict,
        first_pitch: int,
        start_time: int,
        gm_program: int,
        base_ioi: int = 480,
    ) -> List[Dict]:
        """Expand a pattern to notes.

        Args:
            pattern: Pattern dict with pitch_intervals, rhythm_bucket, etc.
            first_pitch: Starting MIDI pitch (0-127)
            start_time: Start time in ticks
            gm_program: GM program number for range clamping
            base_ioi: Base inter-onset interval from corpus tau_offset
        """
        notes = []

        # Get intervals
        intervals = pattern.get('pitch_intervals', [])
        if not intervals:
            # Use pitch_classes as fallback
            pcs = pattern.get('pitch_classes', [])
            if len(pcs) < 2:
                return notes
            # Compute intervals from pitch classes
            intervals = []
            for i in range(1, len(pcs)):
                diff = (pcs[i] - pcs[i-1]) % 12
                if diff > 6:
                    diff -= 12
                intervals.append(diff)

        # Get velocity
        velocity_bucket = pattern.get('velocity_bucket', 4)
        velocity = 60 + velocity_bucket * 10  # Map 0-7 to 60-130
        velocity = min(127, max(40, velocity))

        # Get pitch range for instrument
        pitch_range = GM_RANGES.get(gm_program, DEFAULT_RANGE)

        # Build notes
        current_pitch = first_pitch
        current_time = start_time

        # First note
        notes.append({
            'pitch': max(pitch_range[0], min(pitch_range[1], current_pitch)),
            'velocity': velocity,
            'time': current_time,
            'duration': base_ioi,
        })

        # Remaining notes from intervals
        for interval in intervals:
            current_time += base_ioi
            current_pitch += interval

            # Clamp to instrument range with octave folding
            while current_pitch < pitch_range[0]:
                current_pitch += 12
            while current_pitch > pitch_range[1]:
                current_pitch -= 12

            notes.append({
                'pitch': current_pitch,
                'velocity': velocity,
                'time': current_time,
                'duration': base_ioi,
            })

        return notes

    def generate_track(
        self,
        gm_program: int,
        duration_ticks: int,
        ticks_per_beat: int = 480,
    ) -> List[Dict]:
        """Generate a single track using instrument's vocabulary."""
        if gm_program not in self.vocabularies:
            return []

        vocab = self.vocabularies[gm_program]
        notes = []
        current_time = 0

        while current_time < duration_ticks:
            # Sample pattern
            pattern = vocab.sample_pattern()
            if pattern is None:
                break

            # Sample starting pitch from instrument's distribution
            first_pitch = vocab.sample_pitch()

            # Sample tau (IOI) from instrument's distribution for accurate timing
            base_tau = vocab.sample_tau()

            # Expand pattern to notes
            pattern_notes = self.expand_pattern(
                pattern=pattern,
                first_pitch=first_pitch,
                start_time=current_time,
                gm_program=gm_program,
                base_ioi=base_tau,  # Use corpus-sampled timing
            )

            if not pattern_notes:
                current_time += ticks_per_beat  # Skip ahead
                continue

            notes.extend(pattern_notes)

            # Advance time to end of pattern
            if pattern_notes:
                last_note = pattern_notes[-1]
                current_time = last_note['time'] + last_note['duration']

        return notes

    def generate_multitrack(
        self,
        instruments: List[int] = None,
        bars: int = 32,
        tempo: int = 120,
        ticks_per_beat: int = 480,
        seed: int = None,
    ) -> Dict[int, List[Dict]]:
        """Generate parallel multitrack arrangement.

        Each instrument generates from its own vocabulary INDEPENDENTLY.
        All tracks run in PARALLEL over the same time span.
        """
        if seed is not None:
            random.seed(seed)

        if instruments is None:
            # Use top instruments by pattern count
            instruments = self.get_available_instruments()[:6]

        # Calculate duration
        beats_per_bar = 4
        duration_ticks = bars * beats_per_bar * ticks_per_beat

        if self.verbose:
            print(f"\nGenerating {bars} bars ({duration_ticks} ticks) for {len(instruments)} instruments...")

        tracks = {}

        for gm in instruments:
            if gm not in self.vocabularies:
                if self.verbose:
                    print(f"  Warning: No vocabulary for GM {gm}, skipping")
                continue

            vocab = self.vocabularies[gm]
            track_notes = self.generate_track(
                gm_program=gm,
                duration_ticks=duration_ticks,
                ticks_per_beat=ticks_per_beat,
            )

            tracks[gm] = track_notes

            if self.verbose:
                avg_pitch = np.mean([n['pitch'] for n in track_notes]) if track_notes else 0
                print(f"  GM {gm}: {len(track_notes)} notes, avg_pitch={avg_pitch:.0f}")

        return tracks

    def to_midi(
        self,
        tracks: Dict[int, List[Dict]],
        output_path: str,
        tempo: int = 120,
        ticks_per_beat: int = 480,
    ):
        """Save multitrack to MIDI file."""
        try:
            import mido
        except ImportError:
            print("Error: mido not installed. Run: pip install mido")
            return

        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=1)

        # Tempo track
        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))
        tempo_track.append(mido.MetaMessage('end_of_track', time=0))

        # Add each instrument track
        for gm, notes in tracks.items():
            if not notes:
                continue

            track = mido.MidiTrack()
            mid.tracks.append(track)

            # Set instrument (channel 9 for drums)
            is_drum = (gm >= 128)  # Use 128+ as drum indicator
            channel = 9 if is_drum else (len(mid.tracks) - 2) % 9

            track.append(mido.Message('program_change', program=gm % 128, channel=channel, time=0))

            # Build events
            events = []
            for note in notes:
                events.append((note['time'], 'note_on', note['pitch'], note['velocity'], channel))
                events.append((note['time'] + note['duration'], 'note_off', note['pitch'], 0, channel))

            events.sort(key=lambda e: (e[0], 0 if e[1] == 'note_off' else 1))

            # Convert to delta times
            prev_time = 0
            for event in events:
                abs_time, msg_type, pitch, velocity, ch = event
                delta = abs_time - prev_time

                if msg_type == 'note_on':
                    track.append(mido.Message('note_on', note=pitch, velocity=velocity,
                                             channel=ch, time=delta))
                else:
                    track.append(mido.Message('note_off', note=pitch, velocity=0,
                                             channel=ch, time=delta))
                prev_time = abs_time

            track.append(mido.MetaMessage('end_of_track', time=0))

        mid.save(output_path)

        if self.verbose:
            total_notes = sum(len(notes) for notes in tracks.values())
            print(f"\nSaved MIDI to: {output_path}")
            print(f"  {len(tracks)} tracks, {total_notes} notes")


def main():
    parser = argparse.ArgumentParser(description='Parallel multitrack generator')
    parser.add_argument('checkpoint', help='Path to v53 checkpoint .npz file')
    parser.add_argument('--output', '-o', default='parallel_generated.mid', help='Output MIDI path')
    parser.add_argument('--bars', '-b', type=int, default=32, help='Number of bars')
    parser.add_argument('--tempo', '-t', type=int, default=120, help='Tempo BPM')
    parser.add_argument('--seed', '-s', type=int, help='Random seed')
    parser.add_argument('--instruments', '-i', help='Comma-separated GM program numbers')
    parser.add_argument('--list-instruments', '-l', action='store_true',
                       help='List available instruments and exit')
    args = parser.parse_args()

    print("=" * 60)
    print("PARALLEL MULTITRACK GENERATOR (v53)")
    print("=" * 60)

    # Load generator
    gen = ParallelGenerator(args.checkpoint)

    if args.list_instruments:
        print("\nAvailable instruments (by pattern count):")
        for gm in gen.get_available_instruments()[:20]:
            vocab = gen.vocabularies[gm]
            print(f"  GM {gm}: {len(vocab.patterns)} patterns, avg_pitch={vocab.avg_pitch:.0f}")
        return

    # Parse instruments
    instruments = None
    if args.instruments:
        instruments = [int(x.strip()) for x in args.instruments.split(',')]

    # Generate
    tracks = gen.generate_multitrack(
        instruments=instruments,
        bars=args.bars,
        tempo=args.tempo,
        seed=args.seed,
    )

    # Save
    gen.to_midi(tracks, args.output, tempo=args.tempo)

    print("\nDone!")


if __name__ == '__main__':
    main()
