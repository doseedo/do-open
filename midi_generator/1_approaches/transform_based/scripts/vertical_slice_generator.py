#!/usr/bin/env python3
"""
Vertical Slice Generator (v53 Checkpoint)
==========================================

ARRANGEMENT-AWARE GENERATION: Sample patterns that co-occurred together.

The key insight: In the corpus, when Trumpet plays pattern P1 at time T,
Piano plays pattern P2 at the same time T. These form a "vertical slice".

By sampling entire vertical slices, we preserve:
- Harmonic relationships (piano chords fit under horn melody)
- Rhythmic alignment (instruments lock together)
- Orchestration conventions (who plays when)

Algorithm:
1. Build vertical slices from corpus (patterns at same time in same piece)
2. Sample entire slices weighted by frequency
3. Chain slices together to form complete arrangement

Usage:
    python scripts/vertical_slice_generator.py checkpoint.npz -o output.mid --bars 32
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

# GM instrument ranges for proper octave placement
GM_RANGES = {
    32: (28, 55), 33: (28, 55), 34: (28, 55), 35: (28, 55),  # Bass
    0: (36, 96), 1: (36, 96), 4: (36, 96), 5: (36, 96),      # Piano
    24: (40, 84), 25: (40, 84), 26: (40, 84),                # Guitar
    56: (52, 84), 57: (40, 72), 58: (36, 60),                # Brass
    64: (44, 80), 65: (49, 81), 66: (42, 75), 67: (36, 69),  # Sax
}
DEFAULT_RANGE = (48, 84)


@dataclass
class VerticalSlice:
    """A vertical slice: patterns playing simultaneously."""
    piece_id: str
    onset_time: int
    duration_ticks: int  # Duration until next slice

    # Pattern occurrences in this slice: gm_program -> occurrence data
    instruments: Dict[int, Dict] = field(default_factory=dict)

    @property
    def n_instruments(self) -> int:
        return len(self.instruments)

    def has_instrument(self, gm: int) -> bool:
        return gm in self.instruments


@dataclass
class SliceSequence:
    """A sequence of vertical slices from one piece."""
    piece_id: str
    slices: List[VerticalSlice] = field(default_factory=list)

    @property
    def instruments(self) -> Set[int]:
        """All instruments in this sequence."""
        all_gms = set()
        for s in self.slices:
            all_gms.update(s.instruments.keys())
        return all_gms


class VerticalSliceGenerator:
    """Generate multitrack music by sampling vertical slices."""

    def __init__(self, checkpoint_path: str, verbose: bool = True):
        self.verbose = verbose
        self.load_checkpoint(checkpoint_path)
        self.build_vertical_slices()

    def load_checkpoint(self, checkpoint_path: str):
        """Load v53 checkpoint with patterns."""
        if self.verbose:
            print(f"Loading checkpoint: {checkpoint_path}")

        ckpt = np.load(checkpoint_path, allow_pickle=True)

        # Load patterns from external JSON
        patterns_file = ckpt.get('patterns_json_file', [None])[0]
        if patterns_file:
            patterns_path = os.path.join(os.path.dirname(checkpoint_path), patterns_file)
            with open(patterns_path) as f:
                self.patterns = json.load(f)
        else:
            raise ValueError("No patterns_json_file in checkpoint")

        if self.verbose:
            print(f"  Loaded {len(self.patterns)} patterns")

    def build_vertical_slices(self, time_tolerance: int = 120):
        """Build vertical slices from occurrence data.

        A vertical slice groups patterns that start within time_tolerance
        of each other in the same piece.

        Args:
            time_tolerance: Max ticks difference to consider "simultaneous"
        """
        if self.verbose:
            print("Building vertical slices...")

        # Collect all occurrences with pattern info
        all_occurrences = []
        for pattern_id, pattern in self.patterns.items():
            for occ in pattern.get('occurrences', []):
                all_occurrences.append({
                    'pattern_id': pattern_id,
                    'piece_id': occ.get('piece_id', 'unknown'),
                    'track_id': occ.get('track_id', 0),
                    'onset_time': occ.get('onset_time', 0),
                    'gm_program': occ.get('gm_program', 0),
                    'first_pitch': occ.get('first_pitch', 60),
                    'is_drum': occ.get('is_drum', False),
                    'pattern': pattern,  # Store full pattern
                })

        # Group by piece
        by_piece = defaultdict(list)
        for occ in all_occurrences:
            by_piece[occ['piece_id']].append(occ)

        # Sort each piece by onset time
        for piece_id in by_piece:
            by_piece[piece_id].sort(key=lambda x: x['onset_time'])

        # Build slices within each piece
        self.slice_sequences = {}
        total_slices = 0

        for piece_id, occs in by_piece.items():
            if len(occs) < 2:
                continue

            slices = []
            current_slice = None
            current_base_time = -time_tolerance * 2

            for occ in occs:
                onset = occ['onset_time']

                # Start new slice if too far from current
                if onset > current_base_time + time_tolerance:
                    if current_slice and current_slice.n_instruments > 0:
                        slices.append(current_slice)

                    current_slice = VerticalSlice(
                        piece_id=piece_id,
                        onset_time=onset,
                        duration_ticks=480,  # Default, updated later
                    )
                    current_base_time = onset

                # Add to current slice
                if current_slice is not None:
                    gm = occ['gm_program']
                    # Only keep one pattern per instrument per slice
                    # (prioritize hierarchical patterns)
                    if gm not in current_slice.instruments or \
                       occ['pattern'].get('is_hierarchical', False):
                        current_slice.instruments[gm] = {
                            'pattern_id': occ['pattern_id'],
                            'first_pitch': occ['first_pitch'],
                            'is_drum': occ['is_drum'],
                            'pattern': occ['pattern'],
                            'tau_offset': occ.get('tau_offset', 480),  # Use corpus timing
                        }

            # Add final slice
            if current_slice and current_slice.n_instruments > 0:
                slices.append(current_slice)

            # Update durations (time until next slice)
            for i in range(len(slices) - 1):
                slices[i].duration_ticks = slices[i+1].onset_time - slices[i].onset_time

            if slices:
                self.slice_sequences[piece_id] = SliceSequence(
                    piece_id=piece_id,
                    slices=slices,
                )
                total_slices += len(slices)

        if self.verbose:
            print(f"  Built {total_slices} vertical slices from {len(self.slice_sequences)} pieces")

            # Show slice statistics
            multi_instrument = sum(
                1 for seq in self.slice_sequences.values()
                for s in seq.slices if s.n_instruments >= 2
            )
            print(f"  Multi-instrument slices: {multi_instrument}")

    def get_slice_pool(self, min_instruments: int = 2) -> List[VerticalSlice]:
        """Get pool of vertical slices filtered by instrument count."""
        pool = []
        for seq in self.slice_sequences.values():
            for s in seq.slices:
                if s.n_instruments >= min_instruments:
                    pool.append(s)
        return pool

    def rhythm_bucket_to_ioi(self, bucket: int, ticks_per_beat: int = 480) -> int:
        """Convert rhythm bucket to inter-onset interval."""
        if bucket < 4:
            return ticks_per_beat // 8
        elif bucket < 8:
            return ticks_per_beat // 4
        elif bucket < 12:
            return ticks_per_beat // 2
        else:
            return ticks_per_beat

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
            pcs = pattern.get('pitch_classes', [])
            if len(pcs) < 2:
                return notes
            intervals = []
            for i in range(1, len(pcs)):
                diff = (pcs[i] - pcs[i-1]) % 12
                if diff > 6:
                    diff -= 12
                intervals.append(diff)

        velocity_bucket = pattern.get('velocity_bucket', 4)
        velocity = 60 + velocity_bucket * 10
        velocity = min(127, max(40, velocity))

        pitch_range = GM_RANGES.get(gm_program, DEFAULT_RANGE)

        current_pitch = first_pitch
        current_time = start_time

        # First note
        notes.append({
            'pitch': max(pitch_range[0], min(pitch_range[1], current_pitch)),
            'velocity': velocity,
            'time': current_time,
            'duration': base_ioi,
        })

        # Remaining notes
        for interval in intervals:
            current_time += base_ioi
            current_pitch += interval

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

    def expand_slice(
        self,
        slice_: VerticalSlice,
        start_time: int,
        ticks_per_beat: int = 480,
    ) -> Dict[int, List[Dict]]:
        """Expand a vertical slice to notes for all instruments."""
        tracks = defaultdict(list)

        for gm, occ_data in slice_.instruments.items():
            pattern = occ_data['pattern']
            first_pitch = occ_data['first_pitch']
            tau_offset = occ_data.get('tau_offset', 480)  # Use corpus timing

            notes = self.expand_pattern(
                pattern=pattern,
                first_pitch=first_pitch,
                start_time=start_time,
                gm_program=gm,
                base_ioi=tau_offset,  # Use actual corpus IOI
            )

            tracks[gm].extend(notes)

        return tracks

    def generate_from_slices(
        self,
        bars: int = 32,
        min_instruments: int = 2,
        target_instruments: List[int] = None,
        ticks_per_beat: int = 480,
        seed: int = None,
    ) -> Dict[int, List[Dict]]:
        """Generate by sampling and chaining vertical slices.

        Args:
            bars: Number of bars to generate
            min_instruments: Minimum instruments per slice
            target_instruments: Prefer slices with these instruments
            ticks_per_beat: MIDI resolution
            seed: Random seed
        """
        if seed is not None:
            random.seed(seed)

        # Get slice pool
        pool = self.get_slice_pool(min_instruments)

        if not pool:
            print("ERROR: No slices with enough instruments")
            return {}

        # Filter for target instruments if specified
        if target_instruments:
            target_set = set(target_instruments)
            filtered = [s for s in pool if target_set & set(s.instruments.keys())]
            if filtered:
                pool = filtered

        if self.verbose:
            print(f"\nSampling from {len(pool)} vertical slices...")

        # Calculate duration
        duration_ticks = bars * 4 * ticks_per_beat

        # Sample and chain slices
        all_tracks = defaultdict(list)
        current_time = 0
        n_slices = 0

        while current_time < duration_ticks:
            # Sample a slice
            slice_ = random.choice(pool)

            # Expand slice
            slice_tracks = self.expand_slice(slice_, current_time, ticks_per_beat)

            # Merge into output
            for gm, notes in slice_tracks.items():
                all_tracks[gm].extend(notes)

            # Advance time
            # Use slice duration or calculate from pattern
            slice_duration = slice_.duration_ticks
            if slice_duration < ticks_per_beat // 2:
                slice_duration = ticks_per_beat * 2  # Default 2 beats
            current_time += slice_duration
            n_slices += 1

        if self.verbose:
            print(f"  Generated {n_slices} slices")
            for gm, notes in sorted(all_tracks.items()):
                avg_pitch = np.mean([n['pitch'] for n in notes]) if notes else 0
                print(f"    GM {gm}: {len(notes)} notes, avg_pitch={avg_pitch:.0f}")

        return dict(all_tracks)

    def generate_from_piece(
        self,
        piece_id: str = None,
        bars: int = 32,
        ticks_per_beat: int = 480,
        seed: int = None,
    ) -> Dict[int, List[Dict]]:
        """Generate by sampling slices from a specific piece (REMIX mode).

        This maintains the highest coherence since all slices come from
        the same piece's arrangement.
        """
        if seed is not None:
            random.seed(seed)

        # Select piece
        if piece_id is None:
            # Pick piece with most slices
            piece_id = max(
                self.slice_sequences.keys(),
                key=lambda p: len(self.slice_sequences[p].slices)
            )

        if piece_id not in self.slice_sequences:
            print(f"ERROR: Piece '{piece_id}' not found")
            return {}

        seq = self.slice_sequences[piece_id]

        if self.verbose:
            print(f"\nRemixing piece: {piece_id} ({len(seq.slices)} slices)")
            print(f"  Instruments: {sorted(seq.instruments)}")

        duration_ticks = bars * 4 * ticks_per_beat
        all_tracks = defaultdict(list)
        current_time = 0

        # Sample slices from this piece (with replacement, shuffled order)
        available_slices = list(seq.slices)

        while current_time < duration_ticks:
            slice_ = random.choice(available_slices)

            slice_tracks = self.expand_slice(slice_, current_time, ticks_per_beat)
            for gm, notes in slice_tracks.items():
                all_tracks[gm].extend(notes)

            slice_duration = slice_.duration_ticks
            if slice_duration < ticks_per_beat // 2:
                slice_duration = ticks_per_beat * 2
            current_time += slice_duration

        if self.verbose:
            for gm, notes in sorted(all_tracks.items()):
                avg_pitch = np.mean([n['pitch'] for n in notes]) if notes else 0
                print(f"    GM {gm}: {len(notes)} notes, avg_pitch={avg_pitch:.0f}")

        return dict(all_tracks)

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
        channel_map = {}
        next_channel = 0

        for gm, notes in sorted(tracks.items()):
            if not notes:
                continue

            track = mido.MidiTrack()
            mid.tracks.append(track)

            # Assign channel (avoid channel 9 for melodic, use for drums)
            is_drum = gm >= 128  # Convention: 128+ = drums
            if is_drum:
                channel = 9
            else:
                if gm not in channel_map:
                    if next_channel == 9:
                        next_channel = 10  # Skip drum channel
                    channel_map[gm] = next_channel % 16
                    next_channel += 1
                channel = channel_map[gm]

            track.append(mido.Message('program_change', program=gm % 128,
                                     channel=channel, time=0))

            # Build events
            events = []
            for note in notes:
                events.append((note['time'], 'note_on', note['pitch'],
                              note['velocity'], channel))
                events.append((note['time'] + note['duration'], 'note_off',
                              note['pitch'], 0, channel))

            events.sort(key=lambda e: (e[0], 0 if e[1] == 'note_off' else 1))

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
    parser = argparse.ArgumentParser(description='Vertical slice generator')
    parser.add_argument('checkpoint', help='Path to v53 checkpoint .npz file')
    parser.add_argument('--output', '-o', default='vertical_generated.mid', help='Output MIDI path')
    parser.add_argument('--bars', '-b', type=int, default=32, help='Number of bars')
    parser.add_argument('--tempo', '-t', type=int, default=120, help='Tempo BPM')
    parser.add_argument('--seed', '-s', type=int, help='Random seed')
    parser.add_argument('--piece', '-p', help='Remix specific piece (by piece_id)')
    parser.add_argument('--min-instruments', '-m', type=int, default=2,
                       help='Minimum instruments per slice')
    parser.add_argument('--list-pieces', '-l', action='store_true',
                       help='List available pieces and exit')
    args = parser.parse_args()

    print("=" * 60)
    print("VERTICAL SLICE GENERATOR (Arrangement-Aware)")
    print("=" * 60)

    gen = VerticalSliceGenerator(args.checkpoint)

    if args.list_pieces:
        print("\nAvailable pieces (by slice count):")
        pieces = sorted(
            gen.slice_sequences.items(),
            key=lambda x: len(x[1].slices),
            reverse=True
        )
        for piece_id, seq in pieces[:20]:
            instruments = sorted(seq.instruments)
            print(f"  {piece_id}: {len(seq.slices)} slices, instruments: {instruments[:5]}...")
        return

    # Generate
    if args.piece:
        # Remix mode - sample from specific piece
        tracks = gen.generate_from_piece(
            piece_id=args.piece,
            bars=args.bars,
            seed=args.seed,
        )
    else:
        # General mode - sample from all pieces
        tracks = gen.generate_from_slices(
            bars=args.bars,
            min_instruments=args.min_instruments,
            seed=args.seed,
        )

    # Save
    gen.to_midi(tracks, args.output, tempo=args.tempo)

    print("\nDone!")


if __name__ == '__main__':
    main()
