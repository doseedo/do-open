#!/usr/bin/env python3
"""
HYBRID GENERATOR - v55 Harmony + Clean Rhythm

The Problem:
- v55 has excellent harmony/co-occurrence data (patterns that play together)
- But the rhythm_ratios approach creates wild IOIs when applied multiplicatively
- Result: Harmonically coherent but rhythmically chaotic

The Solution:
- Use v55 for WHAT to play (pattern selection, co-occurrence, pitch)
- Use clean tau_offset values for WHEN to play (rhythm)
- Quantize IOIs to musical divisions (120, 240, 480, 720, 960)

Usage:
    python scripts/hybrid_generator.py --piece "Caravan" --bars 32 -o output.mid
    python scripts/hybrid_generator.py --piece "score - 2025-08-07T204344.144" --bars 16 -o output.mid
"""

import orjson
import json
import random
import argparse
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np


# Quantization grid - standard musical divisions at 480 ticks/beat
QUANT_GRID = [60, 120, 160, 240, 320, 480, 720, 960, 1440, 1920]

# Strict grid - only clean musical values (like GT Caravan)
STRICT_GRID = [120, 240, 480, 720, 960, 1920]

GM_NAMES = {
    0: 'Piano', 32: 'Acoustic Bass', 33: 'Electric Bass',
    56: 'Trumpet', 57: 'Trombone', 60: 'French Horn',
    65: 'Alto Sax', 66: 'Tenor Sax', 67: 'Baritone Sax',
    71: 'Clarinet', 73: 'Flute', 128: 'Drums'
}


def quantize_tau(tau: int, grid: List[int] = QUANT_GRID, strict: bool = False) -> int:
    """Quantize tau to nearest grid value.

    Args:
        tau: Raw tau value in ticks
        grid: Quantization grid
        strict: If True, use STRICT_GRID for cleaner rhythm
    """
    if strict:
        grid = STRICT_GRID
    if tau <= 0:
        return 480  # Default quarter note
    return min(grid, key=lambda g: abs(g - tau))


class HybridGenerator:
    """Generate with v55 harmony + clean rhythm."""

    def __init__(self, patterns_path: str, verbose: bool = True):
        self.verbose = verbose
        self.patterns_path = patterns_path
        self.patterns = None

        # Indices
        self.piece_patterns = defaultdict(list)  # piece -> [occurrences]
        self.pattern_by_gm = defaultdict(list)   # gm -> [pattern_ids]

        self._load()

    def _load(self):
        """Load patterns and build indices."""
        if self.verbose:
            print(f"Loading patterns from {self.patterns_path}...")

        with open(self.patterns_path, 'rb') as f:
            self.patterns = orjson.loads(f.read())

        if self.verbose:
            print(f"  Loaded {len(self.patterns)} patterns")

        # Build indices
        self._build_indices()

    def _build_indices(self):
        """Build piece and instrument indices."""
        if self.verbose:
            print("Building indices...")

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            self.pattern_by_gm[gm].append(pid)

            for occ in p.get('occurrences', []):
                piece_id = occ.get('piece_id', 'unknown')
                tau = occ.get('tau_offset', 480)

                # Skip invalid tau
                if tau <= 0:
                    continue

                self.piece_patterns[piece_id].append({
                    'pattern_id': pid,
                    'gm': occ.get('gm_program', gm),
                    'onset': occ.get('onset_time', 0),
                    'pitch': occ.get('first_pitch', 60),
                    'tau': tau,
                    'intervals': p.get('pitch_intervals', []),
                    'duration_ratios': p.get('duration_ratios', []),
                })

        if self.verbose:
            print(f"  Indexed {len(self.piece_patterns)} pieces")
            print(f"  Indexed {len(self.pattern_by_gm)} instruments")

    def list_pieces(self, min_events: int = 100) -> List[Tuple[str, int]]:
        """List available pieces by event count."""
        pieces = [(pid, len(occs)) for pid, occs in self.piece_patterns.items()]
        pieces = [(p, n) for p, n in pieces if n >= min_events]
        return sorted(pieces, key=lambda x: -x[1])

    def find_piece(self, query: str) -> Optional[str]:
        """Find piece by partial name match."""
        query_lower = query.lower()
        for piece_id in self.piece_patterns:
            if query_lower in piece_id.lower():
                return piece_id
        return None

    def get_piece_section(
        self,
        piece_id: str,
        start_bar: int = 0,
        num_bars: int = 16,
        ticks_per_beat: int = 480,
        deduplicate: bool = True,
    ) -> Dict[int, List[dict]]:
        """Extract a section from a piece.

        This preserves the EXACT timing and harmony from the source piece,
        just time-shifted to start at 0.

        Args:
            deduplicate: If True, only keep the smallest (leaf) pattern at each
                        (gm, onset) to avoid hierarchical pattern duplication.
                        The Re-Pair grammar stores leaf + parent + grandparent
                        patterns at the same time, causing 9-10x note duplication.
        """
        if piece_id not in self.piece_patterns:
            raise ValueError(f"Piece not found: {piece_id}")

        occs = self.piece_patterns[piece_id]

        # Convert bars to ticks
        start_tick = start_bar * 4 * ticks_per_beat
        end_tick = (start_bar + num_bars) * 4 * ticks_per_beat

        # Filter to section
        section_occs = [
            o for o in occs
            if start_tick <= o['onset'] < end_tick
        ]

        # CRITICAL FIX: Deduplicate hierarchical patterns
        # At each (gm, onset), only keep the smallest pattern (leaf)
        if deduplicate and section_occs:
            from collections import defaultdict
            by_moment = defaultdict(list)
            for o in section_occs:
                key = (o['gm'], o['onset'])
                by_moment[key].append(o)

            # Keep only smallest pattern at each moment
            section_occs = []
            for key, patterns in by_moment.items():
                # Sort by number of notes (len(intervals) + 1), keep smallest
                smallest = min(patterns, key=lambda p: len(p.get('intervals', [])))
                section_occs.append(smallest)

        if not section_occs:
            # Find where patterns actually start
            all_onsets = [o['onset'] for o in occs]
            if all_onsets:
                actual_start = min(all_onsets) // (4 * ticks_per_beat)
                print(f"  WARNING: No patterns in bars {start_bar}-{start_bar+num_bars}")
                print(f"  Patterns start at bar {actual_start}")
                # Adjust to actual start
                start_tick = actual_start * 4 * ticks_per_beat
                end_tick = (actual_start + num_bars) * 4 * ticks_per_beat
                section_occs = [
                    o for o in occs
                    if start_tick <= o['onset'] < end_tick
                ]

        # Group by instrument
        by_gm = defaultdict(list)
        for occ in section_occs:
            by_gm[occ['gm']].append({
                **occ,
                'onset': occ['onset'] - start_tick,  # Normalize to 0
            })

        # Sort each track by onset
        for gm in by_gm:
            by_gm[gm] = sorted(by_gm[gm], key=lambda x: x['onset'])

        if self.verbose:
            print(f"\nExtracted section: bars {start_bar}-{start_bar+num_bars}")
            print(f"  Total events: {sum(len(v) for v in by_gm.values())}")
            for gm, events in sorted(by_gm.items()):
                name = GM_NAMES.get(gm, f'GM{gm}')
                print(f"  {name}: {len(events)} events")

        return dict(by_gm)

    def expand_to_notes(
        self,
        events: Dict[int, List[dict]],
        quantize: bool = True,
        strict: bool = False,
    ) -> Dict[int, List[dict]]:
        """Expand pattern events to individual notes.

        Key fix: Use CONSTANT tau per pattern (no multiplicative ratios).

        Args:
            events: Pattern events grouped by GM program
            quantize: Enable quantization to grid
            strict: Use strict grid (120, 240, 480, 720, 960, 1920 only)
        """
        notes_by_gm = defaultdict(list)

        for gm, event_list in events.items():
            for event in event_list:
                onset = event['onset']
                pitch = event['pitch']
                intervals = event.get('intervals', [])
                tau = event.get('tau', 480)

                # Quantize tau to clean musical value
                if quantize:
                    tau = quantize_tau(tau, strict=strict)
                    # Also quantize onset to grid for cleaner rhythm
                    if strict:
                        onset = (onset // 120) * 120  # Quantize to 8th note grid

                # Skip if tau is still 0
                if tau <= 0:
                    tau = 480

                # Get duration ratios (for note length, not IOI)
                dur_ratios = event.get('duration_ratios', [])

                # First note
                current_time = onset
                current_pitch = pitch
                n_notes = len(intervals) + 1

                for i in range(n_notes):
                    # Duration from ratio if available, else 90% of tau
                    if i < len(dur_ratios) and dur_ratios[i] > 0:
                        duration = int(tau * dur_ratios[i] * 0.9)
                    else:
                        duration = int(tau * 0.9)

                    duration = max(30, min(duration, tau * 2))  # Clamp

                    notes_by_gm[gm].append({
                        'pitch': max(0, min(127, current_pitch)),
                        'onset': current_time,
                        'duration': duration,
                        'velocity': 80,
                    })

                    # Advance by CONSTANT tau (key fix!)
                    current_time += tau

                    # Apply interval for next note
                    if i < len(intervals):
                        current_pitch += intervals[i]

        return dict(notes_by_gm)

    def to_midi(
        self,
        notes: Dict[int, List[dict]],
        output_path: str,
        tempo: int = 120,
        ticks_per_beat: int = 480,
    ):
        """Save notes to MIDI file."""
        mid = MidiFile(ticks_per_beat=ticks_per_beat, type=1)

        # Tempo track
        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))
        tempo_track.append(MetaMessage('end_of_track', time=0))

        channel_map = {}
        next_channel = 0

        for gm, note_list in sorted(notes.items()):
            if not note_list:
                continue

            # Assign channel
            is_drum = gm >= 128
            if is_drum:
                channel = 9
            else:
                if gm not in channel_map:
                    if next_channel == 9:
                        next_channel = 10
                    channel_map[gm] = next_channel
                    next_channel = (next_channel + 1) % 16
                channel = channel_map[gm]

            track = MidiTrack()
            mid.tracks.append(track)

            name = GM_NAMES.get(gm, f'GM{gm}')
            track.append(MetaMessage('track_name', name=name, time=0))
            track.append(Message('program_change', program=gm % 128, channel=channel, time=0))

            # Build events
            midi_events = []
            for n in note_list:
                midi_events.append((n['onset'], 'on', n['pitch'], n['velocity']))
                midi_events.append((n['onset'] + n['duration'], 'off', n['pitch'], 0))

            midi_events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))

            last_time = 0
            for abs_time, msg_type, pitch, vel in midi_events:
                delta = abs_time - last_time
                if msg_type == 'on':
                    track.append(Message('note_on', note=pitch, velocity=vel, channel=channel, time=delta))
                else:
                    track.append(Message('note_off', note=pitch, velocity=0, channel=channel, time=delta))
                last_time = abs_time

            track.append(MetaMessage('end_of_track', time=0))

        mid.save(output_path)

        total_notes = sum(len(n) for n in notes.values())
        if self.verbose:
            print(f"\nSaved to: {output_path}")
            print(f"  Tracks: {len(mid.tracks) - 1}")
            print(f"  Notes: {total_notes}")

    def generate_from_piece(
        self,
        piece_id: str,
        start_bar: int = 0,
        num_bars: int = 16,
        output_path: str = None,
        tempo: int = 120,
        quantize: bool = True,
        strict: bool = False,
    ) -> Dict[int, List[dict]]:
        """Generate by extracting from a source piece.

        This is RECONSTRUCTION mode - preserves original harmony and timing.

        Args:
            strict: Use strict quantization grid (cleaner rhythm like GT Caravan)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"HYBRID GENERATOR - Piece Section")
            print(f"{'='*60}")
            print(f"Source: {piece_id}")
            if strict:
                print(f"Mode: STRICT rhythm quantization")

        # Get section
        events = self.get_piece_section(piece_id, start_bar, num_bars)

        # Expand to notes with clean rhythm
        notes = self.expand_to_notes(events, quantize=quantize, strict=strict)

        # Save if output path provided
        if output_path:
            self.to_midi(notes, output_path, tempo=tempo)

        return notes

    def generate_remix(
        self,
        piece_id: str,
        num_bars: int = 32,
        output_path: str = None,
        tempo: int = 120,
        quantize: bool = True,
        strict: bool = False,
    ) -> Dict[int, List[dict]]:
        """Generate by sampling vertical slices from a piece.

        This is REMIX mode - maintains harmonic coherence but reorders.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"HYBRID GENERATOR - Remix Mode")
            print(f"{'='*60}")
            print(f"Source: {piece_id}")

        if piece_id not in self.piece_patterns:
            raise ValueError(f"Piece not found: {piece_id}")

        occs = self.piece_patterns[piece_id]

        # Group by onset time (vertical slices)
        time_tolerance = 120  # ticks
        slices = []
        current_slice = []
        current_base_time = -time_tolerance * 2

        sorted_occs = sorted(occs, key=lambda x: x['onset'])

        for occ in sorted_occs:
            if occ['onset'] > current_base_time + time_tolerance:
                if current_slice:
                    slices.append(current_slice)
                current_slice = [occ]
                current_base_time = occ['onset']
            else:
                current_slice.append(occ)

        if current_slice:
            slices.append(current_slice)

        # Filter to multi-instrument slices
        multi_slices = [s for s in slices if len(set(o['gm'] for o in s)) >= 2]

        if not multi_slices:
            multi_slices = slices

        if self.verbose:
            print(f"  Found {len(slices)} slices, {len(multi_slices)} multi-instrument")

        # Sample and chain slices
        target_ticks = num_bars * 4 * 480
        current_time = 0
        output_events = defaultdict(list)

        while current_time < target_ticks:
            # Sample a slice
            slice_ = random.choice(multi_slices)

            # Get base time of slice
            base_time = min(o['onset'] for o in slice_)

            # Add all events from slice, time-shifted
            for occ in slice_:
                event = {
                    **occ,
                    'onset': current_time + (occ['onset'] - base_time)
                }
                output_events[occ['gm']].append(event)

            # Advance time (use typical slice duration or 2 beats)
            slice_duration = 960  # Default 2 beats
            taus = [o['tau'] for o in slice_ if o['tau'] > 0]
            if taus:
                slice_duration = max(taus) * 2

            current_time += quantize_tau(slice_duration, strict=strict)

        # Expand to notes
        notes = self.expand_to_notes(dict(output_events), quantize=quantize, strict=strict)

        if output_path:
            self.to_midi(notes, output_path, tempo=tempo)

        return notes


def analyze_rhythm(midi_path: str, name: str = ""):
    """Analyze rhythm coherence of a MIDI file."""
    print(f"\n{'='*60}")
    print(f"RHYTHM ANALYSIS: {name or midi_path}")
    print(f"{'='*60}")

    mid = MidiFile(midi_path)

    track_iois = {}

    for track_idx, track in enumerate(mid.tracks):
        current_time = 0
        notes = []
        gm = 0

        for msg in track:
            current_time += msg.time
            if msg.type == 'program_change':
                gm = msg.program
            elif msg.type == 'note_on' and msg.velocity > 0:
                notes.append(current_time)

        if len(notes) < 2:
            continue

        notes = sorted(notes)
        iois = [notes[i+1] - notes[i] for i in range(len(notes)-1) if notes[i+1] > notes[i]]

        if iois:
            track_iois[gm] = iois

    # Analyze
    all_iois = []
    for gm, iois in track_iois.items():
        all_iois.extend(iois)
        name = GM_NAMES.get(gm, f'GM{gm}')
        unique = len(set(iois))
        print(f"\n{name}:")
        print(f"  Notes: {len(iois)+1}")
        print(f"  Unique IOIs: {unique}")
        print(f"  Top IOIs: {Counter(iois).most_common(5)}")

    # Overall stats
    if all_iois:
        print(f"\nOVERALL:")
        print(f"  Unique IOIs: {len(set(all_iois))}")
        print(f"  Top IOIs: {Counter(all_iois).most_common(8)}")

        # Check if quantized to grid
        on_grid = sum(1 for ioi in all_iois if ioi in QUANT_GRID)
        print(f"  On grid: {100*on_grid/len(all_iois):.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Hybrid Generator - v55 Harmony + Clean Rhythm')
    parser.add_argument('--patterns', type=str,
                       default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based/checkpoint_v55_pure_contour_1000files_patterns.json',
                       help='Path to v55 patterns JSON')
    parser.add_argument('--piece', '-p', type=str, help='Source piece (partial match OK)')
    parser.add_argument('--start-bar', type=int, default=0, help='Starting bar')
    parser.add_argument('--bars', '-b', type=int, default=16, help='Number of bars')
    parser.add_argument('--output', '-o', type=str, default='hybrid_output.mid', help='Output path')
    parser.add_argument('--tempo', '-t', type=int, default=120, help='Tempo BPM')
    parser.add_argument('--remix', action='store_true', help='Remix mode (reorder slices)')
    parser.add_argument('--no-quantize', action='store_true', help='Disable rhythm quantization')
    parser.add_argument('--strict', '-s', action='store_true', help='Strict rhythm quantization (cleaner, like GT)')
    parser.add_argument('--list', '-l', action='store_true', help='List available pieces')
    parser.add_argument('--analyze', '-a', type=str, help='Analyze MIDI file rhythm')
    args = parser.parse_args()

    # Analyze mode
    if args.analyze:
        analyze_rhythm(args.analyze)
        return

    # Initialize generator
    gen = HybridGenerator(args.patterns)

    # List mode
    if args.list:
        print("\nAvailable pieces (top 30 by event count):")
        for piece_id, count in gen.list_pieces()[:30]:
            print(f"  {piece_id}: {count} events")
        return

    # Need a piece
    if not args.piece:
        print("ERROR: --piece required")
        print("Use --list to see available pieces")
        return

    # Find piece
    piece_id = gen.find_piece(args.piece)
    if not piece_id:
        print(f"ERROR: No piece matching '{args.piece}'")
        print("Use --list to see available pieces")
        return

    # Generate
    if args.remix:
        gen.generate_remix(
            piece_id=piece_id,
            num_bars=args.bars,
            output_path=args.output,
            tempo=args.tempo,
            quantize=not args.no_quantize,
            strict=args.strict,
        )
    else:
        gen.generate_from_piece(
            piece_id=piece_id,
            start_bar=args.start_bar,
            num_bars=args.bars,
            output_path=args.output,
            strict=args.strict,
            tempo=args.tempo,
            quantize=not args.no_quantize,
        )

    # Analyze output
    analyze_rhythm(args.output, "Generated Output")


if __name__ == '__main__':
    main()
