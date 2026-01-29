#!/usr/bin/env python3
"""
Grid-Based Sampler - Synchronized generation on beat grid.

Problem with previous samplers:
- Each instrument advances time independently
- No vertical synchronization
- Patterns drift apart

Solution:
- Shared beat grid (all instruments step together)
- At each grid point, sample patterns that START together
- Pattern notes fill forward, but next decision is on grid

This produces homophonic texture like real big band arrangements.
"""

import orjson
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


class GridSampler:
    """Generate on shared beat grid for vertical coordination."""

    def __init__(self, patterns: dict, verbose: bool = True):
        self.patterns = patterns
        self.verbose = verbose
        self._build_indices()

    def _build_indices(self):
        """Build indices for grid-based generation."""
        if self.verbose:
            print("Building grid indices...")

        # Patterns by instrument
        self.patterns_by_gm = defaultdict(list)
        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            n_notes = len(p.get('pitch_classes', [1]))
            if n_notes <= 4:
                self.patterns_by_gm[gm].append(pid)

        if self.verbose:
            for gm in sorted(self.patterns_by_gm.keys()):
                print(f"  GM {gm}: {len(self.patterns_by_gm[gm])} short patterns")

        # Collect ACTUAL PITCH combinations (not just PC)
        # Key: (piece, beat) -> {gm: pitch}
        beat_pitches = defaultdict(dict)

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            for occ in p.get('occurrences', []):
                piece = occ.get('piece_id', 'unknown')
                onset = occ.get('onset_time', 0)
                pitch = occ.get('first_pitch', 60)
                beat = onset // 480
                key = (piece, beat)
                if gm not in beat_pitches[key]:
                    beat_pitches[key][gm] = pitch

        # Build VOICING distribution (actual pitches, not PC)
        # Key: tuple of (gm, pitch) sorted by gm
        self.voicings = Counter()
        self.voicing_transitions = defaultdict(Counter)

        piece_sequences = defaultdict(list)
        for (piece, beat), gm_pitches in beat_pitches.items():
            if len(gm_pitches) >= 2:
                voicing = tuple(sorted(gm_pitches.items()))
                piece_sequences[piece].append((beat, voicing))

        for piece, seq in piece_sequences.items():
            seq.sort(key=lambda x: x[0])
            for i, (beat, voicing) in enumerate(seq):
                self.voicings[voicing] += 1
                if i > 0:
                    prev = seq[i-1][1]
                    self.voicing_transitions[prev][voicing] += 1

        if self.verbose:
            print(f"  Voicings: {len(self.voicings)}")
            print(f"  Voicing transitions: {sum(len(v) for v in self.voicing_transitions.values())}")

    def sample_voicing(self, instruments: List[int], previous=None):
        """Sample actual pitch voicing (not just PC) for instruments."""
        instruments_set = set(instruments)

        # Try Markov transition from previous voicing
        if previous and previous in self.voicing_transitions:
            trans = self.voicing_transitions[previous]
            valid = [(v, c) for v, c in trans.items()
                    if instruments_set <= {gm for gm, _ in v}]
            if valid:
                voicings, counts = zip(*valid)
                total = sum(counts)
                return random.choices(voicings, weights=[c/total for c in counts])[0]

        # From marginal distribution
        valid = [(v, c) for v, c in self.voicings.items()
                if instruments_set <= {gm for gm, _ in v}]
        if valid:
            voicings, counts = zip(*valid)
            total = sum(counts)
            return random.choices(voicings, weights=[c/total for c in counts])[0]

        # Fallback: construct a voicing
        root = random.choice([48, 55, 60, 67])  # Common roots
        intervals = [0, 7, 12]  # Octave + fifth voicing
        return tuple((gm, root + intervals[i % 3]) for i, gm in enumerate(sorted(instruments)))

    def sample_pc_joint(self, instruments: List[int], previous=None):
        """Legacy - sample pitch-class joint."""
        # Convert voicing to PC joint for compatibility
        voicing = self.sample_voicing(instruments, previous)
        return frozenset((gm, pitch % 12) for gm, pitch in voicing)

    def sample_pitch(self, gm: int, pc: int) -> int:
        """Sample absolute pitch given instrument and pitch class."""
        key = (gm, pc)
        if key in self.pitch_dist and self.pitch_dist[key]:
            pitches = list(self.pitch_dist[key].keys())
            weights = list(self.pitch_dist[key].values())
            total = sum(weights)
            return random.choices(pitches, weights=[w/total for w in weights])[0]

        # Instrument-appropriate default octave
        if gm in [32, 33, 34, 35, 36, 37, 38, 39]:  # Bass
            return 36 + pc
        elif gm in [56, 57, 58, 59, 60, 61, 62, 63]:  # Brass
            return 60 + pc
        elif gm in [64, 65, 66, 67, 68, 69, 70, 71]:  # Reeds
            return 60 + pc
        return 60 + pc

    def sample_pattern(self, gm: int) -> dict:
        """Sample a short pattern for instrument."""
        candidates = self.patterns_by_gm.get(gm, [])
        if not candidates:
            return {'intervals': [], 'ioi': 480, 'n_notes': 1}

        # Weight by count
        weights = [self.patterns[pid].get('count', 1) for pid in candidates]
        total = sum(weights)
        pid = random.choices(candidates, weights=[w/total for w in weights])[0]

        p = self.patterns[pid]

        # Get IOI from occurrence (use tau_offset)
        occs = p.get('occurrences', [])
        if occs:
            ioi = occs[0].get('tau_offset', 480)
        else:
            ioi = 480

        return {
            'pid': pid,
            'intervals': p.get('pitch_intervals', []),
            'ioi': max(120, min(960, ioi)),  # Clamp to reasonable range
            'n_notes': len(p.get('pitch_classes', [1])),
        }

    def generate(
        self,
        n_beats: int,
        instruments: List[int],
        beats_per_event: int = 2,
        anchor_pattern: List[float] = None
    ) -> Dict[int, List[dict]]:
        """Generate on beat grid with anchor-aware coordination.

        Args:
            n_beats: Total beats to generate
            instruments: GM programs
            beats_per_event: How many beats between coordination events
            anchor_pattern: Probability of all-instrument hit per beat in bar
                           e.g., [1.0, 1.0, 0.6, 0.4] for "In The Mood" style

        Returns:
            {gm: [{'pitch': ..., 'onset': ..., 'duration': ...}, ...]}
        """
        # Default: "In The Mood" style anchors (beat 1,2 strong, 3,4 weaker)
        if anchor_pattern is None:
            anchor_pattern = [0.9, 0.9, 0.6, 0.4]

        output = {gm: [] for gm in instruments}
        previous_voicing = None

        for beat in range(n_beats):
            onset_ticks = beat * 480
            beat_in_bar = beat % 4
            anchor_prob = anchor_pattern[beat_in_bar]

            # Decide if this is an anchor beat (all instruments play)
            is_anchor = random.random() < anchor_prob

            if is_anchor:
                # Sample actual voicing (pitches, not just PC)
                voicing = self.sample_voicing(instruments, previous_voicing)

                for gm, pitch in voicing:
                    if gm in instruments:
                        output[gm].append({
                            'pitch': max(0, min(127, pitch)),
                            'onset': onset_ticks,
                            'duration': 360,
                            'velocity': 85,
                        })

                previous_voicing = voicing
            else:
                # Non-anchor: each instrument independently decides to play
                for gm in instruments:
                    if random.random() < 0.3:  # 30% chance of fill note
                        # Use pitch from previous voicing if available
                        pitch = None
                        if previous_voicing:
                            for vgm, vpitch in previous_voicing:
                                if vgm == gm:
                                    pitch = vpitch
                                    break

                        if pitch is None:
                            # Fallback: instrument-appropriate default
                            if gm in [32, 33, 34, 35, 36, 37, 38, 39]:  # Bass
                                pitch = random.choice([36, 43, 48, 41])
                            elif gm in [56, 57, 58, 59, 60, 61, 62, 63]:  # Brass
                                pitch = random.choice([60, 62, 65, 67])
                            elif gm in [64, 65, 66, 67, 68, 69, 70, 71]:  # Reeds
                                pitch = random.choice([60, 62, 65, 67])
                            else:
                                pitch = random.choice([60, 62, 65, 67])

                        # Shorter duration for fills
                        output[gm].append({
                            'pitch': max(0, min(127, pitch)),
                            'onset': onset_ticks,
                            'duration': 200,
                            'velocity': 70,
                        })

        return output

    def generate_to_midi(
        self,
        n_beats: int,
        instruments: List[int],
        output_path: str,
        beats_per_event: int = 2,
        bpm: int = 120
    ):
        """Generate and save to MIDI."""
        print(f"\nGenerating {n_beats} beats on grid")
        print(f"  Event every {beats_per_event} beats")
        print(f"  Instruments: {instruments}")

        notes = self.generate(n_beats, instruments, beats_per_event)

        # Create MIDI
        mid = MidiFile(ticks_per_beat=480, type=1)

        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
        tempo_track.append(MetaMessage('end_of_track', time=0))

        channel = 0
        for gm, note_list in sorted(notes.items()):
            if not note_list:
                continue

            track = MidiTrack()
            mid.tracks.append(track)
            track.append(Message('program_change', program=gm % 128, channel=channel, time=0))

            # Sort and create events
            sorted_notes = sorted(note_list, key=lambda n: n['onset'])

            events = []
            for n in sorted_notes:
                events.append((n['onset'], 'on', n['pitch'], n['velocity']))
                events.append((n['onset'] + n['duration'], 'off', n['pitch'], 0))

            events.sort(key=lambda x: (x[0], x[1] == 'on'))

            last_time = 0
            for t, etype, pitch, vel in events:
                delta = t - last_time
                if etype == 'on':
                    track.append(Message('note_on', note=pitch, velocity=vel, channel=channel, time=delta))
                else:
                    track.append(Message('note_off', note=pitch, velocity=0, channel=channel, time=delta))
                last_time = t

            track.append(MetaMessage('end_of_track', time=0))
            channel += 1

        mid.save(output_path)

        total_notes = sum(len(n) for n in notes.values())
        print(f"\nSaved to: {output_path}")
        print(f"  Notes: {total_notes}")

        # Verify alignment
        self._verify_alignment(notes, instruments)

        return notes

    def _verify_alignment(self, notes: Dict[int, List[dict]], instruments: List[int]):
        """Verify vertical alignment."""
        onsets_by_gm = {gm: set(n['onset'] for n in notes[gm]) for gm in instruments}

        shared = onsets_by_gm[instruments[0]]
        for gm in instruments[1:]:
            shared = shared & onsets_by_gm[gm]

        total = len(set.union(*onsets_by_gm.values()))
        print(f"\nAlignment: {len(shared)} / {total} onsets shared ({100*len(shared)/total:.1f}%)")

        # Check harmony at shared onsets
        print("\nFirst 5 shared onsets:")
        pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for onset in sorted(shared)[:5]:
            pitches = []
            for gm in instruments:
                for n in notes[gm]:
                    if n['onset'] == onset:
                        pitches.append(n['pitch'])
            pcs = sorted(set(p % 12 for p in pitches))
            pc_str = ", ".join(pc_names[pc] for pc in pcs)
            print(f"  {onset}: pitches={pitches}, PCs={{{pc_str}}}")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python grid_sampler.py <checkpoint.npz> [-o output.mid] [--beats N] [--instruments GM1 GM2 ...]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_path = '/tmp/grid_output.mid'
    n_beats = 64
    instruments = [65, 66, 67]
    beats_per_event = 2

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '-o' and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--beats' and i + 1 < len(sys.argv):
            n_beats = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--event-beats' and i + 1 < len(sys.argv):
            beats_per_event = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--instruments':
            instruments = []
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith('-'):
                instruments.append(int(sys.argv[i]))
                i += 1
        else:
            i += 1

    # Load
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

    # Generate
    sampler = GridSampler(patterns)
    sampler.generate_to_midi(n_beats, instruments, output_path, beats_per_event)


if __name__ == '__main__':
    main()
