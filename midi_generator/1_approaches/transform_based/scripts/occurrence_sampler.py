#!/usr/bin/env python3
"""
Occurrence-Based Sampler - Uses real musical events, not abstracted patterns.

Key insight: Pattern discovery found real melodic phrases with real pitches
in real harmonic contexts. When we sample a pattern and apply a random pitch,
we destroy that context.

This sampler works differently:
1. Index all occurrences by (piece, beat) for co-occurrence lookup
2. Sample a full occurrence (with original first_pitch)
3. Find co-occurring events from same piece/beat (vertical harmony)
4. Use corpus transitions to find what came next (horizontal melody)

This is closer to "recombining learned chunks" than "sampling from distributions"
but that's how real music often works.
"""

import orjson
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


class OccurrenceSampler:
    """Sample real occurrences with their original pitches and context."""

    def __init__(self, patterns: dict, verbose: bool = True):
        self.patterns = patterns
        self.verbose = verbose
        self._build_indices()

    def _build_indices(self):
        """Build indices for occurrence-based sampling."""
        if self.verbose:
            print("Building occurrence indices...")

        # Occurrence index: (piece, beat) -> [(gm, pid, occurrence), ...]
        self.beat_occurrences = defaultdict(list)

        # Pattern occurrences by GM for random sampling
        self.occurrences_by_gm = defaultdict(list)

        # Transitions: (gm, pid, first_pitch) -> [(next_gm, next_pid, next_pitch), ...]
        self.transitions = defaultdict(list)

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            for occ in p.get('occurrences', []):
                piece = occ.get('piece_id', 'unknown')
                onset = occ.get('onset_time', 0)
                first_pitch = occ.get('first_pitch', 60)
                tau = occ.get('tau_offset', 480)
                beat = onset // 480

                occ_data = {
                    'pid': pid,
                    'gm': gm,
                    'piece': piece,
                    'onset': onset,
                    'beat': beat,
                    'first_pitch': first_pitch,
                    'tau': tau,
                    'intervals': p.get('pitch_intervals', []),
                    'rhythm_ratios': p.get('rhythm_ratios', []),
                    'n_notes': len(p.get('pitch_classes', [1])),
                }

                self.beat_occurrences[(piece, beat)].append(occ_data)
                self.occurrences_by_gm[gm].append(occ_data)

        if self.verbose:
            print(f"  Beat locations: {len(self.beat_occurrences)}")
            for gm in sorted(self.occurrences_by_gm.keys()):
                print(f"  GM {gm}: {len(self.occurrences_by_gm[gm])} occurrences")

        # Build transitions (what follows what within same piece+instrument)
        self._build_transitions()

    def _build_transitions(self):
        """Build transition model from corpus sequences."""
        if self.verbose:
            print("Building transition model...")

        # Group occurrences by (piece, gm) to find sequences
        piece_gm_sequences = defaultdict(list)
        for (piece, beat), occs in self.beat_occurrences.items():
            for occ in occs:
                key = (piece, occ['gm'])
                piece_gm_sequences[key].append((beat, occ))

        # Sort by beat and build transitions
        transition_count = 0
        for (piece, gm), seq in piece_gm_sequences.items():
            seq.sort(key=lambda x: x[0])
            for i in range(len(seq) - 1):
                curr_beat, curr_occ = seq[i]
                next_beat, next_occ = seq[i + 1]

                # Only consider nearby transitions (within 8 beats)
                if next_beat - curr_beat <= 8:
                    # Key: (gm, pid, first_pitch)
                    key = (curr_occ['gm'], curr_occ['pid'], curr_occ['first_pitch'])
                    self.transitions[key].append(next_occ)
                    transition_count += 1

        if self.verbose:
            print(f"  Transitions: {transition_count}")
            print(f"  Unique transition sources: {len(self.transitions)}")

    def get_cooccurrences(
        self,
        piece: str,
        beat: int,
        instruments: Optional[Set[int]] = None
    ) -> List[dict]:
        """Get all occurrences at same (piece, beat), optionally filtered by instrument."""
        occs = self.beat_occurrences.get((piece, beat), [])
        if instruments:
            occs = [o for o in occs if o['gm'] in instruments]
        return occs

    def sample_occurrence(self, gm: int, previous: dict = None) -> Optional[dict]:
        """Sample an occurrence, using Markov transition if previous is given."""
        # Try Markov transition
        if previous:
            key = (previous['gm'], previous['pid'], previous['first_pitch'])
            candidates = self.transitions.get(key, [])
            # Filter to same instrument
            candidates = [c for c in candidates if c['gm'] == gm]
            if candidates:
                return random.choice(candidates)

        # Fall back to random occurrence
        if gm in self.occurrences_by_gm:
            return random.choice(self.occurrences_by_gm[gm])
        return None

    def sample_smooth_continuation(self, gm: int, previous: dict = None) -> Optional[dict]:
        """Sample occurrence that maintains melodic smoothness from previous."""
        if gm not in self.occurrences_by_gm:
            return None

        candidates = self.occurrences_by_gm[gm]

        if not previous:
            return random.choice(candidates)

        prev_pitch = previous['first_pitch']

        # Find candidates within small interval (≤5 semitones)
        smooth = [o for o in candidates if abs(o['first_pitch'] - prev_pitch) <= 5]

        if smooth:
            # Weight by how small the interval is
            weights = [1.0 / (1 + abs(o['first_pitch'] - prev_pitch)) for o in smooth]
            total = sum(weights)
            return random.choices(smooth, weights=[w/total for w in weights])[0]

        # Fall back to medium jumps (≤12)
        medium = [o for o in candidates if abs(o['first_pitch'] - prev_pitch) <= 12]
        if medium:
            return random.choice(medium)

        # Last resort: any
        return random.choice(candidates)

    def occurrence_to_notes(
        self,
        occ: dict,
        onset_offset: int = 0
    ) -> List[dict]:
        """Convert an occurrence to a list of notes."""
        notes = []
        intervals = occ.get('intervals', [])
        rhythm_ratios = occ.get('rhythm_ratios', [])
        n_notes = occ.get('n_notes', 1)
        tau = occ.get('tau', 480)

        # First note
        current_pitch = occ['first_pitch']
        current_onset = onset_offset

        notes.append({
            'pitch': max(0, min(127, current_pitch)),
            'onset': current_onset,
            'duration': min(tau, 400),  # Duration capped at tau or 400
            'velocity': 80,
        })

        # Subsequent notes
        for i in range(min(len(intervals), n_notes - 1)):
            current_pitch += intervals[i]
            if i < len(rhythm_ratios):
                ioi = int(tau * rhythm_ratios[i])
            else:
                ioi = tau
            current_onset += max(60, ioi)  # Min IOI of 60 ticks

            notes.append({
                'pitch': max(0, min(127, current_pitch)),
                'onset': current_onset,
                'duration': min(ioi, 400),
                'velocity': 80,
            })

        return notes

    def generate(
        self,
        n_beats: int,
        instruments: List[int],
        anchor_pattern: List[float] = None
    ) -> Dict[int, List[dict]]:
        """Generate by sampling real occurrences with co-occurrence context.

        Args:
            n_beats: Total beats to generate
            instruments: GM programs to generate
            anchor_pattern: Probability of coordination event per beat in bar

        Returns:
            {gm: [{'pitch': ..., 'onset': ..., 'duration': ...}, ...]}
        """
        if anchor_pattern is None:
            anchor_pattern = [0.9, 0.8, 0.5, 0.4]

        output = {gm: [] for gm in instruments}
        previous_by_gm = {gm: None for gm in instruments}

        beat = 0
        while beat < n_beats:
            onset_ticks = beat * 480
            beat_in_bar = beat % 4
            is_anchor = random.random() < anchor_pattern[beat_in_bar]

            if is_anchor:
                # Sample lead instrument occurrence
                lead_gm = instruments[0]
                lead_occ = self.sample_occurrence(lead_gm, previous_by_gm[lead_gm])

                if lead_occ:
                    # Find co-occurrences from same piece/beat
                    co_occs = self.get_cooccurrences(
                        lead_occ['piece'],
                        lead_occ['beat'],
                        set(instruments)
                    )

                    # Build GM -> occurrence mapping
                    gm_to_occ = {}
                    for co in co_occs:
                        if co['gm'] in instruments and co['gm'] not in gm_to_occ:
                            gm_to_occ[co['gm']] = co

                    # OPTION A: Emit full phrases (multi-note) - creates polyphony
                    # OPTION B: Emit only first note at anchor for tight vertical alignment
                    # Using Option B for now to maximize alignment

                    for gm in instruments:
                        if gm in gm_to_occ:
                            occ = gm_to_occ[gm]
                        else:
                            # Fall back to smooth melodic continuation
                            occ = self.sample_smooth_continuation(gm, previous_by_gm[gm])

                        if occ:
                            # Emit only first note for tight alignment
                            output[gm].append({
                                'pitch': max(0, min(127, occ['first_pitch'])),
                                'onset': onset_ticks,
                                'duration': 360,
                                'velocity': 80,
                            })
                            previous_by_gm[gm] = occ

                beat += 1
            else:
                # Non-anchor: individual instruments may play fills
                for gm in instruments:
                    if random.random() < 0.2:  # 20% fill chance
                        occ = self.sample_occurrence(gm, previous_by_gm[gm])
                        if occ:
                            # Just emit first note as a fill
                            output[gm].append({
                                'pitch': max(0, min(127, occ['first_pitch'])),
                                'onset': onset_ticks,
                                'duration': 200,
                                'velocity': 65,
                            })
                beat += 1

        return output

    def generate_to_midi(
        self,
        n_beats: int,
        instruments: List[int],
        output_path: str,
        bpm: int = 120
    ):
        """Generate and save to MIDI."""
        print(f"\nGenerating {n_beats} beats via occurrence sampling")
        print(f"  Instruments: {instruments}")

        notes = self.generate(n_beats, instruments)

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

        all_onsets = set.union(*onsets_by_gm.values()) if onsets_by_gm else set()
        total = len(all_onsets)

        if total > 0:
            print(f"\nAlignment: {len(shared)} / {total} onsets shared ({100*len(shared)/total:.1f}%)")

            # Check harmony at shared onsets
            print("\nFirst 8 shared onsets:")
            pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            for onset in sorted(shared)[:8]:
                pitches = []
                for gm in instruments:
                    for n in notes[gm]:
                        if n['onset'] == onset:
                            pitches.append(n['pitch'])
                            break
                pcs = sorted(set(p % 12 for p in pitches))
                pc_str = ", ".join(pc_names[pc] for pc in pcs)
                interval_str = ""
                if len(pitches) >= 2:
                    intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
                    interval_str = f" intervals={intervals}"
                print(f"  {onset}: pitches={pitches}, PCs={{{pc_str}}}{interval_str}")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python occurrence_sampler.py <checkpoint.npz> [-o output.mid] [--beats N] [--instruments GM1 GM2 ...]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_path = '/tmp/occurrence_output.mid'
    n_beats = 64
    instruments = [65, 66, 67]

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '-o' and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--beats' and i + 1 < len(sys.argv):
            n_beats = int(sys.argv[i + 1])
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
    sampler = OccurrenceSampler(patterns)
    sampler.generate_to_midi(n_beats, instruments, output_path)


if __name__ == '__main__':
    main()
