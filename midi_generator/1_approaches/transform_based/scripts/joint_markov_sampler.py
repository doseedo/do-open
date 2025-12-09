#!/usr/bin/env python3
"""
Joint Markov Sampler - Generation from empirical joint distribution.

Philosophy: Sample directly from the joints that occurred in the corpus.
No prescription, pure discovery.

Hierarchy:
1. PC joints: (gm, pitch_class) combinations - denser, better for transitions
2. Pattern joints: Given PC joint, sample patterns that match

This is philosophically pure: generation is sampling from the codec's
discovered structure, not prescribing rules.
"""

import orjson
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, FrozenSet, Optional
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


class JointMarkovSampler:
    """Generate by sampling from empirical joint distributions."""

    def __init__(self, patterns: dict, verbose: bool = True, max_pattern_length: int = None, max_pattern_beats: float = None):
        self.patterns = patterns
        self.verbose = verbose
        self.max_pattern_length = max_pattern_length
        self.max_pattern_beats = max_pattern_beats

        # Filter patterns by length if specified
        if max_pattern_length:
            original_count = len(self.patterns)
            self.patterns = {
                pid: p for pid, p in self.patterns.items()
                if len(p.get('pitch_classes', [1])) <= max_pattern_length
            }
            if self.verbose:
                print(f"Filtered by note count: {original_count} -> {len(self.patterns)} (max {max_pattern_length} notes)")

        # Filter patterns by duration (beats) if specified
        if max_pattern_beats:
            original_count = len(self.patterns)
            filtered = {}
            for pid, p in self.patterns.items():
                duration_beats = self._estimate_pattern_duration(p)
                if duration_beats <= max_pattern_beats:
                    filtered[pid] = p
            self.patterns = filtered
            if self.verbose:
                print(f"Filtered by duration: {original_count} -> {len(self.patterns)} (max {max_pattern_beats} beats)")

        # Build indices
        self._build_joint_indices()

    def _estimate_pattern_duration(self, p: dict) -> float:
        """Estimate pattern duration in beats from successive ratios."""
        occs = p.get('occurrences', [])
        if not occs:
            return 0
        tau = occs[0].get('tau_offset', 480)
        rhythm_ratios = p.get('rhythm_ratios', [])
        n_notes = len(p.get('pitch_classes', [1]))

        # Successive ratios: IOI[i+1] = IOI[i] * ratio[i]
        total_ioi = tau
        current_ioi = tau
        for r in rhythm_ratios[:n_notes-1]:
            if r > 0:  # Avoid 0 multipliers
                current_ioi = current_ioi * r
                total_ioi += current_ioi

        return total_ioi / 480  # Convert to beats

    def _build_joint_indices(self):
        """Build all indices needed for joint sampling."""
        if self.verbose:
            print("Building joint indices...")

        # (piece, beat) -> [(gm, pattern_id, pitch)]
        beat_data = defaultdict(list)

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            for occ in p.get('occurrences', []):
                piece = occ.get('piece_id', 'unknown')
                onset = occ.get('onset_time', 0)
                pitch = occ.get('first_pitch', 60)
                beat = onset // 480
                beat_data[(piece, beat)].append((gm, pid, pitch))

        # Extract multi-instrument beats only
        self.multi_beats = {}
        for key, items in beat_data.items():
            gms = set(gm for gm, _, _ in items)
            if len(gms) >= 2:
                self.multi_beats[key] = items

        if self.verbose:
            print(f"  Multi-instrument beats: {len(self.multi_beats)}")

        # Build PC joint -> [full joints that have this PC]
        # PC joint: frozenset of (gm, pc)
        # Full joint: list of (gm, pattern_id, pitch)
        self.pc_to_full_joints = defaultdict(list)

        for key, items in self.multi_beats.items():
            pc_joint = frozenset((gm, pitch % 12) for gm, _, pitch in items)
            self.pc_to_full_joints[pc_joint].append(items)

        if self.verbose:
            print(f"  Unique PC joints: {len(self.pc_to_full_joints)}")

        # Build PC joint transitions
        # Group by piece, sort by beat
        piece_sequences = defaultdict(list)
        for (piece, beat), items in self.multi_beats.items():
            pc_joint = frozenset((gm, pitch % 12) for gm, _, pitch in items)
            piece_sequences[piece].append((beat, pc_joint))

        # Sort and extract transitions
        self.pc_transitions = defaultdict(Counter)
        self.pc_initial = Counter()

        for piece, seq in piece_sequences.items():
            seq.sort(key=lambda x: x[0])
            if seq:
                _, first_joint = seq[0]
                self.pc_initial[first_joint] += 1

            for i in range(len(seq) - 1):
                _, j1 = seq[i]
                _, j2 = seq[i + 1]
                self.pc_transitions[j1][j2] += 1

        if self.verbose:
            print(f"  PC transitions: {sum(len(v) for v in self.pc_transitions.values())}")
            print(f"  Initial states: {len(self.pc_initial)}")

        # Build (gm, pc) -> [pattern_ids]
        self.gm_pc_patterns = defaultdict(list)
        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            for occ in p.get('occurrences', []):
                pc = occ.get('first_pitch', 60) % 12
                self.gm_pc_patterns[(gm, pc)].append(pid)

        # Deduplicate
        for key in self.gm_pc_patterns:
            self.gm_pc_patterns[key] = list(set(self.gm_pc_patterns[key]))

        if self.verbose:
            print(f"  (GM, PC) -> pattern mappings: {len(self.gm_pc_patterns)}")

    def sample_initial_pc_joint(self, instruments: List[int]) -> FrozenSet[Tuple[int, int]]:
        """Sample an initial PC joint containing the requested instruments."""
        # Filter to joints that have all requested instruments
        valid = []
        weights = []
        for joint, count in self.pc_initial.items():
            joint_gms = set(gm for gm, _ in joint)
            if all(gm in joint_gms for gm in instruments):
                valid.append(joint)
                weights.append(count)

        if not valid:
            # Fallback: any joint with these instruments
            for joint in self.pc_to_full_joints.keys():
                joint_gms = set(gm for gm, _ in joint)
                if all(gm in joint_gms for gm in instruments):
                    valid.append(joint)
                    weights.append(1)

        if not valid:
            # Ultimate fallback: create a consonant joint
            root = random.randint(0, 11)
            joint = frozenset((gm, (root + i * 4) % 12) for i, gm in enumerate(instruments))
            return joint

        total = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(valid, weights=weights)[0]

    def sample_next_pc_joint(
        self,
        current: FrozenSet[Tuple[int, int]],
        instruments: List[int]
    ) -> FrozenSet[Tuple[int, int]]:
        """Sample next PC joint given current, filtered to include instruments."""
        if current not in self.pc_transitions:
            # No transitions from this state - sample from initial
            return self.sample_initial_pc_joint(instruments)

        # Get transitions, filter to those with our instruments
        transitions = self.pc_transitions[current]
        valid = []
        weights = []

        for next_joint, count in transitions.items():
            joint_gms = set(gm for gm, _ in next_joint)
            if all(gm in joint_gms for gm in instruments):
                valid.append(next_joint)
                weights.append(count)

        if not valid:
            # No valid transitions - sample from initial
            return self.sample_initial_pc_joint(instruments)

        total = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(valid, weights=weights)[0]

    def sample_patterns_for_pc_joint(
        self,
        pc_joint: FrozenSet[Tuple[int, int]],
        instruments: List[int]
    ) -> Dict[int, Tuple[str, int]]:
        """Given a PC joint, sample patterns for each instrument.

        Returns:
            {gm: (pattern_id, pitch)}
        """
        # First try: sample from actual full joints that had this PC
        if pc_joint in self.pc_to_full_joints:
            full_joints = self.pc_to_full_joints[pc_joint]
            if full_joints:
                # Pick a random full joint
                chosen = random.choice(full_joints)

                # Extract (gm, pattern, pitch) for our instruments
                result = {}
                for gm, pid, pitch in chosen:
                    if gm in instruments:
                        result[gm] = (pid, pitch)

                # Fill any missing instruments
                for gm in instruments:
                    if gm not in result:
                        # Find PC for this gm in the joint
                        for jgm, pc in pc_joint:
                            if jgm == gm:
                                pid, pitch = self._sample_pattern_for_gm_pc(gm, pc)
                                result[gm] = (pid, pitch)
                                break

                return result

        # Fallback: sample patterns independently based on PC
        result = {}
        for gm, pc in pc_joint:
            if gm in instruments:
                pid, pitch = self._sample_pattern_for_gm_pc(gm, pc)
                result[gm] = (pid, pitch)

        return result

    def _sample_pattern_for_gm_pc(self, gm: int, pc: int) -> Tuple[str, int]:
        """Sample a pattern that can play at (gm, pc)."""
        key = (gm, pc)
        if key in self.gm_pc_patterns and self.gm_pc_patterns[key]:
            pid = random.choice(self.gm_pc_patterns[key])
        else:
            # Fallback: any pattern for this GM
            gm_patterns = [p for p, d in self.patterns.items()
                         if d.get('gm_program') == gm]
            if gm_patterns:
                pid = random.choice(gm_patterns)
            else:
                pid = list(self.patterns.keys())[0]

        # Get actual pitch
        pattern = self.patterns.get(pid, {})
        occs = pattern.get('occurrences', [])
        if occs:
            for occ in occs:
                fp = occ.get('first_pitch', 60)
                if fp % 12 == pc:
                    return (pid, fp)
            # Transpose if needed
            fp = occs[0].get('first_pitch', 60)
            octave = fp // 12
            return (pid, octave * 12 + pc)
        else:
            return (pid, 60 + pc)

    def generate(
        self,
        length: int,
        instruments: List[int]
    ) -> Dict[int, List[Tuple[str, int]]]:
        """Generate using joint Markov model.

        Args:
            length: Number of coordination events
            instruments: GM programs to generate

        Returns:
            {gm: [(pattern_id, pitch), ...]}
        """
        output = {gm: [] for gm in instruments}

        # Sample initial state
        current_pc = self.sample_initial_pc_joint(instruments)

        for t in range(length):
            # Sample patterns for current PC joint
            patterns = self.sample_patterns_for_pc_joint(current_pc, instruments)

            # Add to output
            for gm in instruments:
                if gm in patterns:
                    output[gm].append(patterns[gm])
                else:
                    # Shouldn't happen but fallback
                    output[gm].append((list(self.patterns.keys())[0], 60))

            # Transition to next PC joint
            current_pc = self.sample_next_pc_joint(current_pc, instruments)

        return output

    def generate_to_midi(
        self,
        length: int,
        instruments: List[int],
        output_path: str,
        bpm: int = 120
    ):
        """Generate and save to MIDI."""
        print(f"\nGenerating {length} events for {instruments}...")

        # Generate pattern/pitch pairs
        pattern_output = self.generate(length, instruments)

        # Expand to notes
        note_output = {}
        for gm, pairs in pattern_output.items():
            notes = []
            current_time = 0

            for pid, target_pitch in pairs:
                pattern = self.patterns.get(pid, {})
                intervals = pattern.get('pitch_intervals', [0])
                rhythm_ratios = pattern.get('rhythm_ratios', [])
                duration_ratios = pattern.get('duration_ratios', [])

                occs = pattern.get('occurrences', [])
                first_occ = occs[0] if occs else {}
                base_ioi = first_occ.get('tau_offset', 480)

                pitch = target_pitch

                # Build IOIs from successive ratios: IOI[i+1] = IOI[i] * ratio[i]
                # First IOI is tau (base_ioi), subsequent IOIs multiply by ratio
                n_notes = len(intervals) + 1
                iois = [base_ioi]  # First note uses tau
                for r in rhythm_ratios[:n_notes - 1]:
                    iois.append(int(iois[-1] * r))

                for i, interval in enumerate([0] + intervals):
                    pitch += interval if i > 0 else 0

                    ioi = iois[i] if i < len(iois) else base_ioi

                    if i < len(duration_ratios):
                        duration = int(ioi * duration_ratios[i] * 0.9)
                    else:
                        duration = int(ioi * 0.9)

                    notes.append({
                        'pitch': max(0, min(127, pitch)),
                        'onset': current_time,
                        'duration': max(1, duration),
                        'velocity': 80,
                    })

                    current_time += ioi

            note_output[gm] = notes

        # Convert to MIDI
        mid = MidiFile(ticks_per_beat=480, type=1)

        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
        tempo_track.append(MetaMessage('end_of_track', time=0))

        channel_map = {}
        next_channel = 0

        for gm, note_list in sorted(note_output.items()):
            if not note_list:
                continue

            if gm not in channel_map:
                channel_map[gm] = next_channel
                next_channel += 1
                if next_channel == 9:
                    next_channel = 10

            channel = channel_map[gm]
            track = MidiTrack()
            mid.tracks.append(track)

            track.append(Message('program_change', program=gm % 128, channel=channel, time=0))

            sorted_notes = sorted(note_list, key=lambda n: n['onset'])

            events = []
            for n in sorted_notes:
                events.append((n['onset'], 'on', n['pitch'], n['velocity']))
                events.append((n['onset'] + n['duration'], 'off', n['pitch'], 0))

            events.sort(key=lambda x: (x[0], x[1] == 'on'))

            last_time = 0
            for event_time, event_type, pitch, vel in events:
                delta = event_time - last_time
                if event_type == 'on':
                    track.append(Message('note_on', note=pitch, velocity=vel, channel=channel, time=delta))
                else:
                    track.append(Message('note_off', note=pitch, velocity=0, channel=channel, time=delta))
                last_time = event_time

            track.append(MetaMessage('end_of_track', time=0))

        mid.save(output_path)

        total_notes = sum(len(n) for n in note_output.values())
        print(f"Saved to: {output_path}")
        print(f"  Tracks: {len(mid.tracks)}")
        print(f"  Notes: {total_notes}")

        # Analyze harmony
        self._analyze_harmony(pattern_output)

        return pattern_output

    def _analyze_harmony(self, output: Dict[int, List[Tuple[str, int]]]):
        """Analyze harmonic content of output."""
        print("\n=== OUTPUT HARMONY ANALYSIS ===")

        pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        interval_names = ["P1", "m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7"]

        length = len(list(output.values())[0])

        interval_counts = Counter()

        print("\nFirst 8 vertical slices:")
        for t in range(min(8, length)):
            pitches = {}
            for gm, pairs in output.items():
                _, pitch = pairs[t]
                pitches[gm] = pitch

            pcs = tuple(sorted(set(p % 12 for p in pitches.values())))
            pc_str = ", ".join(pc_names[pc] for pc in pcs)
            print(f"  t={t}: {{{pc_str}}}")

            # Count intervals
            pc_list = list(pcs)
            for i, pc1 in enumerate(pc_list):
                for pc2 in pc_list[i+1:]:
                    interval = (pc2 - pc1) % 12
                    interval_counts[interval] += 1

        # Full interval distribution
        for t in range(length):
            pcs = set()
            for gm, pairs in output.items():
                _, pitch = pairs[t]
                pcs.add(pitch % 12)

            pc_list = sorted(pcs)
            for i, pc1 in enumerate(pc_list):
                for pc2 in pc_list[i+1:]:
                    interval = (pc2 - pc1) % 12
                    interval_counts[interval] += 1

        print("\nInterval distribution:")
        total = sum(interval_counts.values())
        for iv in sorted(range(12), key=lambda x: -interval_counts.get(x, 0))[:6]:
            count = interval_counts.get(iv, 0)
            pct = 100 * count / total if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {interval_names[iv]:3s}: {pct:5.1f}% {bar}")

        consonant = {0, 3, 4, 5, 7, 8, 9}
        cons = sum(interval_counts.get(iv, 0) for iv in consonant)
        print(f"\nConsonance: {100*cons/total:.1f}%" if total > 0 else "N/A")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python joint_markov_sampler.py <checkpoint.npz> [-o output.mid] [--length N] [--instruments GM1 GM2 ...] [--max-pattern-length N]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    output_path = '/tmp/joint_markov.mid'
    length = 64
    instruments = [0, 32, 56, 57]
    max_pattern_length = None
    max_pattern_beats = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '-o' and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--length' and i + 1 < len(sys.argv):
            length = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--max-pattern-length' and i + 1 < len(sys.argv):
            max_pattern_length = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--max-pattern-beats' and i + 1 < len(sys.argv):
            max_pattern_beats = float(sys.argv[i + 1])
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

    # Create sampler and generate
    sampler = JointMarkovSampler(patterns, max_pattern_length=max_pattern_length, max_pattern_beats=max_pattern_beats)
    sampler.generate_to_midi(length, instruments, output_path)


if __name__ == '__main__':
    main()
