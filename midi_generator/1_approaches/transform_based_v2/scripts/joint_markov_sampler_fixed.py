#!/usr/bin/env python3
"""
Joint Markov Sampler - FIXED VERSION with deduplication.

This version fixes the hierarchical pattern duplication bug:
- At each (piece, onset, gm) location, only keep the LONGEST pattern
- This prevents 2.37x inflation in statistics from nested patterns
- Results in cleaner sampling from true corpus distribution
"""

import orjson
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, FrozenSet, Optional
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


def build_deduplicated_locations(patterns: dict) -> Dict[Tuple, dict]:
    """Build deduplicated location index - only longest pattern at each location.

    Returns: {(piece, onset, gm): {pattern_id, n_notes, first_pitch, tau, pattern}}
    """
    # Group by (piece, onset, gm)
    location_patterns = defaultdict(list)

    for pid, p in patterns.items():
        gm = p.get('gm_program', 0)
        n_notes = len(p.get('pitch_intervals', [])) + 1

        for occ in p.get('occurrences', []):
            piece = occ.get('piece_id', 'unknown')
            onset = occ.get('onset_time', 0)
            pitch = occ.get('first_pitch', 60)
            tau = occ.get('tau_offset', 480)

            location_patterns[(piece, onset, gm)].append({
                'pattern_id': pid,
                'n_notes': n_notes,
                'first_pitch': pitch,
                'tau': tau,
            })

    # Keep only longest at each location
    deduped = {}
    for loc, items in location_patterns.items():
        sorted_items = sorted(items, key=lambda x: -x['n_notes'])
        best = sorted_items[0]
        deduped[loc] = {
            **best,
            'pattern': patterns[best['pattern_id']]
        }

    return deduped


class JointMarkovSamplerFixed:
    """Generate by sampling from empirical joint distributions - DEDUPLICATED."""

    def __init__(self, patterns: dict, verbose: bool = True):
        self.patterns = patterns
        self.verbose = verbose

        # Build deduplicated indices
        self._build_joint_indices()

    def _build_joint_indices(self):
        """Build all indices needed for joint sampling - with deduplication."""
        if self.verbose:
            print("Building deduplicated joint indices...")

        # Build deduplicated location index
        deduped_locations = build_deduplicated_locations(self.patterns)

        if self.verbose:
            print(f"  Deduplicated locations: {len(deduped_locations)}")

        # (piece, beat) -> [(gm, pattern_id, pitch, tau)]
        beat_data = defaultdict(list)

        for (piece, onset, gm), info in deduped_locations.items():
            beat = onset // 480
            beat_data[(piece, beat)].append({
                'gm': gm,
                'pattern_id': info['pattern_id'],
                'pitch': info['first_pitch'],
                'tau': info['tau'],
                'pattern': info['pattern']
            })

        # Extract multi-instrument beats only
        self.multi_beats = {}
        for key, items in beat_data.items():
            gms = set(item['gm'] for item in items)
            if len(gms) >= 2:
                self.multi_beats[key] = items

        if self.verbose:
            print(f"  Multi-instrument beats: {len(self.multi_beats)}")

        # Build PC joint -> [full joints that have this PC]
        self.pc_to_full_joints = defaultdict(list)

        for key, items in self.multi_beats.items():
            pc_joint = frozenset((item['gm'], item['pitch'] % 12) for item in items)
            self.pc_to_full_joints[pc_joint].append(items)

        if self.verbose:
            print(f"  Unique PC joints: {len(self.pc_to_full_joints)}")

        # Build PC joint transitions
        piece_sequences = defaultdict(list)
        for (piece, beat), items in self.multi_beats.items():
            pc_joint = frozenset((item['gm'], item['pitch'] % 12) for item in items)
            piece_sequences[piece].append((beat, pc_joint))

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

        # Build (gm, pc) -> [pattern info]
        self.gm_pc_patterns = defaultdict(list)
        for (piece, onset, gm), info in deduped_locations.items():
            pc = info['first_pitch'] % 12
            self.gm_pc_patterns[(gm, pc)].append({
                'pattern_id': info['pattern_id'],
                'pitch': info['first_pitch'],
                'tau': info['tau'],
                'pattern': info['pattern']
            })

        # Deduplicate by pattern_id (keep one representative for each pattern)
        for key in self.gm_pc_patterns:
            seen_pids = set()
            deduped = []
            for item in self.gm_pc_patterns[key]:
                if item['pattern_id'] not in seen_pids:
                    seen_pids.add(item['pattern_id'])
                    deduped.append(item)
            self.gm_pc_patterns[key] = deduped

        if self.verbose:
            print(f"  (GM, PC) -> pattern mappings: {len(self.gm_pc_patterns)}")

    def sample_initial_pc_joint(self, instruments: List[int]) -> FrozenSet[Tuple[int, int]]:
        """Sample an initial PC joint containing the requested instruments."""
        valid = []
        weights = []
        for joint, count in self.pc_initial.items():
            joint_gms = set(gm for gm, _ in joint)
            if all(gm in joint_gms for gm in instruments):
                valid.append(joint)
                weights.append(count)

        if not valid:
            for joint in self.pc_to_full_joints.keys():
                joint_gms = set(gm for gm, _ in joint)
                if all(gm in joint_gms for gm in instruments):
                    valid.append(joint)
                    weights.append(1)

        if not valid:
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
        """Sample next PC joint given current."""
        if current not in self.pc_transitions:
            return self.sample_initial_pc_joint(instruments)

        transitions = self.pc_transitions[current]
        valid = []
        weights = []

        for next_joint, count in transitions.items():
            joint_gms = set(gm for gm, _ in next_joint)
            if all(gm in joint_gms for gm in instruments):
                valid.append(next_joint)
                weights.append(count)

        if not valid:
            return self.sample_initial_pc_joint(instruments)

        total = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(valid, weights=weights)[0]

    def sample_patterns_for_pc_joint(
        self,
        pc_joint: FrozenSet[Tuple[int, int]],
        instruments: List[int]
    ) -> Dict[int, dict]:
        """Given a PC joint, sample patterns for each instrument.

        Returns: {gm: {pattern_id, pitch, tau, pattern}}
        """
        # Try to get from actual corpus joints
        if pc_joint in self.pc_to_full_joints:
            full_joints = self.pc_to_full_joints[pc_joint]
            if full_joints:
                chosen = random.choice(full_joints)
                result = {}
                for item in chosen:
                    if item['gm'] in instruments:
                        result[item['gm']] = item

                # Fill missing instruments
                for gm in instruments:
                    if gm not in result:
                        for jgm, pc in pc_joint:
                            if jgm == gm:
                                result[gm] = self._sample_pattern_for_gm_pc(gm, pc)
                                break

                return result

        # Fallback: sample independently
        result = {}
        for gm, pc in pc_joint:
            if gm in instruments:
                result[gm] = self._sample_pattern_for_gm_pc(gm, pc)

        return result

    def _sample_pattern_for_gm_pc(self, gm: int, pc: int) -> dict:
        """Sample a pattern for (gm, pc)."""
        key = (gm, pc)
        if key in self.gm_pc_patterns and self.gm_pc_patterns[key]:
            return random.choice(self.gm_pc_patterns[key])

        # Fallback: any pattern for this GM
        gm_patterns = []
        for (g, _), items in self.gm_pc_patterns.items():
            if g == gm:
                gm_patterns.extend(items)

        if gm_patterns:
            return random.choice(gm_patterns)

        # Ultimate fallback
        return {
            'pattern_id': 'fallback',
            'pitch': 60 + pc,
            'tau': 480,
            'pattern': {'pitch_intervals': [], 'rhythm_ratios': [1.0], 'duration_ratios': [0.9]}
        }

    def generate(
        self,
        length: int,
        instruments: List[int]
    ) -> Dict[int, List[dict]]:
        """Generate using joint Markov model.

        Returns: {gm: [{pattern_info}, ...]}
        """
        output = {gm: [] for gm in instruments}

        current_pc = self.sample_initial_pc_joint(instruments)

        for t in range(length):
            patterns = self.sample_patterns_for_pc_joint(current_pc, instruments)

            for gm in instruments:
                if gm in patterns:
                    output[gm].append(patterns[gm])
                else:
                    # Fallback
                    output[gm].append(self._sample_pattern_for_gm_pc(gm, 0))

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

        pattern_output = self.generate(length, instruments)

        # Expand to notes
        note_output = {}
        for gm, items in pattern_output.items():
            notes = []
            current_time = 0

            for item in items:
                p = item.get('pattern', {})
                target_pitch = item['pitch']
                tau = item.get('tau', 480)

                intervals = p.get('pitch_intervals', [0])
                rhythm_ratios = p.get('rhythm_ratios', [])
                duration_ratios = p.get('duration_ratios', [0.9])

                pitch = target_pitch

                # Build IOIs from successive ratios: IOI[i+1] = IOI[i] * ratio[i]
                n_notes = len(intervals) + 1
                iois = [tau]  # First note uses tau
                for r in rhythm_ratios[:n_notes - 1]:
                    iois.append(int(iois[-1] * r))

                for i, interval in enumerate([0] + list(intervals)):
                    pitch += interval if i > 0 else 0

                    ioi = iois[i] if i < len(iois) else tau

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

        gm_names = {
            32: "Acoustic Bass",
            65: "Alto Sax",
            66: "Tenor Sax",
            67: "Baritone Sax"
        }

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

            name = gm_names.get(gm, f"GM {gm}")
            track.append(MetaMessage('track_name', name=name, time=0))
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
        print(f"  Tracks: {len(mid.tracks) - 1}")
        print(f"  Notes: {total_notes}")

        self._analyze_harmony(pattern_output)

        return pattern_output

    def _analyze_harmony(self, output: Dict[int, List[dict]]):
        """Analyze harmonic content of output."""
        print("\n=== OUTPUT HARMONY ANALYSIS ===")

        pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        interval_names = ["P1", "m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7"]

        length = len(list(output.values())[0])

        interval_counts = Counter()

        print("\nFirst 8 vertical slices:")
        for t in range(min(8, length)):
            pitches = {}
            for gm, items in output.items():
                pitches[gm] = items[t]['pitch']

            pcs = tuple(sorted(set(p % 12 for p in pitches.values())))
            pc_str = ", ".join(pc_names[pc] for pc in pcs)
            print(f"  t={t}: {{{pc_str}}}")

            pc_list = list(pcs)
            for i, pc1 in enumerate(pc_list):
                for pc2 in pc_list[i+1:]:
                    interval = (pc2 - pc1) % 12
                    interval_counts[interval] += 1

        # Full interval distribution
        for t in range(length):
            pcs = set()
            for gm, items in output.items():
                pcs.add(items[t]['pitch'] % 12)

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
        print("Usage: python joint_markov_sampler_fixed.py <checkpoint.npz> [-o output.mid] [--length N] [--instruments GM1 GM2 ...]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    output_path = '/tmp/joint_markov_fixed.mid'
    length = 64
    instruments = [65, 66, 67]  # Sax section by default

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '-o' and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--length' and i + 1 < len(sys.argv):
            length = int(sys.argv[i + 1])
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
    sampler = JointMarkovSamplerFixed(patterns)
    sampler.generate_to_midi(length, instruments, output_path)


if __name__ == '__main__':
    main()
