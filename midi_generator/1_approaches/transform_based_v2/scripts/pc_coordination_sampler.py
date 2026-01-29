#!/usr/bin/env python3
"""
Pitch Class Coordination Sampler

Key insight: Full (pattern, pitch) coordination has 66.8% uniqueness (not compressible).
But PITCH CLASS coordination has only 2.3% uniqueness (highly compressible!).

Strategy:
1. Sample a pitch class set for each beat (from corpus distribution)
2. Assign pitch classes to instruments
3. Find patterns that can play at those pitch classes
4. Generate with enforced harmonic consistency

This inverts the current approach:
- Current: Pick patterns → hope pitches align
- New: Pick pitch classes → find patterns that fit
"""

import orjson
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


class PCCoordinationSampler:
    """Generate with pitch class coordination."""

    def __init__(self, patterns: dict, verbose: bool = True):
        self.patterns = patterns
        self.verbose = verbose

        # Build indices
        self._build_pc_index()
        self._build_coordination_distribution()

    def _build_pc_index(self):
        """Build index: (gm, pitch_class) -> [pattern_ids that can play this PC]"""
        if self.verbose:
            print("Building pitch class index...")

        # pattern -> list of pitch classes it can play at
        self.pattern_pcs = {}
        # (gm, pc) -> [patterns]
        self.gm_pc_patterns = defaultdict(list)

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            occs = p.get('occurrences', [])

            if not occs:
                continue

            # Collect all pitch classes this pattern plays at
            pcs = set()
            for occ in occs[:50]:  # Sample for efficiency
                fp = occ.get('first_pitch', 60)
                pcs.add(fp % 12)

            self.pattern_pcs[pid] = list(pcs)

            for pc in pcs:
                self.gm_pc_patterns[(gm, pc)].append(pid)

        if self.verbose:
            print(f"  Indexed {len(self.pattern_pcs)} patterns")
            print(f"  (GM, PC) combinations: {len(self.gm_pc_patterns)}")

    def _build_coordination_distribution(self):
        """Build distribution of pitch class combinations from corpus."""
        if self.verbose:
            print("Building coordination distribution...")

        # Group occurrences by (piece, beat)
        beat_pcs = defaultdict(list)

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            for occ in p.get('occurrences', []):
                piece = occ.get('piece_id', 'unknown')
                onset = occ.get('onset_time', 0)
                pitch = occ.get('first_pitch', 60)

                beat = onset // 480
                beat_pcs[(piece, beat)].append((gm, pitch % 12))

        # Build PC combination distribution
        self.pc_combo_counts = Counter()
        self.gm_set_pc_combos = defaultdict(Counter)  # instrument_set -> pc_combo -> count

        # NEW: Track which GM gets which PC for each instrument set + PC combo
        # gm_pc_assignments[inst_set][(pc_combo, (gm1_pc, gm2_pc, ...))] = count
        self.gm_pc_assignments = defaultdict(Counter)

        for (piece, beat), items in beat_pcs.items():
            gms = tuple(sorted(set(gm for gm, _ in items)))
            if len(gms) < 2:
                continue

            pcs = tuple(sorted(set(pc for _, pc in items)))
            self.pc_combo_counts[pcs] += 1
            self.gm_set_pc_combos[gms][pcs] += 1

            # Track actual GM->PC assignments from corpus
            # Build a dict of gm -> pc for this event
            gm_to_pc = {}
            for gm, pc in items:
                if gm not in gm_to_pc:
                    gm_to_pc[gm] = pc

            # Create sorted assignment tuple matching sorted gms
            assignments = tuple(gm_to_pc[gm] for gm in gms)
            self.gm_pc_assignments[gms][(pcs, assignments)] += 1

        if self.verbose:
            print(f"  Unique PC combinations: {len(self.pc_combo_counts)}")
            print(f"  Instrument set combinations: {len(self.gm_set_pc_combos)}")
            print(f"  Learned GM->PC assignments: {sum(len(v) for v in self.gm_pc_assignments.values())}")

    def sample_pc_combo(
        self,
        instruments: Tuple[int, ...],
        previous_pcs: Optional[Tuple[int, ...]] = None
    ) -> Tuple[int, ...]:
        """Sample a pitch class combination for given instruments.

        Args:
            instruments: Tuple of GM programs
            previous_pcs: Previous pitch classes for continuity

        Returns:
            Tuple of pitch classes to play
        """
        gm_key = tuple(sorted(instruments))

        # Try instrument-specific distribution first
        if gm_key in self.gm_set_pc_combos:
            combos = list(self.gm_set_pc_combos[gm_key].keys())
            counts = list(self.gm_set_pc_combos[gm_key].values())
        else:
            # Fallback to global distribution
            combos = list(self.pc_combo_counts.keys())
            counts = list(self.pc_combo_counts.values())

        if not combos:
            # Ultimate fallback: random consonant combination
            root = random.randint(0, 11)
            return (root, (root + 4) % 12, (root + 7) % 12)

        # Bias toward continuity with previous
        weights = list(counts)
        if previous_pcs is not None:
            for i, combo in enumerate(combos):
                overlap = len(set(combo) & set(previous_pcs))
                weights[i] *= (1 + 0.5 * overlap)

        # Sample
        total = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(combos, weights=weights)[0]

    def assign_pcs_to_instruments(
        self,
        pc_combo: Tuple[int, ...],
        instruments: List[int]
    ) -> Dict[int, int]:
        """Assign pitch classes to instruments.

        Strategy: Use corpus statistics for which PC each instrument typically plays
        when multiple instruments are together.
        """
        # If only one PC, everyone gets it (unison)
        if len(pc_combo) == 1:
            return {gm: pc_combo[0] for gm in instruments}

        # Use learned assignments if available
        inst_key = tuple(sorted(instruments))
        if inst_key in self.gm_pc_assignments:
            # Find best matching PC combo from corpus
            # assignments in gm_pc_assignments are stored as: (pc_combo, (gm1_pc, gm2_pc, ...))
            # where gm1, gm2, ... are in sorted(instruments) order
            best_match = None
            best_score = -1
            for (combo, assignments), count in self.gm_pc_assignments[inst_key].items():
                if set(combo) == set(pc_combo):
                    if count > best_score:
                        best_score = count
                        best_match = assignments

            if best_match:
                # best_match is a tuple of PCs in same order as sorted instruments
                sorted_insts = sorted(instruments)
                return dict(zip(sorted_insts, best_match))

        # Fallback: distribute based on typical instrument ranges
        # Bass gets root, others get remaining chord tones
        gm_ranges = {
            32: (28, 55), 33: (28, 55), 43: (28, 55),  # Bass - low
            0: (48, 84),   # Piano - mid
            56: (55, 82), 57: (45, 72), 58: (36, 60),  # Brass
            65: (49, 81), 66: (42, 75), 67: (36, 65),  # Sax
        }

        def get_range_mid(gm):
            r = gm_ranges.get(gm, (48, 84))
            return (r[0] + r[1]) / 2

        sorted_insts = sorted(instruments, key=get_range_mid)
        sorted_pcs = sorted(pc_combo)

        assignments = {}
        bass_gms = {32, 33, 43}

        # Bass instruments get the root (lowest PC)
        for gm in sorted_insts:
            if gm in bass_gms:
                assignments[gm] = sorted_pcs[0]  # Root

        # Other instruments: distribute remaining PCs
        # Try to give each instrument a different PC if possible
        remaining_insts = [gm for gm in sorted_insts if gm not in assignments]

        # If more instruments than PCs, some will share
        # Distribute to minimize dissonant doublings
        if len(sorted_pcs) >= len(remaining_insts):
            # Each gets a unique PC - assign in order by range
            for i, gm in enumerate(remaining_insts):
                # Skip root if bass already took it
                pc_idx = (i + 1) if bass_gms & set(instruments) else i
                pc_idx = min(pc_idx, len(sorted_pcs) - 1)
                assignments[gm] = sorted_pcs[pc_idx]
        else:
            # More instruments than PCs - need to share
            # Prefer doubling at unison (same PC) over creating dissonant intervals
            for i, gm in enumerate(remaining_insts):
                # Cycle through PCs
                pc_idx = i % len(sorted_pcs)
                assignments[gm] = sorted_pcs[pc_idx]

        return assignments

    def find_pattern_for_gm_pc(self, gm: int, pc: int) -> Tuple[str, int]:
        """Find a pattern that can play at given (GM, PC).

        Returns:
            (pattern_id, actual_pitch)
        """
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

        # Get actual pitch (PC + appropriate octave)
        pattern = self.patterns.get(pid, {})
        occs = pattern.get('occurrences', [])
        if occs:
            # Find an occurrence that matches this PC
            for occ in occs:
                fp = occ.get('first_pitch', 60)
                if fp % 12 == pc:
                    return (pid, fp)

            # If no exact match, transpose
            fp = occs[0].get('first_pitch', 60)
            octave = fp // 12
            return (pid, octave * 12 + pc)
        else:
            return (pid, 60 + pc)

    def generate(
        self,
        length: int,
        instruments: List[int],
    ) -> Dict[int, List[Tuple[str, int]]]:
        """Generate with pitch class coordination.

        Args:
            length: Number of coordination events
            instruments: GM programs to generate

        Returns:
            {gm: [(pattern_id, pitch), ...]}
        """
        output = {gm: [] for gm in instruments}
        inst_tuple = tuple(sorted(instruments))
        previous_pcs = None

        for t in range(length):
            # 1. Sample a pitch class combination
            pc_combo = self.sample_pc_combo(inst_tuple, previous_pcs)

            # 2. Assign PCs to instruments
            gm_to_pc = self.assign_pcs_to_instruments(pc_combo, instruments)

            # 3. Find patterns for each (gm, pc)
            for gm in instruments:
                pc = gm_to_pc[gm]
                pid, pitch = self.find_pattern_for_gm_pc(gm, pc)
                output[gm].append((pid, pitch))

            # Track for continuity
            previous_pcs = pc_combo

        return output

    def generate_to_midi(
        self,
        length: int,
        instruments: List[int],
        output_path: str,
        bpm: int = 120
    ):
        """Generate and save to MIDI file."""
        print(f"\nGenerating {length} coordination events for {instruments}...")

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
                velocity_ratios = pattern.get('velocity_ratios', [])
                duration_ratios = pattern.get('duration_ratios', [])

                occs = pattern.get('occurrences', [])
                first_occ = occs[0] if occs else {}
                base_ioi = first_occ.get('tau_offset', 480)

                pitch = target_pitch

                for i, interval in enumerate([0] + intervals):
                    pitch += interval if i > 0 else 0

                    if i < len(rhythm_ratios):
                        ioi = int(base_ioi * rhythm_ratios[i])
                    else:
                        ioi = base_ioi

                    if i < len(velocity_ratios):
                        velocity = int(80 * velocity_ratios[i])
                    else:
                        velocity = 80

                    if i < len(duration_ratios):
                        duration = int(ioi * duration_ratios[i] * 0.9)
                    else:
                        duration = int(ioi * 0.9)

                    notes.append({
                        'pitch': max(0, min(127, pitch)),
                        'onset': current_time,
                        'duration': max(1, duration),
                        'velocity': max(1, min(127, velocity)),
                    })

                    current_time += ioi

            note_output[gm] = notes

        # Convert to MIDI
        mid = MidiFile(ticks_per_beat=480, type=1)

        # Tempo track
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

        # Analyze output harmony
        self._analyze_output_harmony(pattern_output)

        return pattern_output

    def _analyze_output_harmony(self, output: Dict[int, List[Tuple[str, int]]]):
        """Analyze the harmonic content of generated output."""
        print("\n=== OUTPUT HARMONY ANALYSIS ===")

        pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        interval_names = ["P1", "m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7"]

        # Get length
        length = len(list(output.values())[0])

        # Analyze each time slice
        interval_counts = Counter()
        slice_pcs = []

        print("\nFirst 8 vertical slices:")
        for t in range(min(8, length)):
            pitches = {}
            for gm, pairs in output.items():
                _, pitch = pairs[t]
                pitches[gm] = pitch

            pcs = tuple(sorted(set(p % 12 for p in pitches.values())))
            slice_pcs.append(pcs)

            pc_str = ", ".join(pc_names[pc] for pc in pcs)
            print(f"  t={t}: {{{pc_str}}}")

            # Count intervals
            pc_list = list(pcs)
            for i, pc1 in enumerate(pc_list):
                for pc2 in pc_list[i+1:]:
                    interval = (pc2 - pc1) % 12
                    interval_counts[interval] += 1

        # Overall interval distribution
        print("\nInterval distribution in output:")
        total = sum(interval_counts.values())
        for iv in sorted(range(12), key=lambda x: -interval_counts.get(x, 0))[:6]:
            count = interval_counts.get(iv, 0)
            pct = 100 * count / total if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {interval_names[iv]:3s}: {pct:5.1f}% {bar}")

        # Consonance
        consonant = {0, 3, 4, 5, 7, 8, 9}
        cons = sum(interval_counts.get(iv, 0) for iv in consonant)
        print(f"\nConsonance: {100*cons/total:.1f}%" if total > 0 else "N/A")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pc_coordination_sampler.py <checkpoint.npz> [-o output.mid] [--length N] [--instruments GM1 GM2 ...]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    # Parse args
    output_path = '/tmp/pc_coordinated.mid'
    length = 64
    instruments = [0, 32, 56, 57]

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
    sampler = PCCoordinationSampler(patterns)
    sampler.generate_to_midi(length, instruments, output_path)


if __name__ == '__main__':
    main()
