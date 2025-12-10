#!/usr/bin/env python3
"""
Level-Based Sampler - Multi-resolution generation from hierarchical patterns.

Three purposes from ONE checkpoint:
1. COMPRESSION: Show MDL patterns at max level
2. GENERATION: Sample at low levels (note/bigram) with co-occurrence
3. EDITING: Swap patterns at any level (genome interface)

The hierarchy is the white-box: each level is interpretable and editable.
"""

import orjson
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

N_TERMINALS = 128  # IDs 0-128 are terminals (rhythm × velocity)


def decode_terminal(terminal_id: int) -> Tuple[int, int]:
    """Decode terminal to (rhythm_bucket, velocity_bucket)."""
    rhythm_bucket = terminal_id // 8
    velocity_bucket = terminal_id % 8
    return rhythm_bucket, velocity_bucket


def get_ioi_from_rhythm_bucket(bucket: int) -> int:
    """Convert rhythm bucket to IOI in ticks."""
    # Bucket boundaries (approximate, from corpus)
    bucket_to_ioi = {
        0: 60, 1: 120, 2: 180, 3: 240,
        4: 360, 5: 480, 6: 720, 7: 960,
        8: 480, 9: 480, 10: 480, 11: 480,
        12: 480, 13: 480, 14: 480, 15: 480,
    }
    return bucket_to_ioi.get(bucket, 480)


def get_velocity_from_bucket(bucket: int) -> int:
    """Convert velocity bucket to MIDI velocity."""
    # Linear mapping
    return 40 + bucket * 12  # 40, 52, 64, 76, 88, 100, 112, 124


class LevelSampler:
    """Multi-resolution generation using hierarchy levels."""

    def __init__(self, patterns: dict, verbose: bool = True):
        self.patterns = patterns
        self.verbose = verbose

        # Cache levels
        self._level_cache = {}
        self._build_indices()

    def get_level(self, pid) -> int:
        """Get hierarchy level. Terminals are level 0."""
        if pid in self._level_cache:
            return self._level_cache[pid]

        # Check if terminal
        if isinstance(pid, str) and pid.startswith('GM'):
            local_id = int(pid.split('_')[1])
            if local_id <= N_TERMINALS:
                self._level_cache[pid] = 0
                return 0

        if pid not in self.patterns:
            self._level_cache[pid] = 0
            return 0

        p = self.patterns[pid]
        if not p.get('is_hierarchical', False):
            self._level_cache[pid] = 1
            return 1

        left = p.get('left_child', -1)
        right = p.get('right_child', -1)

        left_level = self.get_level(left) if left != -1 else 0
        right_level = self.get_level(right) if right != -1 else 0

        level = max(left_level, right_level) + 1
        self._level_cache[pid] = level
        return level

    def decompose_to_level(self, pid, target_level: int) -> List[dict]:
        """Decompose pattern to sub-patterns at target_level.

        Returns list of dicts with pattern info for rendering.
        """
        current_level = self.get_level(pid)

        if current_level <= target_level:
            # At or below target - return this pattern
            if pid in self.patterns:
                p = self.patterns[pid]
                return [{
                    'pid': pid,
                    'intervals': p.get('pitch_intervals', []),
                    'rhythm_ratios': p.get('rhythm_ratios', []),
                    'rhythm_bucket': p.get('rhythm_bucket', 8),
                    'velocity_bucket': p.get('velocity_bucket', 4),
                    'n_notes': len(p.get('pitch_classes', [1])),
                    'level': current_level,
                }]
            else:
                # Terminal
                if isinstance(pid, str) and pid.startswith('GM'):
                    local_id = int(pid.split('_')[1])
                    if local_id <= N_TERMINALS:
                        rhythm_bucket, velocity_bucket = decode_terminal(local_id)
                        return [{
                            'pid': pid,
                            'intervals': [],
                            'rhythm_ratios': [],
                            'rhythm_bucket': rhythm_bucket,
                            'velocity_bucket': velocity_bucket,
                            'n_notes': 1,
                            'level': 0,
                            'is_terminal': True,
                        }]
                return []

        # Above target - decompose
        if pid not in self.patterns:
            return []

        p = self.patterns[pid]
        left = p.get('left_child', -1)
        right = p.get('right_child', -1)

        result = []
        if left != -1:
            result.extend(self.decompose_to_level(left, target_level))
        if right != -1:
            result.extend(self.decompose_to_level(right, target_level))

        return result

    def _build_indices(self):
        """Build indices for generation."""
        if self.verbose:
            print("Building level-based indices...")

        # Patterns by instrument and level
        self.patterns_by_gm_level = defaultdict(lambda: defaultdict(list))

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            level = self.get_level(pid)
            self.patterns_by_gm_level[gm][level].append(pid)

        # Co-occurrence for pitch coordination
        beat_pitches = defaultdict(lambda: defaultdict(list))
        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            for occ in p.get('occurrences', []):
                piece = occ.get('piece_id', 'unknown')
                onset = occ.get('onset_time', 0)
                pitch = occ.get('first_pitch', 60)
                beat = onset // 480
                beat_pitches[(piece, beat)][gm].append(pitch)

        # Build PC joints and transitions
        self.pc_joints = Counter()
        self.pc_transitions = defaultdict(Counter)

        piece_beats = defaultdict(list)
        for (piece, beat), gm_pitches in beat_pitches.items():
            if len(gm_pitches) >= 2:
                pc_joint = frozenset(
                    (gm, p % 12)
                    for gm, pitches in gm_pitches.items()
                    for p in pitches
                )
                piece_beats[piece].append((beat, pc_joint))

        for piece, beats in piece_beats.items():
            beats.sort(key=lambda x: x[0])
            for i, (beat, pc_joint) in enumerate(beats):
                self.pc_joints[pc_joint] += 1
                if i > 0:
                    prev_joint = beats[i-1][1]
                    self.pc_transitions[prev_joint][pc_joint] += 1

        # Pitch distribution per (gm, pc)
        self.pitch_dist = defaultdict(Counter)
        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            for occ in p.get('occurrences', []):
                pitch = occ.get('first_pitch', 60)
                pc = pitch % 12
                self.pitch_dist[(gm, pc)][pitch] += 1

        if self.verbose:
            print(f"  Patterns by level: {sum(len(self.patterns_by_gm_level[gm]) for gm in self.patterns_by_gm_level)} gm-level pairs")
            print(f"  PC joints: {len(self.pc_joints)}")
            print(f"  Pitch distributions: {len(self.pitch_dist)}")

    def sample_pattern_at_level(self, gm: int, level: int) -> str:
        """Sample a pattern for instrument at specified level."""
        candidates = self.patterns_by_gm_level[gm].get(level, [])

        if not candidates:
            # Try adjacent levels
            for delta in [1, -1, 2, -2]:
                candidates = self.patterns_by_gm_level[gm].get(level + delta, [])
                if candidates:
                    break

        if not candidates:
            # Any pattern for this GM
            for lvl_patterns in self.patterns_by_gm_level[gm].values():
                candidates.extend(lvl_patterns)

        if not candidates:
            return None

        # Weight by count
        weights = [self.patterns[pid].get('count', 1) for pid in candidates]
        total = sum(weights)
        weights = [w / total for w in weights]

        return random.choices(candidates, weights=weights)[0]

    def sample_pc_joint(
        self,
        instruments: List[int],
        previous=None
    ):
        """Sample pitch-class joint for vertical coordination."""
        if previous and previous in self.pc_transitions:
            transitions = self.pc_transitions[previous]
            valid = [(j, c) for j, c in transitions.items()
                    if all(gm in {g for g, _ in j} for gm in instruments)]
            if valid:
                joints, counts = zip(*valid)
                total = sum(counts)
                weights = [c / total for c in counts]
                return random.choices(joints, weights=weights)[0]

        # From marginal
        valid = [(j, c) for j, c in self.pc_joints.items()
                if all(gm in {g for g, _ in j} for gm in instruments)]
        if valid:
            joints, counts = zip(*valid)
            total = sum(counts)
            weights = [c / total for c in counts]
            return random.choices(joints, weights=weights)[0]

        # Fallback: consonant
        root = random.randint(0, 11)
        return frozenset((gm, (root + i * 4) % 12) for i, gm in enumerate(instruments))

    def sample_pitch(self, gm: int, pc: int) -> int:
        """Sample absolute pitch given instrument and pitch class."""
        key = (gm, pc)
        if key in self.pitch_dist:
            dist = self.pitch_dist[key]
            pitches = list(dist.keys())
            weights = list(dist.values())
            total = sum(weights)
            weights = [w / total for w in weights]
            return random.choices(pitches, weights=weights)[0]

        # Fallback by instrument range
        if gm in [32, 33, 34, 35, 36, 37, 38, 39]:  # Bass
            return 36 + pc
        elif gm in [56, 57, 58, 59, 60, 61, 62, 63]:  # Brass
            return 60 + pc
        elif gm in [64, 65, 66, 67, 68, 69, 70, 71]:  # Reeds
            return 60 + pc
        else:
            return 60 + pc

    def generate_at_level(
        self,
        length: int,
        instruments: List[int],
        level: int = 1
    ) -> Dict[int, List[dict]]:
        """Generate at specified hierarchy level.

        Lower levels = more control, finer granularity
        Higher levels = more structure, coarser chunks

        Args:
            length: Number of events to generate
            instruments: GM programs
            level: Hierarchy level (0=terminal, 1=bigram, 2+=motifs)

        Returns:
            {gm: [{'pattern': ..., 'pitch': ..., 'onset': ...}, ...]}
        """
        output = {gm: [] for gm in instruments}
        current_time = {gm: 0 for gm in instruments}
        previous_pc = None

        for t in range(length):
            # Sample vertical coordination
            pc_joint = self.sample_pc_joint(instruments, previous_pc)

            for gm in instruments:
                # Get target PC from joint
                target_pc = None
                for jgm, pc in pc_joint:
                    if jgm == gm:
                        target_pc = pc
                        break
                if target_pc is None:
                    target_pc = random.randint(0, 11)

                # Sample pattern at level
                pid = self.sample_pattern_at_level(gm, level)

                if pid and pid in self.patterns:
                    p = self.patterns[pid]

                    # Sample absolute pitch
                    pitch = self.sample_pitch(gm, target_pc)

                    output[gm].append({
                        'pid': pid,
                        'pitch': pitch,
                        'onset': current_time[gm],
                        'intervals': p.get('pitch_intervals', []),
                        'rhythm_ratios': p.get('rhythm_ratios', []),
                        'n_notes': len(p.get('pitch_classes', [1])),
                    })

                    # Advance time
                    duration_beats = self._estimate_duration(p)
                    current_time[gm] += int(duration_beats * 480)
                else:
                    # Fallback single note
                    pitch = self.sample_pitch(gm, target_pc)
                    output[gm].append({
                        'pid': None,
                        'pitch': pitch,
                        'onset': current_time[gm],
                        'intervals': [],
                        'rhythm_ratios': [],
                        'n_notes': 1,
                    })
                    current_time[gm] += 480

            previous_pc = pc_joint

        return output

    def _estimate_duration(self, p: dict) -> float:
        """Estimate pattern duration in beats."""
        occs = p.get('occurrences', [])
        if not occs:
            return 1.0
        tau = occs[0].get('tau_offset', 480)
        ratios = p.get('rhythm_ratios', [])
        n_notes = len(p.get('pitch_classes', [1]))

        total = tau
        current = tau
        for r in ratios[:n_notes-1]:
            if r > 0:
                current = current * r
                total += current
        return max(0.5, total / 480)

    def generate_to_midi(
        self,
        length: int,
        instruments: List[int],
        output_path: str,
        level: int = 1,
        bpm: int = 120
    ):
        """Generate at level and save to MIDI."""
        print(f"\nGenerating {length} events at level {level}")
        print(f"  Level 0 = terminals (single notes)")
        print(f"  Level 1 = bigrams (2 notes)")
        print(f"  Level 2+ = motifs/phrases")

        events = self.generate_at_level(length, instruments, level)

        # Expand to notes
        note_output = {}
        for gm, event_list in events.items():
            notes = []

            for event in event_list:
                base_pitch = event['pitch']
                onset = event['onset']
                intervals = event.get('intervals', [])
                rhythm_ratios = event.get('rhythm_ratios', [])
                n_notes = event.get('n_notes', 1)

                # Estimate tau
                if n_notes > 1:
                    tau = 240  # 8th note default for higher levels
                else:
                    tau = 480

                # Build IOIs
                iois = [tau]
                current = tau
                for r in rhythm_ratios[:n_notes-1]:
                    if r > 0:
                        current = int(current * r)
                    iois.append(current)

                # Emit notes
                current_time = onset
                pitch = base_pitch

                for i in range(n_notes):
                    ioi = iois[i] if i < len(iois) else tau
                    duration = int(ioi * 0.9)

                    notes.append({
                        'pitch': max(0, min(127, pitch)),
                        'onset': current_time,
                        'duration': max(1, duration),
                        'velocity': 80,
                    })

                    current_time += ioi
                    if i < len(intervals):
                        pitch += intervals[i]

            note_output[gm] = notes

        # Convert to MIDI
        mid = MidiFile(ticks_per_beat=480, type=1)

        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
        tempo_track.append(MetaMessage('end_of_track', time=0))

        channel = 0
        for gm, note_list in sorted(note_output.items()):
            if not note_list:
                continue

            track = MidiTrack()
            mid.tracks.append(track)
            track.append(Message('program_change', program=gm % 128, channel=channel, time=0))

            sorted_notes = sorted(note_list, key=lambda n: n['onset'])

            midi_events = []
            for n in sorted_notes:
                midi_events.append((n['onset'], 'on', n['pitch'], n['velocity']))
                midi_events.append((n['onset'] + n['duration'], 'off', n['pitch'], 0))

            midi_events.sort(key=lambda x: (x[0], x[1] == 'on'))

            last_time = 0
            for event_time, event_type, pitch, vel in midi_events:
                delta = event_time - last_time
                if event_type == 'on':
                    track.append(Message('note_on', note=pitch, velocity=vel, channel=channel, time=delta))
                else:
                    track.append(Message('note_off', note=pitch, velocity=0, channel=channel, time=delta))
                last_time = event_time

            track.append(MetaMessage('end_of_track', time=0))
            channel += 1

        mid.save(output_path)

        total_notes = sum(len(n) for n in note_output.values())
        print(f"\nSaved to: {output_path}")
        print(f"  Notes: {total_notes}")

        return events

    # =========================================================================
    # EDITING INTERFACE (Genome-like)
    # =========================================================================

    def show_structure(self, pid: str, max_depth: int = 3) -> str:
        """Show hierarchical structure of a pattern (for genome interface)."""
        def _show(pid, depth):
            if depth > max_depth:
                return "..."

            indent = "  " * depth

            if pid not in self.patterns:
                # Terminal
                if isinstance(pid, str) and pid.startswith('GM'):
                    local_id = int(pid.split('_')[1])
                    if local_id <= N_TERMINALS:
                        rb, vb = decode_terminal(local_id)
                        return f"{indent}[T] rhythm={rb} vel={vb}"
                return f"{indent}[?] {pid}"

            p = self.patterns[pid]
            n_notes = len(p.get('pitch_classes', [1]))
            level = self.get_level(pid)

            result = f"{indent}{pid} (L{level}, {n_notes}n)"

            if p.get('is_hierarchical', False):
                left = p.get('left_child', -1)
                right = p.get('right_child', -1)
                if left != -1:
                    result += "\n" + _show(left, depth + 1)
                if right != -1:
                    result += "\n" + _show(right, depth + 1)

            return result

        return _show(pid, 0)

    def find_alternatives(self, pid: str, same_level: bool = True) -> List[str]:
        """Find alternative patterns that could replace this one."""
        if pid not in self.patterns:
            return []

        p = self.patterns[pid]
        gm = p.get('gm_program', 0)
        level = self.get_level(pid)
        n_notes = len(p.get('pitch_classes', [1]))

        alternatives = []

        if same_level:
            candidates = self.patterns_by_gm_level[gm].get(level, [])
        else:
            candidates = []
            for lvl in range(max(1, level-1), level+2):
                candidates.extend(self.patterns_by_gm_level[gm].get(lvl, []))

        for cand in candidates:
            if cand == pid:
                continue
            cp = self.patterns[cand]
            cn = len(cp.get('pitch_classes', [1]))
            # Similar length
            if 0.5 * n_notes <= cn <= 2.0 * n_notes:
                alternatives.append(cand)

        # Sort by count
        alternatives.sort(key=lambda x: -self.patterns[x].get('count', 0))

        return alternatives[:10]


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python level_sampler.py <checkpoint.npz> [-o output.mid] [--length N] [--level L] [--instruments GM1 GM2 ...]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    output_path = '/tmp/level_output.mid'
    length = 32
    level = 1  # Bigram level by default
    instruments = [65, 66, 67]

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '-o' and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--length' and i + 1 < len(sys.argv):
            length = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--level' and i + 1 < len(sys.argv):
            level = int(sys.argv[i + 1])
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

    # Create sampler
    sampler = LevelSampler(patterns)

    # Demo: show structure of a high-level pattern
    print("\n=== STRUCTURE EXAMPLE ===")
    for pid in list(patterns.keys())[:100]:
        if sampler.get_level(pid) >= 4:
            print(sampler.show_structure(pid, max_depth=3))
            alts = sampler.find_alternatives(pid)
            if alts:
                print(f"\nAlternatives: {alts[:5]}")
            break

    # Generate
    sampler.generate_to_midi(length, instruments, output_path, level=level)


if __name__ == '__main__':
    main()
