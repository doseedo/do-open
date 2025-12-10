#!/usr/bin/env python3
"""
Separated Sampler - Rhythm/Contour from patterns, Pitch from co-occurrence.

Philosophy:
- Pattern provides WHEN (rhythm) and HOW (contour/intervals)
- Co-occurrence provides WHAT (absolute pitch at each timestep)
- These are sampled independently, then combined

This separation allows:
1. Rhythm coherence (patterns provide temporal structure)
2. Harmonic coordination (co-occurrence provides vertical alignment)
3. Both discovered from data, neither prescribed
"""

import orjson
import random
import numpy as np
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, FrozenSet, Optional
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


class SeparatedSampler:
    """Generate by separating rhythm/contour selection from pitch selection."""

    def __init__(
        self,
        patterns: dict,
        verbose: bool = True,
        max_pattern_length: int = None,
        max_pattern_beats: float = None,
        ppm_model_path: str = None
    ):
        self.patterns = patterns
        self.verbose = verbose
        self.ppm_model = None

        # Load PPM model if provided
        if ppm_model_path:
            self._load_ppm_model(ppm_model_path)

        # Filter patterns
        if max_pattern_length:
            original = len(self.patterns)
            self.patterns = {
                pid: p for pid, p in self.patterns.items()
                if len(p.get('pitch_classes', [1])) <= max_pattern_length
            }
            if verbose:
                print(f"Filtered by length: {original} -> {len(self.patterns)}")

        if max_pattern_beats:
            original = len(self.patterns)
            self.patterns = {
                pid: p for pid, p in self.patterns.items()
                if self._estimate_duration(p) <= max_pattern_beats
            }
            if verbose:
                print(f"Filtered by duration: {original} -> {len(self.patterns)}")

        self._build_indices()

    def _estimate_duration(self, p: dict) -> float:
        """Estimate pattern duration in beats."""
        occs = p.get('occurrences', [])
        if not occs:
            return 0
        tau = occs[0].get('tau_offset', 480)
        ratios = p.get('rhythm_ratios', [])
        n_notes = len(p.get('pitch_classes', [1]))

        total = tau
        current = tau
        for r in ratios[:n_notes-1]:
            if r > 0:
                current = current * r
                total += current
        return total / 480

    def _build_indices(self):
        """Build separate indices for rhythm and pitch."""
        if self.verbose:
            print("Building separated indices...")

        # =====================================================
        # INDEX 1: Rhythm patterns per instrument
        # Key: gm_program
        # Value: list of (pattern_id, rhythm_signature)
        # =====================================================
        self.rhythm_patterns_by_gm = defaultdict(list)

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)

            # Rhythm signature: (n_notes, rhythm_bucket, duration_estimate)
            n_notes = len(p.get('pitch_classes', [1]))
            rhythm_bucket = p.get('rhythm_bucket', 8)
            duration = self._estimate_duration(p)

            self.rhythm_patterns_by_gm[gm].append({
                'pid': pid,
                'n_notes': n_notes,
                'rhythm_bucket': rhythm_bucket,
                'duration_beats': duration,
                'intervals': p.get('pitch_intervals', []),
                'rhythm_ratios': p.get('rhythm_ratios', []),
                'count': p.get('count', 1),
            })

        if self.verbose:
            print(f"  Rhythm patterns: {sum(len(v) for v in self.rhythm_patterns_by_gm.values())} across {len(self.rhythm_patterns_by_gm)} instruments")

        # =====================================================
        # INDEX 2: Pitch co-occurrence at beat boundaries
        # Key: (piece, beat)
        # Value: {gm: [pitches that occurred]}
        # =====================================================
        beat_pitches = defaultdict(lambda: defaultdict(list))

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            for occ in p.get('occurrences', []):
                piece = occ.get('piece_id', 'unknown')
                onset = occ.get('onset_time', 0)
                pitch = occ.get('first_pitch', 60)
                beat = onset // 480

                beat_pitches[(piece, beat)][gm].append(pitch)

        # =====================================================
        # INDEX 3: Joint pitch distributions (vertical slices)
        # Key: frozenset of (gm, pitch_class) for coordination
        # Value: count of occurrences
        # =====================================================
        self.pc_joints = Counter()
        self.pc_transitions = defaultdict(Counter)

        # Group by piece to get sequences
        piece_beats = defaultdict(list)
        for (piece, beat), gm_pitches in beat_pitches.items():
            if len(gm_pitches) >= 2:  # Multi-instrument
                pc_joint = frozenset(
                    (gm, p % 12)
                    for gm, pitches in gm_pitches.items()
                    for p in pitches
                )
                piece_beats[piece].append((beat, pc_joint, gm_pitches))

        # Build transitions
        for piece, beats in piece_beats.items():
            beats.sort(key=lambda x: x[0])
            for i, (beat, pc_joint, gm_pitches) in enumerate(beats):
                self.pc_joints[pc_joint] += 1
                if i > 0:
                    prev_joint = beats[i-1][1]
                    self.pc_transitions[prev_joint][pc_joint] += 1

        if self.verbose:
            print(f"  Unique PC joints: {len(self.pc_joints)}")
            print(f"  PC transitions: {sum(len(v) for v in self.pc_transitions.values())}")

        # =====================================================
        # INDEX 4: Pitch distributions per (gm, pc)
        # For choosing octave given pitch class
        # =====================================================
        self.pitch_dist_by_gm_pc = defaultdict(Counter)

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            for occ in p.get('occurrences', []):
                pitch = occ.get('first_pitch', 60)
                pc = pitch % 12
                self.pitch_dist_by_gm_pc[(gm, pc)][pitch] += 1

        if self.verbose:
            print(f"  Pitch distributions: {len(self.pitch_dist_by_gm_pc)} (gm, pc) pairs")

    def _load_ppm_model(self, path: str):
        """Load pre-computed PPM* model for pattern sequencing."""
        try:
            with open(path, 'rb') as f:
                self.ppm_model = pickle.load(f)
            if self.verbose:
                print(f"Loaded PPM model: max_order={self.ppm_model['max_order']}, "
                      f"{len(self.ppm_model['gm_programs'])} GM programs")
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load PPM model: {e}")
            self.ppm_model = None

    def sample_rhythm_pattern_ppm(
        self,
        gm: int,
        pattern_history: List[str],
        target_beats: float = None,
        escape_probability: float = 0.1
    ) -> dict:
        """Sample pattern using PPM* for higher-order context.

        PPM* algorithm:
        1. Try longest context (order N) first
        2. If no match or with escape probability, back off to order N-1
        3. Continue until order 0 (unigram)

        Args:
            gm: GM program number
            pattern_history: List of recent pattern IDs for this instrument
            target_beats: Optional target duration filter
            escape_probability: Probability of backing off to shorter context
        """
        if not self.ppm_model or gm not in self.ppm_model['ppm_counts']:
            # Fallback to original method
            return self.sample_rhythm_pattern(gm, target_beats)

        ppm_counts = self.ppm_model['ppm_counts'][gm]
        max_order = self.ppm_model['max_order']

        # Get valid pattern IDs for this instrument
        candidates = self.rhythm_patterns_by_gm.get(gm, [])
        if not candidates:
            return self.sample_rhythm_pattern(gm, target_beats)

        valid_pids = set(p['pid'] for p in candidates)

        # PPM* sampling: try from longest context down
        for order in range(min(max_order, len(pattern_history)), -1, -1):
            # Escape to shorter context with some probability
            if order > 0 and random.random() < escape_probability:
                continue

            if order == 0:
                context = ()
            else:
                context = tuple(pattern_history[-order:])

            # Convert context to string tuple for dict lookup
            context_key = str(context) if context else "()"

            if order not in ppm_counts:
                continue

            # Check if this context exists
            if context not in ppm_counts[order]:
                continue

            transitions = ppm_counts[order][context]

            # Filter to valid patterns (those we have in rhythm_patterns_by_gm)
            valid_transitions = {
                pid: count for pid, count in transitions.items()
                if pid in valid_pids
            }

            if not valid_transitions:
                continue

            # Sample from distribution
            pids = list(valid_transitions.keys())
            counts = list(valid_transitions.values())
            total = sum(counts)
            weights = [c / total for c in counts]

            selected_pid = random.choices(pids, weights=weights)[0]

            # Find the pattern info
            for p in candidates:
                if p['pid'] == selected_pid:
                    return p

        # Ultimate fallback
        return self.sample_rhythm_pattern(gm, target_beats)

    def sample_rhythm_pattern(self, gm: int, target_beats: float = None) -> dict:
        """Sample a rhythm pattern for instrument, independent of pitch.

        Returns pattern skeleton: rhythm_ratios, intervals, n_notes
        """
        candidates = self.rhythm_patterns_by_gm.get(gm, [])
        if not candidates:
            # Fallback to any instrument
            all_patterns = []
            for patterns in self.rhythm_patterns_by_gm.values():
                all_patterns.extend(patterns)
            candidates = all_patterns

        if not candidates:
            return {
                'pid': None,
                'n_notes': 1,
                'intervals': [],
                'rhythm_ratios': [],
                'duration_beats': 1.0,
            }

        # Weight by count (frequency in corpus)
        weights = [p['count'] for p in candidates]

        # Optionally filter by duration
        if target_beats:
            filtered = [(p, w) for p, w in zip(candidates, weights)
                       if 0.5 * target_beats <= p['duration_beats'] <= 2.0 * target_beats]
            if filtered:
                candidates, weights = zip(*filtered)
                candidates, weights = list(candidates), list(weights)

        total = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(candidates, weights=weights)[0]

    def sample_pc_joint(
        self,
        instruments: List[int],
        previous: FrozenSet[Tuple[int, int]] = None
    ) -> FrozenSet[Tuple[int, int]]:
        """Sample a pitch-class joint for vertical coordination.

        Returns: frozenset of (gm, pitch_class) pairs
        """
        if previous and previous in self.pc_transitions:
            # Use Markov transition
            transitions = self.pc_transitions[previous]
            valid = []
            weights = []

            for joint, count in transitions.items():
                joint_gms = set(gm for gm, _ in joint)
                if all(gm in joint_gms for gm in instruments):
                    valid.append(joint)
                    weights.append(count)

            if valid:
                total = sum(weights)
                weights = [w / total for w in weights]
                return random.choices(valid, weights=weights)[0]

        # Sample from marginal distribution
        valid = []
        weights = []

        for joint, count in self.pc_joints.items():
            joint_gms = set(gm for gm, _ in joint)
            if all(gm in joint_gms for gm in instruments):
                valid.append(joint)
                weights.append(count)

        if valid:
            total = sum(weights)
            weights = [w / total for w in weights]
            return random.choices(valid, weights=weights)[0]

        # Ultimate fallback: construct consonant joint
        root = random.randint(0, 11)
        # Simple major triad voicing
        voicings = [0, 4, 7, 0, 4]  # root, 3rd, 5th, root, 3rd
        return frozenset(
            (gm, (root + voicings[i % len(voicings)]) % 12)
            for i, gm in enumerate(instruments)
        )

    def sample_pitch_for_gm_pc(self, gm: int, pc: int) -> int:
        """Sample absolute pitch given instrument and pitch class.

        Uses corpus distribution to choose appropriate octave.
        """
        key = (gm, pc)
        if key in self.pitch_dist_by_gm_pc:
            dist = self.pitch_dist_by_gm_pc[key]
            pitches = list(dist.keys())
            weights = list(dist.values())
            total = sum(weights)
            weights = [w / total for w in weights]
            return random.choices(pitches, weights=weights)[0]

        # Fallback: middle octave for this instrument
        # Bass instruments lower, treble higher
        if gm in [32, 33, 34, 35, 36, 37, 38, 39]:  # Bass
            octave = 2
        elif gm in [56, 57, 58, 59, 60, 61, 62, 63]:  # Brass
            octave = 4
        elif gm in [64, 65, 66, 67, 68, 69, 70, 71]:  # Reeds
            octave = 4
        else:
            octave = 4

        return octave * 12 + pc

    def generate(
        self,
        length: int,
        instruments: List[int],
        beats_per_event: float = 2.0,
        use_ppm: bool = True
    ) -> Dict[int, List[dict]]:
        """Generate using separated sampling.

        Args:
            length: Number of coordination events
            instruments: GM programs to generate
            beats_per_event: Target duration per event
            use_ppm: Whether to use PPM* for pattern sequencing

        Returns:
            {gm: [{'pattern': ..., 'pitch': ..., 'onset': ...}, ...]}
        """
        output = {gm: [] for gm in instruments}
        current_time = {gm: 0 for gm in instruments}
        previous_pc_joint = None

        # Track pattern history per instrument for PPM*
        pattern_history = {gm: [] for gm in instruments}

        for t in range(length):
            # Step 1: Sample PC joint for vertical coordination
            pc_joint = self.sample_pc_joint(instruments, previous_pc_joint)

            # Step 2: For each instrument, sample rhythm pattern
            # Step 3: Apply pitch from PC joint

            for gm in instruments:
                # Get target pitch class from joint
                target_pc = None
                for jgm, pc in pc_joint:
                    if jgm == gm:
                        target_pc = pc
                        break

                if target_pc is None:
                    target_pc = random.randint(0, 11)

                # Sample rhythm pattern - use PPM* if available and enabled
                if use_ppm and self.ppm_model:
                    rhythm = self.sample_rhythm_pattern_ppm(
                        gm,
                        pattern_history[gm],
                        target_beats=beats_per_event
                    )
                else:
                    rhythm = self.sample_rhythm_pattern(gm, target_beats=beats_per_event)

                # Track pattern ID for PPM* context
                if rhythm.get('pid'):
                    pattern_history[gm].append(rhythm['pid'])
                    # Keep only last N patterns for memory efficiency
                    if len(pattern_history[gm]) > 10:
                        pattern_history[gm] = pattern_history[gm][-10:]

                # Sample absolute pitch from distribution
                pitch = self.sample_pitch_for_gm_pc(gm, target_pc)

                output[gm].append({
                    'pattern': rhythm,
                    'pitch': pitch,
                    'onset': current_time[gm],
                })

                # Advance time based on pattern duration
                current_time[gm] += int(rhythm['duration_beats'] * 480)

            previous_pc_joint = pc_joint

        return output

    def generate_to_midi(
        self,
        length: int,
        instruments: List[int],
        output_path: str,
        bpm: int = 120,
        beats_per_event: float = 2.0,
        use_ppm: bool = True
    ):
        """Generate and save to MIDI."""
        print(f"\nGenerating {length} events for {instruments}...")
        if use_ppm and self.ppm_model:
            print(f"  Rhythm: PPM* over pattern sequences (order {self.ppm_model['max_order']})")
        else:
            print(f"  Rhythm: sampled per-instrument from corpus patterns (first-order)")
        print(f"  Pitch: sampled from vertical co-occurrence")

        # Generate
        events = self.generate(length, instruments, beats_per_event, use_ppm=use_ppm)

        # Expand to notes
        note_output = {}
        for gm, event_list in events.items():
            notes = []

            for event in event_list:
                pattern = event['pattern']
                base_pitch = event['pitch']
                onset = event['onset']

                intervals = pattern.get('intervals', [])
                rhythm_ratios = pattern.get('rhythm_ratios', [])
                n_notes = pattern.get('n_notes', 1)

                # Get base IOI from pattern duration
                duration_beats = pattern.get('duration_beats', 1.0)
                if n_notes > 1 and rhythm_ratios:
                    # Estimate tau from duration and ratios
                    # total = tau * (1 + r1 + r1*r2 + ...)
                    # Simplified: tau = duration / n_notes * 480
                    tau = int(duration_beats * 480 / n_notes)
                else:
                    tau = 480

                # Build IOIs using successive ratios
                iois = [tau]
                current_ioi = tau
                for r in rhythm_ratios[:n_notes-1]:
                    if r > 0:
                        current_ioi = int(current_ioi * r)
                    iois.append(current_ioi)

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

                    # Apply interval for next note
                    if i < len(intervals):
                        pitch += intervals[i]

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

        mid.save(output_path)

        total_notes = sum(len(n) for n in note_output.values())
        print(f"\nSaved to: {output_path}")
        print(f"  Tracks: {len(mid.tracks)}")
        print(f"  Notes: {total_notes}")

        # Analyze harmony
        self._analyze_harmony(events, instruments)

        return events

    def _analyze_harmony(self, events: Dict[int, List[dict]], instruments: List[int]):
        """Analyze vertical harmony of output."""
        print("\n=== HARMONY ANALYSIS ===")

        pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        interval_names = ["P1", "m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7"]

        # Get pitch at each event
        n_events = len(events[instruments[0]])
        interval_counts = Counter()

        print("\nFirst 8 vertical slices:")
        for t in range(min(8, n_events)):
            pitches = {}
            for gm in instruments:
                if t < len(events[gm]):
                    pitches[gm] = events[gm][t]['pitch']

            pcs = tuple(sorted(set(p % 12 for p in pitches.values())))
            pc_str = ", ".join(pc_names[pc] for pc in pcs)
            print(f"  t={t}: {{{pc_str}}}")

            # Count intervals
            for i, pc1 in enumerate(pcs):
                for pc2 in pcs[i+1:]:
                    interval = (pc2 - pc1) % 12
                    interval_counts[interval] += 1

        # Full interval distribution
        for t in range(n_events):
            pcs = set()
            for gm in instruments:
                if t < len(events[gm]):
                    pcs.add(events[gm][t]['pitch'] % 12)

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
    import os

    if len(sys.argv) < 2:
        print("Usage: python separated_sampler.py <checkpoint.npz> [-o output.mid] [--length N] [--instruments GM1 GM2 ...] [--ppm-model PPM.pkl] [--no-ppm]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    output_path = '/tmp/separated.mid'
    length = 64
    instruments = [65, 66, 67]  # Sax section
    max_pattern_length = 8
    max_pattern_beats = 4.0
    ppm_model_path = None
    use_ppm = True

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
        elif sys.argv[i] == '--ppm-model' and i + 1 < len(sys.argv):
            ppm_model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--no-ppm':
            use_ppm = False
            i += 1
        elif sys.argv[i] == '--instruments':
            instruments = []
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith('-'):
                instruments.append(int(sys.argv[i]))
                i += 1
        else:
            i += 1

    # Auto-detect PPM model if not specified
    if ppm_model_path is None:
        base_dir = os.path.dirname(checkpoint_path)
        default_ppm = os.path.join(base_dir, 'ppm_pattern_model.pkl')
        if os.path.exists(default_ppm):
            ppm_model_path = default_ppm
            print(f"Auto-detected PPM model: {ppm_model_path}")

    # Load patterns
    print(f"Loading checkpoint: {checkpoint_path}")
    data = np.load(checkpoint_path, allow_pickle=True)
    patterns_file = str(data['patterns_json_file'][0])

    base_dir = os.path.dirname(checkpoint_path)
    json_path = os.path.join(base_dir, patterns_file) if base_dir else patterns_file

    print(f"Loading patterns from: {json_path}")
    with open(json_path, 'rb') as f:
        patterns = orjson.loads(f.read())

    print(f"Loaded {len(patterns)} patterns")

    # Create sampler and generate
    sampler = SeparatedSampler(
        patterns,
        max_pattern_length=max_pattern_length,
        max_pattern_beats=max_pattern_beats,
        ppm_model_path=ppm_model_path
    )
    sampler.generate_to_midi(length, instruments, output_path, use_ppm=use_ppm)


if __name__ == '__main__':
    main()
