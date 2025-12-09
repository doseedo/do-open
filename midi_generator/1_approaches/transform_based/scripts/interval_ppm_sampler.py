#!/usr/bin/env python3
"""
Interval PPM* Sampler - Note-by-note generation with higher-order context.

Key insight from analysis:
- Pattern vocabulary: 4,727 patterns -> sparse transitions (67% count=1 at order-4)
- Interval vocabulary: 77 intervals -> DENSE transitions (32% count=1 at order-4)

This sampler generates note-by-note using:
1. PPM* over interval sequences (horizontal coherence)
2. Co-occurrence lookup for vertical harmony

Architecture:
    for beat in range(n_beats):
        # Vertical: Sample pitch class joint from corpus co-occurrences
        pc_joint = sample_cooccurrence(prev_pc_joint, instruments)

        for gm in instruments:
            target_pc = pc_joint[gm]

            # Horizontal: PPM* samples next interval given context
            interval = sample_interval_ppm(gm, interval_history[gm])

            # Combine: Apply interval to current pitch, constrained by target PC
            pitch = apply_interval_with_pc_constraint(current_pitch, interval, target_pc)

            emit(pitch, beat)
            interval_history[gm].append(interval)
"""

import orjson
import random
import pickle
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, FrozenSet
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


class IntervalPPMSampler:
    """Generate note-by-note using interval PPM* and cross-instrument coordination."""

    def __init__(
        self,
        patterns: dict,
        ppm_model_path: str,
        cross_model_path: str = None,
        verbose: bool = True
    ):
        self.patterns = patterns
        self.verbose = verbose

        # Load interval PPM model
        with open(ppm_model_path, 'rb') as f:
            self.ppm_model = pickle.load(f)

        if verbose:
            print(f"Loaded interval PPM: vocab_size={self.ppm_model['vocab_size']}, "
                  f"max_order={self.ppm_model['max_order']}")

        # Load cross-instrument model
        self.cross_model = None
        if cross_model_path:
            import os
            if os.path.exists(cross_model_path):
                with open(cross_model_path, 'rb') as f:
                    self.cross_model = pickle.load(f)
                if verbose:
                    print(f"Loaded cross-instrument model: {len(self.cross_model['cross_pc_model'])} pairs")

        self._build_indices()

    def _build_indices(self):
        """Build co-occurrence indices from patterns."""
        if self.verbose:
            print("Building co-occurrence indices...")

        # Beat-level pitch co-occurrences (for vertical harmony)
        # Key: (piece, beat) -> {gm: [pitches]}
        beat_pitches = defaultdict(lambda: defaultdict(list))

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            for occ in p.get('occurrences', []):
                piece = occ.get('piece_id', 'unknown')
                onset = occ.get('onset_time', 0)
                pitch = occ.get('first_pitch', 60)
                beat = onset // 480

                beat_pitches[(piece, beat)][gm].append(pitch)

        # Build PC joint distribution (what pitch classes occur together)
        self.pc_joints = Counter()
        self.pc_transitions = defaultdict(Counter)

        piece_beats = defaultdict(list)
        for (piece, beat), gm_pitches in beat_pitches.items():
            if len(gm_pitches) >= 2:  # Multi-instrument
                pc_joint = frozenset(
                    (gm, p % 12)
                    for gm, pitches in gm_pitches.items()
                    for p in pitches
                )
                piece_beats[piece].append((beat, pc_joint, gm_pitches))

        for piece, beats in piece_beats.items():
            beats.sort(key=lambda x: x[0])
            for i, (beat, pc_joint, gm_pitches) in enumerate(beats):
                self.pc_joints[pc_joint] += 1
                if i > 0:
                    prev_joint = beats[i-1][1]
                    self.pc_transitions[prev_joint][pc_joint] += 1

        # Pitch distribution per (gm, pc) for octave selection
        self.pitch_dist_by_gm_pc = defaultdict(Counter)

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            for occ in p.get('occurrences', []):
                pitch = occ.get('first_pitch', 60)
                pc = pitch % 12
                self.pitch_dist_by_gm_pc[(gm, pc)][pitch] += 1

        if self.verbose:
            print(f"  PC joints: {len(self.pc_joints)}")
            print(f"  PC transitions: {sum(len(v) for v in self.pc_transitions.values())}")
            print(f"  Pitch distributions: {len(self.pitch_dist_by_gm_pc)} (gm, pc) pairs")

    def sample_interval_ppm(
        self,
        gm: int,
        interval_history: List[int],
        escape_probability: float = 0.1
    ) -> int:
        """Sample next interval using PPM* with backoff.

        Args:
            gm: GM program number
            interval_history: Recent intervals (as interval values, not IDs)
            escape_probability: Probability of backing off to shorter context

        Returns:
            Next interval (semitones)
        """
        ppm_counts = self.ppm_model['ppm_counts']
        max_order = self.ppm_model['max_order']
        interval_to_id = self.ppm_model['interval_to_id']
        id_to_interval = self.ppm_model['id_to_interval']

        if gm not in ppm_counts:
            # Fallback to random interval
            return random.choice(list(interval_to_id.keys()))

        gm_counts = ppm_counts[gm]

        # Convert history to IDs
        history_ids = [interval_to_id.get(iv, 0) for iv in interval_history if iv in interval_to_id]

        # PPM* sampling: try from longest context down
        for order in range(min(max_order, len(history_ids)), -1, -1):
            # Escape with probability
            if order > 0 and random.random() < escape_probability:
                continue

            if order == 0:
                context = ()
            else:
                context = tuple(history_ids[-order:])

            if order not in gm_counts:
                continue

            if context not in gm_counts[order]:
                continue

            transitions = gm_counts[order][context]

            if not transitions:
                continue

            # Sample from distribution
            ids = list(transitions.keys())
            counts = list(transitions.values())
            total = sum(counts)
            weights = [c / total for c in counts]

            selected_id = random.choices(ids, weights=weights)[0]
            return id_to_interval.get(selected_id, 0)

        # Ultimate fallback: small random interval
        return random.choice([-2, -1, 0, 1, 2])

    def sample_pc_joint(
        self,
        instruments: List[int],
        previous: FrozenSet[Tuple[int, int]] = None
    ) -> FrozenSet[Tuple[int, int]]:
        """Sample pitch-class joint for vertical coordination."""
        if previous and previous in self.pc_transitions:
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

        # Sample from marginal
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

        # Fallback: consonant triad
        root = random.randint(0, 11)
        voicings = [0, 4, 7, 0, 4]
        return frozenset(
            (gm, (root + voicings[i % len(voicings)]) % 12)
            for i, gm in enumerate(instruments)
        )

    def sample_pc_given_others(
        self,
        gm: int,
        other_pitches: Dict[int, int]
    ) -> int:
        """Sample target pitch class for gm conditioned on other instruments' current pitches.

        Args:
            gm: GM program to sample for
            other_pitches: {other_gm: current_pitch} for instruments already processed

        Returns:
            Target pitch class (0-11)
        """
        if not self.cross_model or not other_pitches:
            return random.randint(0, 11)

        cross_pc = self.cross_model['cross_pc_model']

        # Collect votes from each other instrument
        pc_votes = Counter()

        for other_gm, other_pitch in other_pitches.items():
            other_pc = other_pitch % 12
            key = (other_gm, gm)

            if key in cross_pc and other_pc in cross_pc[key]:
                dist = cross_pc[key][other_pc]
                for pc, count in dist.items():
                    pc_votes[pc] += count

        if not pc_votes:
            return random.randint(0, 11)

        # Sample from combined distribution
        pcs = list(pc_votes.keys())
        weights = list(pc_votes.values())
        total = sum(weights)
        weights = [w / total for w in weights]

        return random.choices(pcs, weights=weights)[0]

    def get_likely_pcs_for_others(
        self,
        gm: int,
        other_pitches: Dict[int, int]
    ) -> Dict[int, float]:
        """Get pitch class distribution conditioned on other instruments (DISCOVERED from corpus).

        This is discovery-based, not rule-based. The cross_pc_model was built from
        actual corpus co-occurrences, not prescribed consonance intervals.

        Args:
            gm: GM program for which to sample
            other_pitches: {other_gm: current_pitch}

        Returns:
            Dict mapping pitch class (0-11) to weight
        """
        if not other_pitches or not self.cross_model:
            # Uniform if no conditioning data
            return {pc: 1.0 for pc in range(12)}

        cross_pc = self.cross_model['cross_pc_model']

        # Collect votes from each other instrument based on CORPUS statistics
        pc_weights = Counter()

        for other_gm, other_pitch in other_pitches.items():
            other_pc = other_pitch % 12
            key = (other_gm, gm)

            if key in cross_pc and other_pc in cross_pc[key]:
                # This is what ACTUALLY occurred in the corpus when other_gm played other_pc
                dist = cross_pc[key][other_pc]
                for pc, count in dist.items():
                    pc_weights[pc] += count

        if not pc_weights:
            return {pc: 1.0 for pc in range(12)}

        # Normalize
        total = sum(pc_weights.values())
        return {pc: count / total for pc, count in pc_weights.items()}

    def sample_interval_ppm_vertical(
        self,
        gm: int,
        interval_history: List[int],
        current_pitch: int,
        other_pitches: Dict[int, int],
        escape_probability: float = 0.1,
        vertical_weight: float = 0.5
    ) -> int:
        """Sample next interval using PPM* with vertical harmony weighting.

        Instead of adjusting pitch after sampling, this weights PPM* samples
        toward intervals that produce pitch classes that CO-OCCURRED in the corpus
        with what other instruments are playing. This is DISCOVERY-based (from
        corpus statistics), not PRESCRIPTION-based (from music theory rules).

        Args:
            gm: GM program number
            interval_history: Recent intervals (as interval values, not IDs)
            current_pitch: Current pitch of this instrument
            other_pitches: {other_gm: current_pitch} for other instruments
            escape_probability: PPM* escape probability
            vertical_weight: How much to weight corpus co-occurrence (0-1)

        Returns:
            Next interval (semitones)
        """
        ppm_counts = self.ppm_model['ppm_counts']
        max_order = self.ppm_model['max_order']
        interval_to_id = self.ppm_model['interval_to_id']
        id_to_interval = self.ppm_model['id_to_interval']

        # Get pitch class distribution from CORPUS co-occurrences (discovery-based)
        pc_dist = self.get_likely_pcs_for_others(gm, other_pitches)

        if gm not in ppm_counts:
            return random.choice(list(interval_to_id.keys()))

        gm_counts = ppm_counts[gm]

        # Convert history to IDs
        history_ids = [interval_to_id.get(iv, 0) for iv in interval_history if iv in interval_to_id]

        # PPM* sampling: try from longest context down
        for order in range(min(max_order, len(history_ids)), -1, -1):
            if order > 0 and random.random() < escape_probability:
                continue

            if order == 0:
                context = ()
            else:
                context = tuple(history_ids[-order:])

            if order not in gm_counts:
                continue

            if context not in gm_counts[order]:
                continue

            transitions = gm_counts[order][context]

            if not transitions:
                continue

            # Weight intervals by both PPM count and corpus co-occurrence
            ids = list(transitions.keys())
            ppm_weights = list(transitions.values())
            total_ppm = sum(ppm_weights)
            ppm_weights = [w / total_ppm for w in ppm_weights]

            # Calculate vertical weights from CORPUS statistics
            vertical_weights = []
            for interval_id in ids:
                interval = id_to_interval.get(interval_id, 0)
                new_pitch = current_pitch + interval
                new_pc = new_pitch % 12

                # Use corpus-discovered co-occurrence probability
                vert_w = pc_dist.get(new_pc, 0.01)  # Small floor to avoid zeros
                vertical_weights.append(vert_w)

            # Combine weights: PPM * vertical^strength
            combined_weights = []
            for ppm_w, vert_w in zip(ppm_weights, vertical_weights):
                # Multiplicative combination with strength control
                combined = ppm_w * (vert_w ** vertical_weight)
                combined_weights.append(combined)

            # Normalize
            total = sum(combined_weights)
            if total > 0:
                combined_weights = [w / total for w in combined_weights]
            else:
                combined_weights = ppm_weights

            selected_id = random.choices(ids, weights=combined_weights)[0]
            return id_to_interval.get(selected_id, 0)

        # Ultimate fallback: small random interval
        return random.choice([-2, -1, 0, 1, 2])

    def sample_pitch_for_gm_pc(self, gm: int, pc: int) -> int:
        """Sample absolute pitch given instrument and pitch class."""
        key = (gm, pc)
        if key in self.pitch_dist_by_gm_pc:
            dist = self.pitch_dist_by_gm_pc[key]
            pitches = list(dist.keys())
            weights = list(dist.values())
            total = sum(weights)
            weights = [w / total for w in weights]
            return random.choices(pitches, weights=weights)[0]

        # Fallback by instrument range
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
        n_notes: int,
        instruments: List[int],
        notes_per_beat: float = 2.0,
        vertical_strength: float = 0.5
    ) -> Dict[int, List[dict]]:
        """Generate using interval PPM* + cross-instrument coordination.

        Args:
            n_notes: Number of notes per instrument
            instruments: GM programs to generate
            notes_per_beat: Note density
            vertical_strength: How strongly to weight vertical consonance in PPM* (0-1)
                0 = pure PPM* (ignore other instruments)
                1 = strongly prefer consonant intervals with other instruments

        Returns:
            {gm: [{'pitch': ..., 'onset': ..., 'duration': ...}, ...]}
        """
        output = {gm: [] for gm in instruments}
        current_pitch = {gm: self.sample_pitch_for_gm_pc(gm, random.randint(0, 11)) for gm in instruments}
        interval_history = {gm: [] for gm in instruments}

        tick_per_note = int(480 / notes_per_beat)

        for n in range(n_notes):
            current_time = n * tick_per_note

            # Track pitches generated this timestep for cross-instrument conditioning
            this_timestep_pitches = {}

            for gm in instruments:
                # Sample interval using PPM* with vertical harmony weighting
                if this_timestep_pitches and vertical_strength > 0:
                    interval = self.sample_interval_ppm_vertical(
                        gm, interval_history[gm],
                        current_pitch[gm], this_timestep_pitches,
                        vertical_weight=vertical_strength
                    )
                else:
                    interval = self.sample_interval_ppm(gm, interval_history[gm])

                # Apply interval
                new_pitch = current_pitch[gm] + interval

                # Clamp to MIDI range
                new_pitch = max(24, min(108, new_pitch))

                output[gm].append({
                    'pitch': new_pitch,
                    'onset': current_time,
                    'duration': int(tick_per_note * 0.9),
                    'velocity': 80,
                })

                # Track for cross-instrument conditioning
                this_timestep_pitches[gm] = new_pitch

                # Update state
                actual_interval = new_pitch - current_pitch[gm]
                interval_history[gm].append(actual_interval)
                if len(interval_history[gm]) > 10:
                    interval_history[gm] = interval_history[gm][-10:]

                current_pitch[gm] = new_pitch

        return output

    def generate_to_midi(
        self,
        n_notes: int,
        instruments: List[int],
        output_path: str,
        bpm: int = 120,
        notes_per_beat: float = 2.0,
        vertical_strength: float = 0.8
    ):
        """Generate and save to MIDI."""
        print(f"\nGenerating {n_notes} notes per instrument for {instruments}...")
        print(f"  Horizontal: Interval PPM* (order {self.ppm_model['max_order']}, vocab {self.ppm_model['vocab_size']})")
        print(f"  Vertical: Cross-instrument PC conditioning (strength={vertical_strength})")

        note_output = self.generate(n_notes, instruments, notes_per_beat, vertical_strength)

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

            events = []
            for n in note_list:
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
        print(f"\nSaved to: {output_path}")
        print(f"  Tracks: {len(mid.tracks)}")
        print(f"  Notes: {total_notes}")

        self._analyze_output(note_output, instruments)

        return note_output

    def _analyze_output(self, note_output: Dict[int, List[dict]], instruments: List[int]):
        """Analyze horizontal and vertical properties."""
        print("\n=== ANALYSIS ===")

        # Horizontal: interval statistics per track
        print("\nHorizontal (per-track):")
        for gm in instruments:
            notes = note_output[gm]
            if len(notes) < 2:
                continue

            intervals = [notes[i+1]['pitch'] - notes[i]['pitch'] for i in range(len(notes)-1)]

            step_count = sum(1 for iv in intervals if abs(iv) <= 2)
            step_ratio = step_count / len(intervals)

            # Autocorrelation
            if len(intervals) > 10:
                arr = np.array(intervals)
                arr_norm = (arr - np.mean(arr)) / (np.std(arr) + 1e-6)
                autocorr = np.corrcoef(arr_norm[:-1], arr_norm[1:])[0, 1]
            else:
                autocorr = 0

            print(f"  GM {gm}: {len(notes)} notes, step_ratio={step_ratio:.2f}, autocorr={autocorr:.3f}")

        # Vertical: harmony at sync points
        print("\nVertical (sync points):")
        pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        # Get notes at each beat
        sync_points = set()
        for gm, notes in note_output.items():
            for n in notes:
                if (n['onset'] // 480) % 2 == 0:  # Every 2 beats
                    sync_points.add(n['onset'] // 480 * 480)

        consonant_intervals = {0, 3, 4, 5, 7, 8, 9}
        all_intervals = []

        for sync_time in sorted(sync_points)[:20]:
            pitches_at_sync = {}
            for gm, notes in note_output.items():
                for n in notes:
                    if n['onset'] <= sync_time < n['onset'] + n['duration']:
                        pitches_at_sync[gm] = n['pitch']
                        break

            if len(pitches_at_sync) >= 2:
                pcs = sorted(set(p % 12 for p in pitches_at_sync.values()))
                for i, pc1 in enumerate(pcs):
                    for pc2 in pcs[i+1:]:
                        interval = (pc2 - pc1) % 12
                        all_intervals.append(interval)

        if all_intervals:
            consonant_count = sum(1 for iv in all_intervals if iv in consonant_intervals)
            consonance = consonant_count / len(all_intervals)
            print(f"  Consonance: {100*consonance:.1f}%")

            # Show interval distribution
            interval_counts = Counter(all_intervals)
            interval_names = ["P1", "m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7"]
            print("  Top intervals:")
            for iv, count in interval_counts.most_common(5):
                pct = 100 * count / len(all_intervals)
                print(f"    {interval_names[iv]}: {pct:.1f}%")


def main():
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python interval_ppm_sampler.py <checkpoint.npz> [-o output.mid] [--notes N] [--instruments GM1 GM2 ...] [--vertical-strength 0.8]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_path = '/tmp/interval_ppm.mid'
    n_notes = 256
    instruments = [65, 66, 67]  # Sax section
    vertical_strength = 0.8

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '-o' and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--notes' and i + 1 < len(sys.argv):
            n_notes = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--vertical-strength' and i + 1 < len(sys.argv):
            vertical_strength = float(sys.argv[i + 1])
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

    base_dir = os.path.dirname(checkpoint_path)
    json_path = os.path.join(base_dir, patterns_file) if base_dir else patterns_file

    print(f"Loading patterns from: {json_path}")
    with open(json_path, 'rb') as f:
        patterns = orjson.loads(f.read())

    print(f"Loaded {len(patterns)} patterns")

    # Find PPM model
    ppm_path = os.path.join(base_dir, 'ppm_interval_model.pkl')
    if not os.path.exists(ppm_path):
        print(f"Error: PPM model not found at {ppm_path}")
        print("Run the interval PPM builder first.")
        sys.exit(1)

    # Find cross-instrument model
    cross_path = os.path.join(base_dir, 'cross_instrument_model.pkl')
    if os.path.exists(cross_path):
        print(f"Found cross-instrument model at {cross_path}")
    else:
        cross_path = None
        print("No cross-instrument model found, using independent generation")

    # Create sampler and generate
    sampler = IntervalPPMSampler(patterns, ppm_path, cross_path)
    sampler.generate_to_midi(n_notes, instruments, output_path, vertical_strength=vertical_strength)


if __name__ == '__main__':
    main()
