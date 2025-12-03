#!/usr/bin/env python3
"""
Genome Generator - Generate music from learned patterns
========================================================

Uses checkpoint data:
- Patterns with pitch_classes, rhythm_ratios, duration_ratios, velocity_ratios
- Transform vocabulary with frequencies
- Orchestration rules for multitrack generation

Usage:
    python scripts/generator.py checkpoint_v42.npz --output generated.mid --length 50
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class GenomeGenerator:
    """Generate music from learned genome patterns."""

    def __init__(self, checkpoint_path: str):
        """Load checkpoint and build samplers."""
        self.checkpoint_path = checkpoint_path
        self.load_checkpoint(checkpoint_path)
        self.build_pcfg()
        self.build_transform_sampler()
        self.build_orchestration_sampler()

    def load_checkpoint(self, checkpoint_path: str):
        """Load v4 checkpoint format with external JSON files."""
        ckpt = np.load(checkpoint_path, allow_pickle=True)
        base_path = checkpoint_path.replace('.npz', '')

        # Load patterns from external JSON
        patterns_file = ckpt.get('patterns_json_file', [None])[0]
        if patterns_file:
            patterns_path = os.path.join(os.path.dirname(checkpoint_path), patterns_file)
            with open(patterns_path) as f:
                self.rules = json.load(f)
        else:
            # Fallback: try inline
            self.rules = json.loads(str(ckpt['patterns_json'][0]))

        # Load transforms
        transforms_file = ckpt.get('transforms_json_file', [None])[0]
        if transforms_file:
            transforms_path = os.path.join(os.path.dirname(checkpoint_path), transforms_file)
            with open(transforms_path) as f:
                transforms_data = json.load(f)
        else:
            transforms_data = {'vocabulary': [], 'stats': {}}

        self.transform_vocab = transforms_data.get('vocabulary', [])
        self.transform_stats = transforms_data.get('stats', {})

        # Build transform counts from stats
        self.transform_counts = {}
        for t in self.transform_vocab:
            if t in self.transform_stats:
                self.transform_counts[t] = self.transform_stats[t].get('frequency', 1)
            else:
                self.transform_counts[t] = 1

        # Load orchestration rules
        orch_file = ckpt.get('orchestration_json_file', [None])[0]
        if orch_file:
            orch_path = os.path.join(os.path.dirname(checkpoint_path), orch_file)
            with open(orch_path) as f:
                self.orchestration_rules = json.load(f)
        else:
            self.orchestration_rules = []

        # Load track info
        track_info_file = ckpt.get('track_info_json_file', [None])[0]
        if track_info_file:
            track_info_path = os.path.join(os.path.dirname(checkpoint_path), track_info_file)
            with open(track_info_path) as f:
                self.track_info = json.load(f)
        else:
            self.track_info = []

        # Load meta-patterns (transform sequences for motivic development)
        meta_json = ckpt.get('meta_patterns_json', [None])[0]
        if meta_json:
            meta_data = json.loads(meta_json)
            self.meta_patterns = meta_data.get('rules', {})
        else:
            self.meta_patterns = {}

        # Stats
        self.n_patterns = int(ckpt['n_patterns'][0])
        self.n_notes = int(ckpt['n_notes'][0])

        print(f"Loaded checkpoint: {self.n_patterns} patterns, {len(self.transform_vocab)} transforms")
        print(f"  Orchestration rules: {len(self.orchestration_rules)}")
        print(f"  Meta-patterns: {len(self.meta_patterns)}")

    def build_pcfg(self):
        """Convert rule counts to probabilities for sampling.

        PHILOSOPHICAL APPROACH:
        - Sample only from HIERARCHICAL patterns (composed grammar rules)
        - Terminal 2-note intervals are building blocks, not standalone phrases
        - Use NATURAL corpus counts (no artificial boosting)
        - The grammar already composed intervals correctly via Re-Pair
        """
        # Separate hierarchical (composed) from terminal (base intervals)
        self.hierarchical_patterns = {}
        self.terminal_patterns = {}

        for rid, r in self.rules.items():
            if r.get('is_hierarchical', False):
                self.hierarchical_patterns[rid] = r
            else:
                self.terminal_patterns[rid] = r

        print(f"  Hierarchical patterns: {len(self.hierarchical_patterns)}")
        print(f"  Terminal intervals: {len(self.terminal_patterns)}")

        # Build probability distribution from HIERARCHICAL patterns only
        # Natural corpus weights - no boosting needed
        total = sum(r['count'] for r in self.hierarchical_patterns.values())
        self.rule_probs = {
            rid: r['count'] / total
            for rid, r in self.hierarchical_patterns.items()
        }
        self.rule_ids = list(self.rule_probs.keys())
        self.rule_weights = list(self.rule_probs.values())

        # Categorize hierarchical patterns by PITCH DIVERSITY and rhythm
        # This is the key insight: melodic = diverse pitches, vamping = repeated pitches
        self.walking_patterns = []  # Steady rhythm, can have repeats (bass)
        self.comping_patterns = []  # Varied rhythm with chords (piano/guitar)
        self.melodic_patterns = []  # Diverse pitches = actual melodies (horns)

        for rid, rule in self.hierarchical_patterns.items():
            rr = rule.get('rhythm_ratios', [])
            pc = rule.get('pitch_classes', [])

            if len(pc) < 3:
                continue  # Skip very short patterns

            # KEY METRIC: pitch diversity
            unique_ratio = len(set(pc)) / len(pc)

            # Check for chords (rr=0 means simultaneous notes)
            has_chords = any(r == 0 for r in rr) if rr else False

            # MELODIC: 50%+ unique pitches AND at least 5 notes (real melodic lines)
            if unique_ratio >= 0.5 and len(pc) >= 5:
                self.melodic_patterns.append(rid)
            # COMPING: has chords OR moderate diversity with syncopation
            elif has_chords or (unique_ratio >= 0.4 and any(0 < r < 0.5 for r in rr)):
                self.comping_patterns.append(rid)
            # WALKING: repeated notes with steady rhythm (bass lines, vamps)
            else:
                self.walking_patterns.append(rid)

        # Build weights for each category using NATURAL corpus counts
        def get_weights(pattern_list):
            weights = [self.hierarchical_patterns[r]['count'] for r in pattern_list]
            if weights:
                total = sum(weights)
                return [w/total for w in weights]
            return []

        self.walking_weights = get_weights(self.walking_patterns)
        self.comping_weights = get_weights(self.comping_patterns)
        self.melodic_weights = get_weights(self.melodic_patterns)

        print(f"  Pattern roles: {len(self.walking_patterns)} walking, "
              f"{len(self.comping_patterns)} comping, {len(self.melodic_patterns)} melodic")

        # Build piece sequences for remix-style generation
        self.build_piece_sequences()

    def build_piece_sequences(self):
        """Build sequences of pattern occurrences per piece/track.

        This enables remix-style generation: sample actual sequences
        from the corpus and replay with variations.

        IMPORTANT: We filter to keep only the LONGEST pattern at each
        onset time, avoiding the overlapping Re-Pair fragments.
        """
        # Group occurrences by (piece_id, track_id)
        raw_sequences = defaultdict(list)

        for rule_id, rule in self.rules.items():
            pattern_len = len(rule.get('pitch_classes', []))
            for occ in rule.get('occurrences', []):
                piece_id = occ.get('piece_id', '')
                track_id = occ.get('track_id', 0)
                onset = occ.get('onset_time', 0)
                position = occ.get('position', 0)

                key = (piece_id, track_id)
                raw_sequences[key].append({
                    'rule_id': rule_id,
                    'onset_time': onset,
                    'position': position,
                    'tau_offset': occ.get('tau_offset', 480),
                    'v_offset': occ.get('v_offset', 80),
                    'octave_transform': occ.get('octave_transform', 0),
                    'pitch_offset': occ.get('pitch_offset', 0),
                    'gm_program': occ.get('gm_program', 0),
                    'pattern_len': pattern_len,
                })

        # For each sequence, keep only the LONGEST pattern at each onset
        # BUT don't skip small patterns if they're all we have at that onset
        self.piece_sequences = {}
        for key, occs in raw_sequences.items():
            # Group by onset time
            by_onset = defaultdict(list)
            for occ in occs:
                by_onset[occ['onset_time']].append(occ)

            # Keep longest pattern at each onset (no minimum length filter)
            filtered = []
            for onset_time in sorted(by_onset.keys()):
                best = max(by_onset[onset_time], key=lambda x: x['pattern_len'])
                filtered.append(best)

            if filtered:
                self.piece_sequences[key] = filtered

        # Get list of pieces with enough content
        self.good_sequences = [
            key for key, seq in self.piece_sequences.items()
            if len(seq) >= 8
        ]

        print(f"  Piece sequences: {len(self.good_sequences)} tracks with 8+ patterns (filtered)")

    def build_transform_sampler(self):
        """Normalize transform usage to probabilities."""
        if not self.transform_counts:
            # Default to identity only
            self.transform_counts = {'identity': 1}

        total = sum(self.transform_counts.values())
        self.transform_probs = {
            t: count / total
            for t, count in self.transform_counts.items()
        }
        self.transform_names = list(self.transform_probs.keys())
        self.transform_weights = list(self.transform_probs.values())

    def build_orchestration_sampler(self):
        """Build orchestration lookup for multitrack generation.

        Orchestration rules tell us: when instrument A plays, instrument B
        plays the same pattern with transform T at some probability.

        We build:
        - orch_by_pair[(src_gm, tgt_gm)] -> list of transforms with weights
        - This lets us query: "What transform does Bass use when Trumpet plays?"
        """
        # Group rules by (source_gm, target_gm) pair
        # Each pair has a distribution of transforms
        self.orch_by_pair = defaultdict(lambda: defaultdict(int))

        for rule in self.orchestration_rules:
            src_gm = rule.get('source_instrument', 0) % 128  # Extract actual GM if encoded
            tgt_gm = rule.get('target_instrument', 0) % 128
            transform = rule.get('transform', 'identity')
            freq = rule.get('frequency', 1)

            self.orch_by_pair[(src_gm, tgt_gm)][transform] += freq

        # Convert to probability distributions
        self.orch_transforms = {}
        for pair, transform_counts in self.orch_by_pair.items():
            total = sum(transform_counts.values())
            transforms = list(transform_counts.keys())
            weights = [transform_counts[t] / total for t in transforms]
            self.orch_transforms[pair] = (transforms, weights)

        # Get common instruments from track_info
        self.instrument_counts = defaultdict(int)
        for info in self.track_info:
            gm = info.get('gm_program', 0)
            self.instrument_counts[gm] += 1

        # Top instruments for generation
        self.common_instruments = sorted(
            self.instrument_counts.keys(),
            key=lambda x: self.instrument_counts[x],
            reverse=True
        )[:10]

    def parse_transform(self, transform_name: str) -> Tuple[int, bool, bool]:
        """
        Parse transform name into (transposition, invert, retrograde).

        Examples:
            'identity' -> (0, False, False)
            'T7' -> (7, False, False)
            'I0' -> (0, True, False)
            'T1∘I4' -> (1, True, False) with I pivot at 4
            'R' -> (0, False, True)
            'T5∘R' -> (5, False, True)

        Returns:
            (semitones, invert, retrograde)
        """
        if transform_name == 'identity':
            return (0, False, False)

        semitones = 0
        invert = False
        retrograde = False

        # Check for retrograde
        if '∘R' in transform_name or transform_name == 'R':
            retrograde = True
            transform_name = transform_name.replace('∘R', '').replace('R', '')

        # Check for inversion
        if 'I' in transform_name:
            invert = True
            # Extract inversion pivot (not used in simple transposition model)
            # For now, treat I_n as inversion around pitch class n

        # Extract transposition
        if 'T' in transform_name:
            # Find T followed by digits
            import re
            match = re.search(r'T(\d+)', transform_name)
            if match:
                semitones = int(match.group(1))

        return (semitones, invert, retrograde)

    def apply_transform(
        self,
        pitch_classes: List[int],
        transform_name: str
    ) -> List[int]:
        """Apply a musical transform to pitch classes."""
        semitones, invert, retrograde = self.parse_transform(transform_name)

        result = list(pitch_classes)

        # Apply retrograde (reverse order)
        if retrograde:
            result = result[::-1]

        # Apply inversion (reflect around 0)
        if invert:
            result = [(12 - pc) % 12 for pc in result]

        # Apply transposition
        result = [(pc + semitones) % 12 for pc in result]

        return result

    def sample_pattern(self, role: str = 'any') -> str:
        """
        Sample a pattern weighted by frequency.

        Args:
            role: 'walking' for bass, 'comping' for piano/guitar,
                  'melodic' for horns, 'any' for all patterns
        """
        if role == 'walking' and self.walking_patterns:
            return random.choices(self.walking_patterns, weights=self.walking_weights)[0]
        elif role == 'comping' and self.comping_patterns:
            return random.choices(self.comping_patterns, weights=self.comping_weights)[0]
        elif role == 'melodic' and self.melodic_patterns:
            return random.choices(self.melodic_patterns, weights=self.melodic_weights)[0]
        else:
            return random.choices(self.rule_ids, weights=self.rule_weights)[0]

    def sample_transform(self) -> str:
        """Sample a transform weighted by usage."""
        return random.choices(self.transform_names, weights=self.transform_weights)[0]

    def sample_orchestration_transform(self, lead_gm: int, companion_gm: int) -> str:
        """
        Sample a transform for companion based on learned orchestration.

        Uses the learned distribution: "When lead_gm plays, companion_gm
        typically plays with transform T."

        Args:
            lead_gm: GM program of lead instrument
            companion_gm: GM program of companion instrument

        Returns:
            Transform name (e.g., 'T7', 'identity')
        """
        pair = (lead_gm, companion_gm)

        if pair in self.orch_transforms:
            transforms, weights = self.orch_transforms[pair]
            return random.choices(transforms, weights=weights)[0]

        # Try reverse pair (symmetric relationship)
        reverse_pair = (companion_gm, lead_gm)
        if reverse_pair in self.orch_transforms:
            transforms, weights = self.orch_transforms[reverse_pair]
            # Invert the transform (T7 <-> T5, etc.)
            sampled = random.choices(transforms, weights=weights)[0]
            return self.invert_transform(sampled)

        # Fallback: use corpus-wide transform distribution
        return self.sample_transform()

    def invert_transform(self, transform: str) -> str:
        """Invert a transform (for symmetric orchestration lookup)."""
        if transform == 'identity':
            return 'identity'
        if transform.startswith('T'):
            # T7 -> T5, T5 -> T7, etc. (12 - n)
            try:
                n = int(transform[1:])
                return f'T{(12 - n) % 12}'
            except:
                return transform
        # For inversions and retrogrades, keep as-is (they're self-inverse)
        return transform

    def pattern_to_notes(
        self,
        pattern: Dict,
        transform: str = 'identity',
        tau_offset: int = 480,
        v_offset: int = 80,
        base_octave: int = 5,
        start_time: int = 0,
    ) -> List[Dict]:
        """
        Convert pattern to MIDI notes using ratios × offsets.

        Args:
            pattern: Pattern dict with pitch_classes, rhythm_ratios, etc.
            transform: Transform to apply (e.g., 'T7', 'I0')
            tau_offset: Base IOI in ticks (480 = quarter note)
            v_offset: Base velocity (0-127)
            base_octave: Octave for pitch (5 = middle C area)
            start_time: Starting time in ticks

        Returns:
            List of note dicts with pitch, velocity, time, duration
        """
        notes = []
        time = start_time

        # Get pattern data
        pitch_classes = pattern.get('pitch_classes', [])
        rhythm_ratios = list(pattern.get('rhythm_ratios', [1.0]))
        duration_ratios = list(pattern.get('duration_ratios', [1.0] * len(pitch_classes)))
        velocity_ratios = list(pattern.get('velocity_ratios', [1.0] * len(pitch_classes)))

        # IMPORTANT: rr=0 means CHORD (simultaneous notes) - preserve this!
        # Only sanitize truly problematic values:
        # - Negative ratios -> 0 (chord)
        # - Huge ratios (>8) -> cap to preserve long phrases but avoid excessive gaps
        for i in range(len(rhythm_ratios)):
            if rhythm_ratios[i] < 0:
                rhythm_ratios[i] = 0  # Treat as chord
            elif rhythm_ratios[i] > 8.0:
                rhythm_ratios[i] = 2.0  # Cap at half note for very long gaps

        # Apply transform to pitch classes
        transformed_pcs = self.apply_transform(pitch_classes, transform)

        # Handle retrograde for timing (reverse ratios too)
        _, _, retrograde = self.parse_transform(transform)
        if retrograde:
            rhythm_ratios = rhythm_ratios[::-1]
            duration_ratios = duration_ratios[::-1]
            velocity_ratios = velocity_ratios[::-1]

        # Convert pitch classes to MIDI pitches using SIGNED intervals
        # Use stored intervals only if they match pattern length, else compute heuristic
        stored_intervals = pattern.get('pitch_intervals', [])
        n_notes = len(transformed_pcs)

        # Check if stored intervals are valid (must have n_notes - 1 intervals)
        use_stored = len(stored_intervals) == n_notes - 1

        prev_pitch = base_octave * 12 + transformed_pcs[0] if transformed_pcs else 60

        for i, pc in enumerate(transformed_pcs):
            if i == 0:
                # First note: base octave + pitch class
                pitch = base_octave * 12 + pc
            else:
                if use_stored:
                    # Use stored TRUE interval from canonical pitches
                    diff = stored_intervals[i - 1]
                else:
                    # Compute signed interval heuristic (smallest interval)
                    prev_pc = transformed_pcs[i - 1]
                    diff = (pc - prev_pc) % 12
                    if diff > 6:
                        diff -= 12

                pitch = prev_pitch + diff

            # Clamp to MIDI range
            while pitch > 96:  # Keep in comfortable range
                pitch -= 12
            while pitch < 36:
                pitch += 12

            prev_pitch = pitch

            # Velocity from ratio
            vel_ratio = velocity_ratios[i] if i < len(velocity_ratios) else 1.0
            velocity = int(vel_ratio * v_offset)
            velocity = min(127, max(1, velocity))

            # Duration from ratio - preserve the learned durations
            dur_ratio = duration_ratios[i] if i < len(duration_ratios) else 1.0
            if dur_ratio <= 0:
                dur_ratio = 0.5
            elif dur_ratio > 8.0:
                dur_ratio = 2.0  # Cap very long notes
            duration = int(dur_ratio * tau_offset)
            duration = max(30, duration)  # 30 ticks min (~1/16 note)

            notes.append({
                'pitch': pitch,
                'velocity': velocity,
                'time': time,
                'duration': duration,
            })

            # Advance time using rhythm ratios
            if i < len(rhythm_ratios):
                time += int(rhythm_ratios[i] * tau_offset)

        return notes

    def generate(
        self,
        length: int = 50,
        seed: Optional[int] = None,
        tau_offset: int = 480,
        v_offset: int = 80,
        base_octave: int = 5,
    ) -> List[Dict]:
        """
        Generate a single track of music.

        Args:
            length: Number of patterns to generate
            seed: Random seed for reproducibility
            tau_offset: Base IOI in ticks
            v_offset: Base velocity
            base_octave: Base octave

        Returns:
            List of note dicts
        """
        if seed is not None:
            random.seed(seed)

        output = []
        current_time = 0
        pattern_sequence = []  # Track which patterns were used

        for _ in range(length):
            # Sample pattern
            pattern_id = self.sample_pattern()
            pattern = self.rules[pattern_id]
            pattern_sequence.append(pattern_id)

            # Sample transform
            transform = self.sample_transform()

            # Convert to notes
            notes = self.pattern_to_notes(
                pattern,
                transform=transform,
                tau_offset=tau_offset,
                v_offset=v_offset,
                base_octave=base_octave,
                start_time=current_time,
            )

            output.extend(notes)

            # Advance time to end of this pattern
            if notes:
                current_time = max(n['time'] + n['duration'] for n in notes)

        self._last_pattern_sequence = pattern_sequence
        return output

    def generate_multitrack(
        self,
        length: int = 50,
        n_tracks: int = 4,
        seed: Optional[int] = None,
    ) -> List[List[Dict]]:
        """
        Generate multitrack music via REMIX: replay corpus sequences with variations.

        REMIX APPROACH:
        1. Pick a SOURCE PIECE from corpus
        2. Get ALL tracks from that piece (preserves timing alignment)
        3. Replay with small variations
        4. Fall back to mixing pieces if needed

        This preserves musical coherence because tracks come from
        the SAME piece with aligned timing.

        Args:
            length: Number of patterns (approximate)
            n_tracks: Number of tracks to generate
            seed: Random seed

        Returns:
            List of tracks, each track is a list of note dicts
        """
        if seed is not None:
            random.seed(seed)

        # === REMIX GENERATION ===
        # Find pieces with multiple tracks for coherent multitrack output

        if not self.good_sequences:
            print("WARNING: No good sequences found, falling back to random generation")
            return self._generate_fallback(length, n_tracks)

        # Group sequences by piece_id to find multitrack pieces
        pieces_with_tracks = defaultdict(list)
        for key in self.good_sequences:
            piece_id, track_id = key
            pieces_with_tracks[piece_id].append(key)

        # Find pieces with enough tracks
        multitrack_pieces = [
            (piece_id, track_keys)
            for piece_id, track_keys in pieces_with_tracks.items()
            if len(track_keys) >= min(n_tracks, 2)
        ]

        if not multitrack_pieces:
            print("WARNING: No multitrack pieces, using random tracks")
            multitrack_pieces = [(None, self.good_sequences[:n_tracks])]

        # Sample a piece
        piece_id, available_tracks = random.choice(multitrack_pieces)
        selected_tracks = random.sample(available_tracks, min(n_tracks, len(available_tracks)))

        # If we don't have enough tracks from one piece, add transposed duplicates
        # (better than mixing unrelated pieces)
        base_tracks = list(selected_tracks)
        while len(selected_tracks) < n_tracks:
            # Reuse existing tracks with different transpose
            selected_tracks.append(base_tracks[len(selected_tracks) % len(base_tracks)])

        print(f"  Using piece: {piece_id or 'mixed'} with {len(selected_tracks)} tracks")

        tracks = []

        # Global transposition for the whole piece (key change)
        global_transpose = random.choice([0, 0, 0, 5, 7])

        # Find common time range across all selected tracks
        # Use the SAME time window for all tracks to ensure alignment
        all_onsets = []
        for seq_key in selected_tracks:
            sequence = self.piece_sequences[seq_key]
            for occ in sequence:
                all_onsets.append(occ['onset_time'])

        if all_onsets:
            min_time = min(all_onsets)
            max_time = max(all_onsets)

            # Pick a random start time within the piece
            target_duration = length * 480 * 4  # Approximate duration (4 beats per pattern)
            if max_time - min_time > target_duration:
                start_time = random.randint(min_time, max(min_time, max_time - target_duration))
            else:
                start_time = min_time
            end_time = start_time + target_duration
        else:
            start_time, end_time = 0, 100000

        # Per-track transposes for variety (especially for duplicated tracks)
        track_transposes = [0, 0, 7, 5][:n_tracks]  # Identity, identity, 5th up, 4th up

        for track_idx, seq_key in enumerate(selected_tracks):
            sequence = self.piece_sequences[seq_key]

            # Filter to patterns within the chosen time window
            slice_seq = [
                occ for occ in sequence
                if start_time <= occ['onset_time'] <= end_time
            ][:length]  # Limit number of patterns

            if not slice_seq:
                continue

            track_notes = []
            # Normalize to window start time (not first occurrence in this track)
            # This ensures all tracks align to the same time origin
            base_time = start_time

            # Per-track transpose
            track_transpose = track_transposes[track_idx % len(track_transposes)]

            for occ in slice_seq:
                rule_id = occ['rule_id']
                if rule_id not in self.rules:
                    continue

                pattern = self.rules[rule_id]

                # Use ORIGINAL timing from corpus
                onset = occ['onset_time'] - base_time
                tau = occ.get('tau_offset', 480)

                # v_offset is stored as a small ratio (1-7), scale to MIDI velocity
                v_ratio = occ.get('v_offset', 5)
                vel = int(v_ratio * 15 + 50)  # Scale: 1->65, 5->125, 7->155->127
                vel = min(127, max(40, vel))

                # Compute octave from original occurrence
                octave_transform = occ.get('octave_transform', 0)
                base_octave = 5 + (octave_transform // 12)  # Adjust from canonical

                # Apply global transpose + track transpose + original pitch offset
                pitch_offset = occ.get('pitch_offset', 0)
                total_transpose = (global_transpose + track_transpose + pitch_offset) % 12
                transform = 'identity' if total_transpose == 0 else f'T{total_transpose}'

                notes = self.pattern_to_notes(
                    pattern,
                    transform=transform,
                    tau_offset=tau,
                    v_offset=vel,
                    base_octave=base_octave,
                    start_time=onset,
                )
                track_notes.extend(notes)

            tracks.append(track_notes)

        return tracks

    def _generate_fallback(self, length: int, n_tracks: int) -> List[List[Dict]]:
        """Fallback generation if no sequences available."""
        tracks = []
        for _ in range(n_tracks):
            track_notes = []
            current_time = 0
            for _ in range(length):
                pattern_id = self.sample_pattern()
                pattern = self.rules[pattern_id]
                notes = self.pattern_to_notes(pattern, start_time=current_time)
                track_notes.extend(notes)
                if notes:
                    current_time = max(n['time'] + n['duration'] for n in notes)
            tracks.append(track_notes)
        return tracks


    def compose_transforms(self, t1: str, t2: str) -> str:
        """
        Compose two transforms: result = t1 ∘ t2

        For transpositions: T_a ∘ T_b = T_{(a+b) mod 12}
        """
        # Parse t1
        sem1, inv1, ret1 = self.parse_transform(t1)
        sem2, inv2, ret2 = self.parse_transform(t2)

        # Compose transpositions (simple addition mod 12)
        total_sem = (sem1 + sem2) % 12

        # For now, just handle transposition composition
        # Full composition of inversions/retrogrades is more complex
        if total_sem == 0 and not inv1 and not inv2 and not ret1 and not ret2:
            return 'identity'

        return f'T{total_sem}'

    def to_midi(
        self,
        notes: List[Dict],
        output_path: str,
        ticks_per_beat: int = 480,
        tempo_bpm: int = 120,
    ):
        """
        Save notes to MIDI file.

        Args:
            notes: List of note dicts with pitch, velocity, time, duration
            output_path: Path to save MIDI file
            ticks_per_beat: MIDI resolution
            tempo_bpm: Tempo in BPM
        """
        try:
            import mido
        except ImportError:
            print("Error: mido not installed. Run: pip install mido")
            return

        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Set tempo
        tempo = mido.bpm2tempo(tempo_bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

        # Sort notes by time
        sorted_notes = sorted(notes, key=lambda n: (n['time'], n['pitch']))

        # Build events (note_on and note_off)
        events = []
        for note in sorted_notes:
            events.append((note['time'], 'note_on', note['pitch'], note['velocity']))
            events.append((note['time'] + note['duration'], 'note_off', note['pitch'], 0))

        # Sort by time
        events.sort(key=lambda e: (e[0], 0 if e[1] == 'note_off' else 1))

        # Convert to delta times
        prev_time = 0
        for event in events:
            abs_time, msg_type, pitch, velocity = event
            delta = abs_time - prev_time

            if msg_type == 'note_on':
                track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=delta))
            else:
                track.append(mido.Message('note_off', note=pitch, velocity=0, time=delta))

            prev_time = abs_time

        # End of track
        track.append(mido.MetaMessage('end_of_track', time=0))

        mid.save(output_path)
        print(f"Saved MIDI to: {output_path}")

    def to_midi_multitrack(
        self,
        tracks: List[List[Dict]],
        output_path: str,
        ticks_per_beat: int = 480,
        tempo_bpm: int = 120,
        instruments: Optional[List[int]] = None,
    ):
        """
        Save multitrack to MIDI file.

        Args:
            tracks: List of tracks, each track is list of note dicts
            output_path: Path to save MIDI file
            ticks_per_beat: MIDI resolution
            tempo_bpm: Tempo in BPM
            instruments: GM program numbers for each track
        """
        try:
            import mido
        except ImportError:
            print("Error: mido not installed. Run: pip install mido")
            return

        if instruments is None:
            # Big band arrangement:
            # Rhythm section: Piano(0), Bass(32), Guitar(25), Drums(channel 9)
            # Horn section: Trumpet(56), Alto Sax(65), Tenor Sax(66), Trombone(57), Baritone Sax(67)
            instruments = [
                0,    # Piano
                32,   # Acoustic Bass
                25,   # Steel Guitar
                56,   # Trumpet
                65,   # Alto Sax
                66,   # Tenor Sax
                57,   # Trombone
                67,   # Baritone Sax
                56,   # Trumpet 2
                57,   # Trombone 2
            ]

        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=1)

        # Tempo track
        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo = mido.bpm2tempo(tempo_bpm)
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        tempo_track.append(mido.MetaMessage('end_of_track', time=0))

        # Add each track
        for i, notes in enumerate(tracks):
            track = mido.MidiTrack()
            mid.tracks.append(track)

            # Set instrument
            program = instruments[i % len(instruments)]
            track.append(mido.Message('program_change', program=program, time=0))

            # Sort notes
            sorted_notes = sorted(notes, key=lambda n: (n['time'], n['pitch']))

            # Build events
            events = []
            for note in sorted_notes:
                events.append((note['time'], 'note_on', note['pitch'], note['velocity']))
                events.append((note['time'] + note['duration'], 'note_off', note['pitch'], 0))

            events.sort(key=lambda e: (e[0], 0 if e[1] == 'note_off' else 1))

            # Convert to delta times
            prev_time = 0
            for event in events:
                abs_time, msg_type, pitch, velocity = event
                delta = abs_time - prev_time

                if msg_type == 'note_on':
                    track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=delta))
                else:
                    track.append(mido.Message('note_off', note=pitch, velocity=0, time=delta))

                prev_time = abs_time

            track.append(mido.MetaMessage('end_of_track', time=0))

        mid.save(output_path)
        print(f"Saved multitrack MIDI to: {output_path}")
        print(f"  {len(tracks)} tracks, {sum(len(t) for t in tracks)} notes")


def main():
    parser = argparse.ArgumentParser(description='Generate music from genome patterns')
    parser.add_argument('checkpoint', help='Path to checkpoint .npz file')
    parser.add_argument('--output', '-o', default='generated.mid', help='Output MIDI path')
    parser.add_argument('--length', '-l', type=int, default=50, help='Number of patterns')
    parser.add_argument('--seed', '-s', type=int, help='Random seed')
    parser.add_argument('--tempo', '-t', type=int, default=120, help='Tempo BPM')
    parser.add_argument('--multitrack', '-m', action='store_true', help='Generate multitrack')
    parser.add_argument('--tracks', type=int, default=4, help='Number of tracks (with -m)')
    args = parser.parse_args()

    print("=" * 60)
    print("GENOME GENERATOR")
    print("=" * 60)

    # Load generator
    gen = GenomeGenerator(args.checkpoint)

    # Generate
    print(f"\nGenerating {args.length} patterns...")

    if args.multitrack:
        tracks = gen.generate_multitrack(
            length=args.length,
            n_tracks=args.tracks,
            seed=args.seed,
        )
        gen.to_midi_multitrack(tracks, args.output, tempo_bpm=args.tempo)
    else:
        notes = gen.generate(length=args.length, seed=args.seed)
        gen.to_midi(notes, args.output, tempo_bpm=args.tempo)
        print(f"  Generated {len(notes)} notes")

    print("\nDone!")


if __name__ == '__main__':
    main()
