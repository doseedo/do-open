#!/usr/bin/env python3
"""
Meta-Pattern Generator (v53 Checkpoint)
========================================

FULL CHECKPOINT UTILIZATION:
1. Meta-patterns → Form structure (transform sequences)
2. Transform graph → Pattern relationships
3. Hierarchical patterns → Phrase structure
4. Vertical slices → Harmonic coherence
5. Orchestration rules → Instrument relationships

This generator uses ALL learned structure, not just pattern sampling.

Usage:
    python scripts/meta_pattern_generator.py checkpoint.npz -o output.mid --bars 32
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

# GM instrument ranges
GM_RANGES = {
    32: (28, 55), 33: (28, 55), 34: (28, 55),  # Bass
    0: (36, 96), 1: (36, 96), 4: (36, 96),     # Piano
    56: (52, 84), 57: (40, 72), 58: (36, 60),  # Brass
    64: (44, 80), 65: (49, 81), 66: (42, 75), 67: (36, 69),  # Sax
}
DEFAULT_RANGE = (48, 84)


@dataclass
class TransformRelation:
    """A learned transform relationship between patterns."""
    source_id: str
    target_id: str
    transform: str
    count: int = 1


class MetaPatternGenerator:
    """Generate using full checkpoint structure: meta-patterns, hierarchy, slices."""

    def __init__(self, checkpoint_path: str, verbose: bool = True):
        self.verbose = verbose
        self.load_checkpoint(checkpoint_path)
        self.build_transform_graph()
        self.build_hierarchical_index()

    def load_checkpoint(self, checkpoint_path: str):
        """Load all checkpoint components."""
        if self.verbose:
            print(f"Loading checkpoint: {checkpoint_path}")

        ckpt = np.load(checkpoint_path, allow_pickle=True)
        base_path = checkpoint_path.replace('.npz', '')

        # 1. Load patterns
        patterns_file = ckpt.get('patterns_json_file', [None])[0]
        if patterns_file:
            patterns_path = os.path.join(os.path.dirname(checkpoint_path), patterns_file)
            with open(patterns_path) as f:
                self.patterns = json.load(f)
        else:
            raise ValueError("No patterns in checkpoint")

        # 2. Load transforms
        transforms_file = ckpt.get('transforms_json_file', [None])[0]
        self.transform_vocab = []
        self.transform_relations = []
        if transforms_file:
            transforms_path = os.path.join(os.path.dirname(checkpoint_path), transforms_file)
            if os.path.exists(transforms_path):
                with open(transforms_path) as f:
                    transforms_data = json.load(f)
                    self.transform_vocab = transforms_data.get('vocabulary', [])
                    self.transform_relations = transforms_data.get('relations', [])

        # 3. Load meta-patterns (FORM STRUCTURE)
        self.meta_patterns = []
        meta_file = ckpt.get('meta_patterns_json_file', [None])[0]
        if meta_file:
            meta_path = os.path.join(os.path.dirname(checkpoint_path), meta_file)
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta_data = json.load(f)
                    self.meta_patterns = meta_data.get('interpreted', [])
                    self.orchestration_rules = meta_data.get('orchestration_rules', [])

        # 4. Load track derives
        track_derives_file = ckpt.get('track_derives_json_file', [None])[0]
        self.track_derives = []
        if track_derives_file:
            td_path = os.path.join(os.path.dirname(checkpoint_path), track_derives_file)
            if os.path.exists(td_path):
                with open(td_path) as f:
                    td_data = json.load(f)
                    self.track_derives = td_data.get('derives', [])

        if self.verbose:
            print(f"  Patterns: {len(self.patterns)}")
            print(f"  Transforms: {len(self.transform_vocab)}")
            print(f"  Meta-patterns: {len(self.meta_patterns)}")
            print(f"  Orchestration rules: {len(getattr(self, 'orchestration_rules', []))}")
            print(f"  Track derives: {len(self.track_derives)}")

    def build_transform_graph(self):
        """Build graph: pattern_id → {transform: [target_pattern_ids]}

        This enables walking the learned relationships, not computing new ones.
        """
        if self.verbose:
            print("Building transform graph...")

        # Graph structure: source -> transform -> [targets]
        self.transform_graph = defaultdict(lambda: defaultdict(list))

        # From explicit relations
        for rel in self.transform_relations:
            src = str(rel.get('source', rel.get('source_id', '')))
            tgt = str(rel.get('target', rel.get('target_id', '')))
            t = rel.get('transform', 'identity')
            if src and tgt:
                self.transform_graph[src][t].append(tgt)

        # Also build from occurrence-level derived_from
        for pattern_id, pattern in self.patterns.items():
            for occ in pattern.get('occurrences', []):
                derived = occ.get('derived_from')
                if derived:
                    src_pattern = derived.get('source_pattern_id')
                    transform = derived.get('transform', 'identity')
                    if src_pattern:
                        self.transform_graph[str(src_pattern)][transform].append(pattern_id)

        n_edges = sum(
            len(targets)
            for transforms in self.transform_graph.values()
            for targets in transforms.values()
        )

        if self.verbose:
            print(f"  Transform graph: {len(self.transform_graph)} sources, {n_edges} edges")

    def build_hierarchical_index(self):
        """Index hierarchical patterns by depth for phrase-aware sampling."""
        if self.verbose:
            print("Building hierarchical index...")

        self.patterns_by_depth = defaultdict(list)
        self.pattern_children = {}  # pattern_id -> (left, right, connector)

        for pattern_id, pattern in self.patterns.items():
            if pattern.get('is_hierarchical'):
                left = pattern.get('left_child')
                right = pattern.get('right_child')
                connector = pattern.get('connector_interval', 0)
                self.pattern_children[pattern_id] = (left, right, connector)

            # Estimate depth from pattern length or hierarchy
            depth = self._estimate_depth(pattern_id, pattern)
            self.patterns_by_depth[depth].append(pattern_id)

        if self.verbose:
            for depth in sorted(self.patterns_by_depth.keys())[:5]:
                print(f"  Depth {depth}: {len(self.patterns_by_depth[depth])} patterns")

    def _estimate_depth(self, pattern_id: str, pattern: dict, memo: dict = None) -> int:
        """Estimate hierarchical depth of a pattern."""
        if memo is None:
            memo = {}
        if pattern_id in memo:
            return memo[pattern_id]

        if not pattern.get('is_hierarchical'):
            memo[pattern_id] = 0
            return 0

        left = pattern.get('left_child')
        right = pattern.get('right_child')

        left_depth = 0
        right_depth = 0

        if left and str(left) in self.patterns:
            left_depth = self._estimate_depth(str(left), self.patterns[str(left)], memo)
        if right and str(right) in self.patterns:
            right_depth = self._estimate_depth(str(right), self.patterns[str(right)], memo)

        depth = 1 + max(left_depth, right_depth)
        memo[pattern_id] = depth
        return depth

    def sample_meta_pattern(self) -> Optional[Dict]:
        """Sample a meta-pattern weighted by frequency."""
        if not self.meta_patterns:
            return None

        # Weight by count if available
        weights = []
        for mp in self.meta_patterns:
            if isinstance(mp, dict):
                weights.append(mp.get('count', 1))
            else:
                weights.append(1)

        return random.choices(self.meta_patterns, weights=weights)[0]

    def extract_transform_sequence(self, meta_pattern) -> List[str]:
        """Extract transform sequence from meta-pattern."""
        if isinstance(meta_pattern, dict):
            # Could be {'transforms': [...]} or {'description': 'T5 → T7 → identity'}
            if 'transforms' in meta_pattern:
                return meta_pattern['transforms']
            if 'description' in meta_pattern:
                # Parse "T5 → T7 → identity"
                desc = meta_pattern['description']
                return [t.strip() for t in desc.replace('→', ',').split(',')]
            if 'sequence' in meta_pattern:
                return meta_pattern['sequence']
        elif isinstance(meta_pattern, list):
            return meta_pattern
        elif isinstance(meta_pattern, str):
            return [t.strip() for t in meta_pattern.replace('→', ',').split(',')]

        return ['identity']  # Fallback

    def find_pattern_by_transform(self, source_id: str, transform: str) -> Optional[str]:
        """Find a pattern related to source by the given transform.

        Uses the learned transform graph - doesn't compute new transforms.
        """
        # Direct lookup in transform graph
        if source_id in self.transform_graph:
            targets = self.transform_graph[source_id].get(transform, [])
            if targets:
                return random.choice(targets)

        # Try similar transforms (e.g., T5 ≈ T7 for rough approximation)
        if transform.startswith('T') and len(transform) > 1:
            for alt_t in ['T5', 'T7', 'T2', 'T10', 'identity']:
                targets = self.transform_graph[source_id].get(alt_t, [])
                if targets:
                    return random.choice(targets)

        # Fallback: sample any pattern with high count
        return self.sample_high_count_pattern()

    def sample_high_count_pattern(self, min_depth: int = 0) -> str:
        """Sample a pattern weighted by corpus frequency."""
        candidates = []
        weights = []

        for pattern_id, pattern in self.patterns.items():
            if min_depth > 0:
                depth = self._estimate_depth(pattern_id, pattern)
                if depth < min_depth:
                    continue

            count = pattern.get('count', 1)
            candidates.append(pattern_id)
            weights.append(count)

        if not candidates:
            return list(self.patterns.keys())[0]

        return random.choices(candidates, weights=weights)[0]

    def sample_phrase_pattern(self, target_depth: int = 2) -> str:
        """Sample a hierarchical pattern at approximately the target depth."""
        # Get patterns at or near target depth
        candidates = []
        for d in range(max(0, target_depth - 1), target_depth + 2):
            candidates.extend(self.patterns_by_depth.get(d, []))

        if not candidates:
            return self.sample_high_count_pattern()

        # Weight by count
        weights = [self.patterns.get(pid, {}).get('count', 1) for pid in candidates]
        return random.choices(candidates, weights=weights)[0]

    def expand_hierarchical(self, pattern_id: str, memo: dict = None) -> List[int]:
        """Recursively expand a hierarchical pattern to intervals.

        This respects phrase structure - patterns are expanded through
        the grammar hierarchy, not flattened.
        """
        if memo is None:
            memo = {}
        if pattern_id in memo:
            return memo[pattern_id]

        pattern = self.patterns.get(pattern_id, {})

        if pattern_id not in self.pattern_children:
            # Terminal or non-hierarchical - return intervals directly
            intervals = pattern.get('pitch_intervals', [])
            memo[pattern_id] = intervals
            return intervals

        # Hierarchical - expand children with connector
        left_id, right_id, connector = self.pattern_children[pattern_id]

        left_intervals = self.expand_hierarchical(str(left_id), memo) if left_id else []
        right_intervals = self.expand_hierarchical(str(right_id), memo) if right_id else []

        # Combine: left + connector + right
        result = list(left_intervals) + [connector] + list(right_intervals)
        memo[pattern_id] = result
        return result

    def generate_with_meta_patterns(
        self,
        n_sections: int = 8,
        instruments: List[int] = None,
        ticks_per_beat: int = 480,
        seed: int = None,
    ) -> Dict[int, List[Dict]]:
        """Generate using meta-patterns for form structure.

        Algorithm:
        1. Sample meta-pattern (transform sequence = form)
        2. Sample seed pattern (hierarchical for phrase structure)
        3. Walk transform graph following meta-pattern
        4. Expand patterns through hierarchy
        5. Orchestrate across instruments
        """
        if seed is not None:
            random.seed(seed)

        if instruments is None:
            instruments = [56, 65, 66, 57, 0, 32]  # Trumpet, Alto, Tenor, Trombone, Piano, Bass

        if self.verbose:
            print(f"\nGenerating with meta-patterns...")
            print(f"  Sections: {n_sections}")
            print(f"  Instruments: {instruments}")

        all_tracks = defaultdict(list)
        current_time = 0

        for section_idx in range(n_sections):
            # 1. SAMPLE META-PATTERN (form structure)
            meta = self.sample_meta_pattern()
            transform_seq = self.extract_transform_sequence(meta)

            if self.verbose and section_idx == 0:
                print(f"  Section 0 meta-pattern: {transform_seq[:5]}...")

            # 2. SAMPLE SEED PATTERN (phrase-level, hierarchical)
            seed_pattern_id = self.sample_phrase_pattern(target_depth=2)

            # 3. WALK TRANSFORM GRAPH
            pattern_chain = [seed_pattern_id]
            current_pattern = seed_pattern_id

            for transform in transform_seq[:8]:  # Limit chain length
                next_pattern = self.find_pattern_by_transform(current_pattern, transform)
                if next_pattern:
                    pattern_chain.append(next_pattern)
                    current_pattern = next_pattern

            # 4. EXPAND AND ORCHESTRATE
            for pattern_id in pattern_chain:
                pattern = self.patterns.get(pattern_id, {})

                # Get intervals (through hierarchy if available)
                intervals = self.expand_hierarchical(pattern_id)
                if not intervals:
                    intervals = pattern.get('pitch_intervals', [0])

                # Get timing from occurrences
                occs = pattern.get('occurrences', [])
                base_ioi = 480
                if occs:
                    tau_offsets = [o.get('tau_offset', 480) for o in occs if o.get('tau_offset', 0) > 0]
                    if tau_offsets:
                        base_ioi = random.choice(tau_offsets)

                # Separate instruments by ROLE
                # Horn section: can derive from each other (harmonize)
                # Rhythm section: must play their OWN patterns (different role)
                HORN_SECTION = {56, 57, 58, 59, 60, 61, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73}  # Brass + Sax + Woodwinds
                RHYTHM_SECTION = {0, 1, 2, 3, 4, 5, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39}  # Piano + Guitar + Bass

                horn_instruments = [gm for gm in instruments if gm in HORN_SECTION]
                rhythm_instruments = [gm for gm in instruments if gm in RHYTHM_SECTION]
                other_instruments = [gm for gm in instruments if gm not in HORN_SECTION and gm not in RHYTHM_SECTION]

                # HORN SECTION: Lead + derived harmonies (they CAN share patterns)
                if horn_instruments:
                    lead_gm = horn_instruments[0]
                    lead_pitch = self._sample_pitch_for_instrument(pattern, lead_gm)

                    lead_notes = self._expand_to_notes(
                        intervals, lead_pitch, current_time, lead_gm, base_ioi
                    )
                    all_tracks[lead_gm].extend(lead_notes)

                    # Other horns: derive from lead (harmonize)
                    for follower_gm in horn_instruments[1:]:
                        rule = self._find_orchestration_rule(lead_gm, follower_gm)

                        if rule:
                            transform = rule.get('transform', 'identity')
                        else:
                            transform = 'identity'

                        follower_pitch = self._sample_pitch_for_instrument(pattern, follower_gm)

                        follower_intervals = intervals
                        if transform == 'R':
                            follower_intervals = list(reversed(intervals))
                        elif transform.startswith('I'):
                            follower_intervals = [-i for i in intervals]

                        follower_notes = self._expand_to_notes(
                            follower_intervals, follower_pitch, current_time, follower_gm, base_ioi
                        )
                        all_tracks[follower_gm].extend(follower_notes)

                # RHYTHM SECTION: Sample patterns that CO-OCCURRED with the lead pattern
                # This ensures harmonic compatibility (bass/piano match the melody key)
                for rhythm_gm in rhythm_instruments:
                    # Find patterns this instrument played WHEN the lead pattern was playing
                    rhythm_pattern_id = self._sample_cooccurring_pattern(
                        lead_pattern_id=seed_pattern_id if horn_instruments else pattern_id,
                        target_gm=rhythm_gm
                    )

                    if not rhythm_pattern_id:
                        # Fallback: sample any pattern for this instrument
                        rhythm_pattern_id = self._sample_pattern_for_instrument(rhythm_gm)

                    if rhythm_pattern_id:
                        rhythm_pattern = self.patterns.get(rhythm_pattern_id, {})
                        rhythm_intervals = rhythm_pattern.get('pitch_intervals', [0])
                        rhythm_pitch = self._sample_pitch_for_instrument(rhythm_pattern, rhythm_gm)

                        # Get timing from this pattern's occurrences
                        rhythm_ioi = base_ioi
                        for occ in rhythm_pattern.get('occurrences', []):
                            if occ.get('gm_program') == rhythm_gm:
                                rhythm_ioi = occ.get('tau_offset', base_ioi)
                                break

                        rhythm_notes = self._expand_to_notes(
                            rhythm_intervals, rhythm_pitch, current_time, rhythm_gm, rhythm_ioi
                        )
                        all_tracks[rhythm_gm].extend(rhythm_notes)

                # OTHER INSTRUMENTS: Sample their own patterns too
                for other_gm in other_instruments:
                    other_pattern_id = self._sample_pattern_for_instrument(other_gm)
                    if other_pattern_id:
                        other_pattern = self.patterns.get(other_pattern_id, {})
                        other_intervals = other_pattern.get('pitch_intervals', [0])
                        other_pitch = self._sample_pitch_for_instrument(other_pattern, other_gm)
                        other_notes = self._expand_to_notes(
                            other_intervals, other_pitch, current_time, other_gm, base_ioi
                        )
                        all_tracks[other_gm].extend(other_notes)

                # Advance time
                pattern_duration = base_ioi * (len(intervals) + 1)
                current_time += pattern_duration

        if self.verbose:
            print(f"\nGenerated:")
            for gm, notes in sorted(all_tracks.items()):
                if notes:
                    avg_pitch = np.mean([n['pitch'] for n in notes])
                    print(f"  GM {gm}: {len(notes)} notes, avg_pitch={avg_pitch:.0f}")

        return dict(all_tracks)

    def _find_orchestration_rule(self, lead_gm: int, follower_gm: int) -> Optional[Dict]:
        """Find learned orchestration rule for instrument pair."""
        for rule in getattr(self, 'orchestration_rules', []):
            src = rule.get('source_instrument', -1)
            tgt = rule.get('target_instrument', -1)
            if src == lead_gm and tgt == follower_gm:
                return rule
        return None

    def _sample_pattern_for_instrument(self, gm_program: int) -> Optional[str]:
        """Sample a pattern that this specific instrument plays.

        This ensures each instrument plays patterns appropriate to its ROLE:
        - Bass plays bass patterns (walking lines)
        - Piano plays piano patterns (comping)
        - Horns play horn patterns (melodies)
        """
        # Build per-instrument vocabulary on first call
        if not hasattr(self, '_instrument_patterns'):
            self._instrument_patterns = defaultdict(list)
            self._instrument_weights = defaultdict(list)

            for pattern_id, pattern in self.patterns.items():
                # Group by which instruments play this pattern
                gm_counts = defaultdict(int)
                for occ in pattern.get('occurrences', []):
                    gm = occ.get('gm_program', 0)
                    gm_counts[gm] += 1

                # Add to each instrument's vocabulary
                for gm, count in gm_counts.items():
                    self._instrument_patterns[gm].append(pattern_id)
                    self._instrument_weights[gm].append(count)

        # Sample from this instrument's vocabulary
        patterns = self._instrument_patterns.get(gm_program, [])
        weights = self._instrument_weights.get(gm_program, [])

        if not patterns:
            return None

        return random.choices(patterns, weights=weights)[0]

    def _sample_cooccurring_pattern(self, lead_pattern_id: str, target_gm: int) -> Optional[str]:
        """Sample a pattern that CO-OCCURRED with the lead pattern in the corpus.

        This ensures rhythm section (bass, piano) harmonically matches the melody
        by sampling from patterns that ACTUALLY played together in the training data.

        The co-occurrence index is built from occurrences grouped by (piece_id, onset_time).
        """
        # Build co-occurrence index on first call
        if not hasattr(self, '_cooccurrence_index'):
            if self.verbose:
                print("Building co-occurrence index...")

            # Group occurrences by (piece_id, time_bucket) -> list of (pattern_id, gm)
            time_slices = defaultdict(list)

            for pattern_id, pattern in self.patterns.items():
                for occ in pattern.get('occurrences', []):
                    piece_id = occ.get('piece_id', 'unknown')
                    onset = occ.get('onset', 0)
                    gm = occ.get('gm_program', 0)

                    # Bucket time to 1 beat (480 ticks) for co-occurrence
                    time_bucket = onset // 480

                    time_slices[(piece_id, time_bucket)].append((pattern_id, gm))

            # Build index: lead_pattern -> target_gm -> [cooccurring_patterns]
            self._cooccurrence_index = defaultdict(lambda: defaultdict(list))
            self._cooccurrence_weights = defaultdict(lambda: defaultdict(list))

            for (piece_id, time_bucket), pattern_gm_list in time_slices.items():
                if len(pattern_gm_list) < 2:
                    continue  # Need at least 2 patterns for co-occurrence

                # For each pattern in this time slice, record co-occurrences
                for i, (pattern_id, gm) in enumerate(pattern_gm_list):
                    for j, (other_pattern_id, other_gm) in enumerate(pattern_gm_list):
                        if i != j:
                            # pattern_id co-occurred with other_pattern_id (played by other_gm)
                            self._cooccurrence_index[pattern_id][other_gm].append(other_pattern_id)
                            # Weight by pattern count
                            other_pattern = self.patterns.get(other_pattern_id, {})
                            weight = other_pattern.get('count', 1)
                            self._cooccurrence_weights[pattern_id][other_gm].append(weight)

            if self.verbose:
                n_leads = len(self._cooccurrence_index)
                total_cooc = sum(
                    len(pids)
                    for gm_map in self._cooccurrence_index.values()
                    for pids in gm_map.values()
                )
                print(f"  Co-occurrence index: {n_leads} lead patterns, {total_cooc} co-occurrences")

        # Look up co-occurring patterns for this lead + target instrument
        lead_key = str(lead_pattern_id)
        cooccurring = self._cooccurrence_index.get(lead_key, {}).get(target_gm, [])
        weights = self._cooccurrence_weights.get(lead_key, {}).get(target_gm, [])

        if cooccurring and weights:
            return random.choices(cooccurring, weights=weights)[0]

        # Fallback: no co-occurrence data, return None (caller will use _sample_pattern_for_instrument)
        return None

    def _sample_pitch_for_instrument(self, pattern: dict, gm_program: int) -> int:
        """Sample an appropriate starting pitch for the instrument."""
        pitch_range = GM_RANGES.get(gm_program, DEFAULT_RANGE)

        # Try to find this instrument in pattern occurrences
        for occ in pattern.get('occurrences', []):
            if occ.get('gm_program') == gm_program:
                pitch = occ.get('first_pitch', 60)
                # Verify in range
                if pitch_range[0] <= pitch <= pitch_range[1]:
                    return pitch

        # Fallback: middle of range
        return (pitch_range[0] + pitch_range[1]) // 2

    def _expand_to_notes(
        self,
        intervals: List[int],
        first_pitch: int,
        start_time: int,
        gm_program: int,
        base_ioi: int,
    ) -> List[Dict]:
        """Expand intervals to note events."""
        notes = []
        pitch_range = GM_RANGES.get(gm_program, DEFAULT_RANGE)

        current_pitch = first_pitch
        current_time = start_time
        velocity = 80

        # First note
        notes.append({
            'pitch': max(pitch_range[0], min(pitch_range[1], current_pitch)),
            'velocity': velocity,
            'time': current_time,
            'duration': base_ioi,
        })

        # Remaining notes
        for interval in intervals:
            current_time += base_ioi
            current_pitch += interval

            # Octave fold to stay in range
            while current_pitch < pitch_range[0]:
                current_pitch += 12
            while current_pitch > pitch_range[1]:
                current_pitch -= 12

            notes.append({
                'pitch': current_pitch,
                'velocity': velocity,
                'time': current_time,
                'duration': base_ioi,
            })

        return notes

    def to_midi(
        self,
        tracks: Dict[int, List[Dict]],
        output_path: str,
        tempo: int = 120,
        ticks_per_beat: int = 480,
    ):
        """Save to MIDI file."""
        try:
            import mido
        except ImportError:
            print("Error: mido not installed. Run: pip install mido")
            return

        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=1)

        # Tempo track
        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))
        tempo_track.append(mido.MetaMessage('end_of_track', time=0))

        channel_map = {}
        next_channel = 0

        for gm, notes in sorted(tracks.items()):
            if not notes:
                continue

            track = mido.MidiTrack()
            mid.tracks.append(track)

            if gm not in channel_map:
                if next_channel == 9:
                    next_channel = 10
                channel_map[gm] = next_channel % 16
                next_channel += 1
            channel = channel_map[gm]

            track.append(mido.Message('program_change', program=gm % 128,
                                     channel=channel, time=0))

            events = []
            for note in notes:
                events.append((note['time'], 'note_on', note['pitch'],
                              note['velocity'], channel))
                events.append((note['time'] + note['duration'], 'note_off',
                              note['pitch'], 0, channel))

            events.sort(key=lambda e: (e[0], 0 if e[1] == 'note_off' else 1))

            prev_time = 0
            for event in events:
                abs_time, msg_type, pitch, velocity, ch = event
                delta = abs_time - prev_time

                if msg_type == 'note_on':
                    track.append(mido.Message('note_on', note=pitch, velocity=velocity,
                                             channel=ch, time=delta))
                else:
                    track.append(mido.Message('note_off', note=pitch, velocity=0,
                                             channel=ch, time=delta))
                prev_time = abs_time

            track.append(mido.MetaMessage('end_of_track', time=0))

        mid.save(output_path)

        if self.verbose:
            total_notes = sum(len(notes) for notes in tracks.values())
            print(f"\nSaved MIDI to: {output_path}")
            print(f"  {len(tracks)} tracks, {total_notes} notes")


def main():
    parser = argparse.ArgumentParser(description='Meta-pattern guided generator')
    parser.add_argument('checkpoint', help='Path to v53 checkpoint .npz file')
    parser.add_argument('--output', '-o', default='meta_generated.mid', help='Output MIDI path')
    parser.add_argument('--sections', '-n', type=int, default=8, help='Number of sections')
    parser.add_argument('--tempo', '-t', type=int, default=120, help='Tempo BPM')
    parser.add_argument('--seed', '-s', type=int, help='Random seed')
    parser.add_argument('--instruments', '-i', help='Comma-separated GM program numbers')
    args = parser.parse_args()

    print("=" * 60)
    print("META-PATTERN GENERATOR (Full Checkpoint Utilization)")
    print("=" * 60)

    gen = MetaPatternGenerator(args.checkpoint)

    instruments = None
    if args.instruments:
        instruments = [int(x.strip()) for x in args.instruments.split(',')]

    tracks = gen.generate_with_meta_patterns(
        n_sections=args.sections,
        instruments=instruments,
        seed=args.seed,
    )

    gen.to_midi(tracks, args.output, tempo=args.tempo)

    print("\nDone!")


if __name__ == '__main__':
    main()
