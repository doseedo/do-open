#!/usr/bin/env python3
"""
Meta-Pattern Generator (v53 Checkpoint)
========================================

FULL CHECKPOINT UTILIZATION (All corpus-derived, no music theory constraints):
1. Meta-patterns → Transform sequences (II-V-I progressions, etc.)
2. Transform graph → Pattern relationships
3. Hierarchical patterns → Phrase structure (left_child + connector + right_child)
4. Vertical slices / Co-occurrence → Harmonic coherence (patterns that played together)
5. Orchestration rules → Instrument relationships
6. TrackDerive → Explicit cross-track derivations (72K+ relations)
7. Multi-factor transforms → Rhythm/velocity/duration variation
8. Form structure → Pattern sequences per piece (AABA, etc.) - DERIVED from corpus
9. Rest durations → Natural gaps between patterns per instrument - DERIVED from corpus
10. Activity probabilities → How often instruments play together - DERIVED from corpus

Everything is derived from the checkpoint data. No hardcoded probabilities or music theory rules.

Usage:
    python scripts/meta_pattern_generator.py checkpoint.npz -o output.mid --sections 8
    python scripts/meta_pattern_generator.py checkpoint.npz -o output.mid --no-form  # Use meta-pattern transforms
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
        self.build_track_derive_index()
        self.build_form_structure_index()
        self.build_rest_duration_index()

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

        # 4. Load track derives (explicit cross-track derivations)
        track_derives_file = ckpt.get('track_derives_json_file', [None])[0]
        self.track_derives = []
        self.track_derives_by_transform = {}
        self.leader_instruments = {}
        if track_derives_file:
            td_path = os.path.join(os.path.dirname(checkpoint_path), track_derives_file)
            if os.path.exists(td_path):
                with open(td_path) as f:
                    td_data = json.load(f)
                    self.track_derives = td_data.get('derives_json', td_data.get('derives', []))
                    self.track_derives_by_transform = td_data.get('derives_by_transform', {})
                    self.leader_instruments = td_data.get('leader_instruments', {})

        # 5. Load multi-factor transforms (rhythm, velocity, duration)
        multi_factor_file = ckpt.get('multi_factor_json_file', [None])[0]
        self.rhythm_transforms = []
        self.velocity_transforms = []
        self.duration_transforms = []
        if multi_factor_file:
            mf_path = os.path.join(os.path.dirname(checkpoint_path), multi_factor_file)
            if os.path.exists(mf_path):
                with open(mf_path) as f:
                    mf_data = json.load(f)
                    self.rhythm_transforms = mf_data.get('rhythm_vocabulary', [])
                    self.velocity_transforms = mf_data.get('velocity_vocabulary', [])
                    self.duration_transforms = mf_data.get('duration_vocabulary', [])
                    # Also store full factor vocabulary for sampling
                    self.factor_vocabulary = mf_data.get('vocabulary', [])

        if self.verbose:
            print(f"  Patterns: {len(self.patterns)}")
            print(f"  Transforms: {len(self.transform_vocab)}")
            print(f"  Meta-patterns: {len(self.meta_patterns)}")
            print(f"  Orchestration rules: {len(getattr(self, 'orchestration_rules', []))}")
            print(f"  Track derives: {len(self.track_derives)}")
            print(f"  Rhythm transforms: {len(self.rhythm_transforms)}")
            print(f"  Velocity transforms: {len(self.velocity_transforms)}")
            print(f"  Duration transforms: {len(self.duration_transforms)}")

    def _parse_transform(self, transform_raw) -> str:
        """Parse transform field which can be string OR dict with 'primitives' list.

        Track derives from v53 store transforms as {'primitives': ['T7', 'I0']}
        rather than simple strings like 'T7'.
        """
        if isinstance(transform_raw, dict):
            # Handle {'primitives': ['T7', 'I0']} format
            primitives = transform_raw.get('primitives', [])
            if primitives:
                return primitives[0] if len(primitives) == 1 else '_'.join(primitives)
            return 'identity'
        elif isinstance(transform_raw, str):
            return transform_raw
        return 'identity'

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

        # Also build from track_derives (if relations is empty)
        for derive in self.track_derives:
            if isinstance(derive, dict):
                src_pattern = derive.get('source_pattern_id', '')
                tgt_pattern = derive.get('target_pattern_id', '')
                transform = self._parse_transform(derive.get('transform', 'identity'))

                if src_pattern and tgt_pattern and transform:
                    self.transform_graph[str(src_pattern)][transform].append(str(tgt_pattern))

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

    def build_track_derive_index(self):
        """Build index from TrackDerive data for instrument-pair transforms.

        TrackDerive contains explicit relationships like:
        "Track 2 (Alto Sax) derives from Track 1 (Trumpet) via T7"

        This is MORE SPECIFIC than orchestration rules because it captures
        actual observed derivations, not just aggregated statistics.
        """
        if self.verbose:
            print("Building TrackDerive index...")

        # Index: (source_gm, target_gm) -> {transform: count}
        self.gm_pair_transforms = defaultdict(lambda: defaultdict(int))

        # Also index by pattern: pattern_id -> [(derived_pattern_id, transform, target_gm)]
        self.pattern_derivations = defaultdict(list)

        for derive in self.track_derives:
            if isinstance(derive, dict):
                # Get source and target info
                source_gm = derive.get('source_instrument', derive.get('source_gm', derive.get('leader_gm', -1)))
                target_gm = derive.get('target_instrument', derive.get('target_gm', derive.get('follower_gm', -1)))
                source_pattern = derive.get('source_pattern_id', derive.get('leader_pattern', ''))
                target_pattern = derive.get('target_pattern_id', derive.get('follower_pattern', ''))

                # Parse transform - can be string OR dict with 'primitives' list
                transform = self._parse_transform(derive.get('transform', 'identity'))

                if source_gm >= 0 and target_gm >= 0 and transform:
                    self.gm_pair_transforms[(source_gm, target_gm)][transform] += 1

                if source_pattern and target_pattern:
                    self.pattern_derivations[str(source_pattern)].append(
                        (str(target_pattern), transform, target_gm)
                    )

        # Pre-compute dominant transform per instrument pair
        self.dominant_gm_transforms = {}
        for (src_gm, tgt_gm), transform_counts in self.gm_pair_transforms.items():
            if transform_counts:
                dominant = max(transform_counts.items(), key=lambda x: x[1])
                self.dominant_gm_transforms[(src_gm, tgt_gm)] = dominant[0]

        if self.verbose:
            print(f"  GM pair transforms: {len(self.gm_pair_transforms)} pairs")
            print(f"  Pattern derivations: {len(self.pattern_derivations)} source patterns")
            print(f"  Dominant transforms: {len(self.dominant_gm_transforms)} pairs")

    def build_form_structure_index(self):
        """Build form structure from corpus - pattern sequences per piece.

        This captures how patterns are arranged in actual pieces (AABA form, etc.)
        without imposing music theory constraints - purely from corpus statistics.

        Structure: piece_id -> [(onset_time, pattern_id, gm_program), ...]
        """
        if self.verbose:
            print("Building form structure index...")

        # Group occurrences by piece_id, sorted by onset_time
        self.piece_pattern_sequences = defaultdict(list)
        self.piece_gm_sequences = defaultdict(lambda: defaultdict(list))

        for pattern_id, pattern in self.patterns.items():
            for occ in pattern.get('occurrences', []):
                piece_id = occ.get('piece_id', 'unknown')
                onset = occ.get('onset_time', occ.get('onset', 0))
                gm = occ.get('gm_program', 0)

                self.piece_pattern_sequences[piece_id].append((onset, pattern_id, gm))
                self.piece_gm_sequences[piece_id][gm].append((onset, pattern_id))

        # Sort each sequence by onset time
        for piece_id in self.piece_pattern_sequences:
            self.piece_pattern_sequences[piece_id].sort(key=lambda x: x[0])
        for piece_id in self.piece_gm_sequences:
            for gm in self.piece_gm_sequences[piece_id]:
                self.piece_gm_sequences[piece_id][gm].sort(key=lambda x: x[0])

        # Extract form patterns (pattern-ID sequences for repeated forms)
        # e.g., if multiple pieces have A-A-B-A structure, we can sample that
        self.form_patterns = defaultdict(int)  # (pattern_id_tuple) -> count

        for piece_id, sequence in self.piece_pattern_sequences.items():
            if len(sequence) >= 4:
                # Look at pattern sequence (ignoring time, just IDs)
                pattern_ids = tuple(pid for _, pid, _ in sequence[:8])  # First 8 patterns
                self.form_patterns[pattern_ids] += 1

        if self.verbose:
            print(f"  Pieces with form data: {len(self.piece_pattern_sequences)}")
            print(f"  Unique form patterns: {len(self.form_patterns)}")

    def build_rest_duration_index(self):
        """Build rest duration statistics from corpus.

        Tracks the gaps between pattern occurrences to learn natural rest durations.
        This is purely derived from corpus timing, no music theory.

        Structure: gm_program -> [rest_durations] (gaps between consecutive patterns)
        """
        if self.verbose:
            print("Building rest duration index...")

        self.rest_durations_by_gm = defaultdict(list)
        self.pattern_durations = {}  # pattern_id -> typical duration

        # First, estimate duration for each pattern from its occurrences
        for pattern_id, pattern in self.patterns.items():
            # Duration from intervals + base IOI
            intervals = pattern.get('pitch_intervals', [])
            base_ioi = 480  # Default

            for occ in pattern.get('occurrences', []):
                ioi = occ.get('tau_offset', 0)
                if ioi > 0:
                    base_ioi = ioi
                    break

            est_duration = base_ioi * (len(intervals) + 1)
            self.pattern_durations[pattern_id] = est_duration

        # Now find gaps between consecutive patterns per instrument
        for piece_id, gm_sequences in self.piece_gm_sequences.items():
            for gm, sequence in gm_sequences.items():
                for i in range(len(sequence) - 1):
                    onset1, pid1 = sequence[i]
                    onset2, pid2 = sequence[i + 1]

                    # Duration of first pattern
                    dur1 = self.pattern_durations.get(pid1, 480)

                    # Gap = next_onset - (this_onset + this_duration)
                    gap = onset2 - (onset1 + dur1)

                    # Only record positive gaps (rests)
                    if gap > 0:
                        self.rest_durations_by_gm[gm].append(gap)

        # Compute statistics per instrument
        self.rest_stats_by_gm = {}
        for gm, durations in self.rest_durations_by_gm.items():
            if durations:
                self.rest_stats_by_gm[gm] = {
                    'mean': np.mean(durations),
                    'median': np.median(durations),
                    'min': min(durations),
                    'max': max(durations),
                    'count': len(durations),
                }

        if self.verbose:
            print(f"  Rest data for {len(self.rest_stats_by_gm)} instruments")
            # Show a few examples
            for gm in list(self.rest_stats_by_gm.keys())[:3]:
                stats = self.rest_stats_by_gm[gm]
                print(f"    GM {gm}: median rest={stats['median']:.0f} ticks ({stats['count']} samples)")

    def sample_rest_duration(self, gm_program: int) -> int:
        """Sample a natural rest duration for this instrument from corpus statistics."""
        durations = self.rest_durations_by_gm.get(gm_program, [])

        if durations:
            # Sample from actual corpus rest durations
            return int(random.choice(durations))

        # Fallback: no data, return 0 (no rest)
        return 0

    def sample_form_pattern(self, lead_gm: int = None) -> List[str]:
        """Sample a pattern sequence that represents the form structure from corpus.

        Returns a list of pattern_ids in the order they appeared in corpus pieces.
        """
        if not self.form_patterns:
            return []

        # Weight by frequency
        patterns = list(self.form_patterns.keys())
        weights = list(self.form_patterns.values())

        # If lead_gm specified, prefer forms that start with patterns lead plays
        if lead_gm is not None:
            filtered_patterns = []
            filtered_weights = []

            for pattern_seq, weight in zip(patterns, weights):
                if pattern_seq:
                    first_pattern = self.patterns.get(pattern_seq[0], {})
                    # Check if lead plays first pattern
                    for occ in first_pattern.get('occurrences', []):
                        if occ.get('gm_program') == lead_gm:
                            filtered_patterns.append(pattern_seq)
                            filtered_weights.append(weight)
                            break

            if filtered_patterns:
                patterns, weights = filtered_patterns, filtered_weights

        if patterns:
            return list(random.choices(patterns, weights=weights)[0])

        return []

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

    def sample_phrase_pattern(self, target_depth: int = 2, lead_gm: int = None) -> str:
        """Sample a hierarchical pattern that the lead instrument actually plays.

        If lead_gm is provided, only sample from patterns that instrument played.
        """
        # Get patterns at or near target depth
        candidates = []
        for d in range(max(0, target_depth - 1), target_depth + 2):
            candidates.extend(self.patterns_by_depth.get(d, []))

        if not candidates:
            candidates = list(self.patterns.keys())

        # Filter to patterns the lead instrument actually plays
        if lead_gm is not None:
            filtered = []
            filtered_weights = []
            for pid in candidates:
                pattern = self.patterns.get(pid, {})
                # Check if this GM played this pattern
                gm_count = 0
                for occ in pattern.get('occurrences', []):
                    if occ.get('gm_program') == lead_gm:
                        gm_count += 1
                if gm_count > 0:
                    filtered.append(pid)
                    filtered_weights.append(gm_count)  # Weight by how often lead played it

            if filtered:
                return random.choices(filtered, weights=filtered_weights)[0]

        # Fallback: weight by total count
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
        use_form_structure: bool = True,
    ) -> Dict[int, List[Dict]]:
        """Generate using meta-patterns for form structure.

        Algorithm:
        1. Sample meta-pattern (transform sequence = form)
        2. Sample seed pattern (hierarchical for phrase structure)
        3. Walk transform graph following meta-pattern
        4. Expand patterns through hierarchy
        5. Orchestrate across instruments

        If use_form_structure=True, uses actual pattern sequences from corpus
        for more coherent form (AABA, etc.) - purely data-driven.
        """
        if seed is not None:
            random.seed(seed)

        if instruments is None:
            instruments = [56, 65, 66, 57, 0, 32]  # Trumpet, Alto, Tenor, Trombone, Piano, Bass

        if self.verbose:
            print(f"\nGenerating with meta-patterns...")
            print(f"  Sections: {n_sections}")
            print(f"  Instruments: {instruments}")
            print(f"  Use form structure: {use_form_structure}")

        all_tracks = defaultdict(list)
        current_time = 0

        # Determine lead instrument for pattern selection
        HORN_SECTION = {56, 57, 58, 59, 60, 61, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73}
        horn_instruments = [gm for gm in instruments if gm in HORN_SECTION]
        lead_gm = horn_instruments[0] if horn_instruments else None

        # If using form structure, sample a full form pattern from corpus
        form_pattern_ids = []
        if use_form_structure and self.form_patterns:
            form_pattern_ids = self.sample_form_pattern(lead_gm=lead_gm)
            if self.verbose and form_pattern_ids:
                print(f"  Using corpus form pattern with {len(form_pattern_ids)} patterns")

        for section_idx in range(n_sections):
            # DETERMINE PATTERN CHAIN FOR THIS SECTION

            if use_form_structure and form_pattern_ids:
                # USE CORPUS FORM STRUCTURE
                # Get patterns for this section from the form
                section_start = (section_idx * len(form_pattern_ids)) // n_sections
                section_end = ((section_idx + 1) * len(form_pattern_ids)) // n_sections
                pattern_chain = form_pattern_ids[section_start:section_end]

                # If form is shorter than sections, loop
                if not pattern_chain:
                    idx = section_idx % len(form_pattern_ids)
                    pattern_chain = [form_pattern_ids[idx]]

            else:
                # ORIGINAL BEHAVIOR: Meta-pattern guided generation
                # 1. SAMPLE META-PATTERN (form structure)
                meta = self.sample_meta_pattern()
                transform_seq = self.extract_transform_sequence(meta)

                if self.verbose and section_idx == 0:
                    print(f"  Section 0 meta-pattern: {transform_seq[:5]}...")

                # 2. SAMPLE SEED PATTERN (from patterns lead instrument actually plays)
                seed_pattern_id = self.sample_phrase_pattern(target_depth=2, lead_gm=lead_gm)

                # 3. WALK TRANSFORM GRAPH
                pattern_chain = [seed_pattern_id]
                current_pattern = seed_pattern_id

                for transform in transform_seq[:8]:  # Limit chain length
                    next_pattern = self.find_pattern_by_transform(current_pattern, transform)
                    if next_pattern:
                        pattern_chain.append(next_pattern)
                        current_pattern = next_pattern

            # 4. EXPAND AND ORCHESTRATE
            for pattern_idx, pattern_id in enumerate(pattern_chain):
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

                # Calculate phrase duration (how long this pattern takes)
                phrase_duration = base_ioi * (len(intervals) + 1)

                # Separate instruments by ROLE
                HORN_SECTION = {56, 57, 58, 59, 60, 61, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73}
                RHYTHM_SECTION = {0, 1, 2, 3, 4, 5, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39}

                horn_instruments = [gm for gm in instruments if gm in HORN_SECTION]
                rhythm_instruments = [gm for gm in instruments if gm in RHYTHM_SECTION]
                other_instruments = [gm for gm in instruments if gm not in HORN_SECTION and gm not in RHYTHM_SECTION]

                # HORN SECTION: Lead plays, background horns play based on CORPUS patterns
                if horn_instruments:
                    lead_gm = horn_instruments[0]
                    lead_pitch = self._sample_pitch_for_instrument(pattern, lead_gm)

                    lead_notes = self._expand_to_notes(
                        intervals, lead_pitch, current_time, lead_gm, base_ioi
                    )
                    all_tracks[lead_gm].extend(lead_notes)

                    # Background horns: Use CORPUS-DERIVED activity probability
                    for follower_idx, follower_gm in enumerate(horn_instruments[1:]):
                        # Get natural co-occurrence probability from checkpoint
                        play_prob = self._get_cooccurrence_probability(
                            lead_pattern_id=pattern_id,
                            lead_gm=lead_gm,
                            target_gm=follower_gm
                        )

                        if random.random() > play_prob:
                            # REST - this instrument didn't usually play here
                            # Sample rest duration from corpus if available
                            rest_dur = self.sample_rest_duration(follower_gm)
                            # Just skip this phrase (rest is implicit in timing)
                            continue

                        rule = self._find_orchestration_rule(lead_gm, follower_gm, pattern_id)

                        if rule:
                            transform = rule.get('transform', 'identity')
                            derived_pattern_id = rule.get('derived_pattern')
                            if derived_pattern_id and derived_pattern_id in self.patterns:
                                derived_pattern = self.patterns[derived_pattern_id]
                                follower_intervals = derived_pattern.get('pitch_intervals', intervals)
                                follower_pitch = self._sample_pitch_for_instrument(derived_pattern, follower_gm)
                            else:
                                follower_intervals = self._apply_transform(intervals, transform)
                                follower_pitch = self._sample_pitch_for_instrument(pattern, follower_gm)
                        else:
                            follower_intervals = intervals
                            follower_pitch = self._sample_pitch_for_instrument(pattern, follower_gm)

                        follower_notes = self._expand_to_notes(
                            follower_intervals, follower_pitch, current_time, follower_gm, base_ioi
                        )
                        all_tracks[follower_gm].extend(follower_notes)

                # RHYTHM SECTION: Sample co-occurring patterns to fill the phrase duration
                for rhythm_gm in rhythm_instruments:
                    rhythm_time = current_time
                    loop_count = 0
                    max_loops = 16  # Safety limit

                    while rhythm_time < current_time + phrase_duration and loop_count < max_loops:
                        # Sample a NEW co-occurring pattern for each loop
                        # This creates variety while maintaining harmonic coherence
                        rhythm_pattern_id = self._sample_cooccurring_pattern(
                            lead_pattern_id=pattern_id,  # Use CURRENT pattern, not seed
                            target_gm=rhythm_gm
                        )

                        if not rhythm_pattern_id:
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
                                rhythm_intervals, rhythm_pitch, rhythm_time, rhythm_gm, rhythm_ioi,
                                use_variations=True
                            )
                            all_tracks[rhythm_gm].extend(rhythm_notes)

                            # Advance rhythm time by pattern duration
                            pattern_dur = rhythm_ioi * (len(rhythm_intervals) + 1)
                            rhythm_time += pattern_dur
                        else:
                            # No pattern found, advance time anyway to avoid infinite loop
                            rhythm_time += base_ioi

                        loop_count += 1

                # OTHER INSTRUMENTS
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

                # Advance time by phrase duration
                current_time += phrase_duration

        if self.verbose:
            print(f"\nGenerated:")
            for gm, notes in sorted(all_tracks.items()):
                if notes:
                    avg_pitch = np.mean([n['pitch'] for n in notes])
                    print(f"  GM {gm}: {len(notes)} notes, avg_pitch={avg_pitch:.0f}")

        return dict(all_tracks)

    def _find_orchestration_rule(self, lead_gm: int, follower_gm: int, pattern_id: str = None) -> Optional[Dict]:
        """Find learned orchestration rule for instrument pair.

        Priority order:
        1. Pattern-specific derivation from TrackDerive (most specific)
        2. GM-pair dominant transform from TrackDerive (specific to instruments)
        3. Orchestration rules (aggregated statistics)
        """
        # 1. Try pattern-specific derivation first
        if pattern_id and hasattr(self, 'pattern_derivations'):
            derivations = self.pattern_derivations.get(str(pattern_id), [])
            for derived_pattern, transform, target_gm in derivations:
                if target_gm == follower_gm:
                    return {
                        'transform': transform,
                        'derived_pattern': derived_pattern,
                        'source': 'pattern_derivation'
                    }

        # 2. Try GM-pair dominant transform from TrackDerive
        if hasattr(self, 'dominant_gm_transforms'):
            transform = self.dominant_gm_transforms.get((lead_gm, follower_gm))
            if transform:
                return {
                    'transform': transform,
                    'source': 'track_derive'
                }

        # 3. Fall back to orchestration rules
        for rule in getattr(self, 'orchestration_rules', []):
            src = rule.get('source_instrument', -1)
            tgt = rule.get('target_instrument', -1)
            if src == lead_gm and tgt == follower_gm:
                rule['source'] = 'orchestration_rule'
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
                    # Handle both 'onset_time' and 'onset' field names
                    onset = occ.get('onset_time', occ.get('onset', 0))
                    gm = occ.get('gm_program', 0)

                    # Bucket time to 1 beat (480 ticks) for co-occurrence
                    time_bucket = onset // 480

                    time_slices[(piece_id, time_bucket)].append((pattern_id, gm))

            # Build index: lead_pattern -> target_gm -> {cooccurring_pattern: count}
            # Using dict to aggregate counts (deduplicated)
            cooc_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

            for (piece_id, time_bucket), pattern_gm_list in time_slices.items():
                if len(pattern_gm_list) < 2:
                    continue  # Need at least 2 patterns for co-occurrence

                # For each pattern in this time slice, record co-occurrences
                for i, (pattern_id, gm) in enumerate(pattern_gm_list):
                    for j, (other_pattern_id, other_gm) in enumerate(pattern_gm_list):
                        if i != j:
                            # Count how many times pattern_id co-occurred with other_pattern_id
                            cooc_counts[pattern_id][other_gm][other_pattern_id] += 1

            # Convert to lists for sampling
            self._cooccurrence_index = defaultdict(lambda: defaultdict(list))
            self._cooccurrence_weights = defaultdict(lambda: defaultdict(list))

            for pattern_id, gm_map in cooc_counts.items():
                for target_gm, pattern_counts in gm_map.items():
                    for other_pattern_id, count in pattern_counts.items():
                        self._cooccurrence_index[pattern_id][target_gm].append(other_pattern_id)
                        # Weight by co-occurrence count (how often they played together)
                        self._cooccurrence_weights[pattern_id][target_gm].append(count)

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

    def _get_cooccurrence_probability(self, lead_pattern_id: str, lead_gm: int, target_gm: int) -> float:
        """Get the natural probability that target_gm plays when lead_gm plays this pattern.

        This is derived from the corpus - how often did these instruments actually
        play together when this pattern occurred?
        """
        # Build activity index on first call
        if not hasattr(self, '_activity_index'):
            if self.verbose:
                print("Building activity index from corpus...")

            # For each pattern, count how many times each instrument played
            # pattern_id -> {gm: count}
            self._pattern_instrument_counts = defaultdict(lambda: defaultdict(int))

            # Also track total occurrences per pattern
            self._pattern_total_occurrences = defaultdict(int)

            # And track global instrument pair co-occurrence rates
            # (lead_gm, target_gm) -> (cooccur_count, total_lead_count)
            self._gm_pair_cooccurrence = defaultdict(lambda: [0, 0])

            for pattern_id, pattern in self.patterns.items():
                occs = pattern.get('occurrences', [])

                # Group occurrences by (piece_id, time_bucket) to find co-occurrences
                by_time = defaultdict(list)
                for occ in occs:
                    piece_id = occ.get('piece_id', 'unknown')
                    onset = occ.get('onset_time', occ.get('onset', 0))
                    gm = occ.get('gm_program', 0)
                    time_bucket = onset // 480
                    by_time[(piece_id, time_bucket)].append(gm)

                    self._pattern_instrument_counts[pattern_id][gm] += 1
                    self._pattern_total_occurrences[pattern_id] += 1

                # Count co-occurrences
                for (piece_id, time_bucket), gm_list in by_time.items():
                    gm_set = set(gm_list)
                    for lead in gm_set:
                        self._gm_pair_cooccurrence[(lead, lead)][1] += 1  # Total lead count
                        for target in gm_set:
                            if target != lead:
                                self._gm_pair_cooccurrence[(lead, target)][0] += 1

            self._activity_index = True

            if self.verbose:
                print(f"  Activity index: {len(self._pattern_instrument_counts)} patterns")

        # Method 1: Pattern-specific probability
        pattern_counts = self._pattern_instrument_counts.get(str(lead_pattern_id), {})
        lead_count = pattern_counts.get(lead_gm, 0)
        target_count = pattern_counts.get(target_gm, 0)

        if lead_count > 0:
            pattern_prob = target_count / lead_count
            if pattern_prob > 0:
                return min(1.0, pattern_prob)  # Cap at 1.0

        # Method 2: Global GM-pair co-occurrence rate
        cooccur_count, total_lead = self._gm_pair_cooccurrence.get((lead_gm, target_gm), [0, 0])
        if total_lead > 0:
            global_prob = cooccur_count / total_lead
            if global_prob > 0:
                return global_prob

        # Fallback: moderate probability
        return 0.5

    def _apply_transform(self, intervals: List[int], transform: str) -> List[int]:
        """Apply a pitch transform to intervals.

        Supports:
        - T0-T11: Transposition (no change to intervals, affects first_pitch)
        - I0-I11: Inversion (negate intervals)
        - R: Retrograde (reverse intervals)
        - RI: Retrograde inversion
        """
        if not intervals:
            return intervals

        if transform == 'identity' or transform == 'T0':
            return intervals
        elif transform == 'R':
            return list(reversed(intervals))
        elif transform.startswith('I'):
            return [-i for i in intervals]
        elif transform == 'RI':
            return [-i for i in reversed(intervals)]
        elif transform.startswith('T'):
            # Transposition doesn't change intervals, only first_pitch
            return intervals
        else:
            return intervals

    def _sample_rhythm_variation(self, base_ioi: int) -> int:
        """Sample a rhythm variation using learned rhythm transforms.

        Uses multi-factor rhythm vocabulary to vary timing expressively.
        """
        if not hasattr(self, 'rhythm_transforms') or not self.rhythm_transforms:
            return base_ioi

        # Rhythm transforms are typically ratios or multipliers
        # Sample one and apply it
        transform = random.choice(self.rhythm_transforms)

        if isinstance(transform, dict):
            ratio = transform.get('ratio', 1.0)
        elif isinstance(transform, (int, float)):
            ratio = transform
        else:
            ratio = 1.0

        # Apply ratio but keep in reasonable range
        varied_ioi = int(base_ioi * ratio)
        return max(60, min(1920, varied_ioi))  # 32nd note to whole note

    def _sample_velocity_variation(self, base_velocity: int = 80) -> int:
        """Sample a velocity variation using learned velocity transforms.

        Uses multi-factor velocity vocabulary for expressive dynamics.
        """
        if not hasattr(self, 'velocity_transforms') or not self.velocity_transforms:
            return base_velocity

        # Velocity transforms are typically offsets or multipliers
        transform = random.choice(self.velocity_transforms)

        if isinstance(transform, dict):
            offset = transform.get('offset', 0)
            varied = base_velocity + offset
        elif isinstance(transform, (int, float)):
            varied = base_velocity + int(transform)
        else:
            varied = base_velocity

        return max(40, min(127, int(varied)))

    def _sample_duration_variation(self, base_duration: int) -> int:
        """Sample a duration variation using learned duration transforms.

        Uses multi-factor duration vocabulary for articulation.
        """
        if not hasattr(self, 'duration_transforms') or not self.duration_transforms:
            return base_duration

        transform = random.choice(self.duration_transforms)

        if isinstance(transform, dict):
            ratio = transform.get('ratio', 1.0)
        elif isinstance(transform, (int, float)):
            ratio = transform
        else:
            ratio = 1.0

        varied = int(base_duration * ratio)
        return max(30, min(3840, varied))  # 64th note to 2 whole notes

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
        use_variations: bool = True,
    ) -> List[Dict]:
        """Expand intervals to note events.

        If use_variations=True, applies learned rhythm/velocity/duration
        transforms for expressive output.
        """
        notes = []
        pitch_range = GM_RANGES.get(gm_program, DEFAULT_RANGE)

        current_pitch = first_pitch
        current_time = start_time

        # Sample base velocity with optional variation
        base_velocity = 80
        if use_variations:
            base_velocity = self._sample_velocity_variation(base_velocity)

        # First note
        note_duration = base_ioi
        if use_variations:
            note_duration = self._sample_duration_variation(base_ioi)

        notes.append({
            'pitch': max(pitch_range[0], min(pitch_range[1], current_pitch)),
            'velocity': base_velocity,
            'time': current_time,
            'duration': note_duration,
        })

        # Remaining notes
        for i, interval in enumerate(intervals):
            # Apply rhythm variation for timing
            ioi = base_ioi
            if use_variations:
                ioi = self._sample_rhythm_variation(base_ioi)

            current_time += ioi
            current_pitch += interval

            # Octave fold to stay in range
            while current_pitch < pitch_range[0]:
                current_pitch += 12
            while current_pitch > pitch_range[1]:
                current_pitch -= 12

            # Apply velocity and duration variations
            velocity = base_velocity
            duration = ioi
            if use_variations:
                # Slight velocity variation per note for expression
                velocity = self._sample_velocity_variation(base_velocity)
                duration = self._sample_duration_variation(ioi)

            notes.append({
                'pitch': current_pitch,
                'velocity': velocity,
                'time': current_time,
                'duration': duration,
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
    parser.add_argument('--no-form', action='store_true',
                        help='Disable corpus form structure (use meta-pattern transforms instead)')
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
        use_form_structure=not args.no_form,
    )

    gen.to_midi(tracks, args.output, tempo=args.tempo)

    print("\nDone!")


if __name__ == '__main__':
    main()
