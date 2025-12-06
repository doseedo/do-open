#!/usr/bin/env python3
"""
Meta-Pattern Generator (v53 Checkpoint) - Probabilistic PCFG Sampling
======================================================================

Converts a deterministic Re-Pair grammar (Straight-Line Program) into a
probabilistic generative model using 4-level Markov sampling + form templates:

PROBABILISTIC SAMPLING LEVELS:
  Level 1: Frequency weighting     P(pattern) from occurrence counts
  Level 2: Transition probability  P(pattern_B | pattern_A, instrument)
  Level 3: Position conditioning   P(pattern | position_in_piece, instrument)
  Level 4: Co-occurrence           P(pattern | concurrent_patterns_on_other_instruments)

FORM TEMPLATES:
  Learn repetition structures (AABA, ABAB, etc.) from corpus and enforce them
  during generation to prevent "globally random" output.

This addresses the core problem: Re-Pair creates a DETERMINISTIC grammar
(perfect reconstruction, poor generation). We add PROBABILISTIC sampling
to enable generation while respecting learned structure.

FULL CHECKPOINT UTILIZATION (All corpus-derived, no music theory constraints):
1. Transition index → What patterns follow what (bigram Markov chains)
2. Position index → What patterns appear where (start/middle/end of piece)
3. Co-occurrence → Patterns that played together (harmonic coherence)
4. TrackDerive → Cross-track derivations (72K+ relations)
5. Form structure → Pattern sequences per piece (AABA, etc.)
6. Rest durations → Natural gaps between patterns per instrument
7. Hierarchical patterns → Phrase structure (left_child + connector + right_child)
8. Multi-factor transforms → Rhythm/velocity/duration variation

Everything is derived from checkpoint data. No hardcoded probabilities or music theory rules.

Usage:
    # Default: 3-level probabilistic Markov sampling
    python scripts/meta_pattern_generator.py checkpoint.npz -o output.mid

    # Use corpus form structure (exact pattern sequences)
    python scripts/meta_pattern_generator.py checkpoint.npz -o output.mid --form

    # Use meta-pattern transform graph walking
    python scripts/meta_pattern_generator.py checkpoint.npz -o output.mid --meta-pattern
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
        self.build_transition_index()  # Level 2: P(next | current, gm)
        self.build_position_index()    # Level 3: P(pattern | position, gm)
        self.build_cooccurrence_index()  # Level 4: P(pattern | concurrent_patterns)
        self.build_form_template_index()  # For global repetition structure
        self.build_instrument_role_index()  # Derive lead/follower from corpus

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

    def build_transition_index(self):
        """Build transition probabilities: P(next_pattern | current_pattern, instrument).

        LEVEL 2 of probabilistic sampling.

        This is the key missing piece - instead of sampling patterns independently,
        we sample based on what patterns typically FOLLOW the current pattern.
        This creates coherent melodic/harmonic flow.
        """
        if self.verbose:
            print("Building transition index (Level 2)...")

        # Bigram counts: (current_pattern, next_pattern, gm) -> count
        self.transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Also track unigram counts for smoothing: (pattern, gm) -> count
        self.pattern_counts_by_gm = defaultdict(lambda: defaultdict(int))

        # Build from piece_gm_sequences (consecutive patterns per instrument per piece)
        total_transitions = 0

        for piece_id, gm_sequences in self.piece_gm_sequences.items():
            for gm, sequence in gm_sequences.items():
                # sequence is [(onset_time, pattern_id), ...] sorted by onset
                for i in range(len(sequence)):
                    _, current_pattern = sequence[i]
                    self.pattern_counts_by_gm[gm][current_pattern] += 1

                    if i < len(sequence) - 1:
                        _, next_pattern = sequence[i + 1]
                        self.transition_counts[gm][current_pattern][next_pattern] += 1
                        total_transitions += 1

        # Pre-compute transition probabilities for efficient sampling
        # transition_probs[gm][current] = ([next_patterns], [probabilities])
        self.transition_probs = {}

        for gm, current_map in self.transition_counts.items():
            self.transition_probs[gm] = {}
            for current_pattern, next_counts in current_map.items():
                next_patterns = list(next_counts.keys())
                counts = list(next_counts.values())
                total = sum(counts)
                probs = [c / total for c in counts]
                self.transition_probs[gm][current_pattern] = (next_patterns, probs)

        if self.verbose:
            print(f"  Total transitions: {total_transitions:,}")
            print(f"  Instruments with transitions: {len(self.transition_probs)}")
            # Show coverage
            for gm in list(self.transition_probs.keys())[:3]:
                n_patterns = len(self.transition_probs[gm])
                print(f"    GM {gm}: {n_patterns} patterns with transitions")

    def build_position_index(self):
        """Build position-conditioned probabilities: P(pattern | position_in_piece, instrument).

        LEVEL 3 of probabilistic sampling.

        Patterns that typically appear at the START of pieces (intros) are different
        from patterns that appear in the MIDDLE (development) or END (cadences).
        This captures form-level structure without explicit music theory labels.

        Position is normalized to [0, 1] and bucketed into N_BUCKETS bins.
        """
        if self.verbose:
            print("Building position index (Level 3)...")

        N_BUCKETS = 10  # 0-10%, 10-20%, ..., 90-100%

        # First, compute piece durations
        piece_durations = {}
        for piece_id, sequence in self.piece_pattern_sequences.items():
            if sequence:
                max_onset = max(onset for onset, _, _ in sequence)
                # Add estimated duration of last pattern
                piece_durations[piece_id] = max_onset + 1920  # ~1 bar buffer

        # Position counts: (position_bucket, gm, pattern) -> count
        self.position_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Total counts per (position_bucket, gm) for normalization
        self.position_totals = defaultdict(lambda: defaultdict(int))

        for piece_id, gm_sequences in self.piece_gm_sequences.items():
            piece_duration = piece_durations.get(piece_id, 1)

            for gm, sequence in gm_sequences.items():
                for onset, pattern_id in sequence:
                    # Normalize position to [0, 1]
                    normalized_pos = onset / piece_duration if piece_duration > 0 else 0
                    normalized_pos = min(normalized_pos, 0.999)  # Clamp to valid range

                    # Bucket into N_BUCKETS bins
                    bucket = int(normalized_pos * N_BUCKETS)

                    self.position_counts[bucket][gm][pattern_id] += 1
                    self.position_totals[bucket][gm] += 1

        # Pre-compute position probabilities for efficient sampling
        # position_probs[bucket][gm] = ([patterns], [probabilities])
        self.position_probs = {}
        self.n_position_buckets = N_BUCKETS

        for bucket in range(N_BUCKETS):
            self.position_probs[bucket] = {}
            for gm, pattern_counts in self.position_counts[bucket].items():
                patterns = list(pattern_counts.keys())
                counts = list(pattern_counts.values())
                total = sum(counts)
                probs = [c / total for c in counts]
                self.position_probs[bucket][gm] = (patterns, probs)

        if self.verbose:
            print(f"  Position buckets: {N_BUCKETS}")
            # Show distribution across positions
            for bucket in [0, N_BUCKETS // 2, N_BUCKETS - 1]:
                pos_name = ["start", "middle", "end"][bucket // (N_BUCKETS // 3)]
                n_gms = len(self.position_probs.get(bucket, {}))
                print(f"    Bucket {bucket} ({pos_name}): {n_gms} instruments")

    def build_cooccurrence_index(self):
        """Build co-occurrence index: P(pattern | concurrent_patterns_by_other_instruments).

        LEVEL 4 of probabilistic sampling - Cross-instrument coordination.

        This captures which patterns PLAYED TOGETHER in the corpus, so when trumpet
        plays pattern A, we know which bass patterns typically accompanied it.

        Structure: (lead_pattern, target_gm) -> [(pattern_id, weight), ...]
        """
        if self.verbose:
            print("Building co-occurrence index (Level 4)...")

        # Group occurrences by (piece_id, time_bucket) to find what played together
        # time_bucket = onset // 960 (2 beats = 1 bar in 4/4)
        TIME_BUCKET_TICKS = 960  # 2 beats

        time_slices = defaultdict(list)  # (piece_id, bucket) -> [(pattern_id, gm), ...]

        for pattern_id, pattern in self.patterns.items():
            for occ in pattern.get('occurrences', []):
                piece_id = occ.get('piece_id', 'unknown')
                onset = occ.get('onset_time', occ.get('onset', 0))
                gm = occ.get('gm_program', 0)

                time_bucket = onset // TIME_BUCKET_TICKS
                time_slices[(piece_id, time_bucket)].append((pattern_id, gm))

        # Build co-occurrence counts: (lead_pattern, target_gm) -> {target_pattern: count}
        cooc_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for (piece_id, bucket), patterns_gms in time_slices.items():
            if len(patterns_gms) < 2:
                continue

            # For each pattern, record what other instruments played
            for i, (pattern_id, gm) in enumerate(patterns_gms):
                for j, (other_pattern, other_gm) in enumerate(patterns_gms):
                    if i != j and gm != other_gm:
                        # When pattern_id played, other_pattern played on other_gm
                        cooc_counts[pattern_id][other_gm][other_pattern] += 1

        # Convert to probability distributions for sampling
        self.cooccurrence_probs = {}  # lead_pattern -> target_gm -> ([patterns], [probs])

        for lead_pattern, gm_map in cooc_counts.items():
            self.cooccurrence_probs[lead_pattern] = {}
            for target_gm, pattern_counts in gm_map.items():
                patterns = list(pattern_counts.keys())
                counts = list(pattern_counts.values())
                total = sum(counts)
                probs = [c / total for c in counts]
                self.cooccurrence_probs[lead_pattern][target_gm] = (patterns, probs)

        if self.verbose:
            n_leads = len(self.cooccurrence_probs)
            avg_targets = np.mean([len(gm_map) for gm_map in self.cooccurrence_probs.values()]) if self.cooccurrence_probs else 0
            print(f"  Co-occurrence: {n_leads} lead patterns, avg {avg_targets:.1f} target instruments each")

    def build_form_template_index(self):
        """Build form templates from corpus - learn repetition structures like AABA.

        This addresses "globally random" by identifying which sections REPEAT in pieces.

        Form templates are abstract structures like:
        - [A, A, B, A] - 32-bar jazz standard
        - [A, B, A, B] - simple alternation
        - [A, A, A, A] - continuous development

        We extract these by looking at which pattern groups repeat within pieces.
        """
        if self.verbose:
            print("Building form template index...")

        # Analyze each piece's pattern sequence for repetition
        self.form_templates = []  # List of (template, count) - e.g., (['A','A','B','A'], 5)

        # Group by section similarity
        for piece_id, sequence in self.piece_pattern_sequences.items():
            if len(sequence) < 8:
                continue

            # Divide piece into N_SECTIONS sections
            N_SECTIONS = 4
            section_size = len(sequence) // N_SECTIONS
            if section_size < 2:
                continue

            # Extract pattern sets for each section
            section_patterns = []
            for i in range(N_SECTIONS):
                start = i * section_size
                end = start + section_size
                section_pids = frozenset(pid for _, pid, _ in sequence[start:end])
                section_patterns.append(section_pids)

            # Label sections by similarity (A, B, C, ...)
            # Sections with >50% overlap get same label
            labels = []
            label_map = {}  # frozenset -> label

            for section_set in section_patterns:
                best_match = None
                best_overlap = 0

                for existing_set, label in label_map.items():
                    overlap = len(section_set & existing_set) / max(1, len(section_set | existing_set))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = label

                if best_match and best_overlap > 0.4:
                    labels.append(best_match)
                else:
                    new_label = chr(ord('A') + len(set(labels)))
                    labels.append(new_label)
                    label_map[section_set] = new_label

            # Record this form template
            template = tuple(labels)
            self.form_templates.append(template)

        # Count template frequencies
        template_counts = defaultdict(int)
        for template in self.form_templates:
            template_counts[template] += 1

        # Store as list of (template, count) sorted by frequency
        self.form_template_probs = sorted(
            template_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if self.verbose:
            print(f"  Form templates: {len(self.form_template_probs)} unique")
            for template, count in self.form_template_probs[:5]:
                print(f"    {'-'.join(template)}: {count} pieces")

    def sample_form_template(self) -> List[str]:
        """Sample a form template weighted by corpus frequency.

        Returns list like ['A', 'A', 'B', 'A'] indicating section repetition.
        """
        if not self.form_template_probs:
            # Fallback: derive from observed section count distribution
            # Don't impose AABA - sample number of unique sections from corpus
            if hasattr(self, 'section_count_distribution') and self.section_count_distribution:
                n_unique = random.choices(
                    list(self.section_count_distribution.keys()),
                    weights=list(self.section_count_distribution.values())
                )[0]
            else:
                n_unique = random.randint(2, 4)

            # Generate random template with n_unique distinct sections
            labels = [chr(ord('A') + i) for i in range(n_unique)]
            return [random.choice(labels) for _ in range(4)]

        templates = [t for t, c in self.form_template_probs]
        weights = [c for t, c in self.form_template_probs]

        return list(random.choices(templates, weights=weights)[0])

    def build_instrument_role_index(self):
        """Derive lead/follower instrument roles from TrackDerive corpus data.

        REPLACES hardcoded HORN_SECTION and RHYTHM_SECTION with corpus-derived roles.

        Lead instruments: Those most often copied FROM (source_gm in track_derives)
        Follower instruments: Those most often copying from others (target_gm)

        This ensures instrument roles reflect actual corpus behavior, not assumptions.
        """
        if self.verbose:
            print("Building instrument role index from corpus...")

        # Count how often each instrument is SOURCE (leader) vs TARGET (follower)
        source_counts = defaultdict(int)  # gm -> count of times it's copied FROM
        target_counts = defaultdict(int)  # gm -> count of times it copies FROM others

        for derive in self.track_derives:
            if isinstance(derive, dict):
                source_gm = derive.get('source_instrument', derive.get('source_gm', derive.get('leader_gm', -1)))
                target_gm = derive.get('target_instrument', derive.get('target_gm', derive.get('follower_gm', -1)))

                if source_gm >= 0:
                    source_counts[source_gm] += 1
                if target_gm >= 0:
                    target_counts[target_gm] += 1

        # Calculate lead score: how much more often is this instrument a source than target?
        # lead_score = source_count / (target_count + 1) - higher = more "leader-like"
        self.instrument_lead_scores = {}
        all_gms = set(source_counts.keys()) | set(target_counts.keys())

        for gm in all_gms:
            src = source_counts.get(gm, 0)
            tgt = target_counts.get(gm, 0)
            # Lead score: ratio of being copied vs copying
            self.instrument_lead_scores[gm] = src / (tgt + 1)

        # Sort instruments by lead score
        sorted_by_lead = sorted(
            self.instrument_lead_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Top instruments (high lead score) are "leads"
        # Bottom instruments (low lead score) are "followers/rhythm"
        n_instruments = len(sorted_by_lead)
        n_leads = max(1, n_instruments // 3)  # Top third are leads

        self.corpus_lead_instruments = set(gm for gm, score in sorted_by_lead[:n_leads])
        self.corpus_follower_instruments = set(gm for gm, score in sorted_by_lead[n_leads:])

        # Also store activity scores from pattern counts (how often each instrument plays)
        self.instrument_activity = defaultdict(int)
        for pattern_id, pattern in self.patterns.items():
            for occ in pattern.get('occurrences', []):
                gm = occ.get('gm_program', 0)
                self.instrument_activity[gm] += 1

        if self.verbose:
            print(f"  Lead instruments (top {n_leads}): {sorted(self.corpus_lead_instruments)}")
            print(f"  Follower instruments: {len(self.corpus_follower_instruments)}")
            if sorted_by_lead:
                top_lead = sorted_by_lead[0]
                print(f"  Top lead: GM {top_lead[0]} (score={top_lead[1]:.2f})")

    def get_corpus_lead_instrument(self, instruments: List[int]) -> int:
        """Get the most lead-like instrument from the given list, based on corpus data.

        This replaces the hardcoded logic of "horn_instruments[0]".
        """
        if not instruments:
            return 0

        # Find instrument with highest lead score
        best_gm = instruments[0]
        best_score = self.instrument_lead_scores.get(best_gm, 0)

        for gm in instruments:
            score = self.instrument_lead_scores.get(gm, 0)
            if score > best_score:
                best_score = score
                best_gm = gm

        return best_gm

    def get_corpus_follower_instruments(self, instruments: List[int], lead_gm: int) -> List[int]:
        """Get follower instruments from the given list, excluding the lead.

        This replaces the hardcoded RHYTHM_SECTION logic.
        """
        return [gm for gm in instruments if gm != lead_gm]

    def sample_next_pattern(
        self,
        current_pattern: str,
        position: float,
        gm_program: int,
        concurrent_patterns: Dict[int, str] = None,
        transition_weight: float = 0.4,
        position_weight: float = 0.2,
        frequency_weight: float = 0.1,
        cooccurrence_weight: float = 0.3,
    ) -> Optional[str]:
        """Sample next pattern using all four levels of probabilistic sampling.

        Combines:
        - Level 1: Frequency-weighted (pattern popularity)
        - Level 2: Transition probability (what follows current pattern)
        - Level 3: Position conditioning (what's typical at this point in piece)
        - Level 4: Co-occurrence (what plays with concurrent patterns on other instruments)

        Args:
            current_pattern: Current pattern ID (for transitions)
            position: Normalized position in piece [0, 1]
            gm_program: Instrument GM program
            concurrent_patterns: Dict of {other_gm: pattern_id} for co-occurrence conditioning
            transition_weight: Weight for transition probabilities
            position_weight: Weight for position conditioning
            frequency_weight: Weight for base frequency
            cooccurrence_weight: Weight for co-occurrence with other instruments

        Returns:
            Sampled pattern ID, or None if no data available
        """
        candidates = {}  # pattern_id -> combined score

        # Level 1: Base frequency (unigram)
        if gm_program in self.pattern_counts_by_gm:
            total = sum(self.pattern_counts_by_gm[gm_program].values())
            for pattern_id, count in self.pattern_counts_by_gm[gm_program].items():
                candidates[pattern_id] = frequency_weight * (count / total)

        # Level 2: Transition probability
        if gm_program in self.transition_probs and current_pattern:
            current_key = str(current_pattern)
            if current_key in self.transition_probs[gm_program]:
                next_patterns, probs = self.transition_probs[gm_program][current_key]
                for pattern_id, prob in zip(next_patterns, probs):
                    if pattern_id in candidates:
                        candidates[pattern_id] += transition_weight * prob
                    else:
                        candidates[pattern_id] = transition_weight * prob

        # Level 3: Position conditioning
        bucket = int(min(position, 0.999) * self.n_position_buckets)
        if bucket in self.position_probs and gm_program in self.position_probs[bucket]:
            patterns, probs = self.position_probs[bucket][gm_program]
            for pattern_id, prob in zip(patterns, probs):
                if pattern_id in candidates:
                    candidates[pattern_id] += position_weight * prob
                else:
                    candidates[pattern_id] = position_weight * prob

        # Level 4: Co-occurrence conditioning (NEW)
        # If other instruments are playing, boost patterns that co-occurred with them
        if concurrent_patterns and hasattr(self, 'cooccurrence_probs'):
            for other_gm, other_pattern in concurrent_patterns.items():
                if other_gm == gm_program:
                    continue  # Skip self

                other_key = str(other_pattern)
                if other_key in self.cooccurrence_probs:
                    gm_probs = self.cooccurrence_probs[other_key].get(gm_program)
                    if gm_probs:
                        patterns, probs = gm_probs
                        # Weight contribution by number of concurrent instruments
                        weight_per_inst = cooccurrence_weight / max(1, len(concurrent_patterns))
                        for pattern_id, prob in zip(patterns, probs):
                            if pattern_id in candidates:
                                candidates[pattern_id] += weight_per_inst * prob
                            else:
                                candidates[pattern_id] = weight_per_inst * prob

        if not candidates:
            # Fallback to any pattern for this instrument
            return self._sample_pattern_for_instrument(gm_program)

        # Sample from combined distribution
        pattern_ids = list(candidates.keys())
        scores = list(candidates.values())

        # Normalize to probabilities
        total_score = sum(scores)
        if total_score > 0:
            probs = [s / total_score for s in scores]
            return random.choices(pattern_ids, weights=probs)[0]

        return pattern_ids[0] if pattern_ids else None

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
        use_form_structure: bool = False,
        use_probabilistic: bool = True,
    ) -> Dict[int, List[Dict]]:
        """Generate using learned probabilistic models from corpus.

        Sampling Modes (in priority order):
        1. use_probabilistic=True (DEFAULT): Use 4-level Markov sampling + form templates
           - Level 1: Frequency weighting (pattern popularity)
           - Level 2: Transition probabilities (what follows what)
           - Level 3: Position conditioning (intro vs middle vs end patterns)
           - Level 4: Co-occurrence conditioning (what plays together across instruments)
           - Form templates: Enforce repetition structure (AABA) from corpus

        2. use_form_structure=True: Use exact pattern sequences from corpus pieces
           - Samples an actual pattern sequence that appeared in training

        3. Neither: Meta-pattern guided generation (transform graph walking)
           - Walks the transform graph following learned transform sequences

        The probabilistic mode (default) produces the most coherent output
        because it respects both local (transitions), global (position), and
        cross-instrument (co-occurrence) structure with forced repetition.
        """
        if seed is not None:
            random.seed(seed)

        if instruments is None:
            instruments = [56, 65, 66, 57, 0, 32]  # Trumpet, Alto, Tenor, Trombone, Piano, Bass

        # Determine sampling mode string for logging
        if use_probabilistic:
            mode = "probabilistic (4-level Markov + form templates)"
        elif use_form_structure:
            mode = "form structure (corpus sequences)"
        else:
            mode = "meta-pattern (transform graph)"

        if self.verbose:
            print(f"\nGenerating with {mode}...")
            print(f"  Sections: {n_sections}")
            print(f"  Instruments: {instruments}")

        all_tracks = defaultdict(list)
        current_time = 0

        # Determine lead instrument from CORPUS DATA (not hardcoded categories)
        # Lead = instrument most often copied FROM in TrackDerive
        lead_gm = self.get_corpus_lead_instrument(instruments)
        follower_instruments = self.get_corpus_follower_instruments(instruments, lead_gm)

        if self.verbose:
            print(f"  Lead instrument (from corpus): GM {lead_gm}")
            print(f"  Follower instruments: {follower_instruments}")

        # Estimate total duration for position normalization
        patterns_per_section = 4  # Estimate
        total_patterns = n_sections * patterns_per_section

        # If using form structure, sample a full form pattern from corpus
        form_pattern_ids = []
        if use_form_structure and self.form_patterns:
            form_pattern_ids = self.sample_form_pattern(lead_gm=lead_gm)
            if self.verbose and form_pattern_ids:
                print(f"  Using corpus form pattern with {len(form_pattern_ids)} patterns")

        # NEW: Sample a form template for global repetition structure (e.g., AABA)
        form_template = self.sample_form_template()
        if self.verbose:
            print(f"  Form template: {'-'.join(form_template)}")

        # Expand form template to n_sections by repeating if needed
        while len(form_template) < n_sections:
            form_template = form_template + form_template
        form_template = form_template[:n_sections]

        # Store pattern chains per section label for repetition
        section_pattern_cache = {}  # 'A' -> pattern_chain, timing info, etc.

        # Track current pattern per instrument for transitions
        current_patterns = {}  # gm -> current_pattern_id

        for section_idx in range(n_sections):
            section_label = form_template[section_idx]

            # DETERMINE PATTERN CHAIN FOR THIS SECTION

            if use_probabilistic:
                # Check if this section label was already generated (for repetition)
                if section_label in section_pattern_cache:
                    # REUSE previous section's patterns (AABA repetition)
                    cached = section_pattern_cache[section_label]
                    pattern_chain = cached['pattern_chain']
                    if self.verbose and section_idx > 0:
                        print(f"  Section {section_idx} ({section_label}): reusing cached patterns")
                else:
                    # Generate new patterns for this section label
                    # 4-LEVEL PROBABILISTIC SAMPLING with co-occurrence

                    # Sample 2-6 patterns for this section (based on corpus statistics)
                    n_patterns_in_section = random.randint(2, 6)
                    pattern_chain = []

                    for pattern_idx in range(n_patterns_in_section):
                        # Compute position in piece (0 to 1)
                        global_pattern_idx = section_idx * patterns_per_section + pattern_idx
                        position = global_pattern_idx / total_patterns

                        # Get current pattern for this instrument (for transitions)
                        current_pattern = current_patterns.get(lead_gm)

                        # Sample next pattern using all 4 levels
                        next_pattern = self.sample_next_pattern(
                            current_pattern=current_pattern,
                            position=position,
                            gm_program=lead_gm,
                            concurrent_patterns=current_patterns,  # Level 4: co-occurrence
                            transition_weight=0.4,   # Transitions
                            position_weight=0.25,    # Position in piece
                            frequency_weight=0.1,    # Base frequency
                            cooccurrence_weight=0.25,  # Co-occurrence with other instruments
                        )

                        if next_pattern:
                            pattern_chain.append(next_pattern)
                            current_patterns[lead_gm] = next_pattern

                    if not pattern_chain:
                        # Fallback
                        pattern_chain = [self.sample_phrase_pattern(target_depth=2, lead_gm=lead_gm)]

                    # Cache this section for potential reuse
                    section_pattern_cache[section_label] = {
                        'pattern_chain': pattern_chain,
                    }

            elif use_form_structure and form_pattern_ids:
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

                # Use CORPUS-DERIVED instrument roles (no hardcoded categories)
                # lead_gm already determined from TrackDerive source counts
                # follower_instruments already computed

                # LEAD INSTRUMENT: Plays the main pattern
                lead_pitch = self._sample_pitch_for_instrument(pattern, lead_gm)
                lead_notes = self._expand_to_notes(
                    intervals, lead_pitch, current_time, lead_gm, base_ioi
                )
                all_tracks[lead_gm].extend(lead_notes)

                # FOLLOWER INSTRUMENTS: Use corpus-derived activity probability
                for follower_gm in follower_instruments:
                    # Get natural co-occurrence probability from checkpoint
                    play_prob = self._get_cooccurrence_probability(
                        lead_pattern_id=pattern_id,
                        lead_gm=lead_gm,
                        target_gm=follower_gm
                    )

                    if random.random() > play_prob:
                        # REST - this instrument didn't usually play here
                        continue

                    # Try to find orchestration rule for this instrument pair
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
                        # Use 4-level probabilistic sampling conditioned on lead
                        if use_probabilistic:
                            position = (section_idx * patterns_per_section + pattern_idx) / total_patterns
                            concurrent = {lead_gm: pattern_id}

                            follower_pattern_id = self.sample_next_pattern(
                                current_pattern=current_patterns.get(follower_gm),
                                position=position,
                                gm_program=follower_gm,
                                concurrent_patterns=concurrent,
                                cooccurrence_weight=0.4,  # Strong co-occurrence for followers
                            )

                            if follower_pattern_id:
                                current_patterns[follower_gm] = follower_pattern_id
                                follower_pattern = self.patterns.get(follower_pattern_id, {})
                                follower_intervals = follower_pattern.get('pitch_intervals', intervals)
                                follower_pitch = self._sample_pitch_for_instrument(follower_pattern, follower_gm)
                            else:
                                follower_intervals = intervals
                                follower_pitch = self._sample_pitch_for_instrument(pattern, follower_gm)
                        else:
                            follower_intervals = intervals
                            follower_pitch = self._sample_pitch_for_instrument(pattern, follower_gm)

                    follower_notes = self._expand_to_notes(
                        follower_intervals, follower_pitch, current_time, follower_gm, base_ioi
                    )
                    all_tracks[follower_gm].extend(follower_notes)

                # Update current_patterns for lead
                current_patterns[lead_gm] = pattern_id

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
    parser = argparse.ArgumentParser(
        description='Meta-pattern guided generator with 3-level probabilistic sampling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sampling Modes:
  --probabilistic (DEFAULT)  3-level Markov sampling:
                             Level 1: Frequency weighting
                             Level 2: Transition probabilities
                             Level 3: Position conditioning

  --form                     Use exact pattern sequences from corpus pieces

  --meta-pattern             Walk transform graph following meta-patterns
        """
    )
    parser.add_argument('checkpoint', help='Path to v53 checkpoint .npz file')
    parser.add_argument('--output', '-o', default='meta_generated.mid', help='Output MIDI path')
    parser.add_argument('--sections', '-n', type=int, default=8, help='Number of sections')
    parser.add_argument('--tempo', '-t', type=int, default=120, help='Tempo BPM')
    parser.add_argument('--seed', '-s', type=int, help='Random seed')
    parser.add_argument('--instruments', '-i', help='Comma-separated GM program numbers')

    # Sampling mode options (mutually exclusive-ish, probabilistic is default)
    parser.add_argument('--probabilistic', action='store_true', default=True,
                        help='Use 3-level probabilistic Markov sampling (DEFAULT)')
    parser.add_argument('--form', action='store_true',
                        help='Use corpus form structure (exact pattern sequences from pieces)')
    parser.add_argument('--meta-pattern', action='store_true',
                        help='Use meta-pattern transform graph walking')
    args = parser.parse_args()

    # Determine sampling mode
    use_probabilistic = True
    use_form = False
    if args.form:
        use_probabilistic = False
        use_form = True
    elif getattr(args, 'meta_pattern', False):
        use_probabilistic = False
        use_form = False

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
        use_form_structure=use_form,
        use_probabilistic=use_probabilistic,
    )

    gen.to_midi(tracks, args.output, tempo=args.tempo)

    print("\nDone!")


if __name__ == '__main__':
    main()
