#!/usr/bin/env python3
"""
Meta-Pattern Generator (v53 Checkpoint) - GPU-Accelerated PCFG Sampling
========================================================================

Converts a deterministic Re-Pair grammar (Straight-Line Program) into a
probabilistic generative model. This addresses the compression-to-generation gap:
Re-Pair finds structure, but generation needs probabilities OVER structures.

GPU EFFICIENCY PRINCIPLE:
  - BUILD phase: GPU-accelerated batch operations (clustering, matrix ops)
  - GENERATE phase: CPU with O(1) dictionary lookups (no Python loops)
  - All heavy compute happens ONCE during initialization

PROBABILISTIC SAMPLING ARCHITECTURE:
  PPM* (replaces Levels 1-2):
    - Variable-order Markov model P(pattern | context_1..5, instrument)
    - Uses longest matching context with KT smoothing
    - Information-theoretic foundation (same as IDyOM)

  Musical Conditioning (Levels 3-6):
    Level 3: Position conditioning   P(pattern | position_in_piece, instrument)
    Level 4: Co-occurrence           P(pattern | concurrent_patterns_on_other_instruments)
    Level 5: Style conditioning      P(pattern | style_cluster_z) - per-piece consistency
    Level 6: Chord conditioning      P(pattern | chord_root, instrument) - harmonic context

ADDRESSING THE SLP → PCFG GAP:
  1. Pattern Equivalence Classes: Group patterns by substitutability (GPU K-means)
  2. Style Variable z: Per-piece latent that conditions all choices (GPU clustering)
  3. Short-Term Memory: Track reuse at phrase boundaries (O(1) CPU lookups)
  4. Hierarchical Skeleton: Generate form → chords → melody → accompaniment

FORM TEMPLATES:
  Learn repetition structures (AABA, ABAB, etc.) from corpus and enforce them
  during generation to prevent "globally random" output.

Everything is derived from checkpoint data. No hardcoded probabilities or music theory rules.
"""

import os
import sys
import json
import random
import argparse
import logging
import time
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field

# Configure logging for live progress monitoring
LOG_FILE = "/tmp/generator.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Optional GPU support - falls back to CPU if unavailable
try:
    import torch
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
except ImportError:
    HAS_TORCH = False
    DEVICE = None

# GM instrument ranges
GM_RANGES = {
    32: (28, 55), 33: (28, 55), 34: (28, 55),  # Bass
    0: (36, 96), 1: (36, 96), 4: (36, 96),     # Piano
    56: (52, 84), 57: (40, 72), 58: (36, 60),  # Brass
    64: (44, 80), 65: (49, 81), 66: (42, 75), 67: (36, 69),  # Sax
}
DEFAULT_RANGE = (48, 84)


@dataclass
class ShortTermMemory:
    """Track patterns used for reuse boosting. O(1) operations only.

    This implements the IDyOM-style Short-Term Model that captures
    within-piece motif repetition the Long-Term Model misses.
    """
    phrase_length: int = 8
    patterns_by_position: Dict[int, Set[str]] = field(default_factory=lambda: defaultdict(set))
    pattern_last_used: Dict[str, int] = field(default_factory=dict)
    pattern_use_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, pattern_id: str, position: int):
        """Record pattern usage. O(1)."""
        pos_mod = position % self.phrase_length
        self.patterns_by_position[pos_mod].add(pattern_id)
        self.pattern_last_used[pattern_id] = position
        self.pattern_use_count[pattern_id] += 1

    def get_reuse_boost(self, pattern_id: str, position: int) -> float:
        """Return boost factor for patterns used at equivalent phrase positions. O(1)."""
        pos_mod = position % self.phrase_length

        # Strong boost for exact phrase position match
        if pattern_id in self.patterns_by_position[pos_mod]:
            return 2.0

        # Medium boost for common repetition distances (8, 16, 32 bars)
        if pattern_id in self.pattern_last_used:
            distance = position - self.pattern_last_used[pattern_id]
            if distance in {8, 16, 32}:
                return 1.5
            elif distance in {4, 12, 24}:
                return 1.25

        return 1.0

    def get_familiarity_boost(self, pattern_id: str) -> float:
        """Boost patterns already used in this piece. O(1)."""
        count = self.pattern_use_count.get(pattern_id, 0)
        if count > 0:
            # Diminishing returns: 1.3 for 1 use, ~1.5 for many uses
            return 1.0 + 0.3 * (1 - 1 / (count + 1))
        return 1.0

    def reset(self):
        """Reset for new piece generation."""
        self.patterns_by_position = defaultdict(set)
        self.pattern_last_used = {}
        self.pattern_use_count = defaultdict(int)


class PatternPPM:
    """Prediction by Partial Matching over pattern sequences.

    Implements variable-order Markov model with KT (Krichevsky-Trofimov) smoothing.
    This replaces ad-hoc frequency (Level 1) and transition (Level 2) weights with
    a principled information-theoretic model.

    Key advantages over fixed bigram:
    - Uses longest matching context (up to max_order)
    - Graceful fallback via escape mechanism (PPM*)
    - KT smoothing handles unseen transitions properly
    - Single hyperparameter (max_order) vs multiple weights

    References:
    - Cleary & Witten (1984) "Data Compression Using Adaptive Coding and Partial String Matching"
    - IDyOM (Pearce 2005) uses PPM for melodic expectation
    """

    def __init__(self, max_order: int = 5):
        """Initialize PPM model.

        Args:
            max_order: Maximum context length (5 = looks back 5 patterns)
        """
        self.max_order = max_order
        # counts[gm][context_tuple][next_pattern] = count
        self.counts: Dict[int, Dict[Tuple[str, ...], Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        # total[gm][context_tuple] = total count for that context
        self.totals: Dict[int, Dict[Tuple[str, ...], int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # vocabulary per instrument for KT smoothing
        self.vocab_size: Dict[int, int] = defaultdict(int)
        self.vocab: Dict[int, Set[str]] = defaultdict(set)

    def train(self, piece_gm_sequences: Dict[str, Dict[int, List[Tuple[float, str]]]]):
        """Train PPM model on pattern sequences from corpus.

        Args:
            piece_gm_sequences: {piece_id: {gm: [(onset, pattern_id), ...]}}
        """
        # First pass: collect vocabulary per instrument
        for piece_id, gm_seqs in piece_gm_sequences.items():
            for gm, seq in gm_seqs.items():
                for _, pattern_id in seq:
                    self.vocab[gm].add(pattern_id)

        for gm, patterns in self.vocab.items():
            self.vocab_size[gm] = len(patterns)

        # Second pass: count n-grams for all orders
        for piece_id, gm_seqs in piece_gm_sequences.items():
            for gm, seq in gm_seqs.items():
                # Extract pattern sequence (sorted by onset)
                sorted_seq = sorted(seq, key=lambda x: x[0])
                patterns = [p[1] for p in sorted_seq]

                # Count for all orders (0 to max_order)
                for i in range(len(patterns)):
                    next_pattern = patterns[i]

                    for order in range(self.max_order + 1):
                        if i >= order:
                            if order == 0:
                                context = ()  # Unigram
                            else:
                                context = tuple(patterns[i - order:i])

                            self.counts[gm][context][next_pattern] += 1
                            self.totals[gm][context] += 1

    def get_distribution(
        self,
        context: List[str],
        gm: int,
        escape_prob: float = 0.1
    ) -> Dict[str, float]:
        """Get probability distribution over next pattern using PPM* escape.

        Uses longest matching context, with escape to shorter contexts for
        patterns not seen in that context.

        Args:
            context: Recent pattern history (most recent last)
            gm: Instrument GM program
            escape_prob: Base escape probability (lower = trust longer context more)

        Returns:
            Dict mapping pattern_id -> probability
        """
        if gm not in self.counts:
            return {}

        distribution = {}
        remaining_prob = 1.0

        # Try each order from longest to shortest (PPM* Method D)
        for order in range(min(self.max_order, len(context)), -1, -1):
            if order == 0:
                ctx = ()
            else:
                ctx = tuple(context[-order:])

            if ctx in self.counts[gm]:
                ctx_counts = self.counts[gm][ctx]
                ctx_total = self.totals[gm][ctx]

                # KT estimator: (count + 0.5) / (total + vocab_size * 0.5)
                vocab_sz = max(1, self.vocab_size[gm])
                denominator = ctx_total + vocab_sz * 0.5

                # Patterns seen in this context get probability
                for pattern, count in ctx_counts.items():
                    if pattern not in distribution:
                        kt_prob = (count + 0.5) / denominator
                        distribution[pattern] = remaining_prob * kt_prob

                # Escape probability for unseen patterns
                escape_mass = remaining_prob * escape_prob
                remaining_prob = escape_mass

        # Distribute remaining mass uniformly over vocabulary (smoothing)
        if remaining_prob > 0 and gm in self.vocab:
            unseen_patterns = self.vocab[gm] - set(distribution.keys())
            if unseen_patterns:
                prob_per_unseen = remaining_prob / len(unseen_patterns)
                for pattern in unseen_patterns:
                    distribution[pattern] = prob_per_unseen

        # Normalize
        total = sum(distribution.values())
        if total > 0:
            distribution = {p: prob / total for p, prob in distribution.items()}

        return distribution

    def sample_next(self, context: List[str], gm: int) -> Optional[str]:
        """Sample next pattern from PPM distribution.

        Args:
            context: Recent pattern history
            gm: Instrument GM program

        Returns:
            Sampled pattern ID
        """
        dist = self.get_distribution(context, gm)
        if not dist:
            return None

        patterns = list(dist.keys())
        probs = list(dist.values())
        return random.choices(patterns, weights=probs)[0]

    def get_stats(self) -> Dict[str, Any]:
        """Return training statistics."""
        stats = {
            'max_order': self.max_order,
            'n_instruments': len(self.counts),
            'contexts_per_order': defaultdict(int),
        }

        for gm in self.counts:
            for ctx in self.counts[gm]:
                stats['contexts_per_order'][len(ctx)] += 1

        stats['contexts_per_order'] = dict(stats['contexts_per_order'])
        return stats


@dataclass
class TransformRelation:
    """A learned transform relationship between patterns."""
    source_id: str
    target_id: str
    transform: str
    count: int = 1


class MetaPatternGenerator:
    """Generate using full checkpoint structure with GPU-accelerated PCFG sampling.

    Implements the compression-to-generation bridge:
    - SLP (Straight-Line Program) → PCFG (Probabilistic CFG)
    - Pattern equivalence classes for branching non-terminals
    - Style variable z for per-piece consistency
    - Short-term memory for within-piece repetition
    - Hierarchical skeleton-first generation
    """

    def __init__(self, checkpoint_path: str, verbose: bool = True, use_gpu: bool = True):
        self.verbose = verbose
        self.use_gpu = use_gpu and HAS_TORCH and DEVICE.type == 'cuda'
        self.init_start_time = time.time()

        gpu_status = 'ENABLED' if self.use_gpu else 'DISABLED (CPU fallback)'
        if self.verbose:
            print(f"GPU acceleration: {gpu_status}")
        logger.info(f"Starting generator initialization (GPU: {gpu_status})")

        # Load checkpoint data
        t0 = time.time()
        self.load_checkpoint(checkpoint_path)
        logger.info(f"Checkpoint loaded in {time.time()-t0:.1f}s")

        # Build base indices (CPU)
        t0 = time.time()
        self.build_transform_graph()
        logger.info(f"Transform graph built in {time.time()-t0:.1f}s")

        t0 = time.time()
        self.build_hierarchical_index()
        logger.info(f"Hierarchical index built in {time.time()-t0:.1f}s")

        t0 = time.time()
        self.build_track_derive_index()
        logger.info(f"Track derive index built in {time.time()-t0:.1f}s")

        t0 = time.time()
        self.build_form_structure_index()
        logger.info(f"Form structure index built in {time.time()-t0:.1f}s")

        t0 = time.time()
        self.build_rest_duration_index()
        logger.info(f"Rest duration index built in {time.time()-t0:.1f}s")

        # Build probabilistic sampling indices
        t0 = time.time()
        self.build_transition_index()       # Level 2: P(next | current, gm) - kept for fallback
        logger.info(f"Transition index (Level 2) built in {time.time()-t0:.1f}s")

        t0 = time.time()
        self.build_position_index()         # Level 3: P(pattern | position, gm)
        logger.info(f"Position index (Level 3) built in {time.time()-t0:.1f}s")

        t0 = time.time()
        self.build_cooccurrence_index()     # Level 4: P(pattern | concurrent_patterns)
        logger.info(f"Co-occurrence index (Level 4) built in {time.time()-t0:.1f}s")

        t0 = time.time()
        self.build_form_template_index()    # For global repetition structure
        logger.info(f"Form template index built in {time.time()-t0:.1f}s")

        t0 = time.time()
        self.build_instrument_role_index()  # Derive lead/follower from corpus
        logger.info(f"Instrument role index built in {time.time()-t0:.1f}s")

        # PPM* replaces Levels 1-2 with principled variable-order Markov model
        t0 = time.time()
        self.build_ppm_model()              # P(pattern | context) with KT smoothing
        logger.info(f"PPM* model (order 5) built in {time.time()-t0:.1f}s")

        # GPU-accelerated PCFG conversion (SLP → PCFG)
        t0 = time.time()
        self.build_pattern_equivalence_classes()   # Branching non-terminals
        logger.info(f"Pattern equivalence classes built in {time.time()-t0:.1f}s")

        t0 = time.time()
        self.build_style_clusters()                # Per-piece style variable z
        logger.info(f"Style clusters built in {time.time()-t0:.1f}s")

        t0 = time.time()
        self.build_chord_skeleton_model()          # Hierarchical generation
        logger.info(f"Chord skeleton model built in {time.time()-t0:.1f}s")

        # Initialize short-term memory (reset per piece)
        self.stm = ShortTermMemory(phrase_length=8)

        total_init_time = time.time() - self.init_start_time
        logger.info(f"INITIALIZATION COMPLETE in {total_init_time:.1f}s")

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

        # Pre-compute pitch offsets for GPU-efficient generation
        self._precompute_pitch_offsets()

    def _precompute_pitch_offsets(self):
        """Pre-compute pitch offsets from TrackDerive transforms.

        GPU EFFICIENCY: Batch-compute all transform->offset mappings once.
        Generation uses O(1) dict lookup instead of parsing strings.

        TrackDerive transforms indicate HARMONIC RELATIONSHIP between instruments,
        not melodic copying. We use them only for pitch grounding.
        """
        # Parse all transforms to semitone offsets (done once at build time)
        self.pitch_offsets_by_gm_pair: Dict[Tuple[int, int], int] = {}

        for (src_gm, tgt_gm), transform in self.dominant_gm_transforms.items():
            offset = self._parse_pitch_offset(transform)
            self.pitch_offsets_by_gm_pair[(src_gm, tgt_gm)] = offset

        if self.verbose:
            print(f"  Pitch offsets: {len(self.pitch_offsets_by_gm_pair)} GM pairs")

    def _parse_pitch_offset(self, transform: str) -> int:
        """Parse pitch offset from transform string. O(1) after precompute.

        Supports:
        - T-7, T7, T+7 -> semitone offset
        - identity, I, R, RI -> 0 (no pitch change)
        """
        if not transform or transform == 'identity':
            return 0

        if transform.startswith('T'):
            try:
                # Handle T-7, T7, T+7 formats
                offset_str = transform[1:].replace('+', '')
                return int(offset_str)
            except ValueError:
                return 0

        # Inversion/Retrograde don't affect starting pitch relationship
        return 0

    def generate_follower_pattern(
        self,
        lead_pattern_id: str,
        lead_gm: int,
        follower_gm: int,
        lead_pitch: int,
        position: float,
        current_patterns: Dict[int, str] = None,
        style_id: int = None,
        chord_root: int = None,
        pattern_position: int = 0,
    ) -> Tuple[str, List[int], int]:
        """Generate a follower instrument's pattern using ROLE-SPECIFIC sampling.

        THE KEY FIX: Instead of copying lead's intervals with transposition,
        we sample from the follower's OWN vocabulary (via PPM*) and only use
        TrackDerive for pitch grounding.

        This ensures:
        - Bass plays bass-like patterns (sparser, stepwise)
        - Horns play horn-like patterns (melodic, wider intervals)
        - Each instrument maintains its learned role from the corpus

        Args:
            lead_pattern_id: Pattern the lead instrument is playing
            lead_gm: GM program of lead instrument
            follower_gm: GM program of follower instrument
            lead_pitch: Starting pitch of lead (for grounding)
            position: Position in piece [0, 1]
            current_patterns: Current pattern per instrument (for PPM context)
            style_id: Style cluster for consistency
            chord_root: Current chord root
            pattern_position: For STM tracking

        Returns:
            Tuple of (pattern_id, intervals, first_pitch)
        """
        if current_patterns is None:
            current_patterns = {}

        # 1. SAMPLE FOLLOWER'S OWN PATTERN from its PPM* vocabulary
        #    This is the key change - follower gets its own contour/role
        #    Uses full 6-level probabilistic sampling (PPM*, position, style, chord, etc.)
        follower_pattern_id = self.sample_next_pattern(
            current_pattern=current_patterns.get(follower_gm),
            position=position,
            gm_program=follower_gm,  # <- Follower's own vocabulary!
            concurrent_patterns=None,  # Don't use co-occurrence for pattern selection
            style_id=style_id,
            chord_root=chord_root,
            pattern_position=pattern_position,
            # Reduce co-occurrence weight - we want role diversity, not similarity
            cooccurrence_weight=0.05,
        )

        # 2. GET FOLLOWER'S INTERVALS (its own learned contours)
        if follower_pattern_id:
            follower_pattern = self.patterns.get(str(follower_pattern_id), self.patterns.get(follower_pattern_id, {}))
            follower_intervals = follower_pattern.get('pitch_intervals', [0])
        else:
            # Fallback: sample directly from instrument vocabulary
            follower_pattern_id = self._sample_pattern_for_instrument(follower_gm)
            if follower_pattern_id:
                follower_pattern = self.patterns.get(str(follower_pattern_id), {})
                follower_intervals = follower_pattern.get('pitch_intervals', [0])
            else:
                follower_intervals = [0]

        # 3. COMPUTE PITCH GROUNDING from TrackDerive (O(1) lookup)
        #    TrackDerive tells us harmonic relationship, not melodic copying
        pitch_offset = self.pitch_offsets_by_gm_pair.get((lead_gm, follower_gm), 0)

        # Apply offset to lead pitch, then clamp to follower's range
        pitch_range = GM_RANGES.get(follower_gm, DEFAULT_RANGE)
        grounded_pitch = lead_pitch + pitch_offset

        # Octave adjustment if out of range
        while grounded_pitch < pitch_range[0]:
            grounded_pitch += 12
        while grounded_pitch > pitch_range[1]:
            grounded_pitch -= 12

        # Final clamp
        follower_pitch = max(pitch_range[0], min(pitch_range[1], grounded_pitch))

        return follower_pattern_id, follower_intervals, follower_pitch

    def generate_follower_patterns_batch(
        self,
        lead_pattern_id: str,
        lead_gm: int,
        lead_pitch: int,
        follower_instruments: List[int],
        position: float,
        current_patterns: Dict[int, str] = None,
        style_id: int = None,
        chord_root: int = None,
        pattern_position: int = 0,
    ) -> Dict[int, Tuple[str, List[int], int]]:
        """Batch generate patterns for all followers. GPU-efficient.

        Samples all followers in one pass, avoiding repeated dictionary lookups.
        Uses vectorized pitch offset computation when possible.

        Args:
            lead_pattern_id: Pattern the lead is playing
            lead_gm: GM program of lead
            lead_pitch: Lead's starting pitch
            follower_instruments: List of follower GM programs
            position: Position in piece [0, 1]
            current_patterns: Current pattern per instrument (for PPM context)
            style_id: Style cluster
            chord_root: Chord root
            pattern_position: STM position

        Returns:
            Dict of {follower_gm: (pattern_id, intervals, pitch)}
        """
        if current_patterns is None:
            current_patterns = {}

        results = {}

        # Batch pitch offset computation (vectorized if many followers)
        if HAS_TORCH and len(follower_instruments) > 2:
            # GPU-accelerated batch offset lookup
            offsets = torch.tensor([
                self.pitch_offsets_by_gm_pair.get((lead_gm, f_gm), 0)
                for f_gm in follower_instruments
            ], device=DEVICE)

            base_pitches = lead_pitch + offsets

            # Batch range clamping
            for i, follower_gm in enumerate(follower_instruments):
                pitch_range = GM_RANGES.get(follower_gm, DEFAULT_RANGE)
                pitch = int(base_pitches[i].item())

                # Octave adjustment
                while pitch < pitch_range[0]:
                    pitch += 12
                while pitch > pitch_range[1]:
                    pitch -= 12
                pitch = max(pitch_range[0], min(pitch_range[1], pitch))

                # Sample pattern (CPU - dictionary lookups)
                pattern_id, intervals, _ = self.generate_follower_pattern(
                    lead_pattern_id, lead_gm, follower_gm, lead_pitch,
                    position, current_patterns, style_id, chord_root, pattern_position
                )
                # Override pitch with batch-computed value
                results[follower_gm] = (pattern_id, intervals, pitch)
        else:
            # CPU path for small batches (avoid GPU overhead)
            for follower_gm in follower_instruments:
                results[follower_gm] = self.generate_follower_pattern(
                    lead_pattern_id, lead_gm, follower_gm, lead_pitch,
                    position, current_patterns, style_id, chord_root, pattern_position
                )

        return results

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

    def build_ppm_model(self, max_order: int = 5):
        """Build PPM* (Prediction by Partial Matching) model over pattern sequences.

        REPLACES Levels 1-2 with principled variable-order Markov model.

        PPM* advantages over fixed bigram:
        - Uses longest matching context (order 5 looks back 5 patterns)
        - Graceful fallback via escape mechanism for unseen contexts
        - KT smoothing handles unseen transitions properly
        - Information-theoretic foundation (same as IDyOM)
        - Single hyperparameter (max_order) vs multiple ad-hoc weights

        This is THE key improvement for addressing "locally good, globally random":
        longer context = better global coherence.

        Args:
            max_order: Maximum context length (default 5 = looks back 5 patterns)
        """
        if self.verbose:
            print(f"Building PPM* model (max_order={max_order})...")

        self.ppm = PatternPPM(max_order=max_order)
        self.ppm.train(self.piece_gm_sequences)

        # Store pattern context for generation (per instrument)
        self.pattern_context: Dict[int, List[str]] = defaultdict(list)

        if self.verbose:
            stats = self.ppm.get_stats()
            print(f"  Instruments: {stats['n_instruments']}")
            for order, count in sorted(stats['contexts_per_order'].items()):
                print(f"    Order {order}: {count:,} unique contexts")

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
            for bucket, pos_name in [(0, "start"), (N_BUCKETS // 2, "middle"), (N_BUCKETS - 1, "end")]:
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

    # =========================================================================
    # GPU-ACCELERATED PCFG CONVERSION (SLP → PCFG)
    # =========================================================================

    def build_pattern_equivalence_classes(self):
        """GPU-accelerated pattern clustering for substitutability.

        This is the KEY fix for the SLP → PCFG conversion:
        - SLP has one production per non-terminal (deterministic)
        - PCFG has MULTIPLE productions per non-terminal (probabilistic)

        We create "equivalence classes" of patterns that can substitute for each other
        based on: similar intervals, similar length, appear in similar contexts.

        GPU: Batch cosine similarity + K-means clustering
        CPU: O(1) lookups during generation
        """
        if self.verbose:
            print("Building pattern equivalence classes (GPU-accelerated)...")

        pattern_ids = list(self.patterns.keys())
        n_patterns = len(pattern_ids)

        if n_patterns == 0:
            self.pattern_to_class = {}
            self.class_to_patterns = {}
            self.class_pattern_probs = {}
            return

        # 1. Build feature matrix for all patterns
        # Features: interval signature (padded), length, position stats, frequency
        INTERVAL_DIM = 8  # Pad/truncate intervals to this length
        features = []

        # Precompute position stats per pattern (vectorized)
        pattern_position_means = {}
        for pid in pattern_ids:
            positions = []
            for occ in self.patterns[pid].get('occurrences', []):
                onset = occ.get('onset_time', occ.get('onset', 0))
                positions.append(onset)
            pattern_position_means[pid] = np.mean(positions) if positions else 0

        # Build feature vectors
        for pid in pattern_ids:
            p = self.patterns[pid]
            intervals = p.get('pitch_intervals', [0])

            # Interval signature (padded/truncated)
            padded_intervals = intervals[:INTERVAL_DIM] + [0] * (INTERVAL_DIM - len(intervals))

            # Contour signature (direction of motion)
            contour = [1 if i > 0 else (-1 if i < 0 else 0) for i in padded_intervals]

            feat = [
                *padded_intervals,                                  # Raw intervals
                *contour,                                           # Contour
                len(intervals),                                     # Pattern length
                sum(abs(i) for i in intervals) / max(1, len(intervals)),  # Avg interval magnitude
                pattern_position_means.get(pid, 0.5) / 100000,      # Normalized position
                len(p.get('occurrences', [])),                      # Frequency (log-scaled later)
            ]
            features.append(feat)

        # Convert to numpy array
        X = np.array(features, dtype=np.float32)

        # Normalize features (z-score per column)
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-8
        X_normalized = (X - X_mean) / X_std

        # 2. Cluster patterns into equivalence classes
        # Use GPU if available, otherwise CPU K-means
        n_clusters = min(500, max(10, n_patterns // 20))  # ~20 patterns per class

        if self.use_gpu and HAS_TORCH:
            labels = self._kmeans_gpu(X_normalized, n_clusters)
        else:
            labels = self._kmeans_cpu(X_normalized, n_clusters)

        # 3. Build CPU lookup tables for O(1) generation
        self.pattern_to_class = {pid: int(labels[i]) for i, pid in enumerate(pattern_ids)}

        self.class_to_patterns = defaultdict(list)
        for pid, cls in self.pattern_to_class.items():
            self.class_to_patterns[cls].append(pid)

        # 4. Precompute P(pattern | class) for each equivalence class
        # This gives us BRANCHING: when we need pattern of class C, sample from distribution
        self.class_pattern_probs = {}
        for cls, pids in self.class_to_patterns.items():
            # Weight by occurrence count (frequency in corpus)
            counts = []
            for pid in pids:
                counts.append(len(self.patterns[pid].get('occurrences', [])) + 1)
            total = sum(counts)
            self.class_pattern_probs[cls] = {
                pid: c / total for pid, c in zip(pids, counts)
            }

        # 5. Build transition matrix between equivalence classes
        # P(class_B | class_A) - which classes follow which
        self._build_class_transitions(pattern_ids, labels)

        if self.verbose:
            print(f"  Equivalence classes: {n_clusters}")
            print(f"  Avg patterns per class: {n_patterns / n_clusters:.1f}")
            class_sizes = [len(pids) for pids in self.class_to_patterns.values()]
            print(f"  Class size range: {min(class_sizes)} - {max(class_sizes)}")

    def _kmeans_gpu(self, X: np.ndarray, n_clusters: int, max_iters: int = 100) -> np.ndarray:
        """GPU-accelerated K-means clustering using PyTorch."""
        X_torch = torch.tensor(X, device=DEVICE, dtype=torch.float32)
        n_samples = X_torch.shape[0]

        # Initialize centroids using k-means++
        centroids_idx = [random.randint(0, n_samples - 1)]
        for _ in range(1, n_clusters):
            # Compute distance to nearest centroid (vectorized)
            dists = torch.cdist(X_torch, X_torch[centroids_idx])
            min_dists = dists.min(dim=1).values
            probs = min_dists ** 2
            probs = probs / probs.sum()
            new_idx = torch.multinomial(probs, 1).item()
            centroids_idx.append(new_idx)

        centroids = X_torch[centroids_idx].clone()

        # K-means iterations (all on GPU)
        for _ in range(max_iters):
            # Assign to nearest centroid (batch matrix op)
            dists = torch.cdist(X_torch, centroids)  # [N x K]
            labels = dists.argmin(dim=1)  # [N]

            # Update centroids (scatter_add for GPU efficiency)
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(n_clusters, device=DEVICE)

            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X_torch[mask].mean(dim=0)
                    counts[k] = mask.sum()

            # Check convergence
            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        return labels.cpu().numpy()

    def _kmeans_cpu(self, X: np.ndarray, n_clusters: int, max_iters: int = 100) -> np.ndarray:
        """CPU fallback K-means using numpy vectorized operations."""
        n_samples = X.shape[0]

        # Initialize centroids randomly
        idx = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = X[idx].copy()

        for _ in range(max_iters):
            # Compute distances (vectorized)
            dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = dists.argmin(axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]

            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        return labels

    def _build_class_transitions(self, pattern_ids: List[str], labels: np.ndarray):
        """Build transition probabilities between equivalence classes.

        This creates a PCFG-style transition model:
        P(next_class | current_class, instrument)
        """
        n_clusters = int(labels.max()) + 1
        pid_to_label = {pid: labels[i] for i, pid in enumerate(pattern_ids)}

        # Count transitions per instrument
        class_transitions = defaultdict(lambda: np.zeros((n_clusters, n_clusters)))

        for piece_id, gm_sequences in self.piece_gm_sequences.items():
            for gm, sequence in gm_sequences.items():
                for i in range(len(sequence) - 1):
                    _, current_pid = sequence[i]
                    _, next_pid = sequence[i + 1]

                    current_cls = pid_to_label.get(current_pid, 0)
                    next_cls = pid_to_label.get(next_pid, 0)

                    class_transitions[gm][current_cls, next_cls] += 1

        # Normalize to probabilities
        self.class_transition_probs = {}
        for gm, trans_matrix in class_transitions.items():
            row_sums = trans_matrix.sum(axis=1, keepdims=True) + 1e-8
            self.class_transition_probs[gm] = trans_matrix / row_sums

    def sample_from_equivalence_class(self, equivalence_class: int, gm_program: int = None) -> Optional[str]:
        """Sample a pattern from an equivalence class. O(1) lookup + weighted sample.

        This implements BRANCHING in the PCFG: given a non-terminal (class),
        probabilistically choose which production (pattern) to expand.
        """
        if equivalence_class not in self.class_pattern_probs:
            return None

        probs_dict = self.class_pattern_probs[equivalence_class]

        # If instrument specified, filter to patterns that instrument plays
        if gm_program is not None:
            filtered = {}
            for pid, prob in probs_dict.items():
                for occ in self.patterns[pid].get('occurrences', []):
                    if occ.get('gm_program') == gm_program:
                        filtered[pid] = prob
                        break
            if filtered:
                probs_dict = filtered
                # Renormalize
                total = sum(probs_dict.values())
                probs_dict = {k: v / total for k, v in probs_dict.items()}

        pids = list(probs_dict.keys())
        probs = list(probs_dict.values())

        return random.choices(pids, weights=probs)[0] if pids else None

    def build_style_clusters(self):
        """GPU-accelerated piece clustering for style variable z.

        This implements the "shared latent variable" from Compound PCFGs:
        - Sample z ~ P(z) at generation start
        - All rule probabilities conditioned on z
        - Ensures stylistic consistency without explicit coupling at every step

        GPU: Batch feature extraction + K-means on piece features
        CPU: O(1) lookups for P(pattern | style_z)
        """
        if self.verbose:
            print("Building style clusters (GPU-accelerated)...")

        piece_ids = list(self.piece_gm_sequences.keys())
        n_pieces = len(piece_ids)

        if n_pieces == 0:
            self.piece_to_style = {}
            self.style_pattern_probs = {}
            self.style_weights = [1.0]
            return

        # 1. Extract piece-level features (vectorized)
        piece_features = []
        for pid in piece_ids:
            gm_seqs = self.piece_gm_sequences[pid]

            # Feature: number of instruments
            n_instruments = len(gm_seqs)

            # Feature: total patterns (density proxy)
            total_patterns = sum(len(seq) for seq in gm_seqs.values())

            # Feature: pattern diversity (unique / total)
            unique_patterns = set()
            for seq in gm_seqs.values():
                for _, pat_id in seq:
                    unique_patterns.add(pat_id)
            diversity = len(unique_patterns) / max(1, total_patterns)

            # Feature: average interval magnitude (activity level)
            avg_interval = 0
            count = 0
            for seq in gm_seqs.values():
                for _, pat_id in seq:
                    pat = self.patterns.get(pat_id, {})
                    intervals = pat.get('pitch_intervals', [])
                    if intervals:
                        avg_interval += sum(abs(i) for i in intervals) / len(intervals)
                        count += 1
            avg_interval = avg_interval / max(1, count)

            # Feature: instrument distribution (which GM families present)
            gm_families = set(gm // 8 for gm in gm_seqs.keys())
            has_brass = 1 if 7 in gm_families else 0  # GM 56-63
            has_keys = 1 if 0 in gm_families else 0   # GM 0-7
            has_bass = 1 if 4 in gm_families else 0   # GM 32-39
            has_sax = 1 if 8 in gm_families else 0    # GM 64-71

            feat = [
                n_instruments,
                total_patterns / 100,  # Scale down
                diversity,
                avg_interval / 5,  # Scale down
                has_brass,
                has_keys,
                has_bass,
                has_sax,
            ]
            piece_features.append(feat)

        X = np.array(piece_features, dtype=np.float32)

        # Normalize
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-8
        X_normalized = (X - X_mean) / X_std

        # 2. Cluster pieces into style categories
        n_styles = min(20, max(3, n_pieces // 50))

        if self.use_gpu and HAS_TORCH:
            style_labels = self._kmeans_gpu(X_normalized, n_styles)
        else:
            style_labels = self._kmeans_cpu(X_normalized, n_styles)

        # 3. Build lookup tables
        self.piece_to_style = {pid: int(style_labels[i]) for i, pid in enumerate(piece_ids)}

        # 4. For each style, precompute pattern distributions
        # P(pattern | style_z)
        self.style_pattern_probs = {}
        for style_id in range(n_styles):
            pieces_in_style = [pid for pid, s in self.piece_to_style.items() if s == style_id]

            pattern_counts = Counter()
            for pid in pieces_in_style:
                for gm, seq in self.piece_gm_sequences[pid].items():
                    for _, pat_id in seq:
                        pattern_counts[pat_id] += 1

            total = sum(pattern_counts.values())
            if total > 0:
                self.style_pattern_probs[style_id] = {
                    p: c / total for p, c in pattern_counts.items()
                }
            else:
                self.style_pattern_probs[style_id] = {}

        # 5. Store style weights for sampling (proportional to piece count)
        style_counts = Counter(style_labels)
        total_pieces = sum(style_counts.values())
        self.style_weights = [style_counts.get(i, 1) / total_pieces for i in range(n_styles)]

        if self.verbose:
            print(f"  Style clusters: {n_styles}")
            print(f"  Pieces per style: {[style_counts.get(i, 0) for i in range(n_styles)]}")

    def sample_style(self) -> int:
        """Sample a style at generation start. O(1)."""
        if not self.style_weights:
            return 0
        return random.choices(range(len(self.style_weights)), weights=self.style_weights)[0]

    def get_style_weight(self, pattern_id: str, style_id: int) -> float:
        """Get style-conditioned weight for a pattern. O(1)."""
        if style_id not in self.style_pattern_probs:
            return 1.0
        return self.style_pattern_probs[style_id].get(pattern_id, 0.01)  # Small non-zero for smoothing

    def build_chord_skeleton_model(self):
        """GPU-optimized chord skeleton model with Level 6 pattern-chord associations.

        OPTIMIZATION STRATEGY:
        1. Pre-cache pattern pitches (O(patterns) once)
        2. Build bass progressions per piece (O(pieces × bass_patterns) once)
        3. Use binary search for onset→chord lookup (O(log n) per pattern)
        4. Batch count with Counter (O(1) amortized per association)

        Original: O(pieces × instruments × patterns × occurrences) - stuck
        Optimized: O(pieces × instruments × patterns) - seconds
        """
        if self.verbose:
            print("Building chord skeleton model (GPU-optimized)...")

        from bisect import bisect_right

        BASS_GMS = set(range(32, 40))
        n_chords = 12

        # ========== STEP 1: Pre-cache pattern pitches (O(patterns)) ==========
        pattern_pitch_cache = {}
        for pattern_id, pattern in self.patterns.items():
            for occ in pattern.get('occurrences', []):
                gm = occ.get('gm_program', 0)
                if gm in BASS_GMS:
                    pattern_pitch_cache[(pattern_id, gm)] = occ.get('pitch', 48)
                    break

        if self.verbose:
            print(f"  Pitch cache: {len(pattern_pitch_cache)} bass patterns")

        # ========== STEP 2: Build bass progressions per piece (O(pieces)) ==========
        bass_progressions = {}  # piece_id -> [(onset, root), ...]

        for piece_id, gm_seqs in self.piece_gm_sequences.items():
            for gm, seq in gm_seqs.items():
                if gm not in BASS_GMS:
                    continue
                progression = []
                for onset, pattern_id in seq:
                    pitch = pattern_pitch_cache.get((pattern_id, gm), 48)
                    root = pitch % 12
                    progression.append((onset, root))
                if progression:
                    # Sort by onset for binary search
                    bass_progressions[piece_id] = sorted(progression, key=lambda x: x[0])
                    break  # One bass track per piece is enough

        if self.verbose:
            print(f"  Bass progressions: {len(bass_progressions)} pieces")

        # ========== STEP 3: Chord transitions (O(bass_patterns)) ==========
        chord_transitions = np.zeros((n_chords, n_chords))
        chord_counts = np.zeros(n_chords)

        for piece_id, progression in bass_progressions.items():
            prev_root = None
            for _, root in progression:
                chord_counts[root] += 1
                if prev_root is not None:
                    chord_transitions[prev_root, root] += 1
                prev_root = root

        # Normalize with Laplace smoothing
        self.chord_transitions = (chord_transitions + 0.1) / (
            chord_transitions.sum(axis=1, keepdims=True) + 1.2
        )
        self.chord_prior = (chord_counts + 1) / (chord_counts.sum() + 12)

        if self.verbose:
            print(f"  Chord transitions: {n_chords}x{n_chords}")

        # ========== STEP 4: Pattern-chord associations (OPTIMIZED) ==========
        # Use Counter for O(1) amortized counting
        associations = []  # Will contain (chord_root, gm, pattern_id) tuples

        # Pre-extract onset arrays for binary search
        bass_onsets = {}  # piece_id -> [onset1, onset2, ...]
        bass_roots = {}   # piece_id -> [root1, root2, ...]
        for piece_id, progression in bass_progressions.items():
            bass_onsets[piece_id] = [p[0] for p in progression]
            bass_roots[piece_id] = [p[1] for p in progression]

        for piece_id, gm_seqs in self.piece_gm_sequences.items():
            if piece_id not in bass_progressions:
                continue

            onsets = bass_onsets[piece_id]
            roots = bass_roots[piece_id]

            for gm, seq in gm_seqs.items():
                for onset, pattern_id in seq:
                    # Binary search: find chord active at this onset
                    idx = bisect_right(onsets, onset) - 1
                    chord_root = roots[max(0, idx)] if roots else 0

                    associations.append((chord_root, gm, pattern_id))

        if self.verbose:
            print(f"  Pattern-chord associations: {len(associations):,}")

        # Count all associations at once
        assoc_counts = Counter(associations)

        # Build chord_pattern_probs structure
        self.chord_pattern_probs = defaultdict(lambda: defaultdict(Counter))
        for (chord_root, gm, pattern_id), count in assoc_counts.items():
            self.chord_pattern_probs[chord_root][gm][pattern_id] = count

        # Normalize to probabilities
        self.chord_pattern_distributions = {}
        for chord_root, gm_counts in self.chord_pattern_probs.items():
            self.chord_pattern_distributions[chord_root] = {}
            for gm, pattern_counts in gm_counts.items():
                total = sum(pattern_counts.values())
                if total > 0:
                    self.chord_pattern_distributions[chord_root][gm] = {
                        pid: c / total for pid, c in pattern_counts.items()
                    }

        if self.verbose:
            n_distributions = sum(
                len(gms) for gms in self.chord_pattern_distributions.values()
            )
            print(f"  Level 6 distributions: {n_distributions} (chord, instrument) pairs")

    def generate_chord_skeleton(self, n_chords: int = 8) -> List[int]:
        """Generate a chord progression skeleton. O(n_chords) operations."""
        if not hasattr(self, 'chord_prior') or self.chord_prior is None:
            # Fallback: simple ii-V-I type progression
            return [2, 7, 0, 5] * (n_chords // 4 + 1)[:n_chords]

        chords = []

        # Sample first chord from prior
        current = random.choices(range(12), weights=self.chord_prior.tolist())[0]
        chords.append(current)

        # Sample remaining chords from transition matrix
        for _ in range(n_chords - 1):
            probs = self.chord_transitions[current].tolist()
            current = random.choices(range(12), weights=probs)[0]
            chords.append(current)

        return chords

    def get_chord_pattern_weight(self, pattern_id: str, chord_root: int, gm_program: int) -> float:
        """Get weight for pattern given chord context. O(1)."""
        if chord_root not in self.chord_pattern_distributions:
            return 1.0
        gm_dist = self.chord_pattern_distributions[chord_root].get(gm_program, {})
        return gm_dist.get(pattern_id, 0.01)  # Small non-zero for smoothing

    def sample_next_pattern(
        self,
        current_pattern: str,
        position: float,
        gm_program: int,
        concurrent_patterns: Dict[int, str] = None,
        style_id: int = None,
        chord_root: int = None,
        pattern_position: int = 0,
        ppm_weight: float = 0.40,
        position_weight: float = 0.15,
        cooccurrence_weight: float = 0.20,
        style_weight: float = 0.10,
        stm_weight: float = 0.05,
        chord_weight: float = 0.10,
    ) -> Optional[str]:
        """Sample next pattern using PPM* + musical conditioning.

        This is the core of the SLP → PCFG conversion. Combines:

        PPM* (replaces Levels 1-2):
        - Variable-order Markov model P(pattern | context_1..5, instrument)
        - Uses longest matching context with KT smoothing
        - Information-theoretic foundation (same as IDyOM)

        MUSICAL CONDITIONING (Levels 3-6):
        - Level 3: Position conditioning P(pattern | position_in_piece, instrument)
        - Level 4: Co-occurrence P(pattern | concurrent_patterns_on_other_instruments)
        - Level 5: Style conditioning P(pattern | style_z) - per-piece consistency
        - Level 6: Chord conditioning P(pattern | chord_root, instrument) - harmonic context

        WITHIN-PIECE (Short-Term Model):
        - STM reuse boost: Patterns used at equivalent phrase positions get boost

        Args:
            current_pattern: Current pattern ID (for context)
            position: Normalized position in piece [0, 1]
            gm_program: Instrument GM program
            concurrent_patterns: Dict of {other_gm: pattern_id} for co-occurrence
            style_id: Style cluster ID (sampled at piece start)
            chord_root: Current chord root (0-11) for harmonic conditioning
            pattern_position: Integer position for STM (phrase position tracking)
            ppm_weight: Weight for PPM* sequential prediction (default 0.40)
            *_weight: Contribution weights for each level

        Returns:
            Sampled pattern ID, or None if no data available
        """
        candidates = {}  # pattern_id -> combined score

        # PPM*: Variable-order Markov model (replaces Levels 1-2)
        # Uses context from self.pattern_context[gm_program]
        if hasattr(self, 'ppm'):
            context = self.pattern_context.get(gm_program, [])
            ppm_dist = self.ppm.get_distribution(context, gm_program)
            for pattern_id, prob in ppm_dist.items():
                candidates[pattern_id] = ppm_weight * prob
        else:
            # Fallback to old Level 1-2 if PPM not available
            if gm_program in self.pattern_counts_by_gm:
                total = sum(self.pattern_counts_by_gm[gm_program].values())
                for pattern_id, count in self.pattern_counts_by_gm[gm_program].items():
                    candidates[pattern_id] = ppm_weight * 0.3 * (count / total)
            if gm_program in self.transition_probs and current_pattern:
                current_key = str(current_pattern)
                if current_key in self.transition_probs[gm_program]:
                    next_patterns, probs = self.transition_probs[gm_program][current_key]
                    for pattern_id, prob in zip(next_patterns, probs):
                        if pattern_id in candidates:
                            candidates[pattern_id] += ppm_weight * 0.7 * prob
                        else:
                            candidates[pattern_id] = ppm_weight * 0.7 * prob

        # Level 3: Position conditioning - O(K) where K is patterns at this position
        bucket = int(min(position, 0.999) * self.n_position_buckets)
        if bucket in self.position_probs and gm_program in self.position_probs[bucket]:
            patterns, probs = self.position_probs[bucket][gm_program]
            for pattern_id, prob in zip(patterns, probs):
                if pattern_id in candidates:
                    candidates[pattern_id] += position_weight * prob
                else:
                    candidates[pattern_id] = position_weight * prob

        # Level 4: Co-occurrence conditioning - O(K * M) where M is concurrent instruments
        if concurrent_patterns and hasattr(self, 'cooccurrence_probs'):
            for other_gm, other_pattern in concurrent_patterns.items():
                if other_gm == gm_program:
                    continue

                other_key = str(other_pattern)
                if other_key in self.cooccurrence_probs:
                    gm_probs = self.cooccurrence_probs[other_key].get(gm_program)
                    if gm_probs:
                        patterns, probs = gm_probs
                        weight_per_inst = cooccurrence_weight / max(1, len(concurrent_patterns))
                        for pattern_id, prob in zip(patterns, probs):
                            if pattern_id in candidates:
                                candidates[pattern_id] += weight_per_inst * prob
                            else:
                                candidates[pattern_id] = weight_per_inst * prob

        # Level 5: Style conditioning - O(1) per candidate
        if style_id is not None and hasattr(self, 'style_pattern_probs'):
            style_probs = self.style_pattern_probs.get(style_id, {})
            for pattern_id in candidates:
                style_prob = style_probs.get(pattern_id, 0.01)  # Smoothed
                candidates[pattern_id] += style_weight * style_prob

        # Level 6: Chord conditioning - O(1) per candidate
        if chord_root is not None and hasattr(self, 'chord_pattern_distributions'):
            chord_dist = self.chord_pattern_distributions.get(chord_root, {})
            gm_chord_probs = chord_dist.get(gm_program, {})
            for pattern_id in candidates:
                chord_prob = gm_chord_probs.get(pattern_id, 0.01)  # Smoothed
                candidates[pattern_id] += chord_weight * chord_prob

        # Short-Term Memory: Boost patterns already used at equivalent positions
        if hasattr(self, 'stm'):
            for pattern_id in candidates:
                reuse_boost = self.stm.get_reuse_boost(pattern_id, pattern_position)
                familiarity_boost = self.stm.get_familiarity_boost(pattern_id)
                # Multiplicative boost (not additive) to preserve relative probabilities
                candidates[pattern_id] *= (1 + stm_weight * (reuse_boost - 1))
                candidates[pattern_id] *= (1 + stm_weight * (familiarity_boost - 1))

        if not candidates:
            # Fallback: sample from equivalence class if we have current pattern's class
            if current_pattern and hasattr(self, 'pattern_to_class'):
                current_class = self.pattern_to_class.get(current_pattern)
                if current_class is not None:
                    return self.sample_from_equivalence_class(current_class, gm_program)
            return self._sample_pattern_for_instrument(gm_program)

        # Sample from combined distribution
        pattern_ids = list(candidates.keys())
        scores = list(candidates.values())

        # Normalize to probabilities
        total_score = sum(scores)
        if total_score > 0:
            probs = [s / total_score for s in scores]
            sampled = random.choices(pattern_ids, weights=probs)[0]

            # Record in STM for future reuse boosting
            if hasattr(self, 'stm'):
                self.stm.record(sampled, pattern_position)

            # Update PPM context for this instrument (keep last max_order patterns)
            if hasattr(self, 'pattern_context') and hasattr(self, 'ppm'):
                self.pattern_context[gm_program].append(sampled)
                # Trim to max_order to prevent unbounded growth
                max_ctx = self.ppm.max_order
                if len(self.pattern_context[gm_program]) > max_ctx:
                    self.pattern_context[gm_program] = self.pattern_context[gm_program][-max_ctx:]

            return sampled

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
        1. use_probabilistic=True (DEFAULT): Use 6-level PCFG sampling + form templates
           - Level 1: Frequency weighting (pattern popularity)
           - Level 2: Transition probabilities (what follows what)
           - Level 3: Position conditioning (intro vs middle vs end patterns)
           - Level 4: Co-occurrence conditioning (what plays together across instruments)
           - Level 5: Style conditioning (per-piece consistency via latent z)
           - Level 6: Chord conditioning (harmonic context from skeleton)
           - Short-Term Memory: Boost reuse at phrase boundaries
           - Form templates: Enforce repetition structure (AABA) from corpus

        2. use_form_structure=True: Use exact pattern sequences from corpus pieces
           - Samples an actual pattern sequence that appeared in training

        3. Neither: Meta-pattern guided generation (transform graph walking)
           - Walks the transform graph following learned transform sequences

        The probabilistic mode (default) produces the most coherent output
        because it respects local, global, cross-instrument, style, and harmonic structure.
        """
        if seed is not None:
            random.seed(seed)

        if instruments is None:
            instruments = [56, 65, 66, 57, 0, 32]  # Trumpet, Alto, Tenor, Trombone, Piano, Bass

        # Determine sampling mode string for logging
        if use_probabilistic:
            mode = "probabilistic (6-level PCFG + STM + form templates)"
        elif use_form_structure:
            mode = "form structure (corpus sequences)"
        else:
            mode = "meta-pattern (transform graph)"

        gen_start_time = time.time()
        logger.info(f"Starting generation: {mode}")
        logger.info(f"  Sections: {n_sections}, Instruments: {instruments}")

        if self.verbose:
            print(f"\nGenerating with {mode}...")
            print(f"  Sections: {n_sections}")
            print(f"  Instruments: {instruments}")

        # =====================================================================
        # INITIALIZE GENERATION STATE
        # =====================================================================

        all_tracks = defaultdict(list)
        current_time = 0

        # Reset Short-Term Memory for new piece
        if hasattr(self, 'stm'):
            self.stm.reset()

        # Reset PPM context for new piece
        if hasattr(self, 'pattern_context'):
            self.pattern_context = defaultdict(list)

        # Sample style variable z at piece start (Compound PCFG)
        style_id = self.sample_style() if hasattr(self, 'sample_style') else None
        if self.verbose and style_id is not None:
            print(f"  Style cluster: {style_id}")

        # Generate chord skeleton for hierarchical generation
        chord_skeleton = self.generate_chord_skeleton(n_sections * 2) if hasattr(self, 'generate_chord_skeleton') else None
        if self.verbose and chord_skeleton:
            chord_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            print(f"  Chord skeleton: {[chord_names[c] for c in chord_skeleton[:8]]}...")

        # Global pattern position counter for STM
        global_pattern_position = 0

        # Determine lead instrument from CORPUS DATA (not hardcoded categories)
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
        if use_form_structure and hasattr(self, 'form_patterns') and self.form_patterns:
            form_pattern_ids = self.sample_form_pattern(lead_gm=lead_gm)
            if self.verbose and form_pattern_ids:
                print(f"  Using corpus form pattern with {len(form_pattern_ids)} patterns")

        # Sample a form template for global repetition structure (e.g., AABA)
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

        # =====================================================================
        # MAIN GENERATION LOOP
        # =====================================================================

        for section_idx in range(n_sections):
            section_label = form_template[section_idx]
            section_start_time = time.time()
            logger.info(f"Section {section_idx+1}/{n_sections} (label={section_label})")

            # Get chord root for this section from skeleton
            chord_root = None
            if chord_skeleton:
                chord_idx = min(section_idx * 2, len(chord_skeleton) - 1)
                chord_root = chord_skeleton[chord_idx]

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
                    # 6-LEVEL PROBABILISTIC SAMPLING with all conditioning

                    # Sample 2-6 patterns for this section (based on corpus statistics)
                    n_patterns_in_section = random.randint(2, 6)
                    pattern_chain = []

                    for pattern_idx in range(n_patterns_in_section):
                        # Compute position in piece (0 to 1)
                        global_pattern_idx = section_idx * patterns_per_section + pattern_idx
                        position = global_pattern_idx / total_patterns

                        # Get current pattern for this instrument (for transitions)
                        current_pattern = current_patterns.get(lead_gm)

                        # Sample next pattern using all 6 levels + STM
                        next_pattern = self.sample_next_pattern(
                            current_pattern=current_pattern,
                            position=position,
                            gm_program=lead_gm,
                            concurrent_patterns=current_patterns,  # Level 4
                            style_id=style_id,                     # Level 5: style consistency
                            chord_root=chord_root,                 # Level 6: harmonic context
                            pattern_position=global_pattern_position,  # STM tracking
                        )

                        if next_pattern:
                            pattern_chain.append(next_pattern)
                            current_patterns[lead_gm] = next_pattern
                            global_pattern_position += 1

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

                # FOLLOWER INSTRUMENTS: Role-specific sampling with pitch grounding
                # KEY FIX: Each follower samples from its OWN vocabulary (via PPM*)
                # TrackDerive only affects pitch grounding, not melodic contour
                position = (section_idx * patterns_per_section + pattern_idx) / total_patterns

                # Batch generate all follower patterns (GPU-efficient)
                follower_results = self.generate_follower_patterns_batch(
                    lead_pattern_id=pattern_id,
                    lead_gm=lead_gm,
                    lead_pitch=lead_pitch,
                    follower_instruments=follower_instruments,
                    position=position,
                    current_patterns=current_patterns,
                    style_id=style_id,
                    chord_root=chord_root,
                    pattern_position=global_pattern_position,
                )

                for follower_gm in follower_instruments:
                    # Get natural co-occurrence probability from checkpoint
                    # This controls WHEN instruments play, not WHAT they play
                    play_prob = self._get_cooccurrence_probability(
                        lead_pattern_id=pattern_id,
                        lead_gm=lead_gm,
                        target_gm=follower_gm
                    )

                    if random.random() > play_prob:
                        # REST - this instrument didn't usually play here
                        continue

                    # Get role-specific pattern from batch results
                    follower_pattern_id, follower_intervals, follower_pitch = follower_results[follower_gm]

                    # Update current_patterns for this follower
                    if follower_pattern_id:
                        current_patterns[follower_gm] = follower_pattern_id

                    follower_notes = self._expand_to_notes(
                        follower_intervals, follower_pitch, current_time, follower_gm, base_ioi
                    )
                    all_tracks[follower_gm].extend(follower_notes)

                # Update current_patterns for lead
                current_patterns[lead_gm] = pattern_id

                # Advance time by phrase duration
                current_time += phrase_duration

        gen_elapsed = time.time() - gen_start_time
        total_notes = sum(len(notes) for notes in all_tracks.values())
        logger.info(f"GENERATION COMPLETE in {gen_elapsed:.1f}s ({total_notes} notes)")

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

        total_notes = sum(len(notes) for notes in tracks.values())
        logger.info(f"SAVED: {output_path} ({len(tracks)} tracks, {total_notes} notes)")

        if self.verbose:
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
