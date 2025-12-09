#!/usr/bin/env python3
"""
Energy-Based Multi-Voice Sampler (v54 Compatible)
==================================================

Joint generation using energy-based scoring instead of sequential sampling.
Uses existing v54 checkpoint - no codec changes required.

Key difference from PPM* sequential:
- PPM*: Sample each instrument independently, hope they fit together
- Energy: Score all instruments jointly, pick best combination

Architecture:
    E(voices) = -PPM_likelihood(each voice)           # Term 1: Within-voice coherence
              + cross_track_penalty(voice pairs)       # Term 2: Between-voice compatibility
              + track_derive_penalty(pitch offsets)    # Term 3: Harmonic relationships
              + chord_penalty(vs current chord)        # Term 4: Harmonic context

Sampling methods:
    1. Exhaustive: Try all combinations (k^n, feasible for small k, n)
    2. Greedy: Pick best for each instrument sequentially
    3. Gibbs: MCMC sampling, iteratively resample each voice
    4. Beam: Keep top-b combinations at each step

Usage:
    from energy_sampler import EnergySampler

    sampler = EnergySampler(generator)  # Wrap existing MetaPatternGenerator
    output = sampler.generate(
        length=32,
        instruments=[0, 32, 56, 57],
        method='gibbs',  # or 'exhaustive', 'greedy', 'beam'
        k=10,  # candidates per instrument
    )
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from itertools import product, combinations
from dataclasses import dataclass, field


@dataclass
class EnergyConfig:
    """Configuration for energy function weights."""
    # Term weights (tune these based on results)
    ppm_weight: float = 1.0           # Within-voice coherence
    cooccurrence_weight: float = 0.3  # Simultaneous pattern compatibility (raw)
    harmonic_cooc_weight: float = 1.5 # Harmonic co-occurrence (pitch-interval aware) - THE FIX
    track_derive_weight: float = 0.3  # Pitch relationship alignment
    chord_weight: float = 0.4         # Harmonic context compatibility
    pitch_class_weight: float = 0.0   # DEPRECATED: Replaced by harmonic_cooc

    # Sampling parameters
    temperature: float = 1.0          # Softmax temperature for sampling
    gibbs_iterations: int = 10        # Number of Gibbs sweeps
    beam_width: int = 5               # Beam search width

    # Candidate generation
    top_k: int = 10                   # Candidates per instrument


class EnergySampler:
    """Energy-based joint sampler for multi-voice generation.

    Wraps an existing MetaPatternGenerator and provides alternative
    generation methods that score all voices jointly.
    """

    def __init__(self, generator, config: EnergyConfig = None):
        """Initialize with existing generator.

        Args:
            generator: MetaPatternGenerator instance with loaded checkpoint
            config: Energy function configuration
        """
        self.gen = generator
        self.config = config or EnergyConfig()

        # Cache frequently accessed data
        self.patterns = generator.patterns
        self.ppm = getattr(generator, 'ppm', None)
        self.cooccurrence = getattr(generator, 'pattern_cooccurrence', {})
        self.pitch_offsets = getattr(generator, 'pitch_offsets_by_gm_pair', {})
        self.chord_probs = getattr(generator, 'chord_pattern_probs', {})

        # Build reverse index: pattern -> typical first_pitch
        self._build_pitch_index()

        # Build HARMONIC co-occurrence: tracks pitch intervals between patterns
        # This is THE FIX - learns from corpus what intervals sound good together
        self._build_harmonic_cooccurrence()

    def _build_pitch_index(self):
        """Build index of typical pitches for each pattern.

        Creates:
            pattern_pitches: {pid: average_pitch} - for backward compat
            pattern_pitch_options: {pid: [pitch1, pitch2, ...]} - actual pitches used
        """
        self.pattern_pitches = {}
        self.pattern_pitch_options = {}  # NEW: actual pitch options per pattern

        for pid, pdata in self.patterns.items():
            occs = pdata.get('occurrences', [])
            if occs:
                pitches = [o.get('first_pitch', 60) for o in occs]
                self.pattern_pitches[pid] = sum(pitches) / len(pitches)

                # Store unique pitches (limit to most common 5)
                from collections import Counter
                pitch_counts = Counter(pitches)
                top_pitches = [p for p, _ in pitch_counts.most_common(5)]
                self.pattern_pitch_options[pid] = top_pitches
            else:
                self.pattern_pitches[pid] = 60
                self.pattern_pitch_options[pid] = [60]

    def _build_harmonic_cooccurrence(self, time_tolerance: int = 480):
        """Build co-occurrence that tracks pitch relationships.

        This is THE KEY FIX: Instead of just counting which patterns play together,
        we also track WHAT PITCH INTERVAL they had. This learns from the corpus
        that "these patterns sound good when piano is a 5th above bass."

        Structure:
            harmonic_cooc[(gm1, gm2, interval)][(p1, p2)] = count
            interval_probs[(gm1, gm2)] = {interval: probability}
        """
        print("  Building harmonic co-occurrence index...")

        # Group occurrences by (piece, time_bucket)
        time_buckets = defaultdict(list)

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            for occ in p.get('occurrences', []):
                piece = occ.get('piece_id', occ.get('piece_idx', 'unknown'))
                onset = occ.get('onset_time', occ.get('onset', 0))
                first_pitch = occ.get('first_pitch')
                if first_pitch is not None:
                    bucket = onset // time_tolerance
                    time_buckets[(piece, bucket)].append((gm, pid, first_pitch))

        # Count co-occurrences WITH pitch interval
        self.harmonic_cooc = defaultdict(lambda: defaultdict(int))
        interval_counts_by_pair = defaultdict(lambda: defaultdict(int))

        for (piece, bucket), items in time_buckets.items():
            for i, (gm1, p1, pitch1) in enumerate(items):
                for gm2, p2, pitch2 in items[i+1:]:
                    if gm1 != gm2:
                        # Key includes pitch interval!
                        interval = (pitch2 - pitch1) % 12
                        gm_key = (min(gm1, gm2), max(gm1, gm2))

                        # Normalize interval direction based on gm order
                        if gm1 < gm2:
                            norm_interval = interval
                            pair = (p1, p2)
                        else:
                            norm_interval = (12 - interval) % 12
                            pair = (p2, p1)

                        key = (gm_key[0], gm_key[1], norm_interval)
                        self.harmonic_cooc[key][pair] += 1
                        interval_counts_by_pair[gm_key][norm_interval] += 1

        # Build interval probability distribution per GM pair
        # This tells us: for piano+bass, how often do they play in unison vs 5ths vs 3rds?
        self.interval_probs = {}
        for gm_pair, iv_counts in interval_counts_by_pair.items():
            total = sum(iv_counts.values())
            if total > 0:
                self.interval_probs[gm_pair] = {
                    iv: count / total for iv, count in iv_counts.items()
                }

        # Convert harmonic_cooc to regular dict
        self.harmonic_cooc = {k: dict(v) for k, v in self.harmonic_cooc.items()}

        print(f"    Harmonic co-occurrence keys: {len(self.harmonic_cooc)}")
        print(f"    GM pairs with interval probs: {len(self.interval_probs)}")

        # Show top interval preferences
        if self.interval_probs:
            print("    Top GM pair interval preferences:")
            sorted_pairs = sorted(self.interval_probs.items(),
                                 key=lambda x: sum(interval_counts_by_pair[x[0]].values()),
                                 reverse=True)[:5]
            for gm_pair, probs in sorted_pairs:
                top_3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
                top_str = ", ".join(f"{iv}={p:.0%}" for iv, p in top_3)
                print(f"      GM{gm_pair[0]}+GM{gm_pair[1]}: {top_str}")

    def get_top_k_patterns(
        self,
        gm: int,
        context: List[str],
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get top-k pattern candidates from PPM* for an instrument.

        Args:
            gm: GM program number
            context: List of previous pattern IDs for this instrument
            k: Number of candidates to return

        Returns:
            List of (pattern_id, log_probability) tuples
        """
        if self.ppm is None:
            # Fallback: uniform over instrument's patterns
            gm_patterns = [pid for pid, p in self.patterns.items()
                          if p.get('gm_program') == gm]
            if not gm_patterns:
                gm_patterns = list(self.patterns.keys())[:k]
            return [(pid, -math.log(len(gm_patterns))) for pid in gm_patterns[:k]]

        # Use PPM* to get probabilities
        probs = self.ppm.get_distribution(gm, tuple(context[-5:]))

        if not probs:
            # Fallback to frequency-based
            gm_patterns = [pid for pid, p in self.patterns.items()
                          if p.get('gm_program') == gm]
            return [(pid, -5.0) for pid in gm_patterns[:k]]

        # Sort by probability, take top k
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])[:k]
        return [(pid, math.log(max(p, 1e-10))) for pid, p in sorted_probs]

    def compute_energy(
        self,
        combo: Dict[int, str],
        contexts: Dict[int, List[str]],
        position: float,
        chord_root: Optional[int] = None,
    ) -> float:
        """Compute energy for a combination of patterns.

        Lower energy = better combination.

        Args:
            combo: Dict of {gm_program: pattern_id}
            contexts: Dict of {gm_program: context_list}
            position: Position in piece [0, 1]
            chord_root: Current chord root (0-11) if known

        Returns:
            Energy value (lower is better)
        """
        E = 0.0

        # Term 1: PPM* likelihood (within-voice coherence)
        if self.config.ppm_weight > 0 and self.ppm is not None:
            for gm, pattern_id in combo.items():
                context = contexts.get(gm, [])
                log_prob = self._ppm_log_prob(gm, pattern_id, context)
                E -= self.config.ppm_weight * log_prob

        # Term 2: Co-occurrence (simultaneous pattern compatibility)
        if self.config.cooccurrence_weight > 0:
            for (gm1, p1), (gm2, p2) in combinations(combo.items(), 2):
                log_prob = self._cooccurrence_log_prob(gm1, gm2, p1, p2)
                E -= self.config.cooccurrence_weight * log_prob

        # Term 3: TrackDerive pitch alignment
        if self.config.track_derive_weight > 0:
            penalty = self._track_derive_penalty(combo)
            E += self.config.track_derive_weight * penalty

        # Term 4: Chord compatibility
        if self.config.chord_weight > 0 and chord_root is not None:
            for gm, pattern_id in combo.items():
                log_prob = self._chord_log_prob(chord_root, gm, pattern_id)
                E -= self.config.chord_weight * log_prob

        # Term 5: Pitch class compatibility (DEPRECATED - replaced by harmonic_cooc)
        if self.config.pitch_class_weight > 0:
            penalty = self._pitch_class_compatibility(combo)
            E += self.config.pitch_class_weight * penalty

        # Term 6: HARMONIC CO-OCCURRENCE - THE KEY FIX
        # This learns from corpus what pitch intervals work well together
        if self.config.harmonic_cooc_weight > 0:
            log_prob = self._harmonic_cooccurrence_log_prob(combo)
            E -= self.config.harmonic_cooc_weight * log_prob

        return E

    def compute_energy_with_pitches(
        self,
        combo: Dict[int, str],
        pitches: Dict[int, int],
        contexts: Dict[int, List[str]],
        position: float,
        chord_root: Optional[int] = None,
    ) -> float:
        """Compute energy for a combination of patterns WITH EXPLICIT PITCHES.

        This is the PROPER fix: score the actual (pattern, pitch) pairs that
        will be used during generation, not averaged pitches.

        Args:
            combo: Dict of {gm_program: pattern_id}
            pitches: Dict of {gm_program: actual_pitch} - THE KEY DIFFERENCE
            contexts: Dict of {gm_program: context_list}
            position: Position in piece [0, 1]
            chord_root: Current chord root (0-11) if known

        Returns:
            Energy value (lower is better)
        """
        E = 0.0

        # Term 1: PPM* likelihood (within-voice coherence)
        if self.config.ppm_weight > 0 and self.ppm is not None:
            for gm, pattern_id in combo.items():
                context = contexts.get(gm, [])
                log_prob = self._ppm_log_prob(gm, pattern_id, context)
                E -= self.config.ppm_weight * log_prob

        # Term 2: Raw co-occurrence (pattern pairs regardless of pitch)
        if self.config.cooccurrence_weight > 0:
            for (gm1, p1), (gm2, p2) in combinations(combo.items(), 2):
                log_prob = self._cooccurrence_log_prob(gm1, gm2, p1, p2)
                E -= self.config.cooccurrence_weight * log_prob

        # Term 3: TrackDerive pitch alignment
        if self.config.track_derive_weight > 0:
            penalty = self._track_derive_penalty_with_pitches(combo, pitches)
            E += self.config.track_derive_weight * penalty

        # Term 4: Chord compatibility
        if self.config.chord_weight > 0 and chord_root is not None:
            for gm, pattern_id in combo.items():
                log_prob = self._chord_log_prob(chord_root, gm, pattern_id)
                E -= self.config.chord_weight * log_prob

        # Term 6: HARMONIC CO-OCCURRENCE with explicit pitches
        if self.config.harmonic_cooc_weight > 0:
            log_prob = self._harmonic_cooccurrence_log_prob_with_pitches(combo, pitches)
            E -= self.config.harmonic_cooc_weight * log_prob

        return E

    def _track_derive_penalty_with_pitches(
        self,
        combo: Dict[int, str],
        pitches: Dict[int, int]
    ) -> float:
        """TrackDerive penalty using explicit pitches."""
        penalty = 0.0

        for (gm1, p1), (gm2, p2) in combinations(combo.items(), 2):
            expected_offset = self.pitch_offsets.get((gm1, gm2), None)
            if expected_offset is None:
                expected_offset = self.pitch_offsets.get((gm2, gm1), None)
                if expected_offset is not None:
                    expected_offset = -expected_offset

            if expected_offset is not None:
                pitch1 = pitches[gm1]  # Use explicit pitch
                pitch2 = pitches[gm2]  # Use explicit pitch
                actual_offset = pitch2 - pitch1

                diff = (actual_offset - expected_offset) % 12
                if diff > 6:
                    diff = 12 - diff
                penalty += diff ** 2

        return penalty

    def _harmonic_cooccurrence_log_prob_with_pitches(
        self,
        combo: Dict[int, str],
        pitches: Dict[int, int]
    ) -> float:
        """Harmonic co-occurrence scoring with EXPLICIT pitches.

        This is the PROPER implementation: we use the actual pitches that
        will be generated, not averaged pitches from occurrences.
        """
        if not self.harmonic_cooc:
            return 0.0

        log_prob = 0.0
        n_pairs = 0

        for (gm1, p1), (gm2, p2) in combinations(combo.items(), 2):
            # Use EXPLICIT pitches, not averages
            pitch1 = pitches[gm1]
            pitch2 = pitches[gm2]

            # Normalize GM pair order
            gm_key = (min(gm1, gm2), max(gm1, gm2))
            if gm1 < gm2:
                interval = (pitch2 - pitch1) % 12
                pair = (p1, p2)
            else:
                interval = (pitch1 - pitch2) % 12
                pair = (p2, p1)

            # Score 1: Is this interval common for this instrument pair?
            if gm_key in self.interval_probs:
                interval_prob = self.interval_probs[gm_key].get(interval, 0.001)
                log_prob += 2.0 * math.log(max(interval_prob, 1e-10))
            else:
                log_prob -= 3.0

            # Score 2: Pattern pair at this interval
            key = (gm_key[0], gm_key[1], interval)
            if key in self.harmonic_cooc:
                count = self.harmonic_cooc[key].get(pair, 0)
                total = sum(self.harmonic_cooc[key].values())
                if total > 0 and count > 0:
                    pair_prob = count / total
                    log_prob += 0.3 * math.log(max(pair_prob, 1e-10))

            n_pairs += 1

        if n_pairs > 0:
            log_prob /= n_pairs

        return log_prob

    def _ppm_log_prob(self, gm: int, pattern_id: str, context: List[str]) -> float:
        """Get PPM* log probability for a pattern given context."""
        if self.ppm is None:
            return -5.0  # Default log prob

        probs = self.ppm.get_distribution(gm, tuple(context[-5:]))
        prob = probs.get(pattern_id, 1e-10)
        return math.log(max(prob, 1e-10))

    def _cooccurrence_log_prob(self, gm1: int, gm2: int, p1: str, p2: str) -> float:
        """Get co-occurrence log probability for a pattern pair."""
        key = (min(gm1, gm2), max(gm1, gm2))
        pair_key = (p1, p2) if gm1 < gm2 else (p2, p1)

        if key in self.cooccurrence:
            count = self.cooccurrence[key].get(pair_key, 0)
            total = sum(self.cooccurrence[key].values())
            if total > 0:
                return math.log(max(count / total, 1e-10))

        return -10.0  # Default: low probability for unseen pairs

    def _track_derive_penalty(self, combo: Dict[int, str]) -> float:
        """Compute penalty for violating TrackDerive pitch relationships."""
        penalty = 0.0

        for (gm1, p1), (gm2, p2) in combinations(combo.items(), 2):
            expected_offset = self.pitch_offsets.get((gm1, gm2), None)
            if expected_offset is None:
                expected_offset = self.pitch_offsets.get((gm2, gm1), None)
                if expected_offset is not None:
                    expected_offset = -expected_offset

            if expected_offset is not None:
                pitch1 = self.pattern_pitches.get(p1, 60)
                pitch2 = self.pattern_pitches.get(p2, 60)
                actual_offset = pitch2 - pitch1

                # Penalty is squared difference (allow octave equivalence)
                diff = (actual_offset - expected_offset) % 12
                if diff > 6:
                    diff = 12 - diff
                penalty += diff ** 2

        return penalty

    def _chord_log_prob(self, chord_root: int, gm: int, pattern_id: str) -> float:
        """Get log probability of pattern given chord context."""
        key = (chord_root, gm)
        if key in self.chord_probs:
            prob = self.chord_probs[key].get(pattern_id, 1e-10)
            return math.log(max(prob, 1e-10))
        return -5.0  # Default

    def _pitch_class_compatibility(self, combo: Dict[int, str]) -> float:
        """Penalize dissonant pitch class combinations between voices.

        Lower = more consonant, Higher = more dissonant.
        """
        all_pcs = []

        for gm, pid in combo.items():
            p = self.patterns.get(pid, {})
            occs = p.get('occurrences', [])
            fp = occs[0].get('first_pitch', 60) if occs else 60
            intervals = p.get('pitch_intervals', [])

            # Build all pitches in pattern
            pitches = [fp]
            for iv in intervals:
                pitches.append(pitches[-1] + iv)

            # Convert to pitch classes
            all_pcs.append(set(x % 12 for x in pitches))

        if len(all_pcs) < 2:
            return 0.0

        # Penalize dissonant intervals between voices
        penalty = 0.0
        # m2=1, M2=2, tritone=6, m7=10, M7=11 are dissonant
        dissonant = {1, 2, 6, 10, 11}

        for i, pcs1 in enumerate(all_pcs):
            for pcs2 in all_pcs[i+1:]:
                for a in pcs1:
                    for b in pcs2:
                        interval = min(abs(a - b), 12 - abs(a - b))
                        if interval in dissonant:
                            penalty += 2.0

        return penalty

    def _harmonic_cooccurrence_log_prob(self, combo: Dict[int, str]) -> float:
        """Score based on corpus harmonic relationships.

        THIS IS THE KEY FIX: Instead of hardcoded dissonance rules, we use
        learned statistics about what pitch intervals actually occur between
        instruments in the corpus.

        For each pair of instruments:
        1. Get the actual pitch interval between the patterns
        2. Look up how often this interval occurs for this instrument pair
        3. Look up how often these specific patterns play at this interval

        This gives us both:
        - Interval compatibility (is this interval common for piano+bass?)
        - Pattern pair compatibility at this interval (do these patterns sound good together?)
        """
        if not self.harmonic_cooc:
            return 0.0

        log_prob = 0.0
        n_pairs = 0

        for (gm1, p1), (gm2, p2) in combinations(combo.items(), 2):
            # Get actual pitches from patterns
            pitch1 = self.pattern_pitches.get(p1, 60)
            pitch2 = self.pattern_pitches.get(p2, 60)

            # Normalize GM pair order
            gm_key = (min(gm1, gm2), max(gm1, gm2))
            if gm1 < gm2:
                interval = int(pitch2 - pitch1) % 12
                pair = (p1, p2)
            else:
                interval = int(pitch1 - pitch2) % 12
                pair = (p2, p1)

            # Score 1: Is this interval common for this instrument pair?
            # This is the PRIMARY signal - what intervals work for this instrument combo
            if gm_key in self.interval_probs:
                interval_prob = self.interval_probs[gm_key].get(interval, 0.001)
                # Strong bonus for common intervals (especially unison)
                log_prob += 2.0 * math.log(max(interval_prob, 1e-10))
            else:
                log_prob -= 3.0  # Unseen instrument pair

            # Score 2: Have these patterns been seen together at this interval?
            # This is SECONDARY - specific pattern pairs matter less than interval choice
            key = (gm_key[0], gm_key[1], interval)
            if key in self.harmonic_cooc:
                count = self.harmonic_cooc[key].get(pair, 0)
                total = sum(self.harmonic_cooc[key].values())
                if total > 0 and count > 0:
                    pair_prob = count / total
                    log_prob += 0.3 * math.log(max(pair_prob, 1e-10))
                # Don't penalize unseen pairs too much - interval is more important
            # No else penalty - interval score handles unseen combos

            n_pairs += 1

        # Normalize by number of pairs
        if n_pairs > 0:
            log_prob /= n_pairs

        return log_prob

    # =========================================================================
    # SAMPLING METHODS
    # =========================================================================

    def generate_exhaustive(
        self,
        length: int,
        instruments: List[int],
        k: int = 5,
        chord_skeleton: List[int] = None,
    ) -> Dict[int, List[str]]:
        """Generate by exhaustively scoring all k^n combinations.

        Only feasible for small k and few instruments.
        Complexity: O(length * k^n) where n = len(instruments)

        Args:
            length: Number of patterns to generate per instrument
            instruments: List of GM program numbers
            k: Candidates per instrument
            chord_skeleton: Optional chord progression

        Returns:
            Dict of {gm: [pattern_id, ...]}
        """
        output = {gm: [] for gm in instruments}
        contexts = {gm: [] for gm in instruments}

        for t in range(length):
            position = t / max(1, length - 1)
            chord_root = chord_skeleton[t % len(chord_skeleton)] if chord_skeleton else None

            # Get candidates for each instrument
            candidates = {}
            for gm in instruments:
                candidates[gm] = [pid for pid, _ in
                                 self.get_top_k_patterns(gm, contexts[gm], k)]

            # Score all combinations
            best_combo = None
            best_energy = float('inf')

            for combo_tuple in product(*[candidates[gm] for gm in instruments]):
                combo = dict(zip(instruments, combo_tuple))
                energy = self.compute_energy(combo, contexts, position, chord_root)

                if energy < best_energy:
                    best_energy = energy
                    best_combo = combo

            # Update output and contexts
            for gm, pattern_id in best_combo.items():
                output[gm].append(pattern_id)
                contexts[gm].append(pattern_id)

        return output

    def generate_greedy(
        self,
        length: int,
        instruments: List[int],
        k: int = 10,
        chord_skeleton: List[int] = None,
    ) -> Dict[int, List[str]]:
        """Generate greedily, picking best pattern for each instrument in order.

        Faster than exhaustive but may miss globally optimal combinations.
        Complexity: O(length * n * k)

        Args:
            length: Number of patterns to generate per instrument
            instruments: List of GM program numbers
            k: Candidates per instrument
            chord_skeleton: Optional chord progression

        Returns:
            Dict of {gm: [pattern_id, ...]}
        """
        output = {gm: [] for gm in instruments}
        contexts = {gm: [] for gm in instruments}

        for t in range(length):
            position = t / max(1, length - 1)
            chord_root = chord_skeleton[t % len(chord_skeleton)] if chord_skeleton else None

            # Build combo incrementally
            combo = {}

            for gm in instruments:
                candidates = self.get_top_k_patterns(gm, contexts[gm], k)

                # Pick best given already-chosen instruments
                best_pattern = None
                best_energy = float('inf')

                for pattern_id, _ in candidates:
                    test_combo = {**combo, gm: pattern_id}
                    energy = self.compute_energy(test_combo, contexts, position, chord_root)

                    if energy < best_energy:
                        best_energy = energy
                        best_pattern = pattern_id

                combo[gm] = best_pattern

            # Update output and contexts
            for gm, pattern_id in combo.items():
                output[gm].append(pattern_id)
                contexts[gm].append(pattern_id)

        return output

    def generate_gibbs(
        self,
        length: int,
        instruments: List[int],
        k: int = 10,
        n_iterations: int = None,
        chord_skeleton: List[int] = None,
    ) -> Dict[int, List[str]]:
        """Generate using Gibbs sampling (MCMC).

        Iteratively resample each voice given all others fixed.
        Better exploration than greedy, scales better than exhaustive.

        Args:
            length: Number of patterns to generate per instrument
            instruments: List of GM program numbers
            k: Candidates per instrument
            n_iterations: Number of Gibbs sweeps (default from config)
            chord_skeleton: Optional chord progression

        Returns:
            Dict of {gm: [pattern_id, ...]}
        """
        n_iterations = n_iterations or self.config.gibbs_iterations

        output = {gm: [] for gm in instruments}
        contexts = {gm: [] for gm in instruments}

        for t in range(length):
            position = t / max(1, length - 1)
            chord_root = chord_skeleton[t % len(chord_skeleton)] if chord_skeleton else None

            # Get candidates for each instrument
            all_candidates = {}
            for gm in instruments:
                all_candidates[gm] = [pid for pid, _ in
                                     self.get_top_k_patterns(gm, contexts[gm], k)]

            # Initialize with greedy
            combo = {}
            for gm in instruments:
                if all_candidates[gm]:
                    combo[gm] = all_candidates[gm][0]
                else:
                    combo[gm] = list(self.patterns.keys())[0]

            # Gibbs iterations
            for iteration in range(n_iterations):
                # Sweep through instruments in random order
                gm_order = list(instruments)
                random.shuffle(gm_order)

                for gm in gm_order:
                    candidates = all_candidates[gm]
                    if not candidates:
                        continue

                    # Compute energy for each candidate
                    energies = []
                    for pattern_id in candidates:
                        test_combo = {**combo, gm: pattern_id}
                        energy = self.compute_energy(test_combo, contexts, position, chord_root)
                        energies.append(energy)

                    # Sample proportional to exp(-energy/temperature)
                    min_energy = min(energies)
                    weights = [math.exp(-(e - min_energy) / self.config.temperature)
                              for e in energies]
                    total = sum(weights)
                    weights = [w / total for w in weights]

                    # Weighted random choice
                    r = random.random()
                    cumsum = 0
                    for i, w in enumerate(weights):
                        cumsum += w
                        if r <= cumsum:
                            combo[gm] = candidates[i]
                            break

            # Update output and contexts with final combo
            for gm, pattern_id in combo.items():
                output[gm].append(pattern_id)
                contexts[gm].append(pattern_id)

        return output

    def generate_beam(
        self,
        length: int,
        instruments: List[int],
        k: int = 10,
        beam_width: int = None,
        chord_skeleton: List[int] = None,
    ) -> Dict[int, List[str]]:
        """Generate using beam search.

        Keep top-b combinations at each timestep.
        Good balance between exploration and efficiency.

        Args:
            length: Number of patterns to generate per instrument
            instruments: List of GM program numbers
            k: Candidates per instrument
            beam_width: Number of hypotheses to maintain
            chord_skeleton: Optional chord progression

        Returns:
            Dict of {gm: [pattern_id, ...]}
        """
        beam_width = beam_width or self.config.beam_width

        # Each hypothesis: (total_energy, {gm: [patterns]}, {gm: context})
        beam = [(0.0, {gm: [] for gm in instruments}, {gm: [] for gm in instruments})]

        for t in range(length):
            position = t / max(1, length - 1)
            chord_root = chord_skeleton[t % len(chord_skeleton)] if chord_skeleton else None

            new_beam = []

            for total_energy, output, contexts in beam:
                # Get candidates for each instrument
                candidates = {}
                for gm in instruments:
                    candidates[gm] = [pid for pid, _ in
                                     self.get_top_k_patterns(gm, contexts[gm], k)]

                # Try all combinations (or sample if too many)
                n_combos = 1
                for gm in instruments:
                    n_combos *= len(candidates[gm]) if candidates[gm] else 1

                if n_combos <= 100:  # Exhaustive for small search space
                    combos_to_try = list(product(*[candidates[gm] or [''] for gm in instruments]))
                else:  # Sample for large search space
                    combos_to_try = []
                    for _ in range(100):
                        combo = tuple(random.choice(candidates[gm]) if candidates[gm] else ''
                                     for gm in instruments)
                        combos_to_try.append(combo)

                for combo_tuple in combos_to_try:
                    combo = dict(zip(instruments, combo_tuple))
                    step_energy = self.compute_energy(combo, contexts, position, chord_root)

                    new_output = {gm: output[gm] + [pid] for gm, pid in combo.items()}
                    new_contexts = {gm: contexts[gm] + [pid] for gm, pid in combo.items()}

                    new_beam.append((total_energy + step_energy, new_output, new_contexts))

            # Keep top beam_width hypotheses
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:beam_width]

        # Return best hypothesis
        if beam:
            return beam[0][1]
        else:
            return {gm: [] for gm in instruments}

    # =========================================================================
    # PITCH-CONTROLLED GENERATION (THE PROPER FIX)
    # =========================================================================

    def generate_with_pitch_control(
        self,
        length: int,
        instruments: List[int],
        k: int = 10,
        n_pitch_samples: int = 3,
        n_combo_samples: int = 50,
        chord_skeleton: List[int] = None,
    ) -> Dict[int, List[Tuple[str, int]]]:
        """Generate with explicit pitch control.

        This is the PROPER FIX: we sample (pattern, pitch) pairs together,
        score them with the same pitches we'll use for generation, and
        return both pattern IDs and target pitches.

        Args:
            length: Number of patterns to generate per instrument
            instruments: List of GM program numbers
            k: Pattern candidates per instrument
            n_pitch_samples: Pitch candidates per pattern
            n_combo_samples: Random combos to evaluate per timestep
            chord_skeleton: Optional chord progression [0-11]

        Returns:
            Dict of {gm: [(pattern_id, target_pitch), ...]}
        """
        output = {gm: [] for gm in instruments}
        contexts = {gm: [] for gm in instruments}

        for t in range(length):
            position = t / max(1, length - 1)
            chord_root = chord_skeleton[t % len(chord_skeleton)] if chord_skeleton else None

            # Get pattern candidates for each instrument
            pattern_candidates = {}
            for gm in instruments:
                pattern_candidates[gm] = [pid for pid, _ in
                                         self.get_top_k_patterns(gm, [p for p, _ in output[gm]], k)]

            # Build (pattern, pitch) candidates for each instrument
            pattern_pitch_candidates = {}
            for gm in instruments:
                candidates = []
                for pid in pattern_candidates[gm]:
                    # Get actual pitches this pattern uses
                    pitch_options = self.pattern_pitch_options.get(pid, [60])[:n_pitch_samples]
                    for pitch in pitch_options:
                        candidates.append((pid, pitch))
                pattern_pitch_candidates[gm] = candidates if candidates else [('', 60)]

            # Sample and score combinations
            best_combo = None
            best_pitches = None
            best_energy = float('inf')

            for _ in range(n_combo_samples):
                combo = {}
                pitches = {}
                for gm in instruments:
                    pid, pitch = random.choice(pattern_pitch_candidates[gm])
                    combo[gm] = pid
                    pitches[gm] = pitch

                # Score with EXPLICIT pitches - the key difference
                energy = self.compute_energy_with_pitches(
                    combo, pitches, contexts, position, chord_root
                )

                if energy < best_energy:
                    best_energy = energy
                    best_combo = combo
                    best_pitches = pitches

            # Record winning (pattern, pitch) pairs
            for gm in instruments:
                output[gm].append((best_combo[gm], best_pitches[gm]))
                contexts[gm].append(best_combo[gm])

        return output

    def generate_to_notes_with_pitch_control(
        self,
        length: int,
        instruments: List[int],
        k: int = 10,
        **kwargs
    ) -> Dict[int, List[dict]]:
        """Generate with pitch control and expand to note events.

        This uses the controlled pitches for generation instead of
        random occurrence pitches.
        """
        # Generate (pattern, pitch) pairs
        pattern_output = self.generate_with_pitch_control(length, instruments, k, **kwargs)

        note_output = {}
        for gm, pattern_pitches in pattern_output.items():
            notes = []
            current_time = 0

            for pid, target_pitch in pattern_pitches:
                pattern = self.patterns.get(pid, {})
                intervals = pattern.get('pitch_intervals', [0])
                rhythm_ratios = pattern.get('rhythm_ratios', [])
                velocity_ratios = pattern.get('velocity_ratios', [])
                duration_ratios = pattern.get('duration_ratios', [])

                # Get timing from first occurrence
                occs = pattern.get('occurrences', [])
                first_occ = occs[0] if occs else {}
                base_ioi = first_occ.get('tau_offset', 480)

                # Use TARGET PITCH instead of occurrence pitch
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

        return note_output

    # =========================================================================
    # MAIN INTERFACE
    # =========================================================================

    def generate(
        self,
        length: int,
        instruments: List[int],
        method: str = 'gibbs',
        k: int = None,
        chord_skeleton: List[int] = None,
        **kwargs
    ) -> Dict[int, List[str]]:
        """Generate multi-voice output using specified method.

        Args:
            length: Number of patterns to generate per instrument
            instruments: List of GM program numbers
            method: 'exhaustive', 'greedy', 'gibbs', 'beam', or 'pitch_control'
            k: Candidates per instrument (default from config)
            chord_skeleton: Optional chord progression [0-11]
            **kwargs: Method-specific arguments

        Returns:
            Dict of {gm: [pattern_id, ...]} or {gm: [(pattern_id, pitch), ...]} for pitch_control
        """
        k = k or self.config.top_k

        if method == 'pitch_control':
            return self.generate_with_pitch_control(length, instruments, k, chord_skeleton=chord_skeleton, **kwargs)
        elif method == 'exhaustive':
            return self.generate_exhaustive(length, instruments, k, chord_skeleton)
        elif method == 'greedy':
            return self.generate_greedy(length, instruments, k, chord_skeleton)
        elif method == 'gibbs':
            return self.generate_gibbs(length, instruments, k,
                                       kwargs.get('n_iterations'), chord_skeleton)
        elif method == 'beam':
            return self.generate_beam(length, instruments, k,
                                      kwargs.get('beam_width'), chord_skeleton)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'exhaustive', 'greedy', 'gibbs', or 'beam'")

    def generate_to_notes(
        self,
        length: int,
        instruments: List[int],
        method: str = 'gibbs',
        **kwargs
    ) -> Dict[int, List[dict]]:
        """Generate and expand to note events.

        Returns note events suitable for MIDI conversion.
        """
        pattern_output = self.generate(length, instruments, method, **kwargs)

        note_output = {}
        for gm, pattern_ids in pattern_output.items():
            notes = []
            current_time = 0

            for pid in pattern_ids:
                pattern = self.patterns.get(pid, {})
                intervals = pattern.get('pitch_intervals', [0])
                rhythm_ratios = pattern.get('rhythm_ratios', [])
                velocity_ratios = pattern.get('velocity_ratios', [])
                duration_ratios = pattern.get('duration_ratios', [])

                # Get base values from first occurrence
                occs = pattern.get('occurrences', [])
                first_occ = occs[0] if occs else {}
                first_pitch = first_occ.get('first_pitch', 60)
                base_ioi = first_occ.get('tau_offset', 480)

                # Expand pattern to notes
                pitch = first_pitch
                for i, interval in enumerate([0] + intervals):
                    pitch += interval if i > 0 else 0

                    # Get timing from ratios or defaults
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

        return note_output


# =============================================================================
# CLI for testing
# =============================================================================

def main():
    import argparse
    import sys
    import os

    # Add parent to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.meta_pattern_generator import MetaPatternGenerator

    parser = argparse.ArgumentParser(description='Energy-based multi-voice sampler')
    parser.add_argument('checkpoint', help='Path to v54 checkpoint')
    parser.add_argument('-o', '--output', default='/tmp/energy_test.mid',
                       help='Output MIDI file')
    parser.add_argument('--method', choices=['exhaustive', 'greedy', 'gibbs', 'beam', 'pitch_control'],
                       default='pitch_control', help='Sampling method (pitch_control recommended)')
    parser.add_argument('--length', type=int, default=32,
                       help='Number of patterns to generate')
    parser.add_argument('-k', type=int, default=10,
                       help='Candidates per instrument')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--instruments', type=int, nargs='+',
                       default=[0, 32, 56, 57],
                       help='GM program numbers to generate')
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    gen = MetaPatternGenerator(args.checkpoint, verbose=True)

    print(f"\nInitializing energy sampler...")
    config = EnergyConfig(
        temperature=args.temperature,
        top_k=args.k,
        harmonic_cooc_weight=2.5,  # Strong harmonic guidance
        cooccurrence_weight=0.1,
        pitch_class_weight=0.0,    # Disabled - use harmonic_cooc instead
        ppm_weight=0.5,
    )
    sampler = EnergySampler(gen, config)

    print(f"\nGenerating with method={args.method}, length={args.length}, k={args.k}")
    print(f"Instruments: {args.instruments}")

    # Use pitch_control method for harmonically coherent output
    if args.method == 'pitch_control':
        output = sampler.generate_with_pitch_control(
            length=args.length,
            instruments=args.instruments,
            k=args.k,
            n_pitch_samples=3,
            n_combo_samples=100,
        )
        print(f"\nGenerated (pattern, pitch) pairs:")
        for gm, pairs in output.items():
            unique_patterns = len(set(p for p, _ in pairs))
            unique_pitches = len(set(pitch for _, pitch in pairs))
            print(f"  GM {gm}: {unique_patterns} unique patterns, {unique_pitches} unique pitches")
    else:
        output = sampler.generate(
            length=args.length,
            instruments=args.instruments,
            method=args.method,
            k=args.k,
        )
        print(f"\nGenerated patterns:")
        for gm, patterns in output.items():
            print(f"  GM {gm}: {len(patterns)} patterns")

    # Convert to notes and save MIDI
    print(f"\nExpanding to notes...")
    if args.method == 'pitch_control':
        notes = sampler.generate_to_notes_with_pitch_control(
            length=args.length,
            instruments=args.instruments,
            k=args.k,
            n_pitch_samples=3,
            n_combo_samples=100,
        )
    else:
        notes = sampler.generate_to_notes(
            length=args.length,
            instruments=args.instruments,
            method=args.method,
            k=args.k,
        )

    total_notes = sum(len(n) for n in notes.values())
    print(f"  Total notes: {total_notes}")

    # Save to MIDI
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage

    mid = MidiFile(ticks_per_beat=480, type=1)

    # Tempo track
    tempo_track = MidiTrack()
    mid.tracks.append(tempo_track)
    tempo_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
    tempo_track.append(MetaMessage('end_of_track', time=0))

    # GM program to MIDI channel mapping
    channel_map = {}
    next_channel = 0

    for gm, note_list in sorted(notes.items()):
        if not note_list:
            continue

        # Assign channel
        if gm not in channel_map:
            channel_map[gm] = next_channel
            next_channel += 1
            if next_channel == 9:  # Skip drums channel
                next_channel = 10

        channel = channel_map[gm]
        track = MidiTrack()
        mid.tracks.append(track)

        # Program change
        track.append(Message('program_change', program=gm, channel=channel, time=0))

        # Sort notes by onset
        sorted_notes = sorted(note_list, key=lambda n: n['onset'])

        # Build events
        events = []
        for n in sorted_notes:
            events.append((n['onset'], 'on', n['pitch'], n['velocity']))
            events.append((n['onset'] + n['duration'], 'off', n['pitch'], 0))

        events.sort(key=lambda x: (x[0], x[1] == 'on'))  # offs before ons at same time

        # Convert to delta times
        last_time = 0
        for event_time, event_type, pitch, vel in events:
            delta = event_time - last_time
            if event_type == 'on':
                track.append(Message('note_on', note=pitch, velocity=vel, channel=channel, time=delta))
            else:
                track.append(Message('note_off', note=pitch, velocity=0, channel=channel, time=delta))
            last_time = event_time

        track.append(MetaMessage('end_of_track', time=0))

    mid.save(args.output)
    print(f"\nSaved to: {args.output}")
    print(f"  Tracks: {len(mid.tracks)}")
    print(f"  Notes: {total_notes}")


if __name__ == '__main__':
    main()
