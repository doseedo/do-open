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
    cooccurrence_weight: float = 0.5  # Simultaneous pattern compatibility
    track_derive_weight: float = 0.3  # Pitch relationship alignment
    chord_weight: float = 0.4         # Harmonic context compatibility

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

    def _build_pitch_index(self):
        """Build index of typical pitches for each pattern."""
        self.pattern_pitches = {}
        for pid, pdata in self.patterns.items():
            occs = pdata.get('occurrences', [])
            if occs:
                pitches = [o.get('first_pitch', 60) for o in occs]
                self.pattern_pitches[pid] = sum(pitches) / len(pitches)
            else:
                self.pattern_pitches[pid] = 60

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

        return E

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
            method: 'exhaustive', 'greedy', 'gibbs', or 'beam'
            k: Candidates per instrument (default from config)
            chord_skeleton: Optional chord progression [0-11]
            **kwargs: Method-specific arguments

        Returns:
            Dict of {gm: [pattern_id, ...]}
        """
        k = k or self.config.top_k

        if method == 'exhaustive':
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
                        velocity = int(64 * velocity_ratios[i])
                    else:
                        velocity = 64

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
    parser.add_argument('--method', choices=['exhaustive', 'greedy', 'gibbs', 'beam'],
                       default='gibbs', help='Sampling method')
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
    )
    sampler = EnergySampler(gen, config)

    print(f"\nGenerating with method={args.method}, length={args.length}, k={args.k}")
    print(f"Instruments: {args.instruments}")

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
    notes = sampler.generate_to_notes(
        length=args.length,
        instruments=args.instruments,
        method=args.method,
        k=args.k,
    )

    total_notes = sum(len(n) for n in notes.values())
    print(f"  Total notes: {total_notes}")

    # Save to MIDI (using generator's method if available)
    if hasattr(gen, '_save_midi'):
        gen._save_midi(notes, args.output)
        print(f"\nSaved to: {args.output}")
    else:
        print(f"\nNote: MIDI saving not implemented. Pattern output complete.")
        print(f"Use meta_pattern_generator.py for full MIDI output.")


if __name__ == '__main__':
    main()
