#!/usr/bin/env python3
"""
Coordination Grammar - Extract vertical coordination from existing checkpoint.

The key insight: Pattern occurrences already have (piece_id, onset_time, first_pitch).
We can reconstruct what played together and build a coordination grammar POST-HOC.

No pipeline rerun needed - this works on existing checkpoint data.
"""

import json
import math
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass


@dataclass
class CoordinationEvent:
    """A vertical slice of what's playing at a given beat."""
    beat: int
    instruments: Tuple[Tuple[int, str, int], ...]  # ((gm, pattern_id, pitch_class), ...)

    def __hash__(self):
        return hash(self.instruments)

    def __eq__(self, other):
        return self.instruments == other.instruments


def build_coordination_from_checkpoint(patterns: dict, beat_ticks: int = 480):
    """Extract vertical coordination from existing occurrence data.

    Args:
        patterns: Pattern dict from checkpoint
        beat_ticks: Ticks per beat for quantization

    Returns:
        coordination_sequences: {piece_id: [(beat, event), ...]}
        event_counts: Counter of unique coordination events
    """
    print("=" * 60)
    print("EXTRACTING COORDINATION FROM CHECKPOINT")
    print("=" * 60)

    # Group occurrences by (piece, beat)
    beat_events = defaultdict(list)  # (piece, beat) -> [(gm, pattern_id, pitch_class)]

    total_occs = 0
    for pid, p in patterns.items():
        gm = p.get('gm_program', 0)
        for occ in p.get('occurrences', []):
            piece = occ.get('piece_id', occ.get('piece_idx', 'unknown'))
            onset = occ.get('onset_time', occ.get('onset', 0))
            pitch = occ.get('first_pitch', 60)

            beat = onset // beat_ticks
            beat_events[(piece, beat)].append((gm, pid, pitch % 12))
            total_occs += 1

    print(f"Total occurrences: {total_occs}")
    print(f"Unique (piece, beat) locations: {len(beat_events)}")

    # Extract coordination events (2+ instruments at same beat)
    coordination_sequences = defaultdict(list)  # piece -> [(beat, event), ...]
    single_instrument_beats = 0
    multi_instrument_beats = 0

    for (piece, beat), instruments in sorted(beat_events.items()):
        unique_gms = set(gm for gm, _, _ in instruments)

        if len(unique_gms) >= 2:  # 2+ different instruments
            # Create hashable event - sort for consistency
            event = tuple(sorted(instruments))
            coordination_sequences[piece].append((beat, event))
            multi_instrument_beats += 1
        else:
            single_instrument_beats += 1

    print(f"\nSingle-instrument beats: {single_instrument_beats}")
    print(f"Multi-instrument beats: {multi_instrument_beats}")
    print(f"Pieces with coordination: {len(coordination_sequences)}")

    # Count unique events
    all_events = []
    for piece, events in coordination_sequences.items():
        all_events.extend([e for _, e in events])

    event_counts = Counter(all_events)
    unique_events = len(event_counts)
    total_events = len(all_events)

    print(f"\nTotal coordination events: {total_events}")
    print(f"Unique coordination events: {unique_events}")

    if total_events > 0:
        ratio = unique_events / total_events
        print(f"Uniqueness ratio: {ratio:.1%}")

        if ratio < 0.30:
            print("✓ LOW RATIO - Coordination is highly compressible!")
        elif ratio < 0.50:
            print("~ MODERATE RATIO - Some coordination patterns")
        else:
            print("⚠️ HIGH RATIO - Coordination may not be learnable")

    return coordination_sequences, event_counts


def analyze_coordination_patterns(event_counts: Counter, patterns: dict, top_n: int = 20):
    """Analyze the most common coordination patterns."""
    print("\n" + "=" * 60)
    print("TOP COORDINATION PATTERNS")
    print("=" * 60)

    gm_names = {
        0: "Piano", 32: "A.Bass", 33: "E.Bass",
        56: "Trumpet", 57: "Trombone", 58: "Tuba",
        64: "Sop.Sax", 65: "Alto.Sax", 66: "Ten.Sax", 67: "Bari.Sax",
        40: "Violin", 42: "Cello", 128: "Drums",
    }
    pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    print(f"\nTop {top_n} most common vertical combinations:\n")

    for event, count in event_counts.most_common(top_n):
        # Group by instrument
        by_gm = defaultdict(list)
        for gm, pid, pc in event:
            by_gm[gm].append((pid, pc))

        # Format output
        parts = []
        for gm in sorted(by_gm.keys()):
            name = gm_names.get(gm, f"GM{gm}")
            pcs = [pc_names[pc] for _, pc in by_gm[gm]]
            parts.append(f"{name}:{','.join(pcs)}")

        print(f"  [{count:4d}x] {' | '.join(parts)}")


def analyze_pitch_class_coordination(coordination_sequences: dict):
    """Analyze which pitch classes play together."""
    print("\n" + "=" * 60)
    print("PITCH CLASS COORDINATION ANALYSIS")
    print("=" * 60)

    pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Count pitch class combinations
    pc_combos = Counter()
    interval_counts = Counter()

    for piece, events in coordination_sequences.items():
        for beat, event in events:
            pcs = sorted(set(pc for _, _, pc in event))

            if len(pcs) >= 2:
                pc_combos[tuple(pcs)] += 1

                # Count intervals between all pairs
                for i, pc1 in enumerate(pcs):
                    for pc2 in pcs[i+1:]:
                        interval = (pc2 - pc1) % 12
                        interval_counts[interval] += 1

    print(f"\nUnique PC combinations: {len(pc_combos)}")
    print(f"\nTop 15 pitch class combinations:")
    for combo, count in pc_combos.most_common(15):
        combo_str = ", ".join(pc_names[pc] for pc in combo)
        print(f"  [{count:4d}x] {{{combo_str}}}")

    print(f"\nInterval distribution in coordinated events:")
    interval_names = ["P1", "m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7"]
    total = sum(interval_counts.values())
    for iv in sorted(range(12), key=lambda x: -interval_counts[x]):
        count = interval_counts[iv]
        pct = 100 * count / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {interval_names[iv]:3s}: {pct:5.1f}% {bar}")


def build_simplified_coordination_grammar(
    coordination_sequences: dict,
    patterns: dict,
    min_count: int = 5
) -> dict:
    """Build a simplified coordination grammar for generation.

    Instead of full Re-Pair, we build:
    1. Common instrument combinations
    2. Common pitch class sets for each combination
    3. Pattern options for each (instrument, pitch_class)
    """
    print("\n" + "=" * 60)
    print("BUILDING COORDINATION GRAMMAR")
    print("=" * 60)

    # Track: (instrument_set) -> (pc_set) -> count
    inst_pc_combos = defaultdict(Counter)

    # Track: (gm, pc) -> [pattern_ids]
    gm_pc_patterns = defaultdict(set)

    for piece, events in coordination_sequences.items():
        for beat, event in events:
            # Instrument set
            gms = tuple(sorted(set(gm for gm, _, _ in event)))
            # Pitch class set
            pcs = tuple(sorted(set(pc for _, _, pc in event)))

            inst_pc_combos[gms][pcs] += 1

            # Track which patterns play at each (gm, pc)
            for gm, pid, pc in event:
                gm_pc_patterns[(gm, pc)].add(pid)

    # Filter by minimum count
    grammar = {
        'instrument_combinations': {},
        'gm_pc_patterns': {},
    }

    for gms, pc_counts in inst_pc_combos.items():
        filtered = {pcs: count for pcs, count in pc_counts.items() if count >= min_count}
        if filtered:
            # Convert to probabilities
            total = sum(filtered.values())
            probs = {pcs: count / total for pcs, count in filtered.items()}
            grammar['instrument_combinations'][gms] = probs

    for (gm, pc), pids in gm_pc_patterns.items():
        grammar['gm_pc_patterns'][(gm, pc)] = list(pids)

    print(f"Instrument combinations: {len(grammar['instrument_combinations'])}")
    print(f"(GM, PC) -> pattern mappings: {len(grammar['gm_pc_patterns'])}")

    # Show top instrument combinations
    print("\nTop instrument combinations:")
    sorted_combos = sorted(
        grammar['instrument_combinations'].items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    )[:10]

    gm_names = {
        0: "Piano", 32: "A.Bass", 33: "E.Bass",
        56: "Trumpet", 57: "Trombone", 58: "Tuba",
        65: "Alto.Sax", 66: "Ten.Sax", 128: "Drums",
    }

    for gms, pc_probs in sorted_combos:
        names = [gm_names.get(gm, f"GM{gm}") for gm in gms]
        n_pc_options = len(pc_probs)
        print(f"  {'+'.join(names)}: {n_pc_options} pitch class options")

    return grammar


class CoordinationSampler:
    """Generate using coordination grammar."""

    def __init__(self, grammar: dict, patterns: dict):
        self.grammar = grammar
        self.patterns = patterns

    def sample_coordination_event(
        self,
        instruments: Tuple[int, ...],
        previous_pcs: Optional[Tuple[int, ...]] = None
    ) -> Dict[int, Tuple[str, int]]:
        """Sample a coordinated vertical slice.

        Args:
            instruments: Tuple of GM programs to coordinate
            previous_pcs: Previous pitch classes for continuity

        Returns:
            {gm: (pattern_id, pitch)} for each instrument
        """
        inst_key = tuple(sorted(instruments))

        if inst_key not in self.grammar['instrument_combinations']:
            # Fallback: random pitches
            return self._fallback_sample(instruments)

        # Sample pitch class combination
        pc_probs = self.grammar['instrument_combinations'][inst_key]
        pcs_options = list(pc_probs.keys())
        probs = list(pc_probs.values())

        # Bias toward previous pitch classes for continuity
        if previous_pcs is not None:
            for i, pcs in enumerate(pcs_options):
                overlap = len(set(pcs) & set(previous_pcs))
                probs[i] *= (1 + 0.5 * overlap)

        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]

        chosen_pcs = random.choices(pcs_options, weights=probs)[0]

        # Assign pitch classes to instruments
        # (Simple: round-robin assignment)
        result = {}
        pc_list = list(chosen_pcs)

        for i, gm in enumerate(instruments):
            pc = pc_list[i % len(pc_list)]

            # Find a pattern for this (gm, pc)
            key = (gm, pc)
            if key in self.grammar['gm_pc_patterns']:
                pid = random.choice(self.grammar['gm_pc_patterns'][key])
            else:
                # Fallback: any pattern for this GM
                gm_patterns = [p for p, d in self.patterns.items()
                              if d.get('gm_program') == gm]
                pid = random.choice(gm_patterns) if gm_patterns else list(self.patterns.keys())[0]

            # Convert pitch class to actual pitch (use pattern's typical octave)
            pattern = self.patterns.get(pid, {})
            occs = pattern.get('occurrences', [])
            if occs:
                typical_pitch = occs[0].get('first_pitch', 60)
                octave = typical_pitch // 12
                pitch = octave * 12 + pc
            else:
                pitch = 60 + pc

            result[gm] = (pid, pitch)

        return result

    def _fallback_sample(self, instruments: Tuple[int, ...]) -> Dict[int, Tuple[str, int]]:
        """Fallback sampling when no coordination data."""
        result = {}
        for gm in instruments:
            gm_patterns = [p for p, d in self.patterns.items()
                          if d.get('gm_program') == gm]
            if gm_patterns:
                pid = random.choice(gm_patterns)
                pattern = self.patterns.get(pid, {})
                occs = pattern.get('occurrences', [])
                pitch = occs[0].get('first_pitch', 60) if occs else 60
                result[gm] = (pid, pitch)
            else:
                result[gm] = (list(self.patterns.keys())[0], 60)
        return result

    def generate(
        self,
        length: int,
        instruments: List[int],
        beats_per_pattern: int = 4
    ) -> Dict[int, List[Tuple[str, int]]]:
        """Generate coordinated multi-track output.

        Args:
            length: Number of coordination events
            instruments: GM programs to generate
            beats_per_pattern: How many beats each pattern spans

        Returns:
            {gm: [(pattern_id, pitch), ...]}
        """
        output = {gm: [] for gm in instruments}
        inst_tuple = tuple(sorted(instruments))
        previous_pcs = None

        for t in range(length):
            event = self.sample_coordination_event(inst_tuple, previous_pcs)

            # Record pitch classes for next iteration
            previous_pcs = tuple(sorted(set(
                self.patterns.get(pid, {}).get('occurrences', [{}])[0].get('first_pitch', 60) % 12
                for pid, _ in event.values()
            )))

            for gm in instruments:
                if gm in event:
                    output[gm].append(event[gm])

        return output


def main():
    import sys
    import os
    import numpy as np

    if len(sys.argv) < 2:
        print("Usage: python coordination_grammar.py <checkpoint.npz>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    # Load patterns from checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    data = np.load(checkpoint_path, allow_pickle=True)

    # Get patterns JSON path
    patterns_file = str(data['patterns_json_file'])
    if isinstance(data['patterns_json_file'], np.ndarray):
        patterns_file = str(data['patterns_json_file'][0])

    base_dir = os.path.dirname(checkpoint_path)
    json_path = os.path.join(base_dir, patterns_file)

    print(f"Loading patterns from: {json_path}")
    with open(json_path, 'r') as f:
        patterns = json.load(f)

    print(f"Loaded {len(patterns)} patterns")

    # Extract coordination
    coord_sequences, event_counts = build_coordination_from_checkpoint(patterns)

    # Analyze
    if event_counts:
        analyze_coordination_patterns(event_counts, patterns)
        analyze_pitch_class_coordination(coord_sequences)

        # Build grammar
        grammar = build_simplified_coordination_grammar(coord_sequences, patterns)

        # Test generation
        print("\n" + "=" * 60)
        print("TEST GENERATION")
        print("=" * 60)

        sampler = CoordinationSampler(grammar, patterns)
        output = sampler.generate(
            length=16,
            instruments=[0, 32, 56, 57],
        )

        print("\nGenerated coordination:")
        for gm, pairs in output.items():
            pitches = [p for _, p in pairs]
            pcs = [p % 12 for p in pitches]
            print(f"  GM {gm}: pitch classes = {pcs}")

        # Check vertical alignment
        print("\nVertical slices (first 8):")
        pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for t in range(min(8, len(output[0]))):
            slice_pcs = []
            for gm in sorted(output.keys()):
                _, pitch = output[gm][t]
                slice_pcs.append(pitch % 12)

            pc_str = ", ".join(pc_names[pc] for pc in slice_pcs)
            unique_pcs = len(set(slice_pcs))
            print(f"  t={t}: {{{pc_str}}} ({unique_pcs} unique PCs)")


if __name__ == '__main__':
    main()
