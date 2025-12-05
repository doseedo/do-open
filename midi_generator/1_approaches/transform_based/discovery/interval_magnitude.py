#!/usr/bin/env python3
"""
Interval Magnitude Discovery: Dual Chromatic/Diatonic Transform Representation
===============================================================================

This module provides a dual representation of transforms that allows the system
to DISCOVER diatonic relationships without prescribing music theory.

The Problem:
    - Chromatic: T3 (minor 3rd) and T4 (major 3rd) are different transforms
    - Diatonic: Both are "3rds" that function similarly in scale contexts
    - Current system can't discover this relationship

The Solution:
    - Store BOTH representations for each transform
    - Let Re-Pair/MDL discover which is more compressive for each context
    - If diatonic patterns exist, magnitude representation will compress better

Representation:
    ChromaticTransform: T0, T1, T2, ..., T11 (12 distinct)
    MagnitudeTransform: M0(unison), M1(2nd), M2(3rd), M3(4th), M4(5th), M5(6th), M6(7th) (7 distinct)

The system can now discover:
    - "In this context, T3 and T4 are interchangeable" (both M2)
    - "In this context, T3 and T4 are distinct" (keep chromatic)
    - Modal patterns naturally emerge from magnitude sequences

Author: Interval Magnitude Discovery System
"""

import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import time

# Import from parent package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.primitives import (
    semitones_to_magnitude,
    magnitude_to_name,
    IntervalMagnitude,
    SEMITONE_TO_MAGNITUDE,
    MAGNITUDE_NAMES,
    CompoundTransform,
    Primitive,
    PrimitiveType,
)


@dataclass
class DualTransform:
    """
    A transform with both chromatic and magnitude representations.

    This allows the system to see:
        - Chromatic view: T3 ≠ T4 (different semitones)
        - Magnitude view: T3 ~ T4 (both are "3rds", magnitude 2)

    MDL optimization chooses which view compresses better for each context.
    """
    semitones: int          # Chromatic: 0-11
    magnitude: int          # Diatonic-agnostic: 0-6
    direction: int          # +1 (up) or -1 (down), for directional transforms

    @classmethod
    def from_semitones(cls, semitones: int, direction: int = 1) -> 'DualTransform':
        """Create DualTransform from semitone distance."""
        s = semitones % 12
        return cls(
            semitones=s,
            magnitude=semitones_to_magnitude(s),
            direction=direction,
        )

    @property
    def chromatic_name(self) -> str:
        """Chromatic representation: T0, T1, ..., T11"""
        if self.semitones == 0:
            return "T0"
        return f"T{self.semitones}"

    @property
    def magnitude_name(self) -> str:
        """Magnitude representation: U, 2, 3, 4, 5, 6, 7"""
        return magnitude_to_name(self.magnitude)

    @property
    def dual_name(self) -> str:
        """Combined representation for debugging."""
        return f"{self.chromatic_name}({self.magnitude_name})"

    def __str__(self):
        return self.dual_name

    def __hash__(self):
        return hash((self.semitones, self.magnitude, self.direction))

    def __eq__(self, other):
        if not isinstance(other, DualTransform):
            return False
        return (self.semitones == other.semitones and
                self.magnitude == other.magnitude and
                self.direction == other.direction)


def build_dual_vocabulary() -> Tuple[List[DualTransform], Dict[int, int], Dict[int, List[int]]]:
    """
    Build the dual transform vocabulary.

    Returns:
        - vocab: List of all 12 DualTransforms (T0-T11)
        - chromatic_to_idx: semitones -> vocab index
        - magnitude_to_chromatic: magnitude -> list of semitone values
    """
    vocab = []
    chromatic_to_idx = {}
    magnitude_to_chromatic = defaultdict(list)

    for s in range(12):
        dt = DualTransform.from_semitones(s)
        vocab.append(dt)
        chromatic_to_idx[s] = s
        magnitude_to_chromatic[dt.magnitude].append(s)

    return vocab, chromatic_to_idx, dict(magnitude_to_chromatic)


def convert_chromatic_to_magnitude_sequence(
    chromatic_seq: List[int],
) -> List[int]:
    """
    Convert a sequence of chromatic transforms to magnitude transforms.

    Args:
        chromatic_seq: List of semitone values (0-11)

    Returns:
        List of magnitude values (0-6)
    """
    return [semitones_to_magnitude(s) for s in chromatic_seq]


def build_magnitude_lookup_gpu(
    patterns: List[Dict],
    device: str = 'cuda',
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build BOTH chromatic and magnitude transform lookup tables on GPU.

    This is the key function: instead of just computing chromatic transforms,
    we compute both representations simultaneously.

    Returns:
        chromatic_lookup: [n_patterns, n_patterns] -> semitone transform (0-11, or -1)
        magnitude_lookup: [n_patterns, n_patterns] -> magnitude transform (0-6, or -1)
    """
    if not torch.cuda.is_available():
        device = 'cpu'

    n_patterns = len(patterns)
    if n_patterns == 0:
        empty = torch.empty((0, 0), dtype=torch.int16, device=device)
        return empty, empty

    # Build pattern tensor
    max_len = max(len(p.get('pitch_classes', [])) for p in patterns)
    pattern_np = np.full((n_patterns, max_len), -1, dtype=np.int32)
    lengths_np = np.zeros(n_patterns, dtype=np.int32)

    for i, p in enumerate(patterns):
        pc = p.get('pitch_classes', [])
        if pc:
            pattern_np[i, :len(pc)] = pc
            lengths_np[i] = len(pc)

    pattern_tensor = torch.from_numpy(pattern_np).to(device)
    pattern_lengths = torch.from_numpy(lengths_np).to(device)

    # Initialize lookup tables
    chromatic_lookup = torch.full((n_patterns, n_patterns), -1, dtype=torch.int16, device=device)
    magnitude_lookup = torch.full((n_patterns, n_patterns), -1, dtype=torch.int16, device=device)

    # Semitone to magnitude lookup on GPU
    s2m = torch.tensor(SEMITONE_TO_MAGNITUDE, dtype=torch.int16, device=device)

    # Group by length
    unique_lengths = torch.unique(pattern_lengths)

    for length in unique_lengths:
        length_val = length.item()
        if length_val == 0:
            continue

        len_mask = pattern_lengths == length
        indices = torch.where(len_mask)[0]
        n_same_len = indices.shape[0]

        if n_same_len < 2:
            continue

        # Extract patterns of this length
        sub_patterns = pattern_tensor[indices, :length_val]  # [M, L]
        M, L = sub_patterns.shape

        # Check all 12 transpositions
        for t in range(12):
            transposed = (sub_patterns + t) % 12  # [M, L]

            # Compare all pairs
            # transposed[i] vs sub_patterns[j]
            trans_expanded = transposed.unsqueeze(1)  # [M, 1, L]
            target_expanded = sub_patterns.unsqueeze(0)  # [1, M, L]

            matches = (trans_expanded == target_expanded).all(dim=2)  # [M, M]

            # Update lookups where we found matches and haven't already
            match_pairs = matches.nonzero(as_tuple=False)

            if len(match_pairs) > 0:
                src_local = match_pairs[:, 0]
                tgt_local = match_pairs[:, 1]
                src_global = indices[src_local]
                tgt_global = indices[tgt_local]

                # Only update where not yet set
                not_set = chromatic_lookup[src_global, tgt_global] == -1
                src_update = src_global[not_set]
                tgt_update = tgt_global[not_set]

                if len(src_update) > 0:
                    chromatic_lookup[src_update, tgt_update] = t
                    magnitude_lookup[src_update, tgt_update] = s2m[t]

    if verbose:
        n_chromatic = (chromatic_lookup >= 0).sum().item()
        print(f"  Built dual lookup: {n_chromatic} transform pairs")

    return chromatic_lookup, magnitude_lookup


def extract_dual_transform_sequences(
    patterns: List[Dict],
    chromatic_lookup: torch.Tensor,
    magnitude_lookup: torch.Tensor,
    min_sequence_length: int = 3,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Extract BOTH chromatic and magnitude transform sequences.

    For each piece, builds two parallel sequences:
        - Chromatic: [T4, T3, T7, T5, ...]
        - Magnitude: [M2, M2, M4, M3, ...]  (3rds grouped together!)

    Returns:
        chromatic_sequences: List of chromatic transform sequences
        magnitude_sequences: List of magnitude transform sequences (parallel)
    """
    # Group occurrences by piece
    piece_occurrences = defaultdict(list)

    for p_idx, p in enumerate(patterns):
        for occ in p.get('occurrences', []):
            piece_occurrences[occ['piece_id']].append({
                'pattern_idx': p_idx,
                'onset_time': occ['onset_time'],
            })

    # Sort each piece's occurrences by time
    for piece_id in piece_occurrences:
        piece_occurrences[piece_id].sort(key=lambda x: x['onset_time'])

    chromatic_lookup_cpu = chromatic_lookup.cpu().numpy()
    magnitude_lookup_cpu = magnitude_lookup.cpu().numpy()

    chromatic_sequences = []
    magnitude_sequences = []

    for piece_id, occs in piece_occurrences.items():
        if len(occs) < min_sequence_length:
            continue

        chrom_seq = []
        mag_seq = []

        for i in range(len(occs) - 1):
            src_idx = occs[i]['pattern_idx']
            tgt_idx = occs[i + 1]['pattern_idx']

            if src_idx == tgt_idx:
                continue

            chrom_t = chromatic_lookup_cpu[src_idx, tgt_idx]
            mag_t = magnitude_lookup_cpu[src_idx, tgt_idx]

            if chrom_t >= 0:
                chrom_seq.append(int(chrom_t))
                mag_seq.append(int(mag_t))

        if len(chrom_seq) >= min_sequence_length - 1:
            chromatic_sequences.append(chrom_seq)
            magnitude_sequences.append(mag_seq)

    return chromatic_sequences, magnitude_sequences


def compute_compression_comparison(
    chromatic_sequences: List[List[int]],
    magnitude_sequences: List[List[int]],
    device: str = 'cuda',
    verbose: bool = True,
) -> Dict:
    """
    Run Re-Pair on BOTH representations and compare compression.

    This is the key MDL test: which representation compresses better?

    Returns:
        Dict with compression stats for both representations
    """
    from grammar.v2.repair_gpu_v2 import build_repair_grammar_v2

    results = {
        'chromatic': {'sequences': len(chromatic_sequences), 'tokens': 0, 'rules': 0, 'compression': 0.0},
        'magnitude': {'sequences': len(magnitude_sequences), 'tokens': 0, 'rules': 0, 'compression': 0.0},
    }

    # Chromatic compression
    if chromatic_sequences and len(chromatic_sequences) >= 10:
        total_tokens = sum(len(s) for s in chromatic_sequences)
        results['chromatic']['tokens'] = total_tokens

        try:
            grammar = build_repair_grammar_v2(
                chromatic_sequences,
                device=device,
                max_rules=500,
                min_pair_count=3,
                verbose=False,
            )
            results['chromatic']['rules'] = grammar.n_rules
            results['chromatic']['compression'] = grammar.compression_ratio()
        except Exception as e:
            if verbose:
                print(f"  Chromatic Re-Pair failed: {e}")

    # Magnitude compression
    if magnitude_sequences and len(magnitude_sequences) >= 10:
        total_tokens = sum(len(s) for s in magnitude_sequences)
        results['magnitude']['tokens'] = total_tokens

        try:
            grammar = build_repair_grammar_v2(
                magnitude_sequences,
                device=device,
                max_rules=500,
                min_pair_count=3,
                verbose=False,
            )
            results['magnitude']['rules'] = grammar.n_rules
            results['magnitude']['compression'] = grammar.compression_ratio()
        except Exception as e:
            if verbose:
                print(f"  Magnitude Re-Pair failed: {e}")

    if verbose:
        print(f"  Compression comparison:")
        print(f"    Chromatic (T0-T11): {results['chromatic']['compression']:.2f}x "
              f"({results['chromatic']['rules']} rules)")
        print(f"    Magnitude (U-7):    {results['magnitude']['compression']:.2f}x "
              f"({results['magnitude']['rules']} rules)")

        if results['magnitude']['compression'] > results['chromatic']['compression'] * 1.05:
            print(f"    → Magnitude wins: corpus has diatonic structure!")
        elif results['chromatic']['compression'] > results['magnitude']['compression'] * 1.05:
            print(f"    → Chromatic wins: corpus is chromatically precise")
        else:
            print(f"    → Similar: mixed diatonic/chromatic content")

    return results


def analyze_magnitude_patterns(
    magnitude_sequences: List[List[int]],
    verbose: bool = True,
) -> Dict:
    """
    Analyze patterns in magnitude sequences to discover diatonic regularities.

    Common patterns that might emerge:
        - [2, 2, 1, 2, 2, 2, 1] = major scale step pattern
        - [2, 1, 2, 2, 1, 2, 2] = minor scale step pattern
        - [2, 2, 3] = chord arpeggios (3rd, 3rd, 4th)
    """
    from collections import Counter

    # Count bigrams
    bigram_counts = Counter()
    for seq in magnitude_sequences:
        for i in range(len(seq) - 1):
            bigram = (seq[i], seq[i + 1])
            bigram_counts[bigram] += 1

    # Count trigrams
    trigram_counts = Counter()
    for seq in magnitude_sequences:
        for i in range(len(seq) - 2):
            trigram = (seq[i], seq[i + 1], seq[i + 2])
            trigram_counts[trigram] += 1

    # Convert to readable form
    def mag_name(m):
        return MAGNITUDE_NAMES[m] if 0 <= m < len(MAGNITUDE_NAMES) else f"M{m}"

    top_bigrams = [
        (f"{mag_name(a)}→{mag_name(b)}", count)
        for (a, b), count in bigram_counts.most_common(10)
    ]

    top_trigrams = [
        (f"{mag_name(a)}→{mag_name(b)}→{mag_name(c)}", count)
        for (a, b, c), count in trigram_counts.most_common(10)
    ]

    if verbose:
        print(f"\n  Top magnitude bigrams (diatonic patterns):")
        for pattern, count in top_bigrams[:5]:
            print(f"    {pattern}: {count}")

        print(f"\n  Top magnitude trigrams:")
        for pattern, count in top_trigrams[:5]:
            print(f"    {pattern}: {count}")

    return {
        'bigrams': top_bigrams,
        'trigrams': top_trigrams,
        'unique_bigrams': len(bigram_counts),
        'unique_trigrams': len(trigram_counts),
    }


def run_interval_magnitude_discovery(
    patterns: List[Dict],
    device: str = 'cuda',
    verbose: bool = True,
) -> Dict:
    """
    Run full interval magnitude discovery pipeline.

    This discovers:
    1. Whether magnitude (diatonic) or chromatic representation compresses better
    2. Common diatonic patterns in the corpus
    3. Builds dual vocabulary for downstream use

    Args:
        patterns: List of pattern dicts with occurrences
        device: PyTorch device
        verbose: Print progress

    Returns:
        Dict with:
            - dual_vocab: List of DualTransform objects
            - compression_comparison: Which representation wins
            - magnitude_patterns: Common diatonic patterns found
            - chromatic_sequences: For further processing
            - magnitude_sequences: For further processing
    """
    t0 = time.time()

    if verbose:
        print(f"\n[Interval Magnitude Discovery]")
        print(f"  Building dual transform vocabulary...")

    # Build vocabulary
    dual_vocab, chromatic_to_idx, magnitude_to_chromatic = build_dual_vocabulary()

    if verbose:
        print(f"    Chromatic transforms: 12 (T0-T11)")
        print(f"    Magnitude classes: 7 (U, 2nd, 3rd, 4th, 5th, 6th, 7th)")
        print(f"    Mapping: {magnitude_to_chromatic}")

    # Build dual lookup tables
    if verbose:
        print(f"  Building dual lookup tables...")

    chromatic_lookup, magnitude_lookup = build_magnitude_lookup_gpu(
        patterns, device=device, verbose=verbose
    )

    # Extract dual sequences
    if verbose:
        print(f"  Extracting transform sequences...")

    chromatic_sequences, magnitude_sequences = extract_dual_transform_sequences(
        patterns, chromatic_lookup, magnitude_lookup
    )

    if verbose:
        print(f"    {len(chromatic_sequences)} sequences extracted")

    # Compare compression
    if verbose:
        print(f"  Comparing compression (MDL test)...")

    compression_comparison = compute_compression_comparison(
        chromatic_sequences, magnitude_sequences,
        device=device, verbose=verbose
    )

    # Analyze magnitude patterns
    magnitude_patterns = {}
    if magnitude_sequences:
        magnitude_patterns = analyze_magnitude_patterns(
            magnitude_sequences, verbose=verbose
        )

    elapsed = time.time() - t0

    # Determine which representation is better
    chrom_comp = compression_comparison['chromatic']['compression']
    mag_comp = compression_comparison['magnitude']['compression']

    if mag_comp > chrom_comp * 1.05:
        preferred = 'magnitude'
        reason = 'corpus has diatonic structure'
    elif chrom_comp > mag_comp * 1.05:
        preferred = 'chromatic'
        reason = 'corpus is chromatically precise'
    else:
        preferred = 'both'
        reason = 'mixed content, use both'

    if verbose:
        print(f"\n  Discovery complete in {elapsed:.1f}s")
        print(f"  Preferred representation: {preferred} ({reason})")

    return {
        'dual_vocab': dual_vocab,
        'chromatic_to_idx': chromatic_to_idx,
        'magnitude_to_chromatic': magnitude_to_chromatic,
        'compression_comparison': compression_comparison,
        'magnitude_patterns': magnitude_patterns,
        'chromatic_sequences': chromatic_sequences,
        'magnitude_sequences': magnitude_sequences,
        'preferred_representation': preferred,
        'elapsed_time': elapsed,
    }


if __name__ == "__main__":
    print("Interval Magnitude Discovery Module")
    print("=" * 50)

    # Test with sample data showing diatonic vs chromatic
    patterns = [
        # C major scale fragments
        {'pitch_classes': [0, 2, 4], 'occurrences': [{'piece_id': 'test', 'onset_time': 0}]},  # C-D-E
        {'pitch_classes': [2, 4, 5], 'occurrences': [{'piece_id': 'test', 'onset_time': 100}]},  # D-E-F
        {'pitch_classes': [4, 5, 7], 'occurrences': [{'piece_id': 'test', 'onset_time': 200}]},  # E-F-G
        {'pitch_classes': [5, 7, 9], 'occurrences': [{'piece_id': 'test', 'onset_time': 300}]},  # F-G-A
        {'pitch_classes': [7, 9, 11], 'occurrences': [{'piece_id': 'test', 'onset_time': 400}]},  # G-A-B
        {'pitch_classes': [9, 11, 0], 'occurrences': [{'piece_id': 'test', 'onset_time': 500}]},  # A-B-C
    ]

    # Show the diatonic pattern: these all have magnitude sequence [1, 1] (2nd, 2nd)
    # but chromatic sequences vary: [2,2], [2,1], [1,2], [2,2], [2,2], [2,1]

    print("\nTest patterns (C major scale fragments):")
    for i, p in enumerate(patterns):
        pc = p['pitch_classes']
        intervals = [(pc[j+1] - pc[j]) % 12 for j in range(len(pc)-1)]
        magnitudes = [semitones_to_magnitude(i) for i in intervals]
        print(f"  {i}: {pc} -> intervals {intervals} -> magnitudes {magnitudes}")

    print("\nNote: All have magnitude [1,1] (seconds) but chromatic varies!")
    print("This is exactly what diatonic discovery should find.")

    # Run discovery
    result = run_interval_magnitude_discovery(
        patterns,
        device='cpu',
        verbose=True,
    )

    print(f"\nPreferred representation: {result['preferred_representation']}")
