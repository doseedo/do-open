#!/usr/bin/env python3
"""
Multi-Factor Transform Discovery
================================

Extends primitive-based transform discovery to ALL factor spaces:
- Pitch (T, I, R) - existing in primitives.py
- Rhythm (τ) - time scaling transforms
- Velocity (v) - dynamics scaling transforms
- Duration (d) - note length scaling transforms

Philosophy: Don't prescribe music theory. Let the system DISCOVER
which transforms are productive via MDL optimization.

The Key Equation (from Doseedo philosophy):
    M = Pattern_canonical × T(pitch_offset) × τ(rhythm_offset) × v(velocity_offset) × O(octave_offset)

This module discovers relationships in τ, v, d spaces that the pitch-only
primitives.py cannot find.

Author: Multi-Factor Transform Discovery
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import math


class FactorType(Enum):
    """Types of musical factors that can be transformed."""
    PITCH = "pitch"           # Pitch class transforms (T, I, R)
    RHYTHM = "rhythm"         # Inter-onset interval transforms (τ)
    VELOCITY = "velocity"     # Dynamics transforms (v)
    DURATION = "duration"     # Note length transforms (d)
    OCTAVE = "octave"         # Register transforms (O)


@dataclass(frozen=True)
class FactorTransform:
    """A transform in a specific factor space."""
    factor: FactorType
    transform_type: str       # 'scale', 'retrograde', 'shift', etc.
    param: float              # Scale factor or shift amount

    def __str__(self):
        if self.transform_type == 'identity':
            return f"{self.factor.value}:id"
        elif self.transform_type == 'scale':
            return f"{self.factor.value}:×{self.param:.2f}"
        elif self.transform_type == 'retrograde':
            return f"{self.factor.value}:R"
        elif self.transform_type == 'shift':
            return f"{self.factor.value}:+{self.param}"
        else:
            return f"{self.factor.value}:{self.transform_type}({self.param})"

    def __repr__(self):
        return str(self)


@dataclass
class MultiFactorRelation:
    """A relation between two patterns across multiple factors."""
    source_id: str
    target_id: str
    pitch_transform: Optional[str] = None     # From primitives.py (T, I, R)
    rhythm_transform: Optional[FactorTransform] = None
    velocity_transform: Optional[FactorTransform] = None
    duration_transform: Optional[FactorTransform] = None

    @property
    def is_complete(self) -> bool:
        """True if all factors have transforms."""
        return all([
            self.pitch_transform is not None,
            self.rhythm_transform is not None,
            self.velocity_transform is not None,
            self.duration_transform is not None,
        ])

    @property
    def transforms_str(self) -> str:
        """String representation of all transforms."""
        parts = []
        if self.pitch_transform:
            parts.append(f"P:{self.pitch_transform}")
        if self.rhythm_transform:
            parts.append(str(self.rhythm_transform))
        if self.velocity_transform:
            parts.append(str(self.velocity_transform))
        if self.duration_transform:
            parts.append(str(self.duration_transform))
        return " × ".join(parts) if parts else "identity"


@dataclass
class FactorTransformStats:
    """Statistics for a discovered factor transform."""
    transform: FactorTransform
    frequency: int
    mdl_benefit: float
    example_pairs: List[Tuple[str, str]] = field(default_factory=list)


# =============================================================================
# SCALE FACTOR DETECTION
# =============================================================================

# Standard scale factors to check (musical ratios)
RHYTHM_SCALE_FACTORS = [
    0.25,   # Quadruple diminution
    0.333,  # Triplet diminution
    0.5,    # Diminution (double time)
    0.667,  # Dotted-to-undotted
    0.75,   # Dotted diminution
    1.0,    # Identity
    1.333,  # Undotted-to-dotted
    1.5,    # Dotted augmentation
    2.0,    # Augmentation (half time)
    3.0,    # Triple augmentation
    4.0,    # Quadruple augmentation
]

VELOCITY_SCALE_FACTORS = [
    0.5,    # pp to p, or p to mp
    0.667,  # Soft
    0.75,   # Slightly softer
    0.85,   # Subtle softer
    1.0,    # Identity
    1.15,   # Subtle louder
    1.25,   # Slightly louder
    1.333,  # Louder
    1.5,    # Accent
    2.0,    # Strong accent
]

DURATION_SCALE_FACTORS = [
    0.25,   # Very staccato
    0.5,    # Staccato
    0.75,   # Slightly short
    1.0,    # Identity
    1.25,   # Slightly long
    1.5,    # Tenuto
    2.0,    # Very sustained
]


def find_scale_factor(
    source: np.ndarray,
    target: np.ndarray,
    candidates: List[float],
    tolerance: float = 0.1
) -> Optional[float]:
    """
    Find if target = source × k for some k in candidates.

    Args:
        source: Source ratio array
        target: Target ratio array
        candidates: Scale factors to try
        tolerance: Relative tolerance for matching

    Returns:
        Best matching scale factor, or None if no match
    """
    if len(source) != len(target) or len(source) == 0:
        return None

    # Avoid division by zero
    if np.any(source == 0):
        return None

    # Compute actual ratio
    ratios = target / source

    # Check if all ratios are approximately equal (constant scale)
    if len(ratios) > 1:
        ratio_std = np.std(ratios)
        ratio_mean = np.mean(ratios)
        if ratio_mean > 0 and ratio_std / ratio_mean > tolerance:
            return None  # Not a constant scale

    mean_ratio = np.mean(ratios)

    # Find closest candidate
    best_candidate = None
    best_error = float('inf')

    for k in candidates:
        error = abs(mean_ratio - k) / max(abs(k), 0.01)
        if error < tolerance and error < best_error:
            best_candidate = k
            best_error = error

    return best_candidate


def find_retrograde_match(
    source: np.ndarray,
    target: np.ndarray,
    tolerance: float = 0.1
) -> bool:
    """Check if target is retrograde of source (reversed order)."""
    if len(source) != len(target) or len(source) < 2:
        return False

    source_reversed = source[::-1]

    # Check if reversed source matches target within tolerance
    if np.allclose(source_reversed, target, rtol=tolerance, atol=0.01):
        return True

    return False


# =============================================================================
# RHYTHM (τ) TRANSFORM DISCOVERY
# =============================================================================

def discover_rhythm_transforms_gpu(
    patterns: List[Dict],
    device: str = 'cuda',
    tolerance: float = 0.1,
    verbose: bool = True
) -> Dict[FactorTransform, List[Tuple[str, str]]]:
    """
    GPU-accelerated rhythm (τ) transform discovery.

    Finds pattern pairs where:
    - Same pitch content (pitch_classes match)
    - Rhythm differs by scale factor k: rhythm_b = rhythm_a × τ(k)

    Args:
        patterns: List of pattern dicts with 'pitch_classes' and 'rhythm_ratios'
        device: PyTorch device
        tolerance: Matching tolerance
        verbose: Print progress

    Returns:
        Dict mapping FactorTransform -> list of (source_id, target_id) pairs
    """
    try:
        import torch
    except ImportError:
        return discover_rhythm_transforms_cpu(patterns, tolerance, verbose)

    if verbose:
        print(f"  Discovering rhythm (τ) transforms on {len(patterns)} patterns...", flush=True)

    # Group patterns by pitch content (same pitch = candidates for τ relation)
    by_pitch = defaultdict(list)
    for i, p in enumerate(patterns):
        pc = tuple(p.get('pitch_classes', []))
        rr = p.get('rhythm_ratios', [])
        if pc and len(rr) >= 1:
            by_pitch[pc].append((i, np.array(rr, dtype=np.float32)))

    relations = defaultdict(list)
    total_pairs = 0
    total_found = 0

    for pitch_sig, group in by_pitch.items():
        if len(group) < 2:
            continue

        # Check all pairs within this pitch group
        for i, (idx_a, rr_a) in enumerate(group):
            for idx_b, rr_b in group[i+1:]:
                total_pairs += 1

                if len(rr_a) != len(rr_b):
                    continue

                # Check for scale factor
                k = find_scale_factor(rr_a, rr_b, RHYTHM_SCALE_FACTORS, tolerance)
                if k is not None and k != 1.0:
                    transform = FactorTransform(FactorType.RHYTHM, 'scale', k)
                    relations[transform].append((str(idx_a), str(idx_b)))
                    total_found += 1
                    continue

                # Check reverse direction
                k_rev = find_scale_factor(rr_b, rr_a, RHYTHM_SCALE_FACTORS, tolerance)
                if k_rev is not None and k_rev != 1.0:
                    transform = FactorTransform(FactorType.RHYTHM, 'scale', k_rev)
                    relations[transform].append((str(idx_b), str(idx_a)))
                    total_found += 1
                    continue

                # Check for rhythm retrograde (same notes, reversed timing)
                if find_retrograde_match(rr_a, rr_b, tolerance):
                    transform = FactorTransform(FactorType.RHYTHM, 'retrograde', 0)
                    relations[transform].append((str(idx_a), str(idx_b)))
                    total_found += 1

    if verbose:
        print(f"    Checked {total_pairs} same-pitch pairs, found {total_found} τ relations")
        print(f"    Unique τ transforms: {len(relations)}")
        for t, pairs in sorted(relations.items(), key=lambda x: -len(x[1]))[:5]:
            print(f"      {t}: {len(pairs)} pairs")

    return dict(relations)


def discover_rhythm_transforms_cpu(
    patterns: List[Dict],
    tolerance: float = 0.1,
    verbose: bool = True
) -> Dict[FactorTransform, List[Tuple[str, str]]]:
    """CPU fallback for rhythm transform discovery."""
    return discover_rhythm_transforms_gpu(patterns, 'cpu', tolerance, verbose)


# =============================================================================
# VELOCITY (v) TRANSFORM DISCOVERY
# =============================================================================

def discover_velocity_transforms_gpu(
    patterns: List[Dict],
    device: str = 'cuda',
    tolerance: float = 0.15,
    verbose: bool = True
) -> Dict[FactorTransform, List[Tuple[str, str]]]:
    """
    GPU-accelerated velocity (v) transform discovery.

    Finds pattern pairs where:
    - Same pitch AND rhythm content
    - Velocity differs by scale factor k: velocity_b = velocity_a × v(k)

    Args:
        patterns: List of pattern dicts with pitch_classes, rhythm_ratios, velocity_ratios
        device: PyTorch device
        tolerance: Matching tolerance
        verbose: Print progress

    Returns:
        Dict mapping FactorTransform -> list of (source_id, target_id) pairs
    """
    if verbose:
        print(f"  Discovering velocity (v) transforms on {len(patterns)} patterns...", flush=True)

    # Group patterns by (pitch, rhythm) content
    def make_rhythm_key(rr):
        """Quantize rhythm ratios for grouping."""
        return tuple(round(r * 4) / 4 for r in rr)  # Quarter resolution

    by_pitch_rhythm = defaultdict(list)
    for i, p in enumerate(patterns):
        pc = tuple(p.get('pitch_classes', []))
        rr = p.get('rhythm_ratios', [])
        vr = p.get('velocity_ratios', [])
        if pc and len(vr) >= 1:
            rr_key = make_rhythm_key(rr) if rr else ()
            key = (pc, rr_key)
            by_pitch_rhythm[key].append((i, np.array(vr, dtype=np.float32)))

    relations = defaultdict(list)
    total_pairs = 0
    total_found = 0

    for key, group in by_pitch_rhythm.items():
        if len(group) < 2:
            continue

        for i, (idx_a, vr_a) in enumerate(group):
            for idx_b, vr_b in group[i+1:]:
                total_pairs += 1

                if len(vr_a) != len(vr_b):
                    continue

                # Check for scale factor
                k = find_scale_factor(vr_a, vr_b, VELOCITY_SCALE_FACTORS, tolerance)
                if k is not None and k != 1.0:
                    transform = FactorTransform(FactorType.VELOCITY, 'scale', k)
                    relations[transform].append((str(idx_a), str(idx_b)))
                    total_found += 1
                    continue

                # Check reverse direction
                k_rev = find_scale_factor(vr_b, vr_a, VELOCITY_SCALE_FACTORS, tolerance)
                if k_rev is not None and k_rev != 1.0:
                    transform = FactorTransform(FactorType.VELOCITY, 'scale', k_rev)
                    relations[transform].append((str(idx_b), str(idx_a)))
                    total_found += 1

    if verbose:
        print(f"    Checked {total_pairs} same-pitch-rhythm pairs, found {total_found} v relations")
        print(f"    Unique v transforms: {len(relations)}")
        for t, pairs in sorted(relations.items(), key=lambda x: -len(x[1]))[:5]:
            print(f"      {t}: {len(pairs)} pairs")

    return dict(relations)


# =============================================================================
# DURATION (d) TRANSFORM DISCOVERY
# =============================================================================

def discover_duration_transforms_gpu(
    patterns: List[Dict],
    device: str = 'cuda',
    tolerance: float = 0.15,
    verbose: bool = True
) -> Dict[FactorTransform, List[Tuple[str, str]]]:
    """
    GPU-accelerated duration (d) transform discovery.

    Finds pattern pairs where:
    - Same pitch AND rhythm content
    - Duration differs by scale factor k: duration_b = duration_a × d(k)

    This captures staccato/legato variations of the same melodic content.

    Args:
        patterns: List of pattern dicts
        device: PyTorch device
        tolerance: Matching tolerance
        verbose: Print progress

    Returns:
        Dict mapping FactorTransform -> list of (source_id, target_id) pairs
    """
    if verbose:
        print(f"  Discovering duration (d) transforms on {len(patterns)} patterns...", flush=True)

    # Group patterns by (pitch, rhythm) content
    def make_rhythm_key(rr):
        return tuple(round(r * 4) / 4 for r in rr)

    by_pitch_rhythm = defaultdict(list)
    for i, p in enumerate(patterns):
        pc = tuple(p.get('pitch_classes', []))
        rr = p.get('rhythm_ratios', [])
        dr = p.get('duration_ratios', [])
        if pc and len(dr) >= 1:
            rr_key = make_rhythm_key(rr) if rr else ()
            key = (pc, rr_key)
            by_pitch_rhythm[key].append((i, np.array(dr, dtype=np.float32)))

    relations = defaultdict(list)
    total_pairs = 0
    total_found = 0

    for key, group in by_pitch_rhythm.items():
        if len(group) < 2:
            continue

        for i, (idx_a, dr_a) in enumerate(group):
            for idx_b, dr_b in group[i+1:]:
                total_pairs += 1

                if len(dr_a) != len(dr_b):
                    continue

                # Check for scale factor
                k = find_scale_factor(dr_a, dr_b, DURATION_SCALE_FACTORS, tolerance)
                if k is not None and k != 1.0:
                    transform = FactorTransform(FactorType.DURATION, 'scale', k)
                    relations[transform].append((str(idx_a), str(idx_b)))
                    total_found += 1
                    continue

                # Check reverse direction
                k_rev = find_scale_factor(dr_b, dr_a, DURATION_SCALE_FACTORS, tolerance)
                if k_rev is not None and k_rev != 1.0:
                    transform = FactorTransform(FactorType.DURATION, 'scale', k_rev)
                    relations[transform].append((str(idx_b), str(idx_a)))
                    total_found += 1

    if verbose:
        print(f"    Checked {total_pairs} same-pitch-rhythm pairs, found {total_found} d relations")
        print(f"    Unique d transforms: {len(relations)}")
        for t, pairs in sorted(relations.items(), key=lambda x: -len(x[1]))[:5]:
            print(f"      {t}: {len(pairs)} pairs")

    return dict(relations)


# =============================================================================
# CROSS-FACTOR TRANSFORM DISCOVERY
# =============================================================================

def discover_cross_factor_relations(
    patterns: List[Dict],
    device: str = 'cuda',
    tolerance: float = 0.1,
    verbose: bool = True
) -> List[MultiFactorRelation]:
    """
    Discover patterns related by transforms in DIFFERENT factor spaces.

    Example: Same rhythm, different pitch = "reharmonization"
    Example: Same pitch, different rhythm = "rhythmic variation"

    This enables discovering compositional relationships:
        Pattern_B = Pattern_A × T(5) × τ(2.0) × v(0.8)
        "B is A transposed up a fourth, twice as slow, softer"

    Args:
        patterns: List of pattern dicts
        device: PyTorch device
        tolerance: Matching tolerance
        verbose: Print progress

    Returns:
        List of MultiFactorRelation objects
    """
    if verbose:
        print(f"  Discovering cross-factor relations on {len(patterns)} patterns...", flush=True)

    # Quantization helpers
    def quantize_rhythm(rr):
        return tuple(round(r * 4) / 4 for r in rr)

    # Group by rhythm signature (to find "same rhythm, different pitch")
    by_rhythm = defaultdict(list)
    for i, p in enumerate(patterns):
        rr = p.get('rhythm_ratios', [])
        if len(rr) >= 1:
            rr_key = quantize_rhythm(rr)
            by_rhythm[rr_key].append(i)

    # Group by pitch (to find "same pitch, different rhythm")
    by_pitch = defaultdict(list)
    for i, p in enumerate(patterns):
        pc = tuple(p.get('pitch_classes', []))
        if pc:
            by_pitch[pc].append(i)

    relations = []

    # Find "same rhythm, different pitch" relations
    same_rhythm_different_pitch = 0
    for rr_key, indices in by_rhythm.items():
        if len(indices) < 2 or len(rr_key) < 2:
            continue

        # All pairs in this group have same rhythm
        for i, idx_a in enumerate(indices):
            for idx_b in indices[i+1:]:
                pc_a = tuple(patterns[idx_a].get('pitch_classes', []))
                pc_b = tuple(patterns[idx_b].get('pitch_classes', []))

                if pc_a != pc_b and len(pc_a) == len(pc_b):
                    # Same rhythm, different pitch - check what pitch transform relates them
                    # (This would be discovered by primitives.py, but we note the relation)
                    rel = MultiFactorRelation(
                        source_id=str(idx_a),
                        target_id=str(idx_b),
                        pitch_transform="different",  # Will be filled by pitch discovery
                        rhythm_transform=FactorTransform(FactorType.RHYTHM, 'identity', 1.0),
                    )
                    relations.append(rel)
                    same_rhythm_different_pitch += 1

    # Find "same pitch, different rhythm" relations
    same_pitch_different_rhythm = 0
    for pc_key, indices in by_pitch.items():
        if len(indices) < 2 or len(pc_key) < 2:
            continue

        for i, idx_a in enumerate(indices):
            for idx_b in indices[i+1:]:
                rr_a = tuple(patterns[idx_a].get('rhythm_ratios', []))
                rr_b = tuple(patterns[idx_b].get('rhythm_ratios', []))

                if rr_a != rr_b and len(rr_a) == len(rr_b):
                    # Same pitch, different rhythm
                    rr_a_arr = np.array(rr_a, dtype=np.float32)
                    rr_b_arr = np.array(rr_b, dtype=np.float32)

                    # Try to find rhythm scale factor
                    k = find_scale_factor(rr_a_arr, rr_b_arr, RHYTHM_SCALE_FACTORS, tolerance)
                    if k is not None:
                        rel = MultiFactorRelation(
                            source_id=str(idx_a),
                            target_id=str(idx_b),
                            pitch_transform="identity",
                            rhythm_transform=FactorTransform(FactorType.RHYTHM, 'scale', k),
                        )
                        relations.append(rel)
                        same_pitch_different_rhythm += 1

    if verbose:
        print(f"    Same rhythm, different pitch: {same_rhythm_different_pitch}")
        print(f"    Same pitch, different rhythm: {same_pitch_different_rhythm}")
        print(f"    Total cross-factor relations: {len(relations)}")

    return relations


# =============================================================================
# MDL-BASED VOCABULARY SELECTION FOR FACTOR TRANSFORMS
# =============================================================================

def compute_factor_mdl_benefit(
    transform: FactorTransform,
    frequency: int,
    vocab_size: int,
    bits_per_factor: float = 4.0,
    bits_per_pattern: float = 15.0
) -> float:
    """
    Compute MDL benefit of adding a factor transform to vocabulary.

    Similar to pitch transform MDL, but for factor spaces.
    """
    if frequency < 2:
        return 0.0

    # Cost without: each derivation needs full pattern
    cost_without = frequency * bits_per_pattern

    # Cost with: define transform once, pointer for each use
    if transform.transform_type == 'identity':
        transform_cost = 1.0
    elif transform.transform_type == 'scale':
        transform_cost = bits_per_factor + math.log2(len(RHYTHM_SCALE_FACTORS) + 1)
    elif transform.transform_type == 'retrograde':
        transform_cost = bits_per_factor
    else:
        transform_cost = bits_per_factor * 2

    pointer_cost = math.log2(vocab_size + 1) if vocab_size > 0 else 1
    per_use_cost = pointer_cost + math.log2(max(vocab_size, 1))

    cost_with = transform_cost + frequency * per_use_cost

    return cost_without - cost_with


def select_factor_vocabulary_mdl(
    relations: Dict[FactorTransform, List[Tuple[str, str]]],
    min_frequency: int = 3,
    min_mdl_benefit: float = 0.0,
    verbose: bool = True
) -> Tuple[List[FactorTransform], Dict[str, FactorTransformStats]]:
    """
    Select factor transform vocabulary via MDL optimization.

    Uses greedy covering: iteratively add transform with highest MDL benefit.
    """
    if verbose:
        print(f"\n  Selecting factor vocabulary via MDL from {len(relations)} transforms...", flush=True)

    vocabulary = []
    stats = {}

    # Sort by frequency for greedy selection
    sorted_transforms = sorted(
        relations.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    covered_pairs = set()

    for transform, pairs in sorted_transforms:
        # Filter out already covered pairs
        uncovered = [(s, t) for s, t in pairs if (s, t) not in covered_pairs]
        freq = len(uncovered)

        if freq < min_frequency:
            continue

        benefit = compute_factor_mdl_benefit(transform, freq, len(vocabulary) + 1)

        if benefit > min_mdl_benefit:
            vocabulary.append(transform)

            # Mark pairs as covered
            for pair in uncovered:
                covered_pairs.add(pair)

            stats[str(transform)] = FactorTransformStats(
                transform=transform,
                frequency=freq,
                mdl_benefit=benefit,
                example_pairs=uncovered[:5],
            )

            if verbose and len(vocabulary) <= 15:
                print(f"    +{transform}: {freq} uses, benefit={benefit:.1f} bits", flush=True)

    if verbose:
        total_benefit = sum(s.mdl_benefit for s in stats.values())
        print(f"  Selected {len(vocabulary)} factor transforms, total benefit={total_benefit:.1f} bits")

    return vocabulary, stats


# =============================================================================
# UNIFIED MULTI-FACTOR DISCOVERY PIPELINE
# =============================================================================

def run_multi_factor_discovery(
    patterns: List[Dict],
    device: str = 'cuda',
    tolerance: float = 0.1,
    min_frequency: int = 3,
    verbose: bool = True
) -> Dict:
    """
    Run full multi-factor transform discovery pipeline.

    Discovers transforms in all factor spaces:
    - Rhythm (τ): augmentation, diminution, rhythm retrograde
    - Velocity (v): dynamics scaling
    - Duration (d): staccato/legato variations
    - Cross-factor: reharmonization, rhythmic variation

    Args:
        patterns: List of pattern dicts
        device: PyTorch device
        tolerance: Matching tolerance
        min_frequency: Minimum frequency for vocabulary selection
        verbose: Print progress

    Returns:
        Dict with discovered transforms and statistics
    """
    if verbose:
        print("=" * 60)
        print("MULTI-FACTOR TRANSFORM DISCOVERY")
        print("=" * 60)
        print(f"Input: {len(patterns)} patterns")

    # Discover transforms in each factor space
    rhythm_relations = discover_rhythm_transforms_gpu(patterns, device, tolerance, verbose)
    velocity_relations = discover_velocity_transforms_gpu(patterns, device, tolerance * 1.5, verbose)
    duration_relations = discover_duration_transforms_gpu(patterns, device, tolerance * 1.5, verbose)

    # Discover cross-factor relations
    cross_factor = discover_cross_factor_relations(patterns, device, tolerance, verbose)

    # Combine all factor relations
    all_relations = {}
    all_relations.update(rhythm_relations)
    all_relations.update(velocity_relations)
    all_relations.update(duration_relations)

    # Select vocabulary via MDL
    if verbose:
        print("\n" + "-" * 40)
    vocabulary, stats = select_factor_vocabulary_mdl(
        all_relations,
        min_frequency=min_frequency,
        verbose=verbose
    )

    # Compute statistics
    result = {
        'vocabulary': [str(t) for t in vocabulary],
        'vocabulary_objects': vocabulary,
        'stats': {k: {
            'transform': str(v.transform),
            'frequency': v.frequency,
            'mdl_benefit': v.mdl_benefit,
        } for k, v in stats.items()},

        # Per-factor statistics
        'rhythm_transforms': len(rhythm_relations),
        'velocity_transforms': len(velocity_relations),
        'duration_transforms': len(duration_relations),
        'cross_factor_relations': len(cross_factor),

        # Raw relations for further analysis
        'rhythm_relations': {str(k): v for k, v in rhythm_relations.items()},
        'velocity_relations': {str(k): v for k, v in velocity_relations.items()},
        'duration_relations': {str(k): v for k, v in duration_relations.items()},
    }

    if verbose:
        print("\n" + "=" * 60)
        print("MULTI-FACTOR DISCOVERY COMPLETE")
        print("=" * 60)
        print(f"  Rhythm (τ) transforms: {len(rhythm_relations)}")
        print(f"  Velocity (v) transforms: {len(velocity_relations)}")
        print(f"  Duration (d) transforms: {len(duration_relations)}")
        print(f"  Cross-factor relations: {len(cross_factor)}")
        print(f"  Selected vocabulary: {len(vocabulary)}")

        if vocabulary:
            print("\n  Top factor transforms by frequency:")
            for t in vocabulary[:10]:
                s = stats.get(str(t))
                if s:
                    print(f"    {t}: {s.frequency} uses")

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Multi-Factor Transform Discovery')
    parser.add_argument('checkpoint', help='Path to checkpoint NPZ file')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--tolerance', type=float, default=0.1, help='Matching tolerance')
    parser.add_argument('--min-freq', type=int, default=3, help='Minimum frequency')
    args = parser.parse_args()

    # Load patterns from checkpoint
    import numpy as np
    ckpt = np.load(args.checkpoint, allow_pickle=True)

    # Try to load patterns from JSON file or inline
    patterns_file = ckpt.get('patterns_json_file', [None])[0]
    if patterns_file:
        from pathlib import Path
        patterns_path = Path(args.checkpoint).parent / patterns_file
        with open(patterns_path) as f:
            patterns_raw = json.load(f)
    else:
        patterns_raw = json.loads(str(ckpt['patterns_json'][0]))

    # Convert to list of dicts
    if isinstance(patterns_raw, dict):
        patterns = list(patterns_raw.values())
    else:
        patterns = patterns_raw

    # Run discovery
    result = run_multi_factor_discovery(
        patterns,
        device=args.device,
        tolerance=args.tolerance,
        min_frequency=args.min_freq,
    )

    # Save results
    if args.output:
        # Remove non-serializable objects
        output_data = {k: v for k, v in result.items() if k != 'vocabulary_objects'}
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")
