"""
Primitive-Based MDL Transform Discovery
=======================================

Discovers productive transforms via MDL optimization.
Instead of hardcoding D24, this:

1. Enumerates primitive compositions (T, I, R and combinations)
2. Mines which compositions relate patterns in the corpus
3. Uses MDL to select vocabulary of productive transforms

This realizes the original vision:
    Primitives → Composition → Discovered Transforms
       ↓              ↓              ↓
    +1 semitone   T7 = +1⁷       "Perfect fifth is common"
    reverse       R              "Retrograde appears in fugues"
    invert        I_0            "Inversion around C"

Author: Primitive-based MDL Discovery
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import math
import json
from pathlib import Path

from core.primitives import (
    CompoundTransform,
    Primitive,
    PrimitiveType,
    enumerate_compounds,
    apply_compound,
    apply_compound_batch,
    find_transform,
    find_transforms_batch,
    find_transforms_batch_gpu,
    get_cached_transform_table,
    get_d24_compounds,
    get_d24_with_retrograde,
    simplify,
)


@dataclass
class PatternPair:
    """A pair of patterns with their transform relationship."""
    source_id: str
    target_id: str
    source_pc: np.ndarray
    target_pc: np.ndarray
    transform: Optional[CompoundTransform] = None
    error: float = 0.0


@dataclass
class TransformStats:
    """Statistics for a discovered transform."""
    transform: CompoundTransform
    frequency: int  # How many times it explains a pattern pair
    unique_sources: int  # Number of distinct source patterns
    mdl_benefit: float  # Compression benefit in bits
    example_pairs: List[PatternPair] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    """Result of primitive-based MDL discovery."""
    # Discovered vocabulary
    vocabulary: List[CompoundTransform]
    vocabulary_stats: Dict[str, TransformStats]

    # Coverage statistics
    total_pattern_pairs: int
    covered_pairs: int
    coverage_rate: float

    # Comparison with D24
    d24_coverage: float
    d48_coverage: float
    full_coverage: float

    # MDL metrics
    total_mdl_benefit: float
    bits_per_pattern: float


def extract_canonical_pairs(
    canonicals: List[Dict],
    min_length: int = 3,
    max_length: int = 20
) -> List[PatternPair]:
    """
    Extract pattern pairs from canonical patterns for transform mining.

    Args:
        canonicals: List of canonical pattern dicts with 'pitch_classes'
        min_length: Minimum pattern length
        max_length: Maximum pattern length

    Returns:
        List of PatternPair objects (all pairs of same-length patterns)
    """
    # Group by length
    by_length = defaultdict(list)
    for i, c in enumerate(canonicals):
        pc = c.get('pitch_classes', [])
        if min_length <= len(pc) <= max_length:
            by_length[len(pc)].append((str(i), np.array(pc, dtype=np.int8)))

    # Create pairs within each length group
    pairs = []
    for length, patterns in by_length.items():
        for i, (id_a, pc_a) in enumerate(patterns):
            for id_b, pc_b in patterns[i+1:]:
                pairs.append(PatternPair(
                    source_id=id_a,
                    target_id=id_b,
                    source_pc=pc_a,
                    target_pc=pc_b,
                ))

    return pairs


def mine_transform_relations_gpu(
    canonicals: List[Dict],
    candidates: List[CompoundTransform] = None,
    min_length: int = 3,
    max_length: int = 20,
    chunk_size: int = 100000,
    verbose: bool = True,
    device: str = 'cuda'
) -> Dict[CompoundTransform, List[PatternPair]]:
    """
    GPU-native transform relation mining - avoids O(n²) Python object creation.

    Instead of pre-creating PatternPair objects, this:
    1. Groups patterns by length
    2. Generates pair indices on GPU with torch.triu_indices
    3. Processes in chunks to avoid OOM
    4. Only creates PatternPair objects for successful matches

    Args:
        canonicals: List of canonical pattern dicts with 'pitch_classes'
        candidates: Transform candidates to try (default: depth-2 compounds)
        min_length: Minimum pattern length
        max_length: Maximum pattern length
        chunk_size: Number of pairs to process per GPU chunk
        verbose: Print progress
        device: PyTorch device

    Returns:
        Dict mapping transform -> list of pairs it explains
    """
    import torch

    if candidates is None:
        candidates = enumerate_compounds(max_depth=2)

    if verbose:
        print(f"GPU-native transform mining on {len(canonicals)} patterns...", flush=True)
        print(f"  Using {len(candidates)} transform candidates", flush=True)

    # Group patterns by length
    by_length = defaultdict(list)
    for i, c in enumerate(canonicals):
        pc = c.get('pitch_classes', [])
        if min_length <= len(pc) <= max_length:
            by_length[len(pc)].append((i, np.array(pc, dtype=np.int8)))

    # Count total pairs
    total_pairs = sum(len(pats) * (len(pats) - 1) // 2 for pats in by_length.values())
    if verbose:
        print(f"  Total pairs to process: {total_pairs:,}", flush=True)

    # Pre-cache transform table
    table, retrograde_mask = get_cached_transform_table(candidates, device)
    if verbose:
        print(f"  Transform table: {table.shape}", flush=True)

    relations = defaultdict(list)
    total_found = 0
    pairs_processed = 0

    for length, patterns in by_length.items():
        n = len(patterns)
        if n < 2:
            continue

        # Extract pattern data
        indices = [p[0] for p in patterns]
        pc_arrays = np.stack([p[1] for p in patterns])  # [n, length]

        # Generate all pair indices on GPU
        i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=device)
        n_pairs = i_idx.shape[0]

        if verbose and n_pairs > 10000:
            print(f"  Processing length {length}: {n_pairs:,} pairs from {n} patterns", flush=True)

        # Process in chunks
        for chunk_start in range(0, n_pairs, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_pairs)

            # Get chunk indices
            chunk_i = i_idx[chunk_start:chunk_end].cpu().numpy()
            chunk_j = j_idx[chunk_start:chunk_end].cpu().numpy()

            # Get source and target patterns for this chunk
            sources = pc_arrays[chunk_i]  # [chunk_size, length]
            targets = pc_arrays[chunk_j]  # [chunk_size, length]

            # Find best transforms for all pairs in chunk
            best_indices, best_errors = find_transforms_batch_gpu(
                sources, targets, candidates,
                table=table, retrograde_mask=retrograde_mask, device=device
            )

            # Record matches (only create PatternPair for successful matches)
            for k in range(len(chunk_i)):
                if best_indices[k] >= 0 and best_errors[k] == 0:
                    transform = candidates[best_indices[k]]
                    src_idx = indices[chunk_i[k]]
                    tgt_idx = indices[chunk_j[k]]

                    pair = PatternPair(
                        source_id=str(src_idx),
                        target_id=str(tgt_idx),
                        source_pc=pc_arrays[chunk_i[k]],
                        target_pc=pc_arrays[chunk_j[k]],
                        transform=transform,
                        error=0.0,
                    )
                    relations[transform].append(pair)
                    total_found += 1

            pairs_processed += chunk_end - chunk_start

            # Clear GPU cache periodically
            if chunk_start > 0 and chunk_start % (chunk_size * 10) == 0:
                torch.cuda.empty_cache()

    if verbose:
        print(f"  Found {total_found:,} transform relations across {len(relations)} transforms", flush=True)

    return relations


def mine_cross_track_pairs_gpu(
    pairs: List[Tuple[Dict, Dict]],
    candidates: List[CompoundTransform] = None,
    chunk_size: int = 50000,
    verbose: bool = True,
    device: str = 'cuda'
) -> Dict[CompoundTransform, List[PatternPair]]:
    """
    GPU-native mining for pre-specified cross-track pairs.

    Unlike mine_transform_relations_gpu which generates O(n²) pairs,
    this takes explicit pairs and processes them in batches on GPU.
    Avoids creating PatternPair objects until a match is found.

    Args:
        pairs: List of (dict1, dict2) tuples with 'pitch_classes' and 'pattern_id'
        candidates: Transform candidates to try
        chunk_size: Pairs per GPU batch
        verbose: Print progress
        device: PyTorch device

    Returns:
        Dict mapping transform -> list of PatternPair matches
    """
    import torch

    if not pairs:
        return defaultdict(list)

    if candidates is None:
        candidates = enumerate_compounds(max_depth=2)

    if verbose:
        print(f"  GPU mining {len(pairs):,} cross-track pairs...", flush=True)

    # Group by length for efficient batching
    by_length = defaultdict(list)
    for i, (d1, d2) in enumerate(pairs):
        pc1 = d1.get('pitch_classes', [])
        pc2 = d2.get('pitch_classes', [])
        if len(pc1) == len(pc2) and len(pc1) >= 3:
            by_length[len(pc1)].append((i, d1, d2))

    # Pre-cache transform table
    table, retrograde_mask = get_cached_transform_table(candidates, device)

    relations = defaultdict(list)
    total_found = 0

    for length, length_items in by_length.items():
        n = len(length_items)
        if n == 0:
            continue

        # Extract arrays without creating PatternPair objects
        indices = [item[0] for item in length_items]
        sources = np.array([item[1]['pitch_classes'] for item in length_items], dtype=np.int8)
        targets = np.array([item[2]['pitch_classes'] for item in length_items], dtype=np.int8)

        # Process in chunks
        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)
            chunk_sources = sources[chunk_start:chunk_end]
            chunk_targets = targets[chunk_start:chunk_end]

            # GPU batch transform finding
            best_indices, best_errors = find_transforms_batch_gpu(
                chunk_sources, chunk_targets, candidates,
                table=table, retrograde_mask=retrograde_mask, device=device
            )

            # Only create PatternPair for matches
            for k in range(len(chunk_sources)):
                if best_indices[k] >= 0 and best_errors[k] == 0:
                    transform = candidates[best_indices[k]]
                    orig_idx = chunk_start + k
                    d1 = length_items[orig_idx][1]
                    d2 = length_items[orig_idx][2]

                    pair = PatternPair(
                        source_id=str(d1.get('pattern_id', orig_idx)),
                        target_id=str(d2.get('pattern_id', orig_idx)),
                        source_pc=chunk_sources[k],
                        target_pc=chunk_targets[k],
                        transform=transform,
                        error=0.0,
                    )
                    relations[transform].append(pair)
                    total_found += 1

        # Clear cache periodically
        if length % 5 == 0:
            torch.cuda.empty_cache()

    if verbose:
        print(f"    Found {total_found:,} cross-track relations across {len(relations)} transforms", flush=True)

    return relations


def mine_transform_relations(
    pairs: List,  # Can be List[PatternPair] or List[Tuple[dict, dict]]
    candidates: List[CompoundTransform] = None,
    verbose: bool = True,
    device: str = 'cuda'
) -> Dict[CompoundTransform, List[PatternPair]]:
    """
    Mine which transforms relate pattern pairs.

    Uses GPU acceleration when available for A100 optimization.
    Supports both PatternPair objects and (dict, dict) tuples.

    Args:
        pairs: Pattern pairs to analyze (PatternPair objects or (dict, dict) tuples)
        candidates: Transform candidates to try (default: depth-2 compounds)
        verbose: Print progress
        device: PyTorch device ('cuda' for GPU, 'cpu' for fallback)

    Returns:
        Dict mapping transform -> list of pairs it explains
    """
    if candidates is None:
        candidates = enumerate_compounds(max_depth=2)

    if verbose:
        print(f"Mining transforms for {len(pairs)} pairs using {len(candidates)} candidates...", flush=True)

    # Check for GPU availability
    use_gpu = device == 'cuda'
    try:
        import torch
        if not torch.cuda.is_available():
            use_gpu = False
            if verbose:
                print("  CUDA not available, using CPU", flush=True)
        elif verbose:
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    except ImportError:
        use_gpu = False
        if verbose:
            print("  PyTorch not available, using NumPy CPU", flush=True)

    relations = defaultdict(list)
    total_found = 0

    # Normalize pairs to PatternPair objects - handle both types
    def normalize_pair(pair):
        """Convert tuple (dict, dict) to PatternPair if needed."""
        if isinstance(pair, PatternPair):
            return pair
        elif isinstance(pair, tuple) and len(pair) == 2:
            d1, d2 = pair
            return PatternPair(
                source_id=str(d1.get('pattern_id', id(d1))),
                target_id=str(d2.get('pattern_id', id(d2))),
                source_pc=np.array(d1.get('pitch_classes', []), dtype=np.int8),
                target_pc=np.array(d2.get('pitch_classes', []), dtype=np.int8),
            )
        else:
            raise ValueError(f"Unknown pair type: {type(pair)}")

    # Group pairs by length for batch processing
    pairs_by_length = defaultdict(list)
    for pair in pairs:
        normalized = normalize_pair(pair)
        if len(normalized.source_pc) > 0:  # Skip empty patterns
            pairs_by_length[len(normalized.source_pc)].append(normalized)

    # Pre-cache transform table for GPU efficiency (A100 optimization)
    # This avoids recomputing the table for each length group
    table = None
    retrograde_mask = None
    if use_gpu:
        table, retrograde_mask = get_cached_transform_table(candidates, device)
        if verbose:
            print(f"  Cached transform table: {table.shape}", flush=True)

    for length, length_pairs in pairs_by_length.items():
        if not length_pairs:
            continue

        # Stack for batch processing
        sources = np.stack([p.source_pc for p in length_pairs])
        targets = np.stack([p.target_pc for p in length_pairs])

        # Find best transform for each pair (GPU or CPU)
        if use_gpu:
            best_indices, best_errors = find_transforms_batch_gpu(
                sources, targets, candidates,
                table=table, retrograde_mask=retrograde_mask, device=device
            )
        else:
            best_indices, best_errors = find_transforms_batch(sources, targets, candidates)

        for i, (pair, idx, err) in enumerate(zip(length_pairs, best_indices, best_errors)):
            if idx >= 0 and err == 0:  # Perfect match only
                transform = candidates[idx]
                pair.transform = transform
                pair.error = err
                relations[transform].append(pair)
                total_found += 1

    if verbose:
        print(f"Found {total_found} transform relations across {len(relations)} unique transforms", flush=True)

    return relations


def compute_mdl_benefit(
    transform: CompoundTransform,
    frequency: int,
    vocab_size: int,
    bits_per_primitive: float = 5.0,  # ~log2(25 primitives)
    bits_per_pattern: float = 20.0    # Cost to store a pattern
) -> float:
    """
    Compute MDL benefit of adding transform to vocabulary.

    MDL calculation:
      Cost WITHOUT: frequency × bits_per_pattern (store each target separately)
      Cost WITH: transform_cost + frequency × pointer_cost

    Benefit = cost_without - cost_with

    Args:
        transform: The compound transform
        frequency: How many times it's used
        vocab_size: Current vocabulary size
        bits_per_primitive: Bits to encode a primitive
        bits_per_pattern: Bits to encode a pattern

    Returns:
        MDL benefit in bits (positive = should add)
    """
    if frequency < 2:
        return 0.0  # Need at least 2 uses to benefit

    # Cost without: each use needs a full pattern
    cost_without = frequency * bits_per_pattern

    # Cost with: define transform once, then pointer for each use
    transform_cost = transform.depth * bits_per_primitive + 2  # +2 for structure
    pointer_cost = math.log2(vocab_size + 1) if vocab_size > 0 else 1
    # Each derivation: pointer to source + transform ID
    per_use_cost = pointer_cost + math.log2(max(vocab_size, 1))

    cost_with = transform_cost + frequency * per_use_cost

    return cost_without - cost_with


def select_vocabulary_mdl(
    relations: Dict[CompoundTransform, List[PatternPair]],
    min_frequency: int = 3,
    min_mdl_benefit: float = 0.0,
    verbose: bool = True
) -> Tuple[List[CompoundTransform], Dict[str, TransformStats]]:
    """
    Select transform vocabulary via MDL optimization.

    Uses greedy covering: iteratively add the transform with highest
    MDL benefit until no beneficial transforms remain.

    OPTIMIZED: Uses numpy boolean masks for O(1) coverage checks.

    Args:
        relations: Dict mapping transform -> pairs it explains
        min_frequency: Minimum frequency to consider
        min_mdl_benefit: Minimum MDL benefit to accept

    Returns:
        (vocabulary, stats_dict)
    """
    import numpy as np

    if verbose:
        print(f"\nSelecting vocabulary via MDL from {len(relations)} candidate transforms...", flush=True)

    # Flatten all pairs and assign integer IDs
    all_pairs = []
    pair_to_idx = {}

    for transform, pairs in relations.items():
        for p in pairs:
            key = (p.source_id, p.target_id)
            if key not in pair_to_idx:
                pair_to_idx[key] = len(all_pairs)
                all_pairs.append(p)

    n_pairs = len(all_pairs)
    if verbose:
        print(f"  MDL selection: {n_pairs} unique pairs, {len(relations)} transforms", flush=True)

    # Build coverage arrays: transform -> array of pair indices it covers
    transform_coverage = {}  # transform -> np.array of indices
    for transform, pairs in relations.items():
        indices = np.array([pair_to_idx[(p.source_id, p.target_id)] for p in pairs], dtype=np.int32)
        transform_coverage[transform] = indices

    # Boolean mask of covered pairs - numpy for O(1) indexing
    covered = np.zeros(n_pairs, dtype=bool)

    vocabulary = []
    stats = {}
    active_transforms = set(relations.keys())

    iteration = 0
    total_benefit = 0.0

    while active_transforms:
        iteration += 1

        best_transform = None
        best_benefit = min_mdl_benefit
        best_freq = 0
        best_uncovered_indices = None

        transforms_to_remove = []

        for transform in active_transforms:
            indices = transform_coverage[transform]

            # Count uncovered - numpy boolean indexing is fast
            uncovered_mask = ~covered[indices]
            freq = uncovered_mask.sum()

            if freq < min_frequency:
                transforms_to_remove.append(transform)
                continue

            benefit = compute_mdl_benefit(transform, freq, len(vocabulary) + 1)

            if benefit > best_benefit:
                best_benefit = benefit
                best_transform = transform
                best_freq = freq
                # Store only the uncovered indices
                best_uncovered_indices = indices[uncovered_mask]

        # Remove exhausted transforms
        for t in transforms_to_remove:
            active_transforms.discard(t)

        if best_transform is None:
            break

        # Add to vocabulary
        vocabulary.append(best_transform)
        total_benefit += best_benefit

        # Mark pairs as covered - numpy indexing
        covered[best_uncovered_indices] = True

        # Get example pairs for stats
        example_pairs = [all_pairs[i] for i in best_uncovered_indices[:5]]
        unique_sources = len(set(all_pairs[i].source_id for i in best_uncovered_indices))

        stats[best_transform.name] = TransformStats(
            transform=best_transform,
            frequency=best_freq,
            unique_sources=unique_sources,
            mdl_benefit=best_benefit,
            example_pairs=example_pairs
        )

        # Remove from candidates
        active_transforms.discard(best_transform)

        if verbose and iteration <= 20:
            print(f"  +{best_transform.name}: {best_freq} uses, benefit={best_benefit:.1f} bits", flush=True)

    if verbose:
        print(f"\nSelected {len(vocabulary)} transforms, total benefit={total_benefit:.1f} bits", flush=True)
        print(f"Covered {covered.sum()} of {n_pairs} pairs", flush=True)

    return vocabulary, stats


def compare_with_d24(
    pairs: List[PatternPair],
    discovered_vocab: List[CompoundTransform],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compare discovered vocabulary with hardcoded D24.

    Args:
        pairs: All pattern pairs
        discovered_vocab: MDL-selected vocabulary
        verbose: Print comparison

    Returns:
        Dict with coverage rates for each approach
    """
    d24 = get_d24_compounds()
    d48 = get_d24_with_retrograde()

    def count_coverage(candidates):
        covered = 0
        for pair in pairs:
            result = find_transform(pair.source_pc, pair.target_pc, candidates)
            if result and result[1] == 0:
                covered += 1
        return covered

    d24_covered = count_coverage(d24)
    d48_covered = count_coverage(d48)
    discovered_covered = count_coverage(discovered_vocab)

    total = len(pairs)
    results = {
        'd24_coverage': d24_covered / total if total > 0 else 0,
        'd48_coverage': d48_covered / total if total > 0 else 0,
        'discovered_coverage': discovered_covered / total if total > 0 else 0,
        'd24_count': 24,
        'd48_count': 48,
        'discovered_count': len(discovered_vocab),
    }

    if verbose:
        print(f"\n=== Coverage Comparison ===")
        print(f"D24 (hardcoded):      {d24_covered}/{total} = {results['d24_coverage']:.1%}")
        print(f"D48 (+retrograde):    {d48_covered}/{total} = {results['d48_coverage']:.1%}")
        print(f"Discovered (MDL):     {discovered_covered}/{total} = {results['discovered_coverage']:.1%}")
        print(f"Vocabulary sizes: D24={24}, D48={48}, Discovered={len(discovered_vocab)}")

    return results


def run_primitive_discovery(
    checkpoint_path: str,
    output_path: str = None,
    max_depth: int = 2,
    min_frequency: int = 3,
    verbose: bool = True
) -> DiscoveryResult:
    """
    Run full primitive-based MDL discovery pipeline.

    Args:
        checkpoint_path: Path to checkpoint with canonical patterns
        output_path: Optional path to save results
        max_depth: Maximum compound transform depth
        min_frequency: Minimum frequency for vocabulary selection
        verbose: Print progress

    Returns:
        DiscoveryResult
    """
    if verbose:
        print("=" * 70)
        print("PRIMITIVE-BASED MDL TRANSFORM DISCOVERY")
        print("=" * 70)

    # Load canonicals
    ckpt = np.load(checkpoint_path, allow_pickle=True)
    canonicals = json.loads(str(ckpt['canonical_patterns_json'][0]))

    if verbose:
        print(f"\nLoaded {len(canonicals)} canonical patterns")

    # Extract pairs
    pairs = extract_canonical_pairs(canonicals)
    if verbose:
        print(f"Created {len(pairs)} pattern pairs")

    # Generate candidates
    candidates = enumerate_compounds(max_depth=max_depth)
    if verbose:
        print(f"Generated {len(candidates)} compound candidates (depth ≤ {max_depth})")

    # Mine relations
    relations = mine_transform_relations(pairs, candidates, verbose)

    # Select vocabulary via MDL
    vocabulary, stats = select_vocabulary_mdl(relations, min_frequency, verbose=verbose)

    # Compare with D24
    comparison = compare_with_d24(pairs, vocabulary, verbose)

    # Compute total coverage
    covered_pairs = sum(s.frequency for s in stats.values())
    coverage_rate = covered_pairs / len(pairs) if pairs else 0

    # Compute MDL metrics
    total_mdl = sum(s.mdl_benefit for s in stats.values())
    bits_per = total_mdl / covered_pairs if covered_pairs > 0 else 0

    result = DiscoveryResult(
        vocabulary=vocabulary,
        vocabulary_stats=stats,
        total_pattern_pairs=len(pairs),
        covered_pairs=covered_pairs,
        coverage_rate=coverage_rate,
        d24_coverage=comparison['d24_coverage'],
        d48_coverage=comparison['d48_coverage'],
        full_coverage=comparison['discovered_coverage'],
        total_mdl_benefit=total_mdl,
        bits_per_pattern=bits_per,
    )

    # Save results
    if output_path:
        output = {
            'vocabulary': [t.name for t in vocabulary],
            'vocabulary_size': len(vocabulary),
            'stats': {
                name: {
                    'transform': s.transform.name,
                    'frequency': s.frequency,
                    'unique_sources': s.unique_sources,
                    'mdl_benefit': s.mdl_benefit,
                }
                for name, s in stats.items()
            },
            'total_pairs': len(pairs),
            'covered_pairs': covered_pairs,
            'coverage_rate': coverage_rate,
            'd24_coverage': comparison['d24_coverage'],
            'd48_coverage': comparison['d48_coverage'],
            'discovered_coverage': comparison['discovered_coverage'],
            'total_mdl_benefit': total_mdl,
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"\nResults saved to {output_path}")

    # Final summary
    if verbose:
        print(f"\n{'=' * 70}")
        print("DISCOVERY SUMMARY")
        print(f"{'=' * 70}")
        print(f"Discovered vocabulary: {len(vocabulary)} transforms")
        print(f"Coverage: {coverage_rate:.1%} ({covered_pairs}/{len(pairs)} pairs)")
        print(f"MDL benefit: {total_mdl:.1f} bits total, {bits_per:.2f} bits/pattern")

        print(f"\nTop 10 transforms by frequency:")
        for name, s in sorted(stats.items(), key=lambda x: x[1].frequency, reverse=True)[:10]:
            print(f"  {s.transform.name}: {s.frequency} uses, {s.mdl_benefit:.1f} bits saved")

        # Show novel transforms (not in D24)
        d24_names = {t.name for t in get_d24_compounds()}
        novel = [t for t in vocabulary if t.name not in d24_names]
        if novel:
            print(f"\nNovel transforms (not in D24):")
            for t in novel[:10]:
                s = stats.get(t.name)
                if s:
                    print(f"  {t.name}: {s.frequency} uses")

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Primitive-based MDL Transform Discovery')
    parser.add_argument('checkpoint', help='Path to checkpoint NPZ file')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--max-depth', type=int, default=2, help='Maximum compound depth')
    parser.add_argument('--min-freq', type=int, default=3, help='Minimum frequency')
    args = parser.parse_args()

    result = run_primitive_discovery(
        args.checkpoint,
        args.output,
        max_depth=args.max_depth,
        min_frequency=args.min_freq,
    )
