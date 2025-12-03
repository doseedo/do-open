"""
Vocabulary-First MDL Discovery

The Lewinian approach: find transform vocabulary that minimizes
total corpus description length, rather than greedy per-object assignment.

Key insight: Don't ask "What explains this one object?"
Ask: "What transform vocabulary minimizes total corpus description length?"

This requires searching for frequent compound transforms FIRST,
then assigning objects to them - not the reverse.

Architecture:
  Phase 1: Mine candidate compounds (NO FAISS - small groups)
  Phase 2: MDL Selection (rank and add to vocabulary)
  Phase 3: Greedy Fill (USES FAISS for remaining)

Author: Vocabulary-First MDL Implementation
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import math
from itertools import combinations, product

# Import existing transform library - try both paths
try:
    from core.numpy_transforms import NumpyTransformLibrary
    _transform_lib = NumpyTransformLibrary()
except ImportError:
    # Running from a different path - define local minimal implementation
    _transform_lib = None


def _apply_transform_numpy(batch: np.ndarray, name: str, amount: float) -> np.ndarray:
    """
    Local implementation of transform application.

    This duplicates core.numpy_transforms logic but avoids import issues.
    Args:
        batch: (B, T, F) or (T, F) array
        name: Transform name
        amount: Transform amount
    Returns:
        Transformed array with same shape
    """
    # Use existing library if available
    if _transform_lib is not None:
        return _transform_lib.apply_transform(batch, name, amount)

    # Ensure 3D: (B, T, F)
    needs_squeeze = False
    if batch.ndim == 2:
        batch = np.expand_dims(batch, 0)
        needs_squeeze = True

    B, T, F = batch.shape
    # Assume first 128 dims are pitch
    PITCH_DIM = 128

    result = batch.copy()

    if name == 'transpose_semitone':
        shift = int(amount)
        pitch_part = batch[:, :, :PITCH_DIM]
        rest = batch[:, :, PITCH_DIM:]
        # Roll pitch dimension
        shifted = np.roll(pitch_part, shift, axis=2)
        # Zero out wrapped notes
        if shift > 0:
            shifted[:, :, :shift] = 0
        elif shift < 0:
            shifted[:, :, shift:] = 0
        result = np.concatenate([shifted, rest], axis=2)

    elif name == 'time_shift':
        shift = int(amount)
        result = np.zeros_like(batch)
        if shift > 0 and shift < T:
            result[:, shift:, :] = batch[:, :-shift, :]
        elif shift < 0 and shift > -T:
            result[:, :shift, :] = batch[:, -shift:, :]

    elif name == 'velocity_scale':
        scale = float(amount)
        vel_idx = PITCH_DIM  # velocity is after pitch
        result = batch.copy()
        if F > vel_idx:
            result[:, :, vel_idx] = np.clip(batch[:, :, vel_idx] * scale, 0, 1)

    elif name == 'retrograde':
        result = batch[:, ::-1, :].copy()

    elif name == 'inversion':
        center = int(amount) if amount != 0 else 60
        pitch_part = batch[:, :, :PITCH_DIM]
        rest = batch[:, :, PITCH_DIM:]
        # Flip around center pitch
        inverted = np.flip(pitch_part, axis=2)
        result = np.concatenate([inverted, rest], axis=2)

    else:
        # Unknown transform - return unchanged
        result = batch

    if needs_squeeze:
        result = result[0]

    return result


@dataclass
class CompoundTransform:
    """A composition of primitive transforms."""
    transforms: List[Dict]  # List of {'name': str, 'amount': float}
    name: str = None  # Generated name like "transpose(-7)_time_shift(16)"

    def __post_init__(self):
        if self.name is None:
            parts = [f"{t['name']}({t['amount']})" for t in self.transforms]
            self.name = "_".join(parts)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    @property
    def depth(self):
        return len(self.transforms)


@dataclass
class CandidateMatch:
    """A potential compound transform match."""
    compound: CompoundTransform
    source: 'MusicalObject'
    target: 'MusicalObject'
    error: float


@dataclass
class VocabularyMDLResult:
    """Result from vocabulary-first MDL discovery."""
    vocabulary: Set[CompoundTransform]
    assignments: Dict['MusicalObject', Tuple['MusicalObject', CompoundTransform]]
    sources: Set['MusicalObject']
    stats: Dict


def enumerate_compounds(
    primitives: List[Dict],
    max_depth: int = 2,
    filter_redundant: bool = True
) -> List[CompoundTransform]:
    """
    Enumerate compound transforms up to given depth, with optional filtering.

    Args:
        primitives: List of {'name': str, 'amount': float}
        max_depth: Maximum composition depth (2 = two-step)
        filter_redundant: If True, skip mathematically redundant combinations

    Returns:
        List of CompoundTransform objects

    Filtering removes:
        - transpose(x) ∘ transpose(y) → use transpose(x+y) instead
        - time_shift(x) ∘ time_shift(y) → use time_shift(x+y) instead
        - velocity(x) ∘ velocity(y) → use velocity(x*y) instead
        - retrograde ∘ retrograde → identity
        - quantize ∘ quantize → idempotent

    This reduces ~552 depth-2 compounds to ~150-200, no information loss.
    """
    compounds = []

    # Depth 1: Single transforms (already in vocabulary)
    for p in primitives:
        compounds.append(CompoundTransform([p]))

    # Depth 2: Two-step compositions
    if max_depth >= 2:
        for p1, p2 in product(primitives, primitives):
            # Skip identity compositions (same transform twice often cancels)
            if p1['name'] == p2['name'] and p1['amount'] == -p2['amount']:
                continue

            if filter_redundant:
                # Skip combinations that equal an existing primitive
                n1, n2 = p1['name'], p2['name']

                # Same type: compose amounts instead
                if n1 == n2 and n1 in ('transpose_semitone', 'time_shift'):
                    continue

                # Double retrograde = identity
                if n1 == 'retrograde' and n2 == 'retrograde':
                    continue

                # Double quantize = idempotent
                if n1.startswith('quantize') and n2.startswith('quantize'):
                    continue

                # Double velocity_scale = just multiply (covered by primitives)
                if n1 == 'velocity_scale' and n2 == 'velocity_scale':
                    continue

            compounds.append(CompoundTransform([p1, p2]))

    # Depth 3: Three-step compositions (optional, expensive)
    if max_depth >= 3:
        for p1, p2, p3 in product(primitives, primitives, primitives):
            compounds.append(CompoundTransform([p1, p2, p3]))

    return compounds


def apply_compound_transform(
    tensor: np.ndarray,
    compound: CompoundTransform
) -> np.ndarray:
    """
    Apply a compound transform to a tensor.

    Uses local transform implementation for consistent behavior.

    Args:
        tensor: (T, F) numpy array - single object tensor
        compound: CompoundTransform with list of transforms

    Returns:
        transformed: (T, F) numpy array
    """
    # Add batch dimension: (T, F) -> (1, T, F)
    result = np.expand_dims(tensor, 0)

    for transform in compound.transforms:
        result = _apply_transform_numpy(
            result,
            transform['name'],
            transform['amount']
        )

    # Remove batch dimension: (1, T, F) -> (T, F)
    return result[0]


def compute_reconstruction_error(
    source: np.ndarray,
    target: np.ndarray
) -> float:
    """Compute MSE between transformed source and target."""
    if source.shape != target.shape:
        return float('inf')
    return np.mean((source - target) ** 2)


def group_objects_by_timestep(
    objects: List['MusicalObject'],
    scale: int = 16
) -> Dict[Tuple[str, int], List['MusicalObject']]:
    """
    Group objects by (piece_id, start_time).

    This enables cross-track comparison at the same musical moment.

    Args:
        objects: List of MusicalObject
        scale: Only include objects at this scale

    Returns:
        Dict[(piece_id, start_time) -> [objects at that timestep]]
    """
    groups = defaultdict(list)

    for obj in objects:
        obj_scale = obj.tensor.shape[0]
        if obj_scale == scale:
            key = (obj.piece_id, obj.start_time)
            groups[key].append(obj)

    return groups


def mine_compound_candidates(
    objects: List['MusicalObject'],
    primitives: List[Dict],
    max_error: float = 0.03,
    max_depth: int = 2,
    scale: int = 16,
    verbose: bool = True,
    early_stop_fraction: float = 0.1,
    min_early_matches: int = 2
) -> Dict[CompoundTransform, List[CandidateMatch]]:
    """
    Phase 1: Mine candidate compound transforms from cross-track pairs.

    BATCHED IMPLEMENTATION with PRE-ALLOCATED BUFFER to avoid 174GB allocations.

    Key insight: Look at objects at the SAME TIMESTEP across different tracks.
    This finds relationships like "brass = reed transposed down an octave".

    Args:
        objects: All musical objects
        primitives: Primitive transforms
        max_error: Maximum reconstruction error
        max_depth: Maximum compound depth
        scale: Scale to analyze (16 = 1 bar)
        verbose: Print progress
        early_stop_fraction: Fraction of pairs to check before pruning
        min_early_matches: Min matches in early fraction to continue

    Returns:
        Dict[compound -> [(source, target, error), ...]]
    """
    import time

    if verbose:
        print(f"\n{'='*70}", flush=True)
        print("PHASE 1: MINING COMPOUND CANDIDATES (BATCHED)", flush=True)
        print(f"{'='*70}", flush=True)

    # Generate all candidate compounds
    compounds = enumerate_compounds(primitives, max_depth)
    if verbose:
        print(f"  Generated {len(compounds)} candidate compounds (depth <= {max_depth})", flush=True)

    # Group objects by timestep
    timestep_groups = group_objects_by_timestep(objects, scale)
    if verbose:
        print(f"  Found {len(timestep_groups)} timestep groups at scale {scale}", flush=True)
        avg_group_size = np.mean([len(g) for g in timestep_groups.values()])
        print(f"  Average group size: {avg_group_size:.1f} objects (tracks)", flush=True)

    # Build all pairs (source, target) across all timesteps
    all_pairs = []
    for (piece_id, start_time), group_objects in timestep_groups.items():
        if len(group_objects) < 2:
            continue
        for obj_a, obj_b in combinations(group_objects, 2):
            if obj_a.track_id != obj_b.track_id:
                all_pairs.append((obj_a, obj_b))
                all_pairs.append((obj_b, obj_a))  # Both directions

    if verbose:
        print(f"  Total pairs to check: {len(all_pairs):,}", flush=True)

    if len(all_pairs) == 0:
        return {}

    # Stack all source and target tensors for batch processing
    # Shape: (num_pairs, T, F)
    source_tensors = np.stack([p[0].tensor for p in all_pairs], axis=0)
    target_tensors = np.stack([p[1].tensor for p in all_pairs], axis=0)

    # PRE-ALLOCATE BUFFER - reuse for each compound (avoids 174GB allocations)
    transform_buffer = np.empty_like(source_tensors)

    if verbose:
        print(f"  Batch shape: {source_tensors.shape}", flush=True)
        print(f"  Memory: {source_tensors.nbytes / 1e6:.1f} MB × 2 (source + buffer)", flush=True)

    # Early stopping setup
    early_stop_idx = max(100, int(len(all_pairs) * early_stop_fraction))

    # Mine candidates - iterate over compounds (not pairs!)
    candidates = defaultdict(list)
    total_matches = 0
    pruned_compounds = 0
    start_time = time.time()

    for compound_idx, compound in enumerate(compounds):
        # Skip single-step (already covered by primitives in final FAISS pass)
        if compound.depth == 1:
            continue

        # COPY INTO PRE-ALLOCATED BUFFER (not new allocation)
        np.copyto(transform_buffer, source_tensors)

        # Apply transforms on buffer
        transformed = transform_buffer
        for transform in compound.transforms:
            transformed = _transform_lib.apply_transform(
                transformed,
                transform['name'],
                transform['amount']
            )

        # Compute MSE for all pairs at once
        # errors: (B,) - one error per pair
        errors = np.mean((transformed - target_tensors) ** 2, axis=(1, 2))

        # Early stopping: check first N pairs
        early_matches = np.sum(errors[:early_stop_idx] < max_error)
        if early_matches < min_early_matches:
            pruned_compounds += 1
            continue

        # Find matches below threshold
        match_mask = errors < max_error
        match_indices = np.where(match_mask)[0]

        # Record matches
        for idx in match_indices:
            source_obj, target_obj = all_pairs[idx]
            candidates[compound].append(CandidateMatch(
                compound=compound,
                source=source_obj,
                target=target_obj,
                error=float(errors[idx])
            ))
            total_matches += 1

        # Progress update
        if verbose and (compound_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (compound_idx + 1) / elapsed
            remaining = (len(compounds) - compound_idx - 1) / rate
            print(f"    {compound_idx + 1}/{len(compounds)}: "
                  f"{total_matches:,} matches, {pruned_compounds} pruned, "
                  f"~{remaining:.0f}s remaining", flush=True)

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n  Phase 1 complete in {elapsed:.1f}s", flush=True)
        print(f"  Compounds pruned: {pruned_compounds}", flush=True)
        print(f"  Compounds with matches: {len(candidates)}", flush=True)
        print(f"  Total matches: {total_matches:,}", flush=True)

        # Show top candidates by frequency
        if candidates:
            top_compounds = sorted(candidates.items(), key=lambda x: len(x[1]), reverse=True)[:10]
            print(f"\n  Top 10 compounds by frequency:", flush=True)
            for compound, matches in top_compounds:
                print(f"    {compound.name}: {len(matches)} matches", flush=True)

    return candidates


def apply_transform_torch(tensor, transform: Dict, device):
    """
    Apply a single transform using PyTorch operations.

    tensor: (B, T, F) on GPU
    """
    import torch

    name = transform['name']
    amount = transform['amount']

    if name == 'transpose_semitone':
        # Pitch is one-hot encoded in first 128 dims
        shift = int(amount)
        pitch_part = tensor[:, :, :128]
        rest = tensor[:, :, 128:]

        # Roll pitch dimension (wraps, but close enough for matching)
        pitched_rolled = torch.roll(pitch_part, shifts=shift, dims=2)
        return torch.cat([pitched_rolled, rest], dim=2)

    elif name == 'time_shift':
        shift = int(amount)
        # Shift along time dimension with zero padding
        if shift > 0:
            padded = torch.zeros_like(tensor)
            padded[:, shift:, :] = tensor[:, :-shift, :]
        elif shift < 0:
            padded = torch.zeros_like(tensor)
            padded[:, :shift, :] = tensor[:, -shift:, :]
        else:
            padded = tensor
        return padded

    elif name == 'velocity_scale':
        # Velocity is typically at index 128
        result = tensor.clone()
        if tensor.shape[2] > 128:
            result[:, :, 128] = result[:, :, 128] * amount
        return result

    elif name == 'inversion':
        # Invert pitch around pivot - approximate by flipping
        pivot = int(amount)
        pitch_part = tensor[:, :, :128]
        rest = tensor[:, :, 128:]
        inverted = torch.flip(pitch_part, dims=[2])
        return torch.cat([inverted, rest], dim=2)

    elif name == 'retrograde':
        return torch.flip(tensor, dims=[1])

    else:
        # Unsupported transform - return unchanged
        return tensor


def mine_compound_candidates_gpu(
    objects: List['MusicalObject'],
    primitives: List[Dict],
    max_error: float = 0.03,
    max_depth: int = 2,
    scale: int = 16,
    verbose: bool = True,
    early_stop_fraction: float = 0.1,
    min_early_matches: int = 2
) -> Dict[CompoundTransform, List[CandidateMatch]]:
    """
    Phase 1: Mine candidate compound transforms using GPU acceleration.

    Uses PyTorch for batched transform application on GPU.
    ~10-30x faster than CPU version.

    Args:
        objects: All musical objects
        primitives: Primitive transforms
        max_error: Maximum reconstruction error
        max_depth: Maximum compound depth
        scale: Scale to analyze (16 = 1 bar)
        verbose: Print progress
        early_stop_fraction: Fraction of pairs to check before pruning
        min_early_matches: Min matches in early fraction to continue

    Returns:
        Dict[compound -> [(source, target, error), ...]]
    """
    import torch
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(f"\n{'='*70}", flush=True)
        print(f"PHASE 1: MINING COMPOUND CANDIDATES (GPU: {device})", flush=True)
        print(f"{'='*70}", flush=True)

    # Generate filtered compounds
    compounds = enumerate_compounds(primitives, max_depth, filter_redundant=True)
    depth2_compounds = [c for c in compounds if c.depth == 2]

    if verbose:
        print(f"  Generated {len(compounds)} compounds ({len(depth2_compounds)} depth-2)", flush=True)

    # Group objects by timestep
    timestep_groups = group_objects_by_timestep(objects, scale)

    # Build all pairs
    all_pairs = []
    for (piece_id, start_time), group_objects in timestep_groups.items():
        if len(group_objects) < 2:
            continue
        for obj_a, obj_b in combinations(group_objects, 2):
            if obj_a.track_id != obj_b.track_id:
                all_pairs.append((obj_a, obj_b))
                all_pairs.append((obj_b, obj_a))

    if len(all_pairs) == 0:
        return {}

    if verbose:
        print(f"  Total pairs: {len(all_pairs):,}", flush=True)

    # Stack tensors and move to GPU
    source_np = np.stack([p[0].tensor for p in all_pairs], axis=0)
    target_np = np.stack([p[1].tensor for p in all_pairs], axis=0)

    source_tensors = torch.from_numpy(source_np).float().to(device)
    target_tensors = torch.from_numpy(target_np).float().to(device)

    if verbose:
        mem_mb = source_tensors.element_size() * source_tensors.nelement() / 1e6
        print(f"  Batch shape: {source_tensors.shape}", flush=True)
        print(f"  GPU memory: {mem_mb:.1f} MB × 2", flush=True)

    # Early stopping setup
    early_stop_idx = max(100, int(len(all_pairs) * early_stop_fraction))

    candidates = defaultdict(list)
    total_matches = 0
    pruned_compounds = 0
    start_time = time.time()

    for compound_idx, compound in enumerate(depth2_compounds):
        # Apply compound on GPU
        transformed = source_tensors.clone()
        for transform in compound.transforms:
            transformed = apply_transform_torch(transformed, transform, device)

        # NORMALIZED MSE: Only count error on active timesteps
        # This prevents sparse tensors from matching everything
        # Active = any pitch note is on (sum of pitch dims > 0)
        target_active = target_tensors[:, :, :128].sum(dim=2) > 0.1  # (B, T) bool
        transformed_active = transformed[:, :, :128].sum(dim=2) > 0.1
        either_active = target_active | transformed_active  # (B, T) bool

        # Per-timestep squared error
        per_step_error = ((transformed - target_tensors) ** 2).mean(dim=2)  # (B, T)

        # Mask and compute mean only over active timesteps
        active_counts = either_active.sum(dim=1).float()  # (B,)

        # Mark pairs with no active content - they can't match anything meaningful
        no_content_mask = (active_counts < 1)
        active_counts = torch.clamp(active_counts, min=1)  # Avoid div by zero

        masked_errors = per_step_error * either_active.float()  # Zero out inactive
        errors = masked_errors.sum(dim=1) / active_counts  # Normalized MSE

        # Set error to infinity for empty pairs (they can't match anything)
        errors[no_content_mask] = float('inf')

        # Debug: print stats for first compound
        if compound_idx == 0 and verbose:
            valid_errors = errors[errors < float('inf')]
            print(f"  DEBUG: First compound error stats:", flush=True)
            print(f"    Total pairs: {len(errors)}", flush=True)
            print(f"    Empty pairs (inf): {no_content_mask.sum().item()}", flush=True)
            print(f"    Valid pairs: {len(valid_errors)}", flush=True)
            if len(valid_errors) > 0:
                print(f"    Errors range: [{valid_errors.min().item():.6f}, {valid_errors.max().item():.6f}]", flush=True)
                print(f"    Mean error: {valid_errors.mean().item():.6f}", flush=True)
                print(f"    Median error: {valid_errors.median().item():.6f}", flush=True)
                # Show histogram
                for thresh in [0.01, 0.03, 0.05, 0.1, 0.2]:
                    count = (valid_errors < thresh).sum().item()
                    print(f"    Matches < {thresh}: {count} ({100*count/len(valid_errors):.1f}%)", flush=True)

        # Early stopping check
        early_matches = (errors[:early_stop_idx] < max_error).sum().item()
        if early_matches < min_early_matches:
            pruned_compounds += 1
            continue

        # Find all matches
        match_mask = errors < max_error
        match_indices = torch.where(match_mask)[0].cpu().numpy()
        errors_cpu = errors.cpu().numpy()

        for idx in match_indices:
            source_obj, target_obj = all_pairs[idx]
            candidates[compound].append(CandidateMatch(
                compound=compound,
                source=source_obj,
                target=target_obj,
                error=float(errors_cpu[idx])
            ))
            total_matches += 1

        # Progress
        if verbose and (compound_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (compound_idx + 1) / elapsed
            remaining = (len(depth2_compounds) - compound_idx - 1) / rate
            print(f"    {compound_idx + 1}/{len(depth2_compounds)}: "
                  f"{total_matches:,} matches, {pruned_compounds} pruned, "
                  f"~{remaining:.0f}s remaining", flush=True)

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n  Phase 1 complete in {elapsed:.1f}s", flush=True)
        print(f"  Compounds pruned: {pruned_compounds}", flush=True)
        print(f"  Compounds with matches: {len(candidates)}", flush=True)
        print(f"  Total matches: {total_matches:,}", flush=True)

        if candidates:
            top_compounds = sorted(candidates.items(), key=lambda x: len(x[1]), reverse=True)[:10]
            print(f"\n  Top 10 compounds by frequency:", flush=True)
            for compound, matches in top_compounds:
                print(f"    {compound.name}: {len(matches)} matches", flush=True)

    return candidates


def compute_mdl_benefit(
    compound: CompoundTransform,
    frequency: int,
    vocab_size: int,
    bits_per_primitive: float = 5.0
) -> float:
    """
    Compute MDL benefit of adding compound to vocabulary.

    MDL calculation:
      Cost WITHOUT compound: frequency × (sum of primitive costs)
      Cost WITH compound: compound_cost + frequency × pointer_cost

    Benefit = cost_without - cost_with

    Args:
        compound: The compound transform
        frequency: How many times it's used
        vocab_size: Current vocabulary size
        bits_per_primitive: Bits to encode a primitive (~5 for 33 primitives)

    Returns:
        MDL benefit in bits (positive = should add)
    """
    # Cost without: each use needs all primitives
    cost_per_use_without = compound.depth * bits_per_primitive
    cost_without = frequency * cost_per_use_without

    # Cost with: define compound once, then pointer to it
    compound_definition_cost = compound.depth * bits_per_primitive + 2  # +2 for structure
    pointer_cost = math.log2(vocab_size + 1)  # Point into vocabulary
    cost_with = compound_definition_cost + frequency * pointer_cost

    return cost_without - cost_with


def compute_compression_ratio(
    compound: CompoundTransform,
    num_matches: int,
    exemplar_bits: float = 48.0,  # Typical tensor description length
    bits_per_primitive: float = 5.0
) -> float:
    """
    Compute compression ratio: coverage / (transform_cost + exemplar_cost)

    Higher is better - more coverage per bit of description.

    Args:
        compound: The compound transform
        num_matches: Number of objects explained
        exemplar_bits: Bits to describe one exemplar
        bits_per_primitive: Bits per primitive transform

    Returns:
        Compression ratio (higher = better)
    """
    if num_matches < 2:
        return 0.0

    transform_bits = compound.depth * bits_per_primitive
    if transform_bits == 0:
        transform_bits = 0.1  # Small cost for identity

    total_cost = transform_bits + exemplar_bits
    return num_matches / total_cost


def select_vocabulary_mdl(
    candidates: Dict[CompoundTransform, List[CandidateMatch]],
    primitives: List[Dict],
    min_frequency: int = 10,
    verbose: bool = True
) -> Tuple[Set[CompoundTransform], Dict['MusicalObject', Tuple['MusicalObject', CompoundTransform]]]:
    """
    Phase 2: COSIATEC-style greedy covering with MDL scoring.

    Key change from original: after each compound is selected, we REMOVE
    its targets from consideration and RE-SCORE remaining compounds.
    This prevents double-counting and finds the true compression.

    OPTIMIZATION: We cache the match lists from Phase 1 and just filter
    them per iteration (O(matches) filtering) instead of recomputing
    transforms (O(compounds × objects) GPU ops).

    Algorithm:
    1. Score all compounds by compression ratio on REMAINING objects
    2. Select best compound
    3. Assign its matches and REMOVE those targets
    4. Repeat until no beneficial compounds remain

    Args:
        candidates: Dict[compound -> matches] from Phase 1
        primitives: Primitive transforms
        min_frequency: Minimum frequency to consider
        verbose: Print progress

    Returns:
        (vocabulary, assignments)
    """
    if verbose:
        print(f"\n{'='*70}")
        print("PHASE 2: MDL VOCABULARY SELECTION (COSIATEC-style)")
        print(f"{'='*70}")

    # Initialize vocabulary with primitives
    vocabulary = set()
    for p in primitives:
        vocabulary.add(CompoundTransform([p]))

    assignments = {}  # target -> (source, compound)
    assigned_targets = set()

    # CACHE: Build index from target -> list of (compound, match)
    # This allows O(1) removal when targets are assigned
    target_to_matches = defaultdict(list)
    compound_matches = {}  # compound -> set of target objects (for fast counting)

    for compound, matches in candidates.items():
        if compound.depth < 2:  # Skip depth-1, already primitives
            continue
        compound_matches[compound] = set()
        for m in matches:
            target_to_matches[m.target].append((compound, m))
            compound_matches[compound].add(m.target)

    if verbose:
        print(f"  Candidate compounds: {len(compound_matches)}")
        total_matches = sum(len(targets) for targets in compound_matches.values())
        print(f"  Total UNIQUE TARGETS cached: {total_matches:,}")
        # Show top 5 compounds by unique target count
        sorted_compounds = sorted(compound_matches.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        for comp, targets in sorted_compounds:
            print(f"    {comp.name}: {len(targets)} unique targets")

    iteration = 0
    max_iterations = 100  # Safety limit

    while compound_matches and iteration < max_iterations:
        iteration += 1

        # Score all compounds on REMAINING (unassigned) targets
        # This is O(compounds) since we maintain target sets
        scores = []
        for compound, targets in compound_matches.items():
            # Count unassigned (set difference is O(min(len)))
            unassigned_targets = targets - assigned_targets
            num_unassigned = len(unassigned_targets)

            if num_unassigned < min_frequency:
                continue

            # Compression ratio scoring
            cr = compute_compression_ratio(compound, num_unassigned)

            # Also compute MDL benefit for comparison
            mdl = compute_mdl_benefit(compound, num_unassigned, len(vocabulary))

            scores.append((compound, num_unassigned, cr, mdl))

        if not scores:
            break

        # Select best by compression ratio
        scores.sort(key=lambda x: x[2], reverse=True)
        best_compound, best_count, best_cr, best_mdl = scores[0]

        # Check if beneficial (both CR > threshold and MDL > 0)
        if best_cr < 0.01 or best_mdl <= 0:
            if verbose:
                print(f"  Stopping: best CR={best_cr:.4f}, MDL={best_mdl:.1f}")
            break

        # Add to vocabulary
        vocabulary.add(best_compound)

        # Get the actual matches for this compound (need source info)
        # Use the cached set for consistency
        unassigned_targets = compound_matches[best_compound] - assigned_targets

        # DEBUG: Check consistency
        raw_matches = [m for m in candidates[best_compound] if m.target not in assigned_targets]
        if verbose:
            print(f"    DEBUG: cached_count={best_count}, unassigned_targets={len(unassigned_targets)}, raw_matches={len(raw_matches)}")

        # Build source lookup from matches (pick first/best source for each target)
        target_to_source = {}
        for m in candidates[best_compound]:
            if m.target in unassigned_targets and m.target not in target_to_source:
                target_to_source[m.target] = m.source

        # Assign all UNIQUE targets (not duplicated matches)
        for target in unassigned_targets:
            if target in target_to_source:
                assignments[target] = (target_to_source[target], best_compound)
                assigned_targets.add(target)

        # Remove this compound from working set
        del compound_matches[best_compound]

        if verbose:
            print(f"  + {best_compound.name}: {len(unassigned_targets)} uses, "
                  f"CR={best_cr:.3f}, MDL={best_mdl:.1f} bits")

    if verbose:
        print(f"\n  Iterations: {iteration}")
        print(f"  Compounds added: {len(vocabulary) - len(primitives)}")
        print(f"  Total vocabulary size: {len(vocabulary)}")
        print(f"  Objects assigned via compounds: {len(assignments)}")

    return vocabulary, assignments


def fill_greedy_with_faiss(
    objects: List['MusicalObject'],
    assignments: Dict['MusicalObject', Tuple['MusicalObject', CompoundTransform]],
    scale_indices: Dict,
    vocabulary: Set[CompoundTransform],
    max_error: float = 0.03,
    verbose: bool = True
) -> Dict['MusicalObject', Tuple['MusicalObject', CompoundTransform]]:
    """
    Phase 3: Fill remaining unassigned objects using FAISS.

    Uses existing FAISS infrastructure but now searches with
    the expanded vocabulary (including discovered compounds).

    Args:
        objects: All musical objects
        assignments: Current assignments from Phase 2
        scale_indices: FAISS indices by scale
        vocabulary: Expanded vocabulary with compounds
        max_error: Maximum reconstruction error
        verbose: Print progress

    Returns:
        Updated assignments dict
    """
    if verbose:
        print(f"\n{'='*70}")
        print("PHASE 3: GREEDY FILL WITH FAISS")
        print(f"{'='*70}")

    unassigned = [obj for obj in objects if obj not in assignments]
    if verbose:
        print(f"  Unassigned objects: {len(unassigned)}")
        print(f"  Vocabulary size: {len(vocabulary)}")

    # For now, use simple greedy search
    # In production, integrate with existing FAISS code

    new_assignments = 0
    for obj in unassigned:
        scale = obj.tensor.shape[0]
        if scale not in scale_indices:
            continue

        index_gpu, index_cpu, index_to_object, F = scale_indices[scale]
        index = index_gpu if index_gpu is not None else index_cpu

        best_error = float('inf')
        best_source = None
        best_compound = None

        # Try each compound in vocabulary
        for compound in vocabulary:
            # Apply compound and search
            try:
                transformed = apply_compound_transform(obj.tensor, compound)
                if transformed.shape[0] != scale:
                    continue  # Skip if transform changed length

                query = transformed.flatten().astype(np.float32).reshape(1, -1)
                distances, indices = index.search(query, k=1)

                error = distances[0, 0] / (scale * F)

                if error < best_error and error < max_error:
                    source_idx = indices[0, 0]
                    source = index_to_object[source_idx]

                    # Don't derive from self
                    if source != obj:
                        best_error = error
                        best_source = source
                        best_compound = compound
            except Exception:
                continue

        if best_source is not None:
            assignments[obj] = (best_source, best_compound)
            new_assignments += 1

    if verbose:
        print(f"  New assignments from FAISS: {new_assignments}")
        print(f"  Total assigned: {len(assignments)}")
        print(f"  Remaining sources: {len(objects) - len(assignments)}")

    return assignments


def run_vocabulary_first_mdl(
    objects: List['MusicalObject'],
    primitives: List[Dict],
    scale_indices: Dict = None,
    max_error: float = 0.03,
    max_depth: int = 2,
    min_frequency: int = 10,
    scale: int = 16,
    use_gpu: bool = True,
    verbose: bool = True
) -> VocabularyMDLResult:
    """
    Main entry point for vocabulary-first MDL discovery.

    Args:
        objects: All musical objects
        primitives: Primitive transforms
        scale_indices: FAISS indices (optional, for Phase 3)
        max_error: Maximum reconstruction error
        max_depth: Maximum compound depth
        min_frequency: Minimum frequency for MDL benefit
        scale: Scale to analyze in Phase 1
        use_gpu: Use GPU-accelerated mining (default: True)
        verbose: Print progress

    Returns:
        VocabularyMDLResult with vocabulary, assignments, sources, stats
    """
    if verbose:
        print(f"\n{'='*70}")
        print("VOCABULARY-FIRST MDL DISCOVERY")
        print(f"{'='*70}")
        print(f"  Objects: {len(objects)}")
        print(f"  Primitives: {len(primitives)}")
        print(f"  Max compound depth: {max_depth}")
        print(f"  Min frequency: {min_frequency}")
        print(f"  Max error: {max_error}")

    # Phase 1: Mine compound candidates (GPU or CPU)
    if use_gpu:
        import torch
        if torch.cuda.is_available():
            candidates = mine_compound_candidates_gpu(
                objects, primitives, max_error, max_depth, scale, verbose
            )
        else:
            if verbose:
                print("  Warning: GPU requested but CUDA not available, falling back to CPU")
            candidates = mine_compound_candidates(
                objects, primitives, max_error, max_depth, scale, verbose
            )
    else:
        candidates = mine_compound_candidates(
            objects, primitives, max_error, max_depth, scale, verbose
        )

    # Phase 2: MDL vocabulary selection
    vocabulary, assignments = select_vocabulary_mdl(
        candidates, primitives, min_frequency, verbose
    )

    # Phase 3: Greedy fill with FAISS (if indices provided)
    if scale_indices is not None:
        assignments = fill_greedy_with_faiss(
            objects, assignments, scale_indices, vocabulary, max_error, verbose
        )

    # Compute final statistics
    sources = {obj for obj in objects if obj not in assignments}

    # Count compound usage
    compound_usage = Counter()
    for target, (source, compound) in assignments.items():
        compound_usage[compound.name] += 1

    stats = {
        'total_objects': len(objects),
        'total_assigned': len(assignments),
        'total_sources': len(sources),
        'vocabulary_size': len(vocabulary),
        'compound_usage': dict(compound_usage.most_common(20)),
        'derivation_rate': len(assignments) / len(objects) if objects else 0,
    }

    if verbose:
        print(f"\n{'='*70}")
        print("VOCABULARY-FIRST MDL COMPLETE")
        print(f"{'='*70}")
        print(f"  Final vocabulary: {len(vocabulary)} transforms")
        print(f"  Objects assigned: {len(assignments)} ({stats['derivation_rate']*100:.1f}%)")
        print(f"  Sources remaining: {len(sources)}")
        print(f"\n  Top 10 compound usage:")
        for name, count in list(compound_usage.most_common(10)):
            print(f"    {name}: {count}")

    return VocabularyMDLResult(
        vocabulary=vocabulary,
        assignments=assignments,
        sources=sources,
        stats=stats
    )


# =============================================================================
# HYBRID APPROACH: FAISS pairs + Compound identification
# =============================================================================

def identify_compound_for_pair_gpu(
    source_tensors: 'torch.Tensor',
    target_tensors: 'torch.Tensor',
    compounds: List[CompoundTransform],
    max_error: float,
    device: 'torch.device'
) -> Tuple[List[Optional[CompoundTransform]], List[float]]:
    """
    Given batches of (source, target) pairs, identify the best compound for each.

    Args:
        source_tensors: (B, T, F) tensor of sources
        target_tensors: (B, T, F) tensor of targets
        compounds: List of compound transforms to try
        max_error: Maximum error to accept
        device: PyTorch device

    Returns:
        (best_compounds, best_errors) - lists of length B
    """
    import torch

    B = source_tensors.shape[0]
    best_errors = torch.full((B,), float('inf'), device=device)
    best_compound_idx = torch.full((B,), -1, dtype=torch.long, device=device)

    for c_idx, compound in enumerate(compounds):
        # Apply compound transform
        transformed = source_tensors.clone()
        for transform in compound.transforms:
            transformed = apply_transform_torch(transformed, transform, device)

        # Compute normalized error
        target_active = target_tensors[:, :, :128].sum(dim=2) > 0.1
        transformed_active = transformed[:, :, :128].sum(dim=2) > 0.1
        either_active = target_active | transformed_active

        per_step_error = ((transformed - target_tensors) ** 2).mean(dim=2)
        active_counts = either_active.sum(dim=1).float().clamp(min=1)
        masked_errors = per_step_error * either_active.float()
        errors = masked_errors.sum(dim=1) / active_counts

        # Update best
        improved = errors < best_errors
        best_errors = torch.where(improved, errors, best_errors)
        best_compound_idx = torch.where(improved, torch.tensor(c_idx, device=device), best_compound_idx)

    # Convert to lists
    best_errors_cpu = best_errors.cpu().numpy()
    best_idx_cpu = best_compound_idx.cpu().numpy()

    result_compounds = []
    result_errors = []
    for i in range(B):
        if best_errors_cpu[i] < max_error and best_idx_cpu[i] >= 0:
            result_compounds.append(compounds[best_idx_cpu[i]])
            result_errors.append(float(best_errors_cpu[i]))
        else:
            result_compounds.append(None)
            result_errors.append(float('inf'))

    return result_compounds, result_errors


def mine_compounds_from_faiss_pairs_gpu(
    faiss_pairs: List[Tuple['MusicalObject', 'MusicalObject', float]],
    primitives: List[Dict],
    max_error: float = 0.03,
    max_depth: int = 2,
    batch_size: int = 10000,
    verbose: bool = True
) -> Dict[CompoundTransform, List[CandidateMatch]]:
    """
    HYBRID Phase 1: Given FAISS-discovered pairs, identify which compound explains each.

    This is the key hybrid innovation:
    - FAISS finds candidate pairs (any timestep, cross-piece)
    - This function identifies WHAT compound transform explains each pair

    Args:
        faiss_pairs: List of (source, target, faiss_distance) from FAISS search
        primitives: Primitive transforms
        max_error: Maximum reconstruction error
        max_depth: Maximum compound depth
        batch_size: GPU batch size
        verbose: Print progress

    Returns:
        Dict[compound -> list of CandidateMatch]
    """
    import torch
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(f"\n{'='*70}", flush=True)
        print(f"HYBRID PHASE 1: IDENTIFY COMPOUNDS FOR FAISS PAIRS (GPU: {device})", flush=True)
        print(f"{'='*70}", flush=True)

    # Generate compounds
    compounds = enumerate_compounds(primitives, max_depth, filter_redundant=True)
    depth2_compounds = [c for c in compounds if c.depth >= 2]

    if verbose:
        print(f"  FAISS pairs to analyze: {len(faiss_pairs):,}", flush=True)
        print(f"  Compounds to test: {len(depth2_compounds)}", flush=True)

    candidates = defaultdict(list)
    total_identified = 0
    start_time = time.time()

    # Group pairs by scale (tensors must have same shape to stack)
    pairs_by_scale = defaultdict(list)
    for pair in faiss_pairs:
        scale = pair[0].tensor.shape[0]
        pairs_by_scale[scale].append(pair)

    if verbose:
        print(f"  Pairs by scale: {dict((s, len(p)) for s, p in pairs_by_scale.items())}", flush=True)

    total_batches = sum((len(pairs) + batch_size - 1) // batch_size for pairs in pairs_by_scale.values())
    batch_count = 0

    for scale, scale_pairs in pairs_by_scale.items():
        if not scale_pairs:
            continue

        num_batches = (len(scale_pairs) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(scale_pairs))
            batch_pairs = scale_pairs[batch_start:batch_end]

            # Stack tensors (all same scale now)
            source_np = np.stack([p[0].tensor for p in batch_pairs], axis=0)
            target_np = np.stack([p[1].tensor for p in batch_pairs], axis=0)

            source_tensors = torch.from_numpy(source_np).float().to(device)
            target_tensors = torch.from_numpy(target_np).float().to(device)

            # Identify best compound for each pair
            best_compounds, best_errors = identify_compound_for_pair_gpu(
                source_tensors, target_tensors, depth2_compounds, max_error, device
            )

            # Record matches
            for i, (compound, error) in enumerate(zip(best_compounds, best_errors)):
                if compound is not None:
                    source_obj, target_obj, _ = batch_pairs[i]
                    candidates[compound].append(CandidateMatch(
                        compound=compound,
                        source=source_obj,
                        target=target_obj,
                        error=error
                    ))
                    total_identified += 1

            batch_count += 1
            if verbose and batch_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = batch_count / elapsed
                remaining = (total_batches - batch_count) / rate
                print(f"    Batch {batch_count}/{total_batches} (scale {scale}): "
                      f"{total_identified:,} compound matches identified, "
                      f"~{remaining:.0f}s remaining", flush=True)

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n  Hybrid Phase 1 complete in {elapsed:.1f}s", flush=True)
        print(f"  Compounds with matches: {len(candidates)}", flush=True)
        print(f"  Total compound matches: {total_identified:,}", flush=True)

        if candidates:
            top_compounds = sorted(candidates.items(), key=lambda x: len(x[1]), reverse=True)[:10]
            print(f"\n  Top 10 compounds by frequency:", flush=True)
            for compound, matches in top_compounds:
                print(f"    {compound.name}: {len(matches)} matches", flush=True)

    return candidates


def run_vocabulary_first_hybrid(
    objects: List['MusicalObject'],
    primitives: List[Dict],
    scale_indices: Dict,
    max_error: float = 0.03,
    max_depth: int = 2,
    min_frequency: int = 10,
    k_neighbors: int = 5,
    verbose: bool = True
) -> VocabularyMDLResult:
    """
    HYBRID vocabulary-first MDL discovery.

    Uses FAISS to find candidate pairs, then identifies which compound
    explains each pair. This finds cross-time and cross-piece patterns
    that same-timestep search misses.

    Args:
        objects: All musical objects
        primitives: Primitive transforms
        scale_indices: FAISS indices by scale
        max_error: Maximum reconstruction error
        max_depth: Maximum compound depth
        min_frequency: Minimum frequency for vocabulary
        k_neighbors: Number of FAISS neighbors to check per object
        verbose: Print progress

    Returns:
        VocabularyMDLResult
    """
    import torch
    import time

    if verbose:
        print(f"\n{'='*70}", flush=True)
        print("HYBRID VOCABULARY-FIRST MDL DISCOVERY", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  Objects: {len(objects)}", flush=True)
        print(f"  Primitives: {len(primitives)}", flush=True)
        print(f"  Max compound depth: {max_depth}", flush=True)
        print(f"  k neighbors: {k_neighbors}", flush=True)
        print(f"  Max error: {max_error}", flush=True)

    start_time = time.time()

    # Step 1: Gather FAISS pairs (batched for speed)
    if verbose:
        print(f"\n  Gathering FAISS pairs...", flush=True)

    faiss_pairs = []
    for scale, scale_data in scale_indices.items():
        if len(scale_data) < 4:
            continue

        index_gpu, index_cpu, index_to_object, F = scale_data
        index = index_gpu if index_gpu is not None else index_cpu

        scale_objects = [o for o in objects if o.tensor.shape[0] == scale]
        if not scale_objects:
            continue

        # BATCHED FAISS search - much faster than per-object
        all_queries = np.stack([obj.tensor.flatten().astype(np.float32) for obj in scale_objects])
        all_distances, all_indices = index.search(all_queries, k=k_neighbors + 1)

        for obj_idx, obj in enumerate(scale_objects):
            for dist, target_idx in zip(all_distances[obj_idx], all_indices[obj_idx]):
                if target_idx < 0:
                    continue
                target = index_to_object[target_idx]
                if target != obj:  # Don't include self
                    # Only include same-scale pairs (they must be, but double-check)
                    if target.tensor.shape[0] != scale:
                        continue
                    # Normalize distance
                    error = dist / (scale * F) if F > 0 else dist
                    if error < max_error * 10:  # Loose filter, compound check is strict
                        faiss_pairs.append((obj, target, error))

        if verbose:
            print(f"    Scale {scale}: {len(scale_objects)} objects -> {len([p for p in faiss_pairs if p[0].tensor.shape[0] == scale])} pairs", flush=True)

    if verbose:
        print(f"  Found {len(faiss_pairs):,} FAISS pairs", flush=True)

    if not faiss_pairs:
        return VocabularyMDLResult(
            vocabulary=set(),
            assignments={},
            sources=set(objects),
            stats={'total_objects': len(objects), 'total_assigned': 0}
        )

    # Step 2: Identify compound for each pair
    candidates = mine_compounds_from_faiss_pairs_gpu(
        faiss_pairs, primitives, max_error, max_depth, verbose=verbose
    )

    # Step 3: MDL vocabulary selection (same as before)
    vocabulary, assignments = select_vocabulary_mdl(
        candidates, primitives, min_frequency, verbose
    )

    # Compute final statistics
    sources = {obj for obj in objects if obj not in assignments}
    compound_usage = Counter()
    for target, (source, compound) in assignments.items():
        compound_usage[compound.name] += 1

    stats = {
        'total_objects': len(objects),
        'total_assigned': len(assignments),
        'total_sources': len(sources),
        'vocabulary_size': len(vocabulary),
        'compound_usage': dict(compound_usage.most_common(20)),
        'derivation_rate': len(assignments) / len(objects) if objects else 0,
        'faiss_pairs_analyzed': len(faiss_pairs),
        'total_time': time.time() - start_time,
    }

    if verbose:
        print(f"\n{'='*70}", flush=True)
        print("HYBRID VOCABULARY-FIRST MDL COMPLETE", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  Final vocabulary: {len(vocabulary)} transforms", flush=True)
        print(f"  Objects assigned: {len(assignments)} ({stats['derivation_rate']*100:.1f}%)", flush=True)
        print(f"  Sources remaining: {len(sources)}", flush=True)
        print(f"  FAISS pairs analyzed: {len(faiss_pairs):,}", flush=True)
        print(f"  Total time: {stats['total_time']:.1f}s", flush=True)
        print(f"\n  Top 10 compound usage:", flush=True)
        for name, count in list(compound_usage.most_common(10)):
            print(f"    {name}: {count}", flush=True)

    return VocabularyMDLResult(
        vocabulary=vocabulary,
        assignments=assignments,
        sources=sources,
        stats=stats
    )
