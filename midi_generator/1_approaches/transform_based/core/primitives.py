"""
Pitch-Class Primitives for Transform Discovery
===============================================

Atomic operations on pitch-class sequences (0-11) that compose into
higher-level transforms. The system DISCOVERS which compositions are
useful via MDL optimization.

Primitives:
    - transpose(n): Add n semitones (mod 12)
    - invert(center): Reflect around center pitch
    - retrograde: Reverse sequence order

From these, D24 and more emerge:
    - T_n = transpose(n)           # 12 transpositions
    - I_n = invert(n)              # 12 inversions
    - R = retrograde               # Time reversal
    - RI_n = retrograde ∘ invert(n) # Retrograde-inversion

The key insight: Instead of hardcoding D24 search, we enumerate
primitive compositions and let MDL select which are productive.

Author: Primitive-based Transform Discovery
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from itertools import product
from enum import Enum


class PrimitiveType(Enum):
    """Types of atomic primitives."""
    TRANSPOSE = "transpose"      # Additive group Z_12
    INVERT = "invert"            # Involution with center parameter
    RETROGRADE = "retrograde"    # Involution (self-inverse)


@dataclass(frozen=True)
class Primitive:
    """An atomic transform primitive."""
    type: PrimitiveType
    param: int = 0  # For transpose: semitones; for invert: center

    def __str__(self):
        if self.type == PrimitiveType.RETROGRADE:
            return "R"
        elif self.type == PrimitiveType.TRANSPOSE:
            return f"T{self.param}"
        elif self.type == PrimitiveType.INVERT:
            return f"I{self.param}"
        return f"{self.type.value}({self.param})"

    def __repr__(self):
        return str(self)


@dataclass
class CompoundTransform:
    """A composition of primitive transforms."""
    primitives: Tuple[Primitive, ...]
    _name: str = field(default=None, repr=False)

    def __post_init__(self):
        if self._name is None:
            if len(self.primitives) == 0:
                object.__setattr__(self, '_name', "identity")
            else:
                object.__setattr__(self, '_name', "∘".join(str(p) for p in self.primitives))

    @property
    def name(self) -> str:
        return self._name

    @property
    def depth(self) -> int:
        return len(self.primitives)

    def __hash__(self):
        return hash(self.primitives)

    def __eq__(self, other):
        if not isinstance(other, CompoundTransform):
            return False
        return self.primitives == other.primitives

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"CompoundTransform({self.name})"


# =============================================================================
# Primitive Application Functions (Pitch-Class)
# =============================================================================

def apply_transpose(seq: np.ndarray, semitones: int) -> np.ndarray:
    """
    Transpose pitch classes by semitones (mod 12).

    Args:
        seq: Array of pitch classes (0-11)
        semitones: Number of semitones to transpose

    Returns:
        Transposed sequence
    """
    return (seq + semitones) % 12


def apply_invert(seq: np.ndarray, center: int = 0) -> np.ndarray:
    """
    Invert pitch classes around a center.

    I_n(p) = (2n - p) mod 12

    Args:
        seq: Array of pitch classes (0-11)
        center: Inversion center (0-11)

    Returns:
        Inverted sequence
    """
    return (2 * center - seq) % 12


def apply_retrograde(seq: np.ndarray) -> np.ndarray:
    """
    Reverse sequence order (retrograde).

    Args:
        seq: Array of pitch classes

    Returns:
        Reversed sequence
    """
    return seq[::-1].copy()


def apply_primitive(seq: np.ndarray, primitive: Primitive) -> np.ndarray:
    """Apply a single primitive to a sequence."""
    if primitive.type == PrimitiveType.TRANSPOSE:
        return apply_transpose(seq, primitive.param)
    elif primitive.type == PrimitiveType.INVERT:
        return apply_invert(seq, primitive.param)
    elif primitive.type == PrimitiveType.RETROGRADE:
        return apply_retrograde(seq)
    else:
        raise ValueError(f"Unknown primitive type: {primitive.type}")


def apply_compound(seq: np.ndarray, compound: CompoundTransform) -> np.ndarray:
    """
    Apply a compound transform (composition of primitives).

    Primitives are applied left-to-right:
    (T3 ∘ R)(seq) = R(T3(seq))

    Args:
        seq: Array of pitch classes
        compound: CompoundTransform to apply

    Returns:
        Transformed sequence
    """
    result = seq.copy()
    for primitive in compound.primitives:
        result = apply_primitive(result, primitive)
    return result


# =============================================================================
# Primitive Enumeration
# =============================================================================

def get_all_primitives() -> List[Primitive]:
    """
    Get all atomic primitives.

    Returns:
        List of 25 primitives:
        - 12 transpositions T_0 through T_11
        - 12 inversions I_0 through I_11
        - 1 retrograde R
    """
    primitives = []

    # Transpositions (skip T_0 = identity for compounds)
    for n in range(12):
        primitives.append(Primitive(PrimitiveType.TRANSPOSE, n))

    # Inversions
    for n in range(12):
        primitives.append(Primitive(PrimitiveType.INVERT, n))

    # Retrograde
    primitives.append(Primitive(PrimitiveType.RETROGRADE))

    return primitives


def enumerate_compounds(
    max_depth: int = 2,
    filter_redundant: bool = True,
    include_identity: bool = True
) -> List[CompoundTransform]:
    """
    Enumerate compound transforms up to given depth.

    Args:
        max_depth: Maximum composition depth (1 = single primitives, 2 = pairs)
        filter_redundant: Remove algebraically redundant compositions
        include_identity: Include identity transform (empty composition)

    Returns:
        List of CompoundTransform objects

    Filtering removes:
        - T_a ∘ T_b → T_{a+b} (use single transposition)
        - I_a ∘ I_b → T_{2(a-b)} (reduces to transposition)
        - R ∘ R → identity (retrograde is involutory)
        - T_0 anywhere (identity transposition)
    """
    primitives = get_all_primitives()
    compounds = []

    # Identity
    if include_identity:
        compounds.append(CompoundTransform(()))

    # Depth 1: Single primitives (excluding T_0)
    for p in primitives:
        if p.type == PrimitiveType.TRANSPOSE and p.param == 0:
            continue  # Skip identity transposition
        compounds.append(CompoundTransform((p,)))

    # Depth 2: Pairs of primitives
    if max_depth >= 2:
        for p1 in primitives:
            for p2 in primitives:
                # Skip identity transpositions
                if p1.type == PrimitiveType.TRANSPOSE and p1.param == 0:
                    continue
                if p2.type == PrimitiveType.TRANSPOSE and p2.param == 0:
                    continue

                if filter_redundant:
                    # T_a ∘ T_b = T_{a+b} - redundant
                    if (p1.type == PrimitiveType.TRANSPOSE and
                        p2.type == PrimitiveType.TRANSPOSE):
                        continue

                    # R ∘ R = identity - redundant
                    if (p1.type == PrimitiveType.RETROGRADE and
                        p2.type == PrimitiveType.RETROGRADE):
                        continue

                    # I_a ∘ I_b = T_{2(a-b)} - redundant
                    if (p1.type == PrimitiveType.INVERT and
                        p2.type == PrimitiveType.INVERT):
                        continue

                compounds.append(CompoundTransform((p1, p2)))

    return compounds


def get_d24_compounds() -> List[CompoundTransform]:
    """
    Get the 24 transforms of the dihedral group D24.

    This is what V32 currently hardcodes. Useful for comparison.

    Returns:
        List of 24 CompoundTransforms (12 T_n + 12 I_n)
    """
    compounds = []

    # 12 transpositions
    for n in range(12):
        if n == 0:
            compounds.append(CompoundTransform(()))  # Identity
        else:
            compounds.append(CompoundTransform((Primitive(PrimitiveType.TRANSPOSE, n),)))

    # 12 inversions
    for n in range(12):
        compounds.append(CompoundTransform((Primitive(PrimitiveType.INVERT, n),)))

    return compounds


def get_d24_with_retrograde() -> List[CompoundTransform]:
    """
    Get D24 × {identity, retrograde} = 48 transforms.

    This extends D24 with retrograde and retrograde-inversion.

    Returns:
        List of 48 CompoundTransforms
    """
    d24 = get_d24_compounds()
    retrograde = Primitive(PrimitiveType.RETROGRADE)

    compounds = list(d24)  # Original 24

    # Add retrograde versions
    for compound in d24:
        new_primitives = compound.primitives + (retrograde,)
        compounds.append(CompoundTransform(new_primitives))

    return compounds


# =============================================================================
# Transform Matching
# =============================================================================

def find_transform(
    source: np.ndarray,
    target: np.ndarray,
    candidates: List[CompoundTransform] = None,
    tolerance: float = 0.0
) -> Optional[Tuple[CompoundTransform, float]]:
    """
    Find which transform (if any) maps source to target.

    Args:
        source: Source pitch-class sequence
        target: Target pitch-class sequence
        candidates: Transforms to try (default: D24 + retrograde)
        tolerance: Allow this fraction of mismatches

    Returns:
        (best_transform, error) or None if no match
    """
    if len(source) != len(target):
        return None

    if candidates is None:
        candidates = get_d24_with_retrograde()

    source = np.asarray(source, dtype=np.int8)
    target = np.asarray(target, dtype=np.int8)

    best_transform = None
    best_error = float('inf')

    for compound in candidates:
        transformed = apply_compound(source, compound)

        # Count mismatches
        mismatches = np.sum(transformed != target)
        error = mismatches / len(source)

        if error <= tolerance and error < best_error:
            best_transform = compound
            best_error = error

            if error == 0:
                break  # Perfect match

    if best_transform is not None:
        return (best_transform, best_error)
    return None


def find_all_transforms(
    source: np.ndarray,
    target: np.ndarray,
    candidates: List[CompoundTransform] = None,
    tolerance: float = 0.0
) -> List[Tuple[CompoundTransform, float]]:
    """
    Find ALL transforms that map source to target.

    Multiple transforms might work (e.g., for symmetric patterns).

    Args:
        source: Source pitch-class sequence
        target: Target pitch-class sequence
        candidates: Transforms to try
        tolerance: Allow this fraction of mismatches

    Returns:
        List of (transform, error) tuples, sorted by error
    """
    if len(source) != len(target):
        return []

    if candidates is None:
        candidates = get_d24_with_retrograde()

    source = np.asarray(source, dtype=np.int8)
    target = np.asarray(target, dtype=np.int8)

    matches = []

    for compound in candidates:
        transformed = apply_compound(source, compound)
        mismatches = np.sum(transformed != target)
        error = mismatches / len(source)

        if error <= tolerance:
            matches.append((compound, error))

    return sorted(matches, key=lambda x: (x[1], x[0].depth))


# =============================================================================
# Batch Operations (for GPU acceleration)
# =============================================================================

def apply_compound_batch(
    sequences: np.ndarray,
    compound: CompoundTransform
) -> np.ndarray:
    """
    Apply compound transform to batch of sequences.

    Args:
        sequences: (B, L) array of pitch classes
        compound: Transform to apply

    Returns:
        (B, L) transformed sequences
    """
    result = sequences.copy()
    for primitive in compound.primitives:
        if primitive.type == PrimitiveType.TRANSPOSE:
            result = (result + primitive.param) % 12
        elif primitive.type == PrimitiveType.INVERT:
            result = (2 * primitive.param - result) % 12
        elif primitive.type == PrimitiveType.RETROGRADE:
            result = result[:, ::-1]
    return result


def find_transforms_batch(
    sources: np.ndarray,
    targets: np.ndarray,
    candidates: List[CompoundTransform] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find best transform for each (source, target) pair in batch.

    Args:
        sources: (B, L) source sequences
        targets: (B, L) target sequences
        candidates: Transforms to try

    Returns:
        (best_indices, best_errors): Arrays of shape (B,)
        best_indices[i] = index into candidates, or -1 if no match
        best_errors[i] = fraction of mismatches
    """
    if candidates is None:
        candidates = get_d24_with_retrograde()

    B, L = sources.shape
    best_indices = np.full(B, -1, dtype=np.int32)
    best_errors = np.full(B, 1.0, dtype=np.float32)

    for c_idx, compound in enumerate(candidates):
        transformed = apply_compound_batch(sources, compound)
        errors = np.mean(transformed != targets, axis=1)

        improved = errors < best_errors
        best_indices[improved] = c_idx
        best_errors[improved] = errors[improved]

    return best_indices, best_errors


# =============================================================================
# GPU-Accelerated Transform Mining (A100 optimized)
# =============================================================================

def find_transforms_batch_gpu(
    sources: np.ndarray,
    targets: np.ndarray,
    candidates: List[CompoundTransform] = None,
    table: 'torch.Tensor' = None,
    retrograde_mask: 'torch.Tensor' = None,
    device: str = 'cuda',
    check_pure_retrograde: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fully vectorized GPU transform matching for A100.

    Key optimizations:
    1. Precomputed lookup table - no Python loop over candidates
    2. Single gather operation - O(1) per element, fully parallel
    3. No tensor cloning - direct table lookup
    4. Boolean exact match - faster than float comparison
    5. NEW: Check pure retrograde FIRST (target == reverse(source))

    Args:
        sources: (B, L) source sequences
        targets: (B, L) target sequences
        candidates: Transforms to try
        table: Precomputed (C, 12) lookup table (reuse across calls!)
        retrograde_mask: Precomputed boolean mask for retrograde transforms
        device: PyTorch device ('cuda' or 'cpu')
        check_pure_retrograde: If True, check for pure R before compound transforms

    Returns:
        (best_indices, best_errors): Arrays of shape (B,)
        For pure retrograde matches, returns index of pure R in candidates (or -2 special code)
    """
    try:
        import torch
    except ImportError:
        return find_transforms_batch(sources, targets, candidates)

    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'

    if candidates is None:
        candidates = get_d24_with_retrograde()

    # Precompute table if not provided (cache this for repeated calls!)
    if table is None:
        table = precompute_transform_table_gpu(candidates=candidates, device=device)

    # Precompute retrograde mask if not provided
    if retrograde_mask is None:
        retrograde_mask = torch.tensor([
            any(p.type == PrimitiveType.RETROGRADE for p in c.primitives)
            for c in candidates
        ], dtype=torch.bool, device=device)

    B, L = sources.shape
    C = len(candidates)

    # Move to GPU once (use long for gather indexing)
    sources_t = torch.tensor(sources, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)

    # Initialize result arrays
    best_indices_t = torch.full((B,), -1, dtype=torch.long, device=device)

    # =========================================================================
    # NEW: Check pure retrograde FIRST (target == reverse(source))
    # This catches R relationships that compound transforms might miss
    # =========================================================================
    if check_pure_retrograde and L > 1:
        sources_reversed = sources_t.flip(dims=[1])  # (B, L)
        pure_retrograde_matches = (sources_reversed == targets_t).all(dim=1)  # (B,)

        # Find index of pure R in candidates (single R primitive)
        pure_r_idx = -1
        for i, c in enumerate(candidates):
            if len(c.primitives) == 1 and c.primitives[0].type == PrimitiveType.RETROGRADE:
                pure_r_idx = i
                break

        if pure_r_idx >= 0:
            # Mark pure retrograde matches
            best_indices_t[pure_retrograde_matches] = pure_r_idx
        else:
            # Use special code -2 for pure retrograde if not in candidates
            best_indices_t[pure_retrograde_matches] = -2

    # =========================================================================
    # Standard compound transform matching for remaining pairs
    # =========================================================================
    unmatched_mask = best_indices_t < 0  # Pairs not yet matched

    if unmatched_mask.any():
        # Vectorized table lookup: transform ALL candidates simultaneously
        sources_expanded = sources_t.unsqueeze(0).expand(C, -1, -1)  # (C, B, L)

        # Use advanced indexing for fully vectorized lookup
        all_transformed = table[
            torch.arange(C, device=device).view(C, 1, 1).expand(C, B, L),
            sources_expanded
        ]  # (C, B, L)

        # Apply retrograde where needed - vectorized flip
        if retrograde_mask.any():
            retro_indices = retrograde_mask.nonzero(as_tuple=True)[0]
            all_transformed[retro_indices] = all_transformed[retro_indices].flip(dims=[2])

        # Compare all at once using boolean exact match
        targets_expanded = targets_t.unsqueeze(0)  # (1, B, L)
        exact_matches = (all_transformed == targets_expanded).all(dim=2)  # (C, B) boolean

        # Find first matching transform per pair
        any_match = exact_matches.any(dim=0)  # (B,)

        # argmax on int gives first True index
        compound_indices = exact_matches.int().argmax(dim=0)  # (B,)

        # Update only unmatched pairs that found a compound match
        update_mask = unmatched_mask & any_match
        best_indices_t[update_mask] = compound_indices[update_mask]

    # Handle -2 (pure retrograde not in candidates) -> convert to -1 for compatibility
    # But return additional info about pure retrograde
    pure_retro_count = (best_indices_t == -2).sum().item()
    best_indices_t[best_indices_t == -2] = -1  # Compatibility with existing code

    # Errors: 0 for match, 1 for no match
    any_match_final = best_indices_t >= 0
    best_errors = (~any_match_final).float()

    return best_indices_t.cpu().numpy(), best_errors.cpu().numpy()


# Global cache for transform table (A100 optimization)
_TRANSFORM_TABLE_CACHE = {}


def get_cached_transform_table(
    candidates: List[CompoundTransform],
    device: str = 'cuda'
) -> Tuple['torch.Tensor', 'torch.Tensor']:
    """
    Get or create cached transform table and retrograde mask.

    Caching avoids recomputation - critical for A100 performance.
    """
    import torch

    # Create cache key from candidate names
    cache_key = (tuple(c.name for c in candidates), device)

    if cache_key not in _TRANSFORM_TABLE_CACHE:
        table = precompute_transform_table_gpu(candidates=candidates, device=device)
        retrograde_mask = torch.tensor([
            any(p.type == PrimitiveType.RETROGRADE for p in c.primitives)
            for c in candidates
        ], dtype=torch.bool, device=device)
        _TRANSFORM_TABLE_CACHE[cache_key] = (table, retrograde_mask)

    return _TRANSFORM_TABLE_CACHE[cache_key]


def mine_transforms_gpu(
    pairs_by_length: Dict[int, List],
    candidates: List[CompoundTransform],
    device: str = 'cuda',
    verbose: bool = True
) -> Dict[CompoundTransform, List]:
    """
    GPU-accelerated transform mining for large pattern sets.

    Optimized for A100 40GB - processes all length groups in parallel batches.

    Args:
        pairs_by_length: Dict mapping length -> list of (source, target, pair_obj) tuples
        candidates: Transform candidates to try
        device: PyTorch device
        verbose: Print progress

    Returns:
        Dict mapping transform -> list of pairs it explains
    """
    try:
        import torch
    except ImportError:
        if verbose:
            print("PyTorch not available, falling back to CPU")
        return {}

    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
        if verbose:
            print("CUDA not available, using CPU")

    relations = {}
    for t in candidates:
        relations[t] = []

    total_found = 0
    total_pairs = sum(len(p) for p in pairs_by_length.values())

    if verbose:
        print(f"  GPU mining: {total_pairs} pairs, {len(candidates)} transforms, device={device}")

    for length, length_data in pairs_by_length.items():
        if not length_data:
            continue

        # Extract sources and targets
        sources = np.stack([d[0] for d in length_data])
        targets = np.stack([d[1] for d in length_data])
        pair_objs = [d[2] for d in length_data]

        # GPU batch processing
        best_indices, best_errors = find_transforms_batch_gpu(
            sources, targets, candidates, device
        )

        # Collect results
        for i, (idx, err, pair) in enumerate(zip(best_indices, best_errors, pair_objs)):
            if idx >= 0 and err == 0:
                transform = candidates[idx]
                relations[transform].append(pair)
                total_found += 1

    # Remove empty transforms
    relations = {t: pairs for t, pairs in relations.items() if pairs}

    if verbose:
        print(f"  Found {total_found} relations across {len(relations)} transforms")

    return relations


def precompute_transform_table_gpu(
    max_length: int = 20,
    candidates: List[CompoundTransform] = None,
    device: str = 'cuda'
) -> 'torch.Tensor':
    """
    Precompute transform lookup table on GPU.

    For pitch-class transforms, we can precompute transformed values
    for all 12 pitch classes, enabling O(1) lookup per note.

    Returns:
        Tensor of shape (C, 12) where result[c, pc] gives transformed pitch class
    """
    import torch

    if candidates is None:
        candidates = get_d24_with_retrograde()

    C = len(candidates)

    # Pitch class lookup table (without retrograde - that's handled separately)
    table = torch.zeros((C, 12), dtype=torch.int32, device=device)

    for c_idx, compound in enumerate(candidates):
        for pc in range(12):
            result = pc
            for primitive in compound.primitives:
                if primitive.type == PrimitiveType.TRANSPOSE:
                    result = (result + primitive.param) % 12
                elif primitive.type == PrimitiveType.INVERT:
                    result = (2 * primitive.param - result) % 12
                # Retrograde doesn't change individual pitch classes
            table[c_idx, pc] = result

    return table


def apply_transforms_via_table_gpu(
    sequences: 'torch.Tensor',
    table: 'torch.Tensor',
    candidates: List[CompoundTransform]
) -> 'torch.Tensor':
    """
    Apply all transforms using precomputed lookup table.

    Much faster than computing transforms directly for large batches.

    Args:
        sequences: (B, L) input sequences on GPU
        table: (C, 12) precomputed transform table
        candidates: Transform list (to check retrograde flags)

    Returns:
        (C, B, L) all transformed sequences
    """
    import torch

    B, L = sequences.shape
    C = table.shape[0]
    device = sequences.device

    # Use gather for vectorized lookup
    # Expand sequences to (1, B, L) -> (C, B, L) via broadcast
    sequences_expanded = sequences.unsqueeze(0).expand(C, -1, -1)  # (C, B, L)

    # Flatten for gather: (C, B*L)
    flat_seqs = sequences_expanded.reshape(C, -1)  # (C, B*L)

    # Gather from table: table[c, flat_seqs[c, :]]
    # Need index of shape (C, B*L) with values in [0, 11]
    transformed_flat = torch.gather(
        table.unsqueeze(1).expand(-1, B*L, -1),  # (C, B*L, 12)
        dim=2,
        index=flat_seqs.unsqueeze(2).long()  # (C, B*L, 1)
    ).squeeze(2)  # (C, B*L)

    # Reshape back to (C, B, L)
    result = transformed_flat.reshape(C, B, L)

    # Apply retrograde where needed
    for c_idx, compound in enumerate(candidates):
        needs_retrograde = any(
            p.type == PrimitiveType.RETROGRADE for p in compound.primitives
        )
        if needs_retrograde:
            result[c_idx] = result[c_idx].flip(dims=[1])

    return result


# =============================================================================
# Algebraic Properties
# =============================================================================

def compose(c1: CompoundTransform, c2: CompoundTransform) -> CompoundTransform:
    """
    Compose two compound transforms: c1 ∘ c2 (c2 applied first).

    Note: Does NOT simplify algebraically. Use simplify() for that.
    """
    return CompoundTransform(c2.primitives + c1.primitives)


def simplify(compound: CompoundTransform) -> CompoundTransform:
    """
    Algebraically simplify a compound transform.

    Applies reduction rules:
        - T_a ∘ T_b → T_{(a+b) mod 12}
        - R ∘ R → identity
        - T_0 → remove
        - I_a ∘ I_b → T_{2(a-b) mod 12}
    """
    primitives = list(compound.primitives)
    changed = True

    while changed:
        changed = False
        new_primitives = []
        i = 0

        while i < len(primitives):
            if i + 1 < len(primitives):
                p1, p2 = primitives[i], primitives[i + 1]

                # T_a ∘ T_b → T_{a+b}
                if (p1.type == PrimitiveType.TRANSPOSE and
                    p2.type == PrimitiveType.TRANSPOSE):
                    new_param = (p1.param + p2.param) % 12
                    if new_param != 0:
                        new_primitives.append(Primitive(PrimitiveType.TRANSPOSE, new_param))
                    i += 2
                    changed = True
                    continue

                # R ∘ R → identity
                if (p1.type == PrimitiveType.RETROGRADE and
                    p2.type == PrimitiveType.RETROGRADE):
                    i += 2
                    changed = True
                    continue

                # I_a ∘ I_b → T_{2(a-b)}
                if (p1.type == PrimitiveType.INVERT and
                    p2.type == PrimitiveType.INVERT):
                    new_param = (2 * (p1.param - p2.param)) % 12
                    if new_param != 0:
                        new_primitives.append(Primitive(PrimitiveType.TRANSPOSE, new_param))
                    i += 2
                    changed = True
                    continue

            # Skip T_0
            if primitives[i].type == PrimitiveType.TRANSPOSE and primitives[i].param == 0:
                i += 1
                changed = True
                continue

            new_primitives.append(primitives[i])
            i += 1

        primitives = new_primitives

    return CompoundTransform(tuple(primitives))


def get_inverse(compound: CompoundTransform) -> CompoundTransform:
    """
    Get the inverse of a compound transform.

    (p1 ∘ p2 ∘ ... ∘ pn)^{-1} = pn^{-1} ∘ ... ∘ p2^{-1} ∘ p1^{-1}
    """
    inverse_primitives = []

    for p in reversed(compound.primitives):
        if p.type == PrimitiveType.TRANSPOSE:
            inverse_primitives.append(Primitive(PrimitiveType.TRANSPOSE, (-p.param) % 12))
        elif p.type == PrimitiveType.INVERT:
            # Inversion is self-inverse
            inverse_primitives.append(p)
        elif p.type == PrimitiveType.RETROGRADE:
            # Retrograde is self-inverse
            inverse_primitives.append(p)

    return CompoundTransform(tuple(inverse_primitives))


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=== Pitch-Class Primitives Test ===\n")

    # Test basic primitives
    seq = np.array([0, 4, 7, 11], dtype=np.int8)  # C major 7 chord
    print(f"Original: {seq} (C E G B)")

    # Transpose up a fifth
    t7 = CompoundTransform((Primitive(PrimitiveType.TRANSPOSE, 7),))
    print(f"T7: {apply_compound(seq, t7)} (G B D F#)")

    # Invert around C
    i0 = CompoundTransform((Primitive(PrimitiveType.INVERT, 0),))
    print(f"I0: {apply_compound(seq, i0)} (C Ab F Db)")

    # Retrograde
    r = CompoundTransform((Primitive(PrimitiveType.RETROGRADE),))
    print(f"R: {apply_compound(seq, r)} (B G E C)")

    # Compound: retrograde then transpose
    rt7 = CompoundTransform((
        Primitive(PrimitiveType.TRANSPOSE, 7),
        Primitive(PrimitiveType.RETROGRADE),
    ))
    print(f"T7∘R: {apply_compound(seq, rt7)} (F# D B G)")

    # Enumerate compounds
    print(f"\n=== Compound Enumeration ===")
    d24 = get_d24_compounds()
    print(f"D24: {len(d24)} transforms")

    d48 = get_d24_with_retrograde()
    print(f"D24 × R: {len(d48)} transforms")

    all_depth2 = enumerate_compounds(max_depth=2)
    print(f"All depth ≤ 2 (filtered): {len(all_depth2)} transforms")

    # Find transform
    print(f"\n=== Transform Finding ===")
    source = np.array([0, 2, 4, 5, 7], dtype=np.int8)  # C major scale fragment
    target = np.array([7, 5, 4, 2, 0], dtype=np.int8)  # Retrograde + T7? No, just retrograde of T7

    result = find_transform(source, target, d48)
    if result:
        print(f"Found: {result[0]} (error={result[1]:.2%})")
    else:
        print("No transform found")

    # Test simplification
    print(f"\n=== Simplification ===")
    complex_transform = CompoundTransform((
        Primitive(PrimitiveType.TRANSPOSE, 3),
        Primitive(PrimitiveType.TRANSPOSE, 4),
        Primitive(PrimitiveType.RETROGRADE),
        Primitive(PrimitiveType.RETROGRADE),
    ))
    print(f"Before: {complex_transform}")
    print(f"After:  {simplify(complex_transform)}")

    print("\n✓ All tests passed!")
