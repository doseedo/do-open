"""
Dense Transform Table for O(1) Pattern Transform Lookup.

For pattern sets small enough to fit in GPU memory, we precompute
a dense NxN transform table where table[i, j] gives the transform
that maps pattern[i] to pattern[j].

Memory requirements:
- N patterns, 24 D24 transforms: NxNx1 bytes = N^2 bytes
- N=1000: ~1 MB
- N=5000: ~25 MB
- N=20000: ~400 MB (borderline for some GPUs)

For larger pattern sets, use sparse representation.

Author: Architecture Refactor - Dosedo v2
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.groups.d24_group import D24Group
from core.groups.rhythm_group import RhythmGroup


@dataclass
class DenseTransformTable:
    """
    Dense NxN transform lookup table.

    Stores which D24 transform maps pattern[i] to pattern[j].
    Value of -1 indicates no transform exists.

    For rhythm patterns, a separate table stores rhythm group transforms.
    """
    # Pitch class transforms (D24)
    pitch_transforms: np.ndarray  # [N, N] int8, values -1 to 23

    # Rhythm transforms
    rhythm_transforms: np.ndarray  # [N, N] int8, values -1 to 7

    # Pattern ID mapping
    pattern_ids: List[int]  # Index -> pattern ID

    # GPU versions (optional)
    pitch_transforms_gpu: 'torch.Tensor' = None
    rhythm_transforms_gpu: 'torch.Tensor' = None

    @property
    def n_patterns(self) -> int:
        return len(self.pattern_ids)

    def get_pitch_transform(self, i: int, j: int) -> int:
        """Get D24 transform that maps pattern i to pattern j, or -1 if none."""
        return int(self.pitch_transforms[i, j])

    def get_rhythm_transform(self, i: int, j: int) -> int:
        """Get rhythm transform that maps pattern i to pattern j, or -1 if none."""
        return int(self.rhythm_transforms[i, j])

    def to_gpu(self, device: str = 'cuda'):
        """Upload tables to GPU."""
        if not HAS_TORCH:
            return

        self.pitch_transforms_gpu = torch.tensor(
            self.pitch_transforms,
            device=device,
            dtype=torch.int8
        )
        self.rhythm_transforms_gpu = torch.tensor(
            self.rhythm_transforms,
            device=device,
            dtype=torch.int8
        )

    def get_all_related(self, pattern_idx: int) -> List[Tuple[int, int, str]]:
        """
        Get all patterns related to the given pattern by any transform.

        Returns:
            List of (target_idx, transform_id, transform_type)
        """
        related = []

        # Pitch transforms
        for j in range(self.n_patterns):
            t = self.pitch_transforms[pattern_idx, j]
            if t >= 0:
                related.append((j, int(t), 'd24'))

        # Rhythm transforms
        for j in range(self.n_patterns):
            t = self.rhythm_transforms[pattern_idx, j]
            if t >= 0:
                related.append((j, int(t), 'rhythm'))

        return related


def build_pattern_transform_table(
    patterns: Dict,
    device: str = 'cuda',
    verbose: bool = True
) -> DenseTransformTable:
    """
    Build dense transform table for all patterns.

    Args:
        patterns: Dict of pattern_id -> pattern data
            Each pattern should have:
            - 'pitch_class': array of pitch classes (0-11)
            - 'rhythm': binary onset array

        device: PyTorch device for GPU computation
        verbose: Print progress

    Returns:
        DenseTransformTable with precomputed transforms
    """
    if verbose:
        print(f"\n{'='*70}")
        print("BUILDING DENSE TRANSFORM TABLE")
        print(f"{'='*70}")

    pattern_ids = sorted(patterns.keys())
    N = len(pattern_ids)

    if verbose:
        print(f"  Patterns: {N}")
        memory_mb = 2 * N * N / (1024 * 1024)  # 2 tables, 1 byte each
        print(f"  Memory estimate: {memory_mb:.1f} MB")

    # Initialize tables
    pitch_transforms = np.full((N, N), -1, dtype=np.int8)
    rhythm_transforms = np.full((N, N), -1, dtype=np.int8)

    # Initialize groups
    d24 = D24Group(device=device)
    rhythm_group = RhythmGroup(device=device)

    # Extract pattern data
    pitch_data = []
    rhythm_data = []

    for pid in pattern_ids:
        p = patterns[pid]
        pc = p.get('pitch_class', p.get('pitches', []))
        if len(pc) > 0:
            pc = np.array(pc) % 12
        else:
            pc = np.array([], dtype=np.int32)
        pitch_data.append(pc)

        rhythm = p.get('rhythm', np.array([]))
        rhythm_data.append(np.array(rhythm))

    if verbose:
        print("\n  Computing pitch class transforms...")

    # Compute pitch transforms (D24)
    for i in range(N):
        if len(pitch_data[i]) == 0:
            continue

        for j in range(N):
            if i == j:
                pitch_transforms[i, j] = 0  # Identity
                continue

            if len(pitch_data[j]) != len(pitch_data[i]):
                continue

            # Check each D24 transform
            for t_id in range(24):
                transformed = d24.apply_to_pitch_class_array(t_id, pitch_data[i])
                if np.array_equal(transformed, pitch_data[j]):
                    pitch_transforms[i, j] = t_id
                    break

        if verbose and i % 100 == 0 and i > 0:
            print(f"    Processed {i}/{N} patterns...")

    if verbose:
        print("\n  Computing rhythm transforms...")

    # Compute rhythm transforms
    for i in range(N):
        if len(rhythm_data[i]) == 0:
            continue

        for j in range(N):
            if i == j:
                rhythm_transforms[i, j] = 0  # Identity
                continue

            # Only check exact length matches for now
            # (time scaling would change length)
            if len(rhythm_data[j]) != len(rhythm_data[i]):
                continue

            # Check retrograde
            if np.array_equal(rhythm_data[i][::-1], rhythm_data[j]):
                rhythm_transforms[i, j] = 1  # Retrograde

    # Count relations found
    pitch_relations = np.sum(pitch_transforms >= 0) - N  # Exclude diagonal
    rhythm_relations = np.sum(rhythm_transforms >= 0) - N

    if verbose:
        print(f"\n  Pitch transform relations found: {pitch_relations}")
        print(f"  Rhythm transform relations found: {rhythm_relations}")

    # Create table object
    table = DenseTransformTable(
        pitch_transforms=pitch_transforms,
        rhythm_transforms=rhythm_transforms,
        pattern_ids=pattern_ids
    )

    # Optionally upload to GPU
    if HAS_TORCH and torch.cuda.is_available():
        table.to_gpu(device)
        if verbose:
            print(f"  Uploaded to GPU ({device})")

    return table


def find_transforms_for_pattern_pair(
    table: DenseTransformTable,
    source_idx: int,
    target_idx: int
) -> List[Tuple[str, int]]:
    """
    Find all transforms that map source pattern to target pattern.

    Args:
        table: DenseTransformTable
        source_idx: Index of source pattern
        target_idx: Index of target pattern

    Returns:
        List of (transform_type, transform_id) tuples
    """
    transforms = []

    pitch_t = table.get_pitch_transform(source_idx, target_idx)
    if pitch_t >= 0:
        transforms.append(('d24', pitch_t))

    rhythm_t = table.get_rhythm_transform(source_idx, target_idx)
    if rhythm_t >= 0:
        transforms.append(('rhythm', rhythm_t))

    return transforms


def find_all_transforms_batch(
    table: DenseTransformTable,
    source_indices: np.ndarray,
    target_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch lookup of transforms for multiple pattern pairs.

    Args:
        table: DenseTransformTable
        source_indices: [N] source pattern indices
        target_indices: [N] target pattern indices

    Returns:
        (pitch_transforms, rhythm_transforms) each [N] with values -1 to 23/7
    """
    if HAS_TORCH and table.pitch_transforms_gpu is not None:
        # GPU path
        src = torch.tensor(source_indices, device=table.pitch_transforms_gpu.device)
        tgt = torch.tensor(target_indices, device=table.pitch_transforms_gpu.device)

        pitch_t = table.pitch_transforms_gpu[src, tgt].cpu().numpy()
        rhythm_t = table.rhythm_transforms_gpu[src, tgt].cpu().numpy()
    else:
        # CPU path
        pitch_t = table.pitch_transforms[source_indices, target_indices]
        rhythm_t = table.rhythm_transforms[source_indices, target_indices]

    return pitch_t, rhythm_t


# =============================================================================
# SPARSE ALTERNATIVE FOR LARGE PATTERN SETS
# =============================================================================

@dataclass
class SparseTransformIndex:
    """
    Sparse representation for large pattern sets.

    Instead of NxN table, stores only existing relationships.
    """
    # Map (source_idx, transform_id) -> List[target_idx]
    pitch_forward: Dict[Tuple[int, int], List[int]]

    # Map target_idx -> List[(source_idx, transform_id)]
    pitch_reverse: Dict[int, List[Tuple[int, int]]]

    # Same for rhythm
    rhythm_forward: Dict[Tuple[int, int], List[int]]
    rhythm_reverse: Dict[int, List[Tuple[int, int]]]

    pattern_ids: List[int]


def build_sparse_transform_index(
    patterns: Dict,
    device: str = 'cuda',
    verbose: bool = True
) -> SparseTransformIndex:
    """
    Build sparse transform index for large pattern sets.

    More memory efficient than dense table for N > 5000.
    O(1) lookup by source+transform, O(k) lookup by target.
    """
    if verbose:
        print(f"\n  Building sparse transform index...")

    # For now, return empty index
    # This would be the full implementation for very large pattern sets

    return SparseTransformIndex(
        pitch_forward={},
        pitch_reverse={},
        rhythm_forward={},
        rhythm_reverse={},
        pattern_ids=sorted(patterns.keys())
    )
