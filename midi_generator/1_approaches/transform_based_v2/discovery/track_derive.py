#!/usr/bin/env python3
"""
TrackDerive Discovery: Cross-Track Derivation for Arrangement Patterns
=======================================================================

GPU-OPTIMIZED VERSION (v2)

This module discovers per-occurrence cross-track derivation relationships
and adds them to the pattern graph. Unlike aggregate_orchestration_rules
which produces summary statistics, TrackDerive captures:

    "This trombone occurrence = this trumpet occurrence + T(-7)"

This is the heart of arrangement knowledge:
- Sax section = Brass melody + harmonized intervals
- A section strings = A section piano + octave doubling
- Bridge horn = Verse melody + transposition

Architecture (GPU-optimized):
    1. Find vertical slices via GPU grouping
    2. Extract ALL cross-track pairs from slices in batch
    3. GPU-batch transform matching (T0-T11, I0-I11, R, RT0-RT11)
    4. GPU-vectorized leader selection
    5. Build TrackDerive objects from GPU results

Author: TrackDerive Discovery System
"""

import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import time

# Import from parent package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.primitives import (
    TrackDerive,
    CompoundTransform,
    Primitive,
    PrimitiveType,
)


# ============================================================================
# GPU Transform Tables (precomputed for fast matching)
# ============================================================================

def build_transform_tables(device: str = 'cuda') -> Tuple[torch.Tensor, List[str]]:
    """
    Build GPU tensors for all D24 transforms + retrograde variants.

    Returns:
        transform_table: (48, 12) tensor where transform_table[t] shows how each
                        pitch class maps under transform t
        transform_names: List of transform names for lookup
    """
    transforms = []
    names = []

    # T0-T11: Transpositions
    for t in range(12):
        table = [(pc + t) % 12 for pc in range(12)]
        transforms.append(table)
        names.append(f'T{t}' if t > 0 else 'identity')

    # I0-I11: Inversions (reflect around axis)
    for axis in range(12):
        table = [(2 * axis - pc) % 12 for pc in range(12)]
        transforms.append(table)
        names.append(f'I{axis}')

    # R: Retrograde (handled separately - just reverses order)
    # RT0-RT11: Retrograde + transposition
    for t in range(12):
        table = [(pc + t) % 12 for pc in range(12)]
        transforms.append(table)
        names.append(f'RT{t}' if t > 0 else 'R')

    # RI0-RI11: Retrograde + inversion
    for axis in range(12):
        table = [(2 * axis - pc) % 12 for pc in range(12)]
        transforms.append(table)
        names.append(f'RI{axis}')

    transform_table = torch.tensor(transforms, dtype=torch.int8, device=device)
    return transform_table, names


# GM program number to instrument name (for logging)
GM_INSTRUMENTS = {
    0: 'Piano', 24: 'Nylon Guitar', 25: 'Steel Guitar', 26: 'Jazz Guitar',
    32: 'Acoustic Bass', 33: 'Electric Bass', 40: 'Violin', 41: 'Viola',
    42: 'Cello', 48: 'String Ensemble', 56: 'Trumpet', 57: 'Trombone',
    58: 'Tuba', 60: 'French Horn', 61: 'Brass Section', 64: 'Soprano Sax',
    65: 'Alto Sax', 66: 'Tenor Sax', 67: 'Baritone Sax', 68: 'Oboe',
    71: 'Clarinet', 73: 'Flute',
}


def instrument_name(gm_prog: int) -> str:
    """Get readable instrument name from GM program."""
    return GM_INSTRUMENTS.get(gm_prog, f"GM{gm_prog}")


@dataclass
class VerticalSlice:
    """A moment in time with multiple simultaneous pattern occurrences."""
    piece_id: str
    onset_time: int
    occurrences: List[dict]  # [{track_id, pattern_idx, instrument, pitch_offset}, ...]


def find_vertical_slices_gpu(
    patterns: List[Dict],
    track_instruments: Dict[Tuple[str, int], int] = None,
    device: str = 'cuda',
    verbose: bool = True
) -> List[VerticalSlice]:
    """
    Find all vertical slices (simultaneous patterns across tracks).

    GPU-accelerated grouping by (piece, onset_time).
    OPTIMIZED: Uses numpy for fast flattening, GPU for grouping.

    Args:
        patterns: List of pattern dicts with 'occurrences' field
        track_instruments: (piece_id, track_id) -> GM program mapping
        device: PyTorch device
        verbose: Print progress

    Returns:
        List of VerticalSlice objects
    """
    if not torch.cuda.is_available():
        device = 'cpu'

    # Count total occurrences first for pre-allocation
    total_occs = sum(len(p.get('occurrences', [])) for p in patterns)
    if total_occs < 2:
        return []

    if verbose:
        print(f"    Flattening {total_occs:,} occurrences...")

    # Pre-allocate numpy arrays for speed
    piece_hashes = np.zeros(total_occs, dtype=np.int64)
    track_ids = np.zeros(total_occs, dtype=np.int32)
    onset_times = np.zeros(total_occs, dtype=np.int64)
    pattern_indices = np.zeros(total_occs, dtype=np.int32)
    instruments = np.zeros(total_occs, dtype=np.int32)
    pitch_offsets = np.zeros(total_occs, dtype=np.int32)

    # Build piece_id -> hash mapping (avoid repeated hashing)
    piece_id_to_hash = {}

    idx = 0
    for p_idx, p in enumerate(patterns):
        for occ in p.get('occurrences', []):
            # Handle both dict and PatternOccurrence objects
            if hasattr(occ, 'piece_id'):
                piece_id = occ.piece_id
                track_id = int(occ.track_id)
                onset = int(occ.onset_time)
                pitch_offset = getattr(occ, 'pitch_offset', 0)
            else:
                piece_id = occ['piece_id']
                track_id = int(occ['track_id'])
                onset = int(occ['onset_time'])
                pitch_offset = occ.get('pitch_offset', 0)

            # Cache piece hash
            if piece_id not in piece_id_to_hash:
                piece_id_to_hash[piece_id] = hash(piece_id) % (2**31)

            # Get instrument (GM program)
            if track_instruments:
                instrument = track_instruments.get((piece_id, track_id), track_id)
            else:
                instrument = track_id

            piece_hashes[idx] = piece_id_to_hash[piece_id]
            track_ids[idx] = track_id
            onset_times[idx] = onset
            pattern_indices[idx] = p_idx
            instruments[idx] = instrument
            pitch_offsets[idx] = pitch_offset
            idx += 1

    # Store piece_ids separately (strings can't go in numpy efficiently)
    # We'll use piece_hash -> piece_id reverse lookup
    hash_to_piece_id = {v: k for k, v in piece_id_to_hash.items()}

    if verbose:
        print(f"    GPU grouping...")

    # Convert to tensors for GPU grouping
    piece_hashes_t = torch.from_numpy(piece_hashes).to(device)
    onsets_t = torch.from_numpy(onset_times).to(device)

    # Create slice keys (piece * large_const + onset)
    slice_keys = piece_hashes_t * 10000000 + onsets_t

    # Sort to group same-slice occurrences
    sorted_indices = torch.argsort(slice_keys)
    sorted_keys = slice_keys[sorted_indices]

    # Find slice boundaries
    key_changes = torch.cat([
        torch.tensor([True], device=device),
        sorted_keys[1:] != sorted_keys[:-1]
    ])
    slice_starts = torch.where(key_changes)[0]
    slice_ends = torch.cat([slice_starts[1:], torch.tensor([len(sorted_keys)], device=device)])
    slice_sizes = slice_ends - slice_starts

    # Only keep slices with 2+ patterns (vertical slices)
    valid_mask = slice_sizes >= 2
    valid_starts = slice_starts[valid_mask].cpu().numpy()
    valid_ends = slice_ends[valid_mask].cpu().numpy()
    sorted_indices_cpu = sorted_indices.cpu().numpy()

    if verbose:
        print(f"    Building {len(valid_starts)} slice objects...")

    # Build VerticalSlice objects - vectorized where possible
    slices = []
    for start, end in zip(valid_starts, valid_ends):
        indices = sorted_indices_cpu[start:end]

        # Check unique tracks quickly with numpy
        slice_tracks = track_ids[indices]
        if len(np.unique(slice_tracks)) < 2:
            continue

        # Build occurrence list
        slice_occs = [
            {
                'track_id': int(track_ids[i]),
                'pattern_idx': int(pattern_indices[i]),
                'instrument': int(instruments[i]),
                'pitch_offset': int(pitch_offsets[i]),
            }
            for i in indices
        ]

        piece_hash = piece_hashes[indices[0]]
        slices.append(VerticalSlice(
            piece_id=hash_to_piece_id[piece_hash],
            onset_time=int(onset_times[indices[0]]),
            occurrences=slice_occs,
        ))

    if verbose:
        print(f"  Found {len(slices)} vertical slices with 2+ tracks")

    return slices


def compute_transform_for_pair(
    src_pattern: Dict,
    tgt_pattern: Dict,
    transform_vocab: List[CompoundTransform] = None,
) -> Tuple[Optional[CompoundTransform], int]:
    """
    Compute the transform mapping src_pattern → tgt_pattern.

    Returns:
        (transform, pitch_offset) or (None, 0) if no match
    """
    src_pc = np.array(src_pattern.get('pitch_classes', []))
    tgt_pc = np.array(tgt_pattern.get('pitch_classes', []))

    if len(src_pc) != len(tgt_pc) or len(src_pc) == 0:
        return None, 0

    # Check all 12 transpositions
    for t in range(12):
        transposed = (src_pc + t) % 12
        if np.array_equal(transposed, tgt_pc):
            if t == 0:
                return CompoundTransform(()), 0  # identity
            return CompoundTransform((Primitive(PrimitiveType.TRANSPOSE, t),)), 0

    # Check all 12 inversions
    for axis in range(12):
        inverted = (2 * axis - src_pc) % 12
        if np.array_equal(inverted, tgt_pc):
            return CompoundTransform((Primitive(PrimitiveType.INVERT, axis),)), 0

    # Check retrograde
    retrograde = src_pc[::-1]
    if np.array_equal(retrograde, tgt_pc):
        return CompoundTransform((Primitive(PrimitiveType.RETROGRADE),)), 0

    # Check retrograde + transposition
    for t in range(12):
        rt = (retrograde + t) % 12
        if np.array_equal(rt, tgt_pc):
            return CompoundTransform((
                Primitive(PrimitiveType.RETROGRADE),
                Primitive(PrimitiveType.TRANSPOSE, t),
            )), 0

    return None, 0


def select_leader_mdl(
    slice_occs: List[dict],
    patterns: List[Dict],
    transform_costs: Dict[str, float] = None,
) -> int:
    """
    Select the "leader" pattern in a vertical slice using MDL.

    The leader is the pattern from which others can be most compactly derived.
    This is the pattern that minimizes total description length.

    Heuristics for leader selection:
    1. Patterns that can derive the most others via simple transforms
    2. Patterns with more common instruments (melodic instruments lead)
    3. In ties, prefer lower track number (conventional melody placement)

    Args:
        slice_occs: List of occurrence dicts in this slice
        patterns: Full pattern list
        transform_costs: Optional cost per transform type

    Returns:
        Index into slice_occs of the leader
    """
    if not transform_costs:
        transform_costs = {
            'identity': 0.0,
            'T': 1.0,  # Transposition is cheap
            'I': 2.0,  # Inversion is more expensive
            'R': 2.0,  # Retrograde is expensive
        }

    n = len(slice_occs)
    if n <= 1:
        return 0

    # For each potential leader, compute total cost to derive all others
    leader_costs = []

    for leader_idx in range(n):
        leader_occ = slice_occs[leader_idx]
        leader_pattern = patterns[leader_occ['pattern_idx']]
        total_cost = 0.0
        n_derivable = 0

        for follower_idx in range(n):
            if follower_idx == leader_idx:
                continue

            follower_occ = slice_occs[follower_idx]
            follower_pattern = patterns[follower_occ['pattern_idx']]

            transform, _ = compute_transform_for_pair(leader_pattern, follower_pattern)

            if transform is not None:
                # Cost based on transform complexity
                if len(transform.primitives) == 0:
                    cost = transform_costs['identity']
                elif len(transform.primitives) == 1:
                    p_type = transform.primitives[0].type
                    if p_type == PrimitiveType.TRANSPOSE:
                        cost = transform_costs['T']
                    elif p_type == PrimitiveType.INVERT:
                        cost = transform_costs['I']
                    else:
                        cost = transform_costs['R']
                else:
                    cost = sum(transform_costs.get(p.type.value[0].upper(), 2.0)
                               for p in transform.primitives)

                total_cost += cost
                n_derivable += 1
            else:
                # Can't derive this pattern from leader
                total_cost += 10.0  # High penalty

        # Prefer leaders that can derive more patterns
        # Lower cost = better leader
        adjusted_cost = total_cost - (n_derivable * 0.5)  # Bonus for derivability
        leader_costs.append((adjusted_cost, leader_idx))

    # Select leader with lowest cost
    leader_costs.sort()
    return leader_costs[0][1]


def discover_track_derives_gpu(
    patterns: List[Dict],
    track_instruments: Dict[Tuple[str, int], int] = None,
    device: str = 'cuda',
    min_confidence: float = 0.9,
    verbose: bool = True,
    max_pairs: int = 500000,  # Limit for GPU memory
) -> List[TrackDerive]:
    """
    GPU-OPTIMIZED: Discover TrackDerive relationships from vertical slices.

    Fully batched GPU implementation:
    1. Find vertical slices via GPU grouping
    2. Extract all cross-track pairs in batch
    3. GPU-batch transform matching
    4. GPU leader selection per slice
    5. Build TrackDerive objects

    Args:
        patterns: List of pattern dicts with occurrences
        track_instruments: Mapping of (piece_id, track_id) -> GM program
        device: PyTorch device
        min_confidence: Minimum match quality to create derivation
        verbose: Print progress
        max_pairs: Maximum pairs to process (for GPU memory)

    Returns:
        List of TrackDerive objects
    """
    t0 = time.time()

    if not torch.cuda.is_available():
        device = 'cpu'

    if verbose:
        print("\n[TrackDerive Discovery] (GPU-optimized)")
        print("  Finding vertical slices...")

    # Step 1: Find all vertical slices
    slices = find_vertical_slices_gpu(patterns, track_instruments, device, verbose)

    if not slices:
        if verbose:
            print("  No vertical slices found")
        return []

    # Step 2: Build pattern pitch-class tensor for GPU matching
    # Group patterns by length for efficient batching
    if verbose:
        print(f"  Building pattern tensors for {len(patterns)} patterns...")

    pattern_lengths = {}
    for p_idx, p in enumerate(patterns):
        pc = p.get('pitch_classes', [])
        length = len(pc)
        if length not in pattern_lengths:
            pattern_lengths[length] = []
        pattern_lengths[length].append((p_idx, pc))

    # Step 3: Extract ALL cross-track pairs from slices (OPTIMIZED)
    if verbose:
        print(f"  Extracting cross-track pairs from {len(slices)} slices...")

    # Pre-estimate pair count to avoid list resizing
    estimated_pairs = sum(len(s.occurrences) * (len(s.occurrences) - 1) for s in slices)
    if verbose:
        print(f"    Estimated max pairs: {estimated_pairs:,}")

    # Use numpy arrays for faster pair collection
    # Pre-allocate with estimated size
    max_possible = min(estimated_pairs, max_pairs * 2)  # Over-allocate slightly

    pair_slice_idx = np.zeros(max_possible, dtype=np.int32)
    pair_src_occ = np.zeros(max_possible, dtype=np.int32)
    pair_tgt_occ = np.zeros(max_possible, dtype=np.int32)
    pair_src_pattern = np.zeros(max_possible, dtype=np.int32)
    pair_tgt_pattern = np.zeros(max_possible, dtype=np.int32)
    pair_src_track = np.zeros(max_possible, dtype=np.int32)
    pair_tgt_track = np.zeros(max_possible, dtype=np.int32)
    pair_src_inst = np.zeros(max_possible, dtype=np.int32)
    pair_tgt_inst = np.zeros(max_possible, dtype=np.int32)
    pair_src_pitch = np.zeros(max_possible, dtype=np.int32)
    pair_tgt_pitch = np.zeros(max_possible, dtype=np.int32)

    pair_count = 0
    for slice_idx, slice_obj in enumerate(slices):
        occs = slice_obj.occurrences
        n = len(occs)

        # Extract to arrays for faster nested loop
        track_arr = np.array([o['track_id'] for o in occs], dtype=np.int32)
        pattern_arr = np.array([o['pattern_idx'] for o in occs], dtype=np.int32)
        inst_arr = np.array([o.get('instrument', o['track_id']) for o in occs], dtype=np.int32)
        pitch_arr = np.array([o.get('pitch_offset', 0) for o in occs], dtype=np.int32)

        # Vectorized pair generation using broadcasting
        for i in range(n):
            # Find all j where track differs
            diff_track_mask = track_arr != track_arr[i]
            js = np.where(diff_track_mask)[0]

            for j in js:
                if pair_count >= max_possible:
                    break
                pair_slice_idx[pair_count] = slice_idx
                pair_src_occ[pair_count] = i
                pair_tgt_occ[pair_count] = j
                pair_src_pattern[pair_count] = pattern_arr[i]
                pair_tgt_pattern[pair_count] = pattern_arr[j]
                pair_src_track[pair_count] = track_arr[i]
                pair_tgt_track[pair_count] = track_arr[j]
                pair_src_inst[pair_count] = inst_arr[i]
                pair_tgt_inst[pair_count] = inst_arr[j]
                pair_src_pitch[pair_count] = pitch_arr[i]
                pair_tgt_pitch[pair_count] = pitch_arr[j]
                pair_count += 1

        if pair_count >= max_possible:
            break

    if pair_count == 0:
        if verbose:
            print("  No cross-track pairs found")
        return []

    # Trim to actual size
    pair_slice_idx = pair_slice_idx[:pair_count]
    pair_src_occ = pair_src_occ[:pair_count]
    pair_tgt_occ = pair_tgt_occ[:pair_count]
    pair_src_pattern = pair_src_pattern[:pair_count]
    pair_tgt_pattern = pair_tgt_pattern[:pair_count]
    pair_src_track = pair_src_track[:pair_count]
    pair_tgt_track = pair_tgt_track[:pair_count]
    pair_src_inst = pair_src_inst[:pair_count]
    pair_tgt_inst = pair_tgt_inst[:pair_count]
    pair_src_pitch = pair_src_pitch[:pair_count]
    pair_tgt_pitch = pair_tgt_pitch[:pair_count]

    if verbose:
        print(f"    Collected {pair_count:,} cross-track pairs")

    # Sample if too many pairs
    if pair_count > max_pairs:
        if verbose:
            print(f"  Sampling {max_pairs} from {pair_count} pairs...")
        indices = np.random.choice(pair_count, max_pairs, replace=False)
        pair_slice_idx = pair_slice_idx[indices]
        pair_src_occ = pair_src_occ[indices]
        pair_tgt_occ = pair_tgt_occ[indices]
        pair_src_pattern = pair_src_pattern[indices]
        pair_tgt_pattern = pair_tgt_pattern[indices]
        pair_src_track = pair_src_track[indices]
        pair_tgt_track = pair_tgt_track[indices]
        pair_src_inst = pair_src_inst[indices]
        pair_tgt_inst = pair_tgt_inst[indices]
        pair_src_pitch = pair_src_pitch[indices]
        pair_tgt_pitch = pair_tgt_pitch[indices]
        pair_count = max_pairs

    if verbose:
        print(f"  GPU matching {pair_count} cross-track pairs...")

    # Step 4: GPU-batched transform matching
    # Build transform tables
    transform_table, transform_names = build_transform_tables(device)
    n_transforms = len(transform_names)

    # Process by pattern length (patterns must have same length to compare)
    matched_pair_indices = []  # indices into pair arrays
    matched_transforms = []    # transform index for each match

    for length, patterns_at_length in pattern_lengths.items():
        if length < 2:
            continue

        # Build lookup for patterns at this length
        pattern_idx_to_local = {p_idx: local_idx for local_idx, (p_idx, _) in enumerate(patterns_at_length)}
        pattern_idx_set = set(pattern_idx_to_local.keys())

        # Filter pairs where both patterns have this length (vectorized)
        src_in_length = np.isin(pair_src_pattern, list(pattern_idx_set))
        tgt_in_length = np.isin(pair_tgt_pattern, list(pattern_idx_set))
        pairs_mask = src_in_length & tgt_in_length
        pairs_indices_at_length = np.where(pairs_mask)[0]

        if len(pairs_indices_at_length) == 0:
            continue

        # Get src/tgt pattern indices for pairs at this length
        src_patterns_at_len = pair_src_pattern[pairs_indices_at_length]
        tgt_patterns_at_len = pair_tgt_pattern[pairs_indices_at_length]

        # Map to local indices
        src_local = np.array([pattern_idx_to_local[p] for p in src_patterns_at_len])
        tgt_local = np.array([pattern_idx_to_local[p] for p in tgt_patterns_at_len])

        # Build pitch class tensor for patterns at this length
        pc_tensor = torch.zeros((len(patterns_at_length), length), dtype=torch.int8, device=device)
        for local_idx, (_, pc) in enumerate(patterns_at_length):
            pc_tensor[local_idx] = torch.tensor(pc, dtype=torch.int8)

        # Build pair indices
        src_indices = torch.from_numpy(src_local).long().to(device)
        tgt_indices = torch.from_numpy(tgt_local).long().to(device)

        # Get source and target pitch classes
        src_pcs = pc_tensor[src_indices]  # (n_pairs, length)
        tgt_pcs = pc_tensor[tgt_indices]  # (n_pairs, length)

        # Apply all transforms to source and check against target
        # transform_table: (n_transforms, 12) - maps each PC to new PC
        # We need: for each pair, for each transform, check if transform(src) == tgt

        # Batch process in chunks to manage memory
        chunk_size = 50000
        n_pairs_at_length = len(pairs_indices_at_length)

        for chunk_start in range(0, n_pairs_at_length, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_pairs_at_length)
            chunk_size_actual = chunk_end - chunk_start
            chunk_src = src_pcs[chunk_start:chunk_end]  # (chunk, length)
            chunk_tgt = tgt_pcs[chunk_start:chunk_end]  # (chunk, length)

            # FULLY VECTORIZED: Apply all 24 non-retrograde transforms at once
            # transform_table[:24]: (24, 12) - PC mappings
            # chunk_src: (chunk, length) with values 0-11
            # Use advanced indexing: result[c,t,l] = transform_table[t, chunk_src[c,l]]
            src_long = chunk_src.long()  # (chunk, length)

            # Expand src to (chunk, 24, length) by repeating
            src_expanded = src_long.unsqueeze(1).expand(-1, 24, -1)  # (chunk, 24, length)

            # Reshape for gather: flatten batch and length dims
            # transform_table[:24] is (24, 12)
            # We want transformed_src[c, t, l] = transform_table[t, src[c, l]]
            transformed_src = transform_table[:24][torch.arange(24, device=device).view(1, 24, 1), src_expanded]

            # Check which transforms match target
            tgt_expanded = chunk_tgt.unsqueeze(1).expand(-1, 24, -1)  # (chunk, 24, length)
            matches = (transformed_src == tgt_expanded).all(dim=2)  # (chunk, 24)

            # FULLY VECTORIZED: Apply all 24 retrograde transforms at once
            src_reversed = chunk_src.flip(dims=[1]).long()
            src_rev_expanded = src_reversed.unsqueeze(1).expand(-1, 24, -1)
            transformed_rev = transform_table[24:48][torch.arange(24, device=device).view(1, 24, 1), src_rev_expanded]

            matches_rev = (transformed_rev == tgt_expanded).all(dim=2)  # (chunk, 24)

            # Combine matches
            all_matches = torch.cat([matches, matches_rev], dim=1)  # (chunk, 48)

            # VECTORIZED match extraction
            has_match = all_matches.any(dim=1)  # (chunk,)
            match_indices_local = torch.where(has_match)[0]

            if len(match_indices_local) > 0:
                # Get first matching transform for each matched pair (vectorized)
                # argmax on bool gives first True index
                first_transforms = all_matches[match_indices_local].byte().argmax(dim=1)

                # Convert to numpy and add to results
                local_indices_np = match_indices_local.cpu().numpy()
                transforms_np = first_transforms.cpu().numpy()

                for i, local_idx in enumerate(local_indices_np):
                    global_idx = pairs_indices_at_length[chunk_start + local_idx]
                    matched_pair_indices.append(global_idx)
                    matched_transforms.append(int(transforms_np[i]))

    if verbose:
        print(f"  Found {len(matched_pair_indices)} transform matches")

    # Step 5: Select leaders and build TrackDerive objects
    # Group matched pairs by slice
    pairs_by_slice = defaultdict(list)
    for i, global_idx in enumerate(matched_pair_indices):
        slice_idx = pair_slice_idx[global_idx]
        pairs_by_slice[slice_idx].append((global_idx, matched_transforms[i]))

    if verbose:
        print(f"  Selecting leaders for {len(pairs_by_slice)} slices with matches...")

    all_derives = []
    leader_instruments = defaultdict(int)

    for slice_idx, slice_pairs in pairs_by_slice.items():
        slice_obj = slices[slice_idx]

        # Count how many others each occ can derive (as leader)
        occ_derive_counts = defaultdict(int)
        occ_derive_costs = defaultdict(float)

        for global_idx, t_idx in slice_pairs:
            src_occ_idx = pair_src_occ[global_idx]
            occ_derive_counts[src_occ_idx] += 1
            # Cost: identity=0, T=1, I=2, R=2, compound=3
            t_name = transform_names[t_idx]
            if t_name == 'identity':
                cost = 0.0
            elif t_name.startswith('T') and not t_name.startswith('TI'):
                cost = 1.0
            elif t_name.startswith('I'):
                cost = 2.0
            elif t_name.startswith('R'):
                cost = 2.0
            else:
                cost = 3.0
            occ_derive_costs[src_occ_idx] += cost

        # Select leader: most derivations, lowest cost
        if not occ_derive_counts:
            continue

        leader_occ_idx = max(
            occ_derive_counts.keys(),
            key=lambda i: (occ_derive_counts[i], -occ_derive_costs[i])
        )

        leader_occ = slice_obj.occurrences[leader_occ_idx]
        leader_instruments[leader_occ.get('instrument', leader_occ['track_id'])] += 1

        # Create TrackDerive for each pair where leader is source
        for global_idx, t_idx in slice_pairs:
            if pair_src_occ[global_idx] != leader_occ_idx:
                continue  # Not from leader

            tgt_occ_idx = pair_tgt_occ[global_idx]
            tgt_occ = slice_obj.occurrences[tgt_occ_idx]

            # Build CompoundTransform from transform name
            t_name = transform_names[t_idx]
            transform = _name_to_compound_transform(t_name)

            # Pitch offset
            src_offset = pair_src_pitch[global_idx]
            tgt_offset = pair_tgt_pitch[global_idx]
            voicing_offset = (tgt_offset - src_offset) % 12

            derive = TrackDerive(
                source_piece=slice_obj.piece_id,
                source_track=int(pair_src_track[global_idx]),
                source_instrument=int(pair_src_inst[global_idx]),
                source_time=slice_obj.onset_time,
                source_pattern_id=int(pair_src_pattern[global_idx]),
                target_track=int(pair_tgt_track[global_idx]),
                target_instrument=int(pair_tgt_inst[global_idx]),
                target_time=slice_obj.onset_time,
                target_pattern_id=int(pair_tgt_pattern[global_idx]),
                transform=transform,
                pitch_offset=voicing_offset,
                rhythm_scale=1.0,  # Could compute from patterns if needed
                velocity_scale=1.0,
                confidence=1.0,
            )
            all_derives.append(derive)

    elapsed = time.time() - t0

    if verbose:
        print(f"  Discovered {len(all_derives)} TrackDerive relationships in {elapsed:.1f}s")

        if leader_instruments:
            print("  Leader instruments:")
            sorted_leaders = sorted(leader_instruments.items(), key=lambda x: -x[1])[:5]
            for inst, count in sorted_leaders:
                print(f"    {instrument_name(inst)}: {count} times")

        if all_derives:
            print("  Sample derivations:")
            for derive in all_derives[:5]:
                src_name = instrument_name(derive.source_instrument)
                tgt_name = instrument_name(derive.target_instrument)
                t_str = str(derive.transform)
                extras = []
                if derive.pitch_offset != 0:
                    extras.append(f"+O{derive.pitch_offset}")
                extras_str = " ".join(extras) if extras else ""
                print(f"    {src_name} → {tgt_name}: {t_str} {extras_str}")

    return all_derives


def _name_to_compound_transform(name: str) -> CompoundTransform:
    """Convert transform name to CompoundTransform object."""
    if name == 'identity':
        return CompoundTransform(())
    elif name.startswith('RI'):
        axis = int(name[2:])
        return CompoundTransform((
            Primitive(PrimitiveType.RETROGRADE),
            Primitive(PrimitiveType.INVERT, axis),
        ))
    elif name.startswith('RT'):
        t = int(name[2:])
        return CompoundTransform((
            Primitive(PrimitiveType.RETROGRADE),
            Primitive(PrimitiveType.TRANSPOSE, t),
        ))
    elif name == 'R':
        return CompoundTransform((Primitive(PrimitiveType.RETROGRADE),))
    elif name.startswith('I'):
        axis = int(name[1:])
        return CompoundTransform((Primitive(PrimitiveType.INVERT, axis),))
    elif name.startswith('T'):
        t = int(name[1:])
        return CompoundTransform((Primitive(PrimitiveType.TRANSPOSE, t),))
    else:
        return CompoundTransform(())


def add_derives_to_occurrences(
    patterns: List[Dict],
    derives: List[TrackDerive],
) -> None:
    """
    Add TrackDerive references to pattern occurrences.

    This modifies patterns in-place, adding 'derived_from' field to
    occurrences that are derived from other occurrences.

    Args:
        patterns: List of pattern dicts (modified in-place)
        derives: List of TrackDerive objects
    """
    # Build index: (piece, track, time, pattern_idx) -> occurrence
    occ_index = {}
    for p_idx, p in enumerate(patterns):
        for i, occ in enumerate(p.get('occurrences', [])):
            # Handle both dict and PatternOccurrence objects
            if hasattr(occ, 'piece_id'):
                key = (occ.piece_id, occ.track_id, occ.onset_time, p_idx)
            else:
                key = (occ['piece_id'], occ['track_id'], occ['onset_time'], p_idx)
            occ_index[key] = (p_idx, i)

    # Add derived_from to target occurrences
    for derive in derives:
        target_key = (
            derive.source_piece,  # Same piece
            derive.target_track,
            derive.target_time,
            derive.target_pattern_id,
        )

        if target_key in occ_index:
            p_idx, occ_idx = occ_index[target_key]
            occ = patterns[p_idx]['occurrences'][occ_idx]

            # Add derived_from info
            derived_from = {
                'source_track': derive.source_track,
                'source_pattern': derive.source_pattern_id,
                'transform': str(derive.transform),
                'pitch_offset': derive.pitch_offset,
                'rhythm_scale': derive.rhythm_scale,
                'velocity_scale': derive.velocity_scale,
            }

            # Handle both dict and PatternOccurrence objects
            if isinstance(occ, dict):
                occ['derived_from'] = derived_from
            elif hasattr(occ, '__dict__'):
                # It's a dataclass or object - set attribute directly
                object.__setattr__(occ, 'derived_from', derived_from)


def run_track_derive_discovery(
    patterns: List[Dict],
    track_instruments: Dict[Tuple[str, int], int] = None,
    device: str = 'cuda',
    add_to_occurrences: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Run full TrackDerive discovery pipeline.

    Args:
        patterns: List of pattern dicts with occurrences
        track_instruments: Optional (piece_id, track_id) -> GM program mapping
        device: PyTorch device
        add_to_occurrences: If True, add derived_from to occurrences
        verbose: Print progress

    Returns:
        Dict with:
            - derives: List of TrackDerive objects
            - n_derives: Count
            - derives_by_transform: Counts per transform type
            - leader_instruments: Which instruments lead
    """
    derives = discover_track_derives_gpu(
        patterns,
        track_instruments=track_instruments,
        device=device,
        verbose=verbose,
    )

    if add_to_occurrences and derives:
        add_derives_to_occurrences(patterns, derives)
        if verbose:
            print(f"  Added 'derived_from' to {len(derives)} occurrences")

    # Aggregate statistics
    derives_by_transform = defaultdict(int)
    leader_instruments = defaultdict(int)

    for d in derives:
        t_name = str(d.transform) if d.transform else 'identity'
        # Simplify to base transform type
        if t_name == 'identity' or t_name == '':
            derives_by_transform['identity'] += 1
        elif t_name.startswith('T'):
            derives_by_transform['T'] += 1
        elif t_name.startswith('I'):
            derives_by_transform['I'] += 1
        elif t_name == 'R' or t_name.startswith('R'):
            derives_by_transform['R'] += 1
        else:
            derives_by_transform[t_name] += 1

        leader_instruments[d.source_instrument] += 1

    return {
        'derives': derives,
        'n_derives': len(derives),
        'derives_by_transform': dict(derives_by_transform),
        'leader_instruments': dict(leader_instruments),
        'derives_json': [d.to_dict() for d in derives],
    }


if __name__ == "__main__":
    # Test with sample data
    print("TrackDerive Discovery Module")
    print("=" * 50)

    # Create sample patterns
    patterns = [
        {
            'pitch_classes': [0, 4, 7],  # C major triad
            'rhythm_ratios': [1.0, 1.0, 1.0],
            'velocity_ratios': [1.0, 1.0, 1.0],
            'occurrences': [
                {'piece_id': 'test', 'track_id': 0, 'onset_time': 0, 'pitch_offset': 0},
                {'piece_id': 'test', 'track_id': 0, 'onset_time': 480, 'pitch_offset': 0},
            ]
        },
        {
            'pitch_classes': [7, 11, 2],  # G major triad (T7 of C)
            'rhythm_ratios': [1.0, 1.0, 1.0],
            'velocity_ratios': [1.0, 1.0, 1.0],
            'occurrences': [
                {'piece_id': 'test', 'track_id': 1, 'onset_time': 0, 'pitch_offset': 0},
            ]
        },
        {
            'pitch_classes': [0, 8, 5],  # F major triad (I0 of C -> [0, -4, -7] = [0, 8, 5])
            'rhythm_ratios': [1.0, 1.0, 1.0],
            'velocity_ratios': [0.8, 0.8, 0.8],  # Quieter
            'occurrences': [
                {'piece_id': 'test', 'track_id': 2, 'onset_time': 0, 'pitch_offset': 0},
            ]
        },
    ]

    # Run discovery
    result = run_track_derive_discovery(
        patterns,
        device='cpu',  # CPU for testing
        verbose=True,
    )

    print(f"\nResult: {result['n_derives']} derives")
    print(f"By transform: {result['derives_by_transform']}")
