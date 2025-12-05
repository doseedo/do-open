#!/usr/bin/env python3
"""
TrackDerive Discovery: Cross-Track Derivation for Arrangement Patterns
=======================================================================

This module discovers per-occurrence cross-track derivation relationships
and adds them to the pattern graph. Unlike aggregate_orchestration_rules
which produces summary statistics, TrackDerive captures:

    "This trombone occurrence = this trumpet occurrence + T(-7)"

This is the heart of arrangement knowledge:
- Sax section = Brass melody + harmonized intervals
- A section strings = A section piano + octave doubling
- Bridge horn = Verse melody + transposition

Architecture:
    1. Find vertical slices (patterns at same time across tracks)
    2. For each slice, compute transforms between all pattern pairs
    3. Select "leader" pattern and derive others from it (MDL optimization)
    4. Store TrackDerive objects linking occurrences

The key insight is that arrangement is about WHICH tracks derive from which,
not just statistical correlation. We want to discover the "voice leading"
of arrangement.

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

    # Flatten all occurrences
    all_occs = []
    for p_idx, p in enumerate(patterns):
        for occ in p.get('occurrences', []):
            # Handle both dict and PatternOccurrence objects
            if hasattr(occ, 'piece_id'):
                # It's a PatternOccurrence object
                piece_id = occ.piece_id
                track_id = int(occ.track_id)
                onset = int(occ.onset_time)
                pitch_offset = getattr(occ, 'pitch_offset', 0)
            else:
                # It's a dict
                piece_id = occ['piece_id']
                track_id = int(occ['track_id'])
                onset = int(occ['onset_time'])
                pitch_offset = occ.get('pitch_offset', 0)

            # Get instrument (GM program)
            if track_instruments:
                instrument = track_instruments.get((piece_id, track_id), track_id)
            else:
                instrument = track_id

            all_occs.append({
                'piece_id': piece_id,
                'piece_hash': hash(piece_id) % (2**31),
                'track_id': track_id,
                'onset_time': onset,
                'pattern_idx': p_idx,
                'instrument': instrument,
                'pitch_offset': pitch_offset,
            })

    if len(all_occs) < 2:
        return []

    # Convert to tensors for GPU grouping
    piece_hashes = torch.tensor([o['piece_hash'] for o in all_occs], device=device)
    onsets = torch.tensor([o['onset_time'] for o in all_occs], device=device)

    # Create slice keys (piece * large_const + onset)
    slice_keys = piece_hashes * 10000000 + onsets

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

    # Build VerticalSlice objects
    slices = []
    for start, end in zip(valid_starts, valid_ends):
        indices = sorted_indices_cpu[start:end]
        first_occ = all_occs[indices[0]]

        slice_occs = []
        for idx in indices:
            occ = all_occs[idx]
            slice_occs.append({
                'track_id': occ['track_id'],
                'pattern_idx': occ['pattern_idx'],
                'instrument': occ['instrument'],
                'pitch_offset': occ['pitch_offset'],
            })

        # Only include if we have multiple DIFFERENT tracks
        unique_tracks = set(o['track_id'] for o in slice_occs)
        if len(unique_tracks) >= 2:
            slices.append(VerticalSlice(
                piece_id=first_occ['piece_id'],
                onset_time=first_occ['onset_time'],
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
) -> List[TrackDerive]:
    """
    Discover TrackDerive relationships from vertical slices.

    This is the main entry point. For each vertical slice:
    1. Select the leader pattern (MDL-optimal source)
    2. Compute transforms from leader to all other patterns
    3. Create TrackDerive objects for valid derivations

    Args:
        patterns: List of pattern dicts with occurrences
        track_instruments: Mapping of (piece_id, track_id) -> GM program
        device: PyTorch device
        min_confidence: Minimum match quality to create derivation
        verbose: Print progress

    Returns:
        List of TrackDerive objects
    """
    t0 = time.time()

    if verbose:
        print("\n[TrackDerive Discovery]")
        print("  Finding vertical slices...")

    # Step 1: Find all vertical slices
    slices = find_vertical_slices_gpu(patterns, track_instruments, device, verbose)

    if not slices:
        if verbose:
            print("  No vertical slices found")
        return []

    # Step 2: Process each slice
    all_derives = []
    leader_instruments = defaultdict(int)  # Track which instruments lead

    for slice_obj in slices:
        occs = slice_obj.occurrences
        if len(occs) < 2:
            continue

        # Select leader
        leader_idx = select_leader_mdl(occs, patterns)
        leader_occ = occs[leader_idx]
        leader_pattern = patterns[leader_occ['pattern_idx']]
        leader_instruments[leader_occ['instrument']] += 1

        # Create TrackDerive for each follower
        for follower_idx, follower_occ in enumerate(occs):
            if follower_idx == leader_idx:
                continue

            # Same track = not cross-track
            if follower_occ['track_id'] == leader_occ['track_id']:
                continue

            follower_pattern = patterns[follower_occ['pattern_idx']]

            # Compute transform
            transform, _ = compute_transform_for_pair(leader_pattern, follower_pattern)

            if transform is None:
                continue  # No valid transform found

            # Compute rhythm/velocity scales if available
            rhythm_scale = 1.0
            velocity_scale = 1.0

            src_rhythms = leader_pattern.get('rhythm_ratios', [])
            tgt_rhythms = follower_pattern.get('rhythm_ratios', [])
            if src_rhythms and tgt_rhythms and len(src_rhythms) == len(tgt_rhythms):
                # Compute average scale factor
                src_sum = sum(src_rhythms)
                tgt_sum = sum(tgt_rhythms)
                if src_sum > 0:
                    rhythm_scale = tgt_sum / src_sum

            src_vels = leader_pattern.get('velocity_ratios', [])
            tgt_vels = follower_pattern.get('velocity_ratios', [])
            if src_vels and tgt_vels and len(src_vels) == len(tgt_vels):
                src_sum = sum(src_vels)
                tgt_sum = sum(tgt_vels)
                if src_sum > 0:
                    velocity_scale = tgt_sum / src_sum

            # Compute pitch offset between occurrences
            src_offset = leader_occ.get('pitch_offset', 0)
            tgt_offset = follower_occ.get('pitch_offset', 0)
            voicing_offset = (tgt_offset - src_offset) % 12

            # Create TrackDerive
            derive = TrackDerive(
                source_piece=slice_obj.piece_id,
                source_track=leader_occ['track_id'],
                source_instrument=leader_occ['instrument'],
                source_time=slice_obj.onset_time,
                source_pattern_id=leader_occ['pattern_idx'],
                target_track=follower_occ['track_id'],
                target_instrument=follower_occ['instrument'],
                target_time=slice_obj.onset_time,
                target_pattern_id=follower_occ['pattern_idx'],
                transform=transform,
                pitch_offset=voicing_offset,
                rhythm_scale=round(rhythm_scale, 3),
                velocity_scale=round(velocity_scale, 3),
                confidence=1.0,  # Exact matches only for now
            )

            all_derives.append(derive)

    elapsed = time.time() - t0

    if verbose:
        print(f"  Discovered {len(all_derives)} TrackDerive relationships in {elapsed:.1f}s")

        # Show leader instrument distribution
        if leader_instruments:
            print("  Leader instruments (which tracks lead arrangements):")
            sorted_leaders = sorted(leader_instruments.items(), key=lambda x: -x[1])[:5]
            for inst, count in sorted_leaders:
                print(f"    {instrument_name(inst)}: {count} times")

        # Show sample derives
        if all_derives:
            print("  Sample derivations:")
            for derive in all_derives[:5]:
                src_name = instrument_name(derive.source_instrument)
                tgt_name = instrument_name(derive.target_instrument)
                t_str = str(derive.transform)
                extras = []
                if derive.pitch_offset != 0:
                    extras.append(f"+O{derive.pitch_offset}")
                if derive.rhythm_scale != 1.0:
                    extras.append(f"τ{derive.rhythm_scale:.2f}")
                if derive.velocity_scale != 1.0:
                    extras.append(f"v{derive.velocity_scale:.2f}")
                extras_str = " ".join(extras) if extras else ""
                print(f"    {src_name} → {tgt_name}: {t_str} {extras_str}")

    return all_derives


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
