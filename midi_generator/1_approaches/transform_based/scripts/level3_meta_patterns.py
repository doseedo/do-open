#!/usr/bin/env python3
"""
Level 3: Meta-Pattern Discovery via GPU Re-Pair on Transform Sequences

Discovers recurring transform progressions like:
- M1 = (T5, T7) = "II-V motion"
- M2 = (T7, identity) = "V-I cadence"
- M3 = (T5, T7, identity) = "II-V-I progression"

Re-Pair IS inherently hierarchical - later rules reference earlier rules:
- Iteration 0: M0 = (T5, T7)           ← bigram
- Iteration 1: M1 = (T7, identity)     ← bigram
- Iteration 2: M2 = (M0, identity)     ← USES M0! This is meta-meta
- Iteration 3: M3 = (M0, M1)           ← Composes two meta-patterns

So M2 = (M0, identity) = ((T5, T7), identity) = (T5, T7, identity).
Hierarchy emerges naturally within single Re-Pair pass.
"""

import torch
import numpy as np
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class TransformSequence:
    """A sequence of transforms within one track of a piece (horizontal progression)."""
    piece_id: str
    track_id: int  # Which track/voice this sequence is from
    transforms: List[int]  # Transform IDs in temporal order
    pattern_ids: List[int]  # Which patterns were involved


# ============================================================
# STEP 1: Build Transform Lookup Table (GPU)
# ============================================================

def parse_transform_name(name: str) -> Dict:
    """Parse transform name string into type/param dict."""
    import re
    if name == 'identity':
        return {'type': 'identity', 'param': 0, 'name': name}
    elif name == 'R':
        return {'type': 'retrograde', 'param': 0, 'name': name}
    elif name.startswith('T'):
        # T5, T7, T11, etc.
        match = re.match(r'T(\d+)', name)
        if match:
            return {'type': 'transpose', 'param': int(match.group(1)), 'name': name}
    elif name.startswith('I'):
        # I0, I5, etc. or I5∘R
        match = re.match(r'I(\d+)', name)
        if match:
            return {'type': 'invert', 'param': int(match.group(1)), 'name': name}
    # Compound transforms - treat as identity for now (or skip)
    return {'type': 'unknown', 'param': 0, 'name': name}


def build_transform_lookup_gpu(
    patterns: List[Dict],
    transform_vocab: List,  # Can be List[Dict] or List[str]
    device: str = 'cuda',
    verbose: bool = True
) -> torch.Tensor:
    """
    Build a GPU tensor for fast transform lookup between patterns.

    FULLY GPU-ACCELERATED VERSION - no Python loops over GPU data.

    Returns:
        lookup[i, j] = transform_id that maps pattern_i → pattern_j
                       or -1 if no transform exists
    """
    import time
    t0 = time.time()

    n_patterns = len(patterns)
    if n_patterns == 0:
        return torch.empty((0, 0), dtype=torch.int16, device=device)

    # Handle both dict and string transform formats
    if transform_vocab and isinstance(transform_vocab[0], str):
        transform_vocab = [parse_transform_name(t) for t in transform_vocab]

    n_transforms = len(transform_vocab)
    if verbose:
        print(f"    GPU lookup: {n_patterns} patterns × {n_transforms} transforms", flush=True)

    # ================================================================
    # STEP 1: Build pattern tensor on CPU first, then transfer (faster)
    # ================================================================
    max_len = max(len(p['pitch_classes']) for p in patterns)

    # Build on CPU as numpy (much faster than Python loop with torch)
    import numpy as np
    pattern_np = np.full((n_patterns, max_len), -1, dtype=np.int32)
    lengths_np = np.zeros(n_patterns, dtype=np.int32)

    for i, p in enumerate(patterns):
        pc = p['pitch_classes']
        pattern_np[i, :len(pc)] = pc
        lengths_np[i] = len(pc)

    # Single transfer to GPU
    pattern_tensor = torch.from_numpy(pattern_np).to(device)
    pattern_lengths = torch.from_numpy(lengths_np).to(device)

    if verbose:
        print(f"    Patterns loaded: {time.time()-t0:.1f}s", flush=True)

    # ================================================================
    # STEP 2: Build transform table vectorized
    # ================================================================
    # transform_table[t, pc] = new_pc after applying transform t
    base_pcs = torch.arange(12, device=device, dtype=torch.int32)  # [12]
    transform_table = torch.zeros((n_transforms, 12), dtype=torch.int32, device=device)
    transform_valid = torch.ones(n_transforms, dtype=torch.bool, device=device)

    for t_idx, t in enumerate(transform_vocab):
        if t['type'] == 'transpose':
            transform_table[t_idx] = (base_pcs + t['param']) % 12
        elif t['type'] == 'invert':
            transform_table[t_idx] = (t['param'] - base_pcs) % 12
        elif t['type'] == 'identity':
            transform_table[t_idx] = base_pcs
        elif t['type'] == 'retrograde':
            transform_table[t_idx] = base_pcs  # Skip (needs per-pattern handling)
            transform_valid[t_idx] = False
        else:
            transform_valid[t_idx] = False  # Unknown/compound

    if verbose:
        print(f"    Transform table built: {time.time()-t0:.1f}s", flush=True)

    # ================================================================
    # STEP 3: Group patterns by length (only same-length can match)
    # ================================================================
    unique_lengths = torch.unique(pattern_lengths)
    lookup = torch.full((n_patterns, n_patterns), -1, dtype=torch.int16, device=device)

    for length in unique_lengths:
        length_val = length.item()
        if length_val == 0:
            continue

        # Get pattern indices with this length
        len_mask = pattern_lengths == length
        indices = torch.where(len_mask)[0]  # [M]
        n_same_len = indices.shape[0]

        if n_same_len < 2:
            continue

        # Extract patterns of this length: [M, L]
        sub_patterns = pattern_tensor[indices, :length_val]  # [M, length]

        # ================================================================
        # STEP 4: Apply all transforms at once using gather
        # ================================================================
        # sub_patterns: [M, L], transform_table: [T, 12]
        # We want: transformed[m, t, l] = transform_table[t, sub_patterns[m, l]]

        M, L = sub_patterns.shape
        T = n_transforms

        # Expand for gather: [M, T, L]
        patterns_expanded = sub_patterns.unsqueeze(1).expand(M, T, L)  # [M, T, L]

        # Use advanced indexing for transform application
        # transform_table[t, pc] for each position
        t_indices = torch.arange(T, device=device).view(1, T, 1).expand(M, T, L)  # [M, T, L]
        transformed = transform_table[t_indices, patterns_expanded]  # [M, T, L]

        # ================================================================
        # STEP 5: Compare all transformed patterns to all targets (vectorized)
        # ================================================================
        # transformed: [M, T, L], sub_patterns: [M, L]
        # We want matches[m1, t, m2] = all(transformed[m1, t, :] == sub_patterns[m2, :])

        # Reshape for broadcasting: [M, T, 1, L] vs [1, 1, M, L]
        trans_4d = transformed.unsqueeze(2)  # [M, T, 1, L]
        target_4d = sub_patterns.unsqueeze(0).unsqueeze(0)  # [1, 1, M, L]

        # Element-wise comparison and reduction
        # This creates [M, T, M] boolean tensor
        matches = (trans_4d == target_4d).all(dim=3)  # [M, T, M]

        # Mask out invalid transforms
        matches[:, ~transform_valid, :] = False

        # ================================================================
        # STEP 6: Find first matching transform for each pair (GPU-only)
        # ================================================================
        # matches: [M, T, M] - for each (src, tgt), find first t where matches[src, t, tgt] = True

        # Convert to int and find argmax along transform dimension
        # But we need "first True" not "argmax", so use where + min

        # Create transform priority tensor (lower = higher priority for first match)
        transform_priority = torch.arange(T, device=device).view(1, T, 1).expand(M, T, M)  # [M, T, M]

        # Where no match, use high value
        priority_masked = torch.where(matches, transform_priority, torch.tensor(T + 1, device=device))

        # Find min priority (first matching transform) for each (src, tgt)
        first_match, first_t_idx = priority_masked.min(dim=1)  # [M, M], [M, M]

        # Create mask for pairs that have at least one match
        has_match = first_match < T  # [M, M]

        # Map back to global indices and update lookup table
        # indices[i] gives global pattern index for local index i
        global_src = indices.unsqueeze(1).expand(M, M)  # [M, M]
        global_tgt = indices.unsqueeze(0).expand(M, M)  # [M, M]

        # Update lookup where we found matches
        # Only update where lookup is still -1 (vectorized conditional update)
        update_mask = has_match & (lookup[global_src, global_tgt] == -1)
        lookup[global_src[update_mask], global_tgt[update_mask]] = first_t_idx[update_mask].to(torch.int16)

        # Clear cache periodically
        if length_val % 5 == 0:
            torch.cuda.empty_cache()

    if verbose:
        n_found = (lookup >= 0).sum().item()
        print(f"    Lookup complete: {n_found} relations, {time.time()-t0:.1f}s", flush=True)

    return lookup


# ============================================================
# STEP 2: Extract Transform Sequences (Vectorized)
# ============================================================

def extract_transform_sequences_gpu(
    patterns: List[Dict],
    transform_lookup: torch.Tensor,
    min_sequence_length: int = 3,
    device: str = 'cuda'
) -> List[TransformSequence]:
    """
    Extract HORIZONTAL transform sequences (per-track temporal progressions).

    For each (piece, track) pair:
    1. Get all pattern occurrences on that track, sorted by onset_time
    2. Look up transform between consecutive patterns on the SAME track
    3. Build sequence of transform IDs

    This captures horizontal melodic/harmonic progressions within each voice,
    NOT cross-track (vertical) relations which are handled separately.
    """
    # Build (piece_id, track_id) -> occurrences index
    # Key insight: group by BOTH piece and track to get per-voice sequences
    track_occurrences = defaultdict(list)

    for p_idx, p in enumerate(patterns):
        for occ in p.get('occurrences', []):
            key = (occ['piece_id'], occ['track_id'])
            track_occurrences[key].append({
                'pattern_idx': p_idx,
                'onset_time': occ['onset_time'],
            })

    # Sort each track's occurrences by onset_time
    for key in track_occurrences:
        track_occurrences[key].sort(key=lambda x: x['onset_time'])

    # Extract sequences
    sequences = []
    lookup_cpu = transform_lookup.cpu().numpy()  # Move to CPU for indexing

    for (piece_id, track_id), occs in track_occurrences.items():
        if len(occs) < min_sequence_length:
            continue

        transforms = []
        pattern_ids = []

        for i in range(len(occs) - 1):
            src_idx = occs[i]['pattern_idx']
            tgt_idx = occs[i + 1]['pattern_idx']

            # Skip same-pattern pairs (would always be identity)
            if src_idx == tgt_idx:
                continue

            t_id = lookup_cpu[src_idx, tgt_idx]

            # Skip identity transforms (t_id == 0) - they flood the sequences
            # and prevent Re-Pair from finding interesting progressions
            # (different patterns with same pitch content → identity)
            if t_id == 0:  # identity
                continue

            if t_id >= 0:  # Valid non-identity transform found
                transforms.append(int(t_id))
                pattern_ids.append(src_idx)

        if len(transforms) >= min_sequence_length - 1:
            sequences.append(TransformSequence(
                piece_id=piece_id,
                track_id=track_id,  # Now we track which voice this came from
                transforms=transforms,
                pattern_ids=pattern_ids,
            ))

    return sequences


# ============================================================
# STEP 3: GPU Re-Pair on Transform Sequences
# ============================================================

def run_meta_repair_gpu(
    sequences: List[TransformSequence],
    n_transforms: int,
    max_rules: int = 1000,
    min_frequency: int = 5,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict:
    """
    Run GPU Re-Pair on transform sequences to discover meta-patterns.

    Reuses existing Re-Pair infrastructure with transforms as terminals.
    """
    from grammar.v2.repair_gpu_v2 import build_repair_grammar_v2

    # Convert TransformSequence objects to token lists
    token_sequences = [seq.transforms for seq in sequences if len(seq.transforms) >= 2]

    if verbose:
        total_tokens = sum(len(s) for s in token_sequences)
        print(f"  Meta Re-Pair: {len(token_sequences)} pieces, {total_tokens} transform tokens")
        print(f"  Terminal vocabulary: {n_transforms} transforms")

    if len(token_sequences) < 10:
        if verbose:
            print(f"  Too few sequences for meta-pattern discovery")
        return {'rules': {}, 'n_rules': 0}

    # Run GPU Re-Pair (same algorithm, different terminals)
    grammar = build_repair_grammar_v2(
        token_sequences,
        device=device,
        max_rules=max_rules,
        min_pair_count=min_frequency,  # min_frequency maps to min_pair_count
        verbose=verbose
    )

    # Extract meta-patterns
    meta_rules = {}
    for i in range(grammar.n_rules):
        rule_id = grammar.n_terminals + i
        expansion = grammar.expand_rule(rule_id)

        if expansion is not None and len(expansion) >= 2:
            meta_rules[f"M{i}"] = {
                'transform_ids': list(expansion),
                'expansion_depth': len(expansion),
            }

    if verbose:
        print(f"  Discovered {len(meta_rules)} meta-patterns")
        print(f"  Compression: {grammar.compression_ratio():.2f}x")

    return {
        'rules': meta_rules,
        'n_rules': len(meta_rules),
        'compression_ratio': grammar.compression_ratio(),
        'grammar': grammar,
    }


# ============================================================
# STEP 4: Interpret Meta-Patterns
# ============================================================

def interpret_meta_patterns(
    meta_rules: Dict,
    transform_vocab: List[Dict],
    min_frequency: int = 10,
    verbose: bool = True
) -> List[Dict]:
    """
    Convert meta-pattern IDs to human-readable form.

    Example output:
        M1 = T5 → T7 (II-V motion), frequency: 847
        M2 = T7 → identity (V-I cadence), frequency: 612
    """
    interpreted = []

    for rule_id, rule_data in meta_rules.items():
        transform_ids = rule_data['transform_ids']

        # Convert to names
        transform_names = []
        for t_id in transform_ids:
            if t_id < len(transform_vocab):
                t = transform_vocab[t_id]
                transform_names.append(t.get('name', f"T{t_id}"))
            else:
                # It's a meta-rule reference
                transform_names.append(f"M{t_id - len(transform_vocab)}")

        # Musical interpretation
        interpretation = interpret_progression(transform_names)

        interpreted.append({
            'rule_id': rule_id,
            'transforms': transform_names,
            'sequence': ' → '.join(transform_names),
            'interpretation': interpretation,
        })

    if verbose:
        print("\n  Top Meta-Patterns:")
        for mp in interpreted[:10]:
            print(f"    {mp['rule_id']}: {mp['sequence']}")
            if mp['interpretation']:
                print(f"         ({mp['interpretation']})")

    return interpreted


def interpret_progression(transforms: List[str]) -> Optional[str]:
    """Map transform sequences to musical interpretations."""
    seq = tuple(transforms)

    interpretations = {
        ('T5', 'T7'): 'II-V motion (circle of fifths)',
        ('T7', 'identity'): 'V-I resolution',
        ('T5', 'T7', 'identity'): 'II-V-I cadence',
        ('T5', 'T5'): 'Sequential fourths',
        ('T7', 'T7'): 'Sequential fifths',
        ('T6',): 'Tritone substitution',
        ('T11', 'identity'): 'Chromatic approach → resolution',
        ('T1', 'T11'): 'Chromatic neighbor motion',
    }

    return interpretations.get(seq)


# ============================================================
# STEP 5: Orchestration Rule Aggregation (NEW FIX)
# ============================================================

def aggregate_orchestration_rules_gpu(
    patterns: List[Dict],
    transform_vocab: List,
    min_confidence: float = 0.3,
    min_frequency: int = 5,
    device: str = 'cuda',
    verbose: bool = True,
    track_instruments: Dict[Tuple[str, int], int] = None,  # (piece_id, track_id) -> GM program
) -> Dict:
    """
    GPU-accelerated orchestration rule aggregation.

    Finds dominant transforms between track pairs from vertical slices
    (patterns occurring at the same time across different tracks).

    GPU Strategy:
    1. Build flat arrays of all occurrences with (piece, onset, track, pattern_idx)
    2. Use GPU sorting + unique to find vertical slices efficiently
    3. Batch compute transforms between all same-length pattern pairs on GPU
    4. Aggregate results with GPU scatter operations

    If track_instruments is provided, displays GM instrument names and filters
    same-instrument pairs (which are meaningless for orchestration).
    """
    import torch
    from collections import Counter

    if not torch.cuda.is_available():
        device = 'cpu'

    # Helper function for GM instrument names
    def gm_instrument_name(gm_prog: int) -> str:
        return GM_INSTRUMENTS.get(gm_prog, f"GM{gm_prog}")

    # Step 1: Flatten all occurrences into arrays
    # We need both numeric data for GPU and string piece_ids for instrument lookup
    # IMPORTANT: Include pitch_offset for voicing computation!
    occ_numeric = []  # (piece_id_hash, onset_time, track_id, pattern_idx, pattern_length, gm_program, pitch_offset)
    for p_idx, p in enumerate(patterns):
        pc = p.get('pitch_classes', [])
        plen = len(pc)
        for occ in p.get('occurrences', []):
            piece_id = occ['piece_id']
            piece_hash = hash(piece_id) % (2**31)  # Hash string to int
            track_id = int(occ['track_id'])
            # Get GM program if track_instruments provided, else use track_id as fallback
            if track_instruments:
                gm_prog = track_instruments.get((piece_id, track_id), track_id)
            else:
                gm_prog = track_id  # Fallback: use track_id
            # Get pitch_offset (voicing info) - this is the KEY for proper orchestration!
            # pitch_offset = transposition of this occurrence relative to canonical pattern
            pitch_offset = occ.get('pitch_offset', 0)
            occ_numeric.append((
                piece_hash,
                int(occ['onset_time']),
                track_id,
                p_idx,
                plen,
                gm_prog,  # GM program number (or track_id fallback)
                pitch_offset,  # 0-11: pitch class offset for voicing
            ))

    if len(occ_numeric) < 2:
        return {'rules': [], 'n_rules': 0, 'n_slices': 0}

    # Convert to tensors
    occ_tensor = torch.tensor(occ_numeric, dtype=torch.long, device=device)
    piece_ids = occ_tensor[:, 0]
    onsets = occ_tensor[:, 1]
    track_ids = occ_tensor[:, 2]
    pattern_idxs = occ_tensor[:, 3]
    pattern_lens = occ_tensor[:, 4]
    gm_programs = occ_tensor[:, 5]
    pitch_offsets = occ_tensor[:, 6]  # NEW: voicing offsets

    # Step 2: Create slice keys (piece_id * large_const + onset)
    # Patterns in same slice have same key
    slice_keys = piece_ids * 10000000 + onsets

    # Sort by slice key to group same-slice occurrences
    sorted_indices = torch.argsort(slice_keys)
    sorted_keys = slice_keys[sorted_indices]
    sorted_tracks = track_ids[sorted_indices]
    sorted_patterns = pattern_idxs[sorted_indices]
    sorted_lens = pattern_lens[sorted_indices]
    sorted_gm_progs = gm_programs[sorted_indices]
    sorted_pitch_offsets = pitch_offsets[sorted_indices]  # NEW

    # Find slice boundaries (where key changes)
    key_changes = torch.cat([
        torch.tensor([True], device=device),
        sorted_keys[1:] != sorted_keys[:-1]
    ])
    slice_starts = torch.where(key_changes)[0]
    slice_ends = torch.cat([slice_starts[1:], torch.tensor([len(sorted_keys)], device=device)])
    slice_sizes = slice_ends - slice_starts

    # Only process slices with 2+ patterns (vertical slices)
    valid_slices = slice_sizes >= 2
    n_slices = valid_slices.sum().item()

    if verbose:
        print(f"  Found {n_slices} vertical slices (2+ concurrent patterns)")

    # Step 3: Build pitch class tensor for all patterns (for batch transform computation)
    max_len = max(len(p.get('pitch_classes', [])) for p in patterns) if patterns else 1
    n_patterns = len(patterns)
    pc_tensor = torch.zeros((n_patterns, max_len), dtype=torch.long, device=device)
    for p_idx, p in enumerate(patterns):
        pc = p.get('pitch_classes', [])
        if pc:
            pc_tensor[p_idx, :len(pc)] = torch.tensor(pc, dtype=torch.long, device=device)

    # Step 4: Extract all valid pairs from vertical slices
    # Build pairs: (src_gm_prog, tgt_gm_prog, src_pattern, tgt_pattern, length, src_track, tgt_track, src_pitch_offset, tgt_pitch_offset)
    pairs_list = []
    valid_starts = slice_starts[valid_slices].cpu().numpy()
    valid_ends = slice_ends[valid_slices].cpu().numpy()

    # This loop is unavoidable but operates on slice metadata, not pattern data
    for start, end in zip(valid_starts, valid_ends):
        tracks_in_slice = sorted_tracks[start:end].cpu().numpy()
        patterns_in_slice = sorted_patterns[start:end].cpu().numpy()
        lens_in_slice = sorted_lens[start:end].cpu().numpy()
        gm_progs_in_slice = sorted_gm_progs[start:end].cpu().numpy()
        pitch_offsets_in_slice = sorted_pitch_offsets[start:end].cpu().numpy()  # NEW

        # Generate all pairs within this slice
        for i in range(len(tracks_in_slice)):
            for j in range(i + 1, len(tracks_in_slice)):
                # Filter out same-track pairs (meaningless for cross-track orchestration)
                if tracks_in_slice[i] == tracks_in_slice[j]:
                    continue
                # Only pair same-length patterns
                if lens_in_slice[i] == lens_in_slice[j] and lens_in_slice[i] > 0:
                    pairs_list.append((
                        gm_progs_in_slice[i],   # Source GM program
                        gm_progs_in_slice[j],   # Target GM program
                        patterns_in_slice[i],
                        patterns_in_slice[j],
                        lens_in_slice[i],
                        tracks_in_slice[i],     # Keep track IDs for debugging
                        tracks_in_slice[j],
                        pitch_offsets_in_slice[i],  # NEW: source voicing offset
                        pitch_offsets_in_slice[j],  # NEW: target voicing offset
                    ))

    if not pairs_list:
        return {'rules': [], 'n_rules': 0, 'n_slices': n_slices}

    pairs_tensor = torch.tensor(pairs_list, dtype=torch.long, device=device)
    src_gm_progs = pairs_tensor[:, 0]
    tgt_gm_progs = pairs_tensor[:, 1]
    src_patterns = pairs_tensor[:, 2]
    tgt_patterns = pairs_tensor[:, 3]
    pair_lens = pairs_tensor[:, 4]
    # Columns 5 and 6 are track IDs (for debugging)
    src_pitch_offsets = pairs_tensor[:, 7]  # NEW: voicing offsets
    tgt_pitch_offsets = pairs_tensor[:, 8]  # NEW

    if verbose:
        print(f"  Processing {len(pairs_list)} cross-track pattern pairs on GPU")

    # Step 5: Batch compute transforms on GPU
    # For each pair, check all 12 transpositions and 12 inversions
    src_pcs = pc_tensor[src_patterns]  # (n_pairs, max_len)
    tgt_pcs = pc_tensor[tgt_patterns]  # (n_pairs, max_len)

    # Create length mask
    len_range = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
    mask = len_range < pair_lens.unsqueeze(1)  # (n_pairs, max_len)

    # Check transpositions T0-T11
    # For each t in 0..11: check if (src + t) % 12 == tgt for all positions
    pattern_transform_results = torch.full((len(pairs_list),), -1, dtype=torch.long, device=device)

    for t in range(12):
        transposed = (src_pcs + t) % 12
        matches = ((transposed == tgt_pcs) | ~mask).all(dim=1)
        # Only update where we haven't found a match yet
        update_mask = matches & (pattern_transform_results == -1)
        pattern_transform_results[update_mask] = t  # T0=0, T1=1, ..., T11=11

    # Check inversions I0-I11 (only for pairs not yet matched)
    unmatched = pattern_transform_results == -1
    for axis in range(12):
        inverted = (axis - src_pcs) % 12
        matches = ((inverted == tgt_pcs) | ~mask).all(dim=1)
        update_mask = matches & unmatched
        pattern_transform_results[update_mask] = 12 + axis  # I0=12, I1=13, ..., I11=23

    # ===== KEY FIX: Combine pattern transform with pitch_offset difference =====
    # The REAL voicing transform = pattern_transform + (tgt_pitch_offset - src_pitch_offset)
    # This captures how instruments are ACTUALLY voiced relative to each other!
    #
    # Example: Trumpet plays pattern P at C (offset=0), Trombone plays P at G (offset=7)
    # Pattern comparison gives identity (both play P)
    # But voicing difference is: 7 - 0 = 7 semitones (a 5th)
    # So the TRUE orchestration transform is T7, not identity!

    voicing_delta = (tgt_pitch_offsets - src_pitch_offsets) % 12  # 0-11

    # For transposition transforms (0-11), compose with voicing delta
    # T_a composed with voicing delta d = T_{(a + d) mod 12}
    is_transposition = pattern_transform_results < 12
    final_transform = torch.where(
        is_transposition & (pattern_transform_results >= 0),
        (pattern_transform_results + voicing_delta) % 12,  # Compose transpositions
        pattern_transform_results  # Keep inversions as-is (more complex to compose)
    )

    # Step 6: Aggregate by GM instrument pair (not track pair)
    # Create unique instrument pair keys
    max_gm = 128  # GM programs are 0-127
    pair_keys = src_gm_progs * max_gm + tgt_gm_progs

    # Count transforms per (instrument_pair, transform)
    pair_transforms = defaultdict(Counter)
    valid_mask = final_transform >= 0

    pair_keys_valid = pair_keys[valid_mask].cpu().numpy()
    transforms_valid = final_transform[valid_mask].cpu().numpy()
    src_gm_valid = src_gm_progs[valid_mask].cpu().numpy()
    tgt_gm_valid = tgt_gm_progs[valid_mask].cpu().numpy()

    for pk, t, src_gm, tgt_gm in zip(pair_keys_valid, transforms_valid, src_gm_valid, tgt_gm_valid):
        pair_transforms[(int(src_gm), int(tgt_gm))][int(t)] += 1

    # Step 7: Build orchestration rules with GM instrument names
    def transform_name(t_idx: int) -> str:
        if t_idx < 12:
            return "identity" if t_idx == 0 else f"T{t_idx}"
        else:
            return f"I{t_idx - 12}"

    orchestration_rules = []
    for (src_gm, tgt_gm), counts in pair_transforms.items():
        total = sum(counts.values())
        if total < min_frequency:
            continue

        dominant_t, freq = counts.most_common(1)[0]
        confidence = freq / total

        if confidence >= min_confidence:
            orchestration_rules.append({
                'source_instrument': int(src_gm),
                'target_instrument': int(tgt_gm),
                'source_name': gm_instrument_name(src_gm),
                'target_name': gm_instrument_name(tgt_gm),
                'transform': transform_name(dominant_t),
                'frequency': int(freq),
                'total_pairs': int(total),
                'confidence': float(confidence),
            })

    orchestration_rules.sort(key=lambda r: -r['confidence'])

    if verbose:
        print(f"  Orchestration rules found: {len(orchestration_rules)}")
        for rule in orchestration_rules[:10]:
            print(f"    {rule['source_name']} -> {rule['target_name']}: "
                  f"{rule['transform']} ({rule['confidence']:.1%}, n={rule['frequency']})")

    return {
        'rules': orchestration_rules,
        'n_rules': len(orchestration_rules),
        'n_slices': n_slices,
    }


# Keep CPU version as fallback
def aggregate_orchestration_rules(
    patterns: List[Dict],
    transform_vocab: List,
    min_confidence: float = 0.3,
    min_frequency: int = 5,
    verbose: bool = True,
    track_instruments: Dict[Tuple[str, int], int] = None,
) -> Dict:
    """CPU fallback - delegates to GPU version."""
    return aggregate_orchestration_rules_gpu(
        patterns, transform_vocab, min_confidence, min_frequency,
        device='cuda', verbose=verbose, track_instruments=track_instruments
    )


# ============================================================
# STEP 6: Entangled Cross-Track Relations (MDL-ALIGNED ARCHITECTURE)
# ============================================================
#
# Philosophy: Don't prescribe structure. Let Re-Pair discover it.
#
# 1. Use GM program numbers (instrument identity), not track indices
# 2. Don't bucket time deltas - include raw deltas, let Re-Pair find patterns
# 3. Minimal token encoding: (instrument_pair, transform) only
#    - Time structure emerges from sequence position
#    - Re-Pair discovers which instrument-transform combinations are productive

@dataclass
class EntangledRelation:
    """A temporal cross-track relation between patterns (MDL-aligned)."""
    piece_id: str
    source_instrument: int  # GM program number (meaningful identity)
    target_instrument: int  # GM program number
    source_time: int
    target_time: int
    source_pattern: int
    target_pattern: int
    transform: int
    time_delta: int  # Raw delta, not bucketed


def extract_entangled_relations_gpu(
    patterns: List[Dict],
    transform_lookup: torch.Tensor,
    track_instruments: Dict[Tuple[str, int], int],  # (piece_id, track_id) -> GM program
    max_delta_ticks: int = 960,  # 2 beats - this is a search space limit, not structure
    device: str = 'cuda',
    verbose: bool = True
) -> List[EntangledRelation]:
    """
    Find temporal cross-track relations using GM instruments.

    GPU-efficient: processes all pairs per piece in parallel.

    Key: Uses GM program numbers for instrument identity, not track IDs.
    """
    import time
    t0 = time.time()

    if not torch.cuda.is_available():
        device = 'cpu'

    # Flatten all occurrences with instrument info
    all_occs = []
    for p_idx, p in enumerate(patterns):
        for occ in p.get('occurrences', []):
            piece_id = occ['piece_id']
            track_id = int(occ['track_id'])
            # Get GM program number, default to track_id if not available
            instrument = track_instruments.get((piece_id, track_id), track_id)
            all_occs.append({
                'piece_id': piece_id,
                'instrument': instrument,
                'track_id': track_id,
                'onset_time': int(occ['onset_time']),
                'pattern_idx': p_idx,
            })

    if len(all_occs) < 2:
        return []

    # Group by piece
    piece_groups = defaultdict(list)
    for occ in all_occs:
        piece_groups[occ['piece_id']].append(occ)

    all_relations = []
    lookup_cpu = transform_lookup.cpu().numpy()

    for piece_id, occs in piece_groups.items():
        n = len(occs)
        if n < 2:
            continue

        # Build tensors for this piece
        instruments = torch.tensor([o['instrument'] for o in occs], device=device)
        times = torch.tensor([o['onset_time'] for o in occs], device=device)
        pat_idxs = torch.tensor([o['pattern_idx'] for o in occs], device=device)

        # Compute all pairwise conditions in parallel
        diff_instrument = instruments.unsqueeze(1) != instruments.unsqueeze(0)
        time_delta = times.unsqueeze(1) - times.unsqueeze(0)  # B - A (signed)
        time_valid = time_delta.abs() <= max_delta_ticks

        # Valid candidate pairs: different instruments, close in time
        valid_mask = diff_instrument & time_valid

        # Get pair indices
        pair_idx = valid_mask.nonzero(as_tuple=False)

        if len(pair_idx) == 0:
            continue

        # Batch transform lookup
        src_pats = pat_idxs[pair_idx[:, 0]].cpu().numpy()
        tgt_pats = pat_idxs[pair_idx[:, 1]].cpu().numpy()

        transforms = np.array([
            lookup_cpu[src, tgt] for src, tgt in zip(src_pats, tgt_pats)
        ])

        # Filter to valid transforms
        has_transform = transforms >= 0
        valid_pairs = pair_idx[has_transform].cpu().numpy()
        valid_transforms = transforms[has_transform]

        if len(valid_pairs) == 0:
            continue

        # Extract metadata
        src_instruments = instruments[valid_pairs[:, 0]].cpu().numpy()
        tgt_instruments = instruments[valid_pairs[:, 1]].cpu().numpy()
        src_times = times[valid_pairs[:, 0]].cpu().numpy()
        tgt_times = times[valid_pairs[:, 1]].cpu().numpy()
        deltas = (tgt_times - src_times)

        for i in range(len(valid_pairs)):
            all_relations.append(EntangledRelation(
                piece_id=piece_id,
                source_instrument=int(src_instruments[i]),
                target_instrument=int(tgt_instruments[i]),
                source_time=int(src_times[i]),
                target_time=int(tgt_times[i]),
                source_pattern=int(src_pats[i]),
                target_pattern=int(tgt_pats[i]),
                transform=int(valid_transforms[i]),
                time_delta=int(deltas[i]),
            ))

    if verbose:
        print(f"  Extracted {len(all_relations)} entangled relations in {time.time() - t0:.1f}s")

    return all_relations


def build_entangled_sequences(
    relations: List[EntangledRelation],
    n_transforms: int,
    verbose: bool = True
) -> Tuple[List[List[int]], Dict]:
    """
    Build sequences for Re-Pair using MINIMAL encoding.

    MDL-aligned: Token = (instrument_pair_id * n_transforms) + transform

    NO time buckets - Re-Pair discovers temporal patterns from sequence position.
    The sequence order IS the temporal structure.
    """
    # Assign instrument-pair IDs (using GM program numbers)
    instrument_pairs = set()
    for rel in relations:
        pair = (rel.source_instrument, rel.target_instrument)
        instrument_pairs.add(pair)
    pair_to_id = {p: i for i, p in enumerate(sorted(instrument_pairs))}
    n_pairs = len(pair_to_id)

    def encode_token(rel: EntangledRelation) -> int:
        pair_id = pair_to_id[(rel.source_instrument, rel.target_instrument)]
        # Minimal encoding: just instrument_pair + transform
        return pair_id * n_transforms + rel.transform

    # Group by piece, sort by time, encode
    piece_sequences = defaultdict(list)
    for rel in relations:
        piece_sequences[rel.piece_id].append(rel)

    encoded = []
    for piece_id, rels in piece_sequences.items():
        rels.sort(key=lambda x: x.source_time)
        seq = [encode_token(r) for r in rels]
        if len(seq) >= 3:
            encoded.append(seq)

    vocab_size = max(1, n_pairs * n_transforms)

    vocab_info = {
        'pair_to_id': pair_to_id,
        'id_to_pair': {v: k for k, v in pair_to_id.items()},
        'n_pairs': n_pairs,
        'n_transforms': n_transforms,
        'vocab_size': vocab_size,
    }

    if verbose:
        print(f"  Built {len(encoded)} entangled sequences")
        print(f"  Vocabulary: {n_pairs} instrument-pairs × {n_transforms} transforms = {vocab_size}")

    return encoded, vocab_info


def run_entangled_meta_repair(
    encoded_sequences: List[List[int]],
    vocab_size: int,
    device: str = 'cuda',
    max_rules: int = 500,
    min_frequency: int = 3,
    verbose: bool = True
) -> Dict:
    """Run Re-Pair on entangled sequences to find recurring cross-instrument patterns."""
    from grammar.v2.repair_gpu_v2 import build_repair_grammar_v2

    if len(encoded_sequences) < 10:
        if verbose:
            print(f"  Too few entangled sequences ({len(encoded_sequences)}) for meta-pattern discovery")
        return {'rules': {}, 'n_rules': 0}

    total_tokens = sum(len(s) for s in encoded_sequences)
    if verbose:
        print(f"  Entangled Re-Pair: {len(encoded_sequences)} sequences, {total_tokens} tokens")

    grammar = build_repair_grammar_v2(
        encoded_sequences,
        device=device,
        max_rules=max_rules,
        min_pair_count=min_frequency,
        verbose=verbose,
    )

    return {
        'rules': grammar.get('rules', {}),
        'n_rules': len(grammar.get('rules', {})),
    }


# GM program number to instrument name (subset)
GM_INSTRUMENTS = {
    # Piano (0-7)
    0: 'Piano', 1: 'Bright Piano', 2: 'Electric Grand', 3: 'Honky-tonk',
    4: 'Electric Piano 1', 5: 'Electric Piano 2', 6: 'Harpsichord', 7: 'Clavinet',
    # Chromatic Percussion (8-15)
    8: 'Celesta', 9: 'Glockenspiel', 10: 'Music Box', 11: 'Vibraphone',
    12: 'Marimba', 13: 'Xylophone', 14: 'Tubular Bells', 15: 'Dulcimer',
    # Organ (16-23)
    16: 'Drawbar Organ', 17: 'Percussive Organ', 18: 'Rock Organ', 19: 'Church Organ',
    20: 'Reed Organ', 21: 'Accordion', 22: 'Harmonica', 23: 'Tango Accordion',
    # Guitar (24-31)
    24: 'Nylon Guitar', 25: 'Steel Guitar', 26: 'Jazz Guitar', 27: 'Clean Guitar',
    28: 'Muted Guitar', 29: 'Overdrive Guitar', 30: 'Distortion Guitar', 31: 'Guitar Harmonics',
    # Bass (32-39)
    32: 'Acoustic Bass', 33: 'Electric Bass', 34: 'Pick Bass', 35: 'Fretless Bass',
    36: 'Slap Bass 1', 37: 'Slap Bass 2', 38: 'Synth Bass 1', 39: 'Synth Bass 2',
    # Strings (40-47)
    40: 'Violin', 41: 'Viola', 42: 'Cello', 43: 'Contrabass', 44: 'Tremolo Strings',
    45: 'Pizzicato Strings', 46: 'Orchestral Harp', 47: 'Timpani',
    # Ensemble (48-55)
    48: 'String Ensemble', 49: 'Slow Strings', 50: 'Synth Strings 1', 51: 'Synth Strings 2',
    52: 'Choir Aahs', 53: 'Voice Oohs', 54: 'Synth Voice', 55: 'Orchestra Hit',
    # Brass (56-63)
    56: 'Trumpet', 57: 'Trombone', 58: 'Tuba', 59: 'Muted Trumpet',
    60: 'French Horn', 61: 'Brass Section', 62: 'Synth Brass 1', 63: 'Synth Brass 2',
    # Reed (64-71)
    64: 'Soprano Sax', 65: 'Alto Sax', 66: 'Tenor Sax', 67: 'Baritone Sax',
    68: 'Oboe', 69: 'English Horn', 70: 'Bassoon', 71: 'Clarinet',
    # Pipe (72-79)
    72: 'Piccolo', 73: 'Flute', 74: 'Recorder', 75: 'Pan Flute',
    76: 'Blown Bottle', 77: 'Shakuhachi', 78: 'Whistle', 79: 'Ocarina',
    # Synth Lead (80-87)
    80: 'Lead Square', 81: 'Lead Sawtooth', 82: 'Lead Calliope', 83: 'Lead Chiff',
    84: 'Lead Charang', 85: 'Lead Voice', 86: 'Lead Fifths', 87: 'Lead Bass+Lead',
    # Synth Pad (88-95)
    88: 'Pad New Age', 89: 'Pad Warm', 90: 'Pad Polysynth', 91: 'Pad Choir',
    92: 'Pad Bowed', 93: 'Pad Metallic', 94: 'Pad Halo', 95: 'Pad Sweep',
    # Synth Effects (96-103)
    96: 'FX Rain', 97: 'FX Soundtrack', 98: 'FX Crystal', 99: 'FX Atmosphere',
    100: 'FX Brightness', 101: 'FX Goblins', 102: 'FX Echoes', 103: 'FX Sci-Fi',
    # Ethnic (104-111)
    104: 'Sitar', 105: 'Banjo', 106: 'Shamisen', 107: 'Koto',
    108: 'Kalimba', 109: 'Bagpipe', 110: 'Fiddle', 111: 'Shanai',
    # Percussive (112-119)
    112: 'Tinkle Bell', 113: 'Agogo', 114: 'Steel Drums', 115: 'Woodblock',
    116: 'Taiko Drum', 117: 'Melodic Tom', 118: 'Synth Drum', 119: 'Reverse Cymbal',
    # Sound Effects (120-127)
    120: 'Guitar Fret Noise', 121: 'Breath Noise', 122: 'Seashore', 123: 'Bird Tweet',
    124: 'Telephone Ring', 125: 'Helicopter', 126: 'Applause', 127: 'Gunshot',
}


def decode_entangled_meta_pattern(
    rule_expansion: List[int],
    vocab_info: Dict,
    transform_names: List[str]
) -> str:
    """Convert entangled meta-pattern to human-readable form."""
    n_transforms = vocab_info['n_transforms']
    id_to_pair = vocab_info['id_to_pair']

    def instrument_name(gm_prog: int) -> str:
        return GM_INSTRUMENTS.get(gm_prog, f"GM{gm_prog}")

    parts = []
    for token in rule_expansion:
        pair_id = token // n_transforms
        transform = token % n_transforms

        if pair_id in id_to_pair:
            src_inst, tgt_inst = id_to_pair[pair_id]
            t_name = transform_names[transform] if transform < len(transform_names) else f"T{transform}"
            parts.append(f"{instrument_name(src_inst)}→{instrument_name(tgt_inst)}:{t_name}")
        else:
            parts.append(f"?{token}?")

    return ' | '.join(parts)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_level3_discovery(
    checkpoint_path: str,
    output_path: str = None,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict:
    """
    Full Level 3 pipeline: checkpoint → meta-patterns.
    """
    import time
    start_time = time.time()

    if verbose:
        print("=" * 60)
        print("LEVEL 3: META-PATTERN DISCOVERY")
        print("=" * 60)

    # Load checkpoint
    if verbose:
        print("\n[Step 1] Loading checkpoint...")

    data = np.load(checkpoint_path, allow_pickle=True)

    # Handle JSON-encoded patterns
    if 'canonical_patterns_json' in data:
        patterns_raw = data['canonical_patterns_json']
        if isinstance(patterns_raw, np.ndarray) and len(patterns_raw) == 1:
            patterns = json.loads(patterns_raw[0])
        else:
            patterns = [json.loads(p) if isinstance(p, str) else p for p in patterns_raw]
    else:
        patterns = list(data.get('patterns', []))

    # Handle transform vocabulary
    if 'transform_vocabulary_json' in data:
        transforms_raw = data['transform_vocabulary_json']
        if isinstance(transforms_raw, np.ndarray) and len(transforms_raw) == 1:
            transform_vocab = json.loads(transforms_raw[0])
        else:
            transform_vocab = [json.loads(t) if isinstance(t, str) else t for t in transforms_raw]
    else:
        transform_vocab = list(data.get('transforms', []))

    if verbose:
        print(f"  Loaded {len(patterns)} patterns, {len(transform_vocab)} transforms")

    if len(patterns) == 0 or len(transform_vocab) == 0:
        print("  ERROR: No patterns or transforms found in checkpoint")
        return {'meta_rules': {}, 'interpreted': [], 'sequences': []}

    # Build transform lookup table
    if verbose:
        print("\n[Step 2] Building GPU transform lookup table...")

    lookup_start = time.time()
    transform_lookup = build_transform_lookup_gpu(patterns, transform_vocab, device)

    if verbose:
        print(f"  Lookup table: {transform_lookup.shape}, time: {time.time() - lookup_start:.1f}s")

    # Extract transform sequences
    if verbose:
        print("\n[Step 3] Extracting transform sequences per piece...")

    sequences = extract_transform_sequences_gpu(patterns, transform_lookup, device=device)

    if verbose:
        print(f"  Extracted {len(sequences)} sequences")
        if sequences:
            avg_len = sum(len(s.transforms) for s in sequences) / len(sequences)
            print(f"  Average sequence length: {avg_len:.1f} transforms")

    # Run meta Re-Pair
    if verbose:
        print("\n[Step 4] Running GPU Re-Pair on transform sequences...")

    meta_result = run_meta_repair_gpu(
        sequences,
        n_transforms=len(transform_vocab),
        device=device,
        verbose=verbose
    )

    # Interpret meta-patterns
    if verbose:
        print("\n[Step 5] Interpreting meta-patterns...")

    interpreted = interpret_meta_patterns(
        meta_result['rules'],
        transform_vocab,
        verbose=verbose
    )

    # Save results
    if output_path:
        if verbose:
            print(f"\n[Step 6] Saving to {output_path}...")

        np.savez(
            output_path,
            meta_rules=json.dumps(meta_result['rules']),
            interpreted_patterns=json.dumps(interpreted),
            transform_sequences=json.dumps([{
                'piece_id': s.piece_id,
                'track_id': s.track_id,
                'transforms': s.transforms,
            } for s in sequences]),
            compression_ratio=np.array([meta_result.get('compression_ratio', 0)]),
        )

    total_time = time.time() - start_time

    if verbose:
        print("\n" + "=" * 60)
        print("LEVEL 3 COMPLETE")
        print("=" * 60)
        print(f"  Meta-patterns discovered: {len(meta_result['rules'])}")
        print(f"  Transform sequences: {len(sequences)}")
        print(f"  Compression ratio: {meta_result.get('compression_ratio', 0):.2f}x")
        print(f"  Total time: {total_time:.1f}s")

    return {
        'meta_rules': meta_result['rules'],
        'interpreted': interpreted,
        'sequences': sequences,
        'compression_ratio': meta_result.get('compression_ratio', 0),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Level 3 Meta-Pattern Discovery")
    parser.add_argument("--checkpoint", required=True, help="Path to Level 2 checkpoint")
    parser.add_argument("--output", default="checkpoint_level3.npz", help="Output path")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    run_level3_discovery(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
    )
