#!/usr/bin/env python3
"""
Contour-Normalized Re-Pair Pipeline (v50)
==========================================

TRUE T-normalization: patterns ARE contours (interval sequences).

Key difference from v49:
- v49: Pattern = (pitch_class_A, pitch_class_B) - different pairs are different patterns
- v50: Pattern = (interval, rhythm, velocity) - all pairs with same contour are ONE pattern

This achieves optimal compression because (C,D) and (E,F#) are instances
of the same [+2] pattern with different pitch_offset values.

The equation:
  M = Pattern(contour) × T(pitch_offset) × O(octave_offset)

Usage:
    python scripts/run_contour_pipeline.py /path/to/midi/corpus --output checkpoint_v50.npz
"""

import os
import sys
import time
import json
import glob
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List, Dict

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_factored_pipeline import (
    FactoredTrack,
    load_midi_factored,
    FactoredPattern,
    PatternOccurrence,
    bucket_duration,
    run_factored_mdl_transform_discovery,
    extract_factored_canonical_patterns,
)

# Import the v3 contour-normalized Re-Pair
from grammar.v3.repair_contour_normalized import (
    build_contour_normalized_grammar,
    ContourNormalizedGrammar,
)

# Level 3: Meta-pattern discovery
from scripts.level3_meta_patterns import (
    build_transform_lookup_gpu,
    extract_transform_sequences_gpu,
    run_meta_repair_gpu,
    interpret_meta_patterns,
    parse_transform_name,
    aggregate_orchestration_rules,
)

# Multi-factor transform discovery (τ, v, d)
from discovery.multi_factor_transforms import (
    run_multi_factor_discovery,
    FactorTransform,
    FactorType,
)

# TrackDerive: Cross-track derivation (arrangement patterns)
from discovery.track_derive import (
    run_track_derive_discovery,
)

# Interval Magnitude: Dual chromatic/diatonic transform representation
from discovery.interval_magnitude import (
    run_interval_magnitude_discovery,
)

# Feature Importance: MDL-based conditioning variable discovery
from discovery.feature_importance import (
    run_feature_importance_discovery,
)

# Octave Equivalence: MDL-based test for octave equivalence
from discovery.octave_equivalence import (
    run_octave_equivalence_discovery,
)


def convert_contour_grammar_to_rules(
    grammar: ContourNormalizedGrammar,
    tracks: list,
    verbose: bool = True
) -> dict:
    """
    Convert ContourNormalizedGrammar to rules dictionary format.

    Key difference from v2 converter:
    - Rules are defined by contour (interval, rhythm, velocity), not pitch-class pairs
    - Each occurrence has pitch_offset (starting pitch class)
    - Patterns with same contour are already merged
    """
    rules = {}
    n_terminals = grammar.n_terminals

    # Build lookups from track_info
    track_to_piece = {}
    track_to_midi_id = {}
    track_to_gm = {}
    for i, info in enumerate(grammar.track_info):
        track_to_piece[i] = info.get('piece_id', f'track_{i}')
        track_to_midi_id[i] = info.get('track_id', i)
        track_to_gm[i] = info.get('gm_program', 0)

    # Expand each rule to get pitch classes
    # Rules start at n_terminals + 1 (after separator)
    memo = {}
    for rule_idx in range(grammar.n_rules):
        rule_id = n_terminals + 1 + rule_idx  # +1 for separator (rule 0 is n_terminals+1=13)

        # Get contour definition
        interval, rhythm_bucket, velocity_bucket = grammar.get_rule_contour(rule_id)
        count = grammar.rule_counts[rule_idx].item()

        # Get children for hierarchical
        left_child, right_child = grammar.get_rule_children(rule_id)
        is_hierarchical = left_child >= 0

        # Expand to get interval sequence
        intervals = grammar.expand_rule(rule_id, memo)

        # Build pitch_intervals
        if is_hierarchical:
            # Combine children's intervals + connector
            left_intervals = rules.get(str(left_child), {}).get('pitch_intervals', [])
            right_intervals = rules.get(str(right_child), {}).get('pitch_intervals', [])
            pitch_intervals = list(left_intervals) + [interval] + list(right_intervals)
        else:
            # Base 2-note pattern
            pitch_intervals = [interval]

        # Compute pitch_classes from a canonical occurrence (first one)
        # For contour patterns, we use pitch_offset to determine starting pitch
        canonical_pitches = []
        if grammar.rule_occurrences and rule_id in grammar.rule_occurrences:
            occurrences = grammar.rule_occurrences[rule_id]
            if occurrences:
                first_occ = occurrences[0]
                first_pitch = first_occ.get('first_pitch', 60)
                # Build pitch sequence from intervals
                pitches = [first_pitch]
                for iv in pitch_intervals:
                    pitches.append(pitches[-1] + iv)
                canonical_pitches = pitches

        # Derive pitch_classes from canonical_pitches
        pitch_classes = [p % 12 for p in canonical_pitches] if canonical_pitches else []

        rules[str(rule_id)] = {
            'pitch_classes': pitch_classes,
            'pitch_intervals': pitch_intervals,
            'canonical_pitches': canonical_pitches,
            'rhythm_bucket': rhythm_bucket,
            'velocity_bucket': velocity_bucket,
            'rhythm_ratios': [],
            'duration_ratios': [],
            'velocity_ratios': [],
            'count': count,
            'occurrences': [],
            'is_hierarchical': is_hierarchical,
            'left_child': left_child,
            'right_child': right_child,
            'connector_interval': interval if is_hierarchical else 0,
            # v3 specific: contour definition
            'contour': (interval, rhythm_bucket, velocity_bucket),
        }

    # Add occurrences with pitch_offset
    if grammar.rule_occurrences:
        for rule_id, occurrences in grammar.rule_occurrences.items():
            rule_str = str(rule_id)
            if rule_str not in rules:
                continue

            for occ in occurrences:
                track_idx = occ.get('track_idx', -1)
                orig_pos = occ.get('orig_pos', -1)

                if track_idx < 0 or orig_pos < 0:
                    continue

                piece_id = track_to_piece.get(track_idx, f'track_{track_idx}')
                midi_track_id = track_to_midi_id.get(track_idx, track_idx)
                gm_program = track_to_gm.get(track_idx, 0)

                occ_data = {
                    'piece_id': piece_id,
                    'track_id': midi_track_id,
                    'gm_program': gm_program,
                    'onset_time': occ.get('onset', 0),
                    'position': orig_pos,
                    # v3 key addition: pitch_offset (0-11)
                    'pitch_offset': occ.get('pitch_offset', 0),
                    'octave': occ.get('octave', 4),
                    'first_pitch': occ.get('first_pitch', 60),
                }

                # Timing offsets
                if occ.get('ioi', 0) > 0:
                    occ_data['tau_offset'] = occ['ioi']
                if occ.get('duration', 0) > 0:
                    occ_data['duration_offset'] = occ['duration']
                if occ.get('velocity', 0) > 0:
                    occ_data['v_offset'] = occ['velocity']

                # Compute transform from canonical
                canonical_pitches = rules[rule_str].get('canonical_pitches', [])
                if canonical_pitches and 'first_pitch' in occ:
                    canonical_first = canonical_pitches[0]
                    occ_first = occ['first_pitch']
                    full_offset = occ_first - canonical_first
                    occ_data['octave_transform'] = (full_offset // 12) * 12
                    occ_data['pitch_offset'] = full_offset % 12

                rules[rule_str]['occurrences'].append(occ_data)

        if verbose:
            print(f"  Using inline occurrence tracking with pitch_offset")

    # Fill in timing ratios from occurrences
    for rule_str, rule in rules.items():
        pattern_length = len(rule['pitch_intervals']) + 1  # intervals + 1 = notes
        if not rule['rhythm_ratios']:
            rule['rhythm_ratios'] = [1.0] * max(1, pattern_length - 1)
        if not rule['duration_ratios']:
            rule['duration_ratios'] = [1.0] * pattern_length
        if not rule['velocity_ratios']:
            rule['velocity_ratios'] = [1.0] * pattern_length

    # Stats
    if verbose:
        n_hier = sum(1 for r in rules.values() if r['is_hierarchical'])
        total_occ = sum(len(r['occurrences']) for r in rules.values())

        # Count unique pitch_offsets per rule (shows merging)
        merged_rules = 0
        for r in rules.values():
            offsets = set(o.get('pitch_offset', 0) for o in r['occurrences'])
            if len(offsets) > 1:
                merged_rules += 1

        print(f"  Converted {len(rules)} rules ({n_hier} hierarchical)")
        print(f"  Total tracked occurrences: {total_occ:,}")
        print(f"  Rules with merged pitch_offsets: {merged_rules} (TRUE T-normalization)")

    return {
        'rules': rules,
        'n_rules': len(rules),
    }


def convert_normalized_to_factored_patterns(
    normalized_rules: dict,
    tracks: list,
) -> list:
    """Convert normalized rules to FactoredPattern objects."""
    patterns = []

    for rule_id, rule_data in normalized_rules.items():
        occurrences = [
            PatternOccurrence(
                piece_id=occ['piece_id'],
                track_id=occ['track_id'],
                onset_time=occ['onset_time'],
                position=occ['position'],
                tau_offset=occ.get('tau_offset', 480),
                duration_offset=occ.get('duration_offset', 480),
                v_offset=occ.get('v_offset', 4),
            )
            for occ in rule_data.get('occurrences', [])
        ]

        rhythm_ratios = rule_data.get('rhythm_ratios', [])
        duration_ratios = rule_data.get('duration_ratios', [])
        velocity_ratios = rule_data.get('velocity_ratios', [])

        rhythm_ioi = rule_data.get('rhythm_ioi', [])
        if not rhythm_ioi:
            length = len(rule_data.get('pitch_classes', []))
            if rhythm_ratios and len(rhythm_ratios) == length - 1:
                rhythm_ioi = [int(r * 480) for r in rhythm_ratios]
            else:
                rhythm_ioi = [480] * max(0, length - 1)

        pattern = FactoredPattern(
            pattern_id=int(rule_id) if rule_id.isdigit() else hash(rule_id) % 100000,
            rule_id=rule_id,
            pitch_classes=rule_data.get('pitch_classes', []),
            octaves=rule_data.get('octaves', [4] * len(rule_data.get('pitch_classes', []))),
            velocities=rule_data.get('velocities', [4] * len(rule_data.get('pitch_classes', []))),
            durations=rule_data.get('duration_buckets', [3] * len(rule_data.get('pitch_classes', []))),
            rhythm_ioi=rhythm_ioi,
            occurrences=occurrences,
        )

        pattern._rhythm_ratios = rhythm_ratios
        pattern._duration_ratios = duration_ratios
        pattern._velocity_ratios = velocity_ratios

        patterns.append(pattern)

    return patterns


def save_normalized_checkpoint(
    path: str,
    grammar_result: dict,
    tracks: list,
    stats: dict,
    transform_discovery: dict = None,
    meta_patterns: dict = None,
    multi_factor_discovery: dict = None,
    track_derive_discovery: dict = None,
    feature_importance_discovery: dict = None,
    verbose: bool = True
):
    """Save checkpoint with full normalized pattern data."""

    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    rules_data = convert_numpy(grammar_result['rules'])

    track_info = []
    for track in tracks:
        track_info.append({
            'piece_id': track.piece_id,
            'track_id': track.track_id,
            'length': len(track),
            'gm_program': track.gm_program,
            'is_drum': track.is_drum,
        })

    base_path = path.replace('.npz', '')
    patterns_path = f'{base_path}_patterns.json'
    track_info_path = f'{base_path}_track_info.json'

    if verbose:
        print(f"  Saving patterns JSON to {patterns_path}...")

    with open(patterns_path, 'w') as f:
        json.dump(rules_data, f)

    with open(track_info_path, 'w') as f:
        json.dump(track_info, f)

    data = {
        'version': np.array(['contour_normalized_v50']),

        'n_files': np.array([stats.get('n_files', 0)]),
        'n_tracks': np.array([len(tracks)]),
        'n_notes': np.array([sum(len(t) for t in tracks)]),
        'n_patterns': np.array([len(grammar_result['rules'])]),
        'n_transform_vocabulary': np.array([stats.get('n_transform_vocabulary', 0)]),
        'total_time': np.array([stats.get('total_time', 0)]),

        'normalization': np.array(['contour_T+tau+v']),

        'patterns_json_file': np.array([os.path.basename(patterns_path)]),
        'track_info_json_file': np.array([os.path.basename(track_info_path)]),
    }

    if transform_discovery:
        transform_data = convert_numpy(transform_discovery)
        transforms_path = f'{base_path}_transforms.json'
        with open(transforms_path, 'w') as f:
            json.dump(transform_data, f)
        data['transforms_json_file'] = np.array([os.path.basename(transforms_path)])

    if multi_factor_discovery:
        mf_data = convert_numpy({
            k: v for k, v in multi_factor_discovery.items()
            if k != 'vocabulary_objects'
        })
        multi_factor_path = f'{base_path}_multi_factor.json'
        with open(multi_factor_path, 'w') as f:
            json.dump(mf_data, f)
        data['multi_factor_json_file'] = np.array([os.path.basename(multi_factor_path)])
        data['n_factor_vocabulary'] = np.array([len(multi_factor_discovery.get('vocabulary', []))])
        data['n_rhythm_transforms'] = np.array([multi_factor_discovery.get('rhythm_transforms', 0)])
        data['n_velocity_transforms'] = np.array([multi_factor_discovery.get('velocity_transforms', 0)])
        data['n_duration_transforms'] = np.array([multi_factor_discovery.get('duration_transforms', 0)])

    if track_derive_discovery and track_derive_discovery.get('n_derives', 0) > 0:
        td_data = convert_numpy({
            'derives_json': track_derive_discovery.get('derives_json', []),
            'n_derives': track_derive_discovery.get('n_derives', 0),
            'derives_by_transform': track_derive_discovery.get('derives_by_transform', {}),
            'leader_instruments': track_derive_discovery.get('leader_instruments', {}),
        })
        track_derives_path = f'{base_path}_track_derives.json'
        with open(track_derives_path, 'w') as f:
            json.dump(td_data, f)
        data['track_derives_json_file'] = np.array([os.path.basename(track_derives_path)])
        data['n_track_derives'] = np.array([track_derive_discovery.get('n_derives', 0)])

        if verbose:
            print(f"  Saved {track_derive_discovery.get('n_derives', 0)} TrackDerive relations")

    if feature_importance_discovery and feature_importance_discovery.get('useful_features'):
        fi_data = convert_numpy({
            'useful_features': feature_importance_discovery.get('useful_features', []),
            'feature_gains': feature_importance_discovery.get('feature_gains', {}),
            'feature_clusters': feature_importance_discovery.get('feature_clusters', {}),
            'feature_shifts': feature_importance_discovery.get('feature_shifts', {}),
        })
        feature_importance_path = f'{base_path}_feature_importance.json'
        with open(feature_importance_path, 'w') as f:
            json.dump(fi_data, f)
        data['feature_importance_json_file'] = np.array([os.path.basename(feature_importance_path)])
        data['n_useful_features'] = np.array([len(feature_importance_discovery.get('useful_features', []))])

        if verbose:
            useful = feature_importance_discovery.get('useful_features', [])
            print(f"  Saved {len(useful)} useful features: {', '.join(useful)}")

    piece_prefs = stats.get('piece_interval_preferences', {})
    if piece_prefs:
        prefs_path = f'{base_path}_interval_prefs.json'
        with open(prefs_path, 'w') as f:
            json.dump(convert_numpy(piece_prefs), f)
        data['interval_prefs_json_file'] = np.array([os.path.basename(prefs_path)])
        data['preferred_interval_repr'] = np.array([stats.get('preferred_interval_repr', 'chromatic')])

    if meta_patterns:
        meta_data = convert_numpy(meta_patterns)
        if 'orchestration_rules' in meta_data and len(meta_data.get('orchestration_rules', [])) > 1000:
            orch_path = f'{base_path}_orchestration.json'
            with open(orch_path, 'w') as f:
                json.dump(meta_data['orchestration_rules'], f)
            data['orchestration_json_file'] = np.array([os.path.basename(orch_path)])
            meta_data_small = {k: v for k, v in meta_data.items() if k != 'orchestration_rules'}
            meta_data_small['orchestration_rules_external'] = True
        else:
            meta_data_small = meta_data

        meta_json = json.dumps(meta_data_small)
        if len(meta_json) < 100_000_000:
            data['meta_patterns_json'] = np.array([meta_json])
        else:
            meta_path = f'{base_path}_meta.json'
            with open(meta_path, 'w') as f:
                json.dump(meta_data_small, f)
            data['meta_patterns_json_file'] = np.array([os.path.basename(meta_path)])

        if 'n_orchestration_rules' in meta_patterns:
            data['n_orchestration_rules'] = np.array([meta_patterns['n_orchestration_rules']])
        if 'n_vertical_slices' in meta_patterns:
            data['n_vertical_slices'] = np.array([meta_patterns['n_vertical_slices']])

    np.savez_compressed(path, **data)

    if verbose:
        print(f"\n  Checkpoint saved to: {path}")
        print(f"  Patterns JSON: {patterns_path}")
        print(f"  Size: {Path(path).stat().st_size / 1024 / 1024:.2f}MB")


def run_contour_pipeline(
    corpus_path: str,
    output_path: str = 'checkpoint_contour.npz',
    max_files: int = 500,
    min_length: int = 3,
    max_length: int = 32,
    min_count: int = 2,
    max_rules: int = 20000,
    verbose: bool = True
) -> dict:
    """
    Run the contour-normalized pattern discovery pipeline.

    This uses TRUE T-normalization where patterns ARE contours.
    """
    total_start = time.time()
    stats = {}

    if verbose:
        print("=" * 70)
        print("CONTOUR-NORMALIZED PATTERN DISCOVERY PIPELINE (v50)")
        print("=" * 70)
        print(f"Corpus: {corpus_path}")
        print(f"Max files: {max_files}")
        print(f"Length range: {min_length}-{max_length}")
        print(f"Min count: {min_count}")
        print()

    # =========================================================================
    # PHASE 1: Load MIDI files
    # =========================================================================
    phase_start = time.time()
    if verbose:
        print("[Phase 1] Loading MIDI files...", flush=True)

    midi_files = sorted(glob.glob(str(Path(corpus_path) / "*.mid")))
    midi_files += sorted(glob.glob(str(Path(corpus_path) / "*.midi")))
    midi_files += sorted(glob.glob(str(Path(corpus_path) / "**/*.mid"), recursive=True))
    midi_files += sorted(glob.glob(str(Path(corpus_path) / "**/*.midi"), recursive=True))
    midi_files = list(dict.fromkeys(midi_files))

    if max_files:
        midi_files = midi_files[:max_files]

    if verbose:
        print(f"  Found {len(midi_files)} MIDI files", flush=True)

    all_tracks = []
    n_files_loaded = 0
    n_files_failed = 0
    num_workers = min(8, max(1, len(midi_files)))

    batch_size = 50
    total_batches = (len(midi_files) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(midi_files))
        batch_files = midi_files[batch_start:batch_end]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_midi_factored, f): f for f in batch_files}

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    if result:
                        all_tracks.extend(result)
                        n_files_loaded += 1
                    else:
                        n_files_failed += 1
                except Exception:
                    n_files_failed += 1

        if verbose:
            elapsed = time.time() - phase_start
            print(f"    [{batch_end}/{len(midi_files)}] {n_files_loaded} loaded, "
                  f"{len(all_tracks)} tracks ({elapsed:.1f}s)", flush=True)

    stats['n_files'] = n_files_loaded
    stats['n_files_failed'] = n_files_failed
    stats['n_tracks'] = len(all_tracks)
    stats['n_notes'] = sum(len(t) for t in all_tracks)
    stats['phase1_time'] = time.time() - phase_start

    if verbose:
        print(f"  Loaded {n_files_loaded} files, {len(all_tracks)} tracks, "
              f"{stats['n_notes']:,} notes", flush=True)

    # =========================================================================
    # PHASE 2: Run Contour-Normalized Re-Pair
    # =========================================================================
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 2] Running Contour-Normalized Re-Pair (TRUE T-normalization)...", flush=True)

    # Extract sequences from tracks
    pitch_sequences = []
    ioi_sequences = []
    velocity_sequences = []
    duration_sequences = []
    onset_sequences = []
    octave_sequences = []
    track_info_list = []

    for track in all_tracks:
        pitch_sequences.append(track.pitch_classes.tolist())
        # rhythm_ioi is (N-1,), need to pad
        iois = track.rhythm_ioi.tolist()
        iois.append(480)  # Pad last note
        ioi_sequences.append(iois)
        velocity_sequences.append(track.velocities.tolist())
        duration_sequences.append(track.durations.tolist() if hasattr(track, 'durations') else [480] * len(track))
        onset_sequences.append(track.onsets.tolist() if hasattr(track, 'onsets') else [i * 480 for i in range(len(track))])
        octave_sequences.append(track.octaves.tolist() if hasattr(track, 'octaves') else [4] * len(track))
        track_info_list.append({
            'piece_id': track.piece_id,
            'track_id': track.track_id,
            'gm_program': getattr(track, 'gm_program', 0),
            'is_drum': getattr(track, 'is_drum', False),
        })

    # Run v3 contour-normalized Re-Pair
    grammar = build_contour_normalized_grammar(
        pitch_sequences=pitch_sequences,
        ioi_sequences=ioi_sequences,
        velocity_sequences=velocity_sequences,
        track_info=track_info_list,
        duration_sequences=duration_sequences,
        onset_sequences=onset_sequences,
        octave_sequences=octave_sequences,
        device='cuda',
        min_pair_count=min_count,
        max_rules=max_rules,
        verbose=verbose
    )

    # Convert grammar to rules dict format
    grammar_result = convert_contour_grammar_to_rules(grammar, all_tracks, verbose)

    stats['n_patterns'] = grammar_result['n_rules']
    stats['compression'] = grammar.compression_ratio()
    stats['phase2_time'] = time.time() - phase_start

    if verbose:
        print(f"  Compression: {stats['compression']:.2f}x", flush=True)

    # =========================================================================
    # PHASE 3: Analyze patterns
    # =========================================================================
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 3] Analyzing patterns...", flush=True)

    rules = grammar_result['rules']
    total_occurrences = sum(r['count'] for r in rules.values())

    multi_piece = sum(
        1 for r in rules.values()
        if len(set(o['piece_id'] for o in r['occurrences'])) > 1
    )

    all_pieces = set()
    for r in rules.values():
        for occ in r['occurrences']:
            all_pieces.add(occ['piece_id'])

    length_dist = defaultdict(int)
    for r in rules.values():
        length_dist[len(r['pitch_intervals'])] += 1

    top_patterns = sorted(rules.items(), key=lambda x: x[1]['count'], reverse=True)[:20]

    stats['total_occurrences'] = total_occurrences
    stats['multi_piece_patterns'] = multi_piece
    stats['pieces_covered'] = len(all_pieces)
    stats['phase3_time'] = time.time() - phase_start

    if verbose:
        print(f"  Total occurrences: {total_occurrences:,}")
        print(f"  Multi-piece patterns: {multi_piece}")
        print(f"  Pieces covered: {len(all_pieces)}")
        print(f"\n  Top 10 patterns (by contour):")
        for rule_id, rule in top_patterns[:10]:
            intervals = rule['pitch_intervals']
            count = rule['count']
            n_pieces = len(set(o['piece_id'] for o in rule['occurrences']))
            n_offsets = len(set(o.get('pitch_offset', 0) for o in rule['occurrences']))
            contour = rule.get('contour', intervals)
            print(f"    R{rule_id}: contour={contour} intervals={intervals[:5]}... "
                  f"count={count}, pieces={n_pieces}, pitch_offsets={n_offsets}")

    # =========================================================================
    # PHASE 4: Convert to FactoredPattern for MDL Transform Discovery
    # =========================================================================
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 4] Extracting canonical patterns for transform discovery...", flush=True)

    canonicals = convert_normalized_to_factored_patterns(grammar_result['rules'], all_tracks)
    stats['n_canonical_patterns'] = len(canonicals)
    stats['phase4_time'] = time.time() - phase_start

    if verbose:
        print(f"  Extracted {len(canonicals)} canonical patterns", flush=True)

    # =========================================================================
    # PHASE 5: MDL Transform Discovery
    # =========================================================================
    phase_start = time.time()
    transform_discovery = {'vocabulary': [], 'n_transforms': 0}

    if len(canonicals) > 10:
        if verbose:
            print(f"\n[Phase 5] MDL Transform Discovery on {len(canonicals)} patterns...", flush=True)

        try:
            transform_discovery = run_factored_mdl_transform_discovery(
                canonicals,
                max_depth=2,
                min_frequency=3,
                verbose=verbose
            )
            stats['n_transform_vocabulary'] = len(transform_discovery.get('vocabulary', []))
            if verbose:
                print(f"  Discovered {stats['n_transform_vocabulary']} transforms", flush=True)
        except Exception as e:
            if verbose:
                print(f"  Transform discovery failed: {e}", flush=True)
            stats['n_transform_vocabulary'] = 0
    else:
        if verbose:
            print(f"\n[Phase 5] Skipping transform discovery (need >10 patterns)", flush=True)
        stats['n_transform_vocabulary'] = 0

    stats['phase5_time'] = time.time() - phase_start

    # =========================================================================
    # PHASE 5b: Multi-Factor Transform Discovery (τ, v, d)
    # =========================================================================
    phase_start = time.time()
    multi_factor_discovery = {'vocabulary': [], 'rhythm_transforms': 0, 'velocity_transforms': 0, 'duration_transforms': 0}

    if len(canonicals) > 10:
        if verbose:
            print(f"\n[Phase 5b] Multi-Factor Transform Discovery (τ, v, d)...", flush=True)

        try:
            # Convert canonicals to dict format for multi-factor discovery
            patterns_for_mf = [p.to_dict() if hasattr(p, 'to_dict') else {
                'pitch_classes': p.pitch_classes,
                'rhythm_ratios': getattr(p, '_rhythm_ratios', None) or p.rhythm_ratios,
                'velocity_ratios': getattr(p, '_velocity_ratios', None) or p.velocity_ratios,
                'duration_ratios': getattr(p, '_duration_ratios', None) or [1.0] * len(p.durations),
            } for p in canonicals]

            multi_factor_discovery = run_multi_factor_discovery(
                patterns_for_mf,
                device='cuda',
                tolerance=0.1,
                min_frequency=3,
                verbose=verbose
            )

            stats['n_rhythm_transforms'] = multi_factor_discovery.get('rhythm_transforms', 0)
            stats['n_velocity_transforms'] = multi_factor_discovery.get('velocity_transforms', 0)
            stats['n_duration_transforms'] = multi_factor_discovery.get('duration_transforms', 0)
            stats['n_factor_vocabulary'] = len(multi_factor_discovery.get('vocabulary', []))

            if verbose:
                print(f"  Discovered {stats['n_factor_vocabulary']} factor transforms", flush=True)
                print(f"    τ (rhythm): {stats['n_rhythm_transforms']}", flush=True)
                print(f"    v (velocity): {stats['n_velocity_transforms']}", flush=True)
                print(f"    d (duration): {stats['n_duration_transforms']}", flush=True)

        except Exception as e:
            if verbose:
                print(f"  Multi-factor discovery failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            stats['n_factor_vocabulary'] = 0
    else:
        if verbose:
            print(f"\n[Phase 5b] Skipping multi-factor discovery (need >10 patterns)", flush=True)
        stats['n_factor_vocabulary'] = 0

    stats['phase5b_time'] = time.time() - phase_start

    # =========================================================================
    # PHASE 5c: TrackDerive Discovery (Cross-Track Arrangement Patterns)
    # =========================================================================
    phase_start = time.time()
    track_derive_discovery = {'derives': [], 'n_derives': 0}

    if len(canonicals) > 5:
        if verbose:
            print(f"\n[Phase 5c] TrackDerive Discovery (Cross-Track Arrangements)...", flush=True)

        try:
            # Convert canonicals to dict format with occurrences
            patterns_for_td = []
            for p in canonicals:
                p_dict = p.to_dict() if hasattr(p, 'to_dict') else {
                    'pitch_classes': p.pitch_classes,
                    'rhythm_ratios': getattr(p, '_rhythm_ratios', None) or p.rhythm_ratios,
                    'velocity_ratios': getattr(p, '_velocity_ratios', None) or p.velocity_ratios,
                    'duration_ratios': getattr(p, '_duration_ratios', None) or [1.0] * len(p.durations),
                }
                # Add occurrences from grammar result
                if hasattr(p, 'occurrences'):
                    p_dict['occurrences'] = p.occurrences
                patterns_for_td.append(p_dict)

            # Build track_instruments mapping from all_tracks
            track_instruments = {}
            for track in all_tracks:
                piece_id = getattr(track, 'piece_id', getattr(track, 'source_file', 'unknown'))
                track_id = getattr(track, 'track_id', 0)
                program = getattr(track, 'gm_program', getattr(track, 'program', track_id))
                track_instruments[(piece_id, track_id)] = program

            track_derive_discovery = run_track_derive_discovery(
                patterns_for_td,
                track_instruments=track_instruments if track_instruments else None,
                device='cuda',
                add_to_occurrences=True,
                verbose=verbose
            )

            stats['n_track_derives'] = track_derive_discovery.get('n_derives', 0)
            stats['track_derive_transforms'] = track_derive_discovery.get('derives_by_transform', {})

            if verbose and stats['n_track_derives'] > 0:
                print(f"  Added {stats['n_track_derives']} cross-track derivations to graph", flush=True)

        except Exception as e:
            if verbose:
                print(f"  TrackDerive discovery failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            stats['n_track_derives'] = 0
    else:
        if verbose:
            print(f"\n[Phase 5c] Skipping TrackDerive discovery (need >5 patterns)", flush=True)
        stats['n_track_derives'] = 0

    stats['phase5c_time'] = time.time() - phase_start

    # =========================================================================
    # PHASE 5d: Interval Magnitude Discovery (Diatonic vs Chromatic)
    # =========================================================================
    phase_start = time.time()
    interval_magnitude_discovery = {'preferred_representation': 'chromatic'}

    if len(canonicals) > 10:
        if verbose:
            print(f"\n[Phase 5d] Interval Magnitude Discovery (Diatonic vs Chromatic)...", flush=True)

        try:
            patterns_for_im = [{
                'pitch_classes': p.pitch_classes,
                'occurrences': p.occurrences if hasattr(p, 'occurrences') else [],
            } for p in canonicals]

            interval_magnitude_discovery = run_interval_magnitude_discovery(
                patterns_for_im,
                device='cuda',
                verbose=verbose
            )

            stats['preferred_interval_repr'] = interval_magnitude_discovery.get('preferred_representation', 'chromatic')
            comp = interval_magnitude_discovery.get('compression_comparison', {})
            stats['chromatic_compression'] = comp.get('chromatic', {}).get('compression', 0.0)
            stats['magnitude_compression'] = comp.get('magnitude', {}).get('compression', 0.0)

        except Exception as e:
            if verbose:
                print(f"  Interval magnitude discovery failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            stats['preferred_interval_repr'] = 'chromatic'
    else:
        if verbose:
            print(f"\n[Phase 5d] Skipping interval magnitude discovery (need >10 patterns)", flush=True)
        stats['preferred_interval_repr'] = 'chromatic'

    stats['phase5d_time'] = time.time() - phase_start

    # =========================================================================
    # PHASE 5e: Feature Importance Discovery (MDL-Based)
    # =========================================================================
    phase_start = time.time()
    feature_importance_discovery = {'useful_features': []}

    if len(canonicals) > 20 and stats.get('n_transform_vocabulary', 0) > 0:
        if verbose:
            print(f"\n[Phase 5e] Feature Importance Discovery (MDL-Based)...", flush=True)

        try:
            # Need transform lookup for this phase
            if transform_discovery and 'lookup' in transform_discovery:
                transform_lookup = transform_discovery['lookup']
            else:
                # Build a simple lookup from patterns
                patterns_for_fi = [{
                    'pitch_classes': p.pitch_classes,
                    'occurrences': [{'piece_id': o.piece_id, 'track_id': o.track_id,
                                     'onset_time': o.onset_time, 'pitch_offset': getattr(o, 'pitch_offset', 0)}
                                    for o in p.occurrences] if hasattr(p, 'occurrences') else [],
                } for p in canonicals]

                transform_vocab = transform_discovery.get('vocabulary', []) if transform_discovery else []
                if transform_vocab:
                    transform_lookup = build_transform_lookup_gpu(
                        patterns_for_fi, transform_vocab, device='cuda', verbose=False
                    )
                else:
                    transform_lookup = None

            if transform_lookup is not None:
                patterns_for_fi = [{
                    'pitch_classes': p.pitch_classes,
                    'occurrences': [{'piece_id': o.piece_id, 'track_id': o.track_id,
                                     'onset_time': o.onset_time,
                                     'pitch_offset': getattr(o, 'pitch_offset', 0),
                                     'instrument': getattr(o, 'instrument', o.track_id)}
                                    for o in p.occurrences] if hasattr(p, 'occurrences') else [],
                } for p in canonicals]

                transform_names = transform_discovery.get('vocabulary', []) if transform_discovery else None

                feature_importance_discovery = run_feature_importance_discovery(
                    patterns_for_fi,
                    transform_lookup,
                    transform_names=transform_names,
                    min_gain_bits=50.0,
                    device='cuda',
                    verbose=verbose
                )

                stats['useful_features'] = feature_importance_discovery.get('useful_features', [])
                stats['n_useful_features'] = len(stats['useful_features'])

                # Check if pitch_offset_relative emerged as useful (implies "keys are real")
                if 'pitch_offset_relative' in stats['useful_features']:
                    stats['keys_discovered'] = True
                    if verbose:
                        print(f"  → pitch_offset_relative is useful: 'scale degrees' emerged!", flush=True)
                else:
                    stats['keys_discovered'] = False

        except Exception as e:
            if verbose:
                print(f"  Feature importance discovery failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            stats['useful_features'] = []
            stats['n_useful_features'] = 0
    else:
        if verbose:
            print(f"\n[Phase 5e] Skipping feature importance discovery (need >20 patterns and transforms)", flush=True)
        stats['useful_features'] = []
        stats['n_useful_features'] = 0

    stats['phase5e_time'] = time.time() - phase_start

    # =========================================================================
    # PHASE 5g: Octave Equivalence Discovery (MDL-based)
    # =========================================================================
    phase_start = time.time()
    octave_equivalence_discovery = {'octave_equivalence_useful': False}

    if len(normalized_rules) > 10:
        if verbose:
            print(f"\n[Phase 5g] Octave Equivalence Discovery (MDL-Based)...", flush=True)

        try:
            # Prepare patterns with their pitch intervals
            patterns_for_octave = [
                {
                    'pitch_intervals': rule.get('pitch_intervals', []),
                    'count': rule.get('count', 1),
                }
                for rule in normalized_rules.values()
                if rule.get('pitch_intervals')
            ]

            if patterns_for_octave:
                octave_equivalence_discovery = run_octave_equivalence_discovery(
                    patterns=patterns_for_octave,
                    min_benefit_bits=100.0,
                    verbose=verbose,
                )

                stats['octave_equivalence_useful'] = octave_equivalence_discovery.get('octave_equivalence_useful', False)
                stats['octave_mdl_benefit'] = octave_equivalence_discovery.get('mdl_benefit_bits', 0.0)
                stats['octave_offset_entropy'] = octave_equivalence_discovery.get('octave_offset_entropy', 0.0)
                stats['multi_octave_patterns'] = octave_equivalence_discovery.get('multi_octave_pattern_count', 0)

                if octave_equivalence_discovery.get('octave_equivalence_useful', False):
                    if verbose:
                        print(f"  -> Octave equivalence DISCOVERED: {stats['octave_mdl_benefit']:.1f} bits saved", flush=True)
                else:
                    if verbose:
                        print(f"  -> Octave equivalence not beneficial for this corpus", flush=True)
            else:
                if verbose:
                    print(f"  No patterns with intervals found", flush=True)

        except Exception as e:
            if verbose:
                print(f"  Octave equivalence discovery failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
    else:
        if verbose:
            print(f"\n[Phase 5g] Skipping octave equivalence discovery (need >10 patterns)", flush=True)

    stats['phase5g_time'] = time.time() - phase_start

    # =========================================================================
    # PHASE 6: Level 3 Meta-Pattern Discovery
    # =========================================================================
    phase_start = time.time()
    meta_patterns = {}

    if verbose:
        print(f"\n[Phase 6] Level 3 Meta-Pattern Discovery...", flush=True)

    if stats.get('n_transform_vocabulary', 0) > 0 and len(canonicals) > 10:
        try:
            # Convert canonicals to dict format for Level 3
            patterns_for_l3 = [p.to_dict() if hasattr(p, 'to_dict') else {
                'pitch_classes': p.pitch_classes,
                'octaves': p.octaves,
                'velocities': p.velocities,
                'durations': p.durations,
                'rhythm_ioi': p.rhythm_ioi,
                'occurrences': [{'piece_id': o.piece_id, 'track_id': o.track_id,
                                 'onset_time': o.onset_time, 'position': o.position}
                                for o in p.occurrences],
            } for p in canonicals]

            # Parse transform vocab for Level 3
            transform_vocab_parsed = [
                parse_transform_name(t) if isinstance(t, str) else t
                for t in transform_discovery.get('vocabulary', [])
            ]

            # Step 1: Build transform lookup table
            if verbose:
                print(f"  Building GPU transform lookup table...", flush=True)
            transform_lookup = build_transform_lookup_gpu(
                patterns_for_l3, transform_vocab_parsed, device='cuda'
            )
            if verbose:
                print(f"    Lookup table: {transform_lookup.shape}", flush=True)

            # Step 2: Extract transform sequences
            if verbose:
                print(f"  Extracting transform sequences per piece...", flush=True)
            sequences = extract_transform_sequences_gpu(
                patterns_for_l3, transform_lookup, device='cuda'
            )
            if verbose:
                print(f"    {len(sequences)} sequences extracted", flush=True)

            # Step 3: Run meta Re-Pair
            if len(sequences) >= 10:
                if verbose:
                    print(f"  Running GPU Re-Pair on transform sequences...", flush=True)
                meta_result = run_meta_repair_gpu(
                    sequences,
                    n_transforms=len(transform_vocab_parsed),
                    device='cuda',
                    verbose=verbose
                )

                # Step 4: Interpret
                if meta_result['n_rules'] > 0:
                    interpreted = interpret_meta_patterns(
                        meta_result['rules'],
                        transform_vocab_parsed,
                        verbose=verbose
                    )
                    meta_patterns = {
                        'rules': meta_result['rules'],
                        'interpreted': interpreted,
                        'compression_ratio': meta_result.get('compression_ratio', 0),
                        'n_sequences': len(sequences),
                    }
                    if verbose:
                        print(f"  Discovered {meta_result['n_rules']} horizontal meta-patterns", flush=True)
            else:
                if verbose:
                    print(f"  Too few sequences for meta-pattern discovery ({len(sequences)})", flush=True)

            # Step 5: Orchestration Rule Aggregation (vertical/cross-track relations)
            if verbose:
                print(f"  Aggregating orchestration rules (vertical slices)...", flush=True)

            # Build track instruments lookup
            track_instruments = {}
            for track in all_tracks:
                key = (track.piece_id, track.track_id)
                track_instruments[key] = track.gm_program

            orchestration_result = aggregate_orchestration_rules(
                patterns_for_l3,
                transform_vocab_parsed,
                min_confidence=0.3,
                min_frequency=5,
                verbose=verbose,
                track_instruments=track_instruments,
            )
            if orchestration_result['n_rules'] > 0:
                meta_patterns['orchestration_rules'] = orchestration_result['rules']
                meta_patterns['n_orchestration_rules'] = orchestration_result['n_rules']
                meta_patterns['n_vertical_slices'] = orchestration_result['n_slices']
                if verbose:
                    print(f"  Found {orchestration_result['n_rules']} orchestration rules from {orchestration_result['n_slices']} vertical slices", flush=True)

        except Exception as e:
            if verbose:
                print(f"  Level 3 discovery failed: {e}", flush=True)
                import traceback
                traceback.print_exc()
    else:
        if verbose:
            print(f"  Skipping (need transforms and patterns)", flush=True)

    stats['phase6_time'] = time.time() - phase_start

    # =========================================================================
    # PHASE 7: Save checkpoint
    # =========================================================================
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 7] Saving checkpoint...", flush=True)

    stats['total_time'] = time.time() - total_start

    save_normalized_checkpoint(
        output_path,
        grammar_result,
        all_tracks,
        stats,
        transform_discovery=transform_discovery,
        meta_patterns=meta_patterns,
        multi_factor_discovery=multi_factor_discovery,
        track_derive_discovery=track_derive_discovery,
        feature_importance_discovery=feature_importance_discovery,
        verbose=verbose
    )

    stats['phase7_time'] = time.time() - phase_start

    # Final summary
    if verbose:
        print()
        print("=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  Files: {stats['n_files']} loaded, {stats['n_files_failed']} failed")
        print(f"  Tracks: {stats['n_tracks']}")
        print(f"  Notes: {stats['n_notes']:,}")
        print(f"  Patterns: {stats['n_patterns']}")
        print(f"  Compression: {stats['compression']:.2f}x")
        print(f"  Total occurrences: {stats['total_occurrences']:,}")
        print(f"  Multi-piece patterns: {stats['multi_piece_patterns']}")
        print(f"  Transform vocabulary (pitch): {stats.get('n_transform_vocabulary', 0)}")
        print(f"  Total time: {stats['total_time']:.1f}s")
        print(f"\n  Output: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Contour-Normalized Pattern Discovery (v50)')
    parser.add_argument('corpus', help='Path to MIDI corpus')
    parser.add_argument('--output', '-o', default='checkpoint_contour.npz',
                        help='Output checkpoint path')
    parser.add_argument('--max-files', type=int, default=500,
                        help='Maximum files to process')
    parser.add_argument('--min-length', type=int, default=3,
                        help='Minimum pattern length')
    parser.add_argument('--max-length', type=int, default=32,
                        help='Maximum pattern length')
    parser.add_argument('--min-count', type=int, default=2,
                        help='Minimum occurrence count')
    parser.add_argument('--max-rules', type=int, default=20000,
                        help='Maximum patterns to discover')
    args = parser.parse_args()

    run_contour_pipeline(
        corpus_path=args.corpus,
        output_path=args.output,
        max_files=args.max_files,
        min_length=args.min_length,
        max_length=args.max_length,
        min_count=args.min_count,
        max_rules=args.max_rules,
        verbose=True
    )


if __name__ == '__main__':
    main()
