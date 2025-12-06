#!/usr/bin/env python3
"""
T+τ+v Factored Hierarchical Re-Pair Pipeline
==============================================

This pipeline uses TRUE RE-PAIR ALGORITHM with full T+τ+v normalization.

Patterns are INVARIANT under:
- Transposition (T): C-E-G and D-F#-A are the same pattern [0,4,7]
- Tempo scaling (τ): quarter-eighth-eighth at any tempo is the same rhythm
- Velocity scaling (v): f-mf-p and ff-f-mf are the same contour

Uses GPU-accelerated hierarchical Re-Pair:
- Iteratively finds most frequent T+τ+v normalized pairs
- Creates hierarchical rules (rules can contain other rules)
- Tracks piece/track boundaries for piece mapping

Usage:
    python scripts/run_normalized_pipeline.py /path/to/midi/corpus --output checkpoint_v31_factored.npz
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

# Import the GPU-optimized factored Re-Pair
from grammar.v2.repair_factored_hierarchical import (
    build_factored_hierarchical_grammar,
    FactoredHierarchicalGrammar,
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


def convert_normalized_to_factored_patterns(
    normalized_rules: dict,
    tracks: list,
) -> list:
    """Convert normalized rules to FactoredPattern objects.

    Now includes timing offsets (tau_offset, duration_offset, v_offset) from
    occurrences for lossless reconstruction.
    """
    patterns = []

    for rule_id, rule_data in normalized_rules.items():
        # Build occurrences WITH timing offsets
        occurrences = [
            PatternOccurrence(
                piece_id=occ['piece_id'],
                track_id=occ['track_id'],
                onset_time=occ['onset_time'],
                position=occ['position'],
                # Include timing offsets for lossless reconstruction
                tau_offset=occ.get('tau_offset', 480),
                duration_offset=occ.get('duration_offset', 480),
                v_offset=occ.get('v_offset', 4),
            )
            for occ in rule_data.get('occurrences', [])
        ]

        # Get timing ratios from rule data
        rhythm_ratios = rule_data.get('rhythm_ratios', [])
        duration_ratios = rule_data.get('duration_ratios', [])
        velocity_ratios = rule_data.get('velocity_ratios', [])

        # Get rhythm_ioi from ratios or estimate
        rhythm_ioi = rule_data.get('rhythm_ioi', [])
        if not rhythm_ioi:
            length = len(rule_data.get('pitch_classes', []))
            if rhythm_ratios and len(rhythm_ratios) == length - 1:
                # Convert ratios to IOIs using 480 as base
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

        # Store actual ratios for to_dict() serialization
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
    """Save checkpoint with full normalized pattern data.

    For large datasets, JSON data is saved to separate files and referenced
    in the checkpoint to avoid numpy string array overflow issues.
    """

    # Convert numpy types for JSON
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

    # Build track info for piece mapping
    track_info = []
    for track in tracks:
        track_info.append({
            'piece_id': track.piece_id,
            'track_id': track.track_id,
            'length': len(track),
            'gm_program': track.gm_program,
            'is_drum': track.is_drum,  # Channel 9 = drums in General MIDI
        })

    # Save large JSON data to separate files to avoid numpy string overflow
    base_path = path.replace('.npz', '')
    patterns_path = f'{base_path}_patterns.json'
    track_info_path = f'{base_path}_track_info.json'

    if verbose:
        print(f"  Saving patterns JSON to {patterns_path}...")

    with open(patterns_path, 'w') as f:
        json.dump(rules_data, f)  # No indent to save space

    with open(track_info_path, 'w') as f:
        json.dump(track_info, f)

    data = {
        'version': np.array(['normalized_v4_external']),  # v4: external JSON files

        # Stats
        'n_files': np.array([stats.get('n_files', 0)]),
        'n_tracks': np.array([len(tracks)]),
        'n_notes': np.array([sum(len(t) for t in tracks)]),
        'n_patterns': np.array([len(grammar_result['rules'])]),
        'n_transform_vocabulary': np.array([stats.get('n_transform_vocabulary', 0)]),
        'total_time': np.array([stats.get('total_time', 0)]),

        # Normalization type
        'normalization': np.array(['T+tau+v']),

        # Reference to external JSON files (relative paths)
        'patterns_json_file': np.array([os.path.basename(patterns_path)]),
        'track_info_json_file': np.array([os.path.basename(track_info_path)]),
    }

    # Add transform discovery if available
    if transform_discovery:
        transform_data = convert_numpy(transform_discovery)
        transforms_path = f'{base_path}_transforms.json'
        with open(transforms_path, 'w') as f:
            json.dump(transform_data, f)
        data['transforms_json_file'] = np.array([os.path.basename(transforms_path)])

    # Add multi-factor discovery (τ, v, d) if available
    if multi_factor_discovery:
        mf_data = convert_numpy({
            k: v for k, v in multi_factor_discovery.items()
            if k != 'vocabulary_objects'  # Remove non-serializable
        })
        multi_factor_path = f'{base_path}_multi_factor.json'
        with open(multi_factor_path, 'w') as f:
            json.dump(mf_data, f)
        data['multi_factor_json_file'] = np.array([os.path.basename(multi_factor_path)])
        data['n_factor_vocabulary'] = np.array([len(multi_factor_discovery.get('vocabulary', []))])
        data['n_rhythm_transforms'] = np.array([multi_factor_discovery.get('rhythm_transforms', 0)])
        data['n_velocity_transforms'] = np.array([multi_factor_discovery.get('velocity_transforms', 0)])
        data['n_duration_transforms'] = np.array([multi_factor_discovery.get('duration_transforms', 0)])

    # Add TrackDerive discovery (cross-track arrangement derivations)
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

    # Add feature importance discovery (MDL-based feature selection)
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

    # Add per-piece interval preferences (for context-aware generation)
    piece_prefs = stats.get('piece_interval_preferences', {})
    if piece_prefs:
        prefs_path = f'{base_path}_interval_prefs.json'
        with open(prefs_path, 'w') as f:
            json.dump(convert_numpy(piece_prefs), f)
        data['interval_prefs_json_file'] = np.array([os.path.basename(prefs_path)])
        data['preferred_interval_repr'] = np.array([stats.get('preferred_interval_repr', 'chromatic')])

        if verbose:
            from collections import Counter
            pref_counts = Counter(piece_prefs.values())
            print(f"  Saved per-piece interval preferences: "
                  f"magnitude={pref_counts.get('magnitude', 0)}, "
                  f"chromatic={pref_counts.get('chromatic', 0)}, "
                  f"both={pref_counts.get('both', 0)}")

    # Add meta patterns if available (can also be large)
    if meta_patterns:
        meta_data = convert_numpy(meta_patterns)
        # Only save orchestration rules to external file if large
        if 'orchestration_rules' in meta_data and len(meta_data.get('orchestration_rules', [])) > 1000:
            orch_path = f'{base_path}_orchestration.json'
            with open(orch_path, 'w') as f:
                json.dump(meta_data['orchestration_rules'], f)
            data['orchestration_json_file'] = np.array([os.path.basename(orch_path)])
            # Remove from meta_data to avoid double storage
            meta_data_small = {k: v for k, v in meta_data.items() if k != 'orchestration_rules'}
            meta_data_small['orchestration_rules_external'] = True
        else:
            meta_data_small = meta_data

        # Store small meta_patterns inline
        meta_json = json.dumps(meta_data_small)
        if len(meta_json) < 100_000_000:  # 100MB limit for inline
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


def convert_grammar_to_rules(
    grammar: FactoredHierarchicalGrammar,
    tracks: list,
    verbose: bool = True
) -> dict:
    """
    Convert FactoredHierarchicalGrammar to rules dictionary format.

    NEW in v36: Uses grammar.rule_occurrences captured DURING compression
    for accurate timing data. This solves the hierarchical position loss problem
    where walking final_sequence misses positions that were merged into parent rules.

    Pattern format:
        {
            'pitch_classes': [...],
            'pitch_intervals': [...],
            'rhythm_ratios': [...],      # Per-note, τ-normalized (IOI[i]/IOI[0])
            'duration_ratios': [...],    # Per-note, normalized (dur[i]/dur[0])
            'velocity_ratios': [...],    # Per-note, v-normalized (vel[i]/vel[0])
            'occurrences': [
                {
                    'piece_id': ...,
                    'track_id': ...,
                    'onset_time': ...,
                    'position': ...,
                    'tau_offset': 480,       # First IOI in ticks (for reconstruction)
                    'duration_offset': 480,  # First duration in ticks
                    'v_offset': 100,         # First velocity (MIDI 0-127)
                }
            ]
        }
    """
    rules = {}
    n_terminals = grammar.n_terminals

    # Build lookups from track_info
    # The index into our track list (i) maps to actual MIDI track info
    track_to_piece = {}      # i -> piece_id
    track_to_midi_id = {}    # i -> actual MIDI track_id
    track_to_gm = {}         # i -> GM program number
    for i, info in enumerate(grammar.track_info):
        track_to_piece[i] = info.get('piece_id', f'track_{i}')
        track_to_midi_id[i] = info.get('track_id', i)  # Actual MIDI track number
        track_to_gm[i] = info.get('gm_program', 0)     # GM instrument

    # Expand each rule to get pitch classes
    memo = {}
    for rule_idx in range(grammar.n_rules):
        rule_id = n_terminals + rule_idx

        # Get normalized factors (buckets - used for matching, not reconstruction)
        pitch_interval = grammar.rule_pitch_intervals[rule_idx].item()
        rhythm_bucket = grammar.rule_rhythm_buckets[rule_idx].item()
        velocity_bucket = grammar.rule_velocity_buckets[rule_idx].item()
        count = grammar.rule_counts[rule_idx].item()

        # Expand to get pitch classes (terminals)
        expanded = grammar.expand_rule(rule_id, memo)

        # Compute SIGNED intervals from expanded pitch classes
        # This preserves melodic direction (up vs down) for generation
        pitch_intervals = []
        for i in range(len(expanded) - 1):
            diff = (expanded[i + 1] - expanded[i]) % 12
            if diff > 6:
                diff -= 12  # Convert 7-11 to -5 to -1 (down instead of up)
            pitch_intervals.append(diff)

        # Get children IDs for hierarchical expansion
        left_child = grammar.rule_table[rule_idx, 0].item()
        right_child = grammar.rule_table[rule_idx, 1].item()
        is_hierarchical = (left_child >= n_terminals or right_child >= n_terminals)

        # Compute the REAL connector interval between children
        # For hierarchical patterns: connector is between left's last note and right's first note
        # This is pitch_intervals[left_length - 1]
        connector_interval = 0
        if is_hierarchical:
            # Get left child's length
            left_expanded = grammar.expand_rule(left_child, memo) if left_child >= n_terminals else [left_child]
            left_length = len(left_expanded)
            # The connector interval is at position (left_length - 1) in pitch_intervals
            if left_length > 0 and left_length - 1 < len(pitch_intervals):
                connector_interval = pitch_intervals[left_length - 1]

        rules[str(rule_id)] = {
            'pitch_classes': expanded,
            'pitch_intervals': pitch_intervals,
            'canonical_pitches': [],  # Absolute MIDI pitches from first (canonical) occurrence
            'rhythm_bucket': rhythm_bucket,
            'velocity_bucket': velocity_bucket,
            # Per-note ratios will be computed from first occurrence
            'rhythm_ratios': [],
            'duration_ratios': [],
            'velocity_ratios': [],
            'count': count,
            'occurrences': [],  # Will be populated from rule_occurrences
            'is_hierarchical': is_hierarchical,
            # Store children for recursive expansion (for hierarchical patterns)
            'left_child': left_child,
            'right_child': right_child,
            # Connector interval: pitch offset from left's last note to right's first note
            # This is the SIGNED interval, computed from the actual pitch classes
            'connector_interval': connector_interval,
        }

    # === KEY FIX: Use rule_occurrences captured during compression ===
    # This contains accurate timing data captured at the moment each pair was replaced
    if grammar.rule_occurrences:
        for rule_id, occurrences in grammar.rule_occurrences.items():
            rule_str = str(rule_id)
            if rule_str not in rules:
                continue

            pattern_length = len(rules[rule_str]['pitch_classes'])

            for occ in occurrences:
                track_idx = occ.get('track_idx', -1)
                orig_pos = occ.get('orig_pos', -1)

                if track_idx < 0 or orig_pos < 0:
                    continue

                piece_id = track_to_piece.get(track_idx, f'track_{track_idx}')
                midi_track_id = track_to_midi_id.get(track_idx, track_idx)
                gm_program = track_to_gm.get(track_idx, 0)

                # Build occurrence data from inline-captured timing
                occ_data = {
                    'piece_id': piece_id,
                    'track_id': midi_track_id,  # Actual MIDI track number (for track_info lookup)
                    'gm_program': gm_program,   # GM instrument for this occurrence
                    'onset_time': occ.get('onset', 0),
                    'position': orig_pos,
                }

                # Extract timing offsets (first note's values for reconstruction)
                ioi = occ.get('ioi', 0)
                duration = occ.get('duration', 0)
                velocity = occ.get('velocity', 0)

                if ioi > 0:
                    occ_data['tau_offset'] = ioi
                if duration > 0:
                    occ_data['duration_offset'] = duration
                if velocity > 0:
                    occ_data['v_offset'] = velocity

                # Compute ratios from inline data if available (for 2-note patterns)
                # For longer patterns expanded from hierarchical rules, ratios come from first occurrence
                ioi_right = occ.get('ioi_right', 0)
                duration_right = occ.get('duration_right', 0)
                velocity_right = occ.get('velocity_right', 0)

                # Extract octave data for transformational pitch reconstruction
                octave_left = occ.get('octave', 4)  # Default to octave 4
                octave_right = occ.get('octave_right', octave_left)

                # Compute absolute MIDI pitches for this occurrence
                pitch_classes = rules[rule_str]['pitch_classes']
                if len(pitch_classes) >= 2:
                    # MIDI pitch = pitch_class + octave * 12
                    # (octave is stored as pitch // 12, so this directly recovers the original pitch)
                    occ_midi_pitches = [
                        pitch_classes[0] + octave_left * 12,
                        pitch_classes[1] + octave_right * 12,
                    ]

                    # Set canonical_pitches from FIRST occurrence (the canonical one)
                    if not rules[rule_str]['canonical_pitches']:
                        rules[rule_str]['canonical_pitches'] = occ_midi_pitches
                        # First occurrence has zero transform (it IS the canonical)
                        occ_data['octave_transform'] = 0
                        occ_data['pitch_offset'] = 0  # Pitch class offset (0-11)
                    else:
                        # Compute full semitone offset from canonical
                        # Use first note to determine the transform
                        canonical_first = rules[rule_str]['canonical_pitches'][0]
                        occ_first = occ_midi_pitches[0]
                        full_offset = occ_first - canonical_first

                        # Split into octave (multiple of 12) and pitch class (0-11)
                        # This preserves voicing information for orchestration mining!
                        octave_transform = (full_offset // 12) * 12
                        pitch_offset = full_offset % 12

                        occ_data['octave_transform'] = octave_transform
                        occ_data['pitch_offset'] = pitch_offset  # 0-11, used for voicing

                # Set pattern-level ratios from first occurrence with valid data
                if not rules[rule_str]['rhythm_ratios'] and ioi > 0 and ioi_right > 0:
                    # For base 2-note patterns, ratio is [1.0] (relative IOI)
                    rules[rule_str]['rhythm_ratios'] = [1.0]  # One ratio for 2 notes

                if not rules[rule_str]['duration_ratios'] and duration > 0 and duration_right > 0:
                    # Duration ratios for 2 notes
                    rules[rule_str]['duration_ratios'] = [1.0, duration_right / duration]

                if not rules[rule_str]['velocity_ratios'] and velocity > 0 and velocity_right > 0:
                    # Velocity ratios for 2 notes
                    rules[rule_str]['velocity_ratios'] = [1.0, velocity_right / velocity]

                rules[rule_str]['occurrences'].append(occ_data)

        if verbose:
            print(f"  Using inline occurrence tracking (v36 fix)")
    else:
        # Fallback: walk final_sequence (less accurate for hierarchical rules)
        if verbose:
            print(f"  WARNING: No rule_occurrences - using fallback final_sequence walk")

        final_seq = grammar.final_sequence.cpu().numpy()
        track_boundaries = grammar.track_boundaries
        track_starts = [0] + [b + 1 for b in track_boundaries[:-1]]
        track_ends = track_boundaries

        for track_idx in range(len(track_starts)):
            start = track_starts[track_idx]
            end = track_ends[track_idx] if track_idx < len(track_ends) else len(final_seq)
            piece_id = track_to_piece.get(track_idx, f'track_{track_idx}')
            local_pos = 0

            for pos in range(start, min(end, len(final_seq))):
                symbol = final_seq[pos]
                if symbol < 0:
                    continue
                if symbol >= n_terminals:
                    rule_str = str(symbol)
                    if rule_str in rules:
                        pattern_length = len(rules[rule_str]['pitch_classes'])
                        occ_data = {
                            'piece_id': piece_id,
                            'track_id': track_idx,
                            'onset_time': local_pos * 480,
                            'position': local_pos,
                        }
                        rules[rule_str]['occurrences'].append(occ_data)
                        local_pos += pattern_length
                else:
                    local_pos += 1

    # === HIERARCHICAL TIMING EXPANSION ===
    # For hierarchical rules, expand timing ratios from child rules
    # Uses REAL connector IOI from rule_occurrences captured during compression
    # This must be done after all rules are created so we can reference children

    # Build connector IOI lookup from rule_occurrences
    # The connector IOI is the 'ioi' field captured when a hierarchical pair was replaced
    # This represents the IOI from left child's last note to right child's first note
    connector_ioi_lookup = {}
    if grammar.rule_occurrences:
        for rule_id, occurrences in grammar.rule_occurrences.items():
            if occurrences:
                # Use first occurrence's IOI as the canonical connector
                # This is the IOI captured at compression time between left and right
                first_occ = occurrences[0]
                connector_ioi_lookup[rule_id] = first_occ.get('ioi', 480)

    def expand_timing_ratios(rule_id: int, memo: dict = None) -> Tuple[List[float], List[float], List[float]]:
        """Recursively expand timing ratios for a hierarchical rule.

        Uses real connector IOI from rule_occurrences for accurate rhythm.
        """
        if memo is None:
            memo = {}
        if rule_id in memo:
            return memo[rule_id]

        rule_str = str(rule_id)
        if rule_str not in rules:
            # Terminal - no timing ratios
            memo[rule_id] = ([], [1.0], [1.0])
            return memo[rule_id]

        rule = rules[rule_str]

        # If already has valid ratios, use them
        n_notes = len(rule['pitch_classes'])
        rr = rule['rhythm_ratios']
        dr = rule['duration_ratios']
        vr = rule['velocity_ratios']

        if len(rr) >= max(1, n_notes - 1) and len(dr) >= n_notes:
            memo[rule_id] = (rr, dr, vr)
            return memo[rule_id]

        # Get children from rule_table
        rule_idx = rule_id - n_terminals
        if rule_idx < 0 or rule_idx >= len(grammar.rule_table):
            # Not a valid rule
            memo[rule_id] = ([1.0] * max(1, n_notes - 1), [1.0] * n_notes, [1.0] * n_notes)
            return memo[rule_id]

        left_id = grammar.rule_table[rule_idx, 0].item()
        right_id = grammar.rule_table[rule_idx, 1].item()

        # Expand children
        left_rr, left_dr, left_vr = expand_timing_ratios(left_id, memo)
        right_rr, right_dr, right_vr = expand_timing_ratios(right_id, memo)

        # === KEY FIX: Use REAL connector IOI from rule_occurrences ===
        # The connector_ioi is the IOI between left's last note and right's first note
        # captured at the moment this hierarchical pair was created during compression
        connector_ioi = connector_ioi_lookup.get(rule_id, 480)

        # Get base IOI for ratio computation (use left child's base or default 480)
        # For the canonical pattern, we normalize relative to 480 ticks (quarter note)
        base_ioi = 480
        connector_ratio = connector_ioi / base_ioi

        # Concatenate with real connector ratio
        combined_rr = list(left_rr) + [connector_ratio] + list(right_rr)
        combined_dr = list(left_dr) + list(right_dr)
        combined_vr = list(left_vr) + list(right_vr)

        memo[rule_id] = (combined_rr, combined_dr, combined_vr)
        return memo[rule_id]

    # Expand timing for hierarchical rules
    expansion_memo = {}
    n_hier_with_real_timing = 0
    for rule_str, rule in rules.items():
        if rule.get('is_hierarchical'):
            rule_id = int(rule_str)
            rr, dr, vr = expand_timing_ratios(rule_id, expansion_memo)
            n_notes = len(rule['pitch_classes'])

            # Track if we used real timing data
            if rule_id in connector_ioi_lookup:
                n_hier_with_real_timing += 1

            # Only update if expansion gives better data
            if len(rr) >= max(1, n_notes - 1) and (not rule['rhythm_ratios'] or len(rule['rhythm_ratios']) < len(rr)):
                rule['rhythm_ratios'] = rr[:n_notes - 1] if n_notes > 1 else []
            if len(dr) >= n_notes and (not rule['duration_ratios'] or len(rule['duration_ratios']) < len(dr)):
                rule['duration_ratios'] = dr[:n_notes]
            if len(vr) >= n_notes and (not rule['velocity_ratios'] or len(rule['velocity_ratios']) < len(vr)):
                rule['velocity_ratios'] = vr[:n_notes]

    if verbose:
        n_hier = sum(1 for r in rules.values() if r.get('is_hierarchical'))
        print(f"  Hierarchical timing: {n_hier_with_real_timing}/{n_hier} rules with real connector IOI")

    # Fill in default ratios for patterns still missing data
    for rule_str, rule in rules.items():
        pattern_length = len(rule['pitch_classes'])
        if not rule['rhythm_ratios'] or len(rule['rhythm_ratios']) < max(1, pattern_length - 1):
            rule['rhythm_ratios'] = [1.0] * max(1, pattern_length - 1)
        if not rule['duration_ratios'] or len(rule['duration_ratios']) < pattern_length:
            rule['duration_ratios'] = [1.0] * pattern_length
        if not rule['velocity_ratios'] or len(rule['velocity_ratios']) < pattern_length:
            rule['velocity_ratios'] = [1.0] * pattern_length

    # === RECURSIVE EXPANSION OF CANONICAL PITCHES ===
    # For hierarchical patterns, we need to combine children's pitches
    # with the connector interval to get the full melodic contour
    def expand_canonical_pitches(rule_id_str, memo={}):
        """Recursively expand canonical_pitches for a rule."""
        if rule_id_str in memo:
            return memo[rule_id_str]

        rule = rules.get(rule_id_str)
        if not rule:
            # Terminal (pitch class) - use default octave 5
            pc = int(rule_id_str)
            memo[rule_id_str] = [pc + 60]
            return memo[rule_id_str]

        # If already has correct canonical_pitches (from occurrence data), use it
        if rule['canonical_pitches'] and len(rule['canonical_pitches']) == len(rule['pitch_classes']):
            memo[rule_id_str] = rule['canonical_pitches']
            return rule['canonical_pitches']

        left_child = rule.get('left_child', -1)
        right_child = rule.get('right_child', -1)

        if left_child < 0 or right_child < 0:
            # No children info, use heuristic
            result = [pc + 60 for pc in rule['pitch_classes']]
            memo[rule_id_str] = result
            return result

        # Expand children
        left_pitches = expand_canonical_pitches(str(left_child), memo)
        right_pitches = expand_canonical_pitches(str(right_child), memo)

        # Connector interval determines how right_pitches attach to left_pitches
        # connector_interval is already a signed interval (computed from pitch_intervals)
        connector = rule.get('connector_interval', 0)

        # Adjust right_pitches so that:
        # right_pitches[0] = left_pitches[-1] + connector
        if left_pitches and right_pitches:
            target_first_right = left_pitches[-1] + connector
            current_first_right = right_pitches[0]
            offset = target_first_right - current_first_right

            adjusted_right = [p + offset for p in right_pitches]
            result = left_pitches + adjusted_right
        else:
            result = left_pitches + right_pitches

        memo[rule_id_str] = result
        return result

    # Expand canonical_pitches for all rules
    expansion_memo = {}
    for rule_str, rule in rules.items():
        if not rule['canonical_pitches'] or len(rule['canonical_pitches']) != len(rule['pitch_classes']):
            rule['canonical_pitches'] = expand_canonical_pitches(rule_str, expansion_memo)

    if verbose:
        # Count how many patterns now have full canonical_pitches
        n_full = sum(1 for r in rules.values()
                    if len(r['canonical_pitches']) == len(r['pitch_classes']))
        print(f"  Canonical pitches expanded: {n_full}/{len(rules)} patterns have full data")

    # === COMPUTE TRUE SIGNED INTERVALS FROM CANONICAL PITCHES ===
    # This uses actual MIDI pitches to preserve melodic direction (up vs down)
    # Much better than the heuristic (smallest interval) computed from pitch classes
    for rule_str, rule in rules.items():
        canonical = rule['canonical_pitches']
        if len(canonical) >= 2:
            # Compute TRUE signed intervals from absolute MIDI pitches
            true_intervals = []
            for i in range(len(canonical) - 1):
                interval = canonical[i + 1] - canonical[i]
                true_intervals.append(interval)
            rule['pitch_intervals'] = true_intervals
        # else: keep the heuristic intervals computed earlier

    if verbose:
        n_hier = sum(1 for r in rules.values() if r['is_hierarchical'])
        total_occ = sum(len(r['occurrences']) for r in rules.values())
        n_with_timing = sum(1 for r in rules.values()
                          if any('tau_offset' in o for o in r['occurrences']))
        n_with_transforms = sum(1 for r in rules.values()
                               if any('octave_transform' in o for o in r['occurrences']))
        # Count unique octave transforms used
        all_transforms = set()
        for r in rules.values():
            for o in r['occurrences']:
                if 'octave_transform' in o:
                    all_transforms.add(o['octave_transform'])
        print(f"  Converted {len(rules)} rules ({n_hier} hierarchical)")
        print(f"  Total tracked occurrences: {total_occ:,}")
        print(f"  Patterns with timing data: {n_with_timing}")
        print(f"  Occurrences with octave_transform: {n_with_transforms}")
        print(f"  Unique octave transforms: {sorted(all_transforms)}")

    return {
        'rules': rules,
        'n_rules': len(rules),
    }


def run_normalized_pipeline(
    corpus_path: str,
    output_path: str = 'checkpoint_normalized.npz',
    max_files: int = 500,
    min_length: int = 3,
    max_length: int = 32,
    min_count: int = 2,
    max_rules: int = 20000,
    verbose: bool = True
) -> dict:
    """
    Run the T+τ normalized pattern discovery pipeline.

    Args:
        corpus_path: Path to MIDI corpus directory
        output_path: Output checkpoint path
        max_files: Maximum MIDI files to process
        min_length: Minimum pattern length
        max_length: Maximum pattern length
        min_count: Minimum occurrence count
        max_rules: Maximum patterns to discover
        verbose: Print progress

    Returns:
        Stats dictionary
    """
    total_start = time.time()
    stats = {}

    if verbose:
        print("=" * 70)
        print("T+τ NORMALIZED PATTERN DISCOVERY PIPELINE")
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

    # Find MIDI files
    midi_files = sorted(glob.glob(str(Path(corpus_path) / "*.mid")))
    midi_files += sorted(glob.glob(str(Path(corpus_path) / "*.midi")))
    midi_files += sorted(glob.glob(str(Path(corpus_path) / "**/*.mid"), recursive=True))
    midi_files += sorted(glob.glob(str(Path(corpus_path) / "**/*.midi"), recursive=True))
    midi_files = list(dict.fromkeys(midi_files))  # Dedupe

    if max_files:
        midi_files = midi_files[:max_files]

    if verbose:
        print(f"  Found {len(midi_files)} MIDI files", flush=True)

    # Parallel loading
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
    # PHASE 2: Run T+τ+v factored hierarchical Re-Pair
    # =========================================================================
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 2] Running T+τ+v Factored Hierarchical Re-Pair...", flush=True)

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
        ioi_sequences.append(track.rhythm_ioi.tolist())
        velocity_sequences.append(track.velocities.tolist())
        # Also pass duration, onset, and octave for inline occurrence tracking
        duration_sequences.append(track.durations.tolist() if hasattr(track, 'durations') else [480] * len(track))
        onset_sequences.append(track.onsets.tolist() if hasattr(track, 'onsets') else [i * 480 for i in range(len(track))])
        octave_sequences.append(track.octaves.tolist() if hasattr(track, 'octaves') else [4] * len(track))
        track_info_list.append({
            'piece_id': track.piece_id,
            'track_id': track.track_id,
            'gm_program': getattr(track, 'gm_program', 0),  # GM instrument number
            'is_drum': getattr(track, 'is_drum', False),    # Include drum flag
        })

    # Run GPU Re-Pair with duration/onset/octave for inline occurrence tracking
    grammar = build_factored_hierarchical_grammar(
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
    grammar_result = convert_grammar_to_rules(grammar, all_tracks, verbose)

    stats['n_patterns'] = grammar_result['n_rules']
    stats['compression'] = grammar.compression_ratio()
    stats['phase2_time'] = time.time() - phase_start

    # =========================================================================
    # PHASE 3: Analyze patterns
    # =========================================================================
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 3] Analyzing patterns...", flush=True)

    # Count statistics
    rules = grammar_result['rules']
    total_occurrences = sum(r['count'] for r in rules.values())

    # Multi-piece patterns
    multi_piece = sum(
        1 for r in rules.values()
        if len(set(o['piece_id'] for o in r['occurrences'])) > 1
    )

    # Unique pieces covered
    all_pieces = set()
    for r in rules.values():
        for occ in r['occurrences']:
            all_pieces.add(occ['piece_id'])

    # Length distribution
    length_dist = defaultdict(int)
    for r in rules.values():
        length_dist[len(r['pitch_intervals'])] += 1

    # Top patterns by count
    top_patterns = sorted(rules.items(), key=lambda x: x[1]['count'], reverse=True)[:20]

    stats['total_occurrences'] = total_occurrences
    stats['multi_piece_patterns'] = multi_piece
    stats['pieces_covered'] = len(all_pieces)
    stats['phase3_time'] = time.time() - phase_start

    if verbose:
        print(f"  Total occurrences: {total_occurrences:,}")
        print(f"  Multi-piece patterns: {multi_piece}")
        print(f"  Pieces covered: {len(all_pieces)}")
        print(f"\n  Top 10 patterns:")
        for rule_id, rule in top_patterns[:10]:
            pc = rule['pitch_classes']
            intervals = rule['pitch_intervals']
            count = rule['count']
            n_pieces = len(set(o['piece_id'] for o in rule['occurrences']))
            print(f"    R{rule_id}: {pc[:8]}... intervals={intervals[:5]}... "
                  f"count={count}, pieces={n_pieces}")

    # =========================================================================
    # PHASE 4: Convert to FactoredPattern for MDL Transform Discovery
    # =========================================================================
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 4] Extracting canonical patterns for transform discovery...", flush=True)

    # Convert rules to FactoredPattern objects
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
            # Note: all_tracks contains FactoredTrack objects, not dicts
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
                add_to_occurrences=True,  # Adds 'derived_from' to occurrences
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
            # Reuse patterns_for_td if available, else rebuild
            if 'patterns_for_td' not in dir():
                patterns_for_im = [{
                    'pitch_classes': p.pitch_classes,
                    'occurrences': p.occurrences if hasattr(p, 'occurrences') else [],
                } for p in canonicals]
            else:
                patterns_for_im = patterns_for_td

            interval_magnitude_discovery = run_interval_magnitude_discovery(
                patterns_for_im,
                device='cuda',
                verbose=verbose
            )

            stats['preferred_interval_repr'] = interval_magnitude_discovery.get('preferred_representation', 'chromatic')
            comp = interval_magnitude_discovery.get('compression_comparison', {})
            stats['chromatic_compression'] = comp.get('chromatic', {}).get('compression', 0.0)
            stats['magnitude_compression'] = comp.get('magnitude', {}).get('compression', 0.0)

            # Per-piece preferences for context-aware generation
            stats['piece_interval_preferences'] = interval_magnitude_discovery.get('piece_preferences', {})
            stats['n_magnitude_added'] = interval_magnitude_discovery.get('n_magnitude_added', 0)

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
    # This discovers WHICH features help predict transforms, without prescribing
    # what those features mean. If "keys" are real, pitch_offset_relative will
    # emerge as useful. If music is atonal, it will be ignored.
    phase_start = time.time()
    feature_importance_discovery = {'useful_features': []}

    if len(canonicals) > 20 and stats.get('n_transform_vocabulary', 0) > 0:
        if verbose:
            print(f"\n[Phase 5e] Feature Importance Discovery (MDL-Based)...", flush=True)

        try:
            # Need transform lookup for this phase
            # Build it from the transform discovery if available
            if transform_discovery and 'lookup' in transform_discovery:
                transform_lookup = transform_discovery['lookup']
            else:
                # Build a simple lookup from patterns
                from scripts.level3_meta_patterns import build_transform_lookup_gpu
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
                # Prepare patterns with occurrences
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
        print(f"  Total occurrences: {stats['total_occurrences']:,}")
        print(f"  Multi-piece patterns: {stats['multi_piece_patterns']}")
        print(f"  Transform vocabulary (pitch): {stats.get('n_transform_vocabulary', 0)}")
        if stats.get('n_factor_vocabulary', 0) > 0:
            print(f"  Factor vocabulary (τ,v,d): {stats.get('n_factor_vocabulary', 0)}")
            print(f"    τ (rhythm): {stats.get('n_rhythm_transforms', 0)}")
            print(f"    v (velocity): {stats.get('n_velocity_transforms', 0)}")
            print(f"    d (duration): {stats.get('n_duration_transforms', 0)}")
        if stats.get('n_track_derives', 0) > 0:
            print(f"  TrackDerive relations: {stats.get('n_track_derives', 0)}")
            if stats.get('track_derive_transforms'):
                td_transforms = stats['track_derive_transforms']
                parts = [f"{k}:{v}" for k, v in sorted(td_transforms.items(), key=lambda x: -x[1])[:4]]
                print(f"    By transform: {', '.join(parts)}")
        if stats.get('preferred_interval_repr'):
            pref = stats['preferred_interval_repr']
            chrom = stats.get('chromatic_compression', 0)
            mag = stats.get('magnitude_compression', 0)
            print(f"  Interval representation (global): {pref}")
            print(f"    Chromatic compression: {chrom:.2f}x")
            print(f"    Magnitude compression: {mag:.2f}x")
            if pref == 'magnitude':
                print(f"    → Corpus has diatonic structure (b3/3 treated as '3rds')")
            # Show per-piece breakdown
            piece_prefs = stats.get('piece_interval_preferences', {})
            if piece_prefs:
                from collections import Counter
                pref_counts = Counter(piece_prefs.values())
                print(f"    Per-piece: magnitude={pref_counts.get('magnitude', 0)}, "
                      f"chromatic={pref_counts.get('chromatic', 0)}, "
                      f"both={pref_counts.get('both', 0)}")
        if stats.get('useful_features'):
            useful = stats['useful_features']
            print(f"  Useful features (MDL): {', '.join(useful)}")
            if 'pitch_offset_relative' in useful:
                print(f"    → 'Keys' emerged as useful (pitch relative to piece mode)")
            if 'pitch_offset_delta' in useful:
                print(f"    → Melodic motion patterns emerged as useful")
        if meta_patterns.get('n_orchestration_rules'):
            print(f"  Orchestration rules (summary): {meta_patterns['n_orchestration_rules']}")
        print(f"  Total time: {stats['total_time']:.1f}s")
        print(f"\n  Output: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='T+τ Normalized Pattern Discovery')
    parser.add_argument('corpus', help='Path to MIDI corpus')
    parser.add_argument('--output', '-o', default='checkpoint_normalized.npz',
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

    run_normalized_pipeline(
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
