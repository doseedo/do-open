#!/usr/bin/env python3
"""
Pure Contour Re-Pair Pipeline (v54 - Per-Instrument)
=====================================================

Per-instrument pattern discovery for role-specific generation.

Key change from v53:
- v53: All instruments share one pattern vocabulary (merged)
       Bass [+2,+1,-1] = Trumpet [+2,+1,-1] = same pattern
- v54: Each instrument has its OWN vocabulary (separated)
       Bass has GM32_P42, Trumpet has GM56_P42 (different patterns!)

This enables:
- Role-specific generation (bass plays bass patterns, not melody patterns)
- Chord preservation (piano patterns stay as chords, not arpeggios)
- Better TrackDerive discovery (cross-vocabulary relationships)

Architecture (3-layer):
  Layer 1: Per-instrument grammar (this file)
  Layer 2: Temporal alignment (onset_time for simultaneity)
  Layer 3: Cross-track relationships (TrackDerive, after grammar)

Usage:
    python scripts/run_pure_contour_pipeline.py /path/to/midi/corpus \\
        --output checkpoint_v54_per_instrument.npz --per-instrument
"""

import os
import sys
import time
import json
import glob
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List, Dict

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_factored_pipeline import (
    FactoredTrack,
    load_midi_factored,
    bucket_duration,
    FactoredPattern,
    PatternOccurrence,
    run_factored_mdl_transform_discovery,
)

# Import the v4 pure contour Re-Pair
from grammar.v4.repair_pure_contour import (
    build_pure_contour_grammar,
    PureContourGrammar,
    N_TERMINALS,
    RHYTHM_BUCKETS,
    VELOCITY_BUCKETS,
    decode_terminal,
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


# =============================================================================
# PER-INSTRUMENT PATTERN DISCOVERY (v54)
# =============================================================================

def group_tracks_by_instrument(tracks: list) -> Dict[int, list]:
    """Group tracks by GM program (instrument).

    This is the foundation of per-instrument pattern discovery.
    Each instrument gets its own vocabulary.

    Args:
        tracks: List of FactoredTrack objects

    Returns:
        Dict mapping GM program -> list of tracks for that instrument
    """
    by_gm = defaultdict(list)
    for track in tracks:
        gm = getattr(track, 'gm_program', 0)
        by_gm[gm].append(track)
    return dict(by_gm)


def build_per_instrument_grammars(
    tracks: list,
    device: str = 'cuda',
    min_pair_count: int = 2,
    max_rules_per_instrument: int = 2000,
    top_instruments: int = 15,
    max_notes_per_instrument: int = 100000,  # Cap notes to avoid slow processing
    verbose: bool = True,
) -> Dict[int, 'PureContourGrammar']:
    """Build SEPARATE Re-Pair grammar for each instrument.

    TRUE per-instrument isolation:
    - Bass [+2, +5] → GM32_129
    - Trumpet [+2, +5] → GM56_129 (DIFFERENT pattern!)

    Each instrument gets its own vocabulary. This enables:
    - Role-specific generation (bass samples from bass-only patterns)
    - Instrument idiom capture (patterns reflect instrument-specific usage)
    - Clean PPM context (PPM[bass] has only bass patterns)

    Args:
        tracks: List of FactoredTrack objects
        device: 'cuda' or 'cpu'
        min_pair_count: Minimum pair frequency for rule creation
        max_rules_per_instrument: Max rules per instrument
        top_instruments: Only process top N instruments by note count (0 = all)
        verbose: Print progress

    Returns:
        Dict mapping GM program -> PureContourGrammar for that instrument
    """
    grammars_by_gm = {}
    tracks_by_gm = group_tracks_by_instrument(tracks)

    # Sort instruments by total note count (descending) and take top N
    gm_note_counts = []
    for gm, gm_tracks in tracks_by_gm.items():
        n_notes = sum(len(t.pitch_classes) for t in gm_tracks)
        n_tracks = len(gm_tracks)
        gm_note_counts.append((gm, n_notes, n_tracks, gm_tracks))

    gm_note_counts.sort(key=lambda x: x[1], reverse=True)

    if top_instruments > 0:
        gm_note_counts = gm_note_counts[:top_instruments]
        if verbose:
            print(f"  Processing top {top_instruments} instruments by note count (of {len(tracks_by_gm)} total)...")
    else:
        if verbose:
            print(f"  Building per-instrument grammars for {len(tracks_by_gm)} instruments...")

    for gm, n_notes, n_tracks, gm_tracks in gm_note_counts:
        if n_tracks < 2:
            if verbose:
                print(f"    GM {gm}: Skipping (only {n_tracks} track)")
            continue

        if n_notes < 10:
            if verbose:
                print(f"    GM {gm}: Skipping (only {n_notes} notes)")
            continue

        # Subsample tracks if too many notes
        tracks_to_use = gm_tracks
        if max_notes_per_instrument > 0 and n_notes > max_notes_per_instrument:
            sorted_tracks = sorted(gm_tracks, key=lambda t: len(t.pitch_classes), reverse=True)
            tracks_to_use = []
            note_count = 0
            for t in sorted_tracks:
                if note_count + len(t.pitch_classes) > max_notes_per_instrument:
                    break
                tracks_to_use.append(t)
                note_count += len(t.pitch_classes)
            n_notes_used = note_count
            if verbose:
                print(f"    GM {gm}: {len(tracks_to_use)}/{n_tracks} tracks, {n_notes_used:,}/{n_notes:,} notes (capped)...", end=' ', flush=True)
        else:
            if verbose:
                print(f"    GM {gm}: {n_tracks} tracks, {n_notes:,} notes...", end=' ', flush=True)

        try:
            grammar = build_pure_contour_grammar(
                tracks_to_use,
                device=device,
                min_pair_count=min_pair_count,
                max_rules=max_rules_per_instrument,
                verbose=False,  # Quiet per-instrument
            )
            grammars_by_gm[gm] = grammar

            if verbose:
                print(f"→ {grammar.n_rules} patterns, {grammar.compression_ratio():.1f}x compression")

        except Exception as e:
            if verbose:
                print(f"→ FAILED: {e}")

    return grammars_by_gm


def convert_per_instrument_grammars_to_rules(
    grammars_by_gm: Dict[int, 'PureContourGrammar'],
    tracks_by_gm: Dict[int, list],
    verbose: bool = True,
) -> dict:
    """Convert per-instrument grammars to unified rules dict with GM-prefixed IDs.

    Each pattern ID is prefixed with its GM program to ensure uniqueness:
    - GM32_129 = bass pattern 129
    - GM56_129 = trumpet pattern 129 (DIFFERENT pattern, same local ID!)

    Args:
        grammars_by_gm: Dict of GM -> PureContourGrammar (separate grammar per instrument)
        tracks_by_gm: Dict of GM -> list of tracks
        verbose: Print progress

    Returns:
        Dict with 'rules' containing all patterns with GM-prefixed IDs
    """
    all_rules = {}

    for gm, grammar in grammars_by_gm.items():
        # Build track lookups for this instrument's grammar
        track_to_piece = {}
        track_to_midi_id = {}
        for i, info in enumerate(grammar.track_info):
            track_to_piece[i] = info.get('piece_id', f'track_{i}')
            track_to_midi_id[i] = info.get('track_id', i)

        n_terminals = grammar.n_terminals
        memo = {}

        for rule_idx in range(grammar.n_rules):
            rule_id = n_terminals + 1 + rule_idx

            # Create GM-prefixed pattern ID (unique across all instruments)
            prefixed_id = f"GM{gm}_{rule_id}"

            # Get contour definition
            interval, rhythm_bucket, velocity_bucket = grammar.get_rule_contour(rule_id)
            count = grammar.rule_counts[rule_idx].item()

            # Get children for hierarchical (also prefix with same GM)
            left_child, right_child = grammar.get_rule_children(rule_id)
            is_hierarchical = left_child >= 0

            prefixed_left = f"GM{gm}_{left_child}" if left_child >= 0 else -1
            prefixed_right = f"GM{gm}_{right_child}" if right_child >= 0 else -1

            # Expand to get interval sequence
            intervals = grammar.expand_rule(rule_id, memo)

            # Build pitch_intervals
            if is_hierarchical:
                left_key = f"GM{gm}_{left_child}"
                right_key = f"GM{gm}_{right_child}"
                left_intervals = all_rules.get(left_key, {}).get('pitch_intervals', [])
                right_intervals = all_rules.get(right_key, {}).get('pitch_intervals', [])
                pitch_intervals = list(left_intervals) + [interval] + list(right_intervals)
            else:
                pitch_intervals = [interval]

            # Compute canonical_pitches from first occurrence
            canonical_pitches = []
            rhythm_ratios = []
            velocity_ratios = []
            duration_ratios = []

            if grammar.rule_occurrences and rule_id in grammar.rule_occurrences:
                occurrences = grammar.rule_occurrences[rule_id]
                if occurrences:
                    first_occ = occurrences[0]
                    first_pitch = first_occ.get('first_pitch', 60)
                    pitches = [first_pitch]
                    for iv in pitch_intervals:
                        pitches.append(pitches[-1] + iv)
                    canonical_pitches = pitches

                    rhythm_ratios = first_occ.get('rhythm_ratios', [])
                    velocity_ratios = first_occ.get('velocity_ratios', [])
                    duration_ratios = first_occ.get('duration_ratios', [])

            pitch_classes = [p % 12 for p in canonical_pitches] if canonical_pitches else []

            all_rules[prefixed_id] = {
                'pitch_classes': pitch_classes,
                'pitch_intervals': pitch_intervals,
                'canonical_pitches': canonical_pitches,
                'rhythm_bucket': rhythm_bucket,
                'velocity_bucket': velocity_bucket,
                'rhythm_ratios': rhythm_ratios,
                'duration_ratios': duration_ratios,
                'velocity_ratios': velocity_ratios,
                'count': count,
                'occurrences': [],
                'is_hierarchical': is_hierarchical,
                'left_child': prefixed_left,
                'right_child': prefixed_right,
                'connector_interval': interval if is_hierarchical else 0,
                'contour': (interval, rhythm_bucket, velocity_bucket),
                'is_pure_contour': True,
                'is_per_instrument': True,
                'gm_program': gm,
            }

        # Add occurrences with full timing data
        if grammar.rule_occurrences:
            for rule_id, occurrences in grammar.rule_occurrences.items():
                prefixed_id = f"GM{gm}_{rule_id}"
                if prefixed_id not in all_rules:
                    continue

                for occ in occurrences:
                    track_idx = occ.get('track_idx', -1)
                    orig_pos = occ.get('orig_pos', -1)

                    if track_idx < 0 or orig_pos < 0:
                        continue

                    piece_id = track_to_piece.get(track_idx, f'track_{track_idx}')
                    midi_track_id = track_to_midi_id.get(track_idx, track_idx)

                    occ_data = {
                        'piece_id': piece_id,
                        'track_id': midi_track_id,
                        'gm_program': gm,
                        'is_drum': gm == 128,
                        'position': orig_pos,
                        'last_position': occ.get('last_orig_pos', orig_pos),
                        'first_pitch': occ.get('first_pitch', 60),
                        'last_pitch': occ.get('last_pitch', 60),
                        'onset_time': occ.get('onset_time', 0),
                        'tau_offset': occ.get('first_ioi', 480),
                        'pitch_offset': occ.get('first_pitch', 60) % 12,
                        'rhythm_ratios': occ.get('rhythm_ratios', []),
                        'velocity_ratios': occ.get('velocity_ratios', []),
                        'duration_ratios': occ.get('duration_ratios', []),
                    }
                    all_rules[prefixed_id]['occurrences'].append(occ_data)

    if verbose:
        total_occurrences = sum(len(r['occurrences']) for r in all_rules.values())
        hier_count = sum(1 for r in all_rules.values() if r['is_hierarchical'])
        print(f"  Converted {len(all_rules)} patterns ({hier_count} hierarchical)")
        print(f"    Total occurrences: {total_occurrences:,}")

        # Show per-instrument breakdown
        by_gm = defaultdict(int)
        for r in all_rules.values():
            by_gm[r['gm_program']] += 1
        for gm, count in sorted(by_gm.items()):
            print(f"    GM {gm}: {count} patterns")

    return {'rules': all_rules, 'n_rules': len(all_rules)}


def convert_pure_contour_grammar_to_rules(
    grammar: PureContourGrammar,
    tracks: list,
    verbose: bool = True
) -> dict:
    """
    Convert PureContourGrammar to rules dictionary format.

    Key difference from v3 converter:
    - Terminals are (rhythm, velocity) pairs, NOT pitch classes
    - Each occurrence has first_pitch (absolute MIDI pitch 0-127)
    - Patterns are purely contour-based
    """
    rules = {}
    n_terminals = grammar.n_terminals

    # Build lookups from track_info
    track_to_piece = {}
    track_to_midi_id = {}
    track_to_gm = {}
    track_to_is_drum = {}
    for i, info in enumerate(grammar.track_info):
        track_to_piece[i] = info.get('piece_id', f'track_{i}')
        track_to_midi_id[i] = info.get('track_id', i)
        track_to_gm[i] = info.get('gm_program', 0)
        track_to_is_drum[i] = info.get('is_drum', False)

    # Expand each rule to get pitch intervals
    memo = {}
    for rule_idx in range(grammar.n_rules):
        rule_id = n_terminals + 1 + rule_idx

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
            left_intervals = rules.get(str(left_child), {}).get('pitch_intervals', [])
            right_intervals = rules.get(str(right_child), {}).get('pitch_intervals', [])
            pitch_intervals = list(left_intervals) + [interval] + list(right_intervals)
        else:
            pitch_intervals = [interval]

        # Compute canonical_pitches from first occurrence
        canonical_pitches = []
        if grammar.rule_occurrences and rule_id in grammar.rule_occurrences:
            occurrences = grammar.rule_occurrences[rule_id]
            if occurrences:
                first_occ = occurrences[0]
                first_pitch = first_occ.get('first_pitch', 60)
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
            'contour': (interval, rhythm_bucket, velocity_bucket),
            'is_pure_contour': True,  # v53 marker
        }

    # Add occurrences with first_pitch
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
                is_drum = track_to_is_drum.get(track_idx, False)

                occ_data = {
                    'piece_id': piece_id,
                    'track_id': midi_track_id,
                    'gm_program': gm_program,
                    'is_drum': is_drum,  # From MIDI channel 9
                    'position': orig_pos,
                    'last_position': occ.get('last_orig_pos', orig_pos),
                    'first_pitch': occ.get('first_pitch', 60),  # Absolute MIDI pitch
                    'last_pitch': occ.get('last_pitch', 60),
                    'onset_time': occ.get('onset_time', 0),
                    'tau_offset': occ.get('first_ioi', 480),  # First IOI for timing
                    # Compute pitch_offset from canonical
                    'pitch_offset': occ.get('first_pitch', 60) % 12,
                }
                rules[rule_str]['occurrences'].append(occ_data)

    if verbose:
        total_occurrences = sum(len(r['occurrences']) for r in rules.values())
        hier_count = sum(1 for r in rules.values() if r['is_hierarchical'])
        print(f"  Converted {len(rules)} rules ({hier_count} hierarchical) with {total_occurrences:,} occurrences")

    return {'rules': rules, 'n_rules': len(rules)}


def convert_to_factored_patterns(rules: dict, tracks: list) -> list:
    """Convert rules to FactoredPattern objects for transform discovery."""
    patterns = []

    for rule_id, rule_data in rules.items():
        occurrences = [
            PatternOccurrence(
                piece_id=occ['piece_id'],
                track_id=occ['track_id'],
                onset_time=occ.get('onset_time', 0),
                position=occ.get('position', 0),
                tau_offset=occ.get('tau_offset', 480),  # Use actual IOI from grammar
                duration_offset=480,
                v_offset=4,
            )
            for occ in rule_data.get('occurrences', [])
        ]

        rhythm_ratios = rule_data.get('rhythm_ratios', [])
        pitch_classes = rule_data.get('pitch_classes', [])
        length = len(pitch_classes)

        rhythm_ioi = [480] * max(0, length - 1)

        pattern = FactoredPattern(
            pattern_id=int(rule_id) if rule_id.isdigit() else hash(rule_id) % 100000,
            rule_id=rule_id,
            pitch_classes=pitch_classes,
            octaves=[4] * length,
            velocities=[4] * length,
            durations=[3] * length,
            rhythm_ioi=rhythm_ioi,
            occurrences=occurrences,
        )

        pattern._rhythm_ratios = rhythm_ratios
        pattern._duration_ratios = rule_data.get('duration_ratios', [])
        pattern._velocity_ratios = rule_data.get('velocity_ratios', [])

        patterns.append(pattern)

    return patterns


def save_checkpoint(
    path: str,
    grammar_result: dict,
    grammar: PureContourGrammar,
    tracks: list,
    stats: dict,
    transform_discovery: dict = None,
    meta_patterns: dict = None,
    multi_factor_discovery: dict = None,
    track_derive_discovery: dict = None,
    feature_importance_discovery: dict = None,
    verbose: bool = True
):
    """Save checkpoint with full data.

    For per-instrument mode (v54):
    - Version is 'v54_per_instrument'
    - Pattern IDs are GM-prefixed (e.g., GM32_129)
    - Each pattern has gm_program field
    - Patterns have is_per_instrument=True marker

    For unified mode (v53):
    - Version is 'v53_pure_contour'
    - Pattern IDs are numeric (e.g., 129)
    - All instruments share same pattern vocabulary
    """
    is_per_instrument = stats.get('per_instrument', False)

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
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            # Handle dataclass-like objects (TrackDerive, CompoundTransform, etc.)
            result = {}
            for k, v in obj.__dict__.items():
                if not k.startswith('_'):
                    result[k] = convert_numpy(v)
            # Also handle slot-based objects
            if hasattr(obj, '__slots__'):
                for slot in obj.__slots__:
                    if hasattr(obj, slot) and not slot.startswith('_'):
                        result[slot] = convert_numpy(getattr(obj, slot))
            return result
        else:
            # Try to convert to string for non-serializable objects
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    rules_data = convert_numpy(grammar_result['rules'])

    base_path = path.replace('.npz', '')
    patterns_path = f'{base_path}_patterns.json'

    if verbose:
        print(f"  Saving patterns JSON to {patterns_path}...")

    with open(patterns_path, 'w') as f:
        json.dump(rules_data, f, indent=2)

    # Convert tensors to numpy (handle None grammar for per-instrument mode)
    if grammar is not None:
        final_sequence = grammar.final_sequence.cpu().numpy() if hasattr(grammar.final_sequence, 'cpu') else np.array(grammar.final_sequence)
        rule_contours = grammar.rule_contours.cpu().numpy() if hasattr(grammar.rule_contours, 'cpu') else np.array(grammar.rule_contours)
        rule_children = grammar.rule_children.cpu().numpy() if hasattr(grammar.rule_children, 'cpu') else np.array(grammar.rule_children)
        rule_counts = grammar.rule_counts.cpu().numpy() if hasattr(grammar.rule_counts, 'cpu') else np.array(grammar.rule_counts)
    else:
        # For per-instrument mode, we don't have a single unified grammar
        # Store empty arrays - the actual data is in the patterns JSON
        final_sequence = np.array([])
        rule_contours = np.array([])
        rule_children = np.array([])
        rule_counts = np.array([])

    # Set version based on mode
    version = 'v54_per_instrument' if is_per_instrument else 'v53_pure_contour'

    data = {
        'version': np.array([version]),
        'is_per_instrument': np.array([is_per_instrument]),
        'n_terminals': np.array([grammar.n_terminals if grammar else 128]),
        'n_rules': np.array([grammar.n_rules if grammar else grammar_result['n_rules']]),
        'original_length': np.array([grammar.original_length if grammar else 0]),
        'compressed_length': np.array([grammar.compressed_length if grammar else 0]),
        'final_sequence': final_sequence,
        'rule_contours': rule_contours,
        'rule_children': rule_children,
        'rule_counts': rule_counts,
        'n_files': np.array([stats.get('n_files', 0)]),
        'n_tracks': np.array([len(tracks)]),
        'n_notes': np.array([sum(len(t) for t in tracks)]),
        'n_patterns': np.array([len(grammar_result['rules'])]),
        'n_transform_vocabulary': np.array([stats.get('n_transform_vocabulary', 0)]),
        'total_time': np.array([stats.get('total_time', 0)]),
        'patterns_json_file': np.array([os.path.basename(patterns_path)]),
    }

    if transform_discovery:
        transform_data = convert_numpy(transform_discovery)
        transforms_path = f'{base_path}_transforms.json'
        with open(transforms_path, 'w') as f:
            json.dump(transform_data, f)
        data['transforms_json_file'] = np.array([os.path.basename(transforms_path)])

    if multi_factor_discovery:
        mf_data = convert_numpy({k: v for k, v in multi_factor_discovery.items() if k != 'vocabulary_objects'})
        multi_factor_path = f'{base_path}_multi_factor.json'
        with open(multi_factor_path, 'w') as f:
            json.dump(mf_data, f)
        data['multi_factor_json_file'] = np.array([os.path.basename(multi_factor_path)])

    if track_derive_discovery and track_derive_discovery.get('n_derives', 0) > 0:
        td_data = convert_numpy(track_derive_discovery)
        track_derives_path = f'{base_path}_track_derives.json'
        with open(track_derives_path, 'w') as f:
            json.dump(td_data, f)
        data['track_derives_json_file'] = np.array([os.path.basename(track_derives_path)])

    if feature_importance_discovery and feature_importance_discovery.get('useful_features'):
        fi_data = convert_numpy(feature_importance_discovery)
        feature_importance_path = f'{base_path}_feature_importance.json'
        with open(feature_importance_path, 'w') as f:
            json.dump(fi_data, f)
        data['feature_importance_json_file'] = np.array([os.path.basename(feature_importance_path)])

    if meta_patterns:
        meta_data = convert_numpy(meta_patterns)
        meta_path = f'{base_path}_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f)
        data['meta_patterns_json_file'] = np.array([os.path.basename(meta_path)])

    np.savez_compressed(path, **data)

    if verbose:
        print(f"  Checkpoint saved to: {path}")


def run_pure_contour_pipeline(
    corpus_path: str,
    output_path: str = 'checkpoint_v54_per_instrument.npz',
    max_files: int = 100,
    min_count: int = 2,
    max_rules: int = 5000,
    device: str = 'cuda',
    workers: int = 4,
    per_instrument: bool = True,  # v54: Per-instrument pattern discovery
    top_instruments: int = 15,    # Only process top N instruments (0 = all)
    max_notes_per_instrument: int = 100000,  # Cap notes per instrument
    verbose: bool = True
) -> dict:
    """Run the full pure contour pipeline with all phases.

    Args:
        corpus_path: Path to MIDI corpus directory
        output_path: Output checkpoint path
        max_files: Maximum number of files to process
        min_count: Minimum pair count for rule creation
        max_rules: Maximum number of rules (total or per-instrument)
        device: 'cuda' or 'cpu'
        workers: Number of parallel workers for loading
        per_instrument: If True, run Re-Pair separately per GM program (v54)
                       If False, run unified Re-Pair on all tracks (v53)
        verbose: Print progress
    """

    total_start = time.time()
    stats = {}

    version = "v54 (Per-Instrument)" if per_instrument else "v53 (Unified)"

    if verbose:
        print("=" * 70)
        print(f"Pure Contour Re-Pair Pipeline {version}")
        print("=" * 70)
        print(f"Corpus: {corpus_path}")
        print(f"Output: {output_path}")
        print(f"Max files: {max_files}")
        print(f"Max rules: {max_rules} {'per instrument' if per_instrument else 'total'}")
        print(f"Device: {device}")
        print(f"Per-instrument: {per_instrument}")
        print()

    # =========================================================================
    # PHASE 1: Load MIDI files
    # =========================================================================
    phase_start = time.time()
    if verbose:
        print("[Phase 1] Loading MIDI files...", flush=True)

    midi_files = glob.glob(f"{corpus_path}/**/*.mid", recursive=True)
    midi_files = midi_files[:max_files]
    stats['n_files'] = len(midi_files)

    if verbose:
        print(f"  Found {len(midi_files)} MIDI files")

    all_tracks = []
    n_failed = 0
    n_loaded = 0
    total_files = len(midi_files)
    last_progress = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(load_midi_factored, f): f for f in midi_files}
        for future in as_completed(futures):
            try:
                tracks = future.result()
                if tracks:
                    all_tracks.extend(tracks)
                n_loaded += 1
            except Exception as e:
                n_failed += 1
                n_loaded += 1

            # Progress update every 10%
            if verbose:
                progress = (n_loaded * 100) // total_files
                if progress >= last_progress + 10:
                    last_progress = (progress // 10) * 10
                    print(f"    Loading: {n_loaded}/{total_files} files ({progress}%), {len(all_tracks)} tracks...", flush=True)

    stats['n_tracks'] = len(all_tracks)
    stats['n_notes'] = sum(len(t.pitch_classes) for t in all_tracks)
    stats['n_files_failed'] = n_failed
    stats['phase1_time'] = time.time() - phase_start

    if verbose:
        print(f"  Loaded {len(all_tracks)} tracks with {stats['n_notes']:,} notes in {stats['phase1_time']:.1f}s")

    # =========================================================================
    # PHASE 2: Build pure contour grammar (per-instrument or unified)
    # =========================================================================
    phase_start = time.time()

    if per_instrument:
        # v54: Per-instrument pattern discovery
        if verbose:
            print("\n[Phase 2] Building per-instrument grammars (v54)...", flush=True)

        tracks_by_gm = group_tracks_by_instrument(all_tracks)
        grammars_by_gm = build_per_instrument_grammars(
            all_tracks,
            device=device,
            min_pair_count=min_count,
            max_rules_per_instrument=max_rules,
            top_instruments=top_instruments,
            verbose=verbose,
        )

        grammar_result = convert_per_instrument_grammars_to_rules(
            grammars_by_gm, tracks_by_gm, verbose
        )

        # Store grammars for later use (e.g., checkpoint saving)
        # Use the first grammar for compatibility, but mark as multi-grammar
        grammar = list(grammars_by_gm.values())[0] if grammars_by_gm else None
        stats['n_instruments'] = len(grammars_by_gm)
        stats['per_instrument'] = True

        # Compute aggregate compression
        total_original = sum(g.original_length for g in grammars_by_gm.values())
        total_compressed = sum(g.compressed_length for g in grammars_by_gm.values())
        stats['compression'] = total_original / max(1, total_compressed)
        stats['n_patterns'] = grammar_result['n_rules']

        if verbose:
            print(f"\n  Per-instrument grammar stats:")
            print(f"    Instruments: {len(grammars_by_gm)}")
            print(f"    Total patterns: {grammar_result['n_rules']}")
            print(f"    Aggregate compression: {stats['compression']:.2f}x")
            for gm, g in sorted(grammars_by_gm.items()):
                print(f"    GM {gm}: {g.n_rules} patterns, {g.compression_ratio():.1f}x")

    else:
        # v53: Unified grammar (original behavior)
        if verbose:
            print("\n[Phase 2] Building unified grammar (v53)...", flush=True)

        grammar = build_pure_contour_grammar(
            all_tracks,
            device=device,
            min_pair_count=min_count,
            max_rules=max_rules,
            verbose=verbose,
        )

        grammar_result = convert_pure_contour_grammar_to_rules(grammar, all_tracks, verbose)
        grammars_by_gm = None
        tracks_by_gm = None

        stats['n_patterns'] = grammar_result['n_rules']
        stats['compression'] = grammar.compression_ratio()
        stats['per_instrument'] = False

        if verbose:
            print(f"\n  Grammar stats:")
            print(f"    Terminals: {grammar.n_terminals} (rhythm × velocity buckets)")
            print(f"    Rules: {grammar.n_rules}")
            print(f"    Compression: {grammar.compression_ratio():.2f}x")

    stats['phase2_time'] = time.time() - phase_start

    # =========================================================================
    # PHASE 3: Analyze patterns
    # =========================================================================
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 3] Analyzing patterns...", flush=True)

    rules = grammar_result['rules']
    total_occurrences = sum(r['count'] for r in rules.values())
    multi_piece = sum(1 for r in rules.values() if len(set(o['piece_id'] for o in r['occurrences'])) > 1)

    all_pieces = set()
    for r in rules.values():
        for occ in r['occurrences']:
            all_pieces.add(occ['piece_id'])

    # Count unique first_pitches per rule (shows transposition merging)
    merged_rules = 0
    for r in rules.values():
        first_pitches = set(o.get('first_pitch', 60) for o in r['occurrences'])
        if len(first_pitches) > 1:
            merged_rules += 1

    stats['total_occurrences'] = total_occurrences
    stats['multi_piece_patterns'] = multi_piece
    stats['pieces_covered'] = len(all_pieces)
    stats['merged_transposition_rules'] = merged_rules
    stats['phase3_time'] = time.time() - phase_start

    if verbose:
        print(f"  Total occurrences: {total_occurrences:,}")
        print(f"  Multi-piece patterns: {multi_piece}")
        print(f"  Pieces covered: {len(all_pieces)}")
        print(f"  Rules with merged transpositions: {merged_rules} (TRUE pitch-agnostic)")

    # =========================================================================
    # PHASE 4: Convert to FactoredPattern for MDL Transform Discovery
    # =========================================================================
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 4] Extracting canonical patterns for transform discovery...", flush=True)

    canonicals = convert_to_factored_patterns(rules, all_tracks)
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
            patterns_for_mf = [p.to_dict() if hasattr(p, 'to_dict') else {
                'pitch_classes': p.pitch_classes,
                'rhythm_ratios': getattr(p, '_rhythm_ratios', None) or getattr(p, 'rhythm_ratios', []),
                'velocity_ratios': getattr(p, '_velocity_ratios', None) or getattr(p, 'velocity_ratios', []),
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

        except Exception as e:
            if verbose:
                print(f"  Multi-factor discovery failed: {e}", flush=True)
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
            patterns_for_td = []
            for p in canonicals:
                p_dict = p.to_dict() if hasattr(p, 'to_dict') else {
                    'pitch_classes': p.pitch_classes,
                    'rhythm_ratios': getattr(p, '_rhythm_ratios', None) or [],
                    'velocity_ratios': getattr(p, '_velocity_ratios', None) or [],
                    'duration_ratios': getattr(p, '_duration_ratios', None) or [],
                }
                if hasattr(p, 'occurrences'):
                    p_dict['occurrences'] = [{'piece_id': o.piece_id, 'track_id': o.track_id, 'onset_time': o.onset_time} for o in p.occurrences]
                patterns_for_td.append(p_dict)

            track_instruments = {}
            for track in all_tracks:
                key = (track.piece_id, track.track_id)
                track_instruments[key] = track.gm_program

            track_derive_discovery = run_track_derive_discovery(
                patterns_for_td,
                track_instruments=track_instruments,
                device='cuda',
                add_to_occurrences=True,
                verbose=verbose
            )

            stats['n_track_derives'] = track_derive_discovery.get('n_derives', 0)

            if verbose and stats['n_track_derives'] > 0:
                print(f"  Added {stats['n_track_derives']} cross-track derivations to graph", flush=True)

        except Exception as e:
            if verbose:
                print(f"  TrackDerive discovery failed: {e}", flush=True)
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

    # Clear GPU memory before Phase 5d
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(canonicals) > 10:
        if verbose:
            print(f"\n[Phase 5d] Interval Magnitude Discovery (Diatonic vs Chromatic)...", flush=True)

        try:
            patterns_for_im = [{
                'pitch_classes': p.pitch_classes,
                'occurrences': [{'piece_id': o.piece_id, 'track_id': o.track_id, 'onset_time': o.onset_time, 'pitch_offset': getattr(o, 'pitch_offset', 0)} for o in p.occurrences] if hasattr(p, 'occurrences') else [],
            } for p in canonicals]

            interval_magnitude_discovery = run_interval_magnitude_discovery(
                patterns_for_im,
                device='cuda',
                verbose=verbose
            )

            stats['preferred_interval_repr'] = interval_magnitude_discovery.get('preferred_representation', 'chromatic')

        except Exception as e:
            if verbose:
                print(f"  Interval magnitude discovery failed: {e}", flush=True)
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

    # Clear GPU memory before Phase 5e
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(canonicals) > 20 and stats.get('n_transform_vocabulary', 0) > 0:
        if verbose:
            print(f"\n[Phase 5e] Feature Importance Discovery (MDL-Based)...", flush=True)

        try:
            if transform_discovery and 'lookup' in transform_discovery:
                transform_lookup = transform_discovery['lookup']
            else:
                patterns_for_fi = [{
                    'pitch_classes': p.pitch_classes,
                    'occurrences': [{'piece_id': o.piece_id, 'track_id': o.track_id, 'onset_time': o.onset_time}
                                    for o in p.occurrences] if hasattr(p, 'occurrences') else [],
                } for p in canonicals]

                transform_vocab = transform_discovery.get('vocabulary', [])
                if transform_vocab:
                    transform_lookup = build_transform_lookup_gpu(
                        patterns_for_fi, transform_vocab, device='cuda', verbose=False
                    )
                else:
                    transform_lookup = None

            if transform_lookup is not None:
                patterns_for_fi = [{
                    'pitch_classes': p.pitch_classes,
                    'occurrences': [{'piece_id': o.piece_id, 'track_id': o.track_id, 'onset_time': o.onset_time,
                                     'pitch_offset': 0, 'instrument': o.track_id}
                                    for o in p.occurrences] if hasattr(p, 'occurrences') else [],
                } for p in canonicals]

                feature_importance_discovery = run_feature_importance_discovery(
                    patterns_for_fi,
                    transform_lookup,
                    transform_names=transform_discovery.get('vocabulary', []),
                    min_gain_bits=50.0,
                    device='cuda',
                    verbose=verbose
                )

                stats['useful_features'] = feature_importance_discovery.get('useful_features', [])
                stats['n_useful_features'] = len(stats['useful_features'])

        except Exception as e:
            if verbose:
                print(f"  Feature importance discovery failed: {e}", flush=True)
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

    if len(rules) > 10:
        if verbose:
            print(f"\n[Phase 5g] Octave Equivalence Discovery (MDL-Based)...", flush=True)

        try:
            patterns_for_octave = [
                {'pitch_intervals': rule.get('pitch_intervals', []), 'count': rule.get('count', 1)}
                for rule in rules.values() if rule.get('pitch_intervals')
            ]

            if patterns_for_octave:
                octave_equivalence_discovery = run_octave_equivalence_discovery(
                    patterns=patterns_for_octave,
                    min_benefit_bits=100.0,
                    verbose=verbose,
                )

                stats['octave_equivalence_useful'] = octave_equivalence_discovery.get('octave_equivalence_useful', False)
                stats['octave_mdl_benefit'] = octave_equivalence_discovery.get('mdl_benefit_bits', 0.0)

                if octave_equivalence_discovery.get('octave_equivalence_useful', False):
                    if verbose:
                        print(f"  -> Octave equivalence DISCOVERED: {stats['octave_mdl_benefit']:.1f} bits saved", flush=True)
                else:
                    if verbose:
                        print(f"  -> Octave equivalence not beneficial for this corpus", flush=True)

        except Exception as e:
            if verbose:
                print(f"  Octave equivalence discovery failed: {e}", flush=True)
    else:
        if verbose:
            print(f"\n[Phase 5g] Skipping octave equivalence discovery (need >10 patterns)", flush=True)

    stats['phase5g_time'] = time.time() - phase_start

    # =========================================================================
    # PHASE 6: Level 3 Meta-Pattern Discovery
    # =========================================================================
    phase_start = time.time()
    meta_patterns = {}

    # Clear GPU memory before Phase 6
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"\n[Phase 6] Level 3 Meta-Pattern Discovery...", flush=True)

    if stats.get('n_transform_vocabulary', 0) > 0 and len(canonicals) > 10:
        try:
            patterns_for_l3 = [p.to_dict() if hasattr(p, 'to_dict') else {
                'pitch_classes': p.pitch_classes,
                'octaves': p.octaves,
                'velocities': p.velocities,
                'durations': p.durations,
                'rhythm_ioi': p.rhythm_ioi,
                'occurrences': [{'piece_id': o.piece_id, 'track_id': o.track_id, 'onset_time': o.onset_time, 'position': o.position}
                                for o in p.occurrences],
            } for p in canonicals]

            transform_vocab_parsed = [
                parse_transform_name(t) if isinstance(t, str) else t
                for t in transform_discovery.get('vocabulary', [])
            ]

            if verbose:
                print(f"  Building GPU transform lookup table...", flush=True)
            transform_lookup = build_transform_lookup_gpu(
                patterns_for_l3, transform_vocab_parsed, device='cuda', verbose=verbose
            )

            if verbose:
                print(f"  Extracting transform sequences per piece...", flush=True)
            sequences = extract_transform_sequences_gpu(
                patterns_for_l3, transform_lookup, device='cuda'
            )
            if verbose:
                print(f"    {len(sequences)} sequences extracted", flush=True)

            if len(sequences) >= 10:
                if verbose:
                    print(f"  Running GPU Re-Pair on transform sequences...", flush=True)
                meta_result = run_meta_repair_gpu(
                    sequences,
                    n_transforms=len(transform_vocab_parsed),
                    device='cuda',
                    verbose=verbose
                )

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

            # Orchestration rules
            if verbose:
                print(f"  Aggregating orchestration rules (vertical slices)...", flush=True)

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
                if verbose:
                    print(f"  Found {orchestration_result['n_rules']} orchestration rules", flush=True)

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

    save_checkpoint(
        output_path,
        grammar_result,
        grammar,
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
        print(f"  Files: {stats['n_files']} loaded, {stats.get('n_files_failed', 0)} failed")
        print(f"  Tracks: {stats['n_tracks']}")
        print(f"  Notes: {stats['n_notes']:,}")
        print(f"  Patterns: {stats['n_patterns']}")
        print(f"  Compression: {stats['compression']:.2f}x")
        print(f"  Total occurrences: {stats['total_occurrences']:,}")
        print(f"  Multi-piece patterns: {stats['multi_piece_patterns']}")
        print(f"  Rules with merged transpositions: {stats.get('merged_transposition_rules', 0)}")
        print(f"  Transform vocabulary (pitch): {stats.get('n_transform_vocabulary', 0)}")
        print(f"  Total time: {stats['total_time']:.1f}s")
        print(f"\n  Output: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Pure Contour Re-Pair Pipeline (v54 - Per-Instrument)',
        epilog='''
Examples:
  # Per-instrument mode (recommended for generation)
  python run_pure_contour_pipeline.py /path/to/corpus --per-instrument -o checkpoint_v54.npz

  # Unified mode (original v53 behavior, for analysis)
  python run_pure_contour_pipeline.py /path/to/corpus --no-per-instrument -o checkpoint_v53.npz
        '''
    )
    parser.add_argument('corpus_path', help='Path to MIDI corpus directory')
    parser.add_argument('--output', '-o', default='checkpoint_v54_per_instrument.npz',
                       help='Output checkpoint path')
    parser.add_argument('--max-files', type=int, default=100,
                       help='Maximum number of files to process')
    parser.add_argument('--max-rules', type=int, default=5000,
                       help='Maximum number of rules (per instrument if --per-instrument)')
    parser.add_argument('--min-count', type=int, default=2,
                       help='Minimum pair count for rule creation')
    parser.add_argument('--device', default='cuda',
                       help='Device for computation (cuda/cpu)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers for loading')
    parser.add_argument('--per-instrument', action='store_true', default=True,
                       help='Run Re-Pair separately per instrument (v54, default)')
    parser.add_argument('--no-per-instrument', dest='per_instrument', action='store_false',
                       help='Run unified Re-Pair on all tracks (v53 behavior)')
    parser.add_argument('--top-instruments', type=int, default=15,
                       help='Only process top N instruments by note count (0=all, default=15)')
    parser.add_argument('--max-notes', type=int, default=100000,
                       help='Max notes per instrument (0=unlimited, default=100000)')
    args = parser.parse_args()

    run_pure_contour_pipeline(
        corpus_path=args.corpus_path,
        output_path=args.output,
        max_files=args.max_files,
        min_count=args.min_count,
        max_rules=args.max_rules,
        device=args.device,
        workers=args.workers,
        per_instrument=args.per_instrument,
        top_instruments=args.top_instruments,
        verbose=True,
    )


if __name__ == '__main__':
    main()
