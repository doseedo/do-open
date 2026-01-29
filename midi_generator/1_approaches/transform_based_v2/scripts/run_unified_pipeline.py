#!/usr/bin/env python3
"""
Unified MDL Pipeline with Proper Vertical Orchestration Discovery.

Key fix: Separates horizontal (temporal) from vertical (cross-track) relations.

Level 1-2: Same as before (pattern discovery, grammar)
Level 3 NEW:
  - Per-track horizontal sequences (temporal patterns within one instrument)
  - Per-time-slice vertical slices (patterns at same onset across tracks)
  - Orchestration rule aggregation (dominant transform per track pair)
"""

import argparse
import json
import time
import sys
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# =============================================================================
# IMPORTS FROM EXISTING PIPELINE
# =============================================================================

from scripts.run_factored_pipeline import (
    load_midi_factored,
    run_repair_gpu_factored,
    run_factored_mdl_transform_discovery,
    extract_factored_canonical_patterns,
    _convert_numpy_types,
    FactoredTrack,
    factored_track_to_tokens,
)


# =============================================================================
# LEVEL 3: ORCHESTRATION DISCOVERY (FIXED)
# =============================================================================

@dataclass
class OrchestrationRule:
    """A persistent transform relationship between two tracks."""
    source_track: int  # Track ID
    target_track: int  # Track ID
    transform: str  # Dominant transform name
    frequency: int  # How many times this transform occurs
    total_pairs: int  # Total pairs analyzed
    confidence: float  # frequency / total_pairs

    def to_dict(self):
        return {
            'source_track': self.source_track,
            'target_track': self.target_track,
            'transform': self.transform,
            'frequency': self.frequency,
            'total_pairs': self.total_pairs,
            'confidence': self.confidence,
        }


@dataclass
class HorizontalSequence:
    """Transform sequence within a single track (temporal)."""
    piece_id: str
    track_id: int
    transforms: List[str]  # Transform names in temporal order
    pattern_ids: List[int]  # Pattern IDs involved


@dataclass
class VerticalSlice:
    """Patterns occurring at the same time across tracks."""
    piece_id: str
    onset_time: int
    track_patterns: Dict[int, int]  # track_id -> pattern_id


def get_transform_between_patterns(
    p1: FactoredPattern,
    p2: FactoredPattern,
    transform_vocab: List[str]
) -> Optional[str]:
    """
    Find which transform maps p1 -> p2 (pitch class only).
    Returns transform name or None if no match.
    """
    if len(p1.pitch_classes) != len(p2.pitch_classes):
        return None

    pc1 = np.array(p1.pitch_classes)
    pc2 = np.array(p2.pitch_classes)

    # Check transpositions T0-T11
    for t in range(12):
        if np.array_equal((pc1 + t) % 12, pc2):
            return f"T{t}" if t > 0 else "identity"

    # Check inversions I0-I11
    for axis in range(12):
        if np.array_equal((axis - pc1) % 12, pc2):
            return f"I{axis}"

    return None


def extract_per_track_sequences(
    patterns: List[FactoredPattern],
    transform_vocab: List[str],
    min_seq_length: int = 3,
    verbose: bool = True
) -> Tuple[List[HorizontalSequence], List[VerticalSlice]]:
    """
    Extract horizontal sequences (per-track) and vertical slices (same-time).

    This is the FIX: Instead of sorting all patterns by time across tracks,
    we process per-track for horizontal and per-time-slice for vertical.
    """
    # Build occurrence index: piece_id -> track_id -> [(onset, pattern_idx)]
    occurrence_index = defaultdict(lambda: defaultdict(list))

    for p_idx, pattern in enumerate(patterns):
        for occ in pattern.occurrences:
            piece_id = occ['piece_id']
            track_id = occ['track_id']
            onset = occ['onset_time']
            occurrence_index[piece_id][track_id].append((onset, p_idx))

    # === HORIZONTAL: Per-track temporal sequences ===
    horizontal_sequences = []

    for piece_id, tracks in occurrence_index.items():
        for track_id, occurrences in tracks.items():
            if len(occurrences) < min_seq_length:
                continue

            # Sort by onset within this track
            occurrences.sort(key=lambda x: x[0])

            # Find transforms between consecutive patterns
            transforms = []
            pattern_ids = []

            for i in range(len(occurrences) - 1):
                _, p1_idx = occurrences[i]
                _, p2_idx = occurrences[i + 1]

                t = get_transform_between_patterns(
                    patterns[p1_idx], patterns[p2_idx], transform_vocab
                )

                if t is not None:
                    transforms.append(t)
                    pattern_ids.append(p1_idx)

            if len(transforms) >= min_seq_length - 1:
                horizontal_sequences.append(HorizontalSequence(
                    piece_id=piece_id,
                    track_id=track_id,
                    transforms=transforms,
                    pattern_ids=pattern_ids,
                ))

    # === VERTICAL: Same-time slices across tracks ===
    vertical_slices = []

    for piece_id, tracks in occurrence_index.items():
        # Build time -> track -> pattern_idx
        time_slices = defaultdict(dict)

        for track_id, occurrences in tracks.items():
            for onset, p_idx in occurrences:
                time_slices[onset][track_id] = p_idx

        # Keep slices with multiple tracks
        for onset, track_patterns in time_slices.items():
            if len(track_patterns) >= 2:
                vertical_slices.append(VerticalSlice(
                    piece_id=piece_id,
                    onset_time=onset,
                    track_patterns=dict(track_patterns),
                ))

    if verbose:
        print(f"    Horizontal sequences: {len(horizontal_sequences)}")
        print(f"    Vertical slices: {len(vertical_slices)}")

    return horizontal_sequences, vertical_slices


def aggregate_orchestration_rules(
    vertical_slices: List[VerticalSlice],
    patterns: List[FactoredPattern],
    transform_vocab: List[str],
    min_confidence: float = 0.3,
    min_frequency: int = 5,
    verbose: bool = True
) -> List[OrchestrationRule]:
    """
    Aggregate cross-track relations by instrument pair to find dominant transforms.

    Key insight: If sax and strings consistently have I5 relationship,
    this emerges as an orchestration rule with high confidence.
    """
    # Count transforms per track pair
    pair_transforms = defaultdict(Counter)  # (src_track, tgt_track) -> Counter of transforms

    for slice_ in vertical_slices:
        track_ids = sorted(slice_.track_patterns.keys())

        # Check each pair of tracks in this slice
        for i, src_track in enumerate(track_ids):
            for tgt_track in track_ids[i+1:]:
                src_pattern_idx = slice_.track_patterns[src_track]
                tgt_pattern_idx = slice_.track_patterns[tgt_track]

                t = get_transform_between_patterns(
                    patterns[src_pattern_idx],
                    patterns[tgt_pattern_idx],
                    transform_vocab
                )

                if t is not None:
                    pair_transforms[(src_track, tgt_track)][t] += 1

    # Find dominant transform per pair
    orchestration_rules = []

    for (src_track, tgt_track), counts in pair_transforms.items():
        total = sum(counts.values())
        if total < min_frequency:
            continue

        # Get most common transform
        dominant_transform, freq = counts.most_common(1)[0]
        confidence = freq / total

        if confidence >= min_confidence:
            orchestration_rules.append(OrchestrationRule(
                source_track=src_track,
                target_track=tgt_track,
                transform=dominant_transform,
                frequency=freq,
                total_pairs=total,
                confidence=confidence,
            ))

    # Sort by confidence
    orchestration_rules.sort(key=lambda r: -r.confidence)

    if verbose:
        print(f"    Orchestration rules: {len(orchestration_rules)}")
        for rule in orchestration_rules[:10]:
            print(f"      Track {rule.source_track} -> {rule.target_track}: "
                  f"{rule.transform} ({rule.confidence:.1%}, n={rule.frequency})")

    return orchestration_rules


def run_horizontal_meta_repair(
    horizontal_sequences: List[HorizontalSequence],
    transform_vocab: List[str],
    device: str = 'cuda',
    verbose: bool = True
) -> Dict:
    """
    Run Re-Pair on per-track horizontal sequences.

    This finds temporal meta-patterns like II-V-I progressions.
    """
    from grammar.v2.repair_gpu_v2 import build_repair_grammar_v2

    # Convert transform names to IDs
    t_to_id = {t: i for i, t in enumerate(transform_vocab)}

    token_sequences = []
    for seq in horizontal_sequences:
        tokens = [t_to_id.get(t, 0) for t in seq.transforms if t in t_to_id]
        if len(tokens) >= 2:
            token_sequences.append(tokens)

    if verbose:
        total_tokens = sum(len(s) for s in token_sequences)
        print(f"    Horizontal meta-repair: {len(token_sequences)} sequences, "
              f"{total_tokens} transform tokens")

    if len(token_sequences) < 10:
        if verbose:
            print(f"    Too few sequences for meta-pattern discovery")
        return {'rules': [], 'n_rules': 0, 'compression_ratio': 1.0}

    try:
        grammar = build_repair_grammar_v2(
            token_sequences,
            device=device,
            max_rules=500,
            min_frequency=5,
            verbose=False
        )

        # Extract meta-patterns
        meta_rules = []
        for i in range(grammar.n_rules):
            rule_id = grammar.n_terminals + i
            expansion = grammar.expand_rule(rule_id)

            if expansion is not None and len(expansion) >= 2:
                # Convert IDs back to transform names
                transform_names = [transform_vocab[t] if t < len(transform_vocab) else f"T{t}"
                                   for t in expansion]
                meta_rules.append({
                    'id': f"M{i}",
                    'transforms': transform_names,
                    'length': len(expansion),
                })

        if verbose:
            print(f"    Discovered {len(meta_rules)} horizontal meta-patterns")
            print(f"    Compression: {grammar.compression_ratio():.2f}x")

        return {
            'rules': meta_rules,
            'n_rules': len(meta_rules),
            'compression_ratio': grammar.compression_ratio(),
        }
    except Exception as e:
        if verbose:
            print(f"    Meta-repair error: {e}")
        return {'rules': [], 'n_rules': 0, 'compression_ratio': 1.0}


def run_level3_unified(
    patterns: List[FactoredPattern],
    transform_vocab: List[str],
    device: str = 'cuda',
    verbose: bool = True
) -> Dict:
    """
    Run Level 3 with proper separation of horizontal and vertical relations.
    """
    if verbose:
        print(f"\n[Level 3] Unified Relation Discovery")
        print(f"  Patterns: {len(patterns)}")
        print(f"  Transforms: {len(transform_vocab)}")

    # Step 1: Extract horizontal and vertical
    if verbose:
        print(f"\n  Step 1: Extracting sequences...")
    horizontal, vertical = extract_per_track_sequences(
        patterns, transform_vocab, verbose=verbose
    )

    # Step 2: Aggregate orchestration rules (VERTICAL)
    if verbose:
        print(f"\n  Step 2: Aggregating orchestration rules...")
    orchestration_rules = aggregate_orchestration_rules(
        vertical, patterns, transform_vocab, verbose=verbose
    )

    # Step 3: Meta-repair on horizontal sequences
    if verbose:
        print(f"\n  Step 3: Meta-pattern discovery (horizontal)...")
    meta_patterns = run_horizontal_meta_repair(
        horizontal, transform_vocab, device=device, verbose=verbose
    )

    return {
        'horizontal_sequences': len(horizontal),
        'vertical_slices': len(vertical),
        'orchestration_rules': [r.to_dict() for r in orchestration_rules],
        'n_orchestration_rules': len(orchestration_rules),
        'meta_patterns': meta_patterns['rules'],
        'n_meta_patterns': meta_patterns['n_rules'],
        'meta_compression': meta_patterns['compression_ratio'],
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_unified_pipeline(
    corpus_path: str,
    output_path: str = 'checkpoint_unified.npz',
    max_files: int = 500,
    verbose: bool = True
) -> Dict:
    """
    Run the unified MDL pipeline with proper orchestration discovery.
    """
    stats = PipelineStats()
    total_start = time.time()

    if verbose:
        print("=" * 70)
        print("UNIFIED MDL PIPELINE (with orchestration fix)")
        print("=" * 70)

    # Phase 1: Load MIDI files
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 1] Loading MIDI files...")

    all_objects = load_midi_factored(corpus_path, max_files, verbose)

    stats.n_files = len(set(obj.piece_id for obj in all_objects))
    stats.n_tracks = len(set((obj.piece_id, obj.track_id) for obj in all_objects))
    stats.n_notes = sum(obj.num_notes for obj in all_objects)
    stats.phase_times['load'] = time.time() - phase_start

    if verbose:
        print(f"  Loaded: {stats.n_files} files, {stats.n_tracks} tracks, {stats.n_notes} notes")

    # Phase 2: Factored Re-Pair
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 2] Factored Re-Pair Grammar...")

    canonicals, grammar = run_factored_repair_gpu(all_objects, device='cuda', verbose=verbose)

    stats.n_canonical_patterns = len(canonicals)
    stats.n_grammar_rules = grammar['n_rules']
    stats.phase_times['repair'] = time.time() - phase_start

    # Phase 3: Transform Vocabulary
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 3] Building Transform Vocabulary...")

    transform_discovery = build_transform_vocabulary(canonicals, device='cuda', verbose=verbose)

    stats.n_transform_vocabulary = len(transform_discovery['vocabulary'])
    stats.phase_times['transform'] = time.time() - phase_start

    # Phase 4: Level 3 - Unified Relations (THE FIX)
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 4] Level 3: Unified Relation Discovery...")

    level3_results = run_level3_unified(
        canonicals,
        transform_discovery['vocabulary'],
        device='cuda',
        verbose=verbose
    )

    stats.phase_times['level3'] = time.time() - phase_start

    # Phase 5: Save checkpoint
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 5] Saving checkpoint...")

    # Prepare data for saving
    canonicals_data = [p.to_dict() for p in canonicals]
    grammar_rules_data = grammar.get('rules_list', [])

    data = {
        # Basic stats
        'n_files': np.array([stats.n_files]),
        'n_tracks': np.array([stats.n_tracks]),
        'n_notes': np.array([stats.n_notes]),
        'n_canonicals': np.array([len(canonicals)]),
        'n_grammar_rules': np.array([grammar['n_rules']]),
        'n_transform_vocab': np.array([len(transform_discovery['vocabulary'])]),
        'n_orchestration_rules': np.array([level3_results['n_orchestration_rules']]),
        'n_meta_patterns': np.array([level3_results['n_meta_patterns']]),

        # Flags
        'is_factored': np.array([True]),
        'has_orchestration': np.array([True]),

        # JSON data
        'canonical_patterns_json': np.array([json.dumps(_convert_numpy_types(canonicals_data))]),
        'grammar_rules_json': np.array([json.dumps(_convert_numpy_types(grammar_rules_data))]),
        'transform_vocabulary_json': np.array([json.dumps(_convert_numpy_types(transform_discovery['vocabulary']))]),
        'orchestration_rules_json': np.array([json.dumps(_convert_numpy_types(level3_results['orchestration_rules']))]),
        'meta_patterns_json': np.array([json.dumps(_convert_numpy_types(level3_results['meta_patterns']))]),
    }

    np.savez_compressed(output_path, **data)

    stats.phase_times['save'] = time.time() - phase_start
    stats.total_time = time.time() - total_start

    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  Files: {stats.n_files}")
        print(f"  Tracks: {stats.n_tracks}")
        print(f"  Notes: {stats.n_notes}")
        print(f"  Canonical patterns: {stats.n_canonical_patterns}")
        print(f"  Grammar rules: {stats.n_grammar_rules}")
        print(f"  Transform vocabulary: {stats.n_transform_vocabulary}")
        print(f"  Orchestration rules: {level3_results['n_orchestration_rules']}")
        print(f"  Meta-patterns: {level3_results['n_meta_patterns']}")
        print(f"  Total time: {stats.total_time:.1f}s")
        print(f"\n  Saved to: {output_path}")

    return {
        'stats': stats,
        'level3': level3_results,
    }


def main():
    parser = argparse.ArgumentParser(description='Unified MDL Pipeline')
    parser.add_argument('--corpus', required=True, help='Path to MIDI corpus')
    parser.add_argument('--output', default='checkpoint_unified.npz', help='Output checkpoint path')
    parser.add_argument('--max-files', type=int, default=500, help='Max MIDI files to process')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    args = parser.parse_args()

    run_unified_pipeline(
        corpus_path=args.corpus,
        output_path=args.output,
        max_files=args.max_files,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
