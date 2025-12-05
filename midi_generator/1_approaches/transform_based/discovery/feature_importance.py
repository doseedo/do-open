#!/usr/bin/env python3
"""
Feature Importance Discovery: MDL-Based Conditioning Variable Selection
========================================================================

This module discovers which stored features help predict transforms,
WITHOUT prescribing what those features mean.

Philosophy:
    - Don't implement "key discovery"
    - Implement "test if pitch_offset helps prediction"
    - If keys are real, pitch_offset will emerge as useful
    - If music is atonal, pitch_offset will be ignored

The system tests ALL stored occurrence-level features:
    - pitch_offset: Might emerge as "keys" or "tonal center"
    - track_id: Might emerge as "orchestration rules"
    - position_in_piece: Might emerge as "form structure"
    - instrument: Might emerge as "instrument-specific patterns"

MDL decides: Does conditioning on feature X reduce description length?

What Emerges (Without Prescription):
    - "pitch_offset=7 is 5x more common" → Human: "piece is in G"
    - "{0,4,5,7,11} behave similarly" → Human: "major scale degrees"
    - "pitch_offset predicts T3 vs T4" → Human: "diatonic thirds"
    - "pitch_offset shifts at bar 17" → Human: "modulation"

The system never learns "key" — it learns "pitch_offset is a useful
conditioning variable that clusters and predicts transforms."

Author: Feature Importance Discovery System
"""

import torch
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import math
import time

# Import from parent package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class FeatureImportance:
    """Result of testing a feature's predictive value."""
    feature_name: str
    compression_gain_bits: float  # How many bits saved by conditioning
    unconditional_entropy: float  # H(transform)
    conditional_entropy: float    # H(transform | feature)
    is_useful: bool               # gain > threshold
    feature_clusters: Dict[int, List[int]]  # feature_value -> common transforms

    def __str__(self):
        status = "✓ USEFUL" if self.is_useful else "✗ not useful"
        return (f"{self.feature_name}: {self.compression_gain_bits:.1f} bits "
                f"({self.unconditional_entropy:.2f} → {self.conditional_entropy:.2f}) "
                f"{status}")


@dataclass
class DiscoveredConditioning:
    """A discovered conditioning relationship."""
    feature_name: str
    feature_value: int
    predicted_transform: int
    probability: float
    count: int


def compute_entropy(counts: Dict[int, int]) -> float:
    """Compute Shannon entropy from count distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def compute_conditional_entropy(
    joint_counts: Dict[Tuple[int, int], int],
    feature_counts: Dict[int, int]
) -> float:
    """
    Compute H(Y|X) = sum_x P(x) * H(Y|X=x)

    Args:
        joint_counts: (feature_value, transform) -> count
        feature_counts: feature_value -> count
    """
    total = sum(feature_counts.values())
    if total == 0:
        return 0.0

    # Group by feature value
    transform_counts_given_feature = defaultdict(lambda: defaultdict(int))
    for (fv, t), count in joint_counts.items():
        transform_counts_given_feature[fv][t] = count

    # Weighted sum of conditional entropies
    conditional_entropy = 0.0
    for fv, fv_count in feature_counts.items():
        p_fv = fv_count / total
        h_given_fv = compute_entropy(dict(transform_counts_given_feature[fv]))
        conditional_entropy += p_fv * h_given_fv

    return conditional_entropy


def extract_transform_transitions(
    patterns: List[Dict],
    transform_lookup: np.ndarray,
) -> List[Dict]:
    """
    Extract all transform transitions with their context features.

    For each consecutive pattern pair in a piece/track, record:
        - The transform that occurred
        - The source pattern's features (pitch_offset, track_id, etc.)
        - DERIVED features (pitch_offset_relative, delta, etc.)

    Derived features allow cross-piece generalization:
        - pitch_offset = 7 means nothing across pieces
        - pitch_offset_relative = 0 means "on the tonal center" everywhere

    Returns:
        List of transition records with all available features
    """
    # Group occurrences by (piece, track) and sort by time
    track_sequences = defaultdict(list)

    for p_idx, p in enumerate(patterns):
        for occ in p.get('occurrences', []):
            key = (occ['piece_id'], occ.get('track_id', 0))
            track_sequences[key].append({
                'pattern_idx': p_idx,
                'onset_time': occ['onset_time'],
                'pitch_offset': occ.get('pitch_offset', 0),
                'track_id': occ.get('track_id', 0),
                'piece_id': occ['piece_id'],
                'instrument': occ.get('instrument', occ.get('track_id', 0)),
            })

    # Sort each track by time
    for key in track_sequences:
        track_sequences[key].sort(key=lambda x: x['onset_time'])

    # Compute pitch_offset mode (most common) per piece
    # This is NOT "key detection" — it's just "most frequent pitch class"
    piece_pitch_offsets = defaultdict(list)
    for (piece_id, track_id), occs in track_sequences.items():
        for occ in occs:
            piece_pitch_offsets[piece_id].append(occ['pitch_offset'])

    piece_mode = {}
    for piece_id, offsets in piece_pitch_offsets.items():
        if offsets:
            # Mode = most frequent value
            counts = Counter(offsets)
            piece_mode[piece_id] = counts.most_common(1)[0][0]
        else:
            piece_mode[piece_id] = 0

    # Extract transitions with derived features
    transitions = []

    for (piece_id, track_id), occs in track_sequences.items():
        piece_length = occs[-1]['onset_time'] if occs else 1
        mode = piece_mode.get(piece_id, 0)

        prev_pitch_offset = None

        for i in range(len(occs) - 1):
            src = occs[i]
            tgt = occs[i + 1]

            src_idx = src['pattern_idx']
            tgt_idx = tgt['pattern_idx']

            # Skip self-transitions
            if src_idx == tgt_idx:
                continue

            # Get transform from lookup
            transform = transform_lookup[src_idx, tgt_idx]
            if transform < 0:
                continue  # No valid transform

            # Compute position as fraction of piece (0.0 to 1.0)
            position = src['onset_time'] / piece_length if piece_length > 0 else 0
            position_bucket = int(position * 10)  # 10 buckets

            # DERIVED FEATURES (not prescribed, just computed)
            pitch_offset = src['pitch_offset']

            # pitch_offset_relative: offset from piece mode
            # This is "scale degree" without calling it that
            pitch_offset_relative = (pitch_offset - mode) % 12

            # pitch_offset_delta: change from previous occurrence
            if prev_pitch_offset is not None:
                pitch_offset_delta = (pitch_offset - prev_pitch_offset) % 12
            else:
                pitch_offset_delta = 0

            prev_pitch_offset = pitch_offset

            transitions.append({
                'transform': int(transform),

                # Stored features
                'pitch_offset': pitch_offset,
                'track_id': src['track_id'],
                'instrument': src['instrument'],
                'position_bucket': position_bucket,
                'piece_id': piece_id,
                'src_pattern': src_idx,

                # DERIVED features (computed, not prescribed)
                'pitch_offset_relative': pitch_offset_relative,  # "scale degree" without saying it
                'pitch_offset_delta': pitch_offset_delta,        # melodic motion
                'piece_mode': mode,                              # for reference
            })

    return transitions


def test_feature_importance(
    transitions: List[Dict],
    feature_name: str,
    min_gain_bits: float = 10.0,
) -> FeatureImportance:
    """
    Test if conditioning on a feature improves transform prediction.

    Args:
        transitions: List of transition records
        feature_name: Which feature to test ('pitch_offset', 'track_id', etc.)
        min_gain_bits: Minimum bits saved to consider feature useful

    Returns:
        FeatureImportance with compression gain and usefulness
    """
    if not transitions:
        return FeatureImportance(
            feature_name=feature_name,
            compression_gain_bits=0.0,
            unconditional_entropy=0.0,
            conditional_entropy=0.0,
            is_useful=False,
            feature_clusters={},
        )

    # Count transforms unconditionally
    transform_counts = Counter(t['transform'] for t in transitions)

    # Count (feature, transform) pairs
    joint_counts = Counter(
        (t[feature_name], t['transform']) for t in transitions
    )

    # Count feature values
    feature_counts = Counter(t[feature_name] for t in transitions)

    # Compute entropies
    h_unconditional = compute_entropy(dict(transform_counts))
    h_conditional = compute_conditional_entropy(dict(joint_counts), dict(feature_counts))

    # Information gain (bits saved per transition)
    info_gain_per_transition = h_unconditional - h_conditional

    # Total bits saved across all transitions
    total_bits_saved = info_gain_per_transition * len(transitions)

    # Build feature clusters: which transforms are common for each feature value
    feature_clusters = {}
    for fv in feature_counts:
        transforms_for_fv = [t['transform'] for t in transitions if t[feature_name] == fv]
        transform_dist = Counter(transforms_for_fv)
        # Store top 3 transforms for this feature value
        feature_clusters[fv] = [t for t, _ in transform_dist.most_common(3)]

    return FeatureImportance(
        feature_name=feature_name,
        compression_gain_bits=total_bits_saved,
        unconditional_entropy=h_unconditional,
        conditional_entropy=h_conditional,
        is_useful=total_bits_saved > min_gain_bits,
        feature_clusters=feature_clusters,
    )


def discover_feature_clusters(
    transitions: List[Dict],
    feature_name: str,
    transform_names: List[str] = None,
) -> Dict[str, any]:
    """
    Discover how feature values cluster based on transform behavior.

    This is where "keys" emerge without being prescribed:
    - If pitch_offsets {0, 4, 5, 7, 11} all prefer T4 (major third)
    - And pitch_offsets {2, 9} prefer T3 (minor third)
    - The system discovers these clusters without knowing "major scale"

    Returns:
        Dict with cluster info and behavioral patterns
    """
    # Group transitions by feature value
    by_feature = defaultdict(list)
    for t in transitions:
        by_feature[t[feature_name]].append(t['transform'])

    # Compute transform distribution for each feature value
    distributions = {}
    for fv, transforms in by_feature.items():
        dist = Counter(transforms)
        total = len(transforms)
        distributions[fv] = {t: c / total for t, c in dist.items()}

    # Find similar feature values (cluster by behavior)
    # Using simple heuristic: which transform is most common?
    clusters_by_dominant = defaultdict(list)
    for fv, dist in distributions.items():
        if dist:
            dominant = max(dist, key=dist.get)
            clusters_by_dominant[dominant].append(fv)

    # Format output
    cluster_info = {
        'n_feature_values': len(distributions),
        'n_behavioral_clusters': len(clusters_by_dominant),
        'clusters': {},
    }

    for dominant_transform, feature_values in clusters_by_dominant.items():
        t_name = transform_names[dominant_transform] if transform_names and dominant_transform < len(transform_names) else f"T{dominant_transform}"
        cluster_info['clusters'][t_name] = {
            'feature_values': sorted(feature_values),
            'size': len(feature_values),
        }

    return cluster_info


def detect_feature_shifts(
    transitions: List[Dict],
    feature_name: str,
    window_size: int = 20,
) -> List[Dict]:
    """
    Detect where a feature's distribution shifts within pieces.

    This is where "modulation" emerges without being prescribed:
    - If pitch_offset distribution changes at bar 17
    - The system notices the shift without knowing "modulation"

    Returns:
        List of detected shifts with position and magnitude
    """
    # Group by piece
    by_piece = defaultdict(list)
    for t in transitions:
        by_piece[t['piece_id']].append(t)

    shifts = []

    for piece_id, piece_transitions in by_piece.items():
        if len(piece_transitions) < window_size * 2:
            continue

        # Sort by position
        piece_transitions.sort(key=lambda x: x.get('position_bucket', 0))

        # Sliding window comparison
        for i in range(window_size, len(piece_transitions) - window_size):
            window_before = piece_transitions[i - window_size:i]
            window_after = piece_transitions[i:i + window_size]

            # Compare feature distributions
            dist_before = Counter(t[feature_name] for t in window_before)
            dist_after = Counter(t[feature_name] for t in window_after)

            # Find mode (most common value) in each window
            mode_before = dist_before.most_common(1)[0][0] if dist_before else None
            mode_after = dist_after.most_common(1)[0][0] if dist_after else None

            if mode_before != mode_after and mode_before is not None:
                # Detected a shift!
                shifts.append({
                    'piece_id': piece_id,
                    'position': i / len(piece_transitions),
                    'feature': feature_name,
                    'from_value': mode_before,
                    'to_value': mode_after,
                    'delta': (mode_after - mode_before) % 12 if feature_name == 'pitch_offset' else mode_after - mode_before,
                })

    return shifts


def run_feature_importance_discovery(
    patterns: List[Dict],
    transform_lookup: np.ndarray,
    transform_names: List[str] = None,
    min_gain_bits: float = 50.0,
    device: str = 'cuda',
    verbose: bool = True,
) -> Dict:
    """
    Discover which features help predict transforms.

    This is Phase 5e: pure MDL-based feature selection.
    No music theory prescribed — useful features emerge from compression.

    Args:
        patterns: List of pattern dicts with occurrences
        transform_lookup: [n_patterns, n_patterns] -> transform_id
        transform_names: Optional names for transforms
        min_gain_bits: Minimum total bits saved to consider useful
        device: PyTorch device (unused currently, for future GPU acceleration)
        verbose: Print progress

    Returns:
        Dict with:
            - useful_features: List of features that help compression
            - feature_results: Full FeatureImportance for each tested feature
            - discovered_clusters: Behavioral clusters for useful features
            - detected_shifts: Feature distribution shifts (e.g., modulations)
    """
    t0 = time.time()

    if verbose:
        print(f"\n[Feature Importance Discovery]")
        print(f"  Extracting transform transitions...")

    # Convert torch tensor to numpy if needed
    if hasattr(transform_lookup, 'cpu'):
        transform_lookup = transform_lookup.cpu().numpy()

    # Extract all transitions with features
    transitions = extract_transform_transitions(patterns, transform_lookup)

    if verbose:
        print(f"  Found {len(transitions)} transform transitions")

    if len(transitions) < 100:
        if verbose:
            print(f"  Too few transitions for meaningful analysis")
        return {
            'useful_features': [],
            'feature_results': {},
            'discovered_clusters': {},
            'detected_shifts': [],
        }

    # Features to test (stored AND derived)
    # Derived features allow cross-piece generalization
    candidate_features = [
        # Stored features
        'pitch_offset',           # Might emerge as "piece-specific key"
        'track_id',               # Might emerge as "orchestration rules"
        'instrument',             # Might emerge as "instrument-specific behavior"
        'position_bucket',        # Might emerge as "form structure"

        # DERIVED features (computed, not prescribed)
        'pitch_offset_relative',  # Might emerge as "scale degrees" (generalizes across pieces!)
        'pitch_offset_delta',     # Might emerge as "melodic motion patterns"
    ]

    if verbose:
        print(f"  Testing {len(candidate_features)} candidate features...")

    # Test each feature
    feature_results = {}
    useful_features = []

    for feature in candidate_features:
        # Check if feature exists in transitions
        if feature not in transitions[0]:
            continue

        result = test_feature_importance(
            transitions,
            feature,
            min_gain_bits=min_gain_bits,
        )

        feature_results[feature] = result

        if result.is_useful:
            useful_features.append(feature)

        if verbose:
            print(f"    {result}")

    # Discover clusters for useful features
    discovered_clusters = {}
    for feature in useful_features:
        clusters = discover_feature_clusters(
            transitions,
            feature,
            transform_names=transform_names,
        )
        discovered_clusters[feature] = clusters

        if verbose:
            print(f"\n  {feature} clusters:")
            for t_name, info in clusters['clusters'].items():
                vals = info['feature_values'][:8]  # Show first 8
                suffix = "..." if len(info['feature_values']) > 8 else ""
                print(f"    {t_name}-preferring: {vals}{suffix}")

    # Detect shifts in useful features (modulations, etc.)
    detected_shifts = []
    for feature in useful_features:
        if feature == 'pitch_offset':  # Most interesting for shifts
            shifts = detect_feature_shifts(transitions, feature)
            detected_shifts.extend(shifts)

    if verbose and detected_shifts:
        print(f"\n  Detected {len(detected_shifts)} feature shifts (potential modulations):")
        for shift in detected_shifts[:5]:
            print(f"    {shift['piece_id']}: {shift['feature']} "
                  f"{shift['from_value']} → {shift['to_value']} "
                  f"at position {shift['position']:.1%}")

    elapsed = time.time() - t0

    if verbose:
        print(f"\n  Discovery complete in {elapsed:.1f}s")
        print(f"  Useful features: {useful_features}")

    return {
        'useful_features': useful_features,
        'feature_results': {k: {
            'name': v.feature_name,
            'gain_bits': v.compression_gain_bits,
            'unconditional_entropy': v.unconditional_entropy,
            'conditional_entropy': v.conditional_entropy,
            'is_useful': v.is_useful,
        } for k, v in feature_results.items()},
        'discovered_clusters': discovered_clusters,
        'detected_shifts': detected_shifts[:100],  # Limit for storage
        'n_transitions': len(transitions),
        'elapsed_time': elapsed,
    }


if __name__ == "__main__":
    print("Feature Importance Discovery Module")
    print("=" * 50)
    print()
    print("This module discovers which features help predict transforms")
    print("WITHOUT prescribing music theory.")
    print()
    print("If 'keys' are real, pitch_offset will emerge as useful.")
    print("If music is atonal, pitch_offset will be ignored.")
    print()
    print("What emerges vs human interpretation:")
    print("  'pitch_offset clusters'     → 'piece is in G'")
    print("  '{0,4,5,7,11} similar'      → 'major scale degrees'")
    print("  'pitch_offset predicts T3/T4' → 'diatonic thirds'")
    print("  'pitch_offset shifts'       → 'modulation'")
