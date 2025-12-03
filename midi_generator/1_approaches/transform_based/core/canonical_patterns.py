"""
Canonical Pattern Normalization and Deduplication
==================================================

The core insight: patterns that differ only by transposition are THE SAME pattern.

Before (broken):
    [0,4,7], [1,5,8], [2,6,9], ... = 12 different patterns

After (correct):
    [0,4,7] = canonical pattern
    [1,5,8] = T1([0,4,7])
    [2,6,9] = T2([0,4,7])

This dramatically improves:
1. Coverage: Same melody in 12 keys = 1 pattern (not 12)
2. Compression: Fewer patterns, more transform edges
3. Cross-track linking: Trumpet in C + Sax in Eb = same pattern + T transform
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CanonicalPattern:
    """A pattern normalized to start on pitch-class 0."""
    id: str
    pitch_classes: Tuple[int, ...]  # Always starts with 0
    length: int

    # Optional: preserve other factors for reconstruction
    duration_buckets: Optional[Tuple[int, ...]] = None
    velocity_buckets: Optional[Tuple[int, ...]] = None

    # Statistics
    occurrence_count: int = 0

    def __hash__(self):
        return hash(self.pitch_classes)

    def __eq__(self, other):
        if not isinstance(other, CanonicalPattern):
            return False
        return self.pitch_classes == other.pitch_classes


@dataclass
class PatternOccurrence:
    """An occurrence of a canonical pattern with transform info."""
    pattern_id: str
    piece_id: str
    track_id: int
    onset_time: int

    # The key insight: store the transposition offset
    t_offset: int  # 0-11, how much to transpose canonical pattern

    # Optional: octave info for full reconstruction
    octaves: Optional[Tuple[int, ...]] = None
    velocities: Optional[Tuple[int, ...]] = None
    durations: Optional[Tuple[int, ...]] = None


class CanonicalPatternRegistry:
    """
    Registry for T-normalized patterns with cross-track deduplication.

    Usage:
        registry = CanonicalPatternRegistry()

        # Add patterns - automatically normalizes and deduplicates
        pid, t_offset = registry.add_pattern([3, 7, 10])  # Eb major
        # Returns: ("P_0_4_7", 3) - canonical C major + T3

        pid2, t_offset2 = registry.add_pattern([0, 4, 7])  # C major
        # Returns: ("P_0_4_7", 0) - same pattern + T0
    """

    def __init__(self):
        # Canonical patterns by their normalized form
        self._patterns: Dict[Tuple[int, ...], CanonicalPattern] = {}

        # All occurrences
        self._occurrences: List[PatternOccurrence] = []

        # Index: pattern_id -> list of occurrence indices
        self._occurrences_by_pattern: Dict[str, List[int]] = defaultdict(list)

        # Counter for unique pattern IDs
        self._next_id = 0

    def normalize_pitch_classes(self, pitch_classes: List[int]) -> Tuple[Tuple[int, ...], int]:
        """
        Normalize pitch classes so first note = 0.

        Args:
            pitch_classes: List of pitch classes (0-11)

        Returns:
            (normalized_tuple, t_offset) where:
            - normalized_tuple: pitch classes transposed so first = 0
            - t_offset: transposition to apply to get back to original
        """
        if not pitch_classes:
            return tuple(), 0

        base = pitch_classes[0]
        normalized = tuple((pc - base) % 12 for pc in pitch_classes)
        return normalized, base

    def get_pattern_id(self, normalized: Tuple[int, ...]) -> str:
        """Generate a consistent pattern ID from normalized pitch classes."""
        return f"P_{'_'.join(str(p) for p in normalized)}"

    def add_pattern(
        self,
        pitch_classes: List[int],
        piece_id: str = "",
        track_id: int = 0,
        onset_time: int = 0,
        octaves: Optional[List[int]] = None,
        velocities: Optional[List[int]] = None,
        durations: Optional[List[int]] = None,
        duration_buckets: Optional[List[int]] = None,
        velocity_buckets: Optional[List[int]] = None,
    ) -> Tuple[str, int]:
        """
        Add a pattern, automatically normalizing and deduplicating.

        Returns:
            (pattern_id, t_offset) - the canonical pattern ID and transposition
        """
        # Normalize
        normalized, t_offset = self.normalize_pitch_classes(pitch_classes)

        # Check if canonical form exists
        if normalized not in self._patterns:
            # Create new canonical pattern
            pattern_id = self.get_pattern_id(normalized)
            self._patterns[normalized] = CanonicalPattern(
                id=pattern_id,
                pitch_classes=normalized,
                length=len(normalized),
                duration_buckets=tuple(duration_buckets) if duration_buckets else None,
                velocity_buckets=tuple(velocity_buckets) if velocity_buckets else None,
            )
            self._next_id += 1

        pattern = self._patterns[normalized]
        pattern.occurrence_count += 1

        # Record occurrence if we have location info
        if piece_id:
            occ = PatternOccurrence(
                pattern_id=pattern.id,
                piece_id=piece_id,
                track_id=track_id,
                onset_time=onset_time,
                t_offset=t_offset,
                octaves=tuple(octaves) if octaves else None,
                velocities=tuple(velocities) if velocities else None,
                durations=tuple(durations) if durations else None,
            )
            occ_idx = len(self._occurrences)
            self._occurrences.append(occ)
            self._occurrences_by_pattern[pattern.id].append(occ_idx)

        return pattern.id, t_offset

    def get_pattern(self, pattern_id: str) -> Optional[CanonicalPattern]:
        """Get a pattern by ID."""
        for p in self._patterns.values():
            if p.id == pattern_id:
                return p
        return None

    def get_pattern_by_pitches(self, normalized: Tuple[int, ...]) -> Optional[CanonicalPattern]:
        """Get a pattern by its normalized pitch classes."""
        return self._patterns.get(normalized)

    def get_occurrences(self, pattern_id: str) -> List[PatternOccurrence]:
        """Get all occurrences of a pattern."""
        indices = self._occurrences_by_pattern.get(pattern_id, [])
        return [self._occurrences[i] for i in indices]

    def get_occurrences_for_piece(self, piece_id: str) -> List[PatternOccurrence]:
        """Get all occurrences in a specific piece."""
        return [occ for occ in self._occurrences if occ.piece_id == piece_id]

    @property
    def num_patterns(self) -> int:
        return len(self._patterns)

    @property
    def num_occurrences(self) -> int:
        return len(self._occurrences)

    def get_stats(self) -> Dict:
        """Get statistics about the registry."""
        if not self._patterns:
            return {"patterns": 0, "occurrences": 0}

        lengths = [p.length for p in self._patterns.values()]
        counts = [p.occurrence_count for p in self._patterns.values()]

        return {
            "patterns": len(self._patterns),
            "occurrences": len(self._occurrences),
            "avg_pattern_length": np.mean(lengths),
            "max_pattern_length": max(lengths),
            "avg_occurrences_per_pattern": np.mean(counts),
            "max_occurrences": max(counts),
        }

    def to_dict(self) -> Dict:
        """Export registry to dictionary for checkpointing."""
        patterns = {}
        for normalized, pattern in self._patterns.items():
            patterns[pattern.id] = {
                "pitch_classes": list(pattern.pitch_classes),
                "length": pattern.length,
                "occurrence_count": pattern.occurrence_count,
                "duration_buckets": list(pattern.duration_buckets) if pattern.duration_buckets else None,
                "velocity_buckets": list(pattern.velocity_buckets) if pattern.velocity_buckets else None,
            }

        occurrences = []
        for occ in self._occurrences:
            occurrences.append({
                "pattern_id": occ.pattern_id,
                "piece_id": occ.piece_id,
                "track_id": occ.track_id,
                "onset_time": occ.onset_time,
                "t_offset": occ.t_offset,
                "octaves": list(occ.octaves) if occ.octaves else None,
                "velocities": list(occ.velocities) if occ.velocities else None,
                "durations": list(occ.durations) if occ.durations else None,
            })

        return {
            "patterns": patterns,
            "occurrences": occurrences,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CanonicalPatternRegistry':
        """Load registry from dictionary."""
        registry = cls()

        # Load patterns
        for pid, pdata in data.get("patterns", {}).items():
            normalized = tuple(pdata["pitch_classes"])
            registry._patterns[normalized] = CanonicalPattern(
                id=pid,
                pitch_classes=normalized,
                length=pdata["length"],
                occurrence_count=pdata.get("occurrence_count", 0),
                duration_buckets=tuple(pdata["duration_buckets"]) if pdata.get("duration_buckets") else None,
                velocity_buckets=tuple(pdata["velocity_buckets"]) if pdata.get("velocity_buckets") else None,
            )

        # Load occurrences
        for odata in data.get("occurrences", []):
            occ = PatternOccurrence(
                pattern_id=odata["pattern_id"],
                piece_id=odata["piece_id"],
                track_id=odata["track_id"],
                onset_time=odata["onset_time"],
                t_offset=odata["t_offset"],
                octaves=tuple(odata["octaves"]) if odata.get("octaves") else None,
                velocities=tuple(odata["velocities"]) if odata.get("velocities") else None,
                durations=tuple(odata["durations"]) if odata.get("durations") else None,
            )
            occ_idx = len(registry._occurrences)
            registry._occurrences.append(occ)
            registry._occurrences_by_pattern[occ.pattern_id].append(occ_idx)

        return registry


def deduplicate_patterns(
    patterns: List[Dict],
    verbose: bool = False
) -> Tuple[CanonicalPatternRegistry, Dict[str, Tuple[str, int]]]:
    """
    Deduplicate a list of patterns using T-normalization.

    Args:
        patterns: List of pattern dicts with 'pitch_classes', 'occurrences', etc.
        verbose: Print progress

    Returns:
        (registry, mapping) where:
        - registry: CanonicalPatternRegistry with deduplicated patterns
        - mapping: old_pattern_id -> (canonical_id, t_offset)
    """
    registry = CanonicalPatternRegistry()
    mapping = {}

    for i, pattern in enumerate(patterns):
        old_id = pattern.get('id', str(i))
        pitch_classes = pattern.get('pitch_classes', [])

        if not pitch_classes:
            continue

        # Add each occurrence
        for occ in pattern.get('occurrences', []):
            canonical_id, t_offset = registry.add_pattern(
                pitch_classes=pitch_classes,
                piece_id=occ.get('piece_id', ''),
                track_id=occ.get('track_id', 0),
                onset_time=occ.get('onset_time', 0),
                octaves=pattern.get('octaves'),
                velocities=pattern.get('velocities'),
                durations=pattern.get('durations'),
            )

            if old_id not in mapping:
                mapping[old_id] = (canonical_id, t_offset)

    if verbose:
        stats = registry.get_stats()
        print(f"Deduplicated {len(patterns)} patterns -> {stats['patterns']} canonical patterns")
        print(f"Total occurrences: {stats['occurrences']}")

    return registry, mapping


def convert_existing_checkpoint(
    checkpoint_path: str,
    output_path: str,
    verbose: bool = True
) -> Dict:
    """
    Convert an existing v24 checkpoint to use canonical patterns.

    Args:
        checkpoint_path: Path to existing checkpoint
        output_path: Where to save new checkpoint
        verbose: Print progress

    Returns:
        Statistics about the conversion
    """
    import json

    # Load existing checkpoint
    data = np.load(checkpoint_path, allow_pickle=True)
    rules_json = data['grammar_rules_json'][0]
    rules = json.loads(rules_json)

    if verbose:
        print(f"Loaded {len(rules)} patterns from {checkpoint_path}")

    # Build pattern list
    patterns = []
    for rid, rule in rules.items():
        patterns.append({
            'id': rid,
            'pitch_classes': rule.get('pitch_classes', []),
            'octaves': rule.get('octaves', []),
            'velocities': rule.get('velocities', []),
            'durations': rule.get('durations', []),
            'occurrences': rule.get('occurrences', []),
        })

    # Deduplicate
    registry, mapping = deduplicate_patterns(patterns, verbose=verbose)

    # Save new checkpoint
    registry_data = registry.to_dict()

    np.savez(
        output_path,
        version=np.array(['v25_canonical']),
        canonical_patterns_json=np.array([json.dumps(registry_data['patterns'])]),
        occurrences_json=np.array([json.dumps(registry_data['occurrences'])]),
        mapping_json=np.array([json.dumps({k: list(v) for k, v in mapping.items()})]),
        n_patterns=np.array([registry.num_patterns]),
        n_occurrences=np.array([registry.num_occurrences]),
    )

    if verbose:
        stats = registry.get_stats()
        print(f"\nSaved to {output_path}")
        print(f"Compression: {len(rules)} -> {stats['patterns']} patterns ({stats['patterns']/len(rules)*100:.1f}%)")

    return {
        'original_patterns': len(rules),
        'canonical_patterns': registry.num_patterns,
        'occurrences': registry.num_occurrences,
        'compression_ratio': len(rules) / registry.num_patterns if registry.num_patterns > 0 else 0,
    }


if __name__ == '__main__':
    # Test the normalization
    print("=== Testing T-Normalization ===\n")

    registry = CanonicalPatternRegistry()

    # Add C major chord in different transpositions
    test_patterns = [
        ([0, 4, 7], "C major"),
        ([3, 7, 10], "Eb major"),
        ([7, 11, 2], "G major"),
        ([5, 9, 0], "F major"),
    ]

    for pcs, name in test_patterns:
        pid, t_offset = registry.add_pattern(pcs, piece_id="test", track_id=0, onset_time=0)
        print(f"{name} {pcs} -> Pattern {pid}, T{t_offset}")

    print(f"\nTotal patterns: {registry.num_patterns} (should be 1)")
    print(f"Total occurrences: {registry.num_occurrences} (should be 4)")

    # Test with longer pattern
    print("\n=== Testing Longer Pattern ===")

    # Major scale in C
    c_scale = [0, 2, 4, 5, 7, 9, 11]
    g_scale = [7, 9, 11, 0, 2, 4, 6]  # G major (T7)

    pid1, t1 = registry.add_pattern(c_scale, piece_id="test2", track_id=0, onset_time=0)
    pid2, t2 = registry.add_pattern(g_scale, piece_id="test2", track_id=1, onset_time=0)

    print(f"C major scale {c_scale} -> {pid1}, T{t1}")
    print(f"G major scale {g_scale} -> {pid2}, T{t2}")

    print(f"\nTotal patterns now: {registry.num_patterns}")

    # Show stats
    print("\n=== Stats ===")
    for k, v in registry.get_stats().items():
        print(f"  {k}: {v}")
