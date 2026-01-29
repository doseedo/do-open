"""
Path canonicalizer for musical transformation compositions.

Simplifies composition paths using group-theoretic rules to eliminate
degenerate patterns like T(12) ∘ T(-12) ∘ T(12) ∘ T(-12).

Key operations:
- Combine adjacent same-type transforms (T(7) ∘ T(5) = T(12))
- Cancel inverse pairs (T(12) ∘ T(-12) = identity)
- Remove identity transforms
- Filter trivial paths (length ≤ 1)

Also stores BOTH canonical and raw temporal paths for dual analysis:
- Canonical: For MDL basis discovery (what IS the relationship)
- Temporal: For harmonic rhythm/narrative analysis (how did we GET there)

Author: Agent - Path Canonicalizer
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from .transform_algebra import Transform, combine, are_inverses, is_identity


@dataclass
class PathRecord:
    """
    Records both canonical and temporal representations of a path.

    Attributes:
        canonical: Simplified form (for MDL discovery)
        raw_paths: All temporal variants that reduce to this canonical form
        frequency: Total count across all raw variants
    """
    canonical: Tuple[Transform, ...]
    raw_paths: List[Dict[str, Any]]  # [{path: [...], count: int, context: ...}]
    frequency: int


def canonicalize_path(path: List[Transform]) -> List[Transform]:
    """
    Simplify a composition path to canonical form.

    Algorithm:
        1. Repeatedly combine adjacent same-type transforms
        2. Cancel inverse pairs
        3. Remove identity transforms
        4. Repeat until no changes

    Args:
        path: List of transforms in composition order

    Returns:
        Simplified list of transforms

    Examples:
        >>> canonicalize_path([T(12), T(-12)])
        []  # Identity

        >>> canonicalize_path([T(7), T(5)])
        [T(12)]

        >>> canonicalize_path([T(7), T(5), T(-12)])
        []  # Identity

        >>> canonicalize_path([I(60), I(60)])
        []  # Involution
    """
    if not path:
        return []

    # Make a copy to avoid mutating input
    current = list(path)

    # Iterate until no more simplifications possible
    max_iterations = len(current) * 2  # Safety bound
    for _ in range(max_iterations):
        simplified = _simplify_once(current)

        # No change means we're done
        if len(simplified) == len(current) and all(
            s.type == c.type and abs(s.param - c.param) < 1e-9
            for s, c in zip(simplified, current)
        ):
            break

        current = simplified

    return current


def _simplify_once(path: List[Transform]) -> List[Transform]:
    """
    Perform one pass of simplification on a path.

    Args:
        path: List of transforms

    Returns:
        Path after one simplification pass
    """
    if len(path) <= 1:
        return [t for t in path if not is_identity(t)]

    result = []
    i = 0

    while i < len(path):
        # Check if current transform is identity
        if is_identity(path[i]):
            i += 1
            continue

        # Try to combine with next transform
        if i + 1 < len(path):
            t1, t2 = path[i], path[i + 1]

            # Check if they cancel (inverses)
            if are_inverses(t1, t2):
                i += 2  # Skip both
                continue

            # Try to combine same-type transforms
            combined = combine(t1, t2)
            if combined is not None:
                # Check if result is identity
                if is_identity(combined):
                    i += 2  # Skip both (they cancel to identity)
                    continue
                else:
                    result.append(combined)
                    i += 2
                    continue

        # Can't simplify - keep the transform
        result.append(path[i])
        i += 1

    return result


def is_trivial(path: List[Transform]) -> bool:
    """
    Check if a path is trivial (length ≤ 1 after simplification).

    Args:
        path: Path to check (should already be canonical)

    Returns:
        True if path has 0 or 1 transforms
    """
    return len(path) <= 1


def filter_paths(
    paths: List[List[Transform]],
    min_len: int = 2,
    max_len: int = 4
) -> List[List[Transform]]:
    """
    Filter paths to remove trivial and overly long compositions.

    Args:
        paths: List of canonical paths
        min_len: Minimum path length (inclusive)
        max_len: Maximum path length (inclusive)

    Returns:
        Filtered list of paths
    """
    return [
        p for p in paths
        if min_len <= len(p) <= max_len
    ]


def count_canonical_paths(
    raw_paths: List[List[Transform]],
    min_len: int = 2,
    max_len: int = 4,
    store_temporal: bool = True
) -> Tuple[Dict[Tuple[Transform, ...], int], Dict[Tuple[Transform, ...], List[List[Transform]]]]:
    """
    Count frequencies of canonical paths and optionally store raw variants.

    Args:
        raw_paths: List of raw paths from graph traversal
        min_len: Minimum canonical path length
        max_len: Maximum canonical path length
        store_temporal: Whether to store raw path variants

    Returns:
        Tuple of (canonical_counts, temporal_variants)
        - canonical_counts: {canonical_path: frequency}
        - temporal_variants: {canonical_path: [list of raw paths]}
    """
    canonical_counts = {}
    temporal_variants = {} if store_temporal else None

    for raw_path in raw_paths:
        # Canonicalize
        canonical = canonicalize_path(raw_path)

        # Filter by length
        if len(canonical) < min_len or len(canonical) > max_len:
            continue

        # Convert to hashable key
        key = tuple(canonical)

        # Count
        canonical_counts[key] = canonical_counts.get(key, 0) + 1

        # Store temporal variant
        if store_temporal:
            if key not in temporal_variants:
                temporal_variants[key] = []
            temporal_variants[key].append(raw_path)

    return canonical_counts, temporal_variants


def store_identity_processes(
    raw_paths: List[List[Transform]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Store paths that canonicalize to identity - these represent
    "journeys that return home" (tension-release, oscillations).

    Args:
        raw_paths: List of raw paths

    Returns:
        Dictionary categorizing identity processes by pattern type
    """
    identity_processes = {
        'oscillation': [],  # A → B → A → B → ...
        'cycle': [],        # A → B → C → A
        'simple': [],       # A → B → A
        'other': []
    }

    for raw_path in raw_paths:
        canonical = canonicalize_path(raw_path)

        # Only interested in paths that reduce to identity
        if len(canonical) > 0:
            continue

        # Classify the pattern
        pattern_type = _classify_identity_pattern(raw_path)

        identity_processes[pattern_type].append({
            'path': raw_path,
            'length': len(raw_path)
        })

    return identity_processes


def _classify_identity_pattern(path: List[Transform]) -> str:
    """
    Classify what kind of identity process this is.

    Args:
        path: Raw path that canonicalizes to identity

    Returns:
        Pattern type: 'oscillation', 'cycle', 'simple', or 'other'
    """
    if len(path) == 2:
        return 'simple'

    # Check for oscillation (alternating between two transforms)
    if len(path) >= 4:
        unique = list(set((t.type, t.param) for t in path))
        if len(unique) == 2:
            # Check if alternating
            alternates = all(
                (path[i].type, path[i].param) != (path[i+1].type, path[i+1].param)
                for i in range(len(path) - 1)
            )
            if alternates:
                return 'oscillation'

    # Check for cycle (more than 2 unique transforms)
    unique = list(set((t.type, t.param) for t in path))
    if len(unique) >= 3:
        return 'cycle'

    return 'other'


def path_to_string(path: List[Transform]) -> str:
    """
    Convert a path to a readable string representation.

    Args:
        path: List of transforms

    Returns:
        String like "T(7) ∘ shift(16) ∘ V(0.7)"
    """
    if not path:
        return "identity"

    return " ∘ ".join(str(t) for t in path)
