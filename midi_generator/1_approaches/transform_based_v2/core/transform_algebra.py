"""
Transform algebra for musical transformations.

Defines the mathematical structure of transforms:
- Which transforms combine (composition)
- Which are inverses
- Which are involutory (self-inverse)
- Which commute

This enables algebraic simplification of composition paths.

Author: Agent - Transform Algebra
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class Transform:
    """
    A musical transform with a type and parameter.

    Types:
        - transpose_semitone: Pitch transposition (additive group)
        - time_shift: Temporal shift (additive group)
        - time_scale: Temporal scaling (multiplicative group)
        - velocity_scale: Velocity scaling (multiplicative group)
        - inversion: Pitch inversion around a center (involutory)
        - retrograde: Time reversal (involutory)
        - track_derive: Cross-track derivation (metadata in 'meta')
    """
    type: str
    param: float
    meta: Optional[str] = None  # For track_derive: "source_track→target_track"

    def __str__(self):
        if self.type == "section_track_derive" and self.meta:
            return f"SectionTrackDerive({self.meta})"
        if self.type == "section_derive" and self.meta:
            return f"SectionDerive({self.meta})"
        if self.type == "track_derive" and self.meta:
            return f"TrackDerive({self.meta})"
        return f"{self.type}({self.param})"

    def __repr__(self):
        if self.meta:
            return f"Transform('{self.type}', {self.param}, '{self.meta}')"
        return f"Transform('{self.type}', {self.param})"


# Involutory transforms (self-inverse: T ∘ T = identity)
INVOLUTORY_TRANSFORMS = {'inversion', 'retrograde'}


def combine(t1: Transform, t2: Transform) -> Optional[Transform]:
    """
    Combine two transforms if they're composable.

    Rules:
        - transpose_semitone: T(a) ∘ T(b) = T(a + b) [additive]
        - time_shift: shift(a) ∘ shift(b) = shift(a + b) [additive]
        - time_scale: scale(a) ∘ scale(b) = scale(a * b) [multiplicative]
        - velocity_scale: V(a) ∘ V(b) = V(a * b) [multiplicative]
        - Different types: None (not composable in simple way)

    Args:
        t1: First transform
        t2: Second transform (applied after t1)

    Returns:
        Combined transform if composable, else None
    """
    # Must be same type to combine
    if t1.type != t2.type:
        return None

    # Involutory transforms don't combine (they cancel)
    if t1.type in INVOLUTORY_TRANSFORMS:
        return None

    # Additive groups
    if t1.type in ['transpose_semitone', 'time_shift']:
        return Transform(t1.type, t1.param + t2.param)

    # Multiplicative groups
    if t1.type in ['time_scale', 'velocity_scale']:
        return Transform(t1.type, t1.param * t2.param)

    # Unknown type - can't combine
    return None


def get_inverse(t: Transform) -> Transform:
    """
    Get the inverse of a transform.

    Rules:
        - transpose_semitone: T(a)^-1 = T(-a)
        - time_shift: shift(a)^-1 = shift(-a)
        - time_scale: scale(a)^-1 = scale(1/a)
        - velocity_scale: V(a)^-1 = V(1/a)
        - inversion: I(x)^-1 = I(x) [self-inverse]
        - retrograde: R^-1 = R [self-inverse]

    Args:
        t: Transform to invert

    Returns:
        Inverse transform
    """
    # Involutory transforms are self-inverse
    if t.type in INVOLUTORY_TRANSFORMS:
        return t

    # Additive groups: negate
    if t.type in ['transpose_semitone', 'time_shift']:
        return Transform(t.type, -t.param)

    # Multiplicative groups: reciprocal
    if t.type in ['time_scale', 'velocity_scale']:
        if t.param == 0:
            raise ValueError(f"Cannot invert {t.type} with parameter 0")
        return Transform(t.type, 1.0 / t.param)

    # Unknown type - return identity as fallback
    return Transform(t.type, 0 if 'shift' in t.type or 'transpose' in t.type else 1.0)


def is_identity(t: Transform, octave_equivalence: bool = False) -> bool:
    """
    Check if a transform is the identity (does nothing).

    Rules:
        - transpose_semitone(0): identity
        - transpose_semitone(±12): identity if octave_equivalence=True
          (DEFAULT: False for big band - octaves matter!)
        - time_shift(0): identity
        - time_scale(1.0): identity
        - velocity_scale(1.0): identity

    Args:
        t: Transform to check
        octave_equivalence: Treat octave transpositions as identity
                           (Default False - octaves are meaningful in arranging)

    Returns:
        True if transform is identity
    """
    # Track/section derivations are NEVER identity (always meaningful)
    if t.type in ['track_derive', 'section_track_derive', 'section_derive']:
        return False

    # Additive groups: check for zero
    if t.type == 'transpose_semitone':
        if abs(t.param) < 1e-9:
            return True
        # Octave equivalence: ±12 semitones
        # NOTE: Default is False because in big band arranging,
        # octave transpositions are musically meaningful (bass doubling, etc.)
        if octave_equivalence and abs(abs(t.param) - 12) < 1e-9:
            return True
        return False

    if t.type == 'time_shift':
        return abs(t.param) < 1e-9

    # Multiplicative groups: check for one
    if t.type in ['time_scale', 'velocity_scale']:
        return abs(t.param - 1.0) < 1e-9

    # Involutory transforms are never identity by themselves
    return False


def are_inverses(t1: Transform, t2: Transform) -> bool:
    """
    Check if two transforms are inverses of each other.

    Args:
        t1: First transform
        t2: Second transform

    Returns:
        True if t1 and t2 are inverses
    """
    # Same type required
    if t1.type != t2.type:
        return False

    # Involutory transforms: must be identical
    if t1.type in INVOLUTORY_TRANSFORMS:
        return abs(t1.param - t2.param) < 1e-9

    # Additive groups: check if parameters sum to zero
    if t1.type in ['transpose_semitone', 'time_shift']:
        return abs(t1.param + t2.param) < 1e-9

    # Multiplicative groups: check if parameters multiply to one
    if t1.type in ['time_scale', 'velocity_scale']:
        return abs(t1.param * t2.param - 1.0) < 1e-9

    return False


def simplify_pair(t1: Transform, t2: Transform) -> Optional[Transform]:
    """
    Simplify a pair of adjacent transforms.

    Returns:
        - Combined transform if they compose
        - None if they cancel (inverses)
        - t1 if they don't interact

    This is used by the path canonicalizer for local simplification.
    """
    # Check if they cancel
    if are_inverses(t1, t2):
        return None  # Signal: remove both

    # Try to combine
    combined = combine(t1, t2)
    if combined is not None:
        # Check if result is identity
        if is_identity(combined):
            return None  # Signal: remove both (they simplify to identity)
        return combined

    # Can't simplify - return t1 (keep both)
    return t1


def parse_transform_string(s: str) -> Transform:
    """
    Parse a transform string like "transpose_semitone(12)", "TrackDerive(trumpet→trombone)",
    or "SectionTrackDerive(verse.trumpet→chorus.trombone)" into a Transform object.

    Args:
        s: String representation

    Returns:
        Transform object
    """
    # Extract type and parameter
    if '(' not in s or ')' not in s:
        raise ValueError(f"Invalid transform string: {s}")

    type_name = s[:s.index('(')].strip()
    param_str = s[s.index('(')+1:s.index(')')].strip()

    # Special case for SectionTrackDerive - parameter is "section.track→section.track"
    if type_name == "SectionTrackDerive":
        return Transform("section_track_derive", 0.0, meta=param_str)

    # Special case for SectionDerive - parameter is "section→section"
    if type_name == "SectionDerive":
        return Transform("section_derive", 0.0, meta=param_str)

    # Special case for TrackDerive - parameter is a string "source→target"
    if type_name == "TrackDerive":
        # Store track relationship in metadata
        return Transform("track_derive", 0.0, meta=param_str)

    try:
        param = float(param_str)
    except ValueError:
        raise ValueError(f"Invalid parameter in transform string: {s}")

    return Transform(type_name, param)
