"""
Transform-Relative Encoding Data Structures

Instead of opaque rule IDs like [Rule_17, Rule_42, Rule_17, Rule_89],
we encode: [(INTRO, Motif_A), (TRANSFORM, ref=0, T₇), (REPEAT, ref=0), ...]

This makes transforms visible and enables:
- "MIDI Gene Editor" interface where editing one canonical updates all derivations
- Explicit transform relationships in the encoding
- Better compression through transform-based deduplication
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple
import numpy as np


class TokenType(Enum):
    """Types of tokens in transform-relative encoding."""

    # First occurrence - introduces a new canonical pattern
    INTRO = auto()

    # Reference to earlier pattern via D24 transform
    # If transform_id=0 (identity), this is effectively a REPEAT
    TRANSFORM = auto()

    # Exact repeat of earlier pattern (special case of TRANSFORM with identity)
    REPEAT = auto()

    # Structural markers
    SECTION_BOUNDARY = auto()
    TRACK_BOUNDARY = auto()

    # Cross-track reference (pattern in another track)
    CROSS_TRACK_REF = auto()


@dataclass
class EncodingToken:
    """A single token in the transform-relative encoding."""

    token_type: TokenType

    # For INTRO: index into canonical_patterns list
    # For TRANSFORM/REPEAT: index of the token being referenced (backreference)
    pattern_idx: int = -1

    # For TRANSFORM: which D24 transform (0-23)
    # 0-11: transpositions T₀-T₁₁
    # 12-23: inversions I₀-I₁₁
    transform_id: int = 0

    # For CROSS_TRACK_REF: which track the reference points to
    source_track: int = -1

    # Optional metadata
    position_ticks: int = 0  # Position in original MIDI
    duration_ticks: int = 0  # Duration of this segment

    def __repr__(self):
        if self.token_type == TokenType.INTRO:
            return f"INTRO(pattern={self.pattern_idx})"
        elif self.token_type == TokenType.TRANSFORM:
            t = self.transform_id
            if t < 12:
                transform_str = f"T{t}"
            else:
                transform_str = f"I{t-12}"
            return f"TRANSFORM(ref={self.pattern_idx}, {transform_str})"
        elif self.token_type == TokenType.REPEAT:
            return f"REPEAT(ref={self.pattern_idx})"
        elif self.token_type == TokenType.CROSS_TRACK_REF:
            return f"CROSS_TRACK(track={self.source_track}, ref={self.pattern_idx})"
        elif self.token_type == TokenType.SECTION_BOUNDARY:
            return "SECTION"
        elif self.token_type == TokenType.TRACK_BOUNDARY:
            return "TRACK"
        return f"{self.token_type.name}"


@dataclass
class CanonicalPattern:
    """A canonical pattern that can be referenced by transforms."""

    # Unique ID for this canonical
    pattern_id: int

    # The actual pitch-class sequence (0-11)
    pitch_classes: np.ndarray

    # Original grammar rule ID (for debugging/tracing)
    original_rule_id: int = -1

    # How many times this canonical appears (directly or via transform)
    usage_count: int = 0

    # Full note data for reconstruction (pitch, duration, velocity)
    # Shape: (n_notes, 3) or None if not available
    full_note_data: Optional[np.ndarray] = None

    # Rhythm pattern (inter-onset intervals)
    rhythm_pattern: Optional[np.ndarray] = None

    def __repr__(self):
        pc_str = ','.join(str(p) for p in self.pitch_classes[:8])
        if len(self.pitch_classes) > 8:
            pc_str += f"...({len(self.pitch_classes)} notes)"
        return f"Canon_{self.pattern_id}[{pc_str}]"

    def __len__(self):
        return len(self.pitch_classes)


@dataclass
class TransformRelativeEncoding:
    """
    Complete transform-relative encoding of a piece.

    This is the "MIDI Gene" representation that makes transform
    relationships explicit and editable.
    """

    # Metadata
    piece_id: str = ""
    n_tracks: int = 1

    # The canonical patterns (the "genes")
    canonical_patterns: List[CanonicalPattern] = field(default_factory=list)

    # The encoding sequence (how patterns are assembled)
    # One list per track
    tokens: List[List[EncodingToken]] = field(default_factory=list)

    # D24 transform table reference (for decoding)
    # Maps (pattern_idx, transform_id) -> transformed pitch classes
    # Stored separately to avoid redundancy

    # Cross-track relationships
    # List of (track_a, token_idx_a, track_b, token_idx_b, relation_type)
    cross_track_relations: List[Tuple[int, int, int, int, str]] = field(default_factory=list)

    # Statistics
    @property
    def n_canonicals(self) -> int:
        return len(self.canonical_patterns)

    @property
    def n_tokens(self) -> int:
        return sum(len(t) for t in self.tokens)

    @property
    def n_intros(self) -> int:
        return sum(1 for track in self.tokens for t in track if t.token_type == TokenType.INTRO)

    @property
    def n_transforms(self) -> int:
        return sum(1 for track in self.tokens for t in track if t.token_type == TokenType.TRANSFORM)

    @property
    def n_repeats(self) -> int:
        return sum(1 for track in self.tokens for t in track if t.token_type == TokenType.REPEAT)

    @property
    def compression_ratio(self) -> float:
        """Ratio of total pattern occurrences to canonical count."""
        if self.n_canonicals == 0:
            return 1.0
        total_refs = self.n_intros + self.n_transforms + self.n_repeats
        return total_refs / self.n_canonicals

    def summary(self) -> str:
        """Human-readable summary of the encoding."""
        lines = [
            f"Transform-Relative Encoding: {self.piece_id}",
            f"  Tracks: {self.n_tracks}",
            f"  Canonical patterns: {self.n_canonicals}",
            f"  Total tokens: {self.n_tokens}",
            f"    - INTRO: {self.n_intros}",
            f"    - TRANSFORM: {self.n_transforms}",
            f"    - REPEAT: {self.n_repeats}",
            f"  Cross-track relations: {len(self.cross_track_relations)}",
            f"  Compression ratio: {self.compression_ratio:.2f}x",
        ]
        return '\n'.join(lines)

    def get_token_sequence_repr(self, track_idx: int = 0, max_tokens: int = 20) -> str:
        """Get string representation of token sequence for a track."""
        if track_idx >= len(self.tokens):
            return "Track not found"

        tokens = self.tokens[track_idx][:max_tokens]
        parts = [str(t) for t in tokens]

        if len(self.tokens[track_idx]) > max_tokens:
            parts.append(f"... ({len(self.tokens[track_idx]) - max_tokens} more)")

        return ' → '.join(parts)


def apply_d24_transform(pitch_classes: np.ndarray, transform_id: int) -> np.ndarray:
    """
    Apply a D24 dihedral group transform to pitch classes.

    Args:
        pitch_classes: Array of pitch classes (0-11)
        transform_id: 0-23 (0-11 = transposition, 12-23 = inversion)

    Returns:
        Transformed pitch classes
    """
    if transform_id < 12:
        # Transposition: T_n(p) = (p + n) mod 12
        return (pitch_classes + transform_id) % 12
    else:
        # Inversion: I_n(p) = (n - p) mod 12
        n = transform_id - 12
        return (n - pitch_classes) % 12


def find_transform(source: np.ndarray, target: np.ndarray) -> Optional[int]:
    """
    Find which D24 transform maps source to target.

    Args:
        source: Source pitch class sequence
        target: Target pitch class sequence

    Returns:
        Transform ID (0-23) or None if no transform matches
    """
    if len(source) != len(target):
        return None

    if len(source) == 0:
        return 0  # Identity for empty

    # Try all 24 transforms
    for t in range(24):
        transformed = apply_d24_transform(source, t)
        if np.array_equal(transformed, target):
            return t

    return None
