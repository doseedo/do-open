"""
Step 1: Decoder
===============

Takes transform-relative encoding and reconstructs the original note sequence.

Flow:
1. Walk through tokens in order
2. For each INTRO: look up the canonical pattern's pitch classes
3. For each REPEAT: resolve the reference, copy those pitches
4. For each TRANSFORM: resolve the reference, apply D24 transform
5. Reconstruct full notes by combining pitch_class + octave + rhythm + duration + velocity

Success: Function that takes encoding -> returns list of notes with pitch, onset, duration, velocity.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import json


@dataclass
class DecodedNote:
    """A fully decoded MIDI note."""
    pitch: int          # MIDI pitch (0-127)
    pitch_class: int    # Pitch class (0-11)
    octave: int         # Octave number
    onset: int          # Start time in ticks
    duration: int       # Duration in ticks
    velocity: int       # MIDI velocity (0-127)
    track: int = 0      # Track index
    pattern_idx: int = -1  # Which pattern this came from

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return (pitch, onset, duration, velocity) for comparison."""
        return (self.pitch, self.onset, self.duration, self.velocity)


@dataclass
class DecodedPattern:
    """A decoded pattern instance."""
    pattern_idx: int
    token_type: str
    transform_id: int
    notes: List[DecodedNote]
    start_time: int


@dataclass
class DecodingResult:
    """Complete result of decoding."""
    notes: List[DecodedNote]
    patterns: List[DecodedPattern]
    n_tracks: int
    stats: Dict[str, Any] = field(default_factory=dict)


def apply_d24_transform(pitch_classes: np.ndarray, transform_id: int) -> np.ndarray:
    """
    Apply a D24 dihedral group transform to pitch classes.

    Args:
        pitch_classes: Array of pitch classes (0-11)
        transform_id: 0-23 (0-11 = transposition, 12-23 = inversion)

    Returns:
        Transformed pitch classes
    """
    pitch_classes = np.asarray(pitch_classes)
    if transform_id < 12:
        # Transposition: T_n(p) = (p + n) mod 12
        return (pitch_classes + transform_id) % 12
    else:
        # Inversion: I_n(p) = (n - p) mod 12
        n = transform_id - 12
        return (n - pitch_classes) % 12


class ValidationDecoder:
    """
    Decoder for transform-relative encoding with full note reconstruction.

    Unlike the basic decoder, this one:
    - Tracks pattern boundaries
    - Preserves full note data (octave, duration, velocity) when available
    - Provides detailed reconstruction statistics
    """

    def __init__(
        self,
        ticks_per_beat: int = 480,
        default_velocity: int = 100,
        default_duration_ticks: int = 120,  # 16th note at 480 tpb
        base_octave: int = 4,
    ):
        self.ticks_per_beat = ticks_per_beat
        self.default_velocity = default_velocity
        self.default_duration = default_duration_ticks
        self.base_octave = base_octave

    def decode_from_checkpoint(
        self,
        checkpoint_path: str,
        canonical_note_data: Optional[Dict[int, np.ndarray]] = None,
    ) -> DecodingResult:
        """
        Decode from a saved checkpoint file.

        Args:
            checkpoint_path: Path to .npz checkpoint
            canonical_note_data: Optional mapping of pattern_id to full note data
                                 Each entry: (n_notes, 4) with [pitch, onset, duration, velocity]

        Returns:
            DecodingResult with all decoded notes and statistics
        """
        # Load checkpoint
        ckpt = np.load(checkpoint_path, allow_pickle=True)

        # Parse canonical patterns
        canonical_json = str(ckpt['canonical_patterns_json'][0])
        canonicals = json.loads(canonical_json)

        # Parse encoding tokens
        encoding_json = str(ckpt['encoding_tokens_json'][0])
        tokens_by_track = json.loads(encoding_json)

        return self.decode(
            canonicals=canonicals,
            tokens_by_track=tokens_by_track,
            canonical_note_data=canonical_note_data,
        )

    def decode(
        self,
        canonicals: List[Dict],
        tokens_by_track: List[List[Dict]],
        canonical_note_data: Optional[Dict[int, np.ndarray]] = None,
    ) -> DecodingResult:
        """
        Decode transform-relative encoding to notes.

        Args:
            canonicals: List of canonical pattern dicts with 'pitch_classes'
            tokens_by_track: List of token lists, one per track
            canonical_note_data: Optional full note data per canonical

        Returns:
            DecodingResult
        """
        all_notes = []
        all_patterns = []

        # Process each track
        for track_idx, track_tokens in enumerate(tokens_by_track):
            track_notes, track_patterns = self._decode_track(
                track_tokens=track_tokens,
                canonicals=canonicals,
                canonical_note_data=canonical_note_data,
                track_idx=track_idx,
            )
            all_notes.extend(track_notes)
            all_patterns.extend(track_patterns)

        # Compute statistics
        stats = {
            'n_notes': len(all_notes),
            'n_patterns': len(all_patterns),
            'n_tracks': len(tokens_by_track),
            'n_canonicals': len(canonicals),
            'token_counts': self._count_token_types(tokens_by_track),
        }

        return DecodingResult(
            notes=all_notes,
            patterns=all_patterns,
            n_tracks=len(tokens_by_track),
            stats=stats,
        )

    def _decode_track(
        self,
        track_tokens: List[Dict],
        canonicals: List[Dict],
        canonical_note_data: Optional[Dict[int, np.ndarray]],
        track_idx: int,
    ) -> Tuple[List[DecodedNote], List[DecodedPattern]]:
        """Decode a single track."""
        notes = []
        patterns = []
        current_time = 0

        # Map token indices to their emitted pitch classes (for backreferences)
        emitted: Dict[int, Tuple[np.ndarray, int, int]] = {}  # idx -> (pitch_classes, start_time, pattern_idx)

        # Map canonical IDs to their first introduction token index
        canonical_first_token: Dict[int, int] = {}

        for token_idx, token in enumerate(track_tokens):
            token_type = token.get('type', '')
            pattern_idx = token.get('pattern_idx', -1)
            transform_id = token.get('transform_id', 0)

            if token_type == 'TRACK_BOUNDARY':
                continue
            if token_type == 'SECTION_BOUNDARY':
                current_time += self.ticks_per_beat * 4
                continue

            pattern_notes = []
            pitch_classes = None
            source_pattern_idx = pattern_idx

            if token_type == 'INTRO':
                # Get canonical pattern
                if pattern_idx < 0 or pattern_idx >= len(canonicals):
                    continue

                canonical = canonicals[pattern_idx]
                pitch_classes = np.array(canonical.get('pitch_classes', []))

                # Track first introduction of this canonical
                if pattern_idx not in canonical_first_token:
                    canonical_first_token[pattern_idx] = token_idx

                # Get full note data if available
                full_data = None
                if canonical_note_data and pattern_idx in canonical_note_data:
                    full_data = canonical_note_data[pattern_idx]

                pattern_notes = self._pitch_classes_to_notes(
                    pitch_classes=pitch_classes,
                    start_time=current_time,
                    full_note_data=full_data,
                    track_idx=track_idx,
                    pattern_idx=pattern_idx,
                )

                # Store for backreference
                emitted[token_idx] = (pitch_classes.copy(), current_time, pattern_idx)

            elif token_type == 'REPEAT':
                # Resolve backreference
                ref_idx = pattern_idx

                # The pattern_idx might refer to a canonical or a token index
                # Try to resolve it
                resolved = self._resolve_reference(
                    ref_idx, emitted, canonical_first_token, canonicals
                )
                if resolved is None:
                    continue

                pitch_classes, source_pattern_idx = resolved

                pattern_notes = self._pitch_classes_to_notes(
                    pitch_classes=pitch_classes,
                    start_time=current_time,
                    full_note_data=None,
                    track_idx=track_idx,
                    pattern_idx=source_pattern_idx,
                )

                emitted[token_idx] = (pitch_classes.copy(), current_time, source_pattern_idx)

            elif token_type == 'TRANSFORM':
                # Resolve backreference and apply transform
                ref_idx = pattern_idx

                resolved = self._resolve_reference(
                    ref_idx, emitted, canonical_first_token, canonicals
                )
                if resolved is None:
                    continue

                source_pc, source_pattern_idx = resolved
                pitch_classes = apply_d24_transform(source_pc, transform_id)

                pattern_notes = self._pitch_classes_to_notes(
                    pitch_classes=pitch_classes,
                    start_time=current_time,
                    full_note_data=None,
                    track_idx=track_idx,
                    pattern_idx=source_pattern_idx,
                )

                emitted[token_idx] = (pitch_classes.copy(), current_time, source_pattern_idx)

            # Record pattern and advance time
            if pattern_notes:
                notes.extend(pattern_notes)

                pattern = DecodedPattern(
                    pattern_idx=source_pattern_idx,
                    token_type=token_type,
                    transform_id=transform_id,
                    notes=pattern_notes,
                    start_time=current_time,
                )
                patterns.append(pattern)

                # Advance time by pattern duration
                max_end = max(n.onset + n.duration for n in pattern_notes)
                current_time = max_end

        return notes, patterns

    def _resolve_reference(
        self,
        ref_idx: int,
        emitted: Dict[int, Tuple[np.ndarray, int, int]],
        canonical_first_token: Dict[int, int],
        canonicals: List[Dict],
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Resolve a backreference to get pitch classes and source pattern index.

        The reference might be:
        1. A token index (backreference to earlier in sequence)
        2. A canonical pattern index

        Returns (pitch_classes, canonical_idx) or None.
        """
        # First try as token index
        if ref_idx in emitted:
            pc, _, pattern_idx = emitted[ref_idx]
            return pc, pattern_idx

        # Try as canonical index
        if ref_idx < len(canonicals):
            # Find the token that introduced this canonical
            if ref_idx in canonical_first_token:
                first_token = canonical_first_token[ref_idx]
                if first_token in emitted:
                    pc, _, _ = emitted[first_token]
                    return pc, ref_idx

            # Fall back to canonical directly
            canonical = canonicals[ref_idx]
            pc = np.array(canonical.get('pitch_classes', []))
            return pc, ref_idx

        return None

    def _pitch_classes_to_notes(
        self,
        pitch_classes: np.ndarray,
        start_time: int,
        full_note_data: Optional[np.ndarray],
        track_idx: int,
        pattern_idx: int,
    ) -> List[DecodedNote]:
        """Convert pitch classes to full notes."""
        notes = []

        if full_note_data is not None and len(full_note_data) == len(pitch_classes):
            # Use provided full note data
            for i, (pitch, onset_offset, duration, velocity) in enumerate(full_note_data):
                notes.append(DecodedNote(
                    pitch=int(pitch),
                    pitch_class=int(pitch) % 12,
                    octave=int(pitch) // 12 - 1,
                    onset=start_time + int(onset_offset),
                    duration=int(duration),
                    velocity=int(velocity),
                    track=track_idx,
                    pattern_idx=pattern_idx,
                ))
        else:
            # Reconstruct from pitch classes only
            # Use default octave, duration, velocity
            note_spacing = self.default_duration

            for i, pc in enumerate(pitch_classes):
                midi_pitch = int(pc) + (self.base_octave + 1) * 12
                notes.append(DecodedNote(
                    pitch=midi_pitch,
                    pitch_class=int(pc),
                    octave=self.base_octave,
                    onset=start_time + i * note_spacing,
                    duration=self.default_duration,
                    velocity=self.default_velocity,
                    track=track_idx,
                    pattern_idx=pattern_idx,
                ))

        return notes

    def _count_token_types(self, tokens_by_track: List[List[Dict]]) -> Dict[str, int]:
        """Count token types across all tracks."""
        counts = {}
        for track in tokens_by_track:
            for token in track:
                ttype = token.get('type', 'UNKNOWN')
                counts[ttype] = counts.get(ttype, 0) + 1
        return counts


def decode_encoding(
    checkpoint_path: str,
    canonical_note_data: Optional[Dict[int, np.ndarray]] = None,
) -> DecodingResult:
    """
    Convenience function to decode a checkpoint.

    Args:
        checkpoint_path: Path to .npz checkpoint file
        canonical_note_data: Optional full note data for canonicals

    Returns:
        DecodingResult with decoded notes and statistics
    """
    decoder = ValidationDecoder()
    return decoder.decode_from_checkpoint(checkpoint_path, canonical_note_data)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python decoder.py <checkpoint_path>")
        sys.exit(1)

    result = decode_encoding(sys.argv[1])

    print(f"=== Decoding Result ===")
    print(f"Notes decoded: {result.stats['n_notes']}")
    print(f"Patterns decoded: {result.stats['n_patterns']}")
    print(f"Tracks: {result.stats['n_tracks']}")
    print(f"Canonicals: {result.stats['n_canonicals']}")
    print(f"Token counts: {result.stats['token_counts']}")

    if result.notes:
        print(f"\nFirst 10 notes:")
        for note in result.notes[:10]:
            print(f"  pitch={note.pitch}, onset={note.onset}, dur={note.duration}, vel={note.velocity}")
