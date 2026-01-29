"""
Transform-Relative Decoder

Converts transform-relative encoding back to MIDI.

Flow:
1. For each token in sequence:
   - INTRO: emit canonical pattern directly
   - TRANSFORM: apply D24 transform to referenced pattern
   - REPEAT: copy referenced pattern
2. Reconstruct MIDI with proper timing
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .transform_relative import (
    TokenType,
    EncodingToken,
    TransformRelativeEncoding,
    CanonicalPattern,
    apply_d24_transform,
)


@dataclass
class DecodedNote:
    """A decoded MIDI note."""
    pitch: int
    start_time: int  # In ticks
    duration: int
    velocity: int = 100
    track: int = 0


@dataclass
class DecodedPattern:
    """A decoded pattern with full note data."""
    notes: List[DecodedNote]
    start_time: int
    duration: int


class TransformRelativeDecoder:
    """
    Decodes transform-relative encoding to MIDI.
    """

    def __init__(
        self,
        ticks_per_beat: int = 480,
        default_velocity: int = 100,
        default_duration: int = 480,  # Quarter note
        base_octave: int = 4,  # Middle C octave
    ):
        self.ticks_per_beat = ticks_per_beat
        self.default_velocity = default_velocity
        self.default_duration = default_duration
        self.base_octave = base_octave

    def decode(
        self,
        encoding: TransformRelativeEncoding,
        canonical_note_data: Optional[Dict[int, np.ndarray]] = None,
    ) -> List[List[DecodedNote]]:
        """
        Decode transform-relative encoding to note lists.

        Args:
            encoding: The transform-relative encoding
            canonical_note_data: Optional dict mapping canonical_id to
                                 full note data (pitch, duration, velocity)

        Returns:
            List of note lists, one per track
        """
        all_tracks = []

        for track_idx, track_tokens in enumerate(encoding.tokens):
            track_notes = self._decode_track(
                track_tokens,
                encoding.canonical_patterns,
                canonical_note_data,
                track_idx,
            )
            all_tracks.append(track_notes)

        return all_tracks

    def _decode_track(
        self,
        tokens: List[EncodingToken],
        canonicals: List[CanonicalPattern],
        canonical_note_data: Optional[Dict[int, np.ndarray]],
        track_idx: int,
    ) -> List[DecodedNote]:
        """Decode a single track."""
        notes = []
        current_time = 0

        # Keep track of emitted patterns for backreferences
        emitted_patterns: Dict[int, Tuple[np.ndarray, int]] = {}  # token_idx -> (pitch_classes, start_time)

        for token_idx, token in enumerate(tokens):
            if token.token_type == TokenType.TRACK_BOUNDARY:
                continue
            if token.token_type == TokenType.SECTION_BOUNDARY:
                current_time += self.ticks_per_beat * 4  # Gap between sections
                continue

            pattern_notes = []

            if token.token_type == TokenType.INTRO:
                # Get canonical pattern
                if token.pattern_idx >= len(canonicals):
                    continue
                canonical = canonicals[token.pattern_idx]
                pitch_classes = canonical.pitch_classes

                # Store for backreference
                emitted_patterns[token_idx] = (pitch_classes.copy(), current_time)

                # Convert to notes
                pattern_notes = self._pitch_classes_to_notes(
                    pitch_classes,
                    current_time,
                    canonical_note_data.get(token.pattern_idx) if canonical_note_data else None,
                    track_idx,
                )

            elif token.token_type == TokenType.REPEAT:
                # Get referenced pattern
                ref_idx = token.pattern_idx
                if ref_idx not in emitted_patterns:
                    # Find by canonical ID instead
                    for idx, (pc, _) in emitted_patterns.items():
                        if idx == ref_idx:
                            break
                    else:
                        continue

                pitch_classes, _ = emitted_patterns[ref_idx]
                emitted_patterns[token_idx] = (pitch_classes.copy(), current_time)

                pattern_notes = self._pitch_classes_to_notes(
                    pitch_classes,
                    current_time,
                    None,
                    track_idx,
                )

            elif token.token_type == TokenType.TRANSFORM:
                # Get referenced pattern and apply transform
                ref_idx = token.pattern_idx
                if ref_idx not in emitted_patterns:
                    continue

                source_pc, _ = emitted_patterns[ref_idx]
                transformed_pc = apply_d24_transform(source_pc, token.transform_id)
                emitted_patterns[token_idx] = (transformed_pc.copy(), current_time)

                pattern_notes = self._pitch_classes_to_notes(
                    transformed_pc,
                    current_time,
                    None,
                    track_idx,
                )

            elif token.token_type == TokenType.CROSS_TRACK_REF:
                # Cross-track references would need access to other track data
                # For now, skip
                continue

            # Add notes and advance time
            if pattern_notes:
                notes.extend(pattern_notes)
                # Advance time by pattern duration
                if pattern_notes:
                    max_end = max(n.start_time + n.duration for n in pattern_notes)
                    current_time = max_end

        return notes

    def _pitch_classes_to_notes(
        self,
        pitch_classes: np.ndarray,
        start_time: int,
        full_note_data: Optional[np.ndarray],
        track_idx: int,
    ) -> List[DecodedNote]:
        """Convert pitch classes to actual notes."""
        notes = []

        if full_note_data is not None and len(full_note_data) == len(pitch_classes):
            # Use full note data (pitch, duration, velocity)
            for i, (pitch, duration, velocity) in enumerate(full_note_data):
                notes.append(DecodedNote(
                    pitch=int(pitch),
                    start_time=start_time + i * self.default_duration // 4,
                    duration=int(duration),
                    velocity=int(velocity),
                    track=track_idx,
                ))
        else:
            # Convert pitch classes to MIDI pitches
            for i, pc in enumerate(pitch_classes):
                midi_pitch = int(pc) + self.base_octave * 12 + 12  # +12 for MIDI offset
                notes.append(DecodedNote(
                    pitch=midi_pitch,
                    start_time=start_time + i * self.default_duration // 4,
                    duration=self.default_duration,
                    velocity=self.default_velocity,
                    track=track_idx,
                ))

        return notes

    def to_midi(
        self,
        encoding: TransformRelativeEncoding,
        output_path: str,
        canonical_note_data: Optional[Dict[int, np.ndarray]] = None,
    ) -> bool:
        """
        Decode encoding and save as MIDI file.

        Args:
            encoding: The transform-relative encoding
            output_path: Path to save MIDI file
            canonical_note_data: Optional full note data for canonicals

        Returns:
            True if successful
        """
        try:
            import mido

            # Decode to notes
            all_tracks = self.decode(encoding, canonical_note_data)

            # Create MIDI file
            mid = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)

            for track_idx, track_notes in enumerate(all_tracks):
                track = mido.MidiTrack()
                mid.tracks.append(track)

                # Sort notes by start time
                track_notes.sort(key=lambda n: n.start_time)

                # Convert to MIDI messages
                events = []
                for note in track_notes:
                    events.append((note.start_time, 'note_on', note.pitch, note.velocity))
                    events.append((note.start_time + note.duration, 'note_off', note.pitch, 0))

                events.sort(key=lambda e: (e[0], 0 if e[1] == 'note_off' else 1))

                # Convert to delta times
                current_time = 0
                for time, msg_type, pitch, velocity in events:
                    delta = time - current_time
                    if msg_type == 'note_on':
                        track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=delta))
                    else:
                        track.append(mido.Message('note_off', note=pitch, velocity=0, time=delta))
                    current_time = time

            mid.save(output_path)
            return True

        except ImportError:
            print("mido not installed, cannot save MIDI")
            return False
        except Exception as e:
            print(f"Error saving MIDI: {e}")
            return False


def decode_to_midi(
    encoding: TransformRelativeEncoding,
    output_path: str,
    ticks_per_beat: int = 480,
    canonical_note_data: Optional[Dict[int, np.ndarray]] = None,
) -> bool:
    """
    Convenience function to decode and save MIDI.

    Args:
        encoding: The transform-relative encoding
        output_path: Path to save MIDI file
        ticks_per_beat: MIDI ticks per beat
        canonical_note_data: Optional full note data

    Returns:
        True if successful
    """
    decoder = TransformRelativeDecoder(ticks_per_beat=ticks_per_beat)
    return decoder.to_midi(encoding, output_path, canonical_note_data)


def visualize_encoding(encoding: TransformRelativeEncoding) -> str:
    """
    Create a visual representation of the encoding.

    Returns a string showing the structure of the piece with transforms.
    """
    lines = [encoding.summary(), "", "Token Sequences:"]

    for track_idx, track_tokens in enumerate(encoding.tokens):
        lines.append(f"\nTrack {track_idx}:")
        lines.append(encoding.get_token_sequence_repr(track_idx, max_tokens=30))

    if encoding.canonical_patterns:
        lines.append("\nCanonical Patterns:")
        for i, canon in enumerate(encoding.canonical_patterns[:10]):
            lines.append(f"  {canon}")
        if len(encoding.canonical_patterns) > 10:
            lines.append(f"  ... ({len(encoding.canonical_patterns) - 10} more)")

    return '\n'.join(lines)
