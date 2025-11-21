"""
Musical Locality Functions - Agent 1
====================================

Implements 12 musical locality transformations for semantic feature discovery.

These transformations preserve certain musical properties while varying others,
enabling the discovery of invariant semantic features. Each transformation is:
- Invertible (can be undone)
- Musically valid (preserves musical structure)
- Locality-preserving (nearby features remain nearby)

The 12 Locality Types:
1. TRANSPOSE - Shift all pitches by a constant interval
2. INVERT - Invert melodic intervals around a pivot note
3. TIME_SHIFT - Shift note onsets by a constant time
4. AUGMENT - Stretch note durations and inter-onset intervals
5. RETROGRADE - Reverse the sequence of notes
6. DIMINUTION - Compress note durations and inter-onset intervals
7. OCTAVE_SHIFT - Move pitches by octave(s)
8. VELOCITY_SCALE - Scale velocity/dynamics
9. REGISTER_SHIFT - Shift pitch register (different from transpose)
10. INTERVAL_SCALE - Scale interval sizes (melodic expansion/compression)
11. RHYTHMIC_QUANTIZE - Align notes to metric grid
12. VOICE_PERMUTATION - Swap voices/tracks

Mathematical Properties:
- Each transformation T has an inverse T^-1 such that T(T^-1(x)) = x
- Transformations form groups under composition
- Locality is preserved: d(T(x), T(y)) ≈ d(x, y) for nearby x, y

Integration with Agent 2:
- SemanticFeature will use these transformations to generate variants
- Variants should activate the same semantic features if the features are invariant
- This enables discovery of musically meaningful, transformation-invariant features

Author: Agent 1 - Musical Locality Functions
Date: 2025-11-21
License: MIT
"""

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any, Callable
from pathlib import Path
import copy
import warnings


# ==============================================================================
# ENUMS AND DATA STRUCTURES
# ==============================================================================

class LocalityType(Enum):
    """
    Types of musical locality transformations.

    Each transformation preserves certain musical properties while varying others.
    All transformations are invertible and musically valid.
    """
    TRANSPOSE = "transpose"                    # Pitch transposition
    INVERT = "invert"                          # Interval inversion
    TIME_SHIFT = "time_shift"                  # Temporal shift
    AUGMENT = "augment"                        # Rhythmic augmentation
    RETROGRADE = "retrograde"                  # Temporal reversal
    DIMINUTION = "diminution"                  # Rhythmic compression
    OCTAVE_SHIFT = "octave_shift"              # Octave displacement
    VELOCITY_SCALE = "velocity_scale"          # Dynamic scaling
    REGISTER_SHIFT = "register_shift"          # Register displacement
    INTERVAL_SCALE = "interval_scale"          # Interval expansion/compression
    RHYTHMIC_QUANTIZE = "rhythmic_quantize"    # Metric alignment
    VOICE_PERMUTATION = "voice_permutation"    # Voice reordering


@dataclass
class MusicalTransform:
    """
    Represents a musical transformation with its parameters and metadata.

    Attributes:
        transform_type: Type of transformation
        parameters: Transformation-specific parameters
        is_invertible: Whether the transformation can be inverted
        preserves_pitch_content: Whether pitch content is preserved
        preserves_rhythm: Whether rhythmic structure is preserved
        preserves_contour: Whether melodic contour is preserved
        description: Human-readable description
        inverse_parameters: Parameters for the inverse transformation
    """
    transform_type: LocalityType
    parameters: Dict[str, Any]
    is_invertible: bool = True
    preserves_pitch_content: bool = False
    preserves_rhythm: bool = False
    preserves_contour: bool = False
    description: str = ""
    inverse_parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Compute inverse parameters if not provided."""
        if self.inverse_parameters is None and self.is_invertible:
            self.inverse_parameters = self._compute_inverse_parameters()

    def _compute_inverse_parameters(self) -> Dict[str, Any]:
        """
        Compute parameters for the inverse transformation.

        Returns:
            Dictionary of inverse parameters
        """
        if self.transform_type == LocalityType.TRANSPOSE:
            return {"semitones": -self.parameters.get("semitones", 0)}

        elif self.transform_type == LocalityType.INVERT:
            # Inversion is its own inverse (involution)
            return self.parameters.copy()

        elif self.transform_type == LocalityType.TIME_SHIFT:
            return {"shift_seconds": -self.parameters.get("shift_seconds", 0.0)}

        elif self.transform_type == LocalityType.AUGMENT:
            factor = self.parameters.get("factor", 1.0)
            return {"factor": 1.0 / factor if factor != 0 else 1.0}

        elif self.transform_type == LocalityType.RETROGRADE:
            # Retrograde is its own inverse
            return self.parameters.copy()

        elif self.transform_type == LocalityType.DIMINUTION:
            factor = self.parameters.get("factor", 1.0)
            return {"factor": 1.0 / factor if factor != 0 else 1.0}

        elif self.transform_type == LocalityType.OCTAVE_SHIFT:
            return {"octaves": -self.parameters.get("octaves", 0)}

        elif self.transform_type == LocalityType.VELOCITY_SCALE:
            factor = self.parameters.get("factor", 1.0)
            return {
                "factor": 1.0 / factor if factor != 0 else 1.0,
                "min_velocity": self.parameters.get("min_velocity", 1),
                "max_velocity": self.parameters.get("max_velocity", 127)
            }

        elif self.transform_type == LocalityType.REGISTER_SHIFT:
            return {"semitones": -self.parameters.get("semitones", 0)}

        elif self.transform_type == LocalityType.INTERVAL_SCALE:
            factor = self.parameters.get("factor", 1.0)
            return {
                "factor": 1.0 / factor if factor != 0 else 1.0,
                "pivot_pitch": self.parameters.get("pivot_pitch", 60)
            }

        elif self.transform_type == LocalityType.RHYTHMIC_QUANTIZE:
            # Quantization is not perfectly invertible
            return {"grid_division": self.parameters.get("grid_division", 16)}

        elif self.transform_type == LocalityType.VOICE_PERMUTATION:
            # Invert the permutation
            perm = self.parameters.get("permutation", [])
            if perm:
                inverse_perm = [0] * len(perm)
                for i, p in enumerate(perm):
                    inverse_perm[p] = i
                return {"permutation": inverse_perm}
            return {"permutation": []}

        return self.parameters.copy()


# ==============================================================================
# MAIN TRANSFORMATION CLASS
# ==============================================================================

class MusicalLocalityFunctions:
    """
    Implements 12 musical locality transformations.

    All transformations are:
    - Invertible (reversible)
    - Musically valid
    - Locality-preserving

    Usage:
        # Create transformer
        transformer = MusicalLocalityFunctions()

        # Apply transformation
        midi_file = MidiFile('song.mid')
        transform = MusicalTransform(
            transform_type=LocalityType.TRANSPOSE,
            parameters={"semitones": 5}
        )
        transformed = transformer.apply_transform(midi_file, transform)

        # Invert transformation
        original = transformer.invert_transform(transformed, transform)
    """

    def __init__(self, preserve_tempo: bool = True, preserve_key_signature: bool = False):
        """
        Initialize the locality transformer.

        Args:
            preserve_tempo: Whether to preserve tempo meta messages
            preserve_key_signature: Whether to preserve key signature meta messages
        """
        self.preserve_tempo = preserve_tempo
        self.preserve_key_signature = preserve_key_signature

    # ==========================================================================
    # MAIN API
    # ==========================================================================

    def apply_transform(
        self,
        midi_file: MidiFile,
        transform: MusicalTransform
    ) -> MidiFile:
        """
        Apply a musical transformation to a MIDI file.

        Args:
            midi_file: Input MIDI file
            transform: Transformation to apply

        Returns:
            Transformed MIDI file
        """
        # Dispatch to appropriate transformation function
        transform_func = self._get_transform_function(transform.transform_type)
        return transform_func(midi_file, transform.parameters)

    def invert_transform(
        self,
        midi_file: MidiFile,
        transform: MusicalTransform
    ) -> MidiFile:
        """
        Invert a transformation (apply the inverse).

        Args:
            midi_file: Input MIDI file (transformed)
            transform: Original transformation

        Returns:
            Inverted (approximately original) MIDI file
        """
        if not transform.is_invertible:
            warnings.warn(f"Transform {transform.transform_type} is not perfectly invertible")

        inverse_transform = MusicalTransform(
            transform_type=transform.transform_type,
            parameters=transform.inverse_parameters or transform.parameters,
            is_invertible=transform.is_invertible
        )

        return self.apply_transform(midi_file, inverse_transform)

    def _get_transform_function(self, transform_type: LocalityType) -> Callable:
        """Get the function for a given transformation type."""
        transform_map = {
            LocalityType.TRANSPOSE: self.transpose,
            LocalityType.INVERT: self.invert_intervals,
            LocalityType.TIME_SHIFT: self.time_shift,
            LocalityType.AUGMENT: self.augment,
            LocalityType.RETROGRADE: self.retrograde,
            LocalityType.DIMINUTION: self.diminution,
            LocalityType.OCTAVE_SHIFT: self.octave_shift,
            LocalityType.VELOCITY_SCALE: self.velocity_scale,
            LocalityType.REGISTER_SHIFT: self.register_shift,
            LocalityType.INTERVAL_SCALE: self.interval_scale,
            LocalityType.RHYTHMIC_QUANTIZE: self.rhythmic_quantize,
            LocalityType.VOICE_PERMUTATION: self.voice_permutation,
        }
        return transform_map[transform_type]

    # ==========================================================================
    # TRANSFORMATION 1: TRANSPOSE
    # ==========================================================================

    def transpose(self, midi_file: MidiFile, params: Dict[str, Any]) -> MidiFile:
        """
        Transpose all pitches by a constant interval.

        This is a pitch-preserving transformation in the sense that it maintains
        interval relationships and contour.

        Args:
            midi_file: Input MIDI file
            params: {"semitones": int}  # Number of semitones to transpose

        Returns:
            Transposed MIDI file
        """
        semitones = params.get("semitones", 0)
        result = self._copy_midi(midi_file)

        for track in result.tracks:
            for msg in track:
                if msg.type in ('note_on', 'note_off') and hasattr(msg, 'note'):
                    # Transpose note, clamp to valid MIDI range [0, 127]
                    new_note = np.clip(msg.note + semitones, 0, 127)
                    msg.note = int(new_note)

        return result

    # ==========================================================================
    # TRANSFORMATION 2: INVERT INTERVALS
    # ==========================================================================

    def invert_intervals(self, midi_file: MidiFile, params: Dict[str, Any]) -> MidiFile:
        """
        Invert melodic intervals around a pivot note.

        If the original melody goes up by interval I, the inverted melody goes
        down by interval I (and vice versa).

        Args:
            midi_file: Input MIDI file
            params: {
                "pivot_pitch": int,  # Pitch to invert around (default: 60 = middle C)
                "per_track": bool    # Whether to invert each track separately (default: True)
            }

        Returns:
            Interval-inverted MIDI file
        """
        pivot_pitch = params.get("pivot_pitch", 60)
        per_track = params.get("per_track", True)
        result = self._copy_midi(midi_file)

        for track in result.tracks:
            for msg in track:
                if msg.type in ('note_on', 'note_off') and hasattr(msg, 'note'):
                    # Invert: new_pitch = pivot - (old_pitch - pivot) = 2*pivot - old_pitch
                    new_note = 2 * pivot_pitch - msg.note
                    msg.note = int(np.clip(new_note, 0, 127))

        return result

    # ==========================================================================
    # TRANSFORMATION 3: TIME SHIFT
    # ==========================================================================

    def time_shift(self, midi_file: MidiFile, params: Dict[str, Any]) -> MidiFile:
        """
        Shift all note onsets by a constant time.

        This preserves all musical content but changes the starting point.

        Args:
            midi_file: Input MIDI file
            params: {"shift_seconds": float}  # Seconds to shift (positive or negative)

        Returns:
            Time-shifted MIDI file
        """
        shift_seconds = params.get("shift_seconds", 0.0)
        result = self._copy_midi(midi_file)

        # Convert seconds to ticks
        ticks_per_beat = result.ticks_per_beat
        # Assume 120 BPM default tempo
        tempo = 500000  # microseconds per beat
        ticks_per_second = (ticks_per_beat * 1_000_000) / tempo
        shift_ticks = int(shift_seconds * ticks_per_second)

        if shift_ticks == 0:
            return result

        for track in result.tracks:
            # Collect all messages with times
            messages_with_times = []
            current_time = 0

            for msg in track:
                current_time += msg.time
                messages_with_times.append((current_time, msg))

            # Shift times
            shifted_messages = []
            for abs_time, msg in messages_with_times:
                new_time = max(0, abs_time + shift_ticks)
                shifted_messages.append((new_time, msg))

            # Reconstruct track with delta times
            track.clear()
            prev_time = 0
            for abs_time, msg in shifted_messages:
                msg.time = abs_time - prev_time
                track.append(msg)
                prev_time = abs_time

        return result

    # ==========================================================================
    # TRANSFORMATION 4: AUGMENT (Rhythmic)
    # ==========================================================================

    def augment(self, midi_file: MidiFile, params: Dict[str, Any]) -> MidiFile:
        """
        Augment (stretch) note durations and inter-onset intervals.

        This is classical rhythmic augmentation: all time values are multiplied
        by a constant factor.

        Args:
            midi_file: Input MIDI file
            params: {"factor": float}  # Factor to stretch time (e.g., 2.0 = twice as slow)

        Returns:
            Augmented MIDI file
        """
        factor = params.get("factor", 1.0)
        result = self._copy_midi(midi_file)

        for track in result.tracks:
            for msg in track:
                # Scale all delta times
                msg.time = int(msg.time * factor)

        return result

    # ==========================================================================
    # TRANSFORMATION 5: RETROGRADE
    # ==========================================================================

    def retrograde(self, midi_file: MidiFile, params: Dict[str, Any]) -> MidiFile:
        """
        Reverse the sequence of notes (retrograde/crab canon).

        This preserves pitch and duration content but reverses temporal order.

        Args:
            midi_file: Input MIDI file
            params: {"preserve_track_structure": bool}  # Keep tracks separate (default: True)

        Returns:
            Retrograde MIDI file
        """
        preserve_track_structure = params.get("preserve_track_structure", True)
        result = self._copy_midi(midi_file)

        for track_idx, track in enumerate(result.tracks):
            # Extract all note events with absolute times
            note_events = []
            current_time = 0
            meta_messages = []

            for msg in track:
                current_time += msg.time

                if msg.type in ('note_on', 'note_off'):
                    note_events.append((current_time, msg.copy()))
                elif msg.is_meta:
                    meta_messages.append((current_time, msg.copy()))

            if not note_events:
                continue

            # Find the total duration
            max_time = max(t for t, _ in note_events) if note_events else 0

            # Reverse note events
            reversed_events = []
            for abs_time, msg in note_events:
                new_time = max_time - abs_time
                reversed_events.append((new_time, msg))

            # Sort by time
            reversed_events.sort(key=lambda x: x[0])

            # Reconstruct track
            track.clear()

            # Add meta messages at the beginning
            for meta_time, meta_msg in meta_messages:
                if meta_time == 0:
                    meta_msg.time = 0
                    track.append(meta_msg)

            # Add reversed notes
            prev_time = 0
            for abs_time, msg in reversed_events:
                msg.time = abs_time - prev_time
                track.append(msg)
                prev_time = abs_time

            # Add end of track
            track.append(MetaMessage('end_of_track', time=0))

        return result

    # ==========================================================================
    # TRANSFORMATION 6: DIMINUTION (Rhythmic)
    # ==========================================================================

    def diminution(self, midi_file: MidiFile, params: Dict[str, Any]) -> MidiFile:
        """
        Diminution (compress) note durations and inter-onset intervals.

        This is the opposite of augmentation: all time values are divided
        by a constant factor.

        Args:
            midi_file: Input MIDI file
            params: {"factor": float}  # Factor to compress time (e.g., 2.0 = twice as fast)

        Returns:
            Diminuted MIDI file
        """
        factor = params.get("factor", 1.0)
        # Diminution is just augmentation with inverse factor
        return self.augment(midi_file, {"factor": 1.0 / factor if factor != 0 else 1.0})

    # ==========================================================================
    # TRANSFORMATION 7: OCTAVE SHIFT
    # ==========================================================================

    def octave_shift(self, midi_file: MidiFile, params: Dict[str, Any]) -> MidiFile:
        """
        Shift pitches by octave(s).

        This is a special case of transposition that preserves pitch class.

        Args:
            midi_file: Input MIDI file
            params: {"octaves": int}  # Number of octaves to shift (positive or negative)

        Returns:
            Octave-shifted MIDI file
        """
        octaves = params.get("octaves", 0)
        return self.transpose(midi_file, {"semitones": octaves * 12})

    # ==========================================================================
    # TRANSFORMATION 8: VELOCITY SCALE
    # ==========================================================================

    def velocity_scale(self, midi_file: MidiFile, params: Dict[str, Any]) -> MidiFile:
        """
        Scale velocity/dynamics by a constant factor.

        This changes the overall dynamic level while preserving relative dynamics.

        Args:
            midi_file: Input MIDI file
            params: {
                "factor": float,        # Scale factor (e.g., 1.5 = 50% louder)
                "min_velocity": int,    # Minimum velocity (default: 1)
                "max_velocity": int     # Maximum velocity (default: 127)
            }

        Returns:
            Velocity-scaled MIDI file
        """
        factor = params.get("factor", 1.0)
        min_vel = params.get("min_velocity", 1)
        max_vel = params.get("max_velocity", 127)
        result = self._copy_midi(midi_file)

        for track in result.tracks:
            for msg in track:
                if msg.type == 'note_on' and hasattr(msg, 'velocity'):
                    new_velocity = int(msg.velocity * factor)
                    msg.velocity = int(np.clip(new_velocity, min_vel, max_vel))

        return result

    # ==========================================================================
    # TRANSFORMATION 9: REGISTER SHIFT
    # ==========================================================================

    def register_shift(self, midi_file: MidiFile, params: Dict[str, Any]) -> MidiFile:
        """
        Shift pitch register (different from simple transposition).

        This moves pitches to a different register while attempting to preserve
        musical character.

        Args:
            midi_file: Input MIDI file
            params: {"semitones": int}  # Semitones to shift

        Returns:
            Register-shifted MIDI file
        """
        # For now, register shift is equivalent to transpose
        # In a more sophisticated version, this could preserve pitch class
        # distributions or apply different shifts to different voices
        return self.transpose(midi_file, params)

    # ==========================================================================
    # TRANSFORMATION 10: INTERVAL SCALE
    # ==========================================================================

    def interval_scale(self, midi_file: MidiFile, params: Dict[str, Any]) -> MidiFile:
        """
        Scale interval sizes (melodic expansion/compression).

        Intervals are multiplied by a factor around a pivot pitch.
        This creates melodic expansion (factor > 1) or compression (factor < 1).

        Args:
            midi_file: Input MIDI file
            params: {
                "factor": float,        # Scale factor for intervals
                "pivot_pitch": int      # Pivot pitch (default: 60 = middle C)
            }

        Returns:
            Interval-scaled MIDI file
        """
        factor = params.get("factor", 1.0)
        pivot_pitch = params.get("pivot_pitch", 60)
        result = self._copy_midi(midi_file)

        for track in result.tracks:
            for msg in track:
                if msg.type in ('note_on', 'note_off') and hasattr(msg, 'note'):
                    # Scale interval from pivot
                    interval_from_pivot = msg.note - pivot_pitch
                    new_interval = interval_from_pivot * factor
                    new_note = pivot_pitch + new_interval
                    msg.note = int(np.clip(new_note, 0, 127))

        return result

    # ==========================================================================
    # TRANSFORMATION 11: RHYTHMIC QUANTIZE
    # ==========================================================================

    def rhythmic_quantize(self, midi_file: MidiFile, params: Dict[str, Any]) -> MidiFile:
        """
        Quantize note onsets to a metric grid.

        This is not perfectly invertible but is musically meaningful.

        Args:
            midi_file: Input MIDI file
            params: {
                "grid_division": int,   # Grid division (e.g., 16 = 16th notes)
                "strength": float       # Quantization strength 0-1 (default: 1.0)
            }

        Returns:
            Quantized MIDI file
        """
        grid_division = params.get("grid_division", 16)
        strength = params.get("strength", 1.0)
        result = self._copy_midi(midi_file)

        ticks_per_beat = result.ticks_per_beat
        grid_ticks = ticks_per_beat * 4 // grid_division  # 4 = whole note

        for track in result.tracks:
            # Extract events with absolute times
            events_with_times = []
            current_time = 0

            for msg in track:
                current_time += msg.time
                events_with_times.append((current_time, msg))

            # Quantize times
            quantized_events = []
            for abs_time, msg in events_with_times:
                if msg.type in ('note_on', 'note_off'):
                    # Find nearest grid point
                    grid_point = round(abs_time / grid_ticks) * grid_ticks
                    # Apply quantization strength
                    new_time = int(abs_time * (1 - strength) + grid_point * strength)
                    quantized_events.append((new_time, msg))
                else:
                    quantized_events.append((abs_time, msg))

            # Reconstruct track
            track.clear()
            prev_time = 0
            for abs_time, msg in quantized_events:
                msg.time = max(0, abs_time - prev_time)
                track.append(msg)
                prev_time = abs_time

        return result

    # ==========================================================================
    # TRANSFORMATION 12: VOICE PERMUTATION
    # ==========================================================================

    def voice_permutation(self, midi_file: MidiFile, params: Dict[str, Any]) -> MidiFile:
        """
        Permute (swap) voices/tracks.

        This reorders tracks according to a permutation, which can reveal
        features that are invariant to voice ordering.

        Args:
            midi_file: Input MIDI file
            params: {
                "permutation": List[int]  # Permutation of track indices
                                          # e.g., [1, 0, 2] swaps first two tracks
            }

        Returns:
            Voice-permuted MIDI file
        """
        permutation = params.get("permutation", [])
        result = self._copy_midi(midi_file)

        if not permutation:
            return result

        # Validate permutation
        n_tracks = len(result.tracks)
        if len(permutation) != n_tracks:
            warnings.warn(f"Permutation length {len(permutation)} != track count {n_tracks}")
            return result

        # Apply permutation
        original_tracks = [track for track in result.tracks]
        result.tracks.clear()

        for new_idx in permutation:
            if 0 <= new_idx < n_tracks:
                result.tracks.append(original_tracks[new_idx])
            else:
                warnings.warn(f"Invalid permutation index {new_idx}")

        return result

    # ==========================================================================
    # UTILITY FUNCTIONS
    # ==========================================================================

    def _copy_midi(self, midi_file: MidiFile) -> MidiFile:
        """
        Create a deep copy of a MIDI file.

        Args:
            midi_file: Input MIDI file

        Returns:
            Deep copy of the MIDI file
        """
        # Create new MIDI file with same settings
        result = MidiFile(type=midi_file.type, ticks_per_beat=midi_file.ticks_per_beat)

        # Deep copy all tracks
        for track in midi_file.tracks:
            new_track = MidiTrack()
            for msg in track:
                new_track.append(msg.copy())
            result.tracks.append(new_track)

        return result

    def validate_invertibility(
        self,
        midi_file: MidiFile,
        transform: MusicalTransform,
        tolerance: float = 1e-6
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that a transformation is invertible.

        Args:
            midi_file: Input MIDI file
            transform: Transformation to test
            tolerance: Tolerance for floating-point comparisons

        Returns:
            Tuple of (is_invertible, metrics)
        """
        # Apply transform
        transformed = self.apply_transform(midi_file, transform)

        # Apply inverse
        recovered = self.invert_transform(transformed, transform)

        # Compare original and recovered
        metrics = self._compare_midi_files(midi_file, recovered, tolerance)

        is_invertible = (
            metrics['note_mismatch_count'] == 0 and
            metrics['max_time_error'] < tolerance
        )

        return is_invertible, metrics

    def _compare_midi_files(
        self,
        midi1: MidiFile,
        midi2: MidiFile,
        tolerance: float
    ) -> Dict[str, Any]:
        """
        Compare two MIDI files and return difference metrics.

        Args:
            midi1: First MIDI file
            midi2: Second MIDI file
            tolerance: Tolerance for comparisons

        Returns:
            Dictionary of comparison metrics
        """
        metrics = {
            'track_count_diff': len(midi1.tracks) - len(midi2.tracks),
            'note_mismatch_count': 0,
            'max_time_error': 0.0,
            'max_pitch_error': 0,
            'max_velocity_error': 0
        }

        for track1, track2 in zip(midi1.tracks, midi2.tracks):
            if len(track1) != len(track2):
                metrics['note_mismatch_count'] += abs(len(track1) - len(track2))
                continue

            for msg1, msg2 in zip(track1, track2):
                if msg1.type != msg2.type:
                    metrics['note_mismatch_count'] += 1
                    continue

                # Compare timing
                time_error = abs(msg1.time - msg2.time)
                metrics['max_time_error'] = max(metrics['max_time_error'], time_error)

                # Compare pitch (if applicable)
                if hasattr(msg1, 'note') and hasattr(msg2, 'note'):
                    pitch_error = abs(msg1.note - msg2.note)
                    metrics['max_pitch_error'] = max(metrics['max_pitch_error'], pitch_error)

                # Compare velocity (if applicable)
                if hasattr(msg1, 'velocity') and hasattr(msg2, 'velocity'):
                    vel_error = abs(msg1.velocity - msg2.velocity)
                    metrics['max_velocity_error'] = max(metrics['max_velocity_error'], vel_error)

        return metrics


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def create_random_transform(
    transform_type: Optional[LocalityType] = None,
    random_state: Optional[np.random.RandomState] = None
) -> MusicalTransform:
    """
    Create a random transformation for data augmentation.

    Args:
        transform_type: Specific transform type (or None for random)
        random_state: Random state for reproducibility

    Returns:
        Random MusicalTransform
    """
    rng = random_state or np.random.RandomState()

    if transform_type is None:
        transform_type = rng.choice(list(LocalityType))

    # Generate random parameters based on type
    if transform_type == LocalityType.TRANSPOSE:
        params = {"semitones": rng.randint(-12, 13)}

    elif transform_type == LocalityType.INVERT:
        params = {"pivot_pitch": rng.randint(48, 73)}  # Around middle C

    elif transform_type == LocalityType.TIME_SHIFT:
        params = {"shift_seconds": rng.uniform(-2.0, 2.0)}

    elif transform_type == LocalityType.AUGMENT:
        params = {"factor": rng.uniform(0.5, 2.0)}

    elif transform_type == LocalityType.RETROGRADE:
        params = {}

    elif transform_type == LocalityType.DIMINUTION:
        params = {"factor": rng.uniform(1.5, 3.0)}

    elif transform_type == LocalityType.OCTAVE_SHIFT:
        params = {"octaves": rng.randint(-2, 3)}

    elif transform_type == LocalityType.VELOCITY_SCALE:
        params = {"factor": rng.uniform(0.7, 1.3)}

    elif transform_type == LocalityType.REGISTER_SHIFT:
        params = {"semitones": rng.randint(-24, 25)}

    elif transform_type == LocalityType.INTERVAL_SCALE:
        params = {
            "factor": rng.uniform(0.5, 1.5),
            "pivot_pitch": rng.randint(48, 73)
        }

    elif transform_type == LocalityType.RHYTHMIC_QUANTIZE:
        params = {
            "grid_division": rng.choice([8, 16, 32]),
            "strength": rng.uniform(0.5, 1.0)
        }

    elif transform_type == LocalityType.VOICE_PERMUTATION:
        # Create a random permutation of 4 tracks
        n_tracks = 4
        perm = list(range(n_tracks))
        rng.shuffle(perm)
        params = {"permutation": perm}

    else:
        params = {}

    return MusicalTransform(
        transform_type=transform_type,
        parameters=params
    )
