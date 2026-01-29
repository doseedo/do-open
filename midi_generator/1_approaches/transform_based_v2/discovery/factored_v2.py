"""
Factored Object V2 - Refactored Factorization for Algebraic Transform Architecture

Key Changes from V1:
1. Pitch is factored into pitch_class (0-11) and octave (0-9)
2. Contour is COMPUTED from pitch, not stored (reduces redundancy)
3. Velocity is quantized to 8 levels (ppp, pp, p, mp, mf, f, ff, fff)
4. Pattern matching works in pitch-class space (mod 12)

This enables:
- Efficient D24 group operations (work in pitch-class space)
- Octave as transform parameter, not pattern component
- Reduced storage and cleaner algebraic structure
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
import torch


# =============================================================================
# VELOCITY QUANTIZATION
# =============================================================================

# Standard dynamic levels mapping MIDI velocity (0-127) to 8 levels
VELOCITY_LEVELS = {
    0: (0, 15, 'ppp'),      # pianississimo
    1: (16, 31, 'pp'),      # pianissimo
    2: (32, 47, 'p'),       # piano
    3: (48, 63, 'mp'),      # mezzo-piano
    4: (64, 79, 'mf'),      # mezzo-forte
    5: (80, 95, 'f'),       # forte
    6: (96, 111, 'ff'),     # fortissimo
    7: (112, 127, 'fff'),   # fortississimo
}

def quantize_velocity(velocity: float) -> int:
    """
    Quantize velocity (0.0-1.0) to 8 levels (0-7).

    Args:
        velocity: Float in range [0, 1] or int in range [0, 127]

    Returns:
        Integer level 0-7
    """
    # Convert to 0-127 scale if in 0-1 range
    if velocity <= 1.0:
        midi_vel = int(velocity * 127)
    else:
        midi_vel = int(velocity)

    midi_vel = max(0, min(127, midi_vel))

    for level, (low, high, _) in VELOCITY_LEVELS.items():
        if low <= midi_vel <= high:
            return level
    return 4  # Default to mf if something goes wrong


def dequantize_velocity(level: int) -> float:
    """
    Convert quantized level back to velocity (as float 0-1).

    Args:
        level: Integer 0-7

    Returns:
        Float velocity in [0, 1]
    """
    level = max(0, min(7, level))
    low, high, _ = VELOCITY_LEVELS[level]
    midi_vel = (low + high) // 2
    return midi_vel / 127.0


# =============================================================================
# FACTORED OBJECT V2
# =============================================================================

@dataclass
class FactoredObjectV2:
    """
    A musical object with algebraic factorization.

    Key differences from V1:
    - pitch_class: Pitch mod 12 (0-11)
    - octave: Pitch // 12 (0-9)
    - velocity: Quantized to 8 levels (0-7)
    - contour: COMPUTED property, not stored

    This enables efficient D24 group operations and cleaner algebraic structure.
    """
    # Identity
    piece_id: str
    track_id: int
    start_time: int
    scale: int  # temporal scale (16, 32, 64, etc.)

    # Core components
    rhythm: np.ndarray          # (T,) binary onset pattern (unchanged from V1)
    pitch_class: np.ndarray     # (N,) pitch class 0-11 (new: computed as pitch % 12)
    octave: np.ndarray          # (N,) octave 0-9 (new: computed as pitch // 12)
    duration: np.ndarray        # (N,) duration per note in timesteps
    velocity: np.ndarray        # (N,) quantized velocity 0-7 (new: 8 levels)
    onset_times: np.ndarray     # (N,) onset time for each note

    # Metadata
    is_drum: bool = False

    # Original tensor for reconstruction/verification
    original_tensor: np.ndarray = None

    # Derived quantities (computed lazily)
    _rhythm_hash: int = None
    _pitch_class_hash: int = None

    @property
    def num_notes(self) -> int:
        return len(self.pitch_class)

    @property
    def pitches(self) -> np.ndarray:
        """Reconstruct absolute pitches from pitch_class and octave."""
        return self.pitch_class + self.octave * 12

    def get_contour(self) -> np.ndarray:
        """
        Compute contour (intervals) from pitch_class + octave.

        This is a method, not stored, to avoid redundancy.
        Contour is the first difference of absolute pitches.
        """
        if self.num_notes <= 1:
            return np.array([], dtype=np.int32)
        absolute = self.pitch_class.astype(np.int32) + self.octave.astype(np.int32) * 12
        return np.diff(absolute)

    def get_pitch_class_contour(self) -> np.ndarray:
        """
        Compute pitch-class contour (intervals mod 12).

        Useful for matching in pitch-class space where octave doesn't matter.
        """
        if self.num_notes <= 1:
            return np.array([], dtype=np.int32)
        # Use signed mod for intervals in range [-6, 5]
        intervals = np.diff(self.pitch_class.astype(np.int32))
        # Wrap to closest interval (e.g., +9 becomes -3)
        intervals = ((intervals + 6) % 12) - 6
        return intervals

    @property
    def rhythm_hash(self) -> int:
        """Hash for fast rhythm comparison."""
        if self._rhythm_hash is None:
            self._rhythm_hash = hash(self.rhythm.tobytes())
        return self._rhythm_hash

    @property
    def pitch_class_hash(self) -> int:
        """Hash for fast pitch-class pattern comparison."""
        if self._pitch_class_hash is None:
            self._pitch_class_hash = hash(self.pitch_class.tobytes())
        return self._pitch_class_hash

    # Backward compatibility aliases for V1 code
    @property
    def durations(self) -> np.ndarray:
        """Alias for duration (V1 compatibility)."""
        return self.duration

    @property
    def velocities(self) -> np.ndarray:
        """Alias for velocity (V1 compatibility), scaled to 0-1."""
        return self.velocity / 7.0  # Convert 0-7 back to 0-1 range

    @property
    def pitch_contour(self) -> np.ndarray:
        """Alias for computed contour (V1 compatibility)."""
        return self.get_contour()

    def __hash__(self):
        return hash((self.piece_id, self.track_id, self.start_time, self.scale))

    def __eq__(self, other):
        if not isinstance(other, FactoredObjectV2):
            return False
        return (self.piece_id == other.piece_id and
                self.track_id == other.track_id and
                self.start_time == other.start_time and
                self.scale == other.scale)

    def to_v1_format(self) -> dict:
        """
        Convert to V1 format for backwards compatibility.

        Returns dict with V1 fields: pitches, pitch_contour, velocities (0-1 scale)
        """
        return {
            'rhythm': self.rhythm,
            'pitches': self.pitches,
            'pitch_contour': self.get_contour(),
            'velocities': np.array([dequantize_velocity(v) for v in self.velocity]),
            'durations': self.duration,
            'onset_times': self.onset_times,
        }


# =============================================================================
# FACTORIZATION FUNCTIONS
# =============================================================================

def factor_tensor_v2(
    tensor: np.ndarray,
    piece_id: str,
    track_id: int,
    start_time: int,
    scale: int,
    is_drum: bool = False
) -> FactoredObjectV2:
    """
    Factor a piano roll tensor into FactoredObjectV2 components.

    Args:
        tensor: (T, F) piano roll tensor
            - [:, 0:128]: one-hot pitch
            - [:, 128]: velocity
            - [:, 129+]: other features
        piece_id: identifier for the piece
        track_id: track number within piece
        start_time: start timestep
        scale: temporal scale
        is_drum: True if drum/percussion track

    Returns:
        FactoredObjectV2 with separated components
    """
    T, F = tensor.shape

    # Extract rhythm: binary onset pattern
    pitch_activity = tensor[:, :128]
    rhythm = (pitch_activity.sum(axis=1) > 0).astype(np.float32)

    # Extract note events: (timestep, pitch, velocity, duration)
    note_events = []
    active_notes = {}  # pitch -> (start_time, velocity)

    for t in range(T):
        current_pitches = set(np.where(pitch_activity[t] > 0.5)[0])

        # Check for note-offs
        for pitch in list(active_notes.keys()):
            if pitch not in current_pitches:
                start_t, vel = active_notes[pitch]
                duration = t - start_t
                note_events.append((start_t, pitch, vel, duration))
                del active_notes[pitch]

        # Check for note-ons
        for pitch in current_pitches:
            if pitch not in active_notes:
                vel = tensor[t, 128] if F > 128 else 0.8
                active_notes[pitch] = (t, vel)

    # Close remaining active notes
    for pitch, (start_t, vel) in active_notes.items():
        duration = T - start_t
        note_events.append((start_t, pitch, vel, duration))

    # Sort by time, then pitch
    note_events.sort(key=lambda x: (x[0], x[1]))

    if not note_events:
        # Empty object
        return FactoredObjectV2(
            piece_id=piece_id,
            track_id=track_id,
            start_time=start_time,
            scale=scale,
            rhythm=rhythm,
            pitch_class=np.array([], dtype=np.int8),
            octave=np.array([], dtype=np.int8),
            duration=np.array([], dtype=np.int16),
            velocity=np.array([], dtype=np.int8),
            onset_times=np.array([], dtype=np.int16),
            is_drum=is_drum,
            original_tensor=tensor
        )

    # Extract components
    onset_times = np.array([e[0] for e in note_events], dtype=np.int16)
    pitches = np.array([e[1] for e in note_events], dtype=np.int32)
    velocities_raw = np.array([e[2] for e in note_events], dtype=np.float32)
    durations = np.array([e[3] for e in note_events], dtype=np.int16)

    # NEW: Factor pitch into pitch_class and octave
    pitch_class = (pitches % 12).astype(np.int8)
    octave = (pitches // 12).astype(np.int8)

    # NEW: Quantize velocities to 8 levels
    velocity_quantized = np.array([quantize_velocity(v) for v in velocities_raw], dtype=np.int8)

    return FactoredObjectV2(
        piece_id=piece_id,
        track_id=track_id,
        start_time=start_time,
        scale=scale,
        rhythm=rhythm,
        pitch_class=pitch_class,
        octave=octave,
        duration=durations,
        velocity=velocity_quantized,
        onset_times=onset_times,
        is_drum=is_drum,
        original_tensor=tensor
    )


def factor_tensors_batch_gpu_v2(
    tensors: List[np.ndarray],
    metadata: List[dict],
    device: str = 'cuda'
) -> List[FactoredObjectV2]:
    """
    GPU-accelerated batch factoring with V2 factorization.

    Key GPU optimizations:
    - pitch_class = pitches % 12 (single kernel)
    - octave = pitches // 12 (single kernel)
    - velocity quantization (vectorized)

    Args:
        tensors: List of (T, F) numpy arrays all with same T
        metadata: List of dicts with 'piece_id', 'track_id', 'start_time', 'is_drum'
        device: 'cuda' or 'cpu'

    Returns:
        List of FactoredObjectV2
    """
    if not tensors:
        return []

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    T = tensors[0].shape[0]
    N = len(tensors)

    # Stack tensors: [N, T, F]
    stacked = np.stack(tensors)
    pitch_activity = stacked[:, :, :128]  # [N, T, 128]

    # Move to GPU
    pitch_tensor = torch.tensor(pitch_activity, dtype=torch.float32, device=device)

    # Extract rhythms: [N, T]
    rhythms_gpu = (pitch_tensor > 0.5).any(dim=2)
    rhythms = rhythms_gpu.cpu().numpy().astype(np.float32)

    # Edge detection for note events
    pitch_bool = pitch_tensor > 0.5  # [N, T, 128]
    pitch_bool_shifted = torch.zeros_like(pitch_bool)
    pitch_bool_shifted[:, 1:, :] = pitch_bool[:, :-1, :]

    # Note-on: was 0, now 1
    note_ons = pitch_bool & ~pitch_bool_shifted

    # Get velocity if available
    if stacked.shape[2] > 128:
        velocities_arr = stacked[:, :, 128]  # [N, T]
    else:
        velocities_arr = np.full((N, T), 0.8, dtype=np.float32)

    # Move back to CPU for per-object processing
    note_ons_cpu = note_ons.cpu().numpy()
    pitch_bool_cpu = pitch_bool.cpu().numpy()

    results = []
    for i in range(N):
        rhythm = rhythms[i]
        meta = metadata[i]

        # Find all note-on events
        on_times, on_pitches = np.where(note_ons_cpu[i])

        if len(on_times) == 0:
            results.append(FactoredObjectV2(
                piece_id=meta['piece_id'],
                track_id=meta['track_id'],
                start_time=meta['start_time'],
                scale=T,
                rhythm=rhythm,
                pitch_class=np.array([], dtype=np.int8),
                octave=np.array([], dtype=np.int8),
                duration=np.array([], dtype=np.int16),
                velocity=np.array([], dtype=np.int8),
                onset_times=np.array([], dtype=np.int16),
                is_drum=meta.get('is_drum', False),
                original_tensor=tensors[i]
            ))
            continue

        # For each note-on, find duration
        note_events = []
        for t, pitch in zip(on_times, on_pitches):
            pitch_active = pitch_bool_cpu[i, t:, pitch]
            where_off = np.where(~pitch_active)[0]
            duration = where_off[0] if len(where_off) > 0 else len(pitch_active)
            vel = velocities_arr[i, t]
            note_events.append((int(t), int(pitch), float(vel), int(duration)))

        note_events.sort(key=lambda x: (x[0], x[1]))

        onset_times = np.array([e[0] for e in note_events], dtype=np.int16)
        pitches = np.array([e[1] for e in note_events], dtype=np.int32)
        velocities_raw = np.array([e[2] for e in note_events], dtype=np.float32)
        durations = np.array([e[3] for e in note_events], dtype=np.int16)

        # V2 factorization
        pitch_class = (pitches % 12).astype(np.int8)
        octave = (pitches // 12).astype(np.int8)
        velocity_quantized = np.array([quantize_velocity(v) for v in velocities_raw], dtype=np.int8)

        results.append(FactoredObjectV2(
            piece_id=meta['piece_id'],
            track_id=meta['track_id'],
            start_time=meta['start_time'],
            scale=T,
            rhythm=rhythm,
            pitch_class=pitch_class,
            octave=octave,
            duration=durations,
            velocity=velocity_quantized,
            onset_times=onset_times,
            is_drum=meta.get('is_drum', False),
            original_tensor=tensors[i]
        ))

    return results


# =============================================================================
# V2 COMPONENT INDICES
# =============================================================================

class PitchClassIndex:
    """
    Index for matching patterns in pitch-class space (mod 12).

    This enables finding patterns that are transposition-equivalent
    without needing to check all 12 transpositions.
    """
    def __init__(self, objects: List[FactoredObjectV2]):
        """Build index by pitch-class pattern hash."""
        self.index = defaultdict(list)

        for obj in objects:
            if obj.num_notes == 0:
                continue

            # Normalize pitch-class sequence to start at 0
            # This makes it transposition-invariant
            if len(obj.pitch_class) > 0:
                normalized = (obj.pitch_class - obj.pitch_class[0]) % 12
                key = hash(normalized.tobytes())
                self.index[key].append(obj)

    def get_transposition_candidates(self, obj: FactoredObjectV2) -> List[Tuple[FactoredObjectV2, int]]:
        """
        Find objects with same pitch-class pattern (up to transposition).

        Returns:
            List of (matching_object, transposition_amount)
        """
        if obj.num_notes == 0 or len(obj.pitch_class) == 0:
            return []

        normalized = (obj.pitch_class - obj.pitch_class[0]) % 12
        key = hash(normalized.tobytes())

        matches = []
        for candidate in self.index.get(key, []):
            if candidate == obj:
                continue
            # Compute transposition: difference in first pitch classes
            transposition = (obj.pitch_class[0] - candidate.pitch_class[0]) % 12
            # Normalize to range [-6, 5]
            if transposition > 6:
                transposition -= 12
            matches.append((candidate, transposition))

        return matches

    @property
    def num_groups(self) -> int:
        return len(self.index)


class RhythmIndexV2:
    """
    Index for rhythm patterns with transform-aware matching.

    Stores both original and transformed versions for O(1) lookup.
    """
    def __init__(self, objects: List[FactoredObjectV2],
                 include_retrograde: bool = True,
                 time_shifts: List[int] = None):
        """
        Build rhythm index with optional transforms.

        Args:
            objects: List of FactoredObjectV2
            include_retrograde: Include retrograded versions
            time_shifts: List of time shift amounts (e.g., [-16, -8, 8, 16])
        """
        self.index = defaultdict(list)
        self.include_retrograde = include_retrograde
        self.time_shifts = time_shifts or []

        for obj in objects:
            if obj.num_notes == 0:
                continue

            # Index original rhythm
            key = hash(obj.rhythm.tobytes())
            self.index[key].append((obj, 'identity', 0))

            # Index retrograded version
            if include_retrograde:
                retro = obj.rhythm[::-1]
                key = hash(retro.tobytes())
                self.index[key].append((obj, 'retrograde', 0))

            # Index time-shifted versions
            for shift in self.time_shifts:
                shifted = self._apply_time_shift(obj.rhythm, shift)
                key = hash(shifted.tobytes())
                self.index[key].append((obj, 'time_shift', shift))

    def _apply_time_shift(self, rhythm: np.ndarray, shift: int) -> np.ndarray:
        """Apply time shift to rhythm pattern."""
        result = np.zeros_like(rhythm)
        if shift > 0 and shift < len(rhythm):
            result[shift:] = rhythm[:-shift]
        elif shift < 0 and abs(shift) < len(rhythm):
            result[:shift] = rhythm[-shift:]
        return result

    def find_matches(self, target: FactoredObjectV2) -> List[Tuple[FactoredObjectV2, str, int]]:
        """
        Find all sources whose (possibly transformed) rhythm matches target.

        Returns:
            List of (source_object, transform_name, transform_amount)
        """
        if target.num_notes == 0:
            return []

        key = hash(target.rhythm.tobytes())
        matches = self.index.get(key, [])

        # Filter out self-matches
        return [(obj, t_name, t_amt) for obj, t_name, t_amt in matches if obj != target]

    @property
    def num_entries(self) -> int:
        return sum(len(v) for v in self.index.values())


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================

def convert_v1_to_v2(v1_obj) -> FactoredObjectV2:
    """
    Convert a V1 FactoredObject to V2 format.

    Args:
        v1_obj: FactoredObject (V1 format)

    Returns:
        FactoredObjectV2
    """
    # Factor pitches into pitch_class and octave
    pitches = v1_obj.pitches
    pitch_class = (pitches % 12).astype(np.int8)
    octave = (pitches // 12).astype(np.int8)

    # Quantize velocities
    velocity_quantized = np.array([quantize_velocity(v) for v in v1_obj.velocities], dtype=np.int8)

    return FactoredObjectV2(
        piece_id=v1_obj.piece_id,
        track_id=v1_obj.track_id,
        start_time=v1_obj.start_time,
        scale=v1_obj.scale,
        rhythm=v1_obj.rhythm,
        pitch_class=pitch_class,
        octave=octave,
        duration=v1_obj.durations.astype(np.int16),
        velocity=velocity_quantized,
        onset_times=v1_obj.onset_times.astype(np.int16),
        is_drum=v1_obj.is_drum,
        original_tensor=v1_obj.original_tensor
    )


def batch_convert_v1_to_v2(v1_objects: List) -> List[FactoredObjectV2]:
    """
    Batch convert V1 objects to V2 format.

    Uses vectorized operations where possible.
    """
    return [convert_v1_to_v2(obj) for obj in v1_objects]
