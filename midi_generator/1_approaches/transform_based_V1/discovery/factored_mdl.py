"""
Factored MDL Discovery - Lewinian Multi-Space Architecture

The key insight: Musical objects should live in a PRODUCT SPACE, not a single
fused tensor space. This allows discovering transforms that operate on
individual components (rhythm, pitch, dynamics) independently.

Architecture:
    Object = (Rhythm, PitchContour, Velocities, Durations)

    Transform = T_rhythm × T_pitch × T_velocity × T_duration

    Example: "Trumpet_A → Sax_B = identity(rhythm) × transpose(7)(pitch) × scale(0.8)(velocity)"

This enables:
    1. Same rhythm, different pitches (harmonization)
    2. Same pitches, different rhythm (rhythmic variation)
    3. Cross-component derivation (brass rhythm + sax pitches)
    4. Hierarchical structure (motif → track → section → song)

Author: Factored MDL Implementation
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import math
from itertools import product


# =============================================================================
# FACTORED OBJECT REPRESENTATION
# =============================================================================

@dataclass
class FactoredObject:
    """
    A musical object factored into independent components.

    Each component can be matched/transformed independently, enabling
    much richer pattern discovery than atomic tensor matching.
    """
    # Identity
    piece_id: str
    track_id: int
    start_time: int
    scale: int  # temporal scale (16, 32, 64, etc.)

    # Factored components
    rhythm: np.ndarray          # (T,) binary onset pattern
    pitch_contour: np.ndarray   # (N,) intervals between successive notes
    pitches: np.ndarray         # (N,) absolute MIDI pitches of notes
    velocities: np.ndarray      # (N,) velocity per note
    durations: np.ndarray       # (N,) duration per note in timesteps
    onset_times: np.ndarray     # (N,) onset time for each note - REQUIRED for reconstruction

    # Metadata
    is_drum: bool = False       # True if this is a drum/percussion track (MIDI channel 9)

    # Original tensor for reconstruction/verification
    original_tensor: np.ndarray = None

    # Derived quantities (computed lazily)
    _rhythm_hash: int = None
    _contour_hash: int = None

    @property
    def num_notes(self) -> int:
        return len(self.pitches)

    @property
    def rhythm_hash(self) -> int:
        """Hash for fast rhythm comparison."""
        if self._rhythm_hash is None:
            self._rhythm_hash = hash(self.rhythm.tobytes())
        return self._rhythm_hash

    @property
    def contour_hash(self) -> int:
        """Hash for fast contour comparison."""
        if self._contour_hash is None:
            self._contour_hash = hash(self.pitch_contour.tobytes())
        return self._contour_hash

    def __hash__(self):
        return hash((self.piece_id, self.track_id, self.start_time, self.scale))

    def __eq__(self, other):
        return (self.piece_id == other.piece_id and
                self.track_id == other.track_id and
                self.start_time == other.start_time and
                self.scale == other.scale)


def factor_tensor(tensor: np.ndarray, piece_id: str, track_id: int,
                  start_time: int, scale: int, is_drum: bool = False) -> FactoredObject:
    """
    Factor a piano roll tensor into independent musical components.

    Args:
        tensor: (T, F) piano roll tensor
            - [:, 0:128]: one-hot pitch
            - [:, 128]: velocity
            - [:, 129+]: other features
        piece_id: identifier for the piece
        track_id: track number within piece
        start_time: start timestep
        scale: temporal scale
        is_drum: True if this is a drum/percussion track (MIDI channel 9)

    Returns:
        FactoredObject with separated components
    """
    T, F = tensor.shape

    # Extract rhythm: binary onset pattern
    # A timestep has an onset if any pitch is active
    pitch_activity = tensor[:, :128]
    rhythm = (pitch_activity.sum(axis=1) > 0).astype(np.float32)

    # Extract note events: (timestep, pitch, velocity, duration)
    note_events = []
    active_notes = {}  # pitch -> (start_time, velocity)

    for t in range(T):
        current_pitches = set(np.where(pitch_activity[t] > 0.5)[0])

        # Check for note-offs (pitches that were active but aren't now)
        for pitch in list(active_notes.keys()):
            if pitch not in current_pitches:
                start_t, vel = active_notes[pitch]
                duration = t - start_t
                note_events.append((start_t, pitch, vel, duration))
                del active_notes[pitch]

        # Check for note-ons (new pitches)
        for pitch in current_pitches:
            if pitch not in active_notes:
                vel = tensor[t, 128] if F > 128 else 0.8
                active_notes[pitch] = (t, vel)

    # Close any remaining active notes
    for pitch, (start_t, vel) in active_notes.items():
        duration = T - start_t
        note_events.append((start_t, pitch, vel, duration))

    # Sort by time, then pitch
    note_events.sort(key=lambda x: (x[0], x[1]))

    if not note_events:
        # Empty object
        return FactoredObject(
            piece_id=piece_id,
            track_id=track_id,
            start_time=start_time,
            scale=scale,
            rhythm=rhythm,
            pitch_contour=np.array([], dtype=np.int32),
            pitches=np.array([], dtype=np.int32),
            velocities=np.array([], dtype=np.float32),
            durations=np.array([], dtype=np.int32),
            onset_times=np.array([], dtype=np.int32),
            is_drum=is_drum,
            original_tensor=tensor
        )

    # Extract note data - note_events is (start_t, pitch, velocity, duration)
    onset_times = np.array([e[0] for e in note_events], dtype=np.int32)
    pitches = np.array([e[1] for e in note_events], dtype=np.int32)
    velocities = np.array([e[2] for e in note_events], dtype=np.float32)
    durations = np.array([e[3] for e in note_events], dtype=np.int32)

    # Pitch contour: intervals between successive notes
    if len(pitches) > 1:
        pitch_contour = np.diff(pitches).astype(np.int32)
    else:
        pitch_contour = np.array([], dtype=np.int32)

    return FactoredObject(
        piece_id=piece_id,
        track_id=track_id,
        start_time=start_time,
        scale=scale,
        rhythm=rhythm,
        pitch_contour=pitch_contour,
        pitches=pitches,
        velocities=velocities,
        durations=durations,
        onset_times=onset_times,
        is_drum=is_drum,
        original_tensor=tensor
    )


def factor_tensors_batch_gpu(
    tensors: List[np.ndarray],
    metadata: List[dict],
    verbose: bool = False
) -> List[FactoredObject]:
    """
    GPU-accelerated batch factoring of tensors grouped by scale.

    For rhythm extraction (easy - fully parallel):
    - rhythm = any(tensor[:, :128] > 0, axis=1)

    For note events (harder - need state tracking):
    - Use vectorized edge detection to find note-on/note-off
    - Process in parallel per-pitch

    Args:
        tensors: List of (T, F) numpy arrays all with same T
        metadata: List of dicts with 'piece_id', 'track_id', 'start_time'
        verbose: Print debug info

    Returns:
        List of FactoredObject
    """
    import torch

    if not tensors:
        return []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T = tensors[0].shape[0]  # All tensors same length in this batch
    N = len(tensors)

    # Stack tensors: [N, T, F]
    stacked = np.stack(tensors)
    pitch_activity = stacked[:, :, :128]  # [N, T, 128]

    # Move to GPU
    pitch_tensor = torch.tensor(pitch_activity, dtype=torch.float32, device=device)

    # Extract rhythms: [N, T] - any pitch active at each timestep
    rhythms_gpu = (pitch_tensor > 0.5).any(dim=2)  # [N, T]
    rhythms = rhythms_gpu.cpu().numpy().astype(np.float32)

    # Extract note events using edge detection
    # Note-on: pitch goes from 0 to 1 (or first timestep is 1)
    # Note-off: pitch goes from 1 to 0 (or last timestep ends)

    # Shifted comparison for edge detection
    pitch_bool = pitch_tensor > 0.5  # [N, T, 128]
    pitch_bool_shifted = torch.zeros_like(pitch_bool)
    pitch_bool_shifted[:, 1:, :] = pitch_bool[:, :-1, :]

    # Note-on: was 0, now 1
    note_ons = pitch_bool & ~pitch_bool_shifted  # [N, T, 128]

    # Note-off: was 1, now 0 (or end of tensor)
    note_offs_internal = ~pitch_bool & pitch_bool_shifted  # [N, T, 128]
    # Add note-offs for notes active at end
    note_offs_end = pitch_bool[:, -1:, :].expand(-1, T, -1)  # last timestep
    note_offs_mask = torch.zeros_like(pitch_bool)
    note_offs_mask[:, -1, :] = pitch_bool[:, -1, :]

    # Get velocity if available
    if stacked.shape[2] > 128:
        velocities_arr = stacked[:, :, 128]  # [N, T]
    else:
        velocities_arr = np.full((N, T), 0.8, dtype=np.float32)

    # Move back to CPU for per-object processing
    note_ons_cpu = note_ons.cpu().numpy()
    pitch_bool_cpu = pitch_bool.cpu().numpy()

    # Process each object
    results = []
    for i in range(N):
        rhythm = rhythms[i]
        meta = metadata[i]

        # Find all note-on events
        on_times, on_pitches = np.where(note_ons_cpu[i])

        if len(on_times) == 0:
            # Empty object
            results.append(FactoredObject(
                piece_id=meta['piece_id'],
                track_id=meta['track_id'],
                start_time=meta['start_time'],
                scale=T,
                rhythm=rhythm,
                pitch_contour=np.array([], dtype=np.int32),
                pitches=np.array([], dtype=np.int32),
                velocities=np.array([], dtype=np.float32),
                durations=np.array([], dtype=np.int32),
                onset_times=np.array([], dtype=np.int32),
                is_drum=meta.get('is_drum', False),
                original_tensor=tensors[i]
            ))
            continue

        # For each note-on, find its duration
        note_events = []
        for t, pitch in zip(on_times, on_pitches):
            # Find when this note ends
            pitch_active = pitch_bool_cpu[i, t:, pitch]
            active_len = pitch_active.sum()  # How many timesteps is this pitch active from t onwards

            # But we need consecutive activity, not total
            # Find first 0 after this point (or end of tensor)
            where_off = np.where(~pitch_active)[0]
            if len(where_off) > 0:
                duration = where_off[0]
            else:
                duration = len(pitch_active)

            vel = velocities_arr[i, t]
            note_events.append((int(t), int(pitch), float(vel), int(duration)))

        # Sort by time, then pitch
        note_events.sort(key=lambda x: (x[0], x[1]))

        onset_times = np.array([e[0] for e in note_events], dtype=np.int32)
        pitches = np.array([e[1] for e in note_events], dtype=np.int32)
        velocities = np.array([e[2] for e in note_events], dtype=np.float32)
        durations = np.array([e[3] for e in note_events], dtype=np.int32)

        # Pitch contour
        if len(pitches) > 1:
            pitch_contour = np.diff(pitches).astype(np.int32)
        else:
            pitch_contour = np.array([], dtype=np.int32)

        results.append(FactoredObject(
            piece_id=meta['piece_id'],
            track_id=meta['track_id'],
            start_time=meta['start_time'],
            scale=T,
            rhythm=rhythm,
            pitch_contour=pitch_contour,
            pitches=pitches,
            velocities=velocities,
            durations=durations,
            onset_times=onset_times,
            is_drum=meta.get('is_drum', False),
            original_tensor=tensors[i]
        ))

    return results


def factor_objects_from_corpus(objects: List, verbose: bool = True, use_gpu: bool = True) -> List[FactoredObject]:
    """
    Factor all objects from corpus into FactoredObjects.

    Args:
        objects: List of MusicalObject (from old system)
        verbose: Print progress
        use_gpu: Use GPU-accelerated batch processing

    Returns:
        List of FactoredObject
    """
    if verbose:
        print(f"\n{'='*70}")
        print("FACTORING OBJECTS INTO COMPONENTS")
        print(f"{'='*70}")
        print(f"  Input objects: {len(objects)}")

    if use_gpu and len(objects) > 100:
        # Group objects by scale (tensor length) for GPU batching
        from collections import defaultdict
        import torch

        if torch.cuda.is_available():
            if verbose:
                print(f"  Using GPU acceleration (cuda)")

            objects_by_scale = defaultdict(list)
            for obj in objects:
                scale = obj.tensor.shape[0]
                objects_by_scale[scale].append(obj)

            if verbose:
                print(f"  Grouped into {len(objects_by_scale)} scale groups: {sorted(objects_by_scale.keys())}")

            factored = []
            for scale in sorted(objects_by_scale.keys()):
                group = objects_by_scale[scale]
                tensors = [o.tensor for o in group]
                metadata = [{'piece_id': o.piece_id, 'track_id': o.track_id, 'start_time': o.start_time,
                            'is_drum': getattr(o, 'is_drum', False)}
                           for o in group]

                batch_results = factor_tensors_batch_gpu(tensors, metadata, verbose=False)
                factored.extend(batch_results)

            empty_count = sum(1 for f in factored if f.num_notes == 0)
        else:
            # Fall back to CPU
            if verbose:
                print(f"  GPU not available, using CPU")
            factored = []
            empty_count = 0
            for obj in objects:
                fobj = factor_tensor(
                    tensor=obj.tensor,
                    piece_id=obj.piece_id,
                    track_id=obj.track_id,
                    start_time=obj.start_time,
                    scale=obj.tensor.shape[0],
                    is_drum=getattr(obj, 'is_drum', False)
                )
                factored.append(fobj)
                if fobj.num_notes == 0:
                    empty_count += 1
    else:
        # Original CPU path
        factored = []
        empty_count = 0
        for obj in objects:
            fobj = factor_tensor(
                tensor=obj.tensor,
                piece_id=obj.piece_id,
                track_id=obj.track_id,
                start_time=obj.start_time,
                scale=obj.tensor.shape[0],
                is_drum=getattr(obj, 'is_drum', False)
            )
            factored.append(fobj)
            if fobj.num_notes == 0:
                empty_count += 1

    if verbose:
        print(f"  Factored objects: {len(factored)}")
        print(f"  Empty objects (no notes): {empty_count}")

        # Stats on components
        note_counts = [f.num_notes for f in factored if f.num_notes > 0]
        if note_counts:
            print(f"  Notes per object: min={min(note_counts)}, max={max(note_counts)}, "
                  f"mean={np.mean(note_counts):.1f}")

        # Unique rhythms and contours
        unique_rhythms = len(set(f.rhythm_hash for f in factored if f.num_notes > 0))
        unique_contours = len(set(f.contour_hash for f in factored if f.num_notes > 0))
        print(f"  Unique rhythm patterns: {unique_rhythms}")
        print(f"  Unique pitch contours: {unique_contours}")

    return factored


# =============================================================================
# COMPONENT-LEVEL TRANSFORMS
# =============================================================================

@dataclass
class ComponentTransform:
    """A transform that operates on a single component."""
    component: str  # 'rhythm', 'pitch', 'contour', 'velocity', 'duration'
    name: str       # transform name
    amount: float   # transform parameter

    def __hash__(self):
        return hash((self.component, self.name, self.amount))

    def __eq__(self, other):
        return (self.component == other.component and
                self.name == other.name and
                self.amount == other.amount)

    def __str__(self):
        if self.amount == 0:
            return f"{self.component}:{self.name}"
        return f"{self.component}:{self.name}({self.amount})"


@dataclass
class FactoredTransform:
    """
    A product transform: T_rhythm × T_pitch × T_velocity × T_duration

    Each component can have its own transform (or identity).
    """
    rhythm_transform: Optional[ComponentTransform] = None
    pitch_transform: Optional[ComponentTransform] = None
    contour_transform: Optional[ComponentTransform] = None
    velocity_transform: Optional[ComponentTransform] = None
    duration_transform: Optional[ComponentTransform] = None

    @property
    def name(self) -> str:
        parts = []
        if self.rhythm_transform:
            parts.append(str(self.rhythm_transform))
        if self.pitch_transform:
            parts.append(str(self.pitch_transform))
        if self.contour_transform:
            parts.append(str(self.contour_transform))
        if self.velocity_transform:
            parts.append(str(self.velocity_transform))
        if self.duration_transform:
            parts.append(str(self.duration_transform))
        return " × ".join(parts) if parts else "identity"

    @property
    def is_identity(self) -> bool:
        return (self.rhythm_transform is None and
                self.pitch_transform is None and
                self.contour_transform is None and
                self.velocity_transform is None and
                self.duration_transform is None)

    @property
    def num_components(self) -> int:
        """Number of non-identity component transforms."""
        count = 0
        if self.rhythm_transform: count += 1
        if self.pitch_transform: count += 1
        if self.contour_transform: count += 1
        if self.velocity_transform: count += 1
        if self.duration_transform: count += 1
        return count

    def __hash__(self):
        return hash((self.rhythm_transform, self.pitch_transform,
                     self.contour_transform, self.velocity_transform,
                     self.duration_transform))

    def __eq__(self, other):
        return (self.rhythm_transform == other.rhythm_transform and
                self.pitch_transform == other.pitch_transform and
                self.contour_transform == other.contour_transform and
                self.velocity_transform == other.velocity_transform and
                self.duration_transform == other.duration_transform)


# =============================================================================
# COMPONENT MATCHING
# =============================================================================

def match_rhythm(source: np.ndarray, target: np.ndarray,
                 max_error: float = 0.1) -> Tuple[bool, Optional[ComponentTransform], float]:
    """
    Check if source rhythm matches target, possibly with a transform.

    Returns:
        (matches, transform, error)
    """
    if len(source) != len(target):
        return False, None, float('inf')

    # Exact match
    error = np.mean(np.abs(source - target))
    if error < max_error:
        return True, None, error

    # Try time_shift
    for shift in [-16, -8, -4, 4, 8, 16]:
        if abs(shift) >= len(source):
            continue
        if shift > 0:
            shifted = np.zeros_like(source)
            shifted[shift:] = source[:-shift]
        else:
            shifted = np.zeros_like(source)
            shifted[:shift] = source[-shift:]

        error = np.mean(np.abs(shifted - target))
        if error < max_error:
            transform = ComponentTransform('rhythm', 'time_shift', shift)
            return True, transform, error

    # Try retrograde
    retrograded = source[::-1]
    error = np.mean(np.abs(retrograded - target))
    if error < max_error:
        transform = ComponentTransform('rhythm', 'retrograde', 0)
        return True, transform, error

    return False, None, float('inf')


def match_pitch_contour(source: np.ndarray, target: np.ndarray,
                        max_error: float = 0.1) -> Tuple[bool, Optional[ComponentTransform], float]:
    """
    Check if source contour matches target, possibly with a transform.

    Contour matching is powerful: same contour with different starting pitch
    means the melodic shape is preserved.
    """
    if len(source) != len(target):
        return False, None, float('inf')

    if len(source) == 0:
        return True, None, 0.0

    # Exact contour match (same intervals)
    error = np.mean(np.abs(source - target))
    if error < max_error:
        return True, None, error

    # Inverted contour (negative intervals)
    inverted = -source
    error = np.mean(np.abs(inverted - target))
    if error < max_error:
        transform = ComponentTransform('contour', 'inversion', 0)
        return True, transform, error

    # Retrograde contour
    retrograded = source[::-1]
    error = np.mean(np.abs(retrograded - target))
    if error < max_error:
        transform = ComponentTransform('contour', 'retrograde', 0)
        return True, transform, error

    # Retrograde inversion
    retro_inv = -source[::-1]
    error = np.mean(np.abs(retro_inv - target))
    if error < max_error:
        transform = ComponentTransform('contour', 'retrograde_inversion', 0)
        return True, transform, error

    return False, None, float('inf')


def match_pitches(source: np.ndarray, target: np.ndarray,
                  max_error: float = 1.0) -> Tuple[bool, Optional[ComponentTransform], float]:
    """
    Check if source pitches match target, possibly transposed.

    Returns:
        (matches, transform, error) where transform is transposition amount
    """
    if len(source) != len(target) or len(source) == 0:
        return False, None, float('inf')

    # Compute transposition (difference between means or first notes)
    transpose = int(np.round(np.mean(target) - np.mean(source)))

    transposed = source + transpose
    error = np.mean(np.abs(transposed - target))

    if error < max_error:
        if transpose == 0:
            return True, None, error
        transform = ComponentTransform('pitch', 'transpose_semitone', transpose)
        return True, transform, error

    return False, None, float('inf')


def match_velocities(source: np.ndarray, target: np.ndarray,
                     max_error: float = 0.2) -> Tuple[bool, Optional[ComponentTransform], float]:
    """
    Check if source velocities match target, possibly scaled.
    """
    if len(source) != len(target) or len(source) == 0:
        return False, None, float('inf')

    # Exact match
    error = np.mean(np.abs(source - target))
    if error < max_error:
        return True, None, error

    # Try to find scale factor
    source_mean = np.mean(source)
    target_mean = np.mean(target)

    if source_mean > 0.01:  # Avoid division by near-zero
        scale = target_mean / source_mean

        # Only consider reasonable scales
        if 0.3 <= scale <= 3.0:
            scaled = np.clip(source * scale, 0, 1)
            error = np.mean(np.abs(scaled - target))

            if error < max_error:
                # Round scale to nearest standard value
                standard_scales = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5]
                scale = min(standard_scales, key=lambda s: abs(s - scale))
                if scale != 1.0:
                    transform = ComponentTransform('velocity', 'velocity_scale', scale)
                    return True, transform, error
                return True, None, error

    return False, None, float('inf')


def match_factored_objects(source: FactoredObject, target: FactoredObject,
                          max_rhythm_error: float = 0.1,
                          max_pitch_error: float = 1.0,
                          max_velocity_error: float = 0.2) -> Tuple[bool, Optional[FactoredTransform], Dict[str, float]]:
    """
    Match two factored objects, finding the product transform if possible.

    Returns:
        (matches, factored_transform, component_errors)
    """
    # Must have same number of notes (for now)
    if source.num_notes != target.num_notes:
        return False, None, {}

    if source.num_notes == 0:
        # Both empty - trivial match
        return True, FactoredTransform(), {'rhythm': 0, 'contour': 0, 'pitch': 0, 'velocity': 0}

    # Match each component independently
    rhythm_match, rhythm_transform, rhythm_error = match_rhythm(
        source.rhythm, target.rhythm, max_rhythm_error)

    if not rhythm_match:
        return False, None, {'rhythm': rhythm_error}

    contour_match, contour_transform, contour_error = match_pitch_contour(
        source.pitch_contour, target.pitch_contour)

    if not contour_match:
        return False, None, {'rhythm': rhythm_error, 'contour': contour_error}

    pitch_match, pitch_transform, pitch_error = match_pitches(
        source.pitches, target.pitches, max_pitch_error)

    if not pitch_match:
        return False, None, {'rhythm': rhythm_error, 'contour': contour_error, 'pitch': pitch_error}

    velocity_match, velocity_transform, velocity_error = match_velocities(
        source.velocities, target.velocities, max_velocity_error)

    if not velocity_match:
        return False, None, {'rhythm': rhythm_error, 'contour': contour_error,
                            'pitch': pitch_error, 'velocity': velocity_error}

    # Build factored transform
    factored = FactoredTransform(
        rhythm_transform=rhythm_transform,
        pitch_transform=pitch_transform,
        contour_transform=contour_transform,
        velocity_transform=velocity_transform
    )

    errors = {
        'rhythm': rhythm_error,
        'contour': contour_error,
        'pitch': pitch_error,
        'velocity': velocity_error
    }

    return True, factored, errors


# =============================================================================
# COMPONENT INDICES FOR FAST MATCHING
# =============================================================================

class ComponentIndex:
    """
    Hash-based index for fast component matching.

    Groups objects by component hash, allowing O(1) lookup of potential matches.
    """
    def __init__(self, objects: List[FactoredObject], component: str):
        """
        Build index for a specific component.

        Args:
            objects: List of FactoredObject
            component: 'rhythm' or 'contour'
        """
        self.component = component
        self.index = defaultdict(list)

        for obj in objects:
            if obj.num_notes == 0:
                continue

            if component == 'rhythm':
                key = obj.rhythm_hash
            elif component == 'contour':
                key = obj.contour_hash
            else:
                raise ValueError(f"Unknown component: {component}")

            self.index[key].append(obj)

    def get_candidates(self, obj: FactoredObject) -> List[FactoredObject]:
        """Get objects with matching component."""
        if obj.num_notes == 0:
            return []

        if self.component == 'rhythm':
            key = obj.rhythm_hash
        elif self.component == 'contour':
            key = obj.contour_hash
        else:
            return []

        return self.index.get(key, [])

    @property
    def num_groups(self) -> int:
        return len(self.index)

    @property
    def largest_group(self) -> int:
        if not self.index:
            return 0
        return max(len(v) for v in self.index.values())


# =============================================================================
# TRANSFORM-AWARE COMPONENT MATCHING
# =============================================================================

def apply_rhythm_transform(rhythm: np.ndarray, transform_name: str, amount: float) -> np.ndarray:
    """Apply a transform to a rhythm pattern."""
    if transform_name == 'identity' or transform_name is None:
        return rhythm.copy()
    elif transform_name == 'retrograde':
        return rhythm[::-1].copy()
    elif transform_name == 'time_shift':
        shift = int(amount)
        result = np.zeros_like(rhythm)
        if shift > 0 and shift < len(rhythm):
            result[shift:] = rhythm[:-shift]
        elif shift < 0 and abs(shift) < len(rhythm):
            result[:shift] = rhythm[-shift:]
        return result
    return rhythm.copy()


def apply_contour_transform(contour: np.ndarray, transform_name: str, amount: float) -> np.ndarray:
    """Apply a transform to a pitch contour."""
    if transform_name == 'identity' or transform_name is None:
        return contour.copy()
    elif transform_name == 'inversion':
        return -contour  # Negate intervals
    elif transform_name == 'retrograde':
        return contour[::-1].copy()
    elif transform_name == 'retrograde_inversion':
        return -contour[::-1]
    elif transform_name == 'transpose':
        # Transposition doesn't change contour (intervals stay same)
        return contour.copy()
    return contour.copy()


class TransformAwareIndex:
    """
    Index that stores both original and transformed versions of components.

    This enables finding matches where source.rhythm transformed = target.rhythm.
    """
    def __init__(self, objects: List['FactoredObject'], component: str,
                 transforms: List[Tuple[str, float]]):
        """
        Build index with transforms.

        Args:
            objects: List of FactoredObject
            component: 'rhythm' or 'contour'
            transforms: List of (transform_name, amount) tuples
        """
        self.component = component
        self.transforms = transforms

        # index[hash] -> list of (object, transform_name, transform_amount)
        self.index = defaultdict(list)

        for obj in objects:
            if obj.num_notes == 0:
                continue

            # Get the component array
            if component == 'rhythm':
                original = obj.rhythm
                apply_fn = apply_rhythm_transform
            elif component == 'contour':
                original = obj.pitch_contour
                apply_fn = apply_contour_transform
            else:
                continue

            # Index original (identity transform)
            key = hash(original.tobytes())
            self.index[key].append((obj, 'identity', 0))

            # Index transformed versions
            for t_name, t_amount in transforms:
                transformed = apply_fn(original, t_name, t_amount)
                key = hash(transformed.tobytes())
                self.index[key].append((obj, t_name, t_amount))

    def find_matches(self, target: 'FactoredObject') -> List[Tuple['FactoredObject', str, float]]:
        """
        Find all sources whose (possibly transformed) component matches target's component.

        Returns:
            List of (source_object, transform_name, transform_amount)
        """
        if target.num_notes == 0:
            return []

        if self.component == 'rhythm':
            target_component = target.rhythm
        elif self.component == 'contour':
            target_component = target.pitch_contour
        else:
            return []

        key = hash(target_component.tobytes())
        matches = self.index.get(key, [])

        # Filter out self-matches
        return [(obj, t_name, t_amt) for obj, t_name, t_amt in matches if obj != target]

    @property
    def num_entries(self) -> int:
        return sum(len(v) for v in self.index.values())


@dataclass
class CrossComponentPattern:
    """A pattern where target derives components from different sources."""
    target: 'FactoredObject'
    rhythm_source: Optional['FactoredObject']
    rhythm_transform: Optional[str]
    rhythm_amount: float
    contour_source: Optional['FactoredObject']
    contour_transform: Optional[str]
    contour_amount: float
    pitch_offset: int  # Transposition from contour source to target

    @property
    def is_cross_component(self) -> bool:
        """True if rhythm and contour come from different sources."""
        return (self.rhythm_source is not None and
                self.contour_source is not None and
                self.rhythm_source != self.contour_source)

    @property
    def description(self) -> str:
        parts = []
        if self.rhythm_source:
            r_trans = f"({self.rhythm_transform})" if self.rhythm_transform != 'identity' else ""
            parts.append(f"rhythm{r_trans}({self.rhythm_source.piece_id}:T{self.rhythm_source.track_id})")
        if self.contour_source:
            c_trans = f"({self.contour_transform})" if self.contour_transform != 'identity' else ""
            parts.append(f"contour{c_trans}({self.contour_source.piece_id}:T{self.contour_source.track_id})")
        if self.pitch_offset != 0:
            parts.append(f"transpose({self.pitch_offset})")
        return " × ".join(parts) if parts else "literal"


@dataclass
class PatternCounts:
    """Statistics from pattern discovery without materializing all patterns."""
    total: int = 0
    cross_component: int = 0
    same_source: int = 0
    rhythm_only: int = 0
    contour_only: int = 0
    examples: List = None  # Sample of CrossComponentPattern for display

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


def find_all_factored_patterns(
    objects: List['FactoredObject'],
    rhythm_transforms: List[Tuple[str, float]],
    contour_transforms: List[Tuple[str, float]],
    verbose: bool = True,
    max_examples: int = 10
) -> PatternCounts:
    """
    Find ALL factored patterns including cross-component derivations.

    OPTIMIZED: Only counts patterns and samples examples instead of
    materializing millions of CrossComponentPattern objects.

    This is the key Lewinian insight: an object can derive its rhythm from
    one source and its contour from another source (with transforms).

    Args:
        objects: List of FactoredObject
        rhythm_transforms: List of (name, amount) for rhythm transforms
        contour_transforms: List of (name, amount) for contour transforms
        verbose: Print progress
        max_examples: Maximum cross-component examples to collect

    Returns:
        PatternCounts with statistics and sample examples
    """
    if verbose:
        print(f"\n{'='*70}")
        print("TRANSFORM-AWARE PATTERN DISCOVERY")
        print(f"{'='*70}")
        print(f"  Objects: {len(objects)}")
        print(f"  Rhythm transforms: {len(rhythm_transforms) + 1}")  # +1 for identity
        print(f"  Contour transforms: {len(contour_transforms) + 1}")

    valid_objects = [o for o in objects if o.num_notes > 0]
    if verbose:
        print(f"  Valid objects: {len(valid_objects)}")

    # Build transform-aware indices
    if verbose:
        print(f"\n  Building transform-aware rhythm index...")
    rhythm_index = TransformAwareIndex(valid_objects, 'rhythm', rhythm_transforms)
    if verbose:
        print(f"    Index entries: {rhythm_index.num_entries}")

    if verbose:
        print(f"  Building transform-aware contour index...")
    contour_index = TransformAwareIndex(valid_objects, 'contour', contour_transforms)
    if verbose:
        print(f"    Index entries: {contour_index.num_entries}")

    # Find all patterns - COUNT ONLY, don't materialize
    if verbose:
        print(f"\n  Finding patterns (count-only mode)...")

    counts = PatternCounts()
    import random

    # For reservoir sampling of cross-component examples
    cross_component_reservoir = []
    seen_cross = 0

    for i, target in enumerate(valid_objects):
        if verbose and i > 0 and i % 1000 == 0:
            print(f"    Processed {i}/{len(valid_objects)} objects, {counts.total} patterns so far...")

        # Find rhythm matches
        rhythm_matches = rhythm_index.find_matches(target)

        # Find contour matches
        contour_matches = contour_index.find_matches(target)

        # Count all valid combinations without materializing
        rhythm_options = rhythm_matches + [(None, 'identity', 0)]
        contour_options = contour_matches + [(None, 'identity', 0)]

        for r_src, r_trans, r_amt in rhythm_options:
            for c_src, c_trans, c_amt in contour_options:
                # Skip if both None (literal, not a derivation)
                if r_src is None and c_src is None:
                    continue

                counts.total += 1

                # Determine pattern type
                is_cross = (r_src is not None and c_src is not None and r_src != c_src)
                is_same = (r_src is not None and c_src is not None and r_src == c_src)
                is_rhythm_only = (r_src is not None and c_src is None)
                is_contour_only = (c_src is not None and r_src is None)

                if is_cross:
                    counts.cross_component += 1
                    seen_cross += 1

                    # Reservoir sampling for examples
                    if len(cross_component_reservoir) < max_examples:
                        # Compute pitch offset for example
                        pitch_offset = 0
                        if len(target.pitches) > 0 and len(c_src.pitches) > 0:
                            pitch_offset = int(target.pitches[0] - c_src.pitches[0])

                        pattern = CrossComponentPattern(
                            target=target,
                            rhythm_source=r_src,
                            rhythm_transform=r_trans,
                            rhythm_amount=r_amt,
                            contour_source=c_src,
                            contour_transform=c_trans,
                            contour_amount=c_amt,
                            pitch_offset=pitch_offset
                        )
                        cross_component_reservoir.append(pattern)
                    else:
                        # Reservoir sampling: replace with probability max_examples/seen_cross
                        j = random.randint(0, seen_cross - 1)
                        if j < max_examples:
                            pitch_offset = 0
                            if len(target.pitches) > 0 and len(c_src.pitches) > 0:
                                pitch_offset = int(target.pitches[0] - c_src.pitches[0])

                            pattern = CrossComponentPattern(
                                target=target,
                                rhythm_source=r_src,
                                rhythm_transform=r_trans,
                                rhythm_amount=r_amt,
                                contour_source=c_src,
                                contour_transform=c_trans,
                                contour_amount=c_amt,
                                pitch_offset=pitch_offset
                            )
                            cross_component_reservoir[j] = pattern

                elif is_same:
                    counts.same_source += 1
                elif is_rhythm_only:
                    counts.rhythm_only += 1
                elif is_contour_only:
                    counts.contour_only += 1

    counts.examples = cross_component_reservoir

    if verbose:
        print(f"    Total patterns found: {counts.total}")
        print(f"    Cross-component patterns: {counts.cross_component}")
        print(f"    Same-source patterns: {counts.same_source}")
        print(f"    Rhythm-only patterns: {counts.rhythm_only}")
        print(f"    Contour-only patterns: {counts.contour_only}")

    return counts


# =============================================================================
# FACTORED MDL DISCOVERY
# =============================================================================

@dataclass
class FactoredMatch:
    """A match between two factored objects via a factored transform."""
    source: FactoredObject
    target: FactoredObject
    transform: FactoredTransform
    errors: Dict[str, float]

    @property
    def total_error(self) -> float:
        return sum(self.errors.values())


@dataclass
class FactoredMDLResult:
    """Result from factored MDL discovery."""
    # Discovered transforms
    vocabulary: Set[FactoredTransform]

    # Assignments: target -> (source, transform)
    assignments: Dict[FactoredObject, Tuple[FactoredObject, FactoredTransform]]

    # Unassigned sources
    sources: Set[FactoredObject]

    # Component-level statistics
    rhythm_groups: int      # Number of unique rhythm patterns
    contour_groups: int     # Number of unique pitch contours

    # Match statistics by component
    component_stats: Dict[str, Dict]

    # Overall statistics
    stats: Dict

    # Reconstruction data (patterns + object assignments)
    reconstruction_data: Dict = None


def discover_factored_patterns(
    objects: List[FactoredObject],
    min_group_size: int = 3,
    verbose: bool = True
) -> FactoredMDLResult:
    """
    Discover patterns using factored representation.

    Phase 1: Group by rhythm (same rhythm = potential matches)
    Phase 2: Within rhythm groups, group by contour
    Phase 3: For each (rhythm, contour) group, find pitch transforms
    Phase 4: MDL selection

    Args:
        objects: List of FactoredObject
        min_group_size: Minimum objects in a group to consider
        verbose: Print progress

    Returns:
        FactoredMDLResult
    """
    if verbose:
        print(f"\n{'='*70}")
        print("FACTORED MDL DISCOVERY")
        print(f"{'='*70}")
        print(f"  Objects: {len(objects)}")
        print(f"  Min group size: {min_group_size}")

    # Filter out empty objects
    valid_objects = [o for o in objects if o.num_notes > 0]
    if verbose:
        print(f"  Valid objects (with notes): {len(valid_objects)}")

    # Phase 1: Build rhythm index
    if verbose:
        print(f"\n  Phase 1: Building rhythm index...")

    rhythm_index = ComponentIndex(valid_objects, 'rhythm')

    if verbose:
        print(f"    Unique rhythm patterns: {rhythm_index.num_groups}")
        print(f"    Largest rhythm group: {rhythm_index.largest_group} objects")

    # Phase 2: Build contour index
    if verbose:
        print(f"\n  Phase 2: Building contour index...")

    contour_index = ComponentIndex(valid_objects, 'contour')

    if verbose:
        print(f"    Unique contours: {contour_index.num_groups}")
        print(f"    Largest contour group: {contour_index.largest_group} objects")

    # Phase 3: Find matches within rhythm groups
    if verbose:
        print(f"\n  Phase 3: Finding factored matches...")

    # Group by (rhythm_hash, contour_hash) for efficiency
    combined_groups = defaultdict(list)
    for obj in valid_objects:
        key = (obj.rhythm_hash, obj.contour_hash, obj.num_notes)
        combined_groups[key].append(obj)

    # Find transforms within each group
    all_matches = []
    transform_counts = Counter()

    for (r_hash, c_hash, n_notes), group in combined_groups.items():
        if len(group) < min_group_size:
            continue

        # All objects in group have same rhythm and contour
        # They can only differ in: absolute pitch, velocity, duration

        # Use first object as representative source
        source = group[0]

        for target in group[1:]:
            matches, transform, errors = match_factored_objects(source, target)
            if matches:
                all_matches.append(FactoredMatch(source, target, transform, errors))
                transform_counts[transform.name] += 1

    if verbose:
        print(f"    Found {len(all_matches)} factored matches")
        print(f"    Unique transforms: {len(transform_counts)}")
        if transform_counts:
            print(f"\n    Top 10 transforms:")
            for name, count in transform_counts.most_common(10):
                print(f"      {name}: {count}")

    # Phase 4: Build assignments (greedy covering)
    if verbose:
        print(f"\n  Phase 4: MDL assignment...")

    vocabulary = set()
    assignments = {}
    assigned_targets = set()

    # Sort matches by transform frequency (most common first)
    transform_to_matches = defaultdict(list)
    for match in all_matches:
        transform_to_matches[match.transform].append(match)

    sorted_transforms = sorted(
        transform_to_matches.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    for transform, matches in sorted_transforms:
        if len(matches) < min_group_size:
            continue

        # Count how many new targets this transform covers
        new_targets = [m for m in matches if m.target not in assigned_targets]

        if len(new_targets) >= min_group_size:
            vocabulary.add(transform)

            for match in new_targets:
                assignments[match.target] = (match.source, match.transform)
                assigned_targets.add(match.target)

    sources = {obj for obj in valid_objects if obj not in assigned_targets}

    # Compute statistics
    component_stats = {
        'rhythm': {
            'unique_patterns': rhythm_index.num_groups,
            'largest_group': rhythm_index.largest_group,
        },
        'contour': {
            'unique_patterns': contour_index.num_groups,
            'largest_group': contour_index.largest_group,
        }
    }

    stats = {
        'total_objects': len(objects),
        'valid_objects': len(valid_objects),
        'total_assigned': len(assignments),
        'total_sources': len(sources),
        'vocabulary_size': len(vocabulary),
        'derivation_rate': len(assignments) / len(valid_objects) if valid_objects else 0,
        'combined_groups': len(combined_groups),
    }

    if verbose:
        print(f"\n{'='*70}")
        print("FACTORED MDL COMPLETE")
        print(f"{'='*70}")
        print(f"  Vocabulary size: {len(vocabulary)}")
        print(f"  Objects assigned: {len(assignments)} ({stats['derivation_rate']*100:.1f}%)")
        print(f"  Sources remaining: {len(sources)}")
        print(f"\n  Component reuse:")
        print(f"    Rhythm patterns reused: {len(valid_objects) - rhythm_index.num_groups}")
        print(f"    Contour patterns reused: {len(valid_objects) - contour_index.num_groups}")

    return FactoredMDLResult(
        vocabulary=vocabulary,
        assignments=assignments,
        sources=sources,
        rhythm_groups=rhythm_index.num_groups,
        contour_groups=contour_index.num_groups,
        component_stats=component_stats,
        stats=stats
    )


# =============================================================================
# CROSS-COMPONENT DISCOVERY (Advanced)
# =============================================================================

def discover_cross_component_patterns(
    objects: List[FactoredObject],
    verbose: bool = True
) -> Dict[str, List]:
    """
    Discover patterns where components are mixed from different sources.

    Example: "Sax_B has Trumpet_A's rhythm but Trombone_A's pitches"

    This is the key to finding hierarchical musical structure.
    """
    if verbose:
        print(f"\n{'='*70}")
        print("CROSS-COMPONENT PATTERN DISCOVERY")
        print(f"{'='*70}")

    # Build component registries
    rhythm_registry = {}  # rhythm_hash -> canonical object
    contour_registry = {}  # contour_hash -> canonical object

    for obj in objects:
        if obj.num_notes == 0:
            continue

        if obj.rhythm_hash not in rhythm_registry:
            rhythm_registry[obj.rhythm_hash] = obj

        if obj.contour_hash not in contour_registry:
            contour_registry[obj.contour_hash] = obj

    # Find objects that can be described as combinations
    cross_patterns = []

    for obj in objects:
        if obj.num_notes == 0:
            continue

        rhythm_source = rhythm_registry.get(obj.rhythm_hash)
        contour_source = contour_registry.get(obj.contour_hash)

        # If rhythm and contour come from different canonical sources
        if (rhythm_source is not None and contour_source is not None and
            rhythm_source != contour_source and
            rhythm_source != obj and contour_source != obj):

            cross_patterns.append({
                'target': obj,
                'rhythm_from': rhythm_source,
                'contour_from': contour_source
            })

    if verbose:
        print(f"  Cross-component patterns found: {len(cross_patterns)}")
        if cross_patterns:
            print(f"\n  Examples:")
            for pattern in cross_patterns[:5]:
                print(f"    {pattern['target'].piece_id}:T{pattern['target'].track_id} = "
                      f"rhythm({pattern['rhythm_from'].piece_id}:T{pattern['rhythm_from'].track_id}) × "
                      f"contour({pattern['contour_from'].piece_id}:T{pattern['contour_from'].track_id})")

    return {'cross_patterns': cross_patterns}


# =============================================================================
# FEATURE 1: EXPLICIT BIT COUNTING FOR COMPRESSION
# =============================================================================

@dataclass
class CompressionStats:
    """Compression statistics for a component or full system."""
    literal_bits: float      # Bits needed without compression
    compressed_bits: float   # Bits needed with factored representation
    compression_ratio: float # literal / compressed

    def __str__(self):
        return f"{self.compression_ratio:.2f}x ({self.literal_bits:.0f} -> {self.compressed_bits:.0f} bits)"


def compute_component_compression(
    objects: List[FactoredObject],
    component: str,
    verbose: bool = False
) -> CompressionStats:
    """
    Compute compression ratio for a single component.

    Literal encoding: Each object stores its own component
    Factored encoding: Store unique patterns + index per object

    Args:
        objects: List of FactoredObject
        component: 'rhythm', 'contour', 'pitches', 'velocities', 'durations'
        verbose: Print details

    Returns:
        CompressionStats
    """
    valid_objects = [o for o in objects if o.num_notes > 0]
    n_objects = len(valid_objects)

    if n_objects == 0:
        return CompressionStats(0, 0, 1.0)

    # Get component data and compute bits per value
    if component == 'rhythm':
        # Binary pattern, 1 bit per timestep
        components = [o.rhythm for o in valid_objects]
        bits_per_value = 1.0
        total_values = sum(len(c) for c in components)
    elif component == 'contour':
        # Intervals typically -24 to +24, need ~6 bits
        components = [o.pitch_contour for o in valid_objects]
        bits_per_value = 6.0
        total_values = sum(len(c) for c in components)
    elif component == 'pitches':
        # MIDI pitches 0-127, need 7 bits
        components = [o.pitches for o in valid_objects]
        bits_per_value = 7.0
        total_values = sum(len(c) for c in components)
    elif component == 'velocities':
        # Velocities 0-1 quantized to 32 levels, 5 bits
        components = [o.velocities for o in valid_objects]
        bits_per_value = 5.0
        total_values = sum(len(c) for c in components)
    elif component == 'durations':
        # Durations typically 1-256, need 8 bits
        components = [o.durations for o in valid_objects]
        bits_per_value = 8.0
        total_values = sum(len(c) for c in components)
    else:
        raise ValueError(f"Unknown component: {component}")

    # Literal bits: store every component value
    literal_bits = total_values * bits_per_value

    # Find unique patterns
    pattern_hashes = {}
    for i, comp in enumerate(components):
        h = hash(comp.tobytes())
        if h not in pattern_hashes:
            pattern_hashes[h] = comp

    n_unique = len(pattern_hashes)

    # Factored bits:
    # 1. Store each unique pattern: n_unique * avg_pattern_size * bits_per_value
    # 2. Store index per object: n_objects * log2(n_unique)
    avg_pattern_size = total_values / n_objects if n_objects > 0 else 0
    pattern_bits = n_unique * avg_pattern_size * bits_per_value
    index_bits = n_objects * math.log2(max(n_unique, 1)) if n_unique > 0 else 0
    compressed_bits = pattern_bits + index_bits

    compression_ratio = literal_bits / compressed_bits if compressed_bits > 0 else 1.0

    if verbose:
        print(f"  {component}:")
        print(f"    Objects: {n_objects}, Unique patterns: {n_unique}")
        print(f"    Literal: {literal_bits:.0f} bits")
        print(f"    Compressed: {compressed_bits:.0f} bits (patterns: {pattern_bits:.0f}, indices: {index_bits:.0f})")
        print(f"    Ratio: {compression_ratio:.2f}x")

    return CompressionStats(literal_bits, compressed_bits, compression_ratio)


def compute_full_compression_stats(
    objects: List[FactoredObject],
    verbose: bool = True
) -> Dict[str, CompressionStats]:
    """
    Compute compression ratios for all components and total.

    Returns:
        Dict mapping component name to CompressionStats
    """
    if verbose:
        print(f"\n{'='*70}")
        print("COMPRESSION ANALYSIS (Explicit Bit Counting)")
        print(f"{'='*70}")

    stats = {}
    total_literal = 0
    total_compressed = 0

    for component in ['rhythm', 'contour', 'pitches', 'velocities', 'durations']:
        cs = compute_component_compression(objects, component, verbose)
        stats[component] = cs
        total_literal += cs.literal_bits
        total_compressed += cs.compressed_bits

    # Total compression
    total_ratio = total_literal / total_compressed if total_compressed > 0 else 1.0
    stats['total'] = CompressionStats(total_literal, total_compressed, total_ratio)

    if verbose:
        print(f"\n  TOTAL:")
        print(f"    Literal: {total_literal:.0f} bits")
        print(f"    Factored: {total_compressed:.0f} bits")
        print(f"    Compression ratio: {total_ratio:.2f}x")

    return stats


# =============================================================================
# FEATURE 2: WITHIN-COMPONENT TRANSFORM DISCOVERY
# =============================================================================

def find_within_component_transforms(
    objects: List[FactoredObject],
    component: str,
    transforms: List[Tuple[str, float]],
    verbose: bool = True,
    use_gpu: bool = True
) -> Dict[str, any]:
    """
    After finding unique patterns, check if any pattern is a transform of another.

    This reduces the vocabulary further by recognizing that patterns are
    related by transforms (retrograde, inversion, etc.)

    GPU-OPTIMIZED: Uses batch tensor operations for O(N²) pairwise comparisons.

    Args:
        objects: List of FactoredObject
        component: 'rhythm' or 'contour'
        transforms: List of (transform_name, amount) to check
        verbose: Print details
        use_gpu: Use GPU acceleration if available (default True)

    Returns:
        Dict with discovered transform relationships
    """
    valid_objects = [o for o in objects if o.num_notes > 0]

    # Get unique patterns
    if component == 'rhythm':
        pattern_map = {}  # hash -> (pattern, representative_object)
        for obj in valid_objects:
            h = obj.rhythm_hash
            if h not in pattern_map:
                pattern_map[h] = (obj.rhythm, obj)
        apply_fn = apply_rhythm_transform
    elif component == 'contour':
        pattern_map = {}
        for obj in valid_objects:
            h = obj.contour_hash
            if h not in pattern_map:
                pattern_map[h] = (obj.pitch_contour, obj)
        apply_fn = apply_contour_transform
    else:
        return {}

    patterns = list(pattern_map.values())
    n_patterns = len(patterns)

    if verbose:
        print(f"\n  Within-{component} transform discovery:")
        print(f"    Unique patterns: {n_patterns}")
        import sys
        sys.stdout.flush()

    # For small pattern counts, use simple CPU implementation
    if n_patterns < 100 or not use_gpu:
        return _find_within_component_transforms_cpu(
            patterns, component, transforms, apply_fn, verbose
        )

    # GPU-accelerated implementation for large pattern counts
    return _find_within_component_transforms_gpu(
        patterns, component, transforms, apply_fn, verbose
    )


def _find_within_component_transforms_cpu(
    patterns: List[Tuple[np.ndarray, any]],
    component: str,
    transforms: List[Tuple[str, float]],
    apply_fn,
    verbose: bool
) -> Dict[str, any]:
    """CPU implementation for small pattern counts."""
    n_patterns = len(patterns)

    # Check each pair for transform relationships
    transform_relations = []
    canonical_patterns = set(range(n_patterns))
    derived_patterns = {}

    for i, (pattern_i, obj_i) in enumerate(patterns):
        if i in derived_patterns:
            continue

        for j, (pattern_j, obj_j) in enumerate(patterns):
            if i >= j:
                continue
            if j in derived_patterns:
                continue

            for t_name, t_amount in transforms:
                transformed = apply_fn(pattern_i, t_name, t_amount)

                if len(transformed) == len(pattern_j):
                    if np.allclose(transformed, pattern_j, atol=0.01):
                        transform_relations.append((i, j, t_name, t_amount))
                        derived_patterns[j] = (i, t_name, t_amount)
                        if j in canonical_patterns:
                            canonical_patterns.remove(j)
                        break

    return _finalize_transform_results(
        patterns, n_patterns, canonical_patterns, derived_patterns,
        transform_relations, component, verbose
    )


def _apply_transform_batch_gpu(
    patterns_gpu: 'torch.Tensor',
    t_name: str,
    t_amount: float,
    component: str
) -> 'torch.Tensor':
    """
    Apply transform to batch of patterns entirely on GPU.

    Args:
        patterns_gpu: [N, L] tensor on GPU
        t_name: Transform name
        t_amount: Transform amount
        component: 'rhythm' or 'contour'

    Returns:
        transformed: [N, L] tensor on GPU
    """
    import torch

    if t_name == 'identity':
        return patterns_gpu.clone()

    elif t_name == 'retrograde':
        # Reverse time dimension
        return torch.flip(patterns_gpu, dims=[1])

    elif t_name == 'time_scale' and component == 'rhythm':
        # Time scaling via interpolation
        import torch.nn.functional as F
        N, L = patterns_gpu.shape
        new_L = int(L * t_amount)
        if new_L == L:
            return patterns_gpu.clone()
        # [N, L] -> [N, 1, L] for interpolate -> [N, 1, new_L] -> [N, new_L]
        patterns_3d = patterns_gpu.unsqueeze(1)
        scaled = F.interpolate(patterns_3d, size=new_L, mode='nearest')
        # Pad or truncate to original length
        if new_L < L:
            result = torch.zeros_like(patterns_gpu)
            result[:, :new_L] = scaled.squeeze(1)
        else:
            result = scaled.squeeze(1)[:, :L]
        return result

    elif t_name == 'inversion' and component == 'contour':
        # Invert around axis (negate intervals for contour)
        return -patterns_gpu

    elif t_name == 'transpose' and component == 'contour':
        # Add constant to all values
        return patterns_gpu + t_amount

    elif t_name == 'augmentation' and component == 'rhythm':
        # Double durations (stretch by 2x)
        import torch.nn.functional as F
        N, L = patterns_gpu.shape
        patterns_3d = patterns_gpu.unsqueeze(1)
        scaled = F.interpolate(patterns_3d, scale_factor=2.0, mode='nearest')
        return scaled.squeeze(1)[:, :L]  # Truncate to original length

    elif t_name == 'diminution' and component == 'rhythm':
        # Halve durations (compress by 0.5x)
        import torch.nn.functional as F
        N, L = patterns_gpu.shape
        patterns_3d = patterns_gpu.unsqueeze(1)
        scaled = F.interpolate(patterns_3d, scale_factor=0.5, mode='nearest')
        # Pad to original length
        result = torch.zeros_like(patterns_gpu)
        new_L = scaled.shape[2]
        result[:, :new_L] = scaled.squeeze(1)
        return result

    else:
        # Unknown transform - return unchanged (will not match anything)
        return patterns_gpu.clone()


def _find_within_component_transforms_gpu(
    patterns: List[Tuple[np.ndarray, any]],
    component: str,
    transforms: List[Tuple[str, float]],
    apply_fn,
    verbose: bool
) -> Dict[str, any]:
    """
    GPU-accelerated within-component transform discovery.

    FULLY VECTORIZED: All operations on GPU including transforms.

    Strategy:
    1. Group patterns by length (transforms preserve length for most operations)
    2. Stack patterns into GPU tensors
    3. Apply transforms directly on GPU (batched)
    4. Compute pairwise distances on GPU
    5. Extract matches
    """
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        if verbose:
            print("    [!] PyTorch not available, falling back to CPU")
        return _find_within_component_transforms_cpu(
            patterns, component, transforms, apply_fn, verbose
        )

    n_patterns = len(patterns)

    if verbose:
        print(f"    Using GPU acceleration ({device})")
        import sys
        sys.stdout.flush()

    # Group patterns by length
    by_length = defaultdict(list)
    for idx, (pattern, obj) in enumerate(patterns):
        by_length[len(pattern)].append((idx, pattern))

    transform_relations = []
    canonical_patterns = set(range(n_patterns))
    derived_patterns = {}

    # Process each length group
    for length, group_patterns in by_length.items():
        if len(group_patterns) < 2:
            continue

        # Stack patterns into tensor: [N, L]
        indices = [idx for idx, _ in group_patterns]
        patterns_np = np.array([p for _, p in group_patterns], dtype=np.float32)
        N = len(patterns_np)

        # Estimate memory for full batch: N * N * L * 4 bytes
        # Limit to ~4GB per chunk to avoid OOM
        max_mem_bytes = 4 * 1024**3
        elements_per_pair = length
        bytes_per_pair = elements_per_pair * 4  # float32
        max_pairs = max_mem_bytes // bytes_per_pair

        # Chunk size: process chunk_size rows at a time against all N columns
        chunk_size = max(1, max_pairs // N)
        chunk_size = min(chunk_size, N)  # Don't exceed group size

        # Move to GPU once
        patterns_gpu = torch.tensor(patterns_np, device=device)

        # For each transform, compute transformed versions and check matches
        for t_name, t_amount in transforms:
            # Apply transform ENTIRELY ON GPU (no CPU loop!)
            transformed_gpu = _apply_transform_batch_gpu(
                patterns_gpu, t_name, t_amount, component
            )

            # Ensure same shape after transform
            if transformed_gpu.shape != patterns_gpu.shape:
                continue

            # Process in chunks to avoid OOM on large pattern sets
            for chunk_start in range(0, N, chunk_size):
                chunk_end = min(chunk_start + chunk_size, N)

                # Chunk of source patterns
                patterns_chunk = patterns_gpu[chunk_start:chunk_end]  # [chunk_n, L]

                # Compute pairwise distances: [chunk_n, N]
                # patterns_chunk[i] vs transformed_gpu[j] -> does pattern_i match transform(pattern_j)?
                # We want: original[i] == transform(original[j])
                # So: patterns_chunk[i] - transformed_gpu[j]
                diffs = patterns_chunk.unsqueeze(1) - transformed_gpu.unsqueeze(0)  # [chunk_n, N, L]
                distances = (diffs ** 2).mean(dim=-1)  # [chunk_n, N] MSE

                # Find matches (MSE < threshold)
                threshold = 0.0001  # ~= atol=0.01 squared
                matches = (distances < threshold)

                # Exclude self-matches and ensure i < j (upper triangle logic)
                # For chunked processing: only keep matches where global_i < global_j
                chunk_n = chunk_end - chunk_start
                for local_i in range(chunk_n):
                    global_i = chunk_start + local_i
                    # Zero out matches where j <= i (keep only j > i)
                    matches[local_i, :global_i + 1] = False

                # Extract match indices
                match_indices = torch.nonzero(matches, as_tuple=False).cpu().numpy()

                for i_local, j_local in match_indices:
                    i_global = indices[chunk_start + i_local]
                    j_global = indices[j_local]

                    # Skip if already derived
                    if j_global in derived_patterns:
                        continue

                    transform_relations.append((i_global, j_global, t_name, t_amount))
                    derived_patterns[j_global] = (i_global, t_name, t_amount)
                    if j_global in canonical_patterns:
                        canonical_patterns.remove(j_global)

                # Clear GPU cache between chunks
                torch.cuda.empty_cache()

    return _finalize_transform_results(
        patterns, n_patterns, canonical_patterns, derived_patterns,
        transform_relations, component, verbose
    )


def _finalize_transform_results(
    patterns, n_patterns, canonical_patterns, derived_patterns,
    transform_relations, component, verbose
) -> Dict[str, any]:
    """Common finalization for both CPU and GPU implementations."""
    n_canonical = len(canonical_patterns)
    n_derived = len(derived_patterns)

    if verbose:
        print(f"    Canonical patterns (irreducible): {n_canonical}")
        print(f"    Derived patterns: {n_derived}")
        if transform_relations:
            print(f"    Transform relationships found:")
            for src, tgt, t_name, t_amt in transform_relations[:10]:
                print(f"      Pattern {tgt} = {t_name}({t_amt}) of Pattern {src}")
        import sys
        sys.stdout.flush()

    # Compute additional compression from transforms
    avg_pattern_size = np.mean([len(p) for p, _ in patterns]) if patterns else 0
    bits_per_value = 1.0 if component == 'rhythm' else 6.0
    transform_bits = 4.0

    without_transforms = n_patterns * avg_pattern_size * bits_per_value
    with_transforms = (n_canonical * avg_pattern_size * bits_per_value +
                       n_derived * (math.log2(max(n_canonical, 1)) + transform_bits))

    transform_compression = without_transforms / with_transforms if with_transforms > 0 else 1.0

    if verbose:
        print(f"    Additional compression from transforms: {transform_compression:.2f}x")

    return {
        'component': component,
        'total_patterns': n_patterns,
        'canonical_patterns': n_canonical,
        'derived_patterns': n_derived,
        'transform_relations': transform_relations,
        'compression_from_transforms': transform_compression
    }


# =============================================================================
# FEATURE 3: FACTORED VS ATOMIC COMPRESSION COMPARISON
# =============================================================================

def compute_atomic_compression(
    objects: List[FactoredObject],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compute compression for atomic (fused tensor) representation.

    Atomic approach: Each object is a single tensor, find exact/near matches.

    Args:
        objects: List of FactoredObject (with original_tensor)
        verbose: Print details

    Returns:
        Dict with atomic compression stats
    """
    valid_objects = [o for o in objects if o.num_notes > 0 and o.original_tensor is not None]
    n_objects = len(valid_objects)

    if verbose:
        print(f"\n{'='*70}")
        print("ATOMIC (TENSOR) COMPRESSION")
        print(f"{'='*70}")
        print(f"  Objects with tensors: {n_objects}")

    if n_objects == 0:
        return {'literal_bits': 0, 'compressed_bits': 0, 'compression_ratio': 1.0}

    # Compute tensor statistics
    tensor_shapes = [o.original_tensor.shape for o in valid_objects]
    total_elements = sum(np.prod(s) for s in tensor_shapes)

    # Bits per element: piano roll is mostly 0/1 for pitches + velocity
    # Assume ~2 bits per element on average (sparse binary + velocity)
    bits_per_element = 2.0
    literal_bits = total_elements * bits_per_element

    # Find unique tensors by hash
    tensor_hashes = {}
    for obj in valid_objects:
        # Quantize for matching
        quantized = (obj.original_tensor * 100).astype(np.int8)
        h = hash(quantized.tobytes())
        if h not in tensor_hashes:
            tensor_hashes[h] = obj.original_tensor

    n_unique = len(tensor_hashes)

    # Compressed: store unique tensors + index per object
    avg_tensor_size = total_elements / n_objects
    tensor_bits = n_unique * avg_tensor_size * bits_per_element
    index_bits = n_objects * math.log2(max(n_unique, 1))
    compressed_bits = tensor_bits + index_bits

    compression_ratio = literal_bits / compressed_bits if compressed_bits > 0 else 1.0

    if verbose:
        print(f"  Unique tensors: {n_unique} / {n_objects}")
        print(f"  Literal bits: {literal_bits:.0f}")
        print(f"  Compressed bits: {compressed_bits:.0f}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")

    return {
        'n_objects': n_objects,
        'n_unique_tensors': n_unique,
        'literal_bits': literal_bits,
        'compressed_bits': compressed_bits,
        'compression_ratio': compression_ratio
    }


def compare_factored_vs_atomic(
    objects: List[FactoredObject],
    verbose: bool = True
) -> Dict[str, any]:
    """
    Compare total description length: factored vs atomic approaches.

    Returns:
        Dict with comparison results
    """
    if verbose:
        print(f"\n{'='*70}")
        print("FACTORED vs ATOMIC COMPARISON")
        print(f"{'='*70}")

    # Factored compression (all components)
    factored_stats = compute_full_compression_stats(objects, verbose=False)
    factored_bits = factored_stats['total'].compressed_bits
    factored_literal = factored_stats['total'].literal_bits

    # Atomic compression
    atomic_stats = compute_atomic_compression(objects, verbose=False)
    atomic_bits = atomic_stats['compressed_bits']
    atomic_literal = atomic_stats['literal_bits']

    # Compare
    factored_vs_atomic = atomic_bits / factored_bits if factored_bits > 0 else 1.0

    if verbose:
        print(f"\n  FACTORED APPROACH:")
        print(f"    Literal: {factored_literal:.0f} bits")
        print(f"    Compressed: {factored_bits:.0f} bits")
        print(f"    Ratio: {factored_stats['total'].compression_ratio:.2f}x")

        print(f"\n  ATOMIC APPROACH:")
        print(f"    Literal: {atomic_literal:.0f} bits")
        print(f"    Compressed: {atomic_bits:.0f} bits")
        print(f"    Ratio: {atomic_stats['compression_ratio']:.2f}x")

        print(f"\n  WINNER: {'FACTORED' if factored_bits < atomic_bits else 'ATOMIC'}")
        print(f"    Factored is {factored_vs_atomic:.2f}x {'better' if factored_vs_atomic > 1 else 'worse'} than atomic")

        # Show per-component breakdown
        print(f"\n  Per-component factored compression:")
        for comp in ['rhythm', 'contour', 'pitches', 'velocities', 'durations']:
            cs = factored_stats[comp]
            print(f"    {comp}: {cs.compression_ratio:.2f}x")

    return {
        'factored': {
            'literal_bits': factored_literal,
            'compressed_bits': factored_bits,
            'compression_ratio': factored_stats['total'].compression_ratio,
            'per_component': {k: v.compression_ratio for k, v in factored_stats.items()}
        },
        'atomic': atomic_stats,
        'factored_vs_atomic_ratio': factored_vs_atomic,
        'winner': 'factored' if factored_bits < atomic_bits else 'atomic'
    }


# =============================================================================
# HIERARCHICAL PATTERN DISCOVERY
# =============================================================================

@dataclass
class CompositePattern:
    """A pattern that can be expressed as composition of smaller patterns."""
    pattern_id: int
    pattern: np.ndarray
    composition_type: str  # 'repeat', 'concat', 'time_scale', 'literal'
    children: List[int]    # pattern IDs of children (for concat/repeat)
    param: float = 1.0     # e.g., repeat count or scale factor
    bits_literal: float = 0.0
    bits_composite: float = 0.0

    @property
    def savings(self) -> float:
        return self.bits_literal - self.bits_composite


def discover_hierarchical_patterns(
    patterns: Dict[int, np.ndarray],
    component: str = 'rhythm',
    verbose: bool = True,
    prior_compositions: Optional[Dict[int, 'CompositePattern']] = None
) -> Dict[str, any]:
    """
    Discover hierarchical structure in patterns using compositional operations.

    Operations detected:
    - repeat(P, n): P repeated n times (P + P + ... + P)
    - concat(P1, P2): P1 followed by P2
    - concat_transform(P1, transform(P2)): Lewinian transformed concat
    - time_scale(P, s): P stretched/compressed by factor s

    Implementation:
    - Hash-based substring lookup for O(1) prefix/suffix matching
    - Process patterns by length (shortest first) so children exist before parents
    - MDL criterion: only accept decomposition if bits(composition) < bits(literal)

    Args:
        patterns: Dict of pattern_id -> numpy array
        component: 'rhythm' or 'contour' (affects bit counting)
        verbose: Print progress

    Returns:
        Dict with hierarchical structure and compression stats
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"HIERARCHICAL PATTERN DISCOVERY ({component})")
        print(f"{'='*70}")
        print(f"  Input patterns: {len(patterns)}")

    if len(patterns) == 0:
        return {'primitives': 0, 'composites': 0, 'tree_depth': 0}

    # Sort patterns by length (shortest first)
    sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[1]))

    # Build hash index for fast substring lookup
    # Key: (length, hash of pattern bytes) -> list of pattern IDs
    pattern_index: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for pid, p in patterns.items():
        key = (len(p), hash(p.tobytes()))
        pattern_index[key].append(pid)

    # Also build prefix/suffix index for concat detection
    # prefix_index[hash(first_half)] = [(pid, full_pattern), ...]
    prefix_index: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    suffix_index: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)

    for pid, p in patterns.items():
        if len(p) >= 4:  # Only index patterns long enough to split
            for split_point in [len(p)//4, len(p)//3, len(p)//2, 2*len(p)//3, 3*len(p)//4]:
                if split_point > 0 and split_point < len(p):
                    prefix_hash = hash(p[:split_point].tobytes())
                    suffix_hash = hash(p[split_point:].tobytes())
                    prefix_index[prefix_hash].append((pid, p, split_point))
                    suffix_index[suffix_hash].append((pid, p, split_point))

    # Bits per element (for MDL criterion)
    bits_per_elem = 1.0 if component == 'rhythm' else 4.0  # rhythm is binary, contour needs more bits
    overhead_bits = 8.0  # Cost of storing a composition instruction

    # Result tracking
    composites: List[CompositePattern] = []
    primitives = set()
    pattern_to_composition: Dict[int, CompositePattern] = {}

    # Initialize with prior compositions if provided (for iterative deepening)
    if prior_compositions:
        pattern_to_composition.update(prior_compositions)

    # Track which patterns are used as children (to build tree)
    child_usage_count: Dict[int, int] = defaultdict(int)

    # ==========================================================================
    # GPU-ACCELERATED CONCAT PRECOMPUTATION (using torch.cdist)
    # ==========================================================================
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(f"  Precomputing concat matches (GPU: {device})...")
        import sys
        sys.stdout.flush()

    # Group patterns by length and upload to GPU once
    patterns_by_length: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for pid, p in patterns.items():
        patterns_by_length[len(p)].append((pid, p))

    # Pre-stack all length groups into GPU tensors
    length_tensors: Dict[int, Tuple[List[int], torch.Tensor]] = {}
    for length, items in patterns_by_length.items():
        if len(items) > 0:
            pids = [p[0] for p in items]
            tensor = torch.tensor(
                np.array([p[1] for p in items]),
                dtype=torch.float32, device=device
            )
            length_tensors[length] = (pids, tensor)

    # Precompute all concat matches: concat_matches[target_pid] = (left_pid, right_pid)
    concat_matches: Dict[int, Tuple[int, int]] = {}

    unique_lengths = sorted(length_tensors.keys())
    threshold = 0.01  # sqrt(0.0001) for cdist euclidean

    for target_len in unique_lengths:
        if target_len < 4:
            continue

        target_pids, target_tensor = length_tensors[target_len]  # [N, target_len]
        N = len(target_pids)

        # Find all (left_len, right_len) pairs that sum to target_len
        for left_len in unique_lengths:
            right_len = target_len - left_len
            if right_len not in length_tensors or left_len < 2 or right_len < 2:
                continue

            left_pids, left_tensor = length_tensors[left_len]    # [M1, left_len]
            right_pids, right_tensor = length_tensors[right_len]  # [M2, right_len]
            M1, M2 = len(left_pids), len(right_pids)

            # Extract splits from all targets
            target_lefts = target_tensor[:, :left_len]   # [N, left_len]
            target_rights = target_tensor[:, left_len:]  # [N, right_len]

            # Use torch.cdist for vectorized distance - chunk if memory constrained
            max_mem = 2 * 1024**3  # 2GB
            chunk_size = max(1, max_mem // (max(M1, M2) * max(left_len, right_len) * 4))
            chunk_size = min(chunk_size, N)

            for chunk_start in range(0, N, chunk_size):
                chunk_end = min(chunk_start + chunk_size, N)
                chunk_lefts = target_lefts[chunk_start:chunk_end]
                chunk_rights = target_rights[chunk_start:chunk_end]

                # torch.cdist computes pairwise L2 distance: [chunk_n, M]
                left_dists = torch.cdist(chunk_lefts, left_tensor)    # [chunk_n, M1]
                right_dists = torch.cdist(chunk_rights, right_tensor)  # [chunk_n, M2]

                # Normalize by sqrt(length) for comparison with MSE threshold
                left_dists = left_dists / np.sqrt(left_len)
                right_dists = right_dists / np.sqrt(right_len)

                # Find matches using sparse extraction
                left_matches = (left_dists < threshold).nonzero(as_tuple=False)   # [K1, 2]
                right_matches = (right_dists < threshold).nonzero(as_tuple=False)  # [K2, 2]

                # Build match lookup: target_idx -> list of matching pattern indices
                left_match_dict = defaultdict(list)
                for row, col in left_matches.cpu().numpy():
                    left_match_dict[row].append(col)

                right_match_dict = defaultdict(list)
                for row, col in right_matches.cpu().numpy():
                    right_match_dict[row].append(col)

                # Cross-reference matches
                for i in range(chunk_end - chunk_start):
                    target_idx = chunk_start + i
                    target_pid = target_pids[target_idx]

                    if target_pid in concat_matches:
                        continue

                    if i in left_match_dict and i in right_match_dict:
                        # Take first valid (different) pair
                        for left_idx in left_match_dict[i]:
                            left_pid = left_pids[left_idx]
                            for right_idx in right_match_dict[i]:
                                right_pid = right_pids[right_idx]
                                if left_pid != right_pid:
                                    concat_matches[target_pid] = (left_pid, right_pid)
                                    break
                            if target_pid in concat_matches:
                                break

                del left_dists, right_dists
                torch.cuda.empty_cache()

    if verbose:
        print(f"    Found {len(concat_matches)} concat relationships")
        sys.stdout.flush()

    # ==========================================================================
    # GPU-ACCELERATED CONCAT-WITH-TRANSFORM PRECOMPUTATION (Lewinian)
    # ==========================================================================
    # Finds patterns like: A + transpose(A, k), A + retrograde(A), A + invert(A)
    # Optimized: vectorized checks without Python loops over patterns

    concat_transform_matches: Dict[int, Tuple[int, int, str, float]] = {}

    if verbose:
        print(f"  Precomputing concat-transform matches (GPU)...")
        sys.stdout.flush()

    # Focus on equal-half splits (most common for transformed repeats)
    for target_len in unique_lengths:
        if target_len < 8 or target_len % 2 != 0:
            continue

        half_len = target_len // 2
        if half_len not in length_tensors or target_len not in length_tensors:
            continue

        target_pids, target_tensor = length_tensors[target_len]
        half_pids, half_tensor = length_tensors[half_len]
        N, M = len(target_pids), len(half_pids)

        # Extract halves
        target_lefts = target_tensor[:, :half_len]   # [N, half_len]
        target_rights = target_tensor[:, half_len:]  # [N, half_len]

        # Chunk processing for memory management
        max_mem = 2 * 1024**3
        chunk_size = max(1, max_mem // (M * half_len * 4 * 4))  # Extra factor for transforms
        chunk_size = min(chunk_size, N)

        for chunk_start in range(0, N, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N)
            chunk_lefts = target_lefts[chunk_start:chunk_end]    # [C, half_len]
            chunk_rights = target_rights[chunk_start:chunk_end]  # [C, half_len]
            chunk_n = chunk_end - chunk_start

            # 1. Find left matches using cdist
            left_dists = torch.cdist(chunk_lefts, half_tensor) / np.sqrt(half_len)  # [C, M]
            left_match_mask = left_dists < 0.01

            # 2. Check self-transform patterns (A + transform(A))
            # Instead of checking against all patterns, check if right half is a transform of left half

            # Transpose: right - left should be constant across all positions
            self_diffs = chunk_rights - chunk_lefts  # [C, half_len]
            self_diff_stds = self_diffs.std(dim=1)   # [C]
            self_diff_means = self_diffs.mean(dim=1)  # [C]
            transpose_self_match = (self_diff_stds < 0.01) & (self_diff_means.abs() <= 12) & (self_diff_means.abs() > 0.5)

            # Retrograde: right = reverse(left)
            retro_lefts = chunk_lefts.flip(dims=[1])
            retro_self_dists = ((chunk_rights - retro_lefts) ** 2).mean(dim=1)  # [C]
            retro_self_match = retro_self_dists < 0.0001

            # Inversion (contour only): right = -left
            if component == 'contour':
                invert_self_dists = ((chunk_rights + chunk_lefts) ** 2).mean(dim=1)  # [C]
                invert_self_match = invert_self_dists < 0.0001
            else:
                invert_self_match = torch.zeros(chunk_n, dtype=torch.bool, device=device)

            # 3. Extract matches - vectorized where possible
            # Get indices of self-transform matches
            transpose_indices = torch.where(transpose_self_match)[0]
            retro_indices = torch.where(retro_self_match)[0]
            invert_indices = torch.where(invert_self_match)[0]

            # For transpose matches, we need to find which pattern the left half matches
            for i in transpose_indices.cpu().numpy():
                target_idx = chunk_start + i
                target_pid = target_pids[target_idx]

                if target_pid in concat_transform_matches or target_pid in concat_matches:
                    continue

                # Find which half pattern the left matches
                left_match_indices = torch.where(left_match_mask[i])[0]
                if len(left_match_indices) > 0:
                    left_pid = half_pids[left_match_indices[0].item()]
                    shift = int(round(self_diff_means[i].item()))
                    concat_transform_matches[target_pid] = (left_pid, left_pid, 'transpose', shift)

            # For retrograde matches
            for i in retro_indices.cpu().numpy():
                target_idx = chunk_start + i
                target_pid = target_pids[target_idx]

                if target_pid in concat_transform_matches or target_pid in concat_matches:
                    continue

                left_match_indices = torch.where(left_match_mask[i])[0]
                if len(left_match_indices) > 0:
                    left_pid = half_pids[left_match_indices[0].item()]
                    concat_transform_matches[target_pid] = (left_pid, left_pid, 'retrograde', 1.0)

            # For inversion matches (contour only)
            if component == 'contour':
                for i in invert_indices.cpu().numpy():
                    target_idx = chunk_start + i
                    target_pid = target_pids[target_idx]

                    if target_pid in concat_transform_matches or target_pid in concat_matches:
                        continue

                    left_match_indices = torch.where(left_match_mask[i])[0]
                    if len(left_match_indices) > 0:
                        left_pid = half_pids[left_match_indices[0].item()]
                        concat_transform_matches[target_pid] = (left_pid, left_pid, 'invert', 1.0)

            # Cleanup
            del left_dists, self_diffs, self_diff_stds, self_diff_means
            del retro_lefts, retro_self_dists
            if component == 'contour':
                del invert_self_dists
            torch.cuda.empty_cache()

    if verbose:
        print(f"    Found {len(concat_transform_matches)} concat-transform relationships")
        sys.stdout.flush()

    def find_pattern_by_array(arr: np.ndarray, approx: bool = False, threshold: float = 0.01) -> Optional[int]:
        """O(1) lookup of pattern ID by array content.

        Args:
            arr: Pattern array to find
            approx: If True, allow approximate matches (MSE < threshold)
            threshold: MSE threshold for approximate matching
        """
        # Try exact match first
        key = (len(arr), hash(arr.tobytes()))
        candidates = pattern_index.get(key, [])
        for cid in candidates:
            if np.array_equal(patterns[cid], arr):
                return cid

        # If exact match fails and approx is enabled, try approximate matching
        if approx:
            # Check all patterns of same length
            same_len_patterns = [(pid, p) for pid, p in patterns.items() if len(p) == len(arr)]
            best_match = None
            best_mse = threshold
            for pid, p in same_len_patterns:
                mse = np.mean((p - arr) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_match = pid
            return best_match

        return None

    def find_pattern_with_transform(arr: np.ndarray, transform_type: str = 'any') -> Optional[Tuple[int, str, float]]:
        """Find pattern that matches arr after applying a transform.

        Returns: (pattern_id, transform_name, transform_param) or None
        """
        length = len(arr)
        same_len_patterns = [(pid, p) for pid, p in patterns.items() if len(p) == length]

        for pid, p in same_len_patterns:
            # Check transpose (pitch shift for contour)
            if transform_type in ['any', 'transpose'] and component == 'contour':
                diff = arr - p
                if np.std(diff) < 0.01:  # Constant difference = transpose
                    shift = int(np.round(np.mean(diff)))
                    if shift != 0 and abs(shift) <= 12:
                        return (pid, 'transpose', shift)

            # Check inversion (for contour: -p)
            if transform_type in ['any', 'invert'] and component == 'contour':
                if np.allclose(arr, -p, atol=0.01):
                    return (pid, 'invert', 1.0)

            # Check retrograde (reverse)
            if transform_type in ['any', 'retrograde']:
                if np.allclose(arr, p[::-1], atol=0.01):
                    return (pid, 'retrograde', 1.0)

        return None

    def check_repeat(pattern: np.ndarray) -> Optional[Tuple[int, int]]:
        """Check if pattern = repeat(subpattern, n) for n in [2,3,4]."""
        length = len(pattern)
        for n in [2, 3, 4]:
            if length % n == 0:
                chunk_len = length // n
                chunk = pattern[:chunk_len]
                is_repeat = True
                for i in range(1, n):
                    if not np.array_equal(pattern[i*chunk_len:(i+1)*chunk_len], chunk):
                        is_repeat = False
                        break
                if is_repeat:
                    child_id = find_pattern_by_array(chunk)
                    if child_id is not None:
                        return (child_id, n)
        return None

    def check_concat(pattern: np.ndarray, pid: int) -> Optional[Tuple[int, int]]:
        """Check if pattern = concat(P1, P2) using precomputed GPU matches."""
        # Use precomputed GPU matches for O(1) lookup
        if pid in concat_matches:
            return concat_matches[pid]
        return None

    def check_concat_with_transform(pattern: np.ndarray, pid: int) -> Optional[Tuple[int, int, str, float]]:
        """Check if pattern = concat(P1, transform(P2)) - Lewinian composition.

        Returns: (left_id, right_id, transform_name, transform_param) or None

        This finds structures like:
        - verse = phrase_A + transpose(phrase_A, 5)
        - chorus = melody + invert(melody)
        """
        # Use precomputed GPU matches for O(1) lookup
        if pid in concat_transform_matches:
            return concat_transform_matches[pid]
        return None

    import sys
    import time as time_module

    # Build length-based index for fast time_scale lookup
    patterns_by_length: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for pid, p in patterns.items():
        patterns_by_length[len(p)].append((pid, p))

    # GPU batch time_scale matching - precompute all matches at once
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build time_scale lookup table: timescale_matches[(target_len, factor)] = {target_pid: source_pid}
    timescale_matches: Dict[int, Tuple[int, float]] = {}  # target_pid -> (source_pid, factor)

    if verbose:
        print(f"  Precomputing time_scale matches (GPU batch)...")
        sys.stdout.flush()

    # For each unique length pair that could be related by factor 2.0
    unique_lengths = sorted(set(len(p) for p in patterns.values()))
    for source_len in unique_lengths:
        if source_len < 2 or source_len > 128:  # Skip very long patterns
            continue
        target_len = source_len * 2

        sources = patterns_by_length.get(source_len, [])
        targets = patterns_by_length.get(target_len, [])

        if not sources or not targets:
            continue

        # Stack into tensors
        source_pids = [pid for pid, _ in sources]
        source_arrays = torch.tensor(np.stack([p for _, p in sources]), dtype=torch.float32, device=device)

        target_pids = [pid for pid, _ in targets]
        target_arrays = torch.tensor(np.stack([p for _, p in targets]), dtype=torch.float32, device=device)

        # Stretch sources: repeat each element 2x
        stretched = source_arrays.repeat_interleave(2, dim=1)  # [K, target_len]

        # Batch compare: compute L2 distance
        # targets: [M, target_len], stretched: [K, target_len]
        # distances: [M, K]
        distances = torch.cdist(target_arrays, stretched, p=2)

        # Normalize by length for fair comparison
        threshold = 0.1 * np.sqrt(target_len)  # Scale threshold with length
        matches = distances < threshold

        # Record matches (first match wins for each target)
        for i, target_pid in enumerate(target_pids):
            if target_pid in timescale_matches:
                continue
            match_indices = torch.where(matches[i])[0]
            if len(match_indices) > 0:
                source_idx = match_indices[0].item()
                source_pid = source_pids[source_idx]
                if source_pid != target_pid:
                    timescale_matches[target_pid] = (source_pid, 2.0)

    if verbose:
        print(f"    Found {len(timescale_matches)} time_scale relationships")
        sys.stdout.flush()

    def check_time_scale(pattern: np.ndarray, pid: int) -> Optional[Tuple[int, float]]:
        """Check if pattern = time_scale(P, factor) using precomputed matches."""
        return timescale_matches.get(pid)

    # Debug: pattern length stats
    lengths = [len(p) for p in patterns.values()]
    if verbose:
        print(f"  Pattern lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
        sys.stdout.flush()

    # Skip patterns longer than 256 - they're too expensive and unlikely to have repeat structure
    MAX_PATTERN_LEN = 256
    sorted_patterns = [(pid, p) for pid, p in sorted_patterns if len(p) <= MAX_PATTERN_LEN]
    if verbose:
        print(f"  Patterns <= {MAX_PATTERN_LEN}: {len(sorted_patterns)} (skipping long patterns)")
        sys.stdout.flush()

    # Process patterns (shortest first ensures children exist before parents)
    n_repeat = 0
    n_concat = 0
    n_timescale = 0
    n_literal = 0
    n_skipped_prior = 0
    for i, (pid, pattern) in enumerate(sorted_patterns):
        if verbose and i % 500 == 0:
            print(f"    Processing pattern {i}/{len(sorted_patterns)}...")
            sys.stdout.flush()

        # Skip patterns already decomposed in prior iterations
        # (they're already in pattern_to_composition from prior_compositions)
        if pid in pattern_to_composition and pattern_to_composition[pid].composition_type != 'literal':
            n_skipped_prior += 1
            continue

        # Timing debug for first 5 patterns
        if i < 5 and verbose:
            t0 = time_module.time()

        bits_literal = len(pattern) * bits_per_elem
        best_composition = None
        best_bits = bits_literal

        # Check repeat first (most common in music)
        repeat_result = check_repeat(pattern)
        if i < 5 and verbose:
            t1 = time_module.time()
        if repeat_result:
            child_id, n = repeat_result
            # Cost: overhead + pointer to child + repeat count
            bits_composite = overhead_bits + math.log2(max(len(patterns), 1)) + math.log2(5)
            if bits_composite < best_bits:
                best_composition = ('repeat', [child_id], n)
                best_bits = bits_composite

        # Check concat
        concat_result = check_concat(pattern, pid)
        if i < 5 and verbose:
            t2 = time_module.time()
        if concat_result:
            left_id, right_id = concat_result
            # Cost: overhead + 2 pointers
            bits_composite = overhead_bits + 2 * math.log2(max(len(patterns), 1))
            if bits_composite < best_bits:
                best_composition = ('concat', [left_id, right_id], 1.0)
                best_bits = bits_composite

        # Check concat with transform (Lewinian: A + transpose(A), etc.)
        if best_composition is None or best_composition[0] not in ['repeat']:
            concat_transform_result = check_concat_with_transform(pattern, pid)
            if concat_transform_result:
                left_id, right_id, transform_name, transform_param = concat_transform_result
                # Cost: overhead + 2 pointers + transform info
                bits_composite = overhead_bits + 2 * math.log2(max(len(patterns), 1)) + 8  # extra bits for transform
                if bits_composite < best_bits:
                    # Store as special concat_transform type
                    best_composition = ('concat_transform', [left_id, right_id], (transform_name, transform_param))
                    best_bits = bits_composite

        # Check time_scale (now using precomputed GPU batch matches - O(1) lookup)
        if best_composition is None:
            scale_result = check_time_scale(pattern, pid)
            if scale_result:
                child_id, factor = scale_result
                bits_composite = overhead_bits + math.log2(max(len(patterns), 1)) + 4  # 4 bits for scale
                if bits_composite < best_bits:
                    best_composition = ('time_scale', [child_id], factor)
                    best_bits = bits_composite

        if i < 5 and verbose:
            t3 = time_module.time()
            print(f"    Pattern {i} len={len(pattern)}: repeat={t1-t0:.3f}s, concat={t2-t1:.3f}s, timescale={t3-t2:.3f}s")
            sys.stdout.flush()

        # Record result
        if best_composition:
            comp_type, children, param = best_composition
            comp = CompositePattern(
                pattern_id=pid,
                pattern=pattern,
                composition_type=comp_type,
                children=children,
                param=param,
                bits_literal=bits_literal,
                bits_composite=best_bits
            )
            composites.append(comp)
            pattern_to_composition[pid] = comp

            for child_id in children:
                child_usage_count[child_id] += 1

            if comp_type == 'repeat':
                n_repeat += 1
            elif comp_type == 'concat':
                n_concat += 1
            elif comp_type == 'concat_transform':
                n_concat += 1  # Count with concat for now
            elif comp_type == 'time_scale':
                n_timescale += 1
        else:
            # Literal/primitive pattern
            primitives.add(pid)
            n_literal += 1
            comp = CompositePattern(
                pattern_id=pid,
                pattern=pattern,
                composition_type='literal',
                children=[],
                param=1.0,
                bits_literal=bits_literal,
                bits_composite=bits_literal
            )
            pattern_to_composition[pid] = comp

    # Compute tree depth
    def get_depth(pid: int, visited: Set[int] = None) -> int:
        if visited is None:
            visited = set()
        if pid in visited:
            return 0  # Cycle (shouldn't happen)
        visited.add(pid)

        comp = pattern_to_composition.get(pid)
        if comp is None or comp.composition_type == 'literal':
            return 0

        child_depths = [get_depth(c, visited.copy()) for c in comp.children]
        return 1 + max(child_depths) if child_depths else 1

    max_depth = max((get_depth(pid) for pid in patterns.keys()), default=0)

    # Compute compression from hierarchical structure
    total_literal_bits = sum(len(p) * bits_per_elem for p in patterns.values())
    total_compressed_bits = sum(
        pattern_to_composition[pid].bits_composite
        for pid in patterns.keys()
        if pid in pattern_to_composition
    )
    compression_ratio = total_literal_bits / total_compressed_bits if total_compressed_bits > 0 else 1.0

    if verbose:
        print(f"\n  Composition breakdown:")
        print(f"    Primitives (literal): {n_literal}")
        print(f"    repeat(P, n): {n_repeat}")
        print(f"    concat(P1, P2): {n_concat}")
        print(f"    time_scale(P, s): {n_timescale}")
        print(f"\n  Tree structure:")
        print(f"    Max depth: {max_depth}")
        print(f"    Patterns used as children: {len([k for k,v in child_usage_count.items() if v > 0])}")
        print(f"    Most reused pattern: {max(child_usage_count.values()) if child_usage_count else 0} uses")
        print(f"\n  Hierarchical compression:")
        print(f"    Literal bits: {total_literal_bits:.0f}")
        print(f"    Hierarchical bits: {total_compressed_bits:.0f}")
        print(f"    Compression ratio: {compression_ratio:.2f}x")

        # Show example compositions
        examples = [c for c in composites if c.composition_type != 'literal'][:10]
        if examples:
            print(f"\n  Example compositions:")
            for c in examples:
                if c.composition_type == 'repeat':
                    print(f"    P{c.pattern_id} = repeat(P{c.children[0]}, {int(c.param)}) [saves {c.savings:.1f} bits]")
                elif c.composition_type == 'concat':
                    print(f"    P{c.pattern_id} = concat(P{c.children[0]}, P{c.children[1]}) [saves {c.savings:.1f} bits]")
                elif c.composition_type == 'time_scale':
                    print(f"    P{c.pattern_id} = time_scale(P{c.children[0]}, {c.param}) [saves {c.savings:.1f} bits]")

    return {
        'n_patterns': len(patterns),
        'n_primitives': len(primitives),
        'n_composites': len(composites) - len(primitives),
        'n_repeat': n_repeat,
        'n_concat': n_concat,
        'n_time_scale': n_timescale,
        'max_depth': max_depth,
        'literal_bits': total_literal_bits,
        'hierarchical_bits': total_compressed_bits,
        'compression_ratio': compression_ratio,
        'compositions': pattern_to_composition,
        'child_usage': dict(child_usage_count)
    }


def iterative_hierarchical_discovery(
    patterns: Dict[int, np.ndarray],
    component: str = 'rhythm',
    max_iterations: int = 10,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Iteratively discover hierarchical patterns to find deeper compositions.

    The single-pass `discover_hierarchical_patterns` only finds compositions
    using original vocabulary patterns as children. This wrapper runs multiple
    iterations, adding discovered compositions back to the vocabulary each time.

    This enables finding deeper structures like:
      concat(concat(A, B), concat(C, D))  # depth 2
      concat(concat(concat(A, B), C), D)  # depth 3

    Args:
        patterns: Dict of pattern_id -> numpy array
        component: 'rhythm' or 'contour'
        max_iterations: Maximum number of deepening iterations
        verbose: Print progress

    Returns:
        Dict with final hierarchical structure and iteration stats
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"ITERATIVE HIERARCHICAL DISCOVERY ({component})")
        print(f"{'='*70}")
        print(f"  Initial patterns: {len(patterns)}")
        print(f"  Max iterations: {max_iterations}")

    # Track iteration results
    iteration_stats = []

    # Current vocabulary - start with original patterns
    current_patterns = dict(patterns)
    next_composite_id = max(patterns.keys()) + 1 if patterns else 0

    # Track all compositions across iterations (maps to original IDs)
    all_compositions: Dict[int, CompositePattern] = {}

    # Map from composite pattern ID to its array for child lookup
    composite_arrays: Dict[int, np.ndarray] = {}

    final_result = None

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n  --- Iteration {iteration + 1} ---")
            print(f"  Vocabulary size: {len(current_patterns)}")

        # Run single-pass discovery with prior compositions for depth tracking
        result = discover_hierarchical_patterns(
            current_patterns,
            component=component,
            verbose=False,  # Suppress inner verbose for cleaner output
            prior_compositions=all_compositions if iteration > 0 else None
        )

        # Get compositions from this iteration
        compositions = result.get('compositions', {})

        # Count new compositions (non-literal)
        new_composites = []
        for pid, comp in compositions.items():
            if comp.composition_type != 'literal' and pid not in all_compositions:
                new_composites.append((pid, comp))
                all_compositions[pid] = comp

        # Stats for this iteration
        iter_stat = {
            'iteration': iteration + 1,
            'vocab_size': len(current_patterns),
            'new_composites': len(new_composites),
            'max_depth': result.get('max_depth', 0),
            'compression_ratio': result.get('compression_ratio', 1.0)
        }
        iteration_stats.append(iter_stat)

        if verbose:
            print(f"    New composites found: {len(new_composites)}")
            print(f"    Max depth this iteration: {result.get('max_depth', 0)}")
            print(f"    Compression: {result.get('compression_ratio', 1.0):.2f}x")

        # Check convergence
        if len(new_composites) == 0:
            if verbose:
                print(f"  Converged! No new compositions found.")
            final_result = result
            break

        # Add new compositions to vocabulary for next iteration
        # Each composition becomes a potential child pattern
        for pid, comp in new_composites:
            # Create a new pattern ID for this composition
            composite_arrays[pid] = comp.pattern

            # Also add the composition's pattern to vocabulary with new ID
            # This allows it to be used as a child in the next iteration
            if pid not in current_patterns:
                current_patterns[pid] = comp.pattern

        final_result = result

    else:
        if verbose:
            print(f"  Reached max iterations ({max_iterations})")

    # Compute final depth across all compositions
    def get_total_depth(pid: int, visited: Set[int] = None) -> int:
        if visited is None:
            visited = set()
        if pid in visited:
            return 0
        visited.add(pid)

        comp = all_compositions.get(pid)
        if comp is None or comp.composition_type == 'literal':
            return 0

        child_depths = [get_total_depth(c, visited.copy()) for c in comp.children]
        return 1 + max(child_depths) if child_depths else 1

    final_max_depth = max(
        (get_total_depth(pid) for pid in all_compositions.keys()),
        default=0
    )

    if verbose:
        print(f"\n  Final Results:")
        print(f"    Total iterations: {len(iteration_stats)}")
        print(f"    Final vocabulary: {len(current_patterns)}")
        print(f"    Total compositions: {len(all_compositions)}")
        print(f"    Final max depth: {final_max_depth}")
        if final_result:
            print(f"    Final compression: {final_result.get('compression_ratio', 1.0):.2f}x")

    # Return combined results - include keys compatible with single-pass discover_hierarchical_patterns
    return {
        'n_patterns': len(patterns),  # Original pattern count
        'n_primitives': final_result.get('n_primitives', 0) if final_result else 0,
        'n_composites': len(all_compositions),
        'n_repeat': final_result.get('n_repeat', 0) if final_result else 0,
        'n_concat': final_result.get('n_concat', 0) if final_result else 0,
        'n_time_scale': final_result.get('n_time_scale', 0) if final_result else 0,
        'max_depth': final_max_depth,
        'n_iterations': len(iteration_stats),
        'iteration_stats': iteration_stats,
        'final_vocab_size': len(current_patterns),
        'compositions': all_compositions,
        'literal_bits': final_result.get('literal_bits', 0) if final_result else 0,
        'hierarchical_bits': final_result.get('hierarchical_bits', 0) if final_result else 0,
        'compression_ratio': final_result.get('compression_ratio', 1.0) if final_result else 1.0,
        'child_usage': final_result.get('child_usage', {}) if final_result else {}
    }


# =============================================================================
# GPU-OPTIMIZED TRANSFORM DISCOVERY
# =============================================================================

@dataclass
class TimeShiftRelation:
    """A time-shift relationship between two objects."""
    source_idx: int
    target_idx: int
    delta: int  # target.start_time - source.start_time


@dataclass
class CompoundTransformRelation:
    """A compound transform relationship: transpose(k) + time_shift(d) + velocity_scale(s)."""
    source_idx: int
    target_idx: int
    transpose_k: int
    time_shift_d: int
    velocity_scale_s: float


@dataclass
class CrossComponentRelation:
    """Cross-component pattern: target uses rhythm from src_r and contour from src_c."""
    target_idx: int
    rhythm_source_idx: int
    contour_source_idx: int
    transpose_k: int


def discover_time_shifts_gpu(
    factored_objects: List[FactoredObject],
    verbose: bool = True
) -> List[TimeShiftRelation]:
    """
    GPU-optimized discovery of time-shift relationships.

    Finds objects that are identical except for temporal position.
    Uses grouping by (rhythm_hash, contour_hash, n_notes) for efficiency.

    Args:
        factored_objects: List of FactoredObject
        verbose: Print progress

    Returns:
        List of TimeShiftRelation
    """
    import torch
    import sys

    if verbose:
        print(f"\n{'='*70}")
        print("TIME-SHIFT DISCOVERY (GPU)")
        print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"  Using device: {device}")

    valid_objects = [o for o in factored_objects if o.num_notes > 0]
    n_objects = len(valid_objects)

    if verbose:
        print(f"  Valid objects: {n_objects}")

    # Group by (rhythm_hash, contour_hash, n_notes)
    groups: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for idx, obj in enumerate(valid_objects):
        key = (obj.rhythm_hash, obj.contour_hash, obj.num_notes)
        groups[key].append(idx)

    # Filter to groups with >= 2 members
    candidate_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    n_groups = len(candidate_groups)

    if verbose:
        print(f"  Groups with potential time-shifts: {n_groups}")
        total_candidates = sum(len(v) * (len(v) - 1) // 2 for v in candidate_groups.values())
        print(f"  Total candidate pairs: {total_candidates}")

    relations = []

    for group_idx, (group_key, indices) in enumerate(candidate_groups.items()):
        # Progress logging every 1000 groups
        if verbose and group_idx % 1000 == 0:
            print(f"    Processing group {group_idx}/{n_groups}...")
            sys.stdout.flush()

        if len(indices) < 2:
            continue

        # Get pitches for all objects in this group
        group_objects = [valid_objects[i] for i in indices]

        # Find max notes in group
        max_notes = max(obj.num_notes for obj in group_objects)

        # Pad pitches to same length and stack
        pitch_tensors = []
        for obj in group_objects:
            pitches = np.zeros(max_notes, dtype=np.float32)
            pitches[:len(obj.pitches)] = obj.pitches
            pitch_tensors.append(pitches)

        # Stack and move to GPU: [N_group, max_notes]
        pitches_gpu = torch.tensor(np.stack(pitch_tensors), device=device)

        # Compute pairwise L∞ distance
        # For each pair (i, j), compute max |pitches[i] - pitches[j]|
        n_group = len(indices)

        # Expand for broadcasting: [N, 1, max_notes] vs [1, N, max_notes]
        p1 = pitches_gpu.unsqueeze(1)  # [N, 1, max_notes]
        p2 = pitches_gpu.unsqueeze(0)  # [1, N, max_notes]

        # Pairwise differences: [N, N, max_notes]
        diffs = (p1 - p2).abs()

        # L∞ distance for each pair: [N, N]
        linf_dist = diffs.max(dim=-1).values

        # Find pairs where L∞ distance is 0 (identical pitches)
        # Only consider upper triangle (i < j)
        mask = torch.triu(linf_dist == 0, diagonal=1)
        identical_pairs = torch.nonzero(mask, as_tuple=False).cpu().numpy()

        # Record time-shift relations
        for pair in identical_pairs:
            i_local, j_local = pair
            i_global = indices[i_local]
            j_global = indices[j_local]

            obj_i = valid_objects[i_global]
            obj_j = valid_objects[j_global]

            delta = obj_j.start_time - obj_i.start_time

            # Source is the earlier one
            if delta > 0:
                relations.append(TimeShiftRelation(
                    source_idx=i_global,
                    target_idx=j_global,
                    delta=delta
                ))
            elif delta < 0:
                relations.append(TimeShiftRelation(
                    source_idx=j_global,
                    target_idx=i_global,
                    delta=-delta
                ))
            # If delta == 0, same temporal position, skip

    if verbose:
        print(f"  Time-shift relations found: {len(relations)}")

        # Show delta distribution
        if relations:
            deltas = [r.delta for r in relations]
            delta_counts = Counter(deltas)
            print(f"  Top 10 time-shift deltas:")
            for delta, count in delta_counts.most_common(10):
                print(f"    delta={delta}: {count}")

    return relations


def discover_compound_transforms_gpu(
    factored_objects: List[FactoredObject],
    transpose_range: int = 24,
    velocity_scales: List[float] = None,
    verbose: bool = True
) -> List[CompoundTransformRelation]:
    """
    GPU-optimized discovery of compound transforms using batch operations.

    Finds objects related by: transpose(k) + time_shift(d) + velocity_scale(s)

    NOTE: This is currently SKIPPED because compound transforms are redundant with:
    - Factored matches (already captures transposes)
    - Time-shift discovery (already captures temporal relationships)

    The combination of these two phases covers what compound would find.
    Keeping this function for future use if we want to find truly novel
    relationships (e.g., cross-rhythm patterns with transpose+time_shift).

    Args:
        factored_objects: List of FactoredObject
        transpose_range: Check transpositions in [-range, +range] semitones
        velocity_scales: List of velocity scales to check (default: [0.7, 0.8, 0.9, 1.0, 1.1])
        verbose: Print progress

    Returns:
        List of CompoundTransformRelation (currently empty - skipped for performance)
    """
    if verbose:
        print(f"\n{'='*70}")
        print("COMPOUND TRANSFORM DISCOVERY (SKIPPED)")
        print(f"{'='*70}")
        print("  [Skipping - redundant with factored matches + time-shift discovery]")

    # Return empty list - compound transforms are redundant with existing phases
    return []

    # Original implementation below (kept for reference):
    import torch
    import sys

    if velocity_scales is None:
        velocity_scales = [0.7, 0.8, 0.9, 1.0, 1.1]

    if False:  # Disabled
        print(f"\n{'='*70}")
        print("COMPOUND TRANSFORM DISCOVERY (GPU)")
        print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"  Using device: {device}")
        print(f"  Transpose range: [-{transpose_range}, +{transpose_range}]")
        print(f"  Velocity scales: {velocity_scales}")

    valid_objects = [o for o in factored_objects if o.num_notes > 0]
    n_objects = len(valid_objects)

    if verbose:
        print(f"  Valid objects: {n_objects}")

    # Group by (rhythm_hash, contour_hash) - same structure, different pitch level
    groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, obj in enumerate(valid_objects):
        key = (obj.rhythm_hash, obj.contour_hash)
        groups[key].append(idx)

    # Filter to groups with >= 2 members
    candidate_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    n_groups = len(candidate_groups)

    if verbose:
        print(f"  Groups with same rhythm+contour: {n_groups}")

    relations = []

    # All non-zero transpose amounts to test (skip 0, already covered by time_shift)
    transpose_amounts = [k for k in range(-transpose_range, transpose_range + 1) if k != 0]
    n_transposes = len(transpose_amounts)

    # Process groups with progress logging
    for group_idx, (group_key, indices) in enumerate(candidate_groups.items()):
        # Progress logging every 1000 groups
        if verbose and group_idx % 1000 == 0:
            print(f"    Processing group {group_idx}/{n_groups}...")
            sys.stdout.flush()

        if len(indices) < 2:
            continue

        group_objects = [valid_objects[i] for i in indices]
        n_group = len(indices)

        # Get max notes in group for padding
        max_notes = max(obj.num_notes for obj in group_objects)

        # Skip very large groups to avoid memory issues (will process in chunks if needed)
        if n_group * max_notes > 100000:
            # Process in smaller chunks
            chunk_size = max(2, 100000 // max_notes)
            for chunk_start in range(0, n_group, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_group)
                chunk_indices = indices[chunk_start:chunk_end]
                chunk_objects = group_objects[chunk_start:chunk_end]
                _process_compound_group_gpu(
                    chunk_objects, chunk_indices, valid_objects,
                    transpose_amounts, velocity_scales, device, relations
                )
        else:
            _process_compound_group_gpu(
                group_objects, indices, valid_objects,
                transpose_amounts, velocity_scales, device, relations
            )

    if verbose:
        print(f"  Compound transform relations found: {len(relations)}")

        if relations:
            # Transpose distribution
            transpose_counts = Counter(r.transpose_k for r in relations)
            print(f"  Top 10 transpositions:")
            for k, count in transpose_counts.most_common(10):
                print(f"    transpose({k}): {count}")

    return relations


def _process_compound_group_gpu(
    group_objects: List[FactoredObject],
    indices: List[int],
    valid_objects: List[FactoredObject],
    transpose_amounts: List[int],
    velocity_scales: List[float],
    device,  # torch.device - avoid module-level torch reference
    relations: List[CompoundTransformRelation]
) -> None:
    """
    Process a single group using efficient GPU approach with chunking.

    Key insight: Instead of testing 48 transpose values per pair,
    compute the transpose from the FIRST note of each pair, then verify
    if all other notes match with that transpose.

    For large groups, we chunk to avoid OOM (diff tensor = N*N*max_notes*4 bytes).
    """
    import torch

    n_group = len(indices)
    if n_group < 2:
        return

    max_notes = max(obj.num_notes for obj in group_objects)

    # Build data arrays (keep on CPU initially)
    pitch_list = []
    velocity_list = []
    note_counts = []
    start_times = []

    for obj in group_objects:
        pitches = np.full(max_notes, -999.0, dtype=np.float32)
        pitches[:len(obj.pitches)] = obj.pitches
        pitch_list.append(pitches)

        vels = np.zeros(max_notes, dtype=np.float32)
        vels[:len(obj.velocities)] = obj.velocities
        velocity_list.append(vels)

        note_counts.append(obj.num_notes)
        start_times.append(obj.start_time)

    pitches_np = np.stack(pitch_list)
    velocities_np = np.stack(velocity_list)
    note_counts_np = np.array(note_counts)

    # Compute first pitch for transpose inference
    first_pitch_np = pitches_np[:, 0]

    # Transpose set for validation
    transpose_set = set(transpose_amounts)
    min_k, max_k = min(transpose_amounts), max(transpose_amounts)

    # Estimate memory for full batch: N*N*max_notes*4 bytes
    # Limit to ~4GB per chunk
    max_mem_bytes = 4 * 1024**3
    elements_per_pair = max_notes
    bytes_per_pair = elements_per_pair * 4  # float32
    max_pairs = max_mem_bytes // bytes_per_pair

    # Chunk size: process chunk_size rows at a time
    # Each row processes against all N columns
    chunk_size = max(1, max_pairs // n_group)
    chunk_size = min(chunk_size, n_group)  # Don't exceed group size

    # Move full pitch tensor to GPU (just [N, max_notes], manageable)
    pitches_gpu = torch.tensor(pitches_np, device=device)
    valid_mask = pitches_gpu > -900  # [N, max_notes]

    # Process in chunks
    for chunk_start in range(0, n_group, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_group)
        chunk_n = chunk_end - chunk_start

        # Chunk of source objects
        pitches_chunk = pitches_gpu[chunk_start:chunk_end]  # [chunk_n, max_notes]
        valid_chunk = valid_mask[chunk_start:chunk_end]  # [chunk_n, max_notes]

        # Note counts for chunk
        note_counts_chunk = note_counts_np[chunk_start:chunk_end]

        # Compute candidate transpose: chunk sources vs all targets
        first_pitch_chunk = first_pitch_np[chunk_start:chunk_end]  # [chunk_n]
        candidate_k = first_pitch_np[np.newaxis, :] - first_pitch_chunk[:, np.newaxis]  # [chunk_n, N]
        candidate_k_int = np.round(candidate_k).astype(np.int32)

        # Filter by same note count
        same_count = (note_counts_chunk[:, np.newaxis] == note_counts_np[np.newaxis, :])  # [chunk_n, N]

        # Filter by valid transpose range
        valid_k = (candidate_k_int >= min_k) & (candidate_k_int <= max_k) & same_count

        # Exclude diagonal within chunk
        for local_i in range(chunk_n):
            global_i = chunk_start + local_i
            if global_i < n_group:
                valid_k[local_i, global_i] = False

        if not valid_k.any():
            continue

        # Move candidate_k to GPU for residual computation
        candidate_k_gpu = torch.tensor(candidate_k, device=device)  # [chunk_n, N]

        # Compute diff: pitches[j] - pitches[i] for each pair
        # pitches_chunk: [chunk_n, max_notes]
        # pitches_gpu: [N, max_notes]
        # diff[i,j,n] = pitches_gpu[j,n] - pitches_chunk[i,n]
        diff = pitches_gpu.unsqueeze(0) - pitches_chunk.unsqueeze(1)  # [chunk_n, N, max_notes]

        # Residual = diff - candidate_k
        residual = diff - candidate_k_gpu.unsqueeze(2)  # [chunk_n, N, max_notes]

        # Valid mask for pairs
        valid_pair = valid_chunk.unsqueeze(1) & valid_mask.unsqueeze(0)  # [chunk_n, N, max_notes]

        # Masked residual
        residual_masked = torch.where(valid_pair, residual.abs(), torch.zeros_like(residual))

        # Max residual per pair
        max_residual = residual_masked.max(dim=2).values  # [chunk_n, N]

        # Convert valid_k to tensor
        valid_k_gpu = torch.tensor(valid_k, device=device)

        # Matches
        matches = (max_residual < 0.5) & valid_k_gpu

        if not matches.any():
            continue

        # Get matching pairs
        local_src, tgt = torch.where(matches)
        local_src = local_src.cpu().numpy()
        tgt = tgt.cpu().numpy()

        # Process matches
        for pair_idx in range(len(local_src)):
            local_i = local_src[pair_idx]
            j = tgt[pair_idx]
            i = chunk_start + local_i
            k = int(candidate_k_int[local_i, j])

            # Skip if k not in allowed set
            if k not in transpose_set:
                continue

            # Compute velocity scale
            n_notes = note_counts_np[i]
            v_i = velocities_np[i, :n_notes]
            v_j = velocities_np[j, :n_notes]

            v_i_sum = v_i.sum()
            if v_i_sum > 0:
                avg_scale = v_j.sum() / v_i_sum
                best_scale = min(velocity_scales, key=lambda s: abs(s - avg_scale))
            else:
                best_scale = 1.0

            # Compute time shift
            delta = start_times[j] - start_times[i]

            relations.append(CompoundTransformRelation(
                source_idx=indices[i],
                target_idx=indices[j],
                transpose_k=k,
                time_shift_d=delta,
                velocity_scale_s=best_scale
            ))


def discover_cross_component_gpu(
    factored_objects: List[FactoredObject],
    transpose_range: int = 24,
    verbose: bool = True
) -> List[CrossComponentRelation]:
    """
    GPU-optimized cross-component discovery.

    Finds patterns where target uses rhythm from one source and contour from another:
        target = rhythm(src_r) × contour(src_c) × transpose(k)

    Args:
        factored_objects: List of FactoredObject
        transpose_range: Check transpositions in [-range, +range] semitones
        verbose: Print progress

    Returns:
        List of CrossComponentRelation
    """
    import torch
    import sys

    if verbose:
        print(f"\n{'='*70}")
        print("CROSS-COMPONENT DISCOVERY (GPU)")
        print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"  Using device: {device}")

    valid_objects = [o for o in factored_objects if o.num_notes > 0]
    n_objects = len(valid_objects)

    if verbose:
        print(f"  Valid objects: {n_objects}")

    # Build indices by rhythm and contour
    by_rhythm: Dict[int, List[int]] = defaultdict(list)
    by_contour: Dict[int, List[int]] = defaultdict(list)

    for idx, obj in enumerate(valid_objects):
        by_rhythm[obj.rhythm_hash].append(idx)
        by_contour[obj.contour_hash].append(idx)

    if verbose:
        print(f"  Unique rhythm patterns: {len(by_rhythm)}")
        print(f"  Unique contour patterns: {len(by_contour)}")

    relations = []

    # For each object, check if it can be explained by combining rhythm from one
    # source and contour from another
    for target_idx, target in enumerate(valid_objects):
        # Progress logging every 10000 objects
        if verbose and target_idx % 10000 == 0:
            print(f"    Processing target {target_idx}/{n_objects}...")
            sys.stdout.flush()
        target_rhythm = target.rhythm_hash
        target_contour = target.contour_hash

        # Find objects that share this rhythm but different contour
        rhythm_sources = by_rhythm.get(target_rhythm, [])

        # Find objects that share this contour but different rhythm
        contour_sources = by_contour.get(target_contour, [])

        # Check all combinations
        for r_idx in rhythm_sources:
            if r_idx == target_idx:
                continue
            r_obj = valid_objects[r_idx]

            # Skip if rhythm source has same contour as target (not truly cross-component)
            if r_obj.contour_hash == target_contour:
                continue

            for c_idx in contour_sources:
                if c_idx == target_idx or c_idx == r_idx:
                    continue
                c_obj = valid_objects[c_idx]

                # Skip if contour source has same rhythm as target (not truly cross-component)
                if c_obj.rhythm_hash == target_rhythm:
                    continue

                # This is a TRUE cross-component pattern:
                # 1. target has same rhythm as r_obj (but r_obj has DIFFERENT contour)
                # 2. target has same contour as c_obj (but c_obj has DIFFERENT rhythm)
                # 3. r_obj and c_obj are different objects

                # Compute transpose needed
                if len(target.pitches) > 0 and len(c_obj.pitches) > 0:
                    # Root pitch difference
                    transpose_k = int(target.pitches[0] - c_obj.pitches[0])

                    if abs(transpose_k) <= transpose_range:
                        relations.append(CrossComponentRelation(
                            target_idx=target_idx,
                            rhythm_source_idx=r_idx,
                            contour_source_idx=c_idx,
                            transpose_k=transpose_k
                        ))
                        break  # Found one explanation for this target
            else:
                continue
            break  # Found one explanation for this target

    if verbose:
        print(f"  Cross-component relations found: {len(relations)}")

        if relations:
            transpose_counts = Counter(r.transpose_k for r in relations)
            print(f"  Top 10 transpositions:")
            for k, count in transpose_counts.most_common(10):
                print(f"    transpose({k}): {count}")

    return relations


def run_extended_discovery(
    factored_objects: List[FactoredObject],
    verbose: bool = True
) -> Dict[str, any]:
    """
    Run all three extended discovery methods.

    Args:
        factored_objects: List of FactoredObject
        verbose: Print progress

    Returns:
        Dict with results from all discovery methods
    """
    results = {}

    # 1. Time-shift discovery
    time_shifts = discover_time_shifts_gpu(factored_objects, verbose=verbose)
    results['time_shifts'] = time_shifts

    # 2. Compound transform discovery
    compound_transforms = discover_compound_transforms_gpu(factored_objects, verbose=verbose)
    results['compound_transforms'] = compound_transforms

    # 3. Cross-component discovery
    cross_components = discover_cross_component_gpu(factored_objects, verbose=verbose)
    results['cross_components'] = cross_components

    # Count unique targets across all methods
    valid_objects = [o for o in factored_objects if o.num_notes > 0]
    n_objects = len(valid_objects)

    targets_time_shift = {r.target_idx for r in time_shifts}
    targets_compound = {r.target_idx for r in compound_transforms}
    targets_cross = {r.target_idx for r in cross_components}

    all_targets = targets_time_shift | targets_compound | targets_cross

    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print("EXTENDED DISCOVERY SUMMARY")
        print(f"{'='*70}")

        print(f"  Total valid objects: {n_objects}")
        print(f"  Objects explained by time-shift: {len(targets_time_shift)}")
        print(f"  Objects explained by compound transform: {len(targets_compound)}")
        print(f"  Objects explained by cross-component: {len(targets_cross)}")
        print(f"  Total unique objects explained: {len(all_targets)}")
        print(f"  Coverage: {len(all_targets)/n_objects*100:.1f}%" if n_objects > 0 else "  Coverage: 0.0%")

    results['n_objects'] = n_objects
    results['n_explained'] = len(all_targets)

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

# Standard transforms to search for
RHYTHM_TRANSFORMS = [
    ('retrograde', 0),
    ('time_shift', 16),
    ('time_shift', -16),
    ('time_shift', 32),
    ('time_shift', -32),
]

CONTOUR_TRANSFORMS = [
    ('inversion', 0),
    ('retrograde', 0),
    ('retrograde_inversion', 0),
]


def save_checkpoint(
    path: str,
    factored_objects: List[FactoredObject],
    rhythm_hierarchical: Dict[str, any] = None,
    contour_hierarchical: Dict[str, any] = None,
    verbose: bool = True
):
    """
    Save factored MDL checkpoint as compressed NPZ file.

    This stores everything needed to reconstruct and edit MIDI:
    - All unique rhythm patterns
    - All unique contour patterns
    - Object assignments (which pattern + transform for each object)
    - Transform relations and compositions (for editor visualization)

    Args:
        path: Output path (should end in .npz)
        factored_objects: List of FactoredObject
        rhythm_hierarchical: Results from iterative_hierarchical_discovery for rhythm
        contour_hierarchical: Results from iterative_hierarchical_discovery for contour
        verbose: Print progress
    """
    valid_objects = [o for o in factored_objects if o.num_notes > 0]

    # Collect unique patterns
    rhythm_patterns = {}  # hash -> pattern array
    contour_patterns = {}  # hash -> pattern array

    for obj in valid_objects:
        rh = obj.rhythm_hash
        if rh not in rhythm_patterns:
            rhythm_patterns[rh] = obj.rhythm
        ch = obj.contour_hash
        if ch not in contour_patterns:
            contour_patterns[ch] = obj.pitch_contour

    # Create pattern ID mappings (hash -> R0, R1, etc.)
    rhythm_ids = {h: f"R{i}" for i, h in enumerate(rhythm_patterns.keys())}
    contour_ids = {h: f"C{i}" for i, h in enumerate(contour_patterns.keys())}

    # Build object assignments
    object_data = []
    for obj in valid_objects:
        object_data.append({
            'piece_id': obj.piece_id,
            'track_id': obj.track_id,
            'start_time': obj.start_time,
            'scale': obj.scale,
            'rhythm_id': rhythm_ids[obj.rhythm_hash],
            'contour_id': contour_ids[obj.contour_hash],
            'root_pitch': int(obj.pitches[0]) if len(obj.pitches) > 0 else 0,
            'pitches': obj.pitches.tolist(),
            'velocities': obj.velocities.tolist(),
            'durations': obj.durations.tolist(),
            'onset_times': obj.onset_times.tolist(),  # REQUIRED for reconstruction
            'is_drum': obj.is_drum,  # True if drum/percussion track
        })

    # Convert compositions to serializable format
    # Each composition: {pattern_id: {'type': str, 'children': [ids], 'param': any}}
    def serialize_compositions(hierarchical_result: Dict, id_prefix: str) -> Dict:
        if hierarchical_result is None:
            return {}
        compositions = hierarchical_result.get('compositions', {})
        serialized = {}
        for pid, comp in compositions.items():
            pattern_id = f"{id_prefix}{pid}" if isinstance(pid, int) else pid
            serialized[pattern_id] = {
                'type': comp.composition_type,
                'children': [f"{id_prefix}{c}" if isinstance(c, int) else c for c in comp.children],
                'param': comp.param if hasattr(comp, 'param') else None,
                'depth': comp.depth if hasattr(comp, 'depth') else 0
            }
        return serialized

    rhythm_compositions = serialize_compositions(rhythm_hierarchical, 'R')
    contour_compositions = serialize_compositions(contour_hierarchical, 'C')

    # Build edges list from compositions
    # Each edge: [from_id, to_id, transform_type, param]
    def compositions_to_edges(compositions: Dict) -> List:
        edges = []
        for pid, comp in compositions.items():
            if comp['type'] == 'literal':
                continue
            for child in comp['children']:
                edges.append([child, pid, comp['type'], comp['param']])
        return edges

    rhythm_edges = compositions_to_edges(rhythm_compositions)
    contour_edges = compositions_to_edges(contour_compositions)

    # Save as NPZ
    np.savez_compressed(
        path,
        # Pattern data
        rhythm_pattern_ids=np.array(list(rhythm_ids.values())),
        rhythm_pattern_hashes=np.array(list(rhythm_ids.keys())),
        rhythm_patterns=np.array([rhythm_patterns[h] for h in rhythm_ids.keys()], dtype=object),
        contour_pattern_ids=np.array(list(contour_ids.values())),
        contour_pattern_hashes=np.array(list(contour_ids.keys())),
        contour_patterns=np.array([contour_patterns[h] for h in contour_ids.keys()], dtype=object),
        # Object assignments (as JSON string for flexibility)
        objects_json=np.array([str(object_data)]),
        # Transform relations and compositions (as JSON strings)
        rhythm_compositions_json=np.array([str(rhythm_compositions)]),
        contour_compositions_json=np.array([str(contour_compositions)]),
        rhythm_edges_json=np.array([str(rhythm_edges)]),
        contour_edges_json=np.array([str(contour_edges)]),
    )

    if verbose:
        print(f"\n  Checkpoint saved to: {path}")
        print(f"    Rhythm patterns: {len(rhythm_patterns)}")
        print(f"    Contour patterns: {len(contour_patterns)}")
        print(f"    Objects: {len(object_data)}")
        print(f"    Rhythm compositions: {len(rhythm_compositions)}")
        print(f"    Contour compositions: {len(contour_compositions)}")


def load_checkpoint(path: str) -> dict:
    """
    Load factored MDL checkpoint.

    Returns:
        Dict with 'rhythm_patterns', 'contour_patterns', 'objects'
    """
    import ast
    data = np.load(path, allow_pickle=True)

    # Reconstruct pattern dicts
    rhythm_patterns = dict(zip(
        data['rhythm_pattern_ids'],
        data['rhythm_patterns']
    ))
    contour_patterns = dict(zip(
        data['contour_pattern_ids'],
        data['contour_patterns']
    ))

    # Parse object assignments
    objects_json = str(data['objects_json'][0])
    objects = ast.literal_eval(objects_json)

    return {
        'rhythm_patterns': rhythm_patterns,
        'contour_patterns': contour_patterns,
        'objects': objects
    }


# =============================================================================
# RECONSTRUCTION AND QUALITY VERIFICATION
# =============================================================================

def reconstruct_object_from_components(
    pitches: np.ndarray,
    velocities: np.ndarray,
    durations: np.ndarray,
    onset_times: np.ndarray,
    T: int = None,
    rhythm: np.ndarray = None,  # Optional, only used for T if not provided
) -> np.ndarray:
    """
    Reconstruct a piano roll tensor from factored components.

    For exact reconstruction, we need:
    - pitches: which MIDI note
    - onset_times: when each note starts
    - durations: how long each note lasts
    - velocities: velocity of each note

    Args:
        pitches: (N,) MIDI pitches
        velocities: (N,) velocity per note
        durations: (N,) duration per note in timesteps
        onset_times: (N,) onset time for each note
        T: Total timesteps (if None, inferred from rhythm or max onset+duration)
        rhythm: Optional binary rhythm pattern, used to infer T

    Returns:
        Reconstructed (T, F) tensor where F = 130 (128 pitches + velocity + duration)
    """
    # Determine T
    if T is None:
        if rhythm is not None:
            T = len(rhythm)
        elif len(onset_times) > 0:
            T = int(max(onset_times + durations)) + 1
        else:
            T = 16

    F = 130  # 128 pitches + velocity + duration
    tensor = np.zeros((T, F), dtype=np.float32)

    if len(pitches) == 0:
        return tensor

    # Reconstruct each note
    for i in range(len(pitches)):
        pitch = int(pitches[i])
        vel = float(velocities[i]) if i < len(velocities) else 0.8
        dur = int(durations[i]) if i < len(durations) else 1
        onset_t = int(onset_times[i]) if i < len(onset_times) else 0

        # Clamp values
        pitch = max(0, min(127, pitch))
        onset_t = max(0, min(onset_t, T - 1))
        dur = max(1, min(dur, T - onset_t))

        # Fill in tensor
        for t in range(onset_t, min(onset_t + dur, T)):
            tensor[t, pitch] = 1.0
            tensor[t, 128] = vel  # velocity channel

    return tensor


def measure_reconstruction_quality(
    factored_objects: List[FactoredObject],
    verbose: bool = True
) -> dict:
    """
    Measure how accurately we can reconstruct original objects from factored components.

    The factored representation stores:
    - pitches: absolute MIDI pitches of all notes
    - durations: duration of each note
    - velocities: velocity of each note

    With these, reconstruction should be 100% accurate (lossless).
    The rhythm and contour are DERIVED from these - they're for pattern matching,
    not reconstruction.

    Returns:
        Dict with reconstruction metrics
    """
    if verbose:
        print(f"\n{'='*70}")
        print("RECONSTRUCTION QUALITY VERIFICATION")
        print(f"{'='*70}")

    valid_objects = [o for o in factored_objects if o.num_notes > 0 and o.original_tensor is not None]

    if not valid_objects:
        if verbose:
            print("  No valid objects with original tensors to verify")
        return {'objects_verified': 0, 'reconstruction_accuracy': 1.0}

    total_notes = 0
    reconstructed_notes = 0
    exact_matches = 0
    total_mse = 0.0

    for obj in valid_objects:
        # Reconstruct from factored components using onset_times
        reconstructed = reconstruct_object_from_components(
            pitches=obj.pitches,
            velocities=obj.velocities,
            durations=obj.durations,
            onset_times=obj.onset_times,
            rhythm=obj.rhythm,  # Used to get T
        )

        original = obj.original_tensor

        # Ensure same shape
        T = min(reconstructed.shape[0], original.shape[0])
        F = min(reconstructed.shape[1], original.shape[1])

        reconstructed = reconstructed[:T, :F]
        original = original[:T, :F]

        # Compare pitch channels (0:128)
        pitch_original = original[:, :128]
        pitch_reconstructed = reconstructed[:, :128]

        # Count notes in original
        original_note_count = int(np.sum(pitch_original > 0.5))
        total_notes += original_note_count

        # Count matching notes
        matching = np.logical_and(pitch_original > 0.5, pitch_reconstructed > 0.5)
        matching_count = int(np.sum(matching))
        reconstructed_notes += matching_count

        # Check exact match
        if np.allclose(pitch_original, pitch_reconstructed, atol=0.01):
            exact_matches += 1

        # MSE on pitch channels
        mse = np.mean((pitch_original - pitch_reconstructed) ** 2)
        total_mse += mse

    # Compute metrics
    note_accuracy = reconstructed_notes / total_notes if total_notes > 0 else 1.0
    exact_match_rate = exact_matches / len(valid_objects) if valid_objects else 1.0
    avg_mse = total_mse / len(valid_objects) if valid_objects else 0.0

    if verbose:
        print(f"  Objects verified: {len(valid_objects)}")
        print(f"  Note reconstruction accuracy: {note_accuracy*100:.1f}%")
        print(f"  Exact matches: {exact_matches}/{len(valid_objects)} ({exact_match_rate*100:.1f}%)")
        print(f"  Average MSE: {avg_mse:.6f}")
        print(f"  Total notes: {total_notes}, Correctly reconstructed: {reconstructed_notes}")

    return {
        'objects_verified': len(valid_objects),
        'total_notes': total_notes,
        'reconstructed_notes': reconstructed_notes,
        'note_accuracy': note_accuracy,
        'exact_matches': exact_matches,
        'exact_match_rate': exact_match_rate,
        'average_mse': avg_mse
    }


def get_reconstruction_data(factored_objects: List[FactoredObject]) -> dict:
    """
    Get reconstruction data as a dictionary (for JSON output).

    This is everything needed to reconstruct and edit MIDI.
    """
    valid_objects = [o for o in factored_objects if o.num_notes > 0]

    # Collect unique patterns
    rhythm_patterns = {}
    contour_patterns = {}

    for obj in valid_objects:
        rh = obj.rhythm_hash
        if rh not in rhythm_patterns:
            rhythm_patterns[rh] = obj.rhythm
        ch = obj.contour_hash
        if ch not in contour_patterns:
            contour_patterns[ch] = obj.pitch_contour

    # Create pattern ID mappings
    rhythm_ids = {h: f"R{i}" for i, h in enumerate(rhythm_patterns.keys())}
    contour_ids = {h: f"C{i}" for i, h in enumerate(contour_patterns.keys())}

    # Build reconstruction data
    return {
        'rhythm_patterns': {
            rhythm_ids[h]: p.tolist() for h, p in rhythm_patterns.items()
        },
        'contour_patterns': {
            contour_ids[h]: p.tolist() for h, p in contour_patterns.items()
        },
        'objects': [
            {
                'id': f"{obj.piece_id}:{obj.track_id}@{obj.start_time}",
                'piece_id': obj.piece_id,
                'track_id': obj.track_id,
                'start_time': obj.start_time,
                'scale': obj.scale,
                'rhythm_id': rhythm_ids[obj.rhythm_hash],
                'contour_id': contour_ids[obj.contour_hash],
                'root_pitch': int(obj.pitches[0]) if len(obj.pitches) > 0 else 0,
                # Full note data for exact reconstruction
                'pitches': obj.pitches.tolist(),
                'velocities': obj.velocities.tolist(),
                'durations': obj.durations.tolist(),
                'onset_times': obj.onset_times.tolist(),  # REQUIRED for reconstruction
            }
            for obj in valid_objects
        ]
    }


def run_factored_mdl(
    objects: List,  # MusicalObject from old system
    min_group_size: int = 3,
    verbose: bool = True,
    save_checkpoint_path: str = None
) -> FactoredMDLResult:
    """
    Main entry point for factored MDL discovery.

    Args:
        objects: List of MusicalObject (from old system)
        min_group_size: Minimum objects in a group
        verbose: Print progress
        save_checkpoint_path: If provided, save NPZ checkpoint to this path

    Returns:
        FactoredMDLResult
    """
    # Phase 1: Factor all objects
    factored = factor_objects_from_corpus(objects, verbose)

    # Phase 2: Basic pattern discovery (same rhythm+contour groups)
    result = discover_factored_patterns(factored, min_group_size, verbose)

    # Phase 3: Transform-aware pattern discovery - SKIPPED (stats-only, not used downstream)
    # This was taking minutes to count ~19M patterns that aren't used for MDL optimization
    if verbose:
        print(f"\n  [Skipping transform-aware pattern enumeration - stats only, not used for MDL]")
    result.stats['transform_aware_patterns'] = 0  # Would be ~19M
    result.stats['cross_component_patterns'] = 0
    result.stats['same_source_patterns'] = 0

    # Phase 4: Legacy cross-component discovery - SKIPPED (also stats-only)
    result.stats['cross_patterns'] = 0

    # =========================================================================
    # NEW FEATURES
    # =========================================================================

    # Feature 1: Explicit bit counting
    compression_stats = compute_full_compression_stats(factored, verbose=verbose)
    result.stats['compression'] = {
        'total_ratio': compression_stats['total'].compression_ratio,
        'per_component': {k: v.compression_ratio for k, v in compression_stats.items() if k != 'total'},
        'total_literal_bits': compression_stats['total'].literal_bits,
        'total_compressed_bits': compression_stats['total'].compressed_bits
    }

    # Feature 2: Within-component transform discovery
    rhythm_transform_stats = find_within_component_transforms(
        factored, 'rhythm', RHYTHM_TRANSFORMS, verbose=verbose
    )
    contour_transform_stats = find_within_component_transforms(
        factored, 'contour', CONTOUR_TRANSFORMS, verbose=verbose
    )
    result.stats['within_component_transforms'] = {
        'rhythm': {
            'canonical': rhythm_transform_stats.get('canonical_patterns', 0),
            'derived': rhythm_transform_stats.get('derived_patterns', 0),
            'compression': rhythm_transform_stats.get('compression_from_transforms', 1.0)
        },
        'contour': {
            'canonical': contour_transform_stats.get('canonical_patterns', 0),
            'derived': contour_transform_stats.get('derived_patterns', 0),
            'compression': contour_transform_stats.get('compression_from_transforms', 1.0)
        }
    }

    # Feature 3: Hierarchical pattern discovery (find compositional structure)
    # Build pattern dictionaries for hierarchical discovery
    valid_objects = [o for o in factored if o.num_notes > 0]
    rhythm_patterns_dict = {}
    contour_patterns_dict = {}
    for i, obj in enumerate(valid_objects):
        rh = obj.rhythm_hash
        if rh not in rhythm_patterns_dict:
            rhythm_patterns_dict[rh] = obj.rhythm
        ch = obj.contour_hash
        if ch not in contour_patterns_dict:
            contour_patterns_dict[ch] = obj.pitch_contour

    # Convert to int-keyed dicts for hierarchical discovery
    rhythm_id_map = {h: i for i, h in enumerate(rhythm_patterns_dict.keys())}
    contour_id_map = {h: i for i, h in enumerate(contour_patterns_dict.keys())}

    rhythm_patterns_by_id = {rhythm_id_map[h]: p for h, p in rhythm_patterns_dict.items()}
    contour_patterns_by_id = {contour_id_map[h]: p for h, p in contour_patterns_dict.items()}

    # Run hierarchical discovery on rhythm patterns (with iterative deepening)
    rhythm_hierarchical = iterative_hierarchical_discovery(
        rhythm_patterns_by_id, component='rhythm', max_iterations=10, verbose=verbose
    )

    # Run hierarchical discovery on contour patterns (with iterative deepening)
    contour_hierarchical = iterative_hierarchical_discovery(
        contour_patterns_by_id, component='contour', max_iterations=10, verbose=verbose
    )

    result.stats['hierarchical'] = {
        'rhythm': {
            'n_primitives': rhythm_hierarchical['n_primitives'],
            'n_composites': rhythm_hierarchical['n_composites'],
            'n_repeat': rhythm_hierarchical['n_repeat'],
            'n_concat': rhythm_hierarchical['n_concat'],
            'n_time_scale': rhythm_hierarchical['n_time_scale'],
            'max_depth': rhythm_hierarchical['max_depth'],
            'compression_ratio': rhythm_hierarchical['compression_ratio']
        },
        'contour': {
            'n_primitives': contour_hierarchical['n_primitives'],
            'n_composites': contour_hierarchical['n_composites'],
            'n_repeat': contour_hierarchical['n_repeat'],
            'n_concat': contour_hierarchical['n_concat'],
            'n_time_scale': contour_hierarchical['n_time_scale'],
            'max_depth': contour_hierarchical['max_depth'],
            'compression_ratio': contour_hierarchical['compression_ratio']
        }
    }

    # Feature 4: Factored vs Atomic comparison
    comparison = compare_factored_vs_atomic(factored, verbose=verbose)
    result.stats['factored_vs_atomic'] = {
        'factored_bits': comparison['factored']['compressed_bits'],
        'atomic_bits': comparison['atomic']['compressed_bits'],
        'factored_ratio': comparison['factored']['compression_ratio'],
        'atomic_ratio': comparison['atomic']['compression_ratio'],
        'winner': comparison['winner'],
        'advantage_ratio': comparison['factored_vs_atomic_ratio']
    }

    # Feature 5: Extended GPU-optimized discovery
    # This runs after within-component discovery to find:
    # - Time-shift relations (identical objects at different times)
    # - Compound transforms (transpose + time_shift + velocity_scale)
    # - Cross-component patterns (rhythm from one source, contour from another)
    extended_results = run_extended_discovery(factored, verbose=verbose)
    result.stats['extended_discovery'] = {
        'time_shifts': len(extended_results['time_shifts']),
        'compound_transforms': len(extended_results['compound_transforms']),
        'cross_components': len(extended_results['cross_components']),
        'n_objects': extended_results['n_objects'],
        'n_explained': extended_results.get('n_explained', 0)
    }

    # Feature 6: Reconstruction quality verification
    reconstruction_quality = measure_reconstruction_quality(factored, verbose=verbose)
    result.stats['reconstruction_quality'] = reconstruction_quality

    # Add reconstruction data to result
    result.reconstruction_data = get_reconstruction_data(factored)

    # Save checkpoint if requested
    if save_checkpoint_path:
        save_checkpoint(
            save_checkpoint_path,
            factored,
            rhythm_hierarchical=rhythm_hierarchical,
            contour_hierarchical=contour_hierarchical,
            verbose=verbose
        )

    return result
