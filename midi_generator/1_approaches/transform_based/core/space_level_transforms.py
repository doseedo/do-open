"""
Space-Level Musical Transforms
===============================

Universal transform system for MIDI manipulation.

Each transform:
- Works on ANY MIDI input (universal applicability)
- Has continuous parameter in [0,1] range
- Operates at specific abstraction level (note, phrase, section)
- Is compositionally applicable (can combine transforms)
- Is invertible where possible

Transform Dimensions:
- Pitch: transpose, interval scaling, voice spread, register shift
- Rhythm: tempo, syncopation, density, swing, groove
- Harmony: complexity, tension, extensions, voice leading
- Texture: polyphony, spacing, doubling, orchestration
- Form: structure, development, repetition, variation

Author: Agent 8 - Transform Architecture
Phase: 1 (Foundation)
"""

import copy
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    warnings.warn("mido not available. Install with: pip install mido")


# ============================================================================
# Base Classes
# ============================================================================

@dataclass
class TransformMetadata:
    """Metadata for a transform"""
    name: str
    dimension: str  # pitch, rhythm, harmony, texture, form
    level: str  # note, phrase, section
    description: str
    parameter_range: Tuple[float, float] = (0.0, 1.0)
    default_value: float = 0.5
    is_invertible: bool = True


class SpaceLevelTransform(ABC):
    """
    Base class for all space-level transforms.

    Each transform operates on MIDI data at a specific abstraction level
    and dimension, with a continuous parameter in [0,1].

    Design Principles:
    1. Universal: Works on any MIDI input
    2. Continuous: Parameter varies smoothly from 0 to 1
    3. Interpretable: Parameter meaning is clear
    4. Compositional: Can combine with other transforms
    5. Efficient: Fast enough for real-time editing
    """

    def __init__(self, metadata: TransformMetadata):
        """
        Initialize transform with metadata.

        Args:
            metadata: TransformMetadata describing the transform
        """
        if not MIDO_AVAILABLE:
            raise ImportError("mido required. Install with: pip install mido")

        self.metadata = metadata
        self.name = metadata.name
        self.dimension = metadata.dimension
        self.level = metadata.level
        self.description = metadata.description

    @abstractmethod
    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """
        Apply transform to MIDI file.

        Args:
            midi: Input MIDI file
            amount: Transform amount in [0,1]
                   - 0.5 typically means "no change" or "neutral"
                   - 0.0 means "minimum" or "decrease"
                   - 1.0 means "maximum" or "increase"

        Returns:
            Transformed MIDI file (new object)
        """
        pass

    @abstractmethod
    def get_current_value(self, midi: mido.MidiFile) -> float:
        """
        Extract current parameter value from MIDI file.

        This is the "analysis" or "encoder" direction: given MIDI,
        what is the current value of this transform parameter?

        Args:
            midi: MIDI file to analyze

        Returns:
            Current parameter value in [0,1]
        """
        pass

    def validate_amount(self, amount: float) -> float:
        """
        Validate and clamp amount to valid range.

        Args:
            amount: Input amount

        Returns:
            Clamped amount in [0,1]
        """
        return np.clip(amount, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', dimension='{self.dimension}', level='{self.level}')"


# ============================================================================
# Utility Functions
# ============================================================================

def extract_notes_from_midi(midi: mido.MidiFile) -> List[Dict[str, Any]]:
    """
    Extract note events from MIDI file.

    Args:
        midi: MIDI file

    Returns:
        List of note dictionaries with:
        - pitch: MIDI note number (0-127)
        - velocity: Note velocity (0-127)
        - start_time: Onset time in seconds
        - duration: Note duration in seconds
        - track: Track index
    """
    notes = []
    current_time = [0.0] * len(midi.tracks)
    tempo = 500000  # Default tempo (120 BPM)
    active_notes = {}  # (track, pitch) -> start_time

    for track_idx, track in enumerate(midi.tracks):
        for msg in track:
            # Update time
            current_time[track_idx] += mido.tick2second(
                msg.time, midi.ticks_per_beat, tempo
            )

            # Update tempo
            if msg.type == 'set_tempo':
                tempo = msg.tempo

            # Note on
            elif msg.type == 'note_on' and msg.velocity > 0:
                key = (track_idx, msg.note)
                active_notes[key] = {
                    'start_time': current_time[track_idx],
                    'velocity': msg.velocity
                }

            # Note off (or note_on with velocity 0)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (track_idx, msg.note)
                if key in active_notes:
                    note_info = active_notes[key]
                    notes.append({
                        'pitch': msg.note,
                        'velocity': note_info['velocity'],
                        'start_time': note_info['start_time'],
                        'duration': current_time[track_idx] - note_info['start_time'],
                        'track': track_idx
                    })
                    del active_notes[key]

    return notes


def notes_to_midi(
    notes: List[Dict[str, Any]],
    ticks_per_beat: int = 480,
    tempo: int = 500000
) -> mido.MidiFile:
    """
    Convert note list back to MIDI file.

    Args:
        notes: List of note dictionaries
        ticks_per_beat: MIDI ticks per beat
        tempo: Tempo in microseconds per beat

    Returns:
        MIDI file
    """
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    # Group notes by track
    tracks_notes = {}
    for note in notes:
        track_idx = note.get('track', 0)
        if track_idx not in tracks_notes:
            tracks_notes[track_idx] = []
        tracks_notes[track_idx].append(note)

    # Create tracks
    for track_idx in sorted(tracks_notes.keys()):
        track = mido.MidiTrack()
        midi.tracks.append(track)

        # Add tempo (only in first track)
        if track_idx == 0:
            track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

        # Sort notes by start time
        track_notes = sorted(tracks_notes[track_idx], key=lambda n: n['start_time'])

        # Create note on/off events
        events = []
        for note in track_notes:
            events.append({
                'time': note['start_time'],
                'type': 'note_on',
                'note': note['pitch'],
                'velocity': note['velocity']
            })
            events.append({
                'time': note['start_time'] + note['duration'],
                'type': 'note_off',
                'note': note['pitch'],
                'velocity': 0
            })

        # Sort all events by time
        events.sort(key=lambda e: e['time'])

        # Convert to MIDI messages with delta times
        current_time = 0.0
        for event in events:
            delta_time = event['time'] - current_time
            delta_ticks = mido.second2tick(delta_time, ticks_per_beat, tempo)

            if event['type'] == 'note_on':
                track.append(mido.Message(
                    'note_on',
                    note=event['note'],
                    velocity=event['velocity'],
                    time=int(delta_ticks)
                ))
            else:
                track.append(mido.Message(
                    'note_off',
                    note=event['note'],
                    velocity=0,
                    time=int(delta_ticks)
                ))

            current_time = event['time']

        # End of track
        track.append(mido.MetaMessage('end_of_track', time=0))

    return midi


def compute_tempo_bpm(midi: mido.MidiFile) -> float:
    """
    Extract tempo in BPM from MIDI file.

    Args:
        midi: MIDI file

    Returns:
        Tempo in BPM
    """
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return mido.tempo2bpm(msg.tempo)
    return 120.0  # Default


def set_tempo_bpm(midi: mido.MidiFile, bpm: float) -> mido.MidiFile:
    """
    Set tempo in MIDI file.

    Args:
        midi: MIDI file
        bpm: Tempo in BPM

    Returns:
        MIDI file with updated tempo
    """
    midi_copy = copy.deepcopy(midi)
    tempo = mido.bpm2tempo(bpm)

    for track in midi_copy.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                msg.tempo = int(tempo)
                return midi_copy

    # If no tempo found, add to first track
    if len(midi_copy.tracks) > 0:
        midi_copy.tracks[0].insert(0, mido.MetaMessage('set_tempo', tempo=int(tempo), time=0))

    return midi_copy


# ============================================================================
# Transform Utilities
# ============================================================================

class TransformChain:
    """
    Chain multiple transforms together.

    Allows compositional application of transforms.
    """

    def __init__(self, transforms: Optional[List[SpaceLevelTransform]] = None):
        """
        Initialize transform chain.

        Args:
            transforms: List of transforms to chain
        """
        self.transforms = transforms or []

    def add(self, transform: SpaceLevelTransform):
        """Add transform to chain"""
        self.transforms.append(transform)

    def apply(self, midi: mido.MidiFile, amounts: List[float]) -> mido.MidiFile:
        """
        Apply all transforms in sequence.

        Args:
            midi: Input MIDI
            amounts: List of amounts (one per transform)

        Returns:
            Transformed MIDI
        """
        if len(amounts) != len(self.transforms):
            raise ValueError(f"Expected {len(self.transforms)} amounts, got {len(amounts)}")

        result = midi
        for transform, amount in zip(self.transforms, amounts):
            result = transform.apply(result, amount)

        return result

    def get_current_values(self, midi: mido.MidiFile) -> List[float]:
        """
        Get current values for all transforms.

        Args:
            midi: MIDI file

        Returns:
            List of current values
        """
        return [transform.get_current_value(midi) for transform in self.transforms]


def interpolate_transforms(
    midi1: mido.MidiFile,
    midi2: mido.MidiFile,
    transform: SpaceLevelTransform,
    t: float
) -> mido.MidiFile:
    """
    Interpolate between two MIDI files using a transform.

    Args:
        midi1: First MIDI file
        midi2: Second MIDI file
        transform: Transform to use for interpolation
        t: Interpolation factor (0=midi1, 1=midi2)

    Returns:
        Interpolated MIDI file
    """
    # Get parameter values for both
    value1 = transform.get_current_value(midi1)
    value2 = transform.get_current_value(midi2)

    # Interpolate
    value_interp = (1 - t) * value1 + t * value2

    # Apply to first MIDI
    return transform.apply(midi1, value_interp)
