"""
Differentiable MIDI Representation - Agent 2
============================================

SoftPianoRoll: Differentiable pianoroll representation for MIDI.

This module provides a continuous, differentiable representation of MIDI data
that can be used for training neural networks end-to-end. The pianoroll is
a time-pitch grid where active notes are represented as 1.0 and silence as 0.0.

Key Features:
- Continuous [0, 1] values during training (differentiable)
- Binary {0, 1} values during inference (discrete)
- Efficient conversion between MIDI and tensor formats
- Multi-track support (up to 16 tracks)
- Configurable time resolution and pitch range

Author: Agent 2 - Differentiable MIDI & Utilities Support
Date: November 22, 2025
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pretty_midi
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PianoRollConfig:
    """Configuration for pianoroll representation"""

    # Dimensions
    time_steps: int = 2048                  # Maximum time steps
    num_pitches: int = 88                   # Pitch range (default: piano 88 keys)
    pitch_offset: int = 21                  # MIDI note offset (21 = A0)
    num_tracks: int = 8                     # Number of simultaneous tracks

    # Time resolution
    time_resolution: float = 0.25           # Seconds per time step (0.25 = 16th note @ 120 BPM)
    fps: int = 4                            # Frames per second (1 / time_resolution)

    # MIDI generation
    default_velocity: int = 64              # Default velocity for generated notes
    min_note_duration: float = 0.05         # Minimum note duration (50ms)
    onset_threshold: float = 0.5            # Threshold for note onset detection

    # Instrument programs (GM) for each track
    default_programs: List[int] = field(default_factory=lambda: [0, 0, 32, 48, 56, 64, 71, 0])
    # Piano, Piano, Bass, Strings, Trumpet, Sax, Clarinet, Piano

    def __post_init__(self):
        """Validate configuration"""
        assert self.time_steps > 0, "time_steps must be positive"
        assert self.num_pitches > 0, "num_pitches must be positive"
        assert 0 <= self.pitch_offset <= 127, "pitch_offset must be in MIDI range"
        assert self.num_tracks > 0, "num_tracks must be positive"
        assert self.time_resolution > 0, "time_resolution must be positive"
        assert 1 <= self.default_velocity <= 127, "default_velocity must be in [1, 127]"
        assert self.min_note_duration > 0, "min_note_duration must be positive"
        assert 0 <= self.onset_threshold <= 1, "onset_threshold must be in [0, 1]"


# ============================================================================
# SoftPianoRoll Class
# ============================================================================

class SoftPianoRoll:
    """
    Differentiable pianoroll representation for MIDI.

    This class provides bidirectional conversion between MIDI files and
    differentiable tensor representations suitable for neural network training.

    Shape: (batch_size, num_tracks, time_steps, num_pitches)
    Values: Continuous [0, 1] during training, binary {0, 1} during inference

    Usage:
        # Load from MIDI
        pianoroll = SoftPianoRoll.from_midi('song.mid')
        tensor = pianoroll.to_tensor()  # (1, 8, 2048, 88)

        # Convert decoder output to MIDI
        decoder_output = decoder(dna)  # (batch, 8, 2048, 88)
        pianoroll = SoftPianoRoll(decoder_output, config)
        midi = pianoroll.to_midi()
        midi.write('generated.mid')
    """

    def __init__(
        self,
        data: Optional[torch.Tensor] = None,
        config: Optional[PianoRollConfig] = None,
        device: str = 'cpu'
    ):
        """
        Initialize SoftPianoRoll.

        Args:
            data: Optional tensor data (batch, tracks, time, pitch)
            config: PianoRoll configuration
            device: 'cpu' or 'cuda'
        """
        self.config = config or PianoRollConfig()
        self.device = device

        if data is None:
            # Initialize empty pianoroll
            self.data = torch.zeros(
                1,
                self.config.num_tracks,
                self.config.time_steps,
                self.config.num_pitches,
                device=device
            )
        else:
            # Use provided data
            self.data = data.to(device)
            self._validate_shape()

    def _validate_shape(self):
        """Validate tensor shape"""
        expected_dims = 4
        if len(self.data.shape) != expected_dims:
            raise ValueError(
                f"Expected {expected_dims}D tensor (batch, tracks, time, pitch), "
                f"got {len(self.data.shape)}D"
            )

        batch, tracks, time, pitch = self.data.shape
        if tracks > self.config.num_tracks:
            warnings.warn(f"Data has {tracks} tracks, config specifies {self.config.num_tracks}")
        if pitch != self.config.num_pitches:
            raise ValueError(
                f"Pitch dimension mismatch: data has {pitch}, config specifies {self.config.num_pitches}"
            )

    # ========================================================================
    # MIDI → Pianoroll Conversion
    # ========================================================================

    @classmethod
    def from_midi(
        cls,
        midi_path: Union[str, Path, pretty_midi.PrettyMIDI],
        config: Optional[PianoRollConfig] = None,
        max_time_steps: Optional[int] = None,
        device: str = 'cpu'
    ) -> 'SoftPianoRoll':
        """
        Create SoftPianoRoll from MIDI file.

        Args:
            midi_path: Path to MIDI file or PrettyMIDI object
            config: PianoRoll configuration
            max_time_steps: Maximum time steps (truncate if longer)
            device: 'cpu' or 'cuda'

        Returns:
            SoftPianoRoll object
        """
        config = config or PianoRollConfig()

        # Load MIDI
        if isinstance(midi_path, pretty_midi.PrettyMIDI):
            midi = midi_path
        else:
            try:
                midi = pretty_midi.PrettyMIDI(str(midi_path))
            except Exception as e:
                raise ValueError(f"Failed to load MIDI file: {e}")

        # Override time_steps if specified
        if max_time_steps is not None:
            config.time_steps = max_time_steps

        # Convert to pianoroll tensor
        pianoroll_tensor = cls._midi_to_pianoroll(midi, config)

        return cls(data=pianoroll_tensor, config=config, device=device)

    @staticmethod
    def _midi_to_pianoroll(
        midi: pretty_midi.PrettyMIDI,
        config: PianoRollConfig
    ) -> torch.Tensor:
        """
        Convert MIDI to pianoroll tensor.

        Algorithm:
        1. Extract all notes from all instruments
        2. Group notes by track/instrument
        3. For each note, set pianoroll[track, time_slice, pitch] = 1.0
        4. Return tensor

        Args:
            midi: PrettyMIDI object
            config: Configuration

        Returns:
            Tensor of shape (1, num_tracks, time_steps, num_pitches)
        """
        # Initialize pianoroll
        pianoroll = np.zeros((config.num_tracks, config.time_steps, config.num_pitches))

        # Process each instrument (track)
        num_instruments = min(len(midi.instruments), config.num_tracks)

        for track_idx in range(num_instruments):
            instrument = midi.instruments[track_idx]

            # Skip drum tracks if needed
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                # Convert time to time steps
                start_step = int(note.start / config.time_resolution)
                end_step = int(note.end / config.time_resolution)

                # Convert MIDI pitch to pianoroll pitch
                pitch_idx = note.pitch - config.pitch_offset

                # Validate ranges
                if not (0 <= pitch_idx < config.num_pitches):
                    continue  # Skip notes outside pitch range

                if start_step >= config.time_steps:
                    break  # Past end of pianoroll

                # Clip end step
                end_step = min(end_step, config.time_steps)

                # Set pianoroll values
                if end_step > start_step:
                    pianoroll[track_idx, start_step:end_step, pitch_idx] = 1.0

        # Convert to tensor and add batch dimension
        pianoroll_tensor = torch.from_numpy(pianoroll).float().unsqueeze(0)

        return pianoroll_tensor

    # ========================================================================
    # Pianoroll → MIDI Conversion
    # ========================================================================

    def to_midi(
        self,
        threshold: float = 0.5,
        min_duration: Optional[float] = None,
        default_velocity: Optional[int] = None,
        tempo: int = 120
    ) -> pretty_midi.PrettyMIDI:
        """
        Convert pianoroll to MIDI file.

        Args:
            threshold: Threshold for binarizing continuous values
            min_duration: Minimum note duration (seconds), uses config if None
            default_velocity: Default velocity, uses config if None
            tempo: Tempo in BPM

        Returns:
            pretty_midi.PrettyMIDI object
        """
        min_duration = min_duration or self.config.min_note_duration
        default_velocity = default_velocity or self.config.default_velocity

        # Get first item in batch
        pianoroll = self.data[0]  # (num_tracks, time_steps, num_pitches)

        # Create MIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        # Process each track
        for track_idx in range(pianoroll.shape[0]):
            # Get program for this track
            program = self.config.default_programs[track_idx] if track_idx < len(self.config.default_programs) else 0

            # Create instrument
            instrument = pretty_midi.Instrument(program=program)

            # Extract notes from this track
            track_roll = pianoroll[track_idx]  # (time_steps, num_pitches)

            # Convert to numpy for processing
            track_roll_np = track_roll.detach().cpu().numpy()

            # Find notes for each pitch
            for pitch_idx in range(self.config.num_pitches):
                pitch_roll = track_roll_np[:, pitch_idx]  # (time_steps,)

                # Find note boundaries (onset/offset)
                note_events = self._extract_notes_from_pitch_roll(
                    pitch_roll,
                    pitch_idx,
                    threshold,
                    min_duration
                )

                # Create Note objects
                for start_time, end_time in note_events:
                    note = pretty_midi.Note(
                        velocity=default_velocity,
                        pitch=pitch_idx + self.config.pitch_offset,
                        start=start_time,
                        end=end_time
                    )
                    instrument.notes.append(note)

            # Add instrument if it has notes
            if len(instrument.notes) > 0:
                midi.instruments.append(instrument)

        return midi

    def _extract_notes_from_pitch_roll(
        self,
        pitch_roll: np.ndarray,
        pitch_idx: int,
        threshold: float,
        min_duration: float
    ) -> List[Tuple[float, float]]:
        """
        Extract note events from a single pitch time series.

        Finds contiguous regions where pitch_roll > threshold.

        Args:
            pitch_roll: Time series for one pitch (time_steps,)
            pitch_idx: Pitch index (for validation)
            threshold: Threshold for note detection
            min_duration: Minimum note duration

        Returns:
            List of (start_time, end_time) tuples
        """
        # Binarize
        binary_roll = (pitch_roll > threshold).astype(int)

        # Find onset/offset transitions
        # Pad with zeros to detect start/end
        padded = np.pad(binary_roll, (1, 1), mode='constant', constant_values=0)
        diff = np.diff(padded)

        # Onsets: 0 → 1 (diff = 1)
        onsets = np.where(diff == 1)[0]
        # Offsets: 1 → 0 (diff = -1)
        offsets = np.where(diff == -1)[0]

        # Create note events
        note_events = []
        for onset_step, offset_step in zip(onsets, offsets):
            start_time = onset_step * self.config.time_resolution
            end_time = offset_step * self.config.time_resolution
            duration = end_time - start_time

            # Enforce minimum duration
            if duration >= min_duration:
                note_events.append((start_time, end_time))

        return note_events

    # ========================================================================
    # Tensor Operations
    # ========================================================================

    def to_tensor(self) -> torch.Tensor:
        """
        Get underlying tensor representation.

        Returns:
            torch.Tensor of shape (batch, num_tracks, time_steps, num_pitches)
        """
        return self.data

    def binarize(self, threshold: float = 0.5) -> 'SoftPianoRoll':
        """
        Binarize pianoroll (convert soft probabilities to hard 0/1).

        Args:
            threshold: Threshold for binarization

        Returns:
            New SoftPianoRoll with binary values
        """
        binary_data = (self.data > threshold).float()
        return SoftPianoRoll(data=binary_data, config=self.config, device=self.device)

    def clip(self, start_step: int = 0, end_step: Optional[int] = None) -> 'SoftPianoRoll':
        """
        Clip pianoroll to time range.

        Args:
            start_step: Start time step
            end_step: End time step (None = end of pianoroll)

        Returns:
            Clipped SoftPianoRoll
        """
        end_step = end_step or self.config.time_steps
        clipped_data = self.data[:, :, start_step:end_step, :]
        return SoftPianoRoll(data=clipped_data, config=self.config, device=self.device)

    # ========================================================================
    # Static Utilities
    # ========================================================================

    @staticmethod
    def get_note_events(
        pianoroll: torch.Tensor,
        config: PianoRollConfig,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Extract note events from pianoroll tensor.

        Args:
            pianoroll: Tensor of shape (num_tracks, time_steps, num_pitches)
            config: Configuration
            threshold: Threshold for note detection

        Returns:
            List of note dicts with keys: pitch, start_time, end_time, duration, velocity, track
        """
        pianoroll_np = pianoroll.detach().cpu().numpy()
        note_events = []

        for track_idx in range(pianoroll_np.shape[0]):
            for pitch_idx in range(pianoroll_np.shape[2]):
                pitch_roll = pianoroll_np[track_idx, :, pitch_idx]

                # Binarize
                binary_roll = (pitch_roll > threshold).astype(int)

                # Find transitions
                padded = np.pad(binary_roll, (1, 1), mode='constant', constant_values=0)
                diff = np.diff(padded)
                onsets = np.where(diff == 1)[0]
                offsets = np.where(diff == -1)[0]

                # Create events
                for onset_step, offset_step in zip(onsets, offsets):
                    start_time = onset_step * config.time_resolution
                    end_time = offset_step * config.time_resolution
                    duration = end_time - start_time

                    if duration >= config.min_note_duration:
                        note_events.append({
                            'pitch': pitch_idx + config.pitch_offset,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': duration,
                            'velocity': config.default_velocity,
                            'track': track_idx
                        })

        return note_events

    # ========================================================================
    # Special Methods
    # ========================================================================

    def __repr__(self) -> str:
        batch, tracks, time, pitch = self.data.shape
        return (
            f"SoftPianoRoll(batch={batch}, tracks={tracks}, "
            f"time_steps={time}, pitches={pitch}, device={self.device})"
        )

    def __len__(self) -> int:
        """Return batch size"""
        return self.data.shape[0]
