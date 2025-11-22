"""
MIDI Assembly - Agent 2
=======================

MIDIAssembler: Convert decoder outputs to valid MIDI files.

This module assembles MIDI files from neural network decoder outputs, handling:
- Multi-modal predictions (pitch, onset, duration, velocity)
- Pianoroll representations
- Multi-track coordination
- Post-processing and validation

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
from dataclasses import dataclass
import warnings


# ============================================================================
# MIDI Assembler
# ============================================================================

class MIDIAssembler:
    """
    Assemble MIDI from decoder predictions.

    Handles two primary assembly modes:
    1. Pianoroll-based: Direct conversion from time-pitch grid
    2. Multi-modal: Combine separate pitch, onset, duration, velocity predictions

    Usage:
        # From pianoroll
        assembler = MIDIAssembler(tempo=120)
        midi = assembler.assemble_from_pianoroll(pianoroll_tensor)

        # From multi-modal outputs
        midi = assembler.assemble(
            pitch_probs=decoder_pitch,
            onset_probs=decoder_onset,
            duration_values=decoder_duration,
            velocity_values=decoder_velocity
        )
    """

    def __init__(
        self,
        tempo: int = 120,
        time_signature: Tuple[int, int] = (4, 4),
        time_resolution: float = 0.25,
        pitch_offset: int = 21,
        num_pitches: int = 88,
        min_note_duration: float = 0.05,
        default_programs: Optional[List[int]] = None,
        default_velocity: int = 64
    ):
        """
        Initialize MIDI assembler.

        Args:
            tempo: BPM (default 120)
            time_signature: (numerator, denominator) default 4/4
            time_resolution: Seconds per time step
            pitch_offset: MIDI note offset (21 = A0)
            num_pitches: Number of pitches in pianoroll
            min_note_duration: Minimum note duration in seconds
            default_programs: GM program numbers for tracks
            default_velocity: Default velocity for generated notes
        """
        self.tempo = tempo
        self.time_signature = time_signature
        self.time_resolution = time_resolution
        self.pitch_offset = pitch_offset
        self.num_pitches = num_pitches
        self.min_note_duration = min_note_duration
        self.default_velocity = default_velocity

        # Default instrument programs (Piano, Bass, Strings, Trumpet, Sax, ...)
        self.default_programs = default_programs or [0, 32, 48, 56, 64, 71, 0, 0]

    # ========================================================================
    # Pianoroll-Based Assembly
    # ========================================================================

    def assemble_from_pianoroll(
        self,
        pianoroll: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> pretty_midi.PrettyMIDI:
        """
        Assemble MIDI from pianoroll representation.

        This is the simpler interface when using pianoroll, since onset + pitch + duration
        are all encoded in the pianoroll grid.

        Args:
            pianoroll: Tensor of shape (num_tracks, time_steps, num_pitches)
                      or (batch, num_tracks, time_steps, num_pitches)
            velocities: Optional velocity for each note (same shape as pianoroll)
            threshold: Threshold for note detection

        Returns:
            pretty_midi.PrettyMIDI object
        """
        # Handle batch dimension
        if len(pianoroll.shape) == 4:
            pianoroll = pianoroll[0]  # Take first in batch

        if velocities is not None and len(velocities.shape) == 4:
            velocities = velocities[0]

        # Convert to numpy
        pianoroll_np = pianoroll.detach().cpu().numpy()
        if velocities is not None:
            velocities_np = velocities.detach().cpu().numpy()
        else:
            velocities_np = None

        # Create MIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)

        num_tracks = pianoroll_np.shape[0]

        # Process each track
        for track_idx in range(num_tracks):
            program = self.default_programs[track_idx] if track_idx < len(self.default_programs) else 0
            instrument = pretty_midi.Instrument(program=program)

            track_roll = pianoroll_np[track_idx]  # (time_steps, num_pitches)

            # Extract notes for each pitch
            for pitch_idx in range(track_roll.shape[1]):
                pitch_roll = track_roll[:, pitch_idx]  # (time_steps,)

                # Find note boundaries
                note_events = self._extract_notes_from_roll(
                    pitch_roll,
                    pitch_idx,
                    threshold
                )

                # Create Note objects
                for start_step, end_step in note_events:
                    start_time = start_step * self.time_resolution
                    end_time = end_step * self.time_resolution
                    duration = end_time - start_time

                    # Enforce minimum duration
                    if duration < self.min_note_duration:
                        continue

                    # Get velocity
                    if velocities_np is not None:
                        # Average velocity over note duration
                        vel_slice = velocities_np[track_idx, start_step:end_step, pitch_idx]
                        velocity = int(np.clip(vel_slice.mean() * 127, 1, 127))
                    else:
                        velocity = self.default_velocity

                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch_idx + self.pitch_offset,
                        start=start_time,
                        end=end_time
                    )
                    instrument.notes.append(note)

            # Add instrument if it has notes
            if len(instrument.notes) > 0:
                midi.instruments.append(instrument)

        return midi

    def _extract_notes_from_roll(
        self,
        pitch_roll: np.ndarray,
        pitch_idx: int,
        threshold: float
    ) -> List[Tuple[int, int]]:
        """
        Extract note events (in time steps) from pitch roll.

        Args:
            pitch_roll: (time_steps,) array
            pitch_idx: Pitch index
            threshold: Detection threshold

        Returns:
            List of (start_step, end_step) tuples
        """
        # Binarize
        binary = (pitch_roll > threshold).astype(int)

        # Find transitions
        padded = np.pad(binary, (1, 1), mode='constant', constant_values=0)
        diff = np.diff(padded)

        onsets = np.where(diff == 1)[0]
        offsets = np.where(diff == -1)[0]

        # Pair onsets with offsets
        note_events = list(zip(onsets, offsets))

        return note_events

    # ========================================================================
    # Multi-Modal Assembly
    # ========================================================================

    def assemble(
        self,
        pitch_probs: torch.Tensor,      # (num_tracks, time_steps, num_pitches)
        onset_probs: torch.Tensor,      # (num_tracks, time_steps)
        duration_values: torch.Tensor,  # (num_tracks, time_steps)
        velocity_values: torch.Tensor,  # (num_tracks, time_steps)
        onset_threshold: float = 0.5,
        pitch_threshold: float = 0.5
    ) -> pretty_midi.PrettyMIDI:
        """
        Assemble MIDI from separate multi-modal predictions.

        This method is used when the decoder outputs separate streams for
        pitch, onset, duration, and velocity.

        Args:
            pitch_probs: Pitch probabilities (can be soft or one-hot)
            onset_probs: Note onset probabilities
            duration_values: Note durations in seconds (continuous)
            velocity_values: Note velocities [0, 1] (will be scaled to [1, 127])
            onset_threshold: Threshold for onset detection
            pitch_threshold: Threshold for pitch selection

        Returns:
            pretty_midi.PrettyMIDI object
        """
        # Handle batch dimension
        if len(pitch_probs.shape) == 4:
            pitch_probs = pitch_probs[0]
            onset_probs = onset_probs[0] if len(onset_probs.shape) == 3 else onset_probs
            duration_values = duration_values[0] if len(duration_values.shape) == 3 else duration_values
            velocity_values = velocity_values[0] if len(velocity_values.shape) == 3 else velocity_values

        # Convert to numpy
        pitch_probs_np = pitch_probs.detach().cpu().numpy()
        onset_probs_np = onset_probs.detach().cpu().numpy()
        duration_values_np = duration_values.detach().cpu().numpy()
        velocity_values_np = velocity_values.detach().cpu().numpy()

        # Create MIDI
        midi = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)

        num_tracks = pitch_probs_np.shape[0]

        for track_idx in range(num_tracks):
            program = self.default_programs[track_idx] if track_idx < len(self.default_programs) else 0
            instrument = pretty_midi.Instrument(program=program)

            # Extract notes for this track
            notes = self._extract_notes_multimodal(
                pitch_probs_np[track_idx],
                onset_probs_np[track_idx],
                duration_values_np[track_idx],
                velocity_values_np[track_idx],
                onset_threshold,
                pitch_threshold
            )

            # Add notes to instrument
            for note_dict in notes:
                note = pretty_midi.Note(
                    velocity=note_dict['velocity'],
                    pitch=note_dict['pitch'],
                    start=note_dict['start'],
                    end=note_dict['end']
                )
                instrument.notes.append(note)

            if len(instrument.notes) > 0:
                midi.instruments.append(instrument)

        return midi

    def _extract_notes_multimodal(
        self,
        pitch_probs: np.ndarray,      # (time_steps, num_pitches)
        onset_probs: np.ndarray,      # (time_steps,)
        duration_values: np.ndarray,  # (time_steps,)
        velocity_values: np.ndarray,  # (time_steps,)
        onset_threshold: float,
        pitch_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Extract notes from multi-modal predictions.

        Args:
            pitch_probs: Pitch probabilities at each time step
            onset_probs: Onset probabilities
            duration_values: Predicted durations
            velocity_values: Predicted velocities [0, 1]
            onset_threshold: Threshold for onset
            pitch_threshold: Threshold for pitch

        Returns:
            List of note dictionaries
        """
        notes = []

        # Find onsets
        onset_steps = np.where(onset_probs > onset_threshold)[0]

        for onset_step in onset_steps:
            # Get pitch at this time step (argmax or threshold)
            pitch_dist = pitch_probs[onset_step]

            # Select pitch (can use argmax or threshold)
            pitch_idx = np.argmax(pitch_dist)

            # Check if pitch probability exceeds threshold
            if pitch_dist[pitch_idx] < pitch_threshold:
                continue

            # Get duration
            duration = float(duration_values[onset_step])

            # Enforce minimum duration
            if duration < self.min_note_duration:
                duration = self.min_note_duration

            # Get velocity
            velocity_raw = float(velocity_values[onset_step])
            velocity = int(np.clip(velocity_raw * 127, 1, 127))

            # Create note
            start_time = onset_step * self.time_resolution
            end_time = start_time + duration

            notes.append({
                'pitch': pitch_idx + self.pitch_offset,
                'start': start_time,
                'end': end_time,
                'velocity': velocity
            })

        return notes

    # ========================================================================
    # Post-Processing
    # ========================================================================

    @staticmethod
    def post_process(
        midi: pretty_midi.PrettyMIDI,
        remove_overlap: bool = True,
        quantize_timing: bool = False,
        quantize_grid: float = 0.125,
        sort_notes: bool = True
    ) -> pretty_midi.PrettyMIDI:
        """
        Post-process generated MIDI.

        Args:
            midi: Input MIDI
            remove_overlap: Remove overlapping notes on same pitch
            quantize_timing: Quantize note onsets to grid
            quantize_grid: Grid size in seconds (0.125 = 32nd note @ 120 BPM)
            sort_notes: Sort notes by start time

        Returns:
            Post-processed MIDI
        """
        for instrument in midi.instruments:
            # Sort notes
            if sort_notes:
                instrument.notes.sort(key=lambda n: (n.start, n.pitch))

            # Remove overlaps
            if remove_overlap:
                instrument.notes = MIDIAssembler._remove_overlapping_notes(instrument.notes)

            # Quantize timing
            if quantize_timing:
                for note in instrument.notes:
                    note.start = round(note.start / quantize_grid) * quantize_grid
                    note.end = round(note.end / quantize_grid) * quantize_grid

                    # Ensure end > start
                    if note.end <= note.start:
                        note.end = note.start + quantize_grid

        return midi

    @staticmethod
    def _remove_overlapping_notes(notes: List[pretty_midi.Note]) -> List[pretty_midi.Note]:
        """
        Remove overlapping notes on the same pitch.

        Args:
            notes: List of notes (assumed sorted by start time)

        Returns:
            Cleaned list of notes
        """
        if len(notes) == 0:
            return notes

        # Group by pitch
        pitch_groups = {}
        for note in notes:
            if note.pitch not in pitch_groups:
                pitch_groups[note.pitch] = []
            pitch_groups[note.pitch].append(note)

        cleaned_notes = []

        # Process each pitch separately
        for pitch, pitch_notes in pitch_groups.items():
            # Sort by start time
            pitch_notes.sort(key=lambda n: n.start)

            current_notes = []
            for note in pitch_notes:
                if current_notes and note.start < current_notes[-1].end:
                    # Overlap detected
                    # Option 1: Extend previous note
                    current_notes[-1].end = max(current_notes[-1].end, note.end)
                    # Option 2: Skip new note (uncomment to use)
                    # continue
                else:
                    current_notes.append(note)

            cleaned_notes.extend(current_notes)

        return cleaned_notes


# ============================================================================
# Utilities
# ============================================================================

def demo_assembly():
    """
    Demonstrate MIDI assembly from synthetic decoder outputs.
    """
    print("MIDI Assembly Demo")
    print("=" * 60)

    # Create synthetic decoder outputs
    num_tracks = 4
    time_steps = 256  # ~1 minute at 0.25s resolution
    num_pitches = 88

    # Random pianoroll
    pianoroll = torch.rand(num_tracks, time_steps, num_pitches)
    pianoroll = (pianoroll > 0.95).float()  # Sparse notes

    # Assemble
    assembler = MIDIAssembler(tempo=120)
    midi = assembler.assemble_from_pianoroll(pianoroll, threshold=0.5)

    # Post-process
    midi = MIDIAssembler.post_process(midi, remove_overlap=True, quantize_timing=True)

    print(f"Generated MIDI:")
    print(f"  Duration: {midi.get_end_time():.2f} seconds")
    print(f"  Instruments: {len(midi.instruments)}")
    for i, inst in enumerate(midi.instruments):
        print(f"    Track {i}: {len(inst.notes)} notes, program {inst.program}")

    # Save
    output_path = "/tmp/assembled_demo.mid"
    midi.write(output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    demo_assembly()
