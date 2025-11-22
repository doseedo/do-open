"""
Rhythm Domain Transforms
========================

Space-level transforms operating on rhythm/timing dimension.

Transforms:
1. TempoTransform - Change tempo (BPM)
2. SwingTransform - Add/remove swing feel
3. SyncopationTransform - Increase/decrease syncopation
4. NoteDensityTransform - Change note density (notes per second)
5. QuantizeTransform - Quantize or humanize timing
6. GrooveTransform - Apply rhythmic grooves
7. RubattoTransform - Add tempo flexibility
8. PolyrhythmTransform - Add/remove polyrhythmic elements

Author: Agent 8 - Transform Architecture
"""

import copy
import numpy as np
from typing import List, Dict, Any, Tuple
import mido

from .space_level_transforms import (
    SpaceLevelTransform,
    TransformMetadata,
    extract_notes_from_midi,
    notes_to_midi,
    compute_tempo_bpm,
    set_tempo_bpm
)


# ============================================================================
# 1. Tempo Transform
# ============================================================================

class TempoTransform(SpaceLevelTransform):
    """
    Change tempo (BPM).

    Parameter mapping:
    - 0.0 → very slow (40 BPM)
    - 0.5 → moderate (120 BPM)
    - 1.0 → very fast (200 BPM)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='tempo',
            dimension='rhythm',
            level='section',
            description='Change tempo (BPM)'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply tempo change"""
        amount = self.validate_amount(amount)

        # Map [0,1] to [40, 200] BPM
        target_bpm = 40 + amount * 160

        # Get current tempo
        current_bpm = compute_tempo_bpm(midi)

        # Calculate time scaling factor
        time_scale = current_bpm / target_bpm

        # Extract and scale notes
        notes = extract_notes_from_midi(midi)
        for note in notes:
            note['start_time'] *= time_scale
            note['duration'] *= time_scale

        # Create new MIDI with new tempo
        new_midi = notes_to_midi(notes, midi.ticks_per_beat)
        return set_tempo_bpm(new_midi, target_bpm)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Get current tempo"""
        bpm = compute_tempo_bpm(midi)

        # Map to [0,1]
        amount = (bpm - 40) / 160
        return np.clip(amount, 0.0, 1.0)


# ============================================================================
# 2. Swing Transform
# ============================================================================

class SwingTransform(SpaceLevelTransform):
    """
    Add swing feel to straight rhythm.

    Swings eighth notes (or sixteenth notes).

    Parameter mapping:
    - 0.0 → straight (1:1 ratio)
    - 0.5 → medium swing (2:1 ratio, ~67%)
    - 1.0 → heavy swing (3:1 ratio, ~75%)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='swing',
            dimension='rhythm',
            level='phrase',
            description='Add/remove swing feel'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply swing"""
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return midi

        # Calculate swing ratio
        # 0.0 → 50% (straight), 0.5 → 67%, 1.0 → 75%
        swing_ratio = 0.5 + amount * 0.25

        # Detect beat grid (assume quarter note = 0.5s at 120 BPM)
        beat_duration = 0.5  # Approximate
        subdivision = beat_duration / 2  # Eighth note

        # Apply swing to off-beat notes
        for note in notes:
            # Find position within beat
            beat_position = (note['start_time'] % beat_duration) / beat_duration

            # If on off-beat (0.5 position), delay it
            if 0.4 < beat_position < 0.6:
                # This is an off-beat eighth note
                beat_start = note['start_time'] - (beat_position * beat_duration)
                # Delay off-beat by swing amount
                note['start_time'] = beat_start + (swing_ratio * beat_duration)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Estimate current swing amount"""
        notes = extract_notes_from_midi(midi)
        if len(notes) < 4:
            return 0.0

        # Analyze timing of eighth notes
        # This is a simplified heuristic
        sorted_notes = sorted(notes, key=lambda n: n['start_time'])
        iois = []
        for i in range(1, len(sorted_notes)):
            ioi = sorted_notes[i]['start_time'] - sorted_notes[i-1]['start_time']
            if 0.1 < ioi < 0.6:  # Eighth note range
                iois.append(ioi)

        if len(iois) < 2:
            return 0.0

        # Check for alternating long-short pattern
        # Swing creates uneven IOIs
        variance = np.var(iois)
        amount = min(variance * 10, 1.0)  # Heuristic mapping

        return amount


# ============================================================================
# 3. Syncopation Transform
# ============================================================================

class SyncopationTransform(SpaceLevelTransform):
    """
    Increase or decrease syncopation.

    Shifts notes off/onto strong beats.

    Parameter mapping:
    - 0.0 → minimal syncopation (on-beat)
    - 0.5 → preserve syncopation
    - 1.0 → maximum syncopation (heavily off-beat)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='syncopation',
            dimension='rhythm',
            level='phrase',
            description='Increase or decrease syncopation'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply syncopation"""
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return midi

        beat_duration = 0.5  # Quarter note at 120 BPM

        for note in notes:
            beat_position = (note['start_time'] % beat_duration) / beat_duration

            if amount > 0.5:
                # Increase syncopation: push notes off-beat
                syncopation_strength = (amount - 0.5) * 2
                if beat_position < 0.2:  # On-beat note
                    # Shift slightly off-beat
                    offset = syncopation_strength * 0.2 * beat_duration
                    note['start_time'] += offset

            else:
                # Decrease syncopation: pull notes to beat
                alignment_strength = (0.5 - amount) * 2
                # Quantize to nearest beat position
                nearest_beat = round(note['start_time'] / beat_duration) * beat_duration
                note['start_time'] = (
                    note['start_time'] * (1 - alignment_strength) +
                    nearest_beat * alignment_strength
                )

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Estimate syncopation level"""
        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return 0.5

        beat_duration = 0.5
        off_beat_count = 0

        for note in notes:
            beat_position = (note['start_time'] % beat_duration) / beat_duration
            # Off-beat is roughly 0.2-0.8 range
            if 0.2 < beat_position < 0.8:
                off_beat_count += 1

        off_beat_ratio = off_beat_count / len(notes)
        amount = 0.5 + (off_beat_ratio - 0.5)  # Center at 0.5

        return np.clip(amount, 0.0, 1.0)


# ============================================================================
# 4. Note Density Transform
# ============================================================================

class NoteDensityTransform(SpaceLevelTransform):
    """
    Change note density (notes per second).

    Parameter mapping:
    - 0.0 → very sparse (1 note/sec)
    - 0.5 → moderate (4 notes/sec)
    - 1.0 → very dense (12+ notes/sec)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='note_density',
            dimension='rhythm',
            level='section',
            description='Change note density',
            is_invertible=False
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply density change"""
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return midi

        # Calculate current density
        if len(notes) > 0:
            duration = max(n['start_time'] + n['duration'] for n in notes)
            current_density = len(notes) / max(duration, 1.0)
        else:
            current_density = 1.0

        # Target density
        target_density = 1.0 + amount * 11  # 1-12 notes/sec

        density_ratio = target_density / max(current_density, 0.1)

        if density_ratio > 1:
            # Increase density: duplicate notes
            new_notes = list(notes)
            num_to_add = int(len(notes) * (density_ratio - 1))

            for _ in range(num_to_add):
                # Duplicate random note with slight time offset
                original = notes[np.random.randint(len(notes))].copy()
                original['start_time'] += np.random.uniform(-0.1, 0.1)
                original['velocity'] = int(original['velocity'] * 0.8)
                new_notes.append(original)

            return notes_to_midi(new_notes, midi.ticks_per_beat)

        else:
            # Decrease density: remove notes
            num_to_keep = int(len(notes) * density_ratio)
            # Keep most salient notes (highest velocity)
            notes_sorted = sorted(notes, key=lambda n: n['velocity'], reverse=True)
            kept_notes = notes_sorted[:num_to_keep]

            return notes_to_midi(kept_notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Get current note density"""
        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return 0.5

        duration = max(n['start_time'] + n['duration'] for n in notes)
        density = len(notes) / max(duration, 1.0)

        # Map to [0,1]
        amount = (density - 1) / 11
        return np.clip(amount, 0.0, 1.0)


# ============================================================================
# 5. Quantize Transform
# ============================================================================

class QuantizeTransform(SpaceLevelTransform):
    """
    Quantize or humanize timing.

    Parameter mapping:
    - 0.0 → perfect quantization (mechanical)
    - 0.5 → preserve timing
    - 1.0 → humanized (natural variation)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='quantize',
            dimension='rhythm',
            level='note',
            description='Quantize or humanize timing'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply quantization/humanization"""
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return midi

        grid_size = 0.125  # 16th note at 120 BPM

        if amount < 0.5:
            # Quantize towards grid
            quantize_strength = (0.5 - amount) * 2

            for note in notes:
                # Find nearest grid position
                grid_pos = round(note['start_time'] / grid_size) * grid_size
                # Interpolate towards grid
                note['start_time'] = (
                    note['start_time'] * (1 - quantize_strength) +
                    grid_pos * quantize_strength
                )

        else:
            # Humanize: add random timing variations
            humanize_strength = (amount - 0.5) * 2
            max_deviation = humanize_strength * 0.03  # Up to 30ms

            for note in notes:
                deviation = np.random.normal(0, max_deviation)
                note['start_time'] = max(0, note['start_time'] + deviation)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Estimate quantization level"""
        notes = extract_notes_from_midi(midi)
        if len(notes) < 2:
            return 0.5

        grid_size = 0.125

        # Measure deviation from grid
        deviations = []
        for note in notes:
            nearest_grid = round(note['start_time'] / grid_size) * grid_size
            deviation = abs(note['start_time'] - nearest_grid)
            deviations.append(deviation)

        mean_deviation = np.mean(deviations)

        # Map to [0,1]: 0=perfect quantization, 1=highly humanized
        # Typical humanization is ~10-30ms
        amount = 0.5 + (mean_deviation / 0.03) * 0.5

        return np.clip(amount, 0.0, 1.0)


# ============================================================================
# 6. Groove Transform
# ============================================================================

class GrooveTransform(SpaceLevelTransform):
    """
    Apply rhythmic groove patterns.

    Parameter mapping:
    - 0.0 → minimal groove (even)
    - 0.5 → moderate groove
    - 1.0 → strong groove (funk/shuffle)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='groove',
            dimension='rhythm',
            level='phrase',
            description='Apply rhythmic groove'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply groove"""
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return midi

        # Define groove pattern (simplified)
        # Strong beats get emphasized, weak beats get delayed/softened
        beat_duration = 0.5

        for note in notes:
            beat_position = (note['start_time'] % beat_duration) / beat_duration

            # On downbeat (0.0): emphasize
            if beat_position < 0.1:
                note['velocity'] = min(127, int(note['velocity'] * (1 + 0.2 * amount)))

            # On off-beat (0.5): push slightly
            elif 0.4 < beat_position < 0.6:
                offset = amount * 0.05 * beat_duration
                note['start_time'] += offset
                note['velocity'] = int(note['velocity'] * (1 - 0.1 * amount))

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Estimate groove strength"""
        notes = extract_notes_from_midi(midi)
        if len(notes) < 4:
            return 0.5

        beat_duration = 0.5

        # Analyze velocity variations by beat position
        downbeat_velocities = []
        offbeat_velocities = []

        for note in notes:
            beat_position = (note['start_time'] % beat_duration) / beat_duration
            if beat_position < 0.1:
                downbeat_velocities.append(note['velocity'])
            elif 0.4 < beat_position < 0.6:
                offbeat_velocities.append(note['velocity'])

        if not downbeat_velocities or not offbeat_velocities:
            return 0.5

        # Groove creates larger difference between downbeat and offbeat
        velocity_contrast = (
            np.mean(downbeat_velocities) - np.mean(offbeat_velocities)
        ) / 127

        amount = 0.5 + velocity_contrast
        return np.clip(amount, 0.0, 1.0)


# ============================================================================
# 7. Rubato Transform
# ============================================================================

class RubatoTransform(SpaceLevelTransform):
    """
    Add tempo flexibility (rubato).

    Parameter mapping:
    - 0.0 → strict tempo
    - 0.5 → subtle rubato
    - 1.0 → extreme rubato (expressive)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='rubato',
            dimension='rhythm',
            level='phrase',
            description='Add tempo flexibility'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply rubato"""
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if len(notes) < 2:
            return midi

        # Sort by time
        notes_sorted = sorted(notes, key=lambda n: n['start_time'])

        # Apply smooth tempo curve
        # Slow down at phrase ends, speed up in middle
        total_duration = notes_sorted[-1]['start_time']
        if total_duration == 0:
            return midi

        for note in notes_sorted:
            # Calculate position in phrase (0-1)
            position = note['start_time'] / total_duration

            # Tempo curve: slow at end, fast in middle
            # Using sine curve
            tempo_factor = 1.0 + amount * 0.3 * np.sin(position * np.pi)

            note['start_time'] *= tempo_factor
            note['duration'] *= tempo_factor

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Estimate rubato amount"""
        notes = extract_notes_from_midi(midi)
        if len(notes) < 3:
            return 0.0

        # Analyze IOI (inter-onset interval) variance
        notes_sorted = sorted(notes, key=lambda n: n['start_time'])
        iois = []
        for i in range(1, len(notes_sorted)):
            ioi = notes_sorted[i]['start_time'] - notes_sorted[i-1]['start_time']
            if ioi > 0:
                iois.append(ioi)

        if len(iois) < 2:
            return 0.0

        # Higher variance = more rubato
        variance = np.var(iois) / (np.mean(iois) ** 2)  # Coefficient of variation
        amount = min(variance * 2, 1.0)

        return amount


# ============================================================================
# 8. Polyrhythm Transform
# ============================================================================

class PolyrhythmTransform(SpaceLevelTransform):
    """
    Add polyrhythmic elements.

    Parameter mapping:
    - 0.0 → single rhythm
    - 0.5 → moderate polyrhythm
    - 1.0 → complex polyrhythm (3:2, 4:3, etc.)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='polyrhythm',
            dimension='rhythm',
            level='section',
            description='Add polyrhythmic patterns',
            is_invertible=False
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply polyrhythm"""
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if len(notes) < 4:
            return midi

        if amount < 0.3:
            # No significant polyrhythm
            return midi

        # Add a secondary rhythm layer
        # Choose subset of notes to shift to polyrhythmic grid
        polyrhythm_ratio = 1.5 if amount > 0.7 else 1.33  # 3:2 or 4:3

        # Group notes by track
        track_notes = {}
        for note in notes:
            track = note.get('track', 0)
            if track not in track_notes:
                track_notes[track] = []
            track_notes[track].append(note)

        # Apply polyrhythm to one track
        if len(track_notes) > 1:
            secondary_track = list(track_notes.keys())[1]
            for note in track_notes[secondary_track]:
                # Shift to polyrhythmic grid
                note['start_time'] *= polyrhythm_ratio

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Estimate polyrhythm complexity"""
        notes = extract_notes_from_midi(midi)
        if len(notes) < 4:
            return 0.0

        # Analyze rhythmic complexity across tracks
        # Simplified: return 0 (detecting polyrhythm is complex)
        return 0.0
