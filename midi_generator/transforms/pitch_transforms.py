"""
Pitch Domain Transforms
=======================

Space-level transforms operating on pitch/melody dimension.

Transforms:
1. TransposeTransform - Shift all pitches up/down
2. IntervalScaleTransform - Compress/expand melodic intervals
3. VoiceSpreadTransform - Spread voices across octaves
4. RegisterShiftTransform - Shift overall register (tessitura)
5. PitchRangeTransform - Compress/expand total pitch range
6. MelodicContourTransform - Smooth/exaggerate melodic motion
7. OctaveDoublingTransform - Add/remove octave doublings
8. MicrotonalDetuneTransform - Add microtonal deviations

Author: Agent 8 - Transform Architecture
"""

import copy
import numpy as np
from typing import List, Dict, Any
import mido

from .space_level_transforms import (
    SpaceLevelTransform,
    TransformMetadata,
    extract_notes_from_midi,
    notes_to_midi
)


# ============================================================================
# 1. Transpose Transform
# ============================================================================

class TransposeTransform(SpaceLevelTransform):
    """
    Transpose all pitches by semitones.

    Parameter mapping:
    - 0.0 → -12 semitones (down an octave)
    - 0.5 → 0 semitones (no change)
    - 1.0 → +12 semitones (up an octave)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='transpose',
            dimension='pitch',
            level='note',
            description='Transpose all pitches by semitones',
            default_value=0.5
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply transposition"""
        amount = self.validate_amount(amount)

        # Map [0,1] to [-12, +12] semitones
        semitones = int((amount - 0.5) * 24)

        midi_copy = copy.deepcopy(midi)

        for track in midi_copy.tracks:
            for msg in track:
                if msg.type == 'note_on' or msg.type == 'note_off':
                    msg.note = np.clip(msg.note + semitones, 0, 127)

        return midi_copy

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Get average pitch relative to C4 (60)"""
        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return 0.5

        # Calculate mean pitch
        mean_pitch = np.mean([n['pitch'] for n in notes])

        # Map to [0,1] assuming typical range is C3-C5 (48-72)
        # 0.5 corresponds to C4 (60)
        semitones_from_c4 = mean_pitch - 60
        amount = 0.5 + (semitones_from_c4 / 24)

        return np.clip(amount, 0.0, 1.0)


# ============================================================================
# 2. Interval Scale Transform
# ============================================================================

class IntervalScaleTransform(SpaceLevelTransform):
    """
    Scale melodic intervals (compress/expand melody).

    Parameter mapping:
    - 0.0 → compress to unison (0x intervals)
    - 0.5 → preserve original intervals (1x)
    - 1.0 → expand intervals (2x)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='interval_scale',
            dimension='pitch',
            level='phrase',
            description='Compress or expand melodic intervals'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply interval scaling"""
        amount = self.validate_amount(amount)

        # Map [0,1] to [0, 2] (scale factor)
        scale_factor = amount * 2

        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return midi

        # Calculate reference pitch (mean)
        reference_pitch = np.mean([n['pitch'] for n in notes])

        # Scale intervals around reference
        for note in notes:
            interval = note['pitch'] - reference_pitch
            note['pitch'] = int(reference_pitch + interval * scale_factor)
            note['pitch'] = np.clip(note['pitch'], 0, 127)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Estimate current interval scale"""
        notes = extract_notes_from_midi(midi)
        if len(notes) < 2:
            return 0.5

        # Calculate mean absolute interval
        pitches = sorted([n['pitch'] for n in notes])
        intervals = np.abs(np.diff(pitches))
        mean_interval = np.mean(intervals)

        # Map to [0,1] assuming typical mean interval is 2-3 semitones
        # 0.5 corresponds to ~3 semitones
        amount = mean_interval / 6
        return np.clip(amount, 0.0, 1.0)


# ============================================================================
# 3. Voice Spread Transform
# ============================================================================

class VoiceSpreadTransform(SpaceLevelTransform):
    """
    Spread voices across wider/narrower pitch range.

    Works on polyphonic music by spreading simultaneous notes.

    Parameter mapping:
    - 0.0 → compress to narrow range
    - 0.5 → preserve original spacing
    - 1.0 → spread across full range
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='voice_spread',
            dimension='pitch',
            level='phrase',
            description='Spread voices across wider/narrower range'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply voice spreading"""
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return midi

        # Group notes by time window (100ms = simultaneous)
        time_window = 0.1
        chord_groups = []
        current_group = []
        current_time = None

        for note in sorted(notes, key=lambda n: n['start_time']):
            if current_time is None or abs(note['start_time'] - current_time) < time_window:
                current_group.append(note)
                current_time = note['start_time'] if current_time is None else current_time
            else:
                if len(current_group) > 1:
                    chord_groups.append(current_group)
                current_group = [note]
                current_time = note['start_time']

        if len(current_group) > 1:
            chord_groups.append(current_group)

        # Spread each chord
        for group in chord_groups:
            if len(group) < 2:
                continue

            pitches = [n['pitch'] for n in group]
            min_pitch = min(pitches)
            max_pitch = max(pitches)
            center = (min_pitch + max_pitch) / 2

            # Calculate spread factor
            # amount=0.5 → factor=1 (no change)
            # amount=0 → factor=0.3 (compress)
            # amount=1 → factor=2 (expand)
            spread_factor = 0.3 + amount * 1.7

            # Spread around center
            for note in group:
                interval = note['pitch'] - center
                note['pitch'] = int(center + interval * spread_factor)
                note['pitch'] = np.clip(note['pitch'], 0, 127)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Estimate current voice spread"""
        notes = extract_notes_from_midi(midi)
        if len(notes) < 2:
            return 0.5

        # Calculate pitch range
        pitches = [n['pitch'] for n in notes]
        pitch_range = max(pitches) - min(pitches)

        # Map to [0,1] assuming typical range is 12-48 semitones
        amount = (pitch_range - 12) / 36
        return np.clip(amount, 0.0, 1.0)


# ============================================================================
# 4. Register Shift Transform
# ============================================================================

class RegisterShiftTransform(SpaceLevelTransform):
    """
    Shift overall register (tessitura) while preserving intervals.

    Similar to transpose but specifically targets register/tessitura.

    Parameter mapping:
    - 0.0 → very low register
    - 0.5 → middle register (C4 centered)
    - 1.0 → very high register
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='register_shift',
            dimension='pitch',
            level='section',
            description='Shift overall register/tessitura'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply register shift"""
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return midi

        # Calculate current mean pitch
        current_mean = np.mean([n['pitch'] for n in notes])

        # Target register based on amount
        # 0.0 → C2 (36), 0.5 → C4 (60), 1.0 → C6 (84)
        target_mean = 36 + amount * 48

        # Shift all notes
        shift = int(target_mean - current_mean)
        for note in notes:
            note['pitch'] = np.clip(note['pitch'] + shift, 0, 127)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Get current register"""
        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return 0.5

        mean_pitch = np.mean([n['pitch'] for n in notes])

        # Map to [0,1]
        amount = (mean_pitch - 36) / 48
        return np.clip(amount, 0.0, 1.0)


# ============================================================================
# 5. Pitch Range Transform
# ============================================================================

class PitchRangeTransform(SpaceLevelTransform):
    """
    Compress or expand total pitch range.

    Parameter mapping:
    - 0.0 → minimum range (1 octave)
    - 0.5 → preserve range
    - 1.0 → maximum range (4+ octaves)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='pitch_range',
            dimension='pitch',
            level='section',
            description='Compress or expand total pitch range'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply pitch range transformation"""
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return midi

        # Calculate current range
        pitches = [n['pitch'] for n in notes]
        current_min = min(pitches)
        current_max = max(pitches)
        current_range = current_max - current_min
        center = (current_min + current_max) / 2

        if current_range < 1:
            return midi

        # Target range based on amount
        # 0.0 → 12 semitones, 0.5 → current, 1.0 → 48 semitones
        if amount < 0.5:
            # Compress
            target_range = 12 + (current_range - 12) * (amount / 0.5)
        else:
            # Expand
            target_range = current_range + (48 - current_range) * ((amount - 0.5) / 0.5)

        scale_factor = target_range / current_range

        # Scale around center
        for note in notes:
            interval = note['pitch'] - center
            note['pitch'] = int(center + interval * scale_factor)
            note['pitch'] = np.clip(note['pitch'], 0, 127)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Get current pitch range"""
        notes = extract_notes_from_midi(midi)
        if len(notes) < 2:
            return 0.5

        pitches = [n['pitch'] for n in notes]
        pitch_range = max(pitches) - min(pitches)

        # Map to [0,1] assuming range of 12-48 semitones
        amount = (pitch_range - 12) / 36
        return np.clip(amount, 0.0, 1.0)


# ============================================================================
# 6. Melodic Contour Transform
# ============================================================================

class MelodicContourTransform(SpaceLevelTransform):
    """
    Smooth or exaggerate melodic contour.

    Parameter mapping:
    - 0.0 → very smooth (stepwise motion)
    - 0.5 → preserve contour
    - 1.0 → exaggerated (large leaps)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='melodic_contour',
            dimension='pitch',
            level='phrase',
            description='Smooth or exaggerate melodic motion'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply contour transformation"""
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if len(notes) < 2:
            return midi

        # Sort by time
        notes_sorted = sorted(notes, key=lambda n: n['start_time'])

        # Calculate scaling factor for intervals
        # 0.0 → compress to steps, 1.0 → expand leaps
        if amount < 0.5:
            # Smooth: compress intervals
            scale = amount * 2  # 0-1
        else:
            # Exaggerate: expand intervals
            scale = 1 + (amount - 0.5) * 2  # 1-2

        # Apply to consecutive notes
        for i in range(1, len(notes_sorted)):
            prev_pitch = notes_sorted[i-1]['pitch']
            curr_pitch = notes_sorted[i]['pitch']
            interval = curr_pitch - prev_pitch

            # Scale interval
            new_interval = int(interval * scale)
            notes_sorted[i]['pitch'] = np.clip(prev_pitch + new_interval, 0, 127)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Estimate melodic contour smoothness"""
        notes = extract_notes_from_midi(midi)
        if len(notes) < 2:
            return 0.5

        # Calculate mean absolute interval
        notes_sorted = sorted(notes, key=lambda n: n['start_time'])
        intervals = []
        for i in range(1, len(notes_sorted)):
            intervals.append(abs(notes_sorted[i]['pitch'] - notes_sorted[i-1]['pitch']))

        mean_interval = np.mean(intervals)

        # Map to [0,1]: stepwise=2, typical=4, leaps=8+
        amount = mean_interval / 8
        return np.clip(amount, 0.0, 1.0)


# ============================================================================
# 7. Octave Doubling Transform
# ============================================================================

class OctaveDoublingTransform(SpaceLevelTransform):
    """
    Add or remove octave doublings.

    Parameter mapping:
    - 0.0 → remove all doublings
    - 0.5 → preserve doublings
    - 1.0 → maximum octave doublings
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='octave_doubling',
            dimension='pitch',
            level='note',
            description='Add or remove octave doublings',
            is_invertible=False
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply octave doubling"""
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return midi

        new_notes = list(notes)  # Start with original

        if amount > 0.5:
            # Add doublings
            doubling_probability = (amount - 0.5) * 2  # 0-1

            for note in notes:
                if np.random.random() < doubling_probability:
                    # Add octave above
                    if note['pitch'] + 12 <= 127:
                        doubled_note = note.copy()
                        doubled_note['pitch'] = note['pitch'] + 12
                        doubled_note['velocity'] = int(note['velocity'] * 0.8)
                        new_notes.append(doubled_note)

                    # Add octave below
                    if note['pitch'] - 12 >= 0:
                        doubled_note = note.copy()
                        doubled_note['pitch'] = note['pitch'] - 12
                        doubled_note['velocity'] = int(note['velocity'] * 0.8)
                        new_notes.append(doubled_note)

        elif amount < 0.5:
            # Remove doublings (deduplicate notes at octave intervals)
            removal_probability = (0.5 - amount) * 2

            # Group by time
            time_groups = {}
            for note in new_notes:
                time_key = round(note['start_time'], 2)
                if time_key not in time_groups:
                    time_groups[time_key] = []
                time_groups[time_key].append(note)

            # Remove octave duplicates
            filtered_notes = []
            for group in time_groups.values():
                pitches_seen = set()
                for note in group:
                    pitch_class = note['pitch'] % 12
                    if pitch_class not in pitches_seen or np.random.random() > removal_probability:
                        filtered_notes.append(note)
                        pitches_seen.add(pitch_class)

            new_notes = filtered_notes

        return notes_to_midi(new_notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Estimate amount of octave doubling"""
        notes = extract_notes_from_midi(midi)
        if len(notes) < 2:
            return 0.5

        # Count notes with octave doublings
        time_groups = {}
        for note in notes:
            time_key = round(note['start_time'], 2)
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(note['pitch'])

        doubling_count = 0
        total_opportunities = 0

        for pitches in time_groups.values():
            if len(pitches) < 2:
                continue
            for i, p1 in enumerate(pitches):
                total_opportunities += 1
                for p2 in pitches[i+1:]:
                    if abs(p1 - p2) % 12 == 0:  # Octave related
                        doubling_count += 1
                        break

        if total_opportunities == 0:
            return 0.5

        doubling_ratio = doubling_count / total_opportunities
        return 0.5 + (doubling_ratio * 0.5)  # Map to [0.5, 1.0]


# ============================================================================
# 8. Microtonal Detune Transform
# ============================================================================

class MicrotonalDetuneTransform(SpaceLevelTransform):
    """
    Add microtonal deviations (pitch bend).

    Parameter mapping:
    - 0.0 → perfect tuning
    - 0.5 → slight detuning (humanization)
    - 1.0 → extreme microtonal deviations
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='microtonal_detune',
            dimension='pitch',
            level='note',
            description='Add microtonal pitch deviations'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply microtonal detuning"""
        amount = self.validate_amount(amount)

        # Microtonal effects require pitch bend, which is complex
        # For simplicity, we'll approximate by adding slight pitch variations
        # In a full implementation, this would add pitch bend messages

        notes = extract_notes_from_midi(midi)
        if len(notes) == 0:
            return midi

        # Add random pitch variations
        # amount=0 → no change, amount=1 → up to ±50 cents (±0.5 semitones)
        max_deviation = amount * 0.5

        for note in notes:
            deviation = np.random.uniform(-max_deviation, max_deviation)
            # Round to nearest semitone (MIDI limitation without pitch bend)
            note['pitch'] = np.clip(int(note['pitch'] + round(deviation)), 0, 127)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """
        Microtonal information is not directly encoded in standard MIDI.
        Return neutral value.
        """
        return 0.0  # Assume no detuning
