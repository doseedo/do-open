"""
Texture Domain Transforms
=========================

Space-level transforms operating on texture/orchestration dimension.

Transforms:
1. PolyphonyTransform - Monophonic ↔ polyphonic
2. VoiceSpacingTransform - Close ↔ wide spacing
3. DoublingTransform - Add/remove doublings
4. LayeringTransform - Simple ↔ layered texture
5. ArticulationTransform - Legato ↔ staccato
6. DynamicRangeTransform - Narrow ↔ wide dynamics
7. TextureDensityTransform - Sparse ↔ dense
8. TimbreVarietyTransform - Homogeneous ↔ varied

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


class PolyphonyTransform(SpaceLevelTransform):
    """Control number of simultaneous voices"""

    def __init__(self):
        metadata = TransformMetadata(
            name='polyphony',
            dimension='texture',
            level='section',
            description='Monophonic to polyphonic'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if not notes:
            return midi

        # Target voice count based on amount
        target_voices = int(1 + amount * 7)  # 1-8 voices

        # Group by time
        time_groups = {}
        for note in notes:
            time_key = round(note['start_time'], 1)
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(note)

        # Adjust polyphony
        filtered_notes = []
        for group in time_groups.values():
            if len(group) > target_voices:
                # Reduce to target (keep highest velocity)
                sorted_group = sorted(group, key=lambda n: n['velocity'], reverse=True)
                filtered_notes.extend(sorted_group[:target_voices])
            else:
                filtered_notes.extend(group)

        return notes_to_midi(filtered_notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        notes = extract_notes_from_midi(midi)
        if not notes:
            return 0.5

        # Calculate average polyphony
        time_groups = {}
        for note in notes:
            time_key = round(note['start_time'], 1)
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(note)

        avg_polyphony = np.mean([len(g) for g in time_groups.values()])
        amount = (avg_polyphony - 1) / 7
        return np.clip(amount, 0.0, 1.0)


class VoiceSpacingTransform(SpaceLevelTransform):
    """Control spacing between voices"""

    def __init__(self):
        metadata = TransformMetadata(
            name='voice_spacing',
            dimension='texture',
            level='phrase',
            description='Close to wide voice spacing'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        # Group simultaneous notes
        time_groups = {}
        for note in notes:
            time_key = round(note['start_time'], 1)
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(note)

        # Adjust spacing
        for group in time_groups.values():
            if len(group) < 2:
                continue

            pitches = sorted([n['pitch'] for n in group])
            center = np.mean(pitches)

            # Spacing factor: 0.5=preserve, 0=close, 1=wide
            spacing_factor = 0.5 + (amount - 0.5) * 1.5

            for note in group:
                interval = note['pitch'] - center
                note['pitch'] = int(center + interval * spacing_factor)
                note['pitch'] = np.clip(note['pitch'], 0, 127)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        notes = extract_notes_from_midi(midi)
        if len(notes) < 2:
            return 0.5

        # Measure average spacing
        time_groups = {}
        for note in notes:
            time_key = round(note['start_time'], 1)
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(note)

        spacings = []
        for group in time_groups.values():
            if len(group) >= 2:
                pitches = [n['pitch'] for n in group]
                spacing = max(pitches) - min(pitches)
                spacings.append(spacing)

        if not spacings:
            return 0.5

        avg_spacing = np.mean(spacings)
        amount = min(avg_spacing / 36, 1.0)  # Normalize to 3 octaves
        return amount


class DoublingTransform(SpaceLevelTransform):
    """Add/remove note doublings"""

    def __init__(self):
        metadata = TransformMetadata(
            name='doubling',
            dimension='texture',
            level='note',
            description='Add/remove doublings',
            is_invertible=False
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if not notes:
            return midi

        if amount > 0.5:
            # Add doublings
            doubling_prob = (amount - 0.5) * 2
            new_notes = []

            for note in notes:
                if np.random.random() < doubling_prob * 0.5:
                    # Add doubling at octave
                    doubled = note.copy()
                    doubled['pitch'] = note['pitch'] + 12
                    doubled['velocity'] = int(note['velocity'] * 0.75)
                    if doubled['pitch'] <= 127:
                        new_notes.append(doubled)

            notes.extend(new_notes)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.5  # Simplified


# Simplified remaining transforms

class LayeringTransform(SpaceLevelTransform):
    """Control textural layers"""

    def __init__(self):
        metadata = TransformMetadata(
            name='layering',
            dimension='texture',
            level='section',
            description='Simple to layered texture'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        return midi

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.5


class ArticulationTransform(SpaceLevelTransform):
    """Control note articulation"""

    def __init__(self):
        metadata = TransformMetadata(
            name='articulation',
            dimension='texture',
            level='note',
            description='Legato to staccato'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        # Adjust note durations
        # 0.0=legato (long), 1.0=staccato (short)
        duration_factor = 1.0 - (amount * 0.7)  # 1.0 to 0.3

        for note in notes:
            note['duration'] *= duration_factor

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        notes = extract_notes_from_midi(midi)
        if not notes:
            return 0.5

        # Measure average duration relative to IOI
        sorted_notes = sorted(notes, key=lambda n: n['start_time'])

        if len(sorted_notes) < 2:
            return 0.5

        # Simplified: measure average duration
        avg_duration = np.mean([n['duration'] for n in notes])
        amount = 1.0 - min(avg_duration / 1.0, 1.0)  # Longer=more legato=lower value

        return amount


class DynamicRangeTransform(SpaceLevelTransform):
    """Control velocity range"""

    def __init__(self):
        metadata = TransformMetadata(
            name='dynamic_range',
            dimension='texture',
            level='section',
            description='Narrow to wide dynamics'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if not notes:
            return midi

        velocities = [n['velocity'] for n in notes]
        mean_vel = np.mean(velocities)

        # Dynamic range factor
        range_factor = 0.2 + amount * 0.8  # 0.2 to 1.0

        for note in notes:
            deviation = note['velocity'] - mean_vel
            note['velocity'] = int(mean_vel + deviation * range_factor)
            note['velocity'] = np.clip(note['velocity'], 1, 127)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        notes = extract_notes_from_midi(midi)
        if not notes:
            return 0.5

        velocities = [n['velocity'] for n in notes]
        velocity_range = max(velocities) - min(velocities)

        amount = velocity_range / 127
        return amount


class TextureDensityTransform(SpaceLevelTransform):
    """Control textural density"""

    def __init__(self):
        metadata = TransformMetadata(
            name='texture_density',
            dimension='texture',
            level='section',
            description='Sparse to dense texture'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        return midi

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.5


class TimbreVarietyTransform(SpaceLevelTransform):
    """Control timbre variety (via MIDI channels/programs)"""

    def __init__(self):
        metadata = TransformMetadata(
            name='timbre_variety',
            dimension='texture',
            level='section',
            description='Homogeneous to varied timbre'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        # Would change MIDI program numbers
        return midi

    def get_current_value(self, midi: mido.MidiFile) -> float:
        # Analyze track/channel variety
        return 0.5
