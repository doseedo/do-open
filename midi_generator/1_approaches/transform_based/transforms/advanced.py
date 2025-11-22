"""
Advanced Transforms
===================

Sophisticated transforms that combine multiple musical dimensions.

These transforms represent complex musical operations that don't fit
neatly into a single dimension.

Transforms:
1. MetricModulationTransform - Change meter while preserving pulse
2. ModalMixtureTransform - Borrow from parallel modes
3. CounterpointDensityTransform - Independent voice activity
4. TexturalEvolutionTransform - Gradual textural transformation
5. HarmonicModulationTransform - Key changes
6. Polymeter - Multiple simultaneous meters
7. MicrorhythmTransform - Subtle timing deviations
8. SpectralDensityTransform - Harmonic vs inharmonic content

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


class MetricModulationTransform(SpaceLevelTransform):
    """
    Metric modulation: change tempo while keeping some subdivision constant.

    Parameter mapping:
    - 0.0 → no modulation (original tempo)
    - 0.5 → moderate modulation (3:2)
    - 1.0 → extreme modulation (4:3, 5:4)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='metric_modulation',
            dimension='rhythm',
            level='section',
            description='Metric modulation (tempo change via subdivision)'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        if amount < 0.1:
            return midi

        notes = extract_notes_from_midi(midi)

        # Common metric modulations
        if amount < 0.3:
            ratio = 1.0  # No change
        elif amount < 0.6:
            ratio = 1.5  # Triplet = old eighth
        else:
            ratio = 1.33  # New quarter = old triplet

        # Apply to second half of piece
        if notes:
            midpoint = max(n['start_time'] for n in notes) / 2

            for note in notes:
                if note['start_time'] > midpoint:
                    note['start_time'] = midpoint + (note['start_time'] - midpoint) * ratio
                    note['duration'] *= ratio

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.0  # Simplified


class ModalMixtureTransform(SpaceLevelTransform):
    """
    Modal mixture: borrow chords from parallel mode (major ↔ minor).

    Parameter mapping:
    - 0.0 → diatonic (no borrowing)
    - 0.5 → occasional mixture
    - 1.0 → extensive borrowing
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='modal_mixture',
            dimension='harmony',
            level='phrase',
            description='Modal mixture (parallel mode borrowing)'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        # Identify scale degrees and alter them
        # Simplified: lower 3rd, 6th, 7th (major → minor)
        mixture_prob = amount

        for note in notes:
            pitch_class = note['pitch'] % 12

            # If this is a major 3rd (4 semitones from root)
            if pitch_class == 4 and np.random.random() < mixture_prob:
                note['pitch'] -= 1  # Lower to minor 3rd

            # If this is a major 6th (9 semitones from root)
            elif pitch_class == 9 and np.random.random() < mixture_prob:
                note['pitch'] -= 1  # Lower to minor 6th

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.0  # Simplified


class CounterpointDensityTransform(SpaceLevelTransform):
    """
    Contrapuntal activity: independent voice motion.

    Parameter mapping:
    - 0.0 → homophonic (voices move together)
    - 0.5 → moderate independence
    - 1.0 → highly independent (strict counterpoint)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='counterpoint_density',
            dimension='texture',
            level='phrase',
            description='Homophonic to contrapuntal'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        # Increase rhythmic independence between tracks
        independence_factor = amount

        # Group by track
        track_notes = {}
        for note in notes:
            track = note.get('track', 0)
            if track not in track_notes:
                track_notes[track] = []
            track_notes[track].append(note)

        # Offset each track rhythmically
        for i, (track, track_notes_list) in enumerate(track_notes.items()):
            if i == 0:
                continue  # Keep first track as reference

            offset = independence_factor * 0.2 * i  # Progressive offset

            for note in track_notes_list:
                note['start_time'] += offset

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.5


class TexturalEvolutionTransform(SpaceLevelTransform):
    """
    Gradual textural transformation over time.

    Parameter mapping:
    - 0.0 → static texture
    - 0.5 → gradual evolution
    - 1.0 → dramatic transformation
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='textural_evolution',
            dimension='texture',
            level='section',
            description='Static to evolving texture'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        if amount < 0.1:
            return midi

        # Gradually increase polyphony over time
        max_time = max(n['start_time'] + n['duration'] for n in notes) if notes else 1.0

        evolution_strength = amount

        for note in notes:
            position = note['start_time'] / max_time

            # Start sparse, end dense
            keep_prob = 0.3 + (position * 0.7 * evolution_strength)

            if np.random.random() > keep_prob:
                note['velocity'] = 1  # Effectively remove (will be very quiet)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.0


class HarmonicModulationTransform(SpaceLevelTransform):
    """
    Key changes (modulation).

    Parameter mapping:
    - 0.0 → single key
    - 0.5 → one modulation
    - 1.0 → multiple modulations
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='harmonic_modulation',
            dimension='harmony',
            level='section',
            description='Single key to multiple modulations'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        if amount < 0.1:
            return midi

        # Transpose sections to different keys
        max_time = max(n['start_time'] + n['duration'] for n in notes) if notes else 1.0

        # Number of modulations based on amount
        num_modulations = int(1 + amount * 3)  # 1-4 modulations

        section_length = max_time / (num_modulations + 1)

        modulations = [0]  # Start in original key
        for i in range(num_modulations):
            # Common modulations: +5 (dominant), +7 (fifth), -5 (subdominant)
            modulations.append(np.random.choice([5, 7, -5, -7]))

        for note in notes:
            section = int(note['start_time'] / section_length)
            section = min(section, len(modulations) - 1)

            note['pitch'] += modulations[section]
            note['pitch'] = np.clip(note['pitch'], 0, 127)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.0


class PolymeterTransform(SpaceLevelTransform):
    """
    Multiple simultaneous meters (e.g., 3/4 against 4/4).

    Parameter mapping:
    - 0.0 → single meter
    - 0.5 → subtle polymeter (3:2)
    - 1.0 → complex polymeter (5:4, 7:4)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='polymeter',
            dimension='rhythm',
            level='section',
            description='Single meter to polymeter'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        if amount < 0.1:
            return midi

        # Apply different meters to different tracks
        track_notes = {}
        for note in notes:
            track = note.get('track', 0)
            if track not in track_notes:
                track_notes[track] = []
            track_notes[track].append(note)

        if len(track_notes) < 2:
            return midi  # Need multiple tracks

        # Apply 3:2 polymeter to second track
        track_list = list(track_notes.keys())
        secondary_track = track_list[1]

        for note in track_notes[secondary_track]:
            # Adjust timing to 3/4 while primary is in 4/4
            note['start_time'] *= 0.75

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.0


class MicrorhythmTransform(SpaceLevelTransform):
    """
    Subtle timing deviations (humanization, but more controlled).

    Parameter mapping:
    - 0.0 → perfect timing
    - 0.5 → subtle micro-rhythm
    - 1.0 → pronounced expressive timing
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='microrhythm',
            dimension='rhythm',
            level='note',
            description='Perfect timing to microrhythm'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        # Add correlated timing deviations (not pure random)
        max_deviation = amount * 0.02  # Up to 20ms

        # Sort by time
        sorted_notes = sorted(notes, key=lambda n: n['start_time'])

        # Correlated random walk
        deviation = 0.0
        for note in sorted_notes:
            # Random walk with mean reversion
            deviation += np.random.normal(0, max_deviation)
            deviation *= 0.9  # Mean reversion

            note['start_time'] += deviation
            note['start_time'] = max(0, note['start_time'])

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.0


class SpectralDensityTransform(SpaceLevelTransform):
    """
    Harmonic vs inharmonic spectral content.

    Parameter mapping:
    - 0.0 → pure harmonics (octaves, fifths)
    - 0.5 → mixed
    - 1.0 → inharmonic (clusters, noise-like)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='spectral_density',
            dimension='harmony',
            level='phrase',
            description='Harmonic to inharmonic'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        if amount > 0.5:
            # Add inharmonic content (clusters)
            inharmonic_strength = (amount - 0.5) * 2

            new_notes = []
            for note in notes:
                if np.random.random() < inharmonic_strength * 0.3:
                    # Add cluster around this note
                    cluster_size = int(inharmonic_strength * 3)
                    for offset in range(-cluster_size, cluster_size + 1):
                        cluster_note = note.copy()
                        cluster_note['pitch'] += offset
                        cluster_note['pitch'] = np.clip(cluster_note['pitch'], 0, 127)
                        cluster_note['velocity'] = int(note['velocity'] * 0.5)
                        new_notes.append(cluster_note)

            notes.extend(new_notes)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.0
