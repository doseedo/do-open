"""
Expression Domain Transforms
============================

Space-level transforms operating on expression/dynamics dimension.

This is a NEW dimension focused on performance expression:
articulation, dynamics, phrasing, and interpretive choices.

Transforms:
1. DynamicsContourTransform - Flat ↔ shaped dynamics
2. PhrasingTransform - No phrases ↔ clear phrasing
3. AccentPatternTransform - Even ↔ accented
4. LegatoStaccatoTransform - Legato ↔ staccato (extended)
5. VelocityCurveTransform - Linear ↔ exponential velocity
6. AttackDecayTransform - Immediate ↔ gradual attack/decay
7. PedalingTransform - No pedal ↔ heavy pedaling
8. VibratoTransform - No vibrato ↔ wide vibrato
9. TremoloTransform - No tremolo ↔ rapid tremolo
10. BendTransform - No bends ↔ expressive bends
11. GlissandoTransform - Discrete ↔ glissando
12. OrnamentationTransform - Plain ↔ heavily ornamented

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


class DynamicsContourTransform(SpaceLevelTransform):
    """
    Shape dynamics over time (flat ↔ crescendo/diminuendo).

    Parameter mapping:
    - 0.0 → flat dynamics
    - 0.5 → moderate shaping
    - 1.0 → dramatic crescendo/diminuendo
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='dynamics_contour',
            dimension='expression',
            level='section',
            description='Flat to shaped dynamics'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if len(notes) < 2:
            return midi

        if amount < 0.1:
            return midi  # No shaping

        # Get total duration
        max_time = max(n['start_time'] + n['duration'] for n in notes)

        # Apply dynamics curve
        for note in notes:
            position = note['start_time'] / max(max_time, 1.0)

            # Crescendo-diminuendo curve (arch)
            if amount > 0.5:
                curve_strength = (amount - 0.5) * 2
                # Peak in middle
                curve_value = np.sin(position * np.pi)
                velocity_mult = 1.0 + (curve_value - 0.5) * curve_strength
            else:
                # Gradual crescendo
                curve_strength = amount * 2
                velocity_mult = 1.0 + (position - 0.5) * curve_strength

            note['velocity'] = int(note['velocity'] * velocity_mult)
            note['velocity'] = np.clip(note['velocity'], 1, 127)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        notes = extract_notes_from_midi(midi)
        if len(notes) < 2:
            return 0.0

        # Measure velocity variance over time
        max_time = max(n['start_time'] + n['duration'] for n in notes)

        # Bin notes by time
        n_bins = 10
        bin_velocities = [[] for _ in range(n_bins)]

        for note in notes:
            position = note['start_time'] / max(max_time, 1.0)
            bin_idx = min(int(position * n_bins), n_bins - 1)
            bin_velocities[bin_idx].append(note['velocity'])

        # Calculate mean velocity per bin
        bin_means = [np.mean(v) if v else 64 for v in bin_velocities]

        # Measure curvature
        velocity_variance = np.var(bin_means)
        amount = min(velocity_variance / 1000, 1.0)

        return amount


class PhrasingTransform(SpaceLevelTransform):
    """Add/remove phrase boundaries"""

    def __init__(self):
        metadata = TransformMetadata(
            name='phrasing',
            dimension='expression',
            level='phrase',
            description='No phrases to clear phrasing'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        if amount > 0.5:
            # Add phrase boundaries (small gaps)
            phrase_strength = (amount - 0.5) * 2

            # Identify potential phrase boundaries (every ~4 seconds)
            phrase_length = 4.0
            sorted_notes = sorted(notes, key=lambda n: n['start_time'])

            last_phrase_end = 0.0
            for i, note in enumerate(sorted_notes):
                if note['start_time'] - last_phrase_end > phrase_length:
                    # Add small gap
                    gap = phrase_strength * 0.1
                    for future_note in sorted_notes[i:]:
                        future_note['start_time'] += gap

                    last_phrase_end = note['start_time']

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.5  # Simplified


class AccentPatternTransform(SpaceLevelTransform):
    """Add accent patterns"""

    def __init__(self):
        metadata = TransformMetadata(
            name='accent_pattern',
            dimension='expression',
            level='phrase',
            description='Even to accented'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        # Apply accents to downbeats
        beat_duration = 0.5
        accent_strength = amount

        for note in notes:
            beat_position = (note['start_time'] % beat_duration) / beat_duration

            if beat_position < 0.1:  # Downbeat
                note['velocity'] = int(note['velocity'] * (1 + accent_strength * 0.3))
                note['velocity'] = min(note['velocity'], 127)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.5


# Simplified remaining expression transforms

class LegatoStaccatoTransform(SpaceLevelTransform):
    """Extended articulation control (redundant with ArticulationTransform but with different focus)"""

    def __init__(self):
        metadata = TransformMetadata(
            name='legato_staccato',
            dimension='expression',
            level='note',
            description='Legato to staccato articulation'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        # Similar to ArticulationTransform but focusing on overlap
        for note in notes:
            if amount > 0.5:  # Staccato
                note['duration'] *= (1 - (amount - 0.5))
            else:  # Legato - extend slightly
                note['duration'] *= (1 + (0.5 - amount) * 0.3)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.5


class VelocityCurveTransform(SpaceLevelTransform):
    """Control velocity response curve"""

    def __init__(self):
        metadata = TransformMetadata(
            name='velocity_curve',
            dimension='expression',
            level='note',
            description='Linear to exponential velocity'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        # Apply velocity curve
        # 0.5 = linear, <0.5 = compressed, >0.5 = expanded
        exponent = 0.5 + (amount - 0.5) * 1.5  # Range: 0.5 to 2.0

        for note in notes:
            # Normalize to 0-1
            vel_norm = note['velocity'] / 127.0
            # Apply curve
            vel_curved = vel_norm ** exponent
            # Denormalize
            note['velocity'] = int(vel_curved * 127)
            note['velocity'] = np.clip(note['velocity'], 1, 127)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.5


# Placeholder transforms (full implementation would require pitch bend messages)

class AttackDecayTransform(SpaceLevelTransform):
    """Control attack/decay envelope"""

    def __init__(self):
        metadata = TransformMetadata(
            name='attack_decay',
            dimension='expression',
            level='note',
            description='Immediate to gradual attack'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        return midi  # Requires MIDI CC or synthesizer control

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.5


class PedalingTransform(SpaceLevelTransform):
    """Add/remove sustain pedal"""

    def __init__(self):
        metadata = TransformMetadata(
            name='pedaling',
            dimension='expression',
            level='phrase',
            description='No pedal to heavy pedaling'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        # Would add sustain pedal CC messages
        return midi

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.0


class VibratoTransform(SpaceLevelTransform):
    """Add vibrato"""

    def __init__(self):
        metadata = TransformMetadata(
            name='vibrato',
            dimension='expression',
            level='note',
            description='No vibrato to wide vibrato'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        # Requires pitch bend messages
        return midi

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.0


class TremoloTransform(SpaceLevelTransform):
    """Add tremolo"""

    def __init__(self):
        metadata = TransformMetadata(
            name='tremolo',
            dimension='expression',
            level='note',
            description='No tremolo to rapid tremolo'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        # Would add rapid note repetitions
        return midi

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.0


class BendTransform(SpaceLevelTransform):
    """Add pitch bends"""

    def __init__(self):
        metadata = TransformMetadata(
            name='bend',
            dimension='expression',
            level='note',
            description='No bends to expressive bends'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        # Requires pitch bend messages
        return midi

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.0


class GlissandoTransform(SpaceLevelTransform):
    """Add glissando between notes"""

    def __init__(self):
        metadata = TransformMetadata(
            name='glissando',
            dimension='expression',
            level='note',
            description='Discrete to glissando'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        if amount < 0.1:
            return midi

        notes = extract_notes_from_midi(midi)
        sorted_notes = sorted(notes, key=lambda n: n['start_time'])

        new_notes = list(notes)

        # Add glissando notes between large intervals
        for i in range(len(sorted_notes) - 1):
            n1, n2 = sorted_notes[i], sorted_notes[i+1]
            interval = abs(n2['pitch'] - n1['pitch'])

            if interval > 2 and np.random.random() < amount:
                # Add intermediate notes
                num_steps = min(interval - 1, int(amount * 10))
                step_time = (n2['start_time'] - n1['start_time']) / (num_steps + 1)

                for step in range(1, num_steps + 1):
                    gliss_note = n1.copy()
                    gliss_note['start_time'] = n1['start_time'] + step * step_time
                    gliss_note['pitch'] = n1['pitch'] + step * (1 if n2['pitch'] > n1['pitch'] else -1)
                    gliss_note['duration'] = step_time * 0.8
                    gliss_note['velocity'] = int(n1['velocity'] * 0.7)
                    new_notes.append(gliss_note)

        return notes_to_midi(new_notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.0  # Simplified


class OrnamentationTransform(SpaceLevelTransform):
    """Add ornaments (trills, mordents, turns)"""

    def __init__(self):
        metadata = TransformMetadata(
            name='ornamentation',
            dimension='expression',
            level='note',
            description='Plain to heavily ornamented',
            is_invertible=False
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        if amount < 0.1:
            return midi

        notes = extract_notes_from_midi(midi)
        new_notes = list(notes)

        # Add ornaments to long notes
        ornament_prob = amount

        for note in notes:
            if note['duration'] > 0.3 and np.random.random() < ornament_prob:
                # Add trill (alternating neighbor)
                trill_duration = 0.05
                num_trills = int(note['duration'] / trill_duration)

                for i in range(num_trills):
                    trill_note = note.copy()
                    trill_note['start_time'] = note['start_time'] + i * trill_duration
                    trill_note['duration'] = trill_duration
                    trill_note['pitch'] = note['pitch'] + (1 if i % 2 == 0 else 0)
                    trill_note['velocity'] = int(note['velocity'] * 0.8)
                    new_notes.append(trill_note)

        return notes_to_midi(new_notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.0
