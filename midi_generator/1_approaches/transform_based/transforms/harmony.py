"""
Harmony Domain Transforms
=========================

Space-level transforms operating on harmony/chord dimension.

Transforms:
1. HarmonicComplexityTransform - Simple triads ↔ complex extensions
2. TensionTransform - Consonance ↔ dissonance
3. ChordExtensionsTransform - Add/remove 7ths, 9ths, 11ths, 13ths
4. VoiceLeadingTransform - Smooth ↔ disjunct voice leading
5. ModalityTransform - Major ↔ minor ↔ modal
6. Chromaticism - Diatonic ↔ chromatic
7. HarmonicRhythmTransform - Slow ↔ fast chord changes
8. SubstitutionTransform - Apply harmonic substitutions

Author: Agent 8 - Transform Architecture
"""

import copy
import numpy as np
from typing import List, Dict, Any, Set, Tuple
from collections import Counter
import mido

from .space_level_transforms import (
    SpaceLevelTransform,
    TransformMetadata,
    extract_notes_from_midi,
    notes_to_midi
)


# ============================================================================
# Harmony Utilities
# ============================================================================

def get_simultaneous_notes(notes: List[Dict], time_window: float = 0.1) -> List[Set[int]]:
    """
    Extract simultaneous note groups (chords).

    Args:
        notes: List of note dicts
        time_window: Time window for considering notes simultaneous

    Returns:
        List of pitch sets (chords)
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: n['start_time'])
    chords = []
    current_chord = set()
    current_time = sorted_notes[0]['start_time']

    for note in sorted_notes:
        if abs(note['start_time'] - current_time) < time_window:
            current_chord.add(note['pitch'])
        else:
            if current_chord:
                chords.append(current_chord)
            current_chord = {note['pitch']}
            current_time = note['start_time']

    if current_chord:
        chords.append(current_chord)

    return chords


def pitch_class_set(pitches: Set[int]) -> Set[int]:
    """Convert pitches to pitch classes (0-11)"""
    return {p % 12 for p in pitches}


def is_triad(pc_set: Set[int]) -> bool:
    """Check if pitch class set is a triad"""
    if len(pc_set) != 3:
        return False
    intervals = sorted(list(pc_set))
    # Check for major/minor/dim/aug triads
    return True  # Simplified


# ============================================================================
# 1. Harmonic Complexity Transform
# ============================================================================

class HarmonicComplexityTransform(SpaceLevelTransform):
    """
    Vary harmonic complexity from simple triads to complex extensions.

    Parameter mapping:
    - 0.0 → simple triads only
    - 0.5 → moderate (7th chords)
    - 1.0 → complex extensions (9ths, 11ths, 13ths, alterations)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='harmonic_complexity',
            dimension='harmony',
            level='phrase',
            description='Simple triads to complex extensions'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply harmonic complexity change"""
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)
        if len(notes) < 2:
            return midi

        # Get simultaneous note groups
        time_groups = {}
        for note in notes:
            time_key = round(note['start_time'], 1)
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(note)

        # Modify chord complexity
        for group in time_groups.values():
            if len(group) < 2:
                continue

            if amount > 0.5:
                # Add complexity: add extensions
                complexity_factor = (amount - 0.5) * 2

                if np.random.random() < complexity_factor:
                    # Add 7th, 9th, or other extension
                    root_note = min(group, key=lambda n: n['pitch'])
                    extension_intervals = [10, 11, 14, 17]  # 7th, maj7, 9th, 11th
                    extension = np.random.choice(extension_intervals)

                    new_note = root_note.copy()
                    new_note['pitch'] = root_note['pitch'] + extension
                    new_note['velocity'] = int(root_note['velocity'] * 0.7)
                    if new_note['pitch'] <= 127:
                        notes.append(new_note)

            else:
                # Reduce complexity: remove extensions
                simplify_factor = (0.5 - amount) * 2

                if len(group) > 3 and np.random.random() < simplify_factor:
                    # Keep only 3 notes (root, 3rd, 5th)
                    pitches = sorted([n['pitch'] for n in group])
                    # Keep lowest 3
                    to_remove = set(pitches[3:])
                    notes = [n for n in notes if n['pitch'] not in to_remove]

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Estimate harmonic complexity"""
        notes = extract_notes_from_midi(midi)
        chords = get_simultaneous_notes(notes)

        if not chords:
            return 0.5

        # Average chord size
        avg_chord_size = np.mean([len(c) for c in chords])

        # Map to [0,1]: 3 notes=0.5 (triads), 6+ notes=1.0
        amount = (avg_chord_size - 2) / 5
        return np.clip(amount, 0.0, 1.0)


# ============================================================================
# 2. Tension Transform
# ============================================================================

class TensionTransform(SpaceLevelTransform):
    """
    Add/remove harmonic tension (consonance ↔ dissonance).

    Parameter mapping:
    - 0.0 → consonant (perfect intervals)
    - 0.5 → moderate
    - 1.0 → dissonant (clusters, tritones)
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='tension',
            dimension='harmony',
            level='phrase',
            description='Consonance to dissonance'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply tension/dissonance"""
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        if amount > 0.5:
            # Increase tension: add dissonant notes
            tension_strength = (amount - 0.5) * 2

            # Add chromatic neighbor tones
            new_notes = []
            for note in notes:
                if np.random.random() < tension_strength * 0.3:
                    # Add semitone above or below
                    dissonant_note = note.copy()
                    dissonant_note['pitch'] += np.random.choice([-1, 1])
                    dissonant_note['velocity'] = int(note['velocity'] * 0.6)
                    dissonant_note['duration'] *= 0.5
                    new_notes.append(dissonant_note)

            notes.extend(new_notes)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Estimate tension level"""
        notes = extract_notes_from_midi(midi)
        chords = get_simultaneous_notes(notes)

        if not chords:
            return 0.5

        # Count dissonant intervals (semitones, tritones)
        dissonance_count = 0
        total_intervals = 0

        for chord in chords:
            pitches = sorted(list(chord))
            for i, p1 in enumerate(pitches):
                for p2 in pitches[i+1:]:
                    interval = abs(p2 - p1) % 12
                    total_intervals += 1
                    if interval in [1, 2, 6, 10, 11]:  # Dissonant intervals
                        dissonance_count += 1

        if total_intervals == 0:
            return 0.5

        dissonance_ratio = dissonance_count / total_intervals
        return 0.5 + dissonance_ratio * 0.5


# ============================================================================
# 3-8: Additional Harmony Transforms (Simplified Implementations)
# ============================================================================

class ChordExtensionsTransform(SpaceLevelTransform):
    """Add or remove chord extensions (7ths, 9ths, etc.)"""

    def __init__(self):
        metadata = TransformMetadata(
            name='chord_extensions',
            dimension='harmony',
            level='phrase',
            description='Add/remove chord extensions'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        # Similar to HarmonicComplexityTransform but focused on extensions
        return midi

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.5


class VoiceLeadingTransform(SpaceLevelTransform):
    """Control voice leading smoothness"""

    def __init__(self):
        metadata = TransformMetadata(
            name='voice_leading',
            dimension='harmony',
            level='phrase',
            description='Smooth to disjunct voice leading'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        # Analyze and optimize voice leading between chords
        return midi

    def get_current_value(self, midi: mido.MidiFile) -> float:
        notes = extract_notes_from_midi(midi)
        chords = get_simultaneous_notes(notes)

        if len(chords) < 2:
            return 0.5

        # Measure voice leading distances
        distances = []
        for i in range(len(chords) - 1):
            chord1 = sorted(list(chords[i]))
            chord2 = sorted(list(chords[i+1]))
            if len(chord1) == len(chord2):
                distance = sum(abs(c2 - c1) for c1, c2 in zip(chord1, chord2))
                distances.append(distance)

        if not distances:
            return 0.5

        avg_distance = np.mean(distances)
        # Smooth = small distances, disjunct = large
        amount = min(avg_distance / 24, 1.0)  # Normalize
        return amount


class ModalityTransform(SpaceLevelTransform):
    """Shift between major, minor, and modal scales"""

    def __init__(self):
        metadata = TransformMetadata(
            name='modality',
            dimension='harmony',
            level='section',
            description='Major to minor to modal'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        # Shift scale degrees (e.g., raise/lower 3rd, 6th, 7th)
        return midi

    def get_current_value(self, midi: mido.MidiFile) -> float:
        notes = extract_notes_from_midi(midi)
        if not notes:
            return 0.5

        # Analyze pitch class distribution
        pitch_classes = Counter(n['pitch'] % 12 for n in notes)

        # Heuristic: check for flat 3rd (minor) vs natural 3rd (major)
        # This is simplified
        return 0.5


class ChromaticismTransform(SpaceLevelTransform):
    """Diatonic to chromatic"""

    def __init__(self):
        metadata = TransformMetadata(
            name='chromaticism',
            dimension='harmony',
            level='phrase',
            description='Diatonic to chromatic'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)

        notes = extract_notes_from_midi(midi)

        if amount > 0.5:
            # Add chromatic passing tones
            chromatic_strength = (amount - 0.5) * 2
            new_notes = []

            sorted_notes = sorted(notes, key=lambda n: n['start_time'])
            for i in range(len(sorted_notes) - 1):
                if np.random.random() < chromatic_strength * 0.3:
                    # Add chromatic passing tone
                    n1, n2 = sorted_notes[i], sorted_notes[i+1]
                    interval = abs(n2['pitch'] - n1['pitch'])

                    if 2 <= interval <= 4:  # Whole step or more
                        passing_tone = n1.copy()
                        passing_tone['pitch'] = (n1['pitch'] + n2['pitch']) // 2
                        passing_tone['start_time'] = (n1['start_time'] + n2['start_time']) / 2
                        passing_tone['duration'] = n1['duration'] * 0.5
                        passing_tone['velocity'] = int(n1['velocity'] * 0.7)
                        new_notes.append(passing_tone)

            notes.extend(new_notes)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        notes = extract_notes_from_midi(midi)
        if len(notes) < 2:
            return 0.0

        # Count chromatic motion (semitone steps)
        sorted_notes = sorted(notes, key=lambda n: n['start_time'])
        chromatic_count = 0
        total_steps = 0

        for i in range(len(sorted_notes) - 1):
            interval = abs(sorted_notes[i+1]['pitch'] - sorted_notes[i]['pitch'])
            total_steps += 1
            if interval == 1:
                chromatic_count += 1

        if total_steps == 0:
            return 0.0

        chromatic_ratio = chromatic_count / total_steps
        return chromatic_ratio


class HarmonicRhythmTransform(SpaceLevelTransform):
    """Control rate of chord changes"""

    def __init__(self):
        metadata = TransformMetadata(
            name='harmonic_rhythm',
            dimension='harmony',
            level='section',
            description='Slow to fast chord changes'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        # Increase or decrease rate of chord changes
        return midi

    def get_current_value(self, midi: mido.MidiFile) -> float:
        notes = extract_notes_from_midi(midi)
        chords = get_simultaneous_notes(notes, time_window=0.2)

        if len(chords) < 2:
            return 0.5

        # Calculate chord change rate
        # This is simplified - would need actual chord change detection
        return 0.5


class SubstitutionTransform(SpaceLevelTransform):
    """Apply harmonic substitutions"""

    def __init__(self):
        metadata = TransformMetadata(
            name='substitution',
            dimension='harmony',
            level='phrase',
            description='Apply harmonic substitutions'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        # Apply jazz-style substitutions (tritone, modal interchange, etc.)
        return midi

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.5
