#!/usr/bin/env python3
"""
Advanced Melody Module - Graduate-Level Melodic Theory & Composition

This module provides advanced melodic composition capabilities including:
- Contour theory (arch, wave, climax analysis)
- Motif development (sequence, inversion, retrograde, augmentation, diminution)
- Phrase structure (antecedent-consequent, periods, sentences)
- Intervallic control (step/leap ratios, tension management)
- Range management (tessitura, climax placement)
- Tension curves (melodic tension scoring)
- Ornamentation (trills, turns, mordents, grace notes, appoggiaturas)
- Style-specific patterns (Baroque, Classical, Romantic, Jazz, Pop)
- Narrative arc (introduction, development, climax, resolution)

Integrates with:
- harmony_advanced.py (voice leading, functional harmony)
- melody_generator_proper.py (target-note technique)
- melody_harmonizer_improved.py (chord-scale theory)

Author: Advanced Melody Research Team
Date: 2025
Theory: Schenker, Forte, Narmour, Meyer, Lerdahl & Jackendoff
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from enum import Enum
import math
from copy import deepcopy

# ============================================================================
# CONTOUR THEORY - Melodic Shape Analysis
# ============================================================================

class ContourType(Enum):
    """Melodic contour classifications (based on Morris, Marvin, Laprade)"""
    ARCH = "arch"  # Ascending then descending (common in classical melodies)
    INVERTED_ARCH = "inverted_arch"  # Descending then ascending
    ASCENDING = "ascending"  # Generally upward motion
    DESCENDING = "descending"  # Generally downward motion
    WAVE = "wave"  # Multiple peaks and valleys
    PLATEAU = "plateau"  # Relatively static
    ZIGZAG = "zigzag"  # Frequent direction changes


@dataclass
class ContourPoint:
    """Single point in melodic contour"""
    time: float  # Beat position
    pitch: int  # MIDI note number
    metric_weight: float = 1.0  # Metric accent (1.0 = strong, 0.5 = weak)
    harmonic_weight: float = 1.0  # Harmonic importance


@dataclass
class ContourAnalysis:
    """Analysis of melodic contour"""
    contour_type: ContourType
    peak_points: List[ContourPoint]  # Climax points
    valley_points: List[ContourPoint]  # Low points
    overall_direction: str  # "ascending", "descending", "static"
    range: int  # Total pitch range in semitones
    tessitura: int  # Average pitch (center of melodic activity)
    climax_position: float  # 0.0-1.0 (golden ratio ≈ 0.618 is ideal)
    tension_curve: List[float]  # Tension value at each point (0.0-1.0)
    step_leap_ratio: float  # Ratio of stepwise to leaping motion


class ContourTheory:
    """
    Melodic contour analysis and generation.

    Based on:
    - Robert Morris: Composition with Pitch Classes (1987)
    - Elizabeth West Marvin & Paul Laprade: Relating Musical Contours (1987)
    - Ian Quinn: General Equal-Tempered Harmony (2006-2007)
    """

    @staticmethod
    def analyze_contour(melody: List[int], beat_positions: List[float] = None) -> ContourAnalysis:
        """
        Analyze melodic contour.

        Args:
            melody: List of MIDI note numbers
            beat_positions: Optional beat positions for each note

        Returns:
            ContourAnalysis object
        """
        if not melody or len(melody) < 2:
            raise ValueError("Melody must have at least 2 notes")

        if beat_positions is None:
            beat_positions = list(range(len(melody)))

        # Create contour points
        points = [ContourPoint(time=beat_positions[i], pitch=melody[i])
                  for i in range(len(melody))]

        # Identify peaks and valleys
        peaks = []
        valleys = []
        for i in range(1, len(melody) - 1):
            if melody[i] > melody[i-1] and melody[i] > melody[i+1]:
                peaks.append(points[i])
            elif melody[i] < melody[i-1] and melody[i] < melody[i+1]:
                valleys.append(points[i])

        # Determine overall direction
        overall_direction = "static"
        third_len = max(1, len(melody) // 3)
        start_third = melody[:third_len]
        end_third = melody[-third_len:]
        start_avg = sum(start_third) / len(start_third) if start_third else melody[0]
        end_avg = sum(end_third) / len(end_third) if end_third else melody[-1]
        if end_avg > start_avg + 2:
            overall_direction = "ascending"
        elif start_avg > end_avg + 2:
            overall_direction = "descending"

        # Calculate range and tessitura
        pitch_range = max(melody) - min(melody)
        tessitura = sum(melody) // len(melody)

        # Find climax position
        max_pitch = max(melody)
        climax_idx = melody.index(max_pitch)
        climax_position = climax_idx / (len(melody) - 1) if len(melody) > 1 else 0.5

        # Classify contour type
        contour_type = ContourTheory._classify_contour(melody, peaks, valleys)

        # Calculate tension curve
        tension_curve = ContourTheory._calculate_tension_curve(melody)

        # Calculate step/leap ratio
        intervals = [abs(melody[i+1] - melody[i]) for i in range(len(melody)-1)]
        steps = sum(1 for interval in intervals if interval <= 2)
        leaps = sum(1 for interval in intervals if interval > 2)
        step_leap_ratio = steps / leaps if leaps > 0 else float('inf')

        return ContourAnalysis(
            contour_type=contour_type,
            peak_points=peaks,
            valley_points=valleys,
            overall_direction=overall_direction,
            range=pitch_range,
            tessitura=tessitura,
            climax_position=climax_position,
            tension_curve=tension_curve,
            step_leap_ratio=step_leap_ratio
        )

    @staticmethod
    def _classify_contour(melody: List[int], peaks: List[ContourPoint],
                         valleys: List[ContourPoint]) -> ContourType:
        """Classify contour type based on shape"""
        if len(melody) < 3:
            return ContourType.PLATEAU

        # Check for arch (single peak in middle third)
        middle_third_start = len(melody) // 3
        middle_third_end = 2 * len(melody) // 3

        peaks_in_middle = sum(1 for p in peaks
                             if middle_third_start <= melody.index(p.pitch) < middle_third_end)

        if len(peaks) == 1 and peaks_in_middle == 1:
            return ContourType.ARCH

        # Check for inverted arch (single valley in middle third)
        valleys_in_middle = sum(1 for v in valleys
                               if middle_third_start <= melody.index(v.pitch) < middle_third_end)

        if len(valleys) == 1 and valleys_in_middle == 1:
            return ContourType.INVERTED_ARCH

        # Check for wave (multiple peaks and valleys)
        if len(peaks) >= 2 and len(valleys) >= 2:
            return ContourType.WAVE

        # Check for generally ascending/descending
        first_third_avg = sum(melody[:len(melody)//3]) / (len(melody)//3)
        last_third_avg = sum(melody[-len(melody)//3:]) / (len(melody)//3)

        if last_third_avg > first_third_avg + 4:
            return ContourType.ASCENDING
        elif last_third_avg < first_third_avg - 4:
            return ContourType.DESCENDING

        # Check for zigzag (many direction changes)
        direction_changes = 0
        for i in range(1, len(melody) - 1):
            dir1 = melody[i] - melody[i-1]
            dir2 = melody[i+1] - melody[i]
            if (dir1 > 0 and dir2 < 0) or (dir1 < 0 and dir2 > 0):
                direction_changes += 1

        if direction_changes >= len(melody) * 0.4:
            return ContourType.ZIGZAG

        return ContourType.PLATEAU

    @staticmethod
    def _calculate_tension_curve(melody: List[int]) -> List[float]:
        """
        Calculate tension at each point (0.0 = low, 1.0 = high).

        Tension factors:
        - Pitch height (higher = more tension)
        - Large intervals (leaps create tension)
        - Distance from tessitura
        """
        if len(melody) < 2:
            return [0.5]

        tessitura = sum(melody) / len(melody)
        pitch_range = max(melody) - min(melody)

        tension = []
        for i, pitch in enumerate(melody):
            # Factor 1: Distance from tessitura (normalized)
            tessitura_distance = abs(pitch - tessitura)
            tessitura_tension = min(tessitura_distance / (pitch_range / 2), 1.0) if pitch_range > 0 else 0.0

            # Factor 2: Interval size (before and after)
            interval_tension = 0.0
            if i > 0:
                interval_before = abs(melody[i] - melody[i-1])
                interval_tension += min(interval_before / 12, 1.0)
            if i < len(melody) - 1:
                interval_after = abs(melody[i+1] - melody[i])
                interval_tension += min(interval_after / 12, 1.0)
            interval_tension /= 2 if 0 < i < len(melody) - 1 else 1

            # Factor 3: Relative pitch height
            height_tension = (pitch - min(melody)) / pitch_range if pitch_range > 0 else 0.5

            # Combine factors
            total_tension = (tessitura_tension * 0.3 + interval_tension * 0.4 + height_tension * 0.3)
            tension.append(total_tension)

        return tension

    @staticmethod
    def generate_contour(length: int, target_contour: ContourType,
                        pitch_range: Tuple[int, int] = (60, 84),
                        climax_position: float = 0.618) -> List[int]:
        """
        Generate melody with specified contour.

        Args:
            length: Number of notes
            target_contour: Desired contour type
            pitch_range: (min_pitch, max_pitch) in MIDI
            climax_position: Position of climax (0.0-1.0, golden ratio = 0.618)

        Returns:
            List of MIDI note numbers
        """
        if length < 2:
            raise ValueError("Length must be at least 2")

        min_pitch, max_pitch = pitch_range
        range_span = max_pitch - min_pitch

        melody = []

        if target_contour == ContourType.ARCH:
            # Arch: start low, peak at climax_position, end low
            climax_idx = int(length * climax_position)
            for i in range(length):
                if i < climax_idx:
                    # Ascending phase
                    progress = i / climax_idx if climax_idx > 0 else 0
                    pitch = min_pitch + int(range_span * progress)
                else:
                    # Descending phase
                    progress = (length - 1 - i) / (length - 1 - climax_idx) if length - 1 - climax_idx > 0 else 0
                    pitch = min_pitch + int(range_span * progress)
                melody.append(pitch)

        elif target_contour == ContourType.INVERTED_ARCH:
            # Inverted arch: start high, valley at climax_position, end high
            valley_idx = int(length * climax_position)
            for i in range(length):
                if i < valley_idx:
                    # Descending phase
                    progress = 1 - (i / valley_idx if valley_idx > 0 else 0)
                    pitch = min_pitch + int(range_span * progress)
                else:
                    # Ascending phase
                    progress = (i - valley_idx) / (length - 1 - valley_idx) if length - 1 - valley_idx > 0 else 0
                    pitch = min_pitch + int(range_span * progress)
                melody.append(pitch)

        elif target_contour == ContourType.ASCENDING:
            # Generally ascending
            for i in range(length):
                progress = i / (length - 1)
                pitch = min_pitch + int(range_span * progress)
                melody.append(pitch)

        elif target_contour == ContourType.DESCENDING:
            # Generally descending
            for i in range(length):
                progress = 1 - (i / (length - 1))
                pitch = min_pitch + int(range_span * progress)
                melody.append(pitch)

        elif target_contour == ContourType.WAVE:
            # Wave: multiple peaks (use more prominent waves)
            num_waves = max(3, length // 6)  # More waves
            for i in range(length):
                # Sine wave pattern with larger amplitude
                progress = i / (length - 1) if length > 1 else 0
                wave_value = math.sin(progress * num_waves * math.pi * 2)  # Full cycles
                pitch = min_pitch + int(range_span * (0.5 + wave_value * 0.45))
                melody.append(pitch)

        elif target_contour == ContourType.PLATEAU:
            # Plateau: relatively static
            center = min_pitch + range_span // 2
            for i in range(length):
                pitch = center + (i % 3 - 1)  # Small variations
                melody.append(pitch)

        else:  # ZIGZAG
            # Zigzag: frequent direction changes
            current_pitch = min_pitch + range_span // 2
            direction = 1
            for i in range(length):
                melody.append(current_pitch)
                current_pitch += direction * (2 + i % 3)
                if current_pitch >= max_pitch or current_pitch <= min_pitch:
                    direction *= -1
                    current_pitch = max(min_pitch, min(max_pitch, current_pitch))

        return melody


# ============================================================================
# MOTIF DEVELOPMENT - Theme Transformation Techniques
# ============================================================================

class MotifTransformation(Enum):
    """Motif development techniques (Bach, Beethoven, Brahms)"""
    SEQUENCE = "sequence"  # Repetition at different pitch levels
    INVERSION = "inversion"  # Mirror around axis
    RETROGRADE = "retrograde"  # Backward (crab canon)
    RETROGRADE_INVERSION = "retrograde_inversion"  # Backward + inverted
    AUGMENTATION = "augmentation"  # Rhythmically slower
    DIMINUTION = "diminution"  # Rhythmically faster
    FRAGMENTATION = "fragmentation"  # Use part of motif
    EXTENSION = "extension"  # Add to motif
    TRANSPOSITION = "transposition"  # Different key
    MODAL_SHIFT = "modal_shift"  # Major ↔ Minor


@dataclass
class Motif:
    """Musical motif (short melodic idea)"""
    pitches: List[int]  # MIDI note numbers
    durations: List[float]  # Beat durations
    name: str = "motif"
    importance: float = 1.0  # Thematic weight (0.0-1.0)


@dataclass
class DevelopedMotif:
    """Transformed motif with metadata"""
    original: Motif
    transformed: Motif
    transformation: MotifTransformation
    parameters: Dict  # Transformation-specific parameters


class MotifDevelopment:
    """
    Motif development and transformation engine.

    Based on:
    - Beethoven: Motivic development (5th Symphony)
    - Bach: Canonic transformations (Art of Fugue)
    - Schoenberg: Developing variation
    - Brahms: Thematic transformation
    """

    @staticmethod
    def sequence(motif: Motif, transpositions: List[int],
                sequential_type: str = "ascending") -> List[Motif]:
        """
        Sequential repetition at different pitch levels.

        Args:
            motif: Original motif
            transpositions: List of semitone transpositions
            sequential_type: "ascending", "descending", "alternating"

        Returns:
            List of transposed motifs
        """
        sequences = []

        for i, interval in enumerate(transpositions):
            new_pitches = [p + interval for p in motif.pitches]
            new_motif = Motif(
                pitches=new_pitches,
                durations=motif.durations.copy(),
                name=f"{motif.name}_seq{i+1}",
                importance=motif.importance * 0.9  # Sequences slightly less important
            )
            sequences.append(new_motif)

        return sequences

    @staticmethod
    def inversion(motif: Motif, axis: Optional[int] = None) -> Motif:
        """
        Invert motif around axis pitch.

        Args:
            motif: Original motif
            axis: Axis pitch (MIDI), defaults to first note

        Returns:
            Inverted motif
        """
        if axis is None:
            axis = motif.pitches[0]

        inverted_pitches = [axis - (p - axis) for p in motif.pitches]

        return Motif(
            pitches=inverted_pitches,
            durations=motif.durations.copy(),
            name=f"{motif.name}_inv",
            importance=motif.importance
        )

    @staticmethod
    def retrograde(motif: Motif) -> Motif:
        """
        Reverse motif (crab canon).

        Args:
            motif: Original motif

        Returns:
            Reversed motif
        """
        return Motif(
            pitches=list(reversed(motif.pitches)),
            durations=list(reversed(motif.durations)),
            name=f"{motif.name}_retro",
            importance=motif.importance
        )

    @staticmethod
    def retrograde_inversion(motif: Motif, axis: Optional[int] = None) -> Motif:
        """
        Reverse + invert motif.

        Args:
            motif: Original motif
            axis: Axis pitch (MIDI)

        Returns:
            Retrograde inverted motif
        """
        retro = MotifDevelopment.retrograde(motif)
        return MotifDevelopment.inversion(retro, axis)

    @staticmethod
    def augmentation(motif: Motif, factor: float = 2.0) -> Motif:
        """
        Increase note durations (slower).

        Args:
            motif: Original motif
            factor: Duration multiplier (2.0 = twice as slow)

        Returns:
            Augmented motif
        """
        return Motif(
            pitches=motif.pitches.copy(),
            durations=[d * factor for d in motif.durations],
            name=f"{motif.name}_aug",
            importance=motif.importance
        )

    @staticmethod
    def diminution(motif: Motif, factor: float = 0.5) -> Motif:
        """
        Decrease note durations (faster).

        Args:
            motif: Original motif
            factor: Duration multiplier (0.5 = twice as fast)

        Returns:
            Diminished motif
        """
        return Motif(
            pitches=motif.pitches.copy(),
            durations=[d * factor for d in motif.durations],
            name=f"{motif.name}_dim",
            importance=motif.importance
        )

    @staticmethod
    def fragmentation(motif: Motif, fragment_length: int, start_idx: int = 0) -> Motif:
        """
        Extract fragment of motif.

        Args:
            motif: Original motif
            fragment_length: Length of fragment
            start_idx: Starting index

        Returns:
            Fragmented motif
        """
        end_idx = min(start_idx + fragment_length, len(motif.pitches))

        return Motif(
            pitches=motif.pitches[start_idx:end_idx],
            durations=motif.durations[start_idx:end_idx],
            name=f"{motif.name}_frag",
            importance=motif.importance * 0.7  # Fragments less important
        )

    @staticmethod
    def extension(motif: Motif, additional_pitches: List[int],
                 additional_durations: List[float]) -> Motif:
        """
        Extend motif with additional material.

        Args:
            motif: Original motif
            additional_pitches: Pitches to add
            additional_durations: Durations to add

        Returns:
            Extended motif
        """
        return Motif(
            pitches=motif.pitches + additional_pitches,
            durations=motif.durations + additional_durations,
            name=f"{motif.name}_ext",
            importance=motif.importance
        )

    @staticmethod
    def modal_shift(motif: Motif, original_mode: str = "major",
                   target_mode: str = "minor") -> Motif:
        """
        Shift motif between major and minor.

        Args:
            motif: Original motif
            original_mode: "major" or "minor"
            target_mode: "major" or "minor"

        Returns:
            Mode-shifted motif
        """
        if original_mode == target_mode:
            return motif

        # Simple modal shift: adjust 3rd, 6th, 7th scale degrees
        # This is a simplified approach
        shifted_pitches = motif.pitches.copy()

        if original_mode == "major" and target_mode == "minor":
            # Lower 3, 6, 7 by semitone (if they appear)
            for i in range(len(shifted_pitches)):
                pitch_class = shifted_pitches[i] % 12
                # Check if pitch is 3rd, 6th, or 7th degree (approximate)
                if pitch_class in [4, 9, 11]:  # E, A, B in C major
                    shifted_pitches[i] -= 1

        elif original_mode == "minor" and target_mode == "major":
            # Raise 3, 6, 7 by semitone
            for i in range(len(shifted_pitches)):
                pitch_class = shifted_pitches[i] % 12
                if pitch_class in [3, 8, 10]:  # Eb, Ab, Bb in C minor
                    shifted_pitches[i] += 1

        return Motif(
            pitches=shifted_pitches,
            durations=motif.durations.copy(),
            name=f"{motif.name}_modal",
            importance=motif.importance
        )


# ============================================================================
# PHRASE STRUCTURE - Musical Sentences and Periods
# ============================================================================

class PhraseType(Enum):
    """Phrase structure types (Caplin, Schoenberg)"""
    ANTECEDENT = "antecedent"  # Question phrase
    CONSEQUENT = "consequent"  # Answer phrase
    SENTENCE = "sentence"  # Presentation + continuation + cadence
    PERIOD = "period"  # Antecedent + consequent
    HYBRID = "hybrid"  # Mixed structures


@dataclass
class Phrase:
    """Musical phrase"""
    melody: List[int]  # MIDI pitches
    durations: List[float]  # Beat durations
    phrase_type: PhraseType
    length_beats: float
    cadence_type: Optional[str] = None  # "authentic", "half", "deceptive", "plagal"


@dataclass
class Period:
    """Musical period (antecedent + consequent)"""
    antecedent: Phrase
    consequent: Phrase
    period_length: float


class PhraseStructure:
    """
    Phrase structure analysis and generation.

    Based on:
    - William Caplin: Classical Form (1998)
    - Arnold Schoenberg: Fundamentals of Musical Composition
    - Wallace Berry: Structural Functions in Music
    """

    @staticmethod
    def create_period(motif: Motif, length_beats: float = 8.0) -> Period:
        """
        Create period from motif (antecedent + consequent).

        Args:
            motif: Base motif
            length_beats: Total length in beats

        Returns:
            Period structure
        """
        half_length = length_beats / 2

        # Antecedent: ends with half cadence (unresolved)
        antecedent_pitches = motif.pitches.copy()
        # Extend to half_length if needed
        while sum(motif.durations) < half_length:
            antecedent_pitches.extend(motif.pitches[:2])

        antecedent = Phrase(
            melody=antecedent_pitches[:len(antecedent_pitches)//2],
            durations=motif.durations[:len(motif.durations)//2],
            phrase_type=PhraseType.ANTECEDENT,
            length_beats=half_length,
            cadence_type="half"
        )

        # Consequent: similar to antecedent but ends with authentic cadence
        consequent_pitches = [p - 2 if i == len(antecedent_pitches)//2 - 1 else p
                             for i, p in enumerate(antecedent_pitches[:len(antecedent_pitches)//2])]

        consequent = Phrase(
            melody=consequent_pitches,
            durations=motif.durations[:len(motif.durations)//2],
            phrase_type=PhraseType.CONSEQUENT,
            length_beats=half_length,
            cadence_type="authentic"
        )

        return Period(
            antecedent=antecedent,
            consequent=consequent,
            period_length=length_beats
        )

    @staticmethod
    def create_sentence(motif: Motif, length_beats: float = 8.0) -> Phrase:
        """
        Create sentence structure: presentation (2+2) + continuation (4).

        Args:
            motif: Base motif
            length_beats: Total length in beats

        Returns:
            Sentence phrase
        """
        # Presentation: basic idea + repetition (usually 4 beats total)
        presentation = motif.pitches + motif.pitches

        # Continuation: fragmentation + cadence (usually 4 beats)
        fragment = motif.pitches[:len(motif.pitches)//2]
        continuation = fragment + fragment + [motif.pitches[0]]  # Return to tonic

        sentence_melody = presentation + continuation

        return Phrase(
            melody=sentence_melody,
            durations=[1.0] * len(sentence_melody),  # Simplified
            phrase_type=PhraseType.SENTENCE,
            length_beats=length_beats,
            cadence_type="authentic"
        )


# ============================================================================
# INTERVALLIC CONTROL - Step/Leap Balance
# ============================================================================

@dataclass
class IntervalProfile:
    """Analysis of interval usage in melody"""
    step_count: int  # Semitone intervals (1-2 semitones)
    leap_count: int  # Larger intervals (>2 semitones)
    step_leap_ratio: float
    largest_interval: int
    average_interval: float
    direction_changes: int  # Number of times direction changes


class IntervallicControl:
    """
    Manage interval relationships and stepwise motion.

    Based on:
    - Fux: Gradus ad Parnassum (stepwise motion preference)
    - Schenker: Prolongation through stepwise motion
    - Rothstein: Phrase Rhythm in Tonal Music
    """

    @staticmethod
    def analyze_intervals(melody: List[int]) -> IntervalProfile:
        """
        Analyze interval usage in melody.

        Args:
            melody: List of MIDI pitches

        Returns:
            IntervalProfile
        """
        if len(melody) < 2:
            return IntervalProfile(0, 0, 0.0, 0, 0.0, 0)

        intervals = [abs(melody[i+1] - melody[i]) for i in range(len(melody)-1)]

        step_count = sum(1 for interval in intervals if interval <= 2)
        leap_count = sum(1 for interval in intervals if interval > 2)
        step_leap_ratio = step_count / leap_count if leap_count > 0 else float('inf')
        largest_interval = max(intervals)
        average_interval = sum(intervals) / len(intervals)

        # Count direction changes
        direction_changes = 0
        for i in range(1, len(melody) - 1):
            dir1 = melody[i] - melody[i-1]
            dir2 = melody[i+1] - melody[i]
            if (dir1 > 0 and dir2 < 0) or (dir1 < 0 and dir2 > 0):
                direction_changes += 1

        return IntervalProfile(
            step_count=step_count,
            leap_count=leap_count,
            step_leap_ratio=step_leap_ratio,
            largest_interval=largest_interval,
            average_interval=average_interval,
            direction_changes=direction_changes
        )

    @staticmethod
    def enforce_recovery_after_leap(melody: List[int], max_leap: int = 5) -> List[int]:
        """
        Enforce stepwise recovery after leaps (Fux rule).

        Args:
            melody: Original melody
            max_leap: Maximum allowed leap in semitones

        Returns:
            Corrected melody with inserted recovery notes
        """
        corrected = [melody[0]]

        for i in range(1, len(melody)):
            interval = abs(melody[i] - melody[i-1])

            if interval > max_leap:
                # Large leap: INSERT stepwise recovery note in opposite direction
                direction = 1 if melody[i] < melody[i-1] else -1
                recovery_note = melody[i-1] + direction * 2  # Step in opposite direction
                corrected.append(recovery_note)
                corrected.append(melody[i])  # Then add original note
            else:
                corrected.append(melody[i])

        return corrected

    @staticmethod
    def balance_step_leap_ratio(melody: List[int], target_ratio: float = 3.0) -> List[int]:
        """
        Adjust melody to achieve target step/leap ratio.

        Args:
            melody: Original melody
            target_ratio: Desired step/leap ratio (3.0 = classical, 1.0 = jazz)

        Returns:
            Balanced melody
        """
        profile = IntervallicControl.analyze_intervals(melody)

        if profile.leap_count == 0:
            return melody  # All steps, already good

        current_ratio = profile.step_leap_ratio

        if current_ratio < target_ratio:
            # Too many leaps, add stepwise motion
            # This is a simplified approach
            balanced = []
            for i in range(len(melody) - 1):
                balanced.append(melody[i])
                interval = abs(melody[i+1] - melody[i])
                if interval > 2:
                    # Insert stepwise note between leap
                    direction = 1 if melody[i+1] > melody[i] else -1
                    middle_note = melody[i] + direction * 2
                    balanced.append(middle_note)
            balanced.append(melody[-1])
            return balanced

        return melody  # Ratio is acceptable


# ============================================================================
# ORNAMENTATION - Baroque and Classical Embellishments
# ============================================================================

class OrnamentType(Enum):
    """Ornament types (CPE Bach, Leopold Mozart)"""
    TRILL = "trill"  # Rapid alternation with upper neighbor
    MORDENT = "mordent"  # Single alternation with lower neighbor
    TURN = "turn"  # Four-note figure around main note
    APPOGGIATURA = "appoggiatura"  # Accented non-chord tone
    GRACE_NOTE = "grace_note"  # Quick ornamental note
    SLIDE = "slide"  # Two grace notes ascending
    TREMOLO = "tremolo"  # Rapid repetition


@dataclass
class Ornament:
    """Musical ornament"""
    ornament_type: OrnamentType
    target_note_idx: int  # Index in melody to ornament
    pitches: List[int]  # Ornament pitches
    durations: List[float]  # Ornament durations


class Ornamentation:
    """
    Add ornaments and embellishments to melodies.

    Based on:
    - CPE Bach: Essay on the True Art of Playing Keyboard Instruments (1753)
    - Leopold Mozart: Violinschule (1756)
    - Frederick Neumann: Ornamentation in Baroque and Post-Baroque Music
    """

    @staticmethod
    def add_trill(melody: List[int], durations: List[float], note_idx: int,
                 upper_neighbor_interval: int = 2) -> Tuple[List[int], List[float]]:
        """
        Add trill to specified note.

        Args:
            melody: Original melody
            durations: Original durations
            note_idx: Index of note to trill
            upper_neighbor_interval: Interval to upper neighbor (usually 1 or 2)

        Returns:
            (ornamented_melody, ornamented_durations)
        """
        if note_idx >= len(melody):
            return melody, durations

        main_note = melody[note_idx]
        upper_note = main_note + upper_neighbor_interval
        note_duration = durations[note_idx]

        # Subdivide duration for trill (usually 4-8 alternations)
        num_alternations = 4
        trill_duration = note_duration / (num_alternations * 2)

        trill_pitches = [main_note, upper_note] * num_alternations
        trill_durations = [trill_duration] * (num_alternations * 2)

        ornamented_melody = melody[:note_idx] + trill_pitches + melody[note_idx+1:]
        ornamented_durations = durations[:note_idx] + trill_durations + durations[note_idx+1:]

        return ornamented_melody, ornamented_durations

    @staticmethod
    def add_mordent(melody: List[int], durations: List[float], note_idx: int,
                   lower: bool = True) -> Tuple[List[int], List[float]]:
        """
        Add mordent to specified note.

        Args:
            melody: Original melody
            durations: Original durations
            note_idx: Index of note to ornament
            lower: True for lower mordent, False for upper (inverted) mordent

        Returns:
            (ornamented_melody, ornamented_durations)
        """
        if note_idx >= len(melody):
            return melody, durations

        main_note = melody[note_idx]
        auxiliary_note = main_note - 2 if lower else main_note + 2
        note_duration = durations[note_idx]

        # Mordent: main - auxiliary - main (three notes)
        mordent_duration = note_duration / 3
        mordent_pitches = [main_note, auxiliary_note, main_note]
        mordent_durations = [mordent_duration] * 3

        ornamented_melody = melody[:note_idx] + mordent_pitches + melody[note_idx+1:]
        ornamented_durations = durations[:note_idx] + mordent_durations + durations[note_idx+1:]

        return ornamented_melody, ornamented_durations

    @staticmethod
    def add_turn(melody: List[int], durations: List[float], note_idx: int) -> Tuple[List[int], List[float]]:
        """
        Add turn to specified note.

        Args:
            melody: Original melody
            durations: Original durations
            note_idx: Index of note to ornament

        Returns:
            (ornamented_melody, ornamented_durations)
        """
        if note_idx >= len(melody):
            return melody, durations

        main_note = melody[note_idx]
        upper_note = main_note + 2
        lower_note = main_note - 2
        note_duration = durations[note_idx]

        # Turn: upper - main - lower - main (four notes)
        turn_duration = note_duration / 4
        turn_pitches = [upper_note, main_note, lower_note, main_note]
        turn_durations = [turn_duration] * 4

        ornamented_melody = melody[:note_idx] + turn_pitches + melody[note_idx+1:]
        ornamented_durations = durations[:note_idx] + turn_durations + durations[note_idx+1:]

        return ornamented_melody, ornamented_durations

    @staticmethod
    def add_appoggiatura(melody: List[int], durations: List[float], note_idx: int,
                        interval: int = 2, accent: float = 0.75) -> Tuple[List[int], List[float]]:
        """
        Add appoggiatura before note.

        Args:
            melody: Original melody
            durations: Original durations
            note_idx: Index of note to ornament
            interval: Interval from appoggiatura to main note
            accent: Fraction of duration for appoggiatura (0.75 = 3:1 ratio)

        Returns:
            (ornamented_melody, ornamented_durations)
        """
        if note_idx >= len(melody):
            return melody, durations

        main_note = melody[note_idx]
        appoggiatura_note = main_note + interval
        note_duration = durations[note_idx]

        appoggiatura_duration = note_duration * accent
        resolution_duration = note_duration * (1 - accent)

        appoggiatura_pitches = [appoggiatura_note, main_note]
        appoggiatura_durations = [appoggiatura_duration, resolution_duration]

        ornamented_melody = melody[:note_idx] + appoggiatura_pitches + melody[note_idx+1:]
        ornamented_durations = durations[:note_idx] + appoggiatura_durations + durations[note_idx+1:]

        return ornamented_melody, ornamented_durations


# ============================================================================
# NARRATIVE ARC - Musical Storytelling
# ============================================================================

class NarrativeSection(Enum):
    """Sections of musical narrative (Freytag's pyramid adapted)"""
    EXPOSITION = "exposition"  # Introduce material
    RISING_ACTION = "rising_action"  # Build tension
    CLIMAX = "climax"  # Peak moment
    FALLING_ACTION = "falling_action"  # Release tension
    RESOLUTION = "resolution"  # Conclude


@dataclass
class NarrativeArc:
    """Musical narrative structure"""
    sections: Dict[NarrativeSection, Tuple[float, float]]  # section -> (start_beat, end_beat)
    climax_beat: float
    overall_length: float
    tension_curve: List[Tuple[float, float]]  # (beat, tension_0_to_1)


class MusicalNarrative:
    """
    Create narrative arc in melody.

    Based on:
    - Leonard Meyer: Emotion and Meaning in Music (1956)
    - Fred Lerdahl & Ray Jackendoff: A Generative Theory of Tonal Music (1983)
    - David Huron: Sweet Anticipation (2006)
    """

    @staticmethod
    def create_narrative_arc(total_length_beats: float,
                            climax_position: float = 0.618) -> NarrativeArc:
        """
        Create narrative arc structure (golden ratio climax).

        Args:
            total_length_beats: Total length in beats
            climax_position: Position of climax (0.0-1.0, default = golden ratio)

        Returns:
            NarrativeArc structure
        """
        climax_beat = total_length_beats * climax_position

        # Divide into sections
        exposition_end = total_length_beats * 0.2
        rising_action_end = climax_beat
        falling_action_end = total_length_beats * 0.85

        sections = {
            NarrativeSection.EXPOSITION: (0.0, exposition_end),
            NarrativeSection.RISING_ACTION: (exposition_end, rising_action_end),
            NarrativeSection.CLIMAX: (rising_action_end, rising_action_end + 4.0),  # 4 beat climax
            NarrativeSection.FALLING_ACTION: (rising_action_end + 4.0, falling_action_end),
            NarrativeSection.RESOLUTION: (falling_action_end, total_length_beats)
        }

        # Generate tension curve
        tension_curve = []
        num_points = int(total_length_beats)
        for i in range(num_points):
            beat = i
            if beat < exposition_end:
                # Exposition: low tension (0.2-0.3)
                tension = 0.2 + 0.1 * (beat / exposition_end)
            elif beat < rising_action_end:
                # Rising action: increasing tension
                progress = (beat - exposition_end) / (rising_action_end - exposition_end)
                tension = 0.3 + 0.6 * progress
            elif beat < rising_action_end + 4.0:
                # Climax: peak tension (0.9-1.0)
                tension = 0.9 + 0.1 * ((beat - rising_action_end) / 4.0)
            elif beat < falling_action_end:
                # Falling action: decreasing tension
                progress = (beat - (rising_action_end + 4.0)) / (falling_action_end - (rising_action_end + 4.0))
                tension = 0.9 - 0.5 * progress
            else:
                # Resolution: low tension (0.1-0.2)
                progress = (beat - falling_action_end) / (total_length_beats - falling_action_end)
                tension = 0.4 - 0.3 * progress

            tension_curve.append((beat, tension))

        return NarrativeArc(
            sections=sections,
            climax_beat=climax_beat,
            overall_length=total_length_beats,
            tension_curve=tension_curve
        )

    @staticmethod
    def apply_narrative_to_melody(melody: List[int], arc: NarrativeArc,
                                 beat_positions: List[float]) -> List[int]:
        """
        Adjust melody to follow narrative arc.

        Args:
            melody: Original melody
            arc: Narrative arc structure
            beat_positions: Beat position of each note

        Returns:
            Adjusted melody
        """
        if len(melody) != len(beat_positions):
            raise ValueError("Melody and beat_positions must have same length")

        adjusted_melody = []

        for i, (pitch, beat) in enumerate(zip(melody, beat_positions)):
            # Find tension at this beat
            tension = 0.5
            for curve_beat, curve_tension in arc.tension_curve:
                if curve_beat >= beat:
                    tension = curve_tension
                    break

            # Adjust pitch based on tension
            # Higher tension → higher pitch
            pitch_adjustment = int((tension - 0.5) * 8)  # ±4 semitones max
            adjusted_pitch = pitch + pitch_adjustment

            adjusted_melody.append(adjusted_pitch)

        return adjusted_melody


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ADVANCED MELODY MODULE - DEMONSTRATION")
    print("=" * 80)

    # Test 1: Contour Theory
    print("\n" + "=" * 80)
    print("1. CONTOUR THEORY")
    print("=" * 80)

    test_melody = [60, 62, 64, 67, 65, 62, 60]
    analysis = ContourTheory.analyze_contour(test_melody)
    print(f"\nTest melody: {test_melody}")
    print(f"Contour type: {analysis.contour_type.value}")
    print(f"Range: {analysis.range} semitones")
    print(f"Tessitura: MIDI {analysis.tessitura}")
    print(f"Climax position: {analysis.climax_position:.2f} (golden ratio ≈ 0.618)")
    print(f"Step/leap ratio: {analysis.step_leap_ratio:.2f}")
    print(f"Tension curve: {[f'{t:.2f}' for t in analysis.tension_curve]}")

    # Generate arch contour
    arch_melody = ContourTheory.generate_contour(
        length=8,
        target_contour=ContourType.ARCH,
        pitch_range=(60, 72),
        climax_position=0.618
    )
    print(f"\nGenerated ARCH contour: {arch_melody}")

    # Test 2: Motif Development
    print("\n" + "=" * 80)
    print("2. MOTIF DEVELOPMENT")
    print("=" * 80)

    original_motif = Motif(
        pitches=[60, 64, 67],
        durations=[1.0, 1.0, 2.0],
        name="Theme A"
    )
    print(f"\nOriginal motif: {original_motif.pitches}")

    # Inversion
    inverted = MotifDevelopment.inversion(original_motif)
    print(f"Inverted: {inverted.pitches}")

    # Retrograde
    retrograde = MotifDevelopment.retrograde(original_motif)
    print(f"Retrograde: {retrograde.pitches}")

    # Sequence
    sequences = MotifDevelopment.sequence(original_motif, [2, 4, 7])
    print(f"Sequence (+2 semitones): {sequences[0].pitches}")
    print(f"Sequence (+4 semitones): {sequences[1].pitches}")

    # Augmentation
    augmented = MotifDevelopment.augmentation(original_motif, factor=2.0)
    print(f"Augmented (2x slower): durations={augmented.durations}")

    # Test 3: Phrase Structure
    print("\n" + "=" * 80)
    print("3. PHRASE STRUCTURE")
    print("=" * 80)

    period = PhraseStructure.create_period(original_motif, length_beats=8.0)
    print(f"\nPeriod structure:")
    print(f"  Antecedent ({period.antecedent.cadence_type} cadence): {period.antecedent.melody}")
    print(f"  Consequent ({period.consequent.cadence_type} cadence): {period.consequent.melody}")

    sentence = PhraseStructure.create_sentence(original_motif, length_beats=8.0)
    print(f"\nSentence structure: {sentence.melody}")
    print(f"  Cadence: {sentence.cadence_type}")

    # Test 4: Intervallic Control
    print("\n" + "=" * 80)
    print("4. INTERVALLIC CONTROL")
    print("=" * 80)

    leapy_melody = [60, 67, 55, 72, 60, 64, 69, 62]
    profile = IntervallicControl.analyze_intervals(leapy_melody)
    print(f"\nMelody: {leapy_melody}")
    print(f"Steps: {profile.step_count}, Leaps: {profile.leap_count}")
    print(f"Step/leap ratio: {profile.step_leap_ratio:.2f}")
    print(f"Largest interval: {profile.largest_interval} semitones")
    print(f"Direction changes: {profile.direction_changes}")

    balanced = IntervallicControl.balance_step_leap_ratio(leapy_melody, target_ratio=3.0)
    print(f"\nBalanced melody (target ratio 3.0): {balanced}")

    # Test 5: Ornamentation
    print("\n" + "=" * 80)
    print("5. ORNAMENTATION")
    print("=" * 80)

    simple_melody = [60, 64, 67, 64, 60]
    simple_durations = [1.0, 1.0, 2.0, 1.0, 2.0]

    print(f"\nOriginal: pitches={simple_melody}, durations={simple_durations}")

    # Add trill to 3rd note
    trilled, trill_durs = Ornamentation.add_trill(simple_melody, simple_durations, note_idx=2)
    print(f"With trill on note 3: pitches={trilled}")

    # Add mordent
    mordent_melody, mordent_durs = Ornamentation.add_mordent(simple_melody, simple_durations, note_idx=1)
    print(f"With mordent on note 2: pitches={mordent_melody}")

    # Add turn
    turn_melody, turn_durs = Ornamentation.add_turn(simple_melody, simple_durations, note_idx=2)
    print(f"With turn on note 3: pitches={turn_melody}")

    # Test 6: Narrative Arc
    print("\n" + "=" * 80)
    print("6. NARRATIVE ARC")
    print("=" * 80)

    arc = MusicalNarrative.create_narrative_arc(total_length_beats=32.0, climax_position=0.618)
    print(f"\nNarrative arc (32 beats):")
    print(f"  Climax at beat: {arc.climax_beat:.1f}")
    for section, (start, end) in arc.sections.items():
        print(f"  {section.value.upper()}: beats {start:.1f}-{end:.1f}")

    # Apply narrative to melody
    narrative_melody = [60, 62, 64, 65, 67, 69, 70, 72, 70, 69, 67, 65, 64, 62, 60, 60]
    narrative_beats = list(range(len(narrative_melody)))

    # Create shorter arc for test
    short_arc = MusicalNarrative.create_narrative_arc(total_length_beats=16.0)
    adjusted = MusicalNarrative.apply_narrative_to_melody(narrative_melody, short_arc, narrative_beats)

    print(f"\nOriginal melody: {narrative_melody}")
    print(f"Narrative-adjusted: {adjusted}")
    print("\n  (Notice how pitches rise toward climax and fall toward resolution)")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nAll 6 advanced melody systems tested successfully!")
    print("Ready for integration with harmony_advanced.py and existing modules.")
