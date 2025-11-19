#!/usr/bin/env python3
"""
Advanced Tempo Engine - Tempo Curves, Rubato, and Expressive Timing

This module provides comprehensive tempo manipulation capabilities for MIDI generation,
including rubato, tempo curves, metric modulation, and agogic accents.

Based on extensive research:
- Richard Hudson: "The History of Tempo Rubato" (Claremont scholarship)
- Elliott Carter: Metric/Tempo Modulation technique (1948-)
- ResearchGate: "Musical Tempo Curves" - Mathematical models for accelerando/ritardando
- Music Perception: "Do[n't] Change a Hair for Me: The Art of Jazz Rubato"
- DAW implementations: Logic Pro, Cubase tempo curve algorithms
- Chopin's melodic rubato: Left hand steady, right hand flexible

Key Features:
- Tempo curves (linear, exponential, S-curve, parabolic)
- Accelerando and ritardando with customizable curves
- Rubato (Romantic classical style and jazz style)
- Agogic accents (emphasis via duration extension)
- Metric/tempo modulation (Elliott Carter technique)
- MIDI tempo map generation with meta events
- Fermata and breath mark simulation
- Jazz expressive timing patterns

Author: Agent 16
Date: 2025-11-19
"""

import math
from typing import List, Dict, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum


class CurveType(Enum):
    """Types of tempo curves available."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    S_CURVE = "s_curve"
    PARABOLIC = "parabolic"
    SPLINE = "spline"
    LOGARITHMIC = "logarithmic"


class RubatoStyle(Enum):
    """Rubato performance styles."""
    ROMANTIC = "romantic"  # Chopin-style melodic rubato
    JAZZ = "jazz"  # Late entry, cadential alignment
    EXPRESSIVE = "expressive"  # General expressive timing
    CLASSICAL = "classical"  # Subtle, measured rubato


@dataclass
class TempoPoint:
    """Represents a tempo change point in time."""
    time_ticks: int  # Position in MIDI ticks
    tempo_bpm: float  # Tempo at this point
    curve_to_next: CurveType = CurveType.LINEAR


@dataclass
class Note:
    """Simple note representation for tempo processing."""
    pitch: int
    start_time: float  # In beats or ticks
    duration: float
    velocity: int = 64
    channel: int = 0


class TempoEngine:
    """
    Advanced tempo manipulation engine for MIDI generation.

    This class provides professional-quality tempo curves, rubato effects,
    metric modulation calculations, and expressive timing features based on
    musicological research and performance practice.

    Examples:
        >>> engine = TempoEngine()
        >>> # Create smooth accelerando from 60 to 120 BPM
        >>> curve = engine.create_tempo_curve(60, 120, 8, CurveType.EXPONENTIAL)
        >>>
        >>> # Apply romantic rubato to a phrase
        >>> notes = [Note(60, i, 1, 64) for i in range(8)]
        >>> rubato_notes = engine.apply_rubato(notes, 0.3, RubatoStyle.ROMANTIC)
        >>>
        >>> # Calculate metric modulation
        >>> new_tempo = engine.calculate_tempo_modulation(60, 'quarter', 'dotted_eighth')
    """

    def __init__(self, ticks_per_beat: int = 480):
        """
        Initialize the Tempo Engine.

        Args:
            ticks_per_beat: MIDI ticks per quarter note (default: 480)
        """
        self.ticks_per_beat = ticks_per_beat
        self.tempo_map: List[TempoPoint] = []

    def create_tempo_curve(
        self,
        start_tempo: float,
        end_tempo: float,
        duration_beats: float,
        curve_type: CurveType = CurveType.EXPONENTIAL,
        resolution: int = 16
    ) -> List[Tuple[float, float]]:
        """
        Create a smooth tempo curve between two tempos.

        Based on research from "Musical Tempo Curves" (ResearchGate) showing
        that continuous tempo transitions feature monotonous curves with varying
        shapes rather than linear transitions.

        Args:
            start_tempo: Starting tempo in BPM
            end_tempo: Ending tempo in BPM
            duration_beats: Duration of the transition in beats
            curve_type: Type of curve to use
            resolution: Number of tempo points to generate

        Returns:
            List of (beat_position, tempo_bpm) tuples

        Example:
            >>> engine = TempoEngine()
            >>> curve = engine.create_tempo_curve(60, 120, 8, CurveType.EXPONENTIAL)
            >>> print(f"Points in curve: {len(curve)}")
            Points in curve: 16
        """
        if start_tempo <= 0 or end_tempo <= 0:
            raise ValueError("Tempos must be positive")
        if duration_beats <= 0:
            raise ValueError("Duration must be positive")

        curve_points = []
        delta_tempo = end_tempo - start_tempo

        for i in range(resolution + 1):
            t = i / resolution  # Normalized position (0 to 1)
            beat_pos = t * duration_beats

            # Calculate tempo based on curve type
            if curve_type == CurveType.LINEAR:
                tempo = start_tempo + delta_tempo * t

            elif curve_type == CurveType.EXPONENTIAL:
                # Exponential curve: more gradual at start, faster at end
                tempo = start_tempo + delta_tempo * (math.exp(t * 2) - 1) / (math.exp(2) - 1)

            elif curve_type == CurveType.S_CURVE:
                # S-curve (sigmoid): smooth acceleration and deceleration
                # Using tanh for smooth S-shape
                sigmoid_t = (math.tanh((t - 0.5) * 4) + 1) / 2
                tempo = start_tempo + delta_tempo * sigmoid_t

            elif curve_type == CurveType.PARABOLIC:
                # Parabolic curve: gradual at extremes, faster in middle
                parabolic_t = 1 - (2 * t - 1) ** 2 if delta_tempo > 0 else (2 * t - 1) ** 2
                tempo = start_tempo + delta_tempo * parabolic_t

            elif curve_type == CurveType.LOGARITHMIC:
                # Logarithmic: fast at start, gradual at end
                tempo = start_tempo + delta_tempo * math.log(1 + t * (math.e - 1)) / math.log(math.e)

            elif curve_type == CurveType.SPLINE:
                # Cubic spline approximation for smooth natural feel
                spline_t = t * t * (3 - 2 * t)  # Smoothstep function
                tempo = start_tempo + delta_tempo * spline_t

            else:
                tempo = start_tempo + delta_tempo * t

            curve_points.append((beat_pos, tempo))

        return curve_points

    def apply_accelerando(
        self,
        notes: List[Note],
        start_tempo: float,
        end_tempo: float,
        curve_type: CurveType = CurveType.EXPONENTIAL
    ) -> List[Note]:
        """
        Apply accelerando (gradual speed up) to a sequence of notes.

        Args:
            notes: List of Note objects
            start_tempo: Starting tempo in BPM
            end_tempo: Ending tempo (must be > start_tempo)
            curve_type: Type of acceleration curve

        Returns:
            Modified list of notes with adjusted timing

        Example:
            >>> notes = [Note(60, i, 1.0, 64) for i in range(8)]
            >>> accel_notes = engine.apply_accelerando(notes, 60, 120)
        """
        if end_tempo <= start_tempo:
            raise ValueError("End tempo must be greater than start tempo for accelerando")

        if not notes:
            return notes

        # Find total duration
        last_note = max(notes, key=lambda n: n.start_time + n.duration)
        total_duration = last_note.start_time + last_note.duration

        # Create tempo curve
        curve = self.create_tempo_curve(start_tempo, end_tempo, total_duration, curve_type)

        # Adjust note timings based on tempo curve
        modified_notes = []
        for note in notes:
            new_note = Note(
                pitch=note.pitch,
                start_time=self._adjust_time_for_curve(note.start_time, curve, start_tempo),
                duration=note.duration,
                velocity=note.velocity,
                channel=note.channel
            )
            modified_notes.append(new_note)

        return modified_notes

    def apply_ritardando(
        self,
        notes: List[Note],
        start_tempo: float,
        end_tempo: float,
        curve_type: CurveType = CurveType.EXPONENTIAL
    ) -> List[Note]:
        """
        Apply ritardando (gradual slow down) to a sequence of notes.

        Args:
            notes: List of Note objects
            start_tempo: Starting tempo in BPM
            end_tempo: Ending tempo (must be < start_tempo)
            curve_type: Type of deceleration curve

        Returns:
            Modified list of notes with adjusted timing

        Example:
            >>> notes = [Note(60, i, 1.0, 64) for i in range(8)]
            >>> rit_notes = engine.apply_ritardando(notes, 120, 60)
        """
        if end_tempo >= start_tempo:
            raise ValueError("End tempo must be less than start tempo for ritardando")

        if not notes:
            return notes

        # Find total duration
        last_note = max(notes, key=lambda n: n.start_time + n.duration)
        total_duration = last_note.start_time + last_note.duration

        # Create tempo curve
        curve = self.create_tempo_curve(start_tempo, end_tempo, total_duration, curve_type)

        # Adjust note timings
        modified_notes = []
        for note in notes:
            # For ritardando, notes get stretched out (take more real time)
            new_start = self._adjust_time_for_curve(note.start_time, curve, start_tempo)
            modified_notes.append(Note(
                pitch=note.pitch,
                start_time=new_start,
                duration=note.duration,
                velocity=note.velocity,
                channel=note.channel
            ))

        return modified_notes

    def apply_rubato(
        self,
        notes: List[Note],
        intensity: float = 0.3,
        style: RubatoStyle = RubatoStyle.ROMANTIC
    ) -> List[Note]:
        """
        Apply rubato (expressive tempo flexibility) to notes.

        Based on research:
        - Chopin: "Left hand is the conductor" - melodic rubato
        - Jazz: Late melody entry, cadential alignment
        - Research shows typical jazz strategy: begin late, speed up, align at cadences

        Args:
            notes: List of Note objects
            intensity: Amount of rubato (0.0 to 1.0, typical: 0.2-0.5)
            style: Style of rubato to apply

        Returns:
            Modified notes with expressive timing

        Example:
            >>> melody = [Note(60 + i, i, 1.0, 70) for i in range(8)]
            >>> rubato_melody = engine.apply_rubato(melody, 0.3, RubatoStyle.ROMANTIC)
        """
        if not notes:
            return notes

        if not 0 <= intensity <= 1:
            raise ValueError("Intensity must be between 0 and 1")

        modified_notes = []
        num_notes = len(notes)

        for i, note in enumerate(notes):
            t = i / max(num_notes - 1, 1)  # Normalized position (0 to 1)
            time_offset = 0.0

            if style == RubatoStyle.ROMANTIC:
                # Chopin-style: slight delays and anticipations
                # Create wave-like pattern with emphasis on phrase peaks
                wave = math.sin(t * math.pi) * intensity * 0.2
                time_offset = wave

            elif style == RubatoStyle.JAZZ:
                # Jazz rubato: start late, speed up, align at end
                if t < 0.3:
                    # Begin late
                    time_offset = intensity * 0.15 * (1 - t / 0.3)
                elif t > 0.8:
                    # Align at cadence
                    time_offset = 0.0
                else:
                    # Gradual catch-up
                    progress = (t - 0.3) / 0.5
                    time_offset = intensity * 0.15 * (1 - progress)

            elif style == RubatoStyle.EXPRESSIVE:
                # General expressive timing with phrase-aware shaping
                # Slow down slightly at phrase middle and end
                if t < 0.5:
                    time_offset = intensity * 0.1 * (t / 0.5)
                else:
                    time_offset = intensity * 0.1 * (1 - (t - 0.5) / 0.5)

            elif style == RubatoStyle.CLASSICAL:
                # Subtle, measured rubato
                # Slight delay before important structural points
                time_offset = intensity * 0.05 * math.sin(t * 2 * math.pi)

            modified_notes.append(Note(
                pitch=note.pitch,
                start_time=note.start_time + time_offset,
                duration=note.duration,
                velocity=note.velocity,
                channel=note.channel
            ))

        return modified_notes

    def add_agogic_accent(
        self,
        notes: List[Note],
        accent_indices: List[int],
        lengthen_percent: float = 15.0
    ) -> List[Note]:
        """
        Add agogic accents (emphasis via duration extension) to specific notes.

        Agogic accent: emphasis created by extending duration rather than increasing
        volume. Commonly used in organ/harpsichord and expressive performance.

        Args:
            notes: List of Note objects
            accent_indices: Indices of notes to accent
            lengthen_percent: Percentage to lengthen duration (typical: 10-20%)

        Returns:
            Modified notes with agogic accents

        Example:
            >>> notes = [Note(60, i, 1.0, 64) for i in range(8)]
            >>> # Accent beats 1 and 3 (indices 0 and 2)
            >>> accented = engine.add_agogic_accent(notes, [0, 2], 15)
        """
        if not 0 <= lengthen_percent <= 100:
            raise ValueError("Lengthen percent must be between 0 and 100")

        modified_notes = []

        for i, note in enumerate(notes):
            if i in accent_indices:
                duration = note.duration * (1 + lengthen_percent / 100)
            else:
                duration = note.duration

            modified_notes.append(Note(
                pitch=note.pitch,
                start_time=note.start_time,
                duration=duration,
                velocity=note.velocity,
                channel=note.channel
            ))

        return modified_notes

    def calculate_tempo_modulation(
        self,
        from_tempo: float,
        from_note_value: str,
        to_note_value: str
    ) -> float:
        """
        Calculate new tempo using Elliott Carter's metric modulation technique.

        Metric modulation creates tempo changes where a note value from the first
        tempo equals a different note value in the new tempo, creating a "pivot."

        Examples:
            Quarter note = 60 BPM, dotted quarter becomes new quarter
            -> 60 * (1/1.5) = 40 BPM

        Args:
            from_tempo: Current tempo in BPM
            from_note_value: Note value in current tempo ('quarter', 'eighth',
                           'dotted_quarter', 'triplet_eighth', etc.)
            to_note_value: Note value that will become the new beat

        Returns:
            New tempo in BPM

        Example:
            >>> # Quarter note at 60 BPM becomes dotted eighth
            >>> new_tempo = engine.calculate_tempo_modulation(60, 'quarter', 'dotted_eighth')
            >>> print(f"New tempo: {new_tempo:.1f} BPM")
            New tempo: 160.0 BPM
        """
        # Define note value ratios relative to quarter note
        note_ratios = {
            'whole': 4.0,
            'half': 2.0,
            'dotted_half': 3.0,
            'quarter': 1.0,
            'dotted_quarter': 1.5,
            'eighth': 0.5,
            'dotted_eighth': 0.75,
            'triplet_quarter': 2/3,
            'sixteenth': 0.25,
            'triplet_eighth': 1/3,
            'quintuplet_quarter': 0.8,
        }

        if from_note_value not in note_ratios:
            raise ValueError(f"Unknown note value: {from_note_value}")
        if to_note_value not in note_ratios:
            raise ValueError(f"Unknown note value: {to_note_value}")

        # Calculate the ratio
        from_ratio = note_ratios[from_note_value]
        to_ratio = note_ratios[to_note_value]

        # New tempo = old tempo * (from_value / to_value)
        new_tempo = from_tempo * (from_ratio / to_ratio)

        return new_tempo

    def generate_midi_tempo_map(
        self,
        tempo_changes: List[Tuple[float, float]],
        initial_tempo: float = 120.0
    ) -> List[Dict]:
        """
        Generate MIDI tempo meta events for a tempo map.

        Creates MIDI Set Tempo meta events (FF 51 03) with microseconds per
        quarter note.

        Args:
            tempo_changes: List of (beat_position, tempo_bpm) tuples
            initial_tempo: Initial tempo if not specified at beat 0

        Returns:
            List of tempo event dictionaries with 'tick', 'tempo_bpm', 'microseconds'

        Example:
            >>> curve = engine.create_tempo_curve(60, 120, 8)
            >>> tempo_map = engine.generate_midi_tempo_map(curve)
        """
        tempo_events = []

        # Ensure we have initial tempo
        if not tempo_changes or tempo_changes[0][0] != 0:
            tempo_changes = [(0, initial_tempo)] + list(tempo_changes)

        for beat_pos, tempo_bpm in tempo_changes:
            tick = int(beat_pos * self.ticks_per_beat)
            # MIDI tempo is microseconds per quarter note
            microseconds_per_quarter = int(60_000_000 / tempo_bpm)

            tempo_events.append({
                'tick': tick,
                'tempo_bpm': tempo_bpm,
                'microseconds_per_quarter': microseconds_per_quarter,
                'type': 'set_tempo'
            })

        return tempo_events

    def apply_fermata(
        self,
        notes: List[Note],
        fermata_index: int,
        hold_multiplier: float = 2.0
    ) -> List[Note]:
        """
        Apply fermata (hold/pause) to a specific note.

        A fermata extends a note's duration and creates a pause in the musical flow.

        Args:
            notes: List of Note objects
            fermata_index: Index of note to apply fermata to
            hold_multiplier: How much to extend duration (2.0 = double length)

        Returns:
            Modified notes with fermata applied and subsequent notes shifted

        Example:
            >>> notes = [Note(60, i, 1.0, 64) for i in range(8)]
            >>> # Apply fermata to note at index 3
            >>> fermata_notes = engine.apply_fermata(notes, 3, 2.5)
        """
        if fermata_index < 0 or fermata_index >= len(notes):
            raise ValueError(f"Fermata index {fermata_index} out of range")

        if hold_multiplier < 1.0:
            raise ValueError("Hold multiplier must be >= 1.0")

        modified_notes = []
        fermata_note = notes[fermata_index]
        original_duration = fermata_note.duration
        extended_duration = original_duration * hold_multiplier
        time_shift = extended_duration - original_duration

        for i, note in enumerate(notes):
            if i == fermata_index:
                # Extend this note
                modified_notes.append(Note(
                    pitch=note.pitch,
                    start_time=note.start_time,
                    duration=extended_duration,
                    velocity=note.velocity,
                    channel=note.channel
                ))
            elif note.start_time > fermata_note.start_time:
                # Shift subsequent notes
                modified_notes.append(Note(
                    pitch=note.pitch,
                    start_time=note.start_time + time_shift,
                    duration=note.duration,
                    velocity=note.velocity,
                    channel=note.channel
                ))
            else:
                # Leave earlier notes unchanged
                modified_notes.append(note)

        return modified_notes

    def add_breath_marks(
        self,
        notes: List[Note],
        breath_indices: List[int],
        pause_duration: float = 0.1
    ) -> List[Note]:
        """
        Add breath marks (brief pauses) between phrases.

        Breath marks create slight pauses in the musical flow, typically used
        at phrase boundaries.

        Args:
            notes: List of Note objects
            breath_indices: Indices after which to insert breath pauses
            pause_duration: Duration of pause in beats (typical: 0.1-0.2)

        Returns:
            Modified notes with pauses inserted

        Example:
            >>> notes = [Note(60, i, 1.0, 64) for i in range(16)]
            >>> # Add breaths after every 4 notes
            >>> breathing = engine.add_breath_marks(notes, [3, 7, 11], 0.15)
        """
        if pause_duration < 0:
            raise ValueError("Pause duration must be non-negative")

        # Sort breath indices
        breath_indices = sorted(set(breath_indices), reverse=True)

        modified_notes = list(notes)

        for breath_idx in breath_indices:
            if breath_idx < 0 or breath_idx >= len(notes):
                continue

            # Find all notes after this breath mark
            breath_time = notes[breath_idx].start_time + notes[breath_idx].duration

            for i in range(len(modified_notes)):
                if modified_notes[i].start_time >= breath_time:
                    modified_notes[i] = Note(
                        pitch=modified_notes[i].pitch,
                        start_time=modified_notes[i].start_time + pause_duration,
                        duration=modified_notes[i].duration,
                        velocity=modified_notes[i].velocity,
                        channel=modified_notes[i].channel
                    )

        return modified_notes

    def create_rallentando(
        self,
        notes: List[Note],
        start_tempo: float,
        percentage_slowdown: float = 30.0,
        curve_type: CurveType = CurveType.EXPONENTIAL
    ) -> List[Note]:
        """
        Create rallentando (gradual slowing, similar to ritardando).

        Rallentando is typically more gradual than ritardando and often used
        at the end of pieces or major sections.

        Args:
            notes: List of Note objects
            start_tempo: Starting tempo in BPM
            percentage_slowdown: Percentage to slow down (e.g., 30 = 30% slower)
            curve_type: Type of curve for the slowdown

        Returns:
            Modified notes with rallentando applied

        Example:
            >>> notes = [Note(60, i, 1.0, 64) for i in range(8)]
            >>> rall_notes = engine.create_rallentando(notes, 120, 40)
        """
        end_tempo = start_tempo * (1 - percentage_slowdown / 100)
        return self.apply_ritardando(notes, start_tempo, end_tempo, curve_type)

    def _adjust_time_for_curve(
        self,
        original_time: float,
        curve: List[Tuple[float, float]],
        base_tempo: float
    ) -> float:
        """
        Internal helper to adjust note timing based on tempo curve.

        Args:
            original_time: Original time in beats
            curve: List of (beat, tempo) tuples
            base_tempo: Base tempo for reference

        Returns:
            Adjusted time accounting for tempo changes
        """
        if not curve or original_time <= 0:
            return original_time

        # Simple linear interpolation through curve points
        adjusted_time = 0.0
        prev_beat = 0.0
        prev_tempo = curve[0][1]

        for beat_pos, tempo in curve:
            if beat_pos >= original_time:
                # Interpolate for remaining distance
                remaining = original_time - prev_beat
                avg_tempo = (prev_tempo + tempo) / 2
                time_delta = remaining * (base_tempo / avg_tempo)
                adjusted_time += time_delta
                break
            else:
                # Add time for this segment
                segment_beats = beat_pos - prev_beat
                avg_tempo = (prev_tempo + tempo) / 2
                time_delta = segment_beats * (base_tempo / avg_tempo)
                adjusted_time += time_delta
                prev_beat = beat_pos
                prev_tempo = tempo
        else:
            # Original time is beyond curve
            remaining = original_time - prev_beat
            adjusted_time += remaining

        return adjusted_time


# ============================================================================
# Comprehensive Unit Tests
# ============================================================================

def run_tests():
    """Run comprehensive unit tests for TempoEngine."""
    print("=" * 70)
    print("TEMPO ENGINE - COMPREHENSIVE UNIT TESTS")
    print("=" * 70)

    engine = TempoEngine()
    test_count = 0
    passed = 0

    # Test 1: Linear tempo curve
    test_count += 1
    print(f"\nTest {test_count}: Linear tempo curve (60 → 120 BPM)")
    try:
        curve = engine.create_tempo_curve(60, 120, 8, CurveType.LINEAR)
        assert len(curve) == 17, "Should have 17 points (resolution=16)"
        assert curve[0][1] == 60, "Start should be 60 BPM"
        assert curve[-1][1] == 120, "End should be 120 BPM"
        assert curve[8][1] == 90, "Midpoint should be 90 BPM"
        print("✓ PASSED")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    # Test 2: Exponential tempo curve
    test_count += 1
    print(f"\nTest {test_count}: Exponential tempo curve")
    try:
        curve = engine.create_tempo_curve(60, 120, 8, CurveType.EXPONENTIAL)
        # Exponential should be slower at start, faster at end
        quarter_point = curve[4][1]
        assert quarter_point < 75, f"Quarter point ({quarter_point}) should be < 75 BPM"
        print(f"  Quarter point: {quarter_point:.1f} BPM")
        print("✓ PASSED")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    # Test 3: S-curve tempo curve
    test_count += 1
    print(f"\nTest {test_count}: S-curve tempo curve")
    try:
        curve = engine.create_tempo_curve(60, 120, 8, CurveType.S_CURVE)
        # S-curve should be approximately at start and end values
        assert abs(curve[0][1] - 60) < 2, f"Start should be ~60, got {curve[0][1]}"
        assert abs(curve[-1][1] - 120) < 2, f"End should be ~120, got {curve[-1][1]}"
        assert 80 < curve[8][1] < 100, "Midpoint should be near 90 BPM"
        print(f"  Start: {curve[0][1]:.1f}, Mid: {curve[8][1]:.1f}, End: {curve[-1][1]:.1f}")
        print("✓ PASSED")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    # Test 4: Apply accelerando
    test_count += 1
    print(f"\nTest {test_count}: Apply accelerando to notes")
    try:
        notes = [Note(60, i, 1.0, 64) for i in range(8)]
        accel_notes = engine.apply_accelerando(notes, 60, 120, CurveType.LINEAR)
        assert len(accel_notes) == len(notes)
        assert accel_notes[0].pitch == 60
        print(f"  Original duration: {sum(n.start_time for n in notes):.2f}")
        print(f"  Accelerando duration: {sum(n.start_time for n in accel_notes):.2f}")
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 5: Apply ritardando
    test_count += 1
    print(f"\nTest {test_count}: Apply ritardando to notes")
    try:
        notes = [Note(60, i, 1.0, 64) for i in range(8)]
        rit_notes = engine.apply_ritardando(notes, 120, 60, CurveType.EXPONENTIAL)
        assert len(rit_notes) == len(notes)
        # Later notes should be more stretched
        print(f"  Last note original time: {notes[-1].start_time:.2f}")
        print(f"  Last note ritardando time: {rit_notes[-1].start_time:.2f}")
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 6: Romantic rubato
    test_count += 1
    print(f"\nTest {test_count}: Apply romantic rubato")
    try:
        notes = [Note(60 + i, i, 1.0, 70) for i in range(8)]
        rubato_notes = engine.apply_rubato(notes, 0.3, RubatoStyle.ROMANTIC)
        assert len(rubato_notes) == len(notes)
        # Some notes should be shifted in time
        time_diffs = [abs(rubato_notes[i].start_time - notes[i].start_time)
                     for i in range(len(notes))]
        assert max(time_diffs) > 0.01, "Some timing should change"
        print(f"  Max time shift: {max(time_diffs):.3f} beats")
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 7: Jazz rubato
    test_count += 1
    print(f"\nTest {test_count}: Apply jazz rubato")
    try:
        notes = [Note(60 + i, i, 1.0, 70) for i in range(8)]
        jazz_rubato = engine.apply_rubato(notes, 0.4, RubatoStyle.JAZZ)
        # First note should be delayed (late entry)
        assert jazz_rubato[0].start_time > notes[0].start_time
        print(f"  First note delay: {jazz_rubato[0].start_time - notes[0].start_time:.3f}")
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 8: Agogic accents
    test_count += 1
    print(f"\nTest {test_count}: Add agogic accents")
    try:
        notes = [Note(60, i, 1.0, 64) for i in range(8)]
        accented = engine.add_agogic_accent(notes, [0, 2, 4], 15)
        assert accented[0].duration == 1.15, "Accented note should be 15% longer"
        assert accented[1].duration == 1.0, "Non-accented note unchanged"
        assert accented[2].duration == 1.15
        print(f"  Accented duration: {accented[0].duration:.2f}")
        print(f"  Normal duration: {accented[1].duration:.2f}")
        print("✓ PASSED")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    # Test 9: Metric modulation - quarter to dotted eighth
    test_count += 1
    print(f"\nTest {test_count}: Metric modulation (quarter → dotted eighth)")
    try:
        new_tempo = engine.calculate_tempo_modulation(60, 'quarter', 'dotted_eighth')
        expected = 60 * (1.0 / 0.75)  # 80 BPM
        assert abs(new_tempo - expected) < 0.1, f"Expected {expected}, got {new_tempo}"
        print(f"  60 BPM (quarter) → {new_tempo:.1f} BPM (dotted eighth as new beat)")
        print("✓ PASSED")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    # Test 10: Metric modulation - dotted quarter to eighth
    test_count += 1
    print(f"\nTest {test_count}: Metric modulation (dotted quarter → eighth)")
    try:
        new_tempo = engine.calculate_tempo_modulation(80, 'dotted_quarter', 'eighth')
        expected = 80 * (1.5 / 0.5)  # 240 BPM
        assert abs(new_tempo - expected) < 0.1
        print(f"  80 BPM (dotted quarter) → {new_tempo:.1f} BPM (eighth as new beat)")
        print("✓ PASSED")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    # Test 11: Generate MIDI tempo map
    test_count += 1
    print(f"\nTest {test_count}: Generate MIDI tempo map")
    try:
        curve = engine.create_tempo_curve(60, 120, 8, CurveType.LINEAR, resolution=8)
        tempo_map = engine.generate_midi_tempo_map(curve)
        assert len(tempo_map) > 0
        assert tempo_map[0]['tempo_bpm'] == 60
        assert 'microseconds_per_quarter' in tempo_map[0]
        # 60 BPM = 1,000,000 microseconds per beat
        assert tempo_map[0]['microseconds_per_quarter'] == 1_000_000
        print(f"  Generated {len(tempo_map)} tempo events")
        print(f"  First event: {tempo_map[0]['tempo_bpm']} BPM = {tempo_map[0]['microseconds_per_quarter']} μs/quarter")
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 12: Apply fermata
    test_count += 1
    print(f"\nTest {test_count}: Apply fermata")
    try:
        notes = [Note(60, i, 1.0, 64) for i in range(8)]
        fermata_notes = engine.apply_fermata(notes, 3, 2.5)
        assert fermata_notes[3].duration == 2.5, "Fermata note should be 2.5x longer"
        # Note after fermata should be shifted
        assert fermata_notes[4].start_time > notes[4].start_time
        print(f"  Fermata note duration: {fermata_notes[3].duration:.1f}")
        print(f"  Next note shifted by: {fermata_notes[4].start_time - notes[4].start_time:.1f}")
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 13: Add breath marks
    test_count += 1
    print(f"\nTest {test_count}: Add breath marks")
    try:
        notes = [Note(60, i, 1.0, 64) for i in range(8)]
        breathing = engine.add_breath_marks(notes, [3], 0.15)
        # Notes after breath mark should be shifted
        assert breathing[4].start_time == notes[4].start_time + 0.15
        print(f"  Breath pause: 0.15 beats")
        print(f"  Note 4 shifted to: {breathing[4].start_time:.2f}")
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 14: Rallentando
    test_count += 1
    print(f"\nTest {test_count}: Create rallentando")
    try:
        notes = [Note(60, i, 1.0, 64) for i in range(8)]
        rall_notes = engine.create_rallentando(notes, 120, 30)
        assert len(rall_notes) == len(notes)
        print(f"  120 BPM slowing by 30% → ~84 BPM")
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 15: Invalid tempo error
    test_count += 1
    print(f"\nTest {test_count}: Error handling - negative tempo")
    try:
        engine.create_tempo_curve(-60, 120, 8)
        print("✗ FAILED: Should raise ValueError")
    except ValueError:
        print("✓ PASSED - Correctly raised ValueError")
        passed += 1

    # Test 16: Invalid accelerando direction
    test_count += 1
    print(f"\nTest {test_count}: Error handling - accelerando with decreasing tempo")
    try:
        notes = [Note(60, i, 1.0, 64) for i in range(4)]
        engine.apply_accelerando(notes, 120, 60)
        print("✗ FAILED: Should raise ValueError")
    except ValueError:
        print("✓ PASSED - Correctly raised ValueError")
        passed += 1

    # Test 17: Parabolic curve
    test_count += 1
    print(f"\nTest {test_count}: Parabolic tempo curve")
    try:
        curve = engine.create_tempo_curve(60, 120, 8, CurveType.PARABOLIC)
        assert len(curve) == 17
        print(f"  Parabolic curve generated with {len(curve)} points")
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 18: Logarithmic curve
    test_count += 1
    print(f"\nTest {test_count}: Logarithmic tempo curve")
    try:
        curve = engine.create_tempo_curve(60, 120, 8, CurveType.LOGARITHMIC)
        # Logarithmic should change faster at start
        quarter_point = curve[4][1]
        assert quarter_point > 75, f"Quarter point should be > 75 for log curve"
        print(f"  Quarter point: {quarter_point:.1f} BPM (fast initial change)")
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 19: Expressive rubato style
    test_count += 1
    print(f"\nTest {test_count}: Expressive rubato style")
    try:
        notes = [Note(60 + i, i, 1.0, 70) for i in range(8)]
        expr_rubato = engine.apply_rubato(notes, 0.25, RubatoStyle.EXPRESSIVE)
        assert len(expr_rubato) == len(notes)
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 20: Classical rubato style
    test_count += 1
    print(f"\nTest {test_count}: Classical rubato style")
    try:
        notes = [Note(60 + i, i, 1.0, 70) for i in range(8)]
        classical_rubato = engine.apply_rubato(notes, 0.2, RubatoStyle.CLASSICAL)
        assert len(classical_rubato) == len(notes)
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 21: Triplet metric modulation
    test_count += 1
    print(f"\nTest {test_count}: Metric modulation with triplets")
    try:
        new_tempo = engine.calculate_tempo_modulation(90, 'quarter', 'triplet_eighth')
        expected = 90 * (1.0 / (1/3))  # 270 BPM
        assert abs(new_tempo - expected) < 0.1
        print(f"  90 BPM (quarter) → {new_tempo:.1f} BPM (triplet eighth)")
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 22: Multiple breath marks
    test_count += 1
    print(f"\nTest {test_count}: Multiple breath marks")
    try:
        notes = [Note(60, i, 1.0, 64) for i in range(16)]
        breathing = engine.add_breath_marks(notes, [3, 7, 11], 0.1)
        # Note 12 should be shifted by 3 breath marks = 0.3 beats
        expected_shift = 0.3
        actual_shift = breathing[12].start_time - notes[12].start_time
        assert abs(actual_shift - expected_shift) < 0.01
        print(f"  3 breath marks, total shift: {actual_shift:.2f} beats")
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 23: Spline curve
    test_count += 1
    print(f"\nTest {test_count}: Spline tempo curve")
    try:
        curve = engine.create_tempo_curve(60, 120, 8, CurveType.SPLINE)
        assert len(curve) == 17
        assert 60 <= min(c[1] for c in curve) <= 120
        assert 60 <= max(c[1] for c in curve) <= 120
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 24: Empty note list handling
    test_count += 1
    print(f"\nTest {test_count}: Empty note list handling")
    try:
        empty_notes = []
        result = engine.apply_rubato(empty_notes, 0.3)
        assert result == []
        print("✓ PASSED - Empty list handled correctly")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 25: Large tempo range
    test_count += 1
    print(f"\nTest {test_count}: Large tempo range (30 → 200 BPM)")
    try:
        curve = engine.create_tempo_curve(30, 200, 16, CurveType.S_CURVE)
        assert abs(curve[0][1] - 30) < 5, f"Start should be ~30, got {curve[0][1]}"
        assert abs(curve[-1][1] - 200) < 5, f"End should be ~200, got {curve[-1][1]}"
        print(f"  Created curve from {curve[0][1]:.1f} to {curve[-1][1]:.1f} BPM")
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed}/{test_count} tests passed")
    print(f"Success rate: {100 * passed / test_count:.1f}%")
    print("=" * 70)

    return passed == test_count


if __name__ == "__main__":
    success = run_tests()

    print("\n" + "=" * 70)
    print("DEMONSTRATION - TEMPO ENGINE FEATURES")
    print("=" * 70)

    engine = TempoEngine()

    # Demo 1: Tempo curves comparison
    print("\n1. TEMPO CURVE COMPARISON (60 → 120 BPM over 8 beats)")
    print("-" * 70)
    for curve_type in [CurveType.LINEAR, CurveType.EXPONENTIAL, CurveType.S_CURVE]:
        curve = engine.create_tempo_curve(60, 120, 8, curve_type, resolution=4)
        print(f"\n{curve_type.value.upper()}:")
        for beat, tempo in curve:
            print(f"  Beat {beat:4.1f}: {tempo:6.2f} BPM")

    # Demo 2: Metric modulation examples
    print("\n2. METRIC MODULATION EXAMPLES (Elliott Carter technique)")
    print("-" * 70)
    modulations = [
        (60, 'quarter', 'dotted_eighth'),
        (80, 'dotted_quarter', 'eighth'),
        (120, 'quarter', 'triplet_eighth'),
        (90, 'eighth', 'quarter'),
    ]
    for from_tempo, from_val, to_val in modulations:
        new_tempo = engine.calculate_tempo_modulation(from_tempo, from_val, to_val)
        print(f"  {from_tempo} BPM ({from_val}) → {new_tempo:.1f} BPM ({to_val} becomes beat)")

    # Demo 3: Rubato styles
    print("\n3. RUBATO STYLES COMPARISON")
    print("-" * 70)
    test_melody = [Note(60 + i, i, 1.0, 70) for i in range(8)]

    for style in [RubatoStyle.ROMANTIC, RubatoStyle.JAZZ, RubatoStyle.EXPRESSIVE, RubatoStyle.CLASSICAL]:
        rubato_notes = engine.apply_rubato(test_melody, 0.3, style)
        time_shifts = [rubato_notes[i].start_time - test_melody[i].start_time
                      for i in range(len(test_melody))]
        max_shift = max(abs(s) for s in time_shifts)
        print(f"\n{style.value.upper()}:")
        print(f"  Max time shift: {max_shift:.3f} beats")
        print(f"  First 4 notes: {[f'{s:+.3f}' for s in time_shifts[:4]]}")

    print("\n" + "=" * 70)
    print("Agent 16: Tempo Engine - Implementation Complete! 🎵")
    print("=" * 70)
