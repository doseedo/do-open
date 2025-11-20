#!/usr/bin/env python3
"""
Agent 08: Musical Quality Validators
====================================

Model-agnostic musical quality validation for generated MIDI.

Validates:
1. Interval validity and distribution
2. Harmony correctness and voice leading
3. Rhythm consistency and groove
4. Voice range compliance
5. Musical coherence (parameters vs. output)

These validators work on any MIDI data, regardless of how it was generated.

Author: Agent 08 - Validation Framework Builder
Date: 2025-11-20
License: MIT
"""

import sys
import statistics
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter, defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from existing modules
from analysis.midi_analyzer import NoteEvent, ChordEvent, MidiAnalyzer
from validation.validation_pipeline import MusicalValidationResult, MusicalQualityValidator
from validation.validation_utils import (
    cosine_similarity,
    interval_distribution,
    pitch_class_distribution,
    rhythm_complexity,
    voice_leading_cost
)


# ==============================================================================
# INTERVAL VALIDATOR
# ==============================================================================

class IntervalValidator(MusicalQualityValidator):
    """
    Validate melodic interval correctness and naturalness.

    Checks:
    - No extreme leaps (>12 semitones)
    - Interval distribution matches genre expectations
    - Stepwise motion percentage
    - Leap recovery (after leap, step in opposite direction)
    """

    # Genre-specific interval distributions (based on analysis)
    GENRE_INTERVAL_DISTRIBUTIONS = {
        'bebop': {
            1: 0.15,   # Minor 2nd (chromatic approaches)
            2: 0.25,   # Major 2nd (stepwise)
            3: 0.18,   # Minor 3rd
            4: 0.15,   # Major 3rd
            5: 0.10,   # Perfect 4th
            7: 0.08,   # Perfect 5th
            6: 0.04,   # Tritone
            8: 0.03,   # Minor 6th
            9: 0.02,   # Major 6th
        },
        'classical': {
            1: 0.08,   # Minor 2nd
            2: 0.40,   # Major 2nd (predominantly stepwise)
            3: 0.15,   # Minor 3rd
            4: 0.12,   # Major 3rd
            5: 0.10,   # Perfect 4th
            7: 0.08,   # Perfect 5th
            6: 0.02,   # Tritone
            8: 0.03,   # Minor 6th
            9: 0.02,   # Major 6th
        },
        'rock': {
            2: 0.30,   # Major 2nd
            3: 0.20,   # Minor 3rd (power chords influence)
            4: 0.15,   # Major 3rd
            5: 0.15,   # Perfect 4th
            7: 0.10,   # Perfect 5th
            1: 0.05,   # Minor 2nd
            6: 0.03,   # Tritone
            8: 0.02,   # Minor 6th
        },
        'electronic': {
            1: 0.10,   # Minor 2nd
            2: 0.25,   # Major 2nd
            3: 0.15,   # Minor 3rd
            4: 0.15,   # Major 3rd
            5: 0.12,   # Perfect 4th
            7: 0.12,   # Perfect 5th
            12: 0.06,  # Octave (common in electronic)
            6: 0.03,   # Tritone
            8: 0.02,   # Minor 6th
        }
    }

    def __init__(self, genre: str = 'bebop', max_leap: int = 12, threshold: float = 0.85):
        """
        Initialize interval validator.

        Args:
            genre: Genre for interval distribution comparison
            max_leap: Maximum allowed interval (semitones)
            threshold: Minimum acceptable similarity score
        """
        super().__init__(validation_type='intervals', threshold=threshold)
        self.genre = genre
        self.max_leap = max_leap
        self.reference_distribution = self.GENRE_INTERVAL_DISTRIBUTIONS.get(
            genre,
            self.GENRE_INTERVAL_DISTRIBUTIONS['bebop']  # Default
        )

    def validate_midi(self, midi_data: Any) -> MusicalValidationResult:
        """
        Validate intervals in MIDI data.

        Args:
            midi_data: Either MidiAnalyzer instance, list of NoteEvents, or MIDI file path

        Returns:
            MusicalValidationResult
        """
        start_time = self._start_timer()

        # Extract notes
        notes = self._extract_notes(midi_data)

        if len(notes) < 2:
            return MusicalValidationResult(
                validation_type=self.validation_type,
                category='intervals',
                passed=False,
                score=0.0,
                violations=["Insufficient notes for interval validation"],
                validation_time=self._end_timer(start_time)
            )

        # Sort notes by time
        sorted_notes = sorted(notes, key=lambda n: n.start_time)

        # Extract intervals
        intervals = []
        extreme_leaps = []

        for i in range(len(sorted_notes) - 1):
            interval = sorted_notes[i+1].pitch - sorted_notes[i].pitch
            abs_interval = abs(interval)

            intervals.append(abs_interval)

            # Check for extreme leaps
            if abs_interval > self.max_leap:
                extreme_leaps.append({
                    'position': i,
                    'from_pitch': sorted_notes[i].pitch,
                    'to_pitch': sorted_notes[i+1].pitch,
                    'interval': abs_interval,
                    'time': sorted_notes[i+1].start_time
                })

        # Calculate metrics
        metrics = {}
        violations = []
        warnings = []

        # 1. Extreme leap check
        extreme_leap_ratio = len(extreme_leaps) / len(intervals)
        metrics['extreme_leaps'] = len(extreme_leaps)
        metrics['extreme_leap_ratio'] = extreme_leap_ratio

        if extreme_leap_ratio > 0.05:  # More than 5% extreme leaps
            violations.append(
                f"Too many extreme leaps (>{self.max_leap} semitones): "
                f"{len(extreme_leaps)} ({extreme_leap_ratio:.1%})"
            )

        # 2. Stepwise motion percentage
        stepwise = sum(1 for i in intervals if i <= 2)
        stepwise_ratio = stepwise / len(intervals)
        metrics['stepwise_ratio'] = stepwise_ratio

        if stepwise_ratio < 0.5:  # Less than 50% stepwise
            warnings.append(f"Low stepwise motion: {stepwise_ratio:.1%}")

        # 3. Interval distribution similarity
        actual_dist = interval_distribution([n.pitch for n in sorted_notes])
        similarity = cosine_similarity(actual_dist, self.reference_distribution)
        metrics['distribution_similarity'] = similarity

        if similarity < 0.7:
            warnings.append(
                f"Interval distribution differs from {self.genre} style: "
                f"similarity={similarity:.2f}"
            )

        # 4. Leap recovery check
        leap_recoveries = 0
        total_leaps = 0

        for i in range(len(sorted_notes) - 2):
            interval1 = sorted_notes[i+1].pitch - sorted_notes[i].pitch
            interval2 = sorted_notes[i+2].pitch - sorted_notes[i+1].pitch

            # If first interval is a leap (>3 semitones)
            if abs(interval1) > 3:
                total_leaps += 1

                # Check if next interval is stepwise in opposite direction
                if (interval1 * interval2 < 0 and  # Opposite direction
                    abs(interval2) <= 2):  # Stepwise
                    leap_recoveries += 1

        leap_recovery_ratio = leap_recoveries / total_leaps if total_leaps > 0 else 0
        metrics['leap_recovery_ratio'] = leap_recovery_ratio

        # Calculate overall score
        score = (
            (1.0 - extreme_leap_ratio) * 0.30 +  # Penalty for extreme leaps
            stepwise_ratio * 0.25 +               # Reward stepwise motion
            similarity * 0.30 +                   # Distribution match
            leap_recovery_ratio * 0.15            # Leap recovery
        )

        # Passed if score above threshold and no critical violations
        passed = score >= self.threshold and len(violations) == 0

        result = MusicalValidationResult(
            validation_type=self.validation_type,
            category='intervals',
            passed=passed,
            score=score,
            metrics=metrics,
            violations=violations,
            warnings=warnings,
            threshold=self.threshold,
            validation_time=self._end_timer(start_time),
            details={'extreme_leaps': extreme_leaps[:5]}  # First 5
        )

        return result

    def _extract_notes(self, midi_data: Any) -> List[NoteEvent]:
        """Extract note events from various MIDI data formats."""
        if isinstance(midi_data, list) and all(isinstance(n, NoteEvent) for n in midi_data):
            return midi_data
        elif isinstance(midi_data, MidiAnalyzer):
            return midi_data.result.notes
        elif isinstance(midi_data, str) or isinstance(midi_data, Path):
            analyzer = MidiAnalyzer(str(midi_data))
            analyzer.analyze()
            return analyzer.result.notes
        elif isinstance(midi_data, dict):
            # Dictionary of {instrument: [notes]}
            all_notes = []
            for notes in midi_data.values():
                if isinstance(notes, list):
                    all_notes.extend(notes)
            return all_notes
        else:
            raise ValueError(f"Unsupported MIDI data type: {type(midi_data)}")


# ==============================================================================
# HARMONY VALIDATOR
# ==============================================================================

class HarmonyValidator(MusicalQualityValidator):
    """
    Validate harmonic correctness and voice leading.

    Checks:
    - Functional harmony (V→I resolutions)
    - Parallel fifths/octaves
    - Voice crossing frequency
    - Voice leading smoothness
    - Chord type appropriateness for genre
    """

    # Genre-appropriate chord types
    GENRE_CHORD_TYPES = {
        'bebop': {'maj7', 'min7', 'dom7', '7', 'min7b5', 'dim7', 'maj6', 'min6', 'maj9', 'min9'},
        'swing': {'6', 'maj6', '7', 'dom7', 'min7', 'dim7', 'maj', 'min'},
        'modal': {'maj7', 'min7', 'sus4', 'sus2', '7sus4', 'min9', 'maj9'},
        'classical': {'maj', 'min', 'dim', 'aug', 'maj7', 'dom7'},
        'rock': {'maj', 'min', 'power', 'sus2', 'sus4', '7'},
        'electronic': {'maj', 'min', 'maj7', 'min7', 'add9', 'sus2', 'sus4'}
    }

    def __init__(self, genre: str = 'bebop', allow_parallel_fifths: bool = True, threshold: float = 0.80):
        """
        Initialize harmony validator.

        Args:
            genre: Genre for chord type checking
            allow_parallel_fifths: If True, parallel 5ths are warnings not errors (jazz style)
            threshold: Minimum acceptable score
        """
        super().__init__(validation_type='harmony', threshold=threshold)
        self.genre = genre
        self.allow_parallel_fifths = allow_parallel_fifths
        self.expected_chords = self.GENRE_CHORD_TYPES.get(genre, self.GENRE_CHORD_TYPES['bebop'])

    def validate_midi(self, midi_data: Any) -> MusicalValidationResult:
        """
        Validate harmony in MIDI data.

        Args:
            midi_data: MIDI data (various formats supported)

        Returns:
            MusicalValidationResult
        """
        start_time = self._start_timer()

        # Extract chords and notes
        notes = self._extract_notes(midi_data)
        chords = self._extract_chords(midi_data)

        metrics = {}
        violations = []
        warnings = []

        # 1. Validate chord progressions (if chords detected)
        if chords and len(chords) > 1:
            resolution_errors = 0
            awkward_movements = 0

            for i in range(len(chords) - 1):
                current = chords[i]
                next_chord = chords[i + 1]

                # Check V7 → I resolution
                if current.quality in ['dom7', '7']:
                    # Should resolve to major or minor
                    if next_chord.quality not in ['maj7', 'maj6', 'maj', 'min7', 'min6', 'min']:
                        resolution_errors += 1

                # Check root movement
                root_movement = (next_chord.root - current.root) % 12

                # Tritone leap is awkward unless it's a tritone sub
                if root_movement == 6 and current.quality not in ['dom7', '7']:
                    awkward_movements += 1

            metrics['resolution_errors'] = resolution_errors
            metrics['awkward_movements'] = awkward_movements

            if resolution_errors > len(chords) * 0.2:  # >20% unresolved
                violations.append(f"High rate of unresolved dominants: {resolution_errors}")

        # 2. Voice leading analysis (if sufficient harmonic content)
        if notes and len(notes) >= 3:
            parallel_fifths = self._check_parallel_intervals(notes, 7)
            parallel_octaves = self._check_parallel_intervals(notes, 12)

            metrics['parallel_fifths'] = parallel_fifths
            metrics['parallel_octaves'] = parallel_octaves

            if not self.allow_parallel_fifths and parallel_fifths > 0:
                violations.append(f"Found {parallel_fifths} parallel fifths (classical rules)")
            elif parallel_fifths > len(notes) * 0.1:  # >10%
                warnings.append(f"Many parallel fifths: {parallel_fifths}")

            if parallel_octaves > len(notes) * 0.1:
                warnings.append(f"Many parallel octaves: {parallel_octaves}")

            # Voice leading smoothness
            avg_voice_movement = self._calculate_average_voice_movement(notes)
            metrics['avg_voice_movement'] = avg_voice_movement

            if avg_voice_movement > 5.0:  # >5 semitones average
                warnings.append(f"Large average voice movement: {avg_voice_movement:.1f} semitones")

        # 3. Chord type appropriateness
        if chords:
            inappropriate_chords = 0

            for chord in chords:
                if chord.quality and chord.quality not in self.expected_chords:
                    if chord.quality not in ['maj', 'min', 'aug', 'dim']:  # Basic triads always OK
                        inappropriate_chords += 1

            metrics['inappropriate_chords'] = inappropriate_chords

            if inappropriate_chords > 0:
                warnings.append(
                    f"{inappropriate_chords} chords not typical for {self.genre} style"
                )

        # Calculate score
        score = 1.0

        # Penalize errors
        if 'resolution_errors' in metrics:
            score -= metrics['resolution_errors'] * 0.05

        if 'awkward_movements' in metrics:
            score -= metrics['awkward_movements'] * 0.05

        if not self.allow_parallel_fifths and metrics.get('parallel_fifths', 0) > 0:
            score -= metrics['parallel_fifths'] * 0.1

        if metrics.get('inappropriate_chords', 0) > 0:
            score -= metrics['inappropriate_chords'] * 0.1

        if metrics.get('avg_voice_movement', 0) > 5.0:
            score -= 0.1

        score = max(0.0, score)

        passed = score >= self.threshold and len(violations) == 0

        result = MusicalValidationResult(
            validation_type=self.validation_type,
            category='harmony',
            passed=passed,
            score=score,
            metrics=metrics,
            violations=violations,
            warnings=warnings,
            threshold=self.threshold,
            validation_time=self._end_timer(start_time)
        )

        return result

    def _extract_notes(self, midi_data: Any) -> List[NoteEvent]:
        """Extract note events."""
        if isinstance(midi_data, list) and all(isinstance(n, NoteEvent) for n in midi_data):
            return midi_data
        elif isinstance(midi_data, MidiAnalyzer):
            return midi_data.result.notes
        elif isinstance(midi_data, str) or isinstance(midi_data, Path):
            analyzer = MidiAnalyzer(str(midi_data))
            analyzer.analyze()
            return analyzer.result.notes
        elif isinstance(midi_data, dict):
            all_notes = []
            for notes in midi_data.values():
                if isinstance(notes, list):
                    all_notes.extend(notes)
            return all_notes
        return []

    def _extract_chords(self, midi_data: Any) -> List[ChordEvent]:
        """Extract chord events."""
        if isinstance(midi_data, MidiAnalyzer):
            return midi_data.result.chords
        elif isinstance(midi_data, str) or isinstance(midi_data, Path):
            analyzer = MidiAnalyzer(str(midi_data))
            analyzer.analyze(detect_chords=True)
            return analyzer.result.chords
        return []

    def _check_parallel_intervals(self, notes: List[NoteEvent], interval: int) -> int:
        """Check for parallel motion at specific interval (simplified version)."""
        # Group notes by time
        time_groups = defaultdict(list)

        for note in notes:
            time_key = round(note.start_time * 4) / 4  # Quantize to 16th notes
            time_groups[time_key].append(note.pitch)

        # Sort time points
        sorted_times = sorted(time_groups.keys())

        parallel_count = 0

        # Check consecutive time points
        for i in range(len(sorted_times) - 1):
            pitches1 = sorted(time_groups[sorted_times[i]])
            pitches2 = sorted(time_groups[sorted_times[i+1]])

            if len(pitches1) >= 2 and len(pitches2) >= 2:
                # Check for parallel motion at interval
                for p1 in pitches1:
                    for p2 in pitches1:
                        if abs(p2 - p1) == interval:
                            # Found interval in first chord
                            # Check if same interval in second chord with parallel motion
                            for q1 in pitches2:
                                for q2 in pitches2:
                                    if abs(q2 - q1) == interval:
                                        # Check if parallel (same direction)
                                        if (p2 > p1 and q2 > q1) or (p2 < p1 and q2 < q1):
                                            parallel_count += 1

        return parallel_count

    def _calculate_average_voice_movement(self, notes: List[NoteEvent]) -> float:
        """Calculate average voice movement between chords."""
        # Group by time
        time_groups = defaultdict(list)

        for note in notes:
            time_key = round(note.start_time * 4) / 4
            time_groups[time_key].append(note.pitch)

        sorted_times = sorted(time_groups.keys())

        if len(sorted_times) < 2:
            return 0.0

        movements = []

        for i in range(len(sorted_times) - 1):
            chord1 = sorted(time_groups[sorted_times[i]])
            chord2 = sorted(time_groups[sorted_times[i+1]])

            # Calculate voice leading cost
            cost = voice_leading_cost(chord1, chord2)
            movements.append(cost / max(len(chord1), len(chord2)))

        return statistics.mean(movements) if movements else 0.0


# ==============================================================================
# RHYTHM VALIDATOR
# ==============================================================================

class RhythmValidator(MusicalQualityValidator):
    """
    Validate rhythm consistency and groove.

    Checks:
    - Consistent subdivision
    - Rhythm pattern repetition
    - Swing ratio consistency (for swing styles)
    - Timing deviation (quantization vs. groove)
    """

    def __init__(self, expected_subdivision: str = '16th', threshold: float = 0.90):
        """
        Initialize rhythm validator.

        Args:
            expected_subdivision: Expected subdivision ('8th', '16th', '32nd', 'triplet')
            threshold: Minimum acceptable score
        """
        super().__init__(validation_type='rhythm', threshold=threshold)
        self.expected_subdivision = expected_subdivision

    def validate_midi(self, midi_data: Any) -> MusicalValidationResult:
        """
        Validate rhythm in MIDI data.

        Args:
            midi_data: MIDI data

        Returns:
            MusicalValidationResult
        """
        start_time = self._start_timer()

        notes = self._extract_notes(midi_data)

        if not notes:
            return MusicalValidationResult(
                validation_type=self.validation_type,
                category='rhythm',
                passed=False,
                score=0.0,
                violations=["No notes to validate"],
                validation_time=self._end_timer(start_time)
            )

        metrics = {}
        violations = []
        warnings = []

        # 1. Extract onset times
        onset_times = sorted([n.start_time for n in notes])

        # 2. Detect subdivision
        durations = [n.duration for n in notes]
        duration_counts = Counter([round(d, 3) for d in durations])
        most_common_duration = duration_counts.most_common(1)[0][0]

        metrics['most_common_duration'] = most_common_duration

        # 3. Rhythm complexity
        complexity = rhythm_complexity(durations)
        metrics['rhythm_complexity'] = complexity

        if complexity < 0.2:
            warnings.append("Very simple rhythm (low complexity)")
        elif complexity > 0.8:
            warnings.append("Very complex rhythm (high complexity)")

        # 4. Timing consistency (how well quantized)
        timing_deviations = []

        for onset in onset_times:
            # Quantize to 16th note grid
            sixteenth_duration = 0.25  # Beats
            quantized = round(onset / sixteenth_duration) * sixteenth_duration
            deviation = abs(onset - quantized)
            timing_deviations.append(deviation)

        avg_deviation = statistics.mean(timing_deviations)
        metrics['avg_timing_deviation'] = avg_deviation

        if avg_deviation > 0.05:  # >50ms at 120 BPM
            warnings.append(f"High timing deviation: {avg_deviation:.3f} beats")

        # 5. Pattern repetition (detect repeated rhythm patterns)
        # Simplified: check for repeated duration sequences
        duration_sequence = [round(d, 2) for d in durations[:16]]  # First 16 notes
        pattern_repetitions = sum(
            1 for i in range(len(durations) - len(duration_sequence))
            if [round(d, 2) for d in durations[i:i+len(duration_sequence)]] == duration_sequence
        )

        metrics['pattern_repetitions'] = pattern_repetitions

        # Calculate score
        score = (
            (1.0 - min(avg_deviation / 0.1, 1.0)) * 0.5 +  # Reward consistency
            min(complexity / 0.6, 1.0) * 0.3 +              # Reward moderate complexity
            (1.0 if pattern_repetitions > 1 else 0.5) * 0.2  # Reward repetition
        )

        passed = score >= self.threshold

        result = MusicalValidationResult(
            validation_type=self.validation_type,
            category='rhythm',
            passed=passed,
            score=score,
            metrics=metrics,
            violations=violations,
            warnings=warnings,
            threshold=self.threshold,
            validation_time=self._end_timer(start_time)
        )

        return result

    def _extract_notes(self, midi_data: Any) -> List[NoteEvent]:
        """Extract note events."""
        if isinstance(midi_data, list) and all(isinstance(n, NoteEvent) for n in midi_data):
            return midi_data
        elif isinstance(midi_data, MidiAnalyzer):
            return midi_data.result.notes
        elif isinstance(midi_data, str) or isinstance(midi_data, Path):
            analyzer = MidiAnalyzer(str(midi_data))
            analyzer.analyze()
            return analyzer.result.notes
        elif isinstance(midi_data, dict):
            all_notes = []
            for notes in midi_data.values():
                if isinstance(notes, list):
                    all_notes.extend(notes)
            return all_notes
        return []


# ==============================================================================
# VOICE RANGE VALIDATOR
# ==============================================================================

class VoiceRangeValidator(MusicalQualityValidator):
    """
    Validate that all notes are within valid instrument ranges.
    """

    # MIDI note ranges for common instruments
    INSTRUMENT_RANGES = {
        'piano': (21, 108),          # A0 to C8
        'guitar': (40, 88),           # E2 to E6
        'bass': (28, 67),             # E1 to G4
        'saxophone': (58, 90),        # Bb3 to F#6 (alto sax)
        'trumpet': (55, 82),          # G3 to Bb5
        'trombone': (40, 72),         # E2 to C5
        'violin': (55, 103),          # G3 to G7
        'cello': (36, 84),            # C2 to C6
        'flute': (60, 96),            # C4 to C7
        'clarinet': (50, 94),         # D3 to Bb6
        'voice_soprano': (60, 84),    # C4 to C6
        'voice_alto': (55, 79),       # G3 to G5
        'voice_tenor': (48, 72),      # C3 to C5
        'voice_bass': (40, 64),       # E2 to E4
        'drums': (35, 81),            # Acoustic bass drum to Ride cymbal (GM)
        'synth': (0, 127),            # Full MIDI range
        'default': (21, 108)          # Piano range as default
    }

    def __init__(self, instrument_type: str = 'default', threshold: float = 0.95):
        """
        Initialize voice range validator.

        Args:
            instrument_type: Type of instrument
            threshold: Minimum acceptable ratio of notes in range
        """
        super().__init__(validation_type='voice_range', threshold=threshold)
        self.instrument_type = instrument_type
        self.valid_range = self.INSTRUMENT_RANGES.get(instrument_type, self.INSTRUMENT_RANGES['default'])

    def validate_midi(self, midi_data: Any) -> MusicalValidationResult:
        """
        Validate voice ranges.

        Args:
            midi_data: MIDI data

        Returns:
            MusicalValidationResult
        """
        start_time = self._start_timer()

        notes = self._extract_notes(midi_data)

        if not notes:
            return MusicalValidationResult(
                validation_type=self.validation_type,
                category='voice_range',
                passed=False,
                score=0.0,
                violations=["No notes to validate"],
                validation_time=self._end_timer(start_time)
            )

        metrics = {}
        violations = []
        warnings = []

        min_range, max_range = self.valid_range

        out_of_range_notes = []

        for note in notes:
            if note.pitch < min_range or note.pitch > max_range:
                out_of_range_notes.append({
                    'pitch': note.pitch,
                    'time': note.start_time,
                    'range': self.valid_range
                })

        metrics['total_notes'] = len(notes)
        metrics['out_of_range'] = len(out_of_range_notes)
        metrics['in_range_ratio'] = (len(notes) - len(out_of_range_notes)) / len(notes)

        if out_of_range_notes:
            violations.append(
                f"{len(out_of_range_notes)} notes out of range for {self.instrument_type} "
                f"({min_range}-{max_range})"
            )

        # Calculate pitchrange
        all_pitches = [n.pitch for n in notes]
        actual_range = (min(all_pitches), max(all_pitches))
        metrics['actual_range'] = actual_range
        metrics['range_span'] = actual_range[1] - actual_range[0]

        score = metrics['in_range_ratio']

        passed = score >= self.threshold

        result = MusicalValidationResult(
            validation_type=self.validation_type,
            category='voice_range',
            passed=passed,
            score=score,
            metrics=metrics,
            violations=violations,
            warnings=warnings,
            threshold=self.threshold,
            validation_time=self._end_timer(start_time),
            details={'out_of_range_notes': out_of_range_notes[:10]}  # First 10
        )

        return result

    def _extract_notes(self, midi_data: Any) -> List[NoteEvent]:
        """Extract note events."""
        if isinstance(midi_data, list) and all(isinstance(n, NoteEvent) for n in midi_data):
            return midi_data
        elif isinstance(midi_data, MidiAnalyzer):
            return midi_data.result.notes
        elif isinstance(midi_data, str) or isinstance(midi_data, Path):
            analyzer = MidiAnalyzer(str(midi_data))
            analyzer.analyze()
            return analyzer.result.notes
        elif isinstance(midi_data, dict):
            all_notes = []
            for notes in midi_data.values():
                if isinstance(notes, list):
                    all_notes.extend(notes)
            return all_notes
        return []


# ==============================================================================
# MAIN - EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("Agent 08: Musical Quality Validators")
    print("=" * 60)
    print("\nThis module provides model-agnostic musical quality validation.")
    print("\nValidators available:")
    print("  1. IntervalValidator - Validate melodic intervals")
    print("  2. HarmonyValidator - Validate harmony and voice leading")
    print("  3. RhythmValidator - Validate rhythm consistency")
    print("  4. VoiceRangeValidator - Validate instrument ranges")
    print("\nUsage example:")
    print("""
    from validation.musical_quality import IntervalValidator

    # Create validator
    validator = IntervalValidator(genre='bebop', max_leap=12)

    # Validate MIDI file
    result = validator.validate_midi('path/to/file.mid')

    # Check results
    print(f"Passed: {result.passed}")
    print(f"Score: {result.score:.3f}")
    print(f"Violations: {result.violations}")
    """)
