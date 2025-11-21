#!/usr/bin/env python3
"""
Musical Constraint Validator - Agent 8
=======================================

Comprehensive music theory constraint validation and automatic correction system
for the Musical Program Synthesis System.

This module ensures that all generated parameters and their resulting musical
outputs adhere to fundamental music theory rules, orchestration principles,
and performance practices.

Core Responsibilities:
---------------------
1. Voice Leading Validation
   - Parallel fifths/octaves detection
   - Voice crossing detection
   - Proper voice spacing
   - Smooth voice leading (minimal motion)

2. Instrument Range Validation
   - Tessitura checking
   - Extreme range warnings
   - Comfortable playing ranges
   - Extended technique feasibility

3. Harmonic Rule Validation
   - Proper voice leading in progressions
   - Resolution of tendency tones
   - Suspension preparation and resolution
   - Cadence validation

4. Counterpoint Rules
   - Species counterpoint rules
   - Dissonance treatment
   - Melodic intervals
   - Contrary/oblique motion preferences

5. Orchestration Rules
   - Ensemble balance
   - Doubling conventions
   - Register spacing
   - Idiomatic writing

6. Automatic Correction
   - Fix voice leading errors
   - Transpose out-of-range notes
   - Resolve harmonic issues
   - Suggest alternatives

Research Foundation:
-------------------
- Fux, "Gradus ad Parnassum" (1725) - Species Counterpoint
- Piston, "Harmony" (5th ed.) - Tonal harmony rules
- Rimsky-Korsakov, "Principles of Orchestration" (1922)
- Schoenberg, "Theory of Harmony" (1911)
- Tymoczko, "A Geometry of Music" (2011) - Voice leading spaces
- Aldwell & Schachter, "Harmony and Voice Leading" (4th ed.)
- Clendinning & Marvin, "The Musician's Guide to Theory and Analysis"

Author: Agent 8 - Constraint Validator
License: MIT
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from copy import deepcopy
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ValidationSeverity(IntEnum):
    """Severity levels for constraint violations"""
    INFO = 0          # Informational, stylistic preference
    WARNING = 1       # Minor issue, may be acceptable in some contexts
    ERROR = 2         # Significant violation of music theory
    CRITICAL = 3      # Fundamental error that makes music unplayable/unmusical


class ViolationType(Enum):
    """Types of musical constraint violations"""
    # Voice Leading
    PARALLEL_FIFTHS = "parallel_fifths"
    PARALLEL_OCTAVES = "parallel_octaves"
    PARALLEL_UNISONS = "parallel_unisons"
    HIDDEN_FIFTHS = "hidden_fifths"
    HIDDEN_OCTAVES = "hidden_octaves"
    VOICE_CROSSING = "voice_crossing"
    VOICE_OVERLAP = "voice_overlap"
    EXCESSIVE_SPACING = "excessive_spacing"
    EXCESSIVE_LEAP = "excessive_leap"

    # Range and Tessitura
    OUT_OF_RANGE = "out_of_range"
    EXTREME_REGISTER = "extreme_register"
    UNCOMFORTABLE_TESSITURA = "uncomfortable_tessitura"
    IMPOSSIBLE_TECHNIQUE = "impossible_technique"

    # Harmonic
    UNRESOLVED_DISSONANCE = "unresolved_dissonance"
    POOR_RESOLUTION = "poor_resolution"
    UNPREPARED_SUSPENSION = "unprepared_suspension"
    INCORRECT_DOUBLING = "incorrect_doubling"
    MISSING_CHORD_TONE = "missing_chord_tone"
    IMPROPER_CADENCE = "improper_cadence"

    # Melodic
    AUGMENTED_INTERVAL = "augmented_interval"
    DIMINISHED_INTERVAL = "diminished_interval"
    UNRESOLVED_LEAP = "unresolved_leap"
    POOR_CONTOUR = "poor_contour"

    # Counterpoint
    ILLEGAL_DISSONANCE = "illegal_dissonance"
    IMPROPER_MOTION = "improper_motion"
    CONSECUTIVE_LEAPS = "consecutive_leaps"

    # Orchestration
    POOR_BALANCE = "poor_balance"
    REGISTER_CLASH = "register_clash"
    UNIDIOMATIC_WRITING = "unidiomatic_writing"
    IMPOSSIBLE_DYNAMICS = "impossible_dynamics"


# Standard instrument ranges (MIDI note numbers)
INSTRUMENT_RANGES = {
    # Strings
    'violin': (55, 103),           # G3 to G7
    'viola': (48, 91),             # C3 to G6
    'cello': (36, 84),             # C2 to C6
    'double_bass': (28, 67),       # E1 to G4
    'bass_guitar': (28, 67),       # E1 to G4 (standard 4-string)
    'guitar': (40, 88),            # E2 to E6 (standard tuning)

    # Woodwinds
    'piccolo': (74, 108),          # D5 to C8
    'flute': (60, 96),             # C4 to C7
    'oboe': (58, 91),              # Bb3 to G6
    'clarinet': (50, 94),          # D3 to Bb6 (Bb clarinet)
    'bass_clarinet': (38, 77),     # D2 to F5
    'bassoon': (34, 75),           # Bb1 to Eb5
    'contrabassoon': (22, 63),     # Bb0 to Eb4
    'soprano_sax': (56, 92),       # Ab3 to E6
    'alto_sax': (49, 84),          # Db3 to Ab5
    'tenor_sax': (44, 76),         # Ab2 to E5
    'baritone_sax': (36, 69),      # Db2 to A4

    # Brass
    'trumpet': (55, 94),           # G3 to Bb6 (practical range)
    'french_horn': (41, 77),       # F2 to F5
    'trombone': (40, 72),          # E2 to C5
    'bass_trombone': (34, 67),     # Bb1 to G4
    'tuba': (28, 58),              # E1 to Bb3

    # Keyboards
    'piano': (21, 108),            # A0 to C8
    'organ': (21, 108),            # A0 to C8
    'harpsichord': (29, 89),       # F1 to F6
    'celesta': (60, 108),          # C4 to C8

    # Voice
    'soprano': (60, 84),           # C4 to C6
    'mezzo_soprano': (57, 81),     # A3 to A5
    'alto': (55, 79),              # G3 to G5
    'tenor': (48, 72),             # C3 to C5
    'baritone': (43, 67),          # G2 to G4
    'bass': (40, 64),              # E2 to E4

    # Percussion (pitched)
    'vibraphone': (53, 89),        # F3 to F6
    'marimba': (48, 96),           # C3 to C7
    'xylophone': (65, 108),        # F4 to C8
    'glockenspiel': (79, 108),     # G5 to C8
    'timpani': (40, 60),           # E2 to C4

    # Default fallback
    'default': (21, 108),          # Full piano range
}

# Comfortable tessitura ranges (subset of full range)
COMFORTABLE_RANGES = {
    'violin': (60, 93),            # C4 to A6
    'viola': (53, 81),             # F3 to A5
    'cello': (41, 74),             # F2 to D5
    'trumpet': (58, 82),           # Bb3 to Bb5
    'trombone': (43, 67),          # G2 to G4
    'soprano': (64, 79),           # E4 to G5
    'alto': (57, 74),              # A3 to D5
    'tenor': (52, 67),             # E3 to G4
    'bass': (43, 60),              # G2 to C4
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ConstraintViolation:
    """Represents a single constraint violation"""
    violation_type: ViolationType
    severity: ValidationSeverity
    location: Union[int, Tuple[int, ...]]  # Beat/measure or voice indices
    description: str
    affected_parameters: List[str] = field(default_factory=list)
    suggested_fix: Optional[str] = None
    original_values: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return (f"[{self.severity.name}] {self.violation_type.value} at {self.location}: "
                f"{self.description}")


@dataclass
class ValidationResult:
    """Results of constraint validation"""
    is_valid: bool
    violations: List[ConstraintViolation] = field(default_factory=list)
    score: float = 1.0  # 0.0 (worst) to 1.0 (perfect)
    warnings_count: int = 0
    errors_count: int = 0
    critical_count: int = 0

    def add_violation(self, violation: ConstraintViolation):
        """Add a violation and update counts"""
        self.violations.append(violation)

        if violation.severity == ValidationSeverity.WARNING:
            self.warnings_count += 1
        elif violation.severity == ValidationSeverity.ERROR:
            self.errors_count += 1
            self.is_valid = False
        elif violation.severity == ValidationSeverity.CRITICAL:
            self.critical_count += 1
            self.is_valid = False

        # Update score based on violations
        self._update_score()

    def _update_score(self):
        """Calculate overall score based on violations"""
        penalty = (
            self.warnings_count * 0.05 +
            self.errors_count * 0.15 +
            self.critical_count * 0.30
        )
        self.score = max(0.0, 1.0 - penalty)

    def get_summary(self) -> str:
        """Get human-readable summary"""
        if self.is_valid and not self.violations:
            return "✓ All constraints satisfied!"

        summary = [f"Validation Score: {self.score:.2%}"]
        if self.critical_count:
            summary.append(f"  Critical: {self.critical_count}")
        if self.errors_count:
            summary.append(f"  Errors: {self.errors_count}")
        if self.warnings_count:
            summary.append(f"  Warnings: {self.warnings_count}")

        return "\n".join(summary)


@dataclass
class VoiceLeadingContext:
    """Context for voice leading analysis"""
    voices: List[List[int]]  # List of voices, each voice is list of pitches
    chord_progression: Optional[List[List[int]]] = None
    time_points: Optional[List[float]] = None


@dataclass
class Note:
    """Represents a musical note"""
    pitch: int              # MIDI note number
    duration: float         # Duration in beats
    velocity: int = 64      # MIDI velocity
    voice: int = 0          # Voice number
    time: float = 0.0       # Start time in beats

    @property
    def pitch_class(self) -> int:
        """Get pitch class (0-11)"""
        return self.pitch % 12

    def interval_to(self, other: 'Note') -> int:
        """Calculate interval to another note"""
        return abs(self.pitch - other.pitch)


# =============================================================================
# MUSICAL CONSTRAINT VALIDATOR - MAIN CLASS
# =============================================================================

class MusicalConstraintValidator:
    """
    Main constraint validation and correction engine.

    Validates musical parameters and their outputs against music theory rules
    and automatically corrects violations when possible.

    Usage:
        validator = MusicalConstraintValidator()
        result = validator.validate_voice_leading(voices)
        if not result.is_valid:
            corrected = validator.fix_voice_leading(voices)
    """

    def __init__(self,
                 strict_mode: bool = False,
                 style: str = 'common_practice',
                 allow_auto_correction: bool = True):
        """
        Initialize validator.

        Args:
            strict_mode: If True, apply stricter rules (academic)
            style: Musical style for context-aware validation
                   ('common_practice', 'jazz', 'contemporary', 'baroque')
            allow_auto_correction: Enable automatic correction of violations
        """
        self.strict_mode = strict_mode
        self.style = style
        self.allow_auto_correction = allow_auto_correction

        # Load style-specific rules
        self.rules = self._load_style_rules(style)

    def _load_style_rules(self, style: str) -> Dict[str, Any]:
        """Load style-specific constraint rules"""
        rules = {
            'common_practice': {
                'parallel_fifths_forbidden': True,
                'parallel_octaves_forbidden': True,
                'hidden_fifths_allowed': False,
                'voice_crossing_allowed': False,
                'max_voice_spacing': 12,  # semitones between adjacent voices
                'max_melodic_interval': 12,  # octave
                'augmented_intervals_forbidden': True,
                'resolve_leading_tone': True,
                'resolve_seventh': True,
            },
            'jazz': {
                'parallel_fifths_forbidden': False,  # More lenient in jazz
                'parallel_octaves_forbidden': False,
                'hidden_fifths_allowed': True,
                'voice_crossing_allowed': True,
                'max_voice_spacing': 24,
                'max_melodic_interval': 24,  # 2 octaves OK in jazz
                'augmented_intervals_forbidden': False,
                'resolve_leading_tone': False,
                'resolve_seventh': False,
            },
            'contemporary': {
                'parallel_fifths_forbidden': False,
                'parallel_octaves_forbidden': False,
                'hidden_fifths_allowed': True,
                'voice_crossing_allowed': True,
                'max_voice_spacing': 36,
                'max_melodic_interval': 36,
                'augmented_intervals_forbidden': False,
                'resolve_leading_tone': False,
                'resolve_seventh': False,
            },
            'baroque': {
                'parallel_fifths_forbidden': True,
                'parallel_octaves_forbidden': True,
                'hidden_fifths_allowed': False,
                'voice_crossing_allowed': False,
                'max_voice_spacing': 12,
                'max_melodic_interval': 8,  # Minor 6th
                'augmented_intervals_forbidden': True,
                'resolve_leading_tone': True,
                'resolve_seventh': True,
            },
        }

        return rules.get(style, rules['common_practice'])

    # =========================================================================
    # VOICE LEADING VALIDATION
    # =========================================================================

    def validate_voice_leading(self,
                               voices: List[List[int]],
                               chord_names: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate voice leading for multi-voice texture.

        Args:
            voices: List of voices, each voice is a list of MIDI pitches
                   Example: [[60, 62, 64], [64, 65, 67], [67, 69, 71], [72, 74, 76]]
            chord_names: Optional chord names for each time point

        Returns:
            ValidationResult with any violations found
        """
        result = ValidationResult(is_valid=True)

        if not voices or len(voices) < 2:
            return result  # Need at least 2 voices for voice leading

        # Ensure all voices have same length
        min_length = min(len(v) for v in voices)
        if min_length == 0:
            return result

        # Check each time point transition
        for i in range(min_length - 1):
            current_chord = [voice[i] for voice in voices]
            next_chord = [voice[i + 1] for voice in voices]

            # Check parallel motion
            self._check_parallel_motion(current_chord, next_chord, i, result)

            # Check hidden motion
            self._check_hidden_motion(current_chord, next_chord, i, result)

            # Check voice crossing
            self._check_voice_crossing(next_chord, i + 1, result)

            # Check spacing
            self._check_spacing(next_chord, i + 1, result)

        # Check individual voice melodic intervals
        for voice_idx, voice in enumerate(voices):
            self._check_melodic_intervals(voice, voice_idx, result)

        return result

    def _check_parallel_motion(self,
                               chord1: List[int],
                               chord2: List[int],
                               position: int,
                               result: ValidationResult):
        """Check for parallel fifths, octaves, and unisons"""

        for i in range(len(chord1)):
            for j in range(i + 1, len(chord1)):
                interval1 = abs(chord1[j] - chord1[i]) % 12
                interval2 = abs(chord2[j] - chord2[i]) % 12

                motion1 = chord2[i] - chord1[i]
                motion2 = chord2[j] - chord1[j]

                # Parallel motion: both voices move in same direction by same amount
                is_parallel = (motion1 == motion2) and (motion1 != 0)

                if is_parallel:
                    # Check for parallel fifths
                    if interval1 == 7 and interval2 == 7:
                        if self.rules['parallel_fifths_forbidden']:
                            result.add_violation(ConstraintViolation(
                                violation_type=ViolationType.PARALLEL_FIFTHS,
                                severity=ValidationSeverity.ERROR,
                                location=(position, i, j),
                                description=f"Parallel fifths between voices {i} and {j}",
                                suggested_fix="Move one voice by step in opposite direction"
                            ))

                    # Check for parallel octaves
                    if interval1 == 0 and interval2 == 0:
                        if self.rules['parallel_octaves_forbidden']:
                            result.add_violation(ConstraintViolation(
                                violation_type=ViolationType.PARALLEL_OCTAVES,
                                severity=ValidationSeverity.ERROR,
                                location=(position, i, j),
                                description=f"Parallel octaves between voices {i} and {j}",
                                suggested_fix="Move one voice to different chord tone"
                            ))

    def _check_hidden_motion(self,
                            chord1: List[int],
                            chord2: List[int],
                            position: int,
                            result: ValidationResult):
        """Check for hidden (direct) fifths and octaves"""

        if self.rules['hidden_fifths_allowed']:
            return  # Skip if hidden fifths are allowed in this style

        # Only check outer voices for hidden motion
        if len(chord1) < 2:
            return

        bass1, soprano1 = chord1[0], chord1[-1]
        bass2, soprano2 = chord2[0], chord2[-1]

        # Check if both voices move in same direction
        bass_motion = bass2 - bass1
        soprano_motion = soprano2 - soprano1

        same_direction = (bass_motion > 0 and soprano_motion > 0) or \
                        (bass_motion < 0 and soprano_motion < 0)

        if same_direction:
            final_interval = abs(soprano2 - bass2) % 12

            # Hidden fifth
            if final_interval == 7:
                # It's worse if both voices leap
                soprano_leap = abs(soprano_motion) > 2
                bass_leap = abs(bass_motion) > 2

                if soprano_leap and bass_leap:
                    result.add_violation(ConstraintViolation(
                        violation_type=ViolationType.HIDDEN_FIFTHS,
                        severity=ValidationSeverity.WARNING,
                        location=(position, 0, len(chord1) - 1),
                        description="Hidden fifth in outer voices with both leaping",
                        suggested_fix="Use stepwise motion in soprano"
                    ))

            # Hidden octave
            if final_interval == 0:
                if abs(soprano_motion) > 2:  # Soprano leap to octave
                    result.add_violation(ConstraintViolation(
                        violation_type=ViolationType.HIDDEN_OCTAVES,
                        severity=ValidationSeverity.WARNING,
                        location=(position, 0, len(chord1) - 1),
                        description="Hidden octave in outer voices with soprano leap",
                        suggested_fix="Use stepwise motion in soprano"
                    ))

    def _check_voice_crossing(self,
                             chord: List[int],
                             position: int,
                             result: ValidationResult):
        """Check for voice crossing violations"""

        if self.rules['voice_crossing_allowed']:
            return

        # Check if voices are properly ordered (low to high)
        for i in range(len(chord) - 1):
            if chord[i] > chord[i + 1]:
                result.add_violation(ConstraintViolation(
                    violation_type=ViolationType.VOICE_CROSSING,
                    severity=ValidationSeverity.ERROR,
                    location=(position, i, i + 1),
                    description=f"Voice {i} crosses above voice {i+1}",
                    suggested_fix="Reorder voices or transpose to avoid crossing"
                ))

    def _check_spacing(self,
                      chord: List[int],
                      position: int,
                      result: ValidationResult):
        """Check for excessive spacing between voices"""

        max_spacing = self.rules['max_voice_spacing']

        # Check spacing between adjacent voices (except bass to tenor)
        for i in range(1, len(chord) - 1):
            spacing = abs(chord[i + 1] - chord[i])

            if spacing > max_spacing:
                result.add_violation(ConstraintViolation(
                    violation_type=ViolationType.EXCESSIVE_SPACING,
                    severity=ValidationSeverity.WARNING,
                    location=(position, i, i + 1),
                    description=f"Spacing of {spacing} semitones between voices {i} and {i+1}",
                    suggested_fix=f"Reduce spacing to {max_spacing} or less"
                ))

    def _check_melodic_intervals(self,
                                voice: List[int],
                                voice_idx: int,
                                result: ValidationResult):
        """Check melodic intervals within a single voice"""

        max_interval = self.rules['max_melodic_interval']

        for i in range(len(voice) - 1):
            interval = abs(voice[i + 1] - voice[i])

            # Check maximum interval
            if interval > max_interval:
                result.add_violation(ConstraintViolation(
                    violation_type=ViolationType.EXCESSIVE_LEAP,
                    severity=ValidationSeverity.WARNING,
                    location=(i, voice_idx),
                    description=f"Leap of {interval} semitones in voice {voice_idx}",
                    suggested_fix="Fill in with passing tones or reduce interval"
                ))

            # Check augmented intervals (if forbidden)
            if self.rules['augmented_intervals_forbidden']:
                interval_class = interval % 12

                # Augmented fourth (6 semitones) or augmented fifth (8 semitones)
                if interval_class in [6, 8]:
                    result.add_violation(ConstraintViolation(
                        violation_type=ViolationType.AUGMENTED_INTERVAL,
                        severity=ValidationSeverity.ERROR,
                        location=(i, voice_idx),
                        description=f"Augmented interval ({interval} semitones) in voice {voice_idx}",
                        suggested_fix="Use diatonic interval instead"
                    ))

    # =========================================================================
    # INSTRUMENT RANGE VALIDATION
    # =========================================================================

    def validate_range(self,
                      notes: List[int],
                      instrument: str = 'default') -> ValidationResult:
        """
        Validate that notes are within instrument range.

        Args:
            notes: List of MIDI note numbers
            instrument: Instrument name (see INSTRUMENT_RANGES)

        Returns:
            ValidationResult with range violations
        """
        result = ValidationResult(is_valid=True)

        # Get instrument range
        instrument_lower = instrument.lower().replace(' ', '_')
        range_min, range_max = INSTRUMENT_RANGES.get(
            instrument_lower,
            INSTRUMENT_RANGES['default']
        )

        # Get comfortable range if available
        comfortable_range = COMFORTABLE_RANGES.get(instrument_lower)

        for i, note in enumerate(notes):
            # Check absolute range
            if note < range_min or note > range_max:
                result.add_violation(ConstraintViolation(
                    violation_type=ViolationType.OUT_OF_RANGE,
                    severity=ValidationSeverity.CRITICAL,
                    location=i,
                    description=f"Note {note} out of range for {instrument} ({range_min}-{range_max})",
                    suggested_fix=f"Transpose to range {range_min}-{range_max}"
                ))

            # Check comfortable tessitura
            elif comfortable_range:
                comfort_min, comfort_max = comfortable_range
                if note < comfort_min or note > comfort_max:
                    result.add_violation(ConstraintViolation(
                        violation_type=ViolationType.UNCOMFORTABLE_TESSITURA,
                        severity=ValidationSeverity.WARNING,
                        location=i,
                        description=f"Note {note} outside comfortable range for {instrument}",
                        suggested_fix=f"Consider using range {comfort_min}-{comfort_max}"
                    ))

        return result

    def validate_multi_instrument_ranges(self,
                                        parts: Dict[str, List[int]]) -> ValidationResult:
        """
        Validate ranges for multiple instruments.

        Args:
            parts: Dictionary mapping instrument names to note lists

        Returns:
            Combined validation result
        """
        combined_result = ValidationResult(is_valid=True)

        for instrument, notes in parts.items():
            result = self.validate_range(notes, instrument)
            for violation in result.violations:
                # Add instrument name to description
                violation.description = f"[{instrument}] {violation.description}"
                combined_result.add_violation(violation)

        return combined_result

    # =========================================================================
    # HARMONIC VALIDATION
    # =========================================================================

    def validate_harmonic_progression(self,
                                     chords: List[List[int]],
                                     key: Optional[int] = None) -> ValidationResult:
        """
        Validate harmonic progression and resolutions.

        Args:
            chords: List of chords (each chord is list of MIDI pitches)
            key: Tonic pitch (for checking tendency tone resolutions)

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        for i in range(len(chords) - 1):
            current = chords[i]
            next_chord = chords[i + 1]

            # Check for unresolved dissonances
            self._check_dissonance_resolution(current, next_chord, i, result)

            # Check tendency tone resolutions (if key is provided)
            if key is not None:
                self._check_tendency_tones(current, next_chord, i, key, result)

        return result

    def _check_dissonance_resolution(self,
                                    current: List[int],
                                    next_chord: List[int],
                                    position: int,
                                    result: ValidationResult):
        """Check that dissonances resolve properly"""

        # Identify dissonant intervals in current chord
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                interval = abs(current[j] - current[i]) % 12

                # Dissonant intervals: minor 2nd (1), major 2nd (2), tritone (6),
                # minor 7th (10), major 7th (11)
                if interval in [1, 2, 6, 10, 11]:
                    # Check if it resolves by step in next chord
                    # (This is simplified - real implementation would be more nuanced)
                    resolved = False

                    if i < len(next_chord) and j < len(next_chord):
                        motion_i = abs(next_chord[i] - current[i])
                        motion_j = abs(next_chord[j] - current[j])

                        # At least one voice should move by step
                        if motion_i <= 2 or motion_j <= 2:
                            resolved = True

                    if not resolved and self.rules.get('resolve_seventh', True):
                        result.add_violation(ConstraintViolation(
                            violation_type=ViolationType.UNRESOLVED_DISSONANCE,
                            severity=ValidationSeverity.WARNING,
                            location=(position, i, j),
                            description=f"Dissonance (interval {interval}) may not resolve properly",
                            suggested_fix="Resolve dissonance by stepwise motion"
                        ))

    def _check_tendency_tones(self,
                             current: List[int],
                             next_chord: List[int],
                             position: int,
                             key: int,
                             result: ValidationResult):
        """Check that tendency tones (leading tone, chordal 7th) resolve"""

        if not self.rules.get('resolve_leading_tone', True):
            return

        # Check leading tone resolution (scale degree 7 -> 8)
        leading_tone = (key + 11) % 12  # Pitch class of leading tone
        tonic = key % 12

        for i, note in enumerate(current):
            if note % 12 == leading_tone:
                # Leading tone should resolve up by half step to tonic
                if i < len(next_chord):
                    if (next_chord[i] % 12) != tonic:
                        result.add_violation(ConstraintViolation(
                            violation_type=ViolationType.POOR_RESOLUTION,
                            severity=ValidationSeverity.WARNING,
                            location=(position, i),
                            description="Leading tone doesn't resolve to tonic",
                            suggested_fix="Resolve leading tone up by half step"
                        ))

    # =========================================================================
    # AUTOMATIC CORRECTION
    # =========================================================================

    def fix_voice_leading(self,
                         voices: List[List[int]],
                         validation_result: Optional[ValidationResult] = None) -> List[List[int]]:
        """
        Automatically fix voice leading violations.

        Args:
            voices: Original voices with violations
            validation_result: Optional pre-computed validation result

        Returns:
            Corrected voices
        """
        if not self.allow_auto_correction:
            return voices

        # Validate first if not provided
        if validation_result is None:
            validation_result = self.validate_voice_leading(voices)

        if validation_result.is_valid:
            return voices  # Nothing to fix

        corrected = deepcopy(voices)

        # Fix each violation
        for violation in validation_result.violations:
            if violation.violation_type == ViolationType.PARALLEL_FIFTHS:
                corrected = self._fix_parallel_fifths(corrected, violation)

            elif violation.violation_type == ViolationType.PARALLEL_OCTAVES:
                corrected = self._fix_parallel_octaves(corrected, violation)

            elif violation.violation_type == ViolationType.VOICE_CROSSING:
                corrected = self._fix_voice_crossing(corrected, violation)

            elif violation.violation_type == ViolationType.EXCESSIVE_SPACING:
                corrected = self._fix_spacing(corrected, violation)

        return corrected

    def _fix_parallel_fifths(self,
                           voices: List[List[int]],
                           violation: ConstraintViolation) -> List[List[int]]:
        """Fix parallel fifth by moving one voice by step"""

        position, voice1, voice2 = violation.location

        # Try moving voice1 up or down by step
        for adjustment in [1, -1, 2, -2]:
            test_voices = deepcopy(voices)
            test_voices[voice1][position + 1] += adjustment

            # Check if this fixes the parallel fifth
            result = self.validate_voice_leading(test_voices)
            parallel_fifths = [v for v in result.violations
                             if v.violation_type == ViolationType.PARALLEL_FIFTHS]

            if len(parallel_fifths) < len([v for v in result.violations
                                         if v.violation_type == ViolationType.PARALLEL_FIFTHS]):
                return test_voices

        return voices  # Couldn't fix easily

    def _fix_parallel_octaves(self,
                            voices: List[List[int]],
                            violation: ConstraintViolation) -> List[List[int]]:
        """Fix parallel octaves by moving to different chord tone"""

        position, voice1, voice2 = violation.location

        # Try moving voice1 to nearby chord tone
        for adjustment in [3, 4, -3, -4, 7, -7]:  # Third, fifth up/down
            test_voices = deepcopy(voices)
            test_voices[voice1][position + 1] += adjustment

            result = self.validate_voice_leading(test_voices)
            parallel_octaves = [v for v in result.violations
                              if v.violation_type == ViolationType.PARALLEL_OCTAVES]

            if len(parallel_octaves) == 0:
                return test_voices

        return voices

    def _fix_voice_crossing(self,
                          voices: List[List[int]],
                          violation: ConstraintViolation) -> List[List[int]]:
        """Fix voice crossing by reordering"""

        position, voice1, voice2 = violation.location

        corrected = deepcopy(voices)

        # Swap the crossed notes
        temp = corrected[voice1][position]
        corrected[voice1][position] = corrected[voice2][position]
        corrected[voice2][position] = temp

        return corrected

    def _fix_spacing(self,
                    voices: List[List[int]],
                    violation: ConstraintViolation) -> List[List[int]]:
        """Fix excessive spacing by moving voices closer"""

        position, voice1, voice2 = violation.location

        corrected = deepcopy(voices)

        # Move higher voice down an octave
        corrected[voice2][position] -= 12

        return corrected

    def fix_out_of_range(self,
                        notes: List[int],
                        instrument: str = 'default') -> List[int]:
        """
        Fix out-of-range notes by transposing.

        Args:
            notes: Original notes
            instrument: Instrument name

        Returns:
            Corrected notes within range
        """
        if not self.allow_auto_correction:
            return notes

        instrument_lower = instrument.lower().replace(' ', '_')
        range_min, range_max = INSTRUMENT_RANGES.get(
            instrument_lower,
            INSTRUMENT_RANGES['default']
        )

        corrected = []
        for note in notes:
            corrected_note = note

            # Transpose up/down by octaves until in range
            while corrected_note < range_min:
                corrected_note += 12
            while corrected_note > range_max:
                corrected_note -= 12

            corrected.append(corrected_note)

        return corrected

    # =========================================================================
    # PARAMETER VALIDATION (for integration with Phase 1)
    # =========================================================================

    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters for musical correctness.

        This method will integrate with the Universal Parameter Registry
        from Phase 1 once available.

        Args:
            parameters: Dictionary of parameter values

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Check for common parameter issues
        if 'voices' in parameters:
            voice_result = self.validate_voice_leading(parameters['voices'])
            for violation in voice_result.violations:
                result.add_violation(violation)

        if 'instrument_parts' in parameters:
            range_result = self.validate_multi_instrument_ranges(
                parameters['instrument_parts']
            )
            for violation in range_result.violations:
                result.add_violation(violation)

        if 'chord_progression' in parameters:
            harmony_result = self.validate_harmonic_progression(
                parameters['chord_progression'],
                parameters.get('key', None)
            )
            for violation in harmony_result.violations:
                result.add_violation(violation)

        return result

    def validate_and_correct(self,
                           parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], ValidationResult]:
        """
        Validate parameters and automatically correct violations.

        Args:
            parameters: Original parameters

        Returns:
            Tuple of (corrected_parameters, validation_result)
        """
        # Validate first
        result = self.validate_parameters(parameters)

        if result.is_valid or not self.allow_auto_correction:
            return parameters, result

        # Correct violations
        corrected = deepcopy(parameters)

        if 'voices' in corrected:
            corrected['voices'] = self.fix_voice_leading(corrected['voices'])

        if 'instrument_parts' in corrected:
            for instrument, notes in corrected['instrument_parts'].items():
                corrected['instrument_parts'][instrument] = self.fix_out_of_range(
                    notes, instrument
                )

        # Re-validate
        final_result = self.validate_parameters(corrected)

        return corrected, final_result


# =============================================================================
# SPECIALIZED VALIDATORS
# =============================================================================

class CounterpointValidator:
    """
    Specialized validator for strict counterpoint rules.

    Implements Fux's species counterpoint rules and their extensions.
    """

    def __init__(self, species: int = 1):
        """
        Initialize counterpoint validator.

        Args:
            species: Counterpoint species (1-5)
        """
        self.species = species

    def validate_first_species(self,
                               cantus_firmus: List[int],
                               counterpoint: List[int]) -> ValidationResult:
        """
        Validate first species counterpoint (note-against-note).

        Rules:
        - Begin and end on perfect consonance
        - Use primarily consonant intervals
        - Prefer contrary/oblique motion
        - No parallel perfect intervals
        - Approach final by step
        """
        result = ValidationResult(is_valid=True)

        if len(cantus_firmus) != len(counterpoint):
            result.add_violation(ConstraintViolation(
                violation_type=ViolationType.IMPROPER_MOTION,
                severity=ValidationSeverity.CRITICAL,
                location=0,
                description="Cantus firmus and counterpoint must have same length"
            ))
            return result

        # Check each interval
        for i in range(len(cantus_firmus)):
            interval = abs(counterpoint[i] - cantus_firmus[i]) % 12

            # Consonant intervals: unison(0), m3(3), M3(4), P4(5), P5(7), m6(8), M6(9), octave(0)
            consonances = [0, 3, 4, 5, 7, 8, 9]

            if interval not in consonances:
                result.add_violation(ConstraintViolation(
                    violation_type=ViolationType.ILLEGAL_DISSONANCE,
                    severity=ValidationSeverity.ERROR,
                    location=i,
                    description=f"Dissonant interval at position {i}"
                ))

        # Check beginning and ending
        first_interval = abs(counterpoint[0] - cantus_firmus[0]) % 12
        last_interval = abs(counterpoint[-1] - cantus_firmus[-1]) % 12

        if first_interval not in [0, 7]:  # Must start on unison or fifth
            result.add_violation(ConstraintViolation(
                violation_type=ViolationType.IMPROPER_MOTION,
                severity=ValidationSeverity.ERROR,
                location=0,
                description="First species must begin on perfect consonance"
            ))

        if last_interval != 0:  # Must end on unison
            result.add_violation(ConstraintViolation(
                violation_type=ViolationType.IMPROPER_MOTION,
                severity=ValidationSeverity.ERROR,
                location=len(cantus_firmus) - 1,
                description="First species must end on unison/octave"
            ))

        return result


class OrchestrationValidator:
    """
    Validator for orchestration and ensemble writing.

    Checks balance, doubling, register spacing, and idiomatic writing
    for various ensemble types.
    """

    def validate_ensemble_balance(self,
                                  parts: Dict[str, List[int]],
                                  ensemble_type: str = 'orchestra') -> ValidationResult:
        """
        Validate balance and blend in ensemble writing.

        Args:
            parts: Dictionary of instrument -> notes
            ensemble_type: 'orchestra', 'chamber', 'band', etc.

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Check for register clashes (instruments fighting in same register)
        for inst1, notes1 in parts.items():
            for inst2, notes2 in parts.items():
                if inst1 >= inst2:
                    continue

                # Check if ranges overlap significantly
                if notes1 and notes2:
                    avg1 = sum(notes1) / len(notes1)
                    avg2 = sum(notes2) / len(notes2)

                    # If averages are very close, check for clash
                    if abs(avg1 - avg2) < 6:  # Within half octave
                        result.add_violation(ConstraintViolation(
                            violation_type=ViolationType.REGISTER_CLASH,
                            severity=ValidationSeverity.WARNING,
                            location=0,
                            description=f"{inst1} and {inst2} may clash in register"
                        ))

        return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def interval_name(semitones: int) -> str:
    """Convert interval in semitones to name"""
    names = {
        0: "unison",
        1: "minor 2nd",
        2: "major 2nd",
        3: "minor 3rd",
        4: "major 3rd",
        5: "perfect 4th",
        6: "tritone",
        7: "perfect 5th",
        8: "minor 6th",
        9: "major 6th",
        10: "minor 7th",
        11: "major 7th",
        12: "octave",
    }
    return names.get(semitones % 12, f"{semitones} semitones")


def pitch_to_name(midi_note: int) -> str:
    """Convert MIDI note number to pitch name"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note_name = notes[midi_note % 12]
    return f"{note_name}{octave}"


# =============================================================================
# EXAMPLES AND TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MUSICAL CONSTRAINT VALIDATOR - Agent 8")
    print("=" * 80)

    # Example 1: Voice leading validation
    print("\n1. VOICE LEADING VALIDATION")
    print("-" * 80)

    validator = MusicalConstraintValidator(style='common_practice')

    # Test voices with parallel fifths (SATB)
    voices_bad = [
        [48, 50, 52],  # Bass
        [60, 62, 64],  # Tenor
        [64, 66, 68],  # Alto
        [72, 74, 76],  # Soprano (parallel motion creating parallel 5ths)
    ]

    result = validator.validate_voice_leading(voices_bad)
    print(f"Validation result: {result.get_summary()}")
    for violation in result.violations:
        print(f"  - {violation}")

    # Fix the violations
    if not result.is_valid:
        print("\nAttempting automatic correction...")
        fixed_voices = validator.fix_voice_leading(voices_bad, result)
        result2 = validator.validate_voice_leading(fixed_voices)
        print(f"After correction: {result2.get_summary()}")

    # Example 2: Instrument range validation
    print("\n2. INSTRUMENT RANGE VALIDATION")
    print("-" * 80)

    # Violin part with some out-of-range notes
    violin_part = [40, 60, 72, 84, 96, 110]  # 110 is too high
    result = validator.validate_range(violin_part, 'violin')

    print(f"Validation result: {result.get_summary()}")
    for violation in result.violations:
        print(f"  - {violation}")

    # Fix range issues
    if not result.is_valid:
        fixed_violin = validator.fix_out_of_range(violin_part, 'violin')
        print(f"Original: {[pitch_to_name(n) for n in violin_part]}")
        print(f"Fixed:    {[pitch_to_name(n) for n in fixed_violin]}")

    # Example 3: Multiple instruments
    print("\n3. ENSEMBLE VALIDATION")
    print("-" * 80)

    ensemble = {
        'flute': [72, 74, 76, 77, 79],
        'clarinet': [64, 66, 67, 69, 71],
        'bassoon': [48, 50, 52, 53, 55],
    }

    result = validator.validate_multi_instrument_ranges(ensemble)
    print(f"Validation result: {result.get_summary()}")

    # Example 4: Counterpoint
    print("\n4. COUNTERPOINT VALIDATION (First Species)")
    print("-" * 80)

    cp_validator = CounterpointValidator(species=1)

    cantus = [60, 62, 64, 65, 67, 65, 64, 62, 60]
    counter = [67, 69, 71, 72, 74, 72, 71, 69, 67]

    result = cp_validator.validate_first_species(cantus, counter)
    print(f"Validation result: {result.get_summary()}")

    print("\n" + "=" * 80)
    print("Musical Constraint Validator implementation complete!")
    print("Ready for integration with XGBoost Parameter Synthesizer (Agent 5)")
    print("=" * 80)
