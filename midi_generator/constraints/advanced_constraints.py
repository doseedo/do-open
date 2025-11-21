#!/usr/bin/env python3
"""
Advanced Musical Constraints - Agent 8
======================================

Extended constraint validators for specialized musical contexts:
- Jazz voice leading and harmony
- Extended techniques
- Contemporary music rules
- Genre-specific constraints
- Performance practice considerations

Author: Agent 8 - Constraint Validator
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

from constraints.musical_validator import (
    ValidationResult, ValidationSeverity, ConstraintViolation,
    ViolationType, INSTRUMENT_RANGES
)


# =============================================================================
# JAZZ CONSTRAINT VALIDATOR
# =============================================================================

class JazzVoiceLeadingValidator:
    """
    Validator for jazz-specific voice leading and harmony.

    Jazz has different rules than common practice:
    - Parallel motion more acceptable
    - Voice crossing common in piano voicings
    - Drop voicings and rootless voicings
    - Upper structure triads
    - Altered dominants
    """

    def __init__(self, style: str = 'bebop'):
        """
        Initialize jazz validator.

        Args:
            style: Jazz style ('bebop', 'modal', 'contemporary', 'traditional')
        """
        self.style = style

    def validate_jazz_voicing(self,
                             chord: List[int],
                             chord_symbol: str,
                             voicing_type: str = 'rootless') -> ValidationResult:
        """
        Validate jazz piano/guitar voicing.

        Args:
            chord: MIDI pitches in the voicing
            chord_symbol: Chord symbol (e.g., 'Dm7', 'G7alt', 'Cmaj9')
            voicing_type: 'rootless', 'drop2', 'drop3', 'quartal', 'cluster'

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Parse chord symbol to get root and quality
        root, quality = self._parse_chord_symbol(chord_symbol)

        if voicing_type == 'rootless':
            # Rootless voicings should not include the root
            if any((note % 12) == (root % 12) for note in chord):
                result.add_violation(ConstraintViolation(
                    violation_type=ViolationType.INCORRECT_DOUBLING,
                    severity=ValidationSeverity.WARNING,
                    location=0,
                    description="Rootless voicing contains the root",
                    suggested_fix="Remove root or change voicing type"
                ))

            # Should have 3rd and 7th
            third = self._get_chord_tone(root, quality, 'third')
            seventh = self._get_chord_tone(root, quality, 'seventh')

            has_third = any((note % 12) == (third % 12) for note in chord)
            has_seventh = any((note % 12) == (seventh % 12) for note in chord)

            if not (has_third and has_seventh):
                result.add_violation(ConstraintViolation(
                    violation_type=ViolationType.MISSING_CHORD_TONE,
                    severity=ValidationSeverity.ERROR,
                    location=0,
                    description="Rootless voicing missing 3rd or 7th",
                    suggested_fix="Include 3rd and 7th for definition"
                ))

        elif voicing_type == 'drop2':
            # Drop 2 voicing: 4-note chord with 2nd note from top dropped an octave
            if len(chord) < 4:
                result.add_violation(ConstraintViolation(
                    violation_type=ViolationType.MISSING_CHORD_TONE,
                    severity=ValidationSeverity.ERROR,
                    location=0,
                    description="Drop 2 voicing needs at least 4 notes"
                ))

        elif voicing_type == 'quartal':
            # Quartal voicing: built from 4ths
            for i in range(len(chord) - 1):
                interval = (chord[i + 1] - chord[i]) % 12
                if interval not in [5, 6]:  # Perfect or augmented 4th
                    result.add_violation(ConstraintViolation(
                        violation_type=ViolationType.UNIDIOMATIC_WRITING,
                        severity=ValidationSeverity.WARNING,
                        location=i,
                        description=f"Non-quartal interval in quartal voicing",
                        suggested_fix="Use intervals of 4th (5 or 6 semitones)"
                    ))

        # Check spacing (wider spacing OK in jazz)
        self._check_jazz_spacing(chord, result)

        return result

    def _parse_chord_symbol(self, symbol: str) -> Tuple[int, str]:
        """Parse chord symbol into root and quality"""
        # Simplified parser
        root_map = {
            'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
            'Db': 1, 'Eb': 3, 'Gb': 6, 'Ab': 8, 'Bb': 10,
            'C#': 1, 'D#': 3, 'F#': 6, 'G#': 8, 'A#': 10,
        }

        # Extract root
        for root_name, root_pitch in root_map.items():
            if symbol.startswith(root_name):
                quality = symbol[len(root_name):]
                return root_pitch, quality

        return 0, symbol  # Default to C

    def _get_chord_tone(self, root: int, quality: str, degree: str) -> int:
        """Get pitch class of specific chord tone"""
        intervals = {
            'third': 4 if 'maj' in quality or quality == '' else 3,
            'fifth': 7,
            'seventh': 11 if 'maj7' in quality else 10,
            'ninth': 2,
            'eleventh': 5,
            'thirteenth': 9,
        }

        return (root + intervals.get(degree, 0)) % 12

    def _check_jazz_spacing(self, chord: List[int], result: ValidationResult):
        """Check spacing for jazz voicings"""
        # Jazz voicings can be wider spaced
        for i in range(len(chord) - 1):
            spacing = chord[i + 1] - chord[i]

            # Warn if spacing is too close in low register (muddy)
            if chord[i] < 48 and spacing < 3:  # Below C3
                result.add_violation(ConstraintViolation(
                    violation_type=ViolationType.EXCESSIVE_SPACING,
                    severity=ValidationSeverity.WARNING,
                    location=i,
                    description="Very close spacing in low register (may sound muddy)",
                    suggested_fix="Increase spacing below C3"
                ))


# =============================================================================
# EXTENDED TECHNIQUES VALIDATOR
# =============================================================================

class ExtendedTechniqueValidator:
    """
    Validator for extended techniques and contemporary notation.

    Checks feasibility of:
    - Multiphonics
    - Harmonics
    - Microtones
    - Extreme dynamics
    - Special effects
    """

    def validate_string_harmonics(self,
                                  note: int,
                                  instrument: str = 'violin') -> ValidationResult:
        """
        Validate natural harmonic feasibility for string instruments.

        Args:
            note: Desired harmonic pitch
            instrument: String instrument name

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Get open strings for instrument
        open_strings = {
            'violin': [55, 62, 69, 76],      # G, D, A, E
            'viola': [48, 55, 62, 69],       # C, G, D, A
            'cello': [36, 43, 50, 57],       # C, G, D, A
            'double_bass': [28, 33, 38, 43], # E, A, D, G
        }

        strings = open_strings.get(instrument.lower(), open_strings['violin'])

        # Check if note can be produced as natural harmonic
        is_possible = False
        for string in strings:
            # Natural harmonics: octave (12), P5+octave (19), 2oct (24),
            # M3+2oct (28), P5+2oct (31), etc.
            harmonic_intervals = [12, 19, 24, 28, 31, 36]

            for interval in harmonic_intervals:
                if (string + interval) == note:
                    is_possible = True
                    break

        if not is_possible:
            result.add_violation(ConstraintViolation(
                violation_type=ViolationType.IMPOSSIBLE_TECHNIQUE,
                severity=ValidationSeverity.ERROR,
                location=0,
                description=f"Note {note} cannot be produced as natural harmonic on {instrument}",
                suggested_fix="Use different pitch or artificial harmonic"
            ))

        return result

    def validate_wind_multiphonic(self,
                                 notes: List[int],
                                 instrument: str) -> ValidationResult:
        """
        Validate multiphonic feasibility for wind instruments.

        Note: This is simplified - real multiphonics are instrument-specific
        and require detailed fingering charts.
        """
        result = ValidationResult(is_valid=True)

        if len(notes) < 2:
            return result

        # Basic check: multiphonics typically have intervals of 5th or greater
        for i in range(len(notes) - 1):
            interval = notes[i + 1] - notes[i]
            if interval < 7:  # Less than perfect fifth
                result.add_violation(ConstraintViolation(
                    violation_type=ViolationType.IMPOSSIBLE_TECHNIQUE,
                    severity=ValidationSeverity.WARNING,
                    location=i,
                    description="Multiphonic intervals typically >= perfect fifth",
                    suggested_fix="Widen interval or consult multiphonic charts"
                ))

        # Check if too many notes (most multiphonics are 2-3 notes)
        if len(notes) > 3:
            result.add_violation(ConstraintViolation(
                violation_type=ViolationType.IMPOSSIBLE_TECHNIQUE,
                severity=ValidationSeverity.ERROR,
                location=0,
                description="Multiphonics with >3 notes extremely rare",
                suggested_fix="Reduce to 2-3 simultaneous pitches"
            ))

        return result


# =============================================================================
# PERFORMANCE PRACTICE VALIDATOR
# =============================================================================

class PerformancePracticeValidator:
    """
    Validator for performance practice and idiomatic writing.

    Ensures generated music is actually playable and idiomatic
    for real performers.
    """

    def validate_breathing(self,
                          notes: List[Tuple[int, float]],  # (pitch, duration)
                          instrument: str) -> ValidationResult:
        """
        Validate that wind/vocal parts have adequate breathing points.

        Args:
            notes: List of (pitch, duration) tuples
            instrument: Wind or vocal instrument

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Instruments that need breath
        wind_instruments = {
            'flute', 'oboe', 'clarinet', 'bassoon', 'saxophone',
            'trumpet', 'horn', 'trombone', 'tuba',
            'soprano', 'alto', 'tenor', 'bass'
        }

        if instrument.lower() not in wind_instruments:
            return result  # Not a wind/vocal instrument

        # Calculate phrase lengths
        current_phrase_length = 0.0
        max_phrase_without_breath = 8.0  # beats (conservative)

        for i, (pitch, duration) in enumerate(notes):
            current_phrase_length += duration

            # Check if phrase is too long without rest
            if current_phrase_length > max_phrase_without_breath:
                result.add_violation(ConstraintViolation(
                    violation_type=ViolationType.IMPOSSIBLE_TECHNIQUE,
                    severity=ValidationSeverity.ERROR,
                    location=i,
                    description=f"Phrase too long ({current_phrase_length:.1f} beats) without breath",
                    suggested_fix="Add rest or breath mark"
                ))

            # Reset on rest (pitch = -1 or 0)
            if pitch <= 0:
                current_phrase_length = 0.0

        return result

    def validate_bow_slurs(self,
                          notes: List[Tuple[int, float, bool]],  # (pitch, duration, slurred)
                          instrument: str = 'violin') -> ValidationResult:
        """
        Validate string bowing patterns.

        Args:
            notes: List of (pitch, duration, is_slurred) tuples
            instrument: String instrument

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        string_instruments = {'violin', 'viola', 'cello', 'double_bass'}

        if instrument.lower() not in string_instruments:
            return result

        # Check slur lengths
        current_slur_length = 0.0
        max_slur_length = 8.0  # beats per bow

        in_slur = False

        for i, (pitch, duration, slurred) in enumerate(notes):
            if slurred:
                if not in_slur:
                    current_slur_length = 0.0
                    in_slur = True

                current_slur_length += duration

                if current_slur_length > max_slur_length:
                    result.add_violation(ConstraintViolation(
                        violation_type=ViolationType.IMPOSSIBLE_TECHNIQUE,
                        severity=ValidationSeverity.WARNING,
                        location=i,
                        description=f"Very long slur ({current_slur_length:.1f} beats)",
                        suggested_fix="Break into shorter slurs"
                    ))
            else:
                in_slur = False
                current_slur_length = 0.0

        return result

    def validate_piano_hand_span(self,
                                notes: List[int],
                                hand: str = 'right') -> ValidationResult:
        """
        Validate that piano chord is playable by one hand.

        Args:
            notes: Simultaneous notes in chord
            hand: 'left' or 'right'

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        if not notes:
            return result

        # Maximum comfortable span for average hand
        max_span = 12  # One octave (some can do 13-14)

        span = max(notes) - min(notes)

        if span > max_span:
            result.add_violation(ConstraintViolation(
                violation_type=ViolationType.IMPOSSIBLE_TECHNIQUE,
                severity=ValidationSeverity.ERROR,
                location=0,
                description=f"{hand.capitalize()} hand span of {span} semitones too wide",
                suggested_fix=f"Reduce span to {max_span} or less, or split between hands"
            ))

        # Check number of notes (typical limit is 5 per hand)
        if len(notes) > 5:
            result.add_violation(ConstraintViolation(
                violation_type=ViolationType.IMPOSSIBLE_TECHNIQUE,
                severity=ValidationSeverity.CRITICAL,
                location=0,
                description=f"{len(notes)} notes exceeds 5 fingers",
                suggested_fix="Reduce to 5 notes or arpeggiate"
            ))

        return result


# =============================================================================
# GENRE-SPECIFIC CONSTRAINT SETS
# =============================================================================

class GenreConstraintSet:
    """
    Pre-configured constraint sets for different musical genres.

    Provides quick validation against genre-specific expectations.
    """

    @staticmethod
    def get_baroque_constraints() -> Dict[str, Any]:
        """Constraints for Baroque music (Bach, Handel, Vivaldi)"""
        return {
            'parallel_fifths': 'forbidden',
            'parallel_octaves': 'forbidden',
            'augmented_intervals': 'melodic_only',  # Harmonic OK
            'voice_crossing': 'rare',
            'max_melodic_interval': 8,  # Minor 6th
            'dissonance_treatment': 'strict',
            'suspensions': 'prepared',
            'cadences': 'functional',
        }

    @staticmethod
    def get_romantic_constraints() -> Dict[str, Any]:
        """Constraints for Romantic music (Chopin, Brahms, Wagner)"""
        return {
            'parallel_fifths': 'allowed_with_justification',
            'parallel_octaves': 'allowed_for_effect',
            'augmented_intervals': 'allowed',
            'voice_crossing': 'common',
            'max_melodic_interval': 24,  # Wide leaps OK
            'dissonance_treatment': 'flexible',
            'chromaticism': 'extensive',
            'modulation': 'frequent',
        }

    @staticmethod
    def get_jazz_constraints() -> Dict[str, Any]:
        """Constraints for Jazz"""
        return {
            'parallel_fifths': 'allowed',
            'parallel_octaves': 'allowed',
            'voice_crossing': 'common',
            'extended_harmony': 'expected',
            'altered_dominants': 'common',
            'upper_structures': 'allowed',
            'voice_leading': 'smooth_preferred',
            'substitutions': 'encouraged',
        }

    @staticmethod
    def get_contemporary_constraints() -> Dict[str, Any]:
        """Constraints for Contemporary/Avant-garde"""
        return {
            'traditional_rules': 'optional',
            'atonality': 'allowed',
            'microtonality': 'allowed',
            'extended_techniques': 'encouraged',
            'cluster_chords': 'allowed',
            'unconventional_notation': 'allowed',
        }


# =============================================================================
# ORCHESTRATION-SPECIFIC VALIDATORS
# =============================================================================

class OrchestrationConstraintValidator:
    """
    Advanced orchestration constraint validation.

    Based on:
    - Rimsky-Korsakov: Principles of Orchestration
    - Adler: The Study of Orchestration
    - Berlioz: Grand Traité d'Instrumentation
    """

    def validate_doubling(self,
                         parts: Dict[str, List[int]],
                         register: str = 'middle') -> ValidationResult:
        """
        Validate orchestral doubling practices.

        Args:
            parts: Instrument -> notes mapping
            register: 'low', 'middle', 'high'

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Check for common doubling issues
        # 1. Don't double leading tone
        # 2. Double root in root position
        # 3. Be careful doubling 3rd in major (OK in minor)
        # 4. Winds doubling in unison can be powerful or bland

        # Check for unison doublings
        for inst1, notes1 in parts.items():
            for inst2, notes2 in parts.items():
                if inst1 >= inst2:
                    continue

                # Check if instruments are playing exact same notes
                if notes1 == notes2:
                    result.add_violation(ConstraintViolation(
                        violation_type=ViolationType.INCORRECT_DOUBLING,
                        severity=ValidationSeverity.INFO,
                        location=0,
                        description=f"{inst1} and {inst2} in unison throughout",
                        suggested_fix="Consider octave doubling or independent parts"
                    ))

        return result

    def validate_range_distribution(self,
                                   score: Dict[str, List[int]]) -> ValidationResult:
        """
        Validate that instruments are distributed across range effectively.

        Checks for:
        - Gaps in texture
        - Overcrowding in one register
        - Proper bass foundation
        """
        result = ValidationResult(is_valid=True)

        # Collect all pitches being played
        all_pitches = []
        for notes in score.values():
            all_pitches.extend(notes)

        if not all_pitches:
            return result

        all_pitches.sort()

        # Check for large gaps
        for i in range(len(all_pitches) - 1):
            gap = all_pitches[i + 1] - all_pitches[i]
            if gap > 24:  # 2 octave gap
                result.add_violation(ConstraintViolation(
                    violation_type=ViolationType.EXCESSIVE_SPACING,
                    severity=ValidationSeverity.WARNING,
                    location=i,
                    description=f"Large gap ({gap} semitones) in texture",
                    suggested_fix="Fill gap with additional voice or adjust spacing"
                ))

        # Check bass register coverage (below C3 = 48)
        has_bass = any(pitch < 48 for pitch in all_pitches)
        if not has_bass and len(score) > 2:  # Multiple instruments but no bass
            result.add_violation(ConstraintViolation(
                violation_type=ViolationType.POOR_BALANCE,
                severity=ValidationSeverity.WARNING,
                location=0,
                description="No bass register coverage in ensemble",
                suggested_fix="Add bass instrument or extend range downward"
            ))

        return result


# =============================================================================
# INTEGRATION UTILITIES
# =============================================================================

def create_validator_for_style(style: str) -> 'MusicalConstraintValidator':
    """
    Factory function to create validator with appropriate style settings.

    Args:
        style: Musical style name

    Returns:
        Configured MusicalConstraintValidator
    """
    from constraints.musical_validator import MusicalConstraintValidator

    style_map = {
        'baroque': ('baroque', True),
        'classical': ('common_practice', True),
        'romantic': ('common_practice', False),
        'jazz': ('jazz', False),
        'contemporary': ('contemporary', False),
        'bebop': ('jazz', False),
        'fusion': ('jazz', False),
    }

    style_key, strict = style_map.get(style.lower(), ('common_practice', False))

    return MusicalConstraintValidator(
        style=style_key,
        strict_mode=strict,
        allow_auto_correction=True
    )


if __name__ == "__main__":
    print("=" * 80)
    print("ADVANCED MUSICAL CONSTRAINTS - Agent 8")
    print("=" * 80)

    # Example 1: Jazz voicing validation
    print("\n1. JAZZ VOICING VALIDATION")
    print("-" * 80)

    jazz_validator = JazzVoiceLeadingValidator(style='bebop')

    # Dm7 rootless voicing (F-A-C-E)
    dm7_voicing = [53, 57, 60, 64]  # F3, A3, C4, E4
    result = jazz_validator.validate_jazz_voicing(dm7_voicing, 'Dm7', 'rootless')
    print(f"Dm7 rootless voicing validation: {result.get_summary()}")

    # Example 2: Extended technique validation
    print("\n2. EXTENDED TECHNIQUES VALIDATION")
    print("-" * 80)

    ext_validator = ExtendedTechniqueValidator()

    # Check if note can be played as violin harmonic
    harmonic_note = 74  # D5
    result = ext_validator.validate_string_harmonics(harmonic_note, 'violin')
    print(f"Violin harmonic validation: {result.get_summary()}")

    # Example 3: Performance practice
    print("\n3. PERFORMANCE PRACTICE VALIDATION")
    print("-" * 80)

    perf_validator = PerformancePracticeValidator()

    # Long flute phrase without breath
    long_phrase = [(72, 1.0)] * 10  # 10 beats without rest
    result = perf_validator.validate_breathing(long_phrase, 'flute')
    print(f"Breathing validation: {result.get_summary()}")
    for violation in result.violations:
        print(f"  - {violation}")

    # Piano hand span
    wide_chord = [48, 52, 55, 59, 64]  # C-E-G-B-E (16 semitone span)
    result = perf_validator.validate_piano_hand_span(wide_chord, 'right')
    print(f"\nPiano hand span validation: {result.get_summary()}")
    for violation in result.violations:
        print(f"  - {violation}")

    print("\n" + "=" * 80)
    print("Advanced Constraints implementation complete!")
    print("=" * 80)
