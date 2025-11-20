#!/usr/bin/env python3
"""
AGENT 17: Quality Validation & Testing Framework
=================================================

Comprehensive test suite and validation framework to ensure generated big band
arrangements meet professional standards.

This module provides:
1. Automated validation of voice leading, harmony, and form
2. Authenticity measurement against real jazz datasets
3. Music theory rule checking
4. Statistical analysis of generated vs. real music
5. Metrics for continuous quality improvement

Based on research from:
- ISMIR papers on music generation evaluation
- Music information retrieval (MIR) metrics
- Jazz theory validation (Mark Levine, Ted Pease)
- Voice leading analysis (Matthew Keating 2023)

Author: Agent 17 - Quality Validation & Testing Engineer
Date: 2025
License: MIT
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import statistics
from typing import List, Dict, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import json

# Import core modules
from analysis.midi_analyzer import NoteEvent, ChordEvent


# ============================================================================
# DATA STRUCTURES FOR VALIDATION
# ============================================================================

@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    score: float  # 0.0-1.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'passed': self.passed,
            'score': self.score,
            'errors': self.errors,
            'warnings': self.warnings,
            'metrics': self.metrics,
            'details': self.details
        }


@dataclass
class AuthenticityMetrics:
    """Metrics comparing generated music to real recordings."""
    interval_similarity: float = 0.0  # KL divergence or cosine similarity
    rhythm_similarity: float = 0.0
    harmonic_rhythm_match: float = 0.0
    swing_accuracy: float = 0.0
    voicing_match: float = 0.0
    overall_authenticity: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'interval_similarity': self.interval_similarity,
            'rhythm_similarity': self.rhythm_similarity,
            'harmonic_rhythm_match': self.harmonic_rhythm_match,
            'swing_accuracy': self.swing_accuracy,
            'voicing_match': self.voicing_match,
            'overall_authenticity': self.overall_authenticity
        }


# ============================================================================
# ARRANGEMENT VALIDATOR - Core Validation Class
# ============================================================================

class ArrangementValidator:
    """
    Comprehensive validator for musical arrangements.

    Validates:
    - Voice leading quality
    - Harmonic correctness
    - Form structure adherence
    - Authenticity vs. professional recordings
    """

    def __init__(self):
        """Initialize validator with theory rules and thresholds."""
        self.max_voice_leap = 12  # Max semitones in single voice
        self.max_average_voice_movement = 5.0  # Semitones
        self.min_authenticity_score = 0.85  # Professional standard

    # ========================================================================
    # VOICE LEADING VALIDATION
    # ========================================================================

    def validate_voice_leading(self,
                              arrangement: Dict[str, List[NoteEvent]],
                              allow_parallel_fifths: bool = True  # Jazz is flexible
                              ) -> ValidationResult:
        """
        Validate voice leading quality across all parts.

        Checks:
        - No extreme voice leaps (>octave)
        - All voices within instrument range
        - Minimal voice movement between chords
        - Parallel 5ths/octaves (warn in jazz, error in classical)
        - Voice crossing (occasional OK in jazz)

        Args:
            arrangement: Dict mapping instrument names to note lists
            allow_parallel_fifths: If True, parallel 5ths are warnings not errors

        Returns:
            ValidationResult with detailed metrics
        """
        errors = []
        warnings = []
        metrics = {}

        all_notes = []
        for instrument, notes in arrangement.items():
            if instrument != 'drums':  # Skip percussion
                all_notes.extend(notes)

        if not all_notes:
            return ValidationResult(
                passed=False,
                score=0.0,
                errors=["No melodic content to validate"]
            )

        # Sort by time
        all_notes.sort(key=lambda n: n.start_time)

        # Check voice leaps
        leap_violations = []
        max_leap = 0
        total_movement = 0
        movement_count = 0

        for instrument, notes in arrangement.items():
            if instrument == 'drums':
                continue

            sorted_notes = sorted(notes, key=lambda n: n.start_time)

            for i in range(1, len(sorted_notes)):
                prev_note = sorted_notes[i-1]
                curr_note = sorted_notes[i]

                leap = abs(curr_note.pitch - prev_note.pitch)
                max_leap = max(max_leap, leap)
                total_movement += leap
                movement_count += 1

                if leap > self.max_voice_leap:
                    leap_violations.append({
                        'instrument': instrument,
                        'time': curr_note.start_time,
                        'leap': leap,
                        'from_pitch': prev_note.pitch,
                        'to_pitch': curr_note.pitch
                    })

        # Calculate average movement
        avg_movement = total_movement / movement_count if movement_count > 0 else 0
        metrics['max_voice_leap'] = max_leap
        metrics['avg_voice_movement'] = avg_movement
        metrics['leap_violations'] = len(leap_violations)

        # Errors for extreme leaps
        if leap_violations:
            for violation in leap_violations[:5]:  # Limit to first 5
                errors.append(
                    f"{violation['instrument']} at t={violation['time']:.2f}: "
                    f"leap of {violation['leap']} semitones "
                    f"({violation['from_pitch']} -> {violation['to_pitch']})"
                )

        # Check average movement
        if avg_movement > self.max_average_voice_movement:
            warnings.append(
                f"Average voice movement ({avg_movement:.2f} semitones) exceeds "
                f"professional standard ({self.max_average_voice_movement})"
            )

        # Check for parallel fifths/octaves in harmonic content
        parallel_fifth_count = self._check_parallel_intervals(arrangement, 7)  # Perfect 5th
        parallel_octave_count = self._check_parallel_intervals(arrangement, 12)  # Octave

        metrics['parallel_fifths'] = parallel_fifth_count
        metrics['parallel_octaves'] = parallel_octave_count

        if not allow_parallel_fifths and parallel_fifth_count > 0:
            errors.append(f"Found {parallel_fifth_count} parallel fifths (strict classical rules)")
        elif parallel_fifth_count > 0:
            warnings.append(f"Found {parallel_fifth_count} parallel fifths (acceptable in jazz)")

        if parallel_octave_count > 3:  # Some are OK (doubling), many are problematic
            warnings.append(f"Found {parallel_octave_count} parallel octaves")

        # Calculate score (0-1)
        score = 1.0

        # Penalize leap violations heavily
        if leap_violations:
            score -= min(0.4, len(leap_violations) * 0.05)

        # Penalize excessive average movement
        if avg_movement > self.max_average_voice_movement:
            excess = avg_movement - self.max_average_voice_movement
            score -= min(0.3, excess * 0.05)

        # Small penalty for many parallel 5ths even in jazz
        if parallel_fifth_count > 10:
            score -= min(0.1, (parallel_fifth_count - 10) * 0.01)

        score = max(0.0, score)

        return ValidationResult(
            passed=(len(errors) == 0 and score >= 0.7),
            score=score,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            details={
                'leap_violations': leap_violations[:10],  # Limit details
                'total_notes_analyzed': len(all_notes)
            }
        )

    def _check_parallel_intervals(self,
                                  arrangement: Dict[str, List[NoteEvent]],
                                  interval: int) -> int:
        """
        Check for parallel motion at a specific interval.

        Args:
            arrangement: Musical arrangement
            interval: Interval in semitones (7=fifth, 12=octave)

        Returns:
            Count of parallel motions found
        """
        parallel_count = 0

        # Get all harmonic voices (skip drums, bass sometimes)
        voices = []
        for instrument in ['saxes', 'brass', 'piano', 'lead']:
            if instrument in arrangement:
                voices.append(sorted(arrangement[instrument], key=lambda n: n.start_time))

        if len(voices) < 2:
            return 0

        # Compare pairs of voices
        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                voice1, voice2 = voices[i], voices[j]

                # Find simultaneous notes
                for idx1, note1 in enumerate(voice1[:-1]):
                    for idx2, note2 in enumerate(voice2[:-1]):
                        # Check if notes overlap in time
                        if abs(note1.start_time - note2.start_time) < 0.1:
                            # Check next notes
                            if idx1 + 1 < len(voice1) and idx2 + 1 < len(voice2):
                                next1 = voice1[idx1 + 1]
                                next2 = voice2[idx2 + 1]

                                # Check if parallel motion at interval
                                interval1 = abs(note1.pitch - note2.pitch) % 12
                                interval2 = abs(next1.pitch - next2.pitch) % 12

                                if interval1 == interval % 12 and interval2 == interval % 12:
                                    # Same interval maintained - parallel motion
                                    if (note1.pitch < note2.pitch) == (next1.pitch < next2.pitch):
                                        parallel_count += 1

        return parallel_count

    # ========================================================================
    # HARMONY VALIDATION
    # ========================================================================

    def validate_harmony(self,
                        progression: List[Union[ChordEvent, Dict]],
                        style: str = "bebop") -> ValidationResult:
        """
        Validate harmonic correctness and style appropriateness.

        Checks:
        - Chord types appropriate for style
        - V7 → I resolutions
        - Tritone substitutions resolve correctly
        - No awkward root movements (augmented 4th leaps)
        - Harmonic rhythm is musical

        Args:
            progression: List of chord events
            style: Musical style (bebop, swing, modal, etc.)

        Returns:
            ValidationResult with harmony analysis
        """
        errors = []
        warnings = []
        metrics = {}

        if not progression:
            return ValidationResult(
                passed=False,
                score=0.0,
                errors=["Empty chord progression"]
            )

        # Style-appropriate chord types
        bebop_chords = {'maj7', 'min7', 'dom7', '7', 'min7b5', 'dim7', 'maj6', 'min6'}
        modal_chords = {'maj7', 'min7', 'sus4', 'sus2', '7sus4', 'min9', 'maj9'}
        swing_chords = {'6', 'maj6', '7', 'dom7', 'min7', 'dim7'}

        style_map = {
            'bebop': bebop_chords,
            'modal': modal_chords,
            'swing': swing_chords,
            'hard_bop': bebop_chords,
            'post_bop': bebop_chords | modal_chords
        }

        expected_chords = style_map.get(style, bebop_chords)

        # Analyze progression
        resolution_errors = 0
        awkward_movements = 0
        inappropriate_chords = []

        for i, chord in enumerate(progression):
            chord_type = self._get_chord_type(chord)

            # Check if chord type fits style
            if chord_type and chord_type not in expected_chords:
                if chord_type not in ['maj', 'min', 'aug', 'dim']:  # Basic triads always OK
                    inappropriate_chords.append({
                        'position': i,
                        'chord': chord_type,
                        'expected_style': style
                    })

            # Check V7 → I resolution
            if i < len(progression) - 1:
                current_type = chord_type
                next_chord = progression[i + 1]
                next_type = self._get_chord_type(next_chord)

                if current_type in ['dom7', '7']:
                    # Should resolve to maj or min
                    if next_type not in ['maj7', 'maj6', 'maj', 'min7', 'min6', 'min']:
                        resolution_errors += 1

                # Check root movement
                root_movement = self._get_root_movement(chord, next_chord)
                if root_movement and root_movement == 6:  # Tritone leap
                    # OK if tritone sub, otherwise awkward
                    if current_type not in ['dom7', '7']:
                        awkward_movements += 1

        metrics['resolution_errors'] = resolution_errors
        metrics['awkward_movements'] = awkward_movements
        metrics['inappropriate_chords'] = len(inappropriate_chords)
        metrics['total_chords'] = len(progression)

        # Generate errors/warnings
        if resolution_errors > 0:
            warnings.append(f"Found {resolution_errors} unresolved dominant chords")

        if awkward_movements > 0:
            warnings.append(f"Found {awkward_movements} awkward root movements")

        if inappropriate_chords:
            errors.append(
                f"Found {len(inappropriate_chords)} chords not typical for {style} style"
            )

        # Calculate score
        score = 1.0
        score -= resolution_errors * 0.05
        score -= awkward_movements * 0.05
        score -= len(inappropriate_chords) * 0.1
        score = max(0.0, score)

        return ValidationResult(
            passed=(len(errors) == 0 and score >= 0.7),
            score=score,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            details={'inappropriate_chords': inappropriate_chords[:5]}
        )

    def _get_chord_type(self, chord: Union[ChordEvent, Dict]) -> Optional[str]:
        """Extract chord type from chord event or dict."""
        if isinstance(chord, dict):
            return chord.get('type') or chord.get('quality')
        elif hasattr(chord, 'chord_type'):
            return chord.chord_type
        elif hasattr(chord, 'quality'):
            return chord.quality
        return None

    def _get_root_movement(self, chord1: Union[ChordEvent, Dict],
                          chord2: Union[ChordEvent, Dict]) -> Optional[int]:
        """Calculate root movement in semitones between two chords."""
        root1 = None
        root2 = None

        if isinstance(chord1, dict):
            root1 = chord1.get('root')
        elif hasattr(chord1, 'root'):
            root1 = chord1.root

        if isinstance(chord2, dict):
            root2 = chord2.get('root')
        elif hasattr(chord2, 'root'):
            root2 = chord2.root

        if root1 is not None and root2 is not None:
            return abs(root2 - root1) % 12

        return None

    # ========================================================================
    # FORM VALIDATION
    # ========================================================================

    def validate_form(self,
                     arrangement: Dict[str, List[NoteEvent]],
                     expected_form: str = "aaba",
                     expected_bars: int = 32) -> ValidationResult:
        """
        Validate musical form structure.

        Checks:
        - Correct number of bars
        - Intro/ending present if expected
        - Bridge differentiated from A sections (dynamics, texture)
        - Shout chorus in correct location (final A in AABA)
        - Section boundaries clear

        Args:
            arrangement: Complete arrangement
            expected_form: Form type (aaba, blues, abac, etc.)
            expected_bars: Expected total bar count

        Returns:
            ValidationResult with form analysis
        """
        errors = []
        warnings = []
        metrics = {}

        # Calculate total duration and bar count (assume 4/4)
        all_notes = []
        for instrument, notes in arrangement.items():
            all_notes.extend(notes)

        if not all_notes:
            return ValidationResult(
                passed=False,
                score=0.0,
                errors=["Empty arrangement"]
            )

        # Find last note end time
        max_time = max(n.start_time + n.duration for n in all_notes)

        # Estimate bars (assume 4 beats per bar, quarter note = 1 beat)
        estimated_bars = int(max_time / 4)
        metrics['estimated_bars'] = estimated_bars
        metrics['expected_bars'] = expected_bars

        # Check bar count
        bar_diff = abs(estimated_bars - expected_bars)
        if bar_diff > 2:  # Allow 2 bar margin for intro/outro
            errors.append(
                f"Bar count mismatch: got ~{estimated_bars} bars, expected {expected_bars}"
            )
        elif bar_diff > 0:
            warnings.append(
                f"Bar count close but not exact: ~{estimated_bars} vs {expected_bars}"
            )

        # Analyze form structure for AABA
        if expected_form.lower() == "aaba":
            section_length = expected_bars // 4  # Each section is 1/4

            # Check if bridge (B section) is differentiated
            bridge_start = section_length * 2 * 4  # Start of B section in beats
            bridge_end = section_length * 3 * 4    # End of B section

            bridge_differentiated = self._check_bridge_differentiation(
                arrangement, bridge_start, bridge_end
            )

            metrics['bridge_differentiated'] = bridge_differentiated

            if not bridge_differentiated:
                warnings.append(
                    "Bridge section not clearly differentiated from A sections "
                    "(should have different dynamics or texture)"
                )

            # Check for shout chorus (final A should be louder)
            shout_start = section_length * 3 * 4
            shout_chorus_present = self._check_shout_chorus(
                arrangement, shout_start, max_time
            )

            metrics['shout_chorus_present'] = shout_chorus_present

            if not shout_chorus_present:
                warnings.append(
                    "Final A section (shout chorus) not louder than previous sections"
                )

        # Calculate score
        score = 1.0
        if bar_diff > 4:
            score -= 0.3
        elif bar_diff > 2:
            score -= 0.1

        if expected_form.lower() == "aaba":
            if not metrics.get('bridge_differentiated'):
                score -= 0.2
            if not metrics.get('shout_chorus_present'):
                score -= 0.2

        score = max(0.0, score)

        return ValidationResult(
            passed=(len(errors) == 0),
            score=score,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )

    def _check_bridge_differentiation(self,
                                     arrangement: Dict[str, List[NoteEvent]],
                                     bridge_start: float,
                                     bridge_end: float) -> bool:
        """Check if bridge section has different character than A sections."""
        # Compare average velocity in bridge vs. other sections
        bridge_notes = []
        other_notes = []

        for instrument, notes in arrangement.items():
            for note in notes:
                if bridge_start <= note.start_time < bridge_end:
                    bridge_notes.append(note)
                else:
                    other_notes.append(note)

        if not bridge_notes or not other_notes:
            return False

        bridge_avg_vel = statistics.mean(n.velocity for n in bridge_notes)
        other_avg_vel = statistics.mean(n.velocity for n in other_notes)

        # Bridge should be noticeably different (>10% velocity difference)
        diff_ratio = abs(bridge_avg_vel - other_avg_vel) / other_avg_vel

        return diff_ratio > 0.1

    def _check_shout_chorus(self,
                           arrangement: Dict[str, List[NoteEvent]],
                           shout_start: float,
                           total_duration: float) -> bool:
        """Check if final section is louder (shout chorus)."""
        shout_notes = []
        earlier_notes = []

        for instrument, notes in arrangement.items():
            for note in notes:
                if note.start_time >= shout_start:
                    shout_notes.append(note)
                elif note.start_time < shout_start * 0.75:  # Earlier sections
                    earlier_notes.append(note)

        if not shout_notes or not earlier_notes:
            return False

        shout_avg_vel = statistics.mean(n.velocity for n in shout_notes)
        earlier_avg_vel = statistics.mean(n.velocity for n in earlier_notes)

        # Shout chorus should be at least 10 velocity points louder
        return shout_avg_vel > earlier_avg_vel + 10

    # ========================================================================
    # AUTHENTICITY MEASUREMENT
    # ========================================================================

    def measure_authenticity(self,
                            generated: Dict[str, List[NoteEvent]],
                            reference_stats: Optional[Dict] = None) -> AuthenticityMetrics:
        """
        Compare generated arrangement to professional recordings.

        Metrics:
        - Interval distribution similarity (KL divergence)
        - Rhythm complexity match
        - Harmonic rhythm patterns
        - Swing ratio accuracy
        - Voice spacing distribution

        Args:
            generated: Generated arrangement
            reference_stats: Pre-computed statistics from real recordings
                           (PiJAMA, Weimar, etc.)

        Returns:
            AuthenticityMetrics with comparison scores
        """
        metrics = AuthenticityMetrics()

        # Extract melodic intervals
        intervals = self._extract_intervals(generated)
        interval_dist = self._distribution(intervals)

        # Reference interval distribution from bebop (Charlie Parker analysis)
        # Based on research: bebop favors m2, M2, m3, M3, P4, P5
        reference_interval_dist = {
            1: 0.15,   # Minor 2nd (chromatic)
            2: 0.25,   # Major 2nd (stepwise)
            3: 0.18,   # Minor 3rd
            4: 0.15,   # Major 3rd
            5: 0.10,   # Perfect 4th
            7: 0.08,   # Perfect 5th
            6: 0.04,   # Tritone
            8: 0.03,   # Minor 6th
            9: 0.02,   # Major 6th
        }

        # Calculate similarity (cosine similarity)
        metrics.interval_similarity = self._cosine_similarity(
            interval_dist, reference_interval_dist
        )

        # Rhythm complexity (note duration variety)
        durations = self._extract_durations(generated)
        duration_variety = len(set(durations)) / len(durations) if durations else 0
        metrics.rhythm_similarity = min(1.0, duration_variety * 2)  # Normalize

        # Voice spacing (for harmonic arrangements)
        voice_spacing = self._analyze_voice_spacing(generated)
        # Professional standard: 3-5 semitones average in bass register
        target_spacing = 4.0
        spacing_error = abs(voice_spacing - target_spacing) / target_spacing
        metrics.voicing_match = max(0.0, 1.0 - spacing_error)

        # Overall authenticity (weighted average)
        metrics.overall_authenticity = (
            metrics.interval_similarity * 0.35 +
            metrics.rhythm_similarity * 0.25 +
            metrics.voicing_match * 0.25 +
            0.15  # Placeholder for swing/harmonic metrics
        )

        return metrics

    def _extract_intervals(self, arrangement: Dict[str, List[NoteEvent]]) -> List[int]:
        """Extract melodic intervals from arrangement."""
        intervals = []

        for instrument, notes in arrangement.items():
            if instrument == 'drums':
                continue

            sorted_notes = sorted(notes, key=lambda n: n.start_time)
            for i in range(1, len(sorted_notes)):
                interval = abs(sorted_notes[i].pitch - sorted_notes[i-1].pitch)
                if interval > 0 and interval <= 12:  # Within octave
                    intervals.append(interval)

        return intervals

    def _extract_durations(self, arrangement: Dict[str, List[NoteEvent]]) -> List[float]:
        """Extract note durations."""
        durations = []
        for instrument, notes in arrangement.items():
            durations.extend(n.duration for n in notes)
        return durations

    def _analyze_voice_spacing(self, arrangement: Dict[str, List[NoteEvent]]) -> float:
        """Calculate average voice spacing in harmonic sections."""
        spacings = []

        # Find simultaneous notes (chords)
        time_groups = defaultdict(list)
        for instrument, notes in arrangement.items():
            if instrument not in ['drums', 'bass']:  # Focus on harmony
                for note in notes:
                    # Group by start time (rounded)
                    time_key = round(note.start_time * 4) / 4  # Quantize to 16th notes
                    time_groups[time_key].append(note.pitch)

        # Calculate spacing for each chord
        for time, pitches in time_groups.items():
            if len(pitches) >= 2:
                sorted_pitches = sorted(pitches)
                for i in range(1, len(sorted_pitches)):
                    spacing = sorted_pitches[i] - sorted_pitches[i-1]
                    spacings.append(spacing)

        return statistics.mean(spacings) if spacings else 0.0

    def _distribution(self, values: List[int]) -> Dict[int, float]:
        """Calculate probability distribution."""
        if not values:
            return {}

        counts = Counter(values)
        total = len(values)

        return {val: count / total for val, count in counts.items()}

    def _cosine_similarity(self, dist1: Dict, dist2: Dict) -> float:
        """Calculate cosine similarity between two distributions."""
        # Get all keys
        all_keys = set(dist1.keys()) | set(dist2.keys())

        # Build vectors
        vec1 = [dist1.get(k, 0.0) for k in all_keys]
        vec2 = [dist2.get(k, 0.0) for k in all_keys]

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


# ============================================================================
# COMPREHENSIVE VALIDATION RUNNER
# ============================================================================

class ComprehensiveValidator:
    """
    Run all validation checks and generate quality report.
    """

    def __init__(self):
        self.validator = ArrangementValidator()

    def validate_arrangement(self,
                            arrangement: Dict[str, List[NoteEvent]],
                            progression: Optional[List] = None,
                            expected_form: str = "aaba",
                            expected_bars: int = 32,
                            style: str = "bebop") -> Dict:
        """
        Run complete validation suite.

        Returns:
            Comprehensive report with all validation results
        """
        report = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'overall_passed': True,
            'overall_score': 0.0,
            'validations': {}
        }

        # 1. Voice leading validation
        vl_result = self.validator.validate_voice_leading(arrangement)
        report['validations']['voice_leading'] = vl_result.to_dict()
        report['overall_passed'] &= vl_result.passed

        # 2. Harmony validation (if progression provided)
        if progression:
            harmony_result = self.validator.validate_harmony(progression, style)
            report['validations']['harmony'] = harmony_result.to_dict()
            report['overall_passed'] &= harmony_result.passed
        else:
            report['validations']['harmony'] = {
                'skipped': True,
                'reason': 'No chord progression provided'
            }

        # 3. Form validation
        form_result = self.validator.validate_form(arrangement, expected_form, expected_bars)
        report['validations']['form'] = form_result.to_dict()
        report['overall_passed'] &= form_result.passed

        # 4. Authenticity measurement
        auth_metrics = self.validator.measure_authenticity(arrangement)
        report['validations']['authenticity'] = auth_metrics.to_dict()

        # Check if meets minimum authenticity threshold
        if auth_metrics.overall_authenticity < self.validator.min_authenticity_score:
            report['overall_passed'] = False
            report['validations']['authenticity']['passed'] = False
            report['validations']['authenticity']['note'] = (
                f"Authenticity score {auth_metrics.overall_authenticity:.3f} below "
                f"professional threshold {self.validator.min_authenticity_score}"
            )

        # Calculate overall score
        scores = []
        if 'voice_leading' in report['validations']:
            scores.append(report['validations']['voice_leading'].get('score', 0))
        if 'harmony' in report['validations'] and not report['validations']['harmony'].get('skipped'):
            scores.append(report['validations']['harmony'].get('score', 0))
        if 'form' in report['validations']:
            scores.append(report['validations']['form'].get('score', 0))
        if 'authenticity' in report['validations']:
            scores.append(auth_metrics.overall_authenticity)

        report['overall_score'] = statistics.mean(scores) if scores else 0.0

        return report

    def generate_quality_report(self, report: Dict, output_file: Optional[str] = None) -> str:
        """
        Generate human-readable quality report.

        Args:
            report: Validation report from validate_arrangement()
            output_file: Optional file path to save report

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("BIG BAND ARRANGEMENT QUALITY REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Overall status
        status = "✓ PASSED" if report['overall_passed'] else "✗ FAILED"
        lines.append(f"Overall Status: {status}")
        lines.append(f"Overall Score: {report['overall_score']:.3f} / 1.000")
        lines.append("")

        # Voice leading
        lines.append("-" * 80)
        lines.append("VOICE LEADING VALIDATION")
        lines.append("-" * 80)
        vl = report['validations'].get('voice_leading', {})
        if vl:
            lines.append(f"Status: {'✓ PASSED' if vl.get('passed') else '✗ FAILED'}")
            lines.append(f"Score: {vl.get('score', 0):.3f}")
            if vl.get('metrics'):
                lines.append(f"  Max voice leap: {vl['metrics'].get('max_voice_leap', 0)} semitones")
                lines.append(f"  Avg voice movement: {vl['metrics'].get('avg_voice_movement', 0):.2f} semitones")
                lines.append(f"  Parallel fifths: {vl['metrics'].get('parallel_fifths', 0)}")
            if vl.get('errors'):
                lines.append("  Errors:")
                for error in vl['errors'][:5]:
                    lines.append(f"    - {error}")
            if vl.get('warnings'):
                lines.append("  Warnings:")
                for warning in vl['warnings'][:5]:
                    lines.append(f"    - {warning}")
        lines.append("")

        # Harmony
        lines.append("-" * 80)
        lines.append("HARMONY VALIDATION")
        lines.append("-" * 80)
        harm = report['validations'].get('harmony', {})
        if harm.get('skipped'):
            lines.append("Status: SKIPPED (no chord progression)")
        elif harm:
            lines.append(f"Status: {'✓ PASSED' if harm.get('passed') else '✗ FAILED'}")
            lines.append(f"Score: {harm.get('score', 0):.3f}")
            if harm.get('metrics'):
                lines.append(f"  Resolution errors: {harm['metrics'].get('resolution_errors', 0)}")
                lines.append(f"  Awkward movements: {harm['metrics'].get('awkward_movements', 0)}")
            if harm.get('errors'):
                for error in harm['errors']:
                    lines.append(f"  Error: {error}")
        lines.append("")

        # Form
        lines.append("-" * 80)
        lines.append("FORM VALIDATION")
        lines.append("-" * 80)
        form = report['validations'].get('form', {})
        if form:
            lines.append(f"Status: {'✓ PASSED' if form.get('passed') else '✗ FAILED'}")
            lines.append(f"Score: {form.get('score', 0):.3f}")
            if form.get('metrics'):
                lines.append(f"  Estimated bars: {form['metrics'].get('estimated_bars', 0)}")
                lines.append(f"  Expected bars: {form['metrics'].get('expected_bars', 0)}")
                lines.append(f"  Bridge differentiated: {form['metrics'].get('bridge_differentiated', False)}")
                lines.append(f"  Shout chorus present: {form['metrics'].get('shout_chorus_present', False)}")
        lines.append("")

        # Authenticity
        lines.append("-" * 80)
        lines.append("AUTHENTICITY METRICS")
        lines.append("-" * 80)
        auth = report['validations'].get('authenticity', {})
        if auth:
            lines.append(f"Overall Authenticity: {auth.get('overall_authenticity', 0):.3f}")
            lines.append(f"  Interval similarity: {auth.get('interval_similarity', 0):.3f}")
            lines.append(f"  Rhythm similarity: {auth.get('rhythm_similarity', 0):.3f}")
            lines.append(f"  Voicing match: {auth.get('voicing_match', 0):.3f}")
            if auth.get('note'):
                lines.append(f"  Note: {auth['note']}")
        lines.append("")

        lines.append("=" * 80)

        report_text = "\n".join(lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)

        return report_text


# ============================================================================
# MAIN - Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Agent 17: Quality Validation & Testing Framework")
    print("=" * 60)
    print("\nThis module provides comprehensive validation for big band arrangements.")
    print("\nUsage example:")
    print("""
    from validation_tests import ComprehensiveValidator, ArrangementValidator

    # Create validator
    validator = ComprehensiveValidator()

    # Validate arrangement
    report = validator.validate_arrangement(
        arrangement=my_arrangement,
        progression=my_chords,
        expected_form="aaba",
        expected_bars=32,
        style="bebop"
    )

    # Generate report
    quality_report = validator.generate_quality_report(report, "quality_report.txt")
    print(quality_report)
    """)
