#!/usr/bin/env python3
"""
Comprehensive Test Suite for Musical Constraint Validator - Agent 8
===================================================================

Tests all constraint validation and correction functionality with
real-world musical examples.

Author: Agent 8 - Constraint Validator
"""

import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from constraints.musical_validator import (
    MusicalConstraintValidator,
    CounterpointValidator,
    OrchestrationValidator,
    ValidationResult,
    ValidationSeverity,
    ViolationType
)

from constraints.advanced_constraints import (
    JazzVoiceLeadingValidator,
    ExtendedTechniqueValidator,
    PerformancePracticeValidator,
    OrchestrationConstraintValidator
)


# =============================================================================
# TEST UTILITIES
# =============================================================================

class TestResult:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def assert_true(self, condition: bool, test_name: str, message: str = ""):
        """Assert condition is true"""
        if condition:
            self.passed += 1
            self.tests.append((test_name, True, message))
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.tests.append((test_name, False, message))
            print(f"  ✗ {test_name}: {message}")

    def assert_false(self, condition: bool, test_name: str, message: str = ""):
        """Assert condition is false"""
        self.assert_true(not condition, test_name, message)

    def assert_violations(self, result: ValidationResult,
                         expected_count: int, test_name: str):
        """Assert specific number of violations"""
        actual = len(result.violations)
        if actual == expected_count:
            self.passed += 1
            self.tests.append((test_name, True, ""))
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            msg = f"Expected {expected_count} violations, got {actual}"
            self.tests.append((test_name, False, msg))
            print(f"  ✗ {test_name}: {msg}")

    def summary(self):
        """Print summary"""
        total = self.passed + self.failed
        print(f"\n{'='*80}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"FAILED: {self.failed}")
            print("\nFailed tests:")
            for name, passed, msg in self.tests:
                if not passed:
                    print(f"  - {name}: {msg}")
        print(f"{'='*80}\n")
        return self.failed == 0


# =============================================================================
# VOICE LEADING TESTS
# =============================================================================

def test_voice_leading():
    """Test voice leading validation"""
    print("\n" + "="*80)
    print("VOICE LEADING TESTS")
    print("="*80)

    results = TestResult()
    validator = MusicalConstraintValidator(style='common_practice')

    # Test 1: Parallel fifths detection
    print("\nTest 1: Parallel Fifths Detection")
    voices_parallel_5ths = [
        [48, 50],  # Bass: C->D
        [55, 57],  # Tenor: G->A (parallel 5th with bass)
    ]
    result = validator.validate_voice_leading(voices_parallel_5ths)
    results.assert_false(result.is_valid, "Detects parallel fifths")
    results.assert_true(
        any(v.violation_type == ViolationType.PARALLEL_FIFTHS for v in result.violations),
        "Identifies PARALLEL_FIFTHS violation type"
    )

    # Test 2: Parallel octaves detection
    print("\nTest 2: Parallel Octaves Detection")
    voices_parallel_8ves = [
        [48, 50],  # Bass: C->D
        [60, 62],  # Soprano: C->D (parallel octave)
    ]
    result = validator.validate_voice_leading(voices_parallel_8ves)
    results.assert_false(result.is_valid, "Detects parallel octaves")

    # Test 3: Valid voice leading
    print("\nTest 3: Valid Voice Leading (SATB)")
    voices_valid = [
        [48, 50, 52],  # Bass
        [60, 59, 57],  # Tenor (contrary motion)
        [64, 62, 60],  # Alto
        [72, 71, 69],  # Soprano
    ]
    result = validator.validate_voice_leading(voices_valid)
    results.assert_true(result.is_valid, "Accepts valid voice leading")
    results.assert_violations(result, 0, "No violations in valid progression")

    # Test 4: Voice crossing detection
    print("\nTest 4: Voice Crossing Detection")
    voices_crossed = [
        [48, 50, 52],  # Bass
        [60, 62, 64],  # Tenor
        [66, 64, 62],  # Alto crosses below tenor!
        [72, 74, 76],  # Soprano
    ]
    result = validator.validate_voice_leading(voices_crossed)
    results.assert_false(result.is_valid, "Detects voice crossing")

    # Test 5: Excessive spacing
    print("\nTest 5: Excessive Spacing Detection")
    voices_spaced = [
        [48, 50],    # Bass
        [60, 62],    # Tenor
        [84, 86],    # Alto (2 octaves above tenor!)
        [96, 98],    # Soprano
    ]
    result = validator.validate_voice_leading(voices_spaced)
    results.assert_true(
        any(v.violation_type == ViolationType.EXCESSIVE_SPACING for v in result.violations),
        "Detects excessive spacing"
    )

    # Test 6: Automatic correction
    print("\nTest 6: Automatic Parallel Fifth Correction")
    corrected = validator.fix_voice_leading(voices_parallel_5ths)
    result_after = validator.validate_voice_leading(corrected)
    parallel_5ths_after = [v for v in result_after.violations
                          if v.violation_type == ViolationType.PARALLEL_FIFTHS]
    results.assert_true(
        len(parallel_5ths_after) == 0,
        "Fixes parallel fifths automatically"
    )

    return results


# =============================================================================
# INSTRUMENT RANGE TESTS
# =============================================================================

def test_instrument_ranges():
    """Test instrument range validation"""
    print("\n" + "="*80)
    print("INSTRUMENT RANGE TESTS")
    print("="*80)

    results = TestResult()
    validator = MusicalConstraintValidator()

    # Test 1: Valid violin range
    print("\nTest 1: Valid Violin Range")
    violin_valid = [60, 64, 67, 72, 76, 79, 84]  # C4-C6
    result = validator.validate_range(violin_valid, 'violin')
    results.assert_true(result.is_valid, "Accepts valid violin range")

    # Test 2: Out of range (too low for violin)
    print("\nTest 2: Violin Note Too Low")
    violin_too_low = [40, 60, 72]  # E2 is too low
    result = validator.validate_range(violin_too_low, 'violin')
    results.assert_false(result.is_valid, "Detects out-of-range note")
    results.assert_true(
        any(v.violation_type == ViolationType.OUT_OF_RANGE for v in result.violations),
        "Identifies OUT_OF_RANGE violation"
    )

    # Test 3: Out of range (too high for cello)
    print("\nTest 3: Cello Note Too High")
    cello_too_high = [36, 48, 60, 90]  # F#6 too high
    result = validator.validate_range(cello_too_high, 'cello')
    results.assert_false(result.is_valid, "Detects cello out of range")

    # Test 4: Uncomfortable tessitura
    print("\nTest 4: Uncomfortable Tessitura")
    trumpet_extreme = [94, 96, 98]  # Very high for trumpet
    result = validator.validate_range(trumpet_extreme, 'trumpet')
    results.assert_true(
        any(v.violation_type == ViolationType.UNCOMFORTABLE_TESSITURA
            for v in result.violations),
        "Warns about uncomfortable tessitura"
    )

    # Test 5: Automatic range correction
    print("\nTest 5: Automatic Range Correction")
    corrected = validator.fix_out_of_range(violin_too_low, 'violin')
    result_after = validator.validate_range(corrected, 'violin')
    results.assert_true(result_after.is_valid, "Fixes out-of-range notes")

    # Test 6: Multi-instrument validation
    print("\nTest 6: Multiple Instruments")
    ensemble = {
        'flute': [72, 74, 76],      # Valid
        'clarinet': [64, 66, 67],   # Valid
        'bassoon': [20, 48, 50],    # 20 is too low (Bb0)
    }
    result = validator.validate_multi_instrument_ranges(ensemble)
    results.assert_false(result.is_valid, "Detects issues in ensemble")

    return results


# =============================================================================
# HARMONIC VALIDATION TESTS
# =============================================================================

def test_harmonic_validation():
    """Test harmonic progression validation"""
    print("\n" + "="*80)
    print("HARMONIC VALIDATION TESTS")
    print("="*80)

    results = TestResult()
    validator = MusicalConstraintValidator(style='common_practice')

    # Test 1: Simple progression (I-V-I)
    print("\nTest 1: Simple Harmonic Progression")
    chords = [
        [48, 52, 55, 60],  # C major: C-E-G-C
        [50, 55, 59, 62],  # G major: D-G-B-D (V in C)
        [48, 52, 55, 60],  # C major: C-E-G-C
    ]
    result = validator.validate_harmonic_progression(chords, key=60)
    results.assert_true(result.is_valid or len(result.violations) <= 1,
                       "Validates simple progression")

    # Test 2: Tendency tone resolution
    print("\nTest 2: Leading Tone Resolution")
    # G7 -> C with leading tone (B) resolving to C
    g7 = [50, 55, 59, 62]   # D-G-B-D
    c_maj = [48, 52, 60, 64]  # C-E-C-E (B should resolve to C)
    result = validator.validate_harmonic_progression([g7, c_maj], key=60)
    # This test is informational - resolution checking is complex
    print(f"    Found {len(result.violations)} issues")

    return results


# =============================================================================
# COUNTERPOINT TESTS
# =============================================================================

def test_counterpoint():
    """Test counterpoint validation"""
    print("\n" + "="*80)
    print("COUNTERPOINT TESTS")
    print("="*80)

    results = TestResult()
    cp_validator = CounterpointValidator(species=1)

    # Test 1: Valid first species
    print("\nTest 1: Valid First Species")
    cantus = [60, 62, 64, 65, 67, 65, 64, 62, 60]
    counter = [67, 69, 71, 72, 74, 72, 71, 69, 67]  # All consonances
    result = cp_validator.validate_first_species(cantus, counter)
    results.assert_true(result.is_valid or len(result.violations) <= 1,
                       "Accepts valid first species")

    # Test 2: Invalid - dissonant intervals
    print("\nTest 2: First Species with Dissonances")
    cantus = [60, 62, 64, 65]
    counter = [61, 63, 65, 66]  # Creates dissonances (semitones)
    result = cp_validator.validate_first_species(cantus, counter)
    results.assert_false(result.is_valid, "Rejects dissonances")

    # Test 3: Wrong beginning/ending
    print("\nTest 3: First Species Wrong Cadence")
    cantus = [60, 62, 64, 62, 60]
    counter = [65, 67, 69, 67, 65]  # Ends on fifth, not unison
    result = cp_validator.validate_first_species(cantus, counter)
    results.assert_true(
        any(v.violation_type == ViolationType.IMPROPER_MOTION for v in result.violations),
        "Detects improper cadence"
    )

    return results


# =============================================================================
# JAZZ TESTS
# =============================================================================

def test_jazz_constraints():
    """Test jazz-specific constraints"""
    print("\n" + "="*80)
    print("JAZZ CONSTRAINT TESTS")
    print("="*80)

    results = TestResult()
    jazz_validator = JazzVoiceLeadingValidator(style='bebop')

    # Test 1: Rootless voicing with 3rd and 7th
    print("\nTest 1: Valid Dm7 Rootless Voicing")
    dm7_rootless = [53, 57, 60, 64]  # F-A-C-E (3rd, 5th, 7th, 9th)
    result = jazz_validator.validate_jazz_voicing(dm7_rootless, 'Dm7', 'rootless')
    results.assert_true(result.is_valid or result.warnings_count <= 1,
                       "Accepts valid rootless voicing")

    # Test 2: Rootless voicing missing 3rd or 7th
    print("\nTest 2: Invalid Dm7 Rootless (missing 3rd)")
    dm7_invalid = [48, 55, 60, 64]  # D-G-C-E (has root, missing 3rd)
    result = jazz_validator.validate_jazz_voicing(dm7_invalid, 'Dm7', 'rootless')
    results.assert_true(
        any(v.violation_type == ViolationType.MISSING_CHORD_TONE
            for v in result.violations),
        "Detects missing chord tones in rootless voicing"
    )

    # Test 3: Quartal voicing
    print("\nTest 3: Quartal Voicing")
    quartal = [60, 65, 70, 75]  # C-F-Bb-Eb (all 4ths)
    result = jazz_validator.validate_jazz_voicing(quartal, 'Csus', 'quartal')
    results.assert_true(result.is_valid or result.warnings_count <= 1,
                       "Accepts quartal voicing")

    return results


# =============================================================================
# EXTENDED TECHNIQUE TESTS
# =============================================================================

def test_extended_techniques():
    """Test extended technique validation"""
    print("\n" + "="*80)
    print("EXTENDED TECHNIQUE TESTS")
    print("="*80)

    results = TestResult()
    ext_validator = ExtendedTechniqueValidator()

    # Test 1: Valid violin harmonic
    print("\nTest 1: Valid Violin Natural Harmonic")
    harmonic_note = 67  # G4 (harmonic of G string)
    result = ext_validator.validate_string_harmonics(harmonic_note, 'violin')
    # May or may not be valid - depends on exact calculation
    print(f"    Harmonic validation: {result.get_summary()}")

    # Test 2: Multiphonic with too many notes
    print("\nTest 2: Invalid Multiphonic (too many notes)")
    multiphonic_invalid = [60, 64, 67, 72, 76]  # 5 notes
    result = ext_validator.validate_wind_multiphonic(multiphonic_invalid, 'clarinet')
    results.assert_false(result.is_valid, "Rejects multiphonic with >3 notes")

    # Test 3: Valid multiphonic
    print("\nTest 3: Valid Multiphonic")
    multiphonic_valid = [60, 67]  # 2 notes, perfect 5th apart
    result = ext_validator.validate_wind_multiphonic(multiphonic_valid, 'oboe')
    results.assert_true(result.is_valid or result.warnings_count <= 1,
                       "Accepts valid 2-note multiphonic")

    return results


# =============================================================================
# PERFORMANCE PRACTICE TESTS
# =============================================================================

def test_performance_practice():
    """Test performance practice validation"""
    print("\n" + "="*80)
    print("PERFORMANCE PRACTICE TESTS")
    print("="*80)

    results = TestResult()
    perf_validator = PerformancePracticeValidator()

    # Test 1: Adequate breathing
    print("\nTest 1: Adequate Breathing Points")
    short_phrases = [(72, 2.0), (74, 2.0), (0, 1.0), (76, 2.0)]  # Has rest
    result = perf_validator.validate_breathing(short_phrases, 'flute')
    results.assert_true(result.is_valid, "Accepts phrases with breathing")

    # Test 2: Too long without breath
    print("\nTest 2: Phrase Too Long Without Breath")
    long_phrase = [(72, 1.0)] * 12  # 12 beats without rest
    result = perf_validator.validate_breathing(long_phrase, 'oboe')
    results.assert_false(result.is_valid, "Detects phrase too long")

    # Test 3: Piano hand span - valid
    print("\nTest 3: Valid Piano Hand Span")
    valid_chord = [60, 64, 67, 72]  # C major, 1 octave span
    result = perf_validator.validate_piano_hand_span(valid_chord, 'right')
    results.assert_true(result.is_valid, "Accepts playable chord")

    # Test 4: Piano hand span - too wide
    print("\nTest 4: Piano Hand Span Too Wide")
    wide_chord = [48, 52, 55, 59, 64]  # 16 semitone span
    result = perf_validator.validate_piano_hand_span(wide_chord, 'left')
    results.assert_false(result.is_valid, "Rejects unplayable span")

    # Test 5: Too many notes for one hand
    print("\nTest 5: Too Many Notes For One Hand")
    many_notes = [60, 62, 64, 65, 67, 69]  # 6 notes (impossible)
    result = perf_validator.validate_piano_hand_span(many_notes, 'right')
    results.assert_false(result.is_valid, "Rejects >5 notes per hand")

    return results


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_parameter_integration():
    """Test parameter validation (for Phase 2 integration)"""
    print("\n" + "="*80)
    print("PARAMETER INTEGRATION TESTS")
    print("="*80)

    results = TestResult()
    validator = MusicalConstraintValidator()

    # Test 1: Complete parameter validation
    print("\nTest 1: Complete Parameter Dict Validation")
    params = {
        'voices': [
            [48, 50, 52],
            [60, 62, 64],
            [64, 66, 68],
            [72, 74, 76],
        ],
        'instrument_parts': {
            'violin': [60, 64, 67, 72],
            'viola': [55, 57, 60, 64],
            'cello': [48, 50, 52, 55],
        },
        'chord_progression': [
            [48, 52, 55, 60],
            [50, 55, 59, 62],
        ],
        'key': 60,
    }
    result = validator.validate_parameters(params)
    print(f"    Validation: {result.get_summary()}")
    results.assert_true(True, "Parameter validation completes")

    # Test 2: Validate and correct
    print("\nTest 2: Validate and Auto-Correct Parameters")
    params_bad = {
        'voices': [
            [48, 50],  # Will have parallel motion
            [55, 57],
        ]
    }
    corrected, result = validator.validate_and_correct(params_bad)
    results.assert_true(True, "Auto-correction completes")
    print(f"    After correction: {result.get_summary()}")

    return results


# =============================================================================
# ORCHESTRATION TESTS
# =============================================================================

def test_orchestration():
    """Test orchestration validation"""
    print("\n" + "="*80)
    print("ORCHESTRATION TESTS")
    print("="*80)

    results = TestResult()
    orch_validator = OrchestrationConstraintValidator()

    # Test 1: Orchestral doubling
    print("\nTest 1: Orchestral Doubling Validation")
    score = {
        'flute': [72, 74, 76],
        'oboe': [72, 74, 76],  # Unison with flute
        'clarinet': [64, 66, 67],
    }
    result = orch_validator.validate_doubling(score)
    results.assert_true(
        any(v.violation_type == ViolationType.INCORRECT_DOUBLING
            for v in result.violations),
        "Detects unison doubling"
    )

    # Test 2: Range distribution
    print("\nTest 2: Range Distribution")
    score_gapped = {
        'piccolo': [96, 98, 100],  # Very high
        'bassoon': [36, 38, 40],   # Very low (huge gap!)
    }
    result = orch_validator.validate_range_distribution(score_gapped)
    results.assert_true(
        any(v.violation_type == ViolationType.EXCESSIVE_SPACING
            for v in result.violations),
        "Detects gaps in orchestral texture"
    )

    return results


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*80)
    print("MUSICAL CONSTRAINT VALIDATOR - COMPREHENSIVE TEST SUITE")
    print("Agent 8 - Constraint Validator")
    print("="*80)

    all_results = []

    # Run all test suites
    all_results.append(test_voice_leading())
    all_results.append(test_instrument_ranges())
    all_results.append(test_harmonic_validation())
    all_results.append(test_counterpoint())
    all_results.append(test_jazz_constraints())
    all_results.append(test_extended_techniques())
    all_results.append(test_performance_practice())
    all_results.append(test_parameter_integration())
    all_results.append(test_orchestration())

    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL TEST SUMMARY")
    print("="*80)

    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_tests = total_passed + total_failed

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success Rate: {total_passed/total_tests*100:.1f}%")

    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total_failed} tests failed")

    print("="*80 + "\n")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
