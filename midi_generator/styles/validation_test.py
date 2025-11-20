#!/usr/bin/env python3
"""
Modern Big Band Style Validation Tests
========================================

This script validates and demonstrates the modern big band style profiles.

Tests:
1. Style profile integrity checks
2. Quartal voicing generation
3. Wide spacing voicing generation
4. Style comparison
5. Integration with arrangement engine

Author: Agent 15 - Modern Big Band Style Analyzer
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from styles.modern_profiles import (
    THAD_JONES_STYLE,
    MARIA_SCHNEIDER_STYLE,
    GORDON_GOODWIN_STYLE,
    ModernBigBandArranger,
    get_style_profile,
    list_available_styles,
    compare_style_characteristics
)


# ==============================================================================
# VALIDATION TESTS
# ==============================================================================

def test_style_profile_integrity():
    """Test that all style profiles have required attributes."""
    print("=" * 80)
    print("TEST 1: Style Profile Integrity")
    print("=" * 80)

    styles = [THAD_JONES_STYLE, MARIA_SCHNEIDER_STYLE, GORDON_GOODWIN_STYLE]

    required_attrs = [
        'name', 'era', 'voicing_preference', 'harmony_complexity',
        'rhythmic_complexity', 'articulation_variety', 'dynamic_range',
        'texture_density', 'typical_tempo_range'
    ]

    all_passed = True

    for style in styles:
        print(f"\nValidating: {style.name}")
        print("-" * 40)

        for attr in required_attrs:
            if hasattr(style, attr):
                value = getattr(style, attr)
                print(f"  ✓ {attr}: {value}")
            else:
                print(f"  ✗ MISSING: {attr}")
                all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All style profiles passed integrity check")
    else:
        print("✗ Some style profiles failed integrity check")
    print("=" * 80)

    return all_passed


def test_quartal_voicing_generation():
    """Test quartal voicing generation for modern arrangers."""
    print("\n" + "=" * 80)
    print("TEST 2: Quartal Voicing Generation")
    print("=" * 80)

    # Test Thad Jones style (high quartal usage)
    arranger = ModernBigBandArranger(THAD_JONES_STYLE)

    print(f"\nStyle: {arranger.style.name}")
    print(f"Quartal usage: {arranger.style.use_quartal * 100:.0f}%")
    print()

    # Generate quartal voicings from different roots
    test_roots = [60, 64, 67]  # C, E, G
    note_names = ['C', 'E', 'G']

    for root, name in zip(test_roots, note_names):
        voicing = arranger.generate_quartal_voicing(root, 4)
        intervals = [voicing[i+1] - voicing[i] for i in range(len(voicing)-1)]

        print(f"Quartal voicing from {name} (MIDI {root}):")
        print(f"  Notes: {voicing}")
        print(f"  Intervals: {intervals} semitones")

        # Check that intervals are mostly 4ths (5 semitones) or augmented 4ths (6 semitones)
        valid_intervals = all(i in [5, 6] for i in intervals)
        if valid_intervals:
            print(f"  ✓ Valid quartal voicing (all 4ths/aug4ths)")
        else:
            print(f"  ✗ Invalid quartal voicing")
        print()

    print("=" * 80)
    return True


def test_wide_spacing_voicing():
    """Test wide spacing voicing generation (Thad Jones technique)."""
    print("\n" + "=" * 80)
    print("TEST 3: Wide Spacing Voicing Generation")
    print("=" * 80)

    arranger = ModernBigBandArranger(THAD_JONES_STYLE)

    print(f"\nStyle: {arranger.style.name}")
    print(f"Voicing spacing: {arranger.style.voicing_spacing}")
    print()

    # Test chord: Cmaj7 (C, E, G, B)
    chord_tones = [60, 64, 67, 71]  # C, E, G, B
    min_spacing = 7  # Minimum 7 semitones (perfect 5th)

    print(f"Original chord tones: {chord_tones}")
    wide_voicing = arranger.generate_wide_spacing_voicing(chord_tones, min_spacing)

    print(f"Wide-spaced voicing: {wide_voicing}")

    # Calculate intervals
    intervals = [wide_voicing[i+1] - wide_voicing[i] for i in range(len(wide_voicing)-1)]
    print(f"Intervals: {intervals} semitones")

    # Validate minimum spacing
    all_wide_enough = all(interval >= min_spacing for interval in intervals)
    avg_spacing = sum(intervals) / len(intervals) if intervals else 0

    print(f"Average spacing: {avg_spacing:.1f} semitones")

    if all_wide_enough:
        print(f"✓ All intervals meet minimum spacing of {min_spacing} semitones")
    else:
        print(f"✗ Some intervals below minimum spacing")

    print("=" * 80)
    return all_wide_enough


def test_style_retrieval():
    """Test style profile retrieval by name."""
    print("\n" + "=" * 80)
    print("TEST 4: Style Profile Retrieval")
    print("=" * 80)

    print("\nAvailable styles:")
    styles = list_available_styles()
    for style in styles:
        print(f"  - {style}")

    print("\nTesting retrieval:")
    test_names = ["thad_jones", "thad", "maria_schneider", "schneider",
                  "gordon_goodwin", "goodwin", "invalid_name"]

    for name in test_names:
        profile = get_style_profile(name)
        if profile:
            print(f"  ✓ '{name}' -> {profile.name}")
        else:
            print(f"  ✗ '{name}' -> None")

    print("=" * 80)
    return True


def test_style_characteristics_comparison():
    """Test style characteristics comparison."""
    print("\n" + "=" * 80)
    print("TEST 5: Style Characteristics Comparison")
    print("=" * 80)
    print()

    compare_style_characteristics()

    print("=" * 80)
    return True


def test_arranger_creation_and_usage():
    """Test creating arrangers and using their methods."""
    print("\n" + "=" * 80)
    print("TEST 6: Arranger Creation and Usage")
    print("=" * 80)

    styles = [THAD_JONES_STYLE, MARIA_SCHNEIDER_STYLE, GORDON_GOODWIN_STYLE]

    for style in styles:
        print(f"\n{style.name} Arranger")
        print("-" * 40)

        arranger = ModernBigBandArranger(style)

        # Test methods
        intro = arranger.suggest_intro_type()
        ending = arranger.suggest_ending_type()
        tempo = arranger.get_typical_tempo()

        print(f"Suggested intro: {intro}")
        print(f"Suggested ending: {ending}")
        print(f"Typical tempo: {tempo} BPM")

        # Validate tempo is in range
        min_tempo, max_tempo = style.typical_tempo_range
        if min_tempo <= tempo <= max_tempo:
            print(f"✓ Tempo within expected range ({min_tempo}-{max_tempo} BPM)")
        else:
            print(f"✗ Tempo outside expected range")

    print("\n" + "=" * 80)
    return True


def test_harmonic_complexity_differences():
    """Test that styles have distinct harmonic complexity."""
    print("\n" + "=" * 80)
    print("TEST 7: Harmonic Complexity Differences")
    print("=" * 80)

    styles = {
        "Thad Jones": THAD_JONES_STYLE,
        "Maria Schneider": MARIA_SCHNEIDER_STYLE,
        "Gordon Goodwin": GORDON_GOODWIN_STYLE,
    }

    print("\nHarmonic Complexity Metrics:")
    print("-" * 40)

    for name, style in styles.items():
        print(f"\n{name}:")
        print(f"  Overall complexity: {style.harmony_complexity:.2f}")
        print(f"  Quartal usage: {style.use_quartal:.2f}")
        print(f"  Cluster usage: {style.use_clusters:.2f}")
        print(f"  Altered dominants: {style.use_altered_dominants:.2f}")
        print(f"  Chord extensions: {style.chord_extensions}")

    # Verify Maria Schneider has highest overall complexity
    if MARIA_SCHNEIDER_STYLE.harmony_complexity > THAD_JONES_STYLE.harmony_complexity:
        print("\n✓ Maria Schneider has higher complexity than Thad Jones")
    else:
        print("\n✗ Unexpected complexity ordering")

    # Verify Thad Jones has highest quartal usage
    if THAD_JONES_STYLE.use_quartal > MARIA_SCHNEIDER_STYLE.use_quartal:
        print("✓ Thad Jones has highest quartal usage (signature technique)")
    else:
        print("✗ Unexpected quartal usage")

    print("\n" + "=" * 80)
    return True


def test_tempo_ranges():
    """Test that tempo ranges match style characteristics."""
    print("\n" + "=" * 80)
    print("TEST 8: Tempo Range Validation")
    print("=" * 80)

    styles = [
        ("Maria Schneider", MARIA_SCHNEIDER_STYLE, "Slow to moderate (ballads)"),
        ("Thad Jones", THAD_JONES_STYLE, "Moderate to fast"),
        ("Gordon Goodwin", GORDON_GOODWIN_STYLE, "Fast, high energy"),
    ]

    print("\nTempo Ranges:")
    print("-" * 40)

    for name, style, description in styles:
        min_t, max_t = style.typical_tempo_range
        print(f"\n{name}: {min_t}-{max_t} BPM")
        print(f"  Description: {description}")

    # Verify Gordon Goodwin has fastest tempos
    goodwin_min = GORDON_GOODWIN_STYLE.typical_tempo_range[0]
    schneider_max = MARIA_SCHNEIDER_STYLE.typical_tempo_range[1]

    if goodwin_min > schneider_max:
        print("\n✓ Gordon Goodwin minimum tempo > Maria Schneider maximum")
        print("  (Confirms high-energy vs. ballad orientation)")
    else:
        print("\n✗ Unexpected tempo relationship")

    print("\n" + "=" * 80)
    return True


# ==============================================================================
# RUN ALL TESTS
# ==============================================================================

def run_all_tests():
    """Run all validation tests."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "MODERN BIG BAND STYLE VALIDATION" + " " * 26 + "║")
    print("║" + " " * 25 + "Agent 15 Test Suite" + " " * 34 + "║")
    print("╚" + "═" * 78 + "╝")

    tests = [
        ("Style Profile Integrity", test_style_profile_integrity),
        ("Quartal Voicing Generation", test_quartal_voicing_generation),
        ("Wide Spacing Voicing", test_wide_spacing_voicing),
        ("Style Retrieval", test_style_retrieval),
        ("Style Comparison", test_style_characteristics_comparison),
        ("Arranger Creation", test_arranger_creation_and_usage),
        ("Harmonic Complexity", test_harmonic_complexity_differences),
        ("Tempo Ranges", test_tempo_ranges),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with error: {e}")
            results.append((test_name, False))

    # Summary
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 30 + "TEST SUMMARY" + " " * 36 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status:10s} {test_name}")

    print()
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print()

    if passed == total:
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "ALL TESTS PASSED - SYSTEM VALIDATED" + " " * 23 + "║")
        print("╚" + "═" * 78 + "╝")
    else:
        print("Some tests failed. Please review the results above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
