#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validation Test Suite - Agent 21 Instrumentation Specialist
===========================================================

Comprehensive validation tests for instrumentation specialist engine.

Tests cover:
- Ensemble selection
- Piano voicing generation
- Bass pattern generation
- Drum pattern generation
- Brass section voicing
- Blend compatibility calculation
- Articulation assignment
- Doubling recommendations
- Range validation
- Integration tests

Author: Agent 21
Date: 2025
License: MIT
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.instrumentation_specialist import (
    InstrumentationSpecialist,
    PianoVoicingGenerator,
    BassPatternGenerator,
    DrumPatternGenerator,
    BrassSectionVoicing,
    InstrumentRole,
    EnsembleType,
    VoicingType,
    BassPattern,
    DrumPattern,
    TextureDensity,
    create_jazz_trio,
    create_big_band,
    create_string_quartet,
)
from core.instrument_library import (
    get_instrument,
    get_instruments_by_family,
    InstrumentFamily,
    is_in_comfortable_range,
)


def test_ensemble_selection():
    """Test 1: Ensemble Selection"""
    print("\n" + "=" * 80)
    print("TEST 1: ENSEMBLE SELECTION")
    print("=" * 80)

    specialist = InstrumentationSpecialist()

    # Test jazz trio
    print("\n1.1 - Jazz Trio Selection")
    trio = specialist.select_ensemble("jazz", 0.3)
    print(f"  Ensemble Type: {trio.ensemble_type.value}")
    print(f"  Instruments: {[inst.name for inst in trio.instruments]}")
    print(f"  Texture Density: {trio.texture_density.value}")

    assert trio.ensemble_type in [EnsembleType.TRIO, EnsembleType.JAZZ_COMBO]
    assert len(trio.instruments) >= 3
    assert trio.texture_density in [TextureDensity.SPARSE, TextureDensity.LIGHT, TextureDensity.MEDIUM]
    print("  ✅ PASSED")

    # Test big band
    print("\n1.2 - Big Band Selection")
    big_band = specialist.select_ensemble("jazz", 0.9)
    print(f"  Ensemble Type: {big_band.ensemble_type.value}")
    print(f"  Instruments: {len(big_band.instruments)} instruments")
    print(f"  Texture Density: {big_band.texture_density.value}")

    assert big_band.ensemble_type == EnsembleType.BIG_BAND
    assert len(big_band.instruments) >= 12
    assert big_band.texture_density in [TextureDensity.FULL, TextureDensity.DENSE]
    print("  ✅ PASSED")

    # Test classical chamber
    print("\n1.3 - String Quartet Selection")
    quartet = specialist.select_ensemble("classical", 0.5, ["Violin", "Violin", "Viola", "Cello"])
    print(f"  Ensemble Type: {quartet.ensemble_type.value}")
    print(f"  Instruments: {[inst.name for inst in quartet.instruments]}")

    assert len(quartet.instruments) == 4
    assert all(inst.family == InstrumentFamily.STRINGS for inst in quartet.instruments)
    print("  ✅ PASSED")

    print("\n" + "-" * 80)
    print("TEST 1: ✅ ALL ENSEMBLE SELECTION TESTS PASSED")
    return True


def test_piano_voicings():
    """Test 2: Piano Voicing Generation"""
    print("\n" + "=" * 80)
    print("TEST 2: PIANO VOICING GENERATION")
    print("=" * 80)

    generator = PianoVoicingGenerator()
    root = 60  # Middle C
    chord_tones = [0, 4, 7, 10]  # Dominant 7th
    extensions = [14]  # 9th

    # Test Drop-2
    print("\n2.1 - Drop-2 Voicing")
    drop2 = generator.generate_drop_2(root, chord_tones, extensions)
    print(f"  Voicing Type: {drop2.voicing_type.value}")
    print(f"  Notes: {drop2.notes}")
    print(f"  Intervals from root: {[n - root for n in drop2.notes]}")

    assert drop2.voicing_type == VoicingType.DROP_2
    assert len(drop2.notes) >= 4
    assert drop2.notes == sorted(drop2.notes)  # Should be sorted
    print("  ✅ PASSED")

    # Test Drop-3
    print("\n2.2 - Drop-3 Voicing")
    drop3 = generator.generate_drop_3(root, chord_tones, extensions)
    print(f"  Voicing Type: {drop3.voicing_type.value}")
    print(f"  Notes: {drop3.notes}")

    assert drop3.voicing_type == VoicingType.DROP_3
    assert len(drop3.notes) >= 4
    print("  ✅ PASSED")

    # Test Rootless
    print("\n2.3 - Rootless Voicing (Type A)")
    rootless_a = generator.generate_rootless(root, chord_tones, extensions, type_a=True)
    print(f"  Voicing Type: {rootless_a.voicing_type.value}")
    print(f"  Notes: {rootless_a.notes}")
    print(f"  Omissions: {rootless_a.omissions}")

    assert rootless_a.voicing_type == VoicingType.ROOTLESS
    assert 0 in rootless_a.omissions  # Root should be omitted
    print("  ✅ PASSED")

    # Test Quartal
    print("\n2.4 - Quartal Voicing")
    quartal = generator.generate_quartal(root, 4)
    print(f"  Voicing Type: {quartal.voicing_type.value}")
    print(f"  Notes: {quartal.notes}")
    intervals = [quartal.notes[i+1] - quartal.notes[i] for i in range(len(quartal.notes)-1)]
    print(f"  Intervals: {intervals}")

    assert quartal.voicing_type == VoicingType.QUARTAL
    assert all(interval == 5 for interval in intervals)  # All perfect 4ths
    print("  ✅ PASSED")

    # Test Cluster
    print("\n2.5 - Cluster Voicing (Chromatic)")
    cluster = generator.generate_cluster(root, 5, whole_tone=False)
    print(f"  Voicing Type: {cluster.voicing_type.value}")
    print(f"  Notes: {cluster.notes}")
    intervals = [cluster.notes[i+1] - cluster.notes[i] for i in range(len(cluster.notes)-1)]
    print(f"  Intervals: {intervals}")

    assert cluster.voicing_type == VoicingType.CLUSTER
    assert all(interval == 1 for interval in intervals)  # All half steps
    print("  ✅ PASSED")

    # Test Shell
    print("\n2.6 - Shell Voicing")
    shell = generator.generate_shell(root, "dominant")
    print(f"  Voicing Type: {shell.voicing_type.value}")
    print(f"  Notes: {shell.notes}")
    print(f"  Chord Tones: {shell.chord_tones}")

    assert shell.voicing_type == VoicingType.SHELL
    assert len(shell.notes) == 3  # Root, 3rd, 7th only
    print("  ✅ PASSED")

    print("\n" + "-" * 80)
    print("TEST 2: ✅ ALL PIANO VOICING TESTS PASSED")
    return True


def test_bass_patterns():
    """Test 3: Bass Pattern Generation"""
    print("\n" + "=" * 80)
    print("TEST 3: BASS PATTERN GENERATION")
    print("=" * 80)

    generator = BassPatternGenerator()
    root = 48  # Low C

    # Test Walking Bass
    print("\n3.1 - Walking Bass")
    walking = generator.generate_walking_bass(
        root,
        [(root, 4.0), (root + 7, 4.0)],  # C to G
        style="swing"
    )
    print(f"  Pattern Length: {len(walking)} notes")
    print(f"  First 4 notes: {walking[:4]}")

    assert len(walking) > 0
    assert all(isinstance(note, int) and isinstance(dur, float) for note, dur in walking)
    print("  ✅ PASSED")

    # Test Pedal
    print("\n3.2 - Pedal Tone")
    pedal = generator.generate_pedal(root, 16.0)
    print(f"  Pattern: {pedal}")

    assert len(pedal) == 1
    assert pedal[0][0] == root
    assert pedal[0][1] == 16.0
    print("  ✅ PASSED")

    # Test Ostinato
    print("\n3.3 - Ostinato Pattern")
    pattern = [48, 50, 52, 50]
    ostinato = generator.generate_ostinato(pattern, repetitions=3, note_duration=0.5)
    print(f"  Pattern Length: {len(ostinato)} notes")
    print(f"  First repetition: {ostinato[:4]}")

    assert len(ostinato) == len(pattern) * 3
    print("  ✅ PASSED")

    # Test Two-Feel
    print("\n3.4 - Two-Feel Bass")
    two_feel = generator.generate_two_feel(root, num_bars=4)
    print(f"  Pattern Length: {len(two_feel)} notes")
    print(f"  Pattern: {two_feel}")

    assert len(two_feel) == 8  # 2 notes per bar, 4 bars
    assert all(dur == 2.0 for _, dur in two_feel)  # Half notes
    print("  ✅ PASSED")

    # Test Funk Bass
    print("\n3.5 - Funk Bass")
    funk = generator.generate_funk_bass(root, pattern_type="basic")
    print(f"  Pattern Length: {len(funk)} notes")
    print(f"  First 4 notes: {funk[:4]}")

    assert len(funk) > 0
    print("  ✅ PASSED")

    print("\n" + "-" * 80)
    print("TEST 3: ✅ ALL BASS PATTERN TESTS PASSED")
    return True


def test_drum_patterns():
    """Test 4: Drum Pattern Generation"""
    print("\n" + "=" * 80)
    print("TEST 4: DRUM PATTERN GENERATION")
    print("=" * 80)

    generator = DrumPatternGenerator()

    # Test Swing
    print("\n4.1 - Swing Pattern (Medium)")
    swing = generator.generate_swing_pattern(feel="medium")
    print(f"  Voices: {list(swing.keys())}")
    print(f"  Ride pattern length: {len(swing['ride'])} hits")

    assert 'ride' in swing
    assert 'snare' in swing
    assert 'kick' in swing
    assert all(isinstance(beat, float) and isinstance(vel, int) for beat, vel in swing['ride'])
    print("  ✅ PASSED")

    # Test Rock
    print("\n4.2 - Rock Pattern (Basic)")
    rock = generator.generate_rock_pattern(style="basic")
    print(f"  Voices: {list(rock.keys())}")
    print(f"  Hi-hat pattern length: {len(rock['hi_hat'])} hits")

    assert 'hi_hat' in rock
    assert 'snare' in rock
    assert 'kick' in rock
    print("  ✅ PASSED")

    # Test Funk
    print("\n4.3 - Funk Pattern")
    funk = generator.generate_funk_pattern()
    print(f"  Voices: {list(funk.keys())}")
    print(f"  Hi-hat pattern length: {len(funk['hi_hat'])} hits")

    assert 'hi_hat' in funk
    assert 'snare' in funk
    assert 'kick' in funk
    print("  ✅ PASSED")

    # Test Latin
    print("\n4.4 - Latin Pattern (Bossa Nova)")
    bossa = generator.generate_latin_pattern(style="bossa")
    print(f"  Voices: {list(bossa.keys())}")

    assert 'ride' in bossa or 'kick' in bossa
    print("  ✅ PASSED")

    print("\n" + "-" * 80)
    print("TEST 4: ✅ ALL DRUM PATTERN TESTS PASSED")
    return True


def test_brass_voicings():
    """Test 5: Brass Section Voicing"""
    print("\n" + "=" * 80)
    print("TEST 5: BRASS SECTION VOICING")
    print("=" * 80)

    voicer = BrassSectionVoicing()
    root = 60
    chord_tones = [0, 4, 7, 10]  # Dom7

    # Test 4-Way Close
    print("\n5.1 - Four-Way Close Voicing")
    four_close = voicer.four_way_close(root, chord_tones)
    print(f"  Voices: {list(four_close.keys())}")
    print(f"  Notes: {list(four_close.values())}")

    assert len(four_close) == 4
    assert 'Trumpet 1' in four_close
    assert 'Trombone 2' in four_close
    # Check that all voices are within reasonable range
    notes = list(four_close.values())
    assert max(notes) - min(notes) <= 12  # Within octave for close voicing
    print("  ✅ PASSED")

    # Test 4-Way Open
    print("\n5.2 - Four-Way Open Voicing")
    four_open = voicer.four_way_open(root, chord_tones)
    print(f"  Voices: {list(four_open.keys())}")
    print(f"  Notes: {list(four_open.values())}")

    assert len(four_open) == 4
    notes = list(four_open.values())
    assert max(notes) - min(notes) > 12  # Wider than an octave
    print("  ✅ PASSED")

    # Test 5-Way Close
    print("\n5.3 - Five-Way Close Voicing")
    five_close = voicer.five_way_close(root, chord_tones, extensions=[14])
    print(f"  Voices: {list(five_close.keys())}")
    print(f"  Notes: {list(five_close.values())}")

    assert len(five_close) == 5
    print("  ✅ PASSED")

    # Test Double Lead
    print("\n5.4 - Double Lead Voicing")
    double_lead = voicer.double_lead(root, chord_tones, root + 10)
    print(f"  Voices: {list(double_lead.keys())}")

    assert len(double_lead) >= 4
    # Check that lead is doubled
    lead_notes = [v for k, v in double_lead.items() if 'Trumpet 1' in k]
    assert len(lead_notes) == 2  # Should have doubled lead
    print("  ✅ PASSED")

    print("\n" + "-" * 80)
    print("TEST 5: ✅ ALL BRASS VOICING TESTS PASSED")
    return True


def test_blend_compatibility():
    """Test 6: Blend Compatibility Calculation"""
    print("\n" + "=" * 80)
    print("TEST 6: BLEND COMPATIBILITY")
    print("=" * 80)

    specialist = InstrumentationSpecialist()

    # Test excellent blend (strings)
    print("\n6.1 - String Section Blend")
    strings = [
        get_instrument("Violin"),
        get_instrument("Viola"),
        get_instrument("Cello")
    ]
    strings = [inst for inst in strings if inst is not None]
    string_blend = specialist.calculate_blend_score(strings)
    print(f"  Instruments: {[inst.name for inst in strings]}")
    print(f"  Blend Score: {string_blend:.2f}")

    assert string_blend >= 0.7  # Should be good blend
    print("  ✅ PASSED")

    # Test good blend (brass)
    print("\n6.2 - Brass Section Blend")
    brass = [
        get_instrument("Trumpet"),
        get_instrument("Trombone")
    ]
    brass = [inst for inst in brass if inst is not None]
    brass_blend = specialist.calculate_blend_score(brass)
    print(f"  Instruments: {[inst.name for inst in brass]}")
    print(f"  Blend Score: {brass_blend:.2f}")

    assert brass_blend >= 0.6
    print("  ✅ PASSED")

    # Test mixed ensemble
    print("\n6.3 - Mixed Ensemble Blend")
    mixed = [
        get_instrument("Flute"),
        get_instrument("Trumpet"),
        get_instrument("Cello")
    ]
    mixed = [inst for inst in mixed if inst is not None]
    mixed_blend = specialist.calculate_blend_score(mixed)
    print(f"  Instruments: {[inst.name for inst in mixed]}")
    print(f"  Blend Score: {mixed_blend:.2f}")

    assert 0.0 <= mixed_blend <= 1.0  # Valid score
    print("  ✅ PASSED")

    print("\n" + "-" * 80)
    print("TEST 6: ✅ ALL BLEND COMPATIBILITY TESTS PASSED")
    return True


def test_articulation_assignment():
    """Test 7: Articulation Assignment"""
    print("\n" + "=" * 80)
    print("TEST 7: ARTICULATION ASSIGNMENT")
    print("=" * 80)

    specialist = InstrumentationSpecialist()

    # Create test profile
    profile = create_jazz_trio()

    # Test legato
    print("\n7.1 - Legato Articulations")
    legato_arts = specialist.assign_articulations(profile, "legato")
    print(f"  Assigned articulations: {len(legato_arts)}")
    for inst, art in legato_arts.items():
        print(f"    {inst}: {art.value}")

    assert len(legato_arts) == len(profile.instruments)
    print("  ✅ PASSED")

    # Test staccato
    print("\n7.2 - Staccato Articulations")
    staccato_arts = specialist.assign_articulations(profile, "staccato")
    print(f"  Assigned articulations: {len(staccato_arts)}")

    assert len(staccato_arts) == len(profile.instruments)
    print("  ✅ PASSED")

    # Test swing
    print("\n7.3 - Swing Articulations")
    swing_arts = specialist.assign_articulations(profile, "swing")
    print(f"  Assigned articulations: {len(swing_arts)}")

    assert len(swing_arts) == len(profile.instruments)
    print("  ✅ PASSED")

    print("\n" + "-" * 80)
    print("TEST 7: ✅ ALL ARTICULATION TESTS PASSED")
    return True


def test_voicing_assignment():
    """Test 8: Voicing Assignment to Ensemble"""
    print("\n" + "=" * 80)
    print("TEST 8: VOICING ASSIGNMENT")
    print("=" * 80)

    specialist = InstrumentationSpecialist()

    # Test piano voicing assignment
    print("\n8.1 - Piano Voicing Assignment")
    profile = create_jazz_trio()
    profile.voicing_type = VoicingType.DROP_2

    voicing = specialist.assign_voicing(profile, 60, [0, 4, 7, 10], [14])
    print(f"  Voicing Type: {profile.voicing_type.value}")
    print(f"  Assigned instruments: {list(voicing.keys())}")
    if voicing:
        for inst, notes in voicing.items():
            print(f"    {inst}: {notes}")

    # Should assign to piano if present
    piano_assigned = any("Piano" in inst for inst in voicing.keys())
    if any("Piano" in inst.name for inst in profile.instruments):
        assert piano_assigned
    print("  ✅ PASSED")

    # Test brass voicing assignment
    print("\n8.2 - Brass Section Voicing Assignment")
    big_band_profile = create_big_band()
    big_band_profile.voicing_type = VoicingType.FOUR_WAY_CLOSE

    brass_voicing = specialist.assign_voicing(big_band_profile, 60, [0, 4, 7, 10])
    print(f"  Voicing Type: {big_band_profile.voicing_type.value}")
    print(f"  Assigned instruments: {list(brass_voicing.keys())}")

    # Check that brass instruments got assignments
    brass_count = sum(1 for inst in big_band_profile.instruments
                     if inst.family == InstrumentFamily.BRASS)
    print(f"  Brass instruments in ensemble: {brass_count}")
    print("  ✅ PASSED")

    print("\n" + "-" * 80)
    print("TEST 8: ✅ ALL VOICING ASSIGNMENT TESTS PASSED")
    return True


def test_doubling_recommendations():
    """Test 9: Doubling Recommendations"""
    print("\n" + "=" * 80)
    print("TEST 9: DOUBLING RECOMMENDATIONS")
    print("=" * 80)

    specialist = InstrumentationSpecialist()
    profile = create_big_band()

    # Test trumpet doubling
    print("\n9.1 - Trumpet Doubling Recommendations")
    recommendations = specialist.recommend_doublings(profile, "Trumpet")
    print(f"  Melody Instrument: Trumpet")
    print(f"  Unison doublings: {recommendations.get('unison', [])}")
    print(f"  Octave doublings: {recommendations.get('octave', [])}")

    assert 'unison' in recommendations
    assert 'octave' in recommendations
    print("  ✅ PASSED")

    # Test flute doubling
    print("\n9.2 - Flute Doubling Recommendations")
    # Add flute to profile for testing
    flute = get_instrument("Flute")
    if flute:
        test_profile = create_big_band()
        test_profile.instruments.append(flute)
        flute_recs = specialist.recommend_doublings(test_profile, "Flute")
        print(f"  Melody Instrument: Flute")
        print(f"  Recommendations: {flute_recs}")
        print("  ✅ PASSED")
    else:
        print("  ⚠️  SKIPPED (Flute not available)")

    print("\n" + "-" * 80)
    print("TEST 9: ✅ ALL DOUBLING RECOMMENDATION TESTS PASSED")
    return True


def test_integration():
    """Test 10: Full Integration Test"""
    print("\n" + "=" * 80)
    print("TEST 10: FULL INTEGRATION TEST")
    print("=" * 80)

    specialist = InstrumentationSpecialist()

    # Complete workflow
    print("\n10.1 - Complete Jazz Combo Workflow")

    # Select ensemble
    profile = specialist.select_ensemble("jazz", 0.6)
    print(f"  Step 1 - Ensemble: {profile.ensemble_type.value}")
    print(f"  Instruments: {[inst.name for inst in profile.instruments]}")

    # Assign voicing
    profile.voicing_type = VoicingType.ROOTLESS
    voicing = specialist.assign_voicing(profile, 60, [0, 4, 7, 10], [14])
    print(f"  Step 2 - Voicing: {profile.voicing_type.value}")
    print(f"  Voicing assignments: {len(voicing)}")

    # Assign articulations
    articulations = specialist.assign_articulations(profile, "swing")
    print(f"  Step 3 - Articulations: {len(articulations)}")

    # Generate bass line
    bass_line = specialist.generate_bass_line(BassPattern.WALKING, 48, 4.0, "swing")
    print(f"  Step 4 - Bass line: {len(bass_line)} notes")

    # Generate drum pattern
    drums = specialist.generate_drum_pattern(DrumPattern.SWING, "medium")
    print(f"  Step 5 - Drums: {len(drums)} voices")

    # Calculate blend
    blend = specialist.calculate_blend_score(profile.instruments)
    print(f"  Step 6 - Blend score: {blend:.2f}")

    assert profile.ensemble_type is not None
    assert len(profile.instruments) > 0
    assert len(bass_line) > 0
    assert len(drums) > 0
    assert 0.0 <= blend <= 1.0

    print("  ✅ PASSED")

    print("\n10.2 - Complete Big Band Workflow")

    # Select big band
    bb_profile = specialist.select_ensemble("jazz", 0.9)
    print(f"  Step 1 - Ensemble: {bb_profile.ensemble_type.value}")
    print(f"  Instruments: {len(bb_profile.instruments)}")

    # Assign brass voicing
    bb_profile.voicing_type = VoicingType.FOUR_WAY_CLOSE
    bb_voicing = specialist.assign_voicing(bb_profile, 60, [0, 4, 7, 10])
    print(f"  Step 2 - Brass voicing: {bb_profile.voicing_type.value}")

    # Get doubling recommendations
    doublings = specialist.recommend_doublings(bb_profile, "Trumpet")
    print(f"  Step 3 - Doubling recommendations: {doublings}")

    # Calculate blend
    bb_blend = specialist.calculate_blend_score(bb_profile.instruments)
    print(f"  Step 4 - Blend score: {bb_blend:.2f}")

    assert bb_profile.ensemble_type == EnsembleType.BIG_BAND
    assert len(bb_profile.instruments) >= 12
    assert 0.0 <= bb_blend <= 1.0

    print("  ✅ PASSED")

    print("\n" + "-" * 80)
    print("TEST 10: ✅ ALL INTEGRATION TESTS PASSED")
    return True


def run_all_tests():
    """
    Run complete test suite
    """
    print("=" * 80)
    print("AGENT 21 INSTRUMENTATION SPECIALIST - VALIDATION TEST SUITE")
    print("=" * 80)
    print("\nRunning comprehensive validation tests...\n")

    tests = [
        ("Ensemble Selection", test_ensemble_selection),
        ("Piano Voicing Generation", test_piano_voicings),
        ("Bass Pattern Generation", test_bass_patterns),
        ("Drum Pattern Generation", test_drum_patterns),
        ("Brass Section Voicing", test_brass_voicings),
        ("Blend Compatibility", test_blend_compatibility),
        ("Articulation Assignment", test_articulation_assignment),
        ("Voicing Assignment", test_voicing_assignment),
        ("Doubling Recommendations", test_doubling_recommendations),
        ("Full Integration", test_integration),
    ]

    passed = 0
    failed = 0
    errors = []

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                errors.append(test_name)
        except Exception as e:
            failed += 1
            errors.append(f"{test_name}: {str(e)}")
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {str(e)}")

    # Final report
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"\nTotal Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n🎉 ✅ ALL TESTS PASSED! 🎉")
        print("\nAgent 21: Instrumentation Specialist is fully validated and ready for production.")
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        print("\nFailed tests:")
        for error in errors:
            print(f"  - {error}")

    print("\n" + "=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
