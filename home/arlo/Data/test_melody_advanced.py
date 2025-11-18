#!/usr/bin/env python3
"""
Comprehensive test suite for melody_advanced.py

Tests all 6 major systems:
1. Contour Theory
2. Motif Development
3. Phrase Structure
4. Intervallic Control
5. Ornamentation
6. Narrative Arc
"""

import unittest
from melody_advanced import (
    ContourTheory, ContourType, ContourAnalysis,
    MotifDevelopment, Motif, MotifTransformation,
    PhraseStructure, PhraseType, Phrase, Period,
    IntervallicControl, IntervalProfile,
    Ornamentation, OrnamentType,
    MusicalNarrative, NarrativeSection, NarrativeArc
)


class TestContourTheory(unittest.TestCase):
    """Test contour analysis and generation"""

    def test_analyze_arch_contour(self):
        """Test arch contour detection"""
        arch_melody = [60, 62, 64, 67, 65, 62, 60]
        analysis = ContourTheory.analyze_contour(arch_melody)

        self.assertEqual(analysis.contour_type, ContourType.ARCH)
        self.assertEqual(analysis.range, 7)
        self.assertEqual(analysis.tessitura, 62)
        self.assertGreater(analysis.step_leap_ratio, 0)

    def test_analyze_ascending_contour(self):
        """Test ascending contour detection"""
        ascending_melody = [60, 62, 64, 65, 67, 69, 71, 72]
        analysis = ContourTheory.analyze_contour(ascending_melody)

        self.assertEqual(analysis.overall_direction, "ascending")
        self.assertGreater(len(analysis.tension_curve), 0)

    def test_analyze_descending_contour(self):
        """Test descending contour detection"""
        descending_melody = [72, 71, 69, 67, 65, 64, 62, 60]
        analysis = ContourTheory.analyze_contour(descending_melody)

        self.assertEqual(analysis.overall_direction, "descending")

    def test_generate_arch_contour(self):
        """Test arch contour generation"""
        melody = ContourTheory.generate_contour(
            length=8,
            target_contour=ContourType.ARCH,
            pitch_range=(60, 72)
        )

        self.assertEqual(len(melody), 8)
        self.assertTrue(all(60 <= p <= 72 for p in melody))
        # Should have peak in middle
        max_idx = melody.index(max(melody))
        self.assertGreater(max_idx, 1)
        self.assertLess(max_idx, 7)

    def test_generate_wave_contour(self):
        """Test wave contour generation"""
        melody = ContourTheory.generate_contour(
            length=16,
            target_contour=ContourType.WAVE,
            pitch_range=(60, 72)
        )

        self.assertEqual(len(melody), 16)
        # Wave should have multiple peaks
        analysis = ContourTheory.analyze_contour(melody)
        self.assertGreater(len(analysis.peak_points), 1)

    def test_tension_curve_calculation(self):
        """Test tension curve generation"""
        melody = [60, 64, 67, 72, 67, 64, 60]
        analysis = ContourTheory.analyze_contour(melody)

        # Tension should peak near highest note
        max_tension_idx = analysis.tension_curve.index(max(analysis.tension_curve))
        max_pitch_idx = melody.index(max(melody))
        # Should be close (within 1 index)
        self.assertLessEqual(abs(max_tension_idx - max_pitch_idx), 1)

    def test_step_leap_ratio(self):
        """Test step/leap ratio calculation"""
        stepwise = [60, 61, 62, 63, 64, 65]
        leapy = [60, 67, 55, 72, 60, 64]

        stepwise_analysis = ContourTheory.analyze_contour(stepwise)
        leapy_analysis = ContourTheory.analyze_contour(leapy)

        # Stepwise should have higher ratio
        self.assertGreater(stepwise_analysis.step_leap_ratio,
                          leapy_analysis.step_leap_ratio)


class TestMotifDevelopment(unittest.TestCase):
    """Test motif transformations"""

    def setUp(self):
        """Create test motif"""
        self.motif = Motif(
            pitches=[60, 64, 67],
            durations=[1.0, 1.0, 2.0],
            name="Test Motif"
        )

    def test_sequence(self):
        """Test sequential transposition"""
        sequences = MotifDevelopment.sequence(
            self.motif,
            transpositions=[2, 4, 7]
        )

        self.assertEqual(len(sequences), 3)
        self.assertEqual(sequences[0].pitches, [62, 66, 69])
        self.assertEqual(sequences[1].pitches, [64, 68, 71])
        self.assertEqual(sequences[2].pitches, [67, 71, 74])

    def test_inversion(self):
        """Test melodic inversion"""
        inverted = MotifDevelopment.inversion(self.motif)

        # First note should be same (axis)
        self.assertEqual(inverted.pitches[0], self.motif.pitches[0])
        # Intervals should be inverted
        original_interval1 = self.motif.pitches[1] - self.motif.pitches[0]
        inverted_interval1 = inverted.pitches[1] - inverted.pitches[0]
        self.assertEqual(original_interval1, -inverted_interval1)

    def test_inversion_custom_axis(self):
        """Test inversion around custom axis"""
        axis = 64
        inverted = MotifDevelopment.inversion(self.motif, axis=axis)

        # Check symmetry around axis
        for original, inv in zip(self.motif.pitches, inverted.pitches):
            self.assertEqual(original + inv, 2 * axis)

    def test_retrograde(self):
        """Test retrograde (backward)"""
        retrograde = MotifDevelopment.retrograde(self.motif)

        self.assertEqual(retrograde.pitches, list(reversed(self.motif.pitches)))
        self.assertEqual(retrograde.durations, list(reversed(self.motif.durations)))

    def test_retrograde_inversion(self):
        """Test retrograde inversion"""
        retro_inv = MotifDevelopment.retrograde_inversion(self.motif)

        # Should be reversed AND inverted
        retrograde = MotifDevelopment.retrograde(self.motif)
        inverted = MotifDevelopment.inversion(retrograde)

        self.assertEqual(retro_inv.pitches, inverted.pitches)

    def test_augmentation(self):
        """Test rhythmic augmentation"""
        augmented = MotifDevelopment.augmentation(self.motif, factor=2.0)

        self.assertEqual(augmented.pitches, self.motif.pitches)
        self.assertEqual(augmented.durations, [2.0, 2.0, 4.0])

    def test_diminution(self):
        """Test rhythmic diminution"""
        diminished = MotifDevelopment.diminution(self.motif, factor=0.5)

        self.assertEqual(diminished.pitches, self.motif.pitches)
        self.assertEqual(diminished.durations, [0.5, 0.5, 1.0])

    def test_fragmentation(self):
        """Test motif fragmentation"""
        fragment = MotifDevelopment.fragmentation(
            self.motif,
            fragment_length=2,
            start_idx=0
        )

        self.assertEqual(len(fragment.pitches), 2)
        self.assertEqual(fragment.pitches, self.motif.pitches[:2])

    def test_extension(self):
        """Test motif extension"""
        extended = MotifDevelopment.extension(
            self.motif,
            additional_pitches=[70, 72],
            additional_durations=[1.0, 2.0]
        )

        self.assertEqual(len(extended.pitches), 5)
        self.assertEqual(extended.pitches[-2:], [70, 72])

    def test_modal_shift_major_to_minor(self):
        """Test major to minor modal shift"""
        major_motif = Motif(
            pitches=[60, 64, 67],  # C major triad
            durations=[1.0, 1.0, 1.0],
            name="Major"
        )

        minor = MotifDevelopment.modal_shift(
            major_motif,
            original_mode="major",
            target_mode="minor"
        )

        # Third should be lowered (64 -> 63)
        self.assertEqual(minor.pitches[1], 63)


class TestPhraseStructure(unittest.TestCase):
    """Test phrase and period generation"""

    def setUp(self):
        """Create test motif"""
        self.motif = Motif(
            pitches=[60, 64, 67, 64],
            durations=[1.0, 1.0, 1.0, 1.0],
            name="Test"
        )

    def test_create_period(self):
        """Test period creation"""
        period = PhraseStructure.create_period(self.motif, length_beats=8.0)

        self.assertIsInstance(period, Period)
        self.assertEqual(period.antecedent.phrase_type, PhraseType.ANTECEDENT)
        self.assertEqual(period.consequent.phrase_type, PhraseType.CONSEQUENT)
        self.assertEqual(period.antecedent.cadence_type, "half")
        self.assertEqual(period.consequent.cadence_type, "authentic")

    def test_create_sentence(self):
        """Test sentence structure creation"""
        sentence = PhraseStructure.create_sentence(self.motif, length_beats=8.0)

        self.assertEqual(sentence.phrase_type, PhraseType.SENTENCE)
        self.assertEqual(sentence.cadence_type, "authentic")
        self.assertGreater(len(sentence.melody), len(self.motif.pitches))


class TestIntervallicControl(unittest.TestCase):
    """Test interval analysis and control"""

    def test_analyze_stepwise_melody(self):
        """Test analysis of stepwise melody"""
        stepwise = [60, 61, 62, 63, 64, 65]
        profile = IntervallicControl.analyze_intervals(stepwise)

        self.assertEqual(profile.step_count, 5)
        self.assertEqual(profile.leap_count, 0)
        self.assertEqual(profile.largest_interval, 1)
        self.assertLessEqual(profile.average_interval, 1.0)

    def test_analyze_leapy_melody(self):
        """Test analysis of leapy melody"""
        leapy = [60, 72, 55, 67, 60]
        profile = IntervallicControl.analyze_intervals(leapy)

        self.assertGreater(profile.leap_count, 0)
        self.assertGreater(profile.largest_interval, 7)

    def test_step_leap_ratio(self):
        """Test step/leap ratio calculation"""
        mixed = [60, 62, 64, 72, 70, 69, 67, 60]
        profile = IntervallicControl.analyze_intervals(mixed)

        # Should have both steps and leaps
        self.assertGreater(profile.step_count, 0)
        self.assertGreater(profile.leap_count, 0)
        self.assertGreater(profile.step_leap_ratio, 0)

    def test_direction_changes(self):
        """Test direction change counting"""
        zigzag = [60, 64, 62, 67, 65, 69, 67]
        profile = IntervallicControl.analyze_intervals(zigzag)

        # Zigzag should have many direction changes
        self.assertGreater(profile.direction_changes, 2)

    def test_enforce_recovery_after_leap(self):
        """Test leap recovery enforcement"""
        leapy = [60, 72, 55, 67]  # Large leaps
        corrected = IntervallicControl.enforce_recovery_after_leap(leapy, max_leap=5)

        # Should insert stepwise recovery notes
        self.assertGreater(len(corrected), len(leapy))

    def test_balance_step_leap_ratio(self):
        """Test step/leap ratio balancing"""
        leapy = [60, 72, 55, 67, 60]
        balanced = IntervallicControl.balance_step_leap_ratio(leapy, target_ratio=3.0)

        # Balanced melody should have more notes (inserted steps)
        self.assertGreater(len(balanced), len(leapy))

        balanced_profile = IntervallicControl.analyze_intervals(balanced)
        original_profile = IntervallicControl.analyze_intervals(leapy)

        # Balanced should have better ratio
        self.assertGreater(balanced_profile.step_leap_ratio,
                          original_profile.step_leap_ratio)


class TestOrnamentation(unittest.TestCase):
    """Test ornament application"""

    def setUp(self):
        """Create test melody"""
        self.melody = [60, 64, 67, 64, 60]
        self.durations = [1.0, 1.0, 2.0, 1.0, 2.0]

    def test_add_trill(self):
        """Test trill addition"""
        ornamented, orn_durs = Ornamentation.add_trill(
            self.melody,
            self.durations,
            note_idx=2
        )

        # Should have more notes
        self.assertGreater(len(ornamented), len(self.melody))
        # Should alternate between main note and upper neighbor
        self.assertIn(67, ornamented)
        self.assertIn(69, ornamented)

    def test_add_mordent(self):
        """Test mordent addition"""
        ornamented, orn_durs = Ornamentation.add_mordent(
            self.melody,
            self.durations,
            note_idx=1
        )

        # Should have more notes (main-auxiliary-main = 3 notes)
        self.assertGreater(len(ornamented), len(self.melody))

    def test_add_turn(self):
        """Test turn addition"""
        ornamented, orn_durs = Ornamentation.add_turn(
            self.melody,
            self.durations,
            note_idx=2
        )

        # Turn adds 4 notes (upper-main-lower-main)
        expected_length = len(self.melody) + 3  # 1 note becomes 4
        self.assertEqual(len(ornamented), expected_length)

    def test_add_appoggiatura(self):
        """Test appoggiatura addition"""
        ornamented, orn_durs = Ornamentation.add_appoggiatura(
            self.melody,
            self.durations,
            note_idx=2,
            interval=2
        )

        # Should have more notes
        self.assertGreater(len(ornamented), len(self.melody))
        # Should have note above target
        self.assertIn(69, ornamented)

    def test_ornament_duration_preservation(self):
        """Test that total duration is preserved"""
        original_duration = sum(self.durations)

        _, trill_durs = Ornamentation.add_trill(self.melody, self.durations, 2)
        _, mordent_durs = Ornamentation.add_mordent(self.melody, self.durations, 1)
        _, turn_durs = Ornamentation.add_turn(self.melody, self.durations, 2)

        # Total duration should be same
        self.assertAlmostEqual(sum(trill_durs), original_duration, places=5)
        self.assertAlmostEqual(sum(mordent_durs), original_duration, places=5)
        self.assertAlmostEqual(sum(turn_durs), original_duration, places=5)


class TestMusicalNarrative(unittest.TestCase):
    """Test narrative arc generation"""

    def test_create_narrative_arc(self):
        """Test narrative arc creation"""
        arc = MusicalNarrative.create_narrative_arc(
            total_length_beats=32.0,
            climax_position=0.618
        )

        self.assertIsInstance(arc, NarrativeArc)
        self.assertEqual(arc.overall_length, 32.0)
        self.assertAlmostEqual(arc.climax_beat, 32.0 * 0.618, places=1)

        # Should have all sections
        self.assertEqual(len(arc.sections), 5)
        self.assertIn(NarrativeSection.EXPOSITION, arc.sections)
        self.assertIn(NarrativeSection.CLIMAX, arc.sections)
        self.assertIn(NarrativeSection.RESOLUTION, arc.sections)

    def test_tension_curve_shape(self):
        """Test tension curve follows narrative"""
        arc = MusicalNarrative.create_narrative_arc(32.0)

        tensions = [t for beat, t in arc.tension_curve]

        # Exposition should have low tension
        expo_tensions = tensions[:7]
        self.assertLess(max(expo_tensions), 0.4)

        # Climax should have high tension
        climax_idx = int(arc.climax_beat)
        if climax_idx < len(tensions):
            self.assertGreater(tensions[climax_idx], 0.8)

        # Resolution should have low tension
        resolution_tensions = tensions[-4:]
        if len(resolution_tensions) > 0:
            self.assertLess(max(resolution_tensions), 0.5)

    def test_apply_narrative_to_melody(self):
        """Test applying narrative to melody"""
        melody = [60, 62, 64, 65, 67, 69, 70, 72, 70, 69, 67, 65, 64, 62, 60, 60]
        beats = list(range(len(melody)))

        arc = MusicalNarrative.create_narrative_arc(16.0)
        adjusted = MusicalNarrative.apply_narrative_to_melody(melody, arc, beats)

        self.assertEqual(len(adjusted), len(melody))
        # Pitches should be adjusted based on tension
        # (exact values depend on tension curve, but should differ from original)
        self.assertNotEqual(adjusted, melody)

    def test_golden_ratio_climax(self):
        """Test golden ratio climax position"""
        arc = MusicalNarrative.create_narrative_arc(100.0, climax_position=0.618)

        # Climax should be near golden ratio
        self.assertAlmostEqual(arc.climax_beat, 61.8, places=1)


class TestIntegration(unittest.TestCase):
    """Test integration between systems"""

    def test_contour_with_motif_development(self):
        """Test combining contour generation with motif development"""
        # Generate arch contour
        arch = ContourTheory.generate_contour(8, ContourType.ARCH, (60, 72))

        # Use as motif
        motif = Motif(pitches=arch[:4], durations=[1.0]*4, name="Arch Fragment")

        # Develop motif
        inverted = MotifDevelopment.inversion(motif)
        sequenced = MotifDevelopment.sequence(motif, [2, 4])

        # All should work
        self.assertEqual(len(inverted.pitches), 4)
        self.assertEqual(len(sequenced), 2)

    def test_phrase_with_ornamentation(self):
        """Test adding ornaments to phrases"""
        motif = Motif(pitches=[60, 64, 67, 64], durations=[1.0]*4)
        sentence = PhraseStructure.create_sentence(motif)

        # Add ornament to phrase
        ornamented, _ = Ornamentation.add_trill(
            sentence.melody,
            [1.0] * len(sentence.melody),
            note_idx=2
        )

        self.assertGreater(len(ornamented), len(sentence.melody))

    def test_narrative_with_intervallic_control(self):
        """Test narrative arc with interval control"""
        melody = [60, 67, 55, 72, 60, 64, 69, 62, 60, 57, 64, 60]
        beats = list(range(len(melody)))

        # Balance intervals
        balanced = IntervallicControl.balance_step_leap_ratio(melody, 3.0)

        # Apply narrative
        arc = MusicalNarrative.create_narrative_arc(len(balanced))
        narrative = MusicalNarrative.apply_narrative_to_melody(
            balanced,
            arc,
            list(range(len(balanced)))
        )

        # Should work together
        self.assertEqual(len(narrative), len(balanced))


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestContourTheory))
    suite.addTests(loader.loadTestsFromTestCase(TestMotifDevelopment))
    suite.addTests(loader.loadTestsFromTestCase(TestPhraseStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestIntervallicControl))
    suite.addTests(loader.loadTestsFromTestCase(TestOrnamentation))
    suite.addTests(loader.loadTestsFromTestCase(TestMusicalNarrative))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")

    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
